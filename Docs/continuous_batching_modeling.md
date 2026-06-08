# Continuous Batching — Modeling-Level Analysis (Gemma4 MoE / Mojo)

Scope: what must change at the modeling layer (forward + kernels) to support
continuous batching. A scheduler is assumed to exist and hands `forward` a
well-formed, possibly ragged/packed masked batch of tokens plus KV-cache ids.
KV paging is assumed to be a statically-allocated, fixed-size, NUMA-aware
contiguous block arena; KV caches are id-tagged; reserving a block places it
under that id.

All file:line references are to the real source. The primary entry point is
`Gemma4.forward` in `modeling/gemma_4_moe.mojo:859`.

---

## 1) CURRENT BATCHING MODEL

The model is **single-sequence, chunked-prefill** with a scalar running position
`base_pos`. There is no batch axis anywhere. The only axes are a token axis
(`chunk_len`, a slice of one sequence) and feature/head axes. "Batching" of
generation steps is expressed solely by advancing `base_pos` and re-calling
`forward`; a long prompt is processed by **chunking one sequence** into
`SLIDING_WINDOW`-sized pieces (`gemma_4_moe.mojo:895-971`).

### forward signature

`Gemma4.forward` (`modeling/gemma_4_moe.mojo:859-865`):

```mojo
def forward[tok_origin: ImmutOrigin, //](
    mut self,
    token_ids: Span[Int32, tok_origin],   # one sequence, [total_len]
    base_pos: Int,                        # scalar running offset
) -> TemporalLogitsView[C.VOCAB_SIZE]
```

There is no `batch`, no `num_seqs`, no per-token sequence id, no position-id
array, no explicit mask, no KV-id argument. `base_pos + total_len <=
max_seq_len` is asserted (`gemma_4_moe.mojo:878-881`). The KV cache is implicit
model state (the per-layer `sliding_kv` / `full_kv` slots in the arena), not a
forward argument.

### Chunk loop and the dims that flow

The body is a `while consumed < total_len` loop (`gemma_4_moe.mojo:895`) that
takes `chunk_len = min(remaining, SLIDING_WINDOW)` (`:897`) tokens and advances
`pos += chunk_len` (`:971`). Inside, all activations are **token-major
`[chunk_len, feature]`** living in arena slots, and every kernel takes
`seq_len`/`chunk_len` as a runtime row count:

- **Embedding**: `dispatch_embed_lookup[hidden=C.HIDDEN, scale=embed_scale]`
  into `x_main_ranks`, rows = `shard_rows`, `chunk_len`
  (`gemma_4_moe.mojo:904-908`); followed by `dispatch_allreduce_inplace` over
  `chunk_len * C.HIDDEN` (`:909-910`). So embedding is **vocab-row-sharded across
  ranks** and reduced — a tensor-parallel collective, not single-node.
- **input rms_norm**: `dispatch_rms_norm[hidden=C.HIDDEN]( x_main_ranks,
  x_res_ranks, ..., chunk_len)` (`gemma_4_moe.mojo:924-928`). Per-token rows.
- **Attention** is split by layer kind via `LAYER_SCHEDULE` (every 6th layer is
  FULL, else SLIDING — `gemma4_common.mojo:58-66`):
  - SLIDING: `dispatch_sliding_attention_qkv[...](layout, ctx, pos, chunk_len,
    entry.local_idx, ...)` (`gemma_4_moe.mojo:937-941`, def at `:428`).
  - FULL: `dispatch_full_attention_qkv[...](layout, ctx, pos, chunk_len, ...)`
    (`gemma_4_moe.mojo:931-935`, def at `:519`).
- **Q/K/V projections**: GEMMs into scratch `q/kv` buffers. For sliding,
  `dispatch_gemm_chained_qkv[cols=C.HIDDEN](xs, q_proj, k_proj, v_proj, q_outs,
  k_outs, v_outs, q_rows, kv_rows, seq_len, ...)` (`gemma_4_moe.mojo:472-478`).
  `q_rows = SlidingQ.data_n(degree)` = `Q_DIM_SLIDING/degree` (`:453`). For full,
  two `dispatch_gemm` calls (`:558-563`); Q/K are `Replicated` (head counts are
  comptime) while o_proj is column-sharded (`gemma_4_moe.mojo:88-90`). Critically,
  `GemmKernel` **partitions over the output `rows` axis, with the token count `m`
  a runtime inner loop** (`gemm.mojo:60-78`, `gemm_range` loops `m_panel` over
  `m`). So the token/batch count is already fully runtime-flexible in every GEMM;
  packing more tokens is just a larger `m`.
- **head rms_norm (q_norm/k_norm)**: `dispatch_rms_norm_qkv_heads[head_dim=...]`
  over `num_q_heads`, `num_kv_heads`, `seq_len` (`gemma_4_moe.mojo:480-486`,
  `:566-572`). Per-(token,head) row of length `head_dim`.
- **RoPE + KV write fused**: `dispatch_rope_cache_write[half, pair_stride,
  head_dim, slot_mask]` (`gemma_4_moe.mojo:492-500`, `:578-586`; kernel in
  `kernels/rope.mojo:50-105`). It RoPEs Q in place and writes RoPE'd K + raw V
  into the KV cache slot. See sections 2-3.
- **Attention proper**:
  - SLIDING: `dispatch_sliding_attention[head_dim, max_q, gqa_ratio, window,
    cache_size]` (`gemma_4_moe.mojo:504-509`; kernel
    `attention_dispatch_kernels.mojo:20-96`). Per-token flash over the window.
  - FULL: `dispatch_full_attention[head_dim, num_q, gqa_ratio, kv_stride,
    partial_stride]` producing partials merged by `merge_partials`
    (`gemma_4_moe.mojo:594-599`; kernel `attention_dispatch_kernels.mojo:99+`,
    flash chunk in `flash_attention.mojo`, prefill block in
    `flash_attention_prefill.mojo`, merge in `logsum_merge.mojo`).
- **o_proj**: `dispatch_gemm_cols[rows=C.HIDDEN]` into `xs`
  (`gemma_4_moe.mojo:511-516`, `:601-606`), then attention output is
  allreduced (`:943-944`) and fused into residual via
  `fused_norm_residual_add` (`:946-950`).
- **FFN + MoE**: `dispatch_ffn` (`gemma_4_moe.mojo:952-954`, def `:685`) which
  runs the dense gate/up/gelu (`:716-726`), then `dispatch_moe` (`:728-729`, def
  `:609`): router → merge candidates → expert schedules → phase1 gate_up →
  phase2 down, all over `seq_len` and `experts_per_rank = NUM_EXPERTS/degree`
  (`:624`). MoE output is allreduced (`:680-682`).
- **norms/embedding/logits tail**: only the **last token** of the **last chunk**
  gets final norm + logits: `x_last = x_main_ranks.shifted((chunk_len-1)*HIDDEN)`
  (`gemma_4_moe.mojo:956-957`), `dispatch_rms_norm(..., 1, ...)` (`:959-963`),
  `dispatch_gemv_softcap[cols=C.HIDDEN, cap=LOGIT_SOFTCAP]` producing
  `vocab_per_rank` logits (`:965-968`). Only one row of logits is ever produced.

### Where seq/batch is assumed contiguous/uniform

1. **Position is a single scalar `pos`** threaded as `base_pos` into RoPE
   (`rope.mojo:79` `pos = base_pos + tok`), sliding attention
   (`attention_dispatch_kernels.mojo:43` `q_pos = base_pos + tok`), and the full
   path. One monotonic run, one sequence.
2. **One KV region per layer**, addressed by absolute position derived from
   `pos` (section 2). No id-tag, no block table.
3. **Sliding window assumes a single causal frame**: `window_start = max(0,
   q_pos - window + 1)` (`attention_dispatch_kernels.mojo:44`). Cross-sequence
   membership cannot be expressed.
4. **Activation slots are sized `[SLIDING_WINDOW, HIDDEN]`** (`x_main`,
   `x_residual` — `gemma_4_moe.mojo:183-184`); MoE/FFN/attention scratch are all
   sized `SLIDING_WINDOW * ...` (`gemma_4_moe.mojo:226-324`). The token axis is
   capped at `SLIDING_WINDOW=1024` per chunk and assumed to be one sequence.

There is **no packing/ragged support**: no `cu_seqlens`, no per-token seq id, no
block table. The `Slot` machinery in `modeling/slot.mojo` is a *weight/state
binding* abstraction (offset stamping, per-rank `Binding`), unrelated to
per-sequence KV slots; it carries no `seq_id`.

---

## 2) KV CACHE TODAY

### Allocation — per-layer arena slots, sharded by NUMA degree

KV lives in the arena as typed slots, not a forward argument:

- Sliding: `SlidingKVSlots[max_seq_len]` with `k,v : Slot[BF16,
  Gemma4StateShapes.SlidingKV]` where `SlidingKV =
  TensorColumnSharded[SLIDING_CACHE, KV_DIM_SLIDING]` and `SLIDING_CACHE =
  2*SLIDING_WINDOW` (`gemma_4_moe.mojo:104-105, 165-168`). It is a **2W ring**
  per sliding layer.
- Full: `FullKVSlots[max_seq_len]` with `k,v : Slot[BF16,
  Gemma4StateShapes.FullKV]`, `FullKV = ContextRowSharded[max_seq_len,
  KV_DIM_FULL]` (`gemma_4_moe.mojo:106, 171-174`). A **dense `max_seq_len`**
  context cache, **row(context)-sharded across NUMA ranks**: position `pos` lives
  on rank `pos % degree`.
- Both are `Repeated[...]` per layer (`gemma_4_moe.mojo:199-200`,
  `:386-396`) allocated once at load and bound per layer via
  `layout.sliding_kv.base(...)` / `layout.full_kv.base(...)`
  (`:488-490`, `:574-576`).

### Indexing / position — scalar offset, two regimes

Position is **a running scalar** `pos` (= `base_pos` of the chunk + intra-chunk
`tok`), computed inside `RopeCacheWriteKernel.execute`
(`rope.mojo:78-79`): `pos = base_pos + tok`. The cache slot is then derived:

- Sliding ring: `slot = (pos // cache_degree) & slot_mask` with
  `slot_mask = cache_size-1 = 2W-1` and `cache_degree = 1`
  (`rope.mojo:88-89`; dispatched with `slot_mask=cache_size-1`, third
  numeric arg `1` = `cache_degree` at `gemma_4_moe.mojo:492-500`). So a sliding
  token maps to `pos mod 2W`.
- Full context: `slot_mask = -1` (no masking) and `cache_degree = degree`
  (`gemma_4_moe.mojo:578-586`). The write happens only on the owning rank:
  `if pos % cache_degree == rank: slot = (pos // cache_degree) & slot_mask`
  (`rope.mojo:88-89`). So full KV is **context-sharded round-robin by
  position across NUMA nodes** — token `pos` is written and read on node
  `pos % degree`.

### Write path (prefill vs decode) — unified, fused with RoPE

There is **one** write path. `RopeCacheWriteKernel.execute`
(`rope.mojo:75-101`) loops `for tok in range(start,end)`: RoPEs each Q head
in place (`:84-86`), and when the rank owns the position, writes RoPE'd K head-by-head
via `rope_head_to` (`:92-96`) and copies raw V with `memcpy` of `kv_stride`
(`:98-100`). KV slot stride = `num_kv * head_dim` (`:77`). Prefill and decode
differ only in `seq_len` (chunk size); the same kernel handles both. The write
is **vectorized** over `head_dim` (`rope.mojo:32-47` use `W`-wide SIMD).

### Read path (attention)

- SLIDING (`attention_dispatch_kernels.mojo:40-54`): per token, `q_pos =
  base_pos+tok`, `window_start = max(0, q_pos-window+1)`, `kv_count = q_pos -
  window_start + 1`; per head calls `flash_attention_chunk[head_dim, simd_w]`
  which walks `kv_count` keys in the ring, wrapping at `cache_size`, with online
  softmax. The read assumes the ring holds exactly this one sequence's last `W`
  keys.
- FULL: gathers K/V partials across the context shard and merges via
  `merge_partials` / `MergeSegment` (`logsum_merge.mojo`,
  `flash_attention_prefill.mojo`). Each rank flashes its **local** slice of
  context (the positions it owns) and the partials are log-sum-exp merged. This
  is already a NUMA-distributed flash attention, but over **one sequence's**
  full `[0, kv_len)` context.

### RoPE positions

RoPE reads precomputed `cos/sin` rows by absolute pos: `cos_row = cos_table +
pos*half`, `pos = base_pos + tok` (`rope.mojo:79-81`). Tables are built once
per rank in `model_init` via `init_rope_table` (sliding, θ=10000) and
`init_rope_table_partial_strided` (full, θ=1e6, partial rotary 64 dims —
`gemma_4_moe.mojo:845-857`, `rope.mojo:144-170`). Tables are `Replicated`
per rank (`gemma_4_moe.mojo:107-108, 177-180`): every attention worker reads
them locally — the project's "data closest to the most-frequent reader."

**Summary:** position is a single running scalar; sliding KV is a per-layer 2W
ring keyed by `pos mod 2W`; full KV is a dense context cache **round-robin
context-sharded by position across NUMA nodes**; RoPE indexes the same
scalar-derived absolute pos from per-rank-replicated tables. Nothing is
per-sequence; nothing is id-tagged; nothing is block-paged.

---

## 3) MASKING TODAY

Causality is **implicit / positional**, never an explicit mask tensor:

- SLIDING: enforced by the key range itself — `window_start = max(0, q_pos -
  window + 1)` and `kv_count = q_pos - window_start + 1`
  (`attention_dispatch_kernels.mojo:44-45`). Query at `q_pos` attends to ring
  keys `[window_start, q_pos]`. Causal + windowed by construction; no mask
  array.
- FULL: `flash_prefill_block(... causal_offset ...)`
  (`flash_attention_prefill.mojo:10-20`) takes a `causal_offset` and a
  `[kv_start, kv_end)` range; the per-query causal cutoff is computed from the
  query row + `causal_offset` (the absolute `base_pos` anchor). Again positional,
  no explicit mask tensor.

**Can it express a packed batch?** No. Every causal cutoff is a function of a
single global `base_pos` anchor over one contiguous `[0, kv_len)` (or one ring).
There is no per-token sequence-membership input, so block-diagonal masking
(query of sequence A must not see keys of sequence B even at lower position) is
inexpressible. The sliding ring (`pos mod 2W`) would actively alias keys from
different sequences onto the same ring slots.

---

## 4) WHAT MUST CHANGE FOR CONTINUOUS BATCHING (core deliverable)

Strategy: preserve the existing **token-major packed `[total_tokens, feature]`**
activation layout (matmuls, norms, gelu, router, embedding, allreduce are all
already token-parallel and batching-agnostic) and the existing **NUMA-sharded,
flash, log-sum-merge** attention machinery. Add: (a) per-token sequence metadata,
(b) per-token position ids, (c) an id-tagged paged KV arena with a block table
replacing the dense/ring caches, and (d) sequence-restricted varlen attention
that reuses flash + merge. This is "varlen" attention: one call carries N
independent sequences of mixed prefill/decode lengths packed on the token axis.

### 4.0 New input contract (scheduler → forward)

Introduce a `BatchMeta` carried alongside `token_ids`, all per-token where
possible (small, read-frequent → replicate per NUMA node like the rope tables):

- `total_tokens: Int` — replaces `total_len`/`chunk_len` as packed token count.
- `num_seqs: Int`.
- `cu_seqlens: Span[Int32]` — `[num_seqs+1]` prefix sums of query lengths
  (packed offsets); seq `s` query tokens are `[cu_seqlens[s], cu_seqlens[s+1])`.
- `seq_id_of_token: Span[Int32]` — `[total_tokens]` membership (derivable from
  `cu_seqlens`; precompute once, reuse across all 30 layers).
- `position_ids: Span[Int32]` — `[total_tokens]` absolute position within each
  token's own sequence (replaces `base_pos + tok`). Drives RoPE and the causal
  cutoff.
- `kv_block_table: Span[Int32]` — `[num_seqs * max_blocks_per_seq]` mapping
  (seq, logical block) → physical `block_id` in the id-tagged arena. The paging
  indirection. Separate tables for sliding vs full layers (different cache
  geometry), or one with two stride regions.
- `context_len: Span[Int32]` — `[num_seqs]` KV length already resident per
  sequence (replaces `kv_len = base_pos + seq_len`).
- `kv_write_slot: Span[Int32]` — `[total_tokens]` the absolute KV slot each new
  token writes (= `position_ids` in simple append).

`Slot`/`SlotGroup` in `modeling/slot.mojo` is the weight-binding abstraction and
should **not** be overloaded for this; `BatchMeta` is a new per-call struct.

#### forward signature: before / after

Before (`gemma_4_moe.mojo:859-865`):
```mojo
def forward[tok_origin: ImmutOrigin, //](
    mut self, token_ids: Span[Int32, tok_origin], base_pos: Int,
) -> TemporalLogitsView[C.VOCAB_SIZE]
```
After (sketch):
```mojo
def forward[tok_origin: ImmutOrigin, //](
    mut self,
    token_ids: Span[Int32, tok_origin],     # [total_tokens] packed
    meta: BatchMeta,                          # cu_seqlens, position_ids, seq_id_of_token,
                                              #  kv_block_table, context_len, kv_write_slot, num_seqs
) -> TemporalLogitsView[C.VOCAB_SIZE]         # logits for last token per seq (see 4.6)
```
The `while consumed < total_len` chunk loop (`gemma_4_moe.mojo:895-971`) is
replaced by **one pass over `total_tokens`**; the sequence-vs-chunk semantics
move into `meta`. The `pos += chunk_len` bookkeeping (`:971`) is deleted.

### 4.1 KV cache → id-tagged paged block arena

Replace `SlidingKVSlots`/`FullKVSlots` (`gemma_4_moe.mojo:165-174`) and the
position-keyed addressing (`rope.mojo:88-89`) with a paged arena:

- Statically allocate `num_blocks` blocks per (layer-kind, K/V), each
  `block_size * num_kv_heads * head_dim` bf16, in the NUMA arena. `block_size`
  tokens per block.
- `block_id` is the tag; `kv_block_table[seq, logical_block] -> block_id`.
- Addressing helper (replaces the ring `slot = (pos//cd) & mask` and the
  context-shard `pos % degree`): given a per-token KV slot `s_abs`,
  `logical_block = s_abs / block_size`, `slot = s_abs % block_size`,
  `block_id = kv_block_table[seq*max_blocks + logical_block]`, then
  `k_ptr = block_base(layer, block_id) + slot*num_kv*head_dim + kv_h*head_dim`.
  Keep `head_dim` innermost so the flash SIMD loads in
  `flash_attention_chunk` / `flash_prefill_block` are unchanged.
- The 2W sliding ring's `pos mod 2W` aliasing (`rope.mojo:89`) goes away;
  sliding is just "only the last `window` logical blocks are live," still
  expressed via the block table (scheduler frees old blocks).
- **NUMA**: the existing full-attention design context-shards by `pos % degree`
  (`rope.mojo:88`) and merges partials. With paging, the scheduler chooses
  which node a `block_id` lives on. Two viable mappings: (a) keep round-robin
  position→node so the merge machinery (`logsum_merge.mojo`) is reused
  unchanged, blocks just become the unit; or (b) pin a sequence's blocks to one
  node and skip the cross-node merge for that sequence. (a) preserves the most
  code; (b) is more local. See section 5.8.

### 4.2 Attention kernels: varlen + paged gather + membership

Both `dispatch_sliding_attention_qkv` (`gemma_4_moe.mojo:428`) and
`dispatch_full_attention_qkv` (`:519`) currently take `(base_pos, seq_len)`.
They must take `(meta)` and, per query token, resolve `seq`, `pos`, `context_len`.

- **SlidingAttentionKernel** (`attention_dispatch_kernels.mojo:20-59`):
  - `q_pos = base_pos + tok` (`:43`) → `q_pos = position_ids[tok]`.
  - `window_start = max(0, q_pos - window + 1)` (`:44`) stays, but keys are now
    fetched **through the block table for `seq = seq_id_of_token[tok]`**, not
    the shared ring. `flash_attention_chunk` (`flash_attention.mojo:8-20`) gains
    a block-table + block_size params and gathers each key's block pointer
    (hoist block_id resolution out of the inner `head_dim` loop; iterate keys
    block-by-block so the inner SIMD dot is unchanged).
  - Membership is enforced **structurally**: you only ever walk `seq`'s own
    blocks, so cross-sequence keys are never loaded. No additive mask needed.
- **FullAttentionKernel** (`attention_dispatch_kernels.mojo:99+`) +
  `flash_prefill_block` (`flash_attention_prefill.mojo`):
  - `causal_offset` (`flash_attention_prefill.mojo:19`) currently encodes the
    single `base_pos` anchor; it becomes per-query `position_ids[q_row]`.
  - `[kv_start, kv_end)` ranges become per-sequence: a query of seq `s` ranges
    over `[0, context_len[s] + (local query offset))` of seq `s`'s blocks only.
  - The `MergeSegment`/`merge_partials` log-sum-exp merge
    (`logsum_merge.mojo`) is reused unchanged per (query, head): it already
    merges partials from disjoint KV ranges — now those ranges are this
    sequence's blocks instead of the whole context.
- **Prefill+decode mixed in one batch**: a decode token is a query token with
  `pos = context_len[seq]` (length-1 query for its seq); a prefill token is one
  of many for its seq. Both handled by the same per-token loop with its own
  `q_pos`. There is **no `if seq_len>1` branch to remove** (the current code
  already loops per token), which is favorable — the change is in *where keys
  come from* and *what the cutoff is*, not in adding a new control path.

### 4.3 RoPE / positions

`RopeCacheWriteKernel.execute` (`rope.mojo:75-101`) computes `pos = base_pos +
tok` (`:79`). Change to `pos = position_ids[tok]` (pass `position_ids` and
`seq_id_of_token` into the kernel; drop `base_pos`). The cos/sin table indexing
(`:80-81`) and the SIMD `rope_head`/`rope_head_to` (`:28-47`) are unchanged.
The **fused KV write** in the same kernel (`:88-100`) must switch from
ring/context-shard addressing to the block-table addressing of 4.1: compute
`s_abs = kv_write_slot[tok]`, resolve `(block_id, slot)`, write there. The
per-rank ownership test `pos % cache_degree == rank` (`:88`) is replaced by
"this rank owns `block_id`" (from the block table / scheduler placement). The V
`memcpy` (`:100`) and K `rope_head_to` (`:92-96`) stay vectorized.

### 4.4 KV write path

Already unified and fused into RoPE (section 2/4.3). The only modeling change is
the **address computation**: per new token, `seq = seq_id_of_token[tok]`,
`s_abs = kv_write_slot[tok]`, `logical_block = s_abs / block_size`, `slot = s_abs
% block_size`, `block_id = kv_block_table[seq*max_blocks + logical_block]`, write
each `kv_h`'s `head_dim` run into that block. **Block-boundary crossing** is
inherent-safe: each token resolves its own block independently and a single
head's `head_dim` run is contiguous within one block; the scheduler guarantees
the block exists (reserving places it under the id). Keep the K/V copies
vectorized over `head_dim` exactly as today (`rope.mojo:92-100`).

### 4.5 MoE router / expert FFN

The MoE pipeline is **token-parallel and already batching-agnostic** — confirmed
across `dispatch_moe` (`gemma_4_moe.mojo:609-682`):

- `dispatch_router_expert` / `dispatch_merge_router_candidates`
  (`:639-651`): per-token routing, no cross-token reduction. Operate over
  `total_tokens`.
- `dispatch_build_expert_schedules` (`:659-662`): builds per-expert offset/route
  lists by **grouping tokens by chosen expert** — this is token-choice routing
  with **no capacity cap and no padding** (it sorts tokens into expert buckets;
  `moe_expert_offset` is `[NUM_EXPERTS+1]`, `moe_routes` is `[SLIDING_WINDOW *
  TOP_K]`). Packing many sequences is transparent: tokens are just tokens.
- `dispatch_phase1_gate_up` / `dispatch_phase2_down` (`:664-678`): grouped
  GEMMs over the bucketed tokens; no per-sequence assumption.
- `dispatch_allreduce_inplace` over `seq_len * HIDDEN` (`:680-682`): the NUMA
  collective; size becomes `total_tokens * HIDDEN`.

Only change: the `seq_len` argument everywhere becomes `total_tokens`, and the
scratch sizing (`SLIDING_WINDOW * ...` in `Gemma4FfnMoeScratch`,
`gemma_4_moe.mojo:281-324`) must be bounded by `max_tokens_per_batch` rather
than `SLIDING_WINDOW`. Per-worker scratch (`moe_gate_scratch`
PER_WORKER, `:316-318`) is already per-worker — no race. No expert-capacity
logic exists to break.

### 4.6 Norms / embedding / logits

All token-parallel and **safe** with packed tokens; only the row count changes
`chunk_len → total_tokens`:

- `dispatch_embed_lookup` (`gemma_4_moe.mojo:904-908`): per-token gather, safe;
  it is vocab-sharded + allreduced, which is orthogonal to batching.
- `dispatch_rms_norm` / `fused_norm_residual_add` (`:924-928`, `:946-950`,
  `dispatch_ffn` `:710-760`): per-row over `HIDDEN`; **no cross-token reduction**
  → packing transparent.
- `dispatch_scalar_mul`, `dispatch_gelu_gate_up` (`:724-726`, `:758-760`):
  elementwise, safe.
- **Logits**: today only the **last token of the last chunk** gets logits
  (`gemma_4_moe.mojo:956-968`). Under continuous batching this must become
  **one logits row per sequence's last query token**: gather rows
  `cu_seqlens[s+1]-1` for each `s`, run final norm + `dispatch_gemv_softcap`
  on those `num_seqs` rows, and return `num_seqs` logit vectors (the return type
  `TemporalLogitsView[VOCAB]` (`temporal_scratch.mojo:267`) becomes
  `[num_seqs, vocab_per_rank]`). This is the single biggest tail-path change.

### 4.7 Position/offset bookkeeping to move out of the model

- `base_pos` forward arg (`gemma_4_moe.mojo:864`) and the `pos`/`consumed` chunk
  cursor (`:893-894, 970-971`) → `position_ids[tok]` + `context_len[seq]`.
- `q_pos = base_pos + tok` (`attention_dispatch_kernels.mojo:43`) →
  `position_ids[tok]`.
- `pos = base_pos + tok` in RoPE/write (`rope.mojo:79`) → `position_ids[tok]`
  for RoPE, `kv_write_slot[tok]` for the write address.
- `causal_offset` (`flash_attention_prefill.mojo:19`) → per-query position.
- Ring/context-shard slot math (`rope.mojo:88-89`) → block-table lookup.

### 4.8 NUMA implications of the indirection

- The new `kv_block_table`, `position_ids`, `seq_id_of_token`, `context_len` are
  **hot, read-only, every-layer** structures. Replicate them per NUMA node (like
  the rope tables, `gemma_4_moe.mojo:107-108`) so attention workers read locally.
  They are small.
- The **block-pointer gather** in flash is the new remote-read risk. The
  existing full-attention design already context-shards by `pos % degree`
  (`rope.mojo:88`) and each rank flashes its *local* positions then merges —
  i.e. KV reads are already local-by-construction. Preserve this by mapping
  `block_id → node` consistently (4.1 option (a)): a rank only flashes blocks it
  owns, then `merge_partials` combines across nodes. This keeps the hot KV read
  **local** (a remote-read, never a remote-write) per the project principle. The
  cross-node combine is a small log-sum-exp merge of partials, not a KV-sized
  transfer.
- KV writes (`rope.mojo:88-100`) must stay node-local: write the token into a
  block owned by the writing rank. With paging, the scheduler should reserve a
  token's block on the node whose worker will write/read it — keeping writes
  local (most expensive op to make remote) and reads local.
- The BurstPool model reinforces this: each worker mailbox is on the worker's own
  NUMA node — "Worker reads locally. Dispatcher writes remotely."
  (`threading/burst_threading.mojo:17-23`). Dispatch is already per-node
  (`forward` loops `for r in range(len(pools))` issuing to each rank's pool,
  e.g. `gemma_4_moe.mojo:300-305` in `fanout_dispatch`). So the existing pattern
  is "one pool per NUMA node, each touches its own arena"; the paged KV mapping
  must keep a sequence's blocks on the node whose pool flashes them, so this
  invariant is preserved.

---

## 5) RISKS / SHARP EDGES

1. **Scratch + activation buffers are `SLIDING_WINDOW`-shaped.** `x_main`,
   `x_residual` are `[SLIDING_WINDOW, HIDDEN]` (`gemma_4_moe.mojo:183-184`);
   every scratch buffer in `Gemma4SlidingScratch`/`Gemma4FullScratch`/
   `Gemma4FfnMoeScratch` is `SLIDING_WINDOW * ...` (`:226-324`). The current
   contract caps a chunk at `SLIDING_WINDOW=1024` (`:897`, debug_assert `:459`).
   A packed batch needs these sized to `max_tokens_per_batch`. The scratch
   planner (`temporal_scratch.mojo:108-185`) is parameterized by `(degree,
   workers)` only; it needs a `max_tokens` axis so `derive_scratch_plan` and
   `calculate_peak_scratch` (`gemma_4_moe.mojo:343-344`) reserve enough arena.

2. **`max_workers <= SLIDING_WINDOW` invariant** (`gemma_4_moe.mojo:354-357`):
   full-attention partials are sized assuming `workers <= SLIDING_WINDOW`. Fine
   as long as the per-batch token cap stays ≥ workers, but the relationship must
   be re-derived against `max_tokens_per_batch`.

3. **Sliding ring aliasing across sequences.** `slot = pos mod 2W`
   (`rope.mojo:89`) is correct only for one sequence. Under packing, two
   sequences at the same `pos mod 2W` would collide. Paging (per-seq block
   table, 4.1) is mandatory for sliding layers, not just full.

4. **Causality is single-anchor positional, not membership-aware.** Sliding
   `window_start` (`attention_dispatch_kernels.mojo:44`) and full `causal_offset`
   (`flash_attention_prefill.mojo:19`) both assume one `base_pos`. They must
   become per-query `position_ids` AND key iteration restricted to the query's
   own sequence blocks; otherwise sequences leak into each other's attention.

5. **Logits tail produces exactly one row** (`gemma_4_moe.mojo:956-968`). With
   N sequences you must produce N rows (last token per seq via `cu_seqlens`).
   The return type `TemporalLogitsView` (`temporal_scratch.mojo:267-310`) and
   the `Gemma4HeadScratch.logits` buffer sized `[VOCAB_SIZE]`
   (`gemma_4_moe.mojo:331-332`) must grow to `[num_seqs, vocab_per_rank]`.

6. **SIMD/vectorization tail with ragged counts.** Kernels iterate tokens with
   `for tok in range(start,end)` and vectorize the inner `head_dim`/`HIDDEN`
   (e.g. `rope_head` over `W` at `rope.mojo:32-34`; flash inner loops; norms use
   `vectorize[W](hidden, step)` at `rmsnorm.mojo:35,66`). The token axis is
   scalar-stepped, so ragged `total_tokens` is fine **as long as tokens stay
   densely packed** (no per-sequence padding). If the scheduler pads to a
   rectangle, pad rows waste lanes and—worse—must be excluded from KV writes and
   logits. Recommendation: densely packed varlen, never padded. The hot flash dot
   `dot_to_scalar[cols]` uses the **comptime-`cols`** path `bf16_panel_dot`
   (`dot_products.mojo:38-56, 181-191`) which assumes `cols % (PU*BW) == 0` — true
   for `HEAD_DIM` 256/512 but it has **no scalar tail**; any non-aligned per-head
   length would silently drop lanes. (The runtime-`cols` variant
   `bf16_panel_dot_runtime` at `:60-104` DOES have BW + scalar tails — used by the
   column-sharded matmuls.) Keep `head_dim` and the paged block per-head stride
   SIMD-multiple and 64B-aligned (`SCRATCH_ALIGNMENT`/`DEFAULT_ALIGNMENT=64`,
   `temporal_scratch.mojo:11`, `model_spec.mojo:19`; arena is mbind+first-touch,
   `numa/arena.mojo:6-24`).

7. **Block size vs cache geometry.** Sliding uses `KV_DIM_SLIDING=2048`
   (8 kv heads × 256), full uses `KV_DIM_FULL=1024` (2 kv heads × 512). Block
   layout must keep each kv-head's `head_dim` run contiguous and aligned for the
   flash `load[width=W]`. A `block_size` that is a power of two and a multiple of
   SIMD width avoids unaligned gathers.

8. **NUMA: don't turn the context shard into remote reads/writes.** The full
   path's correctness today depends on `pos % degree` ownership
   (`rope.mojo:88`) so each rank reads/writes only its own positions and merges
   partials (`logsum_merge.mojo`). The paged block→node mapping must preserve
   "a rank only touches blocks it owns" or the hot KV read/write becomes
   cross-domain — the exact thing the project forbids (remote-writes especially).
   The merge of partials across nodes (small) is acceptable; bulk KV traffic is
   not.

9. **Allreduce sizes are token-count dependent.** Every
   `dispatch_allreduce_inplace` uses `chunk_len * HIDDEN` / `seq_len * HIDDEN`
   (`gemma_4_moe.mojo:909-910, 943-944, 680-682, 735-737`). These become
   `total_tokens * HIDDEN`; the collective itself is batching-agnostic but the
   size argument must track the packed count. Embedding + attention-out + MoE-out
   allreduces are per-token and safe.

10. **`Slot` is not the per-sequence record.** `modeling/slot.mojo` is the
    weight/state offset-binding abstraction (no `seq_id`); do not retrofit
    continuous-batching state into it. `BatchMeta` is new and orthogonal.
