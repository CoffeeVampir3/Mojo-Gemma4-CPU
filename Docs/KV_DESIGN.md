# Paged KV for Continuous Batching — Implemented Design

Both KV caches are paged from per-kind pools carved out of the existing
symmetric per-rank arenas. Sequences are tagged with a dense slot id; one
rank-independent page table maps `(seq_id, ordinal)` to a physical page for
full attention, and each sequence pins exactly two sliding pages for its life.
Pages are persistent: pinned for the sequence's life, returned only on release.
No copying, no migration, no per-step reclamation.

Everything below is implemented and verified (`test_kv_paging.mojo`,
`prototypes/kv_metadata_proto.mojo`, `test_gemma4.mojo`).

## 1. Two pools, one mechanism

|  | Full attention | Sliding attention |
|---|---|---|
| Brick shape (per layer) | `ContextRowSharded[batching_seq_len, KV_DIM_FULL]` | `TensorColumnSharded[batching_seq_len, KV_DIM_SLIDING]` |
| Shard axis | position (`pos % degree`) | feature/head slice |
| Pages per sequence | grows with length (`len / PAGE_LEN`) | exactly 2, greedy at admission |
| Policy | growing table (`page_mask = -1`) | 2-page flip-flop ring (`page_mask = 1`) |

`PAGE_LEN = SLIDING_WINDOW`. The sliding 2W ring became the two-page
flip-flop: `slot(p) = base_rows[(p >> shift) & 1] + (p & (PAGE_LEN - 1))`,
with `base_rows[ordinal & 1]` — page 0 holds even ordinals, page 1 odd,
recycled in place forever. Position aliasing modulo 2W is unchanged, so the
in-chunk overwrite safety argument (chunk ≤ W per run) carries over verbatim.
The old single-slab ring is the degenerate one-page case of the same policy.

The caches are disjoint tensors produced by disjoint layers; nothing ever
copies between them. The only bytes that would ever move between allocations
are future lifecycle transplants (prefix restore/fork), which are cold-path
and enabled — not implemented — by this design.

## 2. The two knobs

- `max_seq_len` — logical per-sequence ceiling: rope tables,
  `max_pages_per_seq`, the `forward` bound.
- `batching_seq_len` — physical capacity: rows of every KV brick, both kinds.
  `num_pages = batching_seq_len / PAGE_LEN` per pool;
  `max_seqs = batching_seq_len / 2W` (exactly the sliding pool's capacity at
  two pages per sequence).

## 3. The data plane: one policy struct

`KVSlot.slot()` is an instance method; `LinearKV`/`RingKV` remain as
zero-state policies. The model paths use `PagedKV` everywhere
(`kernels/attention_ops.mojo`):

```
PagedKV { base_rows: UnsafePointer[Int32], shift, row_mask, page_mask }
slot(start_pos, t) = base_rows[((start_pos+t) >> shift) & page_mask]
                   + ((start_pos+t) & row_mask)
```

- Full attention (input = local row): `shift = ctz(PAGE_LEN/degree)`,
  `page_mask = -1` (identity AND), `base_rows[o] = page * (PAGE_LEN/degree)`.
  `full_local_kv_count` is reused unchanged: `PAGE_LEN % degree == 0` keeps
  the round-robin ownership pattern identical in every page.
- Sliding (input = absolute position): `shift = ctz(PAGE_LEN)`,
  `page_mask = 1`.

Degree is power-of-two by the existing model contracts (gcd of the sharded
dims is 8), so `PAGE_LEN/degree` is always a power of two; asserted anyway in
`paging_contracts_ok`.

Kernels need contiguity only within a row; `process_kv_tile` resolves every
row independently through the policy value. A tile spanning a page seam is
nothing special — the ring already wrapped mid-tile before paging.

## 4. Run metadata: pointer to orchestrator-owned Lists

Kernel values are memcpy'd into 256-byte worker mailboxes
(`threading/burst_threading.mojo`), so metadata is never inline or
comptime-sized. The packed-path kernels (both prefill kernels and
`RopeCacheWriteKernel`) carry one `UnsafePointer[KVRunTable]` and walk runs
with a monotone cursor:

```
KVRun      { buf_start, base_pos, base_rows: List[Int32] }
KVRunTable { runs: List[KVRun] }
while t >= runs[r+1].buf_start: r += 1
```

`base_rows.unsafe_ptr()` is hoisted per cursor advance and becomes the
policy's pointer. Decode stays scalar and sequence-pure: the policy value is
built at dispatch from `runs[0]`.

Validated by `prototypes/kv_metadata_proto.mojo` on the real burst pools:
workers read nested host Lists in place, reallocation between bursts is safe,
and the per-cycle cost upper bound is ~360ns against a ~730ns empty
dispatch/join cycle.

Snapshot coherence costs nothing: the tables have one writer (the
orchestrator, between joins) and are rebuilt per chunk; nothing mutates them
with work in flight. Page maps are per-(sequence, kind) — identical for every
layer of a kind — so one table per kind serves all 30 layers of a step.

## 5. Control plane: `KVRuntime` (`modeling/kv_runtime.mojo`)

One allocator + one table per pool, refcounted from day one (refcount ≡ 1 is
the v1 single-owner discipline; `retain` exists so prefix sharing later only
changes the release count). Admission is atomic: `admit()` greedily acquires
the two sliding pages, so the only mid-life allocation is full pages,
reserved by `forward` before each chunk's dispatch (`reserve_full` returning
False is the back-pressure signal; kernels never see an unmapped ordinal).
`release` returns everything; no zeroing — readers are bounded by the new
owner's own counts, so stale bytes are unreachable.

## 6. Writes

The slot policy is threaded through `RopeCacheWriteKernel`
(`slot = kv.slot(0, pos // cache_degree)`), which deletes every chunk
alignment rule: chunks may be any size ≤ W, resume after partial chunks, and
straddle page boundaries freely. Rank `pos % degree` writes the position for
full KV; sliding keeps `cache_degree = 1` into the sequence's two pages.

## 7. Invariant index

| ID | Invariant | Enforced at |
|----|-----------|-------------|
| S1 | A page index resolves to the same row range in every brick of its pool, on every rank | identical per-layer brick layout via `stamp_offsets` |
| S2 | One allocator + one rank-independent table per pool | `KVRuntime` is the only owner |
| P1 | `PAGE_LEN % degree == 0` and `PAGE_LEN/degree` pow2 | `paging_contracts_ok` |
| P2 | `batching_seq_len % 2W == 0`, `max_seq_len % PAGE_LEN == 0`, `batching_seq_len >= max_seq_len` | `paging_contracts_ok` |
| L1 | Positions are linear rows within a page; kernels need contiguity only within a row | `process_kv_tile` resolves rows independently |
| L2 | No metadata in KV payload; identity lives in table + runtime | by construction |
| R1 | Kernels never see an unmapped ordinal: a step's pages are reserved before its write phase dispatches | `forward` calls `reserve_full` before chunk dispatch |
| R2 | Kernels consume single resolved units (decode) or the packed buffer + immutable run table (prefill); the mutable runtime never enters a kernel | dispatch layer |
| W1/W2 | Writes are single-row page-local, sequential per sequence | rope kernel + policy |
| SL1' | Sliding ring layout is position-absolute across its 2 pages; aliasing identical to the 2W slab; chunk ≤ W per run | by construction; `debug_assert` in `dispatch_sliding_attention_qkv` |
| IM | A filled full page is immutable until released | W2 + release discipline; enables future refcount sharing |
| EV | Release only with no work in flight; no zeroing required | step-synchronous bursts |
| M1 | Run tables are never mutated with a burst in flight; `base_rows` pointers are stable for a burst's life | orchestrator protocol |

## 8. Deferred (next passes)

- Scheduler: batching policy, packed multi-run steps (kernels and dispatch
  already consume run tables; `forward` builds 1-run tables), mixed
  prefill+decode chunks, per-run logits gather, B>1 decode partial-slot
  allocation under the 128-item burst budget.
- Weighted (cost-proportional) worker partitioning for packed prefill.
- Page-aligned worker split rounding; next-page software prefetch.
- Prefix sharing/transplant: prerequisites landed (refcounts,
  position-absolute ring, filled-page immutability). A sliding transplant is
  a raw two-page copy per sliding brick; full pages share by refcount.
- Admission-time prefault (`load` still prefaults the whole state region);
  eviction decommit.
