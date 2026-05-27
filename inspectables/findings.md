# Butterquant kernel codegen investigation — findings

Method: each kernel reproduced self-contained in `inspectables/bq_*.mojo`; targets exposed as
`@no_inline @export def probe_*` and disassembled with
`objdump -d -M intel` after `pixi run mojo build -D ASSERT=none --march={alderlake,sapphirerapids}
-g0 -O3`. Alderlake binaries are also run (a `SIGILL` would expose a mis-lowered AVX-512 op). Build +
disasm via `./inspect.fish <file>.mojo <syms...>`.

`compile_info[..., emission_kind="asm"]` was attempted first but fails in this toolchain's
`kgen.compile_offload` fold pass for these targets, so all asm comes from the ELF + objdump route.

Arch facts: Alderlake = AVX2 + AVX-VNNI, **no AVX-512**, 16 YMM, `WI=simd_width_of[int32]()=8`,
`acc_count=VNNI_N_STEP/WI=4`. SapphireRapids = AVX-512 + VNNI-512, 32 ZMM, `WI=16`, `acc_count=2`.

Legend: ✅ no issue / intent confirmed · ⚠️ confirmed issue · 🔧 B variant improves it.

---

## Executive summary

The VNNI math, broadcasts, FWHT shuffles, and f32→i8 packing are all **lowering exactly as intended**
— no mis-selected intrinsics, no accidental scalarization in the arithmetic, no lane-correctness
landmine. On **SapphireRapids the whole module is essentially optimal**: every kernel inspected is
spill-free and uses `vpdpbusd zmm` + `vcvtneps2bf16`. The real issues are **Alderlake-specific** and
fall into three buckets:

1. **🔧 f32→bf16 stores scalarize into per-element `__truncsfbf2` libcalls (highest leverage,
   module-wide).** Alderlake lacks AVX-512-BF16, so every `res.cast[bfloat16]()` store becomes 32–128
   libcalls per panel (GEMM epilogues §2/§5, `scale_cast_row` §6, `bake_split_gain` §6). A vectorized
   round-to-nearest-even helper (`store_bf16_rne`, demonstrated in §2) removes **all** calls and cuts
   the epilogue ~2.9×. Single shared fix, big win.

2. **⚠️ register pressure on the 16-YMM ISA.** Production `MR=4` per-row GEMM spills ~7 accumulators
   every iteration (§2); per-block GEMM spills even at PR=1 (§5); register-resident FWHT (§3) and the
   fused head-prep (§4) spill at block/head_dim ≥ 128. Fix is arch-aware capping: `MR≤2` per-row,
   `PR=1` per-block on AVX2, and a memory-staged transform for large FWHT/head_dim. All free on AVX-512.

3. **⚠️ minor / cold.** `vnni_shifted_dot`'s per-iteration horizontal reduce on the LM-head path (§1,
   ~4× inner-loop bloat, 🔧 fixed with a second `vpdpbusd`); `non_temporal` weight loads are a codegen
   no-op (§2, dead hint); `fwht_rotate_columns` offline scalar transpose (§7, cold).

Build/inspect any file with `./inspect.fish <file>.mojo <probe_syms...>`.

---

## 1. VNNI dot core — `inspectables/bq_vnni_dot.mojo`
Covers `vpdpbusd`, `dot_loaded`, `act_broadcast_vnni`, `vnni_shifted_dot`, `i8_vnni_block_dot`,
`head_logit_row` (`butterquant/dot_products.mojo`).

### ✅ `vpdpbusd` intrinsic lowering (refutes the AVX-512-on-Alderlake correctness worry)
The hardcoded `llvm.x86.avx512.vpdpbusd.{width*32}` lowers correctly on **both** arches and the
Alderlake binary runs (no `SIGILL`, `has_vnni: True`):
- Alderlake (`WI=8`): `{vex} vpdpbusd ymm0,ymm1,ymm2` — VEX-encoded AVX-VNNI, exactly right.
- SapphireRapids (`WI=16`): `vpdpbusd zmm0,zmm1,zmm2` — EVEX AVX-512-VNNI.
`dot_loaded` is identical. **No fix needed.** (The missing `has_vnni()` assert on the gemm path is
therefore not a latent mis-lowering on these two targets — but see note below.)

### ✅ `act_broadcast_vnni` 4-byte lane replication — already optimal
The `comptime for lane in range(width): out.insert[offset=lane*4](b4)` chain is recognized by LLVM
and collapses to a **single broadcast**, not N inserts:
- Alderlake: `vmovss` + `vxorps` (0x80 folded as a const load) + `vbroadcastss ymm,xmm`.
- SPR: same with `vbroadcastss zmm`.
My B variant (explicit `.join` doubling) produces byte-identical code. **No fix needed.**

### ⚠️🔧 `vnni_shifted_dot[..., emit_rhs_sum=True]` — per-iteration horizontal reduce in the hot loop
Used by `i8_vnni_block_dot` → `head_logit_row` (LM-head logits). `rhs_sum += bv.cast[i32]().reduce_add()`
runs **every** k-iteration. Steady-state inner loop, Alderlake (A):
```
vpxor / {vex} vpdpbusd ymm0   <- the actual dot (2 insns)
vpbroadcastq; vpmovsxbd; vpbroadcastq; vpmovsxbd; vpaddd;     <- per-iter reduction tree
vpmovsxbd; vpmovsxbd; vpaddd; vpaddd; vextracti128; vpaddd;
vpshufd; vpaddd; vmovd r8d; add; vpextrd; add                 <- cross-domain GPR accumulate
```
≈16 instructions/iter, only 1 of which is the dot; the `vmovd→GPR→add` bounce serializes each step.

**B** accumulates the rhs sum with a second `vpdpbusd` against a ones-vector and reduces **once after
the loop**. Steady-state inner loop becomes **4 instructions** on both arches:
```
vpxor ymm4,...; vmovdqu ymm5,[weights]; {vex} vpdpbusd ymm0,ymm4,ymm5; {vex} vpdpbusd ymm1,ymm3,ymm5
```
Two independent accumulator chains, no GPR bounce. ~4× fewer inner-loop instructions.
**Recommendation:** for the `emit_rhs_sum=True` path, accumulate rhs into a SIMD vector via
`vpdpbusd(ones, bv)` and reduce after the loop. (Or precompute the weight colsum offline as the gemm
path already does, eliminating the runtime sum entirely.)

Note to verify later: the gemm `dot_loaded` path does not assert `has_vnni()`; harmless on these two
targets but would silently mis-lower on a non-VNNI `--march`. Low priority.

---

## 2. VNNI GEMM accumulation — `inspectables/bq_gemm_tiles.mojo`
Covers `accumulate_tiles`, `accumulate_n_step`, `gemm_i8_per_row_panel` (`butterquant/gemm.mojo`),
N=4096, K=512, PR sweep {1,2,4}.

### ✅ accumulate loop is clean at PR≤2; ⚠️ PR=4 spills accumulators on Alderlake
The `ks`/`dc` loops stay rolled; the inner dot body holds `acc` (PR*acc_count i32 vectors) live.
- PR=1 (4 accs): loop body is `vpbroadcastd` + 4× `{vex} vpdpbusd ymm0/3/4/5`, weights folded as
  memory operands, **zero stack traffic**.
- PR=2 (8 accs): accumulators in `ymm0,6,7,8,9,10,11,12`, broadcasts `ymm2/3`, weights `ymm4/5` —
  still **spill-free**.
- PR=4 (16 accs): exceeds the 16-YMM file. ~7 accumulators spill to stack with a
  load-modify-store **every iteration**:
  `vmovdqu ymm10,[rsp+0xf0]; {vex} vpdpbusd ymm10,...; vmovdqu [rsp+0xf0],ymm10`. The dot chain is
  broken by 14 spill stores + 13 reloads per panel.
On SapphireRapids (32 ZMM, acc_count=2) PR=4 uses only 8 accumulators → no spill.
**Recommendation:** the per-row panel row count (`MR`/`PR`) should be arch-capped — ≤2 on Alderlake
(AVX2/16-YMM), free to go ≥4 on AVX-512. **Production `MR` defaults to 4** everywhere
(`butterquant_kernels/linear.mojo`, `moe.mojo`, `kernels/gemm.mojo`), so this spill is on the live
Alderlake path.

### ✅ `non_temporal=True` weight loads are a codegen no-op here
The `load[..., non_temporal=True]` emits **no `vmovntdqa`** — the `_nt` and `_reg` (non_temporal=False)
probes are byte-identical, and `grep movnt` over the whole binary is 0. On normal write-back memory
LLVM does not emit a streaming load, so the hint neither helps nor hurts; it is dead. (No perf A/B
needed — there is no codegen difference to measure.) If streaming of weights is actually wanted it
would need explicit prefetch/`clflushopt` management, not this attribute.

### ⚠️🔧 f32→bf16 epilogue is scalarized into per-element `__truncsfbf2` libcalls on Alderlake
This is the largest Alderlake finding and is **cross-cutting** (every bf16-output GEMM: linear, qkv,
head, moe-down). The store `res.cast[bfloat16]()` lowers to:
- **SapphireRapids:** a single vectorized `vcvtneps2bf16` per accumulator (2 for PR=1, 8 for PR=4),
  no calls, no stack traffic. Correct and fast.
- **Alderlake (no AVX-512-BF16):** the f32 accumulators are spilled to the stack and each output
  element is converted with a `call __truncsfbf2` (libgcc soft-float). Counts: **32 / 64 / 128**
  calls for PR=1 / 2 / 4, plus ~100–440 stack ops. The whole panel is 248 (PR=1) … 933 (PR=4) insns,
  dominated by the epilogue.

**B (`store_bf16_rne` + `gemm_i8_per_row_panel_bf16`)** does the round-to-nearest-even narrowing in
SIMD (`bits + ((bits>>16)&1) + 0x7FFF` then `>>16`, narrow, store) — fully vectorized
(`vpsrld`/`vpand`/`vpaddd`/narrow), **zero calls**:

| panel | A insns | A libcalls | B insns | B libcalls |
|-------|--------:|-----------:|--------:|-----------:|
| PR=1  | 248     | 32         | 87      | 0          |
| PR=4  | 933     | 128        | 324     | 0          |

~2.9× fewer instructions and the per-element call storm removed. **Recommendation:** route the
bf16 narrowing through an explicit vectorized RNE helper on non-AVX512-BF16 targets (keep
`vcvtneps2bf16` where available). This single change is the highest-leverage Alderlake fix.

---

## 3. Fast Walsh–Hadamard — `inspectables/bq_fwht.mojo`
Covers `fwht_apply`, `fwht_block`, `fwht_width` (`butterquant/fwht.mojo`) at block ∈ {16,64,256}.

### ✅ sign-vector materialization and butterfly shuffles are optimal (refutes issue #8)
For the `stride < width` stages the `sign_buf` InlineArray is **not** a stack round-trip — it is
materialized as a **rodata constant** and loaded directly: `vbroadcastsd ymm,[rip+...]`,
`vbroadcastf128 ymm,[rip+...]`, `vmovaps ymm,[rip+...]` (`vmovss`-to-stack count = 0). Each butterfly
is a single `vshufps`/`vshufpd`/`vpermpd`; each `fma(sign, partner)` is one `vfmadd231ps`. The
cross-register `stride ≥ width` stages are plain `vaddps`/`vsubps`. block=16 → 28 insns, block=64 →
110 insns, **both spill-free** on Alderlake. No fix needed.

### ⚠️ block=256 register-resident FWHT spills on Alderlake (inherent, arch-divergent)
A register-resident FWHT holds `regs = block / simd_width` vectors live across the cross-register
stages (which mix all registers, so they cannot be narrowed). Alderlake f32 width=8:
- block=256 → 32 vectors vs 16 YMM → **811 insns, 358 stack spills**.
- block=128 → 16 vectors = the entire YMM file (borderline; expect light spilling).
SapphireRapids width=16: block=256 → 16 vectors in 32 ZMM → **263 insns, 0 spills**.
This is not a codegen defect — it is the register budget. Relevant only where the FWHT block is large
(notably head_dim FWHT in attention prep, §4). **Recommendation:** on AVX2 keep register-resident
FWHT for block ≤ 64 (≤128 acceptable); for block ≥ 256 a memory-staged / tiled transform avoids the
spill storm. No clean register-only B exists at width=8.

---

## 4. Fused attention head prep — `inspectables/bq_head_prep.mojo`
Covers `head_inv_rms`, `absmax_quantize_head`, `prep_head_qk_i8`, `prep_head_v_i8`
(`butterquant/head_prep.mojo`) at head_dim ∈ {128, 256}. This is the **online** per-token attention
path (`BqAttnPrepKernel`), so it is hot.

### ✅ SapphireRapids — fully register-resident, no spills
The whole load→RMS→γ→RoPE→FWHT→absmax→quantize chain stays in registers: qk_128 = 315 insns,
qk_256 = 601, v_256 = 483, **all with 0 stack ops**. `quantize_i8` is vectorized (no libcalls),
`reduce_add`/`reduce_max` are shuffle-trees, and the only `vmovd`→GPR is the single final qsum scalar
return (not in a loop). Intent confirmed.

### ⚠️ Alderlake — register budget exceeded even at head_dim=128
`regs = head_dim / 8`; plus the live `inv_rms`, per-pair `cos`/`sin`, and `gamma` temporaries.
| probe | insns (ald) | stack ops (ald) | insns (spr) | stack ops (spr) |
|-------|------------:|----------------:|------------:|----------------:|
| qk_128 | 728 | 124 | 315 | 0 |
| qk_256 | 1652 | 490 | 601 | 0 |
| v_256  | 1316 | 445 | 483 | 0 |
head_dim=128 already uses all 16 YMM (16 regs) so the RoPE/γ temporaries spill (124 ops); head_dim=256
(32 regs) spills catastrophically (490 ops). Root cause is identical to §3 (register-resident wide
transform on a 16-register ISA). The reduction and quantize codegen themselves are clean. No clean
register-only B at width=8 — the structural fix is the same as §3 (avoid full register residence for
large head_dim on AVX2). On AVX-512 the kernel is already optimal.

---

## 5. Per-block GEMM — `inspectables/bq_gemm_block.mojo`
Covers `gemm_i8_per_block_panel` (`butterquant/gemm.mojo`), N=4096, K=512, block=128 (nb=4).

### ⚠️ ~2× accumulator pressure vs per-row → Alderlake spills even at PR=1
`facc` (PR*acc_count **f32**) is held live across the whole `nb` loop while `iacc` (PR*acc_count
**i32**) is live inside each block → peak ≈ `2*PR*acc_count` accumulator vectors.
| PR | ald stack ld/st | per-row §2 (ref) | spr stack st |
|----|-----------------|------------------|--------------|
| 1  | 11 / 7          | 0 / 0 (clean)    | 0            |
| 2  | 30 / 22         | 0 / 0 (clean)    | 0            |
| 4  | 76 / 67         | 14 / 13          | 0            |
On Alderlake even PR=1 spills (facc 4 + iacc 4 = 8 vectors + temporaries), where the per-row panel was
clean through PR=2. SPR (32 ZMM) is spill-free through PR=4. **Recommendation:** the per-block path
needs PR=1 on Alderlake, and benefits from reducing dual `facc`/`iacc` residence (e.g. fold the
per-block dequant immediately and avoid keeping a separate f32 bank). Lower priority than the bf16 fix.

### ⚠️🔧 same f32→bf16 epilogue cliff as §2 (32 / 64 / 128 `__truncsfbf2` on Alderlake; `vcvtneps2bf16` on SPR)
The §2 `store_bf16_rne` B applies verbatim here.

---

## 6. Streaming quant / dequant — `inspectables/bq_quant_helpers.mojo`
Covers `quantize_i8` (`simd_math/ops.mojo`), `row_absmax`, `quantize_inv`, `bake_split_gain_in_place`
(`butterquant/kernels.mojo`), `scale_cast_row`, `dequant_weight_row_per_block`
(`butterquant/dequantize.mojo`).

### ✅ `quantize_i8` f32→i8 pack is lane-correct and minimal (refutes the AVX2 pack landmine)
```
vmulps; vroundps; vmaxps; vminps; vcvttps2dq;        <- scale, RNE, clamp, to i32
vextracti128 xmm1,ymm0,0x1; vpackssdw xmm0,xmm0,xmm1; vpacksswb xmm0,xmm0,xmm0
```
The high 128-bit lane is **extracted before** the in-lane `vpackssdw`, so the 8 results stay in
order — no missing-`vpermq` interleave bug. `quantize_inv` shows the same clean pack in its loop.
`row_absmax` (27 insns), `dequant_weight_row_per_block` (43 insns, i8→f32 via sign-extend + `vmulps`)
are fully vectorized with no calls. No fix needed.

### ⚠️🔧 the f32→bf16 cliff is pervasive, not just the GEMM epilogue
Every `.cast[bfloat16]()` store scalarizes on Alderlake:
- `scale_cast_row` (used by `BqEmbedLookupKernel`, runtime embedding): 8 `__truncsfbf2` on Alderlake,
  one `vcvtneps2bf16` per chunk on SPR.
- `bake_split_gain_in_place` (offline γ prep): 8 `__truncsfbf2` on Alderlake (cold path, low priority).
This makes the §2 `store_bf16_rne` fix a **module-wide** recommendation: provide one vectorized RNE
f32→bf16 store helper and use it for all bf16 stores on non-AVX512-BF16 targets.

---

## 7. Offline scalarization — `inspectables/bq_scalar_offline.mojo`
Covers `fwht_rotate_columns` (`butterquant/kernels.mojo`) and the `pack_and_colsum` colsum inner loop
(`butterquant/vnni.mojo`).

### ⚠️ `fwht_rotate_columns` is a scalar strided gather/scatter (cold path, low priority)
The triple-nested loop loads/stores one element at a time: `vmovss xmm0,[r11]` →
`vmovss [r15+r10*4],xmm0` (gather into scratch), then the reverse (scatter back), stride = `cols`
floats. No `vgatherdps`, no wide `vmovups` for the transpose (the 28 `vmovups` in the function are the
inner `fwht_block`, which is vectorized). This is **offline two-sided weight prep**, so it is not on
the hot path — flagged for completeness. A vectorized fix would use a blocked register transpose (like
`transpose_generic`) or `vgatherdps`/`vscatterdps` on AVX-512. Not worth a B at this priority.

### ✅ `pack_and_colsum` colsum inner loop is well vectorized
`vpmovsxbd` (i8→i32) + 4 independent `vpaddd` accumulators reduced at the end — good ILP, no
scalarization. No fix needed. (Note: the surrounding `transpose_generic[int32,16]` packs 16×16 tiles;
on Alderlake a 16-wide int32 SIMD is 2 YMM so this is register-heavy, but it is offline pack — same
low-priority register-budget caveat as §3.)

---

## 8. Runtime activation prep — `inspectables/bq_runtime_prep.mojo`
Covers `prepare_norm_activation` (RMS reduce + γ multiply + `fwht_row` + per-row quantize over an
`InlineArray[Float32, hidden]` work buffer), hidden=4096, block=128 (`butterquant/runtime.mojo`).

### ✅ well-fused, no spurious copies, no bf16 issue
`memcpy`/`memmove`/`rep movs` count = 0 — the `_ = work` keep-alive does **not** materialize a copy of
the 16 KB work buffer. Output is i8 (quantize), so no bf16 cliff (`truncsfbf2 = 0`). RMS reduce, the
`x*fr*g` multiply (`vfma`), FWHT, absmax and quantize are all vectorized, no libcalls.
- SapphireRapids: 222 insns, **0 stack stores** — fully clean.
- Alderlake: 348 insns, 33 stack stores — these are the borderline `fwht_block[128]` register pressure
  (§3, 16 regs = full YMM file) plus the work buffer legitimately living in memory. Not a defect.
The fusion quality is good on both arches; the only Alderlake cost is the §3 FWHT register budget.

---

## 9. Reusable patterns already in `kernels/` and `simd_math/`

The non-butterquant kernels already solve the two structural problems above with established idioms
butterquant should adopt rather than reinvent.

### 🔧 `has_avx512_bf16()` arch gate → the idiomatic fix for the bf16 cliff (§2/§5/§6)
`kernels/dot_products.mojo:9` defines `has_avx512_bf16()` (`CompilationTarget._has_feature["avx512bf16"]`)
and `bf16_pair_dot` uses `comptime if has_avx512_bf16(): <VDPBF16PS intrinsic> else: <deinterleave+fma
fallback>`. The same gate fixes the f32→bf16 *store*. I validated `store_bf16_gated` (one code path):
| arch | result |
|------|--------|
| Alderlake | 0 libcalls, 32 `vpsrld` (vectorized RNE), 324 insns (PR=4 panel) |
| SapphireRapids | 8 `vcvtneps2bf16`, 0 RNE ops, 114 insns |
This is **strictly better than the unconditional-RNE B in §2** — it keeps the native instruction where
the hardware has it. **Recommendation:** add a `store_bf16[width]` next to `bf16_pair_dot` (or reuse it)
and route every butterquant bf16 store through it.

### 🔧 width-aware accumulator pickers → the principled register-pressure fix (§2/§5)
`simd_math/matrixops.mojo` has `pick_port_unroll[width, cols]()` (largest PU∈{1,2,4,8} with
`PU*width | cols`) and `port_unroll_for[count]()` (non-SIMD axis). `kernels/gemm.mojo` composes two
independent knobs — an **M-axis panel `MR`** (register reuse) and a **K-axis `PU`** chosen by
`pick_port_unroll[BW, cols]` (ILP) — each with a `panel=1` tail loop. butterquant's VNNI gemm instead
hardcodes a single `PR * (VNNI_N_STEP/width)` accumulator bank, which is what overflows the 16-YMM file
at PR=4 (§2) and 2× at per-block PR=1 (§5). **Recommendation:** drive the accumulator/panel count from a
`width`-parameterized picker so the bank shrinks on AVX2 (width=8) and grows on AVX-512 (width=16),
instead of a fixed constant. (Note the lever direction: `port_unroll` *adds* accumulators for ILP —
helpful on the 32-ZMM machine, harmful on the spilling 16-YMM one — so it is the picker *discipline*,
not blind unrolling, that transfers.)

### 🔧 tree reductions → cleaner epilogue and the §1 rhs_sum reduce
`tree_merge_accs` (bank → merged **vector**, for cast+store) and `tree_reduce_accs` (bank → **scalar**)
replace ad-hoc `s += accs[...]; s.reduce_add()` loops. `tree_merge_accs` is exactly the shape for a
multi-accumulator bf16 epilogue (merge then one `store_bf16`), and `tree_reduce_accs` is the clean form
for the §1 `vnni_shifted_dot` rhs_sum (accumulate a SIMD bank, tree-reduce once after the loop).
