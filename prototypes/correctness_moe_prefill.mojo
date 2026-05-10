from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from std.math import isnan, isinf

from numa import NumaArena, NumaInfo
from threading import BurstPool
from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from kernels.helpers import (
    OutputPartitionedKernel, DispatchBuffer, NumaPointerArray,
    NumaTypedPointerArray, worker_range,
)
from kernels.moe_router import (
    RouterCandidate, RouterCandidatePtr, SparseRoute, SparseRoutePtr,
    build_expert_schedules,
)
from kernels.moe_experts import (
    Phase1GateUpKernel, Phase2DownKernel,
    dispatch_phase1_gate_up, dispatch_phase2_down,
    PHASE1_TILE_J, PHASE1_MR,
)
from modeling.gemma4_common import Gemma4BaseConfig
from simd_math import pick_port_unroll, tree_reduce_accs
from simd_math.ops import sqrt, gelu_tanh_f32


comptime C = Gemma4BaseConfig
comptime ALIGNMENT = 64
comptime CORR_SEQ = 128
comptime CORR_TP = 1
comptime CORR_LOCAL_EXPERTS = C.NUM_EXPERTS // CORR_TP
comptime ARENA_BYTES = 3 * 1024 * 1024 * 1024
comptime W = simd_width_of[DType.float32]()

comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr  = UnsafePointer[Scalar[DType.float32],  MutAnyOrigin]
comptime I32Ptr  = UnsafePointer[Scalar[DType.int32],    MutAnyOrigin]


def arena_alloc[
    align: Int, //, dtype: DType,
](
    mut arena: NumaArena[alignment=align], count: Int,
) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
    var ptr = arena.alloc[Scalar[dtype]](count)
    if not ptr:
        print("arena alloc failed for", count, "elements")
        return UnsafePointer[Scalar[dtype], MutAnyOrigin].unsafe_dangling()
    return ptr.value()


def arena_alloc_t[
    align: Int, //, T: AnyType,
](
    mut arena: NumaArena[alignment=align], count: Int,
) -> UnsafePointer[T, MutAnyOrigin]:
    var ptr = arena.alloc[T](count)
    if not ptr:
        print("arena alloc failed for", count, "elements")
        return UnsafePointer[T, MutAnyOrigin].unsafe_dangling()
    return ptr.value()


def fill_bf16_small(ptr: BF16Ptr, count: Int, seed: Int):
    for i in range(count):
        var x = (i * 131 + seed * 17) % 257
        ptr[i] = Scalar[DType.bfloat16](Float32(x - 128) * 0.001)


def fill_bf16_unit_scale(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = Scalar[DType.bfloat16](
            1.0 + Float32(i % 13) * 0.001 - 0.006)


def fill_bf16_x_input(ptr: BF16Ptr, count: Int, seed: Int):
    for i in range(count):
        var x = (i * 131 + seed * 17) % 257
        ptr[i] = Scalar[DType.bfloat16](Float32(x - 128) * 0.002)


def fingerprint_f32(ptr: F32Ptr, count: Int) -> Float64:
    var s = Float64(0)
    for i in range(count):
        var v = Float64(ptr[i])
        if v < 0:
            v = -v
        s += v
    return s


def fingerprint_i32(ptr: I32Ptr, count: Int) -> Int:
    var s = 0
    for i in range(count):
        s += Int(ptr[i])
    return s


@fieldwise_init
struct RankState(Copyable, ImplicitlyCopyable):
    var x: BF16Ptr
    var router_proj: BF16Ptr
    var router_scale: BF16Ptr
    var per_expert_scale: BF16Ptr
    var pre_ffn_norm_2: BF16Ptr
    var route_idx: I32Ptr
    var route_w: F32Ptr
    var router_scaled: F32Ptr
    var x_normed: BF16Ptr
    var moe_partial: BF16Ptr
    var cands: RouterCandidatePtr
    var expert_offset: I32Ptr
    var routes: SparseRoutePtr
    var hidden_bucket: BF16Ptr
    var moe_accum: F32Ptr
    var experts_gate_up: BF16Ptr
    var experts_down: BF16Ptr
    var gate_scratch: F32Ptr


@fieldwise_init
struct BaselineSlotState(Copyable, ImplicitlyCopyable):
    var bucket_token_idx: I32Ptr
    var bucket_weight: F32Ptr
    var slot_offset: I32Ptr
    var hidden_scratch: F32Ptr
    var gate_scratch: F32Ptr


@fieldwise_init
struct ExpertRunnerSlotKernel[
    hidden: Int, gate_up_fused: Int, intermediate: Int, num_local_experts: Int,
](OutputPartitionedKernel):
    """Scalar f32 reference kernel: numerical baseline only, single-worker
    slot-phased. Not for production. Used by the correctness harness to
    cross-check the active path against scalar fp32 arithmetic."""
    var x_normed: BF16Ptr
    var expert_offset: I32Ptr
    var bucket_token_idx: I32Ptr
    var bucket_weight: F32Ptr
    var experts_gate_up: BF16Ptr
    var experts_down: BF16Ptr
    var gate_scratch: F32Ptr
    var hidden_scratch: F32Ptr
    var moe_partial: BF16Ptr
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime PU_H = pick_port_unroll[W, Self.hidden]()
        comptime STRIDE_H = PU_H * W
        comptime PU_I = pick_port_unroll[W, Self.intermediate]()
        comptime STRIDE_I = PU_I * W

        var gu_scratch = self.gate_scratch + self.worker_id * Self.gate_up_fused
        var h_scratch = self.hidden_scratch + self.worker_id * Self.intermediate

        for e in range(self.start, self.end):
            var rec_lo = Int(self.expert_offset[e])
            var rec_hi = Int(self.expert_offset[e + 1])
            var gu_w = self.experts_gate_up + e * Self.gate_up_fused * Self.hidden
            var down_w = self.experts_down + e * Self.hidden * Self.intermediate

            for rec in range(rec_lo, rec_hi):
                var tok = Int(self.bucket_token_idx[rec])
                var x_row = self.x_normed + tok * Self.hidden

                for m in range(Self.gate_up_fused):
                    var row = gu_w + m * Self.hidden
                    var accs = InlineArray[SIMD[DType.float32, W], PU_H](
                        fill=SIMD[DType.float32, W](0))
                    for i in range(Self.hidden // STRIDE_H):
                        comptime for p in range(PU_H):
                            var off = i * STRIDE_H + p * W
                            var xv = (x_row + off).load[width=W]().cast[DType.float32]()
                            var wv = (row + off).load[width=W]().cast[DType.float32]()
                            accs[p] = xv.fma(wv, accs[p])
                    gu_scratch[m] = tree_reduce_accs(accs)

                for j in range(0, Self.intermediate, W):
                    var g = (gu_scratch + j).load[width=W]()
                    var u = (gu_scratch + Self.intermediate + j).load[width=W]()
                    (h_scratch + j).store(gelu_tanh_f32[W](g) * u)

                var route_w = SIMD[DType.float32, W](self.bucket_weight[rec])
                var dst = self.moe_partial + tok * Self.hidden
                for m in range(Self.hidden):
                    var row = down_w + m * Self.intermediate
                    var accs = InlineArray[SIMD[DType.float32, W], PU_I](
                        fill=SIMD[DType.float32, W](0))
                    for i in range(Self.intermediate // STRIDE_I):
                        comptime for p in range(PU_I):
                            var off = i * STRIDE_I + p * W
                            var xv = (h_scratch + off).load[width=W]()
                            var wv = (row + off).load[width=W]().cast[DType.float32]()
                            accs[p] = xv.fma(wv, accs[p])
                    var out = tree_reduce_accs(accs)
                    dst[m] = (dst[m].cast[DType.float32]() + out * route_w[0]).cast[DType.bfloat16]()

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(
            self.x_normed, self.expert_offset, self.bucket_token_idx,
            self.bucket_weight, self.experts_gate_up, self.experts_down,
            self.gate_scratch, self.hidden_scratch, self.moe_partial,
            self.worker_id, start, end,
        )


def alloc_state(
    mut arena: NumaArena[alignment=ALIGNMENT], capacity: Int,
) -> Tuple[RankState, BaselineSlotState]:
    var x = arena_alloc[DType.bfloat16](arena, CORR_SEQ * C.HIDDEN)
    var router_proj = arena_alloc[DType.bfloat16](
        arena, CORR_LOCAL_EXPERTS * C.HIDDEN)
    var router_scale = arena_alloc[DType.bfloat16](arena, C.HIDDEN)
    var pes = arena_alloc[DType.bfloat16](arena, C.NUM_EXPERTS)
    var norm_w = arena_alloc[DType.bfloat16](arena, C.HIDDEN)
    var route_idx = arena_alloc[DType.int32](arena, CORR_SEQ * C.TOP_K)
    var route_w = arena_alloc[DType.float32](arena, CORR_SEQ * C.TOP_K)
    var router_scaled = arena_alloc[DType.float32](arena, capacity * C.HIDDEN)
    var x_normed = arena_alloc[DType.bfloat16](arena, CORR_SEQ * C.HIDDEN)
    var partial = arena_alloc[DType.bfloat16](arena, CORR_SEQ * C.HIDDEN)
    var cands = arena_alloc_t[RouterCandidate](arena, CORR_SEQ * C.TOP_K)
    var expert_offset = arena_alloc[DType.int32](arena, CORR_LOCAL_EXPERTS + 1)
    var routes = arena_alloc_t[SparseRoute](arena, CORR_SEQ * C.TOP_K)
    var hidden_bucket = arena_alloc[DType.bfloat16](
        arena, CORR_SEQ * C.TOP_K * C.MOE_INTERMEDIATE)
    var moe_accum = arena_alloc[DType.float32](arena, CORR_SEQ * C.HIDDEN)
    var experts_gu = arena_alloc[DType.bfloat16](
        arena, CORR_LOCAL_EXPERTS * C.MOE_GATE_UP_FUSED * C.HIDDEN)
    var experts_down = arena_alloc[DType.bfloat16](
        arena, CORR_LOCAL_EXPERTS * C.HIDDEN * C.MOE_INTERMEDIATE)
    var gate_scratch = arena_alloc[DType.float32](
        arena, capacity * PHASE1_MR * 2 * PHASE1_TILE_J)

    fill_bf16_x_input(x, CORR_SEQ * C.HIDDEN, 11)
    fill_bf16_small(router_proj, CORR_LOCAL_EXPERTS * C.HIDDEN, 23)
    fill_bf16_unit_scale(router_scale, C.HIDDEN)
    fill_bf16_unit_scale(pes, C.NUM_EXPERTS)
    fill_bf16_unit_scale(norm_w, C.HIDDEN)
    fill_bf16_small(experts_gu,
        CORR_LOCAL_EXPERTS * C.MOE_GATE_UP_FUSED * C.HIDDEN, 41)
    fill_bf16_small(experts_down,
        CORR_LOCAL_EXPERTS * C.HIDDEN * C.MOE_INTERMEDIATE, 59)

    var rank_state = RankState(
        x=x, router_proj=router_proj, router_scale=router_scale,
        per_expert_scale=pes, pre_ffn_norm_2=norm_w,
        route_idx=route_idx, route_w=route_w, router_scaled=router_scaled,
        x_normed=x_normed, moe_partial=partial,
        cands=cands, expert_offset=expert_offset, routes=routes,
        hidden_bucket=hidden_bucket, moe_accum=moe_accum,
        experts_gate_up=experts_gu, experts_down=experts_down,
        gate_scratch=gate_scratch,
    )

    var bucket_idx = arena_alloc[DType.int32](arena, CORR_SEQ)
    var bucket_w = arena_alloc[DType.float32](arena, CORR_SEQ)
    var slot_offset = arena_alloc[DType.int32](arena, CORR_LOCAL_EXPERTS + 1)
    var h_scratch = arena_alloc[DType.float32](arena, C.MOE_INTERMEDIATE)
    var bgu_scratch = arena_alloc[DType.float32](arena, C.MOE_GATE_UP_FUSED)
    var baseline = BaselineSlotState(
        bucket_token_idx=bucket_idx, bucket_weight=bucket_w,
        slot_offset=slot_offset, hidden_scratch=h_scratch,
        gate_scratch=bgu_scratch,
    )

    return (rank_state, baseline)


def numa_one_bf16(ptr: BF16Ptr) -> NumaPointerArray[DType.bfloat16, 1]:
    var bases = InlineArray[Int, 1](uninitialized=True)
    bases[0] = 0
    return NumaPointerArray[DType.bfloat16, 1](ptr, bases)


def numa_one_f32(ptr: F32Ptr) -> NumaPointerArray[DType.float32, 1]:
    var bases = InlineArray[Int, 1](uninitialized=True)
    bases[0] = 0
    return NumaPointerArray[DType.float32, 1](ptr, bases)


def numa_one_i32(ptr: I32Ptr) -> NumaPointerArray[DType.int32, 1]:
    var bases = InlineArray[Int, 1](uninitialized=True)
    bases[0] = 0
    return NumaPointerArray[DType.int32, 1](ptr, bases)


def numa_one_routes(ptr: SparseRoutePtr) -> NumaTypedPointerArray[SparseRoute, 1]:
    var bases = InlineArray[Int, 1](uninitialized=True)
    bases[0] = 0
    return NumaTypedPointerArray[SparseRoute, 1](ptr, bases)


def run_phase1[P: BurstThreadPool](
    state: RankState, mut pools: HeapMoveArray[P],
):
    dispatch_phase1_gate_up[
        hidden=C.HIDDEN, gate_up_fused=C.MOE_GATE_UP_FUSED,
        intermediate=C.MOE_INTERMEDIATE,
        experts_per_rank=CORR_LOCAL_EXPERTS, tp=1,
    ](
        numa_one_bf16(state.x_normed),
        numa_one_i32(state.expert_offset),
        numa_one_routes(state.routes),
        numa_one_bf16(state.experts_gate_up),
        numa_one_f32(state.gate_scratch),
        numa_one_bf16(state.hidden_bucket),
        pools,
    )


def run_phase2[P: BurstThreadPool](
    state: RankState, mut pools: HeapMoveArray[P],
):
    dispatch_phase2_down[
        hidden=C.HIDDEN, intermediate=C.MOE_INTERMEDIATE,
        experts_per_rank=CORR_LOCAL_EXPERTS, tp=1,
    ](
        numa_one_i32(state.expert_offset),
        numa_one_routes(state.routes),
        numa_one_bf16(state.hidden_bucket),
        numa_one_bf16(state.experts_down),
        numa_one_f32(state.moe_accum),
        numa_one_bf16(state.moe_partial),
        CORR_SEQ, pools,
    )


def bucketize_slot_inline(
    route_idx: I32Ptr, route_w: F32Ptr,
    slot: Int, num_local_experts: Int, seq_len: Int,
    expert_offset: I32Ptr,
    bucket_token_idx: I32Ptr, bucket_weight: F32Ptr,
):
    var counts = InlineArray[Int32, CORR_LOCAL_EXPERTS](fill=Int32(0))
    var rank = 0
    var first = rank * num_local_experts
    var last = first + num_local_experts

    for tok in range(seq_len):
        var e = Int(route_idx[tok * C.TOP_K + slot])
        if e >= first and e < last:
            counts[e - first] += Int32(1)

    var running = Int32(0)
    var write_offsets = InlineArray[Int32, CORR_LOCAL_EXPERTS](
        uninitialized=True)
    for e in range(num_local_experts):
        expert_offset[e] = running
        write_offsets[e] = running
        running += counts[e]
    expert_offset[num_local_experts] = running

    for tok in range(seq_len):
        var e = Int(route_idx[tok * C.TOP_K + slot])
        if e >= first and e < last:
            var local = e - first
            var pos = Int(write_offsets[local])
            bucket_token_idx[pos] = Int32(tok)
            bucket_weight[pos] = route_w[tok * C.TOP_K + slot]
            write_offsets[local] = Int32(pos + 1)


def run_baseline_expert[P: BurstThreadPool](
    state: RankState, baseline: BaselineSlotState, mut pool: P,
):
    var buf = DispatchBuffer[
        ExpertRunnerSlotKernel[
            C.HIDDEN, C.MOE_GATE_UP_FUSED, C.MOE_INTERMEDIATE,
            CORR_LOCAL_EXPERTS,
        ]
    ]()
    buf.slot()[] = ExpertRunnerSlotKernel[
        C.HIDDEN, C.MOE_GATE_UP_FUSED, C.MOE_INTERMEDIATE,
        CORR_LOCAL_EXPERTS,
    ](
        state.x_normed, baseline.slot_offset,
        baseline.bucket_token_idx, baseline.bucket_weight,
        state.experts_gate_up, state.experts_down,
        baseline.gate_scratch, baseline.hidden_scratch,
        state.moe_partial, 0, 0, CORR_LOCAL_EXPERTS,
    )
    buf.dispatch(pool)
    pool.join()


def zero_bf16(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = Scalar[DType.bfloat16](0)


def copy_partial(src: BF16Ptr, dst: BF16Ptr, count: Int):
    for i in range(count):
        dst[i] = src[i]


def compute_ground_truth(
    state: RankState,
    gt_out: BF16Ptr,
    mut arena: NumaArena[alignment=ALIGNMENT],
):
    var acc = arena_alloc[DType.float32](arena, CORR_SEQ * C.HIDDEN)
    var gu_temp = arena_alloc[DType.float32](arena, C.MOE_GATE_UP_FUSED)
    var h_temp = arena_alloc[DType.float32](arena, C.MOE_INTERMEDIATE)

    for i in range(CORR_SEQ * C.HIDDEN):
        acc[i] = Float32(0)

    for t in range(CORR_SEQ):
        var x_row = state.x_normed + t * C.HIDDEN
        for slot in range(C.TOP_K):
            var e = Int(state.route_idx[t * C.TOP_K + slot])
            var w = state.route_w[t * C.TOP_K + slot]
            var gu_w_base = state.experts_gate_up + e * C.MOE_GATE_UP_FUSED * C.HIDDEN
            var down_w_base = state.experts_down + e * C.HIDDEN * C.MOE_INTERMEDIATE

            for m in range(C.MOE_GATE_UP_FUSED):
                var s = Float32(0)
                var w_row = gu_w_base + m * C.HIDDEN
                for h in range(C.HIDDEN):
                    s += x_row[h].cast[DType.float32]() * w_row[h].cast[DType.float32]()
                gu_temp[m] = s

            for j in range(C.MOE_INTERMEDIATE):
                var g_simd = SIMD[DType.float32, 1](gu_temp[j])
                var g_out = gelu_tanh_f32[1](g_simd)
                h_temp[j] = g_out[0] * gu_temp[C.MOE_INTERMEDIATE + j]

            var acc_row = acc + t * C.HIDDEN
            for m in range(C.HIDDEN):
                var s = Float32(0)
                var w_row = down_w_base + m * C.MOE_INTERMEDIATE
                for k in range(C.MOE_INTERMEDIATE):
                    s += h_temp[k] * w_row[k].cast[DType.float32]()
                acc_row[m] += w * s

    for i in range(CORR_SEQ * C.HIDDEN):
        gt_out[i] = Scalar[DType.bfloat16](acc[i])


def fingerprint(ptr: BF16Ptr, count: Int) -> Float64:
    var s = Float64(0)
    for i in range(count):
        var v = Float64(ptr[i].cast[DType.float32]())
        if v < 0:
            v = -v
        s += v
    return s


def compare(active: BF16Ptr, baseline: BF16Ptr, count: Int, tol: Float32) -> Bool:
    var max_abs = Float32(0)
    var max_rel = Float32(0)
    var n_bad = 0
    var n_nan_match = 0
    var n_nan_mismatch = 0
    var sum_signed = Float64(0)
    var sum_abs_diff = Float64(0)
    var sum_active_signed = Float64(0)
    var sum_baseline_signed = Float64(0)
    var n_pos = 0
    var n_neg = 0
    var n_zero = 0
    var bk_n = InlineArray[Int, 4](fill=0)
    var bk_sum_signed = InlineArray[Float64, 4](fill=0.0)
    var bk_sum_abs = InlineArray[Float64, 4](fill=0.0)
    var bk_max_abs = InlineArray[Float32, 4](fill=Float32(0.0))
    var bk_sum_baseline = InlineArray[Float64, 4](fill=0.0)
    for i in range(count):
        var a = active[i].cast[DType.float32]()
        var b = baseline[i].cast[DType.float32]()
        var a_nan = isnan(a) or isinf(a)
        var b_nan = isnan(b) or isinf(b)
        if a_nan and b_nan:
            n_nan_match += 1
            continue
        if a_nan != b_nan:
            n_nan_mismatch += 1
            continue
        var d = a - b
        sum_signed += Float64(d)
        sum_active_signed += Float64(a)
        sum_baseline_signed += Float64(b)
        if d > 0: n_pos += 1
        elif d < 0: n_neg += 1
        else: n_zero += 1
        var diff_abs = d
        if diff_abs < 0:
            diff_abs = -diff_abs
        sum_abs_diff += Float64(diff_abs)
        var b_abs = b
        if b_abs < 0: b_abs = -b_abs
        var ref_mag = b_abs
        if ref_mag < 1.0:
            ref_mag = 1.0
        var rel = diff_abs / ref_mag
        if diff_abs > max_abs:
            max_abs = diff_abs
        if rel > max_rel:
            max_rel = rel
        if rel > tol:
            n_bad += 1
        var bk = 0
        if b_abs >= 0.1: bk = 3
        elif b_abs >= 0.01: bk = 2
        elif b_abs >= 0.001: bk = 1
        bk_n[bk] += 1
        bk_sum_signed[bk] += Float64(d)
        bk_sum_abs[bk] += Float64(diff_abs)
        bk_sum_baseline[bk] += Float64(b_abs)
        if diff_abs > bk_max_abs[bk]:
            bk_max_abs[bk] = diff_abs

    print("  elements:        " + String(count))
    print("  max_abs_err:     " + String(max_abs))
    print("  max_rel_err:     " + String(max_rel))
    print("  n_bad:           " + String(n_bad) + " (tol=" + String(tol) + ")")
    print("  n_nan_match:     " + String(n_nan_match))
    print("  n_nan_mismatch:  " + String(n_nan_mismatch))
    print("  fp_active:       " + String(fingerprint(active, count)))
    print("  fp_baseline:     " + String(fingerprint(baseline, count)))
    print("  sum_active_signed:   " + String(sum_active_signed))
    print("  sum_baseline_signed: " + String(sum_baseline_signed))
    print("  sum_signed_diff:     " + String(sum_signed)
        + "   (mean " + String(sum_signed / Float64(count)) + ")")
    print("  sum_abs_diff:        " + String(sum_abs_diff)
        + "   (mean " + String(sum_abs_diff / Float64(count)) + ")")
    print("  diff sign:  pos=" + String(n_pos) + "  neg=" + String(n_neg) + "  zero=" + String(n_zero))
    print("  bucketed by |baseline|:")
    var bk_lo = InlineArray[Float32, 4](uninitialized=True)
    var bk_hi = InlineArray[Float32, 4](uninitialized=True)
    bk_lo[0] = 0.0; bk_hi[0] = 0.001
    bk_lo[1] = 0.001; bk_hi[1] = 0.01
    bk_lo[2] = 0.01; bk_hi[2] = 0.1
    bk_lo[3] = 0.1; bk_hi[3] = 1e30
    for bk in range(4):
        if bk_n[bk] == 0:
            print("    [" + String(bk_lo[bk]) + ", " + String(bk_hi[bk]) + "): empty")
            continue
        var n_f = Float64(bk_n[bk])
        var mean_signed = bk_sum_signed[bk] / n_f
        var mean_abs = bk_sum_abs[bk] / n_f
        var mean_baseline = bk_sum_baseline[bk] / n_f
        var rel_bias = Float64(0)
        if mean_baseline > 0: rel_bias = mean_signed / mean_baseline * 100.0
        print("    [" + String(bk_lo[bk]) + ", " + String(bk_hi[bk]) + "):"
            + "  n=" + String(bk_n[bk])
            + "  mean_|baseline|=" + String(mean_baseline)
            + "  mean_signed_diff=" + String(mean_signed)
            + "  rel_bias_pct=" + String(rel_bias)
            + "  mean_abs_diff=" + String(mean_abs)
            + "  max_abs=" + String(bk_max_abs[bk]))
    return n_bad == 0 and n_nan_mismatch == 0


def run_pipeline_active[P: BurstThreadPool](
    state: RankState, mut pools: HeapMoveArray[P],
):
    var bases = InlineArray[Int, 1](uninitialized=True)
    bases[0] = 0
    build_expert_schedules[1, CORR_LOCAL_EXPERTS, C.TOP_K](
        NumaPointerArray[DType.int32, 1](state.route_idx, bases),
        NumaPointerArray[DType.float32, 1](state.route_w, bases),
        NumaPointerArray[DType.int32, 1](state.expert_offset, bases),
        NumaTypedPointerArray[SparseRoute, 1](state.routes, bases),
        CORR_SEQ,
    )
    run_phase1(state, pools)
    run_phase2(state, pools)


def run_pipeline_baseline[P: BurstThreadPool](
    state: RankState, baseline: BaselineSlotState, mut pools: HeapMoveArray[P],
):
    for slot in range(C.TOP_K):
        bucketize_slot_inline(
            state.route_idx, state.route_w,
            slot, CORR_LOCAL_EXPERTS, CORR_SEQ,
            baseline.slot_offset,
            baseline.bucket_token_idx, baseline.bucket_weight,
        )
        run_baseline_expert(state, baseline, pools[0])


def main():
    print("Gemma4 MoE prefill correctness harness (minimax-style)")
    print("  seq=" + String(CORR_SEQ)
        + " hidden=" + String(C.HIDDEN)
        + " experts=" + String(C.NUM_EXPERTS)
        + " top_k=" + String(C.TOP_K)
        + " local_experts=" + String(CORR_LOCAL_EXPERTS))

    var numa = NumaInfo()
    var topo = numa.plan_topology(numa.num_nodes)
    if numa.num_nodes < 1:
        print("FAIL: no NUMA nodes detected")
        return

    var arena = NumaArena[alignment=ALIGNMENT](topo[0], ARENA_BYTES)
    if not arena:
        print("FAIL: arena allocation failed on node " + String(topo[0]))
        return

    var pools = HeapMoveArray[BurstPool[]](1)
    pools.push(BurstPool[].for_topology(numa, topo[0]))
    print("  workers=" + String(pools[0].get_capacity()))

    var alloc = alloc_state(arena, pools[0].get_capacity())
    var state = alloc[0]
    var baseline = alloc[1]

    fill_bf16_unit_scale(state.x_normed, CORR_SEQ * C.HIDDEN)
    for t in range(CORR_SEQ):
        for k in range(C.TOP_K):
            state.route_idx[t * C.TOP_K + k] = Int32((t * 7 + k * 13) % C.NUM_EXPERTS)
            state.route_w[t * C.TOP_K + k] = Float32(0.125)
    print("  route_idx fp:     " + String(fingerprint_i32(state.route_idx, CORR_SEQ * C.TOP_K)))
    print("  route_w fp:       " + String(fingerprint_f32(state.route_w, CORR_SEQ * C.TOP_K)))
    print("  x_normed fp:      " + String(fingerprint(state.x_normed, CORR_SEQ * C.HIDDEN)))

    var elems = CORR_SEQ * C.HIDDEN
    var snapshot_active = arena_alloc[DType.bfloat16](arena, elems)
    var snapshot_baseline = arena_alloc[DType.bfloat16](arena, elems)
    var gt_out = arena_alloc[DType.bfloat16](arena, elems)

    run_pipeline_active(state, pools)
    copy_partial(state.moe_partial, snapshot_active, elems)

    zero_bf16(state.moe_partial, elems)
    run_pipeline_baseline(state, baseline, pools)
    copy_partial(state.moe_partial, snapshot_baseline, elems)

    print("Computing f64 scalar ground truth (this takes a few seconds)...")
    compute_ground_truth(state, gt_out, arena)
    print("  fp_ground_truth: " + String(fingerprint(gt_out, elems)))

    print("\n========== ACTIVE vs GROUND TRUTH ==========")
    var ok_active = compare(snapshot_active, gt_out, elems, 1.0e-3)
    print("\n========== BASELINE vs GROUND TRUTH ==========")
    var ok_baseline = compare(snapshot_baseline, gt_out, elems, 1.0e-3)
    print("\n========== ACTIVE vs BASELINE ==========")
    _ = compare(snapshot_active, snapshot_baseline, elems, 1.0e-3)

    if ok_active and ok_baseline:
        print("\nPASS")
    else:
        print("\nFAIL")
