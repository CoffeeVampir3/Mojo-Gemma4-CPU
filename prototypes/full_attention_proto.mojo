from std.collections import InlineArray
from std.memory import UnsafePointer, memcpy
from std.time import perf_counter_ns
from std.benchmark import keep
from std.sys.info import simd_width_of

from numa import NumaArena, NumaInfo, NumaTopology
from threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool
from threading.threading_traits import BurstKernel, BurstThreadPool
from notstdcollections import HeapMoveArray
from kernels.kv_tiled_attention import (
    sliding_attention_single_pass, merge_flash_partials,
    FLASH_PARTIAL_STRIDE,
)
from kernels.helpers import join_all


comptime ALIGNMENT = 64
comptime WARMUP = 10
comptime TRIALS = 20
comptime ITERS = 30

comptime HEAD_DIM = 512
comptime NUM_Q = 4
comptime NUM_KV = 2
comptime GQA_RATIO = 2
comptime KV_STRIDE = 1024
comptime MAX_SEQ = 4096
comptime MAX_WORKERS = 128

comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
comptime W = simd_width_of[DType.float32]()
comptime PARTIAL_STRIDE = FLASH_PARTIAL_STRIDE[NUM_Q, HEAD_DIM]


def arena_alloc[dtype: DType](
    mut arena: NumaArena[alignment=ALIGNMENT], count: Int,
) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
    var ptr = arena.alloc[Scalar[dtype]](count)
    if not ptr:
        print("arena alloc failed for", count, "elements")
        return UnsafePointer[Scalar[dtype], MutAnyOrigin].unsafe_dangling()
    return ptr.value()


def fill_pattern(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = Scalar[DType.bfloat16](Float32((i % 127) - 63) * 0.01)


def fmt_ns(ns: Int) -> String:
    if ns < 1000:
        return String(ns) + " ns"
    elif ns < 1000000:
        return String(ns // 1000) + "." + String((ns % 1000) // 100) + " us"
    else:
        return String(ns // 1000000) + "." + String((ns % 1000000) // 100000) + " ms"


def fmt_bw(total_bytes: Int, ns: Int) -> String:
    if ns <= 0:
        return "inf GB/s"
    var bw_100 = total_bytes * 100 // ns
    return String(bw_100 // 100) + "." + String(bw_100 % 100) + " GB/s"


def valid_count(rank: Int, pos: Int, degree: Int) -> Int:
    if pos < 0:
        return 0
    if rank <= pos % degree:
        return pos // degree + 1
    return pos // degree


def full_attention_pipeline[P: BurstThreadPool, //, tp: Int](
    q: BF16Ptr,
    k_caches: InlineArray[BF16Ptr, tp],
    v_caches: InlineArray[BF16Ptr, tp],
    output: BF16Ptr,
    worker_partials: InlineArray[F32Ptr, tp],
    rank_partials: F32Ptr,
    pos: Int,
    mut pools: HeapMoveArray[P],
):
    comptime window = MAX_SEQ // tp

    for r in range(tp):
        var vl = valid_count(r, pos, tp)
        if vl <= 0:
            var rp = rank_partials + r * PARTIAL_STRIDE
            comptime m_off = NUM_Q * HEAD_DIM
            comptime l_off = m_off + NUM_Q
            for h in range(NUM_Q):
                (rp + m_off + h)[] = Scalar[DType.float32](-1e30)
                (rp + l_off + h)[] = Scalar[DType.float32](0)
                for j in range(0, HEAD_DIM, W):
                    (rp + h * HEAD_DIM + j).store(SIMD[DType.float32, W](0))
            continue

        var rank_start_pos = r
        sliding_attention_single_pass[
            head_dim=HEAD_DIM, num_q=NUM_Q, num_kv=NUM_KV,
            gqa_ratio=GQA_RATIO, kv_stride=KV_STRIDE, window=window](
            q, k_caches[r], v_caches[r], output,
            worker_partials[r], rank_start_pos, vl, pools[r])

    join_all[tp](pools)

    # Gather per-rank partials: each rank's merged worker-partials → rank_partials
    # The single-pass already merged workers into partials[0] on each rank
    # but we need the per-rank result. Actually, single_pass writes the
    # final output to `output` (bf16). We need the raw (acc, m, l) instead.
    #
    # Problem: sliding_attention_single_pass calls merge_flash_partials
    # internally and writes bf16 output. We need the pre-merge f32 partials.
    # For the cross-rank merge, we need the per-RANK (acc, m, l), not per-worker.
    #
    # Solution: we need a variant that outputs the merged partial instead of
    # the final bf16. Let me restructure — run the single-pass kernel,
    # then do a local merge into rank_partials, then cross-rank merge.
    pass


def full_attention_local_partial[P: BurstThreadPool, //, tp: Int](
    q: BF16Ptr,
    k_caches: InlineArray[BF16Ptr, tp],
    v_caches: InlineArray[BF16Ptr, tp],
    worker_partials: InlineArray[F32Ptr, tp],
    rank_partials: InlineArray[F32Ptr, tp],
    pos: Int,
    mut pools: HeapMoveArray[P],
):
    comptime window = MAX_SEQ // tp
    from kernels.kv_tiled_attention import FlashDecodeKernel
    from kernels.helpers import DispatchBuffer, worker_range, recommended_workers

    for r in range(tp):
        var vl = valid_count(r, pos, tp)
        comptime m_off = NUM_Q * HEAD_DIM
        comptime l_off = m_off + NUM_Q
        var rp = rank_partials[r]

        if vl <= 0:
            for h in range(NUM_Q):
                (rp + m_off + h)[] = Scalar[DType.float32](-1e30)
                (rp + l_off + h)[] = Scalar[DType.float32](0)
                for j in range(0, HEAD_DIM, W):
                    (rp + h * HEAD_DIM + j).store(SIMD[DType.float32, W](0))
            continue

        var start_pos = r
        var nw = recommended_workers(vl * KV_STRIDE * 2, pools[r].get_capacity())

        var buf = DispatchBuffer[
            FlashDecodeKernel[HEAD_DIM, NUM_Q, NUM_KV, GQA_RATIO, KV_STRIDE, window]]()
        for w in range(nw):
            var wr = worker_range(vl, nw, w)
            buf.slot()[] = FlashDecodeKernel[
                HEAD_DIM, NUM_Q, NUM_KV, GQA_RATIO, KV_STRIDE, window](
                q, k_caches[r], v_caches[r], worker_partials[r],
                PARTIAL_STRIDE, w, start_pos, wr[0], wr[1])
        buf.dispatch(pools[r])

    join_all[tp](pools)

    # Merge worker partials into per-rank partials
    for r in range(tp):
        var vl = valid_count(r, pos, tp)
        if vl <= 0:
            continue
        var nw = recommended_workers(vl * KV_STRIDE * 2, pools[r].get_capacity())
        merge_worker_partials_to_rank(worker_partials[r], rank_partials[r], nw)


def merge_worker_partials_to_rank(
    worker_parts: F32Ptr, rank_out: F32Ptr, num_workers: Int,
):
    from simd_math import fast_exp_softmax_biased

    comptime acc_off = 0
    comptime m_off = NUM_Q * HEAD_DIM
    comptime l_off = m_off + NUM_Q

    for h in range(NUM_Q):
        var global_m = Scalar[DType.float32](-1e30)
        for w in range(num_workers):
            var wm = (worker_parts + w * PARTIAL_STRIDE + m_off + h)[]
            if wm > global_m:
                global_m = wm

        var global_l = Scalar[DType.float32](0)
        for w in range(num_workers):
            var wm = (worker_parts + w * PARTIAL_STRIDE + m_off + h)[]
            var wl = (worker_parts + w * PARTIAL_STRIDE + l_off + h)[]
            var corr = fast_exp_softmax_biased[1](
                SIMD[DType.float32, 1](wm - global_m))[0]
            global_l += wl * corr

        (rank_out + m_off + h)[] = global_m
        (rank_out + l_off + h)[] = global_l

        for j in range(0, HEAD_DIM, W):
            var merged = SIMD[DType.float32, W](0)
            for w in range(num_workers):
                var wm = (worker_parts + w * PARTIAL_STRIDE + m_off + h)[]
                var corr = fast_exp_softmax_biased[1](
                    SIMD[DType.float32, 1](wm - global_m))[0]
                var acc_v = (worker_parts + w * PARTIAL_STRIDE + acc_off
                    + h * HEAD_DIM + j).load[width=W]()
                merged += SIMD[DType.float32, W](corr) * acc_v
            (rank_out + acc_off + h * HEAD_DIM + j).store(merged)


def merge_rank_partials[tp: Int](
    output: BF16Ptr, rank_partials: InlineArray[F32Ptr, tp],
):
    from simd_math import fast_exp_softmax_biased

    comptime acc_off = 0
    comptime m_off = NUM_Q * HEAD_DIM
    comptime l_off = m_off + NUM_Q

    for h in range(NUM_Q):
        var global_m = Scalar[DType.float32](-1e30)
        for r in range(tp):
            var rm = (rank_partials[r] + m_off + h)[]
            if rm > global_m:
                global_m = rm

        var global_l = Scalar[DType.float32](0)
        for r in range(tp):
            var rm = (rank_partials[r] + m_off + h)[]
            var rl = (rank_partials[r] + l_off + h)[]
            var corr = fast_exp_softmax_biased[1](
                SIMD[DType.float32, 1](rm - global_m))[0]
            global_l += rl * corr

        var inv_l = Scalar[DType.float32](1.0) / global_l
        for j in range(0, HEAD_DIM, W):
            var merged = SIMD[DType.float32, W](0)
            for r in range(tp):
                var rm = (rank_partials[r] + m_off + h)[]
                var corr = fast_exp_softmax_biased[1](
                    SIMD[DType.float32, 1](rm - global_m))[0]
                var acc_v = (rank_partials[r] + acc_off
                    + h * HEAD_DIM + j).load[width=W]()
                merged += SIMD[DType.float32, W](corr) * acc_v
            (output + h * HEAD_DIM + j).store(
                (merged * SIMD[DType.float32, W](inv_l)).cast[DType.bfloat16]())


def section_context_sweep[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    q: BF16Ptr,
    k_caches: InlineArray[BF16Ptr, tp],
    v_caches: InlineArray[BF16Ptr, tp],
    output: BF16Ptr,
    worker_partials: InlineArray[F32Ptr, tp],
    rank_partials: InlineArray[F32Ptr, tp],
):
    print("\n=== Full attention: context length sweep (degree=" + String(tp) + ") ===")
    print("  context  | local attn   | merge        | total        | KV/rank  | BW(local)")

    comptime NUM_SIZES = 9
    var sizes = InlineArray[Int, NUM_SIZES](fill=0)
    sizes[0] = 4
    sizes[1] = 32
    sizes[2] = 64
    sizes[3] = 128
    sizes[4] = 256
    sizes[5] = 512
    sizes[6] = 1024
    sizes[7] = 2048
    sizes[8] = 4096

    for s in range(NUM_SIZES):
        var ctx = sizes[s]
        if ctx > MAX_SEQ:
            continue
        var pos = ctx - 1

        for _ in range(WARMUP):
            full_attention_local_partial[tp](
                q, k_caches, v_caches, worker_partials, rank_partials, pos, pools)
            merge_rank_partials[tp](output, rank_partials)

        var best_local = Int(1 << 60)
        var best_merge = Int(1 << 60)
        var best_total = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed_local = 0
            var elapsed_total = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                full_attention_local_partial[tp](
                    q, k_caches, v_caches, worker_partials, rank_partials, pos, pools)
                var t1 = Int(perf_counter_ns())
                merge_rank_partials[tp](output, rank_partials)
                var t2 = Int(perf_counter_ns())
                elapsed_local += t1 - t0
                elapsed_total += t2 - t0
            var avg_local = elapsed_local // ITERS
            var avg_total = elapsed_total // ITERS
            if avg_local < best_local:
                best_local = avg_local
            if avg_total < best_total:
                best_total = avg_total
                best_merge = avg_total - avg_local
        keep(output[0])

        var per_rank_positions = (ctx + tp - 1) // tp
        var kv_bytes_per_rank = per_rank_positions * KV_STRIDE * 2 * 2
        var pad = " " if ctx < 1000 else "" if ctx < 10000 else ""
        print("  " + String(ctx) + pad
            + "    | " + fmt_ns(best_local)
            + " | " + fmt_ns(best_merge)
            + " | " + fmt_ns(best_total)
            + " | " + String(kv_bytes_per_rank // 1024) + " KB"
            + "   | " + fmt_bw(kv_bytes_per_rank, best_local))


def section_validate[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    q: BF16Ptr,
    k_caches: InlineArray[BF16Ptr, tp],
    v_caches: InlineArray[BF16Ptr, tp],
    output: BF16Ptr,
    worker_partials: InlineArray[F32Ptr, tp],
    rank_partials: InlineArray[F32Ptr, tp],
):
    print("\n=== Validation (context=64) ===")
    var pos = 63

    full_attention_local_partial[tp](
        q, k_caches, v_caches, worker_partials, rank_partials, pos, pools)
    merge_rank_partials[tp](output, rank_partials)

    print("  output[0..3]: "
        + String(output[0].cast[DType.float32]()) + " "
        + String(output[1].cast[DType.float32]()) + " "
        + String(output[2].cast[DType.float32]()) + " "
        + String(output[3].cast[DType.float32]()))

    var has_nan = False
    var has_zero = True
    for i in range(NUM_Q * HEAD_DIM):
        var v = output[i].cast[DType.float32]()
        if v != v:
            has_nan = True
        if v != 0:
            has_zero = False
    if has_nan:
        print("  WARNING: NaN in output")
    elif has_zero:
        print("  WARNING: output is all zeros")
    else:
        print("  OK (non-trivial output, no NaN)")


def run_all[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
):
    comptime positions_per_rank = MAX_SEQ // tp

    var q = arena_alloc[DType.bfloat16](arenas[0], NUM_Q * HEAD_DIM)
    var output = arena_alloc[DType.bfloat16](arenas[0], NUM_Q * HEAD_DIM)
    fill_pattern(q, NUM_Q * HEAD_DIM)

    var k_caches = InlineArray[BF16Ptr, tp](uninitialized=True)
    var v_caches = InlineArray[BF16Ptr, tp](uninitialized=True)
    var worker_parts = InlineArray[F32Ptr, tp](uninitialized=True)
    var rank_parts = InlineArray[F32Ptr, tp](uninitialized=True)

    for r in range(tp):
        k_caches[r] = arena_alloc[DType.bfloat16](arenas[r], positions_per_rank * KV_STRIDE)
        v_caches[r] = arena_alloc[DType.bfloat16](arenas[r], positions_per_rank * KV_STRIDE)
        fill_pattern(k_caches[r], positions_per_rank * KV_STRIDE)
        fill_pattern(v_caches[r], positions_per_rank * KV_STRIDE)
        worker_parts[r] = arena_alloc[DType.float32](arenas[r], MAX_WORKERS * PARTIAL_STRIDE)
        rank_parts[r] = arena_alloc[DType.float32](arenas[r], PARTIAL_STRIDE)

    for r in range(tp):
        print("  rank " + String(r) + ": " + String(pools[r].get_capacity()) + " workers"
            + ", KV cache " + String(positions_per_rank * KV_STRIDE * 2 // 1024) + " KB")

    section_validate(pools, q, k_caches, v_caches, output, worker_parts, rank_parts)
    section_context_sweep(pools, q, k_caches, v_caches, output, worker_parts, rank_parts)


def main():
    var numa = NumaInfo()
    var topo = numa.plan_topology(numa.num_nodes)
    var tp = numa.num_nodes

    print("Full attention prototype (context-parallel)")
    print(String(tp) + " NUMA node(s), "
        + String(len(numa.isolated_cpus)) + " isolated cpus")
    print("head_dim=" + String(HEAD_DIM) + " num_q=" + String(NUM_Q)
        + " num_kv=" + String(NUM_KV) + " gqa=" + String(GQA_RATIO)
        + " max_seq=" + String(MAX_SEQ) + "\n")

    comptime ARENA_BYTES = 256 * 1024 * 1024
    var arenas = HeapMoveArray[NumaArena[alignment=ALIGNMENT]](tp)
    for i in range(tp):
        arenas.push(NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    if numa.has_isolation():
        print("mode: isolated")
        var pools = HeapMoveArray[IsolatedBurstPool[]](tp)
        for i in range(tp):
            pools.push(IsolatedBurstPool[].for_topology(numa, topo[i]))
        if tp == 1:
            run_all[tp=1](pools, arenas)
        elif tp == 2:
            run_all[tp=2](pools, arenas)
        elif tp == 4:
            run_all[tp=4](pools, arenas)
        else:
            print("unsupported tp=" + String(tp))
    else:
        print("mode: spin-backoff")
        var pools = HeapMoveArray[BurstPool[]](tp)
        for i in range(tp):
            pools.push(BurstPool[].for_topology(numa, topo[i]))
        if tp == 1:
            run_all[tp=1](pools, arenas)
        elif tp == 2:
            run_all[tp=2](pools, arenas)
        elif tp == 4:
            run_all[tp=4](pools, arenas)
        else:
            print("unsupported tp=" + String(tp))
