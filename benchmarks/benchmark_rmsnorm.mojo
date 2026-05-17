from std.collections import InlineArray
from std.memory import UnsafePointer
from std.time import perf_counter_ns
from std.benchmark import keep
from std.sys.info import simd_width_of

from numa import NumaArena, NumaTopology
from threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool
from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from kernels.helpers import (
    DispatchBuffer, recommended_workers, Binding, ArenaBases,
)
from kernels.rmsnorm import (
    rms_reduce_row, rms_normalize_row, rms_norm_row,
    norm_residual_add_row,
    dispatch_rms_norm, fused_norm_residual_add,
    RmsNormTokenKernel,
)
from simd_math.ops import sqrt


comptime ALIGNMENT = 64
comptime WARMUP = 5
comptime TRIALS = 12
comptime ITERS = 30

comptime HIDDEN = 2816
comptime SQRT_N = sqrt[DType.float32, 1](HIDDEN)
comptime N_EPS = HIDDEN * 1e-6

comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


def arena_alloc[dtype: DType](
    mut arena: NumaArena[alignment=ALIGNMENT], count: Int,
) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
    var ptr = arena.alloc[Scalar[dtype]](count)
    if not ptr:
        print("arena alloc failed for", count, "elements")
        return UnsafePointer[Scalar[dtype], MutAnyOrigin].unsafe_dangling()
    return ptr.value()


def arena_bases[tp: Int](
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
) -> ArenaBases[tp]:
    var bases = ArenaBases[tp].uninitialized()
    for r in range(tp):
        bases[r] = Int(arenas[r].base.value())
    return bases


def arena_alloc_all[dtype: DType, tp: Int](
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]], count: Int,
) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
    var first = UnsafePointer[Scalar[dtype], MutAnyOrigin].unsafe_dangling()
    for r in range(tp):
        var ptr = arena_alloc[dtype](arenas[r], count)
        if r == 0:
            first = ptr
    return first


def fill_norm_input(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = Scalar[DType.bfloat16](Float32((i % 127) - 63) * 0.01)


def fill_ones(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = Scalar[DType.bfloat16](Float32(1.0) + Float32(i % 64) * 0.001)


def fill_norm_input_all[tp: Int](
    ptrs: Binding[Scalar[DType.bfloat16], tp], count: Int,
):
    for r in range(tp):
        fill_norm_input(ptrs[r], count)


def fill_ones_all[tp: Int](
    ptrs: Binding[Scalar[DType.bfloat16], tp], count: Int,
):
    for r in range(tp):
        fill_ones(ptrs[r], count)


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


def section_row_primitives(src: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr):
    print("\n=== Row primitives (single token, HIDDEN=" + String(HIDDEN) + ") ===")

    for _ in range(WARMUP):
        var s = rms_reduce_row[HIDDEN](src)
        keep(s)
    var best_reduce = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            var s = rms_reduce_row[HIDDEN](src)
            keep(s)
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // ITERS
        if avg < best_reduce:
            best_reduce = avg
    print("  reduce:           " + fmt_ns(best_reduce)
        + "  (" + fmt_bw(HIDDEN * 2, best_reduce) + " read)")

    for _ in range(WARMUP):
        rms_normalize_row[HIDDEN](src, dst, weight, Float32(0.5))
    var best_normalize = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            rms_normalize_row[HIDDEN](src, dst, weight, Float32(0.5))
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // ITERS
        if avg < best_normalize:
            best_normalize = avg
    keep(dst[0])
    print("  normalize:        " + fmt_ns(best_normalize)
        + "  (" + fmt_bw(HIDDEN * 2 * 3, best_normalize) + " r+r+w)")

    for _ in range(WARMUP):
        rms_norm_row[HIDDEN, SQRT_N, N_EPS](src, dst, weight)
    var best_full = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            rms_norm_row[HIDDEN, SQRT_N, N_EPS](src, dst, weight)
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // ITERS
        if avg < best_full:
            best_full = avg
    keep(dst[0])
    print("  full norm:        " + fmt_ns(best_full)
        + "  (" + fmt_bw(HIDDEN * 2 * 3, best_full) + " r+r+w)")

    for _ in range(WARMUP):
        rms_norm_row[HIDDEN, SQRT_N, N_EPS, scaled=False](src, dst, weight)
    var best_no_w = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            rms_norm_row[HIDDEN, SQRT_N, N_EPS, scaled=False](src, dst, weight)
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // ITERS
        if avg < best_no_w:
            best_no_w = avg
    keep(dst[0])
    print("  norm (no weight): " + fmt_ns(best_no_w)
        + "  (" + fmt_bw(HIDDEN * 2 * 2, best_no_w) + " r+w)")


def section_fused_primitives(
    src: BF16Ptr, residual: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr,
):
    print("\n=== Fused row primitives (single token) ===")

    for _ in range(WARMUP):
        norm_residual_add_row[HIDDEN, SQRT_N, N_EPS](src, residual, dst, weight)
    var best = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            norm_residual_add_row[HIDDEN, SQRT_N, N_EPS](
                src, residual, dst, weight)
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // ITERS
        if avg < best:
            best = avg
    keep(dst[0])
    print("  norm+residual add: " + fmt_ns(best)
        + "  (" + fmt_bw(HIDDEN * 2 * 4, best) + " 2r+w+w)")

def section_dispatch_overhead[P: BurstThreadPool](
    mut pool: P, src: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr,
):
    print("\n=== Dispatch overhead isolation (seq_len=1) ===")

    for _ in range(WARMUP):
        rms_norm_row[HIDDEN, SQRT_N, N_EPS](src, dst, weight)
    var best_inline = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            rms_norm_row[HIDDEN, SQRT_N, N_EPS](src, dst, weight)
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // ITERS
        if avg < best_inline:
            best_inline = avg
    keep(dst[0])

    var buf = DispatchBuffer[RmsNormTokenKernel[HIDDEN, SQRT_N, N_EPS]]()
    for _ in range(WARMUP):
        buf.slot()[] = RmsNormTokenKernel[HIDDEN, SQRT_N, N_EPS](src, dst, weight, 0, 1)
        buf.dispatch(pool)
        pool.join()
    var best_1w = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            buf.slot()[] = RmsNormTokenKernel[HIDDEN, SQRT_N, N_EPS](src, dst, weight, 0, 1)
            buf.dispatch(pool)
            pool.join()
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // ITERS
        if avg < best_1w:
            best_1w = avg
    keep(dst[0])

    print("  inline:      " + fmt_ns(best_inline))
    print("  1w dispatch: " + fmt_ns(best_1w))
    print("  overhead:    " + fmt_ns(best_1w - best_inline))


def section_seq_sweep[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P], src: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr,
    bases: ArenaBases[tp],
):
    print("\n=== Standalone norm: seq_len sweep ===")
    print("  seq | inline       | dispatched   | workers | tokens/us")

    comptime NUM_SIZES = 9
    var sizes = InlineArray[Int, NUM_SIZES](fill=0)
    sizes[0] = 1
    sizes[1] = 2
    sizes[2] = 4
    sizes[3] = 8
    sizes[4] = 16
    sizes[5] = 32
    sizes[6] = 64
    sizes[7] = 128
    sizes[8] = 256

    for s in range(NUM_SIZES):
        var seq = sizes[s]

        for _ in range(WARMUP):
            for tok in range(seq):
                rms_norm_row[HIDDEN, SQRT_N, N_EPS](
                    src + tok * HIDDEN, dst + tok * HIDDEN, weight)
        var best_inline = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                for tok in range(seq):
                    rms_norm_row[HIDDEN, SQRT_N, N_EPS](
                        src + tok * HIDDEN, dst + tok * HIDDEN, weight)
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // ITERS
            if avg < best_inline:
                best_inline = avg
        keep(dst[0])

        for _ in range(WARMUP):
            dispatch_rms_norm[hidden=HIDDEN, sqrt_n=SQRT_N, n_eps=N_EPS, tp=tp](
                Binding[Scalar[DType.bfloat16], tp](src, bases),
                Binding[Scalar[DType.bfloat16], tp](dst, bases),
                Binding[Scalar[DType.bfloat16], tp](weight, bases),
                seq, pools)
        var best_dispatched = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                dispatch_rms_norm[hidden=HIDDEN, sqrt_n=SQRT_N, n_eps=N_EPS, tp=tp](
                    Binding[Scalar[DType.bfloat16], tp](src, bases),
                    Binding[Scalar[DType.bfloat16], tp](dst, bases),
                    Binding[Scalar[DType.bfloat16], tp](weight, bases),
                    seq, pools)
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // ITERS
            if avg < best_dispatched:
                best_dispatched = avg
        keep(dst[0])

        var data_bytes = seq * HIDDEN * 2
        var nw = recommended_workers(data_bytes, pools[0].get_capacity())
        if seq <= 16:
            nw = 0

        var throughput = 0
        if best_dispatched > 0:
            throughput = seq * 1000 // best_dispatched

        var pad = "  " if seq < 10 else " " if seq < 100 else ""
        print("  " + String(seq) + pad
            + "  | " + fmt_ns(best_inline)
            + " | " + fmt_ns(best_dispatched)
            + " | " + String(nw)
            + "       | " + String(throughput))


def section_fused_sweep[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P], partial: BF16Ptr, residual: BF16Ptr,
    res_dst: BF16Ptr, weight: BF16Ptr,
    bases: ArenaBases[tp],
):
    print("\n=== Norm+residual add: seq_len sweep ===")
    print("  seq | inline       | dispatched   | workers | tokens/us")

    comptime NUM_SIZES = 9
    var sizes = InlineArray[Int, NUM_SIZES](fill=0)
    sizes[0] = 1
    sizes[1] = 2
    sizes[2] = 4
    sizes[3] = 8
    sizes[4] = 16
    sizes[5] = 32
    sizes[6] = 64
    sizes[7] = 128
    sizes[8] = 256

    for s in range(NUM_SIZES):
        var seq = sizes[s]

        for _ in range(WARMUP):
            for tok in range(seq):
                norm_residual_add_row[HIDDEN, SQRT_N, N_EPS](
                    partial + tok * HIDDEN, residual + tok * HIDDEN,
                    res_dst + tok * HIDDEN, weight)
        var best_inline = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                for tok in range(seq):
                    norm_residual_add_row[HIDDEN, SQRT_N, N_EPS](
                        partial + tok * HIDDEN, residual + tok * HIDDEN,
                        res_dst + tok * HIDDEN, weight)
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // ITERS
            if avg < best_inline:
                best_inline = avg
        keep(res_dst[0])

        for _ in range(WARMUP):
            fused_norm_residual_add[
                hidden=HIDDEN, sqrt_n=SQRT_N, n_eps=N_EPS, tp=tp,
            ](
                Binding[Scalar[DType.bfloat16], tp](partial, bases),
                Binding[Scalar[DType.bfloat16], tp](residual, bases),
                Binding[Scalar[DType.bfloat16], tp](res_dst, bases),
                Binding[Scalar[DType.bfloat16], tp](weight, bases),
                seq, pools)
        var best_dispatched = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                fused_norm_residual_add[
                    hidden=HIDDEN, sqrt_n=SQRT_N, n_eps=N_EPS, tp=tp,
                ](
                    Binding[Scalar[DType.bfloat16], tp](partial, bases),
                    Binding[Scalar[DType.bfloat16], tp](residual, bases),
                    Binding[Scalar[DType.bfloat16], tp](res_dst, bases),
                    Binding[Scalar[DType.bfloat16], tp](weight, bases),
                    seq, pools)
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // ITERS
            if avg < best_dispatched:
                best_dispatched = avg
        keep(res_dst[0])

        var data_bytes = seq * HIDDEN * 4
        var nw = recommended_workers(data_bytes, pools[0].get_capacity())
        if seq <= 16:
            nw = 0

        var throughput = 0
        if best_dispatched > 0:
            throughput = seq * 1000 // best_dispatched

        var pad = "  " if seq < 10 else " " if seq < 100 else ""
        print("  " + String(seq) + pad
            + "  | " + fmt_ns(best_inline)
            + " | " + fmt_ns(best_dispatched)
            + " | " + String(nw)
            + "       | " + String(throughput))


def run_all[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
):
    comptime MAX_TOKENS = 256
    var bases = arena_bases[tp](arenas)
    var src = arena_alloc_all[DType.bfloat16, tp](arenas, MAX_TOKENS * HIDDEN)
    var dst = arena_alloc_all[DType.bfloat16, tp](arenas, MAX_TOKENS * HIDDEN)
    var weight = arena_alloc_all[DType.bfloat16, tp](arenas, HIDDEN)
    var partial = arena_alloc_all[DType.bfloat16, tp](arenas, MAX_TOKENS * HIDDEN)
    var residual = arena_alloc_all[DType.bfloat16, tp](arenas, MAX_TOKENS * HIDDEN)
    var res_dst = arena_alloc_all[DType.bfloat16, tp](arenas, MAX_TOKENS * HIDDEN)

    fill_norm_input_all[tp](
        Binding[Scalar[DType.bfloat16], tp](src, bases), MAX_TOKENS * HIDDEN)
    fill_norm_input_all[tp](
        Binding[Scalar[DType.bfloat16], tp](partial, bases), MAX_TOKENS * HIDDEN)
    fill_norm_input_all[tp](
        Binding[Scalar[DType.bfloat16], tp](residual, bases), MAX_TOKENS * HIDDEN)
    fill_ones_all[tp](Binding[Scalar[DType.bfloat16], tp](weight, bases), HIDDEN)
    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    var cap = pools[0].get_capacity()
    print("pool capacity: " + String(cap) + " workers")
    print("hidden: " + String(HIDDEN) + " (" + String(HIDDEN * 2) + " bytes bf16)")
    print("sqrt(N): " + String(SQRT_N) + ", N*eps: " + String(N_EPS))

    section_row_primitives(src, dst, weight)
    section_fused_primitives(partial, residual, res_dst, weight)
    section_dispatch_overhead(pools[0], src, dst, weight)
    section_seq_sweep[tp=tp](pools, src, dst, weight, bases)
    section_fused_sweep[tp=tp](pools, partial, residual, res_dst, weight, bases)


def main():
    var topo = NumaTopology()
    var tp = len(topo)

    print("RMSNorm kernel benchmark")
    print(String(tp) + " NUMA node(s), "
        + String(len(topo.isolated_cpus)) + " isolated cpus\n")

    comptime ARENA_BYTES = 256 * 1024 * 1024
    var arenas = HeapMoveArray[NumaArena[alignment=ALIGNMENT]](tp)
    for i in range(tp):
        arenas.push(NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    if topo.has_isolation():
        print("mode: isolated")
        var pools = HeapMoveArray[IsolatedBurstPool[]](tp)
        for i in range(tp):
            pools.push(IsolatedBurstPool[].for_rank(topo, i))
            print("  node " + String(topo[i]) + ": "
                + String(pools[i].get_capacity()) + " workers")
        print("")
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
            pools.push(BurstPool[].for_rank(topo, i))
            print("  node " + String(topo[i]) + ": "
                + String(pools[i].get_capacity()) + " workers")
        print("")
        if tp == 1:
            run_all[tp=1](pools, arenas)
        elif tp == 2:
            run_all[tp=2](pools, arenas)
        elif tp == 4:
            run_all[tp=4](pools, arenas)
        else:
            print("unsupported tp=" + String(tp))
