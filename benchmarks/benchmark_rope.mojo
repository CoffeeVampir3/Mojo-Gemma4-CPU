from std.collections import InlineArray
from std.memory import UnsafePointer
from std.time import perf_counter_ns
from std.benchmark import keep
from std.sys.info import simd_width_of

from numa import NumaArena, NumaInfo, NumaTopology
from threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool
from threading.threading_traits import BurstKernel, BurstThreadPool
from notstdcollections import HeapMoveArray
from kernels.rope import (
    rope_head, rope_token, rope, init_rope_table, init_rope_table_partial,
)


comptime ALIGNMENT = 64
comptime WARMUP = 5
comptime TRIALS = 12
comptime ITERS = 30

comptime HEAD_DIM_SLIDING = 256
comptime HEAD_DIM_FULL = 512
comptime HALF_SLIDING = 128
comptime HALF_FULL = 64
comptime MAX_POS = 4096

comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]


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


def section_head_primitive(data: BF16Ptr, cos_sl: F32Ptr, sin_sl: F32Ptr,
                           cos_fl: F32Ptr, sin_fl: F32Ptr):
    print("\n=== rope_head primitive (single head) ===")

    for _ in range(WARMUP):
        rope_head[HALF_SLIDING, HALF_SLIDING](data, cos_sl, sin_sl)
    var best_sl = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS * 10):
            var t0 = Int(perf_counter_ns())
            rope_head[HALF_SLIDING, HALF_SLIDING](data, cos_sl, sin_sl)
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // (ITERS * 10)
        if avg < best_sl:
            best_sl = avg
    keep(data[0])
    print("  sliding (256 dim, full rot):  " + fmt_ns(best_sl)
        + "  (" + fmt_bw(HEAD_DIM_SLIDING * 2 + HALF_SLIDING * 4 * 2, best_sl) + " r+w+cos+sin)")

    for _ in range(WARMUP):
        rope_head[HALF_FULL, HEAD_DIM_FULL // 2](data, cos_fl, sin_fl)
    var best_fl = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS * 10):
            var t0 = Int(perf_counter_ns())
            rope_head[HALF_FULL, HEAD_DIM_FULL // 2](data, cos_fl, sin_fl)
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // (ITERS * 10)
        if avg < best_fl:
            best_fl = avg
    keep(data[0])
    print("  full (512 dim, 128 partial):  " + fmt_ns(best_fl)
        + "  (" + fmt_bw(HALF_FULL * 2 * 2 + HALF_FULL * 4 * 2, best_fl) + " rotated portion)")


def section_token_scaling(data: BF16Ptr, cos_sl: F32Ptr, sin_sl: F32Ptr):
    print("\n=== rope_token: head count scaling (sliding, single pos) ===")
    print("  heads | time         | bytes touched | BW")

    comptime NUM_SIZES = 5
    var head_counts = InlineArray[Int, NUM_SIZES](fill=0)
    head_counts[0] = 1
    head_counts[1] = 2
    head_counts[2] = 4
    head_counts[3] = 8
    head_counts[4] = 16

    for s in range(NUM_SIZES):
        var nh = head_counts[s]

        for _ in range(WARMUP):
            for h in range(nh):
                rope_head[HALF_SLIDING, HALF_SLIDING](
                    data + h * HEAD_DIM_SLIDING, cos_sl, sin_sl)

        var best = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS * 10):
                var t0 = Int(perf_counter_ns())
                for h in range(nh):
                    rope_head[HALF_SLIDING, HALF_SLIDING](
                        data + h * HEAD_DIM_SLIDING, cos_sl, sin_sl)
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // (ITERS * 10)
            if avg < best:
                best = avg
        keep(data[0])

        var data_bytes = nh * HEAD_DIM_SLIDING * 2
        var pad = "  " if nh < 10 else " "
        print("  " + String(nh) + pad
            + "    | " + fmt_ns(best)
            + " | " + String(data_bytes)
            + "         | " + fmt_bw(data_bytes * 2, best))


def section_seq_sweep[P: BurstThreadPool](
    mut pool: P, data: BF16Ptr, cos_sl: F32Ptr, sin_sl: F32Ptr,
):
    comptime NUM_Q_HEADS = 4
    print("\n=== rope dispatched: seq_len sweep (sliding, " + String(NUM_Q_HEADS)
        + " Q heads) ===")
    print("  seq | inline       | dispatched   | tokens/us")

    comptime NUM_SIZES = 8
    var sizes = InlineArray[Int, NUM_SIZES](fill=0)
    sizes[0] = 1
    sizes[1] = 2
    sizes[2] = 4
    sizes[3] = 8
    sizes[4] = 16
    sizes[5] = 32
    sizes[6] = 64
    sizes[7] = 128

    for s in range(NUM_SIZES):
        var seq = sizes[s]
        comptime row_stride = NUM_Q_HEADS * HEAD_DIM_SLIDING

        for _ in range(WARMUP):
            for tok in range(seq):
                var cr = cos_sl + tok * HALF_SLIDING
                var sr = sin_sl + tok * HALF_SLIDING
                for h in range(NUM_Q_HEADS):
                    rope_head[HALF_SLIDING, HALF_SLIDING](
                        data + tok * row_stride + h * HEAD_DIM_SLIDING, cr, sr)

        var best_inline = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                for tok in range(seq):
                    var cr = cos_sl + tok * HALF_SLIDING
                    var sr = sin_sl + tok * HALF_SLIDING
                    for h in range(NUM_Q_HEADS):
                        rope_head[HALF_SLIDING, HALF_SLIDING](
                            data + tok * row_stride + h * HEAD_DIM_SLIDING, cr, sr)
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // ITERS
            if avg < best_inline:
                best_inline = avg
        keep(data[0])

        for _ in range(WARMUP):
            rope[half=HALF_SLIDING, pair_stride=HALF_SLIDING,
                 num_heads=NUM_Q_HEADS, head_dim=HEAD_DIM_SLIDING](
                data, cos_sl, sin_sl, 0, seq, pool)

        var best_dispatched = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                rope[half=HALF_SLIDING, pair_stride=HALF_SLIDING,
                     num_heads=NUM_Q_HEADS, head_dim=HEAD_DIM_SLIDING](
                    data, cos_sl, sin_sl, 0, seq, pool)
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // ITERS
            if avg < best_dispatched:
                best_dispatched = avg
        keep(data[0])

        var throughput = 0
        if best_dispatched > 0:
            throughput = seq * 1000 // best_dispatched
        var pad = "  " if seq < 10 else " " if seq < 100 else ""
        print("  " + String(seq) + pad
            + "  | " + fmt_ns(best_inline)
            + " | " + fmt_ns(best_dispatched)
            + " | " + String(throughput))


def section_full_vs_sliding[P: BurstThreadPool](
    mut pool: P, data: BF16Ptr,
    cos_sl: F32Ptr, sin_sl: F32Ptr,
    cos_fl: F32Ptr, sin_fl: F32Ptr,
):
    print("\n=== Sliding vs Full attention RoPE (seq_len=64, TP=4 heads) ===")

    comptime SEQ = 64

    for _ in range(WARMUP):
        rope[half=HALF_SLIDING, pair_stride=HALF_SLIDING,
             num_heads=4, head_dim=HEAD_DIM_SLIDING](
            data, cos_sl, sin_sl, 0, SEQ, pool)
    var best_sl = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            rope[half=HALF_SLIDING, pair_stride=HALF_SLIDING,
                 num_heads=4, head_dim=HEAD_DIM_SLIDING](
                data, cos_sl, sin_sl, 0, SEQ, pool)
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // ITERS
        if avg < best_sl:
            best_sl = avg
    keep(data[0])

    for _ in range(WARMUP):
        rope[half=HALF_FULL, pair_stride=HEAD_DIM_FULL // 2,
             num_heads=4, head_dim=HEAD_DIM_FULL](
            data, cos_fl, sin_fl, 0, SEQ, pool)
    var best_fl = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            rope[half=HALF_FULL, pair_stride=HEAD_DIM_FULL // 2,
                 num_heads=4, head_dim=HEAD_DIM_FULL](
                data, cos_fl, sin_fl, 0, SEQ, pool)
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // ITERS
        if avg < best_fl:
            best_fl = avg
    keep(data[0])

    var sl_bytes = SEQ * 4 * HEAD_DIM_SLIDING * 2
    var fl_bytes = SEQ * 4 * HALF_FULL * 2 * 2
    print("  sliding Q (4h×256, full rot):   " + fmt_ns(best_sl)
        + "  (" + fmt_bw(sl_bytes * 2, best_sl) + " r+w)")
    print("  full Q (4h×512, 128 partial):   " + fmt_ns(best_fl)
        + "  (" + fmt_bw(fl_bytes * 2, best_fl) + " rotated r+w)")


def run_all[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
):
    comptime MAX_SEQ = 128
    comptime MAX_HEADS = 16
    comptime MAX_DATA = MAX_SEQ * MAX_HEADS * HEAD_DIM_FULL

    var data = arena_alloc[DType.bfloat16](arenas[0], MAX_DATA)
    var cos_sl = arena_alloc[DType.float32](arenas[0], MAX_POS * HALF_SLIDING)
    var sin_sl = arena_alloc[DType.float32](arenas[0], MAX_POS * HALF_SLIDING)
    var cos_fl = arena_alloc[DType.float32](arenas[0], MAX_POS * HALF_FULL)
    var sin_fl = arena_alloc[DType.float32](arenas[0], MAX_POS * HALF_FULL)

    fill_pattern(data, MAX_DATA)
    init_rope_table[HALF_SLIDING, MAX_POS](cos_sl, sin_sl, 10000.0)
    init_rope_table_partial[HALF_FULL, MAX_POS](cos_fl, sin_fl, 1000000.0, HEAD_DIM_FULL)

    var cap = pools[0].get_capacity()
    print("pool capacity: " + String(cap) + " workers")
    print("sliding: head_dim=256, half=128, theta=10000")
    print("full:    head_dim=512, rotary_half=64, theta=1000000")

    section_head_primitive(data, cos_sl, sin_sl, cos_fl, sin_fl)
    section_token_scaling(data, cos_sl, sin_sl)
    section_seq_sweep(pools[0], data, cos_sl, sin_sl)
    section_full_vs_sliding(pools[0], data, cos_sl, sin_sl, cos_fl, sin_fl)


def main():
    var numa = NumaInfo()
    var topo = numa.plan_topology(numa.num_nodes)
    var tp = numa.num_nodes

    print("RoPE kernel benchmark")
    print(String(tp) + " NUMA node(s), "
        + String(len(numa.isolated_cpus)) + " isolated cpus\n")

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
            pools.push(BurstPool[].for_topology(numa, topo[i]))
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
