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
from kernels.helpers import DispatchBuffer, RangedKernel, tile_dispatch, recommended_workers
from kernels.gemv import (
    dot_row, gemv_range, gemv, gemv_chained_qkv,
    GemvKernel, ScaledGemvKernel,
)
from simd_math import pick_port_unroll, tree_reduce_accs


comptime ALIGNMENT = 64
comptime WARMUP = 5
comptime TRIALS = 10
comptime ITERS = 10

comptime HIDDEN = 2816
comptime Q_SLIDING = 4096
comptime KV_SLIDING = 2048
comptime Q_FULL = 8192
comptime KV_FULL = 1024

comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime FW = simd_width_of[DType.float32]()


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
        ptr[i] = Scalar[DType.bfloat16](Float32((i % 253) - 126) * 0.005)


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


def section_dot_primitive(x: BF16Ptr, weight: BF16Ptr):
    print("\n=== dot_row primitive (single row, HIDDEN=" + String(HIDDEN) + ") ===")

    comptime PU = pick_port_unroll[FW, HIDDEN]()
    var accs = InlineArray[SIMD[DType.float32, FW], PU](uninitialized=True)

    for _ in range(WARMUP):
        comptime for p in range(PU):
            accs[p] = SIMD[DType.float32, FW](0)
        dot_row[HIDDEN, PU](x, weight, accs)
        keep(tree_reduce_accs(accs))

    var best = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS * 10):
            var t0 = Int(perf_counter_ns())
            comptime for p in range(PU):
                accs[p] = SIMD[DType.float32, FW](0)
            dot_row[HIDDEN, PU](x, weight, accs)
            keep(tree_reduce_accs(accs))
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // (ITERS * 10)
        if avg < best:
            best = avg

    var flops = HIDDEN * 2
    var mflops = flops * 1000 // best
    print("  latency: " + fmt_ns(best)
        + "  (" + fmt_bw(HIDDEN * 2 * 2, best) + " read x+w, "
        + String(mflops) + " MFLOP/s)")


def section_row_sweep(x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr):
    print("\n=== gemv_range inline (row count sweep, no dispatch) ===")
    print("  rows  | time         | weight BW    | rows/us")

    comptime NUM_SIZES = 8
    var sizes = InlineArray[Int, NUM_SIZES](fill=0)
    sizes[0] = 1
    sizes[1] = 4
    sizes[2] = 16
    sizes[3] = 64
    sizes[4] = 256
    sizes[5] = 512
    sizes[6] = 1024
    sizes[7] = 2048

    for s in range(NUM_SIZES):
        var rows = sizes[s]

        for _ in range(WARMUP):
            gemv_range[Q_SLIDING, HIDDEN](x, weight, output, 0, rows)

        var best = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                gemv_range[Q_SLIDING, HIDDEN](x, weight, output, 0, rows)
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // ITERS
            if avg < best:
                best = avg
        keep(output[0])

        var weight_bytes = rows * HIDDEN * 2
        var throughput = 0
        if best > 0:
            throughput = rows * 1000 // best
        var pad = "  " if rows < 10 else " " if rows < 100 else "" if rows < 1000 else ""
        print("  " + String(rows) + pad
            + "   | " + fmt_ns(best)
            + " | " + fmt_bw(weight_bytes, best)
            + " | " + String(throughput))


def section_dispatch_scaling[P: BurstThreadPool](
    mut pool: P, x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr,
):
    var cap = pool.get_capacity()
    print("\n=== Dispatch scaling (Q_SLIDING=" + String(Q_SLIDING)
        + " rows, capacity=" + String(cap) + ") ===")
    print("  workers | time         | weight BW    | speedup")

    for _ in range(WARMUP):
        gemv_range[Q_SLIDING, HIDDEN](x, weight, output, 0, Q_SLIDING)
    var best_inline = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            gemv_range[Q_SLIDING, HIDDEN](x, weight, output, 0, Q_SLIDING)
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // ITERS
        if avg < best_inline:
            best_inline = avg
    keep(output[0])

    var weight_bytes = Q_SLIDING * HIDDEN * 2
    print("  inline  | " + fmt_ns(best_inline)
        + " | " + fmt_bw(weight_bytes, best_inline) + " | 1.0x")

    var nw = 1
    while nw <= cap:
        var buf = DispatchBuffer[GemvKernel[Q_SLIDING, HIDDEN]]()
        for _ in range(WARMUP):
            tile_dispatch(buf,
                GemvKernel[Q_SLIDING, HIDDEN](x, weight, output, 0, 0),
                pool, Q_SLIDING, num_workers=nw)
            pool.join()

        var best = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                tile_dispatch(buf,
                    GemvKernel[Q_SLIDING, HIDDEN](x, weight, output, 0, 0),
                    pool, Q_SLIDING, num_workers=nw)
                pool.join()
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // ITERS
            if avg < best:
                best = avg
        keep(output[0])

        var speedup_10 = best_inline * 10 // best
        var pad = "  " if nw < 10 else " "
        print("  " + String(nw) + pad
            + "      | " + fmt_ns(best)
            + " | " + fmt_bw(weight_bytes, best)
            + " | " + String(speedup_10 // 10) + "." + String(speedup_10 % 10) + "x")

        if nw < 4:
            nw *= 2
        elif nw < cap:
            nw = min(nw * 2, cap)
        else:
            break


def section_projection_sizes[P: BurstThreadPool](
    mut pool: P, x: BF16Ptr,
    mut arena: NumaArena[alignment=ALIGNMENT],
):
    print("\n=== Real projection sizes (full dispatch) ===")
    print("  projection         | rows  | weight MB | time         | BW")

    var cap = pool.get_capacity()

    comptime NUM_PROJS = 4
    var names = InlineArray[String, NUM_PROJS](fill="")
    var row_counts = InlineArray[Int, NUM_PROJS](fill=0)
    names[0] = "Q sliding (TP=4)"
    row_counts[0] = Q_SLIDING // 4
    names[1] = "KV sliding (TP=4)"
    row_counts[1] = KV_SLIDING // 4
    names[2] = "Q full (TP=4)"
    row_counts[2] = Q_FULL // 4
    names[3] = "Q sliding (TP=1)"
    row_counts[3] = Q_SLIDING

    comptime MAX_ROWS = Q_SLIDING
    var weight = arena_alloc[DType.bfloat16](arena, MAX_ROWS * HIDDEN)
    var output = arena_alloc[DType.bfloat16](arena, MAX_ROWS)
    fill_pattern(weight, MAX_ROWS * HIDDEN)

    for p in range(NUM_PROJS):
        var rows = row_counts[p]

        var buf = DispatchBuffer[GemvKernel[MAX_ROWS, HIDDEN]]()
        for _ in range(WARMUP):
            tile_dispatch(buf,
                GemvKernel[MAX_ROWS, HIDDEN](x, weight, output, 0, 0),
                pool, rows, num_workers=cap)
            pool.join()

        var best = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                tile_dispatch(buf,
                    GemvKernel[MAX_ROWS, HIDDEN](x, weight, output, 0, 0),
                    pool, rows, num_workers=cap)
                pool.join()
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // ITERS
            if avg < best:
                best = avg
        keep(output[0])

        var weight_bytes = rows * HIDDEN * 2
        var mb = weight_bytes // (1024 * 1024)
        var name = names[p]
        var pad = ""
        if len(name) < 20:
            pad = " " * (20 - len(name))
        print("  " + name + pad
            + "| " + String(rows)
            + "  | " + String(mb)
            + "        | " + fmt_ns(best)
            + " | " + fmt_bw(weight_bytes, best))


def section_chained_qkv[P: BurstThreadPool](
    mut pool: P, x: BF16Ptr,
    q_weight: BF16Ptr, k_weight: BF16Ptr, v_weight: BF16Ptr,
    q_out: BF16Ptr, k_out: BF16Ptr, v_out: BF16Ptr,
):
    print("\n=== Chained QKV vs 3x separate dispatch (sliding, TP=4) ===")

    comptime Q_R = Q_SLIDING // 4
    comptime KV_R = KV_SLIDING // 4
    var cap = pool.get_capacity()

    # Separate: 3 dispatch+join cycles
    var buf_q = DispatchBuffer[GemvKernel[Q_R, HIDDEN]]()
    var buf_k = DispatchBuffer[GemvKernel[KV_R, HIDDEN]]()
    var buf_v = DispatchBuffer[GemvKernel[KV_R, HIDDEN]]()
    for _ in range(WARMUP):
        tile_dispatch(buf_q, GemvKernel[Q_R, HIDDEN](x, q_weight, q_out, 0, 0),
            pool, Q_R, num_workers=cap)
        pool.join()
        tile_dispatch(buf_k, GemvKernel[KV_R, HIDDEN](x, k_weight, k_out, 0, 0),
            pool, KV_R, num_workers=cap)
        pool.join()
        tile_dispatch(buf_v, GemvKernel[KV_R, HIDDEN](x, v_weight, v_out, 0, 0),
            pool, KV_R, num_workers=cap)
        pool.join()

    var best_separate = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            tile_dispatch(buf_q, GemvKernel[Q_R, HIDDEN](x, q_weight, q_out, 0, 0),
                pool, Q_R, num_workers=cap)
            pool.join()
            tile_dispatch(buf_k, GemvKernel[KV_R, HIDDEN](x, k_weight, k_out, 0, 0),
                pool, KV_R, num_workers=cap)
            pool.join()
            tile_dispatch(buf_v, GemvKernel[KV_R, HIDDEN](x, v_weight, v_out, 0, 0),
                pool, KV_R, num_workers=cap)
            pool.join()
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // ITERS
        if avg < best_separate:
            best_separate = avg
    keep(q_out[0])

    # Chained: 1 dispatch+join cycle
    for _ in range(WARMUP):
        gemv_chained_qkv[q_rows=Q_R, kv_rows=KV_R, cols=HIDDEN](
            x, q_weight, k_weight, v_weight, q_out, k_out, v_out, pool)

    var best_chained = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            gemv_chained_qkv[q_rows=Q_R, kv_rows=KV_R, cols=HIDDEN](
                x, q_weight, k_weight, v_weight, q_out, k_out, v_out, pool)
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // ITERS
        if avg < best_chained:
            best_chained = avg
    keep(q_out[0])

    var total_weight_bytes = (Q_R + KV_R + KV_R) * HIDDEN * 2
    print("  3x separate: " + fmt_ns(best_separate)
        + "  (" + fmt_bw(total_weight_bytes, best_separate) + ")")
    print("  chained:     " + fmt_ns(best_chained)
        + "  (" + fmt_bw(total_weight_bytes, best_chained) + ")")
    print("  savings:     " + fmt_ns(best_separate - best_chained)
        + "  (" + String(100 - best_chained * 100 // best_separate) + "%)")


def run_all[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
):
    comptime MAX_WEIGHT_ELEMS = Q_SLIDING * HIDDEN
    var x = arena_alloc[DType.bfloat16](arenas[0], HIDDEN)
    var weight = arena_alloc[DType.bfloat16](arenas[0], MAX_WEIGHT_ELEMS)
    var output = arena_alloc[DType.bfloat16](arenas[0], Q_SLIDING)

    fill_pattern(x, HIDDEN)
    fill_pattern(weight, MAX_WEIGHT_ELEMS)

    var cap = pools[0].get_capacity()
    print("pool capacity: " + String(cap) + " workers")
    print("hidden: " + String(HIDDEN) + " (" + String(HIDDEN * 2) + " bytes per row)")

    section_dot_primitive(x, weight)
    section_row_sweep(x, weight, output)
    section_dispatch_scaling(pools[0], x, weight, output)
    section_projection_sizes(pools[0], x, arenas[0])

    comptime Q_R = Q_SLIDING // 4
    comptime KV_R = KV_SLIDING // 4
    var q_weight = arena_alloc[DType.bfloat16](arenas[0], Q_R * HIDDEN)
    var k_weight = arena_alloc[DType.bfloat16](arenas[0], KV_R * HIDDEN)
    var v_weight = arena_alloc[DType.bfloat16](arenas[0], KV_R * HIDDEN)
    var q_out = arena_alloc[DType.bfloat16](arenas[0], Q_R)
    var k_out = arena_alloc[DType.bfloat16](arenas[0], KV_R)
    var v_out = arena_alloc[DType.bfloat16](arenas[0], KV_R)
    fill_pattern(q_weight, Q_R * HIDDEN)
    fill_pattern(k_weight, KV_R * HIDDEN)
    fill_pattern(v_weight, KV_R * HIDDEN)

    section_chained_qkv(pools[0], x, q_weight, k_weight, v_weight,
                        q_out, k_out, v_out)


def main():
    var numa = NumaInfo()
    var topo = numa.plan_topology(numa.num_nodes)
    var tp = numa.num_nodes

    print("GEMV kernel benchmark")
    print(String(tp) + " NUMA node(s), "
        + String(len(numa.isolated_cpus)) + " isolated cpus\n")

    comptime ARENA_BYTES = 512 * 1024 * 1024
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
