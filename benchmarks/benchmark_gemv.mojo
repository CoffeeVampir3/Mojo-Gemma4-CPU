from std.collections import InlineArray
from std.memory import UnsafePointer
from std.time import perf_counter_ns
from std.benchmark import keep
from std.sys.info import simd_width_of

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from notstdcollections import HeapMoveArray
from kernels.helpers import (
    DispatchBuffer, tile_dispatch, Binding, ArenaBases, dot_into_accs,
)
from kernels.gemv import (
    gemv_range, dispatch_gemv, dispatch_gemv_chained_qkv, GemvKernel,
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

comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime FW = simd_width_of[DType.float32]()


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


def fill_pattern(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = BFloat16(Float32((i % 253) - 126) * 0.005)


def fill_pattern_all[tp: Int](
    ptrs: Binding[BFloat16, tp], count: Int,
):
    for r in range(tp):
        fill_pattern(ptrs[r], count)


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


def max_last_ts[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
) -> Int:
    var hi = 0
    for r in range(tp):
        var ts = pools[r].last_worker_timestamp()
        if ts > hi:
            hi = ts
    return hi


def section_dot_primitive(x: BF16Ptr, weight: BF16Ptr):
    print("\n=== dot_row primitive (single row, HIDDEN=" + String(HIDDEN) + ") ===")

    comptime PU = pick_port_unroll[FW, HIDDEN]()
    var accs = InlineArray[SIMD[DType.float32, FW], PU](uninitialized=True)

    for _ in range(WARMUP):
        comptime for p in range(PU):
            accs[p] = SIMD[DType.float32, FW](0)
        dot_into_accs[cols=HIDDEN](x, weight, accs)
        keep(tree_reduce_accs(accs))

    var best = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS * 10):
            var t0 = Int(perf_counter_ns())
            comptime for p in range(PU):
                accs[p] = SIMD[DType.float32, FW](0)
            dot_into_accs[cols=HIDDEN](x, weight, accs)
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


def section_row_sweep[rows: Int](x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr):
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
        var n_rows = sizes[s]
        if n_rows > rows:
            continue

        for _ in range(WARMUP):
            gemv_range[rows, HIDDEN](x, weight, output, 0, n_rows)

        var best = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                gemv_range[rows, HIDDEN](x, weight, output, 0, n_rows)
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // ITERS
            if avg < best:
                best = avg
        keep(output[0])

        var weight_bytes = n_rows * HIDDEN * 2
        var throughput = 0
        if best > 0:
            throughput = n_rows * 1000 // best
        var pad = "  " if n_rows < 10 else " " if n_rows < 100 else "" if n_rows < 1000 else ""
        print("  " + String(n_rows) + pad
            + "   | " + fmt_ns(best)
            + " | " + fmt_bw(weight_bytes, best)
            + " | " + String(throughput))


def section_dispatch_scaling[P: BurstThreadPool, //, rows: Int](
    mut pool: P, x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr,
):
    var cap = pool.get_capacity()
    print("\n=== Dispatch scaling (local Q rows=" + String(rows)
        + " rows, capacity=" + String(cap) + ") ===")
    print("  workers | kernel time  | wall time    | weight BW    | speedup")

    for _ in range(WARMUP):
        gemv_range[rows, HIDDEN](x, weight, output, 0, rows)
    var best_inline = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            gemv_range[rows, HIDDEN](x, weight, output, 0, rows)
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // ITERS
        if avg < best_inline:
            best_inline = avg
    keep(output[0])

    var weight_bytes = rows * HIDDEN * 2
    print("  inline  | " + fmt_ns(best_inline)
        + " | " + fmt_ns(best_inline)
        + " | " + fmt_bw(weight_bytes, best_inline) + " | 1.0x")

    var nw = 1
    while nw <= cap:
        var buf = DispatchBuffer[GemvKernel[rows, HIDDEN]]()
        for _ in range(WARMUP):
            _ = tile_dispatch(buf,
                GemvKernel[rows, HIDDEN](x, weight, output, 0, 0),
                pool, rows, num_workers=nw)
            pool.join()

        var best_wall = Int(1 << 60)
        var best_kernel = Int(1 << 60)
        for _ in range(TRIALS):
            var wall_sum = 0
            var kernel_sum = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                _ = tile_dispatch(buf,
                    GemvKernel[rows, HIDDEN](x, weight, output, 0, 0),
                    pool, rows, num_workers=nw)
                pool.join()
                var t1 = Int(perf_counter_ns())
                var t_done = pool.last_worker_timestamp()
                wall_sum += t1 - t0
                kernel_sum += t_done - t0
            var avg_wall = wall_sum // ITERS
            var avg_kernel = kernel_sum // ITERS
            if avg_wall < best_wall:
                best_wall = avg_wall
            if avg_kernel < best_kernel:
                best_kernel = avg_kernel
        keep(output[0])

        var speedup_10 = best_inline * 10 // best_kernel
        var pad = "  " if nw < 10 else " "
        print("  " + String(nw) + pad
            + "      | " + fmt_ns(best_kernel)
            + " | " + fmt_ns(best_wall)
            + " | " + fmt_bw(weight_bytes, best_kernel)
            + " | " + String(speedup_10 // 10) + "." + String(speedup_10 % 10) + "x")

        if nw < 4:
            nw *= 2
        elif nw < cap:
            nw = min(nw * 2, cap)
        else:
            break


def measure_dispatch_gemv[
    P: BurstThreadPool, //, rows: Int, tp: Int,
](
    mut pools: HeapMoveArray[P], x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr,
    bases: ArenaBases[tp],
) -> Tuple[Int, Int]:
    var xs = Binding[BFloat16, tp](x, bases)
    var ws = Binding[BFloat16, tp](weight, bases)
    var outs = Binding[BFloat16, tp](output, bases)

    for _ in range(WARMUP):
        dispatch_gemv[rows=rows, cols=HIDDEN, tp=tp](xs, ws, outs, pools)
        keep(output[0])

    var best_wall = Int(1 << 60)
    var best_kernel = Int(1 << 60)
    for _ in range(TRIALS):
        var wall_sum = 0
        var kernel_sum = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            dispatch_gemv[rows=rows, cols=HIDDEN, tp=tp](xs, ws, outs, pools)
            var t1 = Int(perf_counter_ns())
            var t_done = max_last_ts[tp=tp](pools)
            wall_sum += t1 - t0
            kernel_sum += t_done - t0
        var avg_wall = wall_sum // ITERS
        var avg_kernel = kernel_sum // ITERS
        if avg_wall < best_wall:
            best_wall = avg_wall
        if avg_kernel < best_kernel:
            best_kernel = avg_kernel
    keep(output[0])
    return Tuple[Int, Int](best_kernel, best_wall)


def print_projection_row(name: String, rows: Int, kernel_ns: Int, wall_ns: Int):
    var weight_bytes = rows * HIDDEN * 2
    var mb = weight_bytes // (1024 * 1024)
    var pad = ""
    if name.byte_length() < 20:
        pad = " " * (20 - name.byte_length())
    print("  " + name + pad
        + "| " + String(rows)
        + "  | " + String(mb)
        + "        | " + fmt_ns(kernel_ns)
        + " | " + fmt_ns(wall_ns)
        + " | " + fmt_bw(weight_bytes, kernel_ns))


def section_projection_sizes[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P], x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr,
    bases: ArenaBases[tp],
):
    print("\n=== Real projection sizes (full dispatch) ===")
    print("  projection         | rows  | weight MB | kernel time  | wall time    | BW")

    comptime SL_Q = Q_SLIDING // tp
    comptime SL_KV = KV_SLIDING // tp
    comptime FL_Q = Q_FULL // tp
    comptime FL_K = KV_FULL

    var sl_q = measure_dispatch_gemv[SL_Q, tp](
        pools, x, weight, output, bases)
    print_projection_row("sliding Q", SL_Q, sl_q[0], sl_q[1])

    var sl_kv = measure_dispatch_gemv[SL_KV, tp](
        pools, x, weight, output, bases)
    print_projection_row("sliding KV", SL_KV, sl_kv[0], sl_kv[1])

    var fl_q = measure_dispatch_gemv[FL_Q, tp](
        pools, x, weight, output, bases)
    print_projection_row("full Q", FL_Q, fl_q[0], fl_q[1])

    var fl_k = measure_dispatch_gemv[FL_K, tp](
        pools, x, weight, output, bases)
    print_projection_row("full K replicated", FL_K, fl_k[0], fl_k[1])


def section_chained_qkv[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P], x: BF16Ptr,
    q_weight: BF16Ptr, k_weight: BF16Ptr, v_weight: BF16Ptr,
    q_out: BF16Ptr, k_out: BF16Ptr, v_out: BF16Ptr,
    bases: ArenaBases[tp],
):
    print("\n=== Chained QKV vs 3x separate dispatch (sliding, TP="
        + String(tp) + ") ===")

    comptime Q_R = Q_SLIDING // tp
    comptime KV_R = KV_SLIDING // tp
    var xs = Binding[BFloat16, tp](x, bases)
    var qw = Binding[BFloat16, tp](q_weight, bases)
    var kw = Binding[BFloat16, tp](k_weight, bases)
    var vw = Binding[BFloat16, tp](v_weight, bases)
    var qo = Binding[BFloat16, tp](q_out, bases)
    var ko = Binding[BFloat16, tp](k_out, bases)
    var vo = Binding[BFloat16, tp](v_out, bases)

    for _ in range(WARMUP):
        dispatch_gemv[rows=Q_R, cols=HIDDEN, tp=tp](xs, qw, qo, pools)
        dispatch_gemv[rows=KV_R, cols=HIDDEN, tp=tp](xs, kw, ko, pools)
        dispatch_gemv[rows=KV_R, cols=HIDDEN, tp=tp](xs, vw, vo, pools)

    var best_sep_wall = Int(1 << 60)
    var best_sep_kernel = Int(1 << 60)
    for _ in range(TRIALS):
        var wall_sum = 0
        var kernel_sum = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            dispatch_gemv[rows=Q_R, cols=HIDDEN, tp=tp](xs, qw, qo, pools)
            dispatch_gemv[rows=KV_R, cols=HIDDEN, tp=tp](xs, kw, ko, pools)
            dispatch_gemv[rows=KV_R, cols=HIDDEN, tp=tp](xs, vw, vo, pools)
            var t1 = Int(perf_counter_ns())
            var t_done = max_last_ts[tp=tp](pools)
            wall_sum += t1 - t0
            kernel_sum += t_done - t0
        var avg_wall = wall_sum // ITERS
        var avg_kernel = kernel_sum // ITERS
        if avg_wall < best_sep_wall:
            best_sep_wall = avg_wall
        if avg_kernel < best_sep_kernel:
            best_sep_kernel = avg_kernel
    keep(q_out[0])

    for _ in range(WARMUP):
        dispatch_gemv_chained_qkv[q_rows=Q_R, kv_rows=KV_R, cols=HIDDEN, tp=tp](
            xs, qw, kw, vw, qo, ko, vo, pools)

    var best_ch_wall = Int(1 << 60)
    var best_ch_kernel = Int(1 << 60)
    for _ in range(TRIALS):
        var wall_sum = 0
        var kernel_sum = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            dispatch_gemv_chained_qkv[q_rows=Q_R, kv_rows=KV_R, cols=HIDDEN, tp=tp](
                xs, qw, kw, vw, qo, ko, vo, pools)
            var t1 = Int(perf_counter_ns())
            var t_done = max_last_ts[tp=tp](pools)
            wall_sum += t1 - t0
            kernel_sum += t_done - t0
        var avg_wall = wall_sum // ITERS
        var avg_kernel = kernel_sum // ITERS
        if avg_wall < best_ch_wall:
            best_ch_wall = avg_wall
        if avg_kernel < best_ch_kernel:
            best_ch_kernel = avg_kernel
    keep(q_out[0])

    var total_weight_bytes = (Q_R + KV_R + KV_R) * HIDDEN * 2
    print("  3x separate: kernel=" + fmt_ns(best_sep_kernel)
        + " wall=" + fmt_ns(best_sep_wall)
        + "  (" + fmt_bw(total_weight_bytes, best_sep_kernel) + ")")
    print("  chained:     kernel=" + fmt_ns(best_ch_kernel)
        + " wall=" + fmt_ns(best_ch_wall)
        + "  (" + fmt_bw(total_weight_bytes, best_ch_kernel) + ")")
    print("  savings:     kernel=" + fmt_ns(best_sep_kernel - best_ch_kernel)
        + "  (" + String(100 - best_ch_kernel * 100 // best_sep_kernel) + "%)")


def run_all[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
):
    comptime MAX_ROWS = max(max(Q_SLIDING // tp, Q_FULL // tp), KV_FULL)
    comptime MAX_WEIGHT_ELEMS = MAX_ROWS * HIDDEN
    var bases = arena_bases[tp](arenas)
    var x = arena_alloc_all[DType.bfloat16, tp](arenas, HIDDEN)
    var weight = arena_alloc_all[DType.bfloat16, tp](arenas, MAX_WEIGHT_ELEMS)
    var output = arena_alloc_all[DType.bfloat16, tp](arenas, MAX_ROWS)

    fill_pattern_all[tp](Binding[BFloat16, tp](x, bases), HIDDEN)
    fill_pattern_all[tp](
        Binding[BFloat16, tp](weight, bases), MAX_WEIGHT_ELEMS)

    var cap = pools[0].get_capacity()
    print("pool capacity: " + String(cap) + " workers")
    print("hidden: " + String(HIDDEN) + " (" + String(HIDDEN * 2) + " bytes per row)")

    section_dot_primitive(x, weight)
    section_row_sweep[Q_SLIDING // tp](x, weight, output)
    section_dispatch_scaling[rows=Q_SLIDING // tp](pools[0], x, weight, output)
    section_projection_sizes[tp=tp](pools, x, weight, output, bases)

    comptime Q_R = Q_SLIDING // tp
    comptime KV_R = KV_SLIDING // tp
    var q_weight = arena_alloc_all[DType.bfloat16, tp](arenas, Q_R * HIDDEN)
    var k_weight = arena_alloc_all[DType.bfloat16, tp](arenas, KV_R * HIDDEN)
    var v_weight = arena_alloc_all[DType.bfloat16, tp](arenas, KV_R * HIDDEN)
    var q_out = arena_alloc_all[DType.bfloat16, tp](arenas, Q_R)
    var k_out = arena_alloc_all[DType.bfloat16, tp](arenas, KV_R)
    var v_out = arena_alloc_all[DType.bfloat16, tp](arenas, KV_R)
    fill_pattern_all[tp](
        Binding[BFloat16, tp](q_weight, bases), Q_R * HIDDEN)
    fill_pattern_all[tp](
        Binding[BFloat16, tp](k_weight, bases), KV_R * HIDDEN)
    fill_pattern_all[tp](
        Binding[BFloat16, tp](v_weight, bases), KV_R * HIDDEN)

    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    section_chained_qkv[tp=tp](pools, x, q_weight, k_weight, v_weight,
                               q_out, k_out, v_out, bases)


def main():
    var topo = NumaTopology()
    var tp = len(topo)

    print("GEMV kernel benchmark")
    print(String(tp) + " NUMA node(s), "
        + String(len(topo.isolated_cpus)) + " isolated cpus\n")

    comptime ARENA_BYTES = 512 * 1024 * 1024
    var arenas = HeapMoveArray[NumaArena[alignment=ALIGNMENT]](tp)
    for i in range(tp):
        arenas.push(NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_gemv_tp[
        P: BurstThreadPool, //, degree: Int,
    ](var selected_pools: HeapMoveArray[P]):
        run_all[tp=degree](selected_pools, arenas)

    with_topological_rank_dispatch[
        power_of_two_unrolling=3,
        dispatch=dispatch_gemv_tp,
    ](topo, "mode: isolated", "mode: spin-backoff")
