from std.collections import InlineArray
from std.memory import UnsafePointer
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
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, print_row, max_last_ts, now_ns,
    DEFAULT_SAMPLES,
)


comptime ALIGNMENT = 64
comptime WARMUP = 30
comptime SAMPLES = DEFAULT_SAMPLES

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


def section_dot_primitive(x: BF16Ptr, weight: BF16Ptr):
    print("\n=== dot_row primitive (single row, HIDDEN=" + String(HIDDEN) + ") ===")

    comptime PU = pick_port_unroll[FW, HIDDEN]()
    var accs = InlineArray[SIMD[DType.float32, FW], PU](uninitialized=True)

    for _ in range(WARMUP):
        comptime for p in range(PU):
            accs[p] = SIMD[DType.float32, FW](0)
        dot_into_accs[cols=HIDDEN](x, weight, accs)
        keep(tree_reduce_accs(accs))

    var samples = SampleBuffer(SAMPLES)
    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        comptime for p in range(PU):
            accs[p] = SIMD[DType.float32, FW](0)
        dot_into_accs[cols=HIDDEN](x, weight, accs)
        keep(tree_reduce_accs(accs))
        var t1 = now_ns()
        samples.push(t1 - t0, t1 - t0)

    var ks = compute_stats(samples.kernel_ns, samples.n)
    var ws = compute_stats(samples.wall_ns, samples.n)
    print_row("dot_row HIDDEN=" + String(HIDDEN), ks, ws, HIDDEN * 2 * 2)


def section_row_sweep[rows: Int](x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr):
    print("\n=== gemv_range inline (row count sweep, no dispatch) ===")

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

    var samples = SampleBuffer(SAMPLES)

    for s in range(NUM_SIZES):
        var n_rows = sizes[s]
        if n_rows > rows:
            continue

        for _ in range(WARMUP):
            gemv_range[rows, HIDDEN](x, weight, output, 0, n_rows)

        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            gemv_range[rows, HIDDEN](x, weight, output, 0, n_rows)
            var t1 = now_ns()
            samples.push(t1 - t0, t1 - t0)
        keep(output[0])

        var ks = compute_stats(samples.kernel_ns, samples.n)
        var ws = compute_stats(samples.wall_ns, samples.n)
        var weight_bytes = n_rows * HIDDEN * 2
        print_row("rows=" + String(n_rows), ks, ws, weight_bytes)


def section_dispatch_scaling[P: BurstThreadPool, //, rows: Int](
    mut pool: P, x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr,
):
    var cap = pool.get_capacity()
    print("\n=== Dispatch scaling (local Q rows=" + String(rows)
        + " rows, capacity=" + String(cap) + ") ===")

    var samples = SampleBuffer(SAMPLES)

    for _ in range(WARMUP):
        gemv_range[rows, HIDDEN](x, weight, output, 0, rows)

    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        gemv_range[rows, HIDDEN](x, weight, output, 0, rows)
        var t1 = now_ns()
        samples.push(t1 - t0, t1 - t0)
    keep(output[0])

    var weight_bytes = rows * HIDDEN * 2
    var ks_inline = compute_stats(samples.kernel_ns, samples.n)
    var ws_inline = compute_stats(samples.wall_ns, samples.n)
    print_row("inline", ks_inline, ws_inline, weight_bytes)

    var nw = 1
    while nw <= cap:
        var buf = DispatchBuffer[GemvKernel[rows, HIDDEN]]()
        for _ in range(WARMUP):
            _ = tile_dispatch(buf,
                GemvKernel[rows, HIDDEN](x, weight, output, 0, 0),
                pool, rows, num_workers=nw)
            pool.join()

        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            _ = tile_dispatch(buf,
                GemvKernel[rows, HIDDEN](x, weight, output, 0, 0),
                pool, rows, num_workers=nw)
            pool.join()
            var t1 = now_ns()
            var t_done = pool.last_worker_timestamp()
            samples.push(t_done - t0, t1 - t0)
        keep(output[0])

        var ks = compute_stats(samples.kernel_ns, samples.n)
        var ws = compute_stats(samples.wall_ns, samples.n)
        print_row("workers=" + String(nw), ks, ws, weight_bytes)

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
    bases: ArenaBases[tp], mut samples: SampleBuffer, label: String,
):
    var xs = Binding[BFloat16, tp](x, bases)
    var ws = Binding[BFloat16, tp](weight, bases)
    var outs = Binding[BFloat16, tp](output, bases)

    for _ in range(WARMUP):
        dispatch_gemv[rows=rows, cols=HIDDEN, tp=tp](xs, ws, outs, pools)
        keep(output[0])

    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        dispatch_gemv[rows=rows, cols=HIDDEN, tp=tp](xs, ws, outs, pools)
        var t1 = now_ns()
        var t_done = max_last_ts[tp=tp](pools)
        samples.push(t_done - t0, t1 - t0)
    keep(output[0])

    var ks = compute_stats(samples.kernel_ns, samples.n)
    var wsx = compute_stats(samples.wall_ns, samples.n)
    var weight_bytes = rows * HIDDEN * 2
    print_row(label + " rows=" + String(rows), ks, wsx, weight_bytes)


def section_projection_sizes[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P], x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr,
    bases: ArenaBases[tp],
):
    print("\n=== Real projection sizes (full dispatch) ===")

    comptime SL_Q = Q_SLIDING // tp
    comptime SL_KV = KV_SLIDING // tp
    comptime FL_Q = Q_FULL // tp
    comptime FL_K = KV_FULL

    var samples = SampleBuffer(SAMPLES)

    measure_dispatch_gemv[SL_Q, tp](
        pools, x, weight, output, bases, samples, "sliding Q")
    measure_dispatch_gemv[SL_KV, tp](
        pools, x, weight, output, bases, samples, "sliding KV")
    measure_dispatch_gemv[FL_Q, tp](
        pools, x, weight, output, bases, samples, "full Q")
    measure_dispatch_gemv[FL_K, tp](
        pools, x, weight, output, bases, samples, "full K replicated")


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

    var samples = SampleBuffer(SAMPLES)
    var total_weight_bytes = (Q_R + KV_R + KV_R) * HIDDEN * 2

    for _ in range(WARMUP):
        dispatch_gemv[rows=Q_R, cols=HIDDEN, tp=tp](xs, qw, qo, pools)
        dispatch_gemv[rows=KV_R, cols=HIDDEN, tp=tp](xs, kw, ko, pools)
        dispatch_gemv[rows=KV_R, cols=HIDDEN, tp=tp](xs, vw, vo, pools)

    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        dispatch_gemv[rows=Q_R, cols=HIDDEN, tp=tp](xs, qw, qo, pools)
        dispatch_gemv[rows=KV_R, cols=HIDDEN, tp=tp](xs, kw, ko, pools)
        dispatch_gemv[rows=KV_R, cols=HIDDEN, tp=tp](xs, vw, vo, pools)
        var t1 = now_ns()
        var t_done = max_last_ts[tp=tp](pools)
        samples.push(t_done - t0, t1 - t0)
    keep(q_out[0])

    var ks_sep = compute_stats(samples.kernel_ns, samples.n)
    var ws_sep = compute_stats(samples.wall_ns, samples.n)
    print_row("3x separate", ks_sep, ws_sep, total_weight_bytes)

    for _ in range(WARMUP):
        dispatch_gemv_chained_qkv[q_rows=Q_R, kv_rows=KV_R, cols=HIDDEN, tp=tp](
            xs, qw, kw, vw, qo, ko, vo, pools)

    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        dispatch_gemv_chained_qkv[q_rows=Q_R, kv_rows=KV_R, cols=HIDDEN, tp=tp](
            xs, qw, kw, vw, qo, ko, vo, pools)
        var t1 = now_ns()
        var t_done = max_last_ts[tp=tp](pools)
        samples.push(t_done - t0, t1 - t0)
    keep(q_out[0])

    var ks_ch = compute_stats(samples.kernel_ns, samples.n)
    var ws_ch = compute_stats(samples.wall_ns, samples.n)
    print_row("chained", ks_ch, ws_ch, total_weight_bytes)


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
