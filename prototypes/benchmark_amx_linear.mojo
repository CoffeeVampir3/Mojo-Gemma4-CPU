from std.collections import InlineArray
from std.memory import UnsafePointer
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.helpers import Binding, ArenaBases
from kernels.profiling import Profiler
from modeling.gemma4_common import Gemma4BaseConfig

from butterquant import PackColsumTask, dispatch_pack_colsum
from butterquant.weight import ButterquantWeight, ButterquantActivation
from butterquant_kernels.linear import dispatch_bq_linear
from quant.recipe import (
    QuantRecipe, PerRowQuant, NoGamma, SingleSided, PerRowCs, VnniPacked,
)

from prototypes.amx_intrinsics import init_intel_amx
from prototypes.amx_linear import dispatch_bq_linear_amx
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, print_row, max_last_ts, now_ns,
    DEFAULT_SAMPLES,
)


comptime HIDDEN = Gemma4BaseConfig.HIDDEN
comptime Q_DIM = Gemma4BaseConfig.Q_DIM_SLIDING
comptime INTERMEDIATE = Gemma4BaseConfig.INTERMEDIATE

comptime QUANT: QuantRecipe = PerRowQuant(
    128, NoGamma(), SingleSided(), PerRowCs(), VnniPacked())

comptime ALIGNMENT = 64
comptime WARMUP = 20
comptime SAMPLES = DEFAULT_SAMPLES
comptime MAX_M = 1024
comptime NUM_M = 12

comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime I8Ptr = UnsafePointer[Int8, MutAnyOrigin]


def arena_alloc[T: AnyType](
    mut arena: NumaArena[alignment=ALIGNMENT], count: Int,
) -> UnsafePointer[T, MutAnyOrigin]:
    var ptr = arena.alloc[T](count)
    if not ptr:
        print("arena alloc failed for", count, "elements")
        return UnsafePointer[T, MutAnyOrigin].unsafe_dangling()
    return ptr.value()


def arena_alloc_all[T: AnyType, tp: Int](
    mut arenas: List[NumaArena[alignment=ALIGNMENT]], count: Int,
) -> UnsafePointer[T, MutAnyOrigin]:
    var first = UnsafePointer[T, MutAnyOrigin].unsafe_dangling()
    for r in range(tp):
        var ptr = arena_alloc[T](arenas[r], count)
        if r == 0:
            first = ptr
    return first


def arena_bases[tp: Int](
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
) -> ArenaBases[tp]:
    var bases = ArenaBases[tp].uninitialized()
    for r in range(tp):
        bases[r] = Int(arenas[r].base.value())
    return bases


def fill_i8(ptr: I8Ptr, count: Int):
    for i in range(count):
        ptr[i] = Int8(((i * 7 + 11) % 251) - 125)


def fill_i8_all[tp: Int](ptrs: Binding[Int8, tp], count: Int):
    for r in range(tp):
        fill_i8(ptrs[r], count)


def fill_f32(ptr: F32Ptr, count: Int, scale: Float32):
    for i in range(count):
        ptr[i] = Float32((i % 17) + 1) * scale


def fill_f32_all[tp: Int](ptrs: Binding[Float32, tp], count: Int, scale: Float32):
    for r in range(tp):
        fill_f32(ptrs[r], count, scale)


def measure[
    P: BurstThreadPool, //, N: Int, K: Int, tp: Int, amx: Bool,
](
    mut pools: List[P],
    act: ButterquantActivation[tp],
    weight: ButterquantWeight[QUANT, N, K, tp],
    output: Binding[BFloat16, tp],
    m: Int,
    mut samples: SampleBuffer,
    label: String,
):
    var prof = Profiler[False]()
    for _ in range(WARMUP):
        comptime if amx:
            dispatch_bq_linear_amx[MR=4](act, weight, output, m, pools, prof)
        else:
            dispatch_bq_linear[MR=4](act, weight, output, m, pools, prof)
    keep(output[0][0])

    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        comptime if amx:
            dispatch_bq_linear_amx[MR=4](act, weight, output, m, pools, prof)
        else:
            dispatch_bq_linear[MR=4](act, weight, output, m, pools, prof)
        var t1 = now_ns()
        var t_done = max_last_ts[tp=tp](pools)
        samples.push(t_done - t0, t1 - t0)
    keep(output[0][0])

    var ks = compute_stats(samples.kernel_ns, samples.n)
    var ws = compute_stats(samples.wall_ns, samples.n)
    var payload = (m * K + N * K + m * N * 2)
    print_row(label, ks, ws, payload)


def verify[N: Int, tp: Int](
    out_base: Binding[BFloat16, tp], out_amx: Binding[BFloat16, tp], m: Int,
):
    var pb = out_base[0]
    var pa = out_amx[0]
    var max_abs = Float32(0)
    var max_val = Float32(0)
    for i in range(m * N):
        var b = Float32(pb[i])
        var a = Float32(pa[i])
        var d = abs(b - a)
        if d > max_abs:
            max_abs = d
        if abs(b) > max_val:
            max_val = abs(b)
    var rel = max_abs / max_val if max_val > Float32(0) else Float32(0)
    print(t"    verify M={m}: max_abs_diff={max_abs} max_val={max_val} rel={rel}")


def section_shape[
    P: BurstThreadPool, //, N: Int, K: Int, tp: Int,
](
    mut pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
    nodes: InlineArray[Int, tp],
    label: String,
):
    print(t"\n=== AMX linear vs vpdpbusd, {label} (N={N} K={K} tp={tp}) ===")

    var bases = arena_bases[tp](arenas)
    var weight_ptr = arena_alloc_all[Int8, tp](arenas, N * K)
    var wsc_ptr = arena_alloc_all[Float32, tp](arenas, N)
    var colsum_ptr = arena_alloc_all[Float32, tp](arenas, N)
    var act_ptr = arena_alloc_all[Int8, tp](arenas, MAX_M * K)
    var act_scale_ptr = arena_alloc_all[Float32, tp](arenas, MAX_M)
    var out_base_ptr = arena_alloc_all[BFloat16, tp](arenas, MAX_M * N)
    var out_amx_ptr = arena_alloc_all[BFloat16, tp](arenas, MAX_M * N)

    var weight = Binding[Int8, tp](weight_ptr, bases)
    var wsc = Binding[Float32, tp](wsc_ptr, bases)
    var colsum = Binding[Float32, tp](colsum_ptr, bases)
    var act = Binding[Int8, tp](act_ptr, bases)
    var act_scale = Binding[Float32, tp](act_scale_ptr, bases)
    var out_base = Binding[BFloat16, tp](out_base_ptr, bases)
    var out_amx = Binding[BFloat16, tp](out_amx_ptr, bases)

    fill_i8_all[tp](weight, N * K)
    fill_f32_all[tp](wsc, N, Float32(0.01))
    fill_i8_all[tp](act, MAX_M * K)
    fill_f32_all[tp](act_scale, MAX_M, Float32(0.02))

    var weight_off = Int(weight_ptr) - bases[0]
    var colsum_off = Int(colsum_ptr) - bases[0]
    var tasks = List[PackColsumTask]()
    tasks.append(PackColsumTask(
        weight_off=weight_off, colsum_off=colsum_off,
        rows=N, cols=K, block_cols=K, colsum_row_major=True))

    var prof = Profiler[False]()
    dispatch_pack_colsum[tp](pools, prof, bases, nodes, tasks)

    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    var bq_weight = ButterquantWeight[QUANT, N, K, tp](weight, wsc, colsum)
    var bq_act = ButterquantActivation[tp](act, act_scale)

    var samples = SampleBuffer(SAMPLES)
    var ms = InlineArray[Int, NUM_M](uninitialized=True)
    ms[0] = 1; ms[1] = 2; ms[2] = 4; ms[3] = 8
    ms[4] = 16; ms[5] = 32; ms[6] = 64; ms[7] = 128
    ms[8] = 256; ms[9] = 366; ms[10] = 512; ms[11] = 1024

    for i in range(NUM_M):
        var m = ms[i]
        measure[N=N, K=K, tp=tp, amx=False](
            pools, bq_act, bq_weight, out_base, m, samples,
            String(t"M={m} vpdpbusd"))
        measure[N=N, K=K, tp=tp, amx=True](
            pools, bq_act, bq_weight, out_amx, m, samples,
            String(t"M={m} amx     "))
        verify[N=N, tp=tp](out_base, out_amx, m)


def run_all[P: BurstThreadPool, //, tp: Int](
    mut pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var nodes = InlineArray[Int, tp](uninitialized=True)
    for r in range(tp):
        nodes[r] = arenas[r].node

    var cap = pools[0].get_capacity()
    print(t"pool capacity: {cap} workers")

    section_shape[N=Q_DIM, K=HIDDEN, tp=tp](
        pools, arenas, nodes, "sliding Q proj")

    for r in range(tp):
        arenas[r].reset()

    section_shape[N=INTERMEDIATE, K=HIDDEN, tp=tp](
        pools, arenas, nodes, "dense gate/up proj")


def main():
    if not init_intel_amx():
        print("init_intel_amx failed: AMX xstate permission denied")
        return

    var topo = NumaTopology()
    var tp = len(topo)

    print("AMX linear prototype benchmark")
    var iso = len(topo.isolated_cpus)
    print(t"{tp} NUMA node(s), {iso} isolated cpus\n")

    comptime ARENA_BYTES = 512 * 1024 * 1024
    var arenas = List[NumaArena[alignment=ALIGNMENT]](capacity=tp)
    for i in range(tp):
        arenas.append(NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_amx_tp[
        P: BurstThreadPool, //, degree: Int,
    ](var selected_pools: List[P]):
        run_all[tp=degree](selected_pools, arenas)

    with_topological_rank_dispatch[
        dispatch=dispatch_amx_tp,
    ](topo, "mode: isolated", "mode: spin-backoff")
