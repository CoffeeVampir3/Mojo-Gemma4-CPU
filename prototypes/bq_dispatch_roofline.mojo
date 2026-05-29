from std.collections import InlineArray
from std.memory import UnsafePointer
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.helpers import (
    Binding, ArenaBases, fanout_dispatch, recommended_workers, BF16Ptr, F32Ptr,
)
from kernels.profiling import Profiler
from butterquant_kernels.linear import BqLinearKernel
from butterquant.types import I8Ptr
from butterquant.vnni import VNNI_N_STEP
from benchmarks.bench_harness import SampleBuffer, compute_stats, now_ns


comptime ALIGNMENT = 64
comptime WARMUP = 15
comptime SAMPLES = 120
comptime POOL_BYTES = 256 * 1024 * 1024
comptime NUM_WS = 9
comptime N_MAX = 2816
comptime K_MAX = 2816
comptime M_MAX = 512


@always_inline
def passthrough_w(requested: Int, capacity: Int) -> Int:
    return min(requested, capacity)


comptime MATMUL_DISPATCH_BW_PRODUCT = 22528


@always_inline
def matmul_workers(data_bytes: Int, capacity: Int) -> Int:
    return recommended_workers[MATMUL_DISPATCH_BW_PRODUCT, 1 << 30](
        data_bytes, capacity)


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


def fill_i8_all[tp: Int](ptrs: Binding[Int8, tp], count: Int):
    for r in range(tp):
        var p = ptrs[r]
        for i in range(count):
            p[i] = Int8((i % 251) - 125)


def fill_f32_all[tp: Int](ptrs: Binding[Float32, tp], count: Int):
    for r in range(tp):
        var p = ptrs[r]
        for i in range(count):
            p[i] = Float32((i % 17) - 8) * 0.01


def fill_bf16_all[tp: Int](ptrs: Binding[BFloat16, tp], count: Int):
    for r in range(tp):
        var p = ptrs[r]
        for i in range(count):
            p[i] = BFloat16(0)


def ws_list() -> InlineArray[Int, NUM_WS]:
    var ws = InlineArray[Int, NUM_WS](uninitialized=True)
    ws[0] = 1; ws[1] = 2; ws[2] = 4; ws[3] = 6; ws[4] = 8
    ws[5] = 12; ws[6] = 16; ws[7] = 24; ws[8] = 32
    return ws


def padl(s: String, w: Int) -> String:
    var pad = String("")
    for _ in range(w - s.byte_length()):
        pad += " "
    return pad + s


def us2(ns: Int) -> String:
    var v = ns * 100 // 1000
    var w = v // 100
    var f = v % 100
    var fs = String(f)
    if f < 10:
        fs = "0" + fs
    return String(w) + "." + fs


def probe[
    P: BurstThreadPool, //, N: Int, K: Int, tp: Int,
](
    mut pools: List[P], mut samples: SampleBuffer,
    weight: Binding[Int8, tp], wsc: Binding[Float32, tp],
    colsum: Binding[Float32, tp], act: Binding[Int8, tp],
    act_scale: Binding[Float32, tp], output: Binding[BFloat16, tp],
    positions: Int, num_workers: Int, seq_len: Int,
) -> Tuple[Int, Int, Int]:
    comptime Kern = BqLinearKernel[N, K, 4]
    comptime weight_elems = N * K
    comptime num_tiles = N // VNNI_N_STEP

    var prof = Profiler[True]()
    var weight_off = 0

    @parameter
    def make(r: Int) -> Kern:
        return Kern(act[r], act_scale[r], weight[r] + weight_off,
                    wsc[r], colsum[r], output[r], seq_len, 0, 0)

    for i in range(WARMUP):
        weight_off = (i % positions) * weight_elems
        fanout_dispatch[
            tp, make, worker_policy=passthrough_w, label="bq_probe",
        ](pools, prof, num_tiles, num_workers)
    prof.reset()
    samples.clear()
    for i in range(SAMPLES):
        weight_off = (i % positions) * weight_elems
        var t0 = now_ns()
        fanout_dispatch[
            tp, make, worker_policy=passthrough_w, label="bq_probe",
        ](pools, prof, num_tiles, num_workers)
        var t1 = now_ns()
        samples.push(t1 - t0, t1 - t0)
    keep(output[0][0])

    var lat = compute_stats(samples.kernel_ns, samples.n)
    var cap = pools[0].get_capacity()
    var used = min(min(num_workers, cap), num_tiles)

    var disp_p50 = 0
    var comp_p50 = 0
    var join_p50 = 0
    if prof.count > 0:
        ref rec = prof.records[0]
        disp_p50 = rec.dispatch.quantiles(0.5, 0.99)[0]
        comp_p50 = rec.compute.quantiles(0.5, 0.99)[0]
        join_p50 = rec.join.quantiles(0.5, 0.99)[0]

    print("  " + padl(String(num_workers), 4) + " " + padl(String(used), 4)
          + "   " + padl(us2(disp_p50), 8) + "  " + padl(us2(comp_p50), 9)
          + "  " + padl(us2(join_p50), 7) + "   " + padl(us2(Int(lat.p50)), 9))
    return (Int(lat.p50), comp_p50, used)


def section[
    P: BurstThreadPool, //, N: Int, K: Int, tp: Int,
](
    mut pools: List[P], mut samples: SampleBuffer,
    weight: Binding[Int8, tp], wsc: Binding[Float32, tp],
    colsum: Binding[Float32, tp], act: Binding[Int8, tp],
    act_scale: Binding[Float32, tp], output: Binding[BFloat16, tp],
    label: String, seq_len: Int,
):
    comptime weight_elems = N * K
    comptime weight_kb = weight_elems // 1024
    var positions = POOL_BYTES // weight_elems
    if positions < 1:
        positions = 1

    print(t"\n--- {label}  N={N} K={K}  weight={weight_kb} KB (int8)"
          t"  seq_len={seq_len}  cold-positions={positions} ---")
    print("     W used   dispatch    compute     join    latency")

    var ws = ws_list()
    var best_lat = 0
    var best_w = 0
    var best_used = 0
    for wi in range(NUM_WS):
        var res = probe[N=N, K=K, tp=tp](
            pools, samples, weight, wsc, colsum, act, act_scale, output,
            positions, ws[wi], seq_len)
        var lat = res[0]
        if best_w == 0 or lat < best_lat:
            best_lat = lat
            best_used = res[2]
            best_w = ws[wi]
    var cap = pools[0].get_capacity()
    var picked = matmul_workers(N * K * seq_len, cap)
    print(t"  => W*={best_w} used={best_used}  lat*={us2(best_lat)} us"
          t"   |  matmul_workers picks W={picked}")


def run_shape[
    P: BurstThreadPool, //, N: Int, K: Int, tp: Int,
](
    mut pools: List[P], mut samples: SampleBuffer,
    weight: Binding[Int8, tp], wsc: Binding[Float32, tp],
    colsum: Binding[Float32, tp], act: Binding[Int8, tp],
    act_scale: Binding[Float32, tp], output: Binding[BFloat16, tp],
    label: String,
):
    section[N=N, K=K, tp=tp](
        pools, samples, weight, wsc, colsum, act, act_scale, output,
        label, 1)
    section[N=N, K=K, tp=tp](
        pools, samples, weight, wsc, colsum, act, act_scale, output,
        label, 512)


def run_all[P: BurstThreadPool, //, tp: Int](
    mut pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    comptime pool_elems = POOL_BYTES

    var bases = arena_bases[tp](arenas)
    var weight_ptr = arena_alloc_all[Int8, tp](arenas, pool_elems)
    var wsc_ptr = arena_alloc_all[Float32, tp](arenas, N_MAX)
    var colsum_ptr = arena_alloc_all[Float32, tp](arenas, N_MAX)
    var act_ptr = arena_alloc_all[Int8, tp](arenas, M_MAX * K_MAX)
    var act_scale_ptr = arena_alloc_all[Float32, tp](arenas, M_MAX)
    var output_ptr = arena_alloc_all[BFloat16, tp](arenas, M_MAX * N_MAX)

    var weight = Binding[Int8, tp](weight_ptr, bases)
    var wsc = Binding[Float32, tp](wsc_ptr, bases)
    var colsum = Binding[Float32, tp](colsum_ptr, bases)
    var act = Binding[Int8, tp](act_ptr, bases)
    var act_scale = Binding[Float32, tp](act_scale_ptr, bases)
    var output = Binding[BFloat16, tp](output_ptr, bases)

    fill_i8_all[tp](weight, pool_elems)
    fill_f32_all[tp](wsc, N_MAX)
    fill_f32_all[tp](colsum, N_MAX)
    fill_i8_all[tp](act, M_MAX * K_MAX)
    fill_f32_all[tp](act_scale, M_MAX)
    fill_bf16_all[tp](output, M_MAX * N_MAX)
    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    var cap = pools[0].get_capacity()
    print(t"bq linear dispatch roofline: tp={tp} capacity={cap}/rank "
          t"pool={POOL_BYTES // (1024*1024)} MB/node")
    print("real BqLinearKernel + Profiler[True]; cold weights; p50 us")

    var samples = SampleBuffer(SAMPLES)
    run_shape[N=2048, K=2816, tp=tp](
        pools, samples, weight, wsc, colsum, act, act_scale, output,
        "qkv-sliding/4")
    run_shape[N=2816, K=1024, tp=tp](
        pools, samples, weight, wsc, colsum, act, act_scale, output,
        "o-proj-sliding/4")


def main():
    var topo = NumaTopology()
    var tp = len(topo)

    print("BQ linear dispatch roofline (decode vs prefill)")
    var iso = len(topo.isolated_cpus)
    print(t"{tp} NUMA node(s), {iso} isolated cpus\n")

    var arena_bytes = POOL_BYTES + 64 * 1024 * 1024
    var arenas = List[NumaArena[alignment=ALIGNMENT]](capacity=tp)
    for i in range(tp):
        arenas.append(NumaArena[alignment=ALIGNMENT](topo[i], arena_bytes))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_tp[P: BurstThreadPool, //, degree: Int](
        var selected_pools: List[P]
    ):
        run_all[tp=degree](selected_pools, arenas)

    with_topological_rank_dispatch[
        dispatch=dispatch_tp,
    ](topo, "mode: isolated", "mode: spin-backoff")
