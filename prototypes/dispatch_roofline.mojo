from std.collections import InlineArray
from std.memory import UnsafePointer
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.helpers import Binding, ArenaBases, fanout_dispatch, BF16Ptr
from kernels.gemm import GemmKernel
from kernels.profiling import Profiler
from modeling.gemma4_common import Gemma4BaseConfig
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, now_ns,
)


comptime HIDDEN = Gemma4BaseConfig.HIDDEN
comptime ALIGNMENT = 64
comptime WARMUP = 20
comptime SAMPLES = 200
comptime POOL_BYTES = 512 * 1024 * 1024
comptime NUM_WS = 9


# worker_policy is the API's worker-count knob. passthrough_w treats the
# `data_bytes` argument fanout_dispatch passes it as the requested worker
# count, so we can sweep W at runtime through the real dispatch path with a
# single instantiation (rather than a comptime-W policy per probe).
@always_inline
def passthrough_w(requested: Int, capacity: Int) -> Int:
    return min(requested, capacity)


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


def fill_bf16(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = BFloat16(Float32((i % 253) - 126) * 0.005)


def fill_bf16_all[tp: Int](ptrs: Binding[BFloat16, tp], count: Int):
    for r in range(tp):
        fill_bf16(ptrs[r], count)


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


def gbps2(bytes: Int, ns: Int) -> String:
    if ns <= 0:
        return "0.0"
    var v = Int64(bytes) * 100 // Int64(ns)
    var w = v // 100
    var f = v % 100
    var fs = String(f)
    if f < 10:
        fs = "0" + fs
    return String(w) + "." + fs


# One (rows, W) probe: drive the real fanout_dispatch with the real
# GemmKernel and Profiler[True]; return (latency_p50, compute_p50, used).
def probe[
    P: BurstThreadPool, //, rows: Int, cols: Int, tp: Int,
](
    mut pools: List[P], mut samples: SampleBuffer,
    weight_base: Binding[BFloat16, tp], x: Binding[BFloat16, tp],
    output: Binding[BFloat16, tp], positions: Int, num_workers: Int,
) -> Tuple[Int, Int, Int]:
    comptime K = GemmKernel[rows, cols, 4]
    comptime weight_elems = rows * cols
    comptime node_bytes = rows * cols * 2

    var prof = Profiler[True]()
    var weight_off = 0

    @parameter
    def make(r: Int) -> K:
        return K(x[r], weight_base[r] + weight_off, output[r], 1, 0, 0)

    for i in range(WARMUP):
        weight_off = (i % positions) * weight_elems
        fanout_dispatch[
            tp, make, worker_policy=passthrough_w, label="probe",
        ](pools, prof, rows, num_workers)
    prof.reset()
    samples.clear()
    for i in range(SAMPLES):
        weight_off = (i % positions) * weight_elems
        var t0 = now_ns()
        fanout_dispatch[
            tp, make, worker_policy=passthrough_w, label="probe",
        ](pools, prof, rows, num_workers)
        var t1 = now_ns()
        samples.push(t1 - t0, t1 - t0)
    keep(output[0][0])

    var lat = compute_stats(samples.kernel_ns, samples.n)
    var cap = pools[0].get_capacity()
    var used = min(min(num_workers, cap), rows)

    var disp_p50 = 0
    var comp_p50 = 0
    var join_p50 = 0
    if prof.count > 0:
        ref rec = prof.records[0]
        disp_p50 = rec.dispatch.quantiles(0.5, 0.99)[0]
        comp_p50 = rec.compute.quantiles(0.5, 0.99)[0]
        join_p50 = rec.join.quantiles(0.5, 0.99)[0]

    var bw = gbps2(node_bytes, comp_p50)
    print("  " + padl(String(num_workers), 4) + " " + padl(String(used), 4)
          + "   " + padl(us2(disp_p50), 8) + "  " + padl(us2(comp_p50), 8)
          + "  " + padl(us2(join_p50), 7) + "   " + padl(us2(Int(lat.p50)), 8)
          + "   " + padl(bw, 8) + " GB/s")
    return (Int(lat.p50), comp_p50, used)


def section_rows[
    P: BurstThreadPool, //, rows: Int, cols: Int, tp: Int,
](
    mut pools: List[P], mut samples: SampleBuffer,
    weight: Binding[BFloat16, tp], x: Binding[BFloat16, tp],
    output: Binding[BFloat16, tp], pool_elems: Int,
):
    comptime node_bytes = rows * cols * 2
    comptime weight_elems = rows * cols
    var positions = pool_elems // weight_elems
    if positions < 1:
        positions = 1

    var kb = node_bytes // 1024
    var mb = node_bytes // (1024 * 1024)
    print(t"\n--- rows={rows}  node_bytes={node_bytes} (~{kb} KB / {mb} MB)"
          t"  cold-positions={positions} ---")
    print("     W used   dispatch   compute     join    latency      BW(node)")

    var ws = ws_list()
    var best_lat = 0
    var best_w = 0
    var best_used = 0
    var best_comp = 0
    for wi in range(NUM_WS):
        var res = probe[rows=rows, cols=cols, tp=tp](
            pools, samples, weight, x, output, positions, ws[wi])
        var lat = res[0]
        if best_w == 0 or lat < best_lat:
            best_lat = lat
            best_comp = res[1]
            best_used = res[2]
            best_w = ws[wi]

    var k = node_bytes // (best_used * best_used)
    print(t"  => W*={best_w} used={best_used}  lat*={us2(best_lat)} us  "
          t"compute*={us2(best_comp)} us  k=bytes/W*^2={k}")


def run_all[P: BurstThreadPool, //, tp: Int](
    mut pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    comptime pool_elems = POOL_BYTES // 2
    comptime MAX_ROWS = 65536

    var bases = arena_bases[tp](arenas)
    var weight_ptr = arena_alloc_all[BFloat16, tp](arenas, pool_elems)
    var x_ptr = arena_alloc_all[BFloat16, tp](arenas, HIDDEN)
    var output_ptr = arena_alloc_all[BFloat16, tp](arenas, MAX_ROWS)

    var weight = Binding[BFloat16, tp](weight_ptr, bases)
    var x = Binding[BFloat16, tp](x_ptr, bases)
    var output = Binding[BFloat16, tp](output_ptr, bases)

    fill_bf16_all[tp](weight, pool_elems)
    fill_bf16_all[tp](x, HIDDEN)
    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    var cap = pools[0].get_capacity()
    print(t"dispatch roofline: cols={HIDDEN} tp={tp} capacity={cap}/rank "
          t"pool={POOL_BYTES // (1024*1024)} MB/node")
    print("real fanout_dispatch + Profiler[True]; cold weights; "
          "times are p50, BW from compute")

    var samples = SampleBuffer(SAMPLES)
    section_rows[rows=16, cols=HIDDEN, tp=tp](
        pools, samples, weight, x, output, pool_elems)
    section_rows[rows=64, cols=HIDDEN, tp=tp](
        pools, samples, weight, x, output, pool_elems)
    section_rows[rows=256, cols=HIDDEN, tp=tp](
        pools, samples, weight, x, output, pool_elems)
    section_rows[rows=1024, cols=HIDDEN, tp=tp](
        pools, samples, weight, x, output, pool_elems)
    section_rows[rows=4096, cols=HIDDEN, tp=tp](
        pools, samples, weight, x, output, pool_elems)
    section_rows[rows=16384, cols=HIDDEN, tp=tp](
        pools, samples, weight, x, output, pool_elems)
    section_rows[rows=65536, cols=HIDDEN, tp=tp](
        pools, samples, weight, x, output, pool_elems)


def main():
    var topo = NumaTopology()
    var tp = len(topo)

    print("Dispatch roofline prototype (real API)")
    var iso = len(topo.isolated_cpus)
    print(t"{tp} NUMA node(s), {iso} isolated cpus\n")

    var arena_bytes = POOL_BYTES + 32 * 1024 * 1024
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
