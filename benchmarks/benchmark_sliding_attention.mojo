from std.collections import InlineArray
from std.memory import UnsafePointer
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.attention_ops import flash_partial_stride
from kernels.attention_dispatch_kernels import dispatch_sliding_attention
from kernels.helpers import Binding, ArenaBases
from kernels.profiling import Profiler
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, print_row, max_last_ts, now_ns,
    DEFAULT_SAMPLES,
)


comptime ALIGNMENT = 64
comptime WARMUP = 30
comptime SAMPLES = DEFAULT_SAMPLES

comptime HEAD_DIM = 256
comptime NUM_Q = 4
comptime NUM_KV = 2
comptime GQA_RATIO = 2
comptime KV_STRIDE = 512
comptime WINDOW = 4096
comptime MAX_WORKERS = 128
comptime PSTRIDE = flash_partial_stride[NUM_Q, HEAD_DIM]()

comptime NUM_CTX_SIZES = 8

comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]


def arena_bases[tp: Int](
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
) -> ArenaBases[tp]:
    var bases = ArenaBases[tp].uninitialized()
    for r in range(tp):
        bases[r] = Int(arenas[r].base.value())
    return bases


def arena_alloc[dtype: DType](
    mut arena: NumaArena[alignment=ALIGNMENT], count: Int,
) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
    var ptr = arena.alloc[Scalar[dtype]](count)
    if not ptr:
        print("arena alloc failed for", count, "elements")
        return UnsafePointer[Scalar[dtype], MutAnyOrigin].unsafe_dangling()
    return ptr.value()


def arena_alloc_all[dtype: DType, tp: Int](
    mut arenas: List[NumaArena[alignment=ALIGNMENT]], count: Int,
) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
    var first = UnsafePointer[Scalar[dtype], MutAnyOrigin].unsafe_dangling()
    for r in range(tp):
        var ptr = arena_alloc[dtype](arenas[r], count)
        if r == 0:
            first = ptr
    return first


def fill_pattern(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = BFloat16(Float32((i % 127) - 63) * 0.01)


def fill_pattern_all[tp: Int](
    ptrs: Binding[BFloat16, tp], count: Int,
):
    for r in range(tp):
        fill_pattern(ptrs[r], count)


def section_context_sweep[P: BurstThreadPool, //, tp: Int](
    mut pools: List[P],
    q: Binding[BFloat16, tp],
    k_cache: Binding[BFloat16, tp],
    v_cache: Binding[BFloat16, tp],
    output: Binding[BFloat16, tp],
    partials: Binding[Float32, tp],
):
    print("\n=== Context sweep (dispatch_sliding_attention) ===")

    var sizes = InlineArray[Int, NUM_CTX_SIZES](fill=0)
    sizes[0] = 1; sizes[1] = 8; sizes[2] = 32; sizes[3] = 128
    sizes[4] = 256; sizes[5] = 512; sizes[6] = 1024; sizes[7] = 4096

    var samples = SampleBuffer(SAMPLES)
    var prof = Profiler[False]()

    for s in range(NUM_CTX_SIZES):
        var vl = sizes[s]
        if vl > WINDOW:
            continue
        var pos = vl - 1

        for _ in range(WARMUP):
            dispatch_sliding_attention[
                head_dim=HEAD_DIM, num_q=NUM_Q,
                gqa_ratio=GQA_RATIO, kv_stride=KV_STRIDE,
                window=WINDOW, cache_size=WINDOW,
                partial_stride=PSTRIDE, tp=tp,
            ](q, k_cache, v_cache, output, partials, pos, 1, pools, prof)
            keep(output[0][0])

        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            dispatch_sliding_attention[
                head_dim=HEAD_DIM, num_q=NUM_Q,
                gqa_ratio=GQA_RATIO, kv_stride=KV_STRIDE,
                window=WINDOW, cache_size=WINDOW,
                partial_stride=PSTRIDE, tp=tp,
            ](q, k_cache, v_cache, output, partials, pos, 1, pools, prof)
            var t1 = now_ns()
            var t_done = max_last_ts[tp=tp](pools)
            samples.push(t_done - t0, t1 - t0)
        keep(output[0][0])

        var ks = compute_stats(samples.kernel_ns, samples.n)
        var ws = compute_stats(samples.wall_ns, samples.n)
        var kv_bytes = vl * KV_STRIDE * 2 * 2
        print_row(String(t"seq={vl}"), ks, ws, kv_bytes)


def section_validation[P: BurstThreadPool, //, tp: Int](
    mut pools: List[P],
    q: Binding[BFloat16, tp],
    k_cache: Binding[BFloat16, tp],
    v_cache: Binding[BFloat16, tp],
    output: Binding[BFloat16, tp],
    partials: Binding[Float32, tp],
):
    print("\n=== Validation (valid_len=64) ===")
    comptime VL = 64
    var pos = VL - 1
    var prof = Profiler[False]()

    dispatch_sliding_attention[
        head_dim=HEAD_DIM, num_q=NUM_Q,
        gqa_ratio=GQA_RATIO, kv_stride=KV_STRIDE,
        window=WINDOW, cache_size=WINDOW,
        partial_stride=PSTRIDE, tp=tp,
    ](q, k_cache, v_cache, output, partials, pos, 1, pools, prof)

    var out0 = output[0]
    var o0 = out0[0].cast[DType.float32]()
    var o1 = out0[1].cast[DType.float32]()
    var o2 = out0[2].cast[DType.float32]()
    var o3 = out0[3].cast[DType.float32]()
    print(t"  output[0..3]: {o0} {o1} {o2} {o3}")
    var ok = True
    for i in range(NUM_Q * HEAD_DIM):
        var v = out0[i].cast[DType.float32]()
        if v != v:
            ok = False
            break
    print("  ", "OK (no NaN)" if ok else "FAIL: NaN detected")


def run_all[P: BurstThreadPool, //, tp: Int](
    mut pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var bases = arena_bases[tp](arenas)

    var q_ptr = arena_alloc_all[DType.bfloat16, tp](arenas, NUM_Q * HEAD_DIM)
    var k_cache_ptr = arena_alloc_all[DType.bfloat16, tp](arenas, WINDOW * KV_STRIDE)
    var v_cache_ptr = arena_alloc_all[DType.bfloat16, tp](arenas, WINDOW * KV_STRIDE)
    var output_ptr = arena_alloc_all[DType.bfloat16, tp](arenas, NUM_Q * HEAD_DIM)
    var partials_ptr = arena_alloc_all[DType.float32, tp](
        arenas, MAX_WORKERS * PSTRIDE)

    var q = Binding[BFloat16, tp](q_ptr, bases)
    var k_cache = Binding[BFloat16, tp](k_cache_ptr, bases)
    var v_cache = Binding[BFloat16, tp](v_cache_ptr, bases)
    var output = Binding[BFloat16, tp](output_ptr, bases)
    var partials = Binding[Float32, tp](partials_ptr, bases)

    fill_pattern_all[tp](q, NUM_Q * HEAD_DIM)
    fill_pattern_all[tp](k_cache, WINDOW * KV_STRIDE)
    fill_pattern_all[tp](v_cache, WINDOW * KV_STRIDE)

    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    var cap = pools[0].get_capacity()
    print(t"pool capacity: {cap} workers")
    print(
        t"head_dim={HEAD_DIM} num_q={NUM_Q} num_kv={NUM_KV} "
        t"gqa={GQA_RATIO} window={WINDOW}"
    )

    section_validation[tp=tp](pools, q, k_cache, v_cache, output, partials)
    section_context_sweep[tp=tp](pools, q, k_cache, v_cache, output, partials)


def main():
    var topo = NumaTopology()
    var tp = len(topo)

    print("Sliding attention benchmark (dispatch_sliding_attention)")
    var iso = len(topo.isolated_cpus)
    print(t"{tp} NUMA node(s), {iso} isolated cpus\n")

    comptime ARENA_BYTES = 256 * 1024 * 1024
    var arenas = List[NumaArena[alignment=ALIGNMENT]](capacity=tp)
    for i in range(tp):
        arenas.append(NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_sliding_attention_tp[
        P: BurstThreadPool, //, degree: Int,
    ](var selected_pools: List[P]):
        run_all[tp=degree](selected_pools, arenas)

    with_topological_rank_dispatch[
        power_of_two_unrolling=3,
        dispatch=dispatch_sliding_attention_tp,
    ](topo, "mode: isolated", "mode: spin-backoff")
