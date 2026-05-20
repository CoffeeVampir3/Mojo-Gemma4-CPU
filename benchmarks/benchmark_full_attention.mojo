from std.collections import InlineArray
from std.memory import UnsafePointer
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from notstdcollections import HeapMoveArray
from kernels.attention_ops import LinearKV
from kernels.flash_attention import FlashAttentionKernel
from kernels.attention_dispatch_kernels import dispatch_full_attention
from kernels.helpers import Binding, ArenaBases
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, print_row, max_last_ts, now_ns,
    DEFAULT_SAMPLES,
)


comptime ALIGNMENT = 64
comptime WARMUP = 30
comptime SAMPLES = DEFAULT_SAMPLES

comptime HEAD_DIM = 512
comptime GLOBAL_NUM_Q = 16
comptime NUM_KV = 2
comptime GLOBAL_GQA = GLOBAL_NUM_Q // NUM_KV
comptime KV_STRIDE = 1024
comptime MAX_SEQ = 4096
comptime MAX_WORKERS = 128

comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]


def arena_bases[tp: Int](
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
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
        ptr[i] = BFloat16(Float32((i % 127) - 63) * 0.01)


def fill_pattern_all[tp: Int](
    ptrs: Binding[BFloat16, tp], count: Int,
):
    for r in range(tp):
        fill_pattern(ptrs[r], count)


def section_validation[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P], q: BF16Ptr, k_cache: BF16Ptr, v_cache: BF16Ptr,
    output: BF16Ptr, partials: F32Ptr, bases: ArenaBases[tp],
):
    print("\n=== Validation (valid_len=64) ===")
    comptime VL = 64
    comptime LOCAL_NUM_Q = GLOBAL_NUM_Q // tp

    dispatch_full_attention[
        head_dim=HEAD_DIM, num_q=GLOBAL_NUM_Q,
        local_num_q=LOCAL_NUM_Q, gqa_ratio=GLOBAL_GQA,
        kv_stride=KV_STRIDE, tp=tp,
    ](
        Binding[BFloat16, tp](q, bases),
        Binding[BFloat16, tp](k_cache, bases),
        Binding[BFloat16, tp](v_cache, bases),
        Binding[BFloat16, tp](output, bases),
        Binding[Float32, tp](partials, bases),
        VL - 1, 1, pools)

    print("  output[0..3]: "
        + String(output[0].cast[DType.float32]()) + " "
        + String(output[1].cast[DType.float32]()) + " "
        + String(output[2].cast[DType.float32]()) + " "
        + String(output[3].cast[DType.float32]()))
    var ok = True
    for i in range(LOCAL_NUM_Q * HEAD_DIM):
        var v = output[i].cast[DType.float32]()
        if v != v:
            ok = False
            break
    print("  " + ("OK (no NaN)" if ok else "FAIL: NaN detected"))


def section_context_sweep[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P], q: BF16Ptr, k_cache: BF16Ptr, v_cache: BF16Ptr,
    output: BF16Ptr, partials: F32Ptr, bases: ArenaBases[tp],
):
    print("\n=== Context sweep (replicated Q + context-local attention + merge) ===")
    comptime LOCAL_NUM_Q = GLOBAL_NUM_Q // tp

    var sizes = InlineArray[Int, 8](fill=0)
    sizes[0] = 4; sizes[1] = 32; sizes[2] = 64; sizes[3] = 128
    sizes[4] = 256; sizes[5] = 512; sizes[6] = 1024; sizes[7] = 4096

    var samples = SampleBuffer(SAMPLES)

    for s in range(8):
        var vl = sizes[s]
        if vl > MAX_SEQ:
            continue

        for _ in range(WARMUP):
            dispatch_full_attention[
                head_dim=HEAD_DIM, num_q=GLOBAL_NUM_Q,
                local_num_q=LOCAL_NUM_Q, gqa_ratio=GLOBAL_GQA,
                kv_stride=KV_STRIDE, tp=tp,
            ](
                Binding[BFloat16, tp](q, bases),
                Binding[BFloat16, tp](k_cache, bases),
                Binding[BFloat16, tp](v_cache, bases),
                Binding[BFloat16, tp](output, bases),
                Binding[Float32, tp](partials, bases),
                vl - 1, 1, pools)
            keep(output[0])

        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            dispatch_full_attention[
                head_dim=HEAD_DIM, num_q=GLOBAL_NUM_Q,
                local_num_q=LOCAL_NUM_Q, gqa_ratio=GLOBAL_GQA,
                kv_stride=KV_STRIDE, tp=tp,
            ](
                Binding[BFloat16, tp](q, bases),
                Binding[BFloat16, tp](k_cache, bases),
                Binding[BFloat16, tp](v_cache, bases),
                Binding[BFloat16, tp](output, bases),
                Binding[Float32, tp](partials, bases),
                vl - 1, 1, pools)
            var t1 = now_ns()
            var t_done = max_last_ts[tp=tp](pools)
            samples.push(t_done - t0, t1 - t0)
        keep(output[0])

        var ks = compute_stats(samples.kernel_ns, samples.n)
        var ws = compute_stats(samples.wall_ns, samples.n)
        var kv_bytes = vl * KV_STRIDE * 2 * 2
        print_row("seq=" + String(vl), ks, ws, kv_bytes)


def run_all[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
):
    var bases = arena_bases[tp](arenas)
    comptime LOCAL_NUM_Q = GLOBAL_NUM_Q // tp
    comptime PSTRIDE = FlashAttentionKernel[
        LinearKV, HEAD_DIM, GLOBAL_NUM_Q, GLOBAL_GQA, KV_STRIDE,
    ].PARTIAL_STRIDE
    var q = arena_alloc_all[DType.bfloat16, tp](arenas, GLOBAL_NUM_Q * HEAD_DIM)
    var k_cache = arena_alloc_all[DType.bfloat16, tp](arenas, MAX_SEQ * KV_STRIDE)
    var v_cache = arena_alloc_all[DType.bfloat16, tp](arenas, MAX_SEQ * KV_STRIDE)
    var output = arena_alloc_all[DType.bfloat16, tp](arenas, LOCAL_NUM_Q * HEAD_DIM)
    var partials = arena_alloc_all[DType.float32, tp](arenas, MAX_WORKERS * PSTRIDE)

    fill_pattern_all[tp](
        Binding[BFloat16, tp](q, bases), GLOBAL_NUM_Q * HEAD_DIM)
    fill_pattern_all[tp](
        Binding[BFloat16, tp](k_cache, bases), MAX_SEQ * KV_STRIDE)
    fill_pattern_all[tp](
        Binding[BFloat16, tp](v_cache, bases), MAX_SEQ * KV_STRIDE)
    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    var cap = pools[0].get_capacity()
    print("pool capacity: " + String(cap) + " workers")
    print("head_dim=" + String(HEAD_DIM) + " num_q=" + String(LOCAL_NUM_Q)
        + " num_kv=" + String(NUM_KV) + " gqa=" + String(GLOBAL_GQA)
        + " max_seq=" + String(MAX_SEQ))

    section_validation[tp=tp](pools, q, k_cache, v_cache, output, partials, bases)
    section_context_sweep[tp=tp](pools, q, k_cache, v_cache, output, partials, bases)


def main():
    var topo = NumaTopology()
    var tp = len(topo)

    print("Full attention benchmark (replicated Q + context-local attention + merge)")
    print(String(tp) + " NUMA node(s), "
        + String(len(topo.isolated_cpus)) + " isolated cpus\n")

    comptime ARENA_BYTES = 256 * 1024 * 1024
    var arenas = HeapMoveArray[NumaArena[alignment=ALIGNMENT]](tp)
    for i in range(tp):
        arenas.push(NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_full_attention_tp[
        P: BurstThreadPool, //, degree: Int,
    ](var selected_pools: HeapMoveArray[P]):
        run_all[tp=degree](selected_pools, arenas)

    with_topological_rank_dispatch[
        power_of_two_unrolling=3,
        dispatch=dispatch_full_attention_tp,
    ](topo, "mode: isolated", "mode: spin-backoff")
