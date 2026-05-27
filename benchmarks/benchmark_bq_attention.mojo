from std.collections import InlineArray
from std.memory import UnsafePointer
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.attention_ops import (
    LinearKV, RingKV, flash_partial_stride, full_local_kv_count,
)
from kernels.helpers import Binding, ArenaBases, fanout_dispatch_per_rank
from kernels.logsum_merge import (
    dispatch_merge_flash_partials, dispatch_merge_context_flash_partials,
)
from butterquant_kernels import BqFlashAttentionKernel
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, print_row, max_last_ts, now_ns,
    DEFAULT_SAMPLES,
)

comptime ALIGNMENT = 64
comptime WARMUP = 30
comptime SAMPLES = DEFAULT_SAMPLES
comptime MAX_SEQ = 4096
comptime MAX_WORKERS = 128
comptime NUM_CTX_SIZES = 8

comptime SLIDING_HEAD_DIM = 256
comptime SLIDING_NUM_Q = 4
comptime SLIDING_NUM_KV = 2
comptime SLIDING_GQA_RATIO = 2
comptime SLIDING_KV_STRIDE = 512
comptime SLIDING_WINDOW = 4096

comptime FULL_HEAD_DIM = 512
comptime FULL_GLOBAL_NUM_Q = 16
comptime FULL_NUM_KV = 2
comptime FULL_GQA_RATIO = FULL_GLOBAL_NUM_Q // FULL_NUM_KV
comptime FULL_KV_STRIDE = 1024

comptime I8Ptr = UnsafePointer[Int8, MutAnyOrigin]
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


@always_inline
def sliding_valid_len(pos: Int) -> Int:
    if pos + 1 >= SLIDING_WINDOW:
        return SLIDING_WINDOW
    return pos + 1


def dispatch_bq_sliding_attention[P: BurstThreadPool, //, tp: Int](
    q: Binding[Int8, tp],
    qi_bias: Binding[Float32, tp],
    f_q: Binding[Float32, tp],
    k_cache: Binding[Int8, tp],
    k_scale: Binding[Float32, tp],
    v_cache: Binding[Int8, tp],
    v_scale: Binding[Float32, tp],
    output: Binding[BFloat16, tp],
    partials: Binding[Float32, tp],
    base_pos: Int,
    mut pools: List[P],
):
    comptime pstride = flash_partial_stride[SLIDING_NUM_Q, SLIDING_HEAD_DIM]()
    comptime DecodeK = BqFlashAttentionKernel[
        RingKV[SLIDING_WINDOW], SLIDING_HEAD_DIM, SLIDING_NUM_Q,
        SLIDING_NUM_KV, SLIDING_GQA_RATIO, SLIDING_KV_STRIDE, pstride,
    ]

    var valid_len = sliding_valid_len(base_pos)
    if valid_len <= 0:
        return
    var start_pos = base_pos - valid_len + 1

    @parameter
    def make_decode(r: Int) -> DecodeK:
        return DecodeK(
            q[r], qi_bias[r], f_q[r], k_cache[r], k_scale[r],
            v_cache[r], v_scale[r], partials[r], 0, start_pos, 0, 0,
        )

    @parameter
    def total_for(r: Int) -> Int:
        return valid_len

    @parameter
    def bytes_for(r: Int) -> Int:
        return valid_len * SLIDING_KV_STRIDE * 2

    var nws = fanout_dispatch_per_rank[
        tp, make_decode, total_for, bytes_for,
        max_worker_count=MAX_WORKERS,
    ](pools)

    dispatch_merge_flash_partials[
        head_dim=SLIDING_HEAD_DIM, num_q=SLIDING_NUM_Q,
        partial_stride=pstride, tp=tp, max_worker_count=MAX_WORKERS,
    ](output, partials, nws, pools)


def dispatch_bq_full_attention[P: BurstThreadPool, //, tp: Int](
    q: Binding[Int8, tp],
    qi_bias: Binding[Float32, tp],
    f_q: Binding[Float32, tp],
    k_cache: Binding[Int8, tp],
    k_scale: Binding[Float32, tp],
    v_cache: Binding[Int8, tp],
    v_scale: Binding[Float32, tp],
    output: Binding[BFloat16, tp],
    partials: Binding[Float32, tp],
    base_pos: Int,
    mut pools: List[P],
):
    comptime assert FULL_GLOBAL_NUM_Q % tp == 0, "full Q heads must shard evenly"
    comptime local_num_q = FULL_GLOBAL_NUM_Q // tp
    comptime pstride = flash_partial_stride[FULL_GLOBAL_NUM_Q, FULL_HEAD_DIM]()
    comptime DecodeK = BqFlashAttentionKernel[
        LinearKV, FULL_HEAD_DIM, FULL_GLOBAL_NUM_Q, FULL_NUM_KV,
        FULL_GQA_RATIO, FULL_KV_STRIDE, pstride,
    ]

    var valid_lens = InlineArray[Int, tp](uninitialized=True)
    for rank in range(tp):
        valid_lens[rank] = full_local_kv_count(rank, base_pos, tp)

    @parameter
    def make_decode(r: Int) -> DecodeK:
        return DecodeK(
            q[r], qi_bias[r], f_q[r], k_cache[r], k_scale[r],
            v_cache[r], v_scale[r], partials[r], 0, 0, 0, 0,
        )

    @parameter
    def total_for(r: Int) -> Int:
        return valid_lens[r]

    @parameter
    def bytes_for(r: Int) -> Int:
        return valid_lens[r] * FULL_KV_STRIDE * 2

    var nws = fanout_dispatch_per_rank[
        tp, make_decode, total_for, bytes_for,
        max_worker_count=MAX_WORKERS,
    ](pools)

    dispatch_merge_context_flash_partials[
        head_dim=FULL_HEAD_DIM, num_q=FULL_GLOBAL_NUM_Q,
        local_num_q=local_num_q,
        partial_stride=pstride, tp=tp, max_worker_count=MAX_WORKERS,
    ](output, partials, nws, pools)


def fill_i8(ptr: I8Ptr, count: Int, phase: Int):
    for i in range(count):
        var v = ((i * 17 + phase * 29) % 127) - 63
        ptr[i] = Int8(v)


def fill_i8_all[tp: Int](ptrs: Binding[Int8, tp], count: Int, phase: Int):
    for r in range(tp):
        fill_i8(ptrs[r], count, phase + r)


def fill_scales(ptr: F32Ptr, count: Int, base: Float32):
    for i in range(count):
        ptr[i] = base + Float32((i * 13) % 23) * Float32(0.003)


def fill_scales_all[tp: Int](
    ptrs: Binding[Float32, tp], count: Int, base: Float32,
):
    for r in range(tp):
        fill_scales(ptrs[r], count, base + Float32(r) * Float32(0.002))


def fill_q_aux[head_dim: Int, num_q: Int](
    q: I8Ptr, qi_bias: F32Ptr, f_q: F32Ptr,
):
    comptime inv_sqrt = (
        Float32(0.0625)
        if head_dim == 256 else Float32(0.04419417382415922)
    )
    for h in range(num_q):
        var qsum = Int(0)
        for j in range(head_dim):
            qsum += Int(q[h * head_dim + j])
        qi_bias[h] = Float32(qsum) * Float32(128.0)
        f_q[h] = (Float32(0.22) + Float32(h % 5) * Float32(0.015)) * inv_sqrt


def fill_q_aux_all[head_dim: Int, num_q: Int, tp: Int](
    q: Binding[Int8, tp], qi_bias: Binding[Float32, tp],
    f_q: Binding[Float32, tp],
):
    for r in range(tp):
        fill_q_aux[head_dim, num_q](q[r], qi_bias[r], f_q[r])


@always_inline
def abs_f32(x: Float32) -> Float32:
    return -x if x < Float32(0) else x


def check_single_token(label: StringSlice, output: BF16Ptr, v: I8Ptr, vs: F32Ptr):
    var expected = BFloat16(Float32(Int(v[0])) * vs[0] / Float32(127.0)).cast[DType.float32]()
    var actual = output[0].cast[DType.float32]()
    var diff = abs_f32(actual - expected)
    var ok = diff < Float32(0.004)
    print(t"  {label}: actual={actual} expected={expected} diff={diff} ", "OK" if ok else "FAIL")


def has_nan(ptr: BF16Ptr, count: Int) -> Bool:
    for i in range(count):
        var v = ptr[i].cast[DType.float32]()
        if v != v:
            return True
    return False


def section_validation[P: BurstThreadPool, //, tp: Int](
    mut pools: List[P],
    sliding_q: Binding[Int8, tp],
    sliding_qi_bias: Binding[Float32, tp],
    sliding_f_q: Binding[Float32, tp],
    sliding_k: Binding[Int8, tp],
    sliding_ks: Binding[Float32, tp],
    sliding_v: Binding[Int8, tp],
    sliding_vs: Binding[Float32, tp],
    sliding_output: Binding[BFloat16, tp],
    sliding_partials: Binding[Float32, tp],
    full_q: Binding[Int8, tp],
    full_qi_bias: Binding[Float32, tp],
    full_f_q: Binding[Float32, tp],
    full_k: Binding[Int8, tp],
    full_ks: Binding[Float32, tp],
    full_v: Binding[Int8, tp],
    full_vs: Binding[Float32, tp],
    full_output: Binding[BFloat16, tp],
    full_partials: Binding[Float32, tp],
):
    print("\n=== Validation ===")
    comptime full_local_num_q = FULL_GLOBAL_NUM_Q // tp

    dispatch_bq_sliding_attention[tp=tp](
        sliding_q, sliding_qi_bias, sliding_f_q,
        sliding_k, sliding_ks, sliding_v, sliding_vs,
        sliding_output, sliding_partials, 0, pools)
    check_single_token("sliding seq=1", sliding_output[0], sliding_v[0], sliding_vs[0])

    dispatch_bq_full_attention[tp=tp](
        full_q, full_qi_bias, full_f_q,
        full_k, full_ks, full_v, full_vs,
        full_output, full_partials, 0, pools)
    check_single_token("full seq=1", full_output[0], full_v[0], full_vs[0])

    dispatch_bq_sliding_attention[tp=tp](
        sliding_q, sliding_qi_bias, sliding_f_q,
        sliding_k, sliding_ks, sliding_v, sliding_vs,
        sliding_output, sliding_partials, 63, pools)
    var sliding_bad = has_nan(
        sliding_output[0], SLIDING_NUM_Q * SLIDING_HEAD_DIM)

    dispatch_bq_full_attention[tp=tp](
        full_q, full_qi_bias, full_f_q,
        full_k, full_ks, full_v, full_vs,
        full_output, full_partials, 63, pools)
    var full_bad = has_nan(
        full_output[0], full_local_num_q * FULL_HEAD_DIM)

    print("  sliding seq=64 ", "FAIL: NaN detected" if sliding_bad else "OK (no NaN)")
    print("  full seq=64 ", "FAIL: NaN detected" if full_bad else "OK (no NaN)")


def section_sliding_sweep[P: BurstThreadPool, //, tp: Int](
    mut pools: List[P],
    q: Binding[Int8, tp],
    qi_bias: Binding[Float32, tp],
    f_q: Binding[Float32, tp],
    k: Binding[Int8, tp],
    ks: Binding[Float32, tp],
    v: Binding[Int8, tp],
    vs: Binding[Float32, tp],
    output: Binding[BFloat16, tp],
    partials: Binding[Float32, tp],
):
    print("\n=== Sliding decode sweep (BQ kernel + merge) ===")
    var sizes = InlineArray[Int, NUM_CTX_SIZES](fill=0)
    sizes[0] = 1; sizes[1] = 8; sizes[2] = 32; sizes[3] = 128
    sizes[4] = 256; sizes[5] = 512; sizes[6] = 1024; sizes[7] = 4096

    var samples = SampleBuffer(SAMPLES)
    for s in range(NUM_CTX_SIZES):
        var vl = sizes[s]
        if vl > SLIDING_WINDOW:
            continue
        var pos = vl - 1
        for _ in range(WARMUP):
            dispatch_bq_sliding_attention[tp=tp](
                q, qi_bias, f_q, k, ks, v, vs, output, partials, pos, pools)
            keep(output[0][0])

        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            dispatch_bq_sliding_attention[tp=tp](
                q, qi_bias, f_q, k, ks, v, vs, output, partials, pos, pools)
            var t1 = now_ns()
            var t_done = max_last_ts[tp=tp](pools)
            samples.push(t_done - t0, t1 - t0)
        keep(output[0][0])

        var ks_stats = compute_stats(samples.kernel_ns, samples.n)
        var ws = compute_stats(samples.wall_ns, samples.n)
        var kv_bytes = vl * SLIDING_KV_STRIDE * 2 * 2
        print_row(String(t"seq={vl}"), ks_stats, ws, kv_bytes)


def section_full_sweep[P: BurstThreadPool, //, tp: Int](
    mut pools: List[P],
    q: Binding[Int8, tp],
    qi_bias: Binding[Float32, tp],
    f_q: Binding[Float32, tp],
    k: Binding[Int8, tp],
    ks: Binding[Float32, tp],
    v: Binding[Int8, tp],
    vs: Binding[Float32, tp],
    output: Binding[BFloat16, tp],
    partials: Binding[Float32, tp],
):
    print("\n=== Full decode sweep (BQ kernel + context merge) ===")
    var sizes = InlineArray[Int, NUM_CTX_SIZES](fill=0)
    sizes[0] = 1; sizes[1] = 8; sizes[2] = 32; sizes[3] = 128
    sizes[4] = 256; sizes[5] = 512; sizes[6] = 1024; sizes[7] = 4096

    var samples = SampleBuffer(SAMPLES)
    for s in range(NUM_CTX_SIZES):
        var vl = sizes[s]
        if vl > MAX_SEQ:
            continue
        var pos = vl - 1
        for _ in range(WARMUP):
            dispatch_bq_full_attention[tp=tp](
                q, qi_bias, f_q, k, ks, v, vs, output, partials, pos, pools)
            keep(output[0][0])

        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            dispatch_bq_full_attention[tp=tp](
                q, qi_bias, f_q, k, ks, v, vs, output, partials, pos, pools)
            var t1 = now_ns()
            var t_done = max_last_ts[tp=tp](pools)
            samples.push(t_done - t0, t1 - t0)
        keep(output[0][0])

        var ks_stats = compute_stats(samples.kernel_ns, samples.n)
        var ws = compute_stats(samples.wall_ns, samples.n)
        var kv_bytes = vl * FULL_KV_STRIDE * 2 * 2
        print_row(String(t"seq={vl}"), ks_stats, ws, kv_bytes)


def run_all[P: BurstThreadPool, //, tp: Int](
    mut pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    comptime assert FULL_GLOBAL_NUM_Q % tp == 0, "full Q heads must divide tensor-parallel degree"
    comptime assert MAX_SEQ % tp == 0, "MAX_SEQ must divide tensor-parallel degree"
    var bases = arena_bases[tp](arenas)

    comptime sliding_pstride = flash_partial_stride[
        SLIDING_NUM_Q, SLIDING_HEAD_DIM,
    ]()
    comptime full_local_num_q = FULL_GLOBAL_NUM_Q // tp
    comptime full_pstride = flash_partial_stride[
        FULL_GLOBAL_NUM_Q, FULL_HEAD_DIM,
    ]()

    var sliding_q_ptr = arena_alloc_all[DType.int8, tp](
        arenas, SLIDING_NUM_Q * SLIDING_HEAD_DIM)
    var sliding_qi_bias_ptr = arena_alloc_all[DType.float32, tp](
        arenas, SLIDING_NUM_Q)
    var sliding_f_q_ptr = arena_alloc_all[DType.float32, tp](
        arenas, SLIDING_NUM_Q)
    var sliding_k_ptr = arena_alloc_all[DType.int8, tp](
        arenas, SLIDING_WINDOW * SLIDING_KV_STRIDE)
    var sliding_ks_ptr = arena_alloc_all[DType.float32, tp](
        arenas, SLIDING_WINDOW * SLIDING_NUM_KV)
    var sliding_v_ptr = arena_alloc_all[DType.int8, tp](
        arenas, SLIDING_WINDOW * SLIDING_KV_STRIDE)
    var sliding_vs_ptr = arena_alloc_all[DType.float32, tp](
        arenas, SLIDING_WINDOW * SLIDING_NUM_KV)
    var sliding_output_ptr = arena_alloc_all[DType.bfloat16, tp](
        arenas, SLIDING_NUM_Q * SLIDING_HEAD_DIM)
    var sliding_partials_ptr = arena_alloc_all[DType.float32, tp](
        arenas, MAX_WORKERS * sliding_pstride)

    var full_q_ptr = arena_alloc_all[DType.int8, tp](
        arenas, FULL_GLOBAL_NUM_Q * FULL_HEAD_DIM)
    var full_qi_bias_ptr = arena_alloc_all[DType.float32, tp](
        arenas, FULL_GLOBAL_NUM_Q)
    var full_f_q_ptr = arena_alloc_all[DType.float32, tp](
        arenas, FULL_GLOBAL_NUM_Q)
    var full_k_ptr = arena_alloc_all[DType.int8, tp](
        arenas, MAX_SEQ * FULL_KV_STRIDE)
    var full_ks_ptr = arena_alloc_all[DType.float32, tp](
        arenas, MAX_SEQ * FULL_NUM_KV)
    var full_v_ptr = arena_alloc_all[DType.int8, tp](
        arenas, MAX_SEQ * FULL_KV_STRIDE)
    var full_vs_ptr = arena_alloc_all[DType.float32, tp](
        arenas, MAX_SEQ * FULL_NUM_KV)
    var full_output_ptr = arena_alloc_all[DType.bfloat16, tp](
        arenas, full_local_num_q * FULL_HEAD_DIM)
    var full_partials_ptr = arena_alloc_all[DType.float32, tp](
        arenas, MAX_WORKERS * full_pstride)

    var sliding_q = Binding[Int8, tp](sliding_q_ptr, bases)
    var sliding_qi_bias = Binding[Float32, tp](sliding_qi_bias_ptr, bases)
    var sliding_f_q = Binding[Float32, tp](sliding_f_q_ptr, bases)
    var sliding_k = Binding[Int8, tp](sliding_k_ptr, bases)
    var sliding_ks = Binding[Float32, tp](sliding_ks_ptr, bases)
    var sliding_v = Binding[Int8, tp](sliding_v_ptr, bases)
    var sliding_vs = Binding[Float32, tp](sliding_vs_ptr, bases)
    var sliding_output = Binding[BFloat16, tp](sliding_output_ptr, bases)
    var sliding_partials = Binding[Float32, tp](sliding_partials_ptr, bases)

    var full_q = Binding[Int8, tp](full_q_ptr, bases)
    var full_qi_bias = Binding[Float32, tp](full_qi_bias_ptr, bases)
    var full_f_q = Binding[Float32, tp](full_f_q_ptr, bases)
    var full_k = Binding[Int8, tp](full_k_ptr, bases)
    var full_ks = Binding[Float32, tp](full_ks_ptr, bases)
    var full_v = Binding[Int8, tp](full_v_ptr, bases)
    var full_vs = Binding[Float32, tp](full_vs_ptr, bases)
    var full_output = Binding[BFloat16, tp](full_output_ptr, bases)
    var full_partials = Binding[Float32, tp](full_partials_ptr, bases)

    fill_i8_all[tp](sliding_q, SLIDING_NUM_Q * SLIDING_HEAD_DIM, 1)
    fill_i8_all[tp](sliding_k, SLIDING_WINDOW * SLIDING_KV_STRIDE, 2)
    fill_i8_all[tp](sliding_v, SLIDING_WINDOW * SLIDING_KV_STRIDE, 3)
    fill_scales_all[tp](
        sliding_ks, SLIDING_WINDOW * SLIDING_NUM_KV, Float32(0.18))
    fill_scales_all[tp](
        sliding_vs, SLIDING_WINDOW * SLIDING_NUM_KV, Float32(0.20))
    fill_q_aux_all[SLIDING_HEAD_DIM, SLIDING_NUM_Q, tp](
        sliding_q, sliding_qi_bias, sliding_f_q)

    fill_i8_all[tp](full_q, FULL_GLOBAL_NUM_Q * FULL_HEAD_DIM, 4)
    fill_i8_all[tp](full_k, MAX_SEQ * FULL_KV_STRIDE, 5)
    fill_i8_all[tp](full_v, MAX_SEQ * FULL_KV_STRIDE, 6)
    fill_scales_all[tp](
        full_ks, MAX_SEQ * FULL_NUM_KV, Float32(0.16))
    fill_scales_all[tp](
        full_vs, MAX_SEQ * FULL_NUM_KV, Float32(0.19))
    fill_q_aux_all[FULL_HEAD_DIM, FULL_GLOBAL_NUM_Q, tp](
        full_q, full_qi_bias, full_f_q)

    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    var cap = pools[0].get_capacity()
    print(t"pool capacity: {cap} workers")
    print(
        t"sliding: head_dim={SLIDING_HEAD_DIM} num_q={SLIDING_NUM_Q} "
        t"num_kv={SLIDING_NUM_KV} gqa={SLIDING_GQA_RATIO} "
        t"window={SLIDING_WINDOW}"
    )
    print(
        t"full: head_dim={FULL_HEAD_DIM} num_q={full_local_num_q} "
        t"num_kv={FULL_NUM_KV} gqa={FULL_GQA_RATIO} max_seq={MAX_SEQ}"
    )

    section_validation[tp=tp](
        pools,
        sliding_q, sliding_qi_bias, sliding_f_q,
        sliding_k, sliding_ks, sliding_v, sliding_vs,
        sliding_output, sliding_partials,
        full_q, full_qi_bias, full_f_q,
        full_k, full_ks, full_v, full_vs,
        full_output, full_partials,
    )
    section_sliding_sweep[tp=tp](
        pools, sliding_q, sliding_qi_bias, sliding_f_q,
        sliding_k, sliding_ks, sliding_v, sliding_vs,
        sliding_output, sliding_partials,
    )
    section_full_sweep[tp=tp](
        pools, full_q, full_qi_bias, full_f_q,
        full_k, full_ks, full_v, full_vs,
        full_output, full_partials,
    )


def main():
    var topo = NumaTopology()
    var tp = len(topo)

    print("ButterQuant attention benchmark (apples-apples decode kernel + merge)")
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
    def dispatch_bq_attention_tp[
        P: BurstThreadPool, //, degree: Int,
    ](var selected_pools: List[P]):
        run_all[tp=degree](selected_pools, arenas)

    with_topological_rank_dispatch[
        power_of_two_unrolling=3,
        dispatch=dispatch_bq_attention_tp,
    ](topo, "mode: isolated", "mode: spin-backoff")
