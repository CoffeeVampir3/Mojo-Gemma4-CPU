from std.collections import InlineArray
from std.memory import UnsafePointer
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from notstdcollections import HeapMoveArray
from kernels.flash_attention_prefill import (
    dispatch_flash_prefill_sliding, dispatch_flash_prefill_full,
    FlashPrefillSlidingKernel, FlashPrefillFullKernel,
)
from kernels.attention_ops import flash_partial_stride
from kernels.helpers import Binding, ArenaBases
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, print_row, max_last_ts, now_ns,
    DEFAULT_SAMPLES,
)


comptime ALIGNMENT = 64
comptime WARMUP = 2
comptime SAMPLES = 3
comptime MAX_WORKERS = 128

comptime SLIDING_HEAD_DIM = 256
comptime SLIDING_NUM_Q = 4
comptime SLIDING_NUM_KV = 2
comptime SLIDING_GQA = SLIDING_NUM_Q // SLIDING_NUM_KV
comptime SLIDING_KV_STRIDE = SLIDING_NUM_KV * SLIDING_HEAD_DIM
comptime SLIDING_WINDOW = 4096
comptime SLIDING_CHUNK_SIZE = 4096
comptime SLIDING_MAX_SEQ = 16384

comptime FULL_HEAD_DIM = 512
comptime FULL_NUM_Q = 16
comptime FULL_NUM_KV = 2
comptime FULL_GQA = FULL_NUM_Q // FULL_NUM_KV
comptime FULL_KV_STRIDE = FULL_NUM_KV * FULL_HEAD_DIM
comptime FULL_CHUNK_SIZE = 512
comptime FULL_MAX_SEQ = 4096

comptime NUM_SLIDING_SIZES = 6
comptime NUM_FULL_SIZES = 5

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


def section_sliding_sweep[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    q: Binding[BFloat16, tp],
    k_cache: Binding[BFloat16, tp],
    v_cache: Binding[BFloat16, tp],
    output: Binding[BFloat16, tp],
    worker_scratch: Binding[Float32, tp],
):
    print("\n=== Sliding-window prefill sweep "
        + "(chunk_size=" + String(SLIDING_CHUNK_SIZE) + ") ===")

    var sizes = InlineArray[Int, NUM_SLIDING_SIZES](fill=0)
    sizes[0] = 64; sizes[1] = 256; sizes[2] = 1024
    sizes[3] = 4096; sizes[4] = 8192; sizes[5] = 16384

    var samples = SampleBuffer(SAMPLES)

    for s in range(NUM_SLIDING_SIZES):
        var seq_len = sizes[s]
        if seq_len > SLIDING_MAX_SEQ:
            continue

        for _ in range(WARMUP):
            dispatch_flash_prefill_sliding[
                head_dim=SLIDING_HEAD_DIM, num_q=SLIDING_NUM_Q,
                gqa_ratio=SLIDING_GQA, kv_stride=SLIDING_KV_STRIDE,
                window=SLIDING_WINDOW, tp=tp,
                chunk_size=SLIDING_CHUNK_SIZE,
                max_worker_count=MAX_WORKERS,
            ](q, k_cache, v_cache, output, worker_scratch,
              0, seq_len, pools)
            keep(output[0][0])

        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            dispatch_flash_prefill_sliding[
                head_dim=SLIDING_HEAD_DIM, num_q=SLIDING_NUM_Q,
                gqa_ratio=SLIDING_GQA, kv_stride=SLIDING_KV_STRIDE,
                window=SLIDING_WINDOW, tp=tp,
                chunk_size=SLIDING_CHUNK_SIZE,
                max_worker_count=MAX_WORKERS,
            ](q, k_cache, v_cache, output, worker_scratch,
              0, seq_len, pools)
            var t1 = now_ns()
            var t_done = max_last_ts[tp=tp](pools)
            samples.push(t_done - t0, t1 - t0)
        keep(output[0][0])

        var ks = compute_stats(samples.kernel_ns, samples.n)
        var ws = compute_stats(samples.wall_ns, samples.n)
        var per_q_kv = seq_len if seq_len < SLIDING_WINDOW else SLIDING_WINDOW
        var kv_bytes = seq_len * per_q_kv * SLIDING_KV_STRIDE * 2
        print_row("seq=" + String(seq_len), ks, ws, kv_bytes)


def section_full_sweep[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    q: Binding[BFloat16, tp],
    k_cache: Binding[BFloat16, tp],
    v_cache: Binding[BFloat16, tp],
    output: Binding[BFloat16, tp],
    partials_scratch: Binding[Float32, tp],
):
    print("\n=== Full-attention prefill sweep "
        + "(chunk_size=" + String(FULL_CHUNK_SIZE) + ") ===")
    comptime LOCAL_NUM_Q = FULL_NUM_Q // tp

    var sizes = InlineArray[Int, NUM_FULL_SIZES](fill=0)
    sizes[0] = 64; sizes[1] = 256; sizes[2] = 1024
    sizes[3] = 2048; sizes[4] = 4096

    var samples = SampleBuffer(SAMPLES)

    for s in range(NUM_FULL_SIZES):
        var seq_len = sizes[s]
        if seq_len > FULL_MAX_SEQ:
            continue

        for _ in range(WARMUP):
            dispatch_flash_prefill_full[
                head_dim=FULL_HEAD_DIM, num_q=FULL_NUM_Q,
                local_num_q=LOCAL_NUM_Q, gqa_ratio=FULL_GQA,
                kv_stride=FULL_KV_STRIDE, tp=tp,
                chunk_size=FULL_CHUNK_SIZE,
                max_worker_count=MAX_WORKERS,
            ](q, k_cache, v_cache, output, partials_scratch,
              0, seq_len, pools)
            keep(output[0][0])

        samples.clear()
        for _ in range(SAMPLES):
            var t0 = now_ns()
            dispatch_flash_prefill_full[
                head_dim=FULL_HEAD_DIM, num_q=FULL_NUM_Q,
                local_num_q=LOCAL_NUM_Q, gqa_ratio=FULL_GQA,
                kv_stride=FULL_KV_STRIDE, tp=tp,
                chunk_size=FULL_CHUNK_SIZE,
                max_worker_count=MAX_WORKERS,
            ](q, k_cache, v_cache, output, partials_scratch,
              0, seq_len, pools)
            var t1 = now_ns()
            var t_done = max_last_ts[tp=tp](pools)
            samples.push(t_done - t0, t1 - t0)
        keep(output[0][0])

        var ks = compute_stats(samples.kernel_ns, samples.n)
        var ws = compute_stats(samples.wall_ns, samples.n)
        # Per-rank KV bytes: causal triangle / degree.
        # Average local KV scan per Q ≈ seq_len / (2 * tp); total scans = seq_len.
        var per_rank_kv = (seq_len * (seq_len + 1) // 2) // tp
        var kv_bytes = per_rank_kv * FULL_KV_STRIDE * 2
        print_row("seq=" + String(seq_len), ks, ws, kv_bytes)


def section_validation_sliding[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    q: Binding[BFloat16, tp],
    k_cache: Binding[BFloat16, tp],
    v_cache: Binding[BFloat16, tp],
    output: Binding[BFloat16, tp],
    worker_scratch: Binding[Float32, tp],
):
    print("\n=== Validation (sliding, seq_len=64) ===")
    comptime SL = 64

    dispatch_flash_prefill_sliding[
        head_dim=SLIDING_HEAD_DIM, num_q=SLIDING_NUM_Q,
        gqa_ratio=SLIDING_GQA, kv_stride=SLIDING_KV_STRIDE,
        window=SLIDING_WINDOW, tp=tp,
        chunk_size=SLIDING_CHUNK_SIZE,
        max_worker_count=MAX_WORKERS,
    ](q, k_cache, v_cache, output, worker_scratch, 0, SL, pools)

    var out0 = output[0]
    print("  output[0..3]: "
        + String(out0[0].cast[DType.float32]()) + " "
        + String(out0[1].cast[DType.float32]()) + " "
        + String(out0[2].cast[DType.float32]()) + " "
        + String(out0[3].cast[DType.float32]()))
    var ok = True
    for i in range(SL * SLIDING_NUM_Q * SLIDING_HEAD_DIM):
        var v = out0[i].cast[DType.float32]()
        if v != v:
            ok = False
            break
    print("  " + ("OK (no NaN)" if ok else "FAIL: NaN detected"))


def section_validation_full[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    q: Binding[BFloat16, tp],
    k_cache: Binding[BFloat16, tp],
    v_cache: Binding[BFloat16, tp],
    output: Binding[BFloat16, tp],
    partials_scratch: Binding[Float32, tp],
):
    print("\n=== Validation (full, seq_len=64) ===")
    comptime SL = 64
    comptime LOCAL_NUM_Q = FULL_NUM_Q // tp
    comptime OUT_STRIDE = LOCAL_NUM_Q * FULL_HEAD_DIM
    comptime PSTRIDE = flash_partial_stride[FULL_NUM_Q, FULL_HEAD_DIM]()
    comptime M_OFF = FULL_NUM_Q * FULL_HEAD_DIM
    comptime L_OFF = M_OFF + FULL_NUM_Q

    dispatch_flash_prefill_full[
        head_dim=FULL_HEAD_DIM, num_q=FULL_NUM_Q,
        local_num_q=LOCAL_NUM_Q, gqa_ratio=FULL_GQA,
        kv_stride=FULL_KV_STRIDE, tp=tp,
        chunk_size=FULL_CHUNK_SIZE,
        max_worker_count=MAX_WORKERS,
    ](q, k_cache, v_cache, output, partials_scratch, 0, SL, pools)

    var out0 = output[0]
    print("  output[0..3]: "
        + String(out0[0].cast[DType.float32]()) + " "
        + String(out0[1].cast[DType.float32]()) + " "
        + String(out0[2].cast[DType.float32]()) + " "
        + String(out0[3].cast[DType.float32]()))

    var first_nan = -1
    for i in range(SL * OUT_STRIDE):
        var v = out0[i].cast[DType.float32]()
        if v != v:
            first_nan = i
            break

    if first_nan < 0:
        print("  OK (no NaN)")
        return

    var nan_tok = first_nan // OUT_STRIDE
    var nan_in_tok = first_nan % OUT_STRIDE
    var nan_local_h = nan_in_tok // FULL_HEAD_DIM
    var nan_lane = nan_in_tok % FULL_HEAD_DIM
    print("  FAIL: first NaN at flat=" + String(first_nan)
        + " token=" + String(nan_tok)
        + " local_h=" + String(nan_local_h)
        + " lane=" + String(nan_lane))

    var nan_global_h = nan_local_h
    print("  global_h = q_rank(0) * local(" + String(LOCAL_NUM_Q)
        + ") + " + String(nan_local_h) + " = " + String(nan_global_h))

    print("  per-rank partial m/l at (t=" + String(nan_tok)
        + ", global_h=" + String(nan_global_h) + "):")
    for r in range(tp):
        var pr = partials_scratch[r]
        var slot = pr + nan_tok * PSTRIDE
        var m_val = (slot + M_OFF + nan_global_h)[]
        var l_val = (slot + L_OFF + nan_global_h)[]
        var acc0 = (slot + nan_global_h * FULL_HEAD_DIM)[]
        print("    rank=" + String(r)
            + " m=" + String(m_val)
            + " l=" + String(l_val)
            + " acc[0]=" + String(acc0))


def run_sliding[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
):
    comptime SCRATCH_STRIDE = FlashPrefillSlidingKernel[
        SLIDING_HEAD_DIM, SLIDING_NUM_Q, SLIDING_GQA,
        SLIDING_KV_STRIDE, SLIDING_WINDOW,
    ].SCRATCH_STRIDE
    var bases = arena_bases[tp](arenas)

    var q_ptr = arena_alloc_all[DType.bfloat16, tp](
        arenas, SLIDING_MAX_SEQ * SLIDING_NUM_Q * SLIDING_HEAD_DIM)
    var k_cache_ptr = arena_alloc_all[DType.bfloat16, tp](
        arenas, SLIDING_WINDOW * SLIDING_KV_STRIDE)
    var v_cache_ptr = arena_alloc_all[DType.bfloat16, tp](
        arenas, SLIDING_WINDOW * SLIDING_KV_STRIDE)
    var output_ptr = arena_alloc_all[DType.bfloat16, tp](
        arenas, SLIDING_MAX_SEQ * SLIDING_NUM_Q * SLIDING_HEAD_DIM)
    var worker_scratch_ptr = arena_alloc_all[DType.float32, tp](
        arenas, MAX_WORKERS * SCRATCH_STRIDE)

    var q = Binding[BFloat16, tp](q_ptr, bases)
    var k_cache = Binding[BFloat16, tp](k_cache_ptr, bases)
    var v_cache = Binding[BFloat16, tp](v_cache_ptr, bases)
    var output = Binding[BFloat16, tp](output_ptr, bases)
    var worker_scratch = Binding[Float32, tp](worker_scratch_ptr, bases)

    fill_pattern_all[tp](
        q, SLIDING_MAX_SEQ * SLIDING_NUM_Q * SLIDING_HEAD_DIM)
    fill_pattern_all[tp](k_cache, SLIDING_WINDOW * SLIDING_KV_STRIDE)
    fill_pattern_all[tp](v_cache, SLIDING_WINDOW * SLIDING_KV_STRIDE)

    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    print("sliding: head_dim=" + String(SLIDING_HEAD_DIM)
        + " num_q=" + String(SLIDING_NUM_Q)
        + " num_kv=" + String(SLIDING_NUM_KV)
        + " gqa=" + String(SLIDING_GQA)
        + " window=" + String(SLIDING_WINDOW))

    section_validation_sliding[tp=tp](
        pools, q, k_cache, v_cache, output, worker_scratch)
    section_sliding_sweep[tp=tp](
        pools, q, k_cache, v_cache, output, worker_scratch)


def run_full[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
):
    comptime PSTRIDE = flash_partial_stride[FULL_NUM_Q, FULL_HEAD_DIM]()
    comptime LOCAL_NUM_Q = FULL_NUM_Q // tp
    var bases = arena_bases[tp](arenas)

    var q_ptr = arena_alloc_all[DType.bfloat16, tp](
        arenas, FULL_MAX_SEQ * FULL_NUM_Q * FULL_HEAD_DIM)
    var k_cache_ptr = arena_alloc_all[DType.bfloat16, tp](
        arenas, FULL_MAX_SEQ * FULL_KV_STRIDE)
    var v_cache_ptr = arena_alloc_all[DType.bfloat16, tp](
        arenas, FULL_MAX_SEQ * FULL_KV_STRIDE)
    var output_ptr = arena_alloc_all[DType.bfloat16, tp](
        arenas, FULL_MAX_SEQ * LOCAL_NUM_Q * FULL_HEAD_DIM)
    var partials_ptr = arena_alloc_all[DType.float32, tp](
        arenas, FULL_CHUNK_SIZE * PSTRIDE)

    var q = Binding[BFloat16, tp](q_ptr, bases)
    var k_cache = Binding[BFloat16, tp](k_cache_ptr, bases)
    var v_cache = Binding[BFloat16, tp](v_cache_ptr, bases)
    var output = Binding[BFloat16, tp](output_ptr, bases)
    var partials_scratch = Binding[Float32, tp](partials_ptr, bases)

    fill_pattern_all[tp](
        q, FULL_MAX_SEQ * FULL_NUM_Q * FULL_HEAD_DIM)
    fill_pattern_all[tp](k_cache, FULL_MAX_SEQ * FULL_KV_STRIDE)
    fill_pattern_all[tp](v_cache, FULL_MAX_SEQ * FULL_KV_STRIDE)

    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    print("full: head_dim=" + String(FULL_HEAD_DIM)
        + " num_q=" + String(FULL_NUM_Q)
        + " local_num_q=" + String(LOCAL_NUM_Q)
        + " num_kv=" + String(FULL_NUM_KV)
        + " gqa=" + String(FULL_GQA)
        + " max_seq=" + String(FULL_MAX_SEQ))

    section_validation_full[tp=tp](
        pools, q, k_cache, v_cache, output, partials_scratch)
    section_full_sweep[tp=tp](
        pools, q, k_cache, v_cache, output, partials_scratch)


def run_all[P: BurstThreadPool, //, tp: Int](
    var pools: HeapMoveArray[P],
    mut arenas_sliding: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
    mut arenas_full: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
):
    var cap = pools[0].get_capacity()
    print("pool capacity: " + String(cap) + " workers")
    run_sliding[tp=tp](pools, arenas_sliding)
    run_full[tp=tp](pools, arenas_full)


def main():
    var topo = NumaTopology()
    var tp = len(topo)

    print("Flash-attention prefill benchmark")
    print(String(tp) + " NUMA node(s), "
        + String(len(topo.isolated_cpus)) + " isolated cpus\n")

    comptime ARENA_BYTES = 512 * 1024 * 1024
    var arenas_sliding = HeapMoveArray[
        NumaArena[alignment=ALIGNMENT]](tp)
    var arenas_full = HeapMoveArray[
        NumaArena[alignment=ALIGNMENT]](tp)
    for i in range(tp):
        arenas_sliding.push(
            NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas_sliding[i]:
            print("arena alloc failed on node", topo[i])
            return
        arenas_full.push(
            NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas_full[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_flash_prefill_tp[
        P: BurstThreadPool, //, degree: Int,
    ](var selected_pools: HeapMoveArray[P]):
        run_all[tp=degree](
            selected_pools^, arenas_sliding, arenas_full)

    with_topological_rank_dispatch[
        power_of_two_unrolling=3,
        dispatch=dispatch_flash_prefill_tp,
    ](topo, "mode: isolated", "mode: spin-backoff")
