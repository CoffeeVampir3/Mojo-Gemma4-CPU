from std.collections import InlineArray
from std.memory import UnsafePointer
from std.time import perf_counter_ns
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from notstdcollections import HeapMoveArray
from kernels.helpers import Binding, ArenaBases
from kernels.rope import (
    rope_head, dispatch_rope_cache_write,
    init_rope_table, init_rope_table_partial_strided,
)


comptime ALIGNMENT = 64
comptime WARMUP = 5
comptime TRIALS = 12
comptime ITERS = 30

comptime HEAD_DIM_SLIDING = 256
comptime HEAD_DIM_FULL = 512
comptime HALF_SLIDING = 128
comptime HALF_FULL = 64
comptime Q_DIM_SLIDING = 4096
comptime KV_DIM_SLIDING = 2048
comptime Q_DIM_FULL = 8192
comptime KV_DIM_FULL = 1024
comptime MAX_POS = 4096
comptime SLIDING_WINDOW = 1024

comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]


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
        ptr[i] = BFloat16(Float32((i % 127) - 63) * 0.01)


def fill_pattern_all[tp: Int](
    ptrs: Binding[BFloat16, tp], count: Int,
):
    for r in range(tp):
        fill_pattern(ptrs[r], count)


def init_sliding_tables_all[tp: Int](
    cos_sl: F32Ptr, sin_sl: F32Ptr, bases: ArenaBases[tp],
):
    var cos = Binding[Float32, tp](cos_sl, bases)
    var sin = Binding[Float32, tp](sin_sl, bases)
    for r in range(tp):
        init_rope_table[HALF_SLIDING, MAX_POS](cos[r], sin[r], 10000.0)


def init_full_tables_all[tp: Int](
    cos_fl: F32Ptr, sin_fl: F32Ptr, bases: ArenaBases[tp],
):
    comptime LOCAL_ROWS = MAX_POS // tp
    var cos = Binding[Float32, tp](cos_fl, bases)
    var sin = Binding[Float32, tp](sin_fl, bases)
    for r in range(tp):
        init_rope_table_partial_strided[HALF_FULL, LOCAL_ROWS](
            cos[r], sin[r], 1000000.0, HEAD_DIM_FULL, r, tp)


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


def section_head_primitive(data: BF16Ptr, cos_sl: F32Ptr, sin_sl: F32Ptr,
                           cos_fl: F32Ptr, sin_fl: F32Ptr):
    print("\n=== rope_head primitive (single head) ===")

    for _ in range(WARMUP):
        rope_head[HALF_SLIDING, HALF_SLIDING](data, cos_sl, sin_sl)
    var best_sl = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS * 10):
            var t0 = Int(perf_counter_ns())
            rope_head[HALF_SLIDING, HALF_SLIDING](data, cos_sl, sin_sl)
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // (ITERS * 10)
        if avg < best_sl:
            best_sl = avg
    keep(data[0])
    print("  sliding (256 dim, full rot):  " + fmt_ns(best_sl)
        + "  (" + fmt_bw(HEAD_DIM_SLIDING * 2 + HALF_SLIDING * 4 * 2, best_sl) + " r+w+cos+sin)")

    for _ in range(WARMUP):
        rope_head[HALF_FULL, HEAD_DIM_FULL // 2](data, cos_fl, sin_fl)
    var best_fl = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS * 10):
            var t0 = Int(perf_counter_ns())
            rope_head[HALF_FULL, HEAD_DIM_FULL // 2](data, cos_fl, sin_fl)
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // (ITERS * 10)
        if avg < best_fl:
            best_fl = avg
    keep(data[0])
    print("  full (512 dim, 128 partial):  " + fmt_ns(best_fl)
        + "  (" + fmt_bw(HALF_FULL * 2 * 2 + HALF_FULL * 4 * 2, best_fl) + " rotated portion)")


def section_token_scaling(data: BF16Ptr, cos_sl: F32Ptr, sin_sl: F32Ptr):
    print("\n=== rope_head loop: head count scaling (sliding, single pos) ===")
    print("  heads | time         | bytes touched | BW")

    comptime NUM_SIZES = 5
    var head_counts = InlineArray[Int, NUM_SIZES](fill=0)
    head_counts[0] = 1
    head_counts[1] = 2
    head_counts[2] = 4
    head_counts[3] = 8
    head_counts[4] = 16

    for s in range(NUM_SIZES):
        var nh = head_counts[s]

        for _ in range(WARMUP):
            for h in range(nh):
                rope_head[HALF_SLIDING, HALF_SLIDING](
                    data + h * HEAD_DIM_SLIDING, cos_sl, sin_sl)

        var best = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS * 10):
                var t0 = Int(perf_counter_ns())
                for h in range(nh):
                    rope_head[HALF_SLIDING, HALF_SLIDING](
                        data + h * HEAD_DIM_SLIDING, cos_sl, sin_sl)
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // (ITERS * 10)
            if avg < best:
                best = avg
        keep(data[0])

        var data_bytes = nh * HEAD_DIM_SLIDING * 2
        var pad = "  " if nh < 10 else " "
        print("  " + String(nh) + pad
            + "    | " + fmt_ns(best)
            + " | " + String(data_bytes)
            + "         | " + fmt_bw(data_bytes * 2, best))


def measure_sliding_cache_write[
    P: BurstThreadPool, //, tp: Int,
](
    mut pools: HeapMoveArray[P],
    q: BF16Ptr, k_src: BF16Ptr, v_src: BF16Ptr,
    k_cache: BF16Ptr, v_cache: BF16Ptr,
    cos_sl: F32Ptr, sin_sl: F32Ptr,
    bases: ArenaBases[tp],
) -> Tuple[Int, Int]:
    comptime Q_ROWS = Q_DIM_SLIDING // tp
    comptime KV_ROWS = KV_DIM_SLIDING // tp
    comptime NUM_Q = Q_ROWS // HEAD_DIM_SLIDING
    comptime NUM_KV = KV_ROWS // HEAD_DIM_SLIDING
    comptime POS = 513
    var qs = Binding[BFloat16, tp](q, bases)
    var ks = Binding[BFloat16, tp](k_src, bases)
    var vs = Binding[BFloat16, tp](v_src, bases)
    var kc = Binding[BFloat16, tp](k_cache, bases)
    var vc = Binding[BFloat16, tp](v_cache, bases)
    var cos = Binding[Float32, tp](cos_sl, bases)
    var sin = Binding[Float32, tp](sin_sl, bases)

    for _ in range(WARMUP):
        dispatch_rope_cache_write[
            half=HALF_SLIDING, pair_stride=HEAD_DIM_SLIDING // 2,
            num_q=NUM_Q, num_kv=NUM_KV,
            head_dim=HEAD_DIM_SLIDING, kv_cache_stride=KV_ROWS,
            slot_mask=SLIDING_WINDOW - 1, cache_degree=1, tp=tp,
        ](qs, ks, vs, kc, vc, cos, sin, POS, 1, pools)

    var best_wall = Int(1 << 60)
    var best_kernel = Int(1 << 60)
    for _ in range(TRIALS):
        var wall_sum = 0
        var kernel_sum = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            dispatch_rope_cache_write[
                half=HALF_SLIDING, pair_stride=HEAD_DIM_SLIDING // 2,
                num_q=NUM_Q, num_kv=NUM_KV,
                head_dim=HEAD_DIM_SLIDING, kv_cache_stride=KV_ROWS,
                slot_mask=SLIDING_WINDOW - 1, cache_degree=1, tp=tp,
            ](qs, ks, vs, kc, vc, cos, sin, POS, 1, pools)
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
    keep(q[0])
    return (best_kernel, best_wall)


def measure_full_cache_write[
    P: BurstThreadPool, //, tp: Int,
](
    mut pools: HeapMoveArray[P],
    q: BF16Ptr, k_src: BF16Ptr, v_src: BF16Ptr,
    k_cache: BF16Ptr, v_cache: BF16Ptr,
    cos_fl: F32Ptr, sin_fl: F32Ptr,
    bases: ArenaBases[tp],
) -> Tuple[Int, Int]:
    comptime Q_ROWS = Q_DIM_FULL // tp
    comptime NUM_Q = Q_ROWS // HEAD_DIM_FULL
    comptime NUM_KV = KV_DIM_FULL // HEAD_DIM_FULL
    comptime POS = 513
    var owner = POS % tp
    var owner_bases = ArenaBases[tp].fill(bases[owner])
    var full_cos = Binding[Float32, tp](cos_fl, bases)[owner]
    var full_sin = Binding[Float32, tp](sin_fl, bases)[owner]

    var qs = Binding[BFloat16, tp](q, bases)
    var ks = Binding[BFloat16, tp](k_src, bases)
    var vs = Binding[BFloat16, tp](v_src, bases)
    var kc = Binding[BFloat16, tp](k_cache, bases)
    var vc = Binding[BFloat16, tp](v_cache, bases)
    var cos = Binding[Float32, tp](full_cos, owner_bases)
    var sin = Binding[Float32, tp](full_sin, owner_bases)

    for _ in range(WARMUP):
        dispatch_rope_cache_write[
            half=HALF_FULL, pair_stride=HEAD_DIM_FULL // 2,
            num_q=NUM_Q, num_kv=NUM_KV,
            head_dim=HEAD_DIM_FULL, kv_cache_stride=KV_DIM_FULL,
            slot_mask=-1, cache_degree=tp, tp=tp,
        ](qs, ks, vs, kc, vc, cos, sin, POS, 1, pools)

    var best_wall = Int(1 << 60)
    var best_kernel = Int(1 << 60)
    for _ in range(TRIALS):
        var wall_sum = 0
        var kernel_sum = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            dispatch_rope_cache_write[
                half=HALF_FULL, pair_stride=HEAD_DIM_FULL // 2,
                num_q=NUM_Q, num_kv=NUM_KV,
                head_dim=HEAD_DIM_FULL, kv_cache_stride=KV_DIM_FULL,
                slot_mask=-1, cache_degree=tp, tp=tp,
            ](qs, ks, vs, kc, vc, cos, sin, POS, 1, pools)
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
    keep(q[0])
    return (best_kernel, best_wall)


def section_model_cache_write[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    sliding_q: BF16Ptr, sliding_k: BF16Ptr, sliding_v: BF16Ptr,
    sliding_k_cache: BF16Ptr, sliding_v_cache: BF16Ptr,
    full_q: BF16Ptr, full_k: BF16Ptr, full_v: BF16Ptr,
    full_k_cache: BF16Ptr, full_v_cache: BF16Ptr,
    cos_sl: F32Ptr, sin_sl: F32Ptr, cos_fl: F32Ptr, sin_fl: F32Ptr,
    bases: ArenaBases[tp],
):
    print("\n=== dispatch_rope_cache_write model path (seq_len=1, TP="
        + String(tp) + ") ===")

    var sliding = measure_sliding_cache_write[tp=tp](
        pools, sliding_q, sliding_k, sliding_v,
        sliding_k_cache, sliding_v_cache, cos_sl, sin_sl, bases)
    var full = measure_full_cache_write[tp=tp](
        pools, full_q, full_k, full_v,
        full_k_cache, full_v_cache, cos_fl, sin_fl, bases)

    comptime SL_BYTES = (Q_DIM_SLIDING // tp + 2 * (KV_DIM_SLIDING // tp)) * 2
    comptime FL_BYTES = (Q_DIM_FULL // tp + 2 * KV_DIM_FULL) * 2
    print("  sliding: kernel=" + fmt_ns(sliding[0])
        + " | wall=" + fmt_ns(sliding[1])
        + "  (" + fmt_bw(SL_BYTES, sliding[0]) + " q/k/v touch)")
    print("  full:    kernel=" + fmt_ns(full[0])
        + " | wall=" + fmt_ns(full[1])
        + "  (" + fmt_bw(FL_BYTES, full[0]) + " q/k/v touch)")


def run_all[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
):
    comptime MAX_SEQ = 128
    comptime MAX_HEADS = 16
    comptime MAX_DATA = MAX_SEQ * MAX_HEADS * HEAD_DIM_FULL
    comptime SL_Q_ROWS = Q_DIM_SLIDING // tp
    comptime SL_KV_ROWS = KV_DIM_SLIDING // tp
    comptime FL_Q_ROWS = Q_DIM_FULL // tp
    comptime FL_ROPE_ROWS = MAX_POS // tp

    var bases = arena_bases[tp](arenas)
    var data = arena_alloc_all[DType.bfloat16, tp](arenas, MAX_DATA)
    var cos_sl = arena_alloc_all[DType.float32, tp](
        arenas, MAX_POS * HALF_SLIDING)
    var sin_sl = arena_alloc_all[DType.float32, tp](
        arenas, MAX_POS * HALF_SLIDING)
    var cos_fl = arena_alloc_all[DType.float32, tp](arenas, FL_ROPE_ROWS * HALF_FULL)
    var sin_fl = arena_alloc_all[DType.float32, tp](arenas, FL_ROPE_ROWS * HALF_FULL)

    var sliding_q = arena_alloc_all[DType.bfloat16, tp](arenas, SL_Q_ROWS)
    var sliding_k = arena_alloc_all[DType.bfloat16, tp](arenas, SL_KV_ROWS)
    var sliding_v = arena_alloc_all[DType.bfloat16, tp](arenas, SL_KV_ROWS)
    var sliding_k_cache = arena_alloc_all[DType.bfloat16, tp](
        arenas, SLIDING_WINDOW * SL_KV_ROWS)
    var sliding_v_cache = arena_alloc_all[DType.bfloat16, tp](
        arenas, SLIDING_WINDOW * SL_KV_ROWS)

    var full_q = arena_alloc_all[DType.bfloat16, tp](arenas, FL_Q_ROWS)
    var full_k = arena_alloc_all[DType.bfloat16, tp](arenas, KV_DIM_FULL)
    var full_v = arena_alloc_all[DType.bfloat16, tp](arenas, KV_DIM_FULL)
    var full_k_cache = arena_alloc_all[DType.bfloat16, tp](
        arenas, MAX_POS * KV_DIM_FULL)
    var full_v_cache = arena_alloc_all[DType.bfloat16, tp](
        arenas, MAX_POS * KV_DIM_FULL)

    fill_pattern_all[tp](Binding[BFloat16, tp](data, bases), MAX_DATA)
    fill_pattern_all[tp](
        Binding[BFloat16, tp](sliding_q, bases), SL_Q_ROWS)
    fill_pattern_all[tp](
        Binding[BFloat16, tp](sliding_k, bases), SL_KV_ROWS)
    fill_pattern_all[tp](
        Binding[BFloat16, tp](sliding_v, bases), SL_KV_ROWS)
    fill_pattern_all[tp](
        Binding[BFloat16, tp](full_q, bases), FL_Q_ROWS)
    fill_pattern_all[tp](
        Binding[BFloat16, tp](full_k, bases), KV_DIM_FULL)
    fill_pattern_all[tp](
        Binding[BFloat16, tp](full_v, bases), KV_DIM_FULL)
    init_sliding_tables_all[tp](cos_sl, sin_sl, bases)
    init_full_tables_all[tp](cos_fl, sin_fl, bases)

    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    var cap = pools[0].get_capacity()
    print("pool capacity: " + String(cap) + " workers")
    print("sliding: head_dim=256, half=128, theta=10000")
    print("full:    head_dim=512, rotary_half=64, theta=1000000")

    section_head_primitive(data, cos_sl, sin_sl, cos_fl, sin_fl)
    section_token_scaling(data, cos_sl, sin_sl)
    section_model_cache_write[tp=tp](
        pools,
        sliding_q, sliding_k, sliding_v, sliding_k_cache, sliding_v_cache,
        full_q, full_k, full_v, full_k_cache, full_v_cache,
        cos_sl, sin_sl, cos_fl, sin_fl, bases)


def main():
    var topo = NumaTopology()
    var tp = len(topo)

    print("RoPE kernel benchmark")
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
    def dispatch_rope_tp[
        P: BurstThreadPool, //, degree: Int,
    ](var selected_pools: HeapMoveArray[P]):
        run_all[tp=degree](selected_pools, arenas)

    with_topological_rank_dispatch[
        power_of_two_unrolling=3,
        dispatch=dispatch_rope_tp,
    ](topo, "mode: isolated", "mode: spin-backoff")
