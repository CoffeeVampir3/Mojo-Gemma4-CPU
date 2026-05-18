from std.collections import InlineArray
from std.memory import UnsafePointer
from std.time import perf_counter_ns
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from notstdcollections import HeapMoveArray
from kernels.kv_tiled_attention import dispatch_sliding_attention, FlashDecodeKernel
from kernels.logsum_merge import dispatch_merge_flash_partials
from kernels.helpers import Binding, ArenaBases


comptime ALIGNMENT = 64
comptime WARMUP = 10
comptime TRIALS = 20
comptime ITERS = 50

comptime HEAD_DIM = 256
comptime NUM_Q = 4
comptime NUM_KV = 2
comptime GQA_RATIO = 2
comptime KV_STRIDE = 512
comptime WINDOW = 4096
comptime MAX_WORKERS = 128

comptime NUM_CTX_SIZES = 8

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


def fmt_ns(ns: Int) -> String:
    if ns < 1000:
        return String(ns) + " ns"
    elif ns < 1_000_000:
        return String(ns // 1000) + "." + String((ns % 1000) // 100) + " us"
    else:
        return String(ns // 1_000_000) + "." + String((ns % 1_000_000) // 100000) + " ms"


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


def section_context_sweep[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    q: Binding[BFloat16, tp],
    k_cache: Binding[BFloat16, tp],
    v_cache: Binding[BFloat16, tp],
    output: Binding[BFloat16, tp],
    partials: Binding[Float32, tp],
):
    print("\n=== Context sweep (dispatch_sliding_attention) ===")
    print("  valid_len | kernel time  | wall time    | KV read  | BW")

    var sizes = InlineArray[Int, NUM_CTX_SIZES](fill=0)
    sizes[0] = 1; sizes[1] = 8; sizes[2] = 32; sizes[3] = 128
    sizes[4] = 256; sizes[5] = 512; sizes[6] = 1024; sizes[7] = 4096

    for s in range(NUM_CTX_SIZES):
        var vl = sizes[s]
        if vl > WINDOW:
            continue
        var pos = vl - 1

        for _ in range(WARMUP):
            var nw = dispatch_sliding_attention[
                head_dim=HEAD_DIM, num_q=NUM_Q,
                gqa_ratio=GQA_RATIO, kv_stride=KV_STRIDE, window=WINDOW,
                tp=tp](q, k_cache, v_cache, partials, pos, vl, pools)
            dispatch_merge_flash_partials[HEAD_DIM, NUM_Q, tp=tp](
                output, partials, nw, pools)
            keep(output[0][0])

        var best_wall = Int(1 << 60)
        var best_kernel = Int(1 << 60)
        for _ in range(TRIALS):
            var wall_sum = 0
            var kernel_sum = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                var nw = dispatch_sliding_attention[
                    head_dim=HEAD_DIM, num_q=NUM_Q,
                    gqa_ratio=GQA_RATIO, kv_stride=KV_STRIDE, window=WINDOW,
                    tp=tp](q, k_cache, v_cache, partials, pos, vl, pools)
                dispatch_merge_flash_partials[HEAD_DIM, NUM_Q, tp=tp](
                    output, partials, nw, pools)
                var t1 = Int(perf_counter_ns())
                var t_done = max_last_ts[tp=tp](pools)
                wall_sum += t1 - t0
                kernel_sum += t_done - t0
            keep(output[0][0])
            var avg_wall = wall_sum // ITERS
            var avg_kernel = kernel_sum // ITERS
            if avg_wall < best_wall:
                best_wall = avg_wall
            if avg_kernel < best_kernel:
                best_kernel = avg_kernel

        var kv_bytes = vl * KV_STRIDE * 2 * 2
        var pad = "   " if vl < 10 else "  " if vl < 100 else " " if vl < 1000 else ""
        print("  " + String(vl) + pad
            + "     | " + fmt_ns(best_kernel)
            + " | " + fmt_ns(best_wall)
            + " | " + String(kv_bytes // 1024) + " KB"
            + "  | " + fmt_bw(kv_bytes, best_kernel))


def section_validation[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    q: Binding[BFloat16, tp],
    k_cache: Binding[BFloat16, tp],
    v_cache: Binding[BFloat16, tp],
    output: Binding[BFloat16, tp],
    partials: Binding[Float32, tp],
):
    print("\n=== Validation (valid_len=64) ===")
    comptime VL = 64
    var pos = VL - 1

    var nw = dispatch_sliding_attention[
        head_dim=HEAD_DIM, num_q=NUM_Q,
        gqa_ratio=GQA_RATIO, kv_stride=KV_STRIDE, window=WINDOW,
        tp=tp](q, k_cache, v_cache, partials, pos, VL, pools)
    dispatch_merge_flash_partials[HEAD_DIM, NUM_Q, tp=tp](
        output, partials, nw, pools)

    var out0 = output[0]
    print("  output[0..3]: "
        + String(out0[0].cast[DType.float32]()) + " "
        + String(out0[1].cast[DType.float32]()) + " "
        + String(out0[2].cast[DType.float32]()) + " "
        + String(out0[3].cast[DType.float32]()))
    var ok = True
    for i in range(NUM_Q * HEAD_DIM):
        var v = out0[i].cast[DType.float32]()
        if v != v:
            ok = False
            break
    print("  " + ("OK (no NaN)" if ok else "FAIL: NaN detected"))


def run_all[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
):
    comptime flash_stride = FlashDecodeKernel[
        HEAD_DIM, NUM_Q, GQA_RATIO, KV_STRIDE, WINDOW].PARTIAL_STRIDE
    var bases = arena_bases[tp](arenas)

    var q_ptr = arena_alloc_all[DType.bfloat16, tp](arenas, NUM_Q * HEAD_DIM)
    var k_cache_ptr = arena_alloc_all[DType.bfloat16, tp](arenas, WINDOW * KV_STRIDE)
    var v_cache_ptr = arena_alloc_all[DType.bfloat16, tp](arenas, WINDOW * KV_STRIDE)
    var output_ptr = arena_alloc_all[DType.bfloat16, tp](arenas, NUM_Q * HEAD_DIM)
    var partials_ptr = arena_alloc_all[DType.float32, tp](
        arenas, MAX_WORKERS * flash_stride)

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
    print("pool capacity: " + String(cap) + " workers")
    print("head_dim=" + String(HEAD_DIM) + " num_q=" + String(NUM_Q)
        + " num_kv=" + String(NUM_KV) + " gqa=" + String(GQA_RATIO)
        + " window=" + String(WINDOW))

    section_validation[tp=tp](pools, q, k_cache, v_cache, output, partials)
    section_context_sweep[tp=tp](pools, q, k_cache, v_cache, output, partials)


def main():
    var topo = NumaTopology()
    var tp = len(topo)

    print("Sliding attention benchmark (dispatch_sliding_attention)")
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
    def dispatch_sliding_attention_tp[
        P: BurstThreadPool, //, degree: Int,
    ](var selected_pools: HeapMoveArray[P]):
        run_all[tp=degree](selected_pools, arenas)

    with_topological_rank_dispatch[
        power_of_two_unrolling=3,
        dispatch=dispatch_sliding_attention_tp,
    ](topo, "mode: isolated", "mode: spin-backoff")
