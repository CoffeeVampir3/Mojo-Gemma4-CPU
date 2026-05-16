from std.collections import InlineArray
from std.memory import UnsafePointer
from std.time import perf_counter_ns
from std.benchmark import keep
from std.sys.info import simd_width_of

from numa import NumaArena, NumaTopology
from threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool
from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from kernels.full_attention import dispatch_full_attention, PARTIAL_STRIDE
from kernels.logsum_merge import dispatch_merge_context_flash_partials
from kernels.helpers import NumaPointerArray


comptime ALIGNMENT = 64
comptime WARMUP = 10
comptime TRIALS = 20
comptime ITERS = 30

comptime HEAD_DIM = 512
comptime GLOBAL_NUM_Q = 16
comptime NUM_KV = 2
comptime GLOBAL_GQA = GLOBAL_NUM_Q // NUM_KV
comptime KV_STRIDE = 1024
comptime MAX_SEQ = 4096
comptime MAX_WORKERS = 128

comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]


def arena_bases[tp: Int](
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
) -> InlineArray[Int, tp]:
    var bases = InlineArray[Int, tp](uninitialized=True)
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
        ptr[i] = Scalar[DType.bfloat16](Float32((i % 127) - 63) * 0.01)


def fill_pattern_all[tp: Int](
    ptrs: NumaPointerArray[DType.bfloat16, tp], count: Int,
):
    for r in range(tp):
        fill_pattern(ptrs[r], count)


def full_valid_lens[tp: Int](valid_len: Int) -> InlineArray[Int, tp]:
    var lens = InlineArray[Int, tp](uninitialized=True)
    if valid_len <= 0:
        for r in range(tp):
            lens[r] = 0
        return lens
    var last = valid_len - 1
    for r in range(tp):
        if r <= last % tp:
            lens[r] = last // tp + 1
        else:
            lens[r] = last // tp
    return lens


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


def section_validation[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P], q: BF16Ptr, k_cache: BF16Ptr, v_cache: BF16Ptr,
    output: BF16Ptr, partials: F32Ptr, bases: InlineArray[Int, tp],
):
    print("\n=== Validation (valid_len=64) ===")
    comptime VL = 64
    comptime LOCAL_NUM_Q = GLOBAL_NUM_Q // tp
    comptime PSTRIDE = PARTIAL_STRIDE[GLOBAL_NUM_Q, HEAD_DIM]

    var nw = dispatch_full_attention[
        head_dim=HEAD_DIM, num_q=GLOBAL_NUM_Q,
        gqa_ratio=GLOBAL_GQA, kv_stride=KV_STRIDE, tp=tp](
        NumaPointerArray[DType.bfloat16, tp](q, bases),
        NumaPointerArray[DType.bfloat16, tp](k_cache, bases),
        NumaPointerArray[DType.bfloat16, tp](v_cache, bases),
        NumaPointerArray[DType.float32, tp](partials, bases),
        full_valid_lens[tp](VL), pools)
    dispatch_merge_context_flash_partials[
        head_dim=HEAD_DIM, num_q=GLOBAL_NUM_Q, local_num_q=LOCAL_NUM_Q, tp=tp,
    ](
        NumaPointerArray[DType.bfloat16, tp](output, bases),
        NumaPointerArray[DType.float32, tp](partials, bases),
        PSTRIDE, nw, pools)

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
    output: BF16Ptr, partials: F32Ptr, bases: InlineArray[Int, tp],
):
    print("\n=== Context sweep (replicated Q + context-local attention + merge) ===")
    print("  valid_len | latency      | KV read  | BW")
    comptime LOCAL_NUM_Q = GLOBAL_NUM_Q // tp
    comptime PSTRIDE = PARTIAL_STRIDE[GLOBAL_NUM_Q, HEAD_DIM]

    var sizes = InlineArray[Int, 8](fill=0)
    sizes[0] = 4; sizes[1] = 32; sizes[2] = 64; sizes[3] = 128
    sizes[4] = 256; sizes[5] = 512; sizes[6] = 1024; sizes[7] = 4096

    for s in range(8):
        var vl = sizes[s]
        if vl > MAX_SEQ:
            continue

        for _ in range(WARMUP):
            var nw = dispatch_full_attention[
                head_dim=HEAD_DIM, num_q=GLOBAL_NUM_Q,
                gqa_ratio=GLOBAL_GQA, kv_stride=KV_STRIDE, tp=tp](
                NumaPointerArray[DType.bfloat16, tp](q, bases),
                NumaPointerArray[DType.bfloat16, tp](k_cache, bases),
                NumaPointerArray[DType.bfloat16, tp](v_cache, bases),
                NumaPointerArray[DType.float32, tp](partials, bases),
                full_valid_lens[tp](vl), pools)
            dispatch_merge_context_flash_partials[
                head_dim=HEAD_DIM, num_q=GLOBAL_NUM_Q,
                local_num_q=LOCAL_NUM_Q, tp=tp,
            ](
                NumaPointerArray[DType.bfloat16, tp](output, bases),
                NumaPointerArray[DType.float32, tp](partials, bases),
                PSTRIDE, nw, pools)
            keep(output[0])

        var best = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                var nw = dispatch_full_attention[
                    head_dim=HEAD_DIM, num_q=GLOBAL_NUM_Q,
                    gqa_ratio=GLOBAL_GQA, kv_stride=KV_STRIDE, tp=tp](
                    NumaPointerArray[DType.bfloat16, tp](q, bases),
                    NumaPointerArray[DType.bfloat16, tp](k_cache, bases),
                    NumaPointerArray[DType.bfloat16, tp](v_cache, bases),
                    NumaPointerArray[DType.float32, tp](partials, bases),
                    full_valid_lens[tp](vl), pools)
                dispatch_merge_context_flash_partials[
                    head_dim=HEAD_DIM, num_q=GLOBAL_NUM_Q,
                    local_num_q=LOCAL_NUM_Q, tp=tp,
                ](
                    NumaPointerArray[DType.bfloat16, tp](output, bases),
                    NumaPointerArray[DType.float32, tp](partials, bases),
                    PSTRIDE, nw, pools)
                elapsed += Int(perf_counter_ns()) - t0
            keep(output[0])
            var avg = elapsed // ITERS
            if avg < best:
                best = avg

        var kv_bytes = vl * KV_STRIDE * 2 * 2
        var pad = "   " if vl < 10 else "  " if vl < 100 else " " if vl < 1000 else ""
        print("  " + String(vl) + pad
            + "     | " + fmt_ns(best)
            + " | " + String(kv_bytes // 1024) + " KB"
            + "  | " + fmt_bw(kv_bytes, best))


def run_all[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
):
    var bases = arena_bases[tp](arenas)
    comptime LOCAL_NUM_Q = GLOBAL_NUM_Q // tp
    comptime PSTRIDE = PARTIAL_STRIDE[GLOBAL_NUM_Q, HEAD_DIM]
    var q = arena_alloc_all[DType.bfloat16, tp](arenas, GLOBAL_NUM_Q * HEAD_DIM)
    var k_cache = arena_alloc_all[DType.bfloat16, tp](arenas, MAX_SEQ * KV_STRIDE)
    var v_cache = arena_alloc_all[DType.bfloat16, tp](arenas, MAX_SEQ * KV_STRIDE)
    var output = arena_alloc_all[DType.bfloat16, tp](arenas, LOCAL_NUM_Q * HEAD_DIM)
    var partials = arena_alloc_all[DType.float32, tp](arenas, MAX_WORKERS * PSTRIDE)

    fill_pattern_all[tp](
        NumaPointerArray[DType.bfloat16, tp](q, bases), GLOBAL_NUM_Q * HEAD_DIM)
    fill_pattern_all[tp](
        NumaPointerArray[DType.bfloat16, tp](k_cache, bases), MAX_SEQ * KV_STRIDE)
    fill_pattern_all[tp](
        NumaPointerArray[DType.bfloat16, tp](v_cache, bases), MAX_SEQ * KV_STRIDE)
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

    if topo.has_isolation():
        print("mode: isolated")
        var pools = HeapMoveArray[IsolatedBurstPool[]](tp)
        for i in range(tp):
            pools.push(IsolatedBurstPool[].for_rank(topo, i))
            print("  node " + String(topo[i]) + ": "
                + String(pools[i].get_capacity()) + " workers")
        print("")
        if tp == 1:
            run_all[tp=1](pools, arenas)
        elif tp == 2:
            run_all[tp=2](pools, arenas)
        elif tp == 4:
            run_all[tp=4](pools, arenas)
        else:
            print("unsupported tp=" + String(tp))
    else:
        print("mode: spin-backoff")
        var pools = HeapMoveArray[BurstPool[]](tp)
        for i in range(tp):
            pools.push(BurstPool[].for_rank(topo, i))
            print("  node " + String(topo[i]) + ": "
                + String(pools[i].get_capacity()) + " workers")
        print("")
        if tp == 1:
            run_all[tp=1](pools, arenas)
        elif tp == 2:
            run_all[tp=2](pools, arenas)
        elif tp == 4:
            run_all[tp=4](pools, arenas)
        else:
            print("unsupported tp=" + String(tp))
