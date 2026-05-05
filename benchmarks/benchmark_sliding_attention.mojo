from std.collections import InlineArray
from std.memory import UnsafePointer
from std.time import perf_counter_ns
from std.benchmark import keep
from std.sys.info import simd_width_of

from numa import NumaArena, NumaInfo, NumaTopology
from threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool
from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from kernels.kv_tiled_attention import dispatch_sliding_attention, FLASH_PARTIAL_STRIDE
from kernels.logsum_merge import merge_flash_partials


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

comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]


def arena_alloc[dtype: DType](
    mut arena: NumaArena[alignment=ALIGNMENT], count: Int,
) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
    var ptr = arena.alloc[Scalar[dtype]](count)
    if not ptr:
        print("arena alloc failed for", count, "elements")
        return UnsafePointer[Scalar[dtype], MutAnyOrigin].unsafe_dangling()
    return ptr.value()


def fill_pattern(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = Scalar[DType.bfloat16](Float32((i % 127) - 63) * 0.01)


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


def section_context_sweep[P: BurstThreadPool](
    mut pool: P, q: BF16Ptr, k_cache: BF16Ptr, v_cache: BF16Ptr,
    output: BF16Ptr, partials: F32Ptr,
):
    print("\n=== Context sweep (dispatch_sliding_attention) ===")
    print("  valid_len | latency      | KV read  | BW")

    var sizes = InlineArray[Int, 8](fill=0)
    sizes[0] = 1; sizes[1] = 8; sizes[2] = 32; sizes[3] = 128
    sizes[4] = 256; sizes[5] = 512; sizes[6] = 1024; sizes[7] = 4096

    for s in range(8):
        var vl = sizes[s]
        if vl > WINDOW:
            continue
        var pos = vl - 1

        for _ in range(WARMUP):
            var nw = dispatch_sliding_attention[
                head_dim=HEAD_DIM, num_q=NUM_Q, num_kv=NUM_KV,
                gqa_ratio=GQA_RATIO, kv_stride=KV_STRIDE, window=WINDOW](
                q, k_cache, v_cache, partials, pos, vl, pool)
            merge_flash_partials[HEAD_DIM, NUM_Q](
                output, partials, FLASH_PARTIAL_STRIDE[NUM_Q, HEAD_DIM], nw, pool)
            keep(output[0])

        var best = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                var nw = dispatch_sliding_attention[
                    head_dim=HEAD_DIM, num_q=NUM_Q, num_kv=NUM_KV,
                    gqa_ratio=GQA_RATIO, kv_stride=KV_STRIDE, window=WINDOW](
                    q, k_cache, v_cache, partials, pos, vl, pool)
                merge_flash_partials[HEAD_DIM, NUM_Q](
                    output, partials, FLASH_PARTIAL_STRIDE[NUM_Q, HEAD_DIM], nw, pool)
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


def section_validation[P: BurstThreadPool](
    mut pool: P, q: BF16Ptr, k_cache: BF16Ptr, v_cache: BF16Ptr,
    output: BF16Ptr, partials: F32Ptr,
):
    print("\n=== Validation (valid_len=64) ===")
    comptime VL = 64
    var pos = VL - 1

    var nw = dispatch_sliding_attention[
        head_dim=HEAD_DIM, num_q=NUM_Q, num_kv=NUM_KV,
        gqa_ratio=GQA_RATIO, kv_stride=KV_STRIDE, window=WINDOW](
        q, k_cache, v_cache, partials, pos, VL, pool)
    merge_flash_partials[HEAD_DIM, NUM_Q](
        output, partials, FLASH_PARTIAL_STRIDE[NUM_Q, HEAD_DIM], nw, pool)

    print("  output[0..3]: "
        + String(output[0].cast[DType.float32]()) + " "
        + String(output[1].cast[DType.float32]()) + " "
        + String(output[2].cast[DType.float32]()) + " "
        + String(output[3].cast[DType.float32]()))
    var ok = True
    for i in range(NUM_Q * HEAD_DIM):
        var v = output[i].cast[DType.float32]()
        if v != v:
            ok = False
            break
    print("  " + ("OK (no NaN)" if ok else "FAIL: NaN detected"))


def run_all[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
):
    comptime flash_stride = FLASH_PARTIAL_STRIDE[NUM_Q, HEAD_DIM]
    var q = arena_alloc[DType.bfloat16](arenas[0], NUM_Q * HEAD_DIM)
    var k_cache = arena_alloc[DType.bfloat16](arenas[0], WINDOW * KV_STRIDE)
    var v_cache = arena_alloc[DType.bfloat16](arenas[0], WINDOW * KV_STRIDE)
    var output = arena_alloc[DType.bfloat16](arenas[0], NUM_Q * HEAD_DIM)
    var partials = arena_alloc[DType.float32](arenas[0], MAX_WORKERS * flash_stride)

    fill_pattern(q, NUM_Q * HEAD_DIM)
    fill_pattern(k_cache, WINDOW * KV_STRIDE)
    fill_pattern(v_cache, WINDOW * KV_STRIDE)

    var cap = pools[0].get_capacity()
    print("pool capacity: " + String(cap) + " workers")
    print("head_dim=" + String(HEAD_DIM) + " num_q=" + String(NUM_Q)
        + " num_kv=" + String(NUM_KV) + " gqa=" + String(GQA_RATIO)
        + " window=" + String(WINDOW))

    section_validation(pools[0], q, k_cache, v_cache, output, partials)
    section_context_sweep(pools[0], q, k_cache, v_cache, output, partials)


def main():
    var numa = NumaInfo()
    var topo = numa.plan_topology(numa.num_nodes)
    var tp = numa.num_nodes

    print("Sliding attention benchmark (dispatch_sliding_attention)")
    print(String(tp) + " NUMA node(s), "
        + String(len(numa.isolated_cpus)) + " isolated cpus\n")

    comptime ARENA_BYTES = 256 * 1024 * 1024
    var arenas = HeapMoveArray[NumaArena[alignment=ALIGNMENT]](tp)
    for i in range(tp):
        arenas.push(NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    if numa.has_isolation():
        print("mode: isolated")
        var pools = HeapMoveArray[IsolatedBurstPool[]](tp)
        for i in range(tp):
            pools.push(IsolatedBurstPool[].for_topology(numa, topo[i]))
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
            pools.push(BurstPool[].for_topology(numa, topo[i]))
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
