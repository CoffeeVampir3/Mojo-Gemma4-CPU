from std.collections import InlineArray
from std.memory import UnsafePointer
from std.time import perf_counter_ns
from std.benchmark import keep
from std.sys.info import simd_width_of

from numa import NumaArena, NumaInfo, NumaTopology
from threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool
from threading.threading_traits import BurstKernel, BurstThreadPool
from notstdcollections import HeapMoveArray
from kernels.kv_tiled_attention import (
    sliding_attention_two_pass, sliding_attention_single_pass,
    FLASH_PARTIAL_STRIDE,
)


comptime ALIGNMENT = 64
comptime WARMUP = 5
comptime TRIALS = 10
comptime ITERS = 10

comptime HEAD_DIM = 256
comptime NUM_Q = 4
comptime NUM_KV = 2
comptime GQA_RATIO = 2
comptime KV_STRIDE = 512
comptime WINDOW = 1024
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


def section_validate[P: BurstThreadPool](
    mut pool: P, q: BF16Ptr, k_cache: BF16Ptr, v_cache: BF16Ptr,
    out_2p: BF16Ptr, out_1p: BF16Ptr,
    scores_buf: F32Ptr, partials_2p: F32Ptr, partials_1p: F32Ptr,
):
    print("\n=== Correctness validation (valid_len=64) ===")
    comptime VL = 64
    var pos = VL - 1

    sliding_attention_two_pass[
        head_dim=HEAD_DIM, num_q=NUM_Q, num_kv=NUM_KV,
        gqa_ratio=GQA_RATIO, kv_stride=KV_STRIDE, window=WINDOW](
        q, k_cache, v_cache, out_2p, scores_buf, partials_2p, pos, VL, pool)

    sliding_attention_single_pass[
        head_dim=HEAD_DIM, num_q=NUM_Q, num_kv=NUM_KV,
        gqa_ratio=GQA_RATIO, kv_stride=KV_STRIDE, window=WINDOW](
        q, k_cache, v_cache, out_1p, partials_1p, pos, VL, pool)

    var max_err = Float32(0)
    var max_err_ref = Float32(0)
    var max_err_actual = Float32(0)
    var max_err_rel = Float32(0)
    var max_rel = Float32(0)
    var max_rel_err = Float32(0)
    var max_rel_ref = Float32(0)
    var max_rel_actual = Float32(0)
    var sum_sq_err = Float64(0)
    var sum_sq_ref = Float64(0)
    for i in range(NUM_Q * HEAD_DIM):
        var val_2p = out_2p[i].cast[DType.float32]()
        var val_1p = out_1p[i].cast[DType.float32]()
        var err = (val_1p - val_2p).__abs__()
        var denom = val_2p.__abs__()
        sum_sq_err += Float64(err) * Float64(err)
        sum_sq_ref += Float64(val_2p) * Float64(val_2p)
        if err > max_err:
            max_err = err
            max_err_ref = val_2p
            max_err_actual = val_1p
            max_err_rel = err / max(denom, Float32(1.0e-9))
        if denom > Float32(1.0e-6):
            var rel = err / denom
            if rel > max_rel:
                max_rel = rel
                max_rel_err = err
                max_rel_ref = val_2p
                max_rel_actual = val_1p

    print("  two-pass output[0..3]: "
        + String(out_2p[0].cast[DType.float32]()) + " "
        + String(out_2p[1].cast[DType.float32]()) + " "
        + String(out_2p[2].cast[DType.float32]()) + " "
        + String(out_2p[3].cast[DType.float32]()))
    print("  single-pass max error vs two-pass: " + String(max_err))
    print("  max-error relative: " + String(max_err_rel)
        + " ref=" + String(max_err_ref)
        + " actual=" + String(max_err_actual))
    print("  max relative error: " + String(max_rel)
        + " abs=" + String(max_rel_err)
        + " ref=" + String(max_rel_ref)
        + " actual=" + String(max_rel_actual))
    print("  relative L2 error: " + String((sum_sq_err / sum_sq_ref) ** 0.5))
    if max_err > 0.1:
        print("  WARNING: large error detected, results may be invalid")
    else:
        print("  OK (all within tolerance)")


def section_head_to_head[P: BurstThreadPool](
    mut pool: P, q: BF16Ptr, k_cache: BF16Ptr, v_cache: BF16Ptr,
    output: BF16Ptr, scores_buf: F32Ptr, partials_2p: F32Ptr, partials_1p: F32Ptr,
):
    print("\n=== Head-to-head: valid_len sweep ===")
    print("  valid_len | two-pass     | single-pass  | KV read")

    comptime NUM_SIZES = 7
    var sizes = InlineArray[Int, NUM_SIZES](fill=0)
    sizes[0] = 1
    sizes[1] = 8
    sizes[2] = 32
    sizes[3] = 128
    sizes[4] = 256
    sizes[5] = 512
    sizes[6] = 1024

    for s in range(NUM_SIZES):
        var vl = sizes[s]
        var pos = vl - 1

        for _ in range(WARMUP):
            sliding_attention_two_pass[
                head_dim=HEAD_DIM, num_q=NUM_Q, num_kv=NUM_KV,
                gqa_ratio=GQA_RATIO, kv_stride=KV_STRIDE, window=WINDOW](
                q, k_cache, v_cache, output, scores_buf, partials_2p, pos, vl, pool)
        var best_2p = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                sliding_attention_two_pass[
                    head_dim=HEAD_DIM, num_q=NUM_Q, num_kv=NUM_KV,
                    gqa_ratio=GQA_RATIO, kv_stride=KV_STRIDE, window=WINDOW](
                    q, k_cache, v_cache, output, scores_buf, partials_2p, pos, vl, pool)
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // ITERS
            if avg < best_2p:
                best_2p = avg
        keep(output[0])

        for _ in range(WARMUP):
            sliding_attention_single_pass[
                head_dim=HEAD_DIM, num_q=NUM_Q, num_kv=NUM_KV,
                gqa_ratio=GQA_RATIO, kv_stride=KV_STRIDE, window=WINDOW](
                q, k_cache, v_cache, output, partials_1p, pos, vl, pool)
        var best_1p = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                sliding_attention_single_pass[
                    head_dim=HEAD_DIM, num_q=NUM_Q, num_kv=NUM_KV,
                    gqa_ratio=GQA_RATIO, kv_stride=KV_STRIDE, window=WINDOW](
                    q, k_cache, v_cache, output, partials_1p, pos, vl, pool)
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // ITERS
            if avg < best_1p:
                best_1p = avg
        keep(output[0])

        var kv_bytes = vl * KV_STRIDE * 2 * 2
        var pad = "   " if vl < 10 else "  " if vl < 100 else " " if vl < 1000 else ""
        print("  " + String(vl) + pad
            + "     | " + fmt_ns(best_2p)
            + " | " + fmt_ns(best_1p)
            + " | " + String(kv_bytes // 1024) + " KB")


def section_worker_scaling[P: BurstThreadPool](
    mut pool: P, q: BF16Ptr, k_cache: BF16Ptr, v_cache: BF16Ptr,
    output: BF16Ptr, scores_buf: F32Ptr, partials_2p: F32Ptr, partials_1p: F32Ptr,
):
    print("\n=== Worker scaling (valid_len=1024) ===")
    print("  workers | two-pass     | single-pass")

    comptime VL = 1024
    var pos = VL - 1
    var cap = pool.get_capacity()

    var nw = 1
    while nw <= cap:
        for _ in range(WARMUP):
            sliding_attention_single_pass[
                head_dim=HEAD_DIM, num_q=NUM_Q, num_kv=NUM_KV,
                gqa_ratio=GQA_RATIO, kv_stride=KV_STRIDE, window=WINDOW](
                q, k_cache, v_cache, output, partials_1p, pos, VL, pool, num_workers=nw)

        var best_2p = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                sliding_attention_two_pass[
                    head_dim=HEAD_DIM, num_q=NUM_Q, num_kv=NUM_KV,
                    gqa_ratio=GQA_RATIO, kv_stride=KV_STRIDE, window=WINDOW](
                    q, k_cache, v_cache, output, scores_buf, partials_2p, pos, VL, pool, num_workers=nw)
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // ITERS
            if avg < best_2p:
                best_2p = avg
        keep(output[0])

        var best_1p = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                sliding_attention_single_pass[
                    head_dim=HEAD_DIM, num_q=NUM_Q, num_kv=NUM_KV,
                    gqa_ratio=GQA_RATIO, kv_stride=KV_STRIDE, window=WINDOW](
                    q, k_cache, v_cache, output, partials_1p, pos, VL, pool, num_workers=nw)
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // ITERS
            if avg < best_1p:
                best_1p = avg
        keep(output[0])

        var pad = "  " if nw < 10 else " "
        print("  " + String(nw) + pad
            + "      | " + fmt_ns(best_2p)
            + " | " + fmt_ns(best_1p))

        if nw < 4:
            nw *= 2
        elif nw < cap:
            nw = min(nw * 2, cap)
        else:
            break


def section_long_context[P: BurstThreadPool](
    mut pool: P, q: BF16Ptr, k_cache: BF16Ptr, v_cache: BF16Ptr,
    output: BF16Ptr, scores_buf: F32Ptr, partials_2p: F32Ptr, partials_1p: F32Ptr,
):
    print("\n=== Long context sweep (full pool, window=16384) ===")
    print("  valid_len | two-pass     | single-pass  | KV read  | BW(2P)       | BW(1P)")

    comptime LONG_WINDOW = 16384
    comptime NUM_SIZES = 6
    var sizes = InlineArray[Int, NUM_SIZES](fill=0)
    sizes[0] = 256
    sizes[1] = 1024
    sizes[2] = 2048
    sizes[3] = 4096
    sizes[4] = 8192
    sizes[5] = 16384

    for s in range(NUM_SIZES):
        var vl = sizes[s]
        var pos = vl - 1

        for _ in range(WARMUP):
            sliding_attention_two_pass[
                head_dim=HEAD_DIM, num_q=NUM_Q, num_kv=NUM_KV,
                gqa_ratio=GQA_RATIO, kv_stride=KV_STRIDE, window=LONG_WINDOW](
                q, k_cache, v_cache, output, scores_buf, partials_2p, pos, vl, pool)
        var best_2p = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                sliding_attention_two_pass[
                    head_dim=HEAD_DIM, num_q=NUM_Q, num_kv=NUM_KV,
                    gqa_ratio=GQA_RATIO, kv_stride=KV_STRIDE, window=LONG_WINDOW](
                    q, k_cache, v_cache, output, scores_buf, partials_2p, pos, vl, pool)
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // ITERS
            if avg < best_2p:
                best_2p = avg
        keep(output[0])

        for _ in range(WARMUP):
            sliding_attention_single_pass[
                head_dim=HEAD_DIM, num_q=NUM_Q, num_kv=NUM_KV,
                gqa_ratio=GQA_RATIO, kv_stride=KV_STRIDE, window=LONG_WINDOW](
                q, k_cache, v_cache, output, partials_1p, pos, vl, pool)
        var best_1p = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                sliding_attention_single_pass[
                    head_dim=HEAD_DIM, num_q=NUM_Q, num_kv=NUM_KV,
                    gqa_ratio=GQA_RATIO, kv_stride=KV_STRIDE, window=LONG_WINDOW](
                    q, k_cache, v_cache, output, partials_1p, pos, vl, pool)
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // ITERS
            if avg < best_1p:
                best_1p = avg
        keep(output[0])

        var kv_bytes = vl * KV_STRIDE * 2 * 2
        var pad = " " if vl < 10000 else ""
        print("  " + String(vl) + pad
            + "    | " + fmt_ns(best_2p)
            + " | " + fmt_ns(best_1p)
            + " | " + String(kv_bytes // (1024 * 1024)) + " MB"
            + "     | " + fmt_bw(kv_bytes, best_2p)
            + " | " + fmt_bw(kv_bytes, best_1p))


comptime LONG_WINDOW = 16384


def run_all[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
):
    var q = arena_alloc[DType.bfloat16](arenas[0], NUM_Q * HEAD_DIM)
    var k_cache = arena_alloc[DType.bfloat16](arenas[0], LONG_WINDOW * KV_STRIDE)
    var v_cache = arena_alloc[DType.bfloat16](arenas[0], LONG_WINDOW * KV_STRIDE)
    var out_2p = arena_alloc[DType.bfloat16](arenas[0], NUM_Q * HEAD_DIM)
    var out_1p = arena_alloc[DType.bfloat16](arenas[0], NUM_Q * HEAD_DIM)
    var output = arena_alloc[DType.bfloat16](arenas[0], NUM_Q * HEAD_DIM)
    var scores_buf = arena_alloc[DType.float32](arenas[0], NUM_Q * LONG_WINDOW)
    var partials_2p = arena_alloc[DType.float32](arenas[0], MAX_WORKERS * NUM_Q * HEAD_DIM)
    comptime flash_stride = FLASH_PARTIAL_STRIDE[NUM_Q, HEAD_DIM]
    var partials_1p = arena_alloc[DType.float32](arenas[0], MAX_WORKERS * flash_stride)

    fill_pattern(q, NUM_Q * HEAD_DIM)
    fill_pattern(k_cache, LONG_WINDOW * KV_STRIDE)
    fill_pattern(v_cache, LONG_WINDOW * KV_STRIDE)

    var cap = pools[0].get_capacity()
    print("pool capacity: " + String(cap) + " workers")
    print("head_dim=" + String(HEAD_DIM) + " num_q=" + String(NUM_Q)
        + " num_kv=" + String(NUM_KV) + " gqa=" + String(GQA_RATIO)
        + " window=" + String(WINDOW))

    section_validate(pools[0], q, k_cache, v_cache, out_2p, out_1p,
        scores_buf, partials_2p, partials_1p)
    section_head_to_head(pools[0], q, k_cache, v_cache, output,
        scores_buf, partials_2p, partials_1p)
    section_worker_scaling(pools[0], q, k_cache, v_cache, output,
        scores_buf, partials_2p, partials_1p)
    section_long_context(pools[0], q, k_cache, v_cache, output,
        scores_buf, partials_2p, partials_1p)


def main():
    var numa = NumaInfo()
    var topo = numa.plan_topology(numa.num_nodes)
    var tp = numa.num_nodes

    print("Sliding attention benchmark")
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
