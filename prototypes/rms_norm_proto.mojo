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
from kernels.helpers import (
    RangedKernel, DispatchBuffer, tile_dispatch, worker_range,
)
from simd_math.ops import sqrt
from simd_math import pick_port_unroll, tree_reduce_accs


comptime ALIGNMENT = 64
comptime WARMUP = 5
comptime TRIALS = 12
comptime ITERS = 30

comptime HIDDEN = 2816
comptime RMS_EPS = 1e-6
comptime SQRT_N = sqrt[DType.float32, 1](HIDDEN)
comptime N_EPS = HIDDEN * RMS_EPS

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


def fill_norm_pattern(ptr: BF16Ptr, count: Int):
    for i in range(count):
        var val = Float32((i % 127) - 63) * 0.01
        ptr[i] = Scalar[DType.bfloat16](val)


def fill_weight_pattern(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = Scalar[DType.bfloat16](Float32(1.0) + Float32(i % 64) * 0.001)


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


# === Inline implementations (no dispatch, caller's thread) ===


@always_inline
def rms_reduce_inline(src: BF16Ptr, count: Int) -> Scalar[DType.float32]:
    comptime W = simd_width_of[DType.float32]()
    comptime PU = pick_port_unroll[W, HIDDEN]()
    var accs = InlineArray[SIMD[DType.float32, W], PU](fill=SIMD[DType.float32, W](0))
    var pos = 0
    while pos + PU * W <= count:
        comptime for p in range(PU):
            var v = (src + pos + p * W).load[width=W]().cast[DType.float32]()
            accs[p] = accs[p].fma(v, v)
        pos += PU * W
    var tail_acc = SIMD[DType.float32, W](0)
    while pos + W <= count:
        var v = (src + pos).load[width=W]().cast[DType.float32]()
        tail_acc = tail_acc.fma(v, v)
        pos += W
    accs[0] += tail_acc
    return tree_reduce_accs(accs)


@always_inline
def rms_normalize_inline(
    src: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr,
    count: Int, inv_rms: Scalar[DType.float32],
):
    comptime W = simd_width_of[DType.float32]()
    var factor = SIMD[DType.float32, W](inv_rms)
    var pos = 0
    while pos + W <= count:
        var x = (src + pos).load[width=W]().cast[DType.float32]()
        var w = (weight + pos).load[width=W]().cast[DType.float32]()
        var normed = x * factor * w
        (dst + pos).store(normed.cast[DType.bfloat16]())
        pos += W


@always_inline
def rms_norm_inline(src: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr, count: Int):
    var sum_sq = rms_reduce_inline(src, count)
    var inv_rms = SQRT_N / sqrt[DType.float32, 1](sum_sq + N_EPS)
    rms_normalize_inline(src, dst, weight, count, inv_rms)


def bench_inline(src: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr, seq_len: Int) -> Int:
    for _ in range(WARMUP):
        for t in range(seq_len):
            rms_norm_inline(src + t * HIDDEN, dst + t * HIDDEN, weight, HIDDEN)

    var best_total = Int(1 << 60)
    var best_reduce = Int(1 << 60)
    var best_normalize = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            for t in range(seq_len):
                rms_norm_inline(src + t * HIDDEN, dst + t * HIDDEN, weight, HIDDEN)
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // ITERS
        if avg < best_total:
            best_total = avg
    keep(dst[0])

    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            for t in range(seq_len):
                var s = rms_reduce_inline(src + t * HIDDEN, HIDDEN)
                keep(s)
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // ITERS
        if avg < best_reduce:
            best_reduce = avg

    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            for t in range(seq_len):
                rms_normalize_inline(
                    src + t * HIDDEN, dst + t * HIDDEN, weight, HIDDEN, Float32(0.5))
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // ITERS
        if avg < best_normalize:
            best_normalize = avg
    keep(dst[0])

    var read_bytes = seq_len * HIDDEN * 2
    var write_bytes = seq_len * HIDDEN * 2
    print("    total:     " + fmt_ns(best_total) + "  (" +
          fmt_bw(read_bytes + write_bytes, best_total) + " effective)")
    print("    reduce:    " + fmt_ns(best_reduce) + "  (" +
          fmt_bw(read_bytes, best_reduce) + " read BW)")
    print("    normalize: " + fmt_ns(best_normalize) + "  (" +
          fmt_bw(read_bytes + write_bytes, best_normalize) + " r+w BW)")
    return best_total


# === Dispatched: single worker (measures dispatch overhead vs inline) ===


@fieldwise_init
struct RmsNormKernel(RangedKernel):
    var src: BF16Ptr
    var dst: BF16Ptr
    var weight: BF16Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        var token_start = self.start
        while token_start + HIDDEN <= self.end:
            rms_norm_inline(
                self.src + token_start,
                self.dst + token_start,
                self.weight,
                HIDDEN)
            token_start += HIDDEN

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.src, self.dst, self.weight, start, end)


def bench_dispatched[P: BurstThreadPool](
    mut pool: P, src: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr,
    seq_len: Int, num_workers: Int,
) -> Int:
    var total_elems = seq_len * HIDDEN
    var buf = DispatchBuffer[RmsNormKernel]()

    for _ in range(WARMUP):
        for w in range(num_workers):
            var wr = worker_range(total_elems, num_workers, w)
            var aligned_start = (wr[0] // HIDDEN) * HIDDEN
            var aligned_end = min(((wr[1] + HIDDEN - 1) // HIDDEN) * HIDDEN, total_elems)
            buf.slot()[] = RmsNormKernel(src, dst, weight, aligned_start, aligned_end)
        buf.dispatch(pool)
        pool.join()

    var best = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            for w in range(num_workers):
                var wr = worker_range(total_elems, num_workers, w)
                var aligned_start = (wr[0] // HIDDEN) * HIDDEN
                var aligned_end = min(((wr[1] + HIDDEN - 1) // HIDDEN) * HIDDEN, total_elems)
                buf.slot()[] = RmsNormKernel(src, dst, weight, aligned_start, aligned_end)
            buf.dispatch(pool)
            pool.join()
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // ITERS
        if avg < best:
            best = avg
    keep(dst[0])

    print("    " + String(num_workers) + "w dispatched: " + fmt_ns(best))
    return best


# === Parallel reduce with single-thread normalize (split strategy) ===
# Workers reduce partial sums, host merges + normalizes.


@fieldwise_init
struct RmsReduceKernel(RangedKernel):
    var src: BF16Ptr
    var partial_out: F32Ptr
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var sum = rms_reduce_inline(self.src + self.start, self.end - self.start)
        (self.partial_out + self.worker_id)[] = sum

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.src, self.partial_out, self.worker_id, start, end)


def bench_split_reduce_normalize[P: BurstThreadPool](
    mut pool: P, src: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr,
    partials: F32Ptr, seq_len: Int, num_workers: Int,
) -> Int:
    var buf = DispatchBuffer[RmsReduceKernel]()

    for _ in range(WARMUP):
        for t in range(seq_len):
            var token_src = src + t * HIDDEN
            for w in range(num_workers):
                var wr = worker_range(HIDDEN, num_workers, w)
                buf.slot()[] = RmsReduceKernel(token_src, partials, w, wr[0], wr[1])
            buf.dispatch(pool)
            pool.join()
            var sum_sq = Scalar[DType.float32](0)
            for w in range(num_workers):
                sum_sq += partials[w]
            var inv_rms = SQRT_N / sqrt[DType.float32, 1](sum_sq + N_EPS)
            rms_normalize_inline(token_src, dst + t * HIDDEN, weight, HIDDEN, inv_rms)

    var best = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            for t in range(seq_len):
                var token_src = src + t * HIDDEN
                for w in range(num_workers):
                    var wr = worker_range(HIDDEN, num_workers, w)
                    buf.slot()[] = RmsReduceKernel(token_src, partials, w, wr[0], wr[1])
                buf.dispatch(pool)
                pool.join()
                var sum_sq = Scalar[DType.float32](0)
                for w in range(num_workers):
                    sum_sq += partials[w]
                var inv_rms = SQRT_N / sqrt[DType.float32, 1](sum_sq + N_EPS)
                rms_normalize_inline(token_src, dst + t * HIDDEN, weight, HIDDEN, inv_rms)
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // ITERS
        if avg < best:
            best = avg
    keep(dst[0])

    print("    " + String(num_workers) + "w split (reduce‖, normalize host): " + fmt_ns(best))
    return best


# === Remote read: norm on one rank, data owned by another ===


def bench_inline_remote(
    src: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr,
    seq_len: Int, label: String,
) -> Int:
    for _ in range(WARMUP):
        for t in range(seq_len):
            rms_norm_inline(src + t * HIDDEN, dst + t * HIDDEN, weight, HIDDEN)

    var best = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            for t in range(seq_len):
                rms_norm_inline(src + t * HIDDEN, dst + t * HIDDEN, weight, HIDDEN)
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // ITERS
        if avg < best:
            best = avg
    keep(dst[0])

    print("    " + label + ": " + fmt_ns(best))
    return best


# === Main ===


def run_benchmarks[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
):
    comptime MAX_SEQ = 64
    var src = InlineArray[BF16Ptr, tp](uninitialized=True)
    var dst = InlineArray[BF16Ptr, tp](uninitialized=True)
    var weight = arena_alloc[DType.bfloat16](arenas[0], HIDDEN)
    var partials = arena_alloc[DType.float32](arenas[0], 128)

    for r in range(tp):
        src[r] = arena_alloc[DType.bfloat16](arenas[r], MAX_SEQ * HIDDEN)
        dst[r] = arena_alloc[DType.bfloat16](arenas[r], MAX_SEQ * HIDDEN)
        fill_norm_pattern(src[r], MAX_SEQ * HIDDEN)

    fill_weight_pattern(weight, HIDDEN)

    var cap = pools[0].get_capacity()
    print("pool capacity: " + String(cap) + " workers")
    print("hidden: " + String(HIDDEN) + " (" + String(HIDDEN * 2) + " bytes bf16)")
    print("N-absorbed eps: " + String(N_EPS) + ", sqrt(N): " + String(SQRT_N))

    # --- Section 1: Single-token RMSNorm, inline vs dispatched ---
    print("\n=== Single token (HIDDEN=" + String(HIDDEN) + ", 5.5KB) ===")
    print("  Inline (host thread):")
    var inline_1 = bench_inline(src[0], dst[0], weight, 1)

    print("  Dispatched (worker scaling):")
    _ = bench_dispatched(pools[0], src[0], dst[0], weight, 1, 1)
    if cap >= 2:
        _ = bench_dispatched(pools[0], src[0], dst[0], weight, 1, 2)
    if cap >= 4:
        _ = bench_dispatched(pools[0], src[0], dst[0], weight, 1, 4)

    print("  Split reduce+normalize:")
    _ = bench_split_reduce_normalize(pools[0], src[0], dst[0], weight, partials, 1, 2)
    if cap >= 4:
        _ = bench_split_reduce_normalize(pools[0], src[0], dst[0], weight, partials, 1, 4)

    # --- Section 2: Multi-token sweep (decode batch / small prefill) ---
    print("\n=== Multi-token sweep (inline vs best dispatched) ===")
    print("  seq_len | inline     | 1w dispatch | " + String(cap) + "w dispatch")

    var seq_lens = InlineArray[Int, 7](fill=0)
    seq_lens[0] = 1
    seq_lens[1] = 2
    seq_lens[2] = 4
    seq_lens[3] = 8
    seq_lens[4] = 16
    seq_lens[5] = 32
    seq_lens[6] = 64
    for s in range(7):
        var seq = seq_lens[s]
        if seq > MAX_SEQ:
            break

        # Inline
        for _ in range(WARMUP):
            for t in range(seq):
                rms_norm_inline(src[0] + t * HIDDEN, dst[0] + t * HIDDEN, weight, HIDDEN)
        var best_inline = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                for t in range(seq):
                    rms_norm_inline(src[0] + t * HIDDEN, dst[0] + t * HIDDEN, weight, HIDDEN)
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // ITERS
            if avg < best_inline:
                best_inline = avg
        keep(dst[0][0])

        # 1 worker
        var best_1w = Int(1 << 60)
        var buf1 = DispatchBuffer[RmsNormKernel]()
        for _ in range(WARMUP):
            buf1.slot()[] = RmsNormKernel(src[0], dst[0], weight, 0, seq * HIDDEN)
            buf1.dispatch(pools[0])
            pools[0].join()
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                buf1.slot()[] = RmsNormKernel(src[0], dst[0], weight, 0, seq * HIDDEN)
                buf1.dispatch(pools[0])
                pools[0].join()
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // ITERS
            if avg < best_1w:
                best_1w = avg
        keep(dst[0][0])

        # All workers
        var best_nw = Int(1 << 60)
        var bufn = DispatchBuffer[RmsNormKernel]()
        var total_elems = seq * HIDDEN
        for _ in range(WARMUP):
            for w in range(cap):
                var wr = worker_range(total_elems, cap, w)
                var a_start = (wr[0] // HIDDEN) * HIDDEN
                var a_end = min(((wr[1] + HIDDEN - 1) // HIDDEN) * HIDDEN, total_elems)
                bufn.slot()[] = RmsNormKernel(src[0], dst[0], weight, a_start, a_end)
            bufn.dispatch(pools[0])
            pools[0].join()
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                for w in range(cap):
                    var wr = worker_range(total_elems, cap, w)
                    var a_start = (wr[0] // HIDDEN) * HIDDEN
                    var a_end = min(((wr[1] + HIDDEN - 1) // HIDDEN) * HIDDEN, total_elems)
                    bufn.slot()[] = RmsNormKernel(src[0], dst[0], weight, a_start, a_end)
                bufn.dispatch(pools[0])
                pools[0].join()
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // ITERS
            if avg < best_nw:
                best_nw = avg
        keep(dst[0][0])

        var pad = "  " if seq < 10 else " "
        print("  " + String(seq) + pad + "     | " +
              fmt_ns(best_inline) + " | " + fmt_ns(best_1w) + " | " + fmt_ns(best_nw))

    # --- Section 3: NUMA locality (remote read vs local read) ---
    comptime
    if tp > 1:
        print("\n=== NUMA locality: norm source placement (seq_len=1) ===")
        print("  (compute on node 0, data on node N)")
        for owner in range(tp):
            var label = "src=n" + String(owner) + " dst=n0"
            _ = bench_inline_remote(src[owner], dst[0], weight, 1, label)

        print("\n=== NUMA locality: norm source placement (seq_len=16) ===")
        for owner in range(tp):
            var label = "src=n" + String(owner) + " dst=n0"
            _ = bench_inline_remote(src[owner], dst[0], weight, 16, label)


def main():
    var numa = NumaInfo()
    var topo = numa.plan_topology(numa.num_nodes)
    var tp = numa.num_nodes

    print("RMSNorm prototype benchmark")
    print(String(tp) + " NUMA node(s), "
        + String(len(numa.isolated_cpus)) + " isolated cpus\n")

    comptime ARENA_BYTES = 128 * 1024 * 1024
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
            run_benchmarks[tp=1](pools, arenas)
        elif tp == 2:
            run_benchmarks[tp=2](pools, arenas)
        elif tp == 4:
            run_benchmarks[tp=4](pools, arenas)
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
            run_benchmarks[tp=1](pools, arenas)
        elif tp == 2:
            run_benchmarks[tp=2](pools, arenas)
        elif tp == 4:
            run_benchmarks[tp=4](pools, arenas)
        else:
            print("unsupported tp=" + String(tp))
