from std.collections import InlineArray
from std.memory import UnsafePointer
from std.time import perf_counter_ns
from std.benchmark import keep
from std.sys.info import simd_width_of

from numa import NumaArena, NumaTopology
from threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool
from threading.threading_traits import BurstKernel, BurstThreadPool
from notstdcollections import HeapMoveArray
from kernels.helpers import (
    OutputPartitionedKernel, DispatchBuffer, RankBuffers,
    tile_dispatch, join_all, worker_range,
)
from kernels.reductions import dispatch_allreduce
from modeling.model_spec import BF16, Encoding


comptime ALIGNMENT = 64
comptime WARMUP = 3
comptime TRIALS = 10
comptime ITERS = 20
comptime MAX_ELEMS = 32 * 1024 * 1024
comptime BUF_ELEMS = 2816 * 4096

comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


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
        ptr[i] = Scalar[DType.bfloat16](Float32(i % 256))


def fmt_bw(total_bytes: Int, ns: Int) -> String:
    if ns <= 0:
        return "inf GB/s"
    var bw_100 = total_bytes * 100 // ns
    return String(bw_100 // 100) + "." + String(bw_100 % 100) + " GB/s"


def print_bw(label: String, total_bytes: Int, ns: Int):
    print("  " + label + ": " + String(ns) + " ns  " + fmt_bw(total_bytes, ns))


@fieldwise_init
struct ReadSweepKernel(OutputPartitionedKernel):
    var src: BF16Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        comptime W = simd_width_of[DType.bfloat16]()
        var acc = SIMD[DType.float32, W](0)
        var pos = self.start
        while pos + W <= self.end:
            acc += (self.src + pos).load[width=W]().cast[DType.float32]()
            pos += W
        keep(acc.reduce_add())

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.start = start
        self.end = end


@fieldwise_init
struct WriteSweepKernel(OutputPartitionedKernel):
    var dst: BF16Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        comptime W = simd_width_of[DType.bfloat16]()
        var val = SIMD[DType.bfloat16, W](1)
        var pos = self.start
        while pos + W <= self.end:
            (self.dst + pos).store(val)
            pos += W

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.start = start
        self.end = end


@fieldwise_init
struct CopySweepKernel(OutputPartitionedKernel):
    var dst: BF16Ptr
    var src: BF16Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        comptime W = simd_width_of[DType.bfloat16]()
        var pos = self.start
        while pos + W <= self.end:
            (self.dst + pos).store((self.src + pos).load[width=W]())
            pos += W

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.start = start
        self.end = end


@fieldwise_init
struct NoopKernel(BurstKernel):
    var pad: Int
    def execute(mut self):
        pass


def timed_read[P: BurstThreadPool](mut pool: P, src: BF16Ptr, count: Int) -> Int:
    var buf = DispatchBuffer[ReadSweepKernel]()
    for _ in range(WARMUP):
        tile_dispatch(buf, ReadSweepKernel(src, 0, 0), pool, count)
        pool.join()
    var best = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            tile_dispatch(buf, ReadSweepKernel(src, 0, 0), pool, count)
            pool.join()
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // ITERS
        if avg < best:
            best = avg
    return best


def timed_write[P: BurstThreadPool](mut pool: P, dst: BF16Ptr, count: Int) -> Int:
    var buf = DispatchBuffer[WriteSweepKernel]()
    for _ in range(WARMUP):
        tile_dispatch(buf, WriteSweepKernel(dst, 0, 0), pool, count)
        pool.join()
    var best = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            tile_dispatch(buf, WriteSweepKernel(dst, 0, 0), pool, count)
            pool.join()
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // ITERS
        if avg < best:
            best = avg
    return best


def timed_copy[P: BurstThreadPool](
    mut pool: P, dst: BF16Ptr, src: BF16Ptr, count: Int,
) -> Int:
    var buf = DispatchBuffer[CopySweepKernel]()
    for _ in range(WARMUP):
        tile_dispatch(buf, CopySweepKernel(dst, src, 0, 0), pool, count)
        pool.join()
    var best = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            tile_dispatch(buf, CopySweepKernel(dst, src, 0, 0), pool, count)
            pool.join()
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // ITERS
        if avg < best:
            best = avg
    return best


def section_local_bw[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    src: InlineArray[BF16Ptr, tp],
    dst: InlineArray[BF16Ptr, tp],
    count: Int,
):
    var mb = count * 2 // 1024 // 1024
    print("\n=== Local BW per node (" + String(mb) + " MB bf16) ===")
    for n in range(tp):
        var ns_r = timed_read(pools[n], src[n], count)
        var ns_w = timed_write(pools[n], dst[n], count)
        var ns_c = timed_copy(pools[n], dst[n], src[n], count)
        print_bw("n" + String(n) + " read ", count * 2, ns_r)
        print_bw("n" + String(n) + " write", count * 2, ns_w)
        print_bw("n" + String(n) + " copy ", count * 4, ns_c)


def section_remote_cached[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    src: InlineArray[BF16Ptr, tp],
    count: Int,
):
    comptime
    if tp <= 1:
        return
    print("\n=== Cached remote read BW (data in reader L3 after warmup) ===")
    for reader in range(tp):
        for owner in range(tp):
            if reader == owner:
                continue
            var ns = timed_read(pools[reader], src[owner], count)
            print_bw("reader=n" + String(reader) + " owner=n" + String(owner),
                count * 2, ns)


def section_remote_fresh[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    bufs: InlineArray[BF16Ptr, tp],
    count: Int,
):
    comptime
    if tp <= 1:
        return
    print("\n=== Fresh remote read BW (owner writes, reader reads) ===")
    for reader in range(tp):
        for owner in range(tp):
            if reader == owner:
                continue
            var rbuf = DispatchBuffer[ReadSweepKernel]()
            var wbuf = DispatchBuffer[WriteSweepKernel]()
            for _ in range(WARMUP):
                tile_dispatch(wbuf, WriteSweepKernel(bufs[owner], 0, 0),
                    pools[owner], count)
                pools[owner].join()
                tile_dispatch(rbuf, ReadSweepKernel(bufs[owner], 0, 0),
                    pools[reader], count)
                pools[reader].join()
            var best = Int(1 << 60)
            for _ in range(TRIALS):
                var elapsed = 0
                for _ in range(ITERS):
                    tile_dispatch(wbuf, WriteSweepKernel(bufs[owner], 0, 0),
                        pools[owner], count)
                    pools[owner].join()
                    var t0 = Int(perf_counter_ns())
                    tile_dispatch(rbuf, ReadSweepKernel(bufs[owner], 0, 0),
                        pools[reader], count)
                    pools[reader].join()
                    var t1 = Int(perf_counter_ns())
                    elapsed += t1 - t0
                var avg = elapsed // ITERS
                if avg < best:
                    best = avg
            print_bw("reader=n" + String(reader) + " owner=n" + String(owner),
                count * 2, best)


def section_contended[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    bufs: InlineArray[BF16Ptr, tp],
    count: Int,
):
    comptime
    if tp <= 1:
        return
    print("\n=== Contended read BW (" + String(tp)
        + " pools read one node, chunked) ===")
    var chunk = count // tp
    for src_node in range(tp):
        for _ in range(WARMUP):
            var buf = DispatchBuffer[ReadSweepKernel]()
            for r in range(tp):
                tile_dispatch(buf, ReadSweepKernel(bufs[src_node], 0, 0),
                    pools[r], chunk, chunk * r)
            join_all[tp](pools)
        var best = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var buf = DispatchBuffer[ReadSweepKernel]()
                var t0 = Int(perf_counter_ns())
                for r in range(tp):
                    tile_dispatch(buf, ReadSweepKernel(bufs[src_node], 0, 0),
                        pools[r], chunk, chunk * r)
                join_all[tp](pools)
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // ITERS
            if avg < best:
                best = avg
        print_bw("src=n" + String(src_node) + " (" + String(tp) + " readers)",
            count * 2, best)


def section_dispatch[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
):
    print("\n=== Dispatch + join overhead ===")
    for _ in range(WARMUP):
        var buf = DispatchBuffer[NoopKernel]()
        for r in range(tp):
            buf.slot()[] = NoopKernel(0)
            buf.dispatch(pools[r])
        join_all[tp](pools)
    var best = Int(1 << 60)
    for _ in range(TRIALS * 5):
        var elapsed = 0
        for _ in range(ITERS * 5):
            var buf = DispatchBuffer[NoopKernel]()
            var t0 = Int(perf_counter_ns())
            for r in range(tp):
                buf.slot()[] = NoopKernel(0)
                buf.dispatch(pools[r])
            join_all[tp](pools)
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // (ITERS * 5)
        if avg < best:
            best = avg
    print("  noop dispatch+join (" + String(tp) + " pools x 1 job): "
        + String(best) + " ns")

    var best2 = Int(1 << 60)
    for _ in range(TRIALS * 5):
        var elapsed = 0
        for _ in range(ITERS * 5):
            var buf = DispatchBuffer[NoopKernel]()
            var t0 = Int(perf_counter_ns())
            for r in range(tp):
                var cap = pools[r].get_capacity()
                for _ in range(cap):
                    buf.slot()[] = NoopKernel(0)
                buf.dispatch(pools[r])
            join_all[tp](pools)
            var t1 = Int(perf_counter_ns())
            elapsed += t1 - t0
        var avg = elapsed // (ITERS * 5)
        if avg < best2:
            best2 = avg
    print("  noop dispatch+join (" + String(tp) + " pools x all workers): "
        + String(best2) + " ns")


def section_worker_scaling[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    src: InlineArray[BF16Ptr, tp],
    count: Int,
):
    var cap = pools[0].get_capacity()
    print("\n=== Worker scaling on node 0 (capacity=" + String(cap)
        + ", " + String(count * 2 // 1024 // 1024) + " MB) ===")
    print("  workers | dispatch_ns |   read_ns |    read_GB/s")

    var n = 1
    while n <= cap:
        var noop_buf = DispatchBuffer[NoopKernel]()
        for _ in range(WARMUP * 10):
            for _ in range(n):
                noop_buf.slot()[] = NoopKernel(0)
            noop_buf.dispatch(pools[0])
            pools[0].join()
        var best_dispatch = Int(1 << 60)
        for _ in range(TRIALS * 5):
            var elapsed = 0
            for _ in range(ITERS * 5):
                var t0 = Int(perf_counter_ns())
                for _ in range(n):
                    noop_buf.slot()[] = NoopKernel(0)
                noop_buf.dispatch(pools[0])
                pools[0].join()
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // (ITERS * 5)
            if avg < best_dispatch:
                best_dispatch = avg

        var read_buf = DispatchBuffer[ReadSweepKernel]()
        for _ in range(WARMUP):
            for w in range(n):
                var wr = worker_range(count, n, w)
                read_buf.slot()[] = ReadSweepKernel(src[0], wr[0], wr[1])
            read_buf.dispatch(pools[0])
            pools[0].join()
        var best_read = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                for w in range(n):
                    var wr = worker_range(count, n, w)
                    read_buf.slot()[] = ReadSweepKernel(src[0], wr[0], wr[1])
                read_buf.dispatch(pools[0])
                pools[0].join()
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            var avg = elapsed // ITERS
            if avg < best_read:
                best_read = avg

        var pad = "       " if n < 10 else "      " if n < 100 else "     "
        print("  " + String(n) + pad + "| " + String(best_dispatch)
            + " | " + String(best_read) + " | " + fmt_bw(count * 2, best_read))

        if n < 4:
            n *= 2
        elif n < cap:
            n = min(n * 2, cap)
        else:
            break


def section_sweep[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    src: InlineArray[BF16Ptr, tp],
    dst: InlineArray[BF16Ptr, tp],
):
    print("\n=== Allreduce bf16 sweep (tp=" + String(tp) + ") ===")
    comptime immut = ImmutOrigin(MutAnyOrigin)

    comptime NUM_SIZES = 18
    var sizes = InlineArray[Int, NUM_SIZES](fill=0)
    sizes[0] = 512
    sizes[1] = 1024
    sizes[2] = 2048
    sizes[3] = 4096
    sizes[4] = 8192
    sizes[5] = 16384
    sizes[6] = 32768
    sizes[7] = 65536
    sizes[8] = 131072
    sizes[9] = 262144
    sizes[10] = 524288
    sizes[11] = 1048576
    sizes[12] = 2097152
    sizes[13] = 4194304
    sizes[14] = 8388608
    sizes[15] = BUF_ELEMS
    sizes[16] = 16777216
    sizes[17] = MAX_ELEMS

    for s in range(NUM_SIZES):
        var count = sizes[s]
        var rb = RankBuffers[DType.bfloat16, tp, immut](count=count)
        var db = RankBuffers[DType.bfloat16, tp, MutAnyOrigin](count=count)
        for r in range(tp):
            rb.ptrs[r] = src[r].as_immutable()
            db.ptrs[r] = dst[r]

        for _ in range(WARMUP):
            dispatch_allreduce[BF16, tp](rb, db, pools)

        var best = Int(1 << 60)
        for _ in range(TRIALS):
            var elapsed = 0
            for _ in range(ITERS):
                var t0 = Int(perf_counter_ns())
                dispatch_allreduce[BF16, tp](rb, db, pools)
                var t1 = Int(perf_counter_ns())
                elapsed += t1 - t0
            keep(db[0][0])
            var avg = elapsed // ITERS
            if avg < best:
                best = avg

        var total_bytes = count * 2 * tp * 2
        var sz_kb = count * 2 // 1024
        if sz_kb < 1024:
            print_bw("allreduce " + String(sz_kb) + "KB", total_bytes, best)
        else:
            print_bw("allreduce " + String(sz_kb // 1024) + "MB",
                total_bytes, best)


def run_all[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
):
    var src = InlineArray[BF16Ptr, tp](uninitialized=True)
    var dst = InlineArray[BF16Ptr, tp](uninitialized=True)
    for r in range(tp):
        src[r] = arena_alloc[DType.bfloat16](arenas[r], MAX_ELEMS)
        dst[r] = arena_alloc[DType.bfloat16](arenas[r], MAX_ELEMS)
        fill_pattern(src[r], MAX_ELEMS)

    section_local_bw[tp=tp](pools, src, dst, BUF_ELEMS)
    section_remote_cached[tp=tp](pools, src, BUF_ELEMS)
    section_remote_fresh[tp=tp](pools, dst, BUF_ELEMS)
    section_contended[tp=tp](pools, src, BUF_ELEMS)
    section_dispatch[tp=tp](pools)
    section_worker_scaling[tp=tp](pools, src, BUF_ELEMS)
    section_sweep[tp=tp](pools, src, dst)


def main():
    var topo = NumaTopology()
    var tp = len(topo)

    print(String(tp) + " NUMA node(s), "
        + String(len(topo.isolated_cpus)) + " isolated cpus\n")

    comptime ARENA_BYTES = 512 * 1024 * 1024
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
