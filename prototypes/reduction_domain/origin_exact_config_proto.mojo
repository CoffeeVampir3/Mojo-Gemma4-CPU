from std.collections import InlineArray
from std.memory import Span, UnsafePointer, alloc

from kernels.helpers import RankBuffers
from threading.threading_traits import BurstKernel, BurstThreadPool


@fieldwise_init
struct TestPool(BurstThreadPool):
    var capacity: Int
    var dispatch_count: Int

    def get_capacity(self) -> Int:
        return self.capacity

    def dispatch[K: BurstKernel, origin: MutOrigin](
        mut self, kernels: Span[K, origin], num_jobs: Int
    ):
        for i in range(num_jobs):
            var kernel = kernels[i]
            kernel.execute()
        self.dispatch_count += 1

    def join(mut self):
        pass

    def last_worker_timestamp(self) -> Int:
        return self.dispatch_count

    def wake(mut self):
        pass

    def sleep(mut self):
        pass


struct MoveJobSet[K: BurstKernel]:
    var items: InlineArray[Self.K, 16]
    var count: Int

    def __init__(out self):
        self.items = InlineArray[Self.K, 16](uninitialized=True)
        self.count = 0

    def add(mut self, job: Self.K):
        self.items[self.count] = job
        self.count += 1

    def dispatch(mut self, mut pool: TestPool):
        if self.count > 0:
            pool.dispatch(
                Span(ptr=UnsafePointer(to=self.items[0]), length=self.count),
                self.count,
            )
        self.count = 0


@fieldwise_init
struct ExactReduceConfig[tp: Int, src_origin: MutOrigin, dst_origin: MutOrigin]:
    comptime SrcPtr = UnsafePointer[Float32, Self.src_origin]
    comptime DstPtr = UnsafePointer[Float32, Self.dst_origin]

    var src: InlineArray[Self.SrcPtr, Self.tp]
    var dst: InlineArray[Self.DstPtr, Self.tp]
    var total_elements: Int


@fieldwise_init
struct ExactReduceKernel[
    tp: Int,
](BurstKernel):
    var config_addr: Int
    var rank: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var config = UnsafePointer[
            ExactReduceConfig[Self.tp, MutExternalOrigin, MutExternalOrigin],
            MutAnyOrigin,
        ](unsafe_from_address=self.config_addr)
        reduce_exact_range[
            Self.tp, MutExternalOrigin, MutExternalOrigin, MutAnyOrigin
        ](
            config, self.rank, self.start, self.end
        )


def make_exact_config[
    tp: Int, src_origin: MutOrigin, dst_origin: MutOrigin
](
    src: RankBuffers[DType.float32, tp, src_origin],
    dst: RankBuffers[DType.float32, tp, dst_origin],
) -> ExactReduceConfig[tp, src_origin, dst_origin]:
    var src_ptrs = InlineArray[UnsafePointer[Float32, src_origin], tp](
        uninitialized=True
    )
    var dst_ptrs = InlineArray[UnsafePointer[Float32, dst_origin], tp](
        uninitialized=True
    )
    for r in range(tp):
        src_ptrs[r] = src.ptrs[r]
        dst_ptrs[r] = dst.ptrs[r]
    return ExactReduceConfig[tp, src_origin, dst_origin](
        src=src_ptrs, dst=dst_ptrs, total_elements=src.count
    )


def reduce_exact_range[
    tp: Int,
    src_origin: MutOrigin,
    dst_origin: MutOrigin,
    cfg_origin: MutOrigin,
](
    config: UnsafePointer[
        ExactReduceConfig[tp, src_origin, dst_origin], cfg_origin
    ],
    out_rank: Int,
    start: Int,
    end: Int,
):
    var src0 = config[].src[0]
    var src1 = config[].src[1]
    var dst = config[].dst[out_rank]
    for i in range(start, end):
        dst[i] = src0[i] + src1[i]


def run_exact_external_config[tp: Int](
    src: RankBuffers[DType.float32, tp, MutExternalOrigin],
    dst: RankBuffers[DType.float32, tp, MutExternalOrigin],
    mut pool: TestPool,
):
    var cfg = make_exact_config[tp, MutExternalOrigin, MutExternalOrigin](src, dst)
    var config_addr = Int(UnsafePointer(to=cfg))

    var jobs = MoveJobSet[ExactReduceKernel[tp]]()
    var workers = pool.get_capacity()
    var per_worker = (src.count + workers - 1) // workers
    for w in range(workers):
        var start = w * per_worker
        var end = min(start + per_worker, src.count)
        jobs.add(
            ExactReduceKernel[tp](config_addr, 0, start, end)
        )
    jobs.dispatch(pool)
    pool.join()

    _ = cfg.total_elements


def main():
    comptime TP = 2
    comptime N = 8
    var a = alloc[Float32](N)
    var b = alloc[Float32](N)
    var c = alloc[Float32](N)

    for i in range(N):
        a[i] = Float32(i)
        b[i] = Float32(10 + i)
        c[i] = Float32(-1)

    var src = RankBuffers[DType.float32, TP, MutExternalOrigin](count=N)
    src.insert_next(a)
    src.insert_next(b)

    var dst = RankBuffers[DType.float32, TP, MutExternalOrigin](count=N)
    dst.insert_next(c)
    dst.insert_next(b)

    var pool = TestPool(2, 0)
    run_exact_external_config[TP](src, dst, pool)

    for i in range(N):
        var expected = Float32(10 + 2 * i)
        debug_assert(c[i] == expected, "exact config reduction mismatch")

    print("origin exact config prototype ok")
