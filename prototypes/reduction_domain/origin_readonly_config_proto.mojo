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


struct TinyJobSet[K: BurstKernel]:
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
struct ReadonlySourceConfig[tp: Int, src_origin: ImmutOrigin]:
    comptime SrcPtr = UnsafePointer[Float32, Self.src_origin]
    comptime DstPtr = UnsafePointer[Float32, MutAnyOrigin]

    var src: InlineArray[Self.SrcPtr, Self.tp]
    var dst: InlineArray[Self.DstPtr, Self.tp]
    var total_elements: Int


@fieldwise_init
struct ReadonlySourceKernel[tp: Int, src_origin: ImmutOrigin](BurstKernel):
    var config_addr: Int
    var rank: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var config = UnsafePointer[
            ReadonlySourceConfig[Self.tp, Self.src_origin], MutAnyOrigin
        ](unsafe_from_address=self.config_addr)
        reduce_readonly_source_range[Self.tp, Self.src_origin](
            config, self.rank, self.start, self.end
        )


def make_readonly_source_config[
    src_origin: MutOrigin, dst_origin: MutOrigin, //, tp: Int
](
    src: RankBuffers[DType.float32, tp, src_origin],
    dst: RankBuffers[DType.float32, tp, dst_origin],
) -> ReadonlySourceConfig[tp, ImmutOrigin(src_origin)]:
    var src_ptrs = InlineArray[
        UnsafePointer[Float32, ImmutOrigin(src_origin)], tp
    ](uninitialized=True)
    var dst_ptrs = InlineArray[UnsafePointer[Float32, MutAnyOrigin], tp](
        uninitialized=True
    )
    for r in range(tp):
        src_ptrs[r] = src.ptrs[r].as_immutable()
        dst_ptrs[r] = dst.ptrs[r].as_any_origin()
    return ReadonlySourceConfig[tp, ImmutOrigin(src_origin)](
        src=src_ptrs, dst=dst_ptrs, total_elements=src.count
    )


def reduce_readonly_source_range[tp: Int, src_origin: ImmutOrigin](
    config: UnsafePointer[ReadonlySourceConfig[tp, src_origin], MutAnyOrigin],
    out_rank: Int,
    start: Int,
    end: Int,
):
    var src0 = config[].src[0]
    var src1 = config[].src[1]
    var dst = config[].dst[out_rank]
    for i in range(start, end):
        dst[i] = src0[i] + src1[i]


def run_readonly_source_config[
    src_origin: MutOrigin, dst_origin: MutOrigin, //, tp: Int
](
    src: RankBuffers[DType.float32, tp, src_origin],
    dst: RankBuffers[DType.float32, tp, dst_origin],
    mut pool: TestPool,
):
    var cfg = make_readonly_source_config[tp](src, dst)
    var config_addr = Int(UnsafePointer(to=cfg))

    var jobs = TinyJobSet[ReadonlySourceKernel[tp, ImmutOrigin(src_origin)]]()
    var workers = pool.get_capacity()
    var per_worker = (src.count + workers - 1) // workers
    for w in range(workers):
        var start = w * per_worker
        var end = min(start + per_worker, src.count)
        jobs.add(
            ReadonlySourceKernel[tp, ImmutOrigin(src_origin)](
                config_addr, 0, start, end
            )
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
    run_readonly_source_config[TP](src, dst, pool)

    for i in range(N):
        var expected = Float32(10 + 2 * i)
        debug_assert(c[i] == expected, "readonly source reduction mismatch")

    print("origin readonly config prototype ok")
