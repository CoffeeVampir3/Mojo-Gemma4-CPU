from std.collections import InlineArray
from std.memory import Span, UnsafePointer

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


struct DirectJobSet[K: BurstKernel]:
    var items: InlineArray[Self.K, 16]
    var count: Int

    def __init__(out self):
        self.items = InlineArray[Self.K, 16](uninitialized=True)
        self.count = 0

    def dispatch(mut self, mut pool: TestPool):
        if self.count > 0:
            pool.dispatch(
                Span(ptr=UnsafePointer(to=self.items[0]), length=self.count),
                self.count,
            )
        self.count = 0


@fieldwise_init
struct ExactStoreConfig[src_origin: ImmutOrigin, dst_origin: MutOrigin]:
    var src0: UnsafePointer[Float32, Self.src_origin]
    var src1: UnsafePointer[Float32, Self.src_origin]
    var dst: UnsafePointer[Float32, Self.dst_origin]
    var n: Int


@fieldwise_init
struct ExactStoreJob[
    src_origin: ImmutOrigin, dst_origin: MutOrigin, cfg_origin: MutOrigin,
](BurstKernel):
    var config: UnsafePointer[
        ExactStoreConfig[Self.src_origin, Self.dst_origin], Self.cfg_origin
    ]
    var start: Int
    var end: Int

    def execute(mut self):
        var cfg = self.config
        for i in range(self.start, self.end):
            cfg[].dst[i] = cfg[].src0[i] + cfg[].src1[i]


def run_direct_exact_store[
    src_origin: ImmutOrigin, dst_origin: MutOrigin, //, n: Int
](
    src0: UnsafePointer[Float32, src_origin],
    src1: UnsafePointer[Float32, src_origin],
    dst: UnsafePointer[Float32, dst_origin],
    mut pool: TestPool,
):
    var cfg = ExactStoreConfig[src_origin, dst_origin](src0, src1, dst, n)
    var config = UnsafePointer(to=cfg)
    var jobs = DirectJobSet[
        ExactStoreJob[src_origin, dst_origin, origin_of(cfg)]
    ]()
    var workers = pool.get_capacity()
    var per_worker = (n + workers - 1) // workers
    for w in range(workers):
        var start = w * per_worker
        var end = min(start + per_worker, n)
        jobs.items[jobs.count] = ExactStoreJob[
            src_origin, dst_origin, origin_of(cfg)
        ](config, start, end)
        jobs.count += 1
    jobs.dispatch(pool)
    pool.join()

    _ = cfg.n


def main():
    comptime N = 8
    var src_storage = InlineArray[Float32, N * 2](uninitialized=True)
    var dst_storage = InlineArray[Float32, N](uninitialized=True)
    for i in range(N):
        src_storage[i] = Float32(i)
        src_storage[N + i] = Float32(10 + i)
        dst_storage[i] = Float32(-1)

    var pool = TestPool(2, 0)
    run_direct_exact_store[N](
        UnsafePointer(to=src_storage[0]).as_immutable(),
        UnsafePointer(to=src_storage[N]).as_immutable(),
        UnsafePointer(to=dst_storage[0]),
        pool,
    )

    for i in range(N):
        var expected = Float32(10 + 2 * i)
        debug_assert(dst_storage[i] == expected, "direct exact store mismatch")

    print("origin direct job storage prototype ok")
