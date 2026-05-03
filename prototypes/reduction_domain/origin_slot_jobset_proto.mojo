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


struct SlotJobSet[K: BurstKernel]:
    var items: InlineArray[Self.K, 16]
    var count: Int

    def __init__(out self):
        self.items = InlineArray[Self.K, 16](uninitialized=True)
        self.count = 0

    def reserve(mut self) -> UnsafePointer[Self.K, origin_of(self.items)]:
        var idx = self.count
        self.count += 1
        return UnsafePointer(to=self.items[idx])

    def dispatch(mut self, mut pool: TestPool):
        if self.count > 0:
            pool.dispatch(
                Span(ptr=UnsafePointer(to=self.items[0]), length=self.count),
                self.count,
            )
        self.count = 0


@fieldwise_init
struct ExactStoreJob[src_origin: ImmutOrigin, dst_origin: MutOrigin](BurstKernel):
    var src0: UnsafePointer[Float32, Self.src_origin]
    var src1: UnsafePointer[Float32, Self.src_origin]
    var dst: UnsafePointer[Float32, Self.dst_origin]
    var start: Int
    var end: Int

    def execute(mut self):
        for i in range(self.start, self.end):
            self.dst[i] = self.src0[i] + self.src1[i]


def main():
    comptime N = 8
    var src_storage = InlineArray[Float32, N * 2](uninitialized=True)
    var dst_storage = InlineArray[Float32, N](uninitialized=True)
    for i in range(N):
        src_storage[i] = Float32(i)
        src_storage[N + i] = Float32(10 + i)
        dst_storage[i] = Float32(-1)

    var src0 = UnsafePointer(to=src_storage[0]).as_immutable()
    var src1 = UnsafePointer(to=src_storage[N]).as_immutable()
    var dst = UnsafePointer(to=dst_storage[0])
    var jobs = SlotJobSet[
        ExactStoreJob[ImmutOrigin(origin_of(src_storage)), origin_of(dst_storage)]
    ]()

    for w in range(2):
        var start = w * (N // 2)
        var end = start + (N // 2)
        var slot = jobs.reserve()
        slot[] = ExactStoreJob[
            ImmutOrigin(origin_of(src_storage)), origin_of(dst_storage)
        ](src0, src1, dst, start, end)
        _ = slot

    var pool = TestPool(2, 0)
    jobs.dispatch(pool)
    pool.join()

    for i in range(N):
        debug_assert(
            dst_storage[i] == Float32(10 + 2 * i),
            "slot jobset exact store mismatch",
        )

    print("origin slot jobset prototype ok")
