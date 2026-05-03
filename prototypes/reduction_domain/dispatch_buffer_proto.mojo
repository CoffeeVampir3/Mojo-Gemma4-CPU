from std.collections import InlineArray
from std.memory import Span, UnsafePointer, alloc

from threading.threading_traits import BurstKernel, BurstThreadPool


@fieldwise_init
struct TestPool(BurstThreadPool):
    var capacity: Int
    var ts: Int

    def get_capacity(self) -> Int:
        return self.capacity

    def dispatch[K: BurstKernel, origin: MutOrigin](
        mut self, kernels: Span[K, origin], num_jobs: Int):
        for i in range(num_jobs):
            var k = kernels[i]
            k.execute()

    def join(mut self):
        pass

    def last_worker_timestamp(self) -> Int:
        return self.ts

    def wake(mut self):
        pass

    def sleep(mut self):
        pass


struct DispatchBuffer[K: BurstKernel, capacity: Int = 128]:
    var items: InlineArray[Self.K, Self.capacity]
    var count: Int

    def __init__(out self):
        self.items = InlineArray[Self.K, Self.capacity](uninitialized=True)
        self.count = 0

    @always_inline
    def slot(mut self) -> UnsafePointer[Self.K, origin_of(self.items)]:
        debug_assert(self.count < Self.capacity, "DispatchBuffer overflow")
        var idx = self.count
        self.count += 1
        return UnsafePointer(to=self.items[idx])

    def dispatch[P: BurstThreadPool](mut self, mut pool: P):
        if self.count > 0:
            pool.dispatch(
                Span(ptr=UnsafePointer(to=self.items[0]), length=self.count),
                self.count)
        self.count = 0


@fieldwise_init
struct WriteKernel[dst_origin: MutOrigin](BurstKernel):
    var dst: UnsafePointer[Float32, Self.dst_origin]
    var idx: Int
    var val: Float32

    def execute(mut self):
        self.dst[self.idx] = self.val


@fieldwise_init
struct ScaleKernel[src_origin: ImmutOrigin, dst_origin: MutOrigin](BurstKernel):
    var src: UnsafePointer[Float32, Self.src_origin]
    var dst: UnsafePointer[Float32, Self.dst_origin]
    var start: Int
    var end: Int
    var scale: Float32

    def execute(mut self):
        for i in range(self.start, self.end):
            self.dst[i] = self.src[i] * self.scale


def test_stack_local_mut_origin():
    var storage = InlineArray[Float32, 4](fill=Float32(-1))
    var dst = UnsafePointer(to=storage[0])

    var pool = TestPool(2, 0)
    var buf = DispatchBuffer[WriteKernel[origin_of(storage)]]()

    buf.slot()[] = WriteKernel[origin_of(storage)](dst, 0, Float32(10))
    buf.slot()[] = WriteKernel[origin_of(storage)](dst, 1, Float32(20))
    buf.dispatch(pool)
    pool.join()

    debug_assert(storage[0] == Float32(10), "stack write 0 failed")
    debug_assert(storage[1] == Float32(20), "stack write 1 failed")
    print("  stack local MutOrigin: ok")


def test_heap_exact_origins():
    comptime N = 8
    var src_buf = alloc[Float32](N)
    var dst_buf = alloc[Float32](N)
    for i in range(N):
        src_buf[i] = Float32(i)
        dst_buf[i] = Float32(-1)

    comptime src_ro = ImmutOrigin(MutExternalOrigin)
    var pool = TestPool(2, 0)
    var buf = DispatchBuffer[ScaleKernel[src_ro, MutExternalOrigin]]()

    buf.slot()[] = ScaleKernel[src_ro, MutExternalOrigin](
        src_buf.as_immutable(), dst_buf, 0, N // 2, Float32(3))
    buf.slot()[] = ScaleKernel[src_ro, MutExternalOrigin](
        src_buf.as_immutable(), dst_buf, N // 2, N, Float32(3))
    buf.dispatch(pool)
    pool.join()

    for i in range(N):
        debug_assert(dst_buf[i] == Float32(i * 3), "heap exact origin mismatch")
    print("  heap exact origins: ok")


def test_multi_rank_reuse():
    comptime TP = 3
    comptime N = 6
    var src_buf = alloc[Float32](N)
    var dst_bufs = InlineArray[UnsafePointer[Float32, MutExternalOrigin], TP](
        uninitialized=True)
    for r in range(TP):
        dst_bufs[r] = alloc[Float32](N)
    for i in range(N):
        src_buf[i] = Float32(i)

    comptime src_ro = ImmutOrigin(MutExternalOrigin)
    var pool = TestPool(2, 0)
    var buf = DispatchBuffer[ScaleKernel[src_ro, MutExternalOrigin]]()

    for r in range(TP):
        var scale = Float32(r + 1)
        buf.slot()[] = ScaleKernel[src_ro, MutExternalOrigin](
            src_buf.as_immutable(), dst_bufs[r], 0, N // 2, scale)
        buf.slot()[] = ScaleKernel[src_ro, MutExternalOrigin](
            src_buf.as_immutable(), dst_bufs[r], N // 2, N, scale)
        buf.dispatch(pool)
        pool.join()

    for r in range(TP):
        for i in range(N):
            var expected = Float32(i * (r + 1))
            debug_assert(dst_bufs[r][i] == expected, "multi rank reuse mismatch")
    print("  multi-rank reuse: ok")


def test_zero_dispatch():
    var pool = TestPool(1, 0)
    var buf = DispatchBuffer[WriteKernel[MutExternalOrigin]]()
    buf.dispatch(pool)
    pool.join()
    debug_assert(pool.ts == 0, "zero dispatch should not call pool.dispatch")
    print("  zero dispatch: ok")


def main():
    test_stack_local_mut_origin()
    test_heap_exact_origins()
    test_multi_rank_reuse()
    test_zero_dispatch()
    print("dispatch buffer prototype ok")
