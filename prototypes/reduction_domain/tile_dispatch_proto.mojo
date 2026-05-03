from std.collections import InlineArray
from std.memory import Span, UnsafePointer, alloc

from threading.threading_traits import BurstKernel, BurstThreadPool
from kernels.helpers import DispatchBuffer, worker_range


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


trait TiledKernel(BurstKernel):
    def tile(self, start: Int, end: Int) -> Self: ...


def tile_dispatch[
    K: TiledKernel, P: BurstThreadPool,
](mut buf: DispatchBuffer[K], proto: K, mut pool: P, total: Int, base: Int = 0):
    var workers = pool.get_capacity()
    for w in range(workers):
        var wr = worker_range(total, workers, w, base)
        buf.slot()[] = proto.tile(wr[0], wr[1])
    buf.dispatch(pool)


@fieldwise_init
struct ScaleKernel[src_origin: ImmutOrigin](TiledKernel):
    var src: UnsafePointer[Float32, Self.src_origin]
    var dst: UnsafePointer[Float32, MutAnyOrigin]
    var scale: Float32
    var start: Int
    var end: Int

    def execute(mut self):
        for i in range(self.start, self.end):
            self.dst[i] = self.src[i] * self.scale

    def tile(self, start: Int, end: Int) -> Self:
        return Self(self.src, self.dst, self.scale, start, end)


@fieldwise_init
struct SumKernel[src_origin: ImmutOrigin, cfg_origin: ImmutOrigin](TiledKernel):
    var cfg: UnsafePointer[SumConfig[Self.src_origin], Self.cfg_origin]
    var rank: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var c = self.cfg
        for i in range(self.start, self.end):
            c[].dst[self.rank][i] = c[].src[0][i] + c[].src[1][i]

    def tile(self, start: Int, end: Int) -> Self:
        return Self(self.cfg, self.rank, start, end)


@fieldwise_init
struct SumConfig[src_origin: ImmutOrigin]:
    var src: InlineArray[UnsafePointer[Float32, Self.src_origin], 2]
    var dst: InlineArray[UnsafePointer[Float32, MutAnyOrigin], 2]


def test_single_pool():
    comptime N = 16
    var src_buf = alloc[Float32](N)
    var dst_buf = alloc[Float32](N)
    for i in range(N):
        src_buf[i] = Float32(i)
        dst_buf[i] = Float32(-1)

    comptime src_ro = ImmutOrigin(MutExternalOrigin)
    var pool = TestPool(4, 0)
    var buf = DispatchBuffer[ScaleKernel[src_ro]]()

    var proto = ScaleKernel[src_ro](
        src_buf.as_immutable(), dst_buf.as_any_origin(), Float32(2), 0, 0)
    tile_dispatch(buf, proto, pool, N)
    pool.join()

    for i in range(N):
        debug_assert(dst_buf[i] == Float32(i * 2), "single pool mismatch")
    print("  single pool tile_dispatch: ok")


def test_multi_rank():
    comptime TP = 2
    comptime N = 8
    var a = alloc[Float32](N)
    var b = alloc[Float32](N)
    for i in range(N):
        a[i] = Float32(i)
        b[i] = Float32(10 + i)

    comptime src_ro = ImmutOrigin(MutExternalOrigin)
    var src_ptrs = InlineArray[UnsafePointer[Float32, src_ro], TP](uninitialized=True)
    src_ptrs[0] = a.as_immutable()
    src_ptrs[1] = b.as_immutable()
    var dst_ptrs = InlineArray[UnsafePointer[Float32, MutAnyOrigin], TP](uninitialized=True)
    dst_ptrs[0] = a.as_any_origin()
    dst_ptrs[1] = b.as_any_origin()

    var cfg = SumConfig[src_ro](src=src_ptrs, dst=dst_ptrs)
    var config = UnsafePointer(to=cfg).as_immutable()
    comptime cfg_ro = ImmutOrigin(origin_of(cfg))

    var pool = TestPool(2, 0)
    var chunk = N // TP

    var buf = DispatchBuffer[SumKernel[src_ro, cfg_ro]]()
    for r in range(TP):
        tile_dispatch(buf,
            SumKernel[src_ro, cfg_ro](config, r, 0, 0),
            pool, chunk, r * chunk)
    pool.join()

    for r in range(TP):
        for i in range(chunk):
            var idx = r * chunk + i
            var expected = Float32(10 + 2 * idx)
            debug_assert(dst_ptrs[r][idx] == expected, "multi rank mismatch")
    print("  multi-rank tile_dispatch: ok")


def test_varying_proto_per_rank():
    comptime N = 8
    var src_buf = alloc[Float32](N)
    var dst0 = alloc[Float32](N)
    var dst1 = alloc[Float32](N)
    for i in range(N):
        src_buf[i] = Float32(i)

    comptime src_ro = ImmutOrigin(MutExternalOrigin)
    var pool = TestPool(2, 0)
    var buf = DispatchBuffer[ScaleKernel[src_ro]]()

    tile_dispatch(buf,
        ScaleKernel[src_ro](src_buf.as_immutable(), dst0.as_any_origin(), Float32(3), 0, 0),
        pool, N)
    pool.join()

    tile_dispatch(buf,
        ScaleKernel[src_ro](src_buf.as_immutable(), dst1.as_any_origin(), Float32(5), 0, 0),
        pool, N)
    pool.join()

    for i in range(N):
        debug_assert(dst0[i] == Float32(i * 3), "rank0 scale mismatch")
        debug_assert(dst1[i] == Float32(i * 5), "rank1 scale mismatch")
    print("  varying proto per rank: ok")


def main():
    test_single_pool()
    test_multi_rank()
    test_varying_proto_per_rank()
    print("tile dispatch prototype ok")
