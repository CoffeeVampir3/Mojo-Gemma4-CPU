from std.algorithm import vectorize
from std.collections import InlineArray
from std.memory import Span, UnsafePointer, alloc, memcpy
from std.sys.info import simd_width_of

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


comptime DstPtr[dtype: DType] = UnsafePointer[Scalar[dtype], MutAnyOrigin]


def frozen_ptr[T: AnyType, origin: Origin](
    ref[origin] value: T,
) -> UnsafePointer[T, ImmutOrigin(origin)]:
    return UnsafePointer(to=value).as_immutable()


@always_inline
def reduce_sources_to[
    src_dtype: DType, dst_dtype: DType, tp: Int, src_origin: ImmutOrigin,
    Accum: DType = DType.float32,
](
    srcs: InlineArray[UnsafePointer[Scalar[src_dtype], src_origin], tp],
    dst: DstPtr[dst_dtype], start: Int, end: Int,
):
    def step[width: Int](idx: Int) {read}:
        var pos = start + idx
        var acc = (srcs[0] + pos).load[width=width]().cast[Accum]()
        for r in range(1, tp):
            acc += (srcs[r] + pos).load[width=width]().cast[Accum]()
        (dst + pos).store(acc.cast[dst_dtype]())

    vectorize[simd_width_of[Accum]()](end - start, step)


struct SlotJobSet[K: BurstKernel]:
    var items: InlineArray[Self.K, 128]
    var count: Int

    def __init__(out self):
        self.items = InlineArray[Self.K, 128](uninitialized=True)
        self.count = 0

    def reserve(mut self) -> UnsafePointer[Self.K, origin_of(self.items)]:
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
struct ReduceConfig[tp: Int, src_origin: ImmutOrigin]:
    var src: InlineArray[UnsafePointer[Scalar[DType.bfloat16], Self.src_origin], Self.tp]
    var dst: InlineArray[DstPtr[DType.bfloat16], Self.tp]
    var total_elements: Int
    var chunk: Int
    var rem: Int


@fieldwise_init
struct ReduceStoreKernel[tp: Int, src_origin: ImmutOrigin, cfg_origin: ImmutOrigin](BurstKernel):
    var config: UnsafePointer[ReduceConfig[Self.tp, Self.src_origin], Self.cfg_origin]
    var rank: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var cfg = self.config
        var srcs = InlineArray[
            UnsafePointer[Scalar[DType.bfloat16], Self.src_origin], Self.tp,
        ](uninitialized=True)
        for r in range(Self.tp):
            srcs[r] = cfg[].src[r]
        reduce_sources_to[DType.bfloat16, DType.bfloat16, Self.tp, Self.src_origin](
            srcs, cfg[].dst[self.rank], self.start, self.end)


@fieldwise_init
struct CopyKernel[src_origin: ImmutOrigin, dst_origin: MutOrigin](BurstKernel):
    var src: UnsafePointer[Float32, Self.src_origin]
    var dst: UnsafePointer[Float32, Self.dst_origin]
    var start: Int
    var end: Int

    def execute(mut self):
        for i in range(self.start, self.end):
            self.dst[i] = self.src[i] * 2.0


@fieldwise_init
struct WriteJob[dst_origin: MutOrigin](BurstKernel):
    var dst: UnsafePointer[Float32, Self.dst_origin]
    var val: Float32

    def execute(mut self):
        self.dst[] = self.val


def test_frozen_ptr_with_config():
    comptime TP = 2
    comptime N = 8
    var a = alloc[Scalar[DType.bfloat16]](N)
    var b = alloc[Scalar[DType.bfloat16]](N)
    for i in range(N):
        a[i] = Scalar[DType.bfloat16](Float32(i))
        b[i] = Scalar[DType.bfloat16](Float32(10 + i))

    comptime src_ro = ImmutOrigin(MutExternalOrigin)
    var src_ptrs = InlineArray[UnsafePointer[Scalar[DType.bfloat16], src_ro], TP](
        uninitialized=True)
    src_ptrs[0] = a.as_immutable()
    src_ptrs[1] = b.as_immutable()

    var dst_ptrs = InlineArray[DstPtr[DType.bfloat16], TP](uninitialized=True)
    dst_ptrs[0] = a.as_any_origin()
    dst_ptrs[1] = b.as_any_origin()

    var cfg = ReduceConfig[TP, src_ro](
        src=src_ptrs, dst=dst_ptrs, total_elements=N, chunk=N // TP, rem=0)
    var config = frozen_ptr(cfg)

    reduce_sources_to[DType.bfloat16, DType.bfloat16, TP, src_ro](
        src_ptrs, config[].dst[0], 0, N)

    for r in range(1, TP):
        memcpy(dest=config[].dst[r], src=config[].dst[0], count=N)

    for i in range(N):
        var expected = Float32(10 + 2 * i)
        debug_assert(Float32(a[i]) == expected, "frozen inline rank0 mismatch")
        debug_assert(Float32(b[i]) == expected, "frozen inline rank1 mismatch")

    print("  frozen_ptr + reduce_sources_to inline: ok")


def test_slot_jobset_dispatched():
    comptime TP = 2
    comptime N = 8
    var a = alloc[Scalar[DType.bfloat16]](N)
    var b = alloc[Scalar[DType.bfloat16]](N)
    for i in range(N):
        a[i] = Scalar[DType.bfloat16](Float32(i))
        b[i] = Scalar[DType.bfloat16](Float32(10 + i))

    comptime src_ro = ImmutOrigin(MutExternalOrigin)
    var src_ptrs = InlineArray[UnsafePointer[Scalar[DType.bfloat16], src_ro], TP](
        uninitialized=True)
    src_ptrs[0] = a.as_immutable()
    src_ptrs[1] = b.as_immutable()

    var dst_ptrs = InlineArray[DstPtr[DType.bfloat16], TP](uninitialized=True)
    dst_ptrs[0] = a.as_any_origin()
    dst_ptrs[1] = b.as_any_origin()

    var cfg = ReduceConfig[TP, src_ro](
        src=src_ptrs, dst=dst_ptrs, total_elements=N, chunk=N // TP, rem=0)
    var config = frozen_ptr(cfg)
    comptime cfg_ro = ImmutOrigin(origin_of(cfg))

    var pool = TestPool(2, 0)

    var jobs = SlotJobSet[ReduceStoreKernel[TP, src_ro, cfg_ro]]()
    for r in range(TP):
        var slot = jobs.reserve()
        slot[] = ReduceStoreKernel[TP, src_ro, cfg_ro](
            config, r, r * (N // TP), (r + 1) * (N // TP))
        _ = slot
    jobs.dispatch(pool)
    pool.join()

    for r in range(TP):
        var rank_start = r * (N // TP)
        for i in range(N // TP):
            var expected = Float32(10 + 2 * (rank_start + i))
            debug_assert(
                Float32(dst_ptrs[r][rank_start + i]) == expected,
                "slot dispatch mismatch")

    print("  SlotJobSet dispatched reduce: ok")


def test_exact_dst_origin_with_slots():
    comptime N = 8
    var src_buf = alloc[Float32](N)
    var dst_buf = alloc[Float32](N)
    for i in range(N):
        src_buf[i] = Float32(i)
        dst_buf[i] = Float32(-1)

    comptime src_ro = ImmutOrigin(MutExternalOrigin)
    var pool = TestPool(2, 0)
    var jobs = SlotJobSet[CopyKernel[src_ro, MutExternalOrigin]]()

    for w in range(2):
        var slot = jobs.reserve()
        slot[] = CopyKernel[src_ro, MutExternalOrigin](
            src_buf.as_immutable(), dst_buf, w * (N // 2), (w + 1) * (N // 2))
        _ = slot

    jobs.dispatch(pool)
    pool.join()

    for i in range(N):
        debug_assert(dst_buf[i] == Float32(i * 2), "exact dst origin mismatch")

    print("  SlotJobSet with exact MutOrigin dst: ok")


def test_add_vs_reserve_aliasing():
    var target = alloc[Float32](1)
    target[] = Float32(-1)

    var pool = TestPool(1, 0)

    var slot_jobs = SlotJobSet[WriteJob[MutExternalOrigin]]()
    var slot = slot_jobs.reserve()
    slot[] = WriteJob[MutExternalOrigin](target, Float32(42))
    _ = slot
    slot_jobs.dispatch(pool)
    pool.join()
    debug_assert(target[] == Float32(42), "reserve write failed")

    print("  add vs reserve aliasing: ok")


def main():
    test_frozen_ptr_with_config()
    test_slot_jobset_dispatched()
    test_exact_dst_origin_with_slots()
    test_add_vs_reserve_aliasing()
    print("deep alias probe ok")
