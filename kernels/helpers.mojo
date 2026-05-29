from std.collections import InlineArray
from std.memory import Span, UnsafePointer
from std.sys.info import simd_width_of
from simd_math import pick_port_unroll
from simd_math.fast_flags import set_subnormal_zeroing
from threading.threading_traits import BurstKernel, BurstThreadPool

from .dispatch_heuristics import (
    DISPATCH_BW_PRODUCT, PARALLEL_AMORTIZED_BYTES, MATMUL_DISPATCH_BW_PRODUCT,
)
from .profiling import Profiler, DispatchSpan


comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime F32Ptr  = UnsafePointer[Float32,  MutAnyOrigin]
comptime I32Ptr  = UnsafePointer[Int32,    MutAnyOrigin]
comptime W  = simd_width_of[DType.float32]()
comptime BW = simd_width_of[DType.bfloat16]()


trait OutputPartitionedKernel(BurstKernel):
    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int): ...


trait RangePartitionedKernel(OutputPartitionedKernel):
    @always_inline
    def install_range(mut self, start: Int, end: Int): ...

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.install_range(start, end)


trait WorkerRangePartitionedKernel(OutputPartitionedKernel):
    @always_inline
    def install_worker_range(
        mut self, worker_id: Int, start: Int, end: Int,
    ): ...

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.install_worker_range(worker_id, start, end)


@always_inline
def accumulate_scaled[
    src_dtype: DType, accum: DType, //,
    cols: Int,
](
    src: UnsafePointer[Scalar[src_dtype], MutAnyOrigin],
    weight: Scalar[accum],
    acc: UnsafePointer[Scalar[accum], MutAnyOrigin],
):
    """`acc[i] += weight * src[i]` over `cols` with PU-unrolled SIMD."""
    comptime width = simd_width_of[accum]()
    comptime PU = pick_port_unroll[width, cols]()
    comptime STRIDE = PU * width
    var w_vec = SIMD[accum, width](weight)
    for i in range(cols // STRIDE):
        comptime for p in range(PU):
            var off = i * STRIDE + p * width
            var vv = (src + off).load[width=width]().cast[accum]()
            var av = (acc + off).load[width=width]()
            (acc + off).store(vv.fma(w_vec, av))


@always_inline
def scale_unrolled[
    accum: DType, //,
    cols: Int,
](
    acc: UnsafePointer[Scalar[accum], MutAnyOrigin],
    factor: Scalar[accum],
):
    comptime width = simd_width_of[accum]()
    comptime PU = pick_port_unroll[width, cols]()
    comptime STRIDE = PU * width
    var f = SIMD[accum, width](factor)
    for i in range(cols // STRIDE):
        comptime for p in range(PU):
            var off = i * STRIDE + p * width
            (acc + off).store((acc + off).load[width=width]() * f)


@fieldwise_init
struct Chain[A: OutputPartitionedKernel, B: OutputPartitionedKernel](
    OutputPartitionedKernel
):
    var a: Self.A
    var b: Self.B

    def execute(mut self):
        self.a.execute()
        self.b.execute()

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.a.set_partition(worker_id, start, end)
        self.b.set_partition(worker_id, start, end)


struct RankBuffers[dtype: DType, tp: Int, origin: Origin]:
    var ptrs: InlineArray[UnsafePointer[Scalar[Self.dtype], Self.origin], Self.tp]
    var count: Int

    def __init__(out self, count: Int):
        self.ptrs = InlineArray[
            UnsafePointer[Scalar[Self.dtype], Self.origin], Self.tp,
        ](uninitialized=True)
        self.count = count


struct DispatchBuffer[K: BurstKernel, max_worker_count: Int = 128]:
    var items: InlineArray[Self.K, Self.max_worker_count]
    var count: Int

    def __init__(out self):
        comptime assert Self.max_worker_count > 0, (
            "max_worker_count must be positive")
        self.items = InlineArray[Self.K, Self.max_worker_count](uninitialized=True)
        self.count = 0

    @always_inline
    def slot(mut self) -> UnsafePointer[Self.K, origin_of(self.items)]:
        debug_assert(
            self.count < Self.max_worker_count,
            "DispatchBuffer overflow",
        )
        var idx = self.count
        self.count += 1
        return UnsafePointer(to=self.items[idx])

    def dispatch[P: BurstThreadPool](mut self, mut pool: P):
        if self.count > 0:
            pool.dispatch(
                Span(ptr=UnsafePointer(to=self.items[0]), length=self.count),
                self.count)
        self.count = 0


def join_all[P: BurstThreadPool, //, tp: Int](mut pools: List[P]):
    for r in range(tp):
        pools[r].join()


@fieldwise_init
struct FastFpInitKernel(BurstKernel):
    def execute(mut self):
        set_subnormal_zeroing()


def prime_fp_environment[
    P: BurstThreadPool, //, tp: Int, max_worker_count: Int = 128,
](mut pools: List[P]):
    set_subnormal_zeroing()
    var buf = DispatchBuffer[FastFpInitKernel, max_worker_count]()
    for r in range(tp):
        var cap = min(max_worker_count, pools[r].get_capacity())
        for _ in range(cap):
            buf.slot()[] = FastFpInitKernel()
        buf.dispatch(pools[r])
    join_all[tp](pools)


@always_inline
def worker_range(
    total: Int, num_workers: Int, worker_id: Int, base: Int = 0,
) -> Tuple[Int, Int]:
    var per_worker = (total + num_workers - 1) // num_workers
    var start = base + worker_id * per_worker
    var end = min(start + per_worker, base + total)
    if start >= base + total:
        return (base + total, base + total)
    return (start, end)


@always_inline
def recommended_workers[
    bw_product: Int = DISPATCH_BW_PRODUCT,
    amortized_bytes: Int = PARALLEL_AMORTIZED_BYTES,
](data_bytes: Int, capacity: Int) -> Int:
    if capacity <= 1:
        return capacity
    if data_bytes >= amortized_bytes:
        return capacity
    var target = data_bytes // bw_product
    var n = 1
    while (n + 1) * (n + 1) <= target and n < capacity:
        n += 1
    return n


@fieldwise_init
struct ArenaBases[tp: Int](Copyable, ImplicitlyCopyable):
    var addrs: InlineArray[Int, Self.tp]

    @staticmethod
    def uninitialized() -> Self:
        return Self(addrs=InlineArray[Int, Self.tp](uninitialized=True))

    @always_inline
    def __getitem__(self, rank: Int) -> Int:
        return self.addrs[rank]

    @always_inline
    def __setitem__(mut self, rank: Int, addr: Int):
        self.addrs[rank] = addr

    @always_inline
    def bind[T: AnyType](
        self, ptr: UnsafePointer[T, MutAnyOrigin],
    ) -> Binding[T, Self.tp]:
        return Binding[T, Self.tp](ptr, self)


@fieldwise_init
struct Binding[T: AnyType, tp: Int](Copyable, ImplicitlyCopyable):
    var ptr: UnsafePointer[Self.T, MutAnyOrigin]
    var bases: ArenaBases[Self.tp]

    @always_inline
    def __getitem__(self, rank: Int) -> UnsafePointer[Self.T, MutAnyOrigin]:
        return UnsafePointer[Self.T, MutAnyOrigin](
            unsafe_from_address=Int(self.ptr) + self.bases[rank] - self.bases[0])

    @always_inline
    def shifted(self, n: Int) -> Self:
        return Self(self.ptr + n, self.bases)


def tile_dispatch[
    K: OutputPartitionedKernel, P: BurstThreadPool, //,
    max_worker_count: Int = 128,
](mut buf: DispatchBuffer[K, max_worker_count], proto: K, mut pool: P, total: Int,
  base: Int = 0, num_workers: Int = 0) -> Int:
    """Returns the number of worker partitions actually queued."""
    if total <= 0:
        return 0
    var capacity = min(max_worker_count, pool.get_capacity())
    var workers = capacity if num_workers <= 0 else min(
        num_workers, capacity)
    workers = min(workers, total)
    for w in range(workers):
        var wr = worker_range(total, workers, w, base)
        var item = proto
        item.set_partition(w, wr[0], wr[1])
        buf.slot()[] = item
    buf.dispatch(pool)
    return workers


@always_inline
def saturate_workers(data_bytes: Int, capacity: Int) -> Int:
    return capacity


@always_inline
def matmul_workers(data_bytes: Int, capacity: Int) -> Int:
    return recommended_workers[MATMUL_DISPATCH_BW_PRODUCT, 1 << 30](
        data_bytes, capacity)


def fanout_dispatch[
    K: OutputPartitionedKernel, P: BurstThreadPool, Profile: Bool, N: Int, //,
    tp: Int,
    proto_for: def(Int) capturing [_] -> K,
    max_worker_count: Int = 128,
    worker_policy: def(
        data_bytes: Int, capacity: Int,
    ) thin -> Int = recommended_workers[DISPATCH_BW_PRODUCT, PARALLEL_AMORTIZED_BYTES],
    label: StaticString = "?",
](
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
    total: Int,
    data_bytes: Int,
    inline_threshold_bytes: Int = -1,
):
    if total <= 0:
        return
    var span = DispatchSpan[Profile]()
    if inline_threshold_bytes >= 0 and data_bytes <= inline_threshold_bytes:
        for r in range(tp):
            var k = proto_for(r)
            k.set_partition(0, 0, total)
            k.execute()
        span.finish_inline(prof, label)
        return
    var buf = DispatchBuffer[K, max_worker_count]()
    for r in range(tp):
        var cap = min(max_worker_count, pools[r].get_capacity())
        _ = tile_dispatch(buf, proto_for(r), pools[r], total,
            num_workers=worker_policy(data_bytes, cap))
    span.issued()
    join_all[tp](pools)
    span.finish[tp](prof, pools, label)


def fanout_dispatch_per_rank[
    K: OutputPartitionedKernel, P: BurstThreadPool, Profile: Bool, N: Int, //,
    tp: Int,
    proto_for: def(Int) capturing [_] -> K,
    total_for: def(Int) capturing [_] -> Int,
    data_bytes_for: def(Int) capturing [_] -> Int,
    max_worker_count: Int = 128,
    worker_policy: def(
        data_bytes: Int, capacity: Int,
    ) thin -> Int = recommended_workers[DISPATCH_BW_PRODUCT, PARALLEL_AMORTIZED_BYTES],
    label: StaticString = "?",
](
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
) -> InlineArray[Int, tp]:
    var nws = InlineArray[Int, tp](fill=0)
    var span = DispatchSpan[Profile]()
    var buf = DispatchBuffer[K, max_worker_count]()
    for r in range(tp):
        var total = total_for(r)
        if total <= 0:
            continue
        var cap = min(max_worker_count, pools[r].get_capacity())
        var nw = worker_policy(data_bytes_for(r), cap)
        nws[r] = tile_dispatch(
            buf, proto_for(r), pools[r], total, num_workers=nw)
    span.issued()
    join_all[tp](pools)
    span.finish[tp](prof, pools, label)
    return nws
