from std.collections import InlineArray
from std.memory import Span, UnsafePointer
from std.sys.info import simd_width_of
from notstdcollections import HeapMoveArray
from threading.threading_traits import BurstKernel, BurstThreadPool

from .dispatch_heuristics import (
    DISPATCH_BW_PRODUCT, PARALLEL_AMORTIZED_BYTES,
)


comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime F32Ptr  = UnsafePointer[Float32,  MutAnyOrigin]
comptime I32Ptr  = UnsafePointer[Int32,    MutAnyOrigin]
comptime W  = simd_width_of[DType.float32]()
comptime BW = simd_width_of[DType.bfloat16]()


trait OutputPartitionedKernel(BurstKernel):
    """A kernel that runs over a (worker_id, start, end) slice.

    `set_partition` is called by `tile_dispatch` once per worker, on a
    clone of the proto, to install that worker's indices. Kernels that
    only need (start, end) can ignore `worker_id`. Kernels that maintain
    per-worker scratch (e.g. flash-attention partials) consume it.
    """

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int): ...


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

    @always_inline
    def __getitem__(self, rank: Int) -> UnsafePointer[Scalar[Self.dtype], Self.origin]:
        return self.ptrs[rank]


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


def join_all[P: BurstThreadPool, //, tp: Int](mut pools: HeapMoveArray[P]):
    for r in range(tp):
        pools[r].join()


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
def recommended_workers(data_bytes: Int, capacity: Int) -> Int:
    if capacity <= 1:
        return capacity
    if data_bytes >= PARALLEL_AMORTIZED_BYTES:
        return capacity
    var target = data_bytes // DISPATCH_BW_PRODUCT
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

    @staticmethod
    def fill(addr: Int) -> Self:
        return Self(addrs=InlineArray[Int, Self.tp](fill=addr))

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


def fanout_dispatch[
    K: OutputPartitionedKernel, P: BurstThreadPool, //,
    tp: Int,
    proto_for: def(Int) capturing [_] -> K,
    max_worker_count: Int = 128,
    worker_policy: def(
        data_bytes: Int, capacity: Int,
    ) thin -> Int = recommended_workers,
](
    mut pools: HeapMoveArray[P],
    total: Int,
    data_bytes: Int,
    inline_threshold_bytes: Int = -1,
):
    if total <= 0:
        return
    if inline_threshold_bytes >= 0 and data_bytes <= inline_threshold_bytes:
        for r in range(tp):
            var k = proto_for(r)
            k.set_partition(0, 0, total)
            k.execute()
        return
    var buf = DispatchBuffer[K, max_worker_count]()
    for r in range(tp):
        var cap = min(max_worker_count, pools[r].get_capacity())
        _ = tile_dispatch(buf, proto_for(r), pools[r], total,
            num_workers=worker_policy(data_bytes, cap))
    join_all[tp](pools)


def fanout_dispatch_per_rank[
    K: OutputPartitionedKernel, P: BurstThreadPool, //,
    tp: Int,
    proto_for: def(Int) capturing [_] -> K,
    total_for: def(Int) capturing [_] -> Int,
    data_bytes_for: def(Int) capturing [_] -> Int,
    max_worker_count: Int = 128,
    worker_policy: def(
        data_bytes: Int, capacity: Int,
    ) thin -> Int = recommended_workers,
](
    mut pools: HeapMoveArray[P],
) -> InlineArray[Int, tp]:
    var nws = InlineArray[Int, tp](fill=0)
    var buf = DispatchBuffer[K, max_worker_count]()
    for r in range(tp):
        var total = total_for(r)
        if total <= 0:
            continue
        var cap = min(max_worker_count, pools[r].get_capacity())
        var nw = worker_policy(data_bytes_for(r), cap)
        nws[r] = tile_dispatch(
            buf, proto_for(r), pools[r], total, num_workers=nw)
    join_all[tp](pools)
    return nws
