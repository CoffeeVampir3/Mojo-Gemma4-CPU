from std.collections import InlineArray
from std.memory import UnsafePointer, alloc
from std.sys.info import simd_width_of

from numa import NumaTopology
from notstdcollections import HeapMoveArray
from threading.threading_traits import BurstKernel, BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.helpers import (
    DispatchBuffer, recommended_workers, worker_range, join_all,
)


comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
comptime W = simd_width_of[DType.float32]()


@fieldwise_init
struct WorkRange(Copyable, ImplicitlyCopyable):
    var start: Int
    var end: Int


@fieldwise_init
struct WorkPlan[max_workers: Int](Copyable, ImplicitlyCopyable):
    var inline: Bool
    var num_workers: Int
    var ranges: InlineArray[WorkRange, Self.max_workers]


trait TileStrategy:
    @staticmethod
    def total() -> Int: ...

    @staticmethod
    def data_bytes() -> Int: ...

    @staticmethod
    def map_unit_range(start: Int, end: Int) -> WorkRange:
        return WorkRange(start, end)

    @staticmethod
    def make_plan[max_workers: Int](capacity: Int) -> WorkPlan[max_workers]:
        var total = Self.total()
        var nw = recommended_workers(
            Self.data_bytes(), min(max_workers, capacity))
        nw = min(nw, total)
        var ranges = InlineArray[WorkRange, max_workers](fill=WorkRange(0, 0))
        if nw <= 1:
            ranges[0] = Self.map_unit_range(0, total)
            return WorkPlan[max_workers](True, 1, ranges)
        for w in range(nw):
            var wr = worker_range(total, nw, w)
            ranges[w] = Self.map_unit_range(wr[0], wr[1])
        return WorkPlan[max_workers](False, nw, ranges)


struct RowTile[rows: Int, bytes_per_row: Int](TileStrategy, Copyable):
    @staticmethod
    def total() -> Int: return Self.rows
    @staticmethod
    def data_bytes() -> Int: return Self.rows * Self.bytes_per_row


struct LaneStridedTile[
    total_elements: Int, stride: Int, bytes_per_lane: Int,
](TileStrategy, Copyable):
    @staticmethod
    def total() -> Int: return Self.total_elements // Self.stride
    @staticmethod
    def data_bytes() -> Int: return Self.total_elements * Self.bytes_per_lane
    @staticmethod
    def map_unit_range(start: Int, end: Int) -> WorkRange:
        return WorkRange(start * Self.stride, end * Self.stride)


trait Tiled(TrivialRegisterPassable):
    comptime Tile: TileStrategy
    def execute_range(mut self, start: Int, end: Int): ...


trait WithWorkerScratch:
    comptime SCRATCH_BYTES_PER_WORKER: Int
    def execute_range_with_id(
        mut self, worker_id: Int, start: Int, end: Int,
    ): ...


@fieldwise_init
struct Slice[K: Tiled](BurstKernel):
    var kernel: Self.K
    var start: Int
    var end: Int

    def execute(mut self):
        self.kernel.execute_range(self.start, self.end)


@fieldwise_init
struct ScratchSlice[K: Tiled & WithWorkerScratch](BurstKernel):
    var kernel: Self.K
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        self.kernel.execute_range_with_id(
            self.worker_id, self.start, self.end)


def dispatch[
    K: Tiled, P: BurstThreadPool, //,
    tp: Int, max_workers: Int = 128,
    *,
    build: def[](rank: Int) capturing [_] -> K,
](mut pools: HeapMoveArray[P]):
    var buf = DispatchBuffer[Slice[K], max_workers]()
    for r in range(tp):
        var k = build(r)
        var plan = K.Tile.make_plan[max_workers](pools[r].get_capacity())
        if plan.inline:
            var s = Slice[K](k, plan.ranges[0].start, plan.ranges[0].end)
            s.execute()
            continue
        for w in range(plan.num_workers):
            buf.slot()[] = Slice[K](
                k, plan.ranges[w].start, plan.ranges[w].end)
        buf.dispatch(pools[r])
    join_all[tp](pools)


def dispatch_with_scratch[
    K: Tiled & WithWorkerScratch, P: BurstThreadPool, //,
    tp: Int, max_workers: Int = 128,
    *,
    build: def[](rank: Int) capturing [_] -> K,
](mut pools: HeapMoveArray[P]):
    var buf = DispatchBuffer[ScratchSlice[K], max_workers]()
    for r in range(tp):
        var k = build(r)
        var plan = K.Tile.make_plan[max_workers](pools[r].get_capacity())
        if plan.inline:
            var s = ScratchSlice[K](
                k, 0, plan.ranges[0].start, plan.ranges[0].end)
            s.execute()
            continue
        for w in range(plan.num_workers):
            buf.slot()[] = ScratchSlice[K](
                k, w, plan.ranges[w].start, plan.ranges[w].end)
        buf.dispatch(pools[r])
    join_all[tp](pools)


@fieldwise_init
struct DemoGemv[rows: Int, cols: Int](Tiled):
    comptime Tile = RowTile[Self.rows, Self.cols * 2]

    var x: BF16Ptr
    var w: BF16Ptr
    var out: BF16Ptr

    def execute_range(mut self, start: Int, end: Int):
        for row in range(start, end):
            var acc = Float32(0)
            for c in range(Self.cols):
                acc += (self.x + c)[].cast[DType.float32]() * (
                    self.w + row * Self.cols + c)[].cast[DType.float32]()
            (self.out + row)[] = Scalar[DType.bfloat16](acc)


@fieldwise_init
struct DemoLaneFill[total: Int, stride: Int](Tiled):
    comptime Tile = LaneStridedTile[Self.total, Self.stride, Self.stride * 4]

    var out: F32Ptr
    var val: Float32

    def execute_range(mut self, start: Int, end: Int):
        for i in range(start, end):
            (self.out + i)[] = self.val


@fieldwise_init
struct DemoScratchTouch[total: Int](Tiled, WithWorkerScratch):
    comptime Tile = RowTile[Self.total, 4]
    comptime SCRATCH_BYTES_PER_WORKER = 8

    var out: F32Ptr
    var who: UnsafePointer[Scalar[DType.int32], MutAnyOrigin]

    def execute_range(mut self, start: Int, end: Int):
        self.execute_range_with_id(0, start, end)

    def execute_range_with_id(
        mut self, worker_id: Int, start: Int, end: Int,
    ):
        for i in range(start, end):
            (self.out + i)[] = Float32(i)
            (self.who + i)[] = Int32(worker_id)


def run_check[P: BurstThreadPool, //, degree: Int](
    var pools: HeapMoveArray[P],
):
    comptime ROWS = 32
    comptime COLS = 32
    comptime TOTAL = 256
    comptime STRIDE = 8
    comptime N_TOUCH = 64

    var xs = InlineArray[BF16Ptr, degree](uninitialized=True)
    var ws = InlineArray[BF16Ptr, degree](uninitialized=True)
    var os = InlineArray[BF16Ptr, degree](uninitialized=True)
    var ls = InlineArray[F32Ptr, degree](uninitialized=True)
    var ts = InlineArray[F32Ptr, degree](uninitialized=True)
    var ws_who = InlineArray[
        UnsafePointer[Scalar[DType.int32], MutAnyOrigin], degree,
    ](uninitialized=True)
    for r in range(degree):
        var x = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](COLS)))
        var w = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](ROWS * COLS)))
        var o = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(alloc[Scalar[DType.bfloat16]](ROWS)))
        var l = UnsafePointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(alloc[Scalar[DType.float32]](TOTAL)))
        var t = UnsafePointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(alloc[Scalar[DType.float32]](N_TOUCH)))
        var who = UnsafePointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=Int(alloc[Scalar[DType.int32]](N_TOUCH)))
        for i in range(COLS): (x + i)[] = Scalar[DType.bfloat16](1.0)
        for i in range(ROWS * COLS): (w + i)[] = Scalar[DType.bfloat16](0.5)
        for i in range(ROWS): (o + i)[] = Scalar[DType.bfloat16](-1.0)
        for i in range(TOTAL): (l + i)[] = Float32(-1.0)
        for i in range(N_TOUCH): (t + i)[] = Float32(-1.0)
        for i in range(N_TOUCH): (who + i)[] = Int32(-1)
        xs[r] = x; ws[r] = w; os[r] = o; ls[r] = l; ts[r] = t
        ws_who[r] = who

    @parameter
    def gemv(rank: Int) -> DemoGemv[ROWS, COLS]:
        return DemoGemv[ROWS, COLS](xs[rank], ws[rank], os[rank])
    dispatch[tp=degree, max_workers=128, build=gemv](pools)

    @parameter
    def fill(rank: Int) -> DemoLaneFill[TOTAL, STRIDE]:
        return DemoLaneFill[TOTAL, STRIDE](ls[rank], Float32(7.5))
    dispatch[tp=degree, max_workers=128, build=fill](pools)

    @parameter
    def touch(rank: Int) -> DemoScratchTouch[N_TOUCH]:
        return DemoScratchTouch[N_TOUCH](ts[rank], ws_who[rank])
    dispatch_with_scratch[tp=degree, max_workers=128, build=touch](pools)

    var ok = True
    for r in range(degree):
        for i in range(ROWS):
            if (os[r] + i)[] != Scalar[DType.bfloat16](Float32(COLS) * 0.5):
                ok = False; print("gemv mismatch r=", r, "i=", i); break
        for i in range(TOTAL):
            if (ls[r] + i)[] != Float32(7.5):
                ok = False; print("fill mismatch r=", r, "i=", i); break
        for i in range(N_TOUCH):
            if (ts[r] + i)[] != Float32(i):
                ok = False; print("touch value mismatch r=", r, "i=", i); break
    if ok:
        print("Prototype C [degree=", degree,
              "] passed (gemv + laneStrided + scratch)")


def main():
    var topo = NumaTopology()
    @parameter
    def go[P: BurstThreadPool, //, degree: Int](var pools: HeapMoveArray[P]):
        run_check[degree=degree](pools^)
    with_topological_rank_dispatch[
        power_of_two_unrolling=3, dispatch=go,
    ](topo, "isolated", "cold")
