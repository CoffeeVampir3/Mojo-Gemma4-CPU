from std.collections import InlineArray
from std.memory import Span, UnsafePointer, alloc

from notstdcollections import HeapMoveArray
from threading.threading_traits import (
    BurstKernel, BurstThreadPool, SleepableThreadPool,
)
from kernels.helpers import (
    DispatchBuffer, OutputPartitionedKernel,
    tile_dispatch, recommended_workers, worker_range, join_all,
)


struct SerialPool(BurstThreadPool):
    var capacity: Int

    def __init__(out self, capacity: Int):
        self.capacity = capacity

    def get_capacity(self) -> Int:
        return self.capacity

    def dispatch[K: BurstKernel, origin: MutOrigin](
        mut self, kernels: Span[K, origin], num_jobs: Int,
    ):
        for i in range(num_jobs):
            kernels[i].execute()

    def join(mut self):
        pass

    def last_worker_timestamp(self) -> Int:
        return 0

    def wake(mut self):
        pass

    def sleep(mut self):
        pass


@fieldwise_init
struct AddKernel(OutputPartitionedKernel):
    var dst: UnsafePointer[Int, MutAnyOrigin]
    var addend: Int
    var start: Int
    var end: Int

    def execute(mut self):
        for i in range(self.start, self.end):
            (self.dst + i)[] = (self.dst + i)[] + self.addend

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.dst, self.addend, start, end)


@fieldwise_init
struct WorkerStampKernel(BurstKernel):
    var dst: UnsafePointer[Int, MutAnyOrigin]
    var rank: Int
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        for i in range(self.start, self.end):
            (self.dst + i)[] = self.rank * 100 + self.worker_id


def fan_out_ranks[
    P: BurstThreadPool, K: OutputPartitionedKernel, //,
    *,
    tp: Int,
    proto: def(Int) capturing [_] -> K,
    inline: def(Int) capturing [_] -> None,
](
    mut pools: HeapMoveArray[P],
    total: Int,
    data_bytes: Int,
    inline_threshold: Int = 0,
):
    if inline_threshold > 0 and total <= inline_threshold:
        for r in range(tp):
            inline(r)
        return

    var buf = DispatchBuffer[K]()
    for r in range(tp):
        var nw = recommended_workers(data_bytes, pools[r].get_capacity())
        tile_dispatch(buf, proto(r), pools[r], total, num_workers=nw)
    join_all[tp](pools)


def fan_out_ranks_per_worker[
    P: BurstThreadPool, K: BurstKernel, //,
    *,
    tp: Int,
    proto: def(Int, Int, Int, Int) capturing [_] -> K,
](
    mut pools: HeapMoveArray[P],
    total: Int,
    data_bytes: Int,
    cap_to_total: Bool = False,
):
    var buf = DispatchBuffer[K]()
    for r in range(tp):
        var cap = pools[r].get_capacity()
        var nw = recommended_workers(data_bytes, cap)
        if cap_to_total:
            nw = min(nw, total)
        for w in range(nw):
            var wr = worker_range(total, nw, w)
            buf.slot()[] = proto(r, w, wr[0], wr[1])
        buf.dispatch(pools[r])
    join_all[tp](pools)


def make_pools[tp: Int](capacity: Int) -> HeapMoveArray[SerialPool]:
    var pools = HeapMoveArray[SerialPool](tp)
    for _ in range(tp):
        pools.push(SerialPool(capacity))
    return pools^


def make_zero_buffers[tp: Int, N: Int]() -> InlineArray[
    UnsafePointer[Int, MutAnyOrigin], tp,
]:
    var arr = InlineArray[UnsafePointer[Int, MutAnyOrigin], tp](
        uninitialized=True)
    for r in range(tp):
        var p = alloc[Int](N).unsafe_origin_cast[MutAnyOrigin]()
        for i in range(N):
            (p + i)[] = 0
        arr[r] = p
    return arr


def free_buffers[tp: Int](
    arr: InlineArray[UnsafePointer[Int, MutAnyOrigin], tp],
):
    for r in range(tp):
        arr[r].free()


def expect(label: String, ok: Bool):
    if ok:
        print("  ", label, "OK")
    else:
        print("  ", label, "FAIL")


def demo_fan_out_parallel():
    print("=== fan_out_ranks parallel path (total > inline_threshold) ===")
    comptime TP = 4
    comptime N = 256
    var bufs = make_zero_buffers[TP, N]()
    var pools = make_pools[TP](capacity=4)

    @parameter
    def proto(r: Int) -> AddKernel:
        return AddKernel(bufs[r], r + 1, 0, 0)

    @parameter
    def inline(r: Int):
        for i in range(N):
            (bufs[r] + i)[] = (bufs[r] + i)[] + (r + 1)

    fan_out_ranks[tp=TP, proto=proto, inline=inline](
        pools, total=N, data_bytes=N * 8, inline_threshold=8)

    var ok = True
    for r in range(TP):
        for i in range(N):
            if (bufs[r] + i)[] != r + 1:
                ok = False
    expect("each rank's buffer holds rank+1 in every slot", ok)
    free_buffers[TP](bufs)


def demo_fan_out_inline():
    print("=== fan_out_ranks inline path (total <= inline_threshold) ===")
    comptime TP = 3
    comptime N = 4
    var bufs = make_zero_buffers[TP, N]()
    var pools = make_pools[TP](capacity=4)

    @parameter
    def proto(r: Int) -> AddKernel:
        return AddKernel(bufs[r], (r + 1) * 10, 0, 0)

    @parameter
    def inline(r: Int):
        for i in range(N):
            (bufs[r] + i)[] = (bufs[r] + i)[] + (r + 1) * 10

    fan_out_ranks[tp=TP, proto=proto, inline=inline](
        pools, total=N, data_bytes=N * 8, inline_threshold=8)

    var ok = True
    for r in range(TP):
        for i in range(N):
            if (bufs[r] + i)[] != (r + 1) * 10:
                ok = False
    expect("inline branch wrote (r+1)*10 to every slot", ok)
    free_buffers[TP](bufs)


def demo_per_worker():
    print("=== fan_out_ranks_per_worker (worker_id flows in) ===")
    comptime TP = 2
    comptime N = 32
    comptime CAP = 4
    var bufs = make_zero_buffers[TP, N]()
    for r in range(TP):
        for i in range(N):
            (bufs[r] + i)[] = -1
    var pools = make_pools[TP](capacity=CAP)

    @parameter
    def proto(r: Int, w: Int, s: Int, e: Int) -> WorkerStampKernel:
        return WorkerStampKernel(bufs[r], r, w, s, e)

    fan_out_ranks_per_worker[tp=TP, proto=proto](
        pools, total=N, data_bytes=4 * 1024 * 1024)

    var per_rank_per_worker = (N + CAP - 1) // CAP
    var ok = True
    for r in range(TP):
        for w in range(CAP):
            var s = w * per_rank_per_worker
            var e = min(s + per_rank_per_worker, N)
            for i in range(s, e):
                if (bufs[r] + i)[] != r * 100 + w:
                    ok = False
    expect("each (rank,worker) tile stamped r*100+w", ok)
    free_buffers[TP](bufs)


def main():
    demo_fan_out_parallel()
    demo_fan_out_inline()
    demo_per_worker()
