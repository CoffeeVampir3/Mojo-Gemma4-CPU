from std.collections import InlineArray
from std.memory import Span, UnsafePointer, alloc
from std.os import abort

from kernels.helpers import (
    DispatchBuffer, OutputPartitionedKernel,
    fanout_dispatch_per_rank, saturate_workers, tile_dispatch,
)
from notstdcollections import HeapMoveArray
from threading.threading_traits import BurstKernel, BurstThreadPool


comptime IPtr = UnsafePointer[Int, MutExternalOrigin]


@fieldwise_init
struct TestPool(BurstThreadPool):
    var capacity: Int
    var timestamp: Int

    def get_capacity(self) -> Int:
        return self.capacity

    def dispatch[K: BurstKernel, origin: MutOrigin](
        mut self, kernels: Span[K, origin], num_jobs: Int):
        for i in range(num_jobs):
            var kernel = kernels[i]
            kernel.execute()
        self.timestamp += 1

    def join(mut self):
        pass

    def last_worker_timestamp(self) -> Int:
        return self.timestamp

    def wake(mut self):
        pass

    def sleep(mut self):
        pass


@fieldwise_init
struct CountKernel(OutputPartitionedKernel):
    var out: IPtr
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        self.out[self.worker_id] = self.end - self.start

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.worker_id = worker_id
        self.start = start
        self.end = end


def check(ok: Bool, msg: String):
    if not ok:
        abort("FAIL: " + msg)


def test_tile_dispatch_returns_actual_count():
    var out = alloc[Int](8)
    for i in range(8):
        out[i] = 0
    var pool = TestPool(8, 0)
    var buf = DispatchBuffer[CountKernel]()

    var launched = tile_dispatch(
        buf, CountKernel(out, 0, 0, 0), pool, total=3, num_workers=8)
    pool.join()

    check(launched == 3, "tile_dispatch should return clamped worker count")
    check(out[0] == 1 and out[1] == 1 and out[2] == 1,
        "tile_dispatch should execute exactly the launched partitions")
    check(out[3] == 0, "tile_dispatch should not report/write an extra worker")
    out.free()


def test_fanout_per_rank_returns_actual_counts():
    comptime tp = 2
    var out = alloc[Int](16)
    for i in range(16):
        out[i] = 0

    var pools = HeapMoveArray[TestPool](tp)
    pools.push(TestPool(8, 0))
    pools.push(TestPool(8, 0))

    @parameter
    def make(r: Int) -> CountKernel:
        return CountKernel(out + r * 8, 0, 0, 0)

    @parameter
    def total_for(r: Int) -> Int:
        if r == 0:
            return 2
        return 5

    @parameter
    def bytes_for(r: Int) -> Int:
        return 1024

    var launched = fanout_dispatch_per_rank[
        tp, make, total_for, bytes_for,
        worker_policy=saturate_workers,
    ](pools)

    check(launched[0] == 2, "rank 0 should report two actual workers")
    check(launched[1] == 5, "rank 1 should report five actual workers")
    check(out[2] == 0, "rank 0 should not write a third partial")
    check(out[8 + 4] == 1, "rank 1 fifth worker should run")
    check(out[8 + 5] == 0, "rank 1 should not write a sixth partial")
    out.free()


def main():
    test_tile_dispatch_returns_actual_count()
    test_fanout_per_rank_returns_actual_counts()
    print("dispatch helper tests passed")
