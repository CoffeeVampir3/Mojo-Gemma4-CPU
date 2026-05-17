from std.algorithm import vectorize
from std.memory import Span, UnsafePointer, alloc
from std.os import abort
from std.sys.info import simd_width_of

from kernels.helpers import (
    ArenaBases, DispatchBuffer, Binding, recommended_workers, worker_range,
)
from notstdcollections import HeapMoveArray
from threading.threading_traits import BurstKernel, BurstThreadPool


comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime W = simd_width_of[DType.float32]()
comptime TEST_BASES = ArenaBases[1].fill(0)


@fieldwise_init
struct WorkRange(TrivialRegisterPassable):
    var start: Int
    var end: Int

    @always_inline
    def count(self) -> Int:
        return self.end - self.start


trait RangePartitionedKernel(BurstKernel):
    def set_range(mut self, start: Int, end: Int): ...

    def with_range(self, start: Int, end: Int) -> Self:
        var k = self
        k.set_range(start, end)
        return k


trait RankedRangeJob:
    comptime Kernel: RangePartitionedKernel
    comptime INLINE_UNITS: Int

    def total(self, rank: Int) -> Int: ...
    def data_bytes(self, rank: Int) -> Int: ...
    def inline_rank(self, rank: Int): ...
    def make_kernel(self, rank: Int, worker_id: Int) -> Self.Kernel: ...


def dispatch_ranked_range[
    P: BurstThreadPool, J: RankedRangeJob, //,
    tp: Int, max_worker_count: Int = 128,
](job: J, mut pools: HeapMoveArray[P]):
    var buf = DispatchBuffer[J.Kernel, max_worker_count]()
    var dispatched = False

    for r in range(tp):
        var total = job.total(r)
        if total <= 0:
            continue
        if total <= J.INLINE_UNITS:
            job.inline_rank(r)
            continue

        var cap = min(max_worker_count, pools[r].get_capacity())
        var nw = recommended_workers(job.data_bytes(r), cap)
        nw = min(nw, total)
        for w in range(nw):
            var wr = worker_range(total, nw, w)
            buf.slot()[] = job.make_kernel(r, w).with_range(wr[0], wr[1])
        buf.dispatch(pools[r])
        dispatched = True

    if dispatched:
        for r in range(tp):
            pools[r].join()


@always_inline
def scalar_mul_row_proto[hidden: Int](
    src: BF16Ptr, dst: BF16Ptr, scalar: Scalar[DType.float32],
):
    def step[width: Int](idx: Int) {read}:
        var x = (src + idx).load[width=width]().cast[DType.float32]()
        var factor = SIMD[DType.float32, width](scalar)
        (dst + idx).store((x * factor).cast[DType.bfloat16]())

    vectorize[W](hidden, step)


@fieldwise_init
struct ScalarMulTokenKernelProto[hidden: Int](RangePartitionedKernel):
    var src: BF16Ptr
    var dst: BF16Ptr
    var scalar: Scalar[DType.float32]
    var work: WorkRange

    def execute(mut self):
        for tok in range(self.work.start, self.work.end):
            var off = tok * Self.hidden
            scalar_mul_row_proto[Self.hidden](
                self.src + off, self.dst + off, self.scalar)

    def set_range(mut self, start: Int, end: Int):
        self.work = WorkRange(start, end)


@fieldwise_init
struct ScalarMulJob[hidden: Int, tp: Int](RankedRangeJob):
    comptime Kernel = ScalarMulTokenKernelProto[Self.hidden]
    comptime INLINE_UNITS = 16

    var src: Binding[Scalar[DType.bfloat16], Self.tp]
    var dst: Binding[Scalar[DType.bfloat16], Self.tp]
    var scalar: Scalar[DType.float32]
    var seq_len: Int

    def total(self, rank: Int) -> Int:
        return self.seq_len

    def data_bytes(self, rank: Int) -> Int:
        return self.seq_len * Self.hidden * 4

    def inline_rank(self, rank: Int):
        for tok in range(self.seq_len):
            var off = tok * Self.hidden
            scalar_mul_row_proto[Self.hidden](
                self.src[rank] + off,
                self.dst[rank] + off,
                self.scalar,
            )

    def make_kernel(self, rank: Int, worker_id: Int) -> Self.Kernel:
        return Self.Kernel(
            self.src[rank],
            self.dst[rank],
            self.scalar,
            WorkRange(0, 0),
        )


def dispatch_scalar_mul_proto[
    P: BurstThreadPool, //,
    hidden: Int, tp: Int, max_worker_count: Int = 128,
](
    src: Binding[Scalar[DType.bfloat16], tp],
    dst: Binding[Scalar[DType.bfloat16], tp],
    scalar: Scalar[DType.float32],
    seq_len: Int,
    mut pools: HeapMoveArray[P],
):
    var job = ScalarMulJob[hidden, tp](src, dst, scalar, seq_len)
    dispatch_ranked_range[tp=tp, max_worker_count=max_worker_count](job, pools)


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


def check(ok: Bool, message: String):
    if not ok:
        abort("FAIL: " + message)


def test_scalar_mul_proto():
    comptime hidden = W
    comptime seq_len = 20
    var src = alloc[Scalar[DType.bfloat16]](hidden * seq_len)
    var dst = alloc[Scalar[DType.bfloat16]](hidden * seq_len)

    for i in range(hidden * seq_len):
        src[i] = Scalar[DType.bfloat16](Float32(i % 8 + 1))
        dst[i] = Scalar[DType.bfloat16](Float32(0))

    var pools = HeapMoveArray[TestPool](1)
    pools.push(TestPool(4, 0))

    dispatch_scalar_mul_proto[hidden=hidden, tp=1](
        Binding[Scalar[DType.bfloat16], 1](src.as_any_origin(), TEST_BASES),
        Binding[Scalar[DType.bfloat16], 1](dst.as_any_origin(), TEST_BASES),
        Float32(2.0),
        seq_len,
        pools,
    )

    check(pools[0].timestamp == 1, "parallel dispatch path was not used")
    for i in range(hidden * seq_len):
        var expected = Float32(src[i]) * Float32(2.0)
        check(Float32(dst[i]) == expected, "bad scalar mul at " + String(i))

    src.free()
    dst.free()


def main():
    test_scalar_mul_proto()
    print("kernel dispatch cleanup prototype")
