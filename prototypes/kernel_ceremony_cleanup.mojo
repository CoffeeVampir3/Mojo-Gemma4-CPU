from std.memory import Span, UnsafePointer, alloc
from std.os import abort

from kernels.helpers import (
    ArenaBases, Binding, DispatchBuffer, RankBuffers,
    join_all, recommended_workers, worker_range,
)
from notstdcollections import HeapMoveArray
from threading.threading_traits import BurstKernel, BurstThreadPool


comptime F32Ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
comptime F32ConstAny = ImmutOrigin(MutAnyOrigin)


trait RangeBody(TrivialRegisterPassable):
    """A kernel payload without transport ceremony.

    Production OutputPartitionedKernel structs currently repeat start/end
    fields, execute forwarding, and over_range rebuilding. This trait keeps
    only the operation's payload and range body; RangeTask owns transport.
    """

    @staticmethod
    def inline_units() -> Int:
        return 0

    @staticmethod
    def data_bytes(total_units: Int) -> Int: ...

    def execute_range(mut self, start: Int, end: Int): ...


@fieldwise_init
struct RangeTask[B: RangeBody](BurstKernel):
    var body: Self.B
    var start: Int
    var end: Int

    def execute(mut self):
        self.body.execute_range(self.start, self.end)


@fieldwise_init
struct RangeChain[A: RangeBody, B: RangeBody](RangeBody):
    var a: Self.A
    var b: Self.B

    @staticmethod
    def inline_units() -> Int:
        var ai = Self.A.inline_units()
        var bi = Self.B.inline_units()
        return ai if ai < bi else bi

    @staticmethod
    def data_bytes(total_units: Int) -> Int:
        return Self.A.data_bytes(total_units) + Self.B.data_bytes(total_units)

    def execute_range(mut self, start: Int, end: Int):
        self.a.execute_range(start, end)
        self.b.execute_range(start, end)


@fieldwise_init
struct ScaledRange[B: RangeBody, numer: Int, denom: Int](RangeBody):
    var body: Self.B

    @staticmethod
    def inline_units() -> Int:
        return 0

    @staticmethod
    def data_bytes(total_units: Int) -> Int:
        var n = total_units * Self.numer // Self.denom
        return Self.B.data_bytes(n)

    def execute_range(mut self, start: Int, end: Int):
        self.body.execute_range(
            start * Self.numer // Self.denom,
            end * Self.numer // Self.denom,
        )


def launch_range[
    K: RangeBody, P: BurstThreadPool, //,
    tp: Int, max_worker_count: Int = 128,
    *,
    build: def[](rank: Int) capturing [_] -> K,
](
    mut pools: HeapMoveArray[P],
    total_units: Int,
    requested_workers: Int = 0,
):
    if total_units <= 0:
        return

    if total_units <= K.inline_units():
        for r in range(tp):
            var body = build(r)
            body.execute_range(0, total_units)
        return

    var buf = DispatchBuffer[RangeTask[K], max_worker_count]()
    for r in range(tp):
        var capacity = min(max_worker_count, pools[r].get_capacity())
        var workers = requested_workers
        if workers <= 0:
            workers = recommended_workers(K.data_bytes(total_units), capacity)
        workers = min(workers, capacity)
        workers = min(workers, total_units)
        var body = build(r)
        for w in range(workers):
            var wr = worker_range(total_units, workers, w)
            buf.slot()[] = RangeTask[K](body, wr[0], wr[1])
        buf.dispatch(pools[r])
    join_all[tp](pools)


@fieldwise_init
struct KernelContext[tp: Int, max_worker_count: Int = 128](
    Copyable, ImplicitlyCopyable
):
    """Carries the repeated launch parameters that model callers keep spelling."""

    def fill_rows[
        P: BurstThreadPool, //,
        rows: Int, cols: Int,
    ](
        self,
        output: Binding[Scalar[DType.float32], Self.tp],
        value: Float32,
        mut pools: HeapMoveArray[P],
    ):
        @parameter
        def build(rank: Int) -> FillRows[rows, cols]:
            return FillRows[rows, cols](output[rank], value)

        launch_range[
            tp=Self.tp, max_worker_count=Self.max_worker_count,
            build=build,
        ](pools, rows)

    def qkv_fill[
        P: BurstThreadPool, //,
        q_rows: Int, kv_rows: Int, cols: Int,
    ](
        self,
        q_out: Binding[Scalar[DType.float32], Self.tp],
        k_out: Binding[Scalar[DType.float32], Self.tp],
        v_out: Binding[Scalar[DType.float32], Self.tp],
        mut pools: HeapMoveArray[P],
    ):
        comptime total = q_rows + kv_rows + kv_rows
        comptime QBody = FillRows[q_rows, cols]
        comptime KBody = FillRows[kv_rows, cols]
        comptime VBody = FillRows[kv_rows, cols]
        comptime Q = ScaledRange[QBody, q_rows, total]
        comptime K = ScaledRange[KBody, kv_rows, total]
        comptime V = ScaledRange[VBody, kv_rows, total]
        comptime QK = RangeChain[Q, K]
        comptime QKV = RangeChain[QK, V]

        @parameter
        def build(rank: Int) -> QKV:
            return QKV(
                QK(
                    Q(QBody(q_out[rank], Float32(10.0))),
                    K(KBody(k_out[rank], Float32(20.0))),
                ),
                V(VBody(v_out[rank], Float32(30.0))),
            )

        launch_range[
            tp=Self.tp, max_worker_count=Self.max_worker_count,
            build=build,
        ](pools, total)


@fieldwise_init
struct FillRows[rows: Int, cols: Int](RangeBody):
    var out: F32Ptr
    var value: Float32

    @staticmethod
    def inline_units() -> Int:
        return 2

    @staticmethod
    def data_bytes(total_units: Int) -> Int:
        return total_units * Self.cols * 4

    def execute_range(mut self, start: Int, end: Int):
        for row in range(start, end):
            for col in range(Self.cols):
                (self.out + row * Self.cols + col)[] = self.value


@fieldwise_init
struct AxpyRows[rows: Int, cols: Int](RangeBody):
    var x: F32Ptr
    var y: F32Ptr
    var out: F32Ptr
    var alpha: Float32

    @staticmethod
    def inline_units() -> Int:
        return 2

    @staticmethod
    def data_bytes(total_units: Int) -> Int:
        return total_units * Self.cols * 12

    def execute_range(mut self, start: Int, end: Int):
        for row in range(start, end):
            for col in range(Self.cols):
                var idx = row * Self.cols + col
                (self.out + idx)[] = (
                    (self.x + idx)[] + self.alpha * (self.y + idx)[]
                )


@fieldwise_init
struct TestPool(BurstThreadPool):
    var capacity: Int
    var timestamp: Int

    def get_capacity(self) -> Int:
        return self.capacity

    def dispatch[K: BurstKernel, origin: MutOrigin](
        mut self, kernels: Span[K, origin], num_jobs: Int,
    ):
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


def check(ok: Bool, msg: String):
    if not ok:
        abort("FAIL: " + msg)


def check_all(ptr: F32Ptr, count: Int, expected: Float32, label: String):
    for i in range(count):
        check(
            ptr[i] == expected,
            label + " [" + String(i) + "] expected="
                + String(expected) + " got=" + String(ptr[i]),
        )


def bind_f32_inplace_reduce_inputs[degree: Int](
    binding: Binding[Scalar[DType.float32], degree],
    count: Int,
    mut src: RankBuffers[DType.float32, degree, F32ConstAny],
    mut dst: RankBuffers[DType.float32, degree, MutAnyOrigin],
):
    """Prototype for the repeated model-side RankBuffers setup ceremony."""

    src.count = count
    dst.count = count
    for r in range(degree):
        src.ptrs[r] = binding[r].as_immutable()
        dst.ptrs[r] = binding[r]


def make_pools[degree: Int](capacity: Int) -> HeapMoveArray[TestPool]:
    var pools = HeapMoveArray[TestPool](degree)
    for _ in range(degree):
        pools.push(TestPool(capacity, 0))
    return pools^


def test_generic_launch():
    comptime DEGREE = 2
    comptime ROWS = 5
    comptime COLS = 4
    comptime BASES = ArenaBases[DEGREE].fill(0)

    var x = alloc[Scalar[DType.float32]](ROWS * COLS)
    var y = alloc[Scalar[DType.float32]](ROWS * COLS)
    var out = alloc[Scalar[DType.float32]](ROWS * COLS)
    for i in range(ROWS * COLS):
        x[i] = Float32(i)
        y[i] = Float32(2 * i)
        out[i] = Float32(-1.0)

    var xb = Binding[Scalar[DType.float32], DEGREE](x.as_any_origin(), BASES)
    var yb = Binding[Scalar[DType.float32], DEGREE](y.as_any_origin(), BASES)
    var ob = Binding[Scalar[DType.float32], DEGREE](out.as_any_origin(), BASES)
    var pools = make_pools[DEGREE](4)

    @parameter
    def build(rank: Int) -> AxpyRows[ROWS, COLS]:
        return AxpyRows[ROWS, COLS](xb[rank], yb[rank], ob[rank], Float32(0.5))

    launch_range[
        tp=DEGREE, max_worker_count=8,
        build=build,
    ](pools, ROWS)
    for i in range(ROWS * COLS):
        check(out[i] == Float32(2 * i), "axpy launch mismatch")

    x.free()
    y.free()
    out.free()


def test_callsite_facade():
    comptime DEGREE = 2
    comptime ROWS = 6
    comptime COLS = 3
    comptime BASES = ArenaBases[DEGREE].fill(0)

    var out = alloc[Scalar[DType.float32]](ROWS * COLS)
    for i in range(ROWS * COLS):
        out[i] = Float32(-1.0)

    var ops = KernelContext[DEGREE, 8]()
    var pools = make_pools[DEGREE](4)
    var ob = Binding[Scalar[DType.float32], DEGREE](out.as_any_origin(), BASES)

    ops.fill_rows[rows=ROWS, cols=COLS](ob, Float32(7.0), pools)
    check_all(ob[0], ROWS * COLS, Float32(7.0), "fill_rows")

    out.free()


def test_qkv_chain_facade():
    comptime DEGREE = 2
    comptime Q_ROWS = 6
    comptime KV_ROWS = 2
    comptime COLS = 4
    comptime BASES = ArenaBases[DEGREE].fill(0)

    var q = alloc[Scalar[DType.float32]](Q_ROWS * COLS)
    var k = alloc[Scalar[DType.float32]](KV_ROWS * COLS)
    var v = alloc[Scalar[DType.float32]](KV_ROWS * COLS)
    for i in range(Q_ROWS * COLS):
        q[i] = Float32(-1.0)
    for i in range(KV_ROWS * COLS):
        k[i] = Float32(-1.0)
        v[i] = Float32(-1.0)

    var ops = KernelContext[DEGREE, 8]()
    var pools = make_pools[DEGREE](4)
    var qb = Binding[Scalar[DType.float32], DEGREE](q.as_any_origin(), BASES)
    var kb = Binding[Scalar[DType.float32], DEGREE](k.as_any_origin(), BASES)
    var vb = Binding[Scalar[DType.float32], DEGREE](v.as_any_origin(), BASES)

    ops.qkv_fill[q_rows=Q_ROWS, kv_rows=KV_ROWS, cols=COLS](qb, kb, vb, pools)
    check_all(qb[0], Q_ROWS * COLS, Float32(10.0), "q chain")
    check_all(kb[0], KV_ROWS * COLS, Float32(20.0), "k chain")
    check_all(vb[0], KV_ROWS * COLS, Float32(30.0), "v chain")

    q.free()
    k.free()
    v.free()


def test_rankbuffer_helper():
    comptime DEGREE = 2
    comptime COUNT = 8
    comptime BASES = ArenaBases[DEGREE].fill(0)

    var data = alloc[Scalar[DType.float32]](COUNT)
    var binding = Binding[Scalar[DType.float32], DEGREE](
        data.as_any_origin(), BASES,
    )
    var src = RankBuffers[DType.float32, DEGREE, F32ConstAny](COUNT)
    var dst = RankBuffers[DType.float32, DEGREE, MutAnyOrigin](COUNT)

    bind_f32_inplace_reduce_inputs[DEGREE](binding, COUNT, src, dst)
    for r in range(DEGREE):
        check(src.ptrs[r] == binding[r].as_immutable(), "src buffer bind")
        check(dst.ptrs[r] == binding[r], "dst buffer bind")

    data.free()


def main():
    test_generic_launch()
    test_callsite_facade()
    test_qkv_chain_facade()
    test_rankbuffer_helper()
    print("kernel ceremony cleanup prototype passed")
