from std.collections import InlineArray
from std.memory import Span, UnsafePointer

from threading.threading_traits import BurstKernel, BurstThreadPool


@fieldwise_init
struct TestPool(BurstThreadPool):
    var capacity: Int
    var dispatch_count: Int

    def get_capacity(self) -> Int:
        return self.capacity

    def dispatch[K: BurstKernel, origin: MutOrigin](
        mut self, kernels: Span[K, origin], num_jobs: Int
    ):
        for i in range(num_jobs):
            var kernel = kernels[i]
            kernel.execute()
        self.dispatch_count += 1

    def join(mut self):
        pass

    def last_worker_timestamp(self) -> Int:
        return self.dispatch_count

    def wake(mut self):
        pass

    def sleep(mut self):
        pass


struct TinyJobSet[K: BurstKernel]:
    var items: InlineArray[Self.K, 16]
    var count: Int

    def __init__(out self):
        self.items = InlineArray[Self.K, 16](uninitialized=True)
        self.count = 0

    def add(mut self, job: Self.K):
        self.items[self.count] = job
        self.count += 1

    def dispatch(mut self, mut pool: TestPool):
        if self.count > 0:
            pool.dispatch(
                Span(ptr=UnsafePointer(to=self.items[0]), length=self.count),
                self.count,
            )
        self.count = 0


@fieldwise_init
struct TokenWriteJob[
    src_origin: ImmutOrigin, dst_origin: ImmutOrigin,
](BurstKernel):
    var src0: UnsafePointer[Float32, Self.src_origin]
    var src1: UnsafePointer[Float32, Self.src_origin]
    var dst_keepalive: UnsafePointer[Float32, Self.dst_origin]
    var dst_addr: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var dst = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.dst_addr
        )
        for i in range(self.start, self.end):
            dst[i] = self.src0[i] + self.src1[i]
        _ = self.dst_keepalive


def run_token_write[
    src_origin: ImmutOrigin, dst_origin: MutOrigin, //, n: Int
](
    src0: UnsafePointer[Float32, src_origin],
    src1: UnsafePointer[Float32, src_origin],
    dst: UnsafePointer[Float32, dst_origin],
    mut pool: TestPool,
):
    var jobs = TinyJobSet[
        TokenWriteJob[src_origin, ImmutOrigin(dst_origin)]
    ]()
    var workers = pool.get_capacity()
    var per_worker = (n + workers - 1) // workers
    for w in range(workers):
        var start = w * per_worker
        var end = min(start + per_worker, n)
        jobs.add(
            TokenWriteJob[src_origin, ImmutOrigin(dst_origin)](
                src0,
                src1,
                dst.as_immutable(),
                Int(dst),
                start,
                end,
            )
        )
    jobs.dispatch(pool)
    pool.join()

    _ = dst[]


@fieldwise_init
struct TokenConfig[
    src_origin: ImmutOrigin, dst_origin: ImmutOrigin,
]:
    var src0: UnsafePointer[Float32, Self.src_origin]
    var src1: UnsafePointer[Float32, Self.src_origin]
    var dst_keepalive: UnsafePointer[Float32, Self.dst_origin]
    var dst_addr: Int
    var n: Int


@fieldwise_init
struct TokenConfigJob[
    src_origin: ImmutOrigin, dst_origin: ImmutOrigin,
    cfg_origin: ImmutOrigin,
](BurstKernel):
    var config: UnsafePointer[
        TokenConfig[Self.src_origin, Self.dst_origin], Self.cfg_origin
    ]
    var start: Int
    var end: Int

    def execute(mut self):
        var cfg = self.config
        var dst = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=cfg[].dst_addr
        )
        for i in range(self.start, self.end):
            dst[i] = cfg[].src0[i] + cfg[].src1[i]
        _ = cfg[].dst_keepalive


def run_token_config[
    src_origin: ImmutOrigin, dst_origin: MutOrigin, //, n: Int
](
    src0: UnsafePointer[Float32, src_origin],
    src1: UnsafePointer[Float32, src_origin],
    dst: UnsafePointer[Float32, dst_origin],
    mut pool: TestPool,
):
    var cfg = TokenConfig[src_origin, ImmutOrigin(dst_origin)](
        src0,
        src1,
        dst.as_immutable(),
        Int(dst),
        n,
    )
    var config = UnsafePointer(to=cfg).as_immutable()
    var jobs = TinyJobSet[
        TokenConfigJob[
            src_origin, ImmutOrigin(dst_origin),
            ImmutOrigin(origin_of(cfg)),
        ]
    ]()
    var workers = pool.get_capacity()
    var per_worker = (n + workers - 1) // workers
    for w in range(workers):
        var start = w * per_worker
        var end = min(start + per_worker, n)
        jobs.add(
            TokenConfigJob[
                src_origin, ImmutOrigin(dst_origin),
                ImmutOrigin(origin_of(cfg)),
            ](config, start, end)
        )
    jobs.dispatch(pool)
    pool.join()

    _ = cfg.n


def check_stack_token_write():
    comptime N = 8
    var src_storage = InlineArray[Float32, N * 2](uninitialized=True)
    var dst_storage = InlineArray[Float32, N](uninitialized=True)
    for i in range(N):
        src_storage[i] = Float32(i)
        src_storage[N + i] = Float32(10 + i)
        dst_storage[i] = Float32(-1)

    var pool = TestPool(2, 0)
    run_token_write[N](
        UnsafePointer(to=src_storage[0]).as_immutable(),
        UnsafePointer(to=src_storage[N]).as_immutable(),
        UnsafePointer(to=dst_storage[0]),
        pool,
    )

    for i in range(N):
        debug_assert(
            dst_storage[i] == Float32(10 + 2 * i),
            "stack token write mismatch",
        )


def check_stack_token_config():
    comptime N = 8
    var src_storage = InlineArray[Float32, N * 2](uninitialized=True)
    var dst_storage = InlineArray[Float32, N](uninitialized=True)
    for i in range(N):
        src_storage[i] = Float32(i)
        src_storage[N + i] = Float32(10 + i)
        dst_storage[i] = Float32(-1)

    var pool = TestPool(2, 0)
    run_token_config[N](
        UnsafePointer(to=src_storage[0]).as_immutable(),
        UnsafePointer(to=src_storage[N]).as_immutable(),
        UnsafePointer(to=dst_storage[0]),
        pool,
    )

    for i in range(N):
        debug_assert(
            dst_storage[i] == Float32(10 + 2 * i),
            "stack token config mismatch",
        )


def main():
    check_stack_token_write()
    check_stack_token_config()
    print("origin lifetime token prototype ok")
