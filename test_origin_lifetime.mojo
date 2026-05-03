from std.collections import InlineArray
from std.memory import Span, UnsafePointer, alloc

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


@fieldwise_init
struct Config:
    var src: UnsafePointer[Float32, MutAnyOrigin]
    var dst: UnsafePointer[Float32, MutAnyOrigin]
    var n: Int


@fieldwise_init
struct Job[cfg_origin: Origin](BurstKernel):
    var config: UnsafePointer[Config, Self.cfg_origin]
    var start: Int
    var end: Int

    def execute(mut self):
        var cfg = self.config
        for i in range(self.start, self.end):
            cfg[].dst[i] = cfg[].src[i] * 2.0


def run_with_exact_origin():
    var src = alloc[Float32](4)
    var dst = alloc[Float32](4)
    for i in range(4):
        src[i] = Float32(i)
        dst[i] = Float32(-1)

    var cfg = Config(src, dst, 4)
    var config = UnsafePointer(to=cfg)

    var jobs = InlineArray[Job[origin_of(cfg)], 2](uninitialized=True)
    jobs[0] = Job[origin_of(cfg)](config, 0, 2)
    jobs[1] = Job[origin_of(cfg)](config, 2, 4)

    var pool = TestPool(2, 0)
    pool.dispatch(Span(ptr=UnsafePointer(to=jobs[0]), length=2), 2)
    pool.join()

    for i in range(4):
        debug_assert(dst[i] == Float32(i * 2), "exact origin mismatch")

    print("exact origin ok")


def main():
    run_with_exact_origin()
