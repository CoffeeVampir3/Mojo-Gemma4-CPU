from std.sys.info import simd_width_of, size_of

from threading.threading_traits import BurstThreadPool
from .helpers import RangePartitionedKernel, Binding, fanout_dispatch
from .profiling import Profiler


@fieldwise_init
struct CopyKernel[T: DType](RangePartitionedKernel):
    var src: UnsafePointer[Scalar[Self.T], MutAnyOrigin]
    var dst: UnsafePointer[Scalar[Self.T], MutAnyOrigin]
    var start: Int
    var end: Int

    def execute(mut self):
        comptime width = simd_width_of[Self.T]()
        var i = self.start
        while i + width <= self.end:
            (self.dst + i).store((self.src + i).load[width=width]())
            i += width
        while i < self.end:
            (self.dst + i).store((self.src + i).load())
            i += 1

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_copy[
    T: DType, P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_worker_count: Int = 128,
](
    src: Binding[Scalar[T], o],
    dst: Binding[Scalar[T], o],
    count: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime K = CopyKernel[T]

    @parameter
    def make(r: Int) -> K:
        return K(src[r], dst[r], 0, 0)

    fanout_dispatch[make, max_worker_count=max_worker_count, label="copy"](
        pools, prof, count, count * size_of[Scalar[T]]() * 2)
