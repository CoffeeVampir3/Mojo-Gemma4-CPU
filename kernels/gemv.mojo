from std.collections import InlineArray

from simd_math import pick_port_unroll, tree_reduce_accs
from simd_math.ops import tanh_f32
from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from .helpers import (
    RangePartitionedKernel,
    fanout_dispatch,
    Binding, BF16Ptr, W, dot_into_accs,
)
from .dispatch_heuristics import GEMV_INLINE_ROWS


@always_inline
def softcap_value[
    cap: Float64,
](
    x: SIMD[DType.float32, 1],
) -> SIMD[DType.float32, 1]:
    comptime assert cap > 0.0, "softcap cap must be positive"
    comptime c = SIMD[DType.float32, 1](cap)
    return tanh_f32[1](x / c) * c


@always_inline
def gemv_softcap_range[
    rows: Int, cols: Int, cap: Float64,
](
    x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr,
    start: Int, end: Int,
):
    comptime PU = pick_port_unroll[W, cols]()
    var accs = InlineArray[SIMD[DType.float32, W], PU](uninitialized=True)
    for row in range(start, end):
        comptime for p in range(PU):
            accs[p] = SIMD[DType.float32, W](0)
        dot_into_accs[cols=cols](x, weight + row * cols, accs)
        var capped = softcap_value[cap](tree_reduce_accs(accs))
        (output + row)[] = capped.cast[DType.bfloat16]()


@fieldwise_init
struct GemvSoftcapKernel[
    rows: Int, cols: Int, cap: Float64,
](RangePartitionedKernel):
    var x: BF16Ptr
    var weight: BF16Ptr
    var output: BF16Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        gemv_softcap_range[Self.rows, Self.cols, Self.cap](
            self.x, self.weight, self.output, self.start, self.end)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_gemv_softcap[
    P: BurstThreadPool, //,
    rows: Int, cols: Int, tp: Int, cap: Float64,
    max_worker_count: Int = 128,
](
    x: Binding[BFloat16, tp],
    weight: Binding[BFloat16, tp],
    output: Binding[BFloat16, tp],
    mut pools: HeapMoveArray[P],
):
    comptime K = GemvSoftcapKernel[rows, cols, cap]

    @parameter
    def make(r: Int) -> K:
        return K(x[r], weight[r], output[r], 0, 0)

    fanout_dispatch[tp, make, max_worker_count=max_worker_count](
        pools, rows, rows * cols * 2,
        inline_threshold_bytes=GEMV_INLINE_ROWS * cols * 2)
