from std.collections import InlineArray

from simd_math import pick_port_unroll, tree_reduce_accs
from simd_math.ops import tanh_f32
from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from .helpers import (
    Chain, OutputPartitionedKernel,
    fanout_dispatch, saturate_workers,
    Binding, BF16Ptr, W,
)
from .dispatch_heuristics import GEMV_INLINE_ROWS


@always_inline
def dot_row[cols: Int, PU: Int](
    x: BF16Ptr, weight_row: BF16Ptr,
    mut accs: InlineArray[SIMD[DType.float32, W], PU],
):
    comptime STRIDE = PU * W
    for i in range(cols // STRIDE):
        comptime for p in range(PU):
            var xv = (x + i * STRIDE + p * W).load[width=W]().cast[DType.float32]()
            var wv = (weight_row + i * STRIDE + p * W).load[width=W]().cast[DType.float32]()
            accs[p] = xv.fma(wv, accs[p])


@always_inline
def gemv_range[rows: Int, cols: Int](
    x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr,
    start: Int, end: Int,
):
    comptime PU = pick_port_unroll[W, cols]()
    var accs = InlineArray[SIMD[DType.float32, W], PU](uninitialized=True)
    for row in range(start, end):
        comptime for p in range(PU):
            accs[p] = SIMD[DType.float32, W](0)
        dot_row[cols, PU](x, weight + row * cols, accs)
        (output + row)[] = tree_reduce_accs(accs).cast[DType.bfloat16]()


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
        dot_row[cols, PU](x, weight + row * cols, accs)
        var capped = softcap_value[cap](tree_reduce_accs(accs))
        (output + row)[] = capped.cast[DType.bfloat16]()


@fieldwise_init
struct GemvKernel[rows: Int, cols: Int](OutputPartitionedKernel):
    var x: BF16Ptr
    var weight: BF16Ptr
    var output: BF16Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        gemv_range[Self.rows, Self.cols](
            self.x, self.weight, self.output, self.start, self.end)

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_gemv[
    P: BurstThreadPool, //,
    rows: Int, cols: Int, tp: Int, max_worker_count: Int = 128,
](
    x: Binding[BFloat16, tp],
    weight: Binding[BFloat16, tp],
    output: Binding[BFloat16, tp],
    mut pools: HeapMoveArray[P],
):
    comptime K = GemvKernel[rows, cols]

    @parameter
    def make(r: Int) -> K:
        return K(x[r], weight[r], output[r], 0, 0)

    fanout_dispatch[tp, make, max_worker_count=max_worker_count](
        pools, rows, rows * cols * 2,
        inline_threshold_bytes=GEMV_INLINE_ROWS * cols * 2)


@fieldwise_init
struct GemvSoftcapKernel[
    rows: Int, cols: Int, cap: Float64,
](OutputPartitionedKernel):
    var x: BF16Ptr
    var weight: BF16Ptr
    var output: BF16Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        gemv_softcap_range[Self.rows, Self.cols, Self.cap](
            self.x, self.weight, self.output, self.start, self.end)

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
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


@fieldwise_init
struct ScaledGemvKernel[rows: Int, cols: Int, numer: Int, denom: Int](OutputPartitionedKernel):
    var x: BF16Ptr
    var weight: BF16Ptr
    var output: BF16Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        var my_start = self.start * Self.numer // Self.denom
        var my_end = self.end * Self.numer // Self.denom
        gemv_range[Self.rows, Self.cols](
            self.x, self.weight, self.output, my_start, my_end)

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_gemv_chained_qkv[
    P: BurstThreadPool, //,
    q_rows: Int, kv_rows: Int, cols: Int, tp: Int,
    max_worker_count: Int = 128,
](
    x: Binding[BFloat16, tp],
    q_weight: Binding[BFloat16, tp],
    k_weight: Binding[BFloat16, tp],
    v_weight: Binding[BFloat16, tp],
    q_out: Binding[BFloat16, tp],
    k_out: Binding[BFloat16, tp],
    v_out: Binding[BFloat16, tp],
    mut pools: HeapMoveArray[P],
):
    comptime total_rows = q_rows + kv_rows + kv_rows
    comptime QKernel = ScaledGemvKernel[q_rows, cols, q_rows, total_rows]
    comptime KKernel = ScaledGemvKernel[kv_rows, cols, kv_rows, total_rows]
    comptime VKernel = ScaledGemvKernel[kv_rows, cols, kv_rows, total_rows]
    comptime QK = Chain[QKernel, KKernel]
    comptime QKV = Chain[QK, VKernel]

    @parameter
    def make(r: Int) -> QKV:
        return QKV(
            QK(
                QKernel(x[r], q_weight[r], q_out[r], 0, 0),
                KKernel(x[r], k_weight[r], k_out[r], 0, 0),
            ),
            VKernel(x[r], v_weight[r], v_out[r], 0, 0),
        )

    fanout_dispatch[
        tp, make,
        max_worker_count=max_worker_count,
        worker_policy=saturate_workers,
    ](pools, total_rows, total_rows * cols * 2)
