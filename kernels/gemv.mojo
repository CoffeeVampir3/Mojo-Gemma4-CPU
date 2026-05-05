from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from simd_math import pick_port_unroll, tree_reduce_accs
from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from .helpers import (
    Chain, RangedKernel, DispatchBuffer, tile_dispatch, recommended_workers,
)


comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime W = simd_width_of[DType.float32]()


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
            accs[p] = accs[p].fma(xv, wv)


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


@fieldwise_init
struct GemvKernel[rows: Int, cols: Int](RangedKernel):
    var x: BF16Ptr
    var weight: BF16Ptr
    var output: BF16Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        gemv_range[Self.rows, Self.cols](
            self.x, self.weight, self.output, self.start, self.end)

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.x, self.weight, self.output, start, end)


comptime GEMV_INLINE_ROWS = 4


def gemv[
    P: BurstThreadPool, //,
    rows: Int, cols: Int,
](
    x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr,
    mut pool: P,
):
    if rows <= GEMV_INLINE_ROWS:
        gemv_range[rows, cols](x, weight, output, 0, rows)
        return

    var data_bytes = rows * cols * 2
    var nw = recommended_workers(data_bytes, pool.get_capacity())
    var buf = DispatchBuffer[GemvKernel[rows, cols]]()
    tile_dispatch(buf,
        GemvKernel[rows, cols](x, weight, output, 0, 0),
        pool, rows, num_workers=nw)
    pool.join()


@fieldwise_init
struct ScaledGemvKernel[rows: Int, cols: Int, numer: Int, denom: Int](RangedKernel):
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

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.x, self.weight, self.output, start, end)


def gemv_chained_qkv[
    P: BurstThreadPool, //,
    q_rows: Int, kv_rows: Int, cols: Int,
](
    x: BF16Ptr,
    q_weight: BF16Ptr, k_weight: BF16Ptr, v_weight: BF16Ptr,
    q_out: BF16Ptr, k_out: BF16Ptr, v_out: BF16Ptr,
    mut pool: P,
):
    comptime total_rows = q_rows + kv_rows + kv_rows
    var nw = pool.get_capacity()

    comptime QKernel = ScaledGemvKernel[q_rows, cols, q_rows, total_rows]
    comptime KKernel = ScaledGemvKernel[kv_rows, cols, kv_rows, total_rows]
    comptime VKernel = ScaledGemvKernel[kv_rows, cols, kv_rows, total_rows]
    comptime QK = Chain[QKernel, KKernel]
    comptime QKV = Chain[QK, VKernel]

    var proto = QKV(
        QK(
            QKernel(x, q_weight, q_out, 0, 0),
            KKernel(x, k_weight, k_out, 0, 0),
        ),
        VKernel(x, v_weight, v_out, 0, 0),
    )

    var buf = DispatchBuffer[QKV]()
    tile_dispatch(buf, proto, pool, total_rows, num_workers=nw)
    pool.join()
