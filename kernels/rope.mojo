from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from simd_math import sincos_simd
from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from .helpers import (
    RangedKernel, DispatchBuffer, tile_dispatch, recommended_workers,
)


comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
comptime W = simd_width_of[DType.float32]()

comptime ROPE_INLINE_TOKENS = 16


@always_inline
def rotate_pair[width: Int, pair_stride: Int](
    ptr: BF16Ptr,
    cos_ptr: F32Ptr,
    sin_ptr: F32Ptr,
    j: Int,
):
    var x_lo = (ptr + j).load[width=width]().cast[DType.float32]()
    var x_hi = (ptr + pair_stride + j).load[width=width]().cast[DType.float32]()
    var cv = (cos_ptr + j).load[width=width]()
    var sv = (sin_ptr + j).load[width=width]()
    (ptr + j).store((x_lo * cv - x_hi * sv).cast[DType.bfloat16]())
    (ptr + pair_stride + j).store((x_hi * cv + x_lo * sv).cast[DType.bfloat16]())


@always_inline
def rope_head[half: Int, pair_stride: Int](
    head_ptr: BF16Ptr, cos_row: F32Ptr, sin_row: F32Ptr,
):
    for j in range(0, half, W):
        rotate_pair[W, pair_stride](head_ptr, cos_row, sin_row, j)


@always_inline
def rope_token[half: Int, pair_stride: Int, num_heads: Int, head_dim: Int](
    data: BF16Ptr, cos_row: F32Ptr, sin_row: F32Ptr,
):
    for h in range(num_heads):
        rope_head[half, pair_stride](data + h * head_dim, cos_row, sin_row)


# === Dispatched kernel (token-parallel) ===


@fieldwise_init
struct RopeTokenKernel[
    half: Int, pair_stride: Int, num_heads: Int, head_dim: Int,
](RangedKernel):
    var data: BF16Ptr
    var cos_table: F32Ptr
    var sin_table: F32Ptr
    var base_pos: Int
    var row_stride: Int
    var start: Int
    var end: Int

    def execute(mut self):
        for tok in range(self.start, self.end):
            var cos_row = self.cos_table + (self.base_pos + tok) * Self.half
            var sin_row = self.sin_table + (self.base_pos + tok) * Self.half
            rope_token[Self.half, Self.pair_stride, Self.num_heads, Self.head_dim](
                self.data + tok * self.row_stride, cos_row, sin_row)

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.data, self.cos_table, self.sin_table,
                    self.base_pos, self.row_stride, start, end)


def rope[
    P: BurstThreadPool, //,
    half: Int, pair_stride: Int, num_heads: Int, head_dim: Int,
](
    data: BF16Ptr, cos_table: F32Ptr, sin_table: F32Ptr,
    base_pos: Int, seq_len: Int, mut pool: P,
):
    comptime row_stride = num_heads * head_dim

    if seq_len <= ROPE_INLINE_TOKENS:
        for tok in range(seq_len):
            var cos_row = cos_table + (base_pos + tok) * half
            var sin_row = sin_table + (base_pos + tok) * half
            rope_token[half, pair_stride, num_heads, head_dim](
                data + tok * row_stride, cos_row, sin_row)
        return

    var data_bytes = seq_len * row_stride * 2
    var nw = recommended_workers(data_bytes, pool.get_capacity())
    var buf = DispatchBuffer[RopeTokenKernel[half, pair_stride, num_heads, head_dim]]()
    tile_dispatch(buf,
        RopeTokenKernel[half, pair_stride, num_heads, head_dim](
            data, cos_table, sin_table, base_pos, row_stride, 0, 0),
        pool, seq_len, num_workers=nw)
    pool.join()


# === Static rotation table ===


@fieldwise_init
struct RopeTable[half: Int](Copyable, ImplicitlyCopyable):
    var cos: F32Ptr
    var sin: F32Ptr
    var capacity: Int

    @always_inline
    def cos_row(self, pos: Int) -> F32Ptr:
        return self.cos + pos * Self.half

    @always_inline
    def sin_row(self, pos: Int) -> F32Ptr:
        return self.sin + pos * Self.half


# === Table initialization ===


def init_rope_table[half: Int, max_pos: Int](
    cos_buf: F32Ptr, sin_buf: F32Ptr,
    theta: Float64,
):
    comptime f64w = simd_width_of[DType.float64]()

    for j in range(0, half, f64w):
        var inv = SIMD[DType.float64, f64w]()
        for k in range(f64w):
            var dim_idx = j + k
            inv[k] = 1.0 / (theta ** (Float64(2 * dim_idx) / Float64(half * 2)))

        for pos in range(max_pos):
            var sc = sincos_simd[polynomial_degree=8, width=f64w](SIMD[DType.float64, f64w](Float64(pos)) * inv)
            (cos_buf + pos * half + j).store(sc.cos_val.cast[DType.float32]())
            (sin_buf + pos * half + j).store(sc.sin_val.cast[DType.float32]())


def init_rope_table_partial[rotary_half: Int, max_pos: Int](
    cos_buf: F32Ptr, sin_buf: F32Ptr,
    theta: Float64, full_head_dim: Int,
):
    comptime f64w = simd_width_of[DType.float64]()

    for j in range(0, rotary_half, f64w):
        var inv = SIMD[DType.float64, f64w]()
        for k in range(f64w):
            var dim_idx = j + k
            inv[k] = 1.0 / (theta ** (Float64(2 * dim_idx) / Float64(full_head_dim)))

        for pos in range(max_pos):
            var sc = sincos_simd[polynomial_degree=8, width=f64w](SIMD[DType.float64, f64w](Float64(pos)) * inv)
            (cos_buf + pos * rotary_half + j).store(sc.cos_val.cast[DType.float32]())
            (sin_buf + pos * rotary_half + j).store(sc.sin_val.cast[DType.float32]())
