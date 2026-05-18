from std.memory import memcpy
from std.sys.info import simd_width_of

from simd_math import sincos_simd
from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from .helpers import (
    OutputPartitionedKernel, Binding,
    fanout_dispatch,
    BF16Ptr, F32Ptr, W,
)
from .dispatch_heuristics import ROPE_INLINE_TOKENS


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
def rotate_pair_to[width: Int, pair_stride: Int](
    src: BF16Ptr, dst: BF16Ptr,
    cos_ptr: F32Ptr, sin_ptr: F32Ptr, j: Int,
):
    var x_lo = (src + j).load[width=width]().cast[DType.float32]()
    var x_hi = (src + pair_stride + j).load[width=width]().cast[DType.float32]()
    var cv = (cos_ptr + j).load[width=width]()
    var sv = (sin_ptr + j).load[width=width]()
    (dst + j).store((x_lo * cv - x_hi * sv).cast[DType.bfloat16]())
    (dst + pair_stride + j).store((x_hi * cv + x_lo * sv).cast[DType.bfloat16]())


@always_inline
def rope_head_to[half: Int, pair_stride: Int, head_dim: Int](
    src: BF16Ptr, dst: BF16Ptr, cos_row: F32Ptr, sin_row: F32Ptr,
):
    for j in range(0, half, W):
        rotate_pair_to[W, pair_stride](src, dst, cos_row, sin_row, j)
    comptime
    if half < pair_stride:
        memcpy(dest=dst + half, src=src + half, count=pair_stride - half)
        memcpy(dest=dst + pair_stride + half, src=src + pair_stride + half,
               count=head_dim - pair_stride - half)


@fieldwise_init
struct RopeCacheWriteKernel[
    half: Int,
    pair_stride: Int,
    num_q: Int,
    num_kv: Int,
    head_dim: Int,
    kv_cache_stride: Int,
    slot_mask: Int,
    cache_degree: Int,
](OutputPartitionedKernel):
    var q: BF16Ptr
    var k_src: BF16Ptr
    var v_src: BF16Ptr
    var k_cache: BF16Ptr
    var v_cache: BF16Ptr
    var cos_table: F32Ptr
    var sin_table: F32Ptr
    var base_pos: Int
    var rank: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime q_stride = Self.num_q * Self.head_dim
        comptime kv_stride = Self.num_kv * Self.head_dim
        for tok in range(self.start, self.end):
            var pos = self.base_pos + tok
            var table_idx = pos // Self.cache_degree
            var cos_row = self.cos_table + table_idx * Self.half
            var sin_row = self.sin_table + table_idx * Self.half

            var q_tok = self.q + tok * q_stride
            for h in range(Self.num_q):
                rope_head[Self.half, Self.pair_stride](
                    q_tok + h * Self.head_dim, cos_row, sin_row)

            if pos % Self.cache_degree == self.rank:
                var slot = table_idx & Self.slot_mask
                var k_tok = self.k_src + tok * kv_stride
                var k_dst = self.k_cache + slot * Self.kv_cache_stride
                for h in range(Self.num_kv):
                    rope_head_to[Self.half, Self.pair_stride, Self.head_dim](
                        k_tok + h * Self.head_dim,
                        k_dst + h * Self.head_dim,
                        cos_row, sin_row)

                var v_tok = self.v_src + tok * kv_stride
                var v_dst = self.v_cache + slot * Self.kv_cache_stride
                memcpy(dest=v_dst, src=v_tok, count=kv_stride)

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_rope_cache_write[
    P: BurstThreadPool, //,
    half: Int, pair_stride: Int,
    num_q: Int, num_kv: Int, head_dim: Int,
    kv_cache_stride: Int, slot_mask: Int,
    cache_degree: Int, tp: Int, max_worker_count: Int = 128,
](
    q: Binding[BFloat16, tp],
    k_src: Binding[BFloat16, tp],
    v_src: Binding[BFloat16, tp],
    k_cache: Binding[BFloat16, tp],
    v_cache: Binding[BFloat16, tp],
    cos_table: Binding[Float32, tp],
    sin_table: Binding[Float32, tp],
    base_pos: Int, seq_len: Int,
    mut pools: HeapMoveArray[P],
):
    comptime K = RopeCacheWriteKernel[
        half, pair_stride, num_q, num_kv, head_dim,
        kv_cache_stride, slot_mask, cache_degree]
    comptime row_bytes = (num_q + 2 * num_kv) * head_dim * 2

    @parameter
    def make(r: Int) -> K:
        return K(q[r], k_src[r], v_src[r],
                 k_cache[r], v_cache[r],
                 cos_table[r], sin_table[r],
                 base_pos, r % cache_degree, 0, 0)

    fanout_dispatch[tp, make, max_worker_count=max_worker_count](
        pools, seq_len, seq_len * row_bytes,
        inline_threshold_bytes=ROPE_INLINE_TOKENS * row_bytes)

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


def init_rope_table_partial_strided[rotary_half: Int, rows: Int](
    cos_buf: F32Ptr, sin_buf: F32Ptr,
    theta: Float64, full_head_dim: Int,
    first_pos: Int, stride: Int,
):
    comptime f64w = simd_width_of[DType.float64]()

    for j in range(0, rotary_half, f64w):
        var inv = SIMD[DType.float64, f64w]()
        for k in range(f64w):
            var dim_idx = j + k
            inv[k] = 1.0 / (theta ** (Float64(2 * dim_idx) / Float64(full_head_dim)))

        for row in range(rows):
            var pos = first_pos + row * stride
            var sc = sincos_simd[polynomial_degree=8, width=f64w](
                SIMD[DType.float64, f64w](Float64(pos)) * inv)
            (cos_buf + row * rotary_half + j).store(sc.cos_val.cast[DType.float32]())
            (sin_buf + row * rotary_half + j).store(sc.sin_val.cast[DType.float32]())
