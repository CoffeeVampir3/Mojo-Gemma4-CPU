from std.collections import InlineArray

from simd_math import fast_exp_softmax_biased

from .dot_products import dot_to_scalar
from .helpers import BF16Ptr, F32Ptr, W, accumulate_scaled, scale_unrolled


comptime TILE = W


trait KVSlot:
    @staticmethod
    @always_inline
    def slot(start_pos: Int, pos_t: Int) -> Int: ...


struct LinearKV(KVSlot):
    @staticmethod
    @always_inline
    def slot(start_pos: Int, pos_t: Int) -> Int:
        return pos_t


struct RingKV[window: Int](KVSlot):
    @staticmethod
    @always_inline
    def slot(start_pos: Int, pos_t: Int) -> Int:
        return (start_pos + pos_t) & (Self.window - 1)


@always_inline
def flash_partial_stride[num_q: Int, head_dim: Int]() -> Int:
    return ((num_q * head_dim + 2 * num_q) * 4 + 63) // 64 * 16


@always_inline
def full_local_kv_count(rank: Int, abs_pos: Int, degree: Int) -> Int:
    if abs_pos < 0:
        return 0
    if rank <= abs_pos % degree:
        return abs_pos // degree + 1
    return abs_pos // degree


@always_inline
def online_softmax_tile[
    tile: Int,
](
    scores: SIMD[DType.float32, tile],
    old_m: Float32,
) -> Tuple[Float32, Float32, SIMD[DType.float32, tile]]:
    var tile_max = scores.reduce_max()
    var m_new = tile_max if tile_max > old_m else old_m
    var corr = fast_exp_softmax_biased[1](
        max(SIMD[DType.float32, 1](-87.0),
            SIMD[DType.float32, 1](old_m - m_new)))[0]
    var weights = fast_exp_softmax_biased[tile](
        max(SIMD[DType.float32, tile](-87.0),
            scores - SIMD[DType.float32, tile](m_new)))
    return (m_new, corr, weights)


@always_inline
def process_kv_tile[
    num_q: Int, //,
    KV: KVSlot, head_dim: Int, gqa_ratio: Int, kv_stride: Int,
](
    read q_ptrs: InlineArray[BF16Ptr, num_q],
    k_base: BF16Ptr, v_base: BF16Ptr,
    start_pos: Int, pos: Int, tile_len: Int,
    mut m: InlineArray[Float32, num_q],
    mut l: InlineArray[Float32, num_q],
    read acc_ptrs: InlineArray[F32Ptr, num_q],
):
    comptime for q_idx in range(num_q):
        comptime kv_h = q_idx // gqa_ratio

        var scores = SIMD[DType.float32, TILE](-1e30)
        for t in range(tile_len):
            var s_idx = KV.slot(start_pos, pos + t)
            var k_head = k_base + s_idx * kv_stride + kv_h * head_dim
            scores[t] = dot_to_scalar[head_dim](q_ptrs[q_idx], k_head)

        var sm = online_softmax_tile[TILE](scores, m[q_idx])
        var m_new = sm[0]
        var corr = sm[1]
        var weights = sm[2]

        scale_unrolled[cols=head_dim](acc_ptrs[q_idx], corr)
        l[q_idx] = l[q_idx] * corr + weights.reduce_add()
        m[q_idx] = m_new

        for t in range(tile_len):
            var s_idx = KV.slot(start_pos, pos + t)
            var v_head = v_base + s_idx * kv_stride + kv_h * head_dim
            accumulate_scaled[cols=head_dim](
                v_head, weights[t], acc_ptrs[q_idx])
