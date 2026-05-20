from simd_math import fast_exp_softmax_biased

from .helpers import W


comptime TILE = W


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
    """One online softmax update for a flash-attention score tile.

    Caller fills inactive lanes with a sufficiently negative sentinel.
    Returns `(m_new, correction, weights)` where `correction` rescales the
    previous accumulator and `weights` applies to the current tile's V rows.
    """
    var tile_max = scores.reduce_max()
    var m_new = tile_max if tile_max > old_m else old_m
    var corr = fast_exp_softmax_biased[1](
        SIMD[DType.float32, 1](old_m - m_new))[0]
    var weights = fast_exp_softmax_biased[tile](
        scores - SIMD[DType.float32, tile](m_new))
    return (m_new, corr, weights)
