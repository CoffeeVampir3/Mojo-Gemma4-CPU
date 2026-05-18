from std.collections import InlineArray

from simd_math import pick_port_unroll, tree_reduce_accs
from .helpers import BF16Ptr, W, dot_into_accs


@always_inline
def flash_partial_stride[num_q: Int, head_dim: Int]() -> Int:
    return ((num_q * head_dim + 2 * num_q) * 4 + 63) // 64 * 16


@always_inline
def score_position[head_dim: Int](q: BF16Ptr, k_row: BF16Ptr) -> Float32:
    comptime PU = pick_port_unroll[W, head_dim]()
    var accs = InlineArray[SIMD[DType.float32, W], PU](
        fill=SIMD[DType.float32, W](0))
    dot_into_accs[cols=head_dim](q, k_row, accs)
    return tree_reduce_accs(accs)
