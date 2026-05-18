from std.collections import InlineArray

from simd_math import pick_port_unroll, tree_reduce_accs
from .helpers import BF16Ptr, F32Ptr, W


@always_inline
def score_position[head_dim: Int](q: BF16Ptr, k_row: BF16Ptr) -> Float32:
    comptime PU = pick_port_unroll[W, head_dim]()
    comptime STRIDE = PU * W
    var accs = InlineArray[SIMD[DType.float32, W], PU](fill=SIMD[DType.float32, W](0))
    for i in range(head_dim // STRIDE):
        comptime for p in range(PU):
            var qv = (q + i * STRIDE + p * W).load[width=W]().cast[DType.float32]()
            var kv = (k_row + i * STRIDE + p * W).load[width=W]().cast[DType.float32]()
            accs[p] = qv.fma(kv, accs[p])
    return tree_reduce_accs(accs)


@always_inline
def accumulate_v[head_dim: Int](
    v_row: BF16Ptr, weight: Float32, acc: F32Ptr,
):
    comptime PU = pick_port_unroll[W, head_dim]()
    comptime STRIDE = PU * W
    var w_vec = SIMD[DType.float32, W](weight)
    for i in range(head_dim // STRIDE):
        comptime for p in range(PU):
            var v = (v_row + i * STRIDE + p * W).load[width=W]().cast[DType.float32]()
            var a = (acc + i * STRIDE + p * W).load[width=W]()
            (acc + i * STRIDE + p * W).store(v.fma(w_vec, a))


@always_inline
def scale_acc[head_dim: Int](acc: F32Ptr, factor: Float32):
    comptime PU = pick_port_unroll[W, head_dim]()
    comptime STRIDE = PU * W
    var f = SIMD[DType.float32, W](factor)
    for i in range(head_dim // STRIDE):
        comptime for p in range(PU):
            var a = (acc + i * STRIDE + p * W).load[width=W]()
            (acc + i * STRIDE + p * W).store(a * f)
