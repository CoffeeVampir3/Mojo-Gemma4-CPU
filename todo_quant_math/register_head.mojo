from std.collections import InlineArray
from std.memory import UnsafePointer

from .fwht import fwht_apply, fwht_width
from .i8_quantize import quantize_i8
from .types import BF16Ptr, F32Ptr, I8Ptr


@always_inline
def load_head_bf16[width: Int, regs: Int, //](
    src: BF16Ptr,
    mut r: InlineArray[SIMD[DType.float32, width], regs],
):
    comptime for ri in range(regs):
        r[ri] = (src + ri * width).load[width=width]().cast[DType.float32]()


@always_inline
def regs_sum_sq[width: Int, regs: Int, //](
    r: InlineArray[SIMD[DType.float32, width], regs],
) -> Float32:
    var acc = r[0] * r[0]
    comptime for ri in range(1, regs):
        acc = r[ri].fma(r[ri], acc)
    return acc.reduce_add()


@always_inline
def regs_scale[width: Int, regs: Int, //](
    mut r: InlineArray[SIMD[DType.float32, width], regs],
    s: Float32,
):
    var vs = SIMD[DType.float32, width](s)
    comptime for ri in range(regs):
        r[ri] = r[ri] * vs


@always_inline
def regs_scale_with_gamma_bf16[width: Int, regs: Int, //](
    mut r: InlineArray[SIMD[DType.float32, width], regs],
    s: Float32,
    gamma: BF16Ptr,
):
    var vs = SIMD[DType.float32, width](s)
    comptime for ri in range(regs):
        var g = (gamma + ri * width).load[width=width]().cast[DType.float32]()
        r[ri] = r[ri] * vs * g


@always_inline
def regs_rope_partial[
    width: Int, regs: Int, //,
    rope_regs: Int, pair_reg_stride: Int,
](
    mut r: InlineArray[SIMD[DType.float32, width], regs],
    cos: F32Ptr,
    sin: F32Ptr,
):
    """RoPE over the first rope_regs register pairs."""
    comptime for ri in range(rope_regs):
        var x_lo = r[ri]
        var x_hi = r[pair_reg_stride + ri]
        var cv = (cos + ri * width).load[width=width]()
        var sv = (sin + ri * width).load[width=width]()
        r[ri] = x_lo * cv - x_hi * sv
        r[pair_reg_stride + ri] = x_hi * cv + x_lo * sv


@always_inline
def regs_absmax_quantize_i8[width: Int, regs: Int, //](
    r: InlineArray[SIMD[DType.float32, width], regs],
    qi_out: I8Ptr,
) -> Tuple[Float32, Int32]:
    """Quantize a register bank and return (absmax, sum(qi))."""
    var vmax = SIMD[DType.float32, width](0)
    comptime for ri in range(regs):
        vmax = max(vmax, r[ri].__abs__())
    var absmax = vmax.reduce_max()
    if absmax < Float32(1e-10):
        absmax = Float32(1e-10)
    var inv = SIMD[DType.float32, width](Float32(127.0) / absmax)
    var qsum = SIMD[DType.int32, width](0)
    comptime for ri in range(regs):
        var qi = quantize_i8[width](r[ri], inv)
        (qi_out + ri * width).store(qi)
        qsum += qi.cast[DType.int32]()
    return (absmax, qsum.reduce_add())


@always_inline
def prep_q_head_i8[head_dim: Int, rope_dim: Int, pair_stride: Int](
    q_bf16: BF16Ptr,
    gamma: BF16Ptr,
    cos: F32Ptr,
    sin: F32Ptr,
    inv_rms: Float32,
    qi_out: UnsafePointer[Int8, MutAnyOrigin],
) -> Tuple[Float32, Float32]:
    """MiniMax Q prep: gamma scale, partial RoPE, FWHT, and i8 quantize.

    Returns (qi_bias, absmax). qi_bias is sum(q_i8) * 128 and is subtracted
    from u8-biased VNNI score accumulators.
    """
    comptime width = fwht_width[DType.float32, head_dim]()
    comptime regs = head_dim // width
    comptime pair_reg_stride = pair_stride // width
    comptime rope_regs = (rope_dim // 2) // width

    var r = InlineArray[SIMD[DType.float32, width], regs](uninitialized=True)
    load_head_bf16(q_bf16, r)
    regs_scale_with_gamma_bf16(r, inv_rms, gamma)
    regs_rope_partial[rope_regs=rope_regs, pair_reg_stride=pair_reg_stride](
        r, cos, sin)
    fwht_apply[DType.float32, head_dim](r)
    var quant = regs_absmax_quantize_i8(
        r, qi_out.bitcast[Scalar[DType.int8]]())
    return (Float32(quant[1]) * Float32(128.0), quant[0])


@always_inline
def prep_k_head_i8[head_dim: Int, rope_dim: Int, pair_stride: Int](
    k_bf16: BF16Ptr,
    gamma: BF16Ptr,
    cos: F32Ptr,
    sin: F32Ptr,
    inv_rms: Float32,
    qi_out: I8Ptr,
) -> Float32:
    """MiniMax K prep without cache writes. Returns absmax scale."""
    comptime width = fwht_width[DType.float32, head_dim]()
    comptime regs = head_dim // width
    comptime pair_reg_stride = pair_stride // width
    comptime rope_regs = (rope_dim // 2) // width

    var r = InlineArray[SIMD[DType.float32, width], regs](uninitialized=True)
    load_head_bf16(k_bf16, r)
    regs_scale_with_gamma_bf16(r, inv_rms, gamma)
    regs_rope_partial[rope_regs=rope_regs, pair_reg_stride=pair_reg_stride](
        r, cos, sin)
    fwht_apply[DType.float32, head_dim](r)
    return regs_absmax_quantize_i8(r, qi_out)[0]


@always_inline
def prep_v_head_i8[head_dim: Int](
    v_bf16: BF16Ptr,
    qi_out: I8Ptr,
) -> Float32:
    """MiniMax V prep without cache writes. Returns absmax scale."""
    comptime width = fwht_width[DType.float32, head_dim]()
    comptime regs = head_dim // width

    var r = InlineArray[SIMD[DType.float32, width], regs](uninitialized=True)
    load_head_bf16(v_bf16, r)
    return regs_absmax_quantize_i8(r, qi_out)[0]
