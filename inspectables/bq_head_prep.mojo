from std.collections import InlineArray
from std.math import max
from std.sys import argv, llvm_intrinsic
from std.sys.info import simd_width_of
from std.benchmark import keep
from std.memory import UnsafePointer, alloc
from std.utils import IndexList

comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime I8Ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]


def log2[N: Int]() -> Int:
    comptime if N == 1:
        return 0
    else:
        return 1 + log2[N // 2]()


def butterfly_shuffle[width: Int, stride: Int]() -> IndexList[width]:
    var result = IndexList[width]()
    comptime for i in range(width):
        result[i] = i ^ stride
    return result


@always_inline
def sqrt[dtype: DType, width: Int](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return llvm_intrinsic["llvm.sqrt", SIMD[dtype, width], SIMD[dtype, width]](x)


@always_inline
def roundeven[dtype: DType, width: Int](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return llvm_intrinsic["llvm.nearbyint", SIMD[dtype, width], SIMD[dtype, width]](x)


@always_inline
def quantize_i8[width: Int](
    v: SIMD[DType.float32, width], inv_scale: SIMD[DType.float32, width],
) -> SIMD[DType.int8, width]:
    comptime lo = SIMD[DType.float32, width](-128.0)
    comptime hi = SIMD[DType.float32, width](127.0)
    return min(max(roundeven(v * inv_scale), lo), hi).cast[DType.int8]()


def fwht_width[T: DType, block: Int]() -> Int:
    comptime hw = simd_width_of[T]()
    comptime if block <= hw:
        return block
    else:
        return hw


@always_inline
def fwht_apply[T: DType, block: Int](
    mut r: InlineArray[
        SIMD[T, fwht_width[T, block]()], block // fwht_width[T, block](),
    ],
):
    comptime width = fwht_width[T, block]()
    comptime regs = block // width
    comptime stages = log2[block]()
    comptime for stage in range(stages):
        comptime stride = 1 << stage
        comptime if stride < width:
            comptime mask = butterfly_shuffle[width, stride]()
            var sign_buf = InlineArray[Scalar[T], width](fill=Scalar[T](1.0))
            comptime for k in range(width):
                comptime if (k >> stage) & 1 != 0:
                    sign_buf[k] = Scalar[T](-1.0)
            var sign = UnsafePointer(to=sign_buf).bitcast[Scalar[T]]().load[width=width]()
            comptime for i in range(regs):
                var partner = r[i].shuffle[mask=mask](r[i])
                r[i] = r[i].fma(sign, partner)
        else:
            comptime reg_stride = stride // width
            comptime num_groups = regs // (2 * reg_stride)
            comptime for g in range(num_groups):
                comptime for j in range(reg_stride):
                    comptime a_idx = g * 2 * reg_stride + j
                    comptime b_idx = a_idx + reg_stride
                    var a_val = r[a_idx]
                    var b_val = r[b_idx]
                    r[a_idx] = a_val + b_val
                    r[b_idx] = a_val - b_val
    var sc = Scalar[T](1.0 / Float64(sqrt[T, 1](Scalar[T](block))))
    comptime for i in range(regs):
        r[i] = r[i] * sc


@always_inline
def head_inv_rms[head_dim: Int, width: Int, regs: Int, sqrt_n: Float32, n_eps: Float32](
    read r: InlineArray[SIMD[DType.float32, width], regs],
) -> Float32:
    var ssq = r[0] * r[0]
    comptime for ri in range(1, regs):
        ssq = r[ri].fma(r[ri], ssq)
    return sqrt_n / sqrt[DType.float32, 1](ssq.reduce_add() + n_eps)[0]


@always_inline
def absmax_quantize_head[width: Int, regs: Int](
    read r: InlineArray[SIMD[DType.float32, width], regs], qi_out: I8Ptr,
) -> Tuple[Float32, Int32]:
    var vmax = SIMD[DType.float32, width](0)
    comptime for ri in range(regs):
        vmax = max(vmax, abs(r[ri]))
    var amax = vmax.reduce_max()
    if amax < Float32(1e-10):
        amax = Float32(1e-10)
    var inv = SIMD[DType.float32, width](Float32(127.0) / amax)
    var qsum = SIMD[DType.int32, width](0)
    comptime for ri in range(regs):
        var qi = quantize_i8[width](r[ri], inv)
        (qi_out + ri * width).store(qi)
        qsum += qi.cast[DType.int32]()
    return (amax, qsum.reduce_add())


@always_inline
def prep_head_qk_i8[
    head_dim: Int, rope_half: Int, pair_stride: Int, sqrt_n: Float32, n_eps: Float32,
](
    src: BF16Ptr, gamma: BF16Ptr, cos: F32Ptr, sin: F32Ptr, qi_out: I8Ptr,
) -> Tuple[Float32, Int32]:
    comptime width = fwht_width[DType.float32, head_dim]()
    comptime regs = head_dim // width
    comptime pair_reg_stride = pair_stride // width
    comptime rope_regs = rope_half // width
    var r = InlineArray[SIMD[DType.float32, width], regs](uninitialized=True)
    comptime for ri in range(regs):
        r[ri] = (src + ri * width).load[width=width]().cast[DType.float32]()
    var inv_rms = head_inv_rms[head_dim, width, regs, sqrt_n, n_eps](r)
    var fr = SIMD[DType.float32, width](inv_rms)
    comptime for ri in range(regs):
        var g = (gamma + ri * width).load[width=width]().cast[DType.float32]()
        r[ri] = r[ri] * fr * g
    comptime for ri in range(rope_regs):
        var x_lo = r[ri]
        var x_hi = r[pair_reg_stride + ri]
        var cv = (cos + ri * width).load[width=width]()
        var sv = (sin + ri * width).load[width=width]()
        r[ri] = x_lo * cv - x_hi * sv
        r[pair_reg_stride + ri] = x_hi * cv + x_lo * sv
    fwht_apply[DType.float32, head_dim](r)
    return absmax_quantize_head[width, regs](r, qi_out)


@always_inline
def prep_head_v_i8[head_dim: Int, sqrt_n: Float32, n_eps: Float32](
    src: BF16Ptr, vi_out: I8Ptr,
) -> Float32:
    comptime width = fwht_width[DType.float32, head_dim]()
    comptime regs = head_dim // width
    var r = InlineArray[SIMD[DType.float32, width], regs](uninitialized=True)
    comptime for ri in range(regs):
        r[ri] = (src + ri * width).load[width=width]().cast[DType.float32]()
    var inv_rms = head_inv_rms[head_dim, width, regs, sqrt_n, n_eps](r)
    var fr = SIMD[DType.float32, width](inv_rms)
    comptime for ri in range(regs):
        r[ri] = r[ri] * fr
    fwht_apply[DType.float32, head_dim](r)
    return absmax_quantize_head[width, regs](r, vi_out)[0]


@no_inline
@export
def probe_qk_128(src: BF16Ptr, g: BF16Ptr, c: F32Ptr, s: F32Ptr, o: I8Ptr) -> Int32:
    var r = prep_head_qk_i8[128, 64, 64, 11.3137, 1e-6](src, g, c, s, o)
    return r[1]


@no_inline
@export
def probe_qk_256(src: BF16Ptr, g: BF16Ptr, c: F32Ptr, s: F32Ptr, o: I8Ptr) -> Int32:
    var r = prep_head_qk_i8[256, 128, 128, 16.0, 1e-6](src, g, c, s, o)
    return r[1]


@no_inline
@export
def probe_v_128(src: BF16Ptr, o: I8Ptr) -> Float32:
    return prep_head_v_i8[128, 11.3137, 1e-6](src, o)


@no_inline
@export
def probe_v_256(src: BF16Ptr, o: I8Ptr) -> Float32:
    return prep_head_v_i8[256, 16.0, 1e-6](src, o)


def main():
    var seed = len(argv())
    var src = alloc[BFloat16](256).as_any_origin()
    var g = alloc[BFloat16](256).as_any_origin()
    var c = alloc[Float32](256).as_any_origin()
    var s = alloc[Float32](256).as_any_origin()
    var o = alloc[Int8](256).as_any_origin()
    for i in range(256):
        src[i] = BFloat16(Float32(i % 7 + seed))
        g[i] = BFloat16(Float32(i % 3 + 1))
        c[i] = Float32(i % 5) * 0.1
        s[i] = Float32(i % 4) * 0.1
    keep(probe_qk_128(src, g, c, s, o))
    keep(probe_qk_256(src, g, c, s, o))
    keep(probe_v_128(src, o))
    keep(probe_v_256(src, o))
    keep(o[seed])
