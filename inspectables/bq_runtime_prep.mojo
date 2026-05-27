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
comptime WF = simd_width_of[DType.float32]()


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
    mut r: InlineArray[SIMD[T, fwht_width[T, block]()], block // fwht_width[T, block]()],
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
def fwht_block[block: Int](buf: F32Ptr):
    comptime width = fwht_width[DType.float32, block]()
    comptime regs = block // width
    var r = InlineArray[SIMD[DType.float32, width], regs](fill=SIMD[DType.float32, width](0))
    comptime for i in range(regs):
        r[i] = (buf + i * width).load[width=width]()
    fwht_apply[DType.float32, block](r)
    comptime for i in range(regs):
        (buf + i * width).store(r[i])


@always_inline
def fwht_row[block: Int](buf: F32Ptr, cols: Int):
    for b in range(cols // block):
        fwht_block[block](buf + b * block)


@always_inline
def rms_reduce_row[hidden: Int](src: BF16Ptr) -> Float32:
    var acc = SIMD[DType.float32, WF](0)
    var k = 0
    while k + WF <= hidden:
        var x = (src + k).load[width=WF]().cast[DType.float32]()
        acc = x.fma(x, acc)
        k += WF
    return acc.reduce_add() / Float32(hidden)


@always_inline
def row_absmax(work_row: F32Ptr, cols: Int) -> Float32:
    var vmax = SIMD[DType.float32, WF](0)
    var k = 0
    while k + WF <= cols:
        vmax = max(vmax, abs((work_row + k).load[width=WF]()))
        k += WF
    return vmax.reduce_max()


@always_inline
def quantize_activation_per_row(work: F32Ptr, qi: I8Ptr, sa: F32Ptr, cols: Int):
    var amax = row_absmax(work, cols)
    sa[0] = amax
    var inv = Float32(127.0) / amax if amax > Float32(0) else Float32(0)
    var vinv = SIMD[DType.float32, WF](inv)
    var k = 0
    while k + WF <= cols:
        (qi + k).store(quantize_i8[WF]((work + k).load[width=WF](), vinv))
        k += WF


@always_inline
def prepare_norm_activation[
    hidden: Int, block: Int, sqrt_n: Float32, n_eps: Float32,
](src: BF16Ptr, gamma: BF16Ptr, x_i8: I8Ptr, sa: F32Ptr):
    var work = InlineArray[Float32, hidden](uninitialized=True)
    var wp = UnsafePointer(to=work[0]).as_any_origin()
    var sum_sq = rms_reduce_row[hidden](src)
    var inv_rms = sqrt_n / sqrt[DType.float32, 1](sum_sq + n_eps)
    var fr = SIMD[DType.float32, WF](inv_rms)
    var k = 0
    while k + WF <= hidden:
        var x = (src + k).load[width=WF]().cast[DType.float32]()
        var g = (gamma + k).load[width=WF]().cast[DType.float32]()
        (wp + k).store(x * fr * g)
        k += WF
    fwht_row[block](wp, hidden)
    quantize_activation_per_row(wp, x_i8, sa, hidden)
    _ = work


@no_inline
@export
def probe_norm_quant(src: BF16Ptr, gamma: BF16Ptr, x_i8: I8Ptr, sa: F32Ptr):
    prepare_norm_activation[4096, 128, 64.0, 1e-6](src, gamma, x_i8, sa)


def main():
    var seed = len(argv())
    var src = alloc[BFloat16](4096).as_any_origin()
    var gamma = alloc[BFloat16](4096).as_any_origin()
    var x_i8 = alloc[Int8](4096).as_any_origin()
    var sa = alloc[Float32](1).as_any_origin()
    for i in range(4096):
        src[i] = BFloat16(Float32(i % 11 + seed) - 4.0)
        gamma[i] = BFloat16(Float32(i % 3 + 1))
    probe_norm_quant(src, gamma, x_i8, sa)
    keep(x_i8[seed]); keep(sa[0])
