from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys import llvm_intrinsic
from std.sys.info import simd_width_of
from std.benchmark import keep


@always_inline
def roundeven[dtype: DType, width: Int](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return llvm_intrinsic[
        "llvm.nearbyint",
        SIMD[dtype, width],
        SIMD[dtype, width],
    ](x)


def is_power_of_two[N: Int]() -> Bool:
    return N > 0 and (N & (N - 1)) == 0


@always_inline
def port_unroll_for[count: Int]() -> Int:
    comptime if count >= 8:
        return 8
    elif count >= 4:
        return 4
    elif count >= 2:
        return 2
    else:
        return 1


@always_inline
def pick_port_unroll[width: Int, cols: Int]() -> Int:
    comptime if cols % (8 * width) == 0:
        return 8
    elif cols % (4 * width) == 0:
        return 4
    elif cols % (2 * width) == 0:
        return 2
    else:
        return 1


@always_inline
def quantize_i8[width: Int](
    v: SIMD[DType.float32, width],
    inv_scale: SIMD[DType.float32, width],
) -> SIMD[DType.int8, width]:
    comptime lo = SIMD[DType.float32, width](-128.0)
    comptime hi = SIMD[DType.float32, width](127.0)
    return min(max(roundeven(v * inv_scale), lo), hi).cast[DType.int8]()


@always_inline
def quantize_i8_scalar(v: Float32, inv_scale: Float32) -> Scalar[DType.int8]:
    var q = roundeven[DType.float32, 1](v * inv_scale)
    return min(max(q, Float32(-128.0)), Float32(127.0)).cast[DType.int8]()


@always_inline
def quantize_i8_chunks[cols: Int](
    src: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    inv_scale: Float32,
):
    comptime width = simd_width_of[DType.float32]()
    comptime PU = pick_port_unroll[width, cols]()
    comptime STRIDE = PU * width
    var qs = SIMD[DType.float32, width](inv_scale)
    for i in range(cols // STRIDE):
        comptime for p in range(PU):
            comptime off = p * width
            var v = (src + i * STRIDE + off).load[width=width]()
            (dst + i * STRIDE + off).store(quantize_i8[width](v, qs))
    var k = (cols // STRIDE) * STRIDE
    while k < cols:
        dst[k] = quantize_i8_scalar(src[k], inv_scale)
        k += 1


@always_inline
def absmax_quantize_i8[cols: Int](
    src: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
) -> Float32:
    comptime width = simd_width_of[DType.float32]()
    comptime PU = pick_port_unroll[width, cols]()
    comptime STRIDE = PU * width
    var bank = InlineArray[SIMD[DType.float32, width], PU](
        fill=SIMD[DType.float32, width](0))
    for i in range(cols // STRIDE):
        comptime for p in range(PU):
            comptime off = p * width
            bank[p] = max(
                bank[p],
                (src + i * STRIDE + off).load[width=width]().__abs__())
    var folded = bank[0]
    comptime for p in range(1, PU):
        folded = max(folded, bank[p])
    var absmax = folded.reduce_max()
    var k = (cols // STRIDE) * STRIDE
    while k < cols:
        absmax = max(absmax, src[k].__abs__())
        k += 1
    if absmax < Float32(1e-10):
        absmax = Float32(1e-10)
    quantize_i8_chunks[cols](src, dst, Float32(127.0) / absmax)
    return absmax


def quantize_4096(
    src: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    inv_scale: Float32,
):
    quantize_i8_chunks[4096](src, dst, inv_scale)


def quantize_1024(
    src: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    inv_scale: Float32,
):
    quantize_i8_chunks[1024](src, dst, inv_scale)


def absmax_4096(
    src: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
) -> Float32:
    return absmax_quantize_i8[4096](src, dst)


def main():
    var src = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()
    var dst = UnsafePointer[Scalar[DType.int8], MutAnyOrigin].unsafe_dangling()
    quantize_4096(src, dst, Float32(1.0))
    quantize_1024(src, dst, Float32(1.0))
    var amx = absmax_4096(src, dst)
    keep(amx)
    keep(src)
    keep(dst)
