from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from simd_math.ops import sqrt
from kernels.rmsnorm import rms_reduce_row

from butterquant.fwht import fwht_row
from butterquant.kernels import row_absmax, quantize_inv


comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime I8Ptr = UnsafePointer[Int8, MutAnyOrigin]
comptime WF = simd_width_of[DType.float32]()
comptime WI = simd_width_of[DType.int32]()


@always_inline
def block_absmax(p: F32Ptr, n: Int) -> Float32:
    return row_absmax(p, n)


@always_inline
def quantize_activation_per_block[block: Int](
    work: F32Ptr, qi: I8Ptr, sa: F32Ptr, cols: Int,
):
    comptime assert block % WF == 0, (
        "FWHT block must be a multiple of the f32 SIMD width")
    var nb = cols // block
    for b in range(nb):
        var off = b * block
        var amax = block_absmax(work + off, block)
        sa[b] = amax
        var inv = Float32(127.0) / amax if amax > Float32(0) else Float32(0)
        quantize_inv(work + off, qi + off, inv, block)


@always_inline
def dequant_weight_row_per_block[block: Int](
    qi: I8Ptr, scales: F32Ptr, dst: F32Ptr, cols: Int,
):
    comptime assert block % WF == 0, (
        "FWHT block must be a multiple of the f32 SIMD width")
    var nb = cols // block
    for b in range(nb):
        var off = b * block
        var sb = SIMD[DType.float32, WF](scales[b])
        var k = 0
        while k + WF <= block:
            var v = (qi + off + k).load[width=WF]().cast[DType.float32]()
            (dst + off + k).store(v * sb)
            k += WF


@always_inline
def i8_block_dot(a: I8Ptr, b: I8Ptr, n: Int) -> Int32:
    var acc = SIMD[DType.int32, WI](0)
    var k = 0
    while k + WI <= n:
        var av = (a + k).load[width=WI]().cast[DType.int32]()
        var bv = (b + k).load[width=WI]().cast[DType.int32]()
        acc += av * bv
        k += WI
    return acc.reduce_add()


@always_inline
def head_logit_row[block: Int](
    x_i8: I8Ptr, sa: F32Ptr, weight: I8Ptr, scales: F32Ptr, cols: Int,
) -> Float32:
    comptime assert block % WI == 0, (
        "FWHT block must be a multiple of the i32 SIMD width")
    var nb = cols // block
    var acc = Float32(0)
    for b in range(nb):
        var off = b * block
        var r = i8_block_dot(x_i8 + off, weight + off, block)
        acc += Float32(r) * (sa[b] / Float32(127.0)) * scales[b]
    return acc


@always_inline
def scale_cast_row[hidden: Int, scale: Float64](work: F32Ptr, dst: BF16Ptr):
    comptime assert hidden % WF == 0, (
        "hidden must be a multiple of the f32 SIMD width")
    var f = SIMD[DType.float32, WF](Float32(scale))
    var k = 0
    while k + WF <= hidden:
        (dst + k).store(((work + k).load[width=WF]() * f).cast[DType.bfloat16]())
        k += WF


@always_inline
def zero_row[hidden: Int](dst: BF16Ptr):
    comptime assert hidden % WF == 0, (
        "hidden must be a multiple of the f32 SIMD width")
    var k = 0
    while k + WF <= hidden:
        (dst + k).store(SIMD[DType.bfloat16, WF](0))
        k += WF


@always_inline
def prepare_head_activation[
    hidden: Int, block: Int, sqrt_n: Float32, n_eps: Float32,
](
    src: BF16Ptr, gamma: BF16Ptr, x_i8: I8Ptr, sa: F32Ptr,
):
    comptime assert hidden % WF == 0, (
        "hidden must be a multiple of the f32 SIMD width")
    var work = InlineArray[Float32, hidden](uninitialized=True)
    var wp = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(UnsafePointer(to=work[0])))

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
    quantize_activation_per_block[block](wp, x_i8, sa, hidden)
    _ = work
