from std.math import max
from std.sys import argv, llvm_intrinsic
from std.sys.info import simd_width_of
from std.benchmark import keep
from std.memory import UnsafePointer, alloc

comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime I8Ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime WF = simd_width_of[DType.float32]()


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


@always_inline
def row_absmax(work_row: F32Ptr, cols: Int) -> Float32:
    var vmax = SIMD[DType.float32, WF](0)
    var k = 0
    while k + WF <= cols:
        vmax = max(vmax, abs((work_row + k).load[width=WF]()))
        k += WF
    return vmax.reduce_max()


@always_inline
def quantize_inv(work: F32Ptr, qi: I8Ptr, inv: Float32, n: Int):
    var vinv = SIMD[DType.float32, WF](inv)
    var k = 0
    while k + WF <= n:
        var v = (work + k).load[width=WF]()
        (qi + k).store(quantize_i8[WF](v, vinv))
        k += WF


@always_inline
def bake_split_gain_in_place(gamma: BF16Ptr, cols: Int, eps: Float32 = 1e-12):
    var floor = SIMD[DType.float32, WF](eps)
    var zero = SIMD[DType.float32, WF](0)
    var k = 0
    while k + WF <= cols:
        var g = (gamma + k).load[width=WF]().cast[DType.float32]()
        var s = sqrt[DType.float32, WF](max(abs(g), floor))
        (gamma + k).store(g.lt(zero).select(-s, s).cast[DType.bfloat16]())
        k += WF


@always_inline
def scale_cast_row[hidden: Int, scale: Float64](work: F32Ptr, dst: BF16Ptr):
    var f = SIMD[DType.float32, WF](Float32(scale))
    var k = 0
    while k + WF <= hidden:
        (dst + k).store(((work + k).load[width=WF]() * f).cast[DType.bfloat16]())
        k += WF


@always_inline
def dequant_weight_row_per_block[block: Int](
    qi: I8Ptr, scales: F32Ptr, dst: F32Ptr, cols: Int,
):
    var nb = cols // block
    for b in range(nb):
        var off = b * block
        var sb = SIMD[DType.float32, WF](scales[b])
        var k = 0
        while k + WF <= block:
            var v = (qi + off + k).load[width=WF]().cast[DType.float32]()
            (dst + off + k).store(v * sb)
            k += WF


@no_inline
@export
def probe_quantize_i8(v: SIMD[DType.float32, WF], inv: SIMD[DType.float32, WF]) -> SIMD[DType.int8, WF]:
    return quantize_i8[WF](v, inv)


@no_inline
@export
def probe_quantize_inv(work: F32Ptr, qi: I8Ptr, inv: Float32, n: Int):
    quantize_inv(work, qi, inv, n)


@no_inline
@export
def probe_row_absmax(work: F32Ptr, cols: Int) -> Float32:
    return row_absmax(work, cols)


@no_inline
@export
def probe_bake_split_gain(gamma: BF16Ptr, cols: Int):
    bake_split_gain_in_place(gamma, cols)


@no_inline
@export
def probe_scale_cast_row(work: F32Ptr, dst: BF16Ptr):
    scale_cast_row[4096, 0.5](work, dst)


@no_inline
@export
def probe_dequant_weight(qi: I8Ptr, scales: F32Ptr, dst: F32Ptr, cols: Int):
    dequant_weight_row_per_block[128](qi, scales, dst, cols)


def main():
    var seed = len(argv())
    var work = alloc[Float32](4096).as_any_origin()
    var qi = alloc[Int8](4096).as_any_origin()
    var dst = alloc[Float32](4096).as_any_origin()
    var gamma = alloc[BFloat16](4096).as_any_origin()
    var bdst = alloc[BFloat16](4096).as_any_origin()
    var scales = alloc[Float32](64).as_any_origin()
    for i in range(4096):
        work[i] = Float32(i % 17 + seed) - 8.0
        qi[i] = Int8((i + seed) % 127)
        gamma[i] = BFloat16(Float32(i % 5) - 2.0)
    for i in range(64):
        scales[i] = Float32(i % 7 + 1) * 0.01
    var vv = SIMD[DType.float32, WF](Float32(seed) + 1.5)
    var ii = SIMD[DType.float32, WF](Float32(seed) + 3.0)
    keep(probe_quantize_i8(vv, ii))
    probe_quantize_inv(work, qi, Float32(seed) + 2.0, 4096)
    keep(probe_row_absmax(work, 4096))
    probe_bake_split_gain(gamma, 4096)
    probe_scale_cast_row(work, bdst)
    probe_dequant_weight(qi, scales, dst, 4096)
    keep(qi[seed]); keep(dst[seed]); keep(gamma[seed]); keep(bdst[seed])
