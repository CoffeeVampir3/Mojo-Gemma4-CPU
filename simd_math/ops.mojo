from std.sys import llvm_intrinsic


@always_inline
def sqrt[dtype: DType, width: Int](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """SIMD sqrt — lowers to vsqrtps (f32) or vsqrtpd (f64)."""
    return llvm_intrinsic[
        "llvm.sqrt",
        SIMD[dtype, width],
        SIMD[dtype, width],
    ](x)


@always_inline
def roundeven[dtype: DType, width: Int](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Round to nearest even — lowers to vroundps/vrndscaleps (f32) or
    vroundpd/vrndscalepd (f64)."""
    return llvm_intrinsic[
        "llvm.nearbyint",
        SIMD[dtype, width],
        SIMD[dtype, width],
    ](x)


@always_inline
def quantize_i8[width: Int](
    v: SIMD[DType.float32, width], inv_scale: SIMD[DType.float32, width],
) -> SIMD[DType.int8, width]:
    """Absmax quantize f32 → i8: round(v * inv_scale), clamp [-128, 127].

    inv_scale = 127.0 / absmax (precomputed by caller).
    """
    comptime lo = SIMD[DType.float32, width](-128.0)
    comptime hi = SIMD[DType.float32, width](127.0)
    return min(max(roundeven(v * inv_scale), lo), hi).cast[DType.int8]()
