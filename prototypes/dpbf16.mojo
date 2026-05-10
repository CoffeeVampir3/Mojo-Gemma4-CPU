from std.sys import CompilationTarget, llvm_intrinsic
from std.sys.info import simd_width_of, size_of


comptime F32W = simd_width_of[DType.float32]()
comptime BF16W = simd_width_of[DType.bfloat16]()


@always_inline
def has_avx512_bf16() -> Bool:
    return CompilationTarget._has_feature["avx512bf16"]()


@always_inline
def bf16_pair_dot(
    var acc: SIMD[DType.float32, F32W],
    var a: SIMD[DType.bfloat16, BF16W],
    var b: SIMD[DType.bfloat16, BF16W],
) -> SIMD[DType.float32, F32W]:
    """bf16 pairwise dot-into-f32 accumulator over the target SIMD width.
    Per f32 accumulator lane j:
    `acc[j] += bf16_to_f32(a[2j])   * bf16_to_f32(b[2j])
            +  bf16_to_f32(a[2j+1]) * bf16_to_f32(b[2j+1])`.

    On AVX-512BF16 targets emits one VDPBF16PS for the native SIMD width.
    Otherwise falls back to deinterleave + cast + FMAs with the same
    per-lane semantics."""
    comptime assert BF16W == 2 * F32W, (
        "bf16_pair_dot requires two bf16 lanes per f32 accumulator lane")
    comptime vector_bits = size_of[SIMD[DType.float32, F32W]]() * 8
    comptime if has_avx512_bf16():
        return llvm_intrinsic[
            "llvm.x86.avx512bf16.dpbf16ps." + String(vector_bits),
            SIMD[DType.float32, F32W],
        ](acc, a, b)
    else:
        # `rebind` bridges Mojo's type-checker: deinterleave returns
        # SIMD[.., BF16W // 2], which is the same lane count as the f32
        # accumulator on current CPU targets. The rebind is a runtime no-op.
        var ae_ao = a.deinterleave()
        var be_bo = b.deinterleave()
        var ae = rebind[SIMD[DType.bfloat16, F32W]](ae_ao[0]).cast[DType.float32]()
        var ao = rebind[SIMD[DType.bfloat16, F32W]](ae_ao[1]).cast[DType.float32]()
        var be = rebind[SIMD[DType.bfloat16, F32W]](be_bo[0]).cast[DType.float32]()
        var bo = rebind[SIMD[DType.bfloat16, F32W]](be_bo[1]).cast[DType.float32]()
        var inner = ae.fma(be, acc)
        return ao.fma(bo, inner)
