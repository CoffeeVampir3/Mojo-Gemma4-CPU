from std.sys import CompilationTarget, llvm_intrinsic


@always_inline
fn has_avx512_bf16() -> Bool:
    return CompilationTarget._has_feature["avx512bf16"]()


@always_inline
fn bf16_pair_dot[width: Int](
    var acc: SIMD[DType.float32, 4 * width],
    var a: SIMD[DType.bfloat16, 8 * width],
    var b: SIMD[DType.bfloat16, 8 * width],
) -> SIMD[DType.float32, 4 * width]:
    """bf16 pairwise dot-into-f32 accumulator. `width` is the count of
    128-bit blocks (2 → ymm/256, 4 → zmm/512). Per lane j ∈ [0, 4*width):
    `acc[j] += bf16_to_f32(a[2j])   * bf16_to_f32(b[2j])
            +  bf16_to_f32(a[2j+1]) * bf16_to_f32(b[2j+1])`.

    On AVX-512BF16 targets emits one VDPBF16PS — intrinsic name built at
    comptime from `128 * width`. Otherwise falls back to deinterleave +
    cast + 2 FMAs with the same per-lane semantics."""
    @parameter
    if has_avx512_bf16():
        return llvm_intrinsic[
            "llvm.x86.avx512bf16.dpbf16ps." + String(128 * width),
            SIMD[DType.float32, 4 * width],
        ](acc, a, b)
    else:
        # `rebind` bridges Mojo's type-checker: deinterleave returns
        # SIMD[..,(8*width)//2] which is mathematically 4*width, but the
        # type-checker compares type expressions structurally and won't
        # reduce the integer division. The rebind is a runtime no-op.
        var ae_ao = a.deinterleave()
        var be_bo = b.deinterleave()
        var ae = rebind[SIMD[DType.bfloat16, 4 * width]](
            ae_ao[0]).cast[DType.float32]()
        var ao = rebind[SIMD[DType.bfloat16, 4 * width]](
            ae_ao[1]).cast[DType.float32]()
        var be = rebind[SIMD[DType.bfloat16, 4 * width]](
            be_bo[0]).cast[DType.float32]()
        var bo = rebind[SIMD[DType.bfloat16, 4 * width]](
            be_bo[1]).cast[DType.float32]()
        var inner = ae.fma(be, acc)
        return ao.fma(bo, inner)


from std.memory import UnsafePointer


@always_inline
def prefetch_l1[dtype: DType](ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin]):
    llvm_intrinsic["llvm.prefetch", NoneType](
        ptr.bitcast[UInt8](), Int32(0), Int32(3), Int32(1))


@always_inline
def bcast_bf16_pair_zmm(
    p: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
) -> SIMD[DType.bfloat16, 32]:
    # Load 2 consecutive bf16 elements at p[0..2], pack into a 32-bit dword,
    # broadcast that dword across all 16 i32 lanes, view as 32 bf16.
    # Used as the broadcast operand of `bf16_pair_dot` in M-axis-SIMD outer
    # product GEMMs: every i32 lane sees the same (k, k+1) pair.
    var packed = p.bitcast[Scalar[DType.int32]]()[0]
    var bcast = SIMD[DType.int32, 16](packed)
    return bcast.bitcast[DType.bfloat16, 32]()

