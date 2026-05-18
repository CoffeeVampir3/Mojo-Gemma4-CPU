from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys import CompilationTarget, llvm_intrinsic
from std.sys.info import simd_width_of, size_of


comptime F32W = simd_width_of[DType.float32]()
comptime BF16W = simd_width_of[DType.bfloat16]()
comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]


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

    On AVX-512BF16 emits one VDPBF16PS for the native SIMD width.
    Otherwise falls back to deinterleave + cast + FMAs with the same
    per-lane semantics.
    """
    comptime vector_bits = size_of[SIMD[DType.float32, F32W]]() * 8
    comptime if has_avx512_bf16():
        return llvm_intrinsic[
            "llvm.x86.avx512bf16.dpbf16ps." + String(vector_bits),
            SIMD[DType.float32, F32W],
        ](acc, a, b)
    else:
        var ae_ao = a.deinterleave()
        var be_bo = b.deinterleave()
        var ae = rebind[SIMD[DType.bfloat16, F32W]](ae_ao[0]).cast[DType.float32]()
        var ao = rebind[SIMD[DType.bfloat16, F32W]](ae_ao[1]).cast[DType.float32]()
        var be = rebind[SIMD[DType.bfloat16, F32W]](be_bo[0]).cast[DType.float32]()
        var bo = rebind[SIMD[DType.bfloat16, F32W]](be_bo[1]).cast[DType.float32]()
        var inner = ae.fma(be, acc)
        return ao.fma(bo, inner)


@always_inline
def bf16_panel_dot[
    panel: Int, port_unroll: Int, //,
    cols: Int,
](
    weight_row: BF16Ptr,
    read x_rows: InlineArray[BF16Ptr, panel],
    mut accs: InlineArray[
        InlineArray[SIMD[DType.float32, F32W], port_unroll], panel,
    ],
):
    """PU-unrolled VDPBF16PS over `cols` bf16 elements: one weight column
    is read once and dotted against `panel` x rows in lockstep, FMAing
    into a per-row, per-port f32 accumulator bank. Caller initializes
    `accs` (typically to zero) before the call."""
    comptime STRIDE = port_unroll * BF16W
    for i in range(cols // STRIDE):
        comptime for p in range(port_unroll):
            var off = i * STRIDE + p * BF16W
            var w_v = (weight_row + off).load[width=BF16W]()
            comptime for r in range(panel):
                var x_v = (x_rows[r] + off).load[width=BF16W]()
                accs[r][p] = bf16_pair_dot(accs[r][p], x_v, w_v)


@always_inline
def panel_accs_to_scalars[
    panel: Int, port_unroll: Int, //,
](
    read accs: InlineArray[
        InlineArray[SIMD[DType.float32, F32W], port_unroll], panel,
    ],
) -> InlineArray[Float32, panel]:
    """Per-row: sum the PU partials then horizontal-reduce to a scalar."""
    var out = InlineArray[Float32, panel](uninitialized=True)
    comptime for r in range(panel):
        var s = SIMD[DType.float32, F32W](0)
        comptime for p in range(port_unroll):
            s += accs[r][p]
        out[r] = s.reduce_add()
    return out


@always_inline
def bf16_panel_dot_to_scalars[
    panel: Int, //,
    cols: Int, port_unroll: Int,
](
    weight_row: BF16Ptr,
    read x_rows: InlineArray[BF16Ptr, panel],
) -> InlineArray[Float32, panel]:
    """Initialize, panel-dot, reduce to scalars — the full one-shot
    operation Phase1 and Phase2 both want."""
    var accs = InlineArray[
        InlineArray[SIMD[DType.float32, F32W], port_unroll], panel,
    ](uninitialized=True)
    comptime for r in range(panel):
        comptime for p in range(port_unroll):
            accs[r][p] = SIMD[DType.float32, F32W](0)
    bf16_panel_dot[cols=cols](weight_row, x_rows, accs)
    return panel_accs_to_scalars(accs)