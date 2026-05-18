from std.memory import UnsafePointer, alloc
from std.collections import InlineArray
from std.benchmark import keep
from std.sys.info import simd_width_of, size_of
from std.sys import CompilationTarget, llvm_intrinsic


comptime W = simd_width_of[DType.float32]()
comptime BW = simd_width_of[DType.bfloat16]()
comptime HIDDEN = 2816
comptime INTERMEDIATE = 704
comptime TILE_J = 64
comptime MR = 4
comptime PHASE1_PU_FUSED = 2
comptime PHASE1_PU_SPLIT = 4

comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]


@always_inline
def has_avx512_bf16() -> Bool:
    return CompilationTarget._has_feature["avx512bf16"]()


@always_inline
def bf16_pair_dot(
    var acc: SIMD[DType.float32, W],
    var a: SIMD[DType.bfloat16, BW],
    var b: SIMD[DType.bfloat16, BW],
) -> SIMD[DType.float32, W]:
    comptime vector_bits = size_of[SIMD[DType.float32, W]]() * 8
    comptime if has_avx512_bf16():
        return llvm_intrinsic[
            "llvm.x86.avx512bf16.dpbf16ps." + String(vector_bits),
            SIMD[DType.float32, W],
        ](acc, a, b)
    else:
        var ae_ao = a.deinterleave()
        var be_bo = b.deinterleave()
        var ae = rebind[SIMD[DType.bfloat16, W]](ae_ao[0]).cast[DType.float32]()
        var ao = rebind[SIMD[DType.bfloat16, W]](ae_ao[1]).cast[DType.float32]()
        var be = rebind[SIMD[DType.bfloat16, W]](be_bo[0]).cast[DType.float32]()
        var bo = rebind[SIMD[DType.bfloat16, W]](be_bo[1]).cast[DType.float32]()
        var inner = ae.fma(be, acc)
        return ao.fma(bo, inner)


@always_inline
def phase1_paired_block[
    hidden: Int, tile_j: Int, MR_p: Int,
](
    x_rows: InlineArray[BF16Ptr, MR_p],
    gate_w_base: BF16Ptr,
    up_w_base: BF16Ptr,
    gate_part: F32Ptr,
    up_part: F32Ptr,
):
    comptime if has_avx512_bf16():
        comptime PU = PHASE1_PU_FUSED
        comptime STRIDE = PU * BW
        comptime assert hidden % STRIDE == 0
        for j_off in range(tile_j):
            var g_w_row = gate_w_base + j_off * hidden
            var u_w_row = up_w_base + j_off * hidden
            var g_accs = InlineArray[
                InlineArray[SIMD[DType.float32, W], PU], MR_p,
            ](uninitialized=True)
            var u_accs = InlineArray[
                InlineArray[SIMD[DType.float32, W], PU], MR_p,
            ](uninitialized=True)
            comptime for r in range(MR_p):
                comptime for p in range(PU):
                    g_accs[r][p] = SIMD[DType.float32, W](0)
                    u_accs[r][p] = SIMD[DType.float32, W](0)
            for i in range(hidden // STRIDE):
                comptime for p in range(PU):
                    var off = i * STRIDE + p * BW
                    var x_v = InlineArray[
                        SIMD[DType.bfloat16, BW], MR_p,
                    ](uninitialized=True)
                    comptime for r in range(MR_p):
                        x_v[r] = (x_rows[r] + off).load[width=BW]()
                    var g_w = (g_w_row + off).load[width=BW]()
                    comptime for r in range(MR_p):
                        g_accs[r][p] = bf16_pair_dot(
                            g_accs[r][p], x_v[r], g_w)
                    var u_w = (u_w_row + off).load[width=BW]()
                    comptime for r in range(MR_p):
                        u_accs[r][p] = bf16_pair_dot(
                            u_accs[r][p], x_v[r], u_w)
            comptime for r in range(MR_p):
                var sg = SIMD[DType.float32, W](0)
                var su = SIMD[DType.float32, W](0)
                comptime for p in range(PU):
                    sg += g_accs[r][p]
                    su += u_accs[r][p]
                gate_part[r * tile_j + j_off] = sg.reduce_add()
                up_part[r * tile_j + j_off] = su.reduce_add()
    else:
        comptime PU = PHASE1_PU_SPLIT
        comptime STRIDE = PU * BW
        comptime assert hidden % STRIDE == 0
        for j_off in range(tile_j):
            var w_row = gate_w_base + j_off * hidden
            var accs = InlineArray[
                InlineArray[SIMD[DType.float32, W], PU], MR_p,
            ](uninitialized=True)
            comptime for r in range(MR_p):
                comptime for p in range(PU):
                    accs[r][p] = SIMD[DType.float32, W](0)
            for i in range(hidden // STRIDE):
                comptime for p in range(PU):
                    var off = i * STRIDE + p * BW
                    var w_v = (w_row + off).load[width=BW]()
                    comptime for r in range(MR_p):
                        var x_v = (x_rows[r] + off).load[width=BW]()
                        accs[r][p] = bf16_pair_dot(accs[r][p], x_v, w_v)
            comptime for r in range(MR_p):
                var s = SIMD[DType.float32, W](0)
                comptime for p in range(PU):
                    s += accs[r][p]
                gate_part[r * tile_j + j_off] = s.reduce_add()
        for j_off in range(tile_j):
            var w_row = up_w_base + j_off * hidden
            var accs = InlineArray[
                InlineArray[SIMD[DType.float32, W], PU], MR_p,
            ](uninitialized=True)
            comptime for r in range(MR_p):
                comptime for p in range(PU):
                    accs[r][p] = SIMD[DType.float32, W](0)
            for i in range(hidden // STRIDE):
                comptime for p in range(PU):
                    var off = i * STRIDE + p * BW
                    var w_v = (w_row + off).load[width=BW]()
                    comptime for r in range(MR_p):
                        var x_v = (x_rows[r] + off).load[width=BW]()
                        accs[r][p] = bf16_pair_dot(accs[r][p], x_v, w_v)
            comptime for r in range(MR_p):
                var s = SIMD[DType.float32, W](0)
                comptime for p in range(PU):
                    s += accs[r][p]
                up_part[r * tile_j + j_off] = s.reduce_add()


@no_inline
def phase1_paired_mr4(
    x0: BF16Ptr, x1: BF16Ptr, x2: BF16Ptr, x3: BF16Ptr,
    gate_w_base: BF16Ptr, up_w_base: BF16Ptr,
    gate_part: F32Ptr, up_part: F32Ptr,
):
    var x_rows = InlineArray[BF16Ptr, MR](uninitialized=True)
    x_rows[0] = x0; x_rows[1] = x1; x_rows[2] = x2; x_rows[3] = x3
    phase1_paired_block[HIDDEN, TILE_J, MR](
        x_rows, gate_w_base, up_w_base, gate_part, up_part)


@no_inline
def run_case() -> Int:
    var x0 = alloc[BFloat16](HIDDEN)
    var x1 = alloc[BFloat16](HIDDEN)
    var x2 = alloc[BFloat16](HIDDEN)
    var x3 = alloc[BFloat16](HIDDEN)
    var gate_w = alloc[BFloat16](INTERMEDIATE * HIDDEN)
    var up_w = alloc[BFloat16](INTERMEDIATE * HIDDEN)
    var gate_part = alloc[Float32](MR * TILE_J)
    var up_part = alloc[Float32](MR * TILE_J)

    for i in range(HIDDEN):
        x0[i] = BFloat16(Float32(Float64(i % 127 - 63) * 0.01))
        x1[i] = BFloat16(Float32(Float64(i % 113 - 56) * 0.01))
        x2[i] = BFloat16(Float32(Float64(i % 101 - 50) * 0.01))
        x3[i] = BFloat16(Float32(Float64(i % 89 - 44) * 0.01))
    for i in range(INTERMEDIATE * HIDDEN):
        gate_w[i] = BFloat16(Float32(Float64(i % 97 - 48) * 0.01))
        up_w[i] = BFloat16(Float32(Float64(i % 83 - 41) * 0.01))

    phase1_paired_mr4(x0, x1, x2, x3, gate_w, up_w, gate_part, up_part)

    var checksum = Int(0)
    for i in range(MR * TILE_J):
        checksum += Int(gate_part[i]) + Int(up_part[i])

    x0.free(); x1.free(); x2.free(); x3.free()
    gate_w.free(); up_w.free()
    gate_part.free(); up_part.free()
    return checksum


def main():
    keep(run_case())
