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
comptime PU = 4
comptime PU_FUSED = 2

comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]


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


@no_inline
def phase1_split_mr_block(
    x_rows_0: BF16Ptr, x_rows_1: BF16Ptr, x_rows_2: BF16Ptr, x_rows_3: BF16Ptr,
    gate_w_base: BF16Ptr, up_w_base: BF16Ptr,
    gate_part: F32Ptr, up_part: F32Ptr,
):
    comptime STRIDE = PU * BW

    for j_off in range(TILE_J):
        var w_row = gate_w_base + j_off * HIDDEN
        var accs = InlineArray[
            InlineArray[SIMD[DType.float32, W], PU], MR,
        ](uninitialized=True)
        comptime for r in range(MR):
            comptime for p in range(PU):
                accs[r][p] = SIMD[DType.float32, W](0)
        for i in range(HIDDEN // STRIDE):
            comptime for p in range(PU):
                var off = i * STRIDE + p * BW
                var w_v = (w_row + off).load[width=BW]()
                var x_v0 = (x_rows_0 + off).load[width=BW]()
                var x_v1 = (x_rows_1 + off).load[width=BW]()
                var x_v2 = (x_rows_2 + off).load[width=BW]()
                var x_v3 = (x_rows_3 + off).load[width=BW]()
                accs[0][p] = bf16_pair_dot(accs[0][p], x_v0, w_v)
                accs[1][p] = bf16_pair_dot(accs[1][p], x_v1, w_v)
                accs[2][p] = bf16_pair_dot(accs[2][p], x_v2, w_v)
                accs[3][p] = bf16_pair_dot(accs[3][p], x_v3, w_v)
        comptime for r in range(MR):
            var s = SIMD[DType.float32, W](0)
            comptime for p in range(PU):
                s += accs[r][p]
            gate_part[r * TILE_J + j_off] = s.reduce_add()

    for j_off in range(TILE_J):
        var w_row = up_w_base + j_off * HIDDEN
        var accs = InlineArray[
            InlineArray[SIMD[DType.float32, W], PU], MR,
        ](uninitialized=True)
        comptime for r in range(MR):
            comptime for p in range(PU):
                accs[r][p] = SIMD[DType.float32, W](0)
        for i in range(HIDDEN // STRIDE):
            comptime for p in range(PU):
                var off = i * STRIDE + p * BW
                var w_v = (w_row + off).load[width=BW]()
                var x_v0 = (x_rows_0 + off).load[width=BW]()
                var x_v1 = (x_rows_1 + off).load[width=BW]()
                var x_v2 = (x_rows_2 + off).load[width=BW]()
                var x_v3 = (x_rows_3 + off).load[width=BW]()
                accs[0][p] = bf16_pair_dot(accs[0][p], x_v0, w_v)
                accs[1][p] = bf16_pair_dot(accs[1][p], x_v1, w_v)
                accs[2][p] = bf16_pair_dot(accs[2][p], x_v2, w_v)
                accs[3][p] = bf16_pair_dot(accs[3][p], x_v3, w_v)
        comptime for r in range(MR):
            var s = SIMD[DType.float32, W](0)
            comptime for p in range(PU):
                s += accs[r][p]
            up_part[r * TILE_J + j_off] = s.reduce_add()


@no_inline
def phase1_fused_mr_block(
    x_rows_0: BF16Ptr, x_rows_1: BF16Ptr, x_rows_2: BF16Ptr, x_rows_3: BF16Ptr,
    gate_w_base: BF16Ptr, up_w_base: BF16Ptr,
    gate_part: F32Ptr, up_part: F32Ptr,
):
    comptime STRIDE = PU_FUSED * BW

    for j_off in range(TILE_J):
        var g_w_row = gate_w_base + j_off * HIDDEN
        var u_w_row = up_w_base + j_off * HIDDEN
        var g_accs = InlineArray[
            InlineArray[SIMD[DType.float32, W], PU_FUSED], MR,
        ](uninitialized=True)
        var u_accs = InlineArray[
            InlineArray[SIMD[DType.float32, W], PU_FUSED], MR,
        ](uninitialized=True)
        comptime for r in range(MR):
            comptime for p in range(PU_FUSED):
                g_accs[r][p] = SIMD[DType.float32, W](0)
                u_accs[r][p] = SIMD[DType.float32, W](0)
        for i in range(HIDDEN // STRIDE):
            comptime for p in range(PU_FUSED):
                var off = i * STRIDE + p * BW
                var g_w = (g_w_row + off).load[width=BW]()
                var u_w = (u_w_row + off).load[width=BW]()
                var x_v0 = (x_rows_0 + off).load[width=BW]()
                var x_v1 = (x_rows_1 + off).load[width=BW]()
                var x_v2 = (x_rows_2 + off).load[width=BW]()
                var x_v3 = (x_rows_3 + off).load[width=BW]()
                g_accs[0][p] = bf16_pair_dot(g_accs[0][p], x_v0, g_w)
                g_accs[1][p] = bf16_pair_dot(g_accs[1][p], x_v1, g_w)
                g_accs[2][p] = bf16_pair_dot(g_accs[2][p], x_v2, g_w)
                g_accs[3][p] = bf16_pair_dot(g_accs[3][p], x_v3, g_w)
                u_accs[0][p] = bf16_pair_dot(u_accs[0][p], x_v0, u_w)
                u_accs[1][p] = bf16_pair_dot(u_accs[1][p], x_v1, u_w)
                u_accs[2][p] = bf16_pair_dot(u_accs[2][p], x_v2, u_w)
                u_accs[3][p] = bf16_pair_dot(u_accs[3][p], x_v3, u_w)
        comptime for r in range(MR):
            var sg = SIMD[DType.float32, W](0)
            var su = SIMD[DType.float32, W](0)
            comptime for p in range(PU_FUSED):
                sg += g_accs[r][p]
                su += u_accs[r][p]
            gate_part[r * TILE_J + j_off] = sg.reduce_add()
            up_part[r * TILE_J + j_off] = su.reduce_add()


@no_inline
def run_case() -> Int:
    var x0 = alloc[Scalar[DType.bfloat16]](HIDDEN)
    var x1 = alloc[Scalar[DType.bfloat16]](HIDDEN)
    var x2 = alloc[Scalar[DType.bfloat16]](HIDDEN)
    var x3 = alloc[Scalar[DType.bfloat16]](HIDDEN)
    var gate_w = alloc[Scalar[DType.bfloat16]](INTERMEDIATE * HIDDEN)
    var up_w = alloc[Scalar[DType.bfloat16]](INTERMEDIATE * HIDDEN)
    var gate_part = alloc[Scalar[DType.float32]](MR * TILE_J)
    var up_part = alloc[Scalar[DType.float32]](MR * TILE_J)

    for i in range(HIDDEN):
        x0[i] = Scalar[DType.bfloat16](Float32(Float64(i % 127 - 63) * 0.01))
        x1[i] = Scalar[DType.bfloat16](Float32(Float64(i % 113 - 56) * 0.01))
        x2[i] = Scalar[DType.bfloat16](Float32(Float64(i % 101 - 50) * 0.01))
        x3[i] = Scalar[DType.bfloat16](Float32(Float64(i % 89 - 44) * 0.01))
    for i in range(INTERMEDIATE * HIDDEN):
        gate_w[i] = Scalar[DType.bfloat16](Float32(Float64(i % 97 - 48) * 0.01))
        up_w[i] = Scalar[DType.bfloat16](Float32(Float64(i % 83 - 41) * 0.01))

    phase1_split_mr_block(x0, x1, x2, x3, gate_w, up_w, gate_part, up_part)
    var checksum_a = Int(0)
    for i in range(MR * TILE_J):
        checksum_a += Int(gate_part[i]) + Int(up_part[i])

    phase1_fused_mr_block(x0, x1, x2, x3, gate_w, up_w, gate_part, up_part)
    var checksum_b = Int(0)
    for i in range(MR * TILE_J):
        checksum_b += Int(gate_part[i]) + Int(up_part[i])

    x0.free(); x1.free(); x2.free(); x3.free()
    gate_w.free(); up_w.free()
    gate_part.free(); up_part.free()
    return checksum_a + checksum_b


def main():
    keep(run_case())
