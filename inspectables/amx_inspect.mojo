from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys import llvm_intrinsic
from std.sys.info import simd_width_of
from std.benchmark import keep


@always_inline
def tilezero[tile: Int]():
    llvm_intrinsic["llvm.x86.tilezero", NoneType](Int8(tile))


@always_inline
def tileload[tile: Int, dtype: DType](
    ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    stride: Int,
):
    llvm_intrinsic["llvm.x86.tileloadd64", NoneType](
        Int8(tile), ptr, Int64(stride))


@always_inline
def tilestore[tile: Int, dtype: DType](
    ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    stride: Int,
):
    llvm_intrinsic["llvm.x86.tilestored64", NoneType](
        Int8(tile), ptr, Int64(stride))


@always_inline
def tdpbssd[dst: Int, src_a: Int, src_b: Int]():
    llvm_intrinsic["llvm.x86.tdpbssd", NoneType](
        Int8(dst), Int8(src_a), Int8(src_b))


comptime TILE_M = 16
comptime TILE_K = 64
comptime TILE_N = 16
comptime AMX_K_STEP = TILE_K
comptime AMX_M_STEP = TILE_M * 2


def amx_w1w3_k1024(
    act_tile: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    w1_packed: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    w3_packed: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    n_off: Int,
    K: Int,
    c_out: UnsafePointer[Int32, MutAnyOrigin],
):
    """Mimics amx_fused_w1w3_tile16 inner K-loop pattern at K=1024."""
    var K_const = 1024
    tilezero[4]()
    tilezero[5]()
    tilezero[6]()
    tilezero[7]()
    for k_off in range(0, K_const, AMX_K_STEP):
        tileload[0, DType.int8](act_tile + k_off, K_const)
        tileload[1, DType.int8](act_tile + TILE_M * K_const + k_off, K_const)
        var b_off = n_off + k_off * 16
        tileload[2, DType.int8](w1_packed + b_off, AMX_K_STEP)
        tileload[3, DType.int8](w3_packed + b_off, AMX_K_STEP)
        tdpbssd[4, 0, 2]()
        tdpbssd[5, 1, 2]()
        tdpbssd[6, 0, 3]()
        tdpbssd[7, 1, 3]()
    var c_stride = TILE_N * 4
    tilestore[4, DType.int32](c_out, c_stride)
    tilestore[5, DType.int32](c_out + TILE_M * TILE_N, c_stride)
    tilestore[6, DType.int32](c_out + 2 * TILE_M * TILE_N, c_stride)
    tilestore[7, DType.int32](c_out + 3 * TILE_M * TILE_N, c_stride)


def main():
    var a = UnsafePointer[Scalar[DType.int8], MutAnyOrigin].unsafe_dangling()
    var w1 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin].unsafe_dangling()
    var w3 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin].unsafe_dangling()
    var c = UnsafePointer[Int32, MutAnyOrigin].unsafe_dangling()
    amx_w1w3_k1024(a, w1, w3, 0, 1024, c)
    keep(a)
    keep(w1)
    keep(w3)
    keep(c)
