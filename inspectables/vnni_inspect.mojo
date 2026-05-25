from std.memory import UnsafePointer
from std.sys import llvm_intrinsic
from std.sys.info import CompilationTarget, simd_width_of
from std.benchmark import keep


@always_inline
def vpdpbusd[width: Int](
    acc: SIMD[DType.int32, width],
    a: SIMD[DType.uint8, width * 4],
    b: SIMD[DType.int8, width * 4],
) -> SIMD[DType.int32, width]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd." + String(width * 32),
        SIMD[DType.int32, width],
    ](acc, a, b)


@always_inline
def bcast_4_vnni[width: Int](
    b4: SIMD[DType.uint8, 4],
) -> SIMD[DType.uint8, width * 4]:
    var out = SIMD[DType.uint8, width * 4]()
    comptime for lane in range(width):
        out = out.insert[offset=lane * 4](b4)
    return out


@always_inline
def act_broadcast_vnni[width: Int](
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.uint8, width * 4]:
    return bcast_4_vnni[width](
        (act_row + k_pos).bitcast[UInt8]().load[width=4]()
        ^ SIMD[DType.uint8, 4](0x80))


@always_inline
def dot_vnni_broadcasted[width: Int](
    acc: SIMD[DType.int32, width],
    act_bytes: SIMD[DType.uint8, width * 4],
    wpacked: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
) -> SIMD[DType.int32, width]:
    var w = wpacked.load[width=width * 4, non_temporal=True]()
    return vpdpbusd[width](acc, act_bytes, w)


def vnni_inner_512(
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    dst: UnsafePointer[Int32, MutAnyOrigin],
):
    """512-byte K stripe, 16-wide accumulator: 8 dot ops with vpdpbusd."""
    comptime width = 16  # AVX-512 i32 width
    var acc = SIMD[DType.int32, width](0)
    comptime for i in range(8):
        var act = act_broadcast_vnni[width](act_row, i * 4)
        acc = dot_vnni_broadcasted[width](
            acc, act, wpacked + i * (width * 4))
    (dst).store(acc)


def vnni_inner_256(
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    dst: UnsafePointer[Int32, MutAnyOrigin],
):
    """AVX-VNNI 8-wide accumulator: 8 dot ops."""
    comptime width = 8
    var acc = SIMD[DType.int32, width](0)
    comptime for i in range(8):
        var act = act_broadcast_vnni[width](act_row, i * 4)
        acc = dot_vnni_broadcasted[width](
            acc, act, wpacked + i * (width * 4))
    (dst).store(acc)


def main():
    var act = UnsafePointer[Scalar[DType.int8], MutAnyOrigin].unsafe_dangling()
    var w = UnsafePointer[Scalar[DType.int8], MutAnyOrigin].unsafe_dangling()
    var dst = UnsafePointer[Int32, MutAnyOrigin].unsafe_dangling()
    @parameter
    if simd_width_of[DType.int32]() == 16:
        vnni_inner_512(act, w, dst)
    vnni_inner_256(act, w, dst)
    keep(act)
    keep(w)
    keep(dst)
