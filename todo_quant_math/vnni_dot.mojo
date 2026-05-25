from std.memory import UnsafePointer
from std.sys import llvm_intrinsic
from std.sys.info import CompilationTarget, simd_width_of

from .vnni_layout import VNNI_BLK, VNNI_TILE_N


@always_inline
def vpdpbusd[width: Int](
    acc: SIMD[DType.int32, width],
    a: SIMD[DType.uint8, width * 4],
    b: SIMD[DType.int8, width * 4],
) -> SIMD[DType.int32, width]:
    """x86 VNNI u8*i8 dot product accumulated into i32 lanes."""
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd." + String(width * 32),
        SIMD[DType.int32, width],
    ](acc, a, b)


@always_inline
def bcast_4_vnni[width: Int](
    b4: SIMD[DType.uint8, 4],
) -> SIMD[DType.uint8, width * 4]:
    """Broadcast four bytes to the dword shape expected by vpdpbusd."""
    var out = SIMD[DType.uint8, width * 4]()
    comptime for lane in range(width):
        out = out.insert[offset=lane * 4](b4)
    return out


@always_inline
def bcast_4u8_vnni[width: Int](
    p: UnsafePointer[UInt8, MutAnyOrigin],
) -> SIMD[DType.uint8, width * 4]:
    return bcast_4_vnni[width](p.load[width=4]())


@always_inline
def act_broadcast_vnni[width: Int](
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.uint8, width * 4]:
    """Load four i8 activation bytes, xor-bias to u8, and broadcast."""
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


@always_inline
def dot_vnni[width: Int](
    acc: SIMD[DType.int32, width],
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.int32, width]:
    return dot_vnni_broadcasted[width](
        acc, act_broadcast_vnni[width](act_row, k_pos), wpacked)


@always_inline
def dot_simd[width: Int](
    acc: SIMD[DType.int32, width],
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.int32, width]:
    """Fallback for one VNNI dword: explicit u8/i8 widening and multiply."""
    var wdw = wpacked.bitcast[Scalar[DType.int32]]().load[
        width=width, non_temporal=True]()
    var result = acc
    result += (
        SIMD[DType.int32, width](Int32(act_row[k_pos]) + 128)
        * ((wdw << 24) >> 24)
    )
    result += (
        SIMD[DType.int32, width](Int32(act_row[k_pos + 1]) + 128)
        * ((wdw << 16) >> 24)
    )
    result += (
        SIMD[DType.int32, width](Int32(act_row[k_pos + 2]) + 128)
        * ((wdw << 8) >> 24)
    )
    result += (
        SIMD[DType.int32, width](Int32(act_row[k_pos + 3]) + 128)
        * (wdw >> 24)
    )
    return result


@always_inline
def dot[width: Int](
    acc: SIMD[DType.int32, width],
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.int32, width]:
    comptime if CompilationTarget.has_vnni():
        return dot_vnni[width](acc, act_row, wpacked, k_pos)
    else:
        return dot_simd[width](acc, act_row, wpacked, k_pos)


def gemv_tile_width[T: DType, tile: Int]() -> Int:
    comptime hw = simd_width_of[T]()
    comptime if tile <= hw:
        return tile
    else:
        return hw


@always_inline
def dot_tile_chunked[width: Int](
    acc: SIMD[DType.int32, VNNI_TILE_N],
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    k_pos: Int,
) -> SIMD[DType.int32, VNNI_TILE_N]:
    """VNNI_TILE_N-wide dot split across the hardware SIMD width."""
    comptime assert VNNI_TILE_N % width == 0,
        "dot_tile_chunked requires width to divide VNNI_TILE_N"
    comptime regs = VNNI_TILE_N // width
    var result = acc
    comptime for r in range(regs):
        comptime lane_off = r * width
        result = result.insert[offset=lane_off](
            dot[width](
                result.slice[width, offset=lane_off](),
                act_row,
                wpacked + lane_off * VNNI_BLK,
                k_pos,
            ))
    return result
