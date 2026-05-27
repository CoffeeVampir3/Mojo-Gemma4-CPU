from std.memory import UnsafePointer
from std.sys import llvm_intrinsic
from std.sys.info import CompilationTarget


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
def dot_loaded[width: Int](
    acc: SIMD[DType.int32, width],
    act_bytes: SIMD[DType.uint8, width * 4],
    weights: SIMD[DType.int8, width * 4],
) -> SIMD[DType.int32, width]:
    comptime if CompilationTarget.has_vnni():
        return vpdpbusd[width](acc, act_bytes, weights)
    else:
        var result = acc
        comptime for lane in range(width):
            comptime off = lane * 4
            result[lane] += (
                Int32(act_bytes[off]) * Int32(weights[off])
                + Int32(act_bytes[off + 1]) * Int32(weights[off + 1])
                + Int32(act_bytes[off + 2]) * Int32(weights[off + 2])
                + Int32(act_bytes[off + 3]) * Int32(weights[off + 3])
            )
        return result


@always_inline
def dot_broadcasted[width: Int](
    acc: SIMD[DType.int32, width],
    act_bytes: SIMD[DType.uint8, width * 4],
    wpacked: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
) -> SIMD[DType.int32, width]:
    return dot_loaded[width](
        acc, act_bytes, wpacked.load[width=width * 4, non_temporal=True]())
