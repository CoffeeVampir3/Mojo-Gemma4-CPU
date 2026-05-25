from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys import llvm_intrinsic
from std.sys.info import simd_width_of
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


comptime VNNI_N_STEP = 32
comptime VNNI_K_STEP = 64
comptime VNNI_TILE_N = 16
comptime VNNI_BLK = 4


def gemv_row_small[N: Int, K: Int](
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    act_sc: Float32,
    wsc: UnsafePointer[Float32, MutAnyOrigin],
    wcs: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
):
    """A trimmed gemv_row for a single n-block, exercises the accumulator bank."""
    comptime width = simd_width_of[DType.int32]()
    comptime passes_per_subtile = VNNI_TILE_N // width
    comptime bytes_per_pass = width * VNNI_BLK
    comptime acc_count = VNNI_N_STEP // width
    comptime dc_count = VNNI_K_STEP // VNNI_BLK
    comptime tile_dc_bytes = VNNI_TILE_N * VNNI_BLK
    comptime tile_ks_bytes = dc_count * tile_dc_bytes

    var acc_buf = InlineArray[SIMD[DType.int32, width], acc_count](
        fill=SIMD[DType.int32, width](0))
    var acc = UnsafePointer(to=acc_buf).bitcast[
        SIMD[DType.int32, width]]()
    var packed_off = 0

    for ks in range(0, K, VNNI_K_STEP):
        for dc in range(dc_count):
            var k_pos = ks + dc * VNNI_BLK
            var act_bytes = act_broadcast_vnni[width](act_row, k_pos)
            var t0 = packed_off + dc * tile_dc_bytes
            var t1 = t0 + tile_ks_bytes
            comptime for p in range(passes_per_subtile):
                var off = t0 + p * bytes_per_pass
                acc[p] = dot_vnni_broadcasted[width](
                    acc[p], act_bytes, wpacked + off)
            comptime for p in range(passes_per_subtile):
                var off = t1 + p * bytes_per_pass
                acc[passes_per_subtile + p] = (
                    dot_vnni_broadcasted[width](
                        acc[passes_per_subtile + p], act_bytes,
                        wpacked + off,
                    )
                )
        packed_off += 2 * tile_ks_bytes

    comptime for a in range(acc_count):
        comptime a_off = a * width
        var corrected = (
            acc[a].cast[DType.float32]()
            - Float32(128) * (wcs + a_off).load[width=width]()
        )
        var result = corrected * act_sc * (
            wsc + a_off).load[width=width]()
        (dst + a_off).store(result)


def call_gemv(
    act_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    wpacked: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    act_sc: Float32,
    wsc: UnsafePointer[Float32, MutAnyOrigin],
    wcs: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
):
    gemv_row_small[32, 1024](act_row, wpacked, act_sc, wsc, wcs, dst)


def main():
    var act = UnsafePointer[Scalar[DType.int8], MutAnyOrigin].unsafe_dangling()
    var w = UnsafePointer[Scalar[DType.int8], MutAnyOrigin].unsafe_dangling()
    var wsc = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()
    var wcs = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()
    var dst = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()
    call_gemv(act, w, Float32(1.0), wsc, wcs, dst)
    keep(act)
    keep(w)
    keep(wsc)
    keep(wcs)
    keep(dst)
