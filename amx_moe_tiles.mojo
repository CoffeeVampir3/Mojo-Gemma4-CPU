"""AMX tile math atoms used by MiniMax-M2.7 sparse MoE phases."""

from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from .amx_tiles import (
    TILE_BYTES as AMX_TILE_BYTES,
    TILE_M as AMX_TILE_M,
    TILE_N as AMX_TILE_N,
    K_STEP as AMX_K_STEP,
    tileload,
    tilestore,
    tilezero,
    tdpbssd,
)
from .types import F32Ptr, I8Ptr
from .vnni_layout import VNNI_N_STEP, vnni_amx_half_offset


comptime AMX_M_STEP = AMX_TILE_M * 2


@always_inline
def copy_i8_row[K: Int](src: I8Ptr, dst: I8Ptr):
    comptime width = simd_width_of[DType.int8]()
    var k = 0
    while k + width <= K:
        (dst + k).store((src + k).load[width=width]())
        k += width
    while k < K:
        dst[k] = src[k]
        k += 1


@always_inline
def zero_i8_row[K: Int](dst: I8Ptr):
    comptime width = simd_width_of[DType.int8]()
    var z = SIMD[DType.int8, width](0)
    var k = 0
    while k + width <= K:
        (dst + k).store(z)
        k += width
    while k < K:
        dst[k] = Scalar[DType.int8](0)
        k += 1


@always_inline
def amx_fused_w1w3_tile16[K: Int, out_stride: Int](
    act_tile: I8Ptr,
    w1_packed: I8Ptr,
    w3_packed: I8Ptr,
    act_dequant: F32Ptr,
    w1_scale: F32Ptr,
    w3_scale: F32Ptr,
    n_off: Int,
    gate: F32Ptr,
    up: F32Ptr,
):
    """AMX Mx16 tile for fused W1/W3 projection."""
    comptime assert K % AMX_K_STEP == 0, "K must be AMX K-step aligned"
    comptime width = simd_width_of[DType.float32]()
    comptime c_stride = AMX_TILE_N * 4

    var c_arr = InlineArray[Int32, 4 * AMX_TILE_M * AMX_TILE_N](
        fill=Int32(0))
    var c = UnsafePointer(to=c_arr).bitcast[Int32]()
    var w1_c0 = c
    var w1_c1 = c + AMX_TILE_M * AMX_TILE_N
    var w3_c0 = c + 2 * AMX_TILE_M * AMX_TILE_N
    var w3_c1 = c + 3 * AMX_TILE_M * AMX_TILE_N

    tilezero[4]()
    tilezero[5]()
    tilezero[6]()
    tilezero[7]()

    for k_off in range(0, K, AMX_K_STEP):
        tileload[0, DType.int8](act_tile + k_off, K)
        tileload[1, DType.int8](act_tile + AMX_TILE_M * K + k_off, K)
        var b_off = vnni_amx_half_offset(n_off, k_off, K)
        tileload[2, DType.int8](w1_packed + b_off, AMX_K_STEP)
        tileload[3, DType.int8](w3_packed + b_off, AMX_K_STEP)
        tdpbssd[4, 0, 2]()
        tdpbssd[5, 1, 2]()
        tdpbssd[6, 0, 3]()
        tdpbssd[7, 1, 3]()

    tilestore[4, DType.int32](w1_c0, c_stride)
    tilestore[5, DType.int32](w1_c1, c_stride)
    tilestore[6, DType.int32](w3_c0, c_stride)
    tilestore[7, DType.int32](w3_c1, c_stride)

    var n = 0
    while n + width <= AMX_TILE_N:
        var w1s = (w1_scale + n_off + n).load[width=width]()
        var w3s = (w3_scale + n_off + n).load[width=width]()
        for m in range(AMX_M_STEP):
            var row = m
            var w1_src = w1_c0 + row * AMX_TILE_N
            var w3_src = w3_c0 + row * AMX_TILE_N
            if m >= AMX_TILE_M:
                row = m - AMX_TILE_M
                w1_src = w1_c1 + row * AMX_TILE_N
                w3_src = w3_c1 + row * AMX_TILE_N
            var dq = SIMD[DType.float32, width](act_dequant[m])
            (gate + m * out_stride + n).store(
                (w1_src + n).load[width=width]().cast[DType.float32]()
                * dq
                * w1s)
            (up + m * out_stride + n).store(
                (w3_src + n).load[width=width]().cast[DType.float32]()
                * dq
                * w3s)
        n += width


@always_inline
def amx_down_tile32[K: Int, fwht_blk: Int](
    act_tile: I8Ptr,
    block_scales: F32Ptr,
    down_packed: I8Ptr,
    down_scale: F32Ptr,
    n_off: Int,
    out_tile: F32Ptr,
):
    """AMX Mx32 blocked down-projection tile."""
    comptime assert K % AMX_K_STEP == 0, "K must be AMX K-step aligned"
    comptime assert K % fwht_blk == 0, "K must be divisible by fwht block"
    comptime assert fwht_blk % AMX_K_STEP == 0,
        "fwht block must be AMX K-step aligned"
    debug_assert(n_off >= 0 and n_off % VNNI_N_STEP == 0,
        "amx_down_tile32: n_off must be 32-aligned")
    comptime width = simd_width_of[DType.float32]()
    comptime num_blocks = K // fwht_blk
    comptime k_steps_per_block = fwht_blk // AMX_K_STEP
    comptime c_stride = AMX_TILE_N * 4

    var c_arr = InlineArray[Int32, 4 * AMX_TILE_M * AMX_TILE_N](
        fill=Int32(0))
    var c = UnsafePointer(to=c_arr).bitcast[Int32]()
    var c0_lo = c
    var c1_lo = c + AMX_TILE_M * AMX_TILE_N
    var c0_hi = c + 2 * AMX_TILE_M * AMX_TILE_N
    var c1_hi = c + 3 * AMX_TILE_M * AMX_TILE_N

    for i in range(AMX_M_STEP * VNNI_N_STEP):
        out_tile[i] = Float32(0)

    for blk in range(num_blocks):
        tilezero[4]()
        tilezero[5]()
        tilezero[6]()
        tilezero[7]()
        var k_base = blk * fwht_blk
        for ks in range(k_steps_per_block):
            var k_off = k_base + ks * AMX_K_STEP
            tileload[0, DType.int8](act_tile + k_off, K)
            tileload[1, DType.int8](
                act_tile + AMX_TILE_M * K + k_off, K)
            var b_lo = vnni_amx_half_offset(n_off, k_off, K)
            var b_hi = vnni_amx_half_offset(n_off + AMX_TILE_N, k_off, K)
            tileload[2, DType.int8](down_packed + b_lo, AMX_K_STEP)
            tileload[3, DType.int8](down_packed + b_hi, AMX_K_STEP)
            tdpbssd[4, 0, 2]()
            tdpbssd[5, 1, 2]()
            tdpbssd[6, 0, 3]()
            tdpbssd[7, 1, 3]()

        tilestore[4, DType.int32](c0_lo, c_stride)
        tilestore[5, DType.int32](c1_lo, c_stride)
        tilestore[6, DType.int32](c0_hi, c_stride)
        tilestore[7, DType.int32](c1_hi, c_stride)

        var n = 0
        while n + width <= AMX_TILE_N:
            for m in range(AMX_M_STEP):
                var row = m
                var lo_src = c0_lo + row * AMX_TILE_N
                var hi_src = c0_hi + row * AMX_TILE_N
                if m >= AMX_TILE_M:
                    row = m - AMX_TILE_M
                    lo_src = c1_lo + row * AMX_TILE_N
                    hi_src = c1_hi + row * AMX_TILE_N
                var dq = SIMD[DType.float32, width](
                    block_scales[m * num_blocks + blk] / Float32(127))
                var out_lo = out_tile + m * VNNI_N_STEP + n
                out_lo.store(
                    out_lo.load[width=width]()
                    + (lo_src + n).load[width=width]().cast[DType.float32]()
                    * dq)
                var out_hi = out_tile + m * VNNI_N_STEP + AMX_TILE_N + n
                out_hi.store(
                    out_hi.load[width=width]()
                    + (hi_src + n).load[width=width]().cast[DType.float32]()
                    * dq)
            n += width

    var n = 0
    while n + width <= AMX_TILE_N:
        var ws_lo = (down_scale + n_off + n).load[width=width]()
        var ws_hi = (down_scale + n_off + AMX_TILE_N + n).load[width=width]()
        for m in range(AMX_M_STEP):
            var out_lo = out_tile + m * VNNI_N_STEP + n
            out_lo.store(out_lo.load[width=width]() * ws_lo)
            var out_hi = out_tile + m * VNNI_N_STEP + AMX_TILE_N + n
            out_hi.store(out_hi.load[width=width]() * ws_hi)
        n += width
