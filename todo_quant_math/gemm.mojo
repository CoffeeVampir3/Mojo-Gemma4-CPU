from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from .types import F32Ptr, I8Ptr
from .vnni_dot import act_broadcast_vnni, vpdpbusd
from .vnni_layout import (
    VNNI_BLK, VNNI_K_STEP, VNNI_N_STEP, VNNI_TILE_N, compute_n_block,
)


@always_inline
def accumulate_n_step[
    width: Int, PR: Int, K: Int,
](
    act: I8Ptr,
    m_panel: Int,
    wpacked: I8Ptr,
    mut packed_off: Int,
    k_base: Int,
    k_len: Int,
    mut acc: InlineArray[SIMD[DType.int32, width], PR * (VNNI_N_STEP // width)],
):
    comptime passes = VNNI_TILE_N // width
    comptime bytes_per_pass = width * VNNI_BLK
    comptime acc_count = VNNI_N_STEP // width
    comptime dc_count = VNNI_K_STEP // VNNI_BLK
    comptime tile_dc_bytes = VNNI_TILE_N * VNNI_BLK
    comptime tile_ks_bytes = dc_count * tile_dc_bytes

    for ks in range(0, k_len, VNNI_K_STEP):
        for dc in range(dc_count):
            var k_pos = k_base + ks + dc * VNNI_BLK
            var ab = InlineArray[SIMD[DType.uint8, width * 4], PR](
                uninitialized=True)
            comptime for r in range(PR):
                ab[r] = act_broadcast_vnni[width](
                    act + (m_panel + r) * K, k_pos)
            var t0 = packed_off + dc * tile_dc_bytes
            var t1 = t0 + tile_ks_bytes
            comptime for p in range(passes):
                var w0 = (wpacked + t0 + p * bytes_per_pass).load[
                    width = width * 4, non_temporal=True]()
                comptime for r in range(PR):
                    acc[r * acc_count + p] = vpdpbusd[width](
                        acc[r * acc_count + p], ab[r], w0)
            comptime for p in range(passes):
                var w1 = (wpacked + t1 + p * bytes_per_pass).load[
                    width = width * 4, non_temporal=True]()
                comptime for r in range(PR):
                    acc[r * acc_count + passes + p] = vpdpbusd[width](
                        acc[r * acc_count + passes + p], ab[r], w1)
        packed_off += 2 * tile_ks_bytes


@always_inline
def gemm_i8_panel[
    N: Int, K: Int, fwht_block: Int, PR: Int, Out: DType,
](
    act: I8Ptr,
    m_panel: Int,
    wpacked: I8Ptr,
    block_scales: F32Ptr,
    wsc: F32Ptr,
    block_colsums: F32Ptr,
    dst: UnsafePointer[Scalar[Out], MutAnyOrigin],
    out_stride: Int,
    colsum_stride: Int,
    subrange: Int,
):
    """PR output rows; each weight tile is loaded once and reused across rows.
    Per-K-block dequant; fwht_block == K collapses to the per-row form."""
    comptime width = simd_width_of[DType.int32]()
    comptime num_blocks = K // fwht_block
    comptime acc_count = VNNI_N_STEP // width

    var inv127 = Float32(1.0) / Float32(127.0)
    var n_block = compute_n_block(subrange, K)
    var packed_off = 0

    for nb in range(0, subrange, n_block):
        var nb_size = min(n_block, subrange - nb)
        for ns in range(0, nb_size, VNNI_N_STEP):

            comptime if num_blocks == 1:
                var acc = InlineArray[
                    SIMD[DType.int32, width], PR * acc_count](
                    fill=SIMD[DType.int32, width](0))
                accumulate_n_step[width, PR, K](
                    act, m_panel, wpacked, packed_off, 0, K, acc)

                comptime for r in range(PR):
                    var sc = block_scales[m_panel + r] * inv127
                    comptime for a in range(acc_count):
                        var n_base = nb + ns + a * width
                        var cs = (block_colsums + n_base).load[width=width]()
                        var corrected = (
                            acc[r * acc_count + a].cast[DType.float32]()
                            - Float32(128) * cs)
                        var res = (
                            corrected * sc * (wsc + n_base).load[width=width]())
                        (dst + (m_panel + r) * out_stride + n_base).store(
                            res.cast[Out]())

            else:
                var f32_acc = InlineArray[
                    SIMD[DType.float32, width], PR * acc_count](
                    fill=SIMD[DType.float32, width](0))
                for blk in range(num_blocks):
                    var iacc = InlineArray[
                        SIMD[DType.int32, width], PR * acc_count](
                        fill=SIMD[DType.int32, width](0))
                    accumulate_n_step[width, PR, K](
                        act, m_panel, wpacked, packed_off,
                        blk * fwht_block, fwht_block, iacc)

                    comptime for r in range(PR):
                        var bd = block_scales[
                            (m_panel + r) * num_blocks + blk] * inv127
                        comptime for a in range(acc_count):
                            var n_base = nb + ns + a * width
                            var cs = (block_colsums
                                + blk * colsum_stride + n_base).load[
                                width=width]()
                            var corrected = (
                                iacc[r * acc_count + a].cast[DType.float32]()
                                - Float32(128) * cs)
                            f32_acc[r * acc_count + a] += corrected * bd

                comptime for r in range(PR):
                    comptime for a in range(acc_count):
                        var n_base = nb + ns + a * width
                        var res = (
                            f32_acc[r * acc_count + a]
                            * (wsc + n_base).load[width=width]())
                        (dst + (m_panel + r) * out_stride + n_base).store(
                            res.cast[Out]())


def gemm_i8[
    N: Int, K: Int, fwht_block: Int, MR: Int, Out: DType,
](
    act: I8Ptr,
    m: Int,
    wpacked: I8Ptr,
    block_scales: F32Ptr,
    wsc: F32Ptr,
    block_colsums: F32Ptr,
    dst: UnsafePointer[Scalar[Out], MutAnyOrigin],
    subrange: Int = N,
    colsum_stride: Int = N,
):
    """Stock int8 VNNI GEMM: [m, K] activations x VNNI-packed [N, K] weights.

    block_scales: [m, K/fwht_block] raw per-row per-K-block activation absmax.
    wsc:          [N] per-row weight scale (S_w / 127).
    block_colsums:[K/fwht_block, N] per-K-block weight colsums.
    dst:          [m, N] in Out dtype.

    fwht_block must be a multiple of VNNI_K_STEP (>= 64); fwht_block == K is
    the per-row form. Walks m in MR-row panels with a 1-row tail."""
    debug_assert(K % VNNI_K_STEP == 0, "gemm_i8: K must be a multiple of 64")
    debug_assert(fwht_block % VNNI_K_STEP == 0,
        "gemm_i8: fwht_block must be a multiple of VNNI_K_STEP (>= 64)")
    debug_assert(K % fwht_block == 0, "gemm_i8: fwht_block must divide K")
    debug_assert(subrange % VNNI_N_STEP == 0,
        "gemm_i8: subrange must be a multiple of VNNI_N_STEP")

    var m_panel = 0
    while m_panel + MR <= m:
        gemm_i8_panel[N, K, fwht_block, MR, Out](
            act, m_panel, wpacked, block_scales, wsc, block_colsums,
            dst, N, colsum_stride, subrange)
        m_panel += MR
    while m_panel < m:
        gemm_i8_panel[N, K, fwht_block, 1, Out](
            act, m_panel, wpacked, block_scales, wsc, block_colsums,
            dst, N, colsum_stride, subrange)
        m_panel += 1
