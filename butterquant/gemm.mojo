from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from butterquant.dot_products import act_broadcast_vnni, dot_loaded
from butterquant.types import F32Ptr, I8Ptr
from butterquant.vnni import VNNI_BLK, VNNI_K_STEP, VNNI_N_STEP, VNNI_TILE_N


@always_inline
def accumulate_tiles[
    width: Int, PR: Int, row_ptr: def(Int) capturing [_] -> I8Ptr,
](
    wpacked: I8Ptr,
    packed_base: Int,
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

    var packed_off = packed_base
    for ks in range(0, k_len, VNNI_K_STEP):
        for dc in range(dc_count):
            var k_pos = k_base + ks + dc * VNNI_BLK
            var ab = InlineArray[SIMD[DType.uint8, width * 4], PR](
                uninitialized=True)
            comptime for r in range(PR):
                ab[r] = act_broadcast_vnni[width](row_ptr(r), k_pos)
            var t0 = packed_off + dc * tile_dc_bytes
            var t1 = t0 + tile_ks_bytes
            comptime for p in range(passes):
                var w0 = (wpacked + t0 + p * bytes_per_pass).load[
                    width = width * 4, non_temporal=True]()
                comptime for r in range(PR):
                    acc[r * acc_count + p] = dot_loaded[width](
                        acc[r * acc_count + p], ab[r], w0)
            comptime for p in range(passes):
                var w1 = (wpacked + t1 + p * bytes_per_pass).load[
                    width = width * 4, non_temporal=True]()
                comptime for r in range(PR):
                    acc[r * acc_count + passes + p] = dot_loaded[width](
                        acc[r * acc_count + passes + p], ab[r], w1)
        packed_off += 2 * tile_ks_bytes


@always_inline
def accumulate_n_step[width: Int, PR: Int, K: Int](
    act: I8Ptr,
    m_panel: Int,
    wpacked: I8Ptr,
    packed_base: Int,
    k_base: Int,
    k_len: Int,
    mut acc: InlineArray[SIMD[DType.int32, width], PR * (VNNI_N_STEP // width)],
):
    @parameter
    def row_ptr(r: Int) -> I8Ptr:
        return act + (m_panel + r) * K

    accumulate_tiles[width, PR, row_ptr](
        wpacked, packed_base, k_base, k_len, acc)


@always_inline
def accumulate_n_step_gathered[width: Int, PR: Int](
    rows: InlineArray[I8Ptr, PR],
    wpacked: I8Ptr,
    packed_base: Int,
    k_base: Int,
    k_len: Int,
    mut acc: InlineArray[SIMD[DType.int32, width], PR * (VNNI_N_STEP // width)],
):
    @parameter
    def row_ptr(r: Int) -> I8Ptr:
        return rows[r]

    accumulate_tiles[width, PR, row_ptr](
        wpacked, packed_base, k_base, k_len, acc)


@always_inline
def gemm_i8_per_row_panel[N: Int, K: Int, PR: Int, Out: DType](
    act: I8Ptr,
    m_panel: Int,
    act_scale: F32Ptr,
    wpacked: I8Ptr,
    wsc: F32Ptr,
    colsum: F32Ptr,
    dst: UnsafePointer[Scalar[Out], MutAnyOrigin],
    ns: Int,
):
    comptime width = simd_width_of[DType.int32]()
    comptime acc_count = VNNI_N_STEP // width
    var iacc = InlineArray[SIMD[DType.int32, width], PR * acc_count](
        fill=SIMD[DType.int32, width](0))
    accumulate_n_step[width, PR, K](
        act, m_panel, wpacked, ns * K, 0, K, iacc)

    var inv127 = Float32(1.0) / Float32(127.0)
    comptime for r in range(PR):
        var ad = act_scale[m_panel + r] * inv127
        comptime for a in range(acc_count):
            var n_base = ns + a * width
            var cs = (colsum + n_base).load[width=width]()
            var corrected = (
                iacc[r * acc_count + a].cast[DType.float32]()
                - Float32(128) * cs)
            var res = corrected * ad * (wsc + n_base).load[width=width]()
            (dst + (m_panel + r) * N + n_base).store(res.cast[Out]())


def gemm_i8_per_row[N: Int, K: Int, MR: Int, Out: DType](
    act: I8Ptr,
    m: Int,
    act_scale: F32Ptr,
    wpacked: I8Ptr,
    wsc: F32Ptr,
    colsum: F32Ptr,
    dst: UnsafePointer[Scalar[Out], MutAnyOrigin],
    start_tile: Int,
    end_tile: Int,
):
    var m_panel = 0
    while m_panel + MR <= m:
        for t in range(start_tile, end_tile):
            gemm_i8_per_row_panel[N, K, MR, Out](
                act, m_panel, act_scale, wpacked, wsc, colsum, dst,
                t * VNNI_N_STEP)
        m_panel += MR
    while m_panel < m:
        for t in range(start_tile, end_tile):
            gemm_i8_per_row_panel[N, K, 1, Out](
                act, m_panel, act_scale, wpacked, wsc, colsum, dst,
                t * VNNI_N_STEP)
        m_panel += 1


@always_inline
def gemm_i8_per_block_panel[N: Int, K: Int, block: Int, PR: Int, Out: DType](
    act: I8Ptr,
    m_panel: Int,
    act_scale: F32Ptr,
    wpacked: I8Ptr,
    wsc: F32Ptr,
    colsum: F32Ptr,
    dst: UnsafePointer[Scalar[Out], MutAnyOrigin],
    ns: Int,
):
    comptime width = simd_width_of[DType.int32]()
    comptime acc_count = VNNI_N_STEP // width
    comptime nb = K // block
    comptime inv127 = Float32(1.0) / Float32(127.0)
    comptime blk_bytes = block * VNNI_N_STEP

    var facc = InlineArray[SIMD[DType.float32, width], PR * acc_count](
        fill=SIMD[DType.float32, width](0))

    for b in range(nb):
        var iacc = InlineArray[SIMD[DType.int32, width], PR * acc_count](
            fill=SIMD[DType.int32, width](0))
        accumulate_n_step[width, PR, K](
            act, m_panel, wpacked, ns * K + b * blk_bytes, b * block, block,
            iacc)
        comptime for r in range(PR):
            var adv = SIMD[DType.float32, width](
                act_scale[(m_panel + r) * nb + b] * inv127)
            comptime for a in range(acc_count):
                var n_base = ns + a * width
                var cs = (colsum + b * N + n_base).load[width=width]()
                var corrected = (
                    iacc[r * acc_count + a].cast[DType.float32]()
                    - Float32(128) * cs)
                facc[r * acc_count + a] = corrected.fma(
                    adv, facc[r * acc_count + a])

    comptime for r in range(PR):
        comptime for a in range(acc_count):
            var n_base = ns + a * width
            var res = facc[r * acc_count + a] * (wsc + n_base).load[width=width]()
            (dst + (m_panel + r) * N + n_base).store(res.cast[Out]())


def gemm_i8_per_block[N: Int, K: Int, block: Int, MR: Int, Out: DType](
    act: I8Ptr,
    m: Int,
    act_scale: F32Ptr,
    wpacked: I8Ptr,
    wsc: F32Ptr,
    colsum: F32Ptr,
    dst: UnsafePointer[Scalar[Out], MutAnyOrigin],
    start_tile: Int,
    end_tile: Int,
):
    var m_panel = 0
    while m_panel + MR <= m:
        for t in range(start_tile, end_tile):
            gemm_i8_per_block_panel[N, K, block, MR, Out](
                act, m_panel, act_scale, wpacked, wsc, colsum, dst,
                t * VNNI_N_STEP)
        m_panel += MR
    while m_panel < m:
        for t in range(start_tile, end_tile):
            gemm_i8_per_block_panel[N, K, block, 1, Out](
                act, m_panel, act_scale, wpacked, wsc, colsum, dst,
                t * VNNI_N_STEP)
        m_panel += 1
