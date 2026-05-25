from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from simd_math.matrixops import transpose_generic


comptime L2_TARGET = 256 * 1024
comptime VNNI_N_STEP = 32
comptime VNNI_K_STEP = 64
comptime VNNI_TILE_N = 16
comptime VNNI_BLK = 4
comptime COLSUM_NARROW_WIDTH = 16


@always_inline
def compute_n_block(n: Int, k: Int) -> Int:
    var max_n = L2_TARGET // k
    var n_block = (max_n // VNNI_N_STEP) * VNNI_N_STEP
    if n_block >= n:
        return n
    if n_block >= VNNI_N_STEP:
        return n_block
    return VNNI_N_STEP


@always_inline
def required_pack_scratch_bytes(rows: Int, cols: Int) -> Int:
    return compute_n_block(rows, cols) * cols


@always_inline
def vnni_amx_half_offset(n_off: Int, k_off: Int, K: Int) -> Int:
    """Offset of one 16-column AMX B tile inside the 32-column VNNI pack."""
    debug_assert(n_off >= 0 and n_off % VNNI_TILE_N == 0,
        "vnni_amx_half_offset: n_off must be 16-aligned")
    debug_assert(k_off >= 0 and k_off % VNNI_K_STEP == 0,
        "vnni_amx_half_offset: k_off must be VNNI_K_STEP-aligned")
    var group_base = (n_off // VNNI_N_STEP) * VNNI_N_STEP * K
    var half = (n_off % VNNI_N_STEP) // VNNI_TILE_N
    return group_base + k_off * VNNI_N_STEP + half * (VNNI_TILE_N * VNNI_K_STEP)


@always_inline
def pack_and_colsum_impl[simd_width: Int](
    src: UnsafePointer[UInt8, MutAnyOrigin],
    dst: UnsafePointer[UInt8, MutAnyOrigin],
    scratch: UnsafePointer[UInt8, MutAnyOrigin],
    rows: Int,
    cols: Int,
    block_cols: Int,
    colsum: UnsafePointer[Float32, MutAnyOrigin],
    colsum_row_major: Bool,
):
    var n_block = compute_n_block(rows, cols)
    var num_blocks = cols // block_cols
    var src_i32_stride = cols // VNNI_BLK
    var tile_scratch = InlineArray[SIMD[DType.int32, 16], 16](
        uninitialized=True)

    var src_i8 = src.bitcast[Int8]()
    var scratch_i8 = scratch.bitcast[Int8]()

    for n_block_begin in range(0, rows, n_block):
        var n_block_size = min(n_block, rows - n_block_begin)

        for row in range(n_block_size):
            var global_row = n_block_begin + row
            var src_row = src_i8 + global_row * cols
            var scratch_row = scratch_i8 + row * cols

            for block_idx in range(num_blocks):
                var acc = SIMD[DType.int32, simd_width](0)
                var base = block_idx * block_cols
                for k in range(0, block_cols, simd_width):
                    var v = (src_row + base + k).load[width=simd_width]()
                    (scratch_row + base + k).store(v)
                    acc += v.cast[DType.int32]()
                var sum = Float32(Int(acc.reduce_add()))
                if colsum_row_major:
                    colsum[global_row * num_blocks + block_idx] = sum
                else:
                    colsum[block_idx * rows + global_row] = sum

        for n_begin in range(0, n_block_size, VNNI_N_STEP):
            for k_begin in range(0, cols, VNNI_K_STEP):
                var tile_base = (
                    (n_block_begin + n_begin) * cols
                    + k_begin * VNNI_N_STEP
                )

                var src0 = (
                    scratch_i8 + n_begin * cols + k_begin).bitcast[Int32]()
                var dst0 = (dst + tile_base).bitcast[Int32]()
                transpose_generic[DType.int32, 16](
                    src0, src_i32_stride, dst0, VNNI_TILE_N, tile_scratch)

                var src1 = (
                    scratch_i8 + (n_begin + VNNI_TILE_N) * cols + k_begin
                ).bitcast[Int32]()
                var dst1 = (
                    dst + tile_base + VNNI_TILE_N * VNNI_K_STEP
                ).bitcast[Int32]()
                transpose_generic[DType.int32, 16](
                    src1, src_i32_stride, dst1, VNNI_TILE_N, tile_scratch)


def pack_and_colsum_vnni(
    src: UnsafePointer[UInt8, MutAnyOrigin],
    dst: UnsafePointer[UInt8, MutAnyOrigin],
    scratch: UnsafePointer[UInt8, MutAnyOrigin],
    rows: Int,
    cols: Int,
    block_cols: Int,
    colsum: UnsafePointer[Float32, MutAnyOrigin],
    colsum_row_major: Bool,
):
    """Pack row-major i8 weights into the VNNI tile layout and emit colsums."""
    debug_assert(cols % VNNI_K_STEP == 0,
        "pack_and_colsum_vnni: cols must be a multiple of 64")
    debug_assert(rows % VNNI_N_STEP == 0,
        "pack_and_colsum_vnni: rows must be a multiple of 32")
    debug_assert(block_cols > 0 and cols % block_cols == 0,
        "pack_and_colsum_vnni: block_cols must divide cols")
    debug_assert(block_cols >= COLSUM_NARROW_WIDTH
        and block_cols % COLSUM_NARROW_WIDTH == 0,
        "pack_and_colsum_vnni: block_cols must be a multiple of 16")

    comptime native_i8 = simd_width_of[DType.int8]()
    comptime if native_i8 > COLSUM_NARROW_WIDTH:
        if block_cols % native_i8 == 0:
            pack_and_colsum_impl[native_i8](
                src, dst, scratch, rows, cols, block_cols,
                colsum, colsum_row_major)
        else:
            pack_and_colsum_impl[COLSUM_NARROW_WIDTH](
                src, dst, scratch, rows, cols, block_cols,
                colsum, colsum_row_major)
    else:
        pack_and_colsum_impl[native_i8](
            src, dst, scratch, rows, cols, block_cols,
            colsum, colsum_row_major)
