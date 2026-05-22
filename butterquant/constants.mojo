from std.sys.info import simd_width_of


comptime SIMD_F32_WIDTH = simd_width_of[DType.float32]()

comptime DEFAULT_PANEL_ROWS = 2048
comptime DEFAULT_COPY_CHUNK_BYTES = 16 * 1024 * 1024


def is_supported_fwht_block(block: Int) -> Bool:
    return (
        block == 512 or block == 256 or block == 128
        or block == 64 or block == 32 or block == 16
    )
