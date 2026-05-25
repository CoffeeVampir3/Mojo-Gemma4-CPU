from std.sys.info import simd_width_of


comptime SIMD_F32_WIDTH = simd_width_of[DType.float32]()

comptime DEFAULT_PANEL_ROWS = 2048
comptime DEFAULT_COPY_CHUNK_BYTES = 16 * 1024 * 1024
comptime FWHT_POWER_OF_TWO_UNROLLING = 6


def is_supported_fwht_block(block: Int) -> Bool:
    comptime for i in range(FWHT_POWER_OF_TWO_UNROLLING):
        comptime candidate = SIMD_F32_WIDTH << i
        if block == candidate:
            return True
    return False
