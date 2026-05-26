from std.sys.info import simd_width_of


comptime SIMD_F32_WIDTH = simd_width_of[DType.float32]()

comptime DEFAULT_PANEL_ROWS = 2048
comptime DEFAULT_COPY_CHUNK_BYTES = 16 * 1024 * 1024
