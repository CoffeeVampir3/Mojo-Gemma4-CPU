from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys import llvm_intrinsic


comptime TILE_M = 16
comptime TILE_K = 64
comptime TILE_N = 16
comptime M_STEP = TILE_M * 2
comptime N_STEP = TILE_N * 2
comptime K_STEP = TILE_K
comptime TILE_BYTES = TILE_M * TILE_K


@fieldwise_init
struct TileConfig(Movable):
    var palette_id: UInt8
    var start_row: UInt8
    var reserved: InlineArray[UInt8, 14]
    var colsb: InlineArray[UInt16, 16]
    var rows: InlineArray[UInt8, 16]

    def __init__(out self):
        self.palette_id = 0
        self.start_row = 0
        self.reserved = InlineArray[UInt8, 14](fill=0)
        self.colsb = InlineArray[UInt16, 16](fill=0)
        self.rows = InlineArray[UInt8, 16](fill=0)


def make_224_i8_config() -> TileConfig:
    """2 A tiles, 2 B tiles, 4 C tiles. All tiles are 16x64 bytes."""
    var cfg = TileConfig()
    cfg.palette_id = 1
    for i in range(8):
        cfg.rows[i] = 16
        cfg.colsb[i] = 64
    return cfg^


def make_224_decode_config[hpg: Int]() -> TileConfig:
    """2-2-4 decode config with hpg-row A/C tiles."""
    var cfg = TileConfig()
    cfg.palette_id = 1
    cfg.rows[0] = UInt8(hpg)
    cfg.rows[1] = UInt8(hpg)
    cfg.rows[2] = 16
    cfg.rows[3] = 16
    cfg.rows[4] = UInt8(hpg)
    cfg.rows[5] = UInt8(hpg)
    cfg.rows[6] = UInt8(hpg)
    cfg.rows[7] = UInt8(hpg)
    for i in range(8):
        cfg.colsb[i] = 64
    return cfg^


def make_133_i8_config() -> TileConfig:
    """1 A tile, 3 B tiles, 3 C tiles. TMM7 is unused."""
    var cfg = TileConfig()
    cfg.palette_id = 1
    for i in range(7):
        cfg.rows[i] = 16
        cfg.colsb[i] = 64
    return cfg^


def init_intel_amx() -> Bool:
    """Request Linux xstate permission for AMX tile data."""
    comptime SYS_arch_prctl = 158
    comptime ARCH_REQ_XCOMP_PERM = 0x1023
    comptime XFEATURE_XTILEDATA = 18
    var result = __mlir_op.`pop.external_call`[
        func = "syscall".value,
        _type = Int64,
    ](
        Int64(SYS_arch_prctl),
        Int64(ARCH_REQ_XCOMP_PERM),
        Int64(XFEATURE_XTILEDATA),
    )
    return result == 0


@always_inline
def ldtilecfg(cfg: UnsafePointer[TileConfig, MutAnyOrigin]):
    llvm_intrinsic["llvm.x86.ldtilecfg", NoneType](cfg)


@always_inline
def tilerelease():
    llvm_intrinsic["llvm.x86.tilerelease", NoneType]()


@always_inline
def tilezero[tile: Int]():
    comptime assert tile >= 0 and tile < 8, "tile must be 0-7"
    llvm_intrinsic["llvm.x86.tilezero", NoneType](Int8(tile))


@always_inline
def tileload[tile: Int, dtype: DType](
    ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    stride: Int,
):
    comptime assert tile >= 0 and tile < 8, "tile must be 0-7"
    llvm_intrinsic["llvm.x86.tileloadd64", NoneType](
        Int8(tile), ptr, Int64(stride))


@always_inline
def tilestore[tile: Int, dtype: DType](
    ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    stride: Int,
):
    comptime assert tile >= 0 and tile < 8, "tile must be 0-7"
    llvm_intrinsic["llvm.x86.tilestored64", NoneType](
        Int8(tile), ptr, Int64(stride))


@always_inline
def tdpbssd[dst: Int, src_a: Int, src_b: Int]():
    """AMX i8*i8 -> i32 tile dot."""
    comptime assert dst >= 0 and dst < 8
    comptime assert src_a >= 0 and src_a < 8
    comptime assert src_b >= 0 and src_b < 8
    llvm_intrinsic["llvm.x86.tdpbssd", NoneType](
        Int8(dst), Int8(src_a), Int8(src_b))


@always_inline
def tdpbusd[dst: Int, src_a: Int, src_b: Int]():
    """AMX u8*i8 -> i32 tile dot."""
    comptime assert dst >= 0 and dst < 8
    comptime assert src_a >= 0 and src_a < 8
    comptime assert src_b >= 0 and src_b < 8
    llvm_intrinsic["llvm.x86.tdpbusd", NoneType](
        Int8(dst), Int8(src_a), Int8(src_b))


@always_inline
def tdpbsud[dst: Int, src_a: Int, src_b: Int]():
    """AMX i8*u8 -> i32 tile dot."""
    comptime assert dst >= 0 and dst < 8
    comptime assert src_a >= 0 and src_a < 8
    comptime assert src_b >= 0 and src_b < 8
    llvm_intrinsic["llvm.x86.tdpbsud", NoneType](
        Int8(dst), Int8(src_a), Int8(src_b))


@always_inline
def tdpbf16ps[dst: Int, src_a: Int, src_b: Int]():
    """AMX bf16*bf16 -> f32 tile dot."""
    comptime assert dst >= 0 and dst < 8
    comptime assert src_a >= 0 and src_a < 8
    comptime assert src_b >= 0 and src_b < 8
    llvm_intrinsic["llvm.x86.tdpbf16ps", NoneType](
        Int8(dst), Int8(src_a), Int8(src_b))


@always_inline
def tile_dp[dst: Int, src_a: Int, src_b: Int, a_dtype: DType, b_dtype: DType]():
    """Dispatch to the AMX dot instruction for the A/B tile dtypes."""
    comptime assert dst >= 0 and dst < 8
    comptime assert src_a >= 0 and src_a < 8
    comptime assert src_b >= 0 and src_b < 8
    comptime if a_dtype == DType.int8 and b_dtype == DType.int8:
        tdpbssd[dst, src_a, src_b]()
    elif a_dtype == DType.int8 and b_dtype == DType.uint8:
        tdpbsud[dst, src_a, src_b]()
    elif a_dtype == DType.uint8 and b_dtype == DType.int8:
        tdpbusd[dst, src_a, src_b]()
    elif a_dtype == DType.bfloat16 and b_dtype == DType.bfloat16:
        tdpbf16ps[dst, src_a, src_b]()
    else:
        comptime assert False, "unsupported dtype combination for tile_dp"
