from std.algorithm import vectorize
from std.math import max
from std.memory import UnsafePointer, alloc
from std.reflection import reflect
from std.sys.info import simd_width_of


trait Encoding:
    comptime DTYPE: DType
    comptime ELEMENT_BYTES: Int


struct BF16(Encoding):
    comptime DTYPE = DType.bfloat16
    comptime ELEMENT_BYTES = 2


struct F32(Encoding):
    comptime DTYPE = DType.float32
    comptime ELEMENT_BYTES = 4


struct F8E4M3(Encoding):
    comptime DTYPE = DType.float8_e4m3fn
    comptime ELEMENT_BYTES = 1


trait ShapeLike:
    comptime ROWS: Int
    comptime COLS: Int


struct Shape[rows: Int, cols: Int](ShapeLike):
    comptime ROWS = Self.rows
    comptime COLS = Self.cols


trait QuantSpec:
    pass


trait PerRowKind(QuantSpec):
    comptime FWHT_BLOCK: Int
    comptime GAMMA_NAME: StaticString
    comptime M_BLOCK: Int
    comptime WANT_CS: Bool


struct PerRow[
    fwht_block: Int,
    gamma_name: StaticString = "",
    m_block: Int = 0,
    want_cs: Bool = True,
](PerRowKind):
    comptime FWHT_BLOCK = Self.fwht_block
    comptime GAMMA_NAME = Self.gamma_name
    comptime M_BLOCK = Self.m_block
    comptime WANT_CS = Self.want_cs


trait PerBlockKind(QuantSpec):
    comptime FWHT_BLOCK: Int
    comptime SCALE_BLOCK: Int
    comptime GAMMA_NAME: StaticString
    comptime WANT_CS: Bool


struct PerBlock[
    fwht_block: Int, scale_block: Int,
    gamma_name: StaticString = "", want_cs: Bool = True,
](PerBlockKind):
    comptime FWHT_BLOCK = Self.fwht_block
    comptime SCALE_BLOCK = Self.scale_block
    comptime GAMMA_NAME = Self.gamma_name
    comptime WANT_CS = Self.want_cs


trait RouterCenterKind(QuantSpec):
    pass


struct RouterCenter(RouterCenterKind):
    pass


struct NoQuant(QuantSpec):
    pass


trait SlotLike:
    comptime ENCODING: Encoding
    comptime SHAPE: ShapeLike
    comptime NAME: StaticString
    comptime QUANT: QuantSpec


@fieldwise_init
struct Slot[
    encoding: Encoding, shape: ShapeLike,
    name: StaticString = "",
    quant: QuantSpec = NoQuant,
](SlotLike, Copyable, ImplicitlyCopyable):
    comptime ENCODING = Self.encoding
    comptime SHAPE = Self.shape
    comptime NAME = Self.name
    comptime QUANT = Self.quant
    var offset: Int

    def __init__(out self):
        self.offset = -1


@fieldwise_init
struct DemoModel(Copyable, ImplicitlyCopyable):
    var q_proj:     Slot[BF16,   Shape[64, 128], "q_proj",     PerRow[128, "input_norm"]]
    var v_proj:     Slot[BF16,   Shape[64, 128], "v_proj",     PerRow[128, "input_norm", 64]]
    var o_proj:     Slot[BF16,   Shape[128, 64], "o_proj",     PerRow[64]]
    var ffn_up:     Slot[F8E4M3, Shape[64, 128], "ffn_up",     PerRow[128, "", 0, False]]
    var lm_head:    Slot[BF16,   Shape[64, 128], "lm_head",    PerBlock[128, 32, "final_norm"]]
    var router:    Slot[BF16,    Shape[8, 128],  "router",     RouterCenter]
    var input_norm: Slot[BF16,   Shape[128, 1],  "input_norm"]
    var bias:       Slot[F32,    Shape[64, 1],   "bias"]


@fieldwise_init
struct WeightDesc(Copyable, Movable):
    var name: String
    var dtype: DType
    var rows: Int
    var cols: Int
    var arena_offset: Int


def emit_descs[T: AnyType](mut descs: List[WeightDesc], off_in: Int = 0) -> Int:
    var off = off_in
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, SlotLike):
            comptime if FT.NAME != StaticString(""):
                comptime ROWS = FT.SHAPE.ROWS
                comptime COLS = FT.SHAPE.COLS
                comptime BYTES = ROWS * COLS * FT.ENCODING.ELEMENT_BYTES
                descs.append(WeightDesc(
                    name=String(FT.NAME),
                    dtype=FT.ENCODING.DTYPE,
                    rows=ROWS, cols=COLS,
                    arena_offset=off,
                ))
                off += BYTES
    return off


@fieldwise_init
struct OutputEntry(Copyable, Movable):
    var name: String
    var dtype: DType
    var rows: Int
    var cols: Int
    var data_start: Int
    var data_end: Int


def emit_quant_plan[T: AnyType](mut entries: List[OutputEntry], off_in: Int = 0) -> Int:
    var off = off_in
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, SlotLike):
            comptime ROWS = FT.SHAPE.ROWS
            comptime COLS = FT.SHAPE.COLS
            comptime QT = FT.QUANT
            comptime if conforms_to(QT, PerRowKind):
                var wb = ROWS * COLS
                entries.append(OutputEntry(
                    name=String(FT.NAME), dtype=DType.int8,
                    rows=ROWS, cols=COLS,
                    data_start=off, data_end=off + wb))
                off += wb
                var sb = ROWS * 4
                entries.append(OutputEntry(
                    name=String(FT.NAME) + "_scale", dtype=DType.float32,
                    rows=ROWS, cols=1,
                    data_start=off, data_end=off + sb))
                off += sb
                comptime if QT.WANT_CS:
                    var cb = ROWS * 4
                    entries.append(OutputEntry(
                        name=String(FT.NAME) + "_cs", dtype=DType.float32,
                        rows=ROWS, cols=1,
                        data_start=off, data_end=off + cb))
                    off += cb
            elif conforms_to(QT, PerBlockKind):
                comptime NB = COLS // QT.SCALE_BLOCK
                var wb = ROWS * COLS
                entries.append(OutputEntry(
                    name=String(FT.NAME), dtype=DType.int8,
                    rows=ROWS, cols=COLS,
                    data_start=off, data_end=off + wb))
                off += wb
                var sb = ROWS * NB * 4
                entries.append(OutputEntry(
                    name=String(FT.NAME) + "_scale", dtype=DType.float32,
                    rows=ROWS, cols=NB,
                    data_start=off, data_end=off + sb))
                off += sb
                comptime if QT.WANT_CS:
                    var cb = ROWS * NB * 4
                    entries.append(OutputEntry(
                        name=String(FT.NAME) + "_cs", dtype=DType.float32,
                        rows=ROWS, cols=NB,
                        data_start=off, data_end=off + cb))
                    off += cb
            elif conforms_to(QT, RouterCenterKind):
                var wb = ROWS * COLS * 2
                entries.append(OutputEntry(
                    name=String(FT.NAME), dtype=DType.bfloat16,
                    rows=ROWS, cols=COLS,
                    data_start=off, data_end=off + wb))
                off += wb
                var gb = COLS * 2
                entries.append(OutputEntry(
                    name=String(FT.NAME) + "_gauge", dtype=DType.bfloat16,
                    rows=COLS, cols=1,
                    data_start=off, data_end=off + gb))
                off += gb
            else:
                comptime BYTES = ROWS * COLS * FT.ENCODING.ELEMENT_BYTES
                entries.append(OutputEntry(
                    name=String(FT.NAME), dtype=FT.ENCODING.DTYPE,
                    rows=ROWS, cols=COLS,
                    data_start=off, data_end=off + BYTES))
                off += BYTES
    return off


comptime W = simd_width_of[DType.float32]()


def decode_to_f32[src: DType](
    src_ptr: UnsafePointer[Scalar[src], MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    count: Int,
):
    def step[width: Int](idx: Int) {read}:
        (dst + idx).store(
            (src_ptr + idx).load[width=width]().cast[DType.float32]())
    vectorize[W](count, step)


def quantize_row(
    work_row: UnsafePointer[Float32, MutAnyOrigin],
    qi_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    cols: Int,
) -> Float32:
    var vmax = SIMD[DType.float32, W](0)
    var k = 0
    while k + W <= cols:
        vmax = max(vmax, (work_row + k).load[width=W]().__abs__())
        k += W
    var amax = vmax.reduce_max()
    var inv = Float32(127.0) / amax if amax > Float32(0) else Float32(0)
    var vinv = SIMD[DType.float32, W](inv)
    k = 0
    while k + W <= cols:
        var v = (work_row + k).load[width=W]()
        (qi_row + k).store((v * vinv).cast[DType.int8]())
        k += W
    return amax / Float32(127.0)


def run_per_row[FT: SlotLike, QT: PerRowKind](src_addr: Int):
    comptime SRC_DT = FT.ENCODING.DTYPE
    comptime ROWS = FT.SHAPE.ROWS
    comptime COLS = FT.SHAPE.COLS
    var src = UnsafePointer[Scalar[SRC_DT], MutAnyOrigin](
        unsafe_from_address=src_addr)
    var work = alloc[Float32](ROWS * COLS)
    var qi = alloc[Scalar[DType.int8]](ROWS * COLS)
    var scales = alloc[Float32](ROWS)
    decode_to_f32[SRC_DT](src, work, ROWS * COLS)
    for r in range(ROWS):
        scales[r] = quantize_row(work + r * COLS, qi + r * COLS, COLS)
    print(t"  [{FT.NAME}] PerRow src={SRC_DT} {ROWS}x{COLS} fwht={QT.FWHT_BLOCK} m_block={QT.M_BLOCK} gamma=\"{QT.GAMMA_NAME}\" want_cs={QT.WANT_CS} scale0={scales[0]}")
    work.free(); qi.free(); scales.free()


def run_per_block[FT: SlotLike, QT: PerBlockKind](src_addr: Int):
    comptime SRC_DT = FT.ENCODING.DTYPE
    print(t"  [{FT.NAME}] PerBlock src={SRC_DT} {FT.SHAPE.ROWS}x{FT.SHAPE.COLS} fwht={QT.FWHT_BLOCK} scale_block={QT.SCALE_BLOCK} gamma=\"{QT.GAMMA_NAME}\" want_cs={QT.WANT_CS}")


def run_quant_plan[T: AnyType](
    q_addr: Int, v_addr: Int, o_addr: Int, ffn_addr: Int,
):
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, SlotLike):
            comptime QT = FT.QUANT
            comptime if conforms_to(QT, PerRowKind):
                comptime if FT.NAME == StaticString("q_proj"):
                    run_per_row[FT, QT](q_addr)
                comptime if FT.NAME == StaticString("v_proj"):
                    run_per_row[FT, QT](v_addr)
                comptime if FT.NAME == StaticString("o_proj"):
                    run_per_row[FT, QT](o_addr)
                comptime if FT.NAME == StaticString("ffn_up"):
                    run_per_row[FT, QT](ffn_addr)
            elif conforms_to(QT, PerBlockKind):
                run_per_block[FT, QT](0)
            elif conforms_to(QT, RouterCenterKind):
                print(t"  [{FT.NAME}] RouterCenter src={FT.ENCODING.DTYPE} {FT.SHAPE.ROWS}x{FT.SHAPE.COLS}")
            else:
                print(t"  [{FT.NAME}] Passthrough dtype={FT.ENCODING.DTYPE} {FT.SHAPE.ROWS}x{FT.SHAPE.COLS}")


def main():
    print("--- loader walk ---")
    var descs = List[WeightDesc]()
    var lt = emit_descs[DemoModel](descs)
    print(t"loader: {len(descs)} descs, {lt} bytes")
    for i in range(len(descs)):
        ref d = descs[i]
        print(t"  {d.name}: dtype={d.dtype} rows={d.rows} cols={d.cols} off={d.arena_offset}")

    print("--- quant plan walk ---")
    var entries = List[OutputEntry]()
    var qt = emit_quant_plan[DemoModel](entries)
    print(t"quant: {len(entries)} entries, {qt} bytes")
    for i in range(len(entries)):
        ref e = entries[i]
        print(t"  {e.name}: dtype={e.dtype} rows={e.rows} cols={e.cols} bytes={e.data_end - e.data_start}")

    print("--- executor walk ---")
    var q = alloc[Scalar[DType.bfloat16]](64 * 128)
    var v = alloc[Scalar[DType.bfloat16]](64 * 128)
    var o = alloc[Scalar[DType.bfloat16]](128 * 64)
    var ffn = alloc[Scalar[DType.float8_e4m3fn]](64 * 128)
    for i in range(64 * 128):
        q[i] = Scalar[DType.bfloat16](Float32((i % 17) - 8))
        v[i] = Scalar[DType.bfloat16](Float32((i % 13) - 6))
        ffn[i] = Scalar[DType.float8_e4m3fn](Float32(1.5))
    for i in range(128 * 64):
        o[i] = Scalar[DType.bfloat16](Float32((i % 7) - 3))
    run_quant_plan[DemoModel](Int(q), Int(v), Int(o), Int(ffn))
    q.free(); v.free(); o.free(); ffn.free()
