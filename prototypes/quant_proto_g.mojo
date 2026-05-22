from std.utils.variant import Variant
from std.reflection import reflect


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


@fieldwise_init
struct NoQuant(Copyable, Movable, ImplicitlyCopyable):
    pass


@fieldwise_init
struct PerRow(Copyable, Movable, ImplicitlyCopyable):
    var fwht_block: Int
    var gamma_name: StaticString
    var m_block: Int
    var want_cs: Bool


@fieldwise_init
struct PerBlock(Copyable, Movable, ImplicitlyCopyable):
    var fwht_block: Int
    var scale_block: Int
    var gamma_name: StaticString
    var want_cs: Bool


@fieldwise_init
struct RouterCenter(Copyable, Movable, ImplicitlyCopyable):
    pass


comptime QuantSpec = Variant[NoQuant, PerRow, PerBlock, RouterCenter]


trait SlotLike:
    comptime ENCODING: Encoding
    comptime SHAPE: ShapeLike
    comptime NAME: StaticString
    comptime QUANT: QuantSpec


@fieldwise_init
struct Slot[
    encoding: Encoding, shape: ShapeLike,
    name: StaticString = "",
    quant: QuantSpec = NoQuant(),
](SlotLike, Copyable, Movable, ImplicitlyCopyable):
    comptime ENCODING = Self.encoding
    comptime SHAPE = Self.shape
    comptime NAME = Self.name
    comptime QUANT = Self.quant


@fieldwise_init
struct DemoModel(Copyable, ImplicitlyCopyable):
    var q_proj:     Slot[BF16,   Shape[64, 128], "q_proj",     PerRow(128, "input_norm", 0, True)]
    var v_proj:     Slot[BF16,   Shape[64, 128], "v_proj",     PerRow(128, "input_norm", 64, True)]
    var o_proj:     Slot[BF16,   Shape[128, 64], "o_proj",     PerRow(64, "", 0, True)]
    var ffn_up:     Slot[F8E4M3, Shape[64, 128], "ffn_up",     PerRow(128, "", 0, False)]
    var lm_head:    Slot[BF16,   Shape[64, 128], "lm_head",    PerBlock(128, 32, "final_norm", True)]
    var router:     Slot[BF16,   Shape[8, 128],  "router",     RouterCenter()]
    var input_norm: Slot[BF16,   Shape[128, 1],  "input_norm"]
    var bias:       Slot[F32,    Shape[64, 1],   "bias"]


@fieldwise_init
struct OutputEntry(Copyable, Movable):
    var name: String
    var dtype: DType
    var rows: Int
    var cols: Int
    var bytes: Int


def emit_quant_plan[T: AnyType](mut entries: List[OutputEntry]):
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, SlotLike):
            comptime ROWS = FT.SHAPE.ROWS
            comptime COLS = FT.SHAPE.COLS
            comptime SRC_DT = FT.ENCODING.DTYPE
            comptime QV = FT.QUANT
            comptime if QV.isa[PerRow]():
                comptime QT = QV[PerRow]
                entries.append(OutputEntry(
                    name=String(FT.NAME), dtype=DType.int8,
                    rows=ROWS, cols=COLS, bytes=ROWS * COLS))
                entries.append(OutputEntry(
                    name=String(FT.NAME) + "_scale", dtype=DType.float32,
                    rows=ROWS, cols=1, bytes=ROWS * 4))
                comptime if QT.want_cs:
                    entries.append(OutputEntry(
                        name=String(FT.NAME) + "_cs", dtype=DType.float32,
                        rows=ROWS, cols=1, bytes=ROWS * 4))
            comptime if QV.isa[PerBlock]():
                comptime QT = QV[PerBlock]
                comptime NB = COLS // QT.scale_block
                entries.append(OutputEntry(
                    name=String(FT.NAME), dtype=DType.int8,
                    rows=ROWS, cols=COLS, bytes=ROWS * COLS))
                entries.append(OutputEntry(
                    name=String(FT.NAME) + "_scale", dtype=DType.float32,
                    rows=ROWS, cols=NB, bytes=ROWS * NB * 4))
                comptime if QT.want_cs:
                    entries.append(OutputEntry(
                        name=String(FT.NAME) + "_cs", dtype=DType.float32,
                        rows=ROWS, cols=NB, bytes=ROWS * NB * 4))
            comptime if QV.isa[RouterCenter]():
                entries.append(OutputEntry(
                    name=String(FT.NAME), dtype=DType.bfloat16,
                    rows=ROWS, cols=COLS, bytes=ROWS * COLS * 2))
                entries.append(OutputEntry(
                    name=String(FT.NAME) + "_gauge", dtype=DType.bfloat16,
                    rows=COLS, cols=1, bytes=COLS * 2))
            comptime if QV.isa[NoQuant]():
                entries.append(OutputEntry(
                    name=String(FT.NAME), dtype=SRC_DT,
                    rows=ROWS, cols=COLS, bytes=ROWS * COLS * FT.ENCODING.ELEMENT_BYTES))


def main():
    var entries = List[OutputEntry]()
    emit_quant_plan[DemoModel](entries)
    print(t"plan: {len(entries)} entries")
    for i in range(len(entries)):
        ref e = entries[i]
        print(t"  {e.name}: dtype={e.dtype} rows={e.rows} cols={e.cols} bytes={e.bytes}")
