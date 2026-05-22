from std.utils.variant import Variant


@fieldwise_init
struct PerRow(Copyable, Movable, ImplicitlyCopyable):
    var fwht_block: Int


@fieldwise_init
struct PerBlock(Copyable, Movable, ImplicitlyCopyable):
    var fwht_block: Int
    var scale_block: Int


comptime Recipe = Variant[PerRow, PerBlock]


trait SlotLike:
    comptime DT: DType
    comptime ROWS: Int
    comptime COLS: Int
    comptime QUANT: Recipe


@fieldwise_init
struct QSlot[
    dt: DType, rows: Int, cols: Int, quant: Recipe,
](SlotLike, Copyable, ImplicitlyCopyable, Movable):
    comptime DT = Self.dt
    comptime ROWS = Self.rows
    comptime COLS = Self.cols
    comptime QUANT = Self.quant


comptime MathFn = def(Int, Int) thin -> Int


def math_body[FT: SlotLike](panel_rows: Int, seed: Int) -> Int:
    comptime DT = FT.DT
    comptime ROWS = FT.ROWS
    comptime COLS = FT.COLS
    comptime QV = FT.QUANT
    var x = seed * panel_rows * ROWS
    comptime if QV.isa[PerRow]():
        comptime QT = QV[PerRow]
        x += QT.fwht_block * 1000
    comptime if QV.isa[PerBlock]():
        comptime QT = QV[PerBlock]
        x += QT.fwht_block * 1000 + QT.scale_block
    if DT == DType.bfloat16:
        x += 100000
    if DT == DType.float8_e4m3fn:
        x += 200000
    return x


def call_via_ptr(fp: MathFn, panel_rows: Int, seed: Int) -> Int:
    return fp(panel_rows, seed)


def main():
    comptime FT_A = QSlot[DType.bfloat16, 64, 128, PerRow(128)]
    comptime FT_B = QSlot[DType.float8_e4m3fn, 32, 128, PerBlock(128, 32)]

    var fp_a: MathFn = math_body[FT_A]
    var fp_b: MathFn = math_body[FT_B]

    print(t"A direct: {math_body[FT_A](4, 7)}")
    print(t"B direct: {math_body[FT_B](4, 7)}")
    print(t"A via ptr: {call_via_ptr(fp_a, 4, 7)}")
    print(t"B via ptr: {call_via_ptr(fp_b, 4, 7)}")

    var fps = List[MathFn]()
    fps.append(fp_a)
    fps.append(fp_b)
    for i in range(len(fps)):
        print(t"stored[{i}]({4}, {7}) = {fps[i](4, 7)}")
