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


@fieldwise_init
struct Shape[rows: Int, cols: Int](ShapeLike, Copyable, ImplicitlyCopyable):
    comptime ROWS = Self.rows
    comptime COLS = Self.cols


trait QuantSlotLike:
    comptime SRC: Encoding
    comptime SHAPE: ShapeLike
    comptime NAME: StaticString


trait PerRowInt8Like(QuantSlotLike):
    comptime FWHT_BLOCK: Int
    comptime GAMMA_NAME: StaticString
    comptime GAMMA_SRC: Encoding


trait PassthroughLike(QuantSlotLike):
    pass


@fieldwise_init
struct PerRowInt8[
    src: Encoding, shape: ShapeLike, name: StaticString,
    fwht_block: Int, gamma_name: StaticString = "", gamma_src: Encoding = BF16,
](PerRowInt8Like, Copyable, ImplicitlyCopyable):
    comptime SRC = Self.src
    comptime SHAPE = Self.shape
    comptime NAME = Self.name
    comptime FWHT_BLOCK = Self.fwht_block
    comptime GAMMA_NAME = Self.gamma_name
    comptime GAMMA_SRC = Self.gamma_src


@fieldwise_init
struct Passthrough[
    src: Encoding, shape: ShapeLike, name: StaticString,
](PassthroughLike, Copyable, ImplicitlyCopyable):
    comptime SRC = Self.src
    comptime SHAPE = Self.shape
    comptime NAME = Self.name


@fieldwise_init
struct DemoRecipe(Copyable, ImplicitlyCopyable):
    var q_proj:     PerRowInt8[BF16,    Shape[64, 128], "q_proj",     128, "input_norm"]
    var ffn_up:     PerRowInt8[F8E4M3,  Shape[64, 128], "ffn_up",     128]
    var input_norm: Passthrough[BF16,   Shape[128, 1],  "input_norm"]


@fieldwise_init
struct OutputEntry(Copyable, ImplicitlyCopyable, Movable):
    var name: String
    var dtype: DType
    var rows: Int
    var cols: Int
    var data_start: Int
    var data_end: Int


comptime W = simd_width_of[DType.float32]()


def plan_recipe[T: AnyType](mut entries: List[OutputEntry], off_in: Int = 0) -> Int:
    var off = off_in
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, PerRowInt8Like):
            comptime ROWS = FT.SHAPE.ROWS
            comptime COLS = FT.SHAPE.COLS
            var w_bytes = ROWS * COLS
            entries.append(OutputEntry(
                name=String(FT.NAME),
                dtype=DType.int8, rows=ROWS, cols=COLS,
                data_start=off, data_end=off + w_bytes,
            ))
            off += w_bytes
            var s_bytes = ROWS * 4
            entries.append(OutputEntry(
                name=String(FT.NAME) + "_scale",
                dtype=DType.float32, rows=ROWS, cols=1,
                data_start=off, data_end=off + s_bytes,
            ))
            off += s_bytes
        comptime if conforms_to(FT, PassthroughLike):
            comptime ROWS = FT.SHAPE.ROWS
            comptime COLS = FT.SHAPE.COLS
            comptime BYTES = ROWS * COLS * FT.SRC.ELEMENT_BYTES
            entries.append(OutputEntry(
                name=String(FT.NAME),
                dtype=FT.SRC.DTYPE, rows=ROWS, cols=COLS,
                data_start=off, data_end=off + BYTES,
            ))
            off += BYTES
    return off


def decode_to_f32[src: DType](
    src_ptr: UnsafePointer[Scalar[src], MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    count: Int,
):
    def step[width: Int](idx: Int) {read}:
        (dst + idx).store(
            (src_ptr + idx).load[width=width]().cast[DType.float32]())
    vectorize[W](count, step)


def apply_gamma_sqrt_abs(
    g: UnsafePointer[Float32, MutAnyOrigin],
    work_row: UnsafePointer[Float32, MutAnyOrigin],
    cols: Int,
):
    var k = 0
    while k + W <= cols:
        var gv = (g + k).load[width=W]()
        var sg = (gv * gv).__abs__()
        (work_row + k).store(
            (work_row + k).load[width=W]() * sg)
        k += W


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


@fieldwise_init
struct SourceBuffers(Copyable, ImplicitlyCopyable):
    var q_proj_bytes: Int
    var ffn_up_bytes: Int
    var input_norm_bytes: Int


def execute_per_row_int8_field[
    FT: PerRowInt8Like,
](src_addr: Int, gamma_addr: Int):
    comptime ROWS = FT.SHAPE.ROWS
    comptime COLS = FT.SHAPE.COLS
    comptime SRC_DT = FT.SRC.DTYPE
    comptime HAS_GAMMA = FT.GAMMA_NAME != StaticString("")
    comptime GDT = FT.GAMMA_SRC.DTYPE

    var src = UnsafePointer[Scalar[SRC_DT], MutAnyOrigin](
        unsafe_from_address=src_addr)
    var work = alloc[Float32](ROWS * COLS)
    var qi = alloc[Scalar[DType.int8]](ROWS * COLS)
    var scales = alloc[Float32](ROWS)

    decode_to_f32[SRC_DT](src, work, ROWS * COLS)

    comptime if HAS_GAMMA:
        var g_raw = UnsafePointer[Scalar[GDT], MutAnyOrigin](
            unsafe_from_address=gamma_addr)
        var g_f32 = alloc[Float32](COLS)
        decode_to_f32[GDT](g_raw, g_f32, COLS)
        for r in range(ROWS):
            apply_gamma_sqrt_abs(g_f32, work + r * COLS, COLS)
        g_f32.free()

    for r in range(ROWS):
        scales[r] = quantize_row(work + r * COLS, qi + r * COLS, COLS)

    print(t"  [{FT.NAME}] src_dtype={SRC_DT}  rows={ROWS}  cols={COLS}  gamma={HAS_GAMMA}  scale[0]={scales[0]}")

    work.free()
    qi.free()
    scales.free()


def run_recipe[T: AnyType](
    recipe: T,
    q_proj_addr: Int, ffn_up_addr: Int, input_norm_addr: Int,
):
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, PerRowInt8Like):
            comptime if FT.NAME == StaticString("q_proj"):
                execute_per_row_int8_field[FT](q_proj_addr, input_norm_addr)
            comptime if FT.NAME == StaticString("ffn_up"):
                execute_per_row_int8_field[FT](ffn_up_addr, 0)
        comptime if conforms_to(FT, PassthroughLike):
            print(t"  [{FT.NAME}] passthrough dtype={FT.SRC.DTYPE} rows={FT.SHAPE.ROWS} cols={FT.SHAPE.COLS}")


def main():
    var entries = List[OutputEntry]()
    var total = plan_recipe[DemoRecipe](entries)
    print(t"plan: {len(entries)} entries, {total} bytes")
    for i in range(len(entries)):
        ref e = entries[i]
        print(t"  {e.name}: dtype={e.dtype} rows={e.rows} cols={e.cols} bytes={e.data_end - e.data_start}")

    var q_bytes = alloc[Scalar[DType.bfloat16]](64 * 128)
    var ffn_bytes = alloc[Scalar[DType.float8_e4m3fn]](64 * 128)
    var norm_bytes = alloc[Scalar[DType.bfloat16]](128)
    for i in range(64 * 128):
        q_bytes[i] = Scalar[DType.bfloat16](Float32((i % 17) - 8))
        ffn_bytes[i] = Scalar[DType.float8_e4m3fn](Float32(1.5))
    for i in range(128):
        norm_bytes[i] = Scalar[DType.bfloat16](Float32(0.5))

    print("execute:")
    var recipe = DemoRecipe(
        q_proj=PerRowInt8[BF16, Shape[64, 128], "q_proj", 128, "input_norm"](),
        ffn_up=PerRowInt8[F8E4M3, Shape[64, 128], "ffn_up", 128](),
        input_norm=Passthrough[BF16, Shape[128, 1], "input_norm"](),
    )
    run_recipe(recipe, Int(q_bytes), Int(ffn_bytes), Int(norm_bytes))

    q_bytes.free()
    ffn_bytes.free()
    norm_bytes.free()
