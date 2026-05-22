from std.algorithm import vectorize
from std.collections import InlineArray
from std.memory import UnsafePointer, alloc
from std.math import max
from std.sys.info import simd_width_of


comptime NUM_SRC_DTYPES = 14
comptime SOURCE_DTYPES: InlineArray[DType, NUM_SRC_DTYPES] = [
    DType.bfloat16, DType.float16, DType.float32, DType.float64,
    DType.int8, DType.uint8, DType.int16, DType.uint16,
    DType.int32, DType.uint32, DType.int64, DType.uint64,
    DType.float8_e4m3fn, DType.float8_e5m2,
]

comptime W = simd_width_of[DType.float32]()


@fieldwise_init
struct GammaSpec(Copyable, Movable):
    var dtype: DType
    var bytes: Int
    var addr: Int


@fieldwise_init
struct Int8TaskRuntime(Copyable, Movable):
    var src_dtype: DType
    var rows: Int
    var cols: Int
    var src_addr: Int
    var work_addr: Int
    var qi_addr: Int
    var scales_addr: Int
    var gamma: Optional[GammaSpec]


def decode_panel[src: DType](
    src_ptr: UnsafePointer[Scalar[src], MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    count: Int,
):
    def step[width: Int](idx: Int) {read}:
        (work + idx).store(
            (src_ptr + idx).load[width=width]().cast[DType.float32]())
    vectorize[W](count, step)


def apply_gamma[gdt: DType](
    gamma: UnsafePointer[Scalar[gdt], MutAnyOrigin],
    work_row: UnsafePointer[Float32, MutAnyOrigin],
    cols: Int,
):
    def step[width: Int](idx: Int) {read}:
        var g = (gamma + idx).load[width=width]().cast[DType.float32]()
        (work_row + idx).store((work_row + idx).load[width=width]() * g)
    vectorize[W](cols, step)


def row_absmax(work_row: UnsafePointer[Float32, MutAnyOrigin], cols: Int) -> Float32:
    var vmax = SIMD[DType.float32, W](0)
    var k = 0
    while k + W <= cols:
        vmax = max(vmax, (work_row + k).load[width=W]().__abs__())
        k += W
    return vmax.reduce_max()


def quantize_row(
    work_row: UnsafePointer[Float32, MutAnyOrigin],
    qi_row: UnsafePointer[Scalar[DType.int8], MutAnyOrigin],
    cols: Int,
) -> Float32:
    var amax = row_absmax(work_row, cols)
    var inv = Float32(127.0) / amax if amax > Float32(0) else Float32(0)
    var vinv = SIMD[DType.float32, W](inv)
    var k = 0
    while k + W <= cols:
        var v = (work_row + k).load[width=W]()
        (qi_row + k).store((v * vinv).cast[DType.int8]())
        k += W
    return amax / Float32(127.0)


def run_int8_panel_typed[src: DType, gdt: DType, has_gamma: Bool](
    pt: Int8TaskRuntime, panel_rows: Int,
):
    var src_ptr = UnsafePointer[Scalar[src], MutAnyOrigin](
        unsafe_from_address=pt.src_addr)
    var work = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=pt.work_addr)
    var qi = UnsafePointer[Scalar[DType.int8], MutAnyOrigin](
        unsafe_from_address=pt.qi_addr)
    var scales = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=pt.scales_addr)

    decode_panel[src](src_ptr, work, panel_rows * pt.cols)

    comptime if has_gamma:
        var g_ptr = UnsafePointer[Scalar[gdt], MutAnyOrigin](
            unsafe_from_address=pt.gamma.value().addr)
        for r in range(panel_rows):
            apply_gamma[gdt](g_ptr, work + r * pt.cols, pt.cols)

    for r in range(panel_rows):
        scales[r] = quantize_row(work + r * pt.cols, qi + r * pt.cols, pt.cols)


def run_int8_panel_src[src: DType](pt: Int8TaskRuntime, panel_rows: Int):
    if pt.gamma:
        var gdt = pt.gamma.value().dtype
        comptime for j in range(NUM_SRC_DTYPES):
            comptime GDT = SOURCE_DTYPES[j]
            if gdt == GDT:
                run_int8_panel_typed[src, GDT, True](pt, panel_rows)
                return
        print(t"unsupported gamma dtype {gdt}")
    else:
        run_int8_panel_typed[src, DType.float32, False](pt, panel_rows)


def run_int8_panel(pt: Int8TaskRuntime, panel_rows: Int) -> Bool:
    comptime for i in range(NUM_SRC_DTYPES):
        comptime SRC = SOURCE_DTYPES[i]
        if pt.src_dtype == SRC:
            run_int8_panel_src[SRC](pt, panel_rows)
            return True
    print(t"unsupported src dtype {pt.src_dtype}")
    return False


def main():
    var rows = 4
    var cols = 32
    var n = rows * cols

    var src = alloc[Scalar[DType.bfloat16]](n)
    var gamma = alloc[Scalar[DType.bfloat16]](cols)
    var work = alloc[Float32](n)
    var qi = alloc[Scalar[DType.int8]](n)
    var scales = alloc[Float32](rows)

    for r in range(rows):
        for c in range(cols):
            src[r * cols + c] = Scalar[DType.bfloat16](Float32(c - r))
    for c in range(cols):
        gamma[c] = Scalar[DType.bfloat16](Float32(0.5 if (c % 2) == 0 else 1.0))

    var task = Int8TaskRuntime(
        src_dtype=DType.bfloat16, rows=rows, cols=cols,
        src_addr=Int(src), work_addr=Int(work),
        qi_addr=Int(qi), scales_addr=Int(scales),
        gamma=Optional[GammaSpec](GammaSpec(
            dtype=DType.bfloat16, bytes=cols * 2, addr=Int(gamma))),
    )

    if not run_int8_panel(task, rows):
        print("FAIL")
        return

    for r in range(rows):
        print(t"scale[{r}] = {scales[r]}")
        var s = String("qi[{r}] = ")
        for c in range(cols):
            s += String(Int(qi[r * cols + c])) + " "
        print(s)

    src.free()
    gamma.free()
    work.free()
    qi.free()
    scales.free()
