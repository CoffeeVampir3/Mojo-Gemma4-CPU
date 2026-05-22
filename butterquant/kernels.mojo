from std.memory import UnsafePointer
from std.math import max
from std.os import abort

from simd_math.ops import quantize_i8, sqrt
from butterquant.fwht import fwht_block, fwht_row
from butterquant.constants import SIMD_F32_WIDTH


comptime PtrU8 = UnsafePointer[UInt8, MutAnyOrigin]
comptime PtrF32 = UnsafePointer[Float32, MutAnyOrigin]
comptime PtrBF16 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime PtrI8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime SrcPtr[dtype: DType] = UnsafePointer[Scalar[dtype], MutAnyOrigin]
comptime WIDTH = SIMD_F32_WIDTH


@always_inline
def bf16_to_f32(src: PtrBF16, dst: PtrF32, count: Int):
    var k = 0
    while k + WIDTH <= count:
        (dst + k).store((src + k).load[width=WIDTH]().cast[DType.float32]())
        k += WIDTH


@always_inline
def apply_gamma_in_place(work_row: PtrF32, gamma: PtrF32, cols: Int):
    var k = 0
    while k + WIDTH <= cols:
        (work_row + k).store(
            (work_row + k).load[width=WIDTH]() *
            (gamma + k).load[width=WIDTH]())
        k += WIDTH


@always_inline
def gamma_sqrt_abs_in_place(gamma: PtrF32, cols: Int):
    var k = 0
    while k + WIDTH <= cols:
        var v = (gamma + k).load[width=WIDTH]().__abs__()
        (gamma + k).store(sqrt[DType.float32, WIDTH](v))
        k += WIDTH


@always_inline
def row_absmax(work_row: PtrF32, cols: Int) -> Float32:
    var vmax = SIMD[DType.float32, WIDTH](0)
    var k = 0
    while k + WIDTH <= cols:
        vmax = max(vmax, (work_row + k).load[width=WIDTH]().__abs__())
        k += WIDTH
    return vmax.reduce_max()


@always_inline
def quantize_inv(work: PtrF32, qi: PtrI8, inv: Float32, n: Int):
    var vinv = SIMD[DType.float32, WIDTH](inv)
    var k = 0
    while k + WIDTH <= n:
        var v = (work + k).load[width=WIDTH]()
        (qi + k).store(quantize_i8[WIDTH](v, vinv))
        k += WIDTH


def fwht_rotate_rows[block: Int](work: PtrF32, rows: Int, cols: Int):
    if cols % block != 0:
        abort(t"butterquant: cols={cols} not divisible by K-axis FWHT block={block}")
    for r in range(rows):
        fwht_row[block](work + r * cols, cols)


def fwht_rotate_columns[head_dim: Int](work: PtrF32, rows: Int, cols: Int):
    if rows % head_dim != 0:
        abort(t"butterquant: rows={rows} not divisible by M-axis FWHT block={head_dim}")
    var scratch_buf = List[Float32](length=head_dim, fill=Float32(0))
    var scratch = PtrF32(unsafe_from_address=Int(scratch_buf.unsafe_ptr()))
    var num_heads = rows // head_dim
    for h in range(num_heads):
        var base = h * head_dim
        for c in range(cols):
            for r in range(head_dim):
                (scratch + r).store((work + (base + r) * cols + c).load())
            fwht_block[head_dim](scratch)
            for r in range(head_dim):
                (work + (base + r) * cols + c).store((scratch + r).load())
    _ = scratch_buf^


def quant_rows_per_row(work: PtrF32, qi: PtrI8, scales: PtrF32, rows: Int, cols: Int):
    for r in range(rows):
        var work_row = work + r * cols
        var qi_row = qi + r * cols
        var amax = row_absmax(work_row, cols)
        scales[r] = amax / Float32(127.0)
        var inv = Float32(127.0) / amax if amax > Float32(0) else Float32(0)
        quantize_inv(work_row, qi_row, inv, cols)


def quant_rows_per_block[block: Int](
    work: PtrF32, qi: PtrI8, scales: PtrF32, rows: Int, cols: Int,
):
    var num_blocks = cols // block
    for r in range(rows):
        var work_row = work + r * cols
        var qi_row = qi + r * cols
        var scale_row = scales + r * num_blocks
        for b in range(num_blocks):
            var off = b * block
            var amax = row_absmax(work_row + off, block)
            scale_row[b] = amax / Float32(127.0)
            var inv = Float32(127.0) / amax if amax > Float32(0) else Float32(0)
            quantize_inv(work_row + off, qi_row + off, inv, block)


def rotate_and_quant_per_row[block: Int](
    work: PtrF32, qi: PtrI8, scales: PtrF32,
    rows: Int, cols: Int,
):
    fwht_rotate_rows[block](work, rows, cols)
    quant_rows_per_row(work, qi, scales, rows, cols)


def rotate_and_quant_per_block[block: Int](
    work: PtrF32, qi: PtrI8, scales: PtrF32,
    rows: Int, cols: Int,
):
    fwht_rotate_rows[block](work, rows, cols)
    quant_rows_per_block[block](work, qi, scales, rows, cols)


def rotate_and_quant[per_block: Bool](
    block: Int, work: PtrF32, qi: PtrI8, scales: PtrF32,
    rows: Int, cols: Int, two_sided_head_dim: Int = 0,
):
    @parameter
    def go[b: Int]():
        fwht_rotate_rows[b](work, rows, cols)
        if two_sided_head_dim == 512: fwht_rotate_columns[512](work, rows, cols)
        elif two_sided_head_dim == 256: fwht_rotate_columns[256](work, rows, cols)
        elif two_sided_head_dim == 128: fwht_rotate_columns[128](work, rows, cols)
        elif two_sided_head_dim == 64: fwht_rotate_columns[64](work, rows, cols)
        elif two_sided_head_dim == 32: fwht_rotate_columns[32](work, rows, cols)
        elif two_sided_head_dim == 16: fwht_rotate_columns[16](work, rows, cols)
        elif two_sided_head_dim != 0:
            abort(t"butterquant: unsupported M-axis FWHT block={two_sided_head_dim}")
        comptime if per_block:
            quant_rows_per_block[b](work, qi, scales, rows, cols)
        else:
            quant_rows_per_row(work, qi, scales, rows, cols)
    if block == 512: go[512]()
    elif block == 256: go[256]()
    elif block == 128: go[128]()
    elif block == 64: go[64]()
    elif block == 32: go[32]()
    elif block == 16: go[16]()
    else:
        abort(t"butterquant: unsupported K-axis FWHT block={block}")


def colsum_per_row(qi: PtrI8, cs: PtrF32, rows: Int, cols: Int):
    for r in range(rows):
        var qi_row = qi + r * cols
        var acc = SIMD[DType.int32, WIDTH](0)
        var k = 0
        while k + WIDTH <= cols:
            acc += (qi_row + k).load[width=WIDTH]().cast[DType.int32]()
            k += WIDTH
        cs[r] = Float32(acc.reduce_add())


def colsum_per_block(qi: PtrI8, cs: PtrF32, block: Int, rows: Int, cols: Int):
    var num_blocks = cols // block
    for r in range(rows):
        var qi_row = qi + r * cols
        var cs_row = cs + r * num_blocks
        for b in range(num_blocks):
            var off = b * block
            var acc = SIMD[DType.int32, WIDTH](0)
            var k = 0
            while k + WIDTH <= block:
                acc += (qi_row + off + k).load[width=WIDTH]().cast[DType.int32]()
                k += WIDTH
            cs_row[b] = Float32(acc.reduce_add())


def router_center[src_dtype: DType](
    src: SrcPtr[src_dtype], gauge: PtrF32,
    centered_bf16: PtrBF16, gauge_bf16: PtrBF16,
    rows: Int, cols: Int,
):
    var k = 0
    while k + WIDTH <= cols:
        (gauge + k).store(SIMD[DType.float32, WIDTH](0))
        k += WIDTH

    for r in range(rows):
        var row = src + r * cols
        k = 0
        while k + WIDTH <= cols:
            var v = (row + k).load[width=WIDTH]().cast[DType.float32]()
            (gauge + k).store((gauge + k).load[width=WIDTH]() + v)
            k += WIDTH

    var inv_rows = SIMD[DType.float32, WIDTH](Float32(1.0) / Float32(rows))
    k = 0
    while k + WIDTH <= cols:
        var g = (gauge + k).load[width=WIDTH]() * inv_rows
        (gauge + k).store(g)
        (gauge_bf16 + k).store(g.cast[DType.bfloat16]())
        k += WIDTH

    for r in range(rows):
        var row = src + r * cols
        var out = centered_bf16 + r * cols
        k = 0
        while k + WIDTH <= cols:
            var v = (row + k).load[width=WIDTH]().cast[DType.float32]()
            var c = v - (gauge + k).load[width=WIDTH]()
            (out + k).store(c.cast[DType.bfloat16]())
            k += WIDTH
