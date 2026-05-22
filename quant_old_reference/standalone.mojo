from std.memory import UnsafePointer
from std.collections import InlineArray
from std.utils import IndexList
from std.sys.info import simd_width_of, size_of
from std.sys import llvm_intrinsic
from std.math import max, min
from std.os import abort


comptime PtrU8   = UnsafePointer[UInt8, MutAnyOrigin]
comptime PtrF32  = UnsafePointer[Float32, MutAnyOrigin]
comptime PtrBF16 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime PtrI8   = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime PtrF8   = UnsafePointer[Scalar[DType.float8_e4m3fn], MutAnyOrigin]
comptime WIDTH   = simd_width_of[DType.float32]()


# =============================================================================
# SIMD primitives
# =============================================================================


@always_inline
def sqrt[dtype: DType, width: Int](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return llvm_intrinsic[
        "llvm.sqrt", SIMD[dtype, width], SIMD[dtype, width],
    ](x)


@always_inline
def roundeven[dtype: DType, width: Int](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return llvm_intrinsic[
        "llvm.nearbyint", SIMD[dtype, width], SIMD[dtype, width],
    ](x)


@always_inline
def quantize_i8[width: Int](
    v: SIMD[DType.float32, width], inv_scale: SIMD[DType.float32, width],
) -> SIMD[DType.int8, width]:
    comptime lo = SIMD[DType.float32, width](-128.0)
    comptime hi = SIMD[DType.float32, width](127.0)
    return min(max(roundeven(v * inv_scale), lo), hi).cast[DType.int8]()


@always_inline
def quantize_i8_scalar(v: Float32, inv_scale: Float32) -> Scalar[DType.int8]:
    var q = roundeven[DType.float32, 1](v * inv_scale)
    return min(max(q, Float32(-128.0)), Float32(127.0)).cast[DType.int8]()


# =============================================================================
# FWHT — block-diagonal in-register butterfly
# =============================================================================


def log2[N: Int]() -> Int:
    comptime if N == 1:
        return 0
    else:
        return 1 + log2[N // 2]()


def butterfly_partner[i: Int, stride: Int]() -> Int:
    return i ^ stride


def butterfly_shuffle[width: Int, stride: Int]() -> IndexList[width]:
    var result = IndexList[width]()
    comptime for i in range(width):
        result[i] = butterfly_partner[i, stride]()
    return result


def fwht_width[T: DType, block: Int]() -> Int:
    comptime hw = simd_width_of[T]()
    comptime if block <= hw:
        return block
    else:
        return hw


@always_inline
def fwht_apply[T: DType, block: Int](
    mut r: InlineArray[
        SIMD[T, fwht_width[T, block]()],
        block // fwht_width[T, block](),
    ],
):
    comptime width = fwht_width[T, block]()
    comptime regs = block // width
    comptime stages = log2[block]()

    comptime for stage in range(stages):
        comptime stride = 1 << stage
        comptime if stride < width:
            comptime mask = butterfly_shuffle[width, stride]()
            var sign_buf = InlineArray[Scalar[T], width](fill=Scalar[T](1.0))
            comptime for k in range(width):
                comptime if (k >> stage) & 1 != 0:
                    sign_buf[k] = Scalar[T](-1.0)
            var sign = UnsafePointer(to=sign_buf).bitcast[Scalar[T]]().load[width=width]()
            comptime for i in range(regs):
                var partner = r[i].shuffle[mask=mask](r[i])
                r[i] = r[i].fma(sign, partner)
        else:
            comptime reg_stride = stride // width
            comptime num_groups = regs // (2 * reg_stride)
            comptime for g in range(num_groups):
                comptime for j in range(reg_stride):
                    comptime a_idx = g * 2 * reg_stride + j
                    comptime b_idx = a_idx + reg_stride
                    var a_val = r[a_idx]
                    var b_val = r[b_idx]
                    r[a_idx] = a_val + b_val
                    r[b_idx] = a_val - b_val

    var sc = Scalar[T](1.0 / Float64(sqrt[T, 1](Scalar[T](block))))
    comptime for i in range(regs):
        r[i] = r[i] * sc


@always_inline
def fwht_block[block: Int](buf: PtrF32):
    comptime width = fwht_width[DType.float32, block]()
    comptime regs = block // width
    var r = InlineArray[SIMD[DType.float32, width], regs](
        fill=SIMD[DType.float32, width](0))
    comptime for i in range(regs):
        r[i] = (buf + i * width).load[width=width]()
    fwht_apply[DType.float32, block](r)
    comptime for i in range(regs):
        (buf + i * width).store(r[i])


@always_inline
def fwht_row[block: Int](buf: PtrF32, cols: Int):
    for b in range(cols // block):
        fwht_block[block](buf + b * block)


def is_supported_fwht_block(block: Int) -> Bool:
    return block == 512 or block == 256 or block == 128 \
        or block == 64 or block == 32 or block == 16


# =============================================================================
# Source-format converters — decode vendor dtype into an f32 work buffer
# =============================================================================


trait Converter:
    comptime SOURCE_DTYPE: DType
    comptime AUX_DTYPE: DType
    comptime SOURCE_ELEMENT_BYTES: Int
    comptime AUX_SUFFIX: StaticString
    comptime AUX_ROW_BLOCK: Int

    @staticmethod
    def aux_bytes_for(rows: Int, cols: Int) -> Int: ...

    @staticmethod
    def convert[dst_dtype: DType](
        src: UnsafePointer[Scalar[Self.SOURCE_DTYPE], MutAnyOrigin],
        aux: UnsafePointer[Scalar[Self.AUX_DTYPE], MutAnyOrigin],
        dst: UnsafePointer[Scalar[dst_dtype], MutAnyOrigin],
        rows: Int, cols: Int,
    ): ...


struct Raw[dtype: DType](Converter):
    comptime SOURCE_DTYPE = Self.dtype
    comptime AUX_DTYPE = DType.uint8
    comptime SOURCE_ELEMENT_BYTES = size_of[Scalar[Self.dtype]]()
    comptime AUX_SUFFIX: StaticString = ""
    comptime AUX_ROW_BLOCK = 0

    @staticmethod
    def aux_bytes_for(rows: Int, cols: Int) -> Int:
        return 0

    @staticmethod
    def convert[dst_dtype: DType](
        src: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        aux: UnsafePointer[UInt8, MutAnyOrigin],
        dst: UnsafePointer[Scalar[dst_dtype], MutAnyOrigin],
        rows: Int, cols: Int,
    ):
        comptime width = simd_width_of[DType.float32]()
        var n = rows * cols
        debug_assert(n % width == 0, "Raw.convert: n must be f32-simd-aligned")
        var k = 0
        while k < n:
            (dst + k).store((src + k).load[width=width]().cast[dst_dtype]())
            k += width


comptime Bf16Converter = Raw[DType.bfloat16]
comptime F32Converter  = Raw[DType.float32]
comptime F16Converter  = Raw[DType.float16]


@always_inline
def e4m3fn_to_f32[width: Int](
    raw8: SIMD[DType.uint8, width],
) -> SIMD[DType.float32, width]:
    var raw = raw8.cast[DType.uint32]()
    var zero = SIMD[DType.uint32, width](0)
    var sign_bits = (raw & SIMD[DType.uint32, width](0x80)) << 24
    var exp = (raw >> 3) & SIMD[DType.uint32, width](0x0F)
    var mant = raw & SIMD[DType.uint32, width](0x07)

    var exp_is_15 = exp.eq(SIMD[DType.uint32, width](15))
    var mant_is_7 = mant.eq(SIMD[DType.uint32, width](7))
    var maxfinite_payload = exp_is_15 & mant_is_7
    var mant_finite = maxfinite_payload.select(
        SIMD[DType.uint32, width](6), mant)

    var normal_bits = (
        sign_bits
        | ((exp + SIMD[DType.uint32, width](120)) << 23)
        | (mant_finite << 20)
    )
    var normal = SIMD[DType.float32, width](from_bits=normal_bits)

    var sign_neg = (raw & SIMD[DType.uint32, width](0x80)).ne(zero)
    var sub_mag = mant.cast[DType.float32]() * SIMD[DType.float32, width](0.001953125)
    var subnormal = sign_neg.select(-sub_mag, sub_mag)
    return exp.eq(zero).select(subnormal, normal)


struct Fp8E4M3Block[block: Int](Converter):
    comptime SOURCE_DTYPE = DType.float8_e4m3fn
    comptime AUX_DTYPE = DType.float32
    comptime SOURCE_ELEMENT_BYTES = 1
    comptime AUX_SUFFIX: StaticString = "_scale_inv"
    comptime AUX_ROW_BLOCK = Self.block
    comptime BLOCK = Self.block

    @staticmethod
    def aux_bytes_for(rows: Int, cols: Int) -> Int:
        return (rows // Self.block) * (cols // Self.block) * 4

    @staticmethod
    def decompress_to_bf16(
        src: PtrF8, aux: PtrF32, dst: PtrBF16,
        rows: Int, cols: Int,
    ):
        comptime width = simd_width_of[DType.float32]()
        debug_assert(Self.block % width == 0,
            "Fp8E4M3Block: block must be f32-simd-aligned")
        debug_assert(rows % Self.block == 0 and cols % Self.block == 0,
            "Fp8E4M3Block: rows/cols must be multiples of block")

        var raw_bytes = src.bitcast[UInt8]()
        var tiles_c = cols // Self.block
        var tiles_r = rows // Self.block

        for tr in range(tiles_r):
            for tc in range(tiles_c):
                var scale_vec = SIMD[DType.float32, width](
                    aux[tr * tiles_c + tc])
                var r0 = tr * Self.block
                var c0 = tc * Self.block
                for r_in in range(Self.block):
                    var row_off = (r0 + r_in) * cols + c0
                    var c = 0
                    while c < Self.block:
                        var raw = (raw_bytes + row_off + c).load[width=width]()
                        var v = e4m3fn_to_f32[width](raw) * scale_vec
                        (dst + row_off + c).store(v.cast[DType.bfloat16]())
                        c += width

    @staticmethod
    def convert[dst_dtype: DType](
        src: PtrF8, aux: PtrF32,
        dst: UnsafePointer[Scalar[dst_dtype], MutAnyOrigin],
        rows: Int, cols: Int,
    ):
        var bf16_buf = List[Scalar[DType.bfloat16]](
            length=rows * cols, fill=Scalar[DType.bfloat16](0))
        var bf16 = PtrBF16(unsafe_from_address=Int(bf16_buf.unsafe_ptr()))
        Self.decompress_to_bf16(src, aux, bf16, rows, cols)
        var null_aux = PtrU8()
        Bf16Converter.convert[dst_dtype](bf16, null_aux, dst, rows, cols)
        _ = bf16_buf^


comptime Fp8E4M3Block128Converter = Fp8E4M3Block[128]


# =============================================================================
# Quantization math kernels
# =============================================================================


@always_inline
def bf16_to_f32(src: PtrU8, dst: PtrF32, count: Int):
    var bp = src.bitcast[Scalar[DType.bfloat16]]()
    var k = 0
    while k + WIDTH <= count:
        (dst + k).store((bp + k).load[width=WIDTH]().cast[DType.float32]())
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
def gamma_sqrt_abs(gamma: PtrF32, cols: Int):
    var k = 0
    while k + WIDTH <= cols:
        var v = (gamma + k).load[width=WIDTH]().__abs__()
        (gamma + k).store(sqrt[DType.float32, WIDTH](v))
        k += WIDTH


def router_center_bf16(
    src: PtrF32, gauge: PtrF32,
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
            (gauge + k).store(
                (gauge + k).load[width=WIDTH]()
                + (row + k).load[width=WIDTH]())
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
            var v = (
                (row + k).load[width=WIDTH]()
                - (gauge + k).load[width=WIDTH]()
            )
            (out + k).store(v.cast[DType.bfloat16]())
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
    for r in range(rows):
        fwht_row[block](work + r * cols, cols)


def fwht_rotate_columns[head_dim: Int](work: PtrF32, rows: Int, cols: Int):
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


def quant_rows_per_row(
    work: PtrF32, qi: PtrI8, scales: PtrF32, rows: Int, cols: Int,
):
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


def rotate_and_quant[per_block: Bool](
    block: Int, work: PtrF32, qi: PtrI8, scales: PtrF32,
    rows: Int, cols: Int, two_sided_head_dim: Int = 0,
):
    @parameter
    def go[b: Int]():
        fwht_rotate_rows[b](work, rows, cols)
        if two_sided_head_dim == 128:
            fwht_rotate_columns[128](work, rows, cols)
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
        abort("rotate_and_quant: unsupported block size")


# =============================================================================
# Top-level orchestrator — full pipeline on caller-owned buffers
# =============================================================================


def quantize_panel[T: Converter, per_block: Bool](
    src: UnsafePointer[Scalar[T.SOURCE_DTYPE], MutAnyOrigin],
    aux: UnsafePointer[Scalar[T.AUX_DTYPE], MutAnyOrigin],
    work: PtrF32,
    qi: PtrI8,
    scales: PtrF32,
    rows: Int,
    cols: Int,
    block: Int,
    gamma: PtrF32 = PtrF32(),
    two_sided_head_dim: Int = 0,
):
    T.convert[DType.float32](src, aux, work, rows, cols)
    if gamma:
        for r in range(rows):
            apply_gamma_in_place(work + r * cols, gamma, cols)
    rotate_and_quant[per_block](
        block, work, qi, scales, rows, cols, two_sided_head_dim)
