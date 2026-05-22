from std.memory import UnsafePointer
from std.sys.info import simd_width_of, size_of


comptime PtrU8 = UnsafePointer[UInt8, MutAnyOrigin]
comptime PtrF32 = UnsafePointer[Float32, MutAnyOrigin]
comptime PtrBF16 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime WIDTH = simd_width_of[DType.float32]()


trait Converter:
    comptime SOURCE_DTYPE: DType
    comptime AUX_DTYPE: DType
    comptime SOURCE_ELEMENT_BYTES: Int
    comptime AUX_SUFFIX: StaticString
    comptime AUX_ROW_BLOCK: Int

    @staticmethod
    def aux_bytes_for(rows: Int, cols: Int) -> Int: ...

    @staticmethod
    def convert(
        src: UnsafePointer[Scalar[Self.SOURCE_DTYPE], MutAnyOrigin],
        aux: UnsafePointer[Scalar[Self.AUX_DTYPE], MutAnyOrigin],
        dst: PtrF32,
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
    def convert(
        src: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        aux: UnsafePointer[UInt8, MutAnyOrigin],
        dst: PtrF32,
        rows: Int, cols: Int,
    ):
        var n = rows * cols
        var k = 0
        while k + WIDTH <= n:
            (dst + k).store(
                (src + k).load[width=WIDTH]().cast[DType.float32]())
            k += WIDTH


comptime Bf16Converter = Raw[DType.bfloat16]
comptime F32Converter = Raw[DType.float32]
comptime F16Converter = Raw[DType.float16]


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
    def convert(
        src: UnsafePointer[Scalar[DType.float8_e4m3fn], MutAnyOrigin],
        aux: PtrF32,
        dst: PtrF32,
        rows: Int, cols: Int,
    ):
        debug_assert(Self.block % WIDTH == 0,
            "Fp8E4M3Block: block must be f32-simd-aligned")
        debug_assert(rows % Self.block == 0 and cols % Self.block == 0,
            "Fp8E4M3Block: rows/cols must be multiples of block")

        var raw_bytes = src.bitcast[UInt8]()
        var tiles_c = cols // Self.block
        var tiles_r = rows // Self.block

        for tr in range(tiles_r):
            for tc in range(tiles_c):
                var scale_vec = SIMD[DType.float32, WIDTH](
                    aux[tr * tiles_c + tc])
                var r0 = tr * Self.block
                var c0 = tc * Self.block
                for r_in in range(Self.block):
                    var row_off = (r0 + r_in) * cols + c0
                    var c = 0
                    while c < Self.block:
                        var raw = (raw_bytes + row_off + c).load[width=WIDTH]()
                        var v = e4m3fn_to_f32[WIDTH](raw) * scale_vec
                        (dst + row_off + c).store(v)
                        c += WIDTH


comptime Fp8E4M3Block128Converter = Fp8E4M3Block[128]
