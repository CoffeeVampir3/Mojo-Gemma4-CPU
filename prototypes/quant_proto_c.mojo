from std.algorithm import vectorize
from std.collections import InlineArray
from std.memory import UnsafePointer, alloc
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
struct CompanionSpec(Copyable, Movable):
    var dtype: DType
    var addr: Int
    var block: Int


@fieldwise_init
struct SourceRead(Copyable, Movable):
    var dtype: DType
    var rows: Int
    var cols: Int
    var addr: Int
    var companion: Optional[CompanionSpec]


def decode_raw[src: DType](
    src_ptr: UnsafePointer[Scalar[src], MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    count: Int,
):
    def step[width: Int](idx: Int) {read}:
        (work + idx).store(
            (src_ptr + idx).load[width=width]().cast[DType.float32]())
    vectorize[W](count, step)


def decode_block_scaled[src: DType, comp: DType](
    src_ptr: UnsafePointer[Scalar[src], MutAnyOrigin],
    comp_ptr: UnsafePointer[Scalar[comp], MutAnyOrigin],
    work: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int, cols: Int, block: Int,
):
    var tiles_c = cols // block
    var tiles_r = rows // block
    for tr in range(tiles_r):
        for tc in range(tiles_c):
            var s = Float32(comp_ptr[tr * tiles_c + tc])
            var vs = SIMD[DType.float32, W](s)
            var r0 = tr * block
            var c0 = tc * block
            for r_in in range(block):
                var off = (r0 + r_in) * cols + c0
                var c = 0
                while c < block:
                    var v = (src_ptr + off + c).load[width=W]().cast[DType.float32]()
                    (work + off + c).store(v * vs)
                    c += W


def decode_with_companion[src: DType](
    sr: SourceRead, work: UnsafePointer[Float32, MutAnyOrigin],
) -> Bool:
    var src_ptr = UnsafePointer[Scalar[src], MutAnyOrigin](
        unsafe_from_address=sr.addr)
    ref comp = sr.companion.value()
    comptime for i in range(NUM_SRC_DTYPES):
        comptime CDT = SOURCE_DTYPES[i]
        if comp.dtype == CDT:
            var comp_ptr = UnsafePointer[Scalar[CDT], MutAnyOrigin](
                unsafe_from_address=comp.addr)
            decode_block_scaled[src, CDT](
                src_ptr, comp_ptr, work, sr.rows, sr.cols, comp.block)
            return True
    return False


def decode_raw_dispatch[src: DType](
    sr: SourceRead, work: UnsafePointer[Float32, MutAnyOrigin],
):
    var src_ptr = UnsafePointer[Scalar[src], MutAnyOrigin](
        unsafe_from_address=sr.addr)
    decode_raw[src](src_ptr, work, sr.rows * sr.cols)


def decode(sr: SourceRead, work: UnsafePointer[Float32, MutAnyOrigin]) -> Bool:
    comptime for i in range(NUM_SRC_DTYPES):
        comptime SRC = SOURCE_DTYPES[i]
        if sr.dtype == SRC:
            if sr.companion:
                return decode_with_companion[SRC](sr, work)
            decode_raw_dispatch[SRC](sr, work)
            return True
    print(t"unsupported src dtype {sr.dtype}")
    return False


def main():
    var rows = 8
    var cols = 8
    var n = rows * cols
    var block = 4

    var src = alloc[Scalar[DType.float8_e4m3fn]](n)
    var comp = alloc[Float32]((rows // block) * (cols // block))
    var work = alloc[Float32](n)

    for i in range(n):
        src[i] = Scalar[DType.float8_e4m3fn](Float32(1.5))
    for i in range((rows // block) * (cols // block)):
        comp[i] = Float32(2.0)

    var sr = SourceRead(
        dtype=DType.float8_e4m3fn, rows=rows, cols=cols,
        addr=Int(src),
        companion=Optional[CompanionSpec](CompanionSpec(
            dtype=DType.float32, addr=Int(comp), block=block)),
    )
    if not decode(sr, work):
        print("FAIL companion")
        return
    print(t"fp8 + scale: work[0]={work[0]}  work[63]={work[n-1]}")

    var bf = alloc[Scalar[DType.bfloat16]](n)
    var work2 = alloc[Float32](n)
    for i in range(n):
        bf[i] = Scalar[DType.bfloat16](Float32(0.25 + i * 0.0625))
    var sr2 = SourceRead(
        dtype=DType.bfloat16, rows=rows, cols=cols,
        addr=Int(bf),
        companion=Optional[CompanionSpec](),
    )
    if not decode(sr2, work2):
        print("FAIL raw")
        return
    print(t"bf16 raw  : work2[0]={work2[0]}  work2[63]={work2[n-1]}")

    src.free()
    comp.free()
    work.free()
    bf.free()
    work2.free()
