from std.algorithm import vectorize
from std.collections import InlineArray
from std.memory import UnsafePointer, alloc
from std.sys.info import simd_width_of, size_of


comptime NUM_SRC_DTYPES = 14
comptime SUPPORTED_SOURCE_DTYPES: InlineArray[DType, NUM_SRC_DTYPES] = [
    DType.bfloat16, DType.float16, DType.float32, DType.float64,
    DType.int8, DType.uint8, DType.int16, DType.uint16,
    DType.int32, DType.uint32, DType.int64, DType.uint64,
    DType.float8_e4m3fn, DType.float8_e5m2,
]

comptime W = simd_width_of[DType.float32]()


def is_supported_source_dtype(dt: DType) -> Bool:
    comptime for i in range(NUM_SRC_DTYPES):
        if dt == SUPPORTED_SOURCE_DTYPES[i]:
            return True
    return False


def decode_to_f32[src: DType](
    src_ptr: UnsafePointer[Scalar[src], MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    count: Int,
):
    def step[width: Int](idx: Int) {read}:
        (dst + idx).store(
            (src_ptr + idx).load[width=width]().cast[DType.float32]())
    vectorize[W](count, step)


def decode_panel(
    dt: DType,
    src_bytes: UnsafePointer[UInt8, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    count: Int,
) -> Bool:
    comptime for i in range(NUM_SRC_DTYPES):
        comptime SRC = SUPPORTED_SOURCE_DTYPES[i]
        if dt == SRC:
            decode_to_f32[SRC](src_bytes.bitcast[Scalar[SRC]](), dst, count)
            return True
    return False


def element_bytes(dt: DType) -> Int:
    comptime for i in range(NUM_SRC_DTYPES):
        comptime SRC = SUPPORTED_SOURCE_DTYPES[i]
        if dt == SRC:
            return size_of[Scalar[SRC]]()
    return -1


def main():
    var n = 8
    var src = alloc[Scalar[DType.bfloat16]](n)
    for i in range(n):
        src[i] = Scalar[DType.bfloat16](Float32(i + 1))
    var dst = alloc[Float32](n)
    if not decode_panel(DType.bfloat16, src.bitcast[UInt8](), dst, n):
        print("FAIL: bf16 unsupported")
        return
    for i in range(n):
        print(t"bf16[{i}] -> {dst[i]}")

    print(t"is_supported bf16 = {is_supported_source_dtype(DType.bfloat16)}")
    print(t"is_supported bool = {is_supported_source_dtype(DType.bool)}")
    print(t"bytes bf16     = {element_bytes(DType.bfloat16)}")
    print(t"bytes f32      = {element_bytes(DType.float32)}")
    print(t"bytes f8 e4m3  = {element_bytes(DType.float8_e4m3fn)}")
    src.free()
    dst.free()
