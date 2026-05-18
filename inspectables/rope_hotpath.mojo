from std.memory import UnsafePointer, alloc
from std.benchmark import keep
from std.sys.info import simd_width_of


comptime W = simd_width_of[DType.float32]()
comptime HALF = 128
comptime PAIR_STRIDE = 128
comptime NUM_HEADS = 16
comptime HEAD_DIM = 256

comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]


@always_inline
def rotate_pair[width: Int, pair_stride: Int](
    ptr: BF16Ptr, cos_ptr: F32Ptr, sin_ptr: F32Ptr, j: Int,
):
    var x_lo = (ptr + j).load[width=width]().cast[DType.float32]()
    var x_hi = (ptr + pair_stride + j).load[width=width]().cast[DType.float32]()
    var cv = (cos_ptr + j).load[width=width]()
    var sv = (sin_ptr + j).load[width=width]()
    (ptr + j).store((x_lo * cv - x_hi * sv).cast[DType.bfloat16]())
    (ptr + pair_stride + j).store((x_hi * cv + x_lo * sv).cast[DType.bfloat16]())


@no_inline
def rope_token(data: BF16Ptr, cos_row: F32Ptr, sin_row: F32Ptr):
    for h in range(NUM_HEADS):
        var head_ptr = data + h * HEAD_DIM
        for j in range(0, HALF, W):
            rotate_pair[W, PAIR_STRIDE](head_ptr, cos_row, sin_row, j)


@no_inline
def run_case() -> Int:
    var data = alloc[BFloat16](NUM_HEADS * HEAD_DIM)
    var cos_row = alloc[Float32](HALF)
    var sin_row = alloc[Float32](HALF)

    for i in range(NUM_HEADS * HEAD_DIM):
        data[i] = BFloat16(Float32(Float64(i % 127 - 63) * 0.01))
    for i in range(HALF):
        cos_row[i] = Float32(Float64(i % 50) * 0.02)
        sin_row[i] = Float32(Float64(i % 50) * 0.01)

    rope_token(data, cos_row, sin_row)

    var checksum = Int(0)
    for i in range(NUM_HEADS * HEAD_DIM):
        checksum += Int(data[i].cast[DType.int32]())

    data.free()
    cos_row.free()
    sin_row.free()
    return checksum


def main():
    keep(run_case())
