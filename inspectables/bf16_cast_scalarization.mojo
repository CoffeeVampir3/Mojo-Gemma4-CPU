from std.memory import UnsafePointer, alloc
from std.benchmark import keep
from std.sys.info import simd_width_of


comptime W = simd_width_of[DType.float32]()


@no_inline
def cast_only(
    src: UnsafePointer[Float32, MutAnyOrigin],
) -> SIMD[DType.bfloat16, W]:
    return src.load[width=W]().cast[DType.bfloat16]()


@no_inline
def cast_and_store(
    src: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[BFloat16, MutAnyOrigin],
):
    dst.store(src.load[width=W]().cast[DType.bfloat16]())


@no_inline
def manual_trunc_store(
    src: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[BFloat16, MutAnyOrigin],
):
    var bits = src.load[width=W]().to_bits() >> 16
    var narrow = bits.cast[DType.uint16]()
    dst.bitcast[UInt16]().store(narrow)


def main():
    var src = alloc[Float32](W)
    var dst = alloc[BFloat16](W)
    for i in range(W):
        src[i] = Float32(Float64(i) * 0.1)
    var r = cast_only(src)
    cast_and_store(src, dst)
    manual_trunc_store(src, dst)
    keep(r)
    keep(dst[0])
    src.free()
    dst.free()
