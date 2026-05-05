from std.memory import UnsafePointer, alloc
from std.algorithm import vectorize
from std.collections import InlineArray
from std.benchmark import keep
from std.sys.info import simd_width_of


comptime W = simd_width_of[DType.float32]()
comptime TP = 4
comptime COUNT = 2816

comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
comptime ImmBF16Ptr = UnsafePointer[Scalar[DType.bfloat16], ImmutOrigin(MutAnyOrigin)]


@no_inline
def reduce_sources_to(
    srcs: InlineArray[ImmBF16Ptr, TP],
    dst: BF16Ptr,
    start: Int, end: Int,
):
    def step[width: Int](idx: Int) {read}:
        var pos = start + idx
        var acc = (srcs[0] + pos).load[width=width]().cast[DType.float32]()
        for r in range(1, TP):
            acc += (srcs[r] + pos).load[width=width]().cast[DType.float32]()
        (dst + pos).store(acc.cast[DType.bfloat16]())

    vectorize[W](end - start, step)


@no_inline
def run_case() -> Int:
    var srcs_storage = InlineArray[BF16Ptr, TP](uninitialized=True)
    var srcs = InlineArray[ImmBF16Ptr, TP](uninitialized=True)
    for r in range(TP):
        srcs_storage[r] = alloc[Scalar[DType.bfloat16]](COUNT)
        for i in range(COUNT):
            srcs_storage[r][i] = Scalar[DType.bfloat16](
                Float32(Float64((i + r * 7) % 97 - 48) * 0.01))
        srcs[r] = srcs_storage[r].as_immutable()

    var dst = alloc[Scalar[DType.bfloat16]](COUNT)

    reduce_sources_to(srcs, dst, 0, COUNT)

    var checksum = Int(0)
    for i in range(COUNT):
        checksum += Int(dst[i].cast[DType.int32]())

    for r in range(TP):
        srcs_storage[r].free()
    dst.free()
    return checksum


def main():
    keep(run_case())
