from std.memory import UnsafePointer, alloc
from std.collections import InlineArray
from std.algorithm import vectorize
from std.benchmark import keep
from std.sys.info import simd_width_of

from simd_math.ops import sqrt


comptime W = simd_width_of[DType.float32]()
comptime HIDDEN = 2816

comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]


@always_inline
def port_unroll_for[count: Int]() -> Int:
    comptime assert count > 0, "port_unroll_for requires positive count"
    return 8 if count >= 8 else 4 if count >= 4 else 2 if count >= 2 else 1


@always_inline
def pick_port_unroll[width: Int, cols: Int]() -> Int:
    comptime assert cols >= width, "pick_port_unroll requires cols >= width"
    return port_unroll_for[cols // width]()


@always_inline
def tree_reduce_accs[T: DType, width: Int, port_unroll: Int, //](
    mut accs: InlineArray[SIMD[T, width], port_unroll],
) -> Scalar[T]:
    comptime for stride in range(1, port_unroll):
        comptime if (stride & (stride - 1)) == 0:
            comptime for i in range(0, port_unroll, 2 * stride):
                accs[i] += accs[i + stride]
    return accs[0].reduce_add()


@always_inline
def rms_reduce_row(src: BF16Ptr) -> Float32:
    comptime PU = pick_port_unroll[W, HIDDEN]()
    comptime STRIDE = PU * W
    var accs = InlineArray[SIMD[DType.float32, W], PU](fill=SIMD[DType.float32, W](0))
    for i in range(HIDDEN // STRIDE):
        comptime for p in range(PU):
            var v = (src + i * STRIDE + p * W).load[width=W]().cast[DType.float32]()
            accs[p] = v.fma(v, accs[p])
    return tree_reduce_accs(accs)


@always_inline
def rms_normalize_row(
    src: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr,
    inv_rms: Float32,
):
    def step[width: Int](idx: Int) {read}:
        var x = (src + idx).load[width=width]().cast[DType.float32]()
        var factor = SIMD[DType.float32, width](inv_rms)
        var w = (weight + idx).load[width=width]().cast[DType.float32]()
        (dst + idx).store((x * factor * w).cast[DType.bfloat16]())

    vectorize[W](HIDDEN, step)


comptime SQRT_N = sqrt[DType.float32, 1](HIDDEN)
comptime N_EPS = HIDDEN * 1e-6


@no_inline
def rms_norm_row(src: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr):
    var sum_sq = rms_reduce_row(src)
    var inv_rms = SQRT_N / sqrt[DType.float32, 1](sum_sq + N_EPS)
    rms_normalize_row(src, dst, weight, inv_rms)


@no_inline
def run_case() -> Int:
    var src = alloc[BFloat16](HIDDEN)
    var dst = alloc[BFloat16](HIDDEN)
    var weight = alloc[BFloat16](HIDDEN)

    for i in range(HIDDEN):
        src[i] = BFloat16(Float32(Float64(i % 127 - 63) * 0.01))
        weight[i] = BFloat16(Float32(Float64(i % 53 + 1) * 0.02))

    rms_norm_row(src, dst, weight)

    var checksum = Int(0)
    for i in range(HIDDEN):
        checksum += Int(dst[i].cast[DType.int32]())

    src.free()
    dst.free()
    weight.free()
    return checksum


def main():
    keep(run_case())
