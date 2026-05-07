from std.memory import UnsafePointer, alloc
from std.collections import InlineArray
from std.benchmark import keep
from std.sys.info import simd_width_of


comptime W = simd_width_of[DType.float32]()
comptime COLS = 2816
comptime ROWS = 16

comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


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
def dot_row[cols: Int, PU: Int](
    x: BF16Ptr, weight_row: BF16Ptr,
    mut accs: InlineArray[SIMD[DType.float32, W], PU],
):
    comptime STRIDE = PU * W
    for i in range(cols // STRIDE):
        comptime for p in range(PU):
            var xv = (x + i * STRIDE + p * W).load[width=W]().cast[DType.float32]()
            var wv = (weight_row + i * STRIDE + p * W).load[width=W]().cast[DType.float32]()
            accs[p] = xv.fma(wv, accs[p])


@no_inline
def gemv_range(
    x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr,
    start: Int, end: Int,
):
    comptime PU = pick_port_unroll[W, COLS]()
    var accs = InlineArray[SIMD[DType.float32, W], PU](uninitialized=True)
    for row in range(start, end):
        comptime for p in range(PU):
            accs[p] = SIMD[DType.float32, W](0)
        dot_row[COLS, PU](x, weight + row * COLS, accs)
        (output + row)[] = tree_reduce_accs(accs).cast[DType.bfloat16]()


@no_inline
def run_case() -> Int:
    var x = alloc[Scalar[DType.bfloat16]](COLS)
    var weight = alloc[Scalar[DType.bfloat16]](ROWS * COLS)
    var output = alloc[Scalar[DType.bfloat16]](ROWS)

    for i in range(COLS):
        x[i] = Scalar[DType.bfloat16](Float32(Float64(i % 127 - 63) * 0.01))
    for i in range(ROWS * COLS):
        weight[i] = Scalar[DType.bfloat16](Float32(Float64(i % 97 - 48) * 0.01))

    gemv_range(x, weight, output, 0, ROWS)

    var checksum = Int(0)
    for i in range(ROWS):
        checksum += Int(output[i].cast[DType.int32]())

    x.free()
    weight.free()
    output.free()
    return checksum


def main():
    keep(run_case())
