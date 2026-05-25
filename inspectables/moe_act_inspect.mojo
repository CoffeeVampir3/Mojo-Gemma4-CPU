from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys import llvm_intrinsic
from std.sys.info import simd_width_of
from std.benchmark import keep


@always_inline
def roundeven[dtype: DType, width: Int](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return llvm_intrinsic[
        "llvm.nearbyint",
        SIMD[dtype, width],
        SIMD[dtype, width],
    ](x)


@always_inline
def exp_f32_fast[width: Int](
    x: SIMD[DType.float32, width],
) -> SIMD[DType.float32, width]:
    comptime LN2 = Float32(0.6931471805599453)
    comptime INV_LN2 = Float32(1.4426950408889634)
    var xc = max(x, SIMD[DType.float32, width](-87.0))
    var xn = xc * INV_LN2
    var n = roundeven(xn).cast[DType.int32]()
    var r = xc - n.cast[DType.float32]() * LN2
    var p = SIMD[DType.float32, width](1.0) + r * (
        Float32(0.9999) + r * (
        Float32(0.4985) + r * (
        Float32(0.1681))))
    var n_clamped = max(n, SIMD[DType.int32, width](-126))
    var pow2n = SIMD[DType.float32, width](
        from_bits=(n_clamped + 127).cast[DType.uint32]() << 23
    )
    return p * pow2n


@always_inline
def silu_f32[width: Int](
    x: SIMD[DType.float32, width],
) -> SIMD[DType.float32, width]:
    return x / (SIMD[DType.float32, width](1.0) + exp_f32_fast(-x))


def port_unroll_for[count: Int]() -> Int:
    comptime if count >= 8:
        return 8
    elif count >= 4:
        return 4
    elif count >= 2:
        return 2
    else:
        return 1


def pick_port_unroll[width: Int, cols: Int]() -> Int:
    comptime if cols % (8 * width) == 0:
        return 8
    elif cols % (4 * width) == 0:
        return 4
    elif cols % (2 * width) == 0:
        return 2
    else:
        return 1


@always_inline
def activation_loop[
    chunk_size: Int,
    act_fn: def[w: Int](SIMD[DType.float32, w]) thin
        -> SIMD[DType.float32, w],
](
    gate: UnsafePointer[Float32, MutAnyOrigin],
    up: UnsafePointer[Float32, MutAnyOrigin],
):
    """Mirror of the moe_gateup activation port-unroll loop body."""
    comptime width = simd_width_of[DType.float32]()
    comptime PU = pick_port_unroll[width, chunk_size]()
    comptime STRIDE = PU * width
    for i in range(chunk_size // STRIDE):
        comptime for p in range(PU):
            comptime off = p * width
            var idx = i * STRIDE + off
            var g = (gate + idx).load[width=width]()
            var u = (up + idx).load[width=width]()
            (gate + idx).store(act_fn[width](g) * u)


def silu_mul_64(
    gate: UnsafePointer[Float32, MutAnyOrigin],
    up: UnsafePointer[Float32, MutAnyOrigin],
):
    activation_loop[64, silu_f32](gate, up)


def silu_mul_128(
    gate: UnsafePointer[Float32, MutAnyOrigin],
    up: UnsafePointer[Float32, MutAnyOrigin],
):
    activation_loop[128, silu_f32](gate, up)


def main():
    var g = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()
    var u = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()
    silu_mul_64(g, u)
    silu_mul_128(g, u)
    keep(g)
    keep(u)
