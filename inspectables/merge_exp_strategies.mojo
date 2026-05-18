from std.memory import UnsafePointer, alloc
from std.collections import InlineArray
from std.benchmark import keep
from std.sys.info import simd_width_of


comptime W = simd_width_of[DType.float32]()
comptime HEAD_DIM = 512
comptime NUM_Q = 16
comptime NUM_SOURCES = 4
comptime M_OFF = NUM_Q * HEAD_DIM
comptime L_OFF = M_OFF + NUM_Q

comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]


@always_inline
def port_unroll_for[count: Int]() -> Int:
    comptime assert count > 0, "port_unroll_for requires positive count"
    return 8 if count >= 8 else 4 if count >= 4 else 2 if count >= 2 else 1


@always_inline
def pick_port_unroll[width: Int, cols: Int]() -> Int:
    comptime assert cols >= width, "pick_port_unroll requires cols >= width"
    return port_unroll_for[cols // width]()


comptime PU = pick_port_unroll[W, HEAD_DIM]()
comptime STRIDE = PU * W


@always_inline
def fast_exp_softmax_biased[width: Int](
    x: SIMD[DType.float32, width],
) -> SIMD[DType.float32, width]:
    comptime A_MAGIC = Float32(12102203.16156148)
    comptime BIAS_F = Float32(1059208216.0)
    comptime INV_TWO23 = Float32(1.0) / Float32(1 << 23)
    comptime QC_A = Float32(1.6501418352127075)
    comptime QC_B = Float32(-0.37554836273193359)
    comptime QC_C = Float32(0.38696467876434326)

    var i = (A_MAGIC * x + BIAS_F).cast[DType.int32]()
    var u = i.cast[DType.uint32]()
    var k = SIMD[DType.float32, width](from_bits=u)
    var fbits = u & SIMD[DType.uint32, width](0x7FFFFF)
    var f = fbits.cast[DType.float32]() * INV_TWO23
    return k * (QC_A + f * (QC_B + f * QC_C))


# === Variant A: runtime num_active, scalar exp, serial accumulator chain ===

@no_inline
def merge_runtime_scalar(
    output: BF16Ptr,
    sources: InlineArray[F32Ptr, NUM_SOURCES],
    num_active: Int,
):
    for local_h in range(NUM_Q):
        var h = local_h

        var global_m = Float32(-1e30)
        for s in range(num_active):
            var sm = (sources[s] + M_OFF + h)[]
            if sm > global_m:
                global_m = sm

        var corrections = InlineArray[Float32, NUM_SOURCES](
            fill=Float32(0))
        var global_l = Float32(0)
        for s in range(num_active):
            var sm = (sources[s] + M_OFF + h)[]
            var sl = (sources[s] + L_OFF + h)[]
            if sl > 0:
                corrections[s] = fast_exp_softmax_biased[1](
                    SIMD[DType.float32, 1](sm - global_m))[0]
                global_l += sl * corrections[s]

        if global_l <= 0:
            continue

        var inv_l = Float32(1.0) / global_l
        var dst = output + local_h * HEAD_DIM

        for i in range(HEAD_DIM // STRIDE):
            comptime for p in range(PU):
                var acc = SIMD[DType.float32, W](0)
                for s in range(num_active):
                    if corrections[s] > 0:
                        var v = (sources[s] + h * HEAD_DIM
                            + i * STRIDE + p * W).load[width=W]()
                        acc += v * SIMD[DType.float32, W](corrections[s])
                (dst + i * STRIDE + p * W).store(
                    (acc * SIMD[DType.float32, W](inv_l)).cast[DType.bfloat16]())


# === Variant B: comptime sources, batched exp, independent accumulators ===

@no_inline
def merge_comptime_batched(
    output: BF16Ptr,
    sources: InlineArray[F32Ptr, NUM_SOURCES],
):
    for local_h in range(NUM_Q):
        var h = local_h

        var global_m = Float32(-1e30)
        comptime for s in range(NUM_SOURCES):
            var sm = (sources[s] + M_OFF + h)[]
            if sm > global_m:
                global_m = sm

        var deltas = SIMD[DType.float32, NUM_SOURCES](0)
        var ls = SIMD[DType.float32, NUM_SOURCES](0)
        comptime for s in range(NUM_SOURCES):
            deltas[s] = (sources[s] + M_OFF + h)[] - global_m
            ls[s] = (sources[s] + L_OFF + h)[]

        var corrections = fast_exp_softmax_biased[NUM_SOURCES](deltas)
        corrections = ls.gt(SIMD[DType.float32, NUM_SOURCES](0)).select(
            corrections, SIMD[DType.float32, NUM_SOURCES](0))
        var global_l = (ls * corrections).reduce_add()

        if global_l <= 0:
            continue

        var inv_l = Float32(1.0) / global_l
        var dst = output + local_h * HEAD_DIM

        for i in range(HEAD_DIM // STRIDE):
            comptime for p in range(PU):
                var accs = InlineArray[SIMD[DType.float32, W], NUM_SOURCES](
                    fill=SIMD[DType.float32, W](0))
                comptime for s in range(NUM_SOURCES):
                    var v = (sources[s] + h * HEAD_DIM
                        + i * STRIDE + p * W).load[width=W]()
                    accs[s] = v * SIMD[DType.float32, W](corrections[s])
                var acc = accs[0] + accs[1]
                acc += accs[2] + accs[3]
                (dst + i * STRIDE + p * W).store(
                    (acc * SIMD[DType.float32, W](inv_l)).cast[DType.bfloat16]())


@no_inline
def run_case() -> Int:
    comptime PARTIAL_ELEMS = M_OFF + NUM_Q + NUM_Q
    var sources = InlineArray[F32Ptr, NUM_SOURCES](uninitialized=True)
    for s in range(NUM_SOURCES):
        sources[s] = alloc[Float32](PARTIAL_ELEMS)
        for i in range(M_OFF):
            sources[s][i] = Float32(Float64(i % 97 - 48) * 0.01)
        for h in range(NUM_Q):
            (sources[s] + M_OFF + h)[] = Float32(Float64(s * 3 + h) * 0.1)
            (sources[s] + L_OFF + h)[] = Float32(Float64(h + 1) * 0.5)

    var out_a = alloc[BFloat16](NUM_Q * HEAD_DIM)
    var out_b = alloc[BFloat16](NUM_Q * HEAD_DIM)

    merge_runtime_scalar(out_a, sources, NUM_SOURCES)
    merge_comptime_batched(out_b, sources)

    var checksum = Int(0)
    for i in range(NUM_Q * HEAD_DIM):
        checksum += Int(out_a[i].cast[DType.int32]())
        checksum += Int(out_b[i].cast[DType.int32]())

    for s in range(NUM_SOURCES):
        sources[s].free()
    out_a.free()
    out_b.free()
    return checksum


def main():
    keep(run_case())
