"""Assembly probe for `kernels/logsum_merge.mojo` hot paths.

Build:
    pixi run mojo build -D ASSERT=none inspectables/logsum_merge_hotpaths.mojo

Emit target-specific assembly:
    pixi run mojo build -D ASSERT=none --target-cpu sapphirerapids --emit asm inspectables/logsum_merge_hotpaths.mojo -o /tmp/logsum_merge_hotpaths_spr.s
    pixi run mojo build -D ASSERT=none --target-cpu znver4 --emit asm inspectables/logsum_merge_hotpaths.mojo -o /tmp/logsum_merge_hotpaths_znver4.s

Inspect:
    objdump -Cd logsum_merge_hotpaths | less

Look for the `probe_finalize_bf16` and `probe_reduce_f32` symbols.  The file is
self-contained on purpose: it copies the small helpers used by logsum_merge so
the disassembly is not polluted by threading/dispatch machinery.
"""

from std.benchmark import keep
from std.collections import InlineArray
from std.memory import UnsafePointer, alloc
from std.sys.info import simd_width_of


comptime W = simd_width_of[DType.float32]()
comptime HEAD_DIM = 512
comptime NUM_Q = 16
comptime NUM_SOURCES = 17
comptime M_OFF = NUM_Q * HEAD_DIM
comptime L_OFF = M_OFF + NUM_Q
comptime PARTIAL_STRIDE = ((M_OFF + NUM_Q + NUM_Q) * 4 + 63) // 64 * 16

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


@always_inline
def merge_accumulate[head_dim: Int, num_q: Int](
    base: F32Ptr,
    stride: Int,
    num_sources: Int,
    h: Int,
    mut acc: InlineArray[SIMD[DType.float32, W], head_dim // W],
) -> Tuple[Float32, Float32]:
    comptime m_off = num_q * head_dim
    comptime l_off = m_off + num_q
    comptime PU = pick_port_unroll[W, head_dim]()
    comptime STRIDE = PU * W

    var global_m = Float32(-1e30)
    for s in range(num_sources):
        var sm = (base + s * stride + m_off + h)[]
        if sm > global_m:
            global_m = sm

    var global_l = Float32(0)
    var first = True

    var batch_start = 0
    while batch_start < num_sources:
        var batch_end = min(batch_start + W, num_sources)
        var batch_len = batch_end - batch_start
        var deltas = SIMD[DType.float32, W](-1e30)
        var batch_ls = SIMD[DType.float32, W](0)
        for b in range(batch_len):
            var sp = base + (batch_start + b) * stride
            deltas[b] = (sp + m_off + h)[] - global_m
            batch_ls[b] = (sp + l_off + h)[]
        var corrs = fast_exp_softmax_biased[W](deltas)
        corrs = batch_ls.gt(SIMD[DType.float32, W](0)).select(
            corrs, SIMD[DType.float32, W](0)
        )
        global_l += (batch_ls * corrs).reduce_add()

        for b in range(batch_len):
            var c = corrs[b]
            if c <= 0:
                continue
            var cv = SIMD[DType.float32, W](c)
            var src = base + (batch_start + b) * stride + h * head_dim
            if first:
                for i in range(head_dim // STRIDE):
                    comptime for p in range(PU):
                        acc[i * PU + p] = (
                            src + i * STRIDE + p * W
                        ).load[width=W]() * cv
                first = False
            else:
                for i in range(head_dim // STRIDE):
                    comptime for p in range(PU):
                        var v = (src + i * STRIDE + p * W).load[width=W]()
                        acc[i * PU + p] = v.fma(cv, acc[i * PU + p])
        batch_start += W

    return (global_m, global_l)


@always_inline
def finalize_head[head_dim: Int, num_q: Int](
    dst: BF16Ptr,
    base: F32Ptr,
    stride: Int,
    num_sources: Int,
    h: Int,
    local_h: Int,
):
    comptime PU = pick_port_unroll[W, head_dim]()
    comptime STRIDE = PU * W
    comptime LANES = head_dim // W

    var acc = InlineArray[SIMD[DType.float32, W], LANES](
        uninitialized=True
    )
    var result = merge_accumulate[head_dim, num_q](
        base, stride, num_sources, h, acc
    )
    var global_l = result[1]

    var out = dst + local_h * head_dim
    if global_l <= 0:
        for i in range(head_dim // STRIDE):
            comptime for p in range(PU):
                (out + i * STRIDE + p * W).store(SIMD[DType.bfloat16, W](0))
        return

    var inv_l = SIMD[DType.float32, W](Float32(1.0) / global_l)
    for i in range(head_dim // STRIDE):
        comptime for p in range(PU):
            (out + i * STRIDE + p * W).store(
                (acc[i * PU + p] * inv_l).cast[DType.bfloat16]()
            )


@always_inline
def reduce_head[head_dim: Int, num_q: Int](
    dst: F32Ptr,
    base: F32Ptr,
    stride: Int,
    num_sources: Int,
    h: Int,
):
    comptime m_off = num_q * head_dim
    comptime l_off = m_off + num_q
    comptime PU = pick_port_unroll[W, head_dim]()
    comptime STRIDE = PU * W
    comptime LANES = head_dim // W

    var acc = InlineArray[SIMD[DType.float32, W], LANES](
        uninitialized=True
    )
    var result = merge_accumulate[head_dim, num_q](
        base, stride, num_sources, h, acc
    )

    var out = dst + h * head_dim
    if result[1] <= 0:
        for i in range(head_dim // STRIDE):
            comptime for p in range(PU):
                (out + i * STRIDE + p * W).store(SIMD[DType.float32, W](0))
        (dst + m_off + h)[] = result[0]
        (dst + l_off + h)[] = result[1]
        return

    for i in range(head_dim // STRIDE):
        comptime for p in range(PU):
            (out + i * STRIDE + p * W).store(acc[i * PU + p])
    (dst + m_off + h)[] = result[0]
    (dst + l_off + h)[] = result[1]


@no_inline
def probe_finalize_bf16(
    output: BF16Ptr,
    partials: F32Ptr,
    stride: Int,
    num_sources: Int,
    head_offset: Int,
):
    for local_h in range(NUM_Q):
        finalize_head[HEAD_DIM, NUM_Q](
            output,
            partials,
            stride,
            num_sources,
            head_offset + local_h,
            local_h,
        )


@no_inline
def probe_reduce_f32(
    output_partial: F32Ptr,
    partials: F32Ptr,
    stride: Int,
    num_sources: Int,
):
    for h in range(NUM_Q):
        reduce_head[HEAD_DIM, NUM_Q](
            output_partial, partials, stride, num_sources, h
        )


@always_inline
def fill_partials(buf: F32Ptr, stride: Int, num_sources: Int):
    for s in range(num_sources):
        var sp = buf + s * stride
        for i in range(M_OFF):
            sp[i] = Float32(Float64((i + s * 11) % 127 - 63) * 0.001)
        for h in range(NUM_Q):
            (sp + M_OFF + h)[] = Float32(Float64((s + h) % 9) * 0.25)
            (sp + L_OFF + h)[] = Float32(1.0)


@no_inline
def run_case() -> Int:
    var partials = alloc[Float32](NUM_SOURCES * PARTIAL_STRIDE)
    var output_bf16 = alloc[BFloat16](NUM_Q * HEAD_DIM)
    var output_f32 = alloc[Float32](PARTIAL_STRIDE)

    fill_partials(partials, PARTIAL_STRIDE, NUM_SOURCES)
    probe_finalize_bf16(output_bf16, partials, PARTIAL_STRIDE, NUM_SOURCES, 0)
    probe_reduce_f32(output_f32, partials, PARTIAL_STRIDE, NUM_SOURCES)

    var checksum = Int(0)
    for i in range(NUM_Q * HEAD_DIM):
        checksum += Int(output_bf16[i].cast[DType.int32]())
        checksum += Int(output_f32[i].cast[DType.int32]())

    partials.free()
    output_bf16.free()
    output_f32.free()
    return checksum


def main():
    keep(run_case())
