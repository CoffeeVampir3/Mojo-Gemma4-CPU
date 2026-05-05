from std.collections import InlineArray
from std.memory import UnsafePointer, memset_zero
from std.sys.info import simd_width_of

from simd_math import pick_port_unroll, fast_exp_softmax_biased
from threading.threading_traits import BurstThreadPool
from .helpers import RangedKernel, DispatchBuffer, tile_dispatch, recommended_workers


comptime F32Ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime W = simd_width_of[DType.float32]()
comptime INLINE_MAX_BYTES = 131072
comptime DISPATCH_SATURATE_BYTES = 1048576


@always_inline
def merge_accumulate[head_dim: Int, num_q: Int](
    base: F32Ptr, stride: Int, num_sources: Int, h: Int,
    mut acc: InlineArray[SIMD[DType.float32, W], head_dim // W],
) -> Tuple[Scalar[DType.float32], Scalar[DType.float32]]:
    comptime m_off = num_q * head_dim
    comptime l_off = m_off + num_q
    comptime PU = pick_port_unroll[W, head_dim]()
    comptime STRIDE = PU * W

    var global_m = Scalar[DType.float32](-1e30)
    for s in range(num_sources):
        var sm = (base + s * stride + m_off + h)[]
        if sm > global_m:
            global_m = sm

    var global_l = Scalar[DType.float32](0)
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
            corrs, SIMD[DType.float32, W](0))
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
                        acc[i * PU + p] = (src + i * STRIDE + p * W).load[width=W]() * cv
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
    dst: BF16Ptr, base: F32Ptr, stride: Int, num_sources: Int,
    h: Int, local_h: Int,
):
    comptime PU = pick_port_unroll[W, head_dim]()
    comptime STRIDE = PU * W
    comptime LANES = head_dim // W

    var acc = InlineArray[SIMD[DType.float32, W], LANES](
        uninitialized=True)
    var result = merge_accumulate[head_dim, num_q](base, stride, num_sources, h, acc)
    var global_l = result[1]

    var out = dst + local_h * head_dim
    if global_l <= 0:
        for i in range(head_dim // STRIDE):
            comptime for p in range(PU):
                (out + i * STRIDE + p * W).store(SIMD[DType.bfloat16, W](0))
        return

    var inv_l = SIMD[DType.float32, W](Scalar[DType.float32](1.0) / global_l)
    for i in range(head_dim // STRIDE):
        comptime for p in range(PU):
            (out + i * STRIDE + p * W).store(
                (acc[i * PU + p] * inv_l).cast[DType.bfloat16]())


@always_inline
def reduce_head[head_dim: Int, num_q: Int](
    dst: F32Ptr, base: F32Ptr, stride: Int, num_sources: Int, h: Int,
):
    comptime m_off = num_q * head_dim
    comptime l_off = m_off + num_q
    comptime PU = pick_port_unroll[W, head_dim]()
    comptime STRIDE = PU * W
    comptime LANES = head_dim // W

    var acc = InlineArray[SIMD[DType.float32, W], LANES](
        uninitialized=True)
    var result = merge_accumulate[head_dim, num_q](base, stride, num_sources, h, acc)

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


@fieldwise_init
struct FinalizeKernel[head_dim: Int, num_q: Int](RangedKernel):
    var output: BF16Ptr
    var partials: F32Ptr
    var partial_stride: Int
    var num_sources: Int
    var head_offset: Int
    var start: Int
    var end: Int

    def execute(mut self):
        for local_h in range(self.start, self.end):
            finalize_head[Self.head_dim, Self.num_q](
                self.output, self.partials, self.partial_stride,
                self.num_sources, self.head_offset + local_h, local_h)

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.output, self.partials, self.partial_stride,
            self.num_sources, self.head_offset, start, end)


@fieldwise_init
struct ReduceKernel[head_dim: Int, num_q: Int](RangedKernel):
    var output: F32Ptr
    var partials: F32Ptr
    var partial_stride: Int
    var num_sources: Int
    var start: Int
    var end: Int

    def execute(mut self):
        for h in range(self.start, self.end):
            reduce_head[Self.head_dim, Self.num_q](
                self.output, self.partials, self.partial_stride,
                self.num_sources, h)

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.output, self.partials, self.partial_stride,
            self.num_sources, start, end)


@always_inline
def merge_workers[num_q: Int](data_bytes: Int, capacity: Int) -> Int:
    if data_bytes >= DISPATCH_SATURATE_BYTES:
        return min(num_q, capacity)
    return min(8, min(num_q, capacity))


def merge_flash_partials[
    P: BurstThreadPool, //,
    head_dim: Int, num_q: Int,
](
    output: BF16Ptr,
    partials_buf: F32Ptr,
    partial_stride: Int,
    num_sources: Int,
    mut pool: P,
    head_offset: Int = 0,
    inline_max_bytes: Int = INLINE_MAX_BYTES,
    num_workers: Int = 0,
):
    if num_sources <= 0:
        memset_zero(output, num_q * head_dim)
        return

    var data_bytes = num_sources * (head_dim + 2) * 4 * num_q
    if data_bytes <= inline_max_bytes and num_workers <= 0:
        for local_h in range(num_q):
            finalize_head[head_dim, num_q](
                output, partials_buf, partial_stride,
                num_sources, head_offset + local_h, local_h)
        return

    var nw = num_workers if num_workers > 0 else merge_workers[num_q](
        data_bytes, pool.get_capacity())
    var buf = DispatchBuffer[FinalizeKernel[head_dim, num_q]]()
    tile_dispatch(buf,
        FinalizeKernel[head_dim, num_q](
            output, partials_buf, partial_stride,
            num_sources, head_offset, 0, 0),
        pool, num_q, num_workers=nw)
    pool.join()


def reduce_partials[
    P: BurstThreadPool, //,
    head_dim: Int, num_q: Int,
](
    output_partial: F32Ptr,
    partials_buf: F32Ptr,
    partial_stride: Int,
    num_sources: Int,
    mut pool: P,
    inline_max_bytes: Int = INLINE_MAX_BYTES,
    num_workers: Int = 0,
):
    if num_sources <= 0:
        memset_zero(output_partial, partial_stride)
        return

    var data_bytes = num_sources * (head_dim + 2) * 4 * num_q
    if data_bytes <= inline_max_bytes and num_workers <= 0:
        for h in range(num_q):
            reduce_head[head_dim, num_q](
                output_partial, partials_buf, partial_stride,
                num_sources, h)
        return

    var nw = num_workers if num_workers > 0 else merge_workers[num_q](
        data_bytes, pool.get_capacity())
    var buf = DispatchBuffer[ReduceKernel[head_dim, num_q]]()
    tile_dispatch(buf,
        ReduceKernel[head_dim, num_q](
            output_partial, partials_buf, partial_stride,
            num_sources, 0, 0),
        pool, num_q, num_workers=nw)
    pool.join()
