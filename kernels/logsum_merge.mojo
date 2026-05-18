from std.collections import InlineArray
from std.memory import UnsafePointer, memset_zero

from simd_math import pick_port_unroll, fast_exp_softmax_biased
from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from .helpers import (
    BF16Ptr, F32Ptr, W,
    OutputPartitionedKernel, DispatchBuffer, tile_dispatch,
    fanout_dispatch, join_all, Binding,
)
from .attention_ops import flash_partial_stride
from .dispatch_heuristics import (
    MERGE_INLINE_MAX_BYTES, MERGE_SATURATE_BYTES,
)


@always_inline
def merge_accumulate[head_dim: Int, num_q: Int](
    base: F32Ptr, stride: Int, num_sources: Int, h: Int,
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
    h: Int,
):
    comptime PU = pick_port_unroll[W, head_dim]()
    comptime STRIDE = PU * W
    comptime LANES = head_dim // W

    var acc = InlineArray[SIMD[DType.float32, W], LANES](
        uninitialized=True)
    var result = merge_accumulate[head_dim, num_q](base, stride, num_sources, h, acc)
    var global_l = result[1]

    var out = dst + h * head_dim
    if global_l <= 0:
        for i in range(head_dim // STRIDE):
            comptime for p in range(PU):
                (out + i * STRIDE + p * W).store(SIMD[DType.bfloat16, W](0))
        return

    var inv_l = SIMD[DType.float32, W](Float32(1.0) / global_l)
    for i in range(head_dim // STRIDE):
        comptime for p in range(PU):
            (out + i * STRIDE + p * W).store(
                (acc[i * PU + p] * inv_l).cast[DType.bfloat16]())


@fieldwise_init
struct FinalizeKernel[head_dim: Int, num_q: Int](OutputPartitionedKernel):
    comptime PARTIAL_STRIDE = flash_partial_stride[Self.num_q, Self.head_dim]()

    var output: BF16Ptr
    var partials: F32Ptr
    var num_sources: Int
    var start: Int
    var end: Int

    def execute(mut self):
        for h in range(self.start, self.end):
            finalize_head[Self.head_dim, Self.num_q](
                self.output, self.partials, Self.PARTIAL_STRIDE,
                self.num_sources, h)

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.start = start
        self.end = end


@fieldwise_init
struct ContextFlashMergeConfig[head_dim: Int, num_q: Int, tp: Int]:
    comptime PARTIAL_STRIDE = flash_partial_stride[Self.num_q, Self.head_dim]()

    var output: Binding[BFloat16, Self.tp]
    var partials: Binding[Float32, Self.tp]
    var num_sources: InlineArray[Int, Self.tp]


@always_inline
def merge_context_accumulate[
    head_dim: Int, num_q: Int, tp: Int,
](
    config: UnsafePointer[ContextFlashMergeConfig[head_dim, num_q, tp], _],
    h: Int,
    mut acc: InlineArray[SIMD[DType.float32, W], head_dim // W],
) -> Tuple[Float32, Float32]:
    comptime m_off = num_q * head_dim
    comptime l_off = m_off + num_q
    comptime PU = pick_port_unroll[W, head_dim]()
    comptime STRIDE = PU * W
    comptime PSTRIDE = ContextFlashMergeConfig[head_dim, num_q, tp].PARTIAL_STRIDE

    var global_m = Float32(-1e30)
    for context_rank in range(tp):
        var base = config[].partials[context_rank]
        var ns = config[].num_sources[context_rank]
        for s in range(ns):
            var sm = (base + s * PSTRIDE + m_off + h)[]
            if sm > global_m:
                global_m = sm

    var global_l = Float32(0)
    var first = True
    for context_rank in range(tp):
        var base = config[].partials[context_rank]
        var ns = config[].num_sources[context_rank]
        for s in range(ns):
            var sp = base + s * PSTRIDE
            var l = (sp + l_off + h)[]
            if l <= 0:
                continue
            var corr = fast_exp_softmax_biased[1](
                SIMD[DType.float32, 1]((sp + m_off + h)[] - global_m))[0]
            global_l += l * corr

            var cv = SIMD[DType.float32, W](corr)
            var src = sp + h * head_dim
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

    return (global_m, global_l)


@always_inline
def finalize_context_head[
    head_dim: Int, num_q: Int, local_num_q: Int, tp: Int,
](
    config: UnsafePointer[ContextFlashMergeConfig[head_dim, num_q, tp], _],
    q_rank: Int,
    local_h: Int,
):
    comptime PU = pick_port_unroll[W, head_dim]()
    comptime STRIDE = PU * W
    comptime LANES = head_dim // W

    var global_h = q_rank * local_num_q + local_h
    var acc = InlineArray[SIMD[DType.float32, W], LANES](
        uninitialized=True)
    var result = merge_context_accumulate[head_dim, num_q, tp](
        config, global_h, acc)
    var global_l = result[1]

    var out = config[].output[q_rank] + local_h * head_dim
    if global_l <= 0:
        for i in range(head_dim // STRIDE):
            comptime for p in range(PU):
                (out + i * STRIDE + p * W).store(SIMD[DType.bfloat16, W](0))
        return

    var inv_l = SIMD[DType.float32, W](Float32(1.0) / global_l)
    for i in range(head_dim // STRIDE):
        comptime for p in range(PU):
            (out + i * STRIDE + p * W).store(
                (acc[i * PU + p] * inv_l).cast[DType.bfloat16]())


@fieldwise_init
struct ContextFinalizeKernel[
    head_dim: Int, num_q: Int, local_num_q: Int, tp: Int, cfg_origin: Origin,
](OutputPartitionedKernel):
    var config: UnsafePointer[
        ContextFlashMergeConfig[Self.head_dim, Self.num_q, Self.tp],
        Self.cfg_origin]
    var q_rank: Int
    var start: Int
    var end: Int

    def execute(mut self):
        for local_h in range(self.start, self.end):
            finalize_context_head[
                Self.head_dim, Self.num_q, Self.local_num_q,
                Self.tp,
            ](self.config, self.q_rank, local_h)

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.start = start
        self.end = end


@always_inline
def merge_workers[num_q: Int](data_bytes: Int, capacity: Int) -> Int:
    if data_bytes >= MERGE_SATURATE_BYTES:
        return min(num_q, capacity)
    return min(8, min(num_q, capacity))


def dispatch_merge_flash_partials[
    P: BurstThreadPool, //,
    head_dim: Int, num_q: Int, tp: Int, max_worker_count: Int = 128,
](
    output: Binding[BFloat16, tp],
    partials_buf: Binding[Float32, tp],
    num_sources: InlineArray[Int, tp],
    mut pools: HeapMoveArray[P],
    inline_max_bytes: Int = MERGE_INLINE_MAX_BYTES,
):
    comptime K = FinalizeKernel[head_dim, num_q]
    comptime PSTRIDE = K.PARTIAL_STRIDE
    var buf = DispatchBuffer[K, max_worker_count]()
    for r in range(tp):
        if num_sources[r] <= 0:
            memset_zero(output[r], num_q * head_dim)
            continue
        var data_bytes = num_sources[r] * (head_dim + 2) * 4 * num_q
        if data_bytes <= inline_max_bytes:
            for h in range(num_q):
                finalize_head[head_dim, num_q](
                    output[r], partials_buf[r], PSTRIDE,
                    num_sources[r], h)
            continue
        var nw = merge_workers[num_q](
            data_bytes, min(max_worker_count, pools[r].get_capacity()))
        _ = tile_dispatch(buf,
            K(output[r], partials_buf[r], num_sources[r], 0, 0),
            pools[r], num_q, num_workers=nw)
    join_all[tp](pools)


def dispatch_merge_context_flash_partials[
    P: BurstThreadPool, //,
    head_dim: Int, num_q: Int, local_num_q: Int, tp: Int,
    max_worker_count: Int = 128,
](
    output: Binding[BFloat16, tp],
    partials_buf: Binding[Float32, tp],
    num_sources: InlineArray[Int, tp],
    mut pools: HeapMoveArray[P],
):
    var total_sources = 0
    for r in range(tp):
        total_sources += num_sources[r]

    if total_sources <= 0:
        for r in range(tp):
            memset_zero(output[r], local_num_q * head_dim)
        return

    var cfg = ContextFlashMergeConfig[head_dim, num_q, tp](
        output, partials_buf, num_sources)
    var config = UnsafePointer(to=cfg).as_immutable()
    comptime cfg_ro = ImmutOrigin(origin_of(cfg))
    comptime K = ContextFinalizeKernel[head_dim, num_q, local_num_q, tp, cfg_ro]

    @parameter
    def make(q_rank: Int) -> K:
        return K(config, q_rank, 0, 0)

    fanout_dispatch[
        tp, make,
        max_worker_count=max_worker_count,
        worker_policy=merge_workers[local_num_q],
    ](pools, local_num_q, total_sources * (head_dim + 2) * 4 * local_num_q)
