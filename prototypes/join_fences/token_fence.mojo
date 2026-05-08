from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from kernels.helpers import (
    DispatchBuffer, tile_dispatch, join_all,
    worker_range, recommended_workers,
)
from kernels.gemv import GemvKernel, gemv_range, GEMV_INLINE_ROWS
from kernels.rmsnorm import rms_norm_row, RmsNormTokenKernel, NORM_INLINE_TOKENS
from kernels.kv_tiled_attention import FlashDecodeKernel, FLASH_PARTIAL_STRIDE
from kernels.logsum_merge import FinalizeKernel, finalize_head
from simd_math.ops import sqrt
from modeling.gemma4_common import Gemma4BaseConfig


comptime C = Gemma4BaseConfig
comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]


def nojoin_gemv[P: BurstThreadPool, //, rows: Int, cols: Int](
    x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr, mut pool: P,
):
    if rows <= GEMV_INLINE_ROWS:
        gemv_range[rows, cols](x, weight, output, 0, rows)
        return
    var data_bytes = rows * cols * 2
    var nw = recommended_workers(data_bytes, pool.get_capacity())
    var buf = DispatchBuffer[GemvKernel[rows, cols]]()
    tile_dispatch(buf,
        GemvKernel[rows, cols](x, weight, output, 0, 0),
        pool, rows, num_workers=nw)


def nojoin_rms_norm[
    P: BurstThreadPool, //,
    hidden: Int, sqrt_n: Scalar[DType.float32], n_eps: Scalar[DType.float32],
](
    src: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr,
    count: Int, mut pool: P,
):
    if count <= NORM_INLINE_TOKENS:
        for tok in range(count):
            rms_norm_row[hidden, sqrt_n, n_eps](
                src + tok * hidden, dst + tok * hidden, weight)
        return
    var nw = recommended_workers(count * hidden * 2, pool.get_capacity())
    var buf = DispatchBuffer[RmsNormTokenKernel[hidden, sqrt_n, n_eps]]()
    tile_dispatch(buf,
        RmsNormTokenKernel[hidden, sqrt_n, n_eps](src, dst, weight, 0, 0),
        pool, count, num_workers=nw)


def nojoin_sliding_attention[
    P: BurstThreadPool, //,
    head_dim: Int, num_q: Int, gqa_ratio: Int,
    kv_stride: Int, window: Int,
](
    q: BF16Ptr, k_base: BF16Ptr, v_base: BF16Ptr,
    partials_buf: F32Ptr,
    pos: Int, valid_len: Int,
    mut pool: P,
) -> Int:
    if valid_len <= 0:
        return 0
    var start_pos = pos - valid_len + 1
    var nw = recommended_workers(
        valid_len * kv_stride * 2, pool.get_capacity())
    var buf = DispatchBuffer[
        FlashDecodeKernel[head_dim, num_q, gqa_ratio, kv_stride, window]]()
    for w_idx in range(nw):
        var wr = worker_range(valid_len, nw, w_idx)
        buf.slot()[] = FlashDecodeKernel[
            head_dim, num_q, gqa_ratio, kv_stride, window](
            q, k_base, v_base, partials_buf,
            FLASH_PARTIAL_STRIDE[num_q, head_dim], w_idx, start_pos,
            wr[0], wr[1])
    buf.dispatch(pool)
    return nw


def nojoin_merge[
    P: BurstThreadPool, //, head_dim: Int, num_q: Int,
](
    output: BF16Ptr, partials_buf: F32Ptr,
    partial_stride: Int, num_sources: Int,
    mut pool: P, head_offset: Int = 0,
):
    if num_sources <= 0 or num_q <= 8:
        for local_h in range(num_q):
            finalize_head[head_dim, num_q](
                output, partials_buf, partial_stride,
                num_sources, head_offset + local_h, local_h)
        return
    var nw = min(num_q, pool.get_capacity())
    var buf = DispatchBuffer[FinalizeKernel[head_dim, num_q]]()
    tile_dispatch(buf,
        FinalizeKernel[head_dim, num_q](
            output, partials_buf, partial_stride,
            num_sources, head_offset, 0, 0),
        pool, num_q, num_workers=nw)


def simulate_sliding_layer_nojoin[P: BurstThreadPool, //, degree: Int](
    x_ptrs: InlineArray[BF16Ptr, degree],
    norm_dst_ptrs: InlineArray[BF16Ptr, degree],
    norm_weights: InlineArray[BF16Ptr, degree],
    q_ptrs: InlineArray[BF16Ptr, degree],
    q_weights: InlineArray[BF16Ptr, degree],
    k_bases: InlineArray[BF16Ptr, degree],
    v_bases: InlineArray[BF16Ptr, degree],
    o_weights: InlineArray[BF16Ptr, degree],
    partials: InlineArray[F32Ptr, degree],
    pos: Int,
    valid_len: Int,
    mut pools: HeapMoveArray[P],
):
    comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
    comptime n_eps = Scalar[DType.float32](C.HIDDEN * C.RMS_NORM_EPS)
    comptime q_rows = C.Q_DIM_SLIDING
    comptime kv_rows = C.KV_DIM_SLIDING
    comptime head_dim = C.HEAD_DIM_SLIDING
    comptime num_q = q_rows // head_dim
    comptime num_kv = kv_rows // head_dim
    comptime gqa = num_q // num_kv
    comptime flash_stride = FLASH_PARTIAL_STRIDE[num_q, head_dim]

    # Phase 1: input norm
    for rank in range(degree):
        nojoin_rms_norm[hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps](
            x_ptrs[rank], norm_dst_ptrs[rank], norm_weights[rank],
            1, pools[rank])
    join_all[degree](pools)

    # Phase 2: QKV gemv
    for rank in range(degree):
        nojoin_gemv[rows=q_rows, cols=C.HIDDEN](
            norm_dst_ptrs[rank], q_weights[rank], q_ptrs[rank],
            pools[rank])
    join_all[degree](pools)

    # Phase 3: attention
    var nws = InlineArray[Int, degree](fill=0)
    for rank in range(degree):
        nws[rank] = nojoin_sliding_attention[
            head_dim=head_dim, num_q=num_q,
            gqa_ratio=gqa, kv_stride=kv_rows, window=C.SLIDING_WINDOW](
            q_ptrs[rank], k_bases[rank], v_bases[rank],
            partials[rank], pos, valid_len, pools[rank])
    join_all[degree](pools)

    # Phase 4: merge
    for rank in range(degree):
        nojoin_merge[head_dim=head_dim, num_q=num_q](
            q_ptrs[rank], partials[rank],
            flash_stride, nws[rank], pools[rank])
    join_all[degree](pools)

    # Phase 5: o_proj
    for rank in range(degree):
        nojoin_gemv[rows=C.HIDDEN, cols=q_rows](
            q_ptrs[rank], o_weights[rank], x_ptrs[rank],
            pools[rank])
    join_all[degree](pools)

    # VERDICT for the no-fence / nojoin approach:
    #
    # + Dead simple. No fence struct, no storage, no origin issues.
    # + Each phase is 3 lines: dispatch loop, join_all. Reads cleanly.
    # + nw return from attention "just works" — it's a plain Int.
    # + join_all on pools with no active work is a no-op per pool
    #   (active_jobs==0 → loop body never executes).
    # + Matches the prototype benchmark pattern exactly.
    #
    # - No compiler enforcement that the caller joins. If you forget
    #   join_all, silent data race. The fence designs catch this at
    #   compile time via @explicit_destroy.
    #
    # - The dispatch functions are "fire and forget" from their own
    #   perspective — the join responsibility is entirely on the caller.
    #   For single-rank callers, this means they must remember to call
    #   pool.join() themselves, which is easy to forget.
    #
    # - The inline path still returns normally (no fence), so if the
    #   function ran inline and the caller calls join_all, it's a
    #   harmless no-op. This is correct but not self-documenting.
    #
    # POSSIBLE HYBRID: The nojoin_ functions are the internal
    # primitives. The public dispatch_xyz functions wrap them with
    # pool.join() for single-rank callers. The forward uses nojoin_
    # directly. This splits the API into "composable" and "convenient"
    # layers without needing fences at all.
