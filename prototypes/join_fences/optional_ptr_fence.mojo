from std.collections import InlineArray
from std.memory import Pointer, UnsafePointer
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


@explicit_destroy
struct OptFence[P: BurstThreadPool, origin: MutOrigin](Movable):
    var pool: Optional[Pointer[Self.P, Self.origin]]

    @staticmethod
    def dispatched(ref [origin] pool: P) -> Self:
        return Self {pool: Pointer(to=pool)}

    @staticmethod
    def noop() -> Self:
        return Self {pool: None}

    def join(deinit self):
        if self.pool:
            self.pool.value()[].join()

    def finish(deinit self) -> Int:
        if self.pool:
            self.pool.value()[].join()
            return self.pool.value()[].last_worker_timestamp()
        return 0


def fence_gemv[P: BurstThreadPool, //, rows: Int, cols: Int](
    x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr, mut pool: P,
) -> OptFence[P, origin_of(pool)]:
    if rows <= GEMV_INLINE_ROWS:
        gemv_range[rows, cols](x, weight, output, 0, rows)
        return OptFence.noop()
    var data_bytes = rows * cols * 2
    var nw = recommended_workers(data_bytes, pool.get_capacity())
    var buf = DispatchBuffer[GemvKernel[rows, cols]]()
    tile_dispatch(buf,
        GemvKernel[rows, cols](x, weight, output, 0, 0),
        pool, rows, num_workers=nw)
    return OptFence.dispatched(pool)


def fence_rms_norm[
    P: BurstThreadPool, //,
    hidden: Int, sqrt_n: Scalar[DType.float32], n_eps: Scalar[DType.float32],
](
    src: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr,
    count: Int, mut pool: P,
) -> OptFence[P, origin_of(pool)]:
    if count <= NORM_INLINE_TOKENS:
        for tok in range(count):
            rms_norm_row[hidden, sqrt_n, n_eps](
                src + tok * hidden, dst + tok * hidden, weight)
        return OptFence.noop()
    var nw = recommended_workers(count * hidden * 2, pool.get_capacity())
    var buf = DispatchBuffer[RmsNormTokenKernel[hidden, sqrt_n, n_eps]]()
    tile_dispatch(buf,
        RmsNormTokenKernel[hidden, sqrt_n, n_eps](src, dst, weight, 0, 0),
        pool, count, num_workers=nw)
    return OptFence.dispatched(pool)


def simulate_sliding_layer_opt_fence[P: BurstThreadPool, //, degree: Int](
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

    # Same InlineArray + init_pointee_move ceremony as BoolFence.
    # The OptFence.noop() constructor is slightly cleaner (no pool arg),
    # but the storage problem is identical.

    # ALTERNATIVE APPROACH: skip storing fences entirely. Dispatch
    # functions return fences, but for multi-rank we don't STORE them.
    # Instead: dispatch all ranks, then just call join_all directly.
    # The fences from the inline path are noop — drop them immediately.
    # The fences from the dispatch path — we also drop them, relying on
    # join_all to synchronize instead.
    #
    # This works because: if work was dispatched to pools[rank], then
    # join_all will join it. If work was inline, join_all is a no-op
    # for that pool (active_jobs == 0).
    #
    # The fence's value for multi-rank is then just as a DOCUMENTATION
    # ARTIFACT — it forces the dispatch function to be explicit about
    # whether it dispatched, even if the caller discards it.

    # Phase 1: input norm
    for rank in range(degree):
        fence_rms_norm[hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps](
            x_ptrs[rank], norm_dst_ptrs[rank], norm_weights[rank],
            1, pools[rank])^.join()
        # WAIT — this joins per rank again! We're back to serial.
        # We MUST store the fences to avoid this.

    # OK, so for multi-rank we NEED the storage. Let's try a helper
    # that hides the ceremony.

    # Phase 1 (take 2): input norm with manual fence storage
    var fences = InlineArray[
        OptFence[P, MutAnyOrigin], degree](uninitialized=True)
    for rank in range(degree):
        (UnsafePointer(to=fences[rank])).init_pointee_move(
            fence_rms_norm[hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps](
                x_ptrs[rank], norm_dst_ptrs[rank], norm_weights[rank],
                1, pools[rank]))
    for rank in range(degree):
        (UnsafePointer(to=fences[rank])).take_pointee()^.join()

    # Phase 2: QKV gemv
    for rank in range(degree):
        (UnsafePointer(to=fences[rank])).init_pointee_move(
            fence_gemv[rows=q_rows, cols=C.HIDDEN](
                norm_dst_ptrs[rank], q_weights[rank], q_ptrs[rank],
                pools[rank]))
    for rank in range(degree):
        (UnsafePointer(to=fences[rank])).take_pointee()^.join()

    # Phase 3: attention
    var nws = InlineArray[Int, degree](fill=0)
    for rank in range(degree):
        if valid_len <= 0:
            nws[rank] = 0
            (UnsafePointer(to=fences[rank])).init_pointee_move(
                OptFence[P, MutAnyOrigin].noop())
            continue
        var start_pos = pos - valid_len + 1
        var nw = recommended_workers(
            valid_len * kv_rows * 2, pools[rank].get_capacity())
        nws[rank] = nw
        var buf = DispatchBuffer[
            FlashDecodeKernel[
                head_dim, num_q, gqa, kv_rows, C.SLIDING_WINDOW,
            ]]()
        for w_idx in range(nw):
            var wr = worker_range(valid_len, nw, w_idx)
            buf.slot()[] = FlashDecodeKernel[
                head_dim, num_q, gqa, kv_rows, C.SLIDING_WINDOW](
                q_ptrs[rank], k_bases[rank], v_bases[rank], partials[rank],
                flash_stride, w_idx, start_pos, wr[0], wr[1])
        buf.dispatch(pools[rank])
        (UnsafePointer(to=fences[rank])).init_pointee_move(
            OptFence.dispatched(pools[rank]))
    for rank in range(degree):
        (UnsafePointer(to=fences[rank])).take_pointee()^.join()

    # Phase 4: merge
    for rank in range(degree):
        (UnsafePointer(to=fences[rank])).init_pointee_move(
            fence_merge[head_dim=head_dim, num_q=num_q](
                q_ptrs[rank], partials[rank],
                flash_stride, nws[rank], pools[rank]))
    for rank in range(degree):
        (UnsafePointer(to=fences[rank])).take_pointee()^.join()

    # Phase 5: o_proj
    for rank in range(degree):
        (UnsafePointer(to=fences[rank])).init_pointee_move(
            fence_gemv[rows=C.HIDDEN, cols=q_rows](
                q_ptrs[rank], o_weights[rank], x_ptrs[rank],
                pools[rank]))
    for rank in range(degree):
        (UnsafePointer(to=fences[rank])).take_pointee()^.join()

    # VERDICT for OptFence:
    #
    # The noop() constructor is cleaner than BoolFence.inline(pool) —
    # no pool reference needed for the inline path. Semantically honest:
    # "nothing was dispatched, there is no pool to join."
    #
    # But the multi-rank storage problem is identical to BoolFence.
    # The InlineArray + init_pointee_move + take_pointee ceremony is
    # unavoidable when we need to hold fences across a dispatch loop.
    #
    # Phase 3 (attention) is particularly ugly: the dispatch function
    # returns (nw, fence) so we can't use a wrapper — we inline the
    # DispatchBuffer logic and construct the fence manually. This
    # defeats the purpose of having fence_sliding_attention at all.
    #
    # OBSERVATION: For multi-rank, the fence adds ceremony without
    # adding safety beyond what join_all already provides. The compiler
    # can't verify that we stored the fence correctly in the
    # InlineArray — the init_pointee_move/take_pointee is all unsafe.
    #
    # The fence's real value is for SINGLE-RANK composition: a function
    # that calls dispatch_gemv then dispatch_rms_norm can hold fences
    # and join at the right time. For the multi-rank forward, maybe
    # the right answer is: dispatch functions don't join, and the
    # forward calls join_all explicitly.


def fence_merge[
    P: BurstThreadPool, //, head_dim: Int, num_q: Int,
](
    output: BF16Ptr, partials_buf: F32Ptr,
    partial_stride: Int, num_sources: Int,
    mut pool: P, head_offset: Int = 0,
) -> OptFence[P, origin_of(pool)]:
    if num_sources <= 0 or num_q <= 8:
        for local_h in range(num_q):
            finalize_head[head_dim, num_q](
                output, partials_buf, partial_stride,
                num_sources, head_offset + local_h, local_h)
        return OptFence.noop()
    var nw = min(num_q, pool.get_capacity())
    var buf = DispatchBuffer[FinalizeKernel[head_dim, num_q]]()
    tile_dispatch(buf,
        FinalizeKernel[head_dim, num_q](
            output, partials_buf, partial_stride,
            num_sources, head_offset, 0, 0),
        pool, num_q, num_workers=nw)
    return OptFence.dispatched(pool)
