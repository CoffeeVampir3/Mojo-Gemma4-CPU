from std.collections import InlineArray
from std.memory import Pointer, UnsafePointer, Span
from std.sys.info import simd_width_of

from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from kernels.helpers import (
    DispatchBuffer, OutputPartitionedKernel, tile_dispatch,
    join_all, worker_range, recommended_workers,
)
from kernels.gemv import GemvKernel, gemv_range, GEMV_INLINE_ROWS
from kernels.rmsnorm import (
    RmsNormTokenKernel, rms_norm_row,
    dispatch_rms_norm, NORM_INLINE_TOKENS,
)
from kernels.kv_tiled_attention import (
    FlashDecodeKernel, FLASH_PARTIAL_STRIDE,
)
from kernels.logsum_merge import FinalizeKernel, finalize_head
from simd_math.ops import sqrt
from modeling.gemma4_common import Gemma4BaseConfig


comptime C = Gemma4BaseConfig
comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]


@explicit_destroy
struct BoolFence[P: BurstThreadPool, origin: MutOrigin](Movable):
    var pool: Pointer[Self.P, Self.origin]
    var active: Bool

    @staticmethod
    def dispatched(ref [origin] pool: P) -> Self:
        return Self {pool: Pointer(to=pool), active: True}

    @staticmethod
    def inline(ref [origin] pool: P) -> Self:
        return Self {pool: Pointer(to=pool), active: False}

    def join(deinit self):
        if self.active:
            self.pool[].join()

    def finish(deinit self) -> Int:
        if self.active:
            self.pool[].join()
            return self.pool[].last_worker_timestamp()
        return 0


def fence_gemv[P: BurstThreadPool, //, rows: Int, cols: Int](
    x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr, mut pool: P,
) -> BoolFence[P, origin_of(pool)]:
    if rows <= GEMV_INLINE_ROWS:
        gemv_range[rows, cols](x, weight, output, 0, rows)
        return BoolFence.inline(pool)
    var data_bytes = rows * cols * 2
    var nw = recommended_workers(data_bytes, pool.get_capacity())
    var buf = DispatchBuffer[GemvKernel[rows, cols]]()
    tile_dispatch(buf,
        GemvKernel[rows, cols](x, weight, output, 0, 0),
        pool, rows, num_workers=nw)
    return BoolFence.dispatched(pool)


def fence_rms_norm[
    P: BurstThreadPool, //,
    hidden: Int, sqrt_n: Scalar[DType.float32], n_eps: Scalar[DType.float32],
](
    src: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr,
    count: Int, mut pool: P,
) -> BoolFence[P, origin_of(pool)]:
    if count <= NORM_INLINE_TOKENS:
        for tok in range(count):
            rms_norm_row[hidden, sqrt_n, n_eps](
                src + tok * hidden, dst + tok * hidden, weight)
        return BoolFence.inline(pool)
    var data_bytes = count * hidden * 2
    var nw = recommended_workers(data_bytes, pool.get_capacity())
    var buf = DispatchBuffer[RmsNormTokenKernel[hidden, sqrt_n, n_eps]]()
    tile_dispatch(buf,
        RmsNormTokenKernel[hidden, sqrt_n, n_eps](src, dst, weight, 0, 0),
        pool, count, num_workers=nw)
    return BoolFence.dispatched(pool)


def fence_sliding_attention[
    P: BurstThreadPool, //,
    head_dim: Int, num_q: Int, gqa_ratio: Int,
    kv_stride: Int, window: Int,
](
    q: BF16Ptr, k_base: BF16Ptr, v_base: BF16Ptr,
    partials_buf: F32Ptr,
    pos: Int, valid_len: Int,
    mut pool: P,
) -> Tuple[Int, BoolFence[P, origin_of(pool)]]:
    if valid_len <= 0:
        return (0, BoolFence.inline(pool))
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
    return (nw, BoolFence.dispatched(pool))


def fence_merge[
    P: BurstThreadPool, //,
    head_dim: Int, num_q: Int,
](
    output: BF16Ptr, partials_buf: F32Ptr,
    partial_stride: Int, num_sources: Int,
    mut pool: P,
    head_offset: Int = 0,
) -> BoolFence[P, origin_of(pool)]:
    if num_sources <= 0 or num_q <= 8:
        for local_h in range(num_q):
            finalize_head[head_dim, num_q](
                output, partials_buf, partial_stride,
                num_sources, head_offset + local_h, local_h)
        return BoolFence.inline(pool)
    var nw = min(num_q, pool.get_capacity())
    var buf = DispatchBuffer[FinalizeKernel[head_dim, num_q]]()
    tile_dispatch(buf,
        FinalizeKernel[head_dim, num_q](
            output, partials_buf, partial_stride,
            num_sources, head_offset, 0, 0),
        pool, num_q, num_workers=nw)
    return BoolFence.dispatched(pool)


def simulate_sliding_layer_bool_fence[P: BurstThreadPool, //, degree: Int](
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

    # Phase 1: input norm — dispatch all ranks, then join all.
    #
    # Problem: BoolFence is @explicit_destroy. We need to hold `degree`
    # fences alive across the dispatch loop, then consume them all.
    # InlineArray can hold @explicit_destroy types with uninitialized init,
    # but then WE own destruction — and that means manually moving each
    # element out. Awkward but workable:

    var norm_fences = InlineArray[
        BoolFence[P, MutAnyOrigin], degree](uninitialized=True)
    for rank in range(degree):
        # PROBLEM: BoolFence's origin is tied to origin_of(pool), but
        # pools[rank] has a specific origin from the HeapMoveArray.
        # We'd need to erase or unify origins to store in a homogeneous
        # array. This is the first real friction point.
        #
        # For now, pretend origin erasure works:
        (UnsafePointer(to=norm_fences[rank])).init_pointee_move(
            fence_rms_norm[hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps](
                x_ptrs[rank], norm_dst_ptrs[rank], norm_weights[rank],
                1, pools[rank]))

    for rank in range(degree):
        (UnsafePointer(to=norm_fences[rank])).take_pointee()^.join()

    # VERDICT: The InlineArray + init_pointee_move + take_pointee dance
    # is verbose but correct. The real problem is origin: each fence
    # captures origin_of(pools[rank]), and storing them in one array
    # requires a common origin. pools is HeapMoveArray so all elements
    # share the same backing origin — this SHOULD unify, but it depends
    # on whether the compiler sees pools[0] and pools[1] as the same
    # origin.
    #
    # If origins don't unify, we'd need MutAnyOrigin or a wrapper.

    # Phase 2: QKV gemv — same pattern.
    for rank in range(degree):
        (UnsafePointer(to=norm_fences[rank])).init_pointee_move(
            fence_gemv[rows=q_rows, cols=C.HIDDEN](
                norm_dst_ptrs[rank], q_weights[rank], q_ptrs[rank],
                pools[rank]))
    for rank in range(degree):
        (UnsafePointer(to=norm_fences[rank])).take_pointee()^.join()

    # Phase 3: attention — returns (nw, fence). Need to track nw per rank.
    var nws = InlineArray[Int, degree](fill=0)
    for rank in range(degree):
        var result = fence_sliding_attention[
            head_dim=head_dim, num_q=num_q,
            gqa_ratio=gqa, kv_stride=kv_rows, window=C.SLIDING_WINDOW](
            q_ptrs[rank], k_bases[rank], v_bases[rank],
            partials[rank], pos, valid_len, pools[rank])
        nws[rank] = result[0]
        (UnsafePointer(to=norm_fences[rank])).init_pointee_move(result[1])
    for rank in range(degree):
        (UnsafePointer(to=norm_fences[rank])).take_pointee()^.join()

    # Phase 4: merge partials
    for rank in range(degree):
        (UnsafePointer(to=norm_fences[rank])).init_pointee_move(
            fence_merge[head_dim=head_dim, num_q=num_q](
                q_ptrs[rank], partials[rank],
                flash_stride, nws[rank], pools[rank]))
    for rank in range(degree):
        (UnsafePointer(to=norm_fences[rank])).take_pointee()^.join()

    # Phase 5: o_proj gemv
    for rank in range(degree):
        (UnsafePointer(to=norm_fences[rank])).init_pointee_move(
            fence_gemv[rows=C.HIDDEN, cols=q_rows](
                q_ptrs[rank], o_weights[rank], x_ptrs[rank],
                pools[rank]))
    for rank in range(degree):
        (UnsafePointer(to=norm_fences[rank])).take_pointee()^.join()

    # OVERALL VERDICT for BoolFence:
    #
    # + The fence struct itself is clean and minimal.
    # + Inline no-op is one predicted branch.
    # + @explicit_destroy guarantees consumption.
    #
    # - Storing fences across a dispatch loop requires InlineArray with
    #   uninitialized=True and manual init_pointee_move/take_pointee.
    #   This is 3 lines of ceremony per fence per phase.
    #
    # - Origin unification: all fences in the array need the same origin.
    #   HeapMoveArray elements likely share one origin, but this is fragile.
    #
    # - The fence carries a Pointer it never uses on the inline path.
    #   Not a perf issue, but conceptually wasteful.
    #
    # - Reusing the norm_fences array for all phases works (same type,
    #   same degree) but is semantically confusing — it's not "norm" fences
    #   after phase 1.
    #
    # - The two-loop pattern (dispatch loop + join loop) is clear but
    #   doubles the loop count. For degree=2 or 4 this is fine.
    #
    # KEY QUESTION: Is the InlineArray + take_pointee ceremony acceptable,
    # or should the fence design avoid needing storage entirely?
