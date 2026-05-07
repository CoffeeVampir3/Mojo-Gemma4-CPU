from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from simd_math import pick_port_unroll, tree_reduce_accs
from simd_math import fast_exp_softmax_biased
from threading.threading_traits import BurstThreadPool
from .helpers import (
    RangedKernel, DispatchBuffer, tile_dispatch, recommended_workers, worker_range,
)


comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
comptime W = simd_width_of[DType.float32]()



@always_inline
def score_position[head_dim: Int](q: BF16Ptr, k_row: BF16Ptr) -> Scalar[DType.float32]:
    comptime PU = pick_port_unroll[W, head_dim]()
    comptime STRIDE = PU * W
    var accs = InlineArray[SIMD[DType.float32, W], PU](fill=SIMD[DType.float32, W](0))
    for i in range(head_dim // STRIDE):
        comptime for p in range(PU):
            var qv = (q + i * STRIDE + p * W).load[width=W]().cast[DType.float32]()
            var kv = (k_row + i * STRIDE + p * W).load[width=W]().cast[DType.float32]()
            accs[p] = qv.fma(kv, accs[p])
    return tree_reduce_accs(accs)


@always_inline
def accumulate_v[head_dim: Int](
    v_row: BF16Ptr, weight: Scalar[DType.float32], acc: F32Ptr,
):
    comptime PU = pick_port_unroll[W, head_dim]()
    comptime STRIDE = PU * W
    var w_vec = SIMD[DType.float32, W](weight)
    for i in range(head_dim // STRIDE):
        comptime for p in range(PU):
            var v = (v_row + i * STRIDE + p * W).load[width=W]().cast[DType.float32]()
            var a = (acc + i * STRIDE + p * W).load[width=W]()
            (acc + i * STRIDE + p * W).store(v.fma(w_vec, a))


@always_inline
def accumulate_v_corrected[head_dim: Int](
    v_row: BF16Ptr, weight: Scalar[DType.float32],
    correction: Scalar[DType.float32], acc: F32Ptr,
):
    comptime PU = pick_port_unroll[W, head_dim]()
    comptime STRIDE = PU * W
    var w_vec = SIMD[DType.float32, W](weight)
    var c_vec = SIMD[DType.float32, W](correction)
    for i in range(head_dim // STRIDE):
        comptime for p in range(PU):
            var v = (v_row + i * STRIDE + p * W).load[width=W]().cast[DType.float32]()
            var a = (acc + i * STRIDE + p * W).load[width=W]()
            (acc + i * STRIDE + p * W).store(a.fma(c_vec, v * w_vec))


@always_inline
def scale_acc[head_dim: Int](acc: F32Ptr, factor: Scalar[DType.float32]):
    comptime PU = pick_port_unroll[W, head_dim]()
    comptime STRIDE = PU * W
    var f = SIMD[DType.float32, W](factor)
    for i in range(head_dim // STRIDE):
        comptime for p in range(PU):
            var a = (acc + i * STRIDE + p * W).load[width=W]()
            (acc + i * STRIDE + p * W).store(a * f)


@always_inline
def softmax_inplace(scores: F32Ptr, valid_len: Int):
    var m = scores[0]
    for i in range(1, valid_len):
        var s = scores[i]
        if s > m:
            m = s
    var sum_val = Scalar[DType.float32](0)
    for i in range(0, valid_len, W):
        if i + W <= valid_len:
            var v = (scores + i).load[width=W]()
            var e = fast_exp_softmax_biased(v - SIMD[DType.float32, W](m))
            (scores + i).store(e)
            sum_val += e.reduce_add()
        else:
            for j in range(i, valid_len):
                var e = fast_exp_softmax_biased[1](
                    SIMD[DType.float32, 1](scores[j] - m))
                scores[j] = e[0]
                sum_val += e[0]
    var inv_sum = Scalar[DType.float32](1.0) / sum_val
    for i in range(0, valid_len, W):
        if i + W <= valid_len:
            (scores + i).store((scores + i).load[width=W]() * SIMD[DType.float32, W](inv_sum))
        else:
            for j in range(i, valid_len):
                scores[j] = scores[j] * inv_sum



@fieldwise_init
struct ScoreKernel[
    head_dim: Int, num_q: Int, num_kv: Int, gqa_ratio: Int,
    kv_stride: Int, window: Int,
](RangedKernel):
    var q: BF16Ptr
    var k_base: BF16Ptr
    var scores: F32Ptr
    var valid_len: Int
    var start_pos: Int
    var start: Int
    var end: Int

    def execute(mut self):
        for p in range(self.start, self.end):
            var cache_slot = (self.start_pos + p) & (Self.window - 1)
            var k_row = self.k_base + cache_slot * Self.kv_stride
            comptime for q_idx in range(Self.num_q):
                comptime kv_h = q_idx // Self.gqa_ratio
                var k_head = k_row + kv_h * Self.head_dim
                var s = score_position[Self.head_dim](
                    self.q + q_idx * Self.head_dim, k_head)
                (self.scores + q_idx * self.valid_len + p)[] = s

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.q, self.k_base, self.scores, self.valid_len,
            self.start_pos, start, end)


@fieldwise_init
struct WeightedVKernel[
    head_dim: Int, num_q: Int, num_kv: Int, gqa_ratio: Int,
    kv_stride: Int, window: Int,
](RangedKernel):
    var scores: F32Ptr
    var v_base: BF16Ptr
    var partials: F32Ptr
    var valid_len: Int
    var start_pos: Int
    var start: Int
    var end: Int

    def execute(mut self):
        for i in range(Self.num_q * Self.head_dim):
            (self.partials + i)[] = Scalar[DType.float32](0)
        for p in range(self.start, self.end):
            var cache_slot = (self.start_pos + p) & (Self.window - 1)
            var v_row = self.v_base + cache_slot * Self.kv_stride
            comptime for q_idx in range(Self.num_q):
                comptime kv_h = q_idx // Self.gqa_ratio
                var v_head = v_row + kv_h * Self.head_dim
                var w = (self.scores + q_idx * self.valid_len + p)[]
                accumulate_v[Self.head_dim](
                    v_head, w, self.partials + q_idx * Self.head_dim)

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.scores, self.v_base, self.partials, self.valid_len,
            self.start_pos, start, end)


@always_inline
def merge_sum_partials[head_dim: Int, num_q: Int](
    output: BF16Ptr, partials: F32Ptr, partial_stride: Int, num_workers: Int,
):
    for h in range(num_q):
        for j in range(0, head_dim, W):
            var sum = SIMD[DType.float32, W](0)
            for w_idx in range(num_workers):
                sum += (partials + w_idx * partial_stride + h * head_dim + j).load[width=W]()
            (output + h * head_dim + j).store(sum.cast[DType.bfloat16]())


def sliding_attention_two_pass[
    P: BurstThreadPool, //,
    head_dim: Int, num_q: Int, num_kv: Int, gqa_ratio: Int,
    kv_stride: Int, window: Int,
](
    q: BF16Ptr, k_base: BF16Ptr, v_base: BF16Ptr,
    output: BF16Ptr,
    scores_buf: F32Ptr,
    partials_buf: F32Ptr,
    pos: Int, valid_len: Int,
    mut pool: P,
    num_workers: Int = 0,
):
    var start_pos = pos - valid_len + 1
    var nw = recommended_workers(valid_len * kv_stride * 2, pool.get_capacity()) if num_workers <= 0 else min(num_workers, pool.get_capacity())
    comptime partial_stride = num_q * head_dim

    var score_buf = DispatchBuffer[
        ScoreKernel[head_dim, num_q, num_kv, gqa_ratio, kv_stride, window]]()
    tile_dispatch(score_buf,
        ScoreKernel[head_dim, num_q, num_kv, gqa_ratio, kv_stride, window](
            q, k_base, scores_buf, valid_len, start_pos, 0, 0),
        pool, valid_len, num_workers=nw)
    pool.join()

    for h in range(num_q):
        softmax_inplace(scores_buf + h * valid_len, valid_len)

    var v_buf = DispatchBuffer[
        WeightedVKernel[head_dim, num_q, num_kv, gqa_ratio, kv_stride, window]]()
    for w_idx in range(nw):
        var wr = worker_range(valid_len, nw, w_idx)
        v_buf.slot()[] = WeightedVKernel[
            head_dim, num_q, num_kv, gqa_ratio, kv_stride, window](
            scores_buf, v_base,
            partials_buf + w_idx * partial_stride,
            valid_len, start_pos, wr[0], wr[1])
    v_buf.dispatch(pool)
    pool.join()

    merge_sum_partials[head_dim, num_q](output, partials_buf, partial_stride, nw)



comptime FLASH_PARTIAL_STRIDE[num_q: Int, head_dim: Int]: Int = (
    (num_q * head_dim + num_q + num_q) * 4 + 63) // 64 * 16


@fieldwise_init
struct FlashDecodeKernel[
    head_dim: Int, num_q: Int, num_kv: Int, gqa_ratio: Int,
    kv_stride: Int, window: Int,
](RangedKernel):
    var q: BF16Ptr
    var k_base: BF16Ptr
    var v_base: BF16Ptr
    var partials: F32Ptr
    var partial_stride: Int
    var worker_id: Int
    var start_pos: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var my_partial = self.partials + self.worker_id * self.partial_stride
        comptime m_off = Self.num_q * Self.head_dim
        comptime l_off = m_off + Self.num_q
        comptime TILE = W

        var acc_ptrs = InlineArray[F32Ptr, Self.num_q](uninitialized=True)
        var q_ptrs = InlineArray[BF16Ptr, Self.num_q](uninitialized=True)
        var m = InlineArray[Scalar[DType.float32], Self.num_q](
            fill=Scalar[DType.float32](-1e30))
        var l = InlineArray[Scalar[DType.float32], Self.num_q](
            fill=Scalar[DType.float32](0))

        comptime for h in range(Self.num_q):
            acc_ptrs[h] = my_partial + h * Self.head_dim
            q_ptrs[h] = self.q + h * Self.head_dim
            for j in range(0, Self.head_dim, W):
                (acc_ptrs[h] + j).store(SIMD[DType.float32, W](0))

        var pos = self.start
        while pos < self.end:
            var tile_len = min(TILE, self.end - pos)

            comptime for q_idx in range(Self.num_q):
                comptime kv_h = q_idx // Self.gqa_ratio

                var scores = SIMD[DType.float32, TILE](-1e30)
                for t in range(tile_len):
                    var cache_slot = (self.start_pos + pos + t) & (Self.window - 1)
                    var k_head = self.k_base + cache_slot * Self.kv_stride + kv_h * Self.head_dim
                    scores[t] = score_position[Self.head_dim](q_ptrs[q_idx], k_head)

                var tile_max = scores.reduce_max()
                var m_new = tile_max if tile_max > m[q_idx] else m[q_idx]

                var corr = fast_exp_softmax_biased[1](
                    SIMD[DType.float32, 1](m[q_idx] - m_new))[0]
                var weights = fast_exp_softmax_biased[TILE](
                    scores - SIMD[DType.float32, TILE](m_new))

                scale_acc[Self.head_dim](acc_ptrs[q_idx], corr)
                l[q_idx] = l[q_idx] * corr + weights.reduce_add()
                m[q_idx] = m_new

                for t in range(tile_len):
                    var cache_slot = (self.start_pos + pos + t) & (Self.window - 1)
                    var v_head = self.v_base + cache_slot * Self.kv_stride + kv_h * Self.head_dim
                    accumulate_v[Self.head_dim](v_head, weights[t], acc_ptrs[q_idx])

            pos += TILE

        comptime for h in range(Self.num_q):
            (my_partial + m_off + h)[] = m[h]
            (my_partial + l_off + h)[] = l[h]

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.q, self.k_base, self.v_base, self.partials,
            self.partial_stride, self.worker_id, self.start_pos, start, end)



def dispatch_sliding_attention[
    P: BurstThreadPool, //,
    head_dim: Int, num_q: Int, num_kv: Int, gqa_ratio: Int,
    kv_stride: Int, window: Int,
](
    q: BF16Ptr, k_base: BF16Ptr, v_base: BF16Ptr,
    partials_buf: F32Ptr,
    pos: Int, valid_len: Int,
    mut pool: P,
    num_workers: Int = 0,
) -> Int:
    if valid_len <= 0:
        return 0

    var start_pos = pos - valid_len + 1
    var nw = recommended_workers(valid_len * kv_stride * 2, pool.get_capacity()) if num_workers <= 0 else min(num_workers, pool.get_capacity())

    var buf = DispatchBuffer[
        FlashDecodeKernel[head_dim, num_q, num_kv, gqa_ratio, kv_stride, window]]()
    for w_idx in range(nw):
        var wr = worker_range(valid_len, nw, w_idx)
        buf.slot()[] = FlashDecodeKernel[
            head_dim, num_q, num_kv, gqa_ratio, kv_stride, window](
            q, k_base, v_base, partials_buf,
            FLASH_PARTIAL_STRIDE[num_q, head_dim], w_idx, start_pos, wr[0], wr[1])
    buf.dispatch(pool)
    pool.join()
    return nw
