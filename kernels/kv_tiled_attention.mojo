from std.collections import InlineArray

from simd_math import fast_exp_softmax_biased
from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from .helpers import (
    BF16Ptr, F32Ptr, W,
    OutputPartitionedKernel, fanout_dispatch_per_rank, Binding,
)
from .attention_ops import (
    score_position, accumulate_v, scale_acc, flash_partial_stride,
)


@fieldwise_init
struct FlashDecodeKernel[
    head_dim: Int, num_q: Int, gqa_ratio: Int,
    kv_stride: Int, window: Int,
](OutputPartitionedKernel):
    comptime PARTIAL_STRIDE = flash_partial_stride[Self.num_q, Self.head_dim]()

    var q: BF16Ptr
    var k_base: BF16Ptr
    var v_base: BF16Ptr
    var partials: F32Ptr
    var worker_id: Int
    var start_pos: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var my_partial = self.partials + self.worker_id * Self.PARTIAL_STRIDE
        comptime m_off = Self.num_q * Self.head_dim
        comptime l_off = m_off + Self.num_q
        comptime TILE = W

        var acc_ptrs = InlineArray[F32Ptr, Self.num_q](uninitialized=True)
        var q_ptrs = InlineArray[BF16Ptr, Self.num_q](uninitialized=True)
        var m = InlineArray[Float32, Self.num_q](
            fill=Float32(-1e30))
        var l = InlineArray[Float32, Self.num_q](
            fill=Float32(0))

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

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.worker_id = worker_id
        self.start = start
        self.end = end


def dispatch_sliding_attention[
    P: BurstThreadPool, //,
    head_dim: Int, num_q: Int, gqa_ratio: Int,
    kv_stride: Int, window: Int, tp: Int, max_worker_count: Int = 128,
](
    q: Binding[BFloat16, tp],
    k_base: Binding[BFloat16, tp],
    v_base: Binding[BFloat16, tp],
    partials_buf: Binding[Float32, tp],
    pos: Int, valid_len: Int,
    mut pools: HeapMoveArray[P],
) -> InlineArray[Int, tp]:
    if valid_len <= 0:
        return InlineArray[Int, tp](fill=0)

    var start_pos = pos - valid_len + 1
    comptime K = FlashDecodeKernel[head_dim, num_q, gqa_ratio, kv_stride, window]

    @parameter
    def make(r: Int) -> K:
        return K(q[r], k_base[r], v_base[r], partials_buf[r],
                 0, start_pos, 0, 0)

    @parameter
    def total_for(r: Int) -> Int:
        return valid_len

    @parameter
    def bytes_for(r: Int) -> Int:
        return valid_len * kv_stride * 2

    return fanout_dispatch_per_rank[
        tp, make, total_for, bytes_for,
        max_worker_count=max_worker_count,
    ](pools)
