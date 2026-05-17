from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from simd_math import fast_exp_softmax_biased
from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from .helpers import (
    OutputPartitionedKernel, DispatchBuffer,
    recommended_workers, worker_range, join_all, Binding,
)
from .attention_ops import score_position, accumulate_v, scale_acc


comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
comptime W = simd_width_of[DType.float32]()


comptime FLASH_PARTIAL_STRIDE[num_q: Int, head_dim: Int]: Int = (
    (num_q * head_dim + num_q + num_q) * 4 + 63) // 64 * 16


@fieldwise_init
struct FlashDecodeKernel[
    head_dim: Int, num_q: Int, gqa_ratio: Int,
    kv_stride: Int, window: Int,
](OutputPartitionedKernel):
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
    head_dim: Int, num_q: Int, gqa_ratio: Int,
    kv_stride: Int, window: Int, tp: Int, max_worker_count: Int = 128,
](
    q: Binding[Scalar[DType.bfloat16], tp],
    k_base: Binding[Scalar[DType.bfloat16], tp],
    v_base: Binding[Scalar[DType.bfloat16], tp],
    partials_buf: Binding[Scalar[DType.float32], tp],
    pos: Int, valid_len: Int,
    mut pools: HeapMoveArray[P],
) -> InlineArray[Int, tp]:
    var result = InlineArray[Int, tp](fill=0)
    if valid_len <= 0:
        return result

    var start_pos = pos - valid_len + 1
    comptime stride = FLASH_PARTIAL_STRIDE[num_q, head_dim]
    var buf = DispatchBuffer[
        FlashDecodeKernel[head_dim, num_q, gqa_ratio, kv_stride, window],
        max_worker_count,
    ]()
    for r in range(tp):
        var nw = recommended_workers(
            valid_len * kv_stride * 2,
            min(max_worker_count, pools[r].get_capacity()),
        )
        for w_idx in range(nw):
            var wr = worker_range(valid_len, nw, w_idx)
            buf.slot()[] = FlashDecodeKernel[
                head_dim, num_q, gqa_ratio, kv_stride, window](
                q[r], k_base[r], v_base[r], partials_buf[r],
                stride, w_idx, start_pos, wr[0], wr[1])
        buf.dispatch(pools[r])
        result[r] = nw
    join_all[tp](pools)
    return result
