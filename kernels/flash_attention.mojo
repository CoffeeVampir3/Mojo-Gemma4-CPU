from std.collections import InlineArray

from .helpers import (
    BF16Ptr, F32Ptr, W,
    WorkerRangePartitionedKernel,
    accumulate_scaled, scale_unrolled,
)
from .attention_ops import (
    TILE, flash_partial_stride, online_softmax_tile,
)
from .dot_products import dot_to_scalar


trait KVSlot:
    @staticmethod
    @always_inline
    def slot(start_pos: Int, pos_t: Int) -> Int: ...


struct LinearKV(KVSlot):
    @staticmethod
    @always_inline
    def slot(start_pos: Int, pos_t: Int) -> Int:
        return pos_t


struct RingKV[window: Int](KVSlot):
    @staticmethod
    @always_inline
    def slot(start_pos: Int, pos_t: Int) -> Int:
        return (start_pos + pos_t) & (Self.window - 1)


@fieldwise_init
struct FlashAttentionKernel[
    KV: KVSlot,
    head_dim: Int, num_q: Int, gqa_ratio: Int, kv_stride: Int,
](WorkerRangePartitionedKernel):
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

        var acc_ptrs = InlineArray[F32Ptr, Self.num_q](uninitialized=True)
        var q_ptrs = InlineArray[BF16Ptr, Self.num_q](uninitialized=True)
        var m = InlineArray[Float32, Self.num_q](fill=Float32(-1e30))
        var l = InlineArray[Float32, Self.num_q](fill=Float32(0))

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
                    var s_idx = Self.KV.slot(self.start_pos, pos + t)
                    var k_head = self.k_base + s_idx * Self.kv_stride \
                                 + kv_h * Self.head_dim
                    scores[t] = dot_to_scalar[Self.head_dim](
                        q_ptrs[q_idx], k_head)

                var sm = online_softmax_tile[TILE](scores, m[q_idx])
                var m_new = sm[0]
                var corr = sm[1]
                var weights = sm[2]

                scale_unrolled[cols=Self.head_dim](acc_ptrs[q_idx], corr)
                l[q_idx] = l[q_idx] * corr + weights.reduce_add()
                m[q_idx] = m_new

                for t in range(tile_len):
                    var s_idx = Self.KV.slot(self.start_pos, pos + t)
                    var v_head = self.v_base + s_idx * Self.kv_stride \
                                 + kv_h * Self.head_dim
                    accumulate_scaled[cols=Self.head_dim](
                        v_head, weights[t], acc_ptrs[q_idx])

            pos += TILE

        comptime for h in range(Self.num_q):
            (my_partial + m_off + h)[] = m[h]
            (my_partial + l_off + h)[] = l[h]

    @always_inline
    def install_worker_range(
        mut self, worker_id: Int, start: Int, end: Int,
    ):
        self.worker_id = worker_id
        self.start = start
        self.end = end
