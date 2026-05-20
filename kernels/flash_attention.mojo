from std.collections import InlineArray

from .helpers import (
    BF16Ptr, F32Ptr,
    WorkerRangePartitionedKernel,
)
from .attention_ops import KVSlot, TILE, process_kv_tile, zero_accumulators


@fieldwise_init
struct FlashAttentionKernel[
    KV: KVSlot,
    head_dim: Int, num_q: Int, gqa_ratio: Int, kv_stride: Int,
    partial_stride: Int,
](WorkerRangePartitionedKernel):
    var q: BF16Ptr
    var k_base: BF16Ptr
    var v_base: BF16Ptr
    var partials: F32Ptr
    var worker_id: Int
    var start_pos: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var my_partial = self.partials + self.worker_id * Self.partial_stride
        comptime m_off = Self.num_q * Self.head_dim
        comptime l_off = m_off + Self.num_q

        var acc_ptrs = InlineArray[F32Ptr, Self.num_q](uninitialized=True)
        var q_ptrs = InlineArray[BF16Ptr, Self.num_q](uninitialized=True)
        var m = InlineArray[Float32, Self.num_q](fill=Float32(-1e30))
        var l = InlineArray[Float32, Self.num_q](fill=Float32(0))

        comptime for h in range(Self.num_q):
            acc_ptrs[h] = my_partial + h * Self.head_dim
            q_ptrs[h] = self.q + h * Self.head_dim

        zero_accumulators[Self.num_q, Self.head_dim](acc_ptrs)

        var pos = self.start
        while pos < self.end:
            var tile_len = min(TILE, self.end - pos)
            process_kv_tile[
                Self.KV, Self.head_dim, Self.gqa_ratio, Self.kv_stride,
            ](q_ptrs, self.k_base, self.v_base,
              self.start_pos, pos, tile_len, m, l, acc_ptrs)
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
