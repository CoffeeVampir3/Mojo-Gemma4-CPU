from std.collections import InlineArray
from std.memory import UnsafePointer

from threading.threading_traits import BurstThreadPool
from .helpers import (
    BF16Ptr, F32Ptr, W,
    RangePartitionedKernel, WorkerRangePartitionedKernel,
    fanout_dispatch, Binding,
)
from .attention_ops import (
    LinearKV, RingKV, TILE, full_local_kv_count, process_kv_tile,
    zero_accumulators,
)
from .logsum_merge import MergeSegment, write_finalized_head
from .profiling import Profiler


@fieldwise_init
struct FlashPrefillSlidingKernel[
    head_dim: Int, num_q: Int, gqa_ratio: Int,
    kv_stride: Int, window: Int, cache_size: Int, partial_stride: Int,
](WorkerRangePartitionedKernel):
    """`cache_size` must be a power of two (slot index uses `& (cache_size - 1)`)."""

    var q: BF16Ptr
    var k_base: BF16Ptr
    var v_base: BF16Ptr
    var output: BF16Ptr
    var partials: F32Ptr
    var base_pos: Int
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime q_stride = Self.num_q * Self.head_dim

        var scratch = self.partials + self.worker_id * Self.partial_stride
        var acc_ptrs = InlineArray[F32Ptr, Self.num_q](uninitialized=True)
        var q_ptrs = InlineArray[BF16Ptr, Self.num_q](uninitialized=True)
        comptime for h in range(Self.num_q):
            acc_ptrs[h] = scratch + h * Self.head_dim

        for t in range(self.start, self.end):
            var abs_pos = self.base_pos + t
            var lo = max(0, abs_pos - Self.window + 1)
            var hi = abs_pos + 1

            var q_tok = self.q + t * q_stride
            var out_tok = self.output + t * q_stride

            var m = InlineArray[Float32, Self.num_q](fill=Float32(-1e30))
            var l = InlineArray[Float32, Self.num_q](fill=Float32(0))
            comptime for h in range(Self.num_q):
                q_ptrs[h] = q_tok + h * Self.head_dim

            zero_accumulators[Self.num_q, Self.head_dim](acc_ptrs)

            var pos = lo
            while pos < hi:
                var tile_len = min(TILE, hi - pos)
                process_kv_tile[
                    RingKV[Self.cache_size],
                    Self.head_dim, Self.gqa_ratio, Self.kv_stride,
                ](q_ptrs, self.k_base, self.v_base,
                  0, pos, tile_len, m, l, acc_ptrs)
                pos += TILE

            comptime for h in range(Self.num_q):
                if l[h] > 0:
                    var inv_l = SIMD[DType.float32, W](
                        Float32(1.0) / l[h])
                    for j in range(0, Self.head_dim, W):
                        var v = (acc_ptrs[h] + j).load[width=W]() * inv_l
                        (out_tok + h * Self.head_dim + j).store(
                            v.cast[DType.bfloat16]())
                else:
                    for j in range(0, Self.head_dim, W):
                        (out_tok + h * Self.head_dim + j).store(
                            SIMD[DType.bfloat16, W](0))

    @always_inline
    def install_worker_range(
        mut self, worker_id: Int, start: Int, end: Int,
    ):
        self.worker_id = worker_id
        self.start = start
        self.end = end


@fieldwise_init
struct FlashPrefillFullKernel[
    head_dim: Int, num_q: Int, gqa_ratio: Int,
    kv_stride: Int, degree: Int, partial_stride: Int,
](RangePartitionedKernel):
    """Walks this rank's `pos // degree` local KV slice with causal upper bound;
    writes (acc, m, l) partials for cross-rank logsum merge."""

    var q: BF16Ptr
    var k_base: BF16Ptr
    var v_base: BF16Ptr
    var partials: F32Ptr
    var base_pos: Int
    var rank: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime q_stride = Self.num_q * Self.head_dim
        comptime m_off = Self.num_q * Self.head_dim
        comptime l_off = m_off + Self.num_q

        var acc_ptrs = InlineArray[F32Ptr, Self.num_q](uninitialized=True)
        var q_ptrs = InlineArray[BF16Ptr, Self.num_q](uninitialized=True)

        for t in range(self.start, self.end):
            var abs_pos = self.base_pos + t
            var local_kv_count = full_local_kv_count(
                self.rank, abs_pos, Self.degree)

            var partial_tok = self.partials + t * Self.partial_stride
            var q_tok = self.q + t * q_stride

            var m = InlineArray[Float32, Self.num_q](fill=Float32(-1e30))
            var l = InlineArray[Float32, Self.num_q](fill=Float32(0))

            comptime for h in range(Self.num_q):
                acc_ptrs[h] = partial_tok + h * Self.head_dim
                q_ptrs[h] = q_tok + h * Self.head_dim

            zero_accumulators[Self.num_q, Self.head_dim](acc_ptrs)

            var pos = 0
            while pos < local_kv_count:
                var tile_len = min(TILE, local_kv_count - pos)
                process_kv_tile[
                    LinearKV,
                    Self.head_dim, Self.gqa_ratio, Self.kv_stride,
                ](q_ptrs, self.k_base, self.v_base,
                  0, pos, tile_len, m, l, acc_ptrs)
                pos += TILE

            comptime for h in range(Self.num_q):
                (partial_tok + m_off + h)[] = m[h]
                (partial_tok + l_off + h)[] = l[h]

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


@fieldwise_init
struct PrefillMergeConfig[head_dim: Int, num_q: Int, tp: Int]:
    var output: Binding[BFloat16, Self.tp]
    var partials: Binding[Float32, Self.tp]


@fieldwise_init
struct PrefillMergeKernel[
    head_dim: Int, num_q: Int, local_num_q: Int, tp: Int, partial_stride: Int,
    cfg_origin: Origin,
](RangePartitionedKernel):
    var config: UnsafePointer[
        PrefillMergeConfig[Self.head_dim, Self.num_q, Self.tp],
        Self.cfg_origin,
    ]
    var q_rank: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime out_stride = Self.local_num_q * Self.head_dim

        for flat in range(self.start, self.end):
            var t = flat // Self.local_num_q
            var local_h = flat % Self.local_num_q
            var global_h = self.q_rank * Self.local_num_q + local_h

            var segs = InlineArray[MergeSegment, Self.tp](
                uninitialized=True)
            comptime for r in range(Self.tp):
                segs[r] = MergeSegment(
                    self.config[].partials[r] + t * Self.partial_stride,
                    Self.partial_stride, 1)

            var dst = self.config[].output[self.q_rank] \
                      + t * out_stride + local_h * Self.head_dim
            write_finalized_head[Self.head_dim, Self.num_q, Self.tp](
                dst, segs, global_h)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_merge_flash_prefill_partials[
    P: BurstThreadPool, Profile: Bool, N: Int, //,
    head_dim: Int, num_q: Int, local_num_q: Int, partial_stride: Int, tp: Int,
    max_worker_count: Int = 128,
](
    output: Binding[BFloat16, tp],
    partials: Binding[Float32, tp],
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    if seq_len <= 0:
        return

    var cfg = PrefillMergeConfig[head_dim, num_q, tp](output, partials)
    var config = UnsafePointer(to=cfg).as_immutable()
    comptime cfg_ro = ImmutOrigin(origin_of(cfg))
    comptime K = PrefillMergeKernel[
        head_dim, num_q, local_num_q, tp, partial_stride, cfg_ro,
    ]

    @parameter
    def make(q_rank: Int) -> K:
        return K(config, q_rank, 0, 0)

    var total_units = seq_len * local_num_q
    var data_bytes = total_units * tp * (head_dim + 2) * 4

    fanout_dispatch[
        tp, make, max_worker_count=max_worker_count,
        label="merge_flash_prefill_partials",
    ](pools, prof, total_units, data_bytes)
