from std.collections import InlineArray

from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from .helpers import (
    Binding, fanout_dispatch, fanout_dispatch_per_rank,
)
from .attention_ops import LinearKV, RingKV, full_local_kv_count
from .flash_attention import FlashAttentionKernel
from .flash_attention_prefill import (
    FlashPrefillSlidingKernel, FlashPrefillFullKernel,
    dispatch_merge_flash_prefill_partials,
)
from .logsum_merge import (
    dispatch_merge_flash_partials, dispatch_merge_context_flash_partials,
)


@always_inline
def sliding_valid_len(pos: Int, window: Int) -> Int:
    if pos + 1 >= window:
        return window
    return pos + 1


def dispatch_sliding_attention[
    P: BurstThreadPool, //,
    head_dim: Int, num_q: Int, gqa_ratio: Int,
    kv_stride: Int, window: Int, cache_size: Int, partial_stride: Int, tp: Int,
    max_worker_count: Int = 128,
](
    q: Binding[BFloat16, tp],
    k_base: Binding[BFloat16, tp],
    v_base: Binding[BFloat16, tp],
    output: Binding[BFloat16, tp],
    partials: Binding[Float32, tp],
    base_pos: Int,
    seq_len: Int,
    mut pools: HeapMoveArray[P],
):
    if seq_len <= 0:
        return
    if seq_len == 1:
        var valid_len = sliding_valid_len(base_pos, window)
        if valid_len <= 0:
            return
        var start_pos = base_pos - valid_len + 1
        comptime DecodeK = FlashAttentionKernel[
            RingKV[cache_size],
            head_dim, num_q, gqa_ratio, kv_stride, partial_stride,
        ]

        @parameter
        def make_decode(r: Int) -> DecodeK:
            return DecodeK(q[r], k_base[r], v_base[r], partials[r],
                           0, start_pos, 0, 0)

        @parameter
        def total_for(r: Int) -> Int:
            return valid_len

        @parameter
        def bytes_for(r: Int) -> Int:
            return valid_len * kv_stride * 2

        var nws = fanout_dispatch_per_rank[
            tp, make_decode, total_for, bytes_for,
            max_worker_count=max_worker_count,
        ](pools)

        dispatch_merge_flash_partials[
            head_dim, num_q, partial_stride, tp=tp,
            max_worker_count=max_worker_count,
        ](output, partials, nws, pools)
    else:
        comptime PrefillK = FlashPrefillSlidingKernel[
            head_dim, num_q, gqa_ratio, kv_stride, window, cache_size,
            partial_stride,
        ]

        @parameter
        def make_prefill(r: Int) -> PrefillK:
            return PrefillK(q[r], k_base[r], v_base[r], output[r],
                            partials[r], base_pos, 0, 0, 0)

        var per_q_kv = window if seq_len > window else seq_len
        var data_bytes = seq_len * per_q_kv * kv_stride * 2

        fanout_dispatch[
            tp, make_prefill, max_worker_count=max_worker_count,
        ](pools, seq_len, data_bytes)


def dispatch_full_attention[
    P: BurstThreadPool, //,
    head_dim: Int, num_q: Int, local_num_q: Int, gqa_ratio: Int,
    kv_stride: Int, partial_stride: Int, tp: Int,
    max_worker_count: Int = 128,
](
    q: Binding[BFloat16, tp],
    k_base: Binding[BFloat16, tp],
    v_base: Binding[BFloat16, tp],
    q_local_output: Binding[BFloat16, tp],
    partials: Binding[Float32, tp],
    base_pos: Int,
    seq_len: Int,
    mut pools: HeapMoveArray[P],
):
    """`q_local_output` is the per-rank merged attention output, feeding the
    column-sharded o_proj."""
    if seq_len <= 0:
        return
    if seq_len == 1:
        var valid_lens = InlineArray[Int, tp](uninitialized=True)
        for rank in range(tp):
            valid_lens[rank] = full_local_kv_count(rank, base_pos, tp)

        comptime DecodeK = FlashAttentionKernel[
            LinearKV, head_dim, num_q, gqa_ratio, kv_stride, partial_stride,
        ]

        @parameter
        def make_decode(r: Int) -> DecodeK:
            return DecodeK(q[r], k_base[r], v_base[r], partials[r],
                           0, 0, 0, 0)

        @parameter
        def total_for(r: Int) -> Int:
            return valid_lens[r]

        @parameter
        def bytes_for(r: Int) -> Int:
            return valid_lens[r] * kv_stride * 2

        var nws = fanout_dispatch_per_rank[
            tp, make_decode, total_for, bytes_for,
            max_worker_count=max_worker_count,
        ](pools)

        dispatch_merge_context_flash_partials[
            head_dim=head_dim, num_q=num_q,
            local_num_q=local_num_q, partial_stride=partial_stride, tp=tp,
            max_worker_count=max_worker_count,
        ](q_local_output, partials, nws, pools)
    else:
        comptime PrefillK = FlashPrefillFullKernel[
            head_dim, num_q, gqa_ratio, kv_stride, tp, partial_stride,
        ]

        @parameter
        def make_prefill(r: Int) -> PrefillK:
            return PrefillK(q[r], k_base[r], v_base[r], partials[r],
                            base_pos, r, 0, 0)

        var avg_local_kv = (base_pos + seq_len // 2) // tp + 1
        var data_bytes = seq_len * avg_local_kv * kv_stride * 2

        fanout_dispatch[
            tp, make_prefill, max_worker_count=max_worker_count,
        ](pools, seq_len, data_bytes)

        dispatch_merge_flash_prefill_partials[
            head_dim=head_dim, num_q=num_q,
            local_num_q=local_num_q, partial_stride=partial_stride, tp=tp,
            max_worker_count=max_worker_count,
        ](q_local_output, partials, seq_len, pools)
