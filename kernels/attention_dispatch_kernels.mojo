from std.collections import InlineArray

from threading.threading_traits import BurstThreadPool
from .helpers import (
    Binding, OutputPartitionedKernel, fanout_dispatch, fanout_dispatch_per_rank,
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
from .profiling import Profiler


@always_inline
def sliding_valid_len(pos: Int, window: Int) -> Int:
    if pos + 1 >= window:
        return window
    return pos + 1


def dispatch_flash_sliding[
    P: BurstThreadPool, DecodeK: OutputPartitionedKernel,
    PrefillK: OutputPartitionedKernel, Profile: Bool, N: Int, //,
    head_dim: Int, num_q: Int, partial_stride: Int, kv_stride: Int,
    window: Int, tp: Int, elt_bytes: Int,
    make_decode: def(Int, Int) capturing [_] -> DecodeK,
    make_prefill: def(Int) capturing [_] -> PrefillK,
    decode_label: StaticString, prefill_label: StaticString,
    max_worker_count: Int = 128,
](
    output: Binding[BFloat16, tp],
    partials: Binding[Float32, tp],
    base_pos: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    if seq_len <= 0:
        return
    if seq_len == 1:
        var valid_len = sliding_valid_len(base_pos, window)
        if valid_len <= 0:
            return
        var start_pos = base_pos - valid_len + 1

        @parameter
        def make(r: Int) -> DecodeK:
            return make_decode(r, start_pos)

        @parameter
        def total_for(r: Int) -> Int:
            return valid_len

        @parameter
        def bytes_for(r: Int) -> Int:
            return valid_len * kv_stride * elt_bytes

        var nws = fanout_dispatch_per_rank[
            tp, make, total_for, bytes_for,
            max_worker_count=max_worker_count,
            label=decode_label,
        ](pools, prof)

        dispatch_merge_flash_partials[
            head_dim, num_q, partial_stride, tp=tp,
            max_worker_count=max_worker_count,
        ](output, partials, nws, pools, prof)
    else:
        var per_q_kv = window if seq_len > window else seq_len
        var data_bytes = seq_len * per_q_kv * kv_stride * elt_bytes
        fanout_dispatch[
            tp, make_prefill, max_worker_count=max_worker_count,
            label=prefill_label,
        ](pools, prof, seq_len, data_bytes)


def dispatch_flash_full[
    P: BurstThreadPool, DecodeK: OutputPartitionedKernel,
    PrefillK: OutputPartitionedKernel, Profile: Bool, N: Int, //,
    head_dim: Int, num_q: Int, local_num_q: Int, partial_stride: Int,
    kv_stride: Int, tp: Int, elt_bytes: Int,
    make_decode: def(Int) capturing [_] -> DecodeK,
    make_prefill: def(Int) capturing [_] -> PrefillK,
    decode_label: StaticString, prefill_label: StaticString,
    max_worker_count: Int = 128,
](
    q_local_output: Binding[BFloat16, tp],
    partials: Binding[Float32, tp],
    base_pos: Int,
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    if seq_len <= 0:
        return
    if seq_len == 1:
        var valid_lens = InlineArray[Int, tp](uninitialized=True)
        for rank in range(tp):
            valid_lens[rank] = full_local_kv_count(rank, base_pos, tp)

        @parameter
        def total_for(r: Int) -> Int:
            return valid_lens[r]

        @parameter
        def bytes_for(r: Int) -> Int:
            return valid_lens[r] * kv_stride * elt_bytes

        var nws = fanout_dispatch_per_rank[
            tp, make_decode, total_for, bytes_for,
            max_worker_count=max_worker_count,
            label=decode_label,
        ](pools, prof)

        dispatch_merge_context_flash_partials[
            head_dim=head_dim, num_q=num_q,
            local_num_q=local_num_q, partial_stride=partial_stride, tp=tp,
            max_worker_count=max_worker_count,
        ](q_local_output, partials, nws, pools, prof)
    else:
        var avg_local_kv = (base_pos + seq_len // 2) // tp + 1
        var data_bytes = seq_len * avg_local_kv * kv_stride * elt_bytes
        fanout_dispatch[
            tp, make_prefill, max_worker_count=max_worker_count,
            label=prefill_label,
        ](pools, prof, seq_len, data_bytes)

        dispatch_merge_flash_prefill_partials[
            head_dim=head_dim, num_q=num_q,
            local_num_q=local_num_q, partial_stride=partial_stride, tp=tp,
            max_worker_count=max_worker_count,
        ](q_local_output, partials, seq_len, pools, prof)


def dispatch_sliding_attention[
    P: BurstThreadPool, Profile: Bool, N: Int, //,
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
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime DecodeK = FlashAttentionKernel[
        RingKV[cache_size],
        head_dim, num_q, gqa_ratio, kv_stride, partial_stride,
    ]
    comptime PrefillK = FlashPrefillSlidingKernel[
        head_dim, num_q, gqa_ratio, kv_stride, window, cache_size,
        partial_stride,
    ]

    @parameter
    def make_decode(r: Int, start_pos: Int) -> DecodeK:
        return DecodeK(q[r], k_base[r], v_base[r], partials[r],
                       0, start_pos, 0, 0)

    @parameter
    def make_prefill(r: Int) -> PrefillK:
        return PrefillK(q[r], k_base[r], v_base[r], output[r],
                        partials[r], base_pos, 0, 0, 0)

    dispatch_flash_sliding[
        head_dim, num_q, partial_stride, kv_stride, window, tp, 2,
        make_decode, make_prefill,
        "sliding_attn.flash", "sliding_attn.prefill",
        max_worker_count=max_worker_count,
    ](output, partials, base_pos, seq_len, pools, prof)


def dispatch_full_attention[
    P: BurstThreadPool, Profile: Bool, N: Int, //,
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
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    """`q_local_output` is the per-rank merged attention output, feeding the
    column-sharded o_proj."""
    comptime DecodeK = FlashAttentionKernel[
        LinearKV, head_dim, num_q, gqa_ratio, kv_stride, partial_stride,
    ]
    comptime PrefillK = FlashPrefillFullKernel[
        head_dim, num_q, gqa_ratio, kv_stride, tp, partial_stride,
    ]

    @parameter
    def make_decode(r: Int) -> DecodeK:
        return DecodeK(q[r], k_base[r], v_base[r], partials[r], 0, 0, 0, 0)

    @parameter
    def make_prefill(r: Int) -> PrefillK:
        return PrefillK(q[r], k_base[r], v_base[r], partials[r],
                        base_pos, r, 0, 0)

    dispatch_flash_full[
        head_dim, num_q, local_num_q, partial_stride, kv_stride, tp, 2,
        make_decode, make_prefill,
        "full_attn.flash", "full_attn.prefill",
        max_worker_count=max_worker_count,
    ](q_local_output, partials, base_pos, seq_len, pools, prof)
