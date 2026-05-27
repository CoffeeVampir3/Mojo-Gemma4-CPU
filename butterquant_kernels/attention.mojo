from std.collections import InlineArray

from threading.threading_traits import BurstThreadPool
from kernels.helpers import (
    BF16Ptr, F32Ptr, W, Binding, RangePartitionedKernel,
    WorkerRangePartitionedKernel, fanout_dispatch, fanout_dispatch_per_rank,
    accumulate_scaled, scale_unrolled,
)
from kernels.attention_ops import (
    KVSlot, RingKV, LinearKV, TILE, online_softmax_tile, zero_accumulators,
    full_local_kv_count,
)
from kernels.attention_dispatch_kernels import sliding_valid_len
from kernels.logsum_merge import (
    dispatch_merge_flash_partials, dispatch_merge_context_flash_partials,
)
from kernels.flash_attention_prefill import dispatch_merge_flash_prefill_partials
from kernels.dispatch_heuristics import ROPE_INLINE_TOKENS

from butterquant.dot_products import vpdpbusd
from butterquant.head_prep import prep_head_qk_i8, prep_head_v_i8
from butterquant.types import I8Ptr, WI


@always_inline
def vnni_score_dot[head_dim: Int](k_i8: I8Ptr, q_i8: I8Ptr) -> Int32:
    comptime bytes = WI * 4
    comptime assert head_dim % bytes == 0, (
        "score head_dim must be a multiple of WI*4")
    var acc = SIMD[DType.int32, WI](0)
    for k in range(0, head_dim, bytes):
        var kv = (k_i8 + k).bitcast[UInt8]().load[width=bytes]() ^ SIMD[
            DType.uint8, bytes](0x80)
        var qv = (q_i8 + k).load[width=bytes]()
        acc = vpdpbusd[WI](acc, kv, qv)
    return acc.reduce_add()


@always_inline
def bq_process_kv_tile[
    num_q: Int, //,
    KV: KVSlot, head_dim: Int, gqa_ratio: Int, kv_stride: Int, num_kv: Int,
](
    read q_ptrs: InlineArray[I8Ptr, num_q],
    read qi_bias: InlineArray[Float32, num_q],
    read f_q: InlineArray[Float32, num_q],
    k_base: I8Ptr, v_base: I8Ptr,
    k_scale: F32Ptr, v_scale: F32Ptr,
    start_pos: Int, pos: Int, tile_len: Int,
    mut m: InlineArray[Float32, num_q],
    mut l: InlineArray[Float32, num_q],
    read acc_ptrs: InlineArray[F32Ptr, num_q],
):
    comptime inv127 = Float32(1.0) / Float32(127.0)
    comptime inv127sq = inv127 * inv127

    comptime for q_idx in range(num_q):
        comptime kv_h = q_idx // gqa_ratio

        var scores = SIMD[DType.float32, TILE](-1e30)
        for t in range(tile_len):
            var s_idx = KV.slot(start_pos, pos + t)
            var k_head = k_base + s_idx * kv_stride + kv_h * head_dim
            var r = vnni_score_dot[head_dim](k_head, q_ptrs[q_idx])
            var ks = k_scale[s_idx * num_kv + kv_h]
            scores[t] = (Float32(r) - qi_bias[q_idx]) * f_q[q_idx] * ks * inv127sq

        var sm = online_softmax_tile[TILE](scores, m[q_idx])
        var m_new = sm[0]
        var corr = sm[1]
        var weights = sm[2]

        scale_unrolled[cols=head_dim](acc_ptrs[q_idx], corr)
        l[q_idx] = l[q_idx] * corr + weights.reduce_add()
        m[q_idx] = m_new

        for t in range(tile_len):
            var s_idx = KV.slot(start_pos, pos + t)
            var v_head = v_base + s_idx * kv_stride + kv_h * head_dim
            var vs = v_scale[s_idx * num_kv + kv_h]
            accumulate_scaled[cols=head_dim](
                v_head, weights[t] * vs * inv127, acc_ptrs[q_idx])


@fieldwise_init
struct BqFlashAttentionKernel[
    KV: KVSlot,
    head_dim: Int, num_q: Int, num_kv: Int, gqa_ratio: Int, kv_stride: Int,
    partial_stride: Int,
](WorkerRangePartitionedKernel):
    var q: I8Ptr
    var qi_bias: F32Ptr
    var f_q: F32Ptr
    var k_base: I8Ptr
    var k_scale: F32Ptr
    var v_base: I8Ptr
    var v_scale: F32Ptr
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
        var q_ptrs = InlineArray[I8Ptr, Self.num_q](uninitialized=True)
        var qb = InlineArray[Float32, Self.num_q](uninitialized=True)
        var fq = InlineArray[Float32, Self.num_q](uninitialized=True)
        var m = InlineArray[Float32, Self.num_q](fill=Float32(-1e30))
        var l = InlineArray[Float32, Self.num_q](fill=Float32(0))

        comptime for h in range(Self.num_q):
            acc_ptrs[h] = my_partial + h * Self.head_dim
            q_ptrs[h] = self.q + h * Self.head_dim
            qb[h] = self.qi_bias[h]
            fq[h] = self.f_q[h]

        zero_accumulators[Self.num_q, Self.head_dim](acc_ptrs)

        var pos = self.start
        while pos < self.end:
            var tile_len = min(TILE, self.end - pos)
            bq_process_kv_tile[
                Self.KV, Self.head_dim, Self.gqa_ratio, Self.kv_stride,
                Self.num_kv,
            ](q_ptrs, qb, fq, self.k_base, self.v_base,
              self.k_scale, self.v_scale,
              self.start_pos, pos, tile_len, m, l, acc_ptrs)
            pos += TILE

        comptime for h in range(Self.num_q):
            (my_partial + m_off + h)[] = m[h]
            (my_partial + l_off + h)[] = l[h]

    @always_inline
    def install_worker_range(mut self, worker_id: Int, start: Int, end: Int):
        self.worker_id = worker_id
        self.start = start
        self.end = end


@fieldwise_init
struct BqFlashPrefillSlidingKernel[
    head_dim: Int, num_q: Int, num_kv: Int, gqa_ratio: Int,
    kv_stride: Int, window: Int, cache_size: Int, partial_stride: Int,
](WorkerRangePartitionedKernel):
    var q: I8Ptr
    var qi_bias: F32Ptr
    var f_q: F32Ptr
    var k_base: I8Ptr
    var k_scale: F32Ptr
    var v_base: I8Ptr
    var v_scale: F32Ptr
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
        var q_ptrs = InlineArray[I8Ptr, Self.num_q](uninitialized=True)
        comptime for h in range(Self.num_q):
            acc_ptrs[h] = scratch + h * Self.head_dim

        for t in range(self.start, self.end):
            var abs_pos = self.base_pos + t
            var lo = max(0, abs_pos - Self.window + 1)
            var hi = abs_pos + 1

            var q_tok = self.q + t * q_stride
            var out_tok = self.output + t * q_stride
            var qb = InlineArray[Float32, Self.num_q](uninitialized=True)
            var fq = InlineArray[Float32, Self.num_q](uninitialized=True)
            var m = InlineArray[Float32, Self.num_q](fill=Float32(-1e30))
            var l = InlineArray[Float32, Self.num_q](fill=Float32(0))
            comptime for h in range(Self.num_q):
                q_ptrs[h] = q_tok + h * Self.head_dim
                qb[h] = self.qi_bias[t * Self.num_q + h]
                fq[h] = self.f_q[t * Self.num_q + h]

            zero_accumulators[Self.num_q, Self.head_dim](acc_ptrs)

            var pos = lo
            while pos < hi:
                var tile_len = min(TILE, hi - pos)
                bq_process_kv_tile[
                    RingKV[Self.cache_size], Self.head_dim, Self.gqa_ratio,
                    Self.kv_stride, Self.num_kv,
                ](q_ptrs, qb, fq, self.k_base, self.v_base,
                  self.k_scale, self.v_scale,
                  0, pos, tile_len, m, l, acc_ptrs)
                pos += TILE

            comptime for h in range(Self.num_q):
                if l[h] > 0:
                    var inv_l = SIMD[DType.float32, W](Float32(1.0) / l[h])
                    for j in range(0, Self.head_dim, W):
                        var v = (acc_ptrs[h] + j).load[width=W]() * inv_l
                        (out_tok + h * Self.head_dim + j).store(
                            v.cast[DType.bfloat16]())
                else:
                    for j in range(0, Self.head_dim, W):
                        (out_tok + h * Self.head_dim + j).store(
                            SIMD[DType.bfloat16, W](0))

    @always_inline
    def install_worker_range(mut self, worker_id: Int, start: Int, end: Int):
        self.worker_id = worker_id
        self.start = start
        self.end = end


@fieldwise_init
struct BqFlashPrefillFullKernel[
    head_dim: Int, num_q: Int, num_kv: Int, gqa_ratio: Int,
    kv_stride: Int, degree: Int, partial_stride: Int,
](RangePartitionedKernel):
    var q: I8Ptr
    var qi_bias: F32Ptr
    var f_q: F32Ptr
    var k_base: I8Ptr
    var k_scale: F32Ptr
    var v_base: I8Ptr
    var v_scale: F32Ptr
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
        var q_ptrs = InlineArray[I8Ptr, Self.num_q](uninitialized=True)

        for t in range(self.start, self.end):
            var abs_pos = self.base_pos + t
            var local_kv_count = full_local_kv_count(
                self.rank, abs_pos, Self.degree)

            var partial_tok = self.partials + t * Self.partial_stride
            var q_tok = self.q + t * q_stride
            var qb = InlineArray[Float32, Self.num_q](uninitialized=True)
            var fq = InlineArray[Float32, Self.num_q](uninitialized=True)
            var m = InlineArray[Float32, Self.num_q](fill=Float32(-1e30))
            var l = InlineArray[Float32, Self.num_q](fill=Float32(0))

            comptime for h in range(Self.num_q):
                acc_ptrs[h] = partial_tok + h * Self.head_dim
                q_ptrs[h] = q_tok + h * Self.head_dim
                qb[h] = self.qi_bias[t * Self.num_q + h]
                fq[h] = self.f_q[t * Self.num_q + h]

            zero_accumulators[Self.num_q, Self.head_dim](acc_ptrs)

            var pos = 0
            while pos < local_kv_count:
                var tile_len = min(TILE, local_kv_count - pos)
                bq_process_kv_tile[
                    LinearKV, Self.head_dim, Self.gqa_ratio,
                    Self.kv_stride, Self.num_kv,
                ](q_ptrs, qb, fq, self.k_base, self.v_base,
                  self.k_scale, self.v_scale,
                  0, pos, tile_len, m, l, acc_ptrs)
                pos += TILE

            comptime for h in range(Self.num_q):
                (partial_tok + m_off + h)[] = m[h]
                (partial_tok + l_off + h)[] = l[h]

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_bq_sliding_attention[
    P: BurstThreadPool, //,
    head_dim: Int, num_q: Int, num_kv: Int, gqa_ratio: Int,
    kv_stride: Int, window: Int, cache_size: Int, partial_stride: Int, tp: Int,
    max_worker_count: Int = 128,
](
    q: Binding[Int8, tp],
    qi_bias: Binding[Float32, tp],
    f_q: Binding[Float32, tp],
    k_base: Binding[Int8, tp],
    k_scale: Binding[Float32, tp],
    v_base: Binding[Int8, tp],
    v_scale: Binding[Float32, tp],
    output: Binding[BFloat16, tp],
    partials: Binding[Float32, tp],
    base_pos: Int,
    seq_len: Int,
    mut pools: List[P],
):
    if seq_len <= 0:
        return
    if seq_len == 1:
        var valid_len = sliding_valid_len(base_pos, window)
        if valid_len <= 0:
            return
        var start_pos = base_pos - valid_len + 1
        comptime DecodeK = BqFlashAttentionKernel[
            RingKV[cache_size],
            head_dim, num_q, num_kv, gqa_ratio, kv_stride, partial_stride,
        ]

        @parameter
        def make_decode(r: Int) -> DecodeK:
            return DecodeK(q[r], qi_bias[r], f_q[r], k_base[r], k_scale[r],
                           v_base[r], v_scale[r], partials[r],
                           0, start_pos, 0, 0)

        @parameter
        def total_for(r: Int) -> Int:
            return valid_len

        @parameter
        def bytes_for(r: Int) -> Int:
            return valid_len * kv_stride

        var nws = fanout_dispatch_per_rank[
            tp, make_decode, total_for, bytes_for,
            max_worker_count=max_worker_count,
        ](pools)

        dispatch_merge_flash_partials[
            head_dim, num_q, partial_stride, tp=tp,
            max_worker_count=max_worker_count,
        ](output, partials, nws, pools)
    else:
        comptime PrefillK = BqFlashPrefillSlidingKernel[
            head_dim, num_q, num_kv, gqa_ratio, kv_stride, window, cache_size,
            partial_stride,
        ]

        @parameter
        def make_prefill(r: Int) -> PrefillK:
            return PrefillK(q[r], qi_bias[r], f_q[r], k_base[r], k_scale[r],
                            v_base[r], v_scale[r], output[r], partials[r],
                            base_pos, 0, 0, 0)

        var per_q_kv = window if seq_len > window else seq_len
        fanout_dispatch[
            tp, make_prefill, max_worker_count=max_worker_count,
        ](pools, seq_len, seq_len * per_q_kv * kv_stride)


def dispatch_bq_full_attention[
    P: BurstThreadPool, //,
    head_dim: Int, num_q: Int, num_kv: Int, local_num_q: Int, gqa_ratio: Int,
    kv_stride: Int, partial_stride: Int, tp: Int,
    max_worker_count: Int = 128,
](
    q: Binding[Int8, tp],
    qi_bias: Binding[Float32, tp],
    f_q: Binding[Float32, tp],
    k_base: Binding[Int8, tp],
    k_scale: Binding[Float32, tp],
    v_base: Binding[Int8, tp],
    v_scale: Binding[Float32, tp],
    q_local_output: Binding[BFloat16, tp],
    partials: Binding[Float32, tp],
    base_pos: Int,
    seq_len: Int,
    mut pools: List[P],
):
    if seq_len <= 0:
        return
    if seq_len == 1:
        var valid_lens = InlineArray[Int, tp](uninitialized=True)
        for rank in range(tp):
            valid_lens[rank] = full_local_kv_count(rank, base_pos, tp)

        comptime DecodeK = BqFlashAttentionKernel[
            LinearKV, head_dim, num_q, num_kv, gqa_ratio, kv_stride,
            partial_stride,
        ]

        @parameter
        def make_decode(r: Int) -> DecodeK:
            return DecodeK(q[r], qi_bias[r], f_q[r], k_base[r], k_scale[r],
                           v_base[r], v_scale[r], partials[r], 0, 0, 0, 0)

        @parameter
        def total_for(r: Int) -> Int:
            return valid_lens[r]

        @parameter
        def bytes_for(r: Int) -> Int:
            return valid_lens[r] * kv_stride

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
        comptime PrefillK = BqFlashPrefillFullKernel[
            head_dim, num_q, num_kv, gqa_ratio, kv_stride, tp, partial_stride,
        ]

        @parameter
        def make_prefill(r: Int) -> PrefillK:
            return PrefillK(q[r], qi_bias[r], f_q[r], k_base[r], k_scale[r],
                            v_base[r], v_scale[r], partials[r],
                            base_pos, r, 0, 0)

        var avg_local_kv = (base_pos + seq_len // 2) // tp + 1
        fanout_dispatch[
            tp, make_prefill, max_worker_count=max_worker_count,
        ](pools, seq_len, seq_len * avg_local_kv * kv_stride)

        dispatch_merge_flash_prefill_partials[
            head_dim=head_dim, num_q=num_q,
            local_num_q=local_num_q, partial_stride=partial_stride, tp=tp,
            max_worker_count=max_worker_count,
        ](q_local_output, partials, seq_len, pools)


@fieldwise_init
struct BqAttnPrepKernel[
    head_dim: Int, num_q: Int, num_kv: Int, rope_half: Int, pair_stride: Int,
    slot_mask: Int, cache_degree: Int, sqrt_n: Float32, n_eps: Float32,
](RangePartitionedKernel):
    var q_src: BF16Ptr
    var k_src: BF16Ptr
    var v_src: BF16Ptr
    var q_norm: BF16Ptr
    var k_norm: BF16Ptr
    var q_i8: I8Ptr
    var qi_bias: F32Ptr
    var f_q: F32Ptr
    var k_cache: I8Ptr
    var k_scale: F32Ptr
    var v_cache: I8Ptr
    var v_scale: F32Ptr
    var cos_table: F32Ptr
    var sin_table: F32Ptr
    var base_pos: Int
    var rank: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime q_stride = Self.num_q * Self.head_dim
        comptime kv_stride = Self.num_kv * Self.head_dim
        for tok in range(self.start, self.end):
            var pos = self.base_pos + tok
            var cos_row = self.cos_table + pos * Self.rope_half
            var sin_row = self.sin_table + pos * Self.rope_half

            var q_tok = self.q_src + tok * q_stride
            var qi_tok = self.q_i8 + tok * q_stride
            for h in range(Self.num_q):
                var res = prep_head_qk_i8[
                    Self.head_dim, Self.rope_half, Self.pair_stride,
                    Self.sqrt_n, Self.n_eps,
                ](q_tok + h * Self.head_dim, self.q_norm, cos_row, sin_row,
                  qi_tok + h * Self.head_dim)
                (self.qi_bias + tok * Self.num_q + h)[] = Float32(res[1]) * 128.0
                (self.f_q + tok * Self.num_q + h)[] = res[0]

            if pos % Self.cache_degree == self.rank:
                var slot = (pos // Self.cache_degree) & Self.slot_mask
                var k_tok = self.k_src + tok * kv_stride
                var v_tok = self.v_src + tok * kv_stride
                var k_dst = self.k_cache + slot * kv_stride
                var v_dst = self.v_cache + slot * kv_stride
                var ks_dst = self.k_scale + slot * Self.num_kv
                var vs_dst = self.v_scale + slot * Self.num_kv
                for h in range(Self.num_kv):
                    var sk = prep_head_qk_i8[
                        Self.head_dim, Self.rope_half, Self.pair_stride,
                        Self.sqrt_n, Self.n_eps,
                    ](k_tok + h * Self.head_dim, self.k_norm, cos_row, sin_row,
                      k_dst + h * Self.head_dim)
                    ks_dst[h] = sk[0]
                    var sv = prep_head_v_i8[
                        Self.head_dim, Self.sqrt_n, Self.n_eps,
                    ](v_tok + h * Self.head_dim, v_dst + h * Self.head_dim)
                    vs_dst[h] = sv

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_bq_attn_prep[
    P: BurstThreadPool, //,
    head_dim: Int, num_q: Int, num_kv: Int, rope_half: Int, pair_stride: Int,
    slot_mask: Int, cache_degree: Int, sqrt_n: Float32, n_eps: Float32, tp: Int,
    max_worker_count: Int = 128,
](
    q_src: Binding[BFloat16, tp],
    k_src: Binding[BFloat16, tp],
    v_src: Binding[BFloat16, tp],
    q_norm: Binding[BFloat16, tp],
    k_norm: Binding[BFloat16, tp],
    q_i8: Binding[Int8, tp],
    qi_bias: Binding[Float32, tp],
    f_q: Binding[Float32, tp],
    k_cache: Binding[Int8, tp],
    k_scale: Binding[Float32, tp],
    v_cache: Binding[Int8, tp],
    v_scale: Binding[Float32, tp],
    cos_table: Binding[Float32, tp],
    sin_table: Binding[Float32, tp],
    base_pos: Int, seq_len: Int,
    mut pools: List[P],
):
    comptime K = BqAttnPrepKernel[
        head_dim, num_q, num_kv, rope_half, pair_stride,
        slot_mask, cache_degree, sqrt_n, n_eps]
    comptime row_bytes = (num_q + 2 * num_kv) * head_dim * 6

    @parameter
    def make(r: Int) -> K:
        return K(q_src[r], k_src[r], v_src[r], q_norm[r], k_norm[r],
                 q_i8[r], qi_bias[r], f_q[r],
                 k_cache[r], k_scale[r], v_cache[r], v_scale[r],
                 cos_table[r], sin_table[r],
                 base_pos, r % cache_degree, 0, 0)

    fanout_dispatch[tp, make, max_worker_count=max_worker_count](
        pools, seq_len, seq_len * row_bytes,
        inline_threshold_bytes=ROPE_INLINE_TOKENS * row_bytes)
