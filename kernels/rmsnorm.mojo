from std.algorithm import vectorize

from simd_math.ops import sqrt
from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from .helpers import (
    Chain, RangePartitionedKernel,
    fanout_dispatch, saturate_workers,
    Binding, BF16Ptr, W,
)
from .dispatch_heuristics import NORM_INLINE_TOKENS
from .dot_products import dot_to_scalar


@always_inline
def rms_reduce_row[hidden: Int](src: BF16Ptr) -> Float32:
    return dot_to_scalar[hidden](src, src)


@always_inline
def rms_normalize_row[hidden: Int, scaled: Bool = True](
    src: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr,
    inv_rms: Float32,
):
    def step[width: Int](idx: Int) {read}:
        var x = (src + idx).load[width=width]().cast[DType.float32]()
        var factor = SIMD[DType.float32, width](inv_rms)
        comptime
        if scaled:
            var w = (weight + idx).load[width=width]().cast[DType.float32]()
            (dst + idx).store((x * factor * w).cast[DType.bfloat16]())
        else:
            (dst + idx).store((x * factor).cast[DType.bfloat16]())

    vectorize[W](hidden, step)


@always_inline
def rms_norm_row[
    hidden: Int, sqrt_n: Float32, n_eps: Float32,
    scaled: Bool = True,
](
    src: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr,
):
    var sum_sq = rms_reduce_row[hidden](src)
    var inv_rms = sqrt_n / sqrt[DType.float32, 1](sum_sq + n_eps)
    rms_normalize_row[hidden, scaled](src, dst, weight, inv_rms)


@always_inline
def norm_residual_add_row[
    hidden: Int, sqrt_n: Float32, n_eps: Float32,
](
    src: BF16Ptr, residual: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr,
):
    var sum_sq = rms_reduce_row[hidden](src)
    var inv_rms = sqrt_n / sqrt[DType.float32, 1](sum_sq + n_eps)

    def step[width: Int](idx: Int) {read}:
        var x = (src + idx).load[width=width]().cast[DType.float32]()
        var r = (residual + idx).load[width=width]().cast[DType.float32]()
        var w = (weight + idx).load[width=width]().cast[DType.float32]()
        var factor = SIMD[DType.float32, width](inv_rms)
        (dst + idx).store((r + x * factor * w).cast[DType.bfloat16]())

    vectorize[W](hidden, step)


@fieldwise_init
struct RmsNormTokenKernel[
    hidden: Int, sqrt_n: Float32, n_eps: Float32,
    scaled: Bool = True,
](RangePartitionedKernel):
    var src: BF16Ptr
    var dst: BF16Ptr
    var weight: BF16Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        for tok in range(self.start, self.end):
            rms_norm_row[Self.hidden, Self.sqrt_n, Self.n_eps, Self.scaled](
                self.src + tok * Self.hidden,
                self.dst + tok * Self.hidden,
                self.weight)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


@fieldwise_init
struct NormResidualAddTokenKernel[
    hidden: Int, sqrt_n: Float32, n_eps: Float32,
](RangePartitionedKernel):
    var src: BF16Ptr
    var residual: BF16Ptr
    var dst: BF16Ptr
    var weight: BF16Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        for tok in range(self.start, self.end):
            var off = tok * Self.hidden
            norm_residual_add_row[Self.hidden, Self.sqrt_n, Self.n_eps](
                self.src + off, self.residual + off, self.dst + off,
                self.weight)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_rms_norm[
    P: BurstThreadPool, //,
    hidden: Int, sqrt_n: Float32, n_eps: Float32,
    tp: Int, scaled: Bool = True, max_worker_count: Int = 128,
](
    src: Binding[BFloat16, tp],
    dst: Binding[BFloat16, tp],
    weight: Binding[BFloat16, tp],
    count: Int,
    mut pools: HeapMoveArray[P],
):
    comptime K = RmsNormTokenKernel[hidden, sqrt_n, n_eps, scaled]

    @parameter
    def make(r: Int) -> K:
        return K(src[r], dst[r], weight[r], 0, 0)

    fanout_dispatch[tp, make, max_worker_count=max_worker_count](
        pools, count, count * hidden * 2,
        inline_threshold_bytes=NORM_INLINE_TOKENS * hidden * 2)


@fieldwise_init
struct ScaledNormKernel[
    hidden: Int, sqrt_n: Float32, n_eps: Float32,
    scaled: Bool, numer: Int, denom: Int,
](RangePartitionedKernel):
    var src: BF16Ptr
    var dst: BF16Ptr
    var weight: BF16Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        var my_start = self.start * Self.numer // Self.denom
        var my_end = self.end * Self.numer // Self.denom
        for tok in range(my_start, my_end):
            rms_norm_row[Self.hidden, Self.sqrt_n, Self.n_eps, Self.scaled](
                self.src + tok * Self.hidden,
                self.dst + tok * Self.hidden,
                self.weight)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_rms_norm_qkv_heads[
    P: BurstThreadPool, //,
    head_dim: Int, sqrt_n: Float32, n_eps: Float32,
    num_q: Int, num_kv: Int, tp: Int, max_worker_count: Int = 128,
](
    q_src: Binding[BFloat16, tp],
    q_dst: Binding[BFloat16, tp],
    k_src: Binding[BFloat16, tp],
    k_dst: Binding[BFloat16, tp],
    v_src: Binding[BFloat16, tp],
    v_dst: Binding[BFloat16, tp],
    q_weight: Binding[BFloat16, tp],
    k_weight: Binding[BFloat16, tp],
    seq_len: Int,
    mut pools: HeapMoveArray[P],
):
    if seq_len <= 0:
        return
    comptime heads_per_token = num_q + num_kv + num_kv
    comptime VK = ScaledNormKernel[head_dim, sqrt_n, n_eps, False, num_kv, heads_per_token]
    comptime QK = ScaledNormKernel[head_dim, sqrt_n, n_eps, True, num_q, heads_per_token]
    comptime KK = ScaledNormKernel[head_dim, sqrt_n, n_eps, True, num_kv, heads_per_token]
    comptime VQChain = Chain[VK, QK]
    comptime VQKChain = Chain[VQChain, KK]

    @parameter
    def make(r: Int) -> VQKChain:
        return VQKChain(
            VQChain(
                VK(v_src[r], v_dst[r], k_weight[r], 0, 0),
                QK(q_src[r], q_dst[r], q_weight[r], 0, 0),
            ),
            KK(k_src[r], k_dst[r], k_weight[r], 0, 0),
        )

    var total = seq_len * heads_per_token
    fanout_dispatch[
        tp, make,
        max_worker_count=max_worker_count,
        worker_policy=saturate_workers,
    ](pools, total, total * head_dim * 2,
      inline_threshold_bytes=NORM_INLINE_TOKENS * head_dim * 2)


def fused_norm_residual_add[
    P: BurstThreadPool, //,
    hidden: Int, sqrt_n: Float32, n_eps: Float32,
    tp: Int, max_worker_count: Int = 128,
](
    src: Binding[BFloat16, tp],
    residual: Binding[BFloat16, tp],
    dst: Binding[BFloat16, tp],
    weight: Binding[BFloat16, tp],
    seq_len: Int,
    mut pools: HeapMoveArray[P],
):
    comptime K = NormResidualAddTokenKernel[hidden, sqrt_n, n_eps]

    @parameter
    def make(r: Int) -> K:
        return K(src[r], residual[r], dst[r], weight[r], 0, 0)

    fanout_dispatch[tp, make, max_worker_count=max_worker_count](
        pools, seq_len, seq_len * hidden * 4,
        inline_threshold_bytes=NORM_INLINE_TOKENS * hidden * 4)
