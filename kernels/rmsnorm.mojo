from std.algorithm import vectorize
from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from simd_math.ops import sqrt
from simd_math import pick_port_unroll, tree_reduce_accs
from threading.threading_traits import BurstKernel, BurstThreadPool
from notstdcollections import HeapMoveArray
from .helpers import (
    Chain, RangedKernel, DispatchBuffer, tile_dispatch, recommended_workers,
)


comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
comptime W = simd_width_of[DType.float32]()


@always_inline
def rms_reduce_row[hidden: Int](src: BF16Ptr) -> Scalar[DType.float32]:
    comptime PU = pick_port_unroll[W, hidden]()
    comptime STRIDE = PU * W
    var accs = InlineArray[SIMD[DType.float32, W], PU](fill=SIMD[DType.float32, W](0))
    for i in range(hidden // STRIDE):
        comptime for p in range(PU):
            var v = (src + i * STRIDE + p * W).load[width=W]().cast[DType.float32]()
            accs[p] = accs[p].fma(v, v)
    return tree_reduce_accs(accs)


@always_inline
def rms_normalize_row[hidden: Int, scaled: Bool = True](
    src: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr,
    inv_rms: Scalar[DType.float32],
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
    hidden: Int, sqrt_n: Scalar[DType.float32], n_eps: Scalar[DType.float32],
    scaled: Bool = True,
](
    src: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr,
):
    var sum_sq = rms_reduce_row[hidden](src)
    var inv_rms = sqrt_n / sqrt[DType.float32, 1](sum_sq + n_eps)
    rms_normalize_row[hidden, scaled](src, dst, weight, inv_rms)


@always_inline
def residual_add_rms_reduce_row[hidden: Int](
    partial: BF16Ptr, residual: BF16Ptr, dst: BF16Ptr,
) -> Scalar[DType.float32]:
    comptime PU = pick_port_unroll[W, hidden]()
    comptime STRIDE = PU * W
    var accs = InlineArray[SIMD[DType.float32, W], PU](fill=SIMD[DType.float32, W](0))
    for i in range(hidden // STRIDE):
        comptime for p in range(PU):
            var off = i * STRIDE + p * W
            var p_val = (partial + off).load[width=W]().cast[DType.float32]()
            var r_val = (residual + off).load[width=W]().cast[DType.float32]()
            var sum = p_val + r_val
            (dst + off).store(sum.cast[DType.bfloat16]())
            accs[p] = accs[p].fma(sum, sum)
    return tree_reduce_accs(accs)


@always_inline
def fused_residual_norm_row[
    hidden: Int, sqrt_n: Scalar[DType.float32], n_eps: Scalar[DType.float32],
](
    partial: BF16Ptr, residual: BF16Ptr,
    residual_dst: BF16Ptr, norm_dst: BF16Ptr,
    weight: BF16Ptr,
):
    var sum_sq = residual_add_rms_reduce_row[hidden](partial, residual, residual_dst)
    var inv_rms = sqrt_n / sqrt[DType.float32, 1](sum_sq + n_eps)
    rms_normalize_row[hidden](residual_dst, norm_dst, weight, inv_rms)


# === Dispatched kernels (token-parallel) ===


@fieldwise_init
struct RmsNormTokenKernel[
    hidden: Int, sqrt_n: Scalar[DType.float32], n_eps: Scalar[DType.float32],
    scaled: Bool = True,
](RangedKernel):
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

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.src, self.dst, self.weight, start, end)


@fieldwise_init
struct FusedResidualNormTokenKernel[
    hidden: Int, sqrt_n: Scalar[DType.float32], n_eps: Scalar[DType.float32],
](RangedKernel):
    var partial: BF16Ptr
    var residual: BF16Ptr
    var residual_dst: BF16Ptr
    var norm_dst: BF16Ptr
    var weight: BF16Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        for tok in range(self.start, self.end):
            var off = tok * Self.hidden
            fused_residual_norm_row[Self.hidden, Self.sqrt_n, Self.n_eps](
                self.partial + off, self.residual + off,
                self.residual_dst + off, self.norm_dst + off,
                self.weight)

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.partial, self.residual, self.residual_dst,
                    self.norm_dst, self.weight, start, end)


# TODO: Make this a parameter
comptime NORM_INLINE_TOKENS = 16


def rms_norm[
    P: BurstThreadPool, //,
    hidden: Int, sqrt_n: Scalar[DType.float32], n_eps: Scalar[DType.float32],
    scaled: Bool = True,
](
    src: BF16Ptr, dst: BF16Ptr, weight: BF16Ptr,
    count: Int, mut pool: P,
):
    if count <= NORM_INLINE_TOKENS:
        for tok in range(count):
            rms_norm_row[hidden, sqrt_n, n_eps, scaled](
                src + tok * hidden, dst + tok * hidden, weight)
        return

    var data_bytes = count * hidden * 2
    var nw = recommended_workers(data_bytes, pool.get_capacity())
    var buf = DispatchBuffer[RmsNormTokenKernel[hidden, sqrt_n, n_eps, scaled]]()
    tile_dispatch(buf,
        RmsNormTokenKernel[hidden, sqrt_n, n_eps, scaled](src, dst, weight, 0, 0),
        pool, count, num_workers=nw)
    pool.join()


@fieldwise_init
struct ScaledNormKernel[
    hidden: Int, sqrt_n: Scalar[DType.float32], n_eps: Scalar[DType.float32],
    scaled: Bool, numer: Int, denom: Int,
](RangedKernel):
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

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.src, self.dst, self.weight, start, end)


def rms_norm_qkv_heads[
    P: BurstThreadPool, //,
    head_dim: Int, sqrt_n: Scalar[DType.float32], n_eps: Scalar[DType.float32],
    num_q: Int, num_kv: Int,
](
    q: BF16Ptr, k: BF16Ptr, v: BF16Ptr,
    q_weight: BF16Ptr, k_weight: BF16Ptr,
    mut pool: P,
):
    comptime total = num_q + num_kv + num_kv

    if total <= NORM_INLINE_TOKENS:
        for h in range(num_q):
            rms_norm_row[head_dim, sqrt_n, n_eps](
                q + h * head_dim, q + h * head_dim, q_weight)
        for h in range(num_kv):
            rms_norm_row[head_dim, sqrt_n, n_eps](
                k + h * head_dim, k + h * head_dim, k_weight)
        for h in range(num_kv):
            rms_norm_row[head_dim, sqrt_n, n_eps, scaled=False](
                v + h * head_dim, v + h * head_dim, k_weight)
        return

    var nw = pool.get_capacity()

    comptime QK = ScaledNormKernel[head_dim, sqrt_n, n_eps, True, num_q, total]
    comptime KK = ScaledNormKernel[head_dim, sqrt_n, n_eps, True, num_kv, total]
    comptime VK = ScaledNormKernel[head_dim, sqrt_n, n_eps, False, num_kv, total]
    comptime QKChain = Chain[QK, KK]
    comptime QKVChain = Chain[QKChain, VK]

    var proto = QKVChain(
        QKChain(
            QK(q, q, q_weight, 0, 0),
            KK(k, k, k_weight, 0, 0),
        ),
        VK(v, v, k_weight, 0, 0),
    )

    var buf = DispatchBuffer[QKVChain]()
    tile_dispatch(buf, proto, pool, total, num_workers=nw)
    pool.join()


def fused_residual_norm[
    P: BurstThreadPool, //,
    hidden: Int, sqrt_n: Scalar[DType.float32], n_eps: Scalar[DType.float32],
](
    partial: BF16Ptr, residual: BF16Ptr,
    residual_dst: BF16Ptr, norm_dst: BF16Ptr,
    weight: BF16Ptr, seq_len: Int, mut pool: P,
):
    if seq_len <= NORM_INLINE_TOKENS:
        for tok in range(seq_len):
            var off = tok * hidden
            fused_residual_norm_row[hidden, sqrt_n, n_eps](
                partial + off, residual + off,
                residual_dst + off, norm_dst + off, weight)
        return

    var data_bytes = seq_len * hidden * 4
    var nw = recommended_workers(data_bytes, pool.get_capacity())
    var buf = DispatchBuffer[FusedResidualNormTokenKernel[hidden, sqrt_n, n_eps]]()
    tile_dispatch(buf,
        FusedResidualNormTokenKernel[hidden, sqrt_n, n_eps](
            partial, residual, residual_dst, norm_dst, weight, 0, 0),
        pool, seq_len, num_workers=nw)
    pool.join()
