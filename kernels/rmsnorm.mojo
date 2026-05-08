from std.algorithm import vectorize
from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from simd_math.ops import sqrt
from simd_math import pick_port_unroll, tree_reduce_accs
from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from .helpers import (
    Chain, OutputPartitionedKernel, DispatchBuffer,
    tile_dispatch, recommended_workers, join_all,
    NumaPointerArray,
)


comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime W = simd_width_of[DType.float32]()


@always_inline
def rms_reduce_row[hidden: Int](src: BF16Ptr) -> Scalar[DType.float32]:
    comptime PU = pick_port_unroll[W, hidden]()
    comptime STRIDE = PU * W
    var accs = InlineArray[SIMD[DType.float32, W], PU](fill=SIMD[DType.float32, W](0))
    for i in range(hidden // STRIDE):
        comptime for p in range(PU):
            var v = (src + i * STRIDE + p * W).load[width=W]().cast[DType.float32]()
            accs[p] = v.fma(v, accs[p])
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
def norm_residual_add_row[
    hidden: Int, sqrt_n: Scalar[DType.float32], n_eps: Scalar[DType.float32],
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
    hidden: Int, sqrt_n: Scalar[DType.float32], n_eps: Scalar[DType.float32],
    scaled: Bool = True,
](OutputPartitionedKernel):
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
struct NormResidualAddTokenKernel[
    hidden: Int, sqrt_n: Scalar[DType.float32], n_eps: Scalar[DType.float32],
](OutputPartitionedKernel):
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

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.src, self.residual, self.dst, self.weight, start, end)


comptime NORM_INLINE_TOKENS = 16


def dispatch_rms_norm[
    P: BurstThreadPool, //,
    hidden: Int, sqrt_n: Scalar[DType.float32], n_eps: Scalar[DType.float32],
    tp: Int, scaled: Bool = True,
](
    src: NumaPointerArray[DType.bfloat16, tp],
    dst: NumaPointerArray[DType.bfloat16, tp],
    weight: NumaPointerArray[DType.bfloat16, tp],
    count: Int,
    mut pools: HeapMoveArray[P],
):
    if count <= NORM_INLINE_TOKENS:
        for r in range(tp):
            for tok in range(count):
                rms_norm_row[hidden, sqrt_n, n_eps, scaled](
                    src[r] + tok * hidden, dst[r] + tok * hidden, weight[r])
        return

    var data_bytes = count * hidden * 2
    var buf = DispatchBuffer[RmsNormTokenKernel[hidden, sqrt_n, n_eps, scaled]]()
    for r in range(tp):
        var nw = recommended_workers(data_bytes, pools[r].get_capacity())
        tile_dispatch(buf,
            RmsNormTokenKernel[hidden, sqrt_n, n_eps, scaled](
                src[r], dst[r], weight[r], 0, 0),
            pools[r], count, num_workers=nw)
    join_all[tp](pools)


@fieldwise_init
struct ScaledNormKernel[
    hidden: Int, sqrt_n: Scalar[DType.float32], n_eps: Scalar[DType.float32],
    scaled: Bool, numer: Int, denom: Int,
](OutputPartitionedKernel):
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


def dispatch_rms_norm_qkv_heads[
    P: BurstThreadPool, //,
    head_dim: Int, sqrt_n: Scalar[DType.float32], n_eps: Scalar[DType.float32],
    num_q: Int, num_kv: Int, tp: Int,
](
    q_src: NumaPointerArray[DType.bfloat16, tp],
    q_dst: NumaPointerArray[DType.bfloat16, tp],
    k_src: NumaPointerArray[DType.bfloat16, tp],
    k_dst: NumaPointerArray[DType.bfloat16, tp],
    v_src: NumaPointerArray[DType.bfloat16, tp],
    v_dst: NumaPointerArray[DType.bfloat16, tp],
    q_weight: NumaPointerArray[DType.bfloat16, tp],
    k_weight: NumaPointerArray[DType.bfloat16, tp],
    mut pools: HeapMoveArray[P],
):
    comptime total = num_q + num_kv + num_kv

    if total <= NORM_INLINE_TOKENS:
        for r in range(tp):
            for h in range(num_kv):
                rms_norm_row[head_dim, sqrt_n, n_eps, scaled=False](
                    v_src[r] + h * head_dim, v_dst[r] + h * head_dim, k_weight[r])
            for h in range(num_q):
                rms_norm_row[head_dim, sqrt_n, n_eps](
                    q_src[r] + h * head_dim, q_dst[r] + h * head_dim, q_weight[r])
            for h in range(num_kv):
                rms_norm_row[head_dim, sqrt_n, n_eps](
                    k_src[r] + h * head_dim, k_dst[r] + h * head_dim, k_weight[r])
        return

    comptime VK = ScaledNormKernel[head_dim, sqrt_n, n_eps, False, num_kv, total]
    comptime QK = ScaledNormKernel[head_dim, sqrt_n, n_eps, True, num_q, total]
    comptime KK = ScaledNormKernel[head_dim, sqrt_n, n_eps, True, num_kv, total]
    comptime VQChain = Chain[VK, QK]
    comptime VQKChain = Chain[VQChain, KK]

    var buf = DispatchBuffer[VQKChain]()
    for r in range(tp):
        var nw = pools[r].get_capacity()
        tile_dispatch(buf,
            VQKChain(
                VQChain(
                    VK(v_src[r], v_dst[r], k_weight[r], 0, 0),
                    QK(q_src[r], q_dst[r], q_weight[r], 0, 0),
                ),
                KK(k_src[r], k_dst[r], k_weight[r], 0, 0),
            ),
            pools[r], total, num_workers=nw)
    join_all[tp](pools)


def fused_norm_residual_add[
    P: BurstThreadPool, //,
    hidden: Int, sqrt_n: Scalar[DType.float32], n_eps: Scalar[DType.float32],
    tp: Int,
](
    src: NumaPointerArray[DType.bfloat16, tp],
    residual: NumaPointerArray[DType.bfloat16, tp],
    dst: NumaPointerArray[DType.bfloat16, tp],
    weight: NumaPointerArray[DType.bfloat16, tp],
    seq_len: Int,
    mut pools: HeapMoveArray[P],
):
    if seq_len <= NORM_INLINE_TOKENS:
        for r in range(tp):
            for tok in range(seq_len):
                var off = tok * hidden
                norm_residual_add_row[hidden, sqrt_n, n_eps](
                    src[r] + off, residual[r] + off, dst[r] + off, weight[r])
        return

    var data_bytes = seq_len * hidden * 4
    var buf = DispatchBuffer[NormResidualAddTokenKernel[hidden, sqrt_n, n_eps]]()
    for r in range(tp):
        var nw = recommended_workers(data_bytes, pools[r].get_capacity())
        tile_dispatch(buf,
            NormResidualAddTokenKernel[hidden, sqrt_n, n_eps](
                src[r], residual[r], dst[r], weight[r], 0, 0),
            pools[r], seq_len, num_workers=nw)
    join_all[tp](pools)
