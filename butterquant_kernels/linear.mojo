from threading.threading_traits import BurstThreadPool
from kernels.helpers import (
    Chain, RangePartitionedKernel, Binding, fanout_dispatch,
    saturate_workers, BF16Ptr,
)
from kernels.dispatch_heuristics import NORM_INLINE_TOKENS, GEMV_INLINE_ROWS
from kernels.profiling import Profiler

from butterquant.runtime import (
    prepare_norm_activation_per_row, prepare_block_activation,
)
from butterquant.gemm import gemm_i8_per_row, gemm_i8_per_block
from butterquant.vnni import VNNI_N_STEP
from butterquant.types import F32Ptr, I8Ptr
from butterquant.weight import (
    ButterquantWeight, ButterquantActivation, ButterquantBlockActivation,
    quant_vnni_packed, quant_has_colsum, quant_colsum_per_block, quant_k_block,
)
from quant.recipe import QuantRecipe


@fieldwise_init
struct BqNormQuantKernel[
    hidden: Int, block: Int, sqrt_n: Float32, n_eps: Float32,
](RangePartitionedKernel):
    var src: BF16Ptr
    var gamma: BF16Ptr
    var x_i8: I8Ptr
    var x_sa: F32Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        for tok in range(self.start, self.end):
            prepare_norm_activation_per_row[
                Self.hidden, Self.block, Self.sqrt_n, Self.n_eps,
            ](
                self.src + tok * Self.hidden, self.gamma,
                self.x_i8 + tok * Self.hidden, self.x_sa + tok)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_bq_norm_quant[
    P: BurstThreadPool, Profile: Bool, N: Int, //,
    hidden: Int, block: Int, sqrt_n: Float32, n_eps: Float32, tp: Int,
    max_worker_count: Int = 128,
](
    src: Binding[BFloat16, tp],
    gamma: Binding[BFloat16, tp],
    x_i8: Binding[Int8, tp],
    x_sa: Binding[Float32, tp],
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime Kern = BqNormQuantKernel[hidden, block, sqrt_n, n_eps]

    @parameter
    def make(r: Int) -> Kern:
        return Kern(src[r], gamma[r], x_i8[r], x_sa[r], 0, 0)

    fanout_dispatch[tp, make, max_worker_count=max_worker_count, label="bq_norm_quant"](
        pools, prof, seq_len, seq_len * hidden * 6,
        inline_threshold_bytes=NORM_INLINE_TOKENS * hidden * 6)


@fieldwise_init
struct BqLinearKernel[
    N: Int, K: Int, MR: Int, numer: Int = 1, denom: Int = 1,
](RangePartitionedKernel):
    var act: I8Ptr
    var act_scale: F32Ptr
    var weight: I8Ptr
    var wsc: F32Ptr
    var colsum: F32Ptr
    var output: BF16Ptr
    var m: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var my_start = self.start * Self.numer // Self.denom
        var my_end = self.end * Self.numer // Self.denom
        gemm_i8_per_row[Self.N, Self.K, Self.MR, DType.bfloat16](
            self.act, self.m, self.act_scale, self.weight, self.wsc,
            self.colsum, self.output, my_start, my_end)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_bq_linear[
    P: BurstThreadPool, quant: QuantRecipe, n: Int, m: Int, tp: Int,
    Profile: Bool, N: Int, //,
    MR: Int = 4, max_worker_count: Int = 128,
](
    act: ButterquantActivation[tp],
    weight: ButterquantWeight[quant, n, m, tp],
    output: Binding[BFloat16, tp],
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime assert quant_vnni_packed[quant](), "bq linear consumes a VNNI-packed weight"
    comptime assert quant_has_colsum[quant](), "bq linear requires a colsum sidecar"
    if seq_len <= 0:
        return
    comptime num_tiles = n // VNNI_N_STEP
    comptime Kern = BqLinearKernel[n, m, MR]
    var cs = weight.colsum_checked()

    @parameter
    def make(r: Int) -> Kern:
        return Kern(act.data[r], act.scale[r], weight.data[r], weight.scale[r],
                    cs[r], output[r], seq_len, 0, 0)

    fanout_dispatch[tp, make, max_worker_count=max_worker_count, label="bq_linear"](
        pools, prof, num_tiles, seq_len * m + n * m,
        inline_threshold_bytes=GEMV_INLINE_ROWS * m)


@fieldwise_init
struct BqBlockQuantKernel[cols: Int, block: Int, apply_fwht: Bool](
    RangePartitionedKernel
):
    var src: BF16Ptr
    var x_i8: I8Ptr
    var x_sa: F32Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        comptime nb = Self.cols // Self.block
        for tok in range(self.start, self.end):
            prepare_block_activation[Self.cols, Self.block, Self.apply_fwht](
                self.src + tok * Self.cols, self.x_i8 + tok * Self.cols,
                self.x_sa + tok * nb)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_bq_block_quant[
    P: BurstThreadPool, Profile: Bool, N: Int, //,
    cols: Int, block: Int, apply_fwht: Bool, tp: Int,
    max_worker_count: Int = 128,
](
    src: Binding[BFloat16, tp],
    x_i8: Binding[Int8, tp],
    x_sa: Binding[Float32, tp],
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime Kern = BqBlockQuantKernel[cols, block, apply_fwht]

    @parameter
    def make(r: Int) -> Kern:
        return Kern(src[r], x_i8[r], x_sa[r], 0, 0)

    fanout_dispatch[tp, make, max_worker_count=max_worker_count, label="bq_block_quant"](
        pools, prof, seq_len, seq_len * cols * 6,
        inline_threshold_bytes=NORM_INLINE_TOKENS * cols * 6)


@fieldwise_init
struct BqBlockLinearKernel[N: Int, K: Int, block: Int, MR: Int](
    RangePartitionedKernel
):
    var act: I8Ptr
    var act_scale: F32Ptr
    var weight: I8Ptr
    var wsc: F32Ptr
    var colsum: F32Ptr
    var output: BF16Ptr
    var m: Int
    var start: Int
    var end: Int

    def execute(mut self):
        gemm_i8_per_block[Self.N, Self.K, Self.block, Self.MR, DType.bfloat16](
            self.act, self.m, self.act_scale, self.weight, self.wsc,
            self.colsum, self.output, self.start, self.end)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_bq_block_linear[
    P: BurstThreadPool, quant: QuantRecipe, n: Int, m: Int, tp: Int,
    Profile: Bool, N: Int, //,
    MR: Int = 4, max_worker_count: Int = 128,
](
    act: ButterquantBlockActivation[tp],
    weight: ButterquantWeight[quant, n, m, tp],
    output: Binding[BFloat16, tp],
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime assert quant_vnni_packed[quant](), "bq block linear consumes a VNNI-packed weight"
    comptime assert quant_colsum_per_block[quant](), "bq block linear requires a per-block colsum sidecar"
    if seq_len <= 0:
        return
    comptime num_tiles = n // VNNI_N_STEP
    comptime Kern = BqBlockLinearKernel[n, m, quant_k_block[quant](), MR]
    var cs = weight.colsum_checked()

    @parameter
    def make(r: Int) -> Kern:
        return Kern(act.data[r], act.scale[r], weight.data[r], weight.scale[r],
                    cs[r], output[r], seq_len, 0, 0)

    fanout_dispatch[tp, make, max_worker_count=max_worker_count, label="bq_block_linear"](
        pools, prof, num_tiles, seq_len * m + n * m,
        inline_threshold_bytes=GEMV_INLINE_ROWS * m)


def dispatch_bq_qkv[
    P: BurstThreadPool, quant: QuantRecipe, qn: Int, kvn: Int, m: Int, tp: Int,
    Profile: Bool, N: Int, //,
    MR: Int = 4, max_worker_count: Int = 128,
](
    act: ButterquantActivation[tp],
    q_weight: ButterquantWeight[quant, qn, m, tp],
    k_weight: ButterquantWeight[quant, kvn, m, tp],
    v_weight: ButterquantWeight[quant, kvn, m, tp],
    q_out: Binding[BFloat16, tp],
    k_out: Binding[BFloat16, tp],
    v_out: Binding[BFloat16, tp],
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime assert quant_vnni_packed[quant](), "bq qkv consumes VNNI-packed weights"
    comptime assert quant_has_colsum[quant](), "bq qkv requires colsum sidecars"
    if seq_len <= 0:
        return
    comptime q_tiles = qn // VNNI_N_STEP
    comptime kv_tiles = kvn // VNNI_N_STEP
    comptime total_tiles = q_tiles + kv_tiles + kv_tiles
    comptime QKern = BqLinearKernel[qn, m, MR, q_tiles, total_tiles]
    comptime KKern = BqLinearKernel[kvn, m, MR, kv_tiles, total_tiles]
    comptime VKern = BqLinearKernel[kvn, m, MR, kv_tiles, total_tiles]
    comptime QK = Chain[QKern, KKern]
    comptime QKV = Chain[QK, VKern]

    var qcs = q_weight.colsum_checked()
    var kcs = k_weight.colsum_checked()
    var vcs = v_weight.colsum_checked()

    @parameter
    def make(r: Int) -> QKV:
        return QKV(
            QK(
                QKern(act.data[r], act.scale[r], q_weight.data[r],
                      q_weight.scale[r], qcs[r], q_out[r], seq_len, 0, 0),
                KKern(act.data[r], act.scale[r], k_weight.data[r],
                      k_weight.scale[r], kcs[r], k_out[r], seq_len, 0, 0),
            ),
            VKern(act.data[r], act.scale[r], v_weight.data[r],
                  v_weight.scale[r], vcs[r], v_out[r], seq_len, 0, 0),
        )

    fanout_dispatch[
        tp, make,
        max_worker_count=max_worker_count,
        worker_policy=saturate_workers,
        label="bq_qkv",
    ](pools, prof, total_tiles, seq_len * m + (qn + kvn + kvn) * m)
