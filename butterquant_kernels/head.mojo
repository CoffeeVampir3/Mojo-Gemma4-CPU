from threading.threading_traits import BurstThreadPool
from kernels.helpers import (
    RangePartitionedKernel, Binding, fanout_dispatch, BF16Ptr,
)
from kernels.dispatch_heuristics import GEMV_INLINE_ROWS
from kernels.gemv import softcap_value

from butterquant.convert import store_bf16
from butterquant.dot_products import head_logit_row
from butterquant.runtime import prepare_head_activation
from butterquant.types import F32Ptr, I8Ptr
from butterquant.weight import (
    ButterquantWeight, ButterquantActivation, quant_k_block, quant_per_block,
)
from quant.recipe import QuantRecipe


def dispatch_bq_head_prep[
    tp: Int, //,
    hidden: Int, block: Int, sqrt_n: Float32, n_eps: Float32,
](
    src: Binding[BFloat16, tp],
    gamma: Binding[BFloat16, tp],
    dst: ButterquantActivation[tp],
):
    for r in range(tp):
        prepare_head_activation[hidden, block, sqrt_n, n_eps](
            src[r], gamma[r], dst.data[r], dst.scale[r])


@fieldwise_init
struct BqHeadGemvKernel[
    rows: Int, cols: Int, block: Int, cap: Float64,
](RangePartitionedKernel):
    var x_i8: I8Ptr
    var sa: F32Ptr
    var weight: I8Ptr
    var scales: F32Ptr
    var output: BF16Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        comptime nb = Self.cols // Self.block
        for row in range(self.start, self.end):
            var raw = head_logit_row[Self.block](
                self.x_i8, self.sa,
                self.weight + row * Self.cols,
                self.scales + row * nb,
                Self.cols)
            var capped = softcap_value[Self.cap](SIMD[DType.float32, 1](raw))
            store_bf16[1](capped, self.output + row)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_bq_head_gemv[
    P: BurstThreadPool, quant: QuantRecipe, n: Int, m: Int, tp: Int, //,
    cap: Float64,
    max_worker_count: Int = 128,
](
    act: ButterquantActivation[tp],
    weight: ButterquantWeight[quant, n, m, tp],
    output: Binding[BFloat16, tp],
    mut pools: List[P],
):
    comptime assert quant_per_block[quant](), "head GEMV consumes a per-block weight scale"
    comptime K = BqHeadGemvKernel[n, m, quant_k_block[quant](), cap]

    @parameter
    def make(r: Int) -> K:
        return K(act.data[r], act.scale[r], weight.data[r], weight.scale[r],
                 output[r], 0, 0)

    fanout_dispatch[tp, make, max_worker_count=max_worker_count](
        pools, n, n * m,
        inline_threshold_bytes=GEMV_INLINE_ROWS * m)
