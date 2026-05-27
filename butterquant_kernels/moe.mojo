from std.collections import InlineArray
from std.sys.info import simd_width_of

from threading.threading_traits import BurstThreadPool
from simd_math.ops import gelu_tanh_f32
from kernels.helpers import (
    RangePartitionedKernel, Binding, fanout_dispatch, saturate_workers,
    BF16Ptr, I32Ptr,
)
from kernels.moe_router import SparseRoute, SparseRoutePtr

from butterquant.convert import store_bf16
from butterquant.gemm import accumulate_n_step, accumulate_n_step_gathered
from butterquant.vnni import VNNI_N_STEP
from butterquant.types import F32Ptr, I8Ptr
from butterquant.weight import (
    ButterquantWeight, ButterquantActivation, ButterquantBlockActivation,
    quant_vnni_packed, quant_has_colsum, quant_colsum_per_block, quant_k_block,
)
from quant.recipe import QuantRecipe


@always_inline
def emit_bq_gate_up_panel[
    hidden: Int, gate_up: Int, inter: Int, n_inter_tiles: Int, PR: Int,
](
    it: Int,
    rec_start: Int,
    x_i8: I8Ptr,
    x_sa: F32Ptr,
    routes: SparseRoutePtr,
    w: I8Ptr,
    wsc: F32Ptr,
    cs: F32Ptr,
    bucket: BF16Ptr,
):
    comptime width = simd_width_of[DType.int32]()
    comptime acc_count = VNNI_N_STEP // width
    comptime inv127 = Float32(1.0) / Float32(127.0)
    var gate_tile = it
    var up_tile = n_inter_tiles + it

    var rows = InlineArray[I8Ptr, PR](uninitialized=True)
    var scales = InlineArray[Float32, PR](uninitialized=True)
    comptime for r in range(PR):
        var tok = Int(routes[rec_start + r].token)
        rows[r] = x_i8 + tok * hidden
        scales[r] = x_sa[tok]

    var gacc = InlineArray[SIMD[DType.int32, width], PR * acc_count](
        fill=SIMD[DType.int32, width](0))
    var uacc = InlineArray[SIMD[DType.int32, width], PR * acc_count](
        fill=SIMD[DType.int32, width](0))
    accumulate_n_step_gathered[width, PR](
        rows, w, gate_tile * VNNI_N_STEP * hidden, 0, hidden, gacc)
    accumulate_n_step_gathered[width, PR](
        rows, w, up_tile * VNNI_N_STEP * hidden, 0, hidden, uacc)

    comptime for r in range(PR):
        var ad = scales[r] * inv127
        var bucket_row = bucket + (rec_start + r) * inter + it * VNNI_N_STEP
        comptime for a in range(acc_count):
            var ng = gate_tile * VNNI_N_STEP + a * width
            var nu = up_tile * VNNI_N_STEP + a * width
            var gcs = (cs + ng).load[width=width]()
            var ucs = (cs + nu).load[width=width]()
            var gv = (gacc[r * acc_count + a].cast[DType.float32]()
                - Float32(128) * gcs) * ad * (wsc + ng).load[width=width]()
            var uv = (uacc[r * acc_count + a].cast[DType.float32]()
                - Float32(128) * ucs) * ad * (wsc + nu).load[width=width]()
            var res = gelu_tanh_f32[width](gv) * uv
            store_bf16[width](res, bucket_row + a * width)


@fieldwise_init
struct BqPhase1GateUpKernel[
    hidden: Int, gate_up: Int, inter: Int, experts_per_rank: Int, MR: Int = 4,
](RangePartitionedKernel):
    var x_i8: I8Ptr
    var x_sa: F32Ptr
    var expert_offset: I32Ptr
    var routes: SparseRoutePtr
    var experts_gate_up: I8Ptr
    var gate_up_scale: F32Ptr
    var gate_up_colsum: F32Ptr
    var hidden_bucket: BF16Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        comptime n_inter_tiles = Self.inter // VNNI_N_STEP
        for unit in range(self.start, self.end):
            var expert = unit // n_inter_tiles
            var it = unit % n_inter_tiles
            var rec_lo = Int(self.expert_offset[expert])
            var rec_hi = Int(self.expert_offset[expert + 1])
            var n_tok = rec_hi - rec_lo
            if n_tok <= 0:
                continue
            var w = self.experts_gate_up + expert * Self.gate_up * Self.hidden
            var wsc = self.gate_up_scale + expert * Self.gate_up
            var cs = self.gate_up_colsum + expert * Self.gate_up
            var rb = 0
            while rb + Self.MR <= n_tok:
                emit_bq_gate_up_panel[
                    Self.hidden, Self.gate_up, Self.inter, n_inter_tiles, Self.MR,
                ](it, rec_lo + rb, self.x_i8, self.x_sa, self.routes,
                  w, wsc, cs, self.hidden_bucket)
                rb += Self.MR
            while rb < n_tok:
                emit_bq_gate_up_panel[
                    Self.hidden, Self.gate_up, Self.inter, n_inter_tiles, 1,
                ](it, rec_lo + rb, self.x_i8, self.x_sa, self.routes,
                  w, wsc, cs, self.hidden_bucket)
                rb += 1

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_bq_phase1_gate_up[
    P: BurstThreadPool, quant: QuantRecipe, n: Int, m: Int, tp: Int, //,
    hidden: Int, gate_up: Int, inter: Int, experts_per_rank: Int,
    MR: Int = 4, max_worker_count: Int = 128,
](
    act: ButterquantActivation[tp],
    expert_offset: Binding[Int32, tp],
    routes: Binding[SparseRoute, tp],
    experts_gate_up: ButterquantWeight[quant, n, m, tp],
    hidden_bucket: Binding[BFloat16, tp],
    mut pools: List[P],
):
    comptime assert quant_vnni_packed[quant](), "bq phase1 consumes VNNI-packed experts"
    comptime assert quant_has_colsum[quant](), "bq phase1 requires a colsum sidecar"
    comptime assert n == experts_per_rank * gate_up, "experts_gate_up DATA_N mismatch"
    comptime assert m == hidden, "experts_gate_up DATA_M mismatch"
    comptime n_inter_tiles = inter // VNNI_N_STEP
    comptime total_units = experts_per_rank * n_inter_tiles
    comptime Kern = BqPhase1GateUpKernel[hidden, gate_up, inter, experts_per_rank, MR]
    var cs = experts_gate_up.colsum_checked()

    @parameter
    def make(r: Int) -> Kern:
        return Kern(act.data[r], act.scale[r], expert_offset[r], routes[r],
                    experts_gate_up.data[r], experts_gate_up.scale[r], cs[r],
                    hidden_bucket[r], 0, 0)

    fanout_dispatch[
        tp, make,
        max_worker_count=max_worker_count,
        worker_policy=saturate_workers,
    ](pools, total_units, total_units * hidden * 2)


@always_inline
def emit_bq_down_panel[
    hidden: Int, inter: Int, block: Int, data_n: Int, PR: Int,
](
    rec_start: Int,
    routes: SparseRoutePtr,
    bucket_i8: I8Ptr,
    bucket_sa: F32Ptr,
    w: I8Ptr,
    wsc: F32Ptr,
    cs: F32Ptr,
    e_row_base: Int,
    moe_accum: F32Ptr,
    tile_start: Int,
    tile_end: Int,
):
    comptime width = simd_width_of[DType.int32]()
    comptime acc_count = VNNI_N_STEP // width
    comptime nb = inter // block
    comptime blk_bytes = block * VNNI_N_STEP
    comptime inv127 = Float32(1.0) / Float32(127.0)

    var act = bucket_i8 + rec_start * inter
    var dsts = InlineArray[F32Ptr, PR](uninitialized=True)
    var weights = InlineArray[Float32, PR](uninitialized=True)
    comptime for r in range(PR):
        var rec = routes[rec_start + r]
        dsts[r] = moe_accum + Int(rec.token) * hidden
        weights[r] = rec.weight

    for t in range(tile_start, tile_end):
        var facc = InlineArray[SIMD[DType.float32, width], PR * acc_count](
            fill=SIMD[DType.float32, width](0))
        for b in range(nb):
            var iacc = InlineArray[SIMD[DType.int32, width], PR * acc_count](
                fill=SIMD[DType.int32, width](0))
            accumulate_n_step[width, PR, inter](
                act, 0, w, t * VNNI_N_STEP * inter + b * blk_bytes,
                b * block, block, iacc)
            comptime for r in range(PR):
                var adv = SIMD[DType.float32, width](
                    bucket_sa[(rec_start + r) * nb + b] * inv127)
                comptime for a in range(acc_count):
                    var n = t * VNNI_N_STEP + a * width
                    var ccs = (cs + b * data_n + e_row_base + n).load[width=width]()
                    var corrected = (
                        iacc[r * acc_count + a].cast[DType.float32]()
                        - Float32(128) * ccs)
                    facc[r * acc_count + a] = corrected.fma(
                        adv, facc[r * acc_count + a])

        comptime for r in range(PR):
            comptime for a in range(acc_count):
                var n = t * VNNI_N_STEP + a * width
                var res = (
                    facc[r * acc_count + a] * (wsc + n).load[width=width]()
                    * SIMD[DType.float32, width](weights[r]))
                var prev = (dsts[r] + n).load[width=width]()
                (dsts[r] + n).store(prev + res)


@fieldwise_init
struct BqPhase2DownKernel[
    hidden: Int, inter: Int, block: Int, experts_per_rank: Int, MR: Int = 4,
](RangePartitionedKernel):
    var expert_offset: I32Ptr
    var routes: SparseRoutePtr
    var bucket_i8: I8Ptr
    var bucket_sa: F32Ptr
    var experts_down: I8Ptr
    var down_scale: F32Ptr
    var down_colsum: F32Ptr
    var moe_accum: F32Ptr
    var moe_partial: BF16Ptr
    var seq_len: Int
    var tile_start: Int
    var tile_end: Int

    def execute(mut self):
        comptime width = simd_width_of[DType.int32]()
        comptime data_n = Self.experts_per_rank * Self.hidden
        var col_lo = self.tile_start * VNNI_N_STEP
        var col_hi = self.tile_end * VNNI_N_STEP

        for tok in range(self.seq_len):
            var row = self.moe_accum + tok * Self.hidden
            var c = col_lo
            while c < col_hi:
                (row + c).store(SIMD[DType.float32, width](0))
                c += width

        for e in range(Self.experts_per_rank):
            var rec_lo = Int(self.expert_offset[e])
            var rec_hi = Int(self.expert_offset[e + 1])
            var n_tok = rec_hi - rec_lo
            if n_tok <= 0:
                continue
            var w = self.experts_down + e * Self.hidden * Self.inter
            var wsc = self.down_scale + e * Self.hidden
            var e_row_base = e * Self.hidden
            var rb = 0
            while rb + Self.MR <= n_tok:
                emit_bq_down_panel[
                    Self.hidden, Self.inter, Self.block, data_n, Self.MR,
                ](rec_lo + rb, self.routes, self.bucket_i8, self.bucket_sa,
                  w, wsc, self.down_colsum, e_row_base, self.moe_accum,
                  self.tile_start, self.tile_end)
                rb += Self.MR
            while rb < n_tok:
                emit_bq_down_panel[
                    Self.hidden, Self.inter, Self.block, data_n, 1,
                ](rec_lo + rb, self.routes, self.bucket_i8, self.bucket_sa,
                  w, wsc, self.down_colsum, e_row_base, self.moe_accum,
                  self.tile_start, self.tile_end)
                rb += 1

        for tok in range(self.seq_len):
            var arow = self.moe_accum + tok * Self.hidden
            var drow = self.moe_partial + tok * Self.hidden
            var c = col_lo
            while c < col_hi:
                store_bf16[width]((arow + c).load[width=width](), drow + c)
                c += width

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.tile_start = start
        self.tile_end = end


def dispatch_bq_phase2_down[
    P: BurstThreadPool, quant: QuantRecipe, n: Int, m: Int, tp: Int, //,
    hidden: Int, inter: Int, experts_per_rank: Int,
    MR: Int = 4, max_worker_count: Int = 128,
](
    bucket: ButterquantBlockActivation[tp],
    expert_offset: Binding[Int32, tp],
    routes: Binding[SparseRoute, tp],
    experts_down: ButterquantWeight[quant, n, m, tp],
    moe_accum: Binding[Float32, tp],
    moe_partial: Binding[BFloat16, tp],
    seq_len: Int,
    mut pools: List[P],
):
    comptime assert quant_vnni_packed[quant](), "bq phase2 consumes VNNI-packed experts"
    comptime assert quant_colsum_per_block[quant](), "bq phase2 requires a per-block colsum"
    comptime assert n == experts_per_rank * hidden, "experts_down DATA_N mismatch"
    comptime assert m == inter, "experts_down DATA_M mismatch"
    if seq_len <= 0:
        return
    comptime n_tiles = hidden // VNNI_N_STEP
    comptime Kern = BqPhase2DownKernel[
        hidden, inter, quant_k_block[quant](), experts_per_rank, MR]
    var cs = experts_down.colsum_checked()

    @parameter
    def make(r: Int) -> Kern:
        return Kern(expert_offset[r], routes[r], bucket.data[r], bucket.scale[r],
                    experts_down.data[r], experts_down.scale[r], cs[r],
                    moe_accum[r], moe_partial[r], seq_len, 0, 0)

    fanout_dispatch[
        tp, make,
        max_worker_count=max_worker_count,
        worker_policy=saturate_workers,
    ](pools, n_tiles, seq_len * hidden * 2 + n * m)
