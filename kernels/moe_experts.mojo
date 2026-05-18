from std.collections import InlineArray

from notstdcollections import HeapMoveArray
from threading.threading_traits import BurstThreadPool
from simd_math.ops import gelu_tanh_f32
from .helpers import (
    OutputPartitionedKernel, DispatchBuffer, Binding,
    BF16Ptr, F32Ptr, I32Ptr, W, BW,
    tile_dispatch, join_all,
)
from .dpbf16 import bf16_pair_dot
from .moe_router import SparseRoute, SparseRoutePtr


@fieldwise_init
struct Phase1GateUpKernel[
    hidden: Int, gate_up_fused: Int, intermediate: Int, experts_per_rank: Int,
](OutputPartitionedKernel):
    comptime TILE_J = 64
    comptime MR = 4
    comptime PU_GU = 4
    comptime WORKER_SCRATCH_ELEMS = Self.MR * 2 * Self.TILE_J

    var x_normed: BF16Ptr
    var expert_offset: I32Ptr
    var routes: SparseRoutePtr
    var experts_gate_up: BF16Ptr
    var gate_scratch: F32Ptr
    var hidden_bucket: BF16Ptr
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime PU_GU = Self.PU_GU
        comptime STRIDE_GU = PU_GU * BW
        comptime MR = Self.MR
        comptime tile_j = Self.TILE_J

        comptime assert Self.intermediate % tile_j == 0, (
            "Phase1: intermediate must be divisible by tile_j")
        comptime assert tile_j % W == 0, (
            "Phase1: tile_j must be a multiple of f32 SIMD width")
        comptime assert Self.hidden % STRIDE_GU == 0, (
            "Phase1: hidden must be divisible by STRIDE_GU")
        comptime assert Self.gate_up_fused == 2 * Self.intermediate, (
            "Phase1: gate_up_fused must be 2 * intermediate")
        comptime n_tiles = Self.intermediate // tile_j
        comptime total_tiles = Self.experts_per_rank * n_tiles
        comptime worker_part = MR * 2 * tile_j

        debug_assert(
            self.start >= 0 and self.start <= self.end and self.end <= total_tiles,
            "Phase1: tile range out of bounds",
        )

        var worker_base = self.gate_scratch + self.worker_id * worker_part
        var gate_part = worker_base
        var up_part = worker_base + MR * tile_j

        for tile_idx in range(self.start, self.end):
            var expert = tile_idx // n_tiles
            var t_in_expert = tile_idx % n_tiles
            var j_lo = t_in_expert * tile_j

            var rec_lo = Int(self.expert_offset[expert])
            var rec_hi = Int(self.expert_offset[expert + 1])
            var n_tok = rec_hi - rec_lo
            if n_tok <= 0:
                continue

            var gu_w = self.experts_gate_up + expert * Self.gate_up_fused * Self.hidden
            var gate_w_base = gu_w + j_lo * Self.hidden
            var up_w_base = gu_w + (Self.intermediate + j_lo) * Self.hidden

            var rec_block = 0
            while rec_block + MR <= n_tok:
                var x_rows = InlineArray[BF16Ptr, MR](uninitialized=True)
                comptime for r in range(MR):
                    x_rows[r] = (
                        self.x_normed
                        + Int(self.routes[rec_lo + rec_block + r].token)
                            * Self.hidden)

                for j_off in range(tile_j):
                    var w_row = gate_w_base + j_off * Self.hidden
                    var accs = InlineArray[
                        InlineArray[SIMD[DType.float32, W], PU_GU], MR,
                    ](uninitialized=True)
                    comptime for r in range(MR):
                        comptime for p in range(PU_GU):
                            accs[r][p] = SIMD[DType.float32, W](0)
                    for i in range(Self.hidden // STRIDE_GU):
                        comptime for p in range(PU_GU):
                            var off = i * STRIDE_GU + p * BW
                            var w_v = (w_row + off).load[width=BW]()
                            comptime for r in range(MR):
                                var x_v = (x_rows[r] + off).load[width=BW]()
                                accs[r][p] = bf16_pair_dot(accs[r][p], x_v, w_v)
                    comptime for r in range(MR):
                        var s = SIMD[DType.float32, W](0)
                        comptime for p in range(PU_GU):
                            s += accs[r][p]
                        gate_part[r * tile_j + j_off] = s.reduce_add()

                for j_off in range(tile_j):
                    var w_row = up_w_base + j_off * Self.hidden
                    var accs = InlineArray[
                        InlineArray[SIMD[DType.float32, W], PU_GU], MR,
                    ](uninitialized=True)
                    comptime for r in range(MR):
                        comptime for p in range(PU_GU):
                            accs[r][p] = SIMD[DType.float32, W](0)
                    for i in range(Self.hidden // STRIDE_GU):
                        comptime for p in range(PU_GU):
                            var off = i * STRIDE_GU + p * BW
                            var w_v = (w_row + off).load[width=BW]()
                            comptime for r in range(MR):
                                var x_v = (x_rows[r] + off).load[width=BW]()
                                accs[r][p] = bf16_pair_dot(accs[r][p], x_v, w_v)
                    comptime for r in range(MR):
                        var s = SIMD[DType.float32, W](0)
                        comptime for p in range(PU_GU):
                            s += accs[r][p]
                        up_part[r * tile_j + j_off] = s.reduce_add()

                comptime for r in range(MR):
                    var bucket_row = (
                        self.hidden_bucket
                        + (rec_lo + rec_block + r) * Self.intermediate
                        + j_lo)
                    var src_g = gate_part + r * tile_j
                    var src_u = up_part + r * tile_j
                    for j_off in range(0, tile_j, W):
                        var g = (src_g + j_off).load[width=W]()
                        var u = (src_u + j_off).load[width=W]()
                        var v = gelu_tanh_f32[W](g) * u
                        (bucket_row + j_off).store(v.cast[DType.bfloat16]())

                rec_block += MR

            while rec_block < n_tok:
                var tok = Int(self.routes[rec_lo + rec_block].token)
                var x_row = self.x_normed + tok * Self.hidden

                for j_off in range(tile_j):
                    var w_row = gate_w_base + j_off * Self.hidden
                    var accs = InlineArray[
                        SIMD[DType.float32, W], PU_GU,
                    ](fill=SIMD[DType.float32, W](0))
                    for i in range(Self.hidden // STRIDE_GU):
                        comptime for p in range(PU_GU):
                            var off = i * STRIDE_GU + p * BW
                            var w_v = (w_row + off).load[width=BW]()
                            var x_v = (x_row + off).load[width=BW]()
                            accs[p] = bf16_pair_dot(accs[p], x_v, w_v)
                    var s = SIMD[DType.float32, W](0)
                    comptime for p in range(PU_GU):
                        s += accs[p]
                    gate_part[j_off] = s.reduce_add()

                for j_off in range(tile_j):
                    var w_row = up_w_base + j_off * Self.hidden
                    var accs = InlineArray[
                        SIMD[DType.float32, W], PU_GU,
                    ](fill=SIMD[DType.float32, W](0))
                    for i in range(Self.hidden // STRIDE_GU):
                        comptime for p in range(PU_GU):
                            var off = i * STRIDE_GU + p * BW
                            var w_v = (w_row + off).load[width=BW]()
                            var x_v = (x_row + off).load[width=BW]()
                            accs[p] = bf16_pair_dot(accs[p], x_v, w_v)
                    var s = SIMD[DType.float32, W](0)
                    comptime for p in range(PU_GU):
                        s += accs[p]
                    up_part[j_off] = s.reduce_add()

                var bucket_row = (
                    self.hidden_bucket
                    + (rec_lo + rec_block) * Self.intermediate
                    + j_lo)
                for j_off in range(0, tile_j, W):
                    var g = (gate_part + j_off).load[width=W]()
                    var u = (up_part + j_off).load[width=W]()
                    var v = gelu_tanh_f32[W](g) * u
                    (bucket_row + j_off).store(v.cast[DType.bfloat16]())

                rec_block += 1

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.worker_id = worker_id
        self.start = start
        self.end = end


def dispatch_phase1_gate_up[
    P: BurstThreadPool, //,
    hidden: Int, gate_up_fused: Int, intermediate: Int,
    experts_per_rank: Int, tp: Int, max_worker_count: Int = 128,
](
    x_normed: Binding[BFloat16, tp],
    expert_offset: Binding[Int32, tp],
    routes: Binding[SparseRoute, tp],
    experts_gate_up: Binding[BFloat16, tp],
    gate_scratch: Binding[Float32, tp],
    hidden_bucket: Binding[BFloat16, tp],
    mut pools: HeapMoveArray[P],
):
    comptime Kernel = Phase1GateUpKernel[
        hidden, gate_up_fused, intermediate, experts_per_rank,
    ]
    comptime n_tiles = intermediate // Kernel.TILE_J
    comptime total_units = experts_per_rank * n_tiles

    var buf = DispatchBuffer[Kernel, max_worker_count]()
    for r in range(tp):
        var cap = min(max_worker_count, pools[r].get_capacity())
        var nw = min(cap, total_units)
        tile_dispatch(buf,
            Kernel(
                x_normed[r], expert_offset[r], routes[r],
                experts_gate_up[r], gate_scratch[r], hidden_bucket[r],
                0, 0, 0,
            ),
            pools[r], total_units, num_workers=nw)
    join_all[tp](pools)


@fieldwise_init
struct Phase2DownKernel[
    hidden: Int, intermediate: Int, experts_per_rank: Int,
](OutputPartitionedKernel):
    comptime TOK_TILE = 64
    comptime MR = 4
    comptime PU_DN = 2

    var expert_offset: I32Ptr
    var routes: SparseRoutePtr
    var hidden_bucket: BF16Ptr
    var experts_down: BF16Ptr
    var moe_accum: F32Ptr
    var moe_partial: BF16Ptr
    var seq_len: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime PU_DN = Self.PU_DN
        comptime STRIDE_DN = PU_DN * BW
        comptime MR = Self.MR
        comptime assert Self.intermediate % STRIDE_DN == 0, (
            "Phase2: intermediate must divide STRIDE_DN")

        for tok in range(self.seq_len):
            var acc_row = self.moe_accum + tok * Self.hidden
            var m = self.start
            while m + W <= self.end:
                (acc_row + m).store(SIMD[DType.float32, W](0))
                m += W
            while m < self.end:
                (acc_row + m)[] = Float32(0)
                m += 1

        for e in range(Self.experts_per_rank):
            var rec_lo = Int(self.expert_offset[e])
            var rec_hi = Int(self.expert_offset[e + 1])
            var n_tok_total = rec_hi - rec_lo
            if n_tok_total <= 0:
                continue
            var down_w = self.experts_down + e * Self.hidden * Self.intermediate

            var tok_base = 0
            while tok_base < n_tok_total:
                var n_tok = min(Self.TOK_TILE, n_tok_total - tok_base)
                var route_base = self.routes + rec_lo + tok_base
                var bucket_base = (
                    self.hidden_bucket + (rec_lo + tok_base) * Self.intermediate)

                var rec_block = 0
                while rec_block + MR <= n_tok:
                    var hm_rows = InlineArray[BF16Ptr, MR](uninitialized=True)
                    var dst_rows = InlineArray[F32Ptr, MR](uninitialized=True)
                    var weights = InlineArray[Float32, MR](uninitialized=True)
                    comptime for r in range(MR):
                        hm_rows[r] = bucket_base + (rec_block + r) * Self.intermediate
                        var rec = route_base[rec_block + r]
                        dst_rows[r] = self.moe_accum + Int(rec.token) * Self.hidden
                        weights[r] = rec.weight

                    for m in range(self.start, self.end):
                        var w_row = down_w + m * Self.intermediate
                        var accs = InlineArray[
                            InlineArray[SIMD[DType.float32, W], PU_DN], MR,
                        ](uninitialized=True)
                        comptime for r in range(MR):
                            comptime for p in range(PU_DN):
                                accs[r][p] = SIMD[DType.float32, W](0)
                        for i in range(Self.intermediate // STRIDE_DN):
                            comptime for p in range(PU_DN):
                                var off = i * STRIDE_DN + p * BW
                                var w_v = (w_row + off).load[width=BW]()
                                comptime for r in range(MR):
                                    var x_v = (hm_rows[r] + off).load[width=BW]()
                                    accs[r][p] = bf16_pair_dot(accs[r][p], x_v, w_v)
                        comptime for r in range(MR):
                            var s = SIMD[DType.float32, W](0)
                            comptime for p in range(PU_DN):
                                s += accs[r][p]
                            var out = s.reduce_add()
                            (dst_rows[r] + m)[] = (
                                (dst_rows[r] + m)[] + out * weights[r])
                    rec_block += MR

                while rec_block < n_tok:
                    var hm_row = bucket_base + rec_block * Self.intermediate
                    var rec = route_base[rec_block]
                    var dst_row = self.moe_accum + Int(rec.token) * Self.hidden
                    var weight = rec.weight
                    for m in range(self.start, self.end):
                        var w_row = down_w + m * Self.intermediate
                        var accs = InlineArray[
                            SIMD[DType.float32, W], PU_DN,
                        ](fill=SIMD[DType.float32, W](0))
                        for i in range(Self.intermediate // STRIDE_DN):
                            comptime for p in range(PU_DN):
                                var off = i * STRIDE_DN + p * BW
                                var w_v = (w_row + off).load[width=BW]()
                                var x_v = (hm_row + off).load[width=BW]()
                                accs[p] = bf16_pair_dot(accs[p], x_v, w_v)
                        var s = SIMD[DType.float32, W](0)
                        comptime for p in range(PU_DN):
                            s += accs[p]
                        var out = s.reduce_add()
                        (dst_row + m)[] = (dst_row + m)[] + out * weight
                    rec_block += 1

                tok_base += n_tok

        for tok in range(self.seq_len):
            var acc_row = self.moe_accum + tok * Self.hidden
            var dst_row = self.moe_partial + tok * Self.hidden
            var m = self.start
            while m + W <= self.end:
                var v = (acc_row + m).load[width=W]()
                (dst_row + m).store(v.cast[DType.bfloat16]())
                m += W
            while m < self.end:
                (dst_row + m)[] = (acc_row + m)[].cast[DType.bfloat16]()
                m += 1

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.start = start * W
        self.end = end * W


def dispatch_phase2_down[
    P: BurstThreadPool, //,
    hidden: Int, intermediate: Int, experts_per_rank: Int, tp: Int,
    max_worker_count: Int = 128,
](
    expert_offset: Binding[Int32, tp],
    routes: Binding[SparseRoute, tp],
    hidden_bucket: Binding[BFloat16, tp],
    experts_down: Binding[BFloat16, tp],
    moe_accum: Binding[Float32, tp],
    moe_partial: Binding[BFloat16, tp],
    seq_len: Int,
    mut pools: HeapMoveArray[P],
):
    comptime Kernel = Phase2DownKernel[hidden, intermediate, experts_per_rank]
    comptime hidden_strides = hidden // W

    var buf = DispatchBuffer[Kernel, max_worker_count]()
    for r in range(tp):
        var cap = min(max_worker_count, pools[r].get_capacity())
        var nw = min(cap, hidden_strides)
        tile_dispatch(buf,
            Kernel(
                expert_offset[r], routes[r], hidden_bucket[r],
                experts_down[r], moe_accum[r], moe_partial[r],
                seq_len, 0, 0,
            ),
            pools[r], hidden_strides, num_workers=nw)
    join_all[tp](pools)
