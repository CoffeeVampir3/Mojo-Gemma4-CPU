from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from simd_math import pick_port_unroll, tree_reduce_accs
from simd_math.ops import gelu_tanh_f32
from kernels.helpers import OutputPartitionedKernel
from prototypes.dpbf16 import bf16_pair_dot
from prototypes.moe_phase_kernels import SparseRoutePtr


comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr  = UnsafePointer[Scalar[DType.float32],  MutAnyOrigin]
comptime I32Ptr  = UnsafePointer[Scalar[DType.int32],    MutAnyOrigin]
comptime W = simd_width_of[DType.float32]()
comptime BW = simd_width_of[DType.bfloat16]()


comptime PHASE1_TILE_J = 64
comptime PHASE1_MR = 4
comptime PHASE1_PU_GU = 4
comptime PHASE2_TOK_TILE = 64
comptime PHASE2_MR = 4
comptime PHASE2_PU_DN = 2


@fieldwise_init
struct Phase1GateUpKernel[
    hidden: Int, gate_up_fused: Int, intermediate: Int, experts_per_rank: Int,
](OutputPartitionedKernel):
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
        comptime PU_GU = PHASE1_PU_GU
        comptime STRIDE_GU = PU_GU * BW
        comptime MR = PHASE1_MR
        comptime tile_j = PHASE1_TILE_J

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

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(
            self.x_normed, self.expert_offset, self.routes,
            self.experts_gate_up, self.gate_scratch, self.hidden_bucket,
            self.worker_id, start, end,
        )


@fieldwise_init
struct Phase2DownKernel[
    hidden: Int, intermediate: Int, experts_per_rank: Int,
](OutputPartitionedKernel):
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
        comptime PU_DN = PHASE2_PU_DN
        comptime STRIDE_DN = PU_DN * BW
        comptime MR = PHASE2_MR
        comptime assert Self.intermediate % STRIDE_DN == 0, (
            "Phase2: intermediate must be divisible by STRIDE_DN")

        # Zero this worker's hidden stripe across all tokens.
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
                var n_tok = min(PHASE2_TOK_TILE, n_tok_total - tok_base)
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

        # Final: cast our hidden stripe to bf16 in moe_partial.
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

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(
            self.expert_offset, self.routes, self.hidden_bucket,
            self.experts_down, self.moe_accum, self.moe_partial,
            self.seq_len, start, end,
        )


@fieldwise_init
struct ExpertRunnerSlotKernel[
    hidden: Int, gate_up_fused: Int, intermediate: Int, num_local_experts: Int,
](OutputPartitionedKernel):
    var x_normed: BF16Ptr
    var expert_offset: I32Ptr
    var bucket_token_idx: I32Ptr
    var bucket_weight: F32Ptr
    var experts_gate_up: BF16Ptr
    var experts_down: BF16Ptr
    var gate_scratch: F32Ptr
    var hidden_scratch: F32Ptr
    var moe_partial: BF16Ptr
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime PU_H = pick_port_unroll[W, Self.hidden]()
        comptime STRIDE_H = PU_H * W
        comptime PU_I = pick_port_unroll[W, Self.intermediate]()
        comptime STRIDE_I = PU_I * W

        var gu_scratch = self.gate_scratch + self.worker_id * Self.gate_up_fused
        var h_scratch = self.hidden_scratch + self.worker_id * Self.intermediate

        for e in range(self.start, self.end):
            var rec_lo = Int(self.expert_offset[e])
            var rec_hi = Int(self.expert_offset[e + 1])
            var gu_w = self.experts_gate_up + e * Self.gate_up_fused * Self.hidden
            var down_w = self.experts_down + e * Self.hidden * Self.intermediate

            for rec in range(rec_lo, rec_hi):
                var tok = Int(self.bucket_token_idx[rec])
                var x_row = self.x_normed + tok * Self.hidden

                for m in range(Self.gate_up_fused):
                    var row = gu_w + m * Self.hidden
                    var accs = InlineArray[SIMD[DType.float32, W], PU_H](
                        fill=SIMD[DType.float32, W](0))
                    for i in range(Self.hidden // STRIDE_H):
                        comptime for p in range(PU_H):
                            var off = i * STRIDE_H + p * W
                            var xv = (x_row + off).load[width=W]().cast[DType.float32]()
                            var wv = (row + off).load[width=W]().cast[DType.float32]()
                            accs[p] = xv.fma(wv, accs[p])
                    gu_scratch[m] = tree_reduce_accs(accs)

                for j in range(0, Self.intermediate, W):
                    var g = (gu_scratch + j).load[width=W]()
                    var u = (gu_scratch + Self.intermediate + j).load[width=W]()
                    (h_scratch + j).store(gelu_tanh_f32[W](g) * u)

                var route_w = SIMD[DType.float32, W](self.bucket_weight[rec])
                var dst = self.moe_partial + tok * Self.hidden
                for m in range(Self.hidden):
                    var row = down_w + m * Self.intermediate
                    var accs = InlineArray[SIMD[DType.float32, W], PU_I](
                        fill=SIMD[DType.float32, W](0))
                    for i in range(Self.intermediate // STRIDE_I):
                        comptime for p in range(PU_I):
                            var off = i * STRIDE_I + p * W
                            var xv = (h_scratch + off).load[width=W]()
                            var wv = (row + off).load[width=W]().cast[DType.float32]()
                            accs[p] = xv.fma(wv, accs[p])
                    var out = tree_reduce_accs(accs)
                    dst[m] = (dst[m].cast[DType.float32]() + out * route_w[0]).cast[DType.bfloat16]()

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(
            self.x_normed, self.expert_offset, self.bucket_token_idx,
            self.bucket_weight, self.experts_gate_up, self.experts_down,
            self.gate_scratch, self.hidden_scratch, self.moe_partial,
            self.worker_id, start, end,
        )


comptime BaselineExpertKernel[
    hidden: Int, gate_up_fused: Int, intermediate: Int, num_local_experts: Int,
] = ExpertRunnerSlotKernel[hidden, gate_up_fused, intermediate, num_local_experts]
