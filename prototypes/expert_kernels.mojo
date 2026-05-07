from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from simd_math import pick_port_unroll, tree_reduce_accs
from simd_math.ops import gelu_tanh_f32
from kernels.helpers import RangedKernel
from prototypes.dpbf16 import bf16_pair_dot, prefetch_l1


comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr  = UnsafePointer[Scalar[DType.float32],  MutAnyOrigin]
comptime I32Ptr  = UnsafePointer[Scalar[DType.int32],    MutAnyOrigin]
comptime W = simd_width_of[DType.float32]()


@fieldwise_init
struct ExpertRunnerSlotKernel[
    hidden: Int, gate_up_fused: Int, intermediate: Int, num_local_experts: Int,
](RangedKernel):
    var x_normed: BF16Ptr
    var expert_offset: I32Ptr
    var bucket_token_idx: I32Ptr
    var bucket_weight: F32Ptr
    var experts_gate_up: BF16Ptr
    var experts_down: BF16Ptr
    var gate_scratch: F32Ptr
    var hidden_scratch: F32Ptr
    var hidden_scratch_bf16: BF16Ptr
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
            self.gate_scratch, self.hidden_scratch, self.hidden_scratch_bf16,
            self.moe_partial,
            self.worker_id, start, end,
        )


comptime EXPERT_TOK_TILE = 64
comptime EXPERT_MR = 4
comptime EXPERT_PU = 4


@fieldwise_init
struct ExpertBatchedKernel[
    hidden: Int, gate_up_fused: Int, intermediate: Int, num_local_experts: Int,
](RangedKernel):
    var x_normed: BF16Ptr
    var expert_offset: I32Ptr
    var bucket_token_idx: I32Ptr
    var bucket_weight: F32Ptr
    var experts_gate_up: BF16Ptr
    var experts_down: BF16Ptr
    var gate_scratch: F32Ptr
    var hidden_scratch: F32Ptr
    var hidden_scratch_bf16: BF16Ptr
    var moe_partial: BF16Ptr
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime PU_H = pick_port_unroll[W, Self.hidden]()
        comptime STRIDE_H = PU_H * W
        comptime PU_I = pick_port_unroll[W, Self.intermediate]()
        comptime STRIDE_I = PU_I * W
        comptime MR = EXPERT_MR
        comptime tile_size = EXPERT_TOK_TILE
        comptime gu_per_tile = tile_size * Self.gate_up_fused
        comptime hm_per_tile = tile_size * Self.intermediate

        var gu_scratch = self.gate_scratch + self.worker_id * gu_per_tile
        var h_scratch = self.hidden_scratch + self.worker_id * hm_per_tile

        for e in range(self.start, self.end):
            var rec_lo = Int(self.expert_offset[e])
            var rec_hi = Int(self.expert_offset[e + 1])
            var n_tok_total = rec_hi - rec_lo
            if n_tok_total <= 0:
                continue
            var gu_w = self.experts_gate_up + e * Self.gate_up_fused * Self.hidden
            var down_w = self.experts_down + e * Self.hidden * Self.intermediate

            var tok_base = 0
            while tok_base < n_tok_total:
                var n_tok = min(tile_size, n_tok_total - tok_base)
                var bucket_base = self.bucket_token_idx + rec_lo + tok_base
                var weight_base = self.bucket_weight + rec_lo + tok_base

                # gate_up: weight-row outer with MR-way rec unroll and
                # PU-way K unroll. One W vector read feeds MR FMAs;
                # PU independent K-chains hide FMA latency.
                comptime PU_GU = EXPERT_PU
                comptime STRIDE_GU = PU_GU * W
                var rec_block = 0
                while rec_block + MR <= n_tok:
                    var x_rows = InlineArray[BF16Ptr, MR](uninitialized=True)
                    comptime for r in range(MR):
                        x_rows[r] = self.x_normed + Int(bucket_base[rec_block + r]) * Self.hidden

                    for m in range(Self.gate_up_fused):
                        var w_row = gu_w + m * Self.hidden
                        var accs = InlineArray[
                            InlineArray[SIMD[DType.float32, W], PU_GU], MR,
                        ](uninitialized=True)
                        comptime for r in range(MR):
                            comptime for p in range(PU_GU):
                                accs[r][p] = SIMD[DType.float32, W](0)
                        for i in range(Self.hidden // STRIDE_GU):
                            comptime for p in range(PU_GU):
                                var off = i * STRIDE_GU + p * W
                                var w_v = (w_row + off).load[width=W]().cast[DType.float32]()
                                comptime for r in range(MR):
                                    var x_v = (x_rows[r] + off).load[width=W]().cast[DType.float32]()
                                    accs[r][p] = x_v.fma(w_v, accs[r][p])
                        comptime for r in range(MR):
                            gu_scratch[(rec_block + r) * Self.gate_up_fused + m] = (
                                tree_reduce_accs(accs[r]))
                    rec_block += MR

                # Tail: any recs not divisible by MR fall back to PU unroll.
                while rec_block < n_tok:
                    var tok = Int(bucket_base[rec_block])
                    var x_row = self.x_normed + tok * Self.hidden
                    for m in range(Self.gate_up_fused):
                        var w_row = gu_w + m * Self.hidden
                        var accs = InlineArray[SIMD[DType.float32, W], PU_H](
                            fill=SIMD[DType.float32, W](0))
                        for i in range(Self.hidden // STRIDE_H):
                            comptime for p in range(PU_H):
                                var off = i * STRIDE_H + p * W
                                var xv = (x_row + off).load[width=W]().cast[DType.float32]()
                                var wv = (w_row + off).load[width=W]().cast[DType.float32]()
                                accs[p] = xv.fma(wv, accs[p])
                        gu_scratch[rec_block * Self.gate_up_fused + m] = tree_reduce_accs(accs)
                    rec_block += 1

                # gelu(gate) * up — token-major.
                for rec in range(n_tok):
                    var gu_row = gu_scratch + rec * Self.gate_up_fused
                    var hm_row = h_scratch + rec * Self.intermediate
                    for j in range(0, Self.intermediate, W):
                        var g = (gu_row + j).load[width=W]()
                        var u = (gu_row + Self.intermediate + j).load[width=W]()
                        (hm_row + j).store(gelu_tanh_f32[W](g) * u)

                # down: weight-row outer with MR-way rec unroll and PU-way K unroll.
                # Slot-phased: each token has at most one writer per row.
                comptime PU_DN = EXPERT_PU
                comptime STRIDE_DN = PU_DN * W
                var dn_rec = 0
                while dn_rec + MR <= n_tok:
                    var hm_rows = InlineArray[F32Ptr, MR](uninitialized=True)
                    var dst_rows = InlineArray[BF16Ptr, MR](uninitialized=True)
                    var weights = InlineArray[Scalar[DType.float32], MR](
                        uninitialized=True)
                    comptime for r in range(MR):
                        hm_rows[r] = h_scratch + (dn_rec + r) * Self.intermediate
                        var tok = Int(bucket_base[dn_rec + r])
                        dst_rows[r] = self.moe_partial + tok * Self.hidden
                        weights[r] = weight_base[dn_rec + r]

                    for m in range(Self.hidden):
                        var w_row = down_w + m * Self.intermediate
                        var accs = InlineArray[
                            InlineArray[SIMD[DType.float32, W], PU_DN], MR,
                        ](uninitialized=True)
                        comptime for r in range(MR):
                            comptime for p in range(PU_DN):
                                accs[r][p] = SIMD[DType.float32, W](0)
                        for i in range(Self.intermediate // STRIDE_DN):
                            comptime for p in range(PU_DN):
                                var off = i * STRIDE_DN + p * W
                                var w_v = (w_row + off).load[width=W]().cast[DType.float32]()
                                comptime for r in range(MR):
                                    var x_v = (hm_rows[r] + off).load[width=W]()
                                    accs[r][p] = x_v.fma(w_v, accs[r][p])
                        comptime for r in range(MR):
                            var out = tree_reduce_accs(accs[r])
                            var dst = dst_rows[r] + m
                            dst[] = (dst[].cast[DType.float32]()
                                + out * weights[r]).cast[DType.bfloat16]()
                    dn_rec += MR

                # Tail recs not divisible by MR.
                while dn_rec < n_tok:
                    var hm_row = h_scratch + dn_rec * Self.intermediate
                    var tok = Int(bucket_base[dn_rec])
                    var dst_row = self.moe_partial + tok * Self.hidden
                    var weight = weight_base[dn_rec]
                    for m in range(Self.hidden):
                        var w_row = down_w + m * Self.intermediate
                        var accs = InlineArray[SIMD[DType.float32, W], PU_I](
                            fill=SIMD[DType.float32, W](0))
                        for i in range(Self.intermediate // STRIDE_I):
                            comptime for p in range(PU_I):
                                var off = i * STRIDE_I + p * W
                                var xv = (hm_row + off).load[width=W]()
                                var wv = (w_row + off).load[width=W]().cast[DType.float32]()
                                accs[p] = xv.fma(wv, accs[p])
                        var out = tree_reduce_accs(accs)
                        var dst = dst_row + m
                        dst[] = (dst[].cast[DType.float32]()
                            + out * weight).cast[DType.bfloat16]()
                    dn_rec += 1

                tok_base += n_tok

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(
            self.x_normed, self.expert_offset, self.bucket_token_idx,
            self.bucket_weight, self.experts_gate_up, self.experts_down,
            self.gate_scratch, self.hidden_scratch, self.hidden_scratch_bf16,
            self.moe_partial, self.worker_id, start, end,
        )


@fieldwise_init
struct ExpertBatchedDpKernel[
    hidden: Int, gate_up_fused: Int, intermediate: Int, num_local_experts: Int,
](RangedKernel):
    var x_normed: BF16Ptr
    var expert_offset: I32Ptr
    var bucket_token_idx: I32Ptr
    var bucket_weight: F32Ptr
    var experts_gate_up: BF16Ptr
    var experts_down: BF16Ptr
    var gate_scratch: F32Ptr
    var hidden_scratch: F32Ptr
    var hidden_scratch_bf16: BF16Ptr
    var moe_partial: BF16Ptr
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime DPB_LANES = 16
        comptime DPB_K = 32
        comptime PU_DP = 4
        comptime STRIDE_DP = PU_DP * DPB_K
        comptime PU_I = pick_port_unroll[W, Self.intermediate]()
        comptime STRIDE_I = PU_I * W
        comptime MR = EXPERT_MR
        comptime assert W == DPB_LANES, "VDPBF16PS path requires AVX-512 (W=16)"
        comptime tile_size = EXPERT_TOK_TILE
        comptime gu_per_tile = tile_size * Self.gate_up_fused
        comptime hm_per_tile = tile_size * Self.intermediate

        var gu_scratch = self.gate_scratch + self.worker_id * gu_per_tile
        var h_scratch = self.hidden_scratch + self.worker_id * hm_per_tile
        var h_bf16 = self.hidden_scratch_bf16 + self.worker_id * hm_per_tile

        for e in range(self.start, self.end):
            var rec_lo = Int(self.expert_offset[e])
            var rec_hi = Int(self.expert_offset[e + 1])
            var n_tok_total = rec_hi - rec_lo
            if n_tok_total <= 0:
                continue
            var gu_w = self.experts_gate_up + e * Self.gate_up_fused * Self.hidden
            var down_w = self.experts_down + e * Self.hidden * Self.intermediate

            var tok_base = 0
            while tok_base < n_tok_total:
                var n_tok = min(tile_size, n_tok_total - tok_base)
                var bucket_base = self.bucket_token_idx + rec_lo + tok_base
                var weight_base = self.bucket_weight + rec_lo + tok_base

                # gate_up via DPBF16PS: each call consumes 32 K elements.
                var rec_block = 0
                while rec_block + MR <= n_tok:
                    var x_rows = InlineArray[BF16Ptr, MR](uninitialized=True)
                    comptime for r in range(MR):
                        x_rows[r] = self.x_normed + Int(bucket_base[rec_block + r]) * Self.hidden

                    for m in range(Self.gate_up_fused):
                        var w_row = gu_w + m * Self.hidden
                        var accs = InlineArray[
                            InlineArray[SIMD[DType.float32, DPB_LANES], PU_DP], MR,
                        ](uninitialized=True)
                        comptime for r in range(MR):
                            comptime for p in range(PU_DP):
                                accs[r][p] = SIMD[DType.float32, DPB_LANES](0)
                        for i in range(Self.hidden // STRIDE_DP):
                            comptime for p in range(PU_DP):
                                var off = i * STRIDE_DP + p * DPB_K
                                var w_v = (w_row + off).load[width=DPB_K]()
                                comptime for r in range(MR):
                                    var x_v = (x_rows[r] + off).load[width=DPB_K]()
                                    accs[r][p] = bf16_pair_dot[4](accs[r][p], x_v, w_v)
                        comptime for r in range(MR):
                            var s = SIMD[DType.float32, DPB_LANES](0)
                            comptime for p in range(PU_DP):
                                s += accs[r][p]
                            gu_scratch[(rec_block + r) * Self.gate_up_fused + m] = (
                                s.reduce_add())
                    rec_block += MR

                # Tail: scalar fallback for n_tok % MR.
                while rec_block < n_tok:
                    var tok = Int(bucket_base[rec_block])
                    var x_row = self.x_normed + tok * Self.hidden
                    for m in range(Self.gate_up_fused):
                        var w_row = gu_w + m * Self.hidden
                        var accs = InlineArray[SIMD[DType.float32, DPB_LANES], PU_DP](
                            fill=SIMD[DType.float32, DPB_LANES](0))
                        for i in range(Self.hidden // STRIDE_DP):
                            comptime for p in range(PU_DP):
                                var off = i * STRIDE_DP + p * DPB_K
                                var w_v = (w_row + off).load[width=DPB_K]()
                                var x_v = (x_row + off).load[width=DPB_K]()
                                accs[p] = bf16_pair_dot[4](accs[p], x_v, w_v)
                        var s = SIMD[DType.float32, DPB_LANES](0)
                        comptime for p in range(PU_DP):
                            s += accs[p]
                        gu_scratch[rec_block * Self.gate_up_fused + m] = s.reduce_add()
                    rec_block += 1

                # gelu(gate) * up — f32 storage; down reads f32 with cast+FMA.
                # The DPBF16PS path for down has a known correctness bug
                # (3× fp mismatch) that is not in: bf16 GELU storage, MR unroll,
                # or PU unroll. Reverting to f32 path until root cause found.
                for rec in range(n_tok):
                    var gu_row = gu_scratch + rec * Self.gate_up_fused
                    var hm_row = h_scratch + rec * Self.intermediate
                    for j in range(0, Self.intermediate, W):
                        var g = (gu_row + j).load[width=W]()
                        var u = (gu_row + Self.intermediate + j).load[width=W]()
                        (hm_row + j).store(gelu_tanh_f32[W](g) * u)

                # down: cast+FMA from f32 h_scratch with MR=4 + PU=4 unroll.
                comptime PU_DN = EXPERT_PU
                comptime STRIDE_DN = PU_DN * W
                var dn_rec = 0
                while dn_rec + MR <= n_tok:
                    var hm_rows = InlineArray[F32Ptr, MR](uninitialized=True)
                    var dst_rows = InlineArray[BF16Ptr, MR](uninitialized=True)
                    var weights = InlineArray[Scalar[DType.float32], MR](
                        uninitialized=True)
                    comptime for r in range(MR):
                        hm_rows[r] = h_scratch + (dn_rec + r) * Self.intermediate
                        var tok = Int(bucket_base[dn_rec + r])
                        dst_rows[r] = self.moe_partial + tok * Self.hidden
                        weights[r] = weight_base[dn_rec + r]
                    for m in range(Self.hidden):
                        var w_row = down_w + m * Self.intermediate
                        var accs = InlineArray[
                            InlineArray[SIMD[DType.float32, W], PU_DN], MR,
                        ](uninitialized=True)
                        comptime for r in range(MR):
                            comptime for p in range(PU_DN):
                                accs[r][p] = SIMD[DType.float32, W](0)
                        for i in range(Self.intermediate // STRIDE_DN):
                            comptime for p in range(PU_DN):
                                var off = i * STRIDE_DN + p * W
                                var w_v = (w_row + off).load[width=W]().cast[DType.float32]()
                                comptime for r in range(MR):
                                    var x_v = (hm_rows[r] + off).load[width=W]()
                                    accs[r][p] = x_v.fma(w_v, accs[r][p])
                        comptime for r in range(MR):
                            var out = tree_reduce_accs(accs[r])
                            var dst = dst_rows[r] + m
                            dst[] = (dst[].cast[DType.float32]()
                                + out * weights[r]).cast[DType.bfloat16]()
                    dn_rec += MR

                while dn_rec < n_tok:
                    var hm_row = h_scratch + dn_rec * Self.intermediate
                    var tok = Int(bucket_base[dn_rec])
                    var dst_row = self.moe_partial + tok * Self.hidden
                    var weight = weight_base[dn_rec]
                    for m in range(Self.hidden):
                        var w_row = down_w + m * Self.intermediate
                        var accs = InlineArray[SIMD[DType.float32, W], PU_I](
                            fill=SIMD[DType.float32, W](0))
                        for i in range(Self.intermediate // STRIDE_I):
                            comptime for p in range(PU_I):
                                var off = i * STRIDE_I + p * W
                                var w_v = (w_row + off).load[width=W]().cast[DType.float32]()
                                var x_v = (hm_row + off).load[width=W]()
                                accs[p] = x_v.fma(w_v, accs[p])
                        var out = tree_reduce_accs(accs)
                        var dst = dst_row + m
                        dst[] = (dst[].cast[DType.float32]()
                            + out * weight).cast[DType.bfloat16]()
                    dn_rec += 1

                tok_base += n_tok

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(
            self.x_normed, self.expert_offset, self.bucket_token_idx,
            self.bucket_weight, self.experts_gate_up, self.experts_down,
            self.gate_scratch, self.hidden_scratch, self.hidden_scratch_bf16,
            self.moe_partial, self.worker_id, start, end,
        )


@fieldwise_init
struct ExpertBatchedDpDownKernel[
    hidden: Int, gate_up_fused: Int, intermediate: Int, num_local_experts: Int,
](RangedKernel):
    var x_normed: BF16Ptr
    var expert_offset: I32Ptr
    var bucket_token_idx: I32Ptr
    var bucket_weight: F32Ptr
    var experts_gate_up: BF16Ptr
    var experts_down: BF16Ptr
    var gate_scratch: F32Ptr
    var hidden_scratch: F32Ptr
    var hidden_scratch_bf16: BF16Ptr
    var moe_partial: BF16Ptr
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime DPB_LANES = 16
        comptime DPB_K = 32
        comptime PU_GU = 4
        comptime STRIDE_GU = PU_GU * DPB_K
        comptime PU_DN = 2
        comptime STRIDE_DN = PU_DN * DPB_K
        comptime MR = EXPERT_MR
        comptime assert W == DPB_LANES, "VDPBF16PS path requires AVX-512 (W=16)"
        comptime assert Self.hidden % STRIDE_GU == 0, "hidden must divide STRIDE_GU"
        comptime assert Self.intermediate % STRIDE_DN == 0, "intermediate must divide STRIDE_DN"
        comptime tile_size = EXPERT_TOK_TILE
        comptime gu_per_tile = tile_size * Self.gate_up_fused
        comptime hm_per_tile = tile_size * Self.intermediate

        var gu_scratch = self.gate_scratch + self.worker_id * gu_per_tile
        var h_bf16 = self.hidden_scratch_bf16 + self.worker_id * hm_per_tile

        for e in range(self.start, self.end):
            var rec_lo = Int(self.expert_offset[e])
            var rec_hi = Int(self.expert_offset[e + 1])
            var n_tok_total = rec_hi - rec_lo
            if n_tok_total <= 0:
                continue
            var gu_w = self.experts_gate_up + e * Self.gate_up_fused * Self.hidden
            var down_w = self.experts_down + e * Self.hidden * Self.intermediate

            var tok_base = 0
            while tok_base < n_tok_total:
                var n_tok = min(tile_size, n_tok_total - tok_base)
                var bucket_base = self.bucket_token_idx + rec_lo + tok_base
                var weight_base = self.bucket_weight + rec_lo + tok_base

                var rec_block = 0
                while rec_block + MR <= n_tok:
                    var x_rows = InlineArray[BF16Ptr, MR](uninitialized=True)
                    comptime for r in range(MR):
                        x_rows[r] = self.x_normed + Int(bucket_base[rec_block + r]) * Self.hidden

                    for m in range(Self.gate_up_fused):
                        var w_row = gu_w + m * Self.hidden
                        var accs = InlineArray[
                            InlineArray[SIMD[DType.float32, DPB_LANES], PU_GU], MR,
                        ](uninitialized=True)
                        comptime for r in range(MR):
                            comptime for p in range(PU_GU):
                                accs[r][p] = SIMD[DType.float32, DPB_LANES](0)
                        for i in range(Self.hidden // STRIDE_GU):
                            comptime for p in range(PU_GU):
                                var off = i * STRIDE_GU + p * DPB_K
                                var w_v = (w_row + off).load[width=DPB_K]()
                                comptime for r in range(MR):
                                    var x_v = (x_rows[r] + off).load[width=DPB_K]()
                                    accs[r][p] = bf16_pair_dot[4](accs[r][p], x_v, w_v)
                        comptime for r in range(MR):
                            var s = SIMD[DType.float32, DPB_LANES](0)
                            comptime for p in range(PU_GU):
                                s += accs[r][p]
                            gu_scratch[(rec_block + r) * Self.gate_up_fused + m] = (
                                s.reduce_add())
                    rec_block += MR

                while rec_block < n_tok:
                    var tok = Int(bucket_base[rec_block])
                    var x_row = self.x_normed + tok * Self.hidden
                    for m in range(Self.gate_up_fused):
                        var w_row = gu_w + m * Self.hidden
                        var accs = InlineArray[SIMD[DType.float32, DPB_LANES], PU_GU](
                            fill=SIMD[DType.float32, DPB_LANES](0))
                        for i in range(Self.hidden // STRIDE_GU):
                            comptime for p in range(PU_GU):
                                var off = i * STRIDE_GU + p * DPB_K
                                var w_v = (w_row + off).load[width=DPB_K]()
                                var x_v = (x_row + off).load[width=DPB_K]()
                                accs[p] = bf16_pair_dot[4](accs[p], x_v, w_v)
                        var s = SIMD[DType.float32, DPB_LANES](0)
                        comptime for p in range(PU_GU):
                            s += accs[p]
                        gu_scratch[rec_block * Self.gate_up_fused + m] = s.reduce_add()
                    rec_block += 1

                # GELU output stored as bf16 directly into h_bf16, no f32 stage.
                for rec in range(n_tok):
                    var gu_row = gu_scratch + rec * Self.gate_up_fused
                    var hm_row_bf16 = h_bf16 + rec * Self.intermediate
                    for j in range(0, Self.intermediate, W):
                        var g = (gu_row + j).load[width=W]()
                        var u = (gu_row + Self.intermediate + j).load[width=W]()
                        var v = gelu_tanh_f32[W](g) * u
                        (hm_row_bf16 + j).store(v.cast[DType.bfloat16]())

                # Down GEMM: bf16_pair_dot with MR=4 rec unroll + PU_DN=2
                # K-axis unroll. K=intermediate=704 with PU_DN*DPB_K=64
                # stride → 11 outer iters covering all K (no tail). MR=4
                # hides bf16_pair_dot latency by maintaining 4 independent
                # accumulator chains per K position.
                var dn_rec = 0
                while dn_rec + MR <= n_tok:
                    var hm_rows = InlineArray[BF16Ptr, MR](uninitialized=True)
                    var dst_rows = InlineArray[BF16Ptr, MR](uninitialized=True)
                    var weights = InlineArray[Scalar[DType.float32], MR](
                        uninitialized=True)
                    comptime for r in range(MR):
                        hm_rows[r] = h_bf16 + (dn_rec + r) * Self.intermediate
                        var tok = Int(bucket_base[dn_rec + r])
                        dst_rows[r] = self.moe_partial + tok * Self.hidden
                        weights[r] = weight_base[dn_rec + r]

                    for m in range(Self.hidden):
                        var w_row = down_w + m * Self.intermediate
                        var accs = InlineArray[
                            InlineArray[SIMD[DType.float32, DPB_LANES], PU_DN], MR,
                        ](uninitialized=True)
                        comptime for r in range(MR):
                            comptime for p in range(PU_DN):
                                accs[r][p] = SIMD[DType.float32, DPB_LANES](0)
                        for i in range(Self.intermediate // STRIDE_DN):
                            comptime for p in range(PU_DN):
                                var off = i * STRIDE_DN + p * DPB_K
                                var w_v = (w_row + off).load[width=DPB_K]()
                                comptime for r in range(MR):
                                    var x_v = (hm_rows[r] + off).load[width=DPB_K]()
                                    accs[r][p] = bf16_pair_dot[4](accs[r][p], x_v, w_v)
                        comptime for r in range(MR):
                            var s = SIMD[DType.float32, DPB_LANES](0)
                            comptime for p in range(PU_DN):
                                s += accs[r][p]
                            var out = s.reduce_add()
                            var dst = dst_rows[r] + m
                            dst[] = (dst[].cast[DType.float32]()
                                + out * weights[r]).cast[DType.bfloat16]()
                    dn_rec += MR

                # Tail: any recs not divisible by MR.
                while dn_rec < n_tok:
                    var hm_row = h_bf16 + dn_rec * Self.intermediate
                    var tok = Int(bucket_base[dn_rec])
                    var dst_row = self.moe_partial + tok * Self.hidden
                    var weight = weight_base[dn_rec]
                    for m in range(Self.hidden):
                        var w_row = down_w + m * Self.intermediate
                        var accs = InlineArray[SIMD[DType.float32, DPB_LANES], PU_DN](
                            fill=SIMD[DType.float32, DPB_LANES](0))
                        for i in range(Self.intermediate // STRIDE_DN):
                            comptime for p in range(PU_DN):
                                var off = i * STRIDE_DN + p * DPB_K
                                var w_v = (w_row + off).load[width=DPB_K]()
                                var x_v = (hm_row + off).load[width=DPB_K]()
                                accs[p] = bf16_pair_dot[4](accs[p], x_v, w_v)
                        var s = SIMD[DType.float32, DPB_LANES](0)
                        comptime for p in range(PU_DN):
                            s += accs[p]
                        var out = s.reduce_add()
                        var dst = dst_row + m
                        dst[] = (dst[].cast[DType.float32]()
                            + out * weight).cast[DType.bfloat16]()
                    dn_rec += 1

                tok_base += n_tok

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(
            self.x_normed, self.expert_offset, self.bucket_token_idx,
            self.bucket_weight, self.experts_gate_up, self.experts_down,
            self.gate_scratch, self.hidden_scratch, self.hidden_scratch_bf16,
            self.moe_partial, self.worker_id, start, end,
        )


comptime BaselineExpertKernel[
    hidden: Int, gate_up_fused: Int, intermediate: Int, num_local_experts: Int,
] = ExpertRunnerSlotKernel[hidden, gate_up_fused, intermediate, num_local_experts]


comptime ActiveExpertKernel[
    hidden: Int, gate_up_fused: Int, intermediate: Int, num_local_experts: Int,
] = ExpertBatchedDpDownKernel[hidden, gate_up_fused, intermediate, num_local_experts]
