from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from numa import NumaArena
from kernels.helpers import OutputPartitionedKernel
from kernels.rmsnorm import rms_norm_row
from modeling.gemma4_common import Gemma4BaseConfig
from simd_math import (
    pick_port_unroll, tree_reduce_accs, fast_exp_softmax_biased,
)
from simd_math.ops import sqrt


comptime C = Gemma4BaseConfig
comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr  = UnsafePointer[Scalar[DType.float32],  MutAnyOrigin]
comptime I32Ptr  = UnsafePointer[Scalar[DType.int32],    MutAnyOrigin]
comptime W = simd_width_of[DType.float32]()


def arena_alloc[
    align: Int, //, dtype: DType,
](
    mut arena: NumaArena[alignment=align], count: Int,
) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
    var ptr = arena.alloc[Scalar[dtype]](count)
    if not ptr:
        print("arena alloc failed for", count, "elements")
        return UnsafePointer[Scalar[dtype], MutAnyOrigin].unsafe_dangling()
    return ptr.value()


def fill_bf16(ptr: BF16Ptr, count: Int, seed: Int):
    for i in range(count):
        var x = (i * 131 + seed * 17) % 257
        ptr[i] = Scalar[DType.bfloat16](Float32(x - 128) * 0.002)


def fill_bf16_positive(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = Scalar[DType.bfloat16](1.0 + Float32(i % 13) * 0.001)


@always_inline
def insert_topk[top_k: Int](
    value: Scalar[DType.float32],
    idx: Int,
    mut top_values: InlineArray[Scalar[DType.float32], top_k],
    mut top_indices: InlineArray[Int, top_k],
):
    for k in range(top_k):
        if value > top_values[k] or (value == top_values[k] and idx < top_indices[k]):
            var j = top_k - 1
            while j > k:
                top_values[j] = top_values[j - 1]
                top_indices[j] = top_indices[j - 1]
                j -= 1
            top_values[k] = value
            top_indices[k] = idx
            return


@fieldwise_init
struct RouterStreamKernel[hidden: Int, num_experts: Int, top_k: Int](OutputPartitionedKernel):
    var x: BF16Ptr
    var router_proj: BF16Ptr
    var router_scale: BF16Ptr
    var per_expert_scale: BF16Ptr
    var scaled_scratch: F32Ptr
    var out_idx: I32Ptr
    var out_w: F32Ptr
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime PU = pick_port_unroll[W, Self.hidden]()
        comptime STRIDE = PU * W
        comptime sqrt_n = sqrt[DType.float32, 1](Float32(Self.hidden))
        comptime n_eps = Scalar[DType.float32](
            Float32(Self.hidden) * Float32(C.RMS_NORM_EPS))
        comptime sentinel = Scalar[DType.float32](-1.0e30)

        var scratch = self.scaled_scratch + self.worker_id * Self.hidden
        for tok in range(self.start, self.end):
            var x_row = self.x + tok * Self.hidden

            var ss_accs = InlineArray[SIMD[DType.float32, W], PU](
                fill=SIMD[DType.float32, W](0))
            for i in range(Self.hidden // STRIDE):
                comptime for p in range(PU):
                    var off = i * STRIDE + p * W
                    var xv = (x_row + off).load[width=W]().cast[DType.float32]()
                    ss_accs[p] = xv.fma(xv, ss_accs[p])
            var inv_rms = sqrt_n / sqrt[DType.float32, 1](
                tree_reduce_accs(ss_accs) + n_eps)
            var inv_vec = SIMD[DType.float32, W](inv_rms)
            for j in range(0, Self.hidden, W):
                var xv = (x_row + j).load[width=W]().cast[DType.float32]()
                var sv = (self.router_scale + j).load[width=W]().cast[DType.float32]()
                (scratch + j).store(xv * sv * inv_vec)

            var top_values = InlineArray[Scalar[DType.float32], Self.top_k](
                fill=sentinel)
            var top_indices = InlineArray[Int, Self.top_k](fill=0)
            for e in range(Self.num_experts):
                var row = self.router_proj + e * Self.hidden
                var accs = InlineArray[SIMD[DType.float32, W], PU](
                    fill=SIMD[DType.float32, W](0))
                for i in range(Self.hidden // STRIDE):
                    comptime for p in range(PU):
                        var off = i * STRIDE + p * W
                        var xv = (scratch + off).load[width=W]()
                        var wv = (row + off).load[width=W]().cast[DType.float32]()
                        accs[p] = xv.fma(wv, accs[p])
                insert_topk[Self.top_k](tree_reduce_accs(accs), e, top_values, top_indices)

            var max_v = top_values[0]
            var sum_v = Scalar[DType.float32](0)
            var exp_values = InlineArray[Scalar[DType.float32], Self.top_k](
                uninitialized=True)
            comptime for k in range(Self.top_k):
                var ev = fast_exp_softmax_biased[1](
                    SIMD[DType.float32, 1](top_values[k] - max_v))[0]
                exp_values[k] = ev
                sum_v += ev
            var inv_sum = Scalar[DType.float32](1.0) / sum_v
            var idx_row = self.out_idx + tok * Self.top_k
            var w_row = self.out_w + tok * Self.top_k
            comptime for k in range(Self.top_k):
                var expert = top_indices[k]
                var scale = (self.per_expert_scale + expert)[].cast[DType.float32]()
                (idx_row + k)[] = Int32(expert)
                (w_row + k)[] = exp_values[k] * inv_sum * scale

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(
            self.x, self.router_proj, self.router_scale, self.per_expert_scale,
            self.scaled_scratch, self.out_idx, self.out_w,
            self.worker_id, start, end,
        )


@fieldwise_init
struct RouteGatherConfig[tp: Int](Copyable):
    var src_idx: InlineArray[I32Ptr, Self.tp]
    var src_w: InlineArray[F32Ptr, Self.tp]


@fieldwise_init
struct RouteGatherDstKernel[
    tp: Int, top_k: Int, cfg_origin: ImmutOrigin,
](OutputPartitionedKernel):
    var config: UnsafePointer[RouteGatherConfig[Self.tp], Self.cfg_origin]
    var dst_idx: I32Ptr
    var dst_w: F32Ptr
    var seq_len: Int
    var start: Int
    var end: Int

    def execute(mut self):
        for rec in range(self.start, self.end):
            var tok = rec // Self.top_k
            var owner = tok * Self.tp // self.seq_len
            self.dst_idx[rec] = self.config[].src_idx[owner][rec]
            self.dst_w[rec] = self.config[].src_w[owner][rec]

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(
            self.config, self.dst_idx, self.dst_w, self.seq_len, start, end,
        )


@fieldwise_init
struct FillBF16Kernel(OutputPartitionedKernel):
    var dst: BF16Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        var z = SIMD[DType.bfloat16, simd_width_of[DType.bfloat16]()](0)
        comptime BW = simd_width_of[DType.bfloat16]()
        var pos = self.start
        while pos + BW <= self.end:
            (self.dst + pos).store(z)
            pos += BW
        while pos < self.end:
            self.dst[pos] = Scalar[DType.bfloat16](0)
            pos += 1

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.dst, start, end)


@fieldwise_init
struct RmsNormBenchKernel[
    hidden: Int, sqrt_n: Scalar[DType.float32], n_eps: Scalar[DType.float32],
](OutputPartitionedKernel):
    var src: BF16Ptr
    var dst: BF16Ptr
    var weight: BF16Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        for tok in range(self.start, self.end):
            rms_norm_row[Self.hidden, Self.sqrt_n, Self.n_eps](
                self.src + tok * Self.hidden,
                self.dst + tok * Self.hidden,
                self.weight,
            )

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.src, self.dst, self.weight, start, end)


@fieldwise_init
struct ExpertCountSlotKernel[top_k: Int, num_local_experts: Int](OutputPartitionedKernel):
    var indices: I32Ptr
    var counts_per_worker: I32Ptr
    var rank: Int
    var worker_id: Int
    var slot: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var counts = self.counts_per_worker + self.worker_id * Self.num_local_experts
        for e in range(Self.num_local_experts):
            counts[e] = Int32(0)
        var first = self.rank * Self.num_local_experts
        var last = first + Self.num_local_experts
        for tok in range(self.start, self.end):
            var expert = Int(self.indices[tok * Self.top_k + self.slot])
            if expert >= first and expert < last:
                counts[expert - first] += 1

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(
            self.indices, self.counts_per_worker, self.rank,
            self.worker_id, self.slot, start, end,
        )


@fieldwise_init
struct PrefixKernel[num_local_experts: Int](OutputPartitionedKernel):
    var counts_per_worker: I32Ptr
    var expert_offset: I32Ptr
    var worker_cursor: I32Ptr
    var num_workers: Int
    var start: Int
    var end: Int

    def execute(mut self):
        if self.start != 0:
            return
        var running = 0
        for e in range(Self.num_local_experts):
            self.expert_offset[e] = Int32(running)
            var per_e = 0
            for w in range(self.num_workers):
                var c = Int(self.counts_per_worker[w * Self.num_local_experts + e])
                self.worker_cursor[w * Self.num_local_experts + e] = Int32(running + per_e)
                per_e += c
            running += per_e
        self.expert_offset[Self.num_local_experts] = Int32(running)

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(
            self.counts_per_worker, self.expert_offset, self.worker_cursor,
            self.num_workers, start, end,
        )


@fieldwise_init
struct PlaceSlotKernel[top_k: Int, num_local_experts: Int](OutputPartitionedKernel):
    var indices: I32Ptr
    var weights: F32Ptr
    var worker_cursor: I32Ptr
    var bucket_token_idx: I32Ptr
    var bucket_weight: F32Ptr
    var rank: Int
    var worker_id: Int
    var slot: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var cursors = self.worker_cursor + self.worker_id * Self.num_local_experts
        var first = self.rank * Self.num_local_experts
        var last = first + Self.num_local_experts
        for tok in range(self.start, self.end):
            var row = tok * Self.top_k + self.slot
            var expert = Int(self.indices[row])
            if expert >= first and expert < last:
                var local = expert - first
                var pos = Int(cursors[local])
                cursors[local] = Int32(pos + 1)
                self.bucket_token_idx[pos] = Int32(tok)
                self.bucket_weight[pos] = self.weights[row]

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(
            self.indices, self.weights, self.worker_cursor,
            self.bucket_token_idx, self.bucket_weight, self.rank,
            self.worker_id, self.slot, start, end,
        )


@fieldwise_init
struct RankState(Copyable, ImplicitlyCopyable):
    var x: BF16Ptr
    var router_proj: BF16Ptr
    var router_scale: BF16Ptr
    var per_expert_scale: BF16Ptr
    var pre_ffn_norm_2: BF16Ptr
    var route_idx: I32Ptr
    var route_w: F32Ptr
    var router_scaled: F32Ptr
    var x_normed: BF16Ptr
    var moe_partial: BF16Ptr
    var counts_per_worker: I32Ptr
    var expert_offset: I32Ptr
    var worker_cursor: I32Ptr
    var bucket_token_idx: I32Ptr
    var bucket_weight: F32Ptr
    var experts_gate_up: BF16Ptr
    var experts_down: BF16Ptr
    var gate_scratch: F32Ptr
    var hidden_scratch: F32Ptr
    var hidden_scratch_bf16: BF16Ptr
