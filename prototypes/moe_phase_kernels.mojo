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


@fieldwise_init
struct RouterCandidate(Copyable, ImplicitlyCopyable):
    var expert: Int32
    var logit: Float32


@fieldwise_init
struct SparseRoute(Copyable, ImplicitlyCopyable):
    var token: Int32
    var weight: Float32


comptime RouterCandidatePtr = UnsafePointer[RouterCandidate, MutAnyOrigin]
comptime SparseRoutePtr = UnsafePointer[SparseRoute, MutAnyOrigin]


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


def arena_alloc_t[
    align: Int, //, T: AnyType,
](
    mut arena: NumaArena[alignment=align], count: Int,
) -> UnsafePointer[T, MutAnyOrigin]:
    var ptr = arena.alloc[T](count)
    if not ptr:
        print("arena alloc failed for", count, "elements")
        return UnsafePointer[T, MutAnyOrigin].unsafe_dangling()
    return ptr.value()


def fill_bf16(ptr: BF16Ptr, count: Int, seed: Int):
    for i in range(count):
        var x = (i * 131 + seed * 17) % 257
        ptr[i] = Scalar[DType.bfloat16](Float32(x - 128) * 0.002)


def fill_bf16_positive(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = Scalar[DType.bfloat16](1.0 + Float32(i % 13) * 0.001)


@always_inline
def insert_candidate[top_k: Int](
    expert: Int32,
    logit: Float32,
    mut cands: InlineArray[RouterCandidate, top_k],
):
    """Sorted descending by logit; tie-break to lower expert id."""
    for k in range(top_k):
        var c = cands[k]
        if logit > c.logit or (logit == c.logit and Int(expert) < Int(c.expert)):
            var j = top_k - 1
            while j > k:
                cands[j] = cands[j - 1]
                j -= 1
            cands[k] = RouterCandidate(expert, logit)
            return


@fieldwise_init
struct RouterShardedKernel[
    hidden: Int, experts_per_rank: Int, top_k: Int,
](OutputPartitionedKernel):
    """Per-rank router: rmsnorm-scale, GEMV against expert-sharded
    router_proj, local top_k. Emits RouterCandidate[seq, top_k] with
    GLOBAL expert ids and raw logits. Cross-rank merge + softmax happens
    on the main thread after dispatch."""
    var x: BF16Ptr
    var router_proj: BF16Ptr
    var router_scale: BF16Ptr
    var scaled_scratch: F32Ptr
    var cands_out: RouterCandidatePtr
    var expert_base: Int
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime PU = pick_port_unroll[W, Self.hidden]()
        comptime STRIDE = PU * W
        comptime sqrt_n = sqrt[DType.float32, 1](Float32(Self.hidden))
        comptime n_eps = Scalar[DType.float32](
            Float32(Self.hidden) * Float32(C.RMS_NORM_EPS))
        comptime sentinel = Float32(-1.0e30)

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

            var cands = InlineArray[RouterCandidate, Self.top_k](
                fill=RouterCandidate(Int32(0), sentinel))
            for e in range(Self.experts_per_rank):
                var row = self.router_proj + e * Self.hidden
                var accs = InlineArray[SIMD[DType.float32, W], PU](
                    fill=SIMD[DType.float32, W](0))
                for i in range(Self.hidden // STRIDE):
                    comptime for p in range(PU):
                        var off = i * STRIDE + p * W
                        var xv = (scratch + off).load[width=W]()
                        var wv = (row + off).load[width=W]().cast[DType.float32]()
                        accs[p] = xv.fma(wv, accs[p])
                var logit = tree_reduce_accs(accs)
                insert_candidate[Self.top_k](
                    Int32(self.expert_base + e), logit, cands)

            var dst = self.cands_out + tok * Self.top_k
            comptime for k in range(Self.top_k):
                dst[k] = cands[k]

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(
            self.x, self.router_proj, self.router_scale,
            self.scaled_scratch, self.cands_out, self.expert_base,
            self.worker_id, start, end,
        )


def merge_candidates_inline[tp: Int, top_k: Int](
    cands_per_rank: InlineArray[RouterCandidatePtr, tp],
    per_expert_scale: BF16Ptr,
    route_idx_per_rank: InlineArray[I32Ptr, tp],
    route_w_per_rank: InlineArray[F32Ptr, tp],
    seq_len: Int,
):
    """Single-threaded merge across tp ranks. For each token: gather
    tp*top_k candidates, pick global top_k by logit (ties by expert id),
    softmax over chosen logits, multiply by per_expert_scale, replicate
    result to every rank's route_idx/route_w buffer.
    """
    comptime sentinel = Float32(-1.0e30)
    for tok in range(seq_len):
        var merged = InlineArray[RouterCandidate, top_k](
            fill=RouterCandidate(Int32(0), sentinel))
        for r in range(tp):
            var src = cands_per_rank[r] + tok * top_k
            for k in range(top_k):
                var c = src[k]
                insert_candidate[top_k](c.expert, c.logit, merged)

        var max_logit = merged[0].logit
        var sum_v = Float32(0)
        var exp_values = InlineArray[Float32, top_k](uninitialized=True)
        comptime for k in range(top_k):
            var ev = fast_exp_softmax_biased[1](
                SIMD[DType.float32, 1](merged[k].logit - max_logit))[0]
            exp_values[k] = ev
            sum_v += ev
        var inv_sum = Float32(1.0) / sum_v

        for r in range(tp):
            var idx_dst = route_idx_per_rank[r] + tok * top_k
            var w_dst = route_w_per_rank[r] + tok * top_k
            comptime for k in range(top_k):
                var expert = merged[k].expert
                var scale = (per_expert_scale + Int(expert))[].cast[DType.float32]()
                idx_dst[k] = expert
                w_dst[k] = exp_values[k] * inv_sum * scale


def build_schedule_inline[experts_per_rank: Int, top_k: Int](
    route_idx: I32Ptr,
    route_w: F32Ptr,
    seq_len: Int,
    rank: Int,
    expert_offset: I32Ptr,
    routes: SparseRoutePtr,
) -> Int:
    """Single-threaded count -> prefix -> scatter into rank-local buckets.
    Returns total local route count (= expert_offset[experts_per_rank]).
    """
    var counts = InlineArray[Int32, experts_per_rank](fill=Int32(0))
    var first = rank * experts_per_rank
    var last = first + experts_per_rank

    for tok in range(seq_len):
        for k in range(top_k):
            var e = Int(route_idx[tok * top_k + k])
            if e >= first and e < last:
                counts[e - first] += Int32(1)

    var running = Int32(0)
    var write_offsets = InlineArray[Int32, experts_per_rank](uninitialized=True)
    for e in range(experts_per_rank):
        expert_offset[e] = running
        write_offsets[e] = running
        running += counts[e]
    expert_offset[experts_per_rank] = running

    for tok in range(seq_len):
        for k in range(top_k):
            var idx = tok * top_k + k
            var e = Int(route_idx[idx])
            if e >= first and e < last:
                var local = e - first
                var pos = Int(write_offsets[local])
                routes[pos] = SparseRoute(Int32(tok), route_w[idx])
                write_offsets[local] = Int32(pos + 1)

    return Int(running)


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
    var cands: RouterCandidatePtr
    var expert_offset: I32Ptr
    var routes: SparseRoutePtr
    var hidden_bucket: BF16Ptr
    var moe_accum: F32Ptr
    var experts_gate_up: BF16Ptr
    var experts_down: BF16Ptr
    var gate_scratch: F32Ptr
