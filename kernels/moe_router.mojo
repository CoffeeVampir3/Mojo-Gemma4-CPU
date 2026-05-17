from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from notstdcollections import HeapMoveArray
from threading.threading_traits import BurstThreadPool
from simd_math import pick_port_unroll, tree_reduce_accs, fast_exp_softmax_biased
from simd_math.ops import sqrt
from .helpers import (
    OutputPartitionedKernel, DispatchBuffer, Binding,
    recommended_workers, worker_range, join_all,
)


comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr  = UnsafePointer[Scalar[DType.float32],  MutAnyOrigin]
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


@always_inline
def insert_candidate[top_k: Int](
    expert: Int32,
    logit: Float32,
    mut cands: InlineArray[RouterCandidate, top_k],
):
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
    rms_eps: Scalar[DType.float32],
](OutputPartitionedKernel):
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
            Float32(Self.hidden) * Self.rms_eps)
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


comptime ROUTER_INLINE_TOKENS = 16


def dispatch_router_sharded[
    P: BurstThreadPool, //,
    hidden: Int, experts_per_rank: Int, top_k: Int, tp: Int,
    rms_eps: Scalar[DType.float32], max_worker_count: Int = 128,
](
    x: Binding[Scalar[DType.bfloat16], tp],
    router_proj: Binding[Scalar[DType.bfloat16], tp],
    router_scale: Binding[Scalar[DType.bfloat16], tp],
    scaled_scratch: Binding[Scalar[DType.float32], tp],
    cands_out: Binding[RouterCandidate, tp],
    seq_len: Int,
    mut pools: HeapMoveArray[P],
):
    if seq_len <= 0:
        return

    if seq_len <= ROUTER_INLINE_TOKENS:
        for r in range(tp):
            var k = RouterShardedKernel[
                hidden, experts_per_rank, top_k, rms_eps,
            ](
                x[r], router_proj[r], router_scale[r],
                scaled_scratch[r], cands_out[r],
                r * experts_per_rank, 0, 0, seq_len,
            )
            k.execute()
        return

    var data_bytes = seq_len * (hidden + experts_per_rank * hidden) * 2
    var buf = DispatchBuffer[
        RouterShardedKernel[hidden, experts_per_rank, top_k, rms_eps],
        max_worker_count,
    ]()
    for r in range(tp):
        var cap = min(max_worker_count, pools[r].get_capacity())
        var nw = recommended_workers(data_bytes, cap)
        nw = min(nw, seq_len)
        for w in range(nw):
            var wr = worker_range(seq_len, nw, w)
            buf.slot()[] = RouterShardedKernel[
                hidden, experts_per_rank, top_k, rms_eps,
            ](
                x[r], router_proj[r], router_scale[r],
                scaled_scratch[r], cands_out[r],
                r * experts_per_rank, w, wr[0], wr[1],
            )
        buf.dispatch(pools[r])
    join_all[tp](pools)


def merge_router_candidates[tp: Int, top_k: Int](
    cands_per_rank: Binding[RouterCandidate, tp],
    per_expert_scale: BF16Ptr,
    route_idx_per_rank: Binding[Scalar[DType.int32], tp],
    route_w_per_rank: Binding[Scalar[DType.float32], tp],
    seq_len: Int,
):
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


def build_expert_schedules[tp: Int, experts_per_rank: Int, top_k: Int](
    route_idx: Binding[Scalar[DType.int32], tp],
    route_w: Binding[Scalar[DType.float32], tp],
    expert_offset: Binding[Scalar[DType.int32], tp],
    routes: Binding[SparseRoute, tp],
    seq_len: Int,
):
    for r in range(tp):
        var counts = InlineArray[Int32, experts_per_rank](fill=Int32(0))
        var first = r * experts_per_rank
        var last = first + experts_per_rank
        var idx_r = route_idx[r]
        var w_r = route_w[r]
        var offsets_r = expert_offset[r]
        var routes_r = routes[r]

        for tok in range(seq_len):
            for k in range(top_k):
                var e = Int(idx_r[tok * top_k + k])
                if e >= first and e < last:
                    counts[e - first] += Int32(1)

        var running = Int32(0)
        var write_offsets = InlineArray[Int32, experts_per_rank](
            uninitialized=True)
        for e in range(experts_per_rank):
            offsets_r[e] = running
            write_offsets[e] = running
            running += counts[e]
        offsets_r[experts_per_rank] = running

        for tok in range(seq_len):
            for k in range(top_k):
                var slot = tok * top_k + k
                var e = Int(idx_r[slot])
                if e >= first and e < last:
                    var local = e - first
                    var pos = Int(write_offsets[local])
                    routes_r[pos] = SparseRoute(Int32(tok), w_r[slot])
                    write_offsets[local] = Int32(pos + 1)
