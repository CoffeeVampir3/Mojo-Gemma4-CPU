from std.collections import InlineArray
from std.memory import UnsafePointer
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from simd_math import fast_exp_softmax_biased
from simd_math.ops import sqrt

from kernels.helpers import (
    Binding, ArenaBases, WorkerRangePartitionedKernel,
    DispatchBuffer, tile_dispatch, join_all,
    BF16Ptr, F32Ptr, W,
)
from kernels.dot_products import dot_to_scalar
from kernels.rmsnorm import rms_reduce_row
from kernels.moe_router import (
    RouterCandidate, RouterCandidatePtr, SparseRoute,
    insert_candidate, build_expert_schedules,
)
from kernels.profiling import Profiler
from modeling.gemma4_common import Gemma4BaseConfig
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, print_row, now_ns, DEFAULT_SAMPLES,
)


comptime HIDDEN = Gemma4BaseConfig.HIDDEN
comptime NUM_EXPERTS = Gemma4BaseConfig.NUM_EXPERTS
comptime TOP_K = Gemma4BaseConfig.TOP_K
comptime RMS_EPS = Gemma4BaseConfig.RMS_NORM_EPS

comptime ALIGNMENT = 64
comptime WARMUP = 50
comptime SAMPLES = DEFAULT_SAMPLES
comptime MAX_WORKERS = 64
comptime MAX_SEQ = 512

# Cold sweep cycles through this many distinct router_proj layers so each
# timed call reads weights that the previous calls evicted — matching the
# model, where every layer streams GBs between router calls and router_proj
# is always cold from DRAM. ~92 MB/node working set exceeds the node's L3.
comptime L_LAYERS = 512

comptime I32Ptr = UnsafePointer[Int32, MutAnyOrigin]
comptime SparseRoutePtr = UnsafePointer[SparseRoute, MutAnyOrigin]


# ----------------------------------------------------------------------------
# Improved router: partition over experts (not tokens) so decode (seq_len==1)
# spreads the per-expert dot products across workers instead of running them
# all on one thread. Each worker recomputes the rms-normed + router-scaled x
# into its own scratch slot (rank-local, no cross-worker sharing) and keeps a
# local top_k; the dispatcher merges the per-worker candidate sets.
# ----------------------------------------------------------------------------
@fieldwise_init
struct RouterExpertKernel[
    hidden: Int, sqrt_n: Float32, n_eps: Float32,
    experts_per_rank: Int, top_k: Int,
](WorkerRangePartitionedKernel):
    var x: BF16Ptr
    var router_proj: BF16Ptr
    var router_scale: BF16Ptr
    var scaled_scratch: F32Ptr
    var cands_out: RouterCandidatePtr
    var expert_base: Int
    var seq_len: Int
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        comptime sentinel = Float32(-1.0e30)
        var scratch = self.scaled_scratch + self.worker_id * Self.hidden
        for tok in range(self.seq_len):
            var x_row = self.x + tok * Self.hidden

            var sum_sq = rms_reduce_row[Self.hidden](x_row)
            var inv_rms = Self.sqrt_n / sqrt[DType.float32, 1](
                sum_sq + Self.n_eps)
            var inv_vec = SIMD[DType.float32, W](inv_rms)
            for j in range(0, Self.hidden, W):
                var xv = (x_row + j).load[width=W]().cast[DType.float32]()
                var sv = (self.router_scale + j).load[width=W]().cast[
                    DType.float32]()
                (scratch + j).store(xv * sv * inv_vec)

            var cands = InlineArray[RouterCandidate, Self.top_k](
                fill=RouterCandidate(Int32(0), sentinel))
            for e in range(self.start, self.end):
                var row = self.router_proj + e * Self.hidden
                var logit = dot_to_scalar[Self.hidden](scratch, row)
                insert_candidate[Self.top_k](
                    Int32(self.expert_base + e), logit, cands)

            var dst = self.cands_out + (
                self.worker_id * self.seq_len + tok) * Self.top_k
            comptime for k in range(Self.top_k):
                dst[k] = cands[k]

    @always_inline
    def install_worker_range(mut self, worker_id: Int, start: Int, end: Int):
        self.worker_id = worker_id
        self.start = start
        self.end = end


def dispatch_router_v1[
    P: BurstThreadPool, Profile: Bool, N: Int, //,
    hidden: Int, sqrt_n: Float32, n_eps: Float32,
    experts_per_rank: Int, top_k: Int, tp: Int,
    max_worker_count: Int = 128,
](
    x: Binding[BFloat16, tp],
    router_proj: Binding[BFloat16, tp],
    router_scale: Binding[BFloat16, tp],
    scaled_scratch: Binding[Float32, tp],
    cands_out: Binding[RouterCandidate, tp],
    seq_len: Int,
    num_workers: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
) -> Int:
    comptime K = RouterExpertKernel[
        hidden, sqrt_n, n_eps, experts_per_rank, top_k]
    var buf = DispatchBuffer[K, max_worker_count]()
    var nw_used = 0
    for r in range(tp):
        var cap = min(max_worker_count, pools[r].get_capacity())
        var nw = min(num_workers, cap)
        var proto = K(
            x[r], router_proj[r], router_scale[r],
            scaled_scratch[r], cands_out[r],
            r * experts_per_rank, seq_len, 0, 0, 0)
        nw_used = tile_dispatch(
            buf, proto, pools[r], experts_per_rank, num_workers=nw)
    join_all[tp](pools)
    return nw_used


def merge_router_candidates_v1[tp: Int, top_k: Int](
    cands_per_rank: Binding[RouterCandidate, tp],
    nw: Int, seq_len: Int,
    per_expert_scale: BF16Ptr,
    route_idx_per_rank: Binding[Int32, tp],
    route_w_per_rank: Binding[Float32, tp],
):
    comptime sentinel = Float32(-1.0e30)
    for tok in range(seq_len):
        var merged = InlineArray[RouterCandidate, top_k](
            fill=RouterCandidate(Int32(0), sentinel))
        for r in range(tp):
            for w in range(nw):
                var src = cands_per_rank[r] + (w * seq_len + tok) * top_k
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
                var scale = (per_expert_scale + Int(expert))[].cast[
                    DType.float32]()
                idx_dst[k] = expert
                w_dst[k] = exp_values[k] * inv_sum * scale


# ----------------------------------------------------------------------------
# Full-phase runners (router scores + merge + schedule build). These mirror
# exactly what dispatch_moe does for the routing portion of a decode step.
# ----------------------------------------------------------------------------
def v1_phase[
    P: BurstThreadPool, Profile: Bool, N: Int, //,
    tp: Int, experts_per_rank: Int, sqrt_n: Float32, n_eps: Float32,
](
    mut pools: List[P], seq_len: Int, num_workers: Int,
    x: Binding[BFloat16, tp], router_proj: Binding[BFloat16, tp],
    router_scale: Binding[BFloat16, tp], pes0: BF16Ptr,
    scaled: Binding[Float32, tp], cands: Binding[RouterCandidate, tp],
    route_idx: Binding[Int32, tp], route_w: Binding[Float32, tp],
    expert_offset: Binding[Int32, tp], routes: Binding[SparseRoute, tp],
    mut prof: Profiler[Profile, N],
) -> Int:
    var nw = dispatch_router_v1[
        hidden=HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
        experts_per_rank=experts_per_rank, top_k=TOP_K, tp=tp,
    ](x, router_proj, router_scale, scaled, cands, seq_len, num_workers,
      pools, prof)
    merge_router_candidates_v1[tp, TOP_K](
        cands, nw, seq_len, pes0, route_idx, route_w)
    build_expert_schedules[tp, experts_per_rank, TOP_K](
        route_idx, route_w, expert_offset, routes, seq_len)
    return nw


# ----------------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------------
def arena_alloc[T: AnyType](
    mut arena: NumaArena[alignment=ALIGNMENT], count: Int,
) -> UnsafePointer[T, MutAnyOrigin]:
    var ptr = arena.alloc[T](count)
    if not ptr:
        print("arena alloc failed for", count, "elements")
        return UnsafePointer[T, MutAnyOrigin].unsafe_dangling()
    return ptr.value()


def arena_alloc_all[T: AnyType, tp: Int](
    mut arenas: List[NumaArena[alignment=ALIGNMENT]], count: Int,
) -> UnsafePointer[T, MutAnyOrigin]:
    var first = UnsafePointer[T, MutAnyOrigin].unsafe_dangling()
    for r in range(tp):
        var ptr = arena_alloc[T](arenas[r], count)
        if r == 0:
            first = ptr
    return first


def arena_bases[tp: Int](
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
) -> ArenaBases[tp]:
    var bases = ArenaBases[tp].uninitialized()
    for r in range(tp):
        bases[r] = Int(arenas[r].base.value())
    return bases


def fill_bf16(ptr: BF16Ptr, count: Int):
    for i in range(count):
        ptr[i] = BFloat16(Float32((i % 253) - 126) * 0.005)


def fill_bf16_all[tp: Int](ptrs: Binding[BFloat16, tp], count: Int):
    for r in range(tp):
        fill_bf16(ptrs[r], count)


def topk_matches[tp: Int](
    route_idx: Binding[Int32, tp], route_w: Binding[Float32, tp],
    ref_idx: InlineArray[Int32, TOP_K], ref_w: InlineArray[Float32, TOP_K],
) -> Bool:
    for k in range(TOP_K):
        if route_idx[0][k] != ref_idx[k]:
            return False
        var d = route_w[0][k] - ref_w[k]
        if d < Float32(0):
            d = -d
        if d > Float32(1e-4):
            return False
    return True


def measure_sweep[
    P: BurstThreadPool, //,
    tp: Int, experts_per_rank: Int, sqrt_n: Float32, n_eps: Float32,
](
    mut pools: List[P], label: String, n_layers: Int, seq_len: Int,
    mut samples: SampleBuffer, router_bytes: Int,
    x: Binding[BFloat16, tp], rp_base: Binding[BFloat16, tp],
    router_scale: Binding[BFloat16, tp], pes0: BF16Ptr,
    scaled: Binding[Float32, tp], cands: Binding[RouterCandidate, tp],
    route_idx: Binding[Int32, tp], route_w: Binding[Float32, tp],
    expert_offset: Binding[Int32, tp], routes: Binding[SparseRoute, tp],
    mut prof: Profiler[False],
):
    comptime stride = experts_per_rank * HIDDEN
    print("\n=== " + label + " ===")

    # reference top-k from a single-worker run at layer 0 (algorithm correctness)
    _ = v1_phase[
        tp=tp, experts_per_rank=experts_per_rank, sqrt_n=sqrt_n, n_eps=n_eps,
    ](pools, seq_len, 1, x, rp_base, router_scale, pes0,
      scaled, cands, route_idx, route_w, expert_offset, routes, prof)
    var ref_idx = InlineArray[Int32, TOP_K](uninitialized=True)
    var ref_w = InlineArray[Float32, TOP_K](uninitialized=True)
    for k in range(TOP_K):
        ref_idx[k] = route_idx[0][k]
        ref_w[k] = route_w[0][k]

    # -------- v1 (expert-partitioned dispatch) worker-count sweep --------
    var nws = InlineArray[Int, 6](uninitialized=True)
    nws[0] = 1; nws[1] = 2; nws[2] = 4; nws[3] = 8; nws[4] = 16; nws[5] = 32
    for wi in range(6):
        var num_workers = nws[wi]
        for i in range(WARMUP):
            var rp = rp_base.shifted((i % n_layers) * stride)
            _ = v1_phase[
                tp=tp, experts_per_rank=experts_per_rank,
                sqrt_n=sqrt_n, n_eps=n_eps,
            ](pools, seq_len, num_workers, x, rp, router_scale, pes0,
              scaled, cands, route_idx, route_w, expert_offset, routes, prof)
        samples.clear()
        var nw_used = 0
        for i in range(SAMPLES):
            var rp = rp_base.shifted((i % n_layers) * stride)
            var t0 = now_ns()
            nw_used = v1_phase[
                tp=tp, experts_per_rank=experts_per_rank,
                sqrt_n=sqrt_n, n_eps=n_eps,
            ](pools, seq_len, num_workers, x, rp, router_scale, pes0,
              scaled, cands, route_idx, route_w, expert_offset, routes, prof)
            var t1 = now_ns()
            samples.push(t1 - t0, t1 - t0)
        keep(route_idx[0][0])

        # correctness at layer 0 against the baseline reference
        _ = v1_phase[
            tp=tp, experts_per_rank=experts_per_rank,
            sqrt_n=sqrt_n, n_eps=n_eps,
        ](pools, seq_len, num_workers, x, rp_base, router_scale, pes0,
          scaled, cands, route_idx, route_w, expert_offset, routes, prof)
        var ok = topk_matches[tp](route_idx, route_w, ref_idx, ref_w)

        var v_stats = compute_stats(samples.wall_ns, samples.n)
        print_row(String(t"v1 workers={num_workers} (used={nw_used})"),
                  v_stats, v_stats, router_bytes)
        if ok:
            print("    correctness vs 1-worker ref: PASS")
        else:
            print("    correctness vs 1-worker ref: FAIL")


def run_all[P: BurstThreadPool, //, tp: Int](
    mut pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    comptime experts_per_rank = NUM_EXPERTS // tp
    comptime sqrt_n = sqrt[DType.float32, 1](HIDDEN)
    comptime n_eps = HIDDEN * RMS_EPS

    var bases = arena_bases[tp](arenas)

    var x_ptr = arena_alloc_all[BFloat16, tp](arenas, MAX_SEQ * HIDDEN)
    var rp_ptr = arena_alloc_all[BFloat16, tp](
        arenas, L_LAYERS * experts_per_rank * HIDDEN)
    var rs_ptr = arena_alloc_all[BFloat16, tp](arenas, HIDDEN)
    var pes_ptr = arena_alloc_all[BFloat16, tp](arenas, NUM_EXPERTS)
    var scaled_ptr = arena_alloc_all[Float32, tp](arenas, MAX_WORKERS * HIDDEN)
    var cands_ptr = arena_alloc_all[RouterCandidate, tp](
        arenas, MAX_WORKERS * MAX_SEQ * TOP_K)
    var route_idx_ptr = arena_alloc_all[Int32, tp](arenas, MAX_SEQ * TOP_K)
    var route_w_ptr = arena_alloc_all[Float32, tp](arenas, MAX_SEQ * TOP_K)
    var expert_offset_ptr = arena_alloc_all[Int32, tp](
        arenas, experts_per_rank + 1)
    var routes_ptr = arena_alloc_all[SparseRoute, tp](arenas, MAX_SEQ * TOP_K)

    var x = Binding[BFloat16, tp](x_ptr, bases)
    var router_proj = Binding[BFloat16, tp](rp_ptr, bases)
    var router_scale = Binding[BFloat16, tp](rs_ptr, bases)
    var scaled = Binding[Float32, tp](scaled_ptr, bases)
    var cands = Binding[RouterCandidate, tp](cands_ptr, bases)
    var route_idx = Binding[Int32, tp](route_idx_ptr, bases)
    var route_w = Binding[Float32, tp](route_w_ptr, bases)
    var expert_offset = Binding[Int32, tp](expert_offset_ptr, bases)
    var routes = Binding[SparseRoute, tp](routes_ptr, bases)

    var per_expert_scale = Binding[BFloat16, tp](pes_ptr, bases)
    fill_bf16_all[tp](x, MAX_SEQ * HIDDEN)
    fill_bf16_all[tp](router_proj, L_LAYERS * experts_per_rank * HIDDEN)
    fill_bf16_all[tp](router_scale, HIDDEN)
    for r in range(tp):
        var pes_r = per_expert_scale[r]
        for i in range(NUM_EXPERTS):
            pes_r[i] = BFloat16(1.0)

    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    # per_expert_scale read by merge from rank-0 base (matches model layout).
    var pes0 = pes_ptr

    var cap = pools[0].get_capacity()
    print(
        t"hidden={HIDDEN} experts={NUM_EXPERTS} experts/rank={experts_per_rank}"
        t" top_k={TOP_K} tp={tp} capacity={cap}/rank")

    comptime router_bytes = NUM_EXPERTS * HIDDEN * 2  # aggregate weight read
    var samples = SampleBuffer(SAMPLES)
    var prof = Profiler[False]()

    var seqs = InlineArray[Int, 2](uninitialized=True)
    seqs[0] = 1
    seqs[1] = 512
    for si in range(2):
        var seq_len = seqs[si]
        print(t"\n################ router phase, seq_len={seq_len} ################")
        print("(payload = aggregate router_proj bytes across all ranks)")

        measure_sweep[
            tp=tp, experts_per_rank=experts_per_rank, sqrt_n=sqrt_n, n_eps=n_eps,
        ](pools, "HOT weights (router_proj resident in cache)", 1, seq_len,
          samples, router_bytes, x, router_proj, router_scale, pes0,
          scaled, cands, route_idx, route_w, expert_offset, routes, prof)

        measure_sweep[
            tp=tp, experts_per_rank=experts_per_rank, sqrt_n=sqrt_n, n_eps=n_eps,
        ](pools, "COLD weights (cycled layers, read from DRAM)", L_LAYERS, seq_len,
          samples, router_bytes, x, router_proj, router_scale, pes0,
          scaled, cands, route_idx, route_w, expert_offset, routes, prof)


def main():
    var topo = NumaTopology()
    var tp = len(topo)

    print("Router phase prototype")
    var iso = len(topo.isolated_cpus)
    print(t"{tp} NUMA node(s), {iso} isolated cpus\n")

    # router_proj dominates the arena; experts_per_rank = NUM_EXPERTS // tp.
    var experts_per_rank = NUM_EXPERTS // tp
    var rp_bytes = L_LAYERS * experts_per_rank * HIDDEN * 2
    var arena_bytes = rp_bytes + 32 * 1024 * 1024
    var arenas = List[NumaArena[alignment=ALIGNMENT]](capacity=tp)
    for i in range(tp):
        arenas.append(NumaArena[alignment=ALIGNMENT](topo[i], arena_bytes))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_router_tp[
        P: BurstThreadPool, //, degree: Int,
    ](var selected_pools: List[P]):
        run_all[tp=degree](selected_pools, arenas)

    with_topological_rank_dispatch[
        dispatch=dispatch_router_tp,
    ](topo, "mode: isolated", "mode: spin-backoff")
