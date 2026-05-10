from std.collections import InlineArray
from std.memory import UnsafePointer
from std.time import perf_counter_ns
from std.benchmark import keep
from std.sys.info import simd_width_of

from numa import NumaArena, NumaInfo
from threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool
from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from kernels.helpers import (
    DispatchBuffer, RankBuffers, worker_range, join_all,
    recommended_workers,
)
from kernels.reductions import dispatch_allreduce
from modeling.model_spec import BF16
from modeling.gemma4_common import Gemma4BaseConfig
from simd_math.ops import sqrt
from prototypes.expert_kernels import (
    Phase1GateUpKernel, Phase2DownKernel, PHASE1_TILE_J, PHASE1_MR,
)
from prototypes.moe_phase_kernels import (
    arena_alloc, arena_alloc_t, fill_bf16, fill_bf16_positive,
    RouterShardedKernel, RouterCandidate, RouterCandidatePtr,
    SparseRoute, SparseRoutePtr,
    merge_candidates_inline, build_schedule_inline,
    RmsNormBenchKernel, RankState,
)


comptime C = Gemma4BaseConfig
comptime ALIGNMENT = 64
comptime WARMUP = 1
comptime TRIALS = 3
comptime BENCH_SEQ = C.MAX_SEQ_LEN
comptime MAX_WORKERS = 128

comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
comptime I32Ptr = UnsafePointer[Scalar[DType.int32], MutAnyOrigin]
comptime W = simd_width_of[DType.float32]()


def fmt_ns(ns: Int) -> String:
    if ns < 1000:
        return String(ns) + " ns"
    elif ns < 1000000:
        return String(ns // 1000) + "." + String((ns % 1000) // 100) + " us"
    else:
        return String(ns // 1000000) + "." + String((ns % 1000000) // 100000) + " ms"


def fmt_mib(bytes: Int) -> String:
    var scaled = bytes * 100 // (1024 * 1024)
    var whole = scaled // 100
    var frac = scaled % 100
    var frac_s = String(frac)
    if frac < 10:
        frac_s = "0" + frac_s
    return String(whole) + "." + frac_s


def fmt_bw(total_bytes: Int, ns: Int) -> String:
    if ns <= 0:
        return "inf GB/s"
    var bw_100 = total_bytes * 100 // ns
    return String(bw_100 // 100) + "." + String(bw_100 % 100) + " GB/s"


def print_phase(label: String, wall_ns: Int, worker_ns: Int, bytes: Int = 0):
    var extra = ""
    if bytes > 0:
        extra = "  " + fmt_bw(bytes, wall_ns)
    print(
        "  " + label
        + " wall=" + fmt_ns(wall_ns)
        + " worker=" + fmt_ns(worker_ns)
        + extra
    )


def max_worker_ts[P: BurstThreadPool, //, tp: Int](mut pools: HeapMoveArray[P]) -> Int:
    var m = 0
    for r in range(tp):
        var ts = pools[r].last_worker_timestamp()
        if ts > m:
            m = ts
    return m


@fieldwise_init
struct PhaseTiming(Copyable, ImplicitlyCopyable):
    var wall_ns: Int
    var worker_ns: Int


@fieldwise_init
struct PhaseTotals(Copyable, ImplicitlyCopyable):
    var router: PhaseTiming
    var merge: PhaseTiming
    var norm: PhaseTiming
    var schedule: PhaseTiming
    var phase1: PhaseTiming
    var phase2: PhaseTiming
    var allreduce: PhaseTiming
    var total_wall_ns: Int


def empty_timing() -> PhaseTiming:
    return PhaseTiming(0, 0)


def empty_totals() -> PhaseTotals:
    return PhaseTotals(
        empty_timing(), empty_timing(), empty_timing(), empty_timing(),
        empty_timing(), empty_timing(), empty_timing(), 0,
    )


@always_inline
def better(candidate: PhaseTotals, incumbent: PhaseTotals) -> Bool:
    return incumbent.total_wall_ns == 0 or candidate.total_wall_ns < incumbent.total_wall_ns


def dispatch_router_sharded_phase[
    P: BurstThreadPool, //, tp: Int, num_local_experts: Int,
](
    states: InlineArray[RankState, tp], mut pools: HeapMoveArray[P],
) -> PhaseTiming:
    var t0 = Int(perf_counter_ns())
    for r in range(tp):
        var cap = pools[r].get_capacity()
        var buf = DispatchBuffer[
            RouterShardedKernel[C.HIDDEN, num_local_experts, C.TOP_K]
        ]()
        for w in range(cap):
            var wr = worker_range(BENCH_SEQ, cap, w)
            buf.slot()[] = RouterShardedKernel[
                C.HIDDEN, num_local_experts, C.TOP_K,
            ](
                states[r].x, states[r].router_proj, states[r].router_scale,
                states[r].router_scaled, states[r].cands,
                r * num_local_experts, w, wr[0], wr[1],
            )
        buf.dispatch(pools[r])
    join_all[tp](pools)
    var worker = max_worker_ts[tp=tp](pools) - t0
    var t1 = Int(perf_counter_ns())
    return PhaseTiming(t1 - t0, worker)


def merge_step[tp: Int](
    states: InlineArray[RankState, tp], seq_len: Int,
) -> PhaseTiming:
    var t0 = Int(perf_counter_ns())
    var cands = InlineArray[RouterCandidatePtr, tp](uninitialized=True)
    var idx_per = InlineArray[I32Ptr, tp](uninitialized=True)
    var w_per = InlineArray[F32Ptr, tp](uninitialized=True)
    for r in range(tp):
        cands[r] = states[r].cands
        idx_per[r] = states[r].route_idx
        w_per[r] = states[r].route_w
    merge_candidates_inline[tp, C.TOP_K](
        cands, states[0].per_expert_scale, idx_per, w_per, seq_len)
    var t1 = Int(perf_counter_ns())
    return PhaseTiming(t1 - t0, t1 - t0)


def dispatch_norm_phase[P: BurstThreadPool, //, tp: Int](
    states: InlineArray[RankState, tp], mut pools: HeapMoveArray[P],
) -> PhaseTiming:
    comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
    comptime n_eps = C.HIDDEN * C.RMS_NORM_EPS
    var t0 = Int(perf_counter_ns())
    for r in range(tp):
        var cap = pools[r].get_capacity()
        var nw = recommended_workers(BENCH_SEQ * C.HIDDEN * 2, cap)
        var buf = DispatchBuffer[RmsNormBenchKernel[C.HIDDEN, sqrt_n, n_eps]]()
        for w in range(nw):
            var wr = worker_range(BENCH_SEQ, nw, w)
            buf.slot()[] = RmsNormBenchKernel[C.HIDDEN, sqrt_n, n_eps](
                states[r].x, states[r].x_normed, states[r].pre_ffn_norm_2,
                wr[0], wr[1],
            )
        buf.dispatch(pools[r])
    join_all[tp](pools)
    var worker = max_worker_ts[tp=tp](pools) - t0
    var t1 = Int(perf_counter_ns())
    return PhaseTiming(t1 - t0, worker)


def schedule_step[tp: Int, num_local_experts: Int](
    states: InlineArray[RankState, tp], seq_len: Int,
) -> PhaseTiming:
    var t0 = Int(perf_counter_ns())
    for r in range(tp):
        _ = build_schedule_inline[num_local_experts, C.TOP_K](
            states[r].route_idx, states[r].route_w, seq_len, r,
            states[r].expert_offset, states[r].routes)
    var t1 = Int(perf_counter_ns())
    return PhaseTiming(t1 - t0, t1 - t0)


def dispatch_phase1_step[
    P: BurstThreadPool, //, tp: Int, num_local_experts: Int,
](
    states: InlineArray[RankState, tp], mut pools: HeapMoveArray[P],
) -> PhaseTiming:
    comptime n_tiles = C.MOE_INTERMEDIATE // PHASE1_TILE_J
    comptime total_units = num_local_experts * n_tiles
    var t0 = Int(perf_counter_ns())
    for r in range(tp):
        var cap = pools[r].get_capacity()
        var nw = min(cap, total_units)
        var buf = DispatchBuffer[
            Phase1GateUpKernel[
                C.HIDDEN, C.MOE_GATE_UP_FUSED, C.MOE_INTERMEDIATE,
                num_local_experts,
            ]
        ]()
        for w in range(nw):
            var wr = worker_range(total_units, nw, w)
            buf.slot()[] = Phase1GateUpKernel[
                C.HIDDEN, C.MOE_GATE_UP_FUSED, C.MOE_INTERMEDIATE,
                num_local_experts,
            ](
                states[r].x_normed, states[r].expert_offset, states[r].routes,
                states[r].experts_gate_up, states[r].gate_scratch,
                states[r].hidden_bucket,
                w, wr[0], wr[1],
            )
        buf.dispatch(pools[r])
    join_all[tp](pools)
    var worker = max_worker_ts[tp=tp](pools) - t0
    var t1 = Int(perf_counter_ns())
    return PhaseTiming(t1 - t0, worker)


def dispatch_phase2_step[
    P: BurstThreadPool, //, tp: Int, num_local_experts: Int,
](
    states: InlineArray[RankState, tp], mut pools: HeapMoveArray[P], seq_len: Int,
) -> PhaseTiming:
    comptime hidden_strides = C.HIDDEN // W
    var t0 = Int(perf_counter_ns())
    for r in range(tp):
        var cap = pools[r].get_capacity()
        var nw = min(cap, hidden_strides)
        var buf = DispatchBuffer[
            Phase2DownKernel[C.HIDDEN, C.MOE_INTERMEDIATE, num_local_experts]
        ]()
        for w in range(nw):
            var sr = worker_range(hidden_strides, nw, w)
            buf.slot()[] = Phase2DownKernel[
                C.HIDDEN, C.MOE_INTERMEDIATE, num_local_experts,
            ](
                states[r].expert_offset, states[r].routes,
                states[r].hidden_bucket, states[r].experts_down,
                states[r].moe_accum, states[r].moe_partial,
                seq_len, sr[0] * W, sr[1] * W,
            )
        buf.dispatch(pools[r])
    join_all[tp](pools)
    var worker = max_worker_ts[tp=tp](pools) - t0
    var t1 = Int(perf_counter_ns())
    return PhaseTiming(t1 - t0, worker)


def dispatch_allreduce_phase[P: BurstThreadPool, //, tp: Int](
    states: InlineArray[RankState, tp], mut pools: HeapMoveArray[P],
) -> PhaseTiming:
    comptime immut = ImmutOrigin(MutAnyOrigin)
    var src = RankBuffers[DType.bfloat16, tp, immut](count=BENCH_SEQ * C.HIDDEN)
    var dst = RankBuffers[DType.bfloat16, tp, MutAnyOrigin](count=BENCH_SEQ * C.HIDDEN)
    for r in range(tp):
        src.ptrs[r] = states[r].moe_partial.as_immutable()
        dst.ptrs[r] = states[r].moe_partial
    var t0 = Int(perf_counter_ns())
    dispatch_allreduce[BF16, tp](src, dst, pools)
    var worker = max_worker_ts[tp=tp](pools) - t0
    var t1 = Int(perf_counter_ns())
    return PhaseTiming(t1 - t0, worker)


def run_iteration[P: BurstThreadPool, //, tp: Int, num_local_experts: Int](
    states: InlineArray[RankState, tp], mut pools: HeapMoveArray[P],
) -> PhaseTotals:
    var total = empty_totals()
    var t0 = Int(perf_counter_ns())

    total.router = dispatch_router_sharded_phase[
        tp=tp, num_local_experts=num_local_experts](states, pools)
    total.merge = merge_step[tp=tp](states, BENCH_SEQ)
    total.norm = dispatch_norm_phase[tp=tp](states, pools)
    total.schedule = schedule_step[
        tp=tp, num_local_experts=num_local_experts](states, BENCH_SEQ)
    total.phase1 = dispatch_phase1_step[
        tp=tp, num_local_experts=num_local_experts](states, pools)
    total.phase2 = dispatch_phase2_step[
        tp=tp, num_local_experts=num_local_experts](states, pools, BENCH_SEQ)
    total.allreduce = dispatch_allreduce_phase[tp=tp](states, pools)

    total.total_wall_ns = Int(perf_counter_ns()) - t0
    keep(states[0].moe_partial[0])
    return total


def print_totals[tp: Int](totals: PhaseTotals):
    print("\n=== Best phase timings (seq=" + String(BENCH_SEQ)
        + ", tp=" + String(tp) + ") ===")
    print_phase("router_sharded", totals.router.wall_ns, totals.router.worker_ns)
    print_phase("merge_softmax",  totals.merge.wall_ns, totals.merge.worker_ns)
    print_phase("pre_moe_norm", totals.norm.wall_ns, totals.norm.worker_ns,
        BENCH_SEQ * C.HIDDEN * 4 * tp)
    print_phase("build_schedule", totals.schedule.wall_ns, totals.schedule.worker_ns)
    print_phase("phase1_gate_up", totals.phase1.wall_ns, totals.phase1.worker_ns)
    print_phase("phase2_down",    totals.phase2.wall_ns, totals.phase2.worker_ns)
    print_phase("output_allreduce", totals.allreduce.wall_ns, totals.allreduce.worker_ns,
        BENCH_SEQ * C.HIDDEN * 2 * tp * 2)
    print("  total wall=" + fmt_ns(totals.total_wall_ns))


def alloc_states[tp: Int, num_local_experts: Int](
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
    capacities: InlineArray[Int, tp],
) -> InlineArray[RankState, tp]:
    var states = InlineArray[RankState, tp](uninitialized=True)
    for r in range(tp):
        var x = arena_alloc[DType.bfloat16](arenas[r], BENCH_SEQ * C.HIDDEN)
        var router_proj = arena_alloc[DType.bfloat16](
            arenas[r], num_local_experts * C.HIDDEN)
        var router_scale = arena_alloc[DType.bfloat16](arenas[r], C.HIDDEN)
        var pes = arena_alloc[DType.bfloat16](arenas[r], C.NUM_EXPERTS)
        var norm_w = arena_alloc[DType.bfloat16](arenas[r], C.HIDDEN)
        var route_idx = arena_alloc[DType.int32](arenas[r], BENCH_SEQ * C.TOP_K)
        var route_w = arena_alloc[DType.float32](arenas[r], BENCH_SEQ * C.TOP_K)
        var router_scaled = arena_alloc[DType.float32](
            arenas[r], capacities[r] * C.HIDDEN)
        var x_normed = arena_alloc[DType.bfloat16](arenas[r], BENCH_SEQ * C.HIDDEN)
        var partial = arena_alloc[DType.bfloat16](arenas[r], BENCH_SEQ * C.HIDDEN)
        var cands = arena_alloc_t[RouterCandidate](arenas[r], BENCH_SEQ * C.TOP_K)
        var expert_offset = arena_alloc[DType.int32](
            arenas[r], num_local_experts + 1)
        var routes = arena_alloc_t[SparseRoute](arenas[r], BENCH_SEQ * C.TOP_K)
        var hidden_bucket = arena_alloc[DType.bfloat16](
            arenas[r], BENCH_SEQ * C.TOP_K * C.MOE_INTERMEDIATE)
        var moe_accum = arena_alloc[DType.float32](arenas[r], BENCH_SEQ * C.HIDDEN)
        var experts_gu = arena_alloc[DType.bfloat16](
            arenas[r], num_local_experts * C.MOE_GATE_UP_FUSED * C.HIDDEN)
        var experts_down = arena_alloc[DType.bfloat16](
            arenas[r], num_local_experts * C.HIDDEN * C.MOE_INTERMEDIATE)
        var gate_scratch = arena_alloc[DType.float32](
            arenas[r], capacities[r] * PHASE1_MR * 2 * PHASE1_TILE_J)

        fill_bf16(x, BENCH_SEQ * C.HIDDEN, 11 + r)
        fill_bf16(router_proj, num_local_experts * C.HIDDEN, 23 + r)
        fill_bf16_positive(router_scale, C.HIDDEN)
        fill_bf16_positive(pes, C.NUM_EXPERTS)
        fill_bf16_positive(norm_w, C.HIDDEN)
        fill_bf16(experts_gu, num_local_experts * C.MOE_GATE_UP_FUSED * C.HIDDEN, 41 + r)
        fill_bf16(experts_down, num_local_experts * C.HIDDEN * C.MOE_INTERMEDIATE, 59 + r)

        states[r] = RankState(
            x=x, router_proj=router_proj, router_scale=router_scale,
            per_expert_scale=pes, pre_ffn_norm_2=norm_w,
            route_idx=route_idx, route_w=route_w, router_scaled=router_scaled,
            x_normed=x_normed, moe_partial=partial,
            cands=cands, expert_offset=expert_offset, routes=routes,
            hidden_bucket=hidden_bucket, moe_accum=moe_accum,
            experts_gate_up=experts_gu, experts_down=experts_down,
            gate_scratch=gate_scratch,
        )
    return states


def run_all[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
):
    comptime assert C.NUM_EXPERTS % tp == 0, "tp must divide NUM_EXPERTS"
    comptime num_local_experts = C.NUM_EXPERTS // tp

    var capacities = InlineArray[Int, tp](uninitialized=True)
    for r in range(tp):
        capacities[r] = pools[r].get_capacity()

    var states = alloc_states[tp, num_local_experts](arenas, capacities)

    print("\nBenchmark shape:")
    print("  seq=" + String(BENCH_SEQ)
        + " hidden=" + String(C.HIDDEN)
        + " experts=" + String(C.NUM_EXPERTS)
        + " top_k=" + String(C.TOP_K)
        + " local_experts=" + String(num_local_experts))
    var expert_weight_bytes = (
        num_local_experts * C.MOE_GATE_UP_FUSED * C.HIDDEN * 2
        + num_local_experts * C.HIDDEN * C.MOE_INTERMEDIATE * 2)
    print("  expert weights per rank: " + fmt_mib(expert_weight_bytes) + " MiB")

    for _ in range(WARMUP):
        _ = run_iteration[tp=tp, num_local_experts=num_local_experts](states, pools)

    var best = empty_totals()
    for _ in range(TRIALS):
        var result = run_iteration[tp=tp, num_local_experts=num_local_experts](states, pools)
        if better(result, best):
            best = result

    print_totals[tp](best)


def main():
    var numa = NumaInfo()
    var topo = numa.plan_topology(numa.num_nodes)
    var tp = numa.num_nodes

    print("Gemma4 MoE prefill phase benchmark (minimax-style)")
    print(String(tp) + " NUMA node(s), "
        + String(len(numa.isolated_cpus)) + " isolated cpus")

    if not (tp == 1 or tp == 2 or tp == 4 or tp == 8):
        print("unsupported tp=" + String(tp))
        return

    comptime ARENA_BYTES = 3 * 1024 * 1024 * 1024
    var arenas = HeapMoveArray[NumaArena[alignment=ALIGNMENT]](tp)
    for i in range(tp):
        arenas.push(NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    if numa.has_isolation():
        print("mode: isolated")
        var pools = HeapMoveArray[IsolatedBurstPool[]](tp)
        for i in range(tp):
            pools.push(IsolatedBurstPool[].for_topology(numa, topo[i]))
            print("  node " + String(topo[i]) + ": "
                + String(pools[i].get_capacity()) + " workers")
        if tp == 1:
            run_all[tp=1](pools, arenas)
        elif tp == 2:
            run_all[tp=2](pools, arenas)
        elif tp == 4:
            run_all[tp=4](pools, arenas)
        elif tp == 8:
            run_all[tp=8](pools, arenas)
    else:
        print("mode: spin-backoff")
        var pools = HeapMoveArray[BurstPool[]](tp)
        for i in range(tp):
            pools.push(BurstPool[].for_topology(numa, topo[i]))
            print("  node " + String(topo[i]) + ": "
                + String(pools[i].get_capacity()) + " workers")
        if tp == 1:
            run_all[tp=1](pools, arenas)
        elif tp == 2:
            run_all[tp=2](pools, arenas)
        elif tp == 4:
            run_all[tp=4](pools, arenas)
        elif tp == 8:
            run_all[tp=8](pools, arenas)
