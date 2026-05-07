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
from prototypes.expert_kernels import ActiveExpertKernel, EXPERT_TOK_TILE
from prototypes.moe_phase_kernels import (
    arena_alloc, fill_bf16, fill_bf16_positive,
    RouterStreamKernel, RouteGatherConfig, RouteGatherDstKernel,
    FillBF16Kernel, RmsNormBenchKernel,
    ExpertCountSlotKernel, PrefixKernel, PlaceSlotKernel,
    RankState,
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
    var gather: PhaseTiming
    var norm: PhaseTiming
    var zero: PhaseTiming
    var bucket: PhaseTiming
    var expert: PhaseTiming
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
def add_timing(mut dst: PhaseTiming, src: PhaseTiming):
    dst.wall_ns += src.wall_ns
    dst.worker_ns += src.worker_ns


@always_inline
def better(candidate: PhaseTotals, incumbent: PhaseTotals) -> Bool:
    return incumbent.total_wall_ns == 0 or candidate.total_wall_ns < incumbent.total_wall_ns



def dispatch_router_phase[P: BurstThreadPool, //, tp: Int, num_local_experts: Int](
    states: InlineArray[RankState, tp], mut pools: HeapMoveArray[P],
) -> PhaseTiming:
    var t0 = Int(perf_counter_ns())
    for r in range(tp):
        var cap = pools[r].get_capacity()
        var buf = DispatchBuffer[
            RouterStreamKernel[C.HIDDEN, C.NUM_EXPERTS, C.TOP_K]
        ]()
        var stripe = worker_range(BENCH_SEQ, tp, r)
        for w in range(cap):
            var wr = worker_range(stripe[1] - stripe[0], cap, w, stripe[0])
            buf.slot()[] = RouterStreamKernel[
                C.HIDDEN, C.NUM_EXPERTS, C.TOP_K,
            ](
                states[r].x, states[r].router_proj, states[r].router_scale,
                states[r].per_expert_scale, states[r].router_scaled,
                states[r].route_idx, states[r].route_w, w, wr[0], wr[1],
            )
        buf.dispatch(pools[r])
    join_all[tp](pools)
    var worker = max_worker_ts[tp=tp](pools) - t0
    var t1 = Int(perf_counter_ns())
    return PhaseTiming(t1 - t0, worker)


def dispatch_route_gather_phase[P: BurstThreadPool, //, tp: Int](
    states: InlineArray[RankState, tp], mut pools: HeapMoveArray[P],
) -> PhaseTiming:
    var src_idx = InlineArray[I32Ptr, tp](uninitialized=True)
    var src_w = InlineArray[F32Ptr, tp](uninitialized=True)
    for r in range(tp):
        src_idx[r] = states[r].route_idx
        src_w[r] = states[r].route_w
    var cfg = RouteGatherConfig[tp](src_idx, src_w)
    var config = UnsafePointer(to=cfg).as_immutable()
    comptime cfg_ro = ImmutOrigin(origin_of(cfg))

    var total_records = BENCH_SEQ * C.TOP_K
    var t0 = Int(perf_counter_ns())
    for dst in range(tp):
        var cap = pools[dst].get_capacity()
        var buf = DispatchBuffer[RouteGatherDstKernel[tp, C.TOP_K, cfg_ro]]()
        var nw = recommended_workers(total_records * 8, cap)
        for w in range(nw):
            var wr = worker_range(total_records, nw, w)
            buf.slot()[] = RouteGatherDstKernel[tp, C.TOP_K, cfg_ro](
                config, states[dst].route_idx, states[dst].route_w,
                BENCH_SEQ, wr[0], wr[1],
            )
        buf.dispatch(pools[dst])
    join_all[tp](pools)
    var worker = max_worker_ts[tp=tp](pools) - t0
    var t1 = Int(perf_counter_ns())
    return PhaseTiming(t1 - t0, worker)


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
 

def dispatch_zero_phase[P: BurstThreadPool, //, tp: Int](
    states: InlineArray[RankState, tp], mut pools: HeapMoveArray[P],
) -> PhaseTiming:
    var elems = BENCH_SEQ * C.HIDDEN
    var t0 = Int(perf_counter_ns())
    for r in range(tp):
        var buf = DispatchBuffer[FillBF16Kernel]()
        var nw = recommended_workers(elems * 2, pools[r].get_capacity())
        for w in range(nw):
            var wr = worker_range(elems, nw, w)
            buf.slot()[] = FillBF16Kernel(states[r].moe_partial, wr[0], wr[1])
        buf.dispatch(pools[r])
    join_all[tp](pools)
    var worker = max_worker_ts[tp=tp](pools) - t0
    var t1 = Int(perf_counter_ns())
    return PhaseTiming(t1 - t0, worker)


def dispatch_bucket_slot_phase[
    P: BurstThreadPool, //, tp: Int, num_local_experts: Int,
](
    states: InlineArray[RankState, tp], mut pools: HeapMoveArray[P], slot: Int,
) -> PhaseTiming:
    var t0 = Int(perf_counter_ns())
    for r in range(tp):
        var cap = pools[r].get_capacity()
        var buf = DispatchBuffer[
            ExpertCountSlotKernel[C.TOP_K, num_local_experts]
        ]()
        for w in range(cap):
            var wr = worker_range(BENCH_SEQ, cap, w)
            buf.slot()[] = ExpertCountSlotKernel[C.TOP_K, num_local_experts](
                states[r].route_idx, states[r].counts_per_worker,
                r, w, slot, wr[0], wr[1],
            )
        buf.dispatch(pools[r])
    join_all[tp](pools)

    for r in range(tp):
        var buf = DispatchBuffer[PrefixKernel[num_local_experts]]()
        buf.slot()[] = PrefixKernel[num_local_experts](
            states[r].counts_per_worker, states[r].expert_offset,
            states[r].worker_cursor, pools[r].get_capacity(), 0, 1,
        )
        buf.dispatch(pools[r])
    join_all[tp](pools)

    for r in range(tp):
        var cap = pools[r].get_capacity()
        var buf = DispatchBuffer[PlaceSlotKernel[C.TOP_K, num_local_experts]]()
        for w in range(cap):
            var wr = worker_range(BENCH_SEQ, cap, w)
            buf.slot()[] = PlaceSlotKernel[C.TOP_K, num_local_experts](
                states[r].route_idx, states[r].route_w,
                states[r].worker_cursor, states[r].bucket_token_idx,
                states[r].bucket_weight, r, w, slot, wr[0], wr[1],
            )
        buf.dispatch(pools[r])
    join_all[tp](pools)

    var worker = max_worker_ts[tp=tp](pools) - t0
    var t1 = Int(perf_counter_ns())
    return PhaseTiming(t1 - t0, worker)


def dispatch_expert_slot_phase[
    P: BurstThreadPool, //, tp: Int, num_local_experts: Int,
](
    states: InlineArray[RankState, tp], mut pools: HeapMoveArray[P],
) -> PhaseTiming:
    var t0 = Int(perf_counter_ns())
    for r in range(tp):
        var cap = pools[r].get_capacity()
        var buf = DispatchBuffer[
            ActiveExpertKernel[
                C.HIDDEN, C.MOE_GATE_UP_FUSED, C.MOE_INTERMEDIATE,
                num_local_experts,
            ],
        ]()
        for w in range(cap):
            var wr = worker_range(num_local_experts, cap, w)
            buf.slot()[] = ActiveExpertKernel[
                C.HIDDEN, C.MOE_GATE_UP_FUSED, C.MOE_INTERMEDIATE,
                num_local_experts,
            ](
                states[r].x_normed, states[r].expert_offset,
                states[r].bucket_token_idx, states[r].bucket_weight,
                states[r].experts_gate_up, states[r].experts_down,
                states[r].gate_scratch, states[r].hidden_scratch,
                states[r].hidden_scratch_bf16,
                states[r].moe_partial, w, wr[0], wr[1],
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

    total.router = dispatch_router_phase[tp=tp, num_local_experts=num_local_experts](states, pools)
    total.gather = dispatch_route_gather_phase[tp=tp](states, pools)
    total.norm = dispatch_norm_phase[tp=tp](states, pools)
    total.zero = dispatch_zero_phase[tp=tp](states, pools)

    for slot in range(C.TOP_K):
        var bt = dispatch_bucket_slot_phase[tp=tp, num_local_experts=num_local_experts](
            states, pools, slot)
        add_timing(total.bucket, bt)
        var et = dispatch_expert_slot_phase[tp=tp, num_local_experts=num_local_experts](
            states, pools)
        add_timing(total.expert, et)

    total.allreduce = dispatch_allreduce_phase[tp=tp](states, pools)
    total.total_wall_ns = Int(perf_counter_ns()) - t0
    keep(states[0].moe_partial[0])
    return total


def print_totals[tp: Int](totals: PhaseTotals):
    print("\n=== Best phase timings (seq=" + String(BENCH_SEQ)
        + ", tp=" + String(tp) + ") ===")
    print_phase("router_stream_topk", totals.router.wall_ns, totals.router.worker_ns)
    print_phase("route_metadata_gather", totals.gather.wall_ns, totals.gather.worker_ns,
        BENCH_SEQ * C.TOP_K * 8 * tp)
    print_phase("pre_moe_norm", totals.norm.wall_ns, totals.norm.worker_ns,
        BENCH_SEQ * C.HIDDEN * 4 * tp)
    print_phase("zero_local_partial", totals.zero.wall_ns, totals.zero.worker_ns,
        BENCH_SEQ * C.HIDDEN * 2 * tp)
    print_phase("bucketize_all_slots", totals.bucket.wall_ns, totals.bucket.worker_ns,
        BENCH_SEQ * C.TOP_K * 8 * tp * 2)
    print_phase("expert_runner_all_slots", totals.expert.wall_ns, totals.expert.worker_ns)
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
        var router_proj = arena_alloc[DType.bfloat16](arenas[r], C.NUM_EXPERTS * C.HIDDEN)
        var router_scale = arena_alloc[DType.bfloat16](arenas[r], C.HIDDEN)
        var pes = arena_alloc[DType.bfloat16](arenas[r], C.NUM_EXPERTS)
        var norm_w = arena_alloc[DType.bfloat16](arenas[r], C.HIDDEN)
        var route_idx = arena_alloc[DType.int32](arenas[r], BENCH_SEQ * C.TOP_K)
        var route_w = arena_alloc[DType.float32](arenas[r], BENCH_SEQ * C.TOP_K)
        var router_scaled = arena_alloc[DType.float32](arenas[r], capacities[r] * C.HIDDEN)
        var x_normed = arena_alloc[DType.bfloat16](arenas[r], BENCH_SEQ * C.HIDDEN)
        var partial = arena_alloc[DType.bfloat16](arenas[r], BENCH_SEQ * C.HIDDEN)
        var counts = arena_alloc[DType.int32](arenas[r], capacities[r] * num_local_experts)
        var offsets = arena_alloc[DType.int32](arenas[r], num_local_experts + 1)
        var cursors = arena_alloc[DType.int32](arenas[r], capacities[r] * num_local_experts)
        var bucket_idx = arena_alloc[DType.int32](arenas[r], BENCH_SEQ)
        var bucket_w = arena_alloc[DType.float32](arenas[r], BENCH_SEQ)
        var experts_gu = arena_alloc[DType.bfloat16](
            arenas[r], num_local_experts * C.MOE_GATE_UP_FUSED * C.HIDDEN)
        var experts_down = arena_alloc[DType.bfloat16](
            arenas[r], num_local_experts * C.HIDDEN * C.MOE_INTERMEDIATE)
        var gate_scratch = arena_alloc[DType.float32](
            arenas[r], capacities[r] * EXPERT_TOK_TILE * C.MOE_GATE_UP_FUSED)
        var hidden_scratch = arena_alloc[DType.float32](
            arenas[r], capacities[r] * EXPERT_TOK_TILE * C.MOE_INTERMEDIATE)
        var hidden_scratch_bf16 = arena_alloc[DType.bfloat16](
            arenas[r], capacities[r] * EXPERT_TOK_TILE * C.MOE_INTERMEDIATE)

        fill_bf16(x, BENCH_SEQ * C.HIDDEN, 11 + r)
        fill_bf16(router_proj, C.NUM_EXPERTS * C.HIDDEN, 23 + r)
        fill_bf16_positive(router_scale, C.HIDDEN)
        fill_bf16_positive(pes, C.NUM_EXPERTS)
        fill_bf16_positive(norm_w, C.HIDDEN)
        fill_bf16(experts_gu, num_local_experts * C.MOE_GATE_UP_FUSED * C.HIDDEN, 41 + r)
        fill_bf16(experts_down, num_local_experts * C.HIDDEN * C.MOE_INTERMEDIATE, 59 + r)

        states[r] = RankState(
            x, router_proj, router_scale, pes, norm_w,
            route_idx, route_w, router_scaled, x_normed, partial,
            counts, offsets, cursors, bucket_idx, bucket_w,
            experts_gu, experts_down, gate_scratch, hidden_scratch,
            hidden_scratch_bf16,
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

    print("Gemma4 MoE prefill phase benchmark")
    print(String(tp) + " NUMA node(s), "
        + String(len(numa.isolated_cpus)) + " isolated cpus")

    if not (tp == 1 or tp == 2 or tp == 4 or tp == 8):
        print("unsupported tp=" + String(tp))
        return

    # Full Gemma4 local expert weights dominate the arena requirement.
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
