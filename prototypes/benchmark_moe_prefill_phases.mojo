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
    RankBuffers, NumaPointerArray, NumaTypedPointerArray, join_all,
)
from kernels.reductions import dispatch_allreduce
from kernels.rmsnorm import dispatch_rms_norm
from kernels.moe_router import (
    RouterCandidate, RouterCandidatePtr, SparseRoute, SparseRoutePtr,
    dispatch_router_sharded, merge_router_candidates, build_expert_schedules,
)
from kernels.moe_experts import (
    Phase1GateUpKernel, Phase2DownKernel,
    dispatch_phase1_gate_up, dispatch_phase2_down,
    PHASE1_TILE_J, PHASE1_MR,
)
from modeling.model_spec import BF16
from modeling.gemma4_common import Gemma4BaseConfig
from simd_math.ops import sqrt


comptime C = Gemma4BaseConfig
comptime ALIGNMENT = 64
comptime WARMUP = 1
comptime TRIALS = 3
comptime BENCH_SEQ = C.MAX_SEQ_LEN

comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
comptime I32Ptr = UnsafePointer[Scalar[DType.int32], MutAnyOrigin]
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


def numa_bf16[tp: Int](ptr: BF16Ptr, bases: InlineArray[Int, tp])
        -> NumaPointerArray[DType.bfloat16, tp]:
    return NumaPointerArray[DType.bfloat16, tp](ptr, bases)


def numa_f32[tp: Int](ptr: F32Ptr, bases: InlineArray[Int, tp])
        -> NumaPointerArray[DType.float32, tp]:
    return NumaPointerArray[DType.float32, tp](ptr, bases)


def numa_i32[tp: Int](ptr: I32Ptr, bases: InlineArray[Int, tp])
        -> NumaPointerArray[DType.int32, tp]:
    return NumaPointerArray[DType.int32, tp](ptr, bases)


def numa_cands[tp: Int](ptr: RouterCandidatePtr, bases: InlineArray[Int, tp])
        -> NumaTypedPointerArray[RouterCandidate, tp]:
    return NumaTypedPointerArray[RouterCandidate, tp](ptr, bases)


def numa_routes[tp: Int](ptr: SparseRoutePtr, bases: InlineArray[Int, tp])
        -> NumaTypedPointerArray[SparseRoute, tp]:
    return NumaTypedPointerArray[SparseRoute, tp](ptr, bases)


def dispatch_router_sharded_phase[
    P: BurstThreadPool, //, tp: Int, num_local_experts: Int,
](
    states: InlineArray[RankState, tp], arena_bases: InlineArray[Int, tp],
    mut pools: HeapMoveArray[P],
) -> PhaseTiming:
    comptime rms_eps = Scalar[DType.float32](C.RMS_NORM_EPS)
    var t0 = Int(perf_counter_ns())
    dispatch_router_sharded[
        hidden=C.HIDDEN, experts_per_rank=num_local_experts,
        top_k=C.TOP_K, tp=tp, rms_eps=rms_eps,
    ](
        numa_bf16[tp](states[0].x, arena_bases),
        numa_bf16[tp](states[0].router_proj, arena_bases),
        numa_bf16[tp](states[0].router_scale, arena_bases),
        numa_f32[tp](states[0].router_scaled, arena_bases),
        numa_cands[tp](states[0].cands, arena_bases),
        BENCH_SEQ, pools,
    )
    var worker = max_worker_ts[tp=tp](pools) - t0
    var t1 = Int(perf_counter_ns())
    return PhaseTiming(t1 - t0, worker)


def merge_step[tp: Int](
    states: InlineArray[RankState, tp], arena_bases: InlineArray[Int, tp],
    seq_len: Int,
) -> PhaseTiming:
    var t0 = Int(perf_counter_ns())
    merge_router_candidates[tp, C.TOP_K](
        numa_cands[tp](states[0].cands, arena_bases),
        states[0].per_expert_scale,
        numa_i32[tp](states[0].route_idx, arena_bases),
        numa_f32[tp](states[0].route_w, arena_bases),
        seq_len,
    )
    var t1 = Int(perf_counter_ns())
    return PhaseTiming(t1 - t0, t1 - t0)


def dispatch_norm_phase[P: BurstThreadPool, //, tp: Int](
    states: InlineArray[RankState, tp], arena_bases: InlineArray[Int, tp],
    mut pools: HeapMoveArray[P],
) -> PhaseTiming:
    comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
    comptime n_eps = C.HIDDEN * C.RMS_NORM_EPS
    var t0 = Int(perf_counter_ns())
    dispatch_rms_norm[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps, tp=tp,
    ](
        numa_bf16[tp](states[0].x, arena_bases),
        numa_bf16[tp](states[0].x_normed, arena_bases),
        numa_bf16[tp](states[0].pre_ffn_norm_2, arena_bases),
        BENCH_SEQ, pools,
    )
    var worker = max_worker_ts[tp=tp](pools) - t0
    var t1 = Int(perf_counter_ns())
    return PhaseTiming(t1 - t0, worker)


def schedule_step[tp: Int, num_local_experts: Int](
    states: InlineArray[RankState, tp], arena_bases: InlineArray[Int, tp],
    seq_len: Int,
) -> PhaseTiming:
    var t0 = Int(perf_counter_ns())
    build_expert_schedules[tp, num_local_experts, C.TOP_K](
        numa_i32[tp](states[0].route_idx, arena_bases),
        numa_f32[tp](states[0].route_w, arena_bases),
        numa_i32[tp](states[0].expert_offset, arena_bases),
        numa_routes[tp](states[0].routes, arena_bases),
        seq_len,
    )
    var t1 = Int(perf_counter_ns())
    return PhaseTiming(t1 - t0, t1 - t0)


def dispatch_phase1_step[
    P: BurstThreadPool, //, tp: Int, num_local_experts: Int,
](
    states: InlineArray[RankState, tp], arena_bases: InlineArray[Int, tp],
    mut pools: HeapMoveArray[P],
) -> PhaseTiming:
    var t0 = Int(perf_counter_ns())
    dispatch_phase1_gate_up[
        hidden=C.HIDDEN, gate_up_fused=C.MOE_GATE_UP_FUSED,
        intermediate=C.MOE_INTERMEDIATE,
        experts_per_rank=num_local_experts, tp=tp,
    ](
        numa_bf16[tp](states[0].x_normed, arena_bases),
        numa_i32[tp](states[0].expert_offset, arena_bases),
        numa_routes[tp](states[0].routes, arena_bases),
        numa_bf16[tp](states[0].experts_gate_up, arena_bases),
        numa_f32[tp](states[0].gate_scratch, arena_bases),
        numa_bf16[tp](states[0].hidden_bucket, arena_bases),
        pools,
    )
    var worker = max_worker_ts[tp=tp](pools) - t0
    var t1 = Int(perf_counter_ns())
    return PhaseTiming(t1 - t0, worker)


def dispatch_phase2_step[
    P: BurstThreadPool, //, tp: Int, num_local_experts: Int,
](
    states: InlineArray[RankState, tp], arena_bases: InlineArray[Int, tp],
    mut pools: HeapMoveArray[P], seq_len: Int,
) -> PhaseTiming:
    var t0 = Int(perf_counter_ns())
    dispatch_phase2_down[
        hidden=C.HIDDEN, intermediate=C.MOE_INTERMEDIATE,
        experts_per_rank=num_local_experts, tp=tp,
    ](
        numa_i32[tp](states[0].expert_offset, arena_bases),
        numa_routes[tp](states[0].routes, arena_bases),
        numa_bf16[tp](states[0].hidden_bucket, arena_bases),
        numa_bf16[tp](states[0].experts_down, arena_bases),
        numa_f32[tp](states[0].moe_accum, arena_bases),
        numa_bf16[tp](states[0].moe_partial, arena_bases),
        seq_len, pools,
    )
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
    states: InlineArray[RankState, tp], arena_bases: InlineArray[Int, tp],
    mut pools: HeapMoveArray[P],
) -> PhaseTotals:
    var total = empty_totals()
    var t0 = Int(perf_counter_ns())

    total.router = dispatch_router_sharded_phase[
        tp=tp, num_local_experts=num_local_experts](states, arena_bases, pools)
    total.merge = merge_step[tp=tp](states, arena_bases, BENCH_SEQ)
    total.norm = dispatch_norm_phase[tp=tp](states, arena_bases, pools)
    total.schedule = schedule_step[
        tp=tp, num_local_experts=num_local_experts](states, arena_bases, BENCH_SEQ)
    total.phase1 = dispatch_phase1_step[
        tp=tp, num_local_experts=num_local_experts](states, arena_bases, pools)
    total.phase2 = dispatch_phase2_step[
        tp=tp, num_local_experts=num_local_experts](
        states, arena_bases, pools, BENCH_SEQ)
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
    var arena_bases = InlineArray[Int, tp](uninitialized=True)
    for r in range(tp):
        capacities[r] = pools[r].get_capacity()
        arena_bases[r] = Int(arenas[r].base.value())

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
        _ = run_iteration[tp=tp, num_local_experts=num_local_experts](
            states, arena_bases, pools)

    var best = empty_totals()
    for _ in range(TRIALS):
        var result = run_iteration[tp=tp, num_local_experts=num_local_experts](
            states, arena_bases, pools)
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
