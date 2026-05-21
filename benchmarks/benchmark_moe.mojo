from std.collections import InlineArray
from std.memory import UnsafePointer
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from notstdcollections import HeapMoveArray
from kernels.helpers import Binding, ArenaBases
from kernels.moe_router import SparseRoute
from kernels.moe_experts import dispatch_phase1_gate_up, dispatch_phase2_down
from modeling.gemma4_common import Gemma4BaseConfig
from benchmarks.bench_harness import (
    SampleBuffer, compute_stats, print_row, max_last_ts, now_ns,
    DEFAULT_SAMPLES,
)


comptime HIDDEN = Gemma4BaseConfig.HIDDEN
comptime INTERMEDIATE = Gemma4BaseConfig.MOE_INTERMEDIATE
comptime GATE_UP_FUSED = Gemma4BaseConfig.MOE_GATE_UP_FUSED
comptime NUM_EXPERTS = Gemma4BaseConfig.NUM_EXPERTS
comptime TOP_K = Gemma4BaseConfig.TOP_K

comptime ALIGNMENT = 64
comptime WARMUP = 30
comptime SAMPLES = DEFAULT_SAMPLES

comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime F32Ptr  = UnsafePointer[Float32,  MutAnyOrigin]
comptime I32Ptr  = UnsafePointer[Int32,    MutAnyOrigin]
comptime SparseRoutePtr = UnsafePointer[SparseRoute, MutAnyOrigin]

comptime NUM_SEQ_SIZES = 5
comptime PHASE1_SCRATCH_PER_WORKER = 4 * 2 * 64
comptime MAX_WORKERS = 256


def arena_alloc[T: AnyType](
    mut arena: NumaArena[alignment=ALIGNMENT], count: Int,
) -> UnsafePointer[T, MutAnyOrigin]:
    var ptr = arena.alloc[T](count)
    if not ptr:
        print("arena alloc failed for", count, "elements")
        return UnsafePointer[T, MutAnyOrigin].unsafe_dangling()
    return ptr.value()


def arena_alloc_all[T: AnyType, tp: Int](
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]], count: Int,
) -> UnsafePointer[T, MutAnyOrigin]:
    var first = UnsafePointer[T, MutAnyOrigin].unsafe_dangling()
    for r in range(tp):
        var ptr = arena_alloc[T](arenas[r], count)
        if r == 0:
            first = ptr
    return first


def arena_bases[tp: Int](
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
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


def build_uniform_routing[experts_per_rank: Int, tp: Int](
    seq_len: Int,
    expert_offset: Binding[Int32, tp],
    routes: Binding[SparseRoute, tp],
):
    """Round-robin routing: token t picks experts t*TOP_K..t*TOP_K+TOP_K-1
    mod NUM_EXPERTS. Yields a balanced load when seq_len*TOP_K is a
    multiple of NUM_EXPERTS, and exercises the same MR=4 fast path
    + tail mix that real top-k routing produces."""
    var w = Float32(1.0) / Float32(TOP_K)
    for r in range(tp):
        var first = r * experts_per_rank
        var last = first + experts_per_rank
        var ofs_r = expert_offset[r]
        var routes_r = routes[r]

        var counts = InlineArray[Int, experts_per_rank](fill=0)
        for tok in range(seq_len):
            for k in range(TOP_K):
                var e = (tok * TOP_K + k) % NUM_EXPERTS
                if e >= first and e < last:
                    counts[e - first] += 1

        var running = Int32(0)
        var write_ofs = InlineArray[Int, experts_per_rank](fill=0)
        for e in range(experts_per_rank):
            ofs_r[e] = running
            write_ofs[e] = Int(running)
            running += Int32(counts[e])
        ofs_r[experts_per_rank] = running

        for tok in range(seq_len):
            for k in range(TOP_K):
                var e = (tok * TOP_K + k) % NUM_EXPERTS
                if e >= first and e < last:
                    var local = e - first
                    var pos = write_ofs[local]
                    routes_r[pos] = SparseRoute(Int32(tok), w)
                    write_ofs[local] = pos + 1


def phase1_flops(seq_len: Int) -> Int:
    """Each (token, expert) pair does a gate dot AND an up dot, each
    INTERMEDIATE * HIDDEN MACs => 4 * INTERMEDIATE * HIDDEN FLOPS per
    pair. Routing produces seq_len * TOP_K pairs in total across ranks."""
    return seq_len * TOP_K * 4 * INTERMEDIATE * HIDDEN


def phase2_flops(seq_len: Int) -> Int:
    """Each (token, expert) pair does a down dot of HIDDEN * INTERMEDIATE
    MACs => 2 * HIDDEN * INTERMEDIATE FLOPS per pair."""
    return seq_len * TOP_K * 2 * HIDDEN * INTERMEDIATE


def section_phase1[
    P: BurstThreadPool, //,
    tp: Int, experts_per_rank: Int,
](
    mut pools: HeapMoveArray[P],
    mut samples: SampleBuffer,
    seq_len: Int,
    x_normed: Binding[BFloat16, tp],
    expert_offset: Binding[Int32, tp],
    routes: Binding[SparseRoute, tp],
    experts_gate_up: Binding[BFloat16, tp],
    gate_scratch: Binding[Float32, tp],
    hidden_bucket: Binding[BFloat16, tp],
):
    for _ in range(WARMUP):
        dispatch_phase1_gate_up[
            hidden=HIDDEN, gate_up_fused=GATE_UP_FUSED,
            intermediate=INTERMEDIATE,
            experts_per_rank=experts_per_rank, tp=tp,
        ](x_normed, expert_offset, routes,
          experts_gate_up, gate_scratch, hidden_bucket, pools)

    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        dispatch_phase1_gate_up[
            hidden=HIDDEN, gate_up_fused=GATE_UP_FUSED,
            intermediate=INTERMEDIATE,
            experts_per_rank=experts_per_rank, tp=tp,
        ](x_normed, expert_offset, routes,
          experts_gate_up, gate_scratch, hidden_bucket, pools)
        var t1 = now_ns()
        var t_done = max_last_ts[tp=tp](pools)
        samples.push(t_done - t0, t1 - t0)
    keep(hidden_bucket[0][0])

    var ks = compute_stats(samples.kernel_ns, samples.n)
    var ws = compute_stats(samples.wall_ns, samples.n)
    print_row(String(t"phase1 seq={seq_len}"), ks, ws, 0)


def section_phase2[
    P: BurstThreadPool, //,
    tp: Int, experts_per_rank: Int,
](
    mut pools: HeapMoveArray[P],
    mut samples: SampleBuffer,
    seq_len: Int,
    expert_offset: Binding[Int32, tp],
    routes: Binding[SparseRoute, tp],
    hidden_bucket: Binding[BFloat16, tp],
    experts_down: Binding[BFloat16, tp],
    moe_accum: Binding[Float32, tp],
    moe_out: Binding[BFloat16, tp],
):
    for _ in range(WARMUP):
        dispatch_phase2_down[
            hidden=HIDDEN, intermediate=INTERMEDIATE,
            experts_per_rank=experts_per_rank, tp=tp,
        ](expert_offset, routes, hidden_bucket,
          experts_down, moe_accum, moe_out, seq_len, pools)

    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        dispatch_phase2_down[
            hidden=HIDDEN, intermediate=INTERMEDIATE,
            experts_per_rank=experts_per_rank, tp=tp,
        ](expert_offset, routes, hidden_bucket,
          experts_down, moe_accum, moe_out, seq_len, pools)
        var t1 = now_ns()
        var t_done = max_last_ts[tp=tp](pools)
        samples.push(t_done - t0, t1 - t0)
    keep(moe_out[0][0])

    var ks = compute_stats(samples.kernel_ns, samples.n)
    var ws = compute_stats(samples.wall_ns, samples.n)
    print_row(String(t"phase2 seq={seq_len}"), ks, ws, 0)


def section_combined[
    P: BurstThreadPool, //,
    tp: Int, experts_per_rank: Int,
](
    mut pools: HeapMoveArray[P],
    mut samples: SampleBuffer,
    seq_len: Int,
    x_normed: Binding[BFloat16, tp],
    expert_offset: Binding[Int32, tp],
    routes: Binding[SparseRoute, tp],
    experts_gate_up: Binding[BFloat16, tp],
    experts_down: Binding[BFloat16, tp],
    gate_scratch: Binding[Float32, tp],
    hidden_bucket: Binding[BFloat16, tp],
    moe_accum: Binding[Float32, tp],
    moe_out: Binding[BFloat16, tp],
):
    for _ in range(WARMUP):
        dispatch_phase1_gate_up[
            hidden=HIDDEN, gate_up_fused=GATE_UP_FUSED,
            intermediate=INTERMEDIATE,
            experts_per_rank=experts_per_rank, tp=tp,
        ](x_normed, expert_offset, routes,
          experts_gate_up, gate_scratch, hidden_bucket, pools)
        dispatch_phase2_down[
            hidden=HIDDEN, intermediate=INTERMEDIATE,
            experts_per_rank=experts_per_rank, tp=tp,
        ](expert_offset, routes, hidden_bucket,
          experts_down, moe_accum, moe_out, seq_len, pools)

    samples.clear()
    for _ in range(SAMPLES):
        var t0 = now_ns()
        dispatch_phase1_gate_up[
            hidden=HIDDEN, gate_up_fused=GATE_UP_FUSED,
            intermediate=INTERMEDIATE,
            experts_per_rank=experts_per_rank, tp=tp,
        ](x_normed, expert_offset, routes,
          experts_gate_up, gate_scratch, hidden_bucket, pools)
        dispatch_phase2_down[
            hidden=HIDDEN, intermediate=INTERMEDIATE,
            experts_per_rank=experts_per_rank, tp=tp,
        ](expert_offset, routes, hidden_bucket,
          experts_down, moe_accum, moe_out, seq_len, pools)
        var t1 = now_ns()
        var t_done = max_last_ts[tp=tp](pools)
        samples.push(t_done - t0, t1 - t0)
    keep(moe_out[0][0])

    var ks = compute_stats(samples.kernel_ns, samples.n)
    var ws = compute_stats(samples.wall_ns, samples.n)
    print_row(String(t"p1+p2 seq={seq_len}"), ks, ws, 0)


def run_all[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
):
    comptime experts_per_rank = NUM_EXPERTS // tp
    comptime MAX_SEQ = 1024

    var sizes = InlineArray[Int, NUM_SEQ_SIZES](fill=0)
    sizes[0] = 1
    sizes[1] = 16
    sizes[2] = 64
    sizes[3] = 256
    sizes[4] = MAX_SEQ

    var bases = arena_bases[tp](arenas)

    var x_normed_ptr = arena_alloc_all[BFloat16, tp](arenas, MAX_SEQ * HIDDEN)
    var expert_offset_ptr = arena_alloc_all[Int32, tp](
        arenas, experts_per_rank + 1)
    var routes_ptr = arena_alloc_all[SparseRoute, tp](arenas, MAX_SEQ * TOP_K)
    var experts_gate_up_ptr = arena_alloc_all[BFloat16, tp](
        arenas, experts_per_rank * GATE_UP_FUSED * HIDDEN)
    var experts_down_ptr = arena_alloc_all[BFloat16, tp](
        arenas, experts_per_rank * HIDDEN * INTERMEDIATE)
    var gate_scratch_ptr = arena_alloc_all[Float32, tp](
        arenas, MAX_WORKERS * PHASE1_SCRATCH_PER_WORKER)
    var hidden_bucket_ptr = arena_alloc_all[BFloat16, tp](
        arenas, MAX_SEQ * TOP_K * INTERMEDIATE)
    var moe_accum_ptr = arena_alloc_all[Float32, tp](arenas, MAX_SEQ * HIDDEN)
    var moe_out_ptr = arena_alloc_all[BFloat16, tp](arenas, MAX_SEQ * HIDDEN)

    var x_normed = Binding[BFloat16, tp](x_normed_ptr, bases)
    var expert_offset = Binding[Int32, tp](expert_offset_ptr, bases)
    var routes = Binding[SparseRoute, tp](routes_ptr, bases)
    var experts_gate_up = Binding[BFloat16, tp](experts_gate_up_ptr, bases)
    var experts_down = Binding[BFloat16, tp](experts_down_ptr, bases)
    var gate_scratch = Binding[Float32, tp](gate_scratch_ptr, bases)
    var hidden_bucket = Binding[BFloat16, tp](hidden_bucket_ptr, bases)
    var moe_accum = Binding[Float32, tp](moe_accum_ptr, bases)
    var moe_out = Binding[BFloat16, tp](moe_out_ptr, bases)

    fill_bf16_all[tp](x_normed, MAX_SEQ * HIDDEN)
    fill_bf16_all[tp](experts_gate_up,
        experts_per_rank * GATE_UP_FUSED * HIDDEN)
    fill_bf16_all[tp](experts_down,
        experts_per_rank * HIDDEN * INTERMEDIATE)

    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    print(
        t"hidden={HIDDEN} intermediate={INTERMEDIATE} "
        t"experts/rank={experts_per_rank} top_k={TOP_K}"
    )

    var samples = SampleBuffer(SAMPLES)

    print("\n=== Phase1 (gate+up over routed tokens) ===")
    for s in range(NUM_SEQ_SIZES):
        var seq = sizes[s]
        build_uniform_routing[experts_per_rank=experts_per_rank, tp=tp](
            seq, expert_offset, routes)
        section_phase1[tp=tp, experts_per_rank=experts_per_rank](
            pools, samples, seq, x_normed, expert_offset, routes,
            experts_gate_up, gate_scratch, hidden_bucket)

    print("\n=== Phase2 (down projection, scatter-accumulate) ===")
    for s in range(NUM_SEQ_SIZES):
        var seq = sizes[s]
        build_uniform_routing[experts_per_rank=experts_per_rank, tp=tp](
            seq, expert_offset, routes)
        section_phase2[tp=tp, experts_per_rank=experts_per_rank](
            pools, samples, seq, expert_offset, routes, hidden_bucket,
            experts_down, moe_accum, moe_out)

    print("\n=== Phase1 + Phase2 (no allreduce) ===")
    for s in range(NUM_SEQ_SIZES):
        var seq = sizes[s]
        build_uniform_routing[experts_per_rank=experts_per_rank, tp=tp](
            seq, expert_offset, routes)
        section_combined[tp=tp, experts_per_rank=experts_per_rank](
            pools, samples, seq, x_normed, expert_offset, routes,
            experts_gate_up, experts_down,
            gate_scratch, hidden_bucket, moe_accum, moe_out)


def arena_bytes_for(tp: Int) -> Int:
    var experts_per_rank = NUM_EXPERTS // tp
    var weights = (
        experts_per_rank * GATE_UP_FUSED * HIDDEN * 2
        + experts_per_rank * HIDDEN * INTERMEDIATE * 2)
    return weights + 256 * 1024 * 1024


def main():
    var topo = NumaTopology()
    var tp = len(topo)

    print("MoE kernel benchmark")
    var iso = len(topo.isolated_cpus)
    print(t"{tp} NUMA node(s), {iso} isolated cpus\n")

    var arenas = HeapMoveArray[NumaArena[alignment=ALIGNMENT]](tp)
    var bytes = arena_bytes_for(tp)
    for i in range(tp):
        arenas.push(NumaArena[alignment=ALIGNMENT](topo[i], bytes))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_moe_tp[
        P: BurstThreadPool, //, degree: Int,
    ](var selected_pools: HeapMoveArray[P]):
        run_all[tp=degree](selected_pools, arenas)

    with_topological_rank_dispatch[
        power_of_two_unrolling=3,
        dispatch=dispatch_moe_tp,
    ](topo, "mode: isolated", "mode: spin-backoff")
