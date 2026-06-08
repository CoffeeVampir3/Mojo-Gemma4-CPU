from std.memory import UnsafePointer
from std.os import abort
from std.time import perf_counter_ns

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.helpers import (
    DispatchBuffer, RangePartitionedKernel, join_all, tile_dispatch,
)


comptime I64Ptr = UnsafePointer[Int64, MutAnyOrigin]
comptime PAGE_SHIFT = 4
comptime PAGE_ROWS = 1 << PAGE_SHIFT
comptime ROW_MASK = PAGE_ROWS - 1
comptime POOL_PAGES = 64


struct RunMeta(Copyable, Movable):
    """One packed-buffer run: a contiguous span of tokens from one sequence.
    `base_rows` is the dynamically sized page map (ordinal -> physical base
    row), heap-backed, exactly the shape the paged-KV run table needs."""
    var buf_start: Int32
    var base_pos: Int32
    var base_rows: List[Int32]

    def __init__(out self, buf_start: Int, base_pos: Int):
        self.buf_start = Int32(buf_start)
        self.base_pos = Int32(base_pos)
        self.base_rows = List[Int32]()


struct StepKVMeta(Movable):
    """Orchestrator-owned step metadata. Kernels receive a pointer to this and
    read it in place; nothing here is copied into kernel values or sized at
    comptime."""
    var runs: List[RunMeta]

    def __init__(out self):
        self.runs = List[RunMeta]()


def page_base_row(run_id: Int, ordinal: Int) -> Int:
    return ((ordinal * 7 + run_id * 11 + 3) % POOL_PAGES) * PAGE_ROWS


def append_run(mut meta: StepKVMeta, buf_start: Int, base_pos: Int, length: Int):
    var run = RunMeta(buf_start, base_pos)
    var run_id = len(meta.runs)
    var last_ordinal = (base_pos + length - 1) >> PAGE_SHIFT
    for ordinal in range(last_ordinal + 1):
        run.base_rows.append(Int32(page_base_row(run_id, ordinal)))
    meta.runs.append(run^)


def probe_value(rank: Int, row: Int, pos: Int) -> Int64:
    return Int64(rank * 1_000_000_000 + row * 100_000 + pos)


@fieldwise_init
struct MetaProbeKernel(RangePartitionedKernel):
    """Walks packed tokens through the run cursor, resolving each token's
    physical row through the metadata pointer the way a paged attention kernel
    would: monotone cursor, per-run base-rows pointer hoisted on advance, no
    copies and no allocation in the loop."""
    var meta: UnsafePointer[StepKVMeta, MutAnyOrigin]
    var out: I64Ptr
    var rank: Int
    var start: Int
    var end: Int

    def execute(mut self):
        ref runs = self.meta[].runs
        var num_runs = len(runs)
        var r = 0
        var rows = runs[0].base_rows.unsafe_ptr()
        var run_start = Int(runs[0].buf_start)
        var run_pos = Int(runs[0].base_pos)
        for t in range(self.start, self.end):
            while r + 1 < num_runs and t >= Int(runs[r + 1].buf_start):
                r += 1
                rows = runs[r].base_rows.unsafe_ptr()
                run_start = Int(runs[r].buf_start)
                run_pos = Int(runs[r].base_pos)
            var pos = run_pos + (t - run_start)
            var row = Int(rows[pos >> PAGE_SHIFT]) + (pos & ROW_MASK)
            self.out[t] = probe_value(self.rank, row, pos)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


@fieldwise_init
struct ScalarProbeKernel(RangePartitionedKernel):
    """Baseline for the timing comparison: identical writes, identity row
    resolution, no metadata pointer."""
    var out: I64Ptr
    var rank: Int
    var start: Int
    var end: Int

    def execute(mut self):
        for t in range(self.start, self.end):
            self.out[t] = probe_value(self.rank, t & 1023, t)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_meta_probe[P: BurstThreadPool, //](
    mut pools: List[P],
    meta: UnsafePointer[StepKVMeta, MutAnyOrigin],
    read outs: List[I64Ptr],
    total: Int,
):
    var buf = DispatchBuffer[MetaProbeKernel, 128]()
    for r in range(len(pools)):
        _ = tile_dispatch(
            buf, MetaProbeKernel(meta, outs[r], r, 0, 0), pools[r], total)
    join_all(pools)


def dispatch_scalar_probe[P: BurstThreadPool, //](
    mut pools: List[P],
    read outs: List[I64Ptr],
    total: Int,
):
    var buf = DispatchBuffer[ScalarProbeKernel, 128]()
    for r in range(len(pools)):
        _ = tile_dispatch(
            buf, ScalarProbeKernel(outs[r], r, 0, 0), pools[r], total)
    join_all(pools)


def verify_meta_probe(
    read meta: StepKVMeta,
    read outs: List[I64Ptr],
    total: Int,
    label: String,
) -> Int:
    var mismatches = 0
    for rank in range(len(outs)):
        var r = 0
        for t in range(total):
            while r + 1 < len(meta.runs) and t >= Int(meta.runs[r + 1].buf_start):
                r += 1
            ref run = meta.runs[r]
            var pos = Int(run.base_pos) + (t - Int(run.buf_start))
            var row = Int(run.base_rows[pos >> PAGE_SHIFT]) + (pos & ROW_MASK)
            if outs[rank][t] != probe_value(rank, row, pos):
                mismatches += 1
    var status = "PASS" if mismatches == 0 else "FAIL"
    print(t"  {label}: {status} ({mismatches} mismatches, "
          t"{len(outs)} ranks x {total} tokens)")
    return mismatches


def run_proto[P: BurstThreadPool, //](
    read topo: NumaTopology,
    var pools: List[P],
):
    var degree = len(pools)
    comptime TOTAL_A = 512
    comptime TOTAL_B = 640

    var arenas = List[NumaArena[]](capacity=degree)
    var outs = List[I64Ptr]()
    for r in range(degree):
        var node = topo.node(r % len(topo))
        arenas.append(NumaArena[](node, TOTAL_B * 8 + 4096))
        if not arenas[r]:
            abort(t"proto: arena allocation failed on node {node}")
        var got = arenas[r].alloc[Int64](TOTAL_B)
        if not got:
            abort("proto: out buffer alloc failed")
        outs.append(got.value())

    var meta = StepKVMeta()
    append_run(meta, 0, 0, 200)
    append_run(meta, 200, 37, 150)
    append_run(meta, 350, 1000, 162)
    var meta_ptr = UnsafePointer(to=meta)

    var failures = 0

    print("phase 1: workers resolve rows through heap metadata")
    dispatch_meta_probe(pools, meta_ptr, outs, TOTAL_A)
    failures += verify_meta_probe(meta, outs, TOTAL_A, "initial run table")

    print("phase 2: mutate lists between bursts (realloc + new run)")
    for run_id in range(len(meta.runs)):
        ref run = meta.runs[run_id]
        var first_new = len(run.base_rows)
        for ordinal in range(first_new, first_new + 4):
            run.base_rows.append(Int32(page_base_row(run_id + 5, ordinal)))
    append_run(meta, TOTAL_A, 4321, TOTAL_B - TOTAL_A)
    dispatch_meta_probe(pools, meta_ptr, outs, TOTAL_B)
    failures += verify_meta_probe(meta, outs, TOTAL_B, "mutated run table")

    print("phase 3: burst-frequency timing (dispatch/join cycles)")
    comptime CYCLES = 2000
    comptime TIMING_TOKENS = 128

    dispatch_scalar_probe(pools, outs, TIMING_TOKENS)
    var t0 = perf_counter_ns()
    for _ in range(CYCLES):
        dispatch_scalar_probe(pools, outs, TIMING_TOKENS)
    var scalar_ns = Int(perf_counter_ns() - t0) // CYCLES

    dispatch_meta_probe(pools, meta_ptr, outs, TIMING_TOKENS)
    var t1 = perf_counter_ns()
    for _ in range(CYCLES):
        dispatch_meta_probe(pools, meta_ptr, outs, TIMING_TOKENS)
    var meta_ns = Int(perf_counter_ns() - t1) // CYCLES

    failures += verify_meta_probe(meta, outs, TIMING_TOKENS, "post-timing readback")
    print(t"  scalar baseline: {scalar_ns} ns/cycle")
    print(t"  metadata probe:  {meta_ns} ns/cycle "
          t"(delta {meta_ns - scalar_ns} ns)")

    if failures == 0:
        print("RESULT: PASS -- pointer-to-List metadata is safe under burst dispatch")
    else:
        print(t"RESULT: FAIL -- {failures} mismatches")


def main():
    var topo = NumaTopology()
    var nodes = topo.num_nodes()
    print(t"{nodes} NUMA nodes")

    @parameter
    def dispatch_proto[P: BurstThreadPool, //](var selected_pools: List[P]):
        run_proto(topo, selected_pools^)

    with_topological_rank_dispatch[
        dispatch=dispatch_proto,
    ](topo, "mode: isolated (spin-only)", "mode: cold (spin-backoff)")
