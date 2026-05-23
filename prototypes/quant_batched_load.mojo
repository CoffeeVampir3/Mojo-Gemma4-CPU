from std.collections import List
from std.memory import UnsafePointer
from std.pathlib import Path
from std.time import perf_counter_ns

from numa import NumaArena, NumaTopology
from modeling.model_spec import (
    WeightDesc, DEFAULT_ALIGNMENT, align_up, DISTRIBUTED,
)
from modeling.loader import discover_shards, load_weights_from_descs
from safetensors.parser import parse_safetensors_header, dtype_byte_size


comptime CHECKPOINT = "checkpoints/gemma-4-26B-A4B"
comptime BUDGET_BYTES = 2 * 1024 * 1024 * 1024


def fold_rows_cols(shape: List[Int]) -> Tuple[Int, Int]:
    if len(shape) == 0:
        return (1, 1)
    if len(shape) == 1:
        return (shape[0], 1)
    var rows = 1
    for i in range(len(shape) - 1):
        rows *= shape[i]
    return (rows, shape[len(shape) - 1])


def enumerate_source_descs(
    shards: List[Path],
) -> Optional[List[WeightDesc]]:
    """One WeightDesc per source tensor across all shards. arena_offset is
    left as 0 here; the batcher rewrites it per batch so the batch packs
    into the scratch."""
    var descs = List[WeightDesc]()
    for i in range(len(shards)):
        var h_opt = parse_safetensors_header(shards[i])
        if not h_opt:
            ref p = shards[i]
            print(t"prototype: failed to parse {p}")
            return None
        var h = h_opt.take()
        for item in h.tensors.items():
            ref meta = item.value
            var rc = fold_rows_cols(meta.shape)
            var rows = rc[0]
            var cols = rc[1]
            descs.append(WeightDesc(
                name=String(item.key),
                arena_offset=0,
                dtype=meta.dtype,
                element_bytes=dtype_byte_size(meta.dtype),
                global_rows=rows, global_cols=cols,
                local_cols=cols,
                data_rows=rows, data_cols=cols,
                target_rank=DISTRIBUTED,
            ))
    return descs^


def pack_batch(
    descs: List[WeightDesc],
    start_idx: Int,
    budget: Int,
) -> Tuple[List[WeightDesc], Int]:
    """Greedy walk from start_idx, accumulating tensors until the next one
    would overflow budget. Each desc's arena_offset is set to its cumulative
    aligned offset within the budget. Returns (batch, next_start_idx); next
    is set to -1 if a single tensor exceeds budget (caller would row-split
    that one — out of scope for the prototype)."""
    var batch = List[WeightDesc]()
    var off = 0
    var idx = start_idx
    while idx < len(descs):
        var d = descs[idx].copy()
        var tbytes = d.data_rows * d.data_cols * d.element_bytes
        var aligned = align_up(tbytes)
        if off + aligned > budget:
            if len(batch) == 0:
                ref n = d.name
                print(t"prototype: {n} ({tbytes} bytes) exceeds budget {budget}")
                return (batch^, -1)
            break
        d.arena_offset = off
        batch.append(d^)
        off += aligned
        idx += 1
    return (batch^, idx)


def main():
    var topo = NumaTopology()
    var n_nodes = topo.num_nodes()
    print(t"prototype: NUMA {n_nodes} node(s)")

    var shards = discover_shards(Path(CHECKPOINT))
    if len(shards) == 0:
        print(t"prototype: no shards in {CHECKPOINT}")
        return
    var n_shards = len(shards)
    print(t"prototype: {n_shards} shard(s)")

    var descs_opt = enumerate_source_descs(shards)
    if not descs_opt:
        return
    var descs = descs_opt.take()
    var n_tensors = len(descs)
    print(t"prototype: {n_tensors} source tensors")

    var scratch = NumaArena[](topo.node(0), BUDGET_BYTES)
    if not scratch:
        print(t"prototype: scratch alloc failed")
        return
    var base = Int(scratch.base.value())
    var budget_mib = BUDGET_BYTES // (1024 * 1024)
    var node0 = topo.node(0)
    print(t"prototype: scratch {budget_mib} MiB on node {node0}")

    var arena_bases = List[Int]()
    arena_bases.append(base)

    var idx = 0
    var batch_no = 0
    var total_bytes = 0
    var t0 = perf_counter_ns()

    while idx < len(descs):
        var packed = pack_batch(descs, idx, BUDGET_BYTES)
        var batch = packed[0].copy()
        var next_idx = packed[1]
        if next_idx < 0:
            return
        if len(batch) == 0:
            return

        var batch_bytes = 0
        for i in range(len(batch)):
            batch_bytes += batch[i].data_rows * batch[i].data_cols * batch[i].element_bytes
        var batch_mib = batch_bytes // (1024 * 1024)
        var n = len(batch)

        var t_load = perf_counter_ns()
        var result = load_weights_from_descs(batch, shards, arena_bases, topo)
        if not result:
            print(t"prototype: load failed at batch {batch_no}")
            return
        var load_ms = (perf_counter_ns() - t_load) / 1_000_000
        var load_gbps = Float64(batch_bytes) / (Float64(load_ms) / 1000.0) / 1e9
        print(t"prototype: batch {batch_no}: {n} tensors, {batch_mib} MiB, "
              t"loaded in {load_ms} ms ({load_gbps} GB/s)")

        # Walk the batch: this is where the per-desc quantize+write
        # would dispatch. The source bytes for desc i live at
        # base + batch[i].arena_offset and are valid until the next
        # load_weights_from_descs call.
        var checksum = UInt64(0)
        for i in range(len(batch)):
            ref d = batch[i]
            var src = UnsafePointer[UInt8, MutAnyOrigin](
                unsafe_from_address=base + d.arena_offset)
            checksum ^= UInt64(src.load[width=1]())
        # Suppress unused-warning; in the real pipeline this is where
        # transform + write descriptors flow into the writer ring.
        _ = checksum

        total_bytes += batch_bytes
        batch_no += 1
        idx = next_idx

    var elapsed_ms = (perf_counter_ns() - t0) / 1_000_000
    var total_gib = Float64(total_bytes) / Float64(1024 * 1024 * 1024)
    var avg_gbps = Float64(total_bytes) / (Float64(elapsed_ms) / 1000.0) / 1e9
    print(t"prototype: {batch_no} batch(es), {total_gib} GiB total, "
          t"{elapsed_ms} ms, {avg_gbps} GB/s avg")
