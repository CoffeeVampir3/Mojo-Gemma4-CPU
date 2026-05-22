from std.memory import Span, UnsafePointer
from std.pathlib import Path

from numa import NumaTopology, NumaArena
from threading.burst_threading import BurstPool
from threading.threading_traits import BurstKernel
from linux.io_uring import (
    IoRing, ReadOp, Completion,
    ReadMode, open_files_for_ring, process_read_queue, close_fds,
)
from safetensors.parser import parse_safetensors_header, SafetensorsHeader


comptime CHUNK_SIZE = 8 * 1024 * 1024
comptime IN_FLIGHT = 16
comptime PROBE_ARENA_BYTES = IN_FLIGHT * CHUNK_SIZE
comptime DEFAULT_QUEUE_DEPTH = 256
comptime DEFAULT_MASK_SIZE = 128


@fieldwise_init
struct ReadDesc(Copyable, ImplicitlyCopyable, Movable):
    var shard: Int
    var file_offset: Int
    var bytes: Int


def enumerate_tensors(paths: List[Path]) -> Optional[List[ReadDesc]]:
    var headers = List[SafetensorsHeader]()
    for i in range(len(paths)):
        var h = parse_safetensors_header(paths[i])
        if not h:
            ref p = paths[i]
            print(t"probe: failed to parse {p}")
            return None
        headers.append(h.take())

    var descs = List[ReadDesc]()
    for i in range(len(headers)):
        ref hdr = headers[i]
        for item in hdr.tensors.items():
            ref meta = item.value
            descs.append(ReadDesc(
                shard=i,
                file_offset=hdr.data_offset + meta.start,
                bytes=meta.end - meta.start,
            ))
    return descs^


def partition_round_robin(
    descs: List[ReadDesc], tp: Int,
) -> List[List[ReadDesc]]:
    var out = List[List[ReadDesc]]()
    for _ in range(tp):
        out.append(List[ReadDesc]())
    for i in range(len(descs)):
        out[i % tp].append(descs[i].copy())
    return out^


@fieldwise_init
struct ReadProbeKernel[queue_depth: Int](BurstKernel):
    var rank: Int
    var node: Int
    var arena_base: Int
    var fds: Span[Int32, MutAnyOrigin]
    var descs: Span[ReadDesc, MutAnyOrigin]
    var bytes_out: UnsafePointer[Int, MutAnyOrigin]
    var ops_out: UnsafePointer[Int, MutAnyOrigin]
    var ok_out: UnsafePointer[Int32, MutAnyOrigin]

    def execute(mut self):
        var ring = IoRing[Self.queue_depth]()
        if not ring:
            print(t"probe: rank {self.rank} ring init failed")
            self.ok_out[] = 0
            return
        try:
            _ = ring.register_fds(self.fds)
        except err:
            print(t"probe: rank {self.rank} register_fds failed: {err.error_message()}")
            self.ok_out[] = 0
            return

        var ops = List[ReadOp[]]()
        var next_id = 0
        for d in range(len(self.descs)):
            ref desc = self.descs[d]
            var remaining = desc.bytes
            var off = desc.file_offset
            while remaining > 0:
                var n = CHUNK_SIZE if remaining > CHUNK_SIZE else remaining
                var slot = next_id % IN_FLIGHT
                var dest = UnsafePointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=self.arena_base + slot * CHUNK_SIZE)
                ops.append(ReadOp(
                    file_idx=desc.shard, offset=off, length=n,
                    dest=dest, id=next_id))
                next_id += 1
                off += n
                remaining -= n

        var byte_acc = 0

        @parameter
        def on_complete(c: Completion):
            byte_acc += Int(c.result)

        var ops_span = Span[ReadOp[], MutAnyOrigin](
            ptr=UnsafePointer[ReadOp[], MutAnyOrigin](
                unsafe_from_address=Int(ops.unsafe_ptr())),
            length=len(ops))

        var err_opt = process_read_queue[on_complete](ring, ops_span)
        if err_opt:
            print(t"probe: rank {self.rank} io error: {err_opt.value().msg}")
            self.ok_out[] = 0
            return

        self.bytes_out[] = byte_acc
        self.ops_out[] = len(ops)
        self.ok_out[] = 1
        var op_count = len(ops)
        print(t"probe: rank {self.rank} node {self.node} ops {op_count} bytes {byte_acc}")
        _ = ops^


def make_probe_kernel[queue_depth: Int](
    rank: Int,
    node: Int,
    arena_base: Int,
    fds: Span[Int32, _],
    descs: Span[ReadDesc, _],
    bytes_out: UnsafePointer[Int, MutAnyOrigin],
    ops_out: UnsafePointer[Int, MutAnyOrigin],
    ok_out: UnsafePointer[Int32, MutAnyOrigin],
) -> ReadProbeKernel[queue_depth]:
    var fds_wild = UnsafePointer[Int32, MutAnyOrigin](
        unsafe_from_address=Int(fds.unsafe_ptr()))
    var descs_wild = UnsafePointer[ReadDesc, MutAnyOrigin](
        unsafe_from_address=Int(descs.unsafe_ptr()))
    return ReadProbeKernel[queue_depth](
        rank=rank, node=node, arena_base=arena_base,
        fds=Span[Int32, MutAnyOrigin](ptr=fds_wild, length=len(fds)),
        descs=Span[ReadDesc, MutAnyOrigin](ptr=descs_wild, length=len(descs)),
        bytes_out=bytes_out, ops_out=ops_out, ok_out=ok_out,
    )


def run_read_probe[
    queue_depth: Int = DEFAULT_QUEUE_DEPTH,
    mask_size: Int = DEFAULT_MASK_SIZE,
](paths: List[Path], topo: NumaTopology) -> Bool:
    var descs_opt = enumerate_tensors(paths)
    if not descs_opt:
        return False
    var descs = descs_opt.take()
    var tp = len(topo)
    var num_descs = len(descs)
    var num_shards = len(paths)
    print(t"probe: {num_descs} tensors across {num_shards} shard(s), tp={tp}")

    var partitioned = partition_round_robin(descs, tp)

    var arenas = List[NumaArena[]](capacity=tp)
    for r in range(tp):
        arenas.append(NumaArena[](topo.node(r), PROBE_ARENA_BYTES))
        if not arenas[r]:
            var node = topo.node(r)
            print(t"probe: arena allocation failed on node {node}")
            return False

    var pools = List[BurstPool[mask_size]](capacity=tp)
    for r in range(tp):
        var mask = topo.mask[mask_size](r)
        pools.append(BurstPool[mask_size](
            capacity=1, cpu_mask=mask, numa_node=topo.node(r)))
        if not pools[r]:
            print(t"probe: pool setup failed for rank {r}")
            return False

    var fds: List[Int32]
    try:
        fds = open_files_for_ring[ReadMode](Span(paths))
    except err:
        print(t"probe: open_files_for_ring failed: {err.error_message()}")
        return False

    var bytes_results = List[Int](length=tp, fill=0)
    var ops_results = List[Int](length=tp, fill=0)
    var ok_results = List[Int32](length=tp, fill=Int32(0))

    var bytes_base = UnsafePointer[Int, MutAnyOrigin](
        unsafe_from_address=Int(bytes_results.unsafe_ptr()))
    var ops_base = UnsafePointer[Int, MutAnyOrigin](
        unsafe_from_address=Int(ops_results.unsafe_ptr()))
    var ok_base = UnsafePointer[Int32, MutAnyOrigin](
        unsafe_from_address=Int(ok_results.unsafe_ptr()))

    var kernels = List[ReadProbeKernel[queue_depth]](capacity=tp)
    for r in range(tp):
        var arena_base = Int(arenas[r].base.value())
        kernels.append(make_probe_kernel[queue_depth](
            rank=r,
            node=topo.node(r),
            arena_base=arena_base,
            fds=Span(fds),
            descs=Span(partitioned[r]),
            bytes_out=bytes_base + r,
            ops_out=ops_base + r,
            ok_out=ok_base + r,
        ))

    var pool_base = pools.unsafe_ptr()
    for r in range(tp):
        var span = Span[ReadProbeKernel[queue_depth], MutAnyOrigin](
            ptr=UnsafePointer(to=kernels[r]), length=1)
        (pool_base + r)[].dispatch(span, 1)
    for r in range(tp):
        (pool_base + r)[].join()

    _ = kernels^
    _ = partitioned
    _ = arenas^
    close_fds(fds^)

    var total_bytes = 0
    var total_ops = 0
    var all_ok = True
    for r in range(tp):
        if ok_results[r] == 0:
            all_ok = False
        total_bytes += bytes_results[r]
        total_ops += ops_results[r]
    var total_mb = total_bytes // (1024 * 1024)
    print(t"probe: total ops {total_ops} bytes {total_bytes} ({total_mb} MB) ok={all_ok}")
    return all_ok
