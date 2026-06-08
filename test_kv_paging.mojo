from std.memory import UnsafePointer
from std.os import abort

from numa import NumaArena, NumaTopology
from kernels.attention_ops import (
    KVRun, KVRunTable, LinearKV, PagedKV, RingKV, pow2_shift,
)
from modeling.kv_runtime import KVPageAllocator, KVRuntime


comptime FPtr = UnsafePointer[Float32, MutAnyOrigin]


def policy_equivalence() -> Int:
    print("--- policy equivalence ---")
    var failures = 0

    var ring = RingKV[2048]()
    var ring_rows = List[Int32]()
    ring_rows.append(Int32(0))
    ring_rows.append(Int32(1024))
    var paged_ring = PagedKV(ring_rows.unsafe_ptr(), 10, 1023, 1)
    for start in range(0, 4096, 97):
        for t in range(0, 1024, 13):
            if paged_ring.slot(start, t) != ring.slot(start, t):
                failures += 1

    var linear = LinearKV()
    var table_rows = List[Int32]()
    for ordinal in range(8):
        table_rows.append(Int32(ordinal * 1024))
    var paged_table = PagedKV(table_rows.unsafe_ptr(), 10, 1023, -1)
    for t in range(0, 8192, 7):
        if paged_table.slot(0, t) != linear.slot(0, t):
            failures += 1

    var scattered = List[Int32]()
    scattered.append(Int32(12))
    scattered.append(Int32(4))
    scattered.append(Int32(28))
    var paged_scattered = PagedKV(scattered.unsafe_ptr(), 2, 3, -1)
    for t in range(12):
        var want = Int(scattered[t // 4]) + t % 4
        if paged_scattered.slot(0, t) != want:
            failures += 1

    print(t"  {failures} mismatches")
    return failures


def allocator_behavior() -> Int:
    print("--- allocator refcounts ---")
    var failures = 0
    var pages = KVPageAllocator(4)

    for expected in range(4):
        if pages.acquire() != expected:
            failures += 1
    if pages.acquire() != -1:
        failures += 1

    pages.release(1)
    pages.release(3)
    if pages.available() != 2:
        failures += 1
    if pages.acquire() != 3:
        failures += 1
    if pages.acquire() != 1:
        failures += 1

    pages.retain(0)
    pages.release(0)
    if pages.available() != 0:
        failures += 1
    pages.release(0)
    if pages.available() != 1:
        failures += 1
    if pages.acquire() != 0:
        failures += 1

    print(t"  {failures} mismatches")
    return failures


def runtime_lifecycle() -> Int:
    print("--- runtime admission / reservation / release ---")
    var failures = 0
    var kv = KVRuntime(
        page_len=4, degree=1, num_pages=2, max_seqs=2, max_pages_per_seq=4)

    var a = kv.admit()
    if a != 0:
        failures += 1
    if kv.admit() != -1:
        failures += 1
    if Int(kv.seqs[a].sliding_pages[0]) != 0 or Int(kv.seqs[a].sliding_pages[1]) != 1:
        failures += 1

    if not kv.reserve_full(a, 7):
        failures += 1
    if kv.reserve_full(a, 11):
        failures += 1
    if not kv.reserve_full(a, 7):
        failures += 1

    kv.begin_step()
    kv.push_run(a, 0, 4, 7)
    if len(kv.full_runs.runs) != 1 or len(kv.sliding_runs.runs) != 1:
        failures += 1
    ref full_run = kv.full_runs.runs[0]
    if len(full_run.base_rows) != 2:
        failures += 1
    if Int(full_run.base_rows[0]) != 0 or Int(full_run.base_rows[1]) != 4:
        failures += 1
    ref sliding_run = kv.sliding_runs.runs[0]
    if Int(sliding_run.base_rows[0]) != 0 or Int(sliding_run.base_rows[1]) != 4:
        failures += 1

    kv.release(a)
    if kv.full_pages.available() != 2 or kv.sliding_pages.available() != 2:
        failures += 1
    var b = kv.admit()
    if b != 0:
        failures += 1
    kv.release(b)

    print(t"  {failures} mismatches")
    return failures


@always_inline
def probe_value(rank: Int, seq_id: Int, seq_pos: Int, d: Int) -> Float32:
    return Float32(rank * 100000 + seq_id * 10000 + seq_pos * 100 + d)


def paged_traversal() -> Int:
    print("--- interleaved-page traversal across symmetric arenas ---")
    var failures = 0
    comptime degree = 2
    comptime page_len = 4
    comptime kv_dim = 8
    comptime num_pages = 8
    comptime rows_per_page = page_len // degree
    comptime page_shift = pow2_shift(rows_per_page)
    comptime row_mask = rows_per_page - 1

    var topo = NumaTopology()
    var arenas = List[NumaArena[]](capacity=degree)
    var bricks = List[FPtr]()
    var brick_rows = num_pages * rows_per_page
    for rank in range(degree):
        var node = topo.node(rank % len(topo))
        arenas.append(NumaArena[](node, brick_rows * kv_dim * 4 + 4096))
        if not arenas[rank]:
            abort(t"traversal: arena allocation failed on node {node}")
        var got = arenas[rank].alloc[Float32](brick_rows * kv_dim)
        if not got:
            abort("traversal: brick alloc failed")
        bricks.append(got.value())

    var kv = KVRuntime(
        page_len=page_len, degree=degree, num_pages=num_pages,
        max_seqs=4, max_pages_per_seq=4)
    var seq_a = kv.admit()
    var seq_b = kv.admit()
    var len_a = 10
    var len_b = 6

    _ = kv.reserve_full(seq_a, 3)
    _ = kv.reserve_full(seq_b, 3)
    _ = kv.reserve_full(seq_a, 7)
    _ = kv.reserve_full(seq_b, 5)
    _ = kv.reserve_full(seq_a, 9)

    if kv.full_table.page_index(seq_a, 1) != 2:
        failures += 1
    if kv.full_table.page_index(seq_b, 1) != 3:
        failures += 1

    kv.begin_step()
    kv.push_run(seq_a, 0, 0, len_a - 1)
    kv.push_run(seq_b, 0, 0, len_b - 1)

    var lengths = List[Int]()
    lengths.append(len_a)
    lengths.append(len_b)

    for rank in range(degree):
        for s in range(2):
            ref run = kv.full_runs.runs[s]
            var paged = PagedKV(
                run.base_rows.unsafe_ptr(), page_shift, row_mask, -1)
            for pos in range(lengths[s]):
                if pos % degree != rank:
                    continue
                var slot = paged.slot(0, pos // degree)
                for d in range(kv_dim):
                    (bricks[rank] + slot * kv_dim + d)[] = probe_value(
                        rank, s, pos, d)

    for rank in range(degree):
        for s in range(2):
            ref run = kv.full_runs.runs[s]
            var paged = PagedKV(
                run.base_rows.unsafe_ptr(), page_shift, row_mask, -1)
            for pos in range(lengths[s]):
                if pos % degree != rank:
                    continue
                var slot = paged.slot(0, pos // degree)
                for d in range(kv_dim):
                    var got = (bricks[rank] + slot * kv_dim + d)[]
                    if got != probe_value(rank, s, pos, d):
                        failures += 1

    kv.release(seq_a)
    kv.release(seq_b)
    if kv.full_pages.available() != num_pages:
        failures += 1

    print(t"  {failures} mismatches")
    return failures


def main():
    var total = 0
    total += policy_equivalence()
    total += allocator_behavior()
    total += runtime_lifecycle()
    total += paged_traversal()
    if total == 0:
        print("RESULT: PASS -- paged KV policies, allocator, and runtime hold")
    else:
        print(t"RESULT: FAIL -- {total} mismatches")
