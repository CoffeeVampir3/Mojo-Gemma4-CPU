from std.memory import UnsafePointer
from std.os import abort

from numa import NumaArena, NumaTopology
from kernels.attention_ops import (
    KVRunTable, LinearKV, PagedKV, RingKV, pow2_shift,
)
from continuous_batching.paging import (
    KVPageAccountant, KVPageAllocator, PagePoolSpec, BatchGeometry,
)


comptime FPtr = UnsafePointer[Float32, MutAnyOrigin]
comptime RING_POOL = 0
comptime GROWING_POOL = 1


def two_pool_geometry(
    page_len: Int, max_seqs: Int, num_pages: Int, max_pages_per_seq: Int,
) -> BatchGeometry:
    var pools = List[PagePoolSpec]()
    pools.append(PagePoolSpec(
        num_pages=max_seqs * 2,
        fixed_pages_per_seq=2,
        max_pages_per_seq=2))
    pools.append(PagePoolSpec(
        num_pages=num_pages,
        fixed_pages_per_seq=0,
        max_pages_per_seq=max_pages_per_seq))
    return BatchGeometry(
        max_seqs=max_seqs, max_slots=max_seqs, max_step_tokens=page_len,
        pools=pools^)


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
    print("--- allocator free list ---")
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
    if pages.available() != 0:
        failures += 1

    print(t"  {failures} mismatches")
    return failures


def accountant_lifecycle() -> Int:
    print("--- accountant admission / reservation / release ---")
    var failures = 0
    var kv = KVPageAccountant(two_pool_geometry(
        page_len=4, max_seqs=2, num_pages=2, max_pages_per_seq=4))

    var a = kv.admit()
    if a != 0:
        failures += 1
    if kv.page_index(RING_POOL, a, 0) != 0 or kv.page_index(RING_POOL, a, 1) != 1:
        failures += 1
    var b = kv.admit()
    if b != 1:
        failures += 1
    if kv.admit() != -1:
        failures += 1

    var need = List[Int](length=kv.pool_count(), fill=0)
    kv.pages_needed(a, 1, need)
    if need[RING_POOL] != 0 or need[GROWING_POOL] != 2:
        failures += 1
    if not kv.fits(need):
        failures += 1

    if not kv.reserve(a, 1):
        failures += 1
    if kv.reserve(a, 2):
        failures += 1
    if not kv.reserve(a, 1):
        failures += 1

    var full_runs = KVRunTable()
    var sliding_runs = KVRunTable()
    full_runs.begin_run(0, 4)
    for ordinal in range(2):
        full_runs.add_base_row(Int32(kv.page_index(GROWING_POOL, a, ordinal) * 4))
    sliding_runs.begin_run(0, 4)
    for ordinal in range(2):
        sliding_runs.add_base_row(Int32(kv.page_index(RING_POOL, a, ordinal) * 4))

    if len(full_runs.runs) != 1 or len(sliding_runs.runs) != 1:
        failures += 1
    if Int(full_runs.runs[0].page_count) != 2:
        failures += 1
    if Int(full_runs.row_ptr(0)[0]) != 0 or Int(full_runs.row_ptr(0)[1]) != 4:
        failures += 1
    if Int(sliding_runs.row_ptr(0)[0]) != 0 or Int(sliding_runs.row_ptr(0)[1]) != 4:
        failures += 1

    kv.release(a)
    kv.release(b)
    if kv.pool_available(RING_POOL) != 4 or kv.pool_available(GROWING_POOL) != 2:
        failures += 1
    var c = kv.admit()
    if c != 0:
        failures += 1
    kv.release(c)

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

    var kv = KVPageAccountant(two_pool_geometry(
        page_len=page_len, max_seqs=4, num_pages=num_pages,
        max_pages_per_seq=4))
    var seq_a = kv.admit()
    var seq_b = kv.admit()
    var len_a = 10
    var len_b = 6

    _ = kv.reserve(seq_a, 0)
    _ = kv.reserve(seq_b, 0)
    _ = kv.reserve(seq_a, 1)
    _ = kv.reserve(seq_b, 1)
    _ = kv.reserve(seq_a, 2)

    if kv.page_index(GROWING_POOL, seq_a, 1) != 2:
        failures += 1
    if kv.page_index(GROWING_POOL, seq_b, 1) != 3:
        failures += 1

    var lengths = List[Int]()
    lengths.append(len_a)
    lengths.append(len_b)

    var full_runs = KVRunTable()
    var seq_ids = List[Int]()
    seq_ids.append(seq_a)
    seq_ids.append(seq_b)
    var buf_start = 0
    for s in range(2):
        full_runs.begin_run(buf_start, 0)
        var last_ordinal = (lengths[s] - 1) // page_len
        for ordinal in range(last_ordinal + 1):
            full_runs.add_base_row(Int32(
                kv.page_index(GROWING_POOL, seq_ids[s], ordinal)
                * rows_per_page))
        buf_start += lengths[s]

    for rank in range(degree):
        for s in range(2):
            var paged = PagedKV(
                full_runs.row_ptr(s), page_shift, row_mask, -1)
            for pos in range(lengths[s]):
                if pos % degree != rank:
                    continue
                var slot = paged.slot(0, pos // degree)
                for d in range(kv_dim):
                    (bricks[rank] + slot * kv_dim + d)[] = probe_value(
                        rank, s, pos, d)

    for rank in range(degree):
        for s in range(2):
            var paged = PagedKV(
                full_runs.row_ptr(s), page_shift, row_mask, -1)
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
    if kv.pool_available(GROWING_POOL) != num_pages:
        failures += 1

    print(t"  {failures} mismatches")
    return failures


def main():
    var total = 0
    total += policy_equivalence()
    total += allocator_behavior()
    total += accountant_lifecycle()
    total += paged_traversal()
    if total == 0:
        print("RESULT: PASS -- paged KV policies, allocator, and accountant hold")
    else:
        print(t"RESULT: FAIL -- {total} mismatches")
