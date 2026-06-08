from std.collections import InlineArray

from kernels.attention_ops import KVRun, KVRunTable


struct KVPageAllocator(Movable):
    """Free list over logical page indices with per-page refcounts. One
    instance is the single source of truth that drives every rank, so an
    acquired page is reserved in lockstep on all arenas and a page index never
    means two different sequences across ranks. Refcounts are 1 for the v1
    single-owner discipline; `retain` exists so future prefix sharing only
    changes the release count, not the allocator's shape."""
    var free_stack: List[Int32]
    var refcounts: List[Int32]

    def __init__(out self, num_pages: Int):
        self.free_stack = List[Int32](capacity=num_pages)
        for i in range(num_pages):
            self.free_stack.append(Int32(num_pages - 1 - i))
        self.refcounts = List[Int32](length=num_pages, fill=Int32(0))

    @always_inline
    def acquire(mut self) -> Int:
        if len(self.free_stack) == 0:
            return -1
        var page = Int(self.free_stack.pop())
        self.refcounts[page] = 1
        return page

    @always_inline
    def retain(mut self, page: Int):
        self.refcounts[page] += 1

    @always_inline
    def release(mut self, page: Int):
        var rc = self.refcounts[page] - 1
        self.refcounts[page] = rc
        if rc == 0:
            self.free_stack.append(Int32(page))

    @always_inline
    def available(self) -> Int:
        return len(self.free_stack)


struct KVPageTable(Movable):
    """Maps `(seq_id, page_ordinal)` to a physical `page_index`. Rank- and
    layer-independent shared truth: a page index resolves to the same row range
    in every brick of its pool on every arena. `-1` marks an unmapped
    ordinal."""
    var entries: List[Int32]
    var max_seqs: Int
    var max_pages_per_seq: Int

    def __init__(out self, max_seqs: Int, max_pages_per_seq: Int):
        self.max_seqs = max_seqs
        self.max_pages_per_seq = max_pages_per_seq
        self.entries = List[Int32](
            length=max_seqs * max_pages_per_seq, fill=Int32(-1))

    @always_inline
    def entry_index(self, seq_id: Int, page_ordinal: Int) -> Int:
        return seq_id * self.max_pages_per_seq + page_ordinal

    @always_inline
    def map_page(mut self, seq_id: Int, page_ordinal: Int, page_index: Int):
        self.entries[self.entry_index(seq_id, page_ordinal)] = Int32(page_index)

    @always_inline
    def page_index(self, seq_id: Int, page_ordinal: Int) -> Int:
        return Int(self.entries[self.entry_index(seq_id, page_ordinal)])


struct SeqKVState(Copyable, Movable):
    var active: Bool
    var length: Int
    var sliding_pages: InlineArray[Int32, 2]

    def __init__(out self):
        self.active = False
        self.length = 0
        self.sliding_pages = InlineArray[Int32, 2](fill=Int32(-1))


struct KVRuntime(Movable):
    """Owner of both KV pools' state: the full-attention page table and
    allocator, the sliding page allocator, per-sequence state, and the per-step
    run tables the kernels read by pointer. Sliding pages are acquired greedily
    at admission (two per sequence, even/odd ring slots), so admission is
    atomic and the only mid-life allocation is full pages, reserved before each
    chunk's dispatch. The run tables are rebuilt per chunk and never mutated
    with work in flight."""
    var page_len: Int
    var degree: Int
    var max_seqs: Int
    var max_pages_per_seq: Int
    var full_pages: KVPageAllocator
    var full_table: KVPageTable
    var sliding_pages: KVPageAllocator
    var seqs: List[SeqKVState]
    var full_runs: KVRunTable
    var sliding_runs: KVRunTable

    def __init__(
        out self,
        page_len: Int,
        degree: Int,
        num_pages: Int,
        max_seqs: Int,
        max_pages_per_seq: Int,
    ):
        self.page_len = page_len
        self.degree = degree
        self.max_seqs = max_seqs
        self.max_pages_per_seq = max_pages_per_seq
        self.full_pages = KVPageAllocator(num_pages)
        self.full_table = KVPageTable(max_seqs, max_pages_per_seq)
        self.sliding_pages = KVPageAllocator(num_pages)
        self.seqs = List[SeqKVState](length=max_seqs, fill=SeqKVState())
        self.full_runs = KVRunTable()
        self.sliding_runs = KVRunTable()

    def admit(mut self) -> Int:
        for seq_id in range(self.max_seqs):
            if self.seqs[seq_id].active:
                continue
            var even = self.sliding_pages.acquire()
            if even < 0:
                return -1
            var odd = self.sliding_pages.acquire()
            if odd < 0:
                self.sliding_pages.release(even)
                return -1
            self.seqs[seq_id].active = True
            self.seqs[seq_id].length = 0
            self.seqs[seq_id].sliding_pages[0] = Int32(even)
            self.seqs[seq_id].sliding_pages[1] = Int32(odd)
            return seq_id
        return -1

    def release(mut self, seq_id: Int):
        if not self.seqs[seq_id].active:
            return
        for ordinal in range(self.max_pages_per_seq):
            var page = self.full_table.page_index(seq_id, ordinal)
            if page >= 0:
                self.full_pages.release(page)
                self.full_table.map_page(seq_id, ordinal, -1)
        for i in range(2):
            var page = Int(self.seqs[seq_id].sliding_pages[i])
            if page >= 0:
                self.sliding_pages.release(page)
            self.seqs[seq_id].sliding_pages[i] = Int32(-1)
        self.seqs[seq_id].active = False
        self.seqs[seq_id].length = 0

    def reserve_full(mut self, seq_id: Int, last_pos: Int) -> Bool:
        var last_ordinal = last_pos // self.page_len
        for ordinal in range(last_ordinal + 1):
            if self.full_table.page_index(seq_id, ordinal) >= 0:
                continue
            var page = self.full_pages.acquire()
            if page < 0:
                return False
            self.full_table.map_page(seq_id, ordinal, page)
        return True

    def note_extent(mut self, seq_id: Int, end_pos: Int):
        if end_pos > self.seqs[seq_id].length:
            self.seqs[seq_id].length = end_pos

    def begin_step(mut self):
        self.full_runs.clear()
        self.sliding_runs.clear()

    def push_run(mut self, seq_id: Int, buf_start: Int, base_pos: Int, last_pos: Int):
        debug_assert(self.seqs[seq_id].active, "kv runtime: run for inactive sequence")
        var rows_per_page = self.page_len // self.degree
        var last_ordinal = last_pos // self.page_len

        var full = KVRun(buf_start, base_pos)
        for ordinal in range(last_ordinal + 1):
            var page = self.full_table.page_index(seq_id, ordinal)
            debug_assert(page >= 0, "kv runtime: run references unmapped page")
            full.base_rows.append(Int32(page * rows_per_page))
        self.full_runs.runs.append(full^)

        var sliding = KVRun(buf_start, base_pos)
        for i in range(2):
            var page = Int(self.seqs[seq_id].sliding_pages[i])
            sliding.base_rows.append(Int32(page * self.page_len))
        self.sliding_runs.runs.append(sliding^)
