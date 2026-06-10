from std.memory import Span
from std.time import perf_counter_ns

from kernels.flash_sample import SamplingParams, SampleOutcome

from .schedule import (
    Schedule, BatchSlot, PageCopy, ScheduledModel, MAXIMUM_SAMPLING_LOGITS,
)
from .paging import KVPageAccountant, BatchGeometry
from .slot_registry import SlotRegistry


def common_prefix_len(read a: List[Int32], read b: List[Int32]) -> Int:
    var n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


@fieldwise_init
struct PrefixReuse(Copyable, Movable, ImplicitlyCopyable):
    var seq_id: Int
    var adopted: Bool
    var trim_len: Int
    var copy_start: Int
    var copy_count: Int


@fieldwise_init
struct PagePin(Copyable, Movable, ImplicitlyCopyable):
    var pool: Int
    var page: Int


struct Request(Movable):
    var tokens: List[Int32]
    var generated: List[Int32]
    var sampling: SamplingParams
    var max_new_tokens: Int
    var seq_id: Int
    var done: Bool

    def __init__(
        out self, var tokens: List[Int32], sampling: SamplingParams,
        max_new_tokens: Int,
    ):
        self.tokens = tokens^
        self.generated = List[Int32]()
        self.sampling = sampling
        self.max_new_tokens = max_new_tokens
        self.seq_id = -1
        self.done = False


struct ContinuousBatchScheduler[positions_per_page: Int](Movable):
    var pages: KVPageAccountant
    var registry: SlotRegistry
    var requests: List[Request]
    var schedule: Schedule
    var need: List[Int]
    var max_slots: Int
    var step_budget: Int
    var eos_token: Int32
    var first_pending: Int
    var ring_capacity: Int
    var ring_floor: List[Int]
    var reuse: List[PrefixReuse]
    var reuse_copies: List[PageCopy]
    var pins: List[PagePin]

    def __init__(
        out self, read geometry: BatchGeometry, step_budget: Int,
        eos_token: Int32,
    ):
        self.pages = KVPageAccountant(geometry)
        self.registry = SlotRegistry(geometry.max_seqs)
        self.requests = List[Request]()
        self.schedule = Schedule()
        self.need = List[Int](length=len(geometry.pools), fill=0)
        self.max_slots = geometry.max_slots
        self.step_budget = min(step_budget, geometry.max_step_tokens)
        self.eos_token = eos_token
        self.first_pending = 0
        var ring_capacity = 1 << 62
        for p in range(len(geometry.pools)):
            if geometry.pools[p].fixed_pages_per_seq > 0:
                var cap = (geometry.pools[p].fixed_pages_per_seq
                           * Self.positions_per_page)
                if cap < ring_capacity:
                    ring_capacity = cap
        self.ring_capacity = ring_capacity
        self.ring_floor = List[Int](length=geometry.max_seqs, fill=0)
        self.reuse = List[PrefixReuse]()
        self.reuse_copies = List[PageCopy]()
        self.pins = List[PagePin]()

    def submit(
        mut self, var tokens: List[Int32], sampling: SamplingParams,
        max_new_tokens: Int,
    ) -> Int:
        debug_assert(len(tokens) > 0, "submit requires a non-empty prompt")
        var request_id = len(self.requests)
        self.requests.append(Request(tokens^, sampling, max_new_tokens))
        return request_id

    def pending_work(self) -> Bool:
        for i in range(self.first_pending, len(self.requests)):
            if not self.requests[i].done:
                return True
        return False

    def evict_warm(mut self) -> Bool:
        var victim = self.registry.lru_victim()
        if victim < 0:
            return False
        self.pages.release(victim)
        self.registry.close(victim)
        return True

    @always_inline
    def window_intact(self, seq_id: Int, lcp: Int) -> Bool:
        var seq_len = self.registry.length(seq_id)
        if lcp >= seq_len:
            return True
        var live_start = lcp - Self.positions_per_page
        if live_start < 0:
            live_start = 0
        var floor = self.ring_floor[seq_id]
        if seq_len - self.ring_capacity > floor:
            floor = seq_len - self.ring_capacity
        return live_start >= floor

    def release_copy_pins(mut self):
        for k in range(len(self.pins)):
            self.pages.release_page(self.pins[k].pool, self.pins[k].page)
        self.pins.clear()

    def best_reusable(
        self, request_id: Int, want_owned: Bool,
    ) -> Tuple[Int, Int]:
        var best = -1
        var best_lcp = 0
        for sid in range(self.registry.max_seqs):
            if not self.registry.is_resident(sid):
                continue
            if (self.registry.owner_of(sid) >= 0) != want_owned:
                continue
            var lcp = self.registry.prefix_len(
                sid, Span(self.requests[request_id].tokens))
            if lcp <= best_lcp:
                continue
            if not self.window_intact(sid, lcp):
                continue
            best = sid
            best_lcp = lcp
        return (best, best_lcp)

    def adopt_prefix(
        mut self, sid: Int, request_id: Int, lcp: Int, now: UInt,
    ) -> Bool:
        var req_len = len(self.requests[request_id].tokens)
        var seq_len = self.registry.length(sid)
        var take = lcp
        var resume = take - 1 if take == req_len else take
        var keep_last = (take - 1) // Self.positions_per_page
        var private_ordinal = resume // Self.positions_per_page
        var do_private = False
        if private_ordinal <= keep_last:
            for p in range(self.pages.pool_count()):
                if self.pages.pool_fixed_pages(p) > 0:
                    continue
                var page = self.pages.page_index(p, sid, private_ordinal)
                if self.pages.page_holds(p, page) > 1:
                    do_private = True
        if do_private:
            for p in range(len(self.need)):
                self.need[p] = 0
            for p in range(self.pages.pool_count()):
                if self.pages.pool_fixed_pages(p) > 0:
                    continue
                var page = self.pages.page_index(p, sid, private_ordinal)
                if self.pages.page_holds(p, page) > 1:
                    self.need[p] += 1
            while not self.pages.fits(self.need):
                if self.evict_warm():
                    continue
                take = private_ordinal * Self.positions_per_page
                do_private = False
                break
            if not do_private:
                if take < 1:
                    return False
                keep_last = (take - 1) // Self.positions_per_page
        self.registry.adopt(sid, request_id, now)
        var copy_start = len(self.reuse_copies)
        var trim = take
        if do_private:
            var valid_rows = take - private_ordinal * Self.positions_per_page
            for p in range(self.pages.pool_count()):
                if self.pages.pool_fixed_pages(p) > 0:
                    continue
                var old_page = self.pages.page_index(p, sid, private_ordinal)
                if self.pages.page_holds(p, old_page) <= 1:
                    continue
                var fresh = self.pages.replace_with_private(
                    p, sid, private_ordinal)
                debug_assert(
                    fresh >= 0, "adopt: privatization page must fit")
                self.pages.retain_page(p, old_page)
                self.pins.append(PagePin(p, old_page))
                self.reuse_copies.append(PageCopy(
                    p, old_page, fresh, 0, valid_rows))
            trim = private_ordinal * Self.positions_per_page
        if take < seq_len:
            var stale_floor = seq_len - self.ring_capacity
            if stale_floor > self.ring_floor[sid]:
                self.ring_floor[sid] = stale_floor
            self.pages.truncate(sid, keep_last)
            self.registry.seed(
                sid, Span(self.requests[request_id].tokens), take, now)
        self.reuse.append(PrefixReuse(
            sid, True, trim, copy_start,
            len(self.reuse_copies) - copy_start))
        self.requests[request_id].seq_id = sid
        return True

    def fork_prefix(
        mut self, donor: Int, request_id: Int, lcp: Int, now: UInt,
    ) -> Bool:
        var req_len = len(self.requests[request_id].tokens)
        var resume = lcp - 1 if lcp == req_len else lcp
        var sid = self.pages.admit()
        if sid < 0:
            if not self.evict_warm():
                return False
            sid = self.pages.admit()
            if sid < 0:
                return False
        var sealed = resume // Self.positions_per_page
        var last_ordinal = (lcp - 1) // Self.positions_per_page
        for p in range(self.pages.pool_count()):
            if self.pages.pool_fixed_pages(p) > 0:
                continue
            for ordinal in range(sealed):
                self.pages.share(p, donor, sid, ordinal)
        while True:
            for p in range(len(self.need)):
                self.need[p] = 0
            self.pages.pages_needed(sid, last_ordinal, self.need)
            if self.pages.fits(self.need):
                break
            if not self.evict_warm():
                self.pages.release(sid)
                return False
        if not self.pages.reserve(sid, last_ordinal):
            self.pages.release(sid)
            return False
        var copy_start = len(self.reuse_copies)
        var partial_rows = lcp - sealed * Self.positions_per_page
        if partial_rows > 0:
            for p in range(self.pages.pool_count()):
                if self.pages.pool_fixed_pages(p) > 0:
                    continue
                self.reuse_copies.append(PageCopy(
                    p,
                    self.pages.page_index(p, donor, sealed),
                    self.pages.page_index(p, sid, sealed),
                    0, partial_rows))
        var live_start = lcp - Self.positions_per_page
        if live_start < 0:
            live_start = 0
        for p in range(self.pages.pool_count()):
            var ring_pages = self.pages.pool_fixed_pages(p)
            if ring_pages == 0:
                continue
            var a = live_start
            while a < lcp:
                var page_end = (((a // Self.positions_per_page) + 1)
                                * Self.positions_per_page)
                var seg_end = min(lcp, page_end)
                var ring_slot = (a // Self.positions_per_page) % ring_pages
                self.reuse_copies.append(PageCopy(
                    p,
                    self.pages.page_index(p, donor, ring_slot),
                    self.pages.page_index(p, sid, ring_slot),
                    a % Self.positions_per_page, seg_end - a))
                a = seg_end
        self.registry.open(sid, request_id, now)
        self.registry.seed(
            sid, Span(self.requests[request_id].tokens), lcp, now)
        self.ring_floor[sid] = live_start
        self.requests[request_id].seq_id = sid
        self.reuse.append(PrefixReuse(
            sid, False, 0, copy_start,
            len(self.reuse_copies) - copy_start))
        return True

    def preempt_last_slot(mut self, now: UInt):
        var dropped = self.schedule.slots.pop()
        for _ in range(dropped.n_tokens):
            _ = self.schedule.tokens.pop()
        var sid = dropped.seq_id
        var owner = self.registry.owner_of(sid)
        var adopted_record = -1
        for k in range(len(self.reuse)):
            if self.reuse[k].seq_id != sid:
                continue
            if self.reuse[k].adopted:
                adopted_record = k
            self.reuse[k].seq_id = -1
        if adopted_record >= 0:
            self.registry.seed(
                sid, Span(self.requests[owner].tokens),
                self.reuse[adopted_record].trim_len, now)
            self.registry.set_warm(sid)
        else:
            self.pages.release(sid)
            self.registry.close(sid)
        self.requests[owner].seq_id = -1

    def unwind_prefix_reuse(mut self, now: UInt):
        for k in range(len(self.reuse)):
            var sid = self.reuse[k].seq_id
            if sid < 0:
                continue
            var owner = self.registry.owner_of(sid)
            if self.reuse[k].adopted:
                if owner >= 0:
                    self.requests[owner].seq_id = -1
                    self.registry.seed(
                        sid, Span(self.requests[owner].tokens),
                        self.reuse[k].trim_len, now)
                self.registry.set_warm(sid)
            else:
                if owner >= 0:
                    self.requests[owner].seq_id = -1
                self.pages.release(sid)
                self.registry.close(sid)
        self.reuse.clear()
        self.reuse_copies.clear()
        self.release_copy_pins()

    def build_schedule(mut self, now: UInt) -> Bool:
        while (self.first_pending < len(self.requests)
               and self.requests[self.first_pending].done):
            self.first_pending += 1
        self.reuse.clear()
        self.reuse_copies.clear()

        for i in range(self.first_pending, len(self.requests)):
            if self.requests[i].done or self.requests[i].seq_id >= 0:
                continue
            var warm_match = self.best_reusable(i, False)
            var active_match = self.best_reusable(i, True)
            var warm_sid = warm_match[0]
            var warm_lcp = warm_match[1]
            var active_sid = active_match[0]
            var active_lcp = active_match[1]
            var can_adopt = warm_sid >= 0 and warm_lcp >= 1
            var can_fork = (active_sid >= 0
                            and active_lcp >= Self.positions_per_page)
            var matched = False
            if can_adopt and (not can_fork
                              or warm_lcp + Self.positions_per_page
                              > active_lcp):
                matched = self.adopt_prefix(warm_sid, i, warm_lcp, now)
                if not matched and can_fork:
                    matched = self.fork_prefix(active_sid, i, active_lcp, now)
            elif can_fork:
                matched = self.fork_prefix(active_sid, i, active_lcp, now)
                if not matched and can_adopt:
                    matched = self.adopt_prefix(warm_sid, i, warm_lcp, now)
            if matched:
                continue
            var sid = self.pages.admit()
            if sid < 0:
                if not self.evict_warm():
                    continue
                sid = self.pages.admit()
                if sid < 0:
                    continue
            self.registry.open(sid, i, now)
            self.ring_floor[sid] = 0
            self.requests[i].seq_id = sid

        self.schedule.clear()
        var budget = self.step_budget
        for i in range(self.first_pending, len(self.requests)):
            if self.requests[i].done or self.requests[i].seq_id < 0:
                continue
            if len(self.schedule.slots) >= self.max_slots or budget < 1:
                break
            var sid = self.requests[i].seq_id
            var base_pos = self.registry.length(sid)
            if base_pos >= len(self.requests[i].tokens):
                base_pos = len(self.requests[i].tokens) - 1
            var want = len(self.requests[i].tokens) - base_pos
            var feed = min(want, budget)
            for j in range(self.first_pending, len(self.requests)):
                if (j == i or self.requests[j].done
                        or self.requests[j].seq_id >= 0):
                    continue
                var common = common_prefix_len(
                    self.requests[i].tokens, self.requests[j].tokens)
                if common < Self.positions_per_page:
                    continue
                var limit = common + Self.positions_per_page - base_pos
                if limit >= 1 and limit < feed:
                    feed = limit
            var emit = (base_pos + feed == len(self.requests[i].tokens))
            for t in range(feed):
                self.schedule.tokens.append(
                    self.requests[i].tokens[base_pos + t])
            self.schedule.slots.append(BatchSlot(
                sid, base_pos, feed, emit, self.requests[i].sampling))
            budget -= feed

        while len(self.schedule.slots) > 0:
            for p in range(len(self.need)):
                self.need[p] = 0
            for s in range(len(self.schedule.slots)):
                var last_pos = (self.schedule.slots[s].base_pos
                                + self.schedule.slots[s].n_tokens - 1)
                self.pages.pages_needed(
                    self.schedule.slots[s].seq_id,
                    last_pos // Self.positions_per_page, self.need)
            if self.pages.fits(self.need):
                break
            if self.evict_warm():
                continue
            if len(self.schedule.slots) == 1:
                self.unwind_prefix_reuse(now)
                return False
            self.preempt_last_slot(now)

        if len(self.schedule.slots) == 0:
            self.unwind_prefix_reuse(now)
            return False

        for s in range(len(self.schedule.slots)):
            var last_pos = (self.schedule.slots[s].base_pos
                            + self.schedule.slots[s].n_tokens - 1)
            if not self.pages.reserve(
                    self.schedule.slots[s].seq_id,
                    last_pos // Self.positions_per_page):
                self.unwind_prefix_reuse(now)
                return False

        for s in range(len(self.schedule.slots)):
            var sid = self.schedule.slots[s].seq_id
            var first_ordinal = (self.schedule.slots[s].base_pos
                                 // Self.positions_per_page)
            var last_ordinal = ((self.schedule.slots[s].base_pos
                                 + self.schedule.slots[s].n_tokens - 1)
                                // Self.positions_per_page)
            for p in range(self.pages.pool_count()):
                if self.pages.pool_fixed_pages(p) > 0:
                    continue
                for ordinal in range(first_ordinal, last_ordinal + 1):
                    debug_assert(
                        self.pages.page_holds(
                            p, self.pages.page_index(p, sid, ordinal)) == 1,
                        "schedule: write range crosses a held page",
                    )

        for k in range(len(self.reuse)):
            if self.reuse[k].seq_id < 0:
                continue
            var copy_end = self.reuse[k].copy_start + self.reuse[k].copy_count
            for c in range(self.reuse[k].copy_start, copy_end):
                self.schedule.copies.append(self.reuse_copies[c])
        self.release_copy_pins()
        return True

    def absorb(
        mut self,
        read outs: List[SampleOutcome[MAXIMUM_SAMPLING_LOGITS]],
        now: UInt,
    ):
        var buf_start = 0
        var emit_idx = 0
        for s in range(len(self.schedule.slots)):
            var seq_id = self.schedule.slots[s].seq_id
            var base_pos = self.schedule.slots[s].base_pos
            var fed = self.schedule.slots[s].n_tokens
            var request_id = self.registry.owner_of(seq_id)
            self.registry.extend(
                seq_id, Span(self.schedule.tokens),
                buf_start, base_pos, fed, now)
            buf_start += fed
            if self.schedule.slots[s].emit:
                var token = outs[emit_idx].token_id
                emit_idx += 1
                self.requests[request_id].generated.append(token)
                self.requests[request_id].tokens.append(token)
                if (token == self.eos_token
                    or len(self.requests[request_id].generated)
                        >= self.requests[request_id].max_new_tokens):
                    self.requests[request_id].done = True
                    self.registry.set_warm(seq_id)

    def step[M: ScheduledModel, //](mut self, mut model: M) -> Int:
        comptime assert M.POSITIONS_PER_PAGE == Self.positions_per_page, (
            "scheduler page granularity must match the model's")
        if not self.build_schedule(perf_counter_ns()):
            return 0
        var outs = model.execute(self.schedule, self.pages)
        self.absorb(outs, perf_counter_ns())
        return len(self.schedule.slots)
