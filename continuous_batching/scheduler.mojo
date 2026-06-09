from std.memory import Span
from std.time import perf_counter_ns

from kernels.flash_sample import SamplingParams, SampleOutcome

from .schedule import (
    Schedule, BatchSlot, ScheduledModel, MAXIMUM_SAMPLING_LOGITS,
)
from .paging import KVPageAccountant, BatchGeometry
from .slot_registry import SlotRegistry


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

    def preempt_last_slot(mut self):
        var dropped = self.schedule.slots.pop()
        for _ in range(dropped.n_tokens):
            _ = self.schedule.tokens.pop()
        var owner = self.registry.owner_of(dropped.seq_id)
        self.pages.release(dropped.seq_id)
        self.registry.close(dropped.seq_id)
        self.requests[owner].seq_id = -1

    def build_schedule(mut self, now: UInt) -> Bool:
        while (self.first_pending < len(self.requests)
               and self.requests[self.first_pending].done):
            self.first_pending += 1

        for i in range(self.first_pending, len(self.requests)):
            if self.requests[i].done or self.requests[i].seq_id >= 0:
                continue
            var target = self.registry.match_append(
                Span(self.requests[i].tokens))
            if target >= 0:
                self.registry.adopt(target, i, now)
                self.requests[i].seq_id = target
                continue
            var sid = self.pages.admit()
            if sid < 0:
                if not self.evict_warm():
                    continue
                sid = self.pages.admit()
                if sid < 0:
                    continue
            self.registry.open(sid, i, now)
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
                return False
            self.preempt_last_slot()

        if len(self.schedule.slots) == 0:
            return False

        for s in range(len(self.schedule.slots)):
            var last_pos = (self.schedule.slots[s].base_pos
                            + self.schedule.slots[s].n_tokens - 1)
            if not self.pages.reserve(
                    self.schedule.slots[s].seq_id,
                    last_pos // Self.positions_per_page):
                return False
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
