from kernels.flash_sample import SamplingParams, SampleOutcome
from continuous_batching.schedule import (
    Schedule, ScheduledModel, MAXIMUM_SAMPLING_LOGITS,
)
from continuous_batching.paging import (
    KVPageAccountant, BatchGeometry, PagePoolSpec,
)
from continuous_batching.scheduler import ContinuousBatchScheduler


comptime PPP = 4
comptime RING_POOL = 0
comptime GROW_POOL = 1
comptime RING_PAGES = 8
comptime GROW_PAGES = 16


struct StubModel(Movable, ScheduledModel):
    comptime POSITIONS_PER_PAGE = PPP
    var steps: Int
    var fed_tokens: Int
    var ring_copy_positions: Int
    var grow_copy_positions: Int
    var next_token: Int32

    def __init__(out self):
        self.steps = 0
        self.fed_tokens = 0
        self.ring_copy_positions = 0
        self.grow_copy_positions = 0
        self.next_token = Int32(9000)

    def batch_geometry(self) -> BatchGeometry:
        var pools = List[PagePoolSpec]()
        pools.append(PagePoolSpec(
            num_pages=RING_PAGES, fixed_pages_per_seq=2, max_pages_per_seq=2))
        pools.append(PagePoolSpec(
            num_pages=GROW_PAGES, fixed_pages_per_seq=0,
            max_pages_per_seq=GROW_PAGES))
        return BatchGeometry(
            max_seqs=4, max_slots=4, max_step_tokens=8, pools=pools^)

    def execute(
        mut self,
        read schedule: Schedule,
        read pages: KVPageAccountant,
    ) -> List[SampleOutcome[MAXIMUM_SAMPLING_LOGITS]]:
        self.steps += 1
        self.fed_tokens += len(schedule.tokens)
        for c in range(len(schedule.copies)):
            if schedule.copies[c].pool == RING_POOL:
                self.ring_copy_positions += schedule.copies[c].pos_count
            else:
                self.grow_copy_positions += schedule.copies[c].pos_count
        var outs = List[SampleOutcome[MAXIMUM_SAMPLING_LOGITS]]()
        for s in range(len(schedule.slots)):
            if schedule.slots[s].emit:
                var out = SampleOutcome[MAXIMUM_SAMPLING_LOGITS]()
                out.token_id = self.next_token
                self.next_token += 1
                outs.append(out)
        return outs^


def check(cond: Bool, msg: String) -> Int:
    if cond:
        print("  ok  -", msg)
        return 0
    print("  FAIL-", msg)
    return 1


def run_until_idle(
    mut sched: ContinuousBatchScheduler[PPP], mut model: StubModel,
) -> Int:
    var steps = 0
    while sched.pending_work():
        if sched.step(model) == 0:
            break
        steps += 1
    return steps


def main():
    var failures = 0
    var model = StubModel()
    var greedy = SamplingParams(
        Float32(1.0), Float32(0.0), 0, MAXIMUM_SAMPLING_LOGITS, True)
    var sched = ContinuousBatchScheduler[PPP](
        model.batch_geometry(), 8, Int32(-1))

    print("--- fresh prefill ---")
    var prompt_f = List[Int32]()
    for t in range(9):
        prompt_f.append(Int32(100 + t))
    var rf = sched.submit(prompt_f^, greedy, 1)
    var steps = run_until_idle(sched, model)
    failures += check(steps == 2, "9-token prompt prefills in two steps")
    failures += check(model.fed_tokens == 9, "prefill feeds exactly the prompt")
    failures += check(sched.requests[rf].done, "request completes")
    var sid_f = sched.requests[rf].seq_id

    print("--- adoption (take) and fork (point + copy) ---")
    var prompt_k = List[Int32]()
    for t in range(len(sched.requests[rf].tokens)):
        prompt_k.append(sched.requests[rf].tokens[t])
    prompt_k.append(Int32(111))
    var rk = sched.submit(prompt_k^, greedy, 1)
    var prompt_l = List[Int32]()
    for t in range(9):
        prompt_l.append(Int32(100 + t))
    prompt_l.append(Int32(222))
    var rl = sched.submit(prompt_l^, greedy, 1)

    var fed_before = model.fed_tokens
    steps = run_until_idle(sched, model)
    failures += check(steps == 1, "adoption and fork settle in one step")
    failures += check(
        model.fed_tokens - fed_before == 3,
        "21 requested tokens cost 3 fed tokens")
    failures += check(
        sched.requests[rk].seq_id == sid_f,
        "warm sequence is donated to the continuation")
    var sid_l = sched.requests[rl].seq_id
    failures += check(sid_l != sid_f, "fork gets its own sequence")
    failures += check(
        sched.pages.page_index(GROW_POOL, sid_f, 0)
        == sched.pages.page_index(GROW_POOL, sid_l, 0),
        "fork points at the donor's first sealed page")
    failures += check(
        sched.pages.page_index(GROW_POOL, sid_f, 1)
        == sched.pages.page_index(GROW_POOL, sid_l, 1),
        "fork points at the donor's second sealed page")
    failures += check(
        sched.pages.page_holds(
            GROW_POOL, sched.pages.page_index(GROW_POOL, sid_f, 0)) == 2,
        "shared sealed page carries two holds")
    failures += check(
        sched.pages.page_index(GROW_POOL, sid_f, 2)
        != sched.pages.page_index(GROW_POOL, sid_l, 2),
        "divergence pages stay private")
    failures += check(
        model.grow_copy_positions == 1,
        "divergence copy moves one position")
    failures += check(
        model.ring_copy_positions == 4,
        "ring copy moves exactly the live window")

    print("--- adoption with truncation privatizes a shared page ---")
    var prompt_m = List[Int32]()
    for t in range(7):
        prompt_m.append(Int32(100 + t))
    prompt_m.append(Int32(333))
    var rm = sched.submit(prompt_m^, greedy, 1)
    fed_before = model.fed_tokens
    var grow_before = model.grow_copy_positions
    var ring_before = model.ring_copy_positions
    steps = run_until_idle(sched, model)
    failures += check(steps == 1, "edited retry settles in one step")
    failures += check(
        sched.requests[rm].seq_id == sid_f,
        "intact warm sequence wins over the stale fork")
    failures += check(
        model.fed_tokens - fed_before == 1,
        "8 requested tokens cost 1 fed token")
    failures += check(
        model.grow_copy_positions - grow_before == 3,
        "privatization copies the kept rows of the shared page")
    failures += check(
        model.ring_copy_positions == ring_before,
        "adoption never copies the ring")
    failures += check(
        sched.pages.page_index(GROW_POOL, sid_f, 1)
        != sched.pages.page_index(GROW_POOL, sid_l, 1),
        "divergence page was replaced with a private copy")
    failures += check(
        sched.pages.page_holds(
            GROW_POOL, sched.pages.page_index(GROW_POOL, sid_f, 1)) == 1,
        "private replacement has a single hold")
    failures += check(
        sched.pages.page_holds(
            GROW_POOL, sched.pages.page_index(GROW_POOL, sid_l, 1)) == 1,
        "copy source returns to a single hold once unpinned")

    print("--- hold balance after eviction ---")
    while sched.evict_warm():
        pass
    failures += check(
        sched.pages.pool_available(RING_POOL) == RING_PAGES,
        "ring pool drains to empty")
    failures += check(
        sched.pages.pool_available(GROW_POOL) == GROW_PAGES,
        "growing pool drains to empty")

    print()
    if failures == 0:
        print("RESULT: PASS -- prefix holds, donation, and copies hold")
    else:
        print(t"RESULT: FAIL -- {failures} check(s)")
