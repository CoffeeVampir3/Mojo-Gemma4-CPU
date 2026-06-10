from std.pathlib import Path
from std.time import perf_counter_ns

from numa import NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from tokenizer import load_tokenizer, BPETokenizer, AutoPreTokenizer, AutoByteTransform
from modeling.gemma_4_moe import Gemma4, FULL_POOL
from modeling.gemma4_common import Gemma4BaseConfig
from kernels.flash_sample import SamplingParams
from continuous_batching.schedule import (
    ScheduledModel, MAXIMUM_SAMPLING_LOGITS,
)
from continuous_batching.scheduler import ContinuousBatchScheduler


comptime TOKENIZER_PATH = "checkpoints/gemma-4-26B-A4B/tokenizer.json"
comptime MODEL_DIR = "checkpoints/gemma-4-26B-A4B"
comptime MAX_NEW_TOKENS = 24
comptime LARGE_MAX_NEW = 8
comptime STEP_BUDGET = Gemma4BaseConfig.SLIDING_WINDOW
comptime BOS_TOKEN_ID = 2
comptime EOS_TOKEN_ID = 1

comptime BASE_A_LEN = 3500
comptime FORK_A_AT = 3000
comptime EDIT_A_AT = 3490
comptime STALE_AT = 1500
comptime BASE_C_LEN = 2600
comptime FORK_C_AT = 2400
comptime ADOPT_C_AT = 2000


def elapsed_ms_since(start_ns: UInt) -> Int:
    return Int((perf_counter_ns() - start_ns) / 1_000_000)


def encode_prompt(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    prompt: String,
) -> List[Int32]:
    var token_ids = List[Int32]()
    token_ids.append(Int32(BOS_TOKEN_ID))
    var encoded = tok.encode(prompt)
    for i in range(len(encoded)):
        token_ids.append(Int32(encoded[i]))
    return token_ids^


def decode_int32(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read ids: List[Int32],
) -> String:
    var as_int = List[Int](capacity=len(ids))
    for i in range(len(ids)):
        as_int.append(Int(ids[i]))
    return tok.decode(as_int)


def synth_prompt(count: Int, salt: Int) -> List[Int32]:
    var toks = List[Int32]()
    toks.append(Int32(BOS_TOKEN_ID))
    for t in range(count - 1):
        toks.append(Int32(1000 + ((t * 7 + salt) % 900)))
    return toks^


def prefix_plus_tail(
    read base: List[Int32], prefix_len: Int, tail_salt: Int, tail_len: Int,
) -> List[Int32]:
    var toks = List[Int32]()
    for t in range(prefix_len):
        toks.append(base[t])
    for t in range(tail_len):
        toks.append(Int32(tail_salt + t))
    return toks^


def check(cond: Bool, msg: String) -> Int:
    if cond:
        print(t"  ok  - {msg}")
        return 0
    print(t"  FAIL- {msg}")
    return 1


def generation_mismatches(
    read got: List[Int32], read want: List[Int32],
) -> Int:
    if len(got) != len(want):
        return 1 + abs(len(got) - len(want))
    var mismatches = 0
    for t in range(len(want)):
        if got[t] != want[t]:
            mismatches += 1
    return mismatches


def run_steps[
    M: ScheduledModel, pp: Int, //,
](
    mut sched: ContinuousBatchScheduler[pp],
    mut model: M,
    label: String,
    max_steps: Int = 0,
) -> Int:
    var steps = 0
    while sched.pending_work():
        if max_steps > 0 and steps >= max_steps:
            break
        var t0 = perf_counter_ns()
        var slots = sched.step(model)
        var us = Int((perf_counter_ns() - t0) / 1_000)
        if slots == 0:
            break
        steps += 1
        var fed = len(sched.schedule.tokens)
        var copies = len(sched.schedule.copies)
        print(t"  [{label}] step {steps}: {slots} slot(s) {fed} tok {copies} copies {us} us")
    return steps


def drop_warm[pp: Int, //](mut sched: ContinuousBatchScheduler[pp]):
    while sched.evict_warm():
        pass


def run_scheduler[
    P: BurstThreadPool, //,
](
    topo: NumaTopology,
    var pools: List[P],
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read prompts: List[List[Int32]],
):
    var t0 = perf_counter_ns()
    var model_opt = Gemma4[profile=True, Pool=P].load(Path(MODEL_DIR), topo, pools^)
    if not model_opt:
        print("model load failed")
        return
    var model = model_opt.take()
    print(t"model loaded in {elapsed_ms_since(t0)} ms")
    print()

    var failures = 0
    var greedy = SamplingParams(
        Float32(1.0), Float32(0.0), 0, MAXIMUM_SAMPLING_LOGITS, True)
    var sched = ContinuousBatchScheduler[
        Gemma4[profile=True, Pool=P].POSITIONS_PER_PAGE,
    ](model.batch_geometry(), STEP_BUDGET, Int32(EOS_TOKEN_ID))

    print("=== phase 1: smoke (small prompts, warm continuation) ===")
    var K = len(prompts)
    for k in range(K):
        var toks = List[Int32]()
        for t in range(len(prompts[k])):
            toks.append(prompts[k][t])
        _ = sched.submit(toks^, greedy, MAX_NEW_TOKENS)

    var step_t0 = perf_counter_ns()
    var steps = run_steps(sched, model, String("round1"))
    var step_ms = elapsed_ms_since(step_t0)
    print(t"round 1: {K} requests | {steps} steps | {step_ms} ms")
    print()
    for k in range(K):
        var n = len(sched.requests[k].generated)
        print(t"--- request {k} ({n} new tokens) ---")
        print(decode_int32(tok, sched.requests[k].tokens))
        print()

    var turn2 = List[Int32]()
    for t in range(len(sched.requests[0].tokens)):
        turn2.append(sched.requests[0].tokens[t])
    var follow = encode_prompt(tok, String(" In one word, why?"))
    for t in range(1, len(follow)):
        turn2.append(follow[t])
    var turn2_id = sched.submit(turn2^, greedy, MAX_NEW_TOKENS)

    var cont_t0 = perf_counter_ns()
    var steps2 = run_steps(sched, model, String("cont"))
    var cont_ms = elapsed_ms_since(cont_t0)
    print(t"continuation: {steps2} steps | {cont_ms} ms (warm-prefix reuse)")
    print(decode_int32(tok, sched.requests[turn2_id].tokens))
    print()

    drop_warm(sched)

    print("=== phase 2: multi-page fork with mid-page divergence ===")
    var base_a = synth_prompt(BASE_A_LEN, 0)
    var prompt_a = List[Int32]()
    for t in range(len(base_a)):
        prompt_a.append(base_a[t])
    var req_a = sched.submit(prompt_a^, greedy, LARGE_MAX_NEW)
    _ = run_steps(sched, model, String("prefill-a"), 3)
    var sid_a = sched.requests[req_a].seq_id
    failures += check(
        sched.registry.length(sid_a) >= FORK_A_AT,
        "donor prefilled past the fork point before the fork request lands")

    var req_b = sched.submit(
        prefix_plus_tail(base_a, FORK_A_AT, 3001, 8), greedy, LARGE_MAX_NEW)
    _ = run_steps(sched, model, String("fork-b"), 1)
    var sid_b = sched.requests[req_b].seq_id
    failures += check(sid_b >= 0 and sid_b != sid_a, "fork gets its own sequence")
    failures += check(
        len(sched.requests[req_b].generated) >= 1,
        "3008-token fork emits after a single step")
    failures += check(
        sched.pages.page_index(FULL_POOL, sid_a, 0)
        == sched.pages.page_index(FULL_POOL, sid_b, 0),
        "fork points at the donor's first sealed page")
    failures += check(
        sched.pages.page_index(FULL_POOL, sid_a, 1)
        == sched.pages.page_index(FULL_POOL, sid_b, 1),
        "fork points at the donor's second sealed page")
    failures += check(
        sched.pages.page_holds(
            FULL_POOL, sched.pages.page_index(FULL_POOL, sid_a, 0)) >= 2,
        "shared sealed page carries both holds")
    failures += check(
        sched.pages.page_index(FULL_POOL, sid_a, 2)
        != sched.pages.page_index(FULL_POOL, sid_b, 2),
        "mid-page divergence page is private to the fork")
    var fork_t0 = perf_counter_ns()
    var steps3 = run_steps(sched, model, String("fork"))
    var fork_ms = elapsed_ms_since(fork_t0)
    var na = len(sched.requests[req_a].generated)
    var nb = len(sched.requests[req_b].generated)
    print(t"fork pair: {steps3} steps | {fork_ms} ms | a {na} tokens, b {nb} tokens")
    print()

    print("=== phase 3: stale-window edit falls back to fresh prefill ===")
    var req_stale = sched.submit(
        prefix_plus_tail(base_a, STALE_AT, 4001, 8), greedy, LARGE_MAX_NEW)
    _ = run_steps(sched, model, String("stale"), 1)
    var sid_stale = sched.requests[req_stale].seq_id
    failures += check(
        sid_stale != sid_a and sid_stale != sid_b,
        "deep edit beyond the sliding window starts a fresh sequence")
    failures += check(
        len(sched.requests[req_stale].generated) == 0,
        "fresh 1508-token prefill cannot finish in one step")
    _ = run_steps(sched, model, String("stale"))
    print()

    print("=== phase 4: near-tail edit adopts and truncates in place ===")
    var req_edit = sched.submit(
        prefix_plus_tail(base_a, EDIT_A_AT, 5001, 6), greedy, LARGE_MAX_NEW)
    _ = run_steps(sched, model, String("edit"), 1)
    failures += check(
        sched.requests[req_edit].seq_id == sid_a,
        "near-tail edit is donated the warm sequence")
    failures += check(
        len(sched.requests[req_edit].generated) >= 1,
        "3496-token edited retry emits after a single step")
    _ = run_steps(sched, model, String("edit"))
    print()

    print("=== phase 5: fork and edit exactness vs fresh prefill ===")
    drop_warm(sched)
    var ref_b = sched.submit(
        prefix_plus_tail(base_a, FORK_A_AT, 3001, 8), greedy, LARGE_MAX_NEW)
    var refb_t0 = perf_counter_ns()
    var refb_steps = run_steps(sched, model, String("ref-b"))
    print(t"reference b: {refb_steps} steps | {elapsed_ms_since(refb_t0)} ms")
    var mb = generation_mismatches(
        sched.requests[req_b].generated, sched.requests[ref_b].generated)
    failures += check(mb == 0, "forked output matches fresh prefill exactly")

    drop_warm(sched)
    var ref_edit = sched.submit(
        prefix_plus_tail(base_a, EDIT_A_AT, 5001, 6), greedy, LARGE_MAX_NEW)
    var refe_t0 = perf_counter_ns()
    var refe_steps = run_steps(sched, model, String("ref-edit"))
    print(t"reference edit: {refe_steps} steps | {elapsed_ms_since(refe_t0)} ms")
    var me = generation_mismatches(
        sched.requests[req_edit].generated, sched.requests[ref_edit].generated)
    failures += check(me == 0, "adopted-truncated output matches fresh prefill exactly")
    print()

    print("=== phase 6: adoption privatizes a fork-shared page ===")
    drop_warm(sched)
    var base_c = synth_prompt(BASE_C_LEN, 17)
    var prompt_c = List[Int32]()
    for t in range(len(base_c)):
        prompt_c.append(base_c[t])
    var req_c = sched.submit(prompt_c^, greedy, LARGE_MAX_NEW)
    _ = run_steps(sched, model, String("prefill-c"), 3)
    var sid_c = sched.requests[req_c].seq_id
    failures += check(
        sched.registry.length(sid_c) >= FORK_C_AT,
        "second donor prefilled past its fork point")
    var req_d = sched.submit(
        prefix_plus_tail(base_c, FORK_C_AT, 6001, 8), greedy, LARGE_MAX_NEW)
    _ = run_steps(sched, model, String("fork-d"))
    var sid_d = sched.requests[req_d].seq_id
    var shared_page1 = sched.pages.page_index(FULL_POOL, sid_c, 1)
    failures += check(
        shared_page1 == sched.pages.page_index(FULL_POOL, sid_d, 1),
        "fork shares the donor's second page")
    failures += check(
        sched.pages.page_holds(FULL_POOL, shared_page1) == 2,
        "shared page carries two holds before the deep edit")

    var req_e = sched.submit(
        prefix_plus_tail(base_c, ADOPT_C_AT, 7001, 6), greedy, LARGE_MAX_NEW)
    _ = run_steps(sched, model, String("adopt-e"), 1)
    failures += check(
        sched.requests[req_e].seq_id == sid_c,
        "deep edit inside the window adopts the intact donor")
    failures += check(
        len(sched.requests[req_e].generated) >= 1,
        "2006-token deep edit emits after a single step")
    failures += check(
        sched.pages.page_index(FULL_POOL, sid_c, 1) != shared_page1,
        "divergence inside a shared page is privatized")
    failures += check(
        sched.pages.page_index(FULL_POOL, sid_d, 1) == shared_page1,
        "fork keeps the original shared page")
    failures += check(
        sched.pages.page_holds(FULL_POOL, shared_page1) == 1,
        "copy source returns to a single hold once unpinned")
    _ = run_steps(sched, model, String("adopt-e"))

    drop_warm(sched)
    var ref_e = sched.submit(
        prefix_plus_tail(base_c, ADOPT_C_AT, 7001, 6), greedy, LARGE_MAX_NEW)
    var refp_t0 = perf_counter_ns()
    var refp_steps = run_steps(sched, model, String("ref-e"))
    print(t"reference e: {refp_steps} steps | {elapsed_ms_since(refp_t0)} ms")
    var mp = generation_mismatches(
        sched.requests[req_e].generated, sched.requests[ref_e].generated)
    failures += check(mp == 0, "privatized adoption matches fresh prefill exactly")
    print()

    model.profiler.report("continuous batching")
    print()
    if failures == 0:
        print("RESULT: PASS -- multi-page forks, tears, and adoptions hold")
    else:
        print(t"RESULT: FAIL -- {failures} check(s)")


def main():
    print("Continuous-batching scheduler")
    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print(t"failed to load tokenizer from {TOKENIZER_PATH}")
        return
    var tok = tok_opt.take()

    var raw = List[String]()
    raw.append(String("The capital of France is"))
    raw.append(String("Photosynthesis is the process by which plants"))
    raw.append(String("The largest planet in our solar system is"))
    raw.append(String("A group of wolves is known as a"))

    var prompts = List[List[Int32]]()
    for i in range(len(raw)):
        prompts.append(encode_prompt(tok, raw[i]))

    var topo = NumaTopology()

    @parameter
    def dispatch_cb[
        P: BurstThreadPool, //,
    ](var selected_pools: List[P]):
        run_scheduler(topo, selected_pools^, tok, prompts)

    with_topological_rank_dispatch[
        dispatch=dispatch_cb,
    ](topo, "mode: isolated (spin-only)", "mode: cold (spin-backoff)")
