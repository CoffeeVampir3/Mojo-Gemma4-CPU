from std.pathlib import Path
from std.time import perf_counter_ns

from numa import NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from tokenizer import load_tokenizer, BPETokenizer, AutoPreTokenizer, AutoByteTransform
from modeling.gemma_4_moe import Gemma4
from modeling.gemma4_common import Gemma4BaseConfig
from kernels.flash_sample import SamplingParams
from continuous_batching.schedule import MAXIMUM_SAMPLING_LOGITS
from continuous_batching.scheduler import ContinuousBatchScheduler


comptime TOKENIZER_PATH = "checkpoints/gemma-4-26B-A4B/tokenizer.json"
comptime MODEL_DIR = "checkpoints/gemma-4-26B-A4B"
comptime MAX_NEW_TOKENS = 24
comptime STEP_BUDGET = Gemma4BaseConfig.SLIDING_WINDOW
comptime BOS_TOKEN_ID = 2
comptime EOS_TOKEN_ID = 1


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

    var greedy = SamplingParams(
        Float32(1.0), Float32(0.0), 0, MAXIMUM_SAMPLING_LOGITS, True)
    var sched = ContinuousBatchScheduler[
        Gemma4[profile=True, Pool=P].POSITIONS_PER_PAGE,
    ](model.batch_geometry(), STEP_BUDGET, Int32(EOS_TOKEN_ID))

    var K = len(prompts)
    for k in range(K):
        var toks = List[Int32]()
        for t in range(len(prompts[k])):
            toks.append(prompts[k][t])
        _ = sched.submit(toks^, greedy, MAX_NEW_TOKENS)

    var step_t0 = perf_counter_ns()
    var steps = 0
    while sched.pending_work():
        if sched.step(model) == 0:
            break
        steps += 1
    var step_ms = elapsed_ms_since(step_t0)
    model.profiler.report("continuous batching")
    print()
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

    var steps2 = 0
    while sched.pending_work():
        if sched.step(model) == 0:
            break
        steps2 += 1
    print(t"continuation: {steps2} steps (warm-prefix reuse)")
    print()
    print(decode_int32(tok, sched.requests[turn2_id].tokens))
    print()


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
