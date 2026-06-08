from std.memory import Span
from std.sys.info import simd_width_of
from std.pathlib import Path
from std.time import perf_counter_ns

from numa import NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from tokenizer import load_tokenizer, BPETokenizer, AutoPreTokenizer, AutoByteTransform
from modeling.gemma4_common import Gemma4BaseConfig
from modeling.gemma_4_moe import Gemma4
from modeling.temporal_scratch import TemporalLogitsView


comptime TOKENIZER_PATH = "checkpoints/gemma-4-26B-A4B/tokenizer.json"
comptime MODEL_DIR = "checkpoints/gemma-4-26B-A4B"
comptime VOCAB = Gemma4BaseConfig.VOCAB_SIZE
comptime MAX_NEW_TOKENS = 128
comptime BOS_TOKEN_ID = 2
comptime EOS_TOKEN_ID = 1


def elapsed_ms_since(start_ns: UInt) -> Int:
    return Int((perf_counter_ns() - start_ns) / 1_000_000)


def tokens_per_second(token_count: Int, elapsed_ms: Int) -> Int:
    if elapsed_ms == 0:
        return 0
    return token_count * 1000 // elapsed_ms


def encode_prompt(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    prompt: String,
) -> List[Int]:
    var token_ids = List[Int]()
    token_ids.append(BOS_TOKEN_ID)
    var encoded = tok.encode(prompt)
    for i in range(len(encoded)):
        token_ids.append(encoded[i])
    return token_ids^


def int32_tokens(read token_ids: List[Int]) -> List[Int32]:
    var out = List[Int32](capacity=len(token_ids))
    for i in range(len(token_ids)):
        out.append(Int32(token_ids[i]))
    return out^


def merged_ids(read prompt_ids: List[Int], read generated_ids: List[Int]) -> List[Int]:
    var out = List[Int](capacity=len(prompt_ids) + len(generated_ids))
    for i in range(len(prompt_ids)):
        out.append(prompt_ids[i])
    for i in range(len(generated_ids)):
        out.append(generated_ids[i])
    return out^


def print_prompt(prompt: String, read token_ids: List[Int]):
    var prompt_repr = repr(prompt)
    print(t"prompt: {prompt_repr}")
    var n_tokens = len(token_ids)
    print(t"tokens: {n_tokens} ids:", end="")
    for i in range(n_tokens):
        print("", token_ids[i], end="")
    print()


def greedy_next_token(
    read view: TemporalLogitsView[VOCAB],
) -> Int:
    comptime width = simd_width_of[DType.float32]()
    var best_val: Float32 = -1e30
    var best_idx = 0

    for j in range(0, VOCAB, width):
        var values = view.load_f32[width](j)
        var local_best = values.reduce_max()[0]
        if local_best > best_val:
            best_val = local_best
            for k in range(width):
                if values[k] == local_best:
                    best_idx = j + k
                    break

    return best_idx


def load_and_run[
    P: BurstThreadPool, //,
](
    topo: NumaTopology,
    var pools: List[P],
    read tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read token_ids: List[Int],
):
    var t0 = perf_counter_ns()
    var model_opt = Gemma4[profile=True, Pool=P].load(
        Path(MODEL_DIR), topo, pools^)
    if not model_opt:
        return
    var model = model_opt.take()
    var load_ms = elapsed_ms_since(t0)
    print(t"model loaded in {load_ms} ms")
    print()

    var seq = model.admit_sequence()
    if seq < 0:
        print("sequence admission failed")
        return

    var prompt_len = len(token_ids)

    var tok_buf = int32_tokens(token_ids)
    var step_buf = List[Int32](capacity=1)
    step_buf.append(0)

    var generated = List[Int]()
    var next_id = 0
    var pos = 0
    var prefill_ms = 0
    var decode_start = perf_counter_ns()

    while len(generated) < MAX_NEW_TOKENS:
        var logits: TemporalLogitsView[VOCAB]
        if pos == 0:
            var t1 = perf_counter_ns()
            logits = model.forward(seq, Span(tok_buf), 0)
            prefill_ms = elapsed_ms_since(t1)
            pos = prompt_len
            model.profiler.report("prefill")
            model.profiler.reset()
            decode_start = perf_counter_ns()
        else:
            step_buf[0] = Int32(next_id)
            logits = model.forward(seq, Span(step_buf), pos)
            pos += 1

        next_id = greedy_next_token(logits)
        generated.append(next_id)

        if next_id == EOS_TOKEN_ID:
            break

    model.profiler.report("decode")

    var prefill_tps = tokens_per_second(prompt_len, prefill_ms)
    print(t"prompt  | {prompt_len} tokens | {prefill_ms} ms | {prefill_tps} t/s")

    var decode_elapsed_ms = elapsed_ms_since(decode_start)
    var decode_tokens = len(generated) - 1
    var decode_tps = tokens_per_second(decode_tokens, decode_elapsed_ms)
    print(t"decode  | {decode_tokens} tokens | {decode_elapsed_ms} ms | {decode_tps} t/s")

    var all_ids = merged_ids(token_ids, generated)
    var full_text = tok.decode(all_ids)
    print()
    var n_generated = len(generated)
    print(t"=== generated {n_generated} tokens ===")
    print(full_text)


def main():
    print("Launching.")
    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print(t"failed to load tokenizer from {TOKENIZER_PATH}")
        return
    var tok = tok_opt.take()

    var prompt = """The Roman aqueducts were a system of engineering structures built by the ancient Romans to transport water from distant sources into cities and towns. Constructed from a combination of stone, brick, and a special volcanic cement known as pozzolana, these channels supplied public baths, latrines, fountains, and private households across the empire. The water flowed largely by gravity alone, descending along a very gentle gradient maintained over distances that sometimes exceeded a hundred kilometres.

The earliest aqueduct in Rome, the Aqua Appia, was commissioned in 312 BC and ran almost entirely underground to protect it from contamination and enemy sabotage. As the city's population grew, later aqueducts such as the Aqua Marcia and the Aqua Claudia carried far greater volumes and rose onto towering arched bridges where the terrain dipped. At the height of the empire, eleven major aqueducts served the capital, together delivering an estimated one million cubic metres of water each day.

Beyond Rome itself, provincial cities throughout Gaul, Hispania, and North Africa built their own aqueducts, many of which still stand today. The Pont du Gard in southern France and the aqueduct of Segovia in Spain remain among the best preserved, their multi-tiered arches a testament to the durability of Roman construction. Maintenance was the responsibility of a dedicated office, and a permanent staff of workers inspected the channels, cleared sediment, and repaired leaks.

The decline of the aqueduct network paralleled the broader collapse of Roman administrative power in the West. As central authority weakened, the resources and expertise needed to maintain the channels disappeared, and many fell into disrepair or were deliberately cut during sieges. Nevertheless, the underlying principles of gradient flow and durable masonry influenced water engineering for centuries, and several aqueducts were restored and returned to service during the Renaissance."""
    var token_ids = encode_prompt(tok, prompt)
    print_prompt(prompt, token_ids)

    var topo = NumaTopology()
    var nodes = topo.num_nodes()
    var iso = len(topo.isolated_cpus)
    print(t"{nodes} NUMA nodes, {iso} isolated cpus")

    @parameter
    def dispatch_gemma4_tp[
        P: BurstThreadPool, //,
    ](var selected_pools: List[P]):
        load_and_run(topo, selected_pools^, tok, token_ids)

    with_topological_rank_dispatch[
        dispatch=dispatch_gemma4_tp,
    ](
        topo, "mode: isolated (spin-only)", "mode: cold (spin-backoff)")
