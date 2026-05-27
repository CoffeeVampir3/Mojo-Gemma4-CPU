from std.memory import Span, UnsafePointer
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


def greedy_argmax[degree: Int](
    read view: TemporalLogitsView[VOCAB, degree],
) -> Tuple[Int, Float32]:
    comptime width = simd_width_of[DType.float32]()
    var best_val = Float32(-1e30)
    var best_idx = 0

    for j in range(0, VOCAB, width):
        var v = view.load_f32[width](j)
        for k in range(width):
            if v[k] > best_val:
                best_val = v[k]
                best_idx = j + k

    return (best_idx, best_val)


def load_and_run[
    P: BurstThreadPool, //, degree: Int,
](
    topo: NumaTopology,
    var pools: List[P],
    read tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read token_ids: List[Int],
):
    var t0 = perf_counter_ns()
    var model_opt = Gemma4[degree=degree, Pool=P].load(
        Path(MODEL_DIR), topo, pools^)
    if not model_opt:
        return
    var model = model_opt.take()
    var load_ms = (perf_counter_ns() - t0) / 1_000_000
    print(t"model loaded in {load_ms} ms")
    print()

    var prompt_len = len(token_ids)

    var tok_buf = List[Int32](capacity=prompt_len)
    for i in range(prompt_len):
        tok_buf.append(Int32(token_ids[i]))

    var generated = List[Int]()
    var next_id = 0
    var pos = 0
    var prefill_ms = UInt(0)
    var decode_start = perf_counter_ns()

    while len(generated) < MAX_NEW_TOKENS:
        var logits: TemporalLogitsView[VOCAB, degree]
        if pos == 0:
            var t1 = perf_counter_ns()
            logits = model.forward(
                Span[Int32, origin_of(tok_buf)](
                    ptr=tok_buf.unsafe_ptr(), length=prompt_len),
                0)
            prefill_ms = (perf_counter_ns() - t1) / 1_000_000
            pos = prompt_len
            decode_start = perf_counter_ns()
        else:
            var step_id = Int32(next_id)
            logits = model.forward(
                Span[Int32, origin_of(step_id)](
                    ptr=UnsafePointer(to=step_id), length=1),
                pos)
            pos += 1

        next_id = greedy_argmax[degree](logits)[0]
        logits^.release()
        generated.append(next_id)

        if next_id == 1:
            break

    var prefill_tps = Int(Float64(prompt_len) / (Float64(prefill_ms) / 1000.0))
    print(t"prompt  | {prompt_len} tokens | {prefill_ms} ms | {prefill_tps} t/s")

    var decode_elapsed_ms = (perf_counter_ns() - decode_start) / 1_000_000
    var decode_tokens = len(generated) - 1
    var decode_tps = Int(Float64(decode_tokens) / (Float64(decode_elapsed_ms) / 1000.0))
    print(t"decode  | {decode_tokens} tokens | {decode_elapsed_ms} ms | {decode_tps} t/s")

    var all_ids = List[Int]()
    for i in range(len(token_ids)):
        all_ids.append(token_ids[i])
    for i in range(len(generated)):
        all_ids.append(generated[i])

    var full_text = tok.decode(all_ids)
    print()
    var n_generated = len(generated)
    print(t"=== generated {n_generated} tokens ===")
    print(full_text)


def main():
    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print(t"failed to load tokenizer from {TOKENIZER_PATH}")
        return
    var tok = tok_opt.take()

    var prompt = """Meepington 5000"""


    var token_ids = List[Int]()
    token_ids.append(2)  # <bos>
    var encoded = tok.encode(prompt)
    for i in range(len(encoded)):
        token_ids.append(encoded[i])
    var prompt_repr = repr(prompt)
    print(t"prompt: {prompt_repr}")
    var n_tokens = len(token_ids)
    print(t"tokens: {n_tokens} ids:", end="")
    for i in range(len(token_ids)):
        print("", token_ids[i], end="")
    print()

    var topo = NumaTopology()
    var nodes = topo.num_nodes()
    var iso = len(topo.isolated_cpus)
    print(t"{nodes} NUMA nodes, {iso} isolated cpus")

    @parameter
    def dispatch_gemma4_tp[
        P: BurstThreadPool, //, degree: Int,
    ](var selected_pools: List[P]):
        load_and_run[degree=degree](topo, selected_pools^, tok, token_ids)

    with_topological_rank_dispatch[
        power_of_two_unrolling=3,
        dispatch=dispatch_gemma4_tp,
    ](
        topo, "mode: isolated (spin-only)", "mode: cold (spin-backoff)")
