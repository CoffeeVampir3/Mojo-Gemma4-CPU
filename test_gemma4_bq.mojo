from std.memory import Span, UnsafePointer
from std.pathlib import Path
from std.time import perf_counter_ns

from numa import NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from tokenizer import load_tokenizer, BPETokenizer, AutoPreTokenizer, AutoByteTransform
from modeling.gemma4_common import Gemma4BaseConfig
from modeling.gemma_4_moe_bq import Gemma4


comptime TOKENIZER_PATH = "checkpoints/gemma-4-26B-A4B/tokenizer.json"
comptime MODEL_DIR = "checkpoints/gemma-4-26B-A4B-bq"


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

    var t1 = perf_counter_ns()
    model.forward(
        Span[Int32, origin_of(tok_buf)](
            ptr=tok_buf.unsafe_ptr(), length=prompt_len),
        0)
    var embed_ms = (perf_counter_ns() - t1) / 1_000_000
    print(t"embed | {prompt_len} tokens | {embed_ms} ms")


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
    def dispatch_gemma4_bq_tp[
        P: BurstThreadPool, //, degree: Int,
    ](var selected_pools: List[P]):
        load_and_run[degree=degree](topo, selected_pools^, tok, token_ids)

    with_topological_rank_dispatch[
        power_of_two_unrolling=3,
        dispatch=dispatch_gemma4_bq_tp,
    ](
        topo, "mode: isolated (spin-only)", "mode: cold (spin-backoff)")
