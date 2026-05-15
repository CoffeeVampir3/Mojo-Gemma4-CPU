from std.pathlib import Path
from std.time import perf_counter_ns

from numa import NumaInfo, NumaTopology
from threading.burst_threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool
from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from modeling.gemma_4_moe import Gemma4
from tokenizer import load_gemma4_tokenizer, BPETokenizer


def print_token_ids(name: String, ids: List[Int]):
    print(name, end=" [")
    for i in range(len(ids)):
        if i > 0:
            print(", ", end="")
        print(ids[i], end="")
    print("]")


def run_generation[
    P: BurstThreadPool, //,
    degree: Int,
](
    numa: NumaInfo,
    numa_topo: NumaTopology,
    var pools: HeapMoveArray[P],
    mut tokenizer: BPETokenizer[...],
    var tokens: List[Int],
):
    var ckpt = Path("checkpoints/gemma-4-26B-A4B")
    var loaded = Gemma4[degree, P].load(ckpt, numa, numa_topo, pools^)
    if not loaded:
        print("load failed")
        return
    var model = loaded.take()

    var kv = model.new_kv_cache()
    var next_token = -1
    var prompt_len = len(tokens)
    var prefill_start = perf_counter_ns()
    for i in range(len(tokens)):
        next_token = model.forward(tokens[i], i, kv)
    var prefill_ms = (perf_counter_ns() - prefill_start) / 1_000_000
    var prefill_tps = Float64(prompt_len) / (Float64(prefill_ms) / 1000.0)
    print(
        "prompt  |", prompt_len, "tokens |",
        prefill_ms, "ms |",
        Int(prefill_tps), "t/s",
    )

    var generated = List[Int]()
    var decode_start = perf_counter_ns()
    for i in range(30):
        generated.append(next_token)
        tokens.append(next_token)
        if i + 1 < 30:
            next_token = model.forward(next_token, len(tokens) - 1, kv)
    var decode_elapsed_ms = (perf_counter_ns() - decode_start) / 1_000_000
    var decode_tokens = len(generated) - 1
    var decode_tps = Float64(decode_tokens) / (Float64(decode_elapsed_ms) / 1000.0)
    print(
        "decode  |", decode_tokens, "tokens |",
        decode_elapsed_ms, "ms |",
        Int(decode_tps), "t/s",
    )

    print_token_ids("generated tokens:", generated)

    var response_tokens = List[Int]()
    for i in range(1, len(tokens)):
        response_tokens.append(tokens[i])
    print("full response:")
    print(tokenizer.decode(response_tokens))

    _ = model^


def main():
    var tokenizer_path = Path("checkpoints/gemma-4-26B-A4B/tokenizer.json")

    var loaded_tokenizer = load_gemma4_tokenizer(tokenizer_path)
    if not loaded_tokenizer:
        print("tokenizer load failed")
        return
    var tokenizer = loaded_tokenizer.take()

    var prompt = String("The captital of france is")
    var tokens = List[Int]()
    tokens.append(2)
    var prompt_tokens = tokenizer.encode(prompt)
    for i in range(len(prompt_tokens)):
        tokens.append(prompt_tokens[i])

    print("prompt:", repr(prompt))
    print_token_ids("input tokens:", tokens)

    var numa = NumaInfo()
    var topo = numa.plan_topology(numa.num_nodes)
    var tp = numa.num_nodes

    print(String(tp) + " NUMA node(s), "
        + String(len(numa.isolated_cpus)) + " isolated cpus")

    if numa.has_isolation():
        print("mode: isolated (spin-only)")
        var pools = HeapMoveArray[IsolatedBurstPool[]](tp)
        for i in range(tp):
            pools.push(IsolatedBurstPool[].for_topology(numa, topo[i]))
            print("  node " + String(topo[i]) + ": "
                + String(pools[i].get_capacity()) + " workers")

        if tp == 1:
            run_generation[degree=1](numa, topo, pools^, tokenizer, tokens^)
        elif tp == 2:
            run_generation[degree=2](numa, topo, pools^, tokenizer, tokens^)
        elif tp == 4:
            run_generation[degree=4](numa, topo, pools^, tokenizer, tokens^)
        else:
            print("unsupported tp=" + String(tp))
    else:
        print("mode: cold (spin-backoff)")
        var pools = HeapMoveArray[BurstPool[]](tp)
        for i in range(tp):
            pools.push(BurstPool[].for_topology(numa, topo[i]))
            print("  node " + String(topo[i]) + ": "
                + String(pools[i].get_capacity()) + " workers")

        if tp == 1:
            run_generation[degree=1](numa, topo, pools^, tokenizer, tokens^)
        elif tp == 2:
            run_generation[degree=2](numa, topo, pools^, tokenizer, tokens^)
        elif tp == 4:
            run_generation[degree=4](numa, topo, pools^, tokenizer, tokens^)
        else:
            print("unsupported tp=" + String(tp))
