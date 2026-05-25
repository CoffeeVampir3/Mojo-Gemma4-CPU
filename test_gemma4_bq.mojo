from std.pathlib import Path
from std.time import perf_counter_ns

from numa import NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from modeling.gemma_4_moe_bq import Gemma4


comptime MODEL_DIR = "checkpoints/gemma-4-26B-A4B-bq"


def main():
    var topo = NumaTopology()
    var nodes = len(topo)
    var iso = len(topo.isolated_cpus)
    print(t"{nodes} NUMA nodes, {iso} isolated cpus")

    @parameter
    def dispatch_gemma4_bq_tp[
        P: BurstThreadPool, //, degree: Int,
    ](var selected_pools: List[P]):
        var t0 = perf_counter_ns()
        var model_opt = Gemma4[degree=degree, Pool=P].load(
            Path(MODEL_DIR), topo, selected_pools^)
        if not model_opt:
            return
        var model = model_opt.take()
        var load_ms = (perf_counter_ns() - t0) / 1_000_000
        print(t"model loaded in {load_ms} ms")
        print()
        model.dump_tensors()

    with_topological_rank_dispatch[
        power_of_two_unrolling=3,
        dispatch=dispatch_gemma4_bq_tp,
    ](
        topo, "mode: isolated (spin-only)", "mode: cold (spin-backoff)")
