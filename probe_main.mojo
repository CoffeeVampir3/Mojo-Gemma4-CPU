from std.pathlib import Path
from numa import NumaTopology
from modeling.loader import discover_shards
from quant.probe import run_read_probe


comptime CHECKPOINT_DIR = "checkpoints/gemma-4-26B-A4B"


def main():
    var topo = NumaTopology()
    var tp = len(topo)
    var iso = len(topo.isolated_cpus)
    print(t"{tp} NUMA nodes, {iso} isolated cpus")

    var shards = discover_shards(Path(CHECKPOINT_DIR))
    if len(shards) == 0:
        print(t"probe: no shards under {CHECKPOINT_DIR}")
        return
    print(t"probe: {len(shards)} shard(s)")

    var ok = run_read_probe(shards, topo)
    if not ok:
        print("probe: FAILED")
    else:
        print("probe: ok")
