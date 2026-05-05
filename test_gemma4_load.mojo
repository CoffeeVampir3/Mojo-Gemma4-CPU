from std.pathlib import Path

from numa import NumaInfo
from threading.burst_threading import BurstPool
from notstdcollections import HeapMoveArray
from modeling.gemma_4_moe import Gemma4


def main():
    var numa = NumaInfo()
    var topo = numa.plan_topology(4)
    print("numa nodes:", numa.num_nodes, "degree=4")

    var pools = HeapMoveArray[BurstPool[]](4)
    for i in range(4):
        pools.push(BurstPool[].for_topology(numa, topo[i]))
        print("  rank", i, "node", topo[i], "workers:", pools[i].get_capacity())

    var ckpt = Path("checkpoints/gemma-4-26B-A4B")
    var loaded = Gemma4[4].load(ckpt, numa, topo, pools^)
    if not loaded:
        print("load failed")
        return
    var model = loaded.take()

    var kv = model.new_kv_cache()
    var token_id = 2
    model.forward(token_id, 0, kv)

    _ = model^
