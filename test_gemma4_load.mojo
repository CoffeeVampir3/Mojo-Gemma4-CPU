from std.pathlib import Path

from numa import NumaInfo
from threading.burst_threading import BurstPool
from notstdcollections import HeapMoveArray
from modeling.gemma_4_moe import Gemma4


def main():
    var numa = NumaInfo()
    var topo = numa.plan_topology(1)
    print("numa nodes:", numa.num_nodes, "degree=1 rank0 node:", topo[0])

    var pools = HeapMoveArray[BurstPool[]](1)
    pools.push(BurstPool[].for_topology(numa, topo[0]))
    print("pool capacity:", pools[0].get_capacity())

    var ckpt = Path("checkpoints/gemma-4-26B-A4B")
    var loaded = Gemma4[1].load(ckpt, numa, topo, pools^)
    if not loaded:
        print("load failed")
        return
    var model = loaded.take()
    _ = model^
    print("load ok")
