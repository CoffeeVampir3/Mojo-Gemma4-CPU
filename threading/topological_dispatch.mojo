from numa import NumaTopology
from .burst_threading import BurstPool
from .isolated_burst_pool import IsolatedBurstPool
from .threading_traits import BurstThreadPool


def select_tensor_parallel_degree[
    P: BurstThreadPool,
    //,
    *,
    power_of_two_unrolling: Int,
    dispatch: def[Q: BurstThreadPool, //, degree: Int](
        var List[Q]
    ) capturing [_] -> None,
](tp: Int, var pools: List[P]):
    comptime assert power_of_two_unrolling > 0, (
        "power_of_two_unrolling must be positive"
    )
    comptime for i in range(power_of_two_unrolling):
        comptime degree = 1 << i
        if tp == degree:
            dispatch[degree](pools^)
            return
    print(t"unsupported tp={tp}")


def with_topological_rank_dispatch[
    *,
    power_of_two_unrolling: Int,
    dispatch: def[P: BurstThreadPool, //, degree: Int](
        var List[P]
    ) capturing [_] -> None,
](
    read topo: NumaTopology,
    read isolated_mode: String,
    read cold_mode: String,
):
    var tp = len(topo)
    if topo.has_isolation():
        print(isolated_mode)
        var pools = List[IsolatedBurstPool[]](capacity=tp)
        for i in range(tp):
            pools.append(IsolatedBurstPool[].for_rank(topo, i))
            print(t"  node {topo.node(i)}: {pools[i].get_capacity()} workers")
        print("")

        select_tensor_parallel_degree[
            power_of_two_unrolling=power_of_two_unrolling,
            dispatch=dispatch,
        ](tp, pools^)
    else:
        print(cold_mode)
        var pools = List[BurstPool[]](capacity=tp)
        for i in range(tp):
            pools.append(BurstPool[].for_rank(topo, i))
            print(t"  node {topo.node(i)}: {pools[i].get_capacity()} workers")
        print("")

        select_tensor_parallel_degree[
            power_of_two_unrolling=power_of_two_unrolling,
            dispatch=dispatch,
        ](tp, pools^)
