from std.memory import Span, alloc
from threading_trait_kernel_proto import (
    PoolStub, ScheduledKernel, StoreJob,
)


def main():
    var n = 4
    var outs_buf = alloc[Int](n)
    var jobs = List[ScheduledKernel[StoreJob]]()
    for i in range(n):
        jobs.append(ScheduledKernel[StoreJob](
            inner=StoreJob(
                out_addr=outs_buf + i,
                value=i + 1,
            ),
            node=i % 2,
            cost=10 + i,
        ))

    var pool = PoolStub()
    pool.capacity = n
    pool.dispatch_kernel(Span(jobs))

    print("active_jobs =", pool.active_jobs)
    debug_assert(jobs[3].preferred_node() == 1, "node metadata was not preserved")
    debug_assert(jobs[3].estimated_cost() == 13, "cost metadata was not preserved")
    for i in range(n):
        var v = (outs_buf + i)[]
        print("outs[", i, "] =", v)
        debug_assert(v == i + 1, "kernel did not run")
    _ = outs_buf
    print("ok")
