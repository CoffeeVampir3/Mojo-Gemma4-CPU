from threading.threading_traits import BurstKernel, BurstThreadPool
from kernels.helpers import Chain
from threading.burst_threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool
from std.memory import Span, UnsafePointer, alloc
from std.time import perf_counter_ns
from std.benchmark import keep
from std.sys.info import size_of
from numa import NumaInfo
from notstdcollections import HeapMoveArray


@fieldwise_init
struct WriteKernel(BurstKernel):
    var dest: Int
    var value: Int

    def execute(mut self):
        UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=self.dest)[] = self.value


@fieldwise_init
struct AddKernel(BurstKernel):
    var dest: Int
    var addend: Int

    def execute(mut self):
        var p = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=self.dest)
        p[] = p[] + self.addend


@fieldwise_init
struct MulKernel(BurstKernel):
    var dest: Int
    var factor: Int

    def execute(mut self):
        var p = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=self.dest)
        p[] = p[] * self.factor


def test_chain[P: BurstThreadPool](mut pool: P):
    var cap = pool.get_capacity()
    var results = alloc[Int](cap)
    for i in range(cap):
        results[i] = 0

    var kernels = alloc[Chain[WriteKernel, AddKernel]](cap)
    for i in range(cap):
        kernels[i] = Chain[WriteKernel, AddKernel](
            WriteKernel(Int(results + i), i + 1),
            AddKernel(Int(results + i), 100),
        )

    pool.dispatch(Span(ptr=kernels, length=cap), cap)
    pool.join()
    keep(results[0])

    for i in range(cap):
        var expected = i + 1 + 100
        if results[i] != expected:
            print("FAIL chain: worker", i, "got", results[i], "expected", expected)
            return
    print("  chain[write, add]: ok (" + String(cap) + " workers)")


def test_nested_chain[P: BurstThreadPool](mut pool: P):
    var cap = pool.get_capacity()
    var results = alloc[Int](cap)
    for i in range(cap):
        results[i] = 0

    var kernels = alloc[Chain[Chain[WriteKernel, AddKernel], MulKernel]](cap)
    for i in range(cap):
        kernels[i] = Chain[Chain[WriteKernel, AddKernel], MulKernel](
            Chain[WriteKernel, AddKernel](
                WriteKernel(Int(results + i), i + 1),
                AddKernel(Int(results + i), 10),
            ),
            MulKernel(Int(results + i), 3),
        )

    pool.dispatch(Span(ptr=kernels, length=cap), cap)
    pool.join()
    keep(results[0])

    for i in range(cap):
        var expected = (i + 1 + 10) * 3
        if results[i] != expected:
            print("FAIL nested: worker", i, "got", results[i], "expected", expected)
            return
    print("  chain[chain[write, add], mul]: ok (" + String(cap) + " workers)")


def test_chain_repeated[P: BurstThreadPool](mut pool: P):
    var cap = pool.get_capacity()
    var results = alloc[Int](cap)

    var kernels = alloc[Chain[WriteKernel, AddKernel]](cap)
    for i in range(cap):
        kernels[i] = Chain[WriteKernel, AddKernel](
            WriteKernel(Int(results + i), 1),
            AddKernel(Int(results + i), i),
        )

    comptime ROUNDS = 100
    for _ in range(ROUNDS):
        pool.dispatch(Span(ptr=kernels, length=cap), cap)
        pool.join()
    keep(results[0])

    for i in range(cap):
        var expected = 1 + i
        if results[i] != expected:
            print("FAIL repeated: worker", i, "got", results[i], "expected", expected)
            return
    print("  chain repeated x" + String(ROUNDS) + ": ok")


def test_sizes():
    print("  WriteKernel:                    " + String(size_of[WriteKernel]()) + " bytes")
    print("  Chain[Write, Add]:              " + String(size_of[Chain[WriteKernel, AddKernel]]()) + " bytes")
    print("  Chain[Chain[Write, Add], Mul]:   " + String(size_of[Chain[Chain[WriteKernel, AddKernel], MulKernel]]()) + " bytes")


def test_bench[P: BurstThreadPool](mut pool: P):
    var cap = pool.get_capacity()
    var results = alloc[Int](cap)
    for i in range(cap):
        results[i] = 0

    var single_k = alloc[WriteKernel](cap)
    var chain2_k = alloc[Chain[WriteKernel, AddKernel]](cap)
    var chain3_k = alloc[Chain[Chain[WriteKernel, AddKernel], MulKernel]](cap)
    for i in range(cap):
        single_k[i] = WriteKernel(Int(results + i), i)
        chain2_k[i] = Chain[WriteKernel, AddKernel](
            WriteKernel(Int(results + i), i),
            AddKernel(Int(results + i), 1),
        )
        chain3_k[i] = Chain[Chain[WriteKernel, AddKernel], MulKernel](
            Chain[WriteKernel, AddKernel](
                WriteKernel(Int(results + i), i),
                AddKernel(Int(results + i), 1),
            ),
            MulKernel(Int(results + i), 2),
        )

    comptime WARMUP = 500
    comptime ITERS = 2000

    for _ in range(WARMUP):
        pool.dispatch(Span(ptr=single_k, length=cap), cap)
        pool.join()
    var t0 = Int(perf_counter_ns())
    for _ in range(ITERS):
        pool.dispatch(Span(ptr=single_k, length=cap), cap)
        pool.join()
    var t1 = Int(perf_counter_ns())
    keep(results[0])

    for _ in range(WARMUP):
        pool.dispatch(Span(ptr=chain2_k, length=cap), cap)
        pool.join()
    var t2 = Int(perf_counter_ns())
    for _ in range(ITERS):
        pool.dispatch(Span(ptr=chain2_k, length=cap), cap)
        pool.join()
    var t3 = Int(perf_counter_ns())
    keep(results[0])

    for _ in range(WARMUP):
        pool.dispatch(Span(ptr=chain3_k, length=cap), cap)
        pool.join()
    var t4 = Int(perf_counter_ns())
    for _ in range(ITERS):
        pool.dispatch(Span(ptr=chain3_k, length=cap), cap)
        pool.join()
    var t5 = Int(perf_counter_ns())
    keep(results[0])

    var single_ns = (t1 - t0) // ITERS
    var chain2_ns = (t3 - t2) // ITERS
    var chain3_ns = (t5 - t4) // ITERS

    print("  single kernel:  " + String(single_ns) + " ns/dispatch")
    print("  chain[2]:       " + String(chain2_ns) + " ns/dispatch  (+" + String(chain2_ns - single_ns) + " ns)")
    print("  chain[3]:       " + String(chain3_ns) + " ns/dispatch  (+" + String(chain3_ns - single_ns) + " ns)")

    var two_singles_ns = single_ns * 2
    var three_singles_ns = single_ns * 3
    print("  2x single:      " + String(two_singles_ns) + " ns  (chain[2] saves " + String(two_singles_ns - chain2_ns) + " ns)")
    print("  3x single:      " + String(three_singles_ns) + " ns  (chain[3] saves " + String(three_singles_ns - chain3_ns) + " ns)")


def main():
    var numa = NumaInfo()
    var topo = numa.plan_topology(numa.num_nodes)

    print(String(numa.num_nodes) + " NUMA nodes, "
        + String(len(numa.isolated_cpus)) + " isolated cpus\n")

    if numa.has_isolation():
        print("mode: isolated (spin-only)")
        var pools = HeapMoveArray[IsolatedBurstPool[]](numa.num_nodes)
        for i in range(numa.num_nodes):
            pools.push(IsolatedBurstPool[].for_topology(numa, topo[i]))
            print("  node " + String(topo[i]) + ": "
                + String(pools[i].get_capacity()) + " workers")
        print("\nsizes:")
        test_sizes()
        print("\ncorrectness:")
        test_chain(pools[0])
        test_nested_chain(pools[0])
        test_chain_repeated(pools[0])
        print("\nbenchmark:")
        test_bench(pools[0])
    else:
        print("mode: cold (spin-backoff)")
        var pools = HeapMoveArray[BurstPool[]](numa.num_nodes)
        for i in range(numa.num_nodes):
            pools.push(BurstPool[].for_topology(numa, topo[i]))
            print("  node " + String(topo[i]) + ": "
                + String(pools[i].get_capacity()) + " workers")
        print("\nsizes:")
        test_sizes()
        print("\ncorrectness:")
        test_chain(pools[0])
        test_nested_chain(pools[0])
        test_chain_repeated(pools[0])
        print("\nbenchmark:")
        test_bench(pools[0])
