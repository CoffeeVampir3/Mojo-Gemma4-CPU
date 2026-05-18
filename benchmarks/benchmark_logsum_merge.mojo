from std.collections import InlineArray
from std.memory import UnsafePointer
from std.time import perf_counter_ns
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from notstdcollections import HeapMoveArray
from kernels.logsum_merge import dispatch_merge_flash_partials, FinalizeKernel
from kernels.helpers import (
    OutputPartitionedKernel, DispatchBuffer, tile_dispatch,
    Binding, ArenaBases,
)


comptime ALIGNMENT = 64
comptime WARMUP = 50
comptime TRIALS = 30
comptime ITERS = 100
comptime MAX_SOURCES = 128
comptime FORCE_INLINE = 1 << 30

comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]


@fieldwise_init
struct NoopKernel(OutputPartitionedKernel):
    var dst: F32Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        self.dst[self.start] = Float32(0)

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.start = start
        self.end = end


def warm_pool[P: BurstThreadPool](scratch: F32Ptr, mut pool: P):
    var buf = DispatchBuffer[NoopKernel]()
    _ = tile_dispatch(buf, NoopKernel(scratch, 0, 0), pool, pool.get_capacity())
    pool.join()


def arena_alloc[dtype: DType](
    mut arena: NumaArena[alignment=ALIGNMENT], count: Int,
) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
    var ptr = arena.alloc[Scalar[dtype]](count)
    if not ptr:
        print("arena alloc failed")
        return UnsafePointer[Scalar[dtype], MutAnyOrigin].unsafe_dangling()
    return ptr.value()


def arena_bases[tp: Int](
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
) -> ArenaBases[tp]:
    var bases = ArenaBases[tp].uninitialized()
    for r in range(tp):
        bases[r] = Int(arenas[r].base.value())
    return bases


def arena_alloc_all[dtype: DType, tp: Int](
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]], count: Int,
) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
    var first = UnsafePointer[Scalar[dtype], MutAnyOrigin].unsafe_dangling()
    for r in range(tp):
        var ptr = arena_alloc[dtype](arenas[r], count)
        if r == 0:
            first = ptr
    return first


def fill_partials[head_dim: Int, num_q: Int](
    buf: F32Ptr, stride: Int, num_sources: Int,
):
    comptime m_off = num_q * head_dim
    comptime l_off = m_off + num_q
    for s in range(num_sources):
        var sp = buf + s * stride
        for i in range(num_q * head_dim):
            sp[i] = Float32((i % 127) - 63) * 0.001
        for h in range(num_q):
            (sp + m_off + h)[] = Float32(s % 5) * 0.5 - 1.0
            (sp + l_off + h)[] = Float32(1.0)


def fill_partials_all[head_dim: Int, num_q: Int, tp: Int](
    ptrs: Binding[Float32, tp],
    stride: Int, num_sources: Int,
):
    for r in range(tp):
        fill_partials[head_dim, num_q](ptrs[r], stride, num_sources)


def source_counts[tp: Int](num_sources: Int) -> InlineArray[Int, tp]:
    return InlineArray[Int, tp](fill=num_sources)


def fmt_ns(ns: Int) -> String:
    if ns < 1000:
        return String(ns) + " ns"
    elif ns < 1000000:
        return String(ns // 1000) + "." + String((ns % 1000) // 100) + " us"
    else:
        return String(ns // 1000000) + "." + String((ns % 1000000) // 100000) + " ms"


def max_last_ts[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
) -> Int:
    var hi = 0
    for r in range(tp):
        var ts = pools[r].last_worker_timestamp()
        if ts > hi:
            hi = ts
    return hi


@fieldwise_init
struct MergeTiming(Copyable, ImplicitlyCopyable):
    var kernel_ns: Int
    var wall_ns: Int


def measure_finalize[
    P: BurstThreadPool, //, head_dim: Int, num_q: Int, tp: Int,
](
    output: BF16Ptr, partials: F32Ptr, scratch: F32Ptr,
    num_sources: Int, mut pools: HeapMoveArray[P], bases: ArenaBases[tp],
) -> MergeTiming:
    warm_pool(scratch, pools[0])
    for _ in range(WARMUP):
        dispatch_merge_flash_partials[head_dim, num_q, tp=tp](
            Binding[BFloat16, tp](output, bases),
            Binding[Float32, tp](partials, bases),
            source_counts[tp](num_sources), pools,
            inline_max_bytes=0)
        keep(output[0])

    var best_wall = Int(1 << 60)
    var best_kernel = Int(1 << 60)
    for _ in range(TRIALS):
        var wall_sum = 0
        var kernel_sum = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            dispatch_merge_flash_partials[head_dim, num_q, tp=tp](
                Binding[BFloat16, tp](output, bases),
                Binding[Float32, tp](partials, bases),
                source_counts[tp](num_sources), pools,
                inline_max_bytes=0)
            var t1 = Int(perf_counter_ns())
            var t_done = max_last_ts[tp=tp](pools)
            keep(output[0])
            wall_sum += t1 - t0
            kernel_sum += t_done - t0
        var avg_wall = wall_sum // ITERS
        var avg_kernel = kernel_sum // ITERS
        if avg_wall < best_wall:
            best_wall = avg_wall
        if avg_kernel < best_kernel:
            best_kernel = avg_kernel
    return MergeTiming(best_kernel, best_wall)


def measure_finalize_inline[
    P: BurstThreadPool, //, head_dim: Int, num_q: Int, tp: Int,
](
    output: BF16Ptr, partials: F32Ptr, scratch: F32Ptr,
    num_sources: Int, mut pools: HeapMoveArray[P], bases: ArenaBases[tp],
) -> Int:
    warm_pool(scratch, pools[0])
    for _ in range(WARMUP):
        dispatch_merge_flash_partials[head_dim, num_q, tp=tp](
            Binding[BFloat16, tp](output, bases),
            Binding[Float32, tp](partials, bases),
            source_counts[tp](num_sources), pools,
            inline_max_bytes=FORCE_INLINE)
        keep(output[0])

    var best = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            dispatch_merge_flash_partials[head_dim, num_q, tp=tp](
                Binding[BFloat16, tp](output, bases),
                Binding[Float32, tp](partials, bases),
                source_counts[tp](num_sources), pools,
                inline_max_bytes=FORCE_INLINE)
            keep(output[0])
            elapsed += Int(perf_counter_ns()) - t0
        var avg = elapsed // ITERS
        if avg < best:
            best = avg
    return best


def run_config[
    P: BurstThreadPool, //, head_dim: Int, num_q: Int, tp: Int,
](
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
    mut pools: HeapMoveArray[P],
):
    comptime stride = FinalizeKernel[head_dim, num_q].PARTIAL_STRIDE
    var bases = arena_bases[tp](arenas)
    var partials = arena_alloc_all[DType.float32, tp](arenas, MAX_SOURCES * stride)
    var output = arena_alloc_all[DType.bfloat16, tp](arenas, num_q * head_dim)
    var scratch = arena_alloc[DType.float32](arenas[0], pools[0].get_capacity())

    fill_partials_all[head_dim, num_q, tp](
        Binding[Float32, tp](partials, bases),
        stride, MAX_SOURCES)
    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    print("\n=== head_dim=" + String(head_dim) + " num_q=" + String(num_q)
        + " pool_capacity=" + String(pools[0].get_capacity()) + " ===")
    print("  sources | data     | inline     | kernel     | wall       -> best")

    var counts = InlineArray[Int, 7](fill=0)
    counts[0] = 2; counts[1] = 4; counts[2] = 8
    counts[3] = 16; counts[4] = 32; counts[5] = 64
    counts[6] = 128

    for s in range(7):
        var ns = counts[s]
        var data_bytes = ns * (head_dim + 2) * 4 * num_q

        var t_inline = measure_finalize_inline[head_dim, num_q, tp](
            output, partials, scratch, ns, pools, bases)

        var t_dispatched = measure_finalize[head_dim, num_q, tp](
            output, partials, scratch, ns, pools, bases)

        var line = "  " + String(ns)
        if ns < 100:
            line += " " * (8 - String(ns).byte_length())
        else:
            line += " " * (7 - String(ns).byte_length())
        line += "| " + String(data_bytes // 1024) + "KB"
        var kb_str = String(data_bytes // 1024)
        line += " " * (7 - kb_str.byte_length())
        line += "| " + fmt_ns(t_inline)
        line += " " * max(0, 11 - fmt_ns(t_inline).byte_length())
        line += "| " + fmt_ns(t_dispatched.kernel_ns)
        line += " " * max(0, 11 - fmt_ns(t_dispatched.kernel_ns).byte_length())
        line += "| " + fmt_ns(t_dispatched.wall_ns)
        line += " " * max(0, 11 - fmt_ns(t_dispatched.wall_ns).byte_length())

        var best_label = "inline" if t_inline <= t_dispatched.kernel_ns else "dispatched"
        line += "-> " + best_label

        print(line)


def run_all[P: BurstThreadPool, //, tp: Int](
    mut arenas: HeapMoveArray[NumaArena[alignment=ALIGNMENT]],
    mut pools: HeapMoveArray[P],
):
    run_config[head_dim=256, num_q=8, tp=tp](arenas, pools)
    run_config[head_dim=512, num_q=16, tp=tp](arenas, pools)


def main():
    var topo = NumaTopology()
    var tp = len(topo)

    print("logsum_merge worker count sweep")
    print(String(tp) + " NUMA node(s), "
        + String(len(topo.isolated_cpus)) + " isolated cpus\n")

    comptime ARENA_BYTES = 128 * 1024 * 1024
    var arenas = HeapMoveArray[NumaArena[alignment=ALIGNMENT]](tp)
    for i in range(tp):
        arenas.push(NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_logsum_merge_tp[
        P: BurstThreadPool, //, degree: Int,
    ](var selected_pools: HeapMoveArray[P]):
        run_all[tp=degree](arenas, selected_pools)

    with_topological_rank_dispatch[
        power_of_two_unrolling=3,
        dispatch=dispatch_logsum_merge_tp,
    ](topo, "mode: isolated", "mode: spin-backoff")
