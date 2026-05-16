from std.collections import InlineArray
from std.memory import UnsafePointer
from std.time import perf_counter_ns
from std.benchmark import keep

from numa import NumaArena, NumaTopology
from threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool
from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from kernels.logsum_merge import dispatch_merge_flash_partials
from kernels.helpers import OutputPartitionedKernel, DispatchBuffer, tile_dispatch, NumaPointerArray


comptime ALIGNMENT = 64
comptime WARMUP = 50
comptime TRIALS = 30
comptime ITERS = 100
comptime MAX_SOURCES = 128
comptime FORCE_INLINE = 1 << 30

comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]

comptime PARTIAL_STRIDE[num_q: Int, head_dim: Int]: Int = (
    (num_q * head_dim + num_q + num_q) * 4 + 63) // 64 * 16


@fieldwise_init
struct NoopKernel(OutputPartitionedKernel):
    var dst: F32Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        self.dst[self.start] = Scalar[DType.float32](0)

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.dst, start, end)


def warm_pool[P: BurstThreadPool](scratch: F32Ptr, mut pool: P):
    var buf = DispatchBuffer[NoopKernel]()
    tile_dispatch(buf, NoopKernel(scratch, 0, 0), pool, pool.get_capacity())
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
) -> InlineArray[Int, tp]:
    var bases = InlineArray[Int, tp](uninitialized=True)
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
            sp[i] = Scalar[DType.float32]((i % 127) - 63) * 0.001
        for h in range(num_q):
            (sp + m_off + h)[] = Scalar[DType.float32](s % 5) * 0.5 - 1.0
            (sp + l_off + h)[] = Scalar[DType.float32](1.0)


def fill_partials_all[head_dim: Int, num_q: Int, tp: Int](
    ptrs: NumaPointerArray[DType.float32, tp],
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


def measure_finalize[
    P: BurstThreadPool, //, head_dim: Int, num_q: Int, tp: Int,
](
    output: BF16Ptr, partials: F32Ptr, stride: Int, scratch: F32Ptr,
    num_sources: Int, mut pools: HeapMoveArray[P], bases: InlineArray[Int, tp],
) -> Int:
    warm_pool(scratch, pools[0])
    for _ in range(WARMUP):
        dispatch_merge_flash_partials[head_dim, num_q, tp=tp](
            NumaPointerArray[DType.bfloat16, tp](output, bases),
            NumaPointerArray[DType.float32, tp](partials, bases),
            stride, source_counts[tp](num_sources), pools,
            inline_max_bytes=0)
        keep(output[0])

    var best = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            dispatch_merge_flash_partials[head_dim, num_q, tp=tp](
                NumaPointerArray[DType.bfloat16, tp](output, bases),
                NumaPointerArray[DType.float32, tp](partials, bases),
                stride, source_counts[tp](num_sources), pools,
                inline_max_bytes=0)
            keep(output[0])
            elapsed += Int(perf_counter_ns()) - t0
        var avg = elapsed // ITERS
        if avg < best:
            best = avg
    return best


def measure_finalize_inline[
    P: BurstThreadPool, //, head_dim: Int, num_q: Int, tp: Int,
](
    output: BF16Ptr, partials: F32Ptr, stride: Int, scratch: F32Ptr,
    num_sources: Int, mut pools: HeapMoveArray[P], bases: InlineArray[Int, tp],
) -> Int:
    warm_pool(scratch, pools[0])
    for _ in range(WARMUP):
        dispatch_merge_flash_partials[head_dim, num_q, tp=tp](
            NumaPointerArray[DType.bfloat16, tp](output, bases),
            NumaPointerArray[DType.float32, tp](partials, bases),
            stride, source_counts[tp](num_sources), pools,
            inline_max_bytes=FORCE_INLINE)
        keep(output[0])

    var best = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            dispatch_merge_flash_partials[head_dim, num_q, tp=tp](
                NumaPointerArray[DType.bfloat16, tp](output, bases),
                NumaPointerArray[DType.float32, tp](partials, bases),
                stride, source_counts[tp](num_sources), pools,
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
    comptime stride = PARTIAL_STRIDE[num_q, head_dim]
    var bases = arena_bases[tp](arenas)
    var partials = arena_alloc_all[DType.float32, tp](arenas, MAX_SOURCES * stride)
    var output = arena_alloc_all[DType.bfloat16, tp](arenas, num_q * head_dim)
    var scratch = arena_alloc[DType.float32](arenas[0], pools[0].get_capacity())

    fill_partials_all[head_dim, num_q, tp](
        NumaPointerArray[DType.float32, tp](partials, bases),
        stride, MAX_SOURCES)
    for r in range(tp):
        _ = arenas[r].prefault(0, arenas[r].used())

    print("\n=== head_dim=" + String(head_dim) + " num_q=" + String(num_q)
        + " pool_capacity=" + String(pools[0].get_capacity()) + " ===")
    print("  sources | data     | inline     | dispatched")

    var source_counts = InlineArray[Int, 7](fill=0)
    source_counts[0] = 2; source_counts[1] = 4; source_counts[2] = 8
    source_counts[3] = 16; source_counts[4] = 32; source_counts[5] = 64
    source_counts[6] = 128

    for s in range(7):
        var ns = source_counts[s]
        var data_bytes = ns * (head_dim + 2) * 4 * num_q

        var t_inline = measure_finalize_inline[head_dim, num_q, tp](
            output, partials, stride, scratch, ns, pools, bases)

        var t_dispatched = measure_finalize[head_dim, num_q, tp](
            output, partials, stride, scratch, ns, pools, bases)

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
        line += "| " + fmt_ns(t_dispatched)

        var best_label = "inline" if t_inline <= t_dispatched else "dispatched"
        line += " -> " + best_label

        print(line)


def main():
    var topo = NumaTopology()
    var tp = len(topo)

    print("logsum_merge worker count sweep")
    print(String(tp) + " NUMA node(s), "
        + String(len(topo.isolated_cpus)) + " isolated cpus")

    comptime ARENA_BYTES = 128 * 1024 * 1024
    var arenas = HeapMoveArray[NumaArena[alignment=ALIGNMENT]](tp)
    for i in range(tp):
        arenas.push(NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    if topo.has_isolation():
        print("mode: isolated")
        var pools = HeapMoveArray[IsolatedBurstPool[]](tp)
        for i in range(tp):
            pools.push(IsolatedBurstPool[].for_rank(topo, i))
        if tp == 1:
            run_config[head_dim=256, num_q=8, tp=1](arenas, pools)
            run_config[head_dim=512, num_q=16, tp=1](arenas, pools)
        elif tp == 2:
            run_config[head_dim=256, num_q=8, tp=2](arenas, pools)
            run_config[head_dim=512, num_q=16, tp=2](arenas, pools)
        elif tp == 4:
            run_config[head_dim=256, num_q=8, tp=4](arenas, pools)
            run_config[head_dim=512, num_q=16, tp=4](arenas, pools)
        else:
            print("unsupported tp=" + String(tp))
    else:
        print("mode: spin-backoff")
        var pools = HeapMoveArray[BurstPool[]](tp)
        for i in range(tp):
            pools.push(BurstPool[].for_rank(topo, i))
        if tp == 1:
            run_config[head_dim=256, num_q=8, tp=1](arenas, pools)
            run_config[head_dim=512, num_q=16, tp=1](arenas, pools)
        elif tp == 2:
            run_config[head_dim=256, num_q=8, tp=2](arenas, pools)
            run_config[head_dim=512, num_q=16, tp=2](arenas, pools)
        elif tp == 4:
            run_config[head_dim=256, num_q=8, tp=4](arenas, pools)
            run_config[head_dim=512, num_q=16, tp=4](arenas, pools)
        else:
            print("unsupported tp=" + String(tp))
