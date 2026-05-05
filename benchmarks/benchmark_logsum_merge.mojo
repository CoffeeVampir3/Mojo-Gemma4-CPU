from std.collections import InlineArray
from std.memory import UnsafePointer
from std.time import perf_counter_ns
from std.benchmark import keep

from numa import NumaArena, NumaInfo, NumaTopology
from threading import BurstPool
from threading.isolated_burst_pool import IsolatedBurstPool
from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from kernels.logsum_merge import merge_flash_partials, reduce_partials
from kernels.helpers import RangedKernel, DispatchBuffer, tile_dispatch


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
struct NoopKernel(RangedKernel):
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


def fmt_ns(ns: Int) -> String:
    if ns < 1000:
        return String(ns) + " ns"
    elif ns < 1000000:
        return String(ns // 1000) + "." + String((ns % 1000) // 100) + " us"
    else:
        return String(ns // 1000000) + "." + String((ns % 1000000) // 100000) + " ms"


def measure_finalize[P: BurstThreadPool, //, head_dim: Int, num_q: Int](
    output: BF16Ptr, partials: F32Ptr, stride: Int, scratch: F32Ptr,
    num_sources: Int, nw: Int, mut pool: P,
) -> Int:
    warm_pool(scratch, pool)
    for _ in range(WARMUP):
        merge_flash_partials[head_dim, num_q](
            output, partials, stride, num_sources, pool,
            inline_max_bytes=0, num_workers=nw)
        keep(output[0])

    var best = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            merge_flash_partials[head_dim, num_q](
                output, partials, stride, num_sources, pool,
                inline_max_bytes=0, num_workers=nw)
            keep(output[0])
            elapsed += Int(perf_counter_ns()) - t0
        var avg = elapsed // ITERS
        if avg < best:
            best = avg
    return best


def measure_finalize_inline[P: BurstThreadPool, //, head_dim: Int, num_q: Int](
    output: BF16Ptr, partials: F32Ptr, stride: Int, scratch: F32Ptr,
    num_sources: Int, mut pool: P,
) -> Int:
    warm_pool(scratch, pool)
    for _ in range(WARMUP):
        merge_flash_partials[head_dim, num_q](
            output, partials, stride, num_sources, pool,
            inline_max_bytes=FORCE_INLINE)
        keep(output[0])

    var best = Int(1 << 60)
    for _ in range(TRIALS):
        var elapsed = 0
        for _ in range(ITERS):
            var t0 = Int(perf_counter_ns())
            merge_flash_partials[head_dim, num_q](
                output, partials, stride, num_sources, pool,
                inline_max_bytes=FORCE_INLINE)
            keep(output[0])
            elapsed += Int(perf_counter_ns()) - t0
        var avg = elapsed // ITERS
        if avg < best:
            best = avg
    return best


def run_config[P: BurstThreadPool, //, head_dim: Int, num_q: Int](
    mut arena: NumaArena[alignment=ALIGNMENT],
    mut pool: P,
):
    comptime stride = PARTIAL_STRIDE[num_q, head_dim]
    var partials = arena_alloc[DType.float32](arena, MAX_SOURCES * stride)
    var output = arena_alloc[DType.bfloat16](arena, num_q * head_dim)
    var scratch = arena_alloc[DType.float32](arena, pool.get_capacity())

    fill_partials[head_dim, num_q](partials, stride, MAX_SOURCES)

    print("\n=== head_dim=" + String(head_dim) + " num_q=" + String(num_q)
        + " pool_capacity=" + String(pool.get_capacity()) + " ===")
    print("  sources | data     | inline     | nw=2       | nw=4       | nw=8       | nw=" + String(num_q))

    var source_counts = InlineArray[Int, 7](fill=0)
    source_counts[0] = 2; source_counts[1] = 4; source_counts[2] = 8
    source_counts[3] = 16; source_counts[4] = 32; source_counts[5] = 64
    source_counts[6] = 128

    var worker_counts = InlineArray[Int, 4](fill=0)
    worker_counts[0] = 2; worker_counts[1] = 4
    worker_counts[2] = 8; worker_counts[3] = num_q

    for s in range(7):
        var ns = source_counts[s]
        var data_bytes = ns * (head_dim + 2) * 4 * num_q

        var t_inline = measure_finalize_inline[head_dim, num_q](
            output, partials, stride, scratch, ns, pool)

        var results = InlineArray[Int, 4](fill=0)
        for w in range(4):
            var nw = worker_counts[w]
            if nw > num_q:
                nw = num_q
            results[w] = measure_finalize[head_dim, num_q](
                output, partials, stride, scratch, ns, nw, pool)

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

        for w in range(4):
            var s_val = fmt_ns(results[w])
            line += "| " + s_val
            line += " " * max(0, 11 - s_val.byte_length())

        var best_val = t_inline
        var best_label = "inline"
        for w in range(4):
            if results[w] < best_val:
                best_val = results[w]
                best_label = "nw=" + String(worker_counts[w])
        line += "-> " + best_label

        print(line)


def main():
    var numa = NumaInfo()
    var topo = numa.plan_topology(1)

    print("logsum_merge worker count sweep")
    print(String(numa.num_nodes) + " NUMA node(s), "
        + String(len(numa.isolated_cpus)) + " isolated cpus")

    comptime ARENA_BYTES = 128 * 1024 * 1024
    var arena = NumaArena[alignment=ALIGNMENT](topo[0], ARENA_BYTES)
    if not arena:
        print("arena alloc failed")
        return

    if numa.has_isolation():
        print("mode: isolated")
        var pools = HeapMoveArray[IsolatedBurstPool[]](1)
        pools.push(IsolatedBurstPool[].for_topology(numa, topo[0]))
        run_config[head_dim=256, num_q=8](arena, pools[0])
        run_config[head_dim=512, num_q=16](arena, pools[0])
    else:
        print("mode: spin-backoff")
        var pools = HeapMoveArray[BurstPool[]](1)
        pools.push(BurstPool[].for_topology(numa, topo[0]))
        run_config[head_dim=256, num_q=8](arena, pools[0])
        run_config[head_dim=512, num_q=16](arena, pools[0])
