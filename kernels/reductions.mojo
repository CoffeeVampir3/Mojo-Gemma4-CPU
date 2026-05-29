from std.algorithm import vectorize
from std.memory import UnsafePointer, memcpy
from std.sys.info import simd_width_of

from modeling.model_spec import Encoding
from threading.threading_traits import BurstThreadPool
from .helpers import (
    RangePartitionedKernel, RankBuffers, DispatchBuffer, Binding,
    join_all, tile_dispatch, recommended_workers,
)
from .profiling import Profiler, DispatchSpan


comptime DEFAULT_INLINE_BYTES = 16384
comptime DstPtr[dtype: DType] = UnsafePointer[Scalar[dtype], MutAnyOrigin]


@fieldwise_init
struct ReduceConfig[E: Encoding, src_origin: ImmutOrigin](Copyable, Movable):
    """Per-rank source/destination pointer tables for one all-reduce. Lives on
    the dispatcher stack; the dispatched kernels carry only a pointer to it, so
    they remain trivially-copyable. `len(src)` is the runtime degree."""
    var src: List[UnsafePointer[Scalar[Self.E.DTYPE], Self.src_origin]]
    var dst: List[DstPtr[Self.E.DTYPE]]
    var chunk: Int
    var rem: Int


@fieldwise_init
struct ReduceStoreKernel[
    E: Encoding, src_origin: ImmutOrigin, cfg_origin: ImmutOrigin,
    Accum: DType = DType.float32,
](RangePartitionedKernel):
    var config: UnsafePointer[ReduceConfig[Self.E, Self.src_origin], Self.cfg_origin]
    var rank: Int
    var start: Int
    var end: Int

    def execute(mut self):
        reduce_store_range[Self.E, Self.src_origin, Self.Accum](
            self.config, self.rank, self.start, self.end)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


@fieldwise_init
struct GatherKernel[
    E: Encoding, src_origin: ImmutOrigin, cfg_origin: ImmutOrigin,
](RangePartitionedKernel):
    var config: UnsafePointer[ReduceConfig[Self.E, Self.src_origin], Self.cfg_origin]
    var rank: Int
    var start: Int
    var end: Int

    def execute(mut self):
        gather_chunks[Self.E, Self.src_origin](
            self.config, self.rank, self.start, self.end)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


@always_inline
def rank_chunk_count(chunk: Int, rem: Int, rank: Int, tp: Int) -> Int:
    if rank == tp - 1:
        return chunk + rem
    return chunk


@always_inline
def reduce_store_range[
    E: Encoding, src_origin: ImmutOrigin,
    Accum: DType = DType.float32,
](
    config: UnsafePointer[ReduceConfig[E, src_origin], _],
    out_rank: Int, start: Int, end: Int,
):
    var tp = len(config[].src)
    var srcs = config[].src.unsafe_ptr()
    var dst = config[].dst[out_rank]

    def step[width: Int](idx: Int) {read}:
        var pos = start + idx
        var acc = (srcs[0] + pos).load[width=width]().cast[Accum]()
        for r in range(1, tp):
            acc += (srcs[r] + pos).load[width=width]().cast[Accum]()
        (dst + pos).store(acc.cast[E.DTYPE]())

    vectorize[simd_width_of[Accum]()](end - start, step)


def gather_chunks[E: Encoding, src_origin: ImmutOrigin](
    config: UnsafePointer[ReduceConfig[E, src_origin], _],
    dst_rank: Int, start: Int, end: Int,
):
    var tp = len(config[].src)
    for src_rank in range(tp):
        if src_rank == dst_rank:
            continue
        var src_start = config[].chunk * src_rank
        var src_count = rank_chunk_count(config[].chunk, config[].rem, src_rank, tp)
        var lo = max(start, src_start)
        var hi = min(end, src_start + src_count)
        if lo < hi:
            memcpy(dest=config[].dst[dst_rank] + lo,
                   src=config[].dst[src_rank] + lo, count=hi - lo)


def dispatch_allreduce[
    P: BurstThreadPool, src_origin: ImmutOrigin, dst_origin: MutOrigin,
    Profile: Bool, N: Int, //,
    E: Encoding, Accum: DType = DType.float32,
    max_worker_count: Int = 128,
](
    src: RankBuffers[E.DTYPE, src_origin],
    output: RankBuffers[E.DTYPE, dst_origin],
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
    inline_max_bytes: Int = DEFAULT_INLINE_BYTES,
):
    if src.count <= 0:
        return

    var tp = len(pools)
    var chunk = src.count // tp
    var dst_ptrs = List[DstPtr[E.DTYPE]]()
    var src_ptrs = List[UnsafePointer[Scalar[E.DTYPE], src_origin]]()
    for r in range(tp):
        dst_ptrs.append(output.ptrs[r].as_any_origin())
        src_ptrs.append(src.ptrs[r])
    var cfg = ReduceConfig[E, src_origin](
        src=src_ptrs^, dst=dst_ptrs^, chunk=chunk, rem=src.count - chunk * tp,
    )
    var config = UnsafePointer(to=cfg).as_immutable()
    comptime cfg_ro = ImmutOrigin(origin_of(cfg))

    if src.count * E.ELEMENT_BYTES <= inline_max_bytes or tp <= 1:
        var inline_span = DispatchSpan[Profile]()
        reduce_store_range[E, src_origin, Accum](config, 0, 0, src.count)
        for r in range(1, tp):
            if config[].dst[r] != config[].dst[0]:
                memcpy(dest=config[].dst[r], src=config[].dst[0], count=src.count)
        inline_span.finish_inline(prof, "allreduce")
        return

    var data_bytes = src.count * E.ELEMENT_BYTES
    var reduce_span = DispatchSpan[Profile]()
    var reduce_buf = DispatchBuffer[
        ReduceStoreKernel[E, src_origin, cfg_ro, Accum],
        max_worker_count,
    ]()
    for r in range(tp):
        var rank_start = cfg.chunk * r
        var rank_count = rank_chunk_count(cfg.chunk, cfg.rem, r, tp)
        var nw = recommended_workers(
            rank_count * E.ELEMENT_BYTES,
            min(max_worker_count, pools[r].get_capacity()),
        )
        _ = tile_dispatch(reduce_buf,
            ReduceStoreKernel[E, src_origin, cfg_ro, Accum](config, r, 0, 0),
            pools[r], rank_count, rank_start, nw)
    reduce_span.issued()
    join_all(pools)
    reduce_span.finish(prof, pools, "allreduce.reduce")

    var gather_span = DispatchSpan[Profile]()
    var gather_buf = DispatchBuffer[
        GatherKernel[E, src_origin, cfg_ro], max_worker_count,
    ]()
    for r in range(tp):
        var nw = recommended_workers(
            data_bytes, min(max_worker_count, pools[r].get_capacity()))
        _ = tile_dispatch(gather_buf,
            GatherKernel[E, src_origin, cfg_ro](config, r, 0, 0),
            pools[r], src.count, num_workers=nw)
    gather_span.issued()
    join_all(pools)
    gather_span.finish(prof, pools, "allreduce.gather")


@always_inline
def dispatch_allreduce_inplace[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    E: Encoding, Accum: DType = DType.float32,
    max_worker_count: Int = 128,
](
    buf: Binding[Scalar[E.DTYPE], o], count: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
    inline_max_bytes: Int = DEFAULT_INLINE_BYTES,
):
    """In-place allreduce: each rank's `buf` is both source and destination."""
    comptime immut = ImmutOrigin(MutAnyOrigin)
    var tp = len(pools)
    var src = RankBuffers[E.DTYPE, immut](count=count)
    var dst = RankBuffers[E.DTYPE, MutAnyOrigin](count=count)
    for r in range(tp):
        src.add(buf[r].as_immutable())
        dst.add(buf[r])
    dispatch_allreduce[E, Accum, max_worker_count=max_worker_count](
        src, dst, pools, prof, inline_max_bytes=inline_max_bytes)
