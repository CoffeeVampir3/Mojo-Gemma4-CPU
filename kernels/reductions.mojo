from std.algorithm import vectorize
from std.collections import InlineArray
from std.memory import UnsafePointer, memcpy
from std.sys.info import simd_width_of

from modeling.model_spec import Encoding
from notstdcollections import HeapMoveArray
from threading.threading_traits import BurstKernel, BurstThreadPool
from .helpers import (
    RangedKernel, RankBuffers, DispatchBuffer,
    join_all, tile_dispatch, recommended_workers,
)


comptime DEFAULT_INLINE_BYTES = 16384
comptime DstPtr[dtype: DType] = UnsafePointer[Scalar[dtype], MutAnyOrigin]


trait Writeback:
    @staticmethod
    def apply[
        dtype: DType, dst_origin: MutOrigin, Accum: DType, width: Int,
    ](
        dst: UnsafePointer[Scalar[dtype], dst_origin],
        pos: Int,
        reduced: SIMD[Accum, width],
    ): ...


struct AddIntoDest(Writeback):
    @staticmethod
    @always_inline
    def apply[
        dtype: DType, dst_origin: MutOrigin, Accum: DType, width: Int,
    ](
        dst: UnsafePointer[Scalar[dtype], dst_origin],
        pos: Int,
        reduced: SIMD[Accum, width],
    ):
        var old = (dst + pos).load[width=width]().cast[Accum]()
        (dst + pos).store((old + reduced).cast[dtype]())


@fieldwise_init
struct ReduceConfig[E: Encoding, tp: Int, src_origin: ImmutOrigin]:
    var src: InlineArray[UnsafePointer[Scalar[Self.E.DTYPE], Self.src_origin], Self.tp]
    var dst: InlineArray[DstPtr[Self.E.DTYPE], Self.tp]
    var chunk: Int
    var rem: Int


@fieldwise_init
struct ScratchReduceConfig[E: Encoding, tp: Int, src_origin: ImmutOrigin, Accum: DType]:
    var src: InlineArray[UnsafePointer[Scalar[Self.E.DTYPE], Self.src_origin], Self.tp]
    var scratch: DstPtr[Self.Accum]


@fieldwise_init
struct GatherConfig[E: Encoding, tp: Int, src_origin: ImmutOrigin]:
    var src: InlineArray[UnsafePointer[Scalar[Self.E.DTYPE], Self.src_origin], Self.tp]
    var shard_count: Int


@fieldwise_init
struct ChunkedReduceConfig[E: Encoding, tp: Int, src_origin: ImmutOrigin, Accum: DType]:
    var src: InlineArray[UnsafePointer[Scalar[Self.E.DTYPE], Self.src_origin], Self.tp]
    var scratches: InlineArray[DstPtr[Self.Accum], Self.tp]
    var chunk: Int
    var rem: Int


@fieldwise_init
struct CopyKernel[dtype: DType, src_origin: ImmutOrigin](RangedKernel):
    var dst: DstPtr[Self.dtype]
    var src: UnsafePointer[Scalar[Self.dtype], Self.src_origin]
    var start: Int
    var end: Int

    def execute(mut self):
        memcpy(dest=self.dst + self.start, src=self.src + self.start,
               count=self.end - self.start)

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.dst, self.src, start, end)


@fieldwise_init
struct ReduceStoreKernel[
    E: Encoding, tp: Int, src_origin: ImmutOrigin, cfg_origin: ImmutOrigin,
    Accum: DType = DType.float32,
](RangedKernel):
    var config: UnsafePointer[ReduceConfig[Self.E, Self.tp, Self.src_origin], Self.cfg_origin]
    var rank: Int
    var start: Int
    var end: Int

    def execute(mut self):
        reduce_store_range[Self.E, Self.tp, Self.src_origin, Self.Accum](
            self.config, self.rank, self.start, self.end)

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.config, self.rank, start, end)


@fieldwise_init
struct GatherKernel[E: Encoding, tp: Int, src_origin: ImmutOrigin, cfg_origin: ImmutOrigin](RangedKernel):
    var config: UnsafePointer[ReduceConfig[Self.E, Self.tp, Self.src_origin], Self.cfg_origin]
    var rank: Int
    var start: Int
    var end: Int

    def execute(mut self):
        gather_chunks[Self.E, Self.tp, Self.src_origin](
            self.config, self.rank, self.start, self.end)

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.config, self.rank, start, end)


@fieldwise_init
struct ChunkedReduceKernel[
    E: Encoding, tp: Int, src_origin: ImmutOrigin, cfg_origin: ImmutOrigin,
    Accum: DType = DType.float32,
](RangedKernel):
    var config: UnsafePointer[
        ChunkedReduceConfig[Self.E, Self.tp, Self.src_origin, Self.Accum],
        Self.cfg_origin,
    ]
    var rank: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var rank_start = self.config[].chunk * self.rank
        var srcs = InlineArray[
            UnsafePointer[Scalar[Self.E.DTYPE], Self.src_origin], Self.tp,
        ](uninitialized=True)
        for i in range(Self.tp):
            srcs[i] = self.config[].src[i] + rank_start
        reduce_sources_to[Self.E.DTYPE, Self.Accum, Self.tp, Self.src_origin, Self.Accum](
            srcs, self.config[].scratches[self.rank], self.start, self.end)

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.config, self.rank, start, end)


@fieldwise_init
struct DistributedWritebackKernel[
    E: Encoding, tp: Int, W: Writeback,
    src_origin: ImmutOrigin, cfg_origin: ImmutOrigin,
    Accum: DType = DType.float32,
](RangedKernel):
    var config: UnsafePointer[
        ChunkedReduceConfig[Self.E, Self.tp, Self.src_origin, Self.Accum],
        Self.cfg_origin,
    ]
    var dst: DstPtr[Self.E.DTYPE]
    var start: Int
    var end: Int

    def execute(mut self):
        var chunk_size = self.config[].chunk
        var remainder = self.config[].rem
        for c in range(Self.tp):
            var chunk_start = chunk_size * c
            var chunk_count = chunk_size + (remainder if c == Self.tp - 1 else 0)
            var lo = max(self.start, chunk_start)
            var hi = min(self.end, chunk_start + chunk_count)
            if lo >= hi:
                continue
            var scratch_ptr = self.config[].scratches[c]
            var scratch_off = lo - chunk_start
            var dst_off = lo

            def step[width: Int](idx: Int) {read}:
                var values = (scratch_ptr + scratch_off + idx).load[width=width]()
                Self.W.apply[Self.E.DTYPE, MutAnyOrigin, Self.Accum, width](
                    self.dst, dst_off + idx, values)

            vectorize[simd_width_of[Self.Accum]()](hi - lo, step)

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.config, self.dst, start, end)


@fieldwise_init
struct AllGatherKernel[E: Encoding, tp: Int, src_origin: ImmutOrigin, cfg_origin: ImmutOrigin](RangedKernel):
    var config: UnsafePointer[GatherConfig[Self.E, Self.tp, Self.src_origin], Self.cfg_origin]
    var dst: DstPtr[Self.E.DTYPE]
    var start: Int
    var end: Int

    def execute(mut self):
        var cfg = self.config
        var count = self.end - self.start
        for src_rank in range(Self.tp):
            memcpy(
                dest=self.dst + src_rank * cfg[].shard_count + self.start,
                src=cfg[].src[src_rank] + self.start,
                count=count,
            )

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.config, self.dst, start, end)


@always_inline
def rank_chunk_count[tp: Int](chunk: Int, rem: Int, rank: Int) -> Int:
    if rank == tp - 1:
        return chunk + rem
    return chunk


@always_inline
def reduce_sources_to[
    src_dtype: DType, dst_dtype: DType, tp: Int, src_origin: ImmutOrigin,
    Accum: DType = DType.float32,
](
    srcs: InlineArray[UnsafePointer[Scalar[src_dtype], src_origin], tp],
    dst: DstPtr[dst_dtype], start: Int, end: Int,
):
    def step[width: Int](idx: Int) {read}:
        var pos = start + idx
        var acc = (srcs[0] + pos).load[width=width]().cast[Accum]()
        for r in range(1, tp):
            acc += (srcs[r] + pos).load[width=width]().cast[Accum]()
        (dst + pos).store(acc.cast[dst_dtype]())

    vectorize[simd_width_of[Accum]()](end - start, step)


@always_inline
def reduce_store_range[
    E: Encoding, tp: Int, src_origin: ImmutOrigin,
    Accum: DType = DType.float32,
](
    config: UnsafePointer[ReduceConfig[E, tp, src_origin], _],
    out_rank: Int, start: Int, end: Int,
):
    var srcs = InlineArray[UnsafePointer[Scalar[E.DTYPE], src_origin], tp](
        uninitialized=True)
    for r in range(tp):
        srcs[r] = config[].src[r]
    reduce_sources_to[E.DTYPE, E.DTYPE, tp, src_origin, Accum](
        srcs, config[].dst[out_rank], start, end)


def copy_chunk[E: Encoding, tp: Int, src_origin: ImmutOrigin](
    config: UnsafePointer[ReduceConfig[E, tp, src_origin], _],
    dst_rank: Int, src_rank: Int, start: Int, end: Int,
):
    if start >= end:
        return
    memcpy(dest=config[].dst[dst_rank] + start,
           src=config[].dst[src_rank] + start, count=end - start)


def gather_chunks[E: Encoding, tp: Int, src_origin: ImmutOrigin](
    config: UnsafePointer[ReduceConfig[E, tp, src_origin], _],
    dst_rank: Int, start: Int, end: Int,
):
    for src_rank in range(tp):
        if src_rank == dst_rank:
            continue
        var src_start = config[].chunk * src_rank
        var src_count = rank_chunk_count[tp](config[].chunk, config[].rem, src_rank)
        copy_chunk[E, tp, src_origin](
            config, dst_rank, src_rank,
            max(start, src_start), min(end, src_start + src_count))


@always_inline
def reduce_to_scratch_range[
    E: Encoding, tp: Int, src_origin: ImmutOrigin,
    Accum: DType = DType.float32,
](
    config: UnsafePointer[ScratchReduceConfig[E, tp, src_origin, Accum], _],
    start: Int, end: Int,
):
    var srcs = InlineArray[UnsafePointer[Scalar[E.DTYPE], src_origin], tp](
        uninitialized=True)
    for r in range(tp):
        srcs[r] = config[].src[r]
    reduce_sources_to[E.DTYPE, Accum, tp, src_origin, Accum](
        srcs, config[].scratch, start, end)


def writeback_range[
    E: Encoding, W: Writeback, scratch_origin: ImmutOrigin,
    Accum: DType = DType.float32,
](
    scratch: UnsafePointer[Scalar[Accum], scratch_origin],
    dst: DstPtr[E.DTYPE], start: Int, end: Int,
):
    def step[width: Int](idx: Int) {read}:
        var pos = start + idx
        var values = (scratch + pos).load[width=width]()
        W.apply[E.DTYPE, MutAnyOrigin, Accum, width](dst, pos, values)

    vectorize[simd_width_of[Accum]()](end - start, step)


def make_reduce_config[
    E: Encoding, tp: Int, src_origin: ImmutOrigin, dst_origin: MutOrigin,
](
    src: RankBuffers[E.DTYPE, tp, src_origin],
    dst: RankBuffers[E.DTYPE, tp, dst_origin],
) -> ReduceConfig[E, tp, src_origin]:
    var chunk = src.count // tp
    var dst_ptrs = InlineArray[DstPtr[E.DTYPE], tp](uninitialized=True)
    for r in range(tp):
        dst_ptrs[r] = dst.ptrs[r].as_any_origin()
    return ReduceConfig[E, tp, src_origin](
        src=src.ptrs, dst=dst_ptrs, chunk=chunk, rem=src.count - chunk * tp,
    )


def make_scratch_config[
    E: Encoding, tp: Int, src_origin: ImmutOrigin, Accum: DType,
](
    src: RankBuffers[E.DTYPE, tp, src_origin],
    scratch: DstPtr[Accum],
) -> ScratchReduceConfig[E, tp, src_origin, Accum]:
    return ScratchReduceConfig[E, tp, src_origin, Accum](
        src=src.ptrs, scratch=scratch,
    )


def dispatch_allreduce[
    P: BurstThreadPool, src_origin: ImmutOrigin, dst_origin: MutOrigin, //,
    E: Encoding, tp: Int, Accum: DType = DType.float32,
](
    src: RankBuffers[E.DTYPE, tp, src_origin],
    output: RankBuffers[E.DTYPE, tp, dst_origin],
    mut pools: HeapMoveArray[P],
    inline_max_bytes: Int = DEFAULT_INLINE_BYTES,
):
    if src.count <= 0:
        return

    var cfg = make_reduce_config[E, tp, src_origin, dst_origin](src, output)
    var config = UnsafePointer(to=cfg).as_immutable()
    comptime cfg_ro = ImmutOrigin(origin_of(cfg))

    if src.count * E.ELEMENT_BYTES <= inline_max_bytes or tp <= 1:
        reduce_store_range[E, tp, src_origin, Accum](config, 0, 0, src.count)
        for r in range(1, tp):
            if config[].dst[r] != config[].dst[0]:
                memcpy(dest=config[].dst[r], src=config[].dst[0], count=src.count)
        return

    var data_bytes = src.count * E.ELEMENT_BYTES
    var reduce_buf = DispatchBuffer[ReduceStoreKernel[E, tp, src_origin, cfg_ro, Accum]]()
    for r in range(tp):
        var rank_start = cfg.chunk * r
        var rank_count = rank_chunk_count[tp](cfg.chunk, cfg.rem, r)
        var nw = recommended_workers(rank_count * E.ELEMENT_BYTES, pools[r].get_capacity())
        tile_dispatch(reduce_buf,
            ReduceStoreKernel[E, tp, src_origin, cfg_ro, Accum](config, r, 0, 0),
            pools[r], rank_count, rank_start, nw)
    join_all[tp](pools)

    var gather_buf = DispatchBuffer[GatherKernel[E, tp, src_origin, cfg_ro]]()
    for r in range(tp):
        var nw = recommended_workers(data_bytes, pools[r].get_capacity())
        tile_dispatch(gather_buf,
            GatherKernel[E, tp, src_origin, cfg_ro](config, r, 0, 0),
            pools[r], src.count, num_workers=nw)
    join_all[tp](pools)


def reduce_writeback[
    P: BurstThreadPool, src_origin: ImmutOrigin, dst_origin: MutOrigin, //,
    E: Encoding, tp: Int, W: Writeback, Accum: DType = DType.float32,
](
    src: RankBuffers[E.DTYPE, tp, src_origin],
    output: RankBuffers[E.DTYPE, tp, dst_origin],
    scratch: RankBuffers[Accum, tp, MutAnyOrigin],
    mut pools: HeapMoveArray[P],
    inline_max_bytes: Int = DEFAULT_INLINE_BYTES,
):
    if src.count <= 0:
        return

    comptime scratch_ro = ImmutOrigin(MutAnyOrigin)

    if src.count * E.ELEMENT_BYTES <= inline_max_bytes or tp <= 1:
        var scfg = make_scratch_config[E, tp, src_origin, Accum](src, scratch.ptrs[0])
        var sconfig = UnsafePointer(to=scfg).as_immutable()
        reduce_to_scratch_range[E, tp, src_origin, Accum](sconfig, 0, src.count)
        for r in range(tp):
            writeback_range[E, W, scratch_ro, Accum](
                scratch.ptrs[0].as_immutable(), output[r].as_any_origin(), 0, src.count)
        return

    var chunk = src.count // tp
    var rem = src.count - chunk * tp

    var scratch_ptrs = InlineArray[DstPtr[Accum], tp](uninitialized=True)
    for r in range(tp):
        scratch_ptrs[r] = scratch.ptrs[r]
    var cfg = ChunkedReduceConfig[E, tp, src_origin, Accum](
        src=src.ptrs, scratches=scratch_ptrs, chunk=chunk, rem=rem)
    var config = UnsafePointer(to=cfg).as_immutable()
    comptime cfg_ro = ImmutOrigin(origin_of(cfg))

    var data_bytes = src.count * E.ELEMENT_BYTES
    var reduce_buf = DispatchBuffer[
        ChunkedReduceKernel[E, tp, src_origin, cfg_ro, Accum],
    ]()
    for r in range(tp):
        var rank_count = rank_chunk_count[tp](chunk, rem, r)
        var nw = recommended_workers(rank_count * E.ELEMENT_BYTES, pools[r].get_capacity())
        tile_dispatch(reduce_buf,
            ChunkedReduceKernel[E, tp, src_origin, cfg_ro, Accum](
                config, r, 0, 0),
            pools[r], rank_count, num_workers=nw)
    join_all[tp](pools)

    var wb_buf = DispatchBuffer[
        DistributedWritebackKernel[E, tp, W, src_origin, cfg_ro, Accum],
    ]()
    for r in range(tp):
        var nw = recommended_workers(data_bytes, pools[r].get_capacity())
        tile_dispatch(wb_buf,
            DistributedWritebackKernel[E, tp, W, src_origin, cfg_ro, Accum](
                config, output[r].as_any_origin(), 0, 0),
            pools[r], src.count, num_workers=nw)
    join_all[tp](pools)


def dispatch_broadcast[
    P: BurstThreadPool, src_origin: ImmutOrigin, dst_origin: MutOrigin, //,
    E: Encoding, tp: Int,
](
    src: RankBuffers[E.DTYPE, tp, src_origin],
    dst: RankBuffers[E.DTYPE, tp, dst_origin],
    mut pools: HeapMoveArray[P],
    src_rank: Int = 0,
    inline_max_bytes: Int = DEFAULT_INLINE_BYTES,
):
    if src.count <= 0:
        return

    if src[src_rank] != dst[src_rank]:
        memcpy(
            dest=dst[src_rank].as_any_origin(),
            src=src[src_rank],
            count=src.count,
        )

    if src.count * E.ELEMENT_BYTES <= inline_max_bytes:
        for r in range(tp):
            if r != src_rank:
                memcpy(dest=dst[r].as_any_origin(), src=src[src_rank], count=src.count)
        return

    var data_bytes = src.count * E.ELEMENT_BYTES
    var buf = DispatchBuffer[CopyKernel[E.DTYPE, src_origin]]()
    for r in range(tp):
        if r != src_rank:
            var nw = recommended_workers(data_bytes, pools[r].get_capacity())
            tile_dispatch(buf,
                CopyKernel[E.DTYPE, src_origin](
                    dst[r].as_any_origin(), src[src_rank], 0, 0),
                pools[r], src.count, num_workers=nw)
    for r in range(tp):
        if r != src_rank:
            pools[r].join()


def allgather[
    P: BurstThreadPool, src_origin: ImmutOrigin, dst_origin: MutOrigin, //,
    E: Encoding, tp: Int,
](
    src: RankBuffers[E.DTYPE, tp, src_origin],
    dst: RankBuffers[E.DTYPE, tp, dst_origin],
    mut pools: HeapMoveArray[P],
    inline_max_bytes: Int = DEFAULT_INLINE_BYTES,
):
    if src.count <= 0:
        return

    if tp <= 1:
        if src[0] != dst[0]:
            memcpy(dest=dst[0].as_any_origin(), src=src[0], count=src.count)
        return

    if src.count * E.ELEMENT_BYTES <= inline_max_bytes:
        for r in range(tp):
            var d = dst[r].as_any_origin()
            for src_rank in range(tp):
                memcpy(dest=d + src_rank * src.count, src=src[src_rank],
                       count=src.count)
        return

    var cfg = GatherConfig[E, tp, src_origin](src=src.ptrs, shard_count=src.count)
    var config = UnsafePointer(to=cfg).as_immutable()
    comptime cfg_ro = ImmutOrigin(origin_of(cfg))

    var data_bytes = src.count * E.ELEMENT_BYTES * tp
    var buf = DispatchBuffer[AllGatherKernel[E, tp, src_origin, cfg_ro]]()
    for r in range(tp):
        var nw = recommended_workers(data_bytes, pools[r].get_capacity())
        tile_dispatch(buf,
            AllGatherKernel[E, tp, src_origin, cfg_ro](
                config, dst[r].as_any_origin(), 0, 0),
            pools[r], src.count, num_workers=nw)
    join_all[tp](pools)
