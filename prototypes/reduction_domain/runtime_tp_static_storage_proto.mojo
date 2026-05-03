from std.algorithm import vectorize
from std.collections import InlineArray
from std.memory import Span, UnsafePointer, memcpy
from std.sys.info import simd_width_of

from kernels.helpers import DispatchBuffer, RangedKernel, worker_range
from threading.threading_traits import BurstKernel, BurstThreadPool


comptime MAX_TP = 4
comptime DstPtr[dtype: DType] = UnsafePointer[Scalar[dtype], MutAnyOrigin]


@fieldwise_init
struct TestPool(BurstThreadPool, Copyable, ImplicitlyCopyable):
    var capacity: Int
    var dispatch_count: Int

    def get_capacity(self) -> Int:
        return self.capacity

    def dispatch[K: BurstKernel, origin: MutOrigin](
        mut self, kernels: Span[K, origin], num_jobs: Int
    ):
        for i in range(num_jobs):
            var kernel = kernels[i]
            kernel.execute()
        self.dispatch_count += 1

    def join(mut self):
        pass

    def last_worker_timestamp(self) -> Int:
        return self.dispatch_count

    def wake(mut self):
        pass

    def sleep(mut self):
        pass


struct RuntimeRankBuffers[
    dtype: DType, data_origin: Origin, table_origin: ImmutOrigin,
]:
    comptime DataPtr = UnsafePointer[Scalar[Self.dtype], Self.data_origin]

    var ptrs: UnsafePointer[Self.DataPtr, Self.table_origin]
    var tp: Int
    var count: Int

    @always_inline
    def __init__(
        out self,
        ptrs: UnsafePointer[Self.DataPtr, Self.table_origin],
        tp: Int,
        count: Int,
    ):
        self.ptrs = ptrs
        self.tp = tp
        self.count = count

    @always_inline
    def __getitem__(self, rank: Int) -> Self.DataPtr:
        return self.ptrs[rank]


@fieldwise_init
struct RuntimeReduceConfig[
    dtype: DType,
    src_origin: ImmutOrigin,
    src_table_origin: ImmutOrigin,
    dst_table_origin: ImmutOrigin,
]:
    var src: UnsafePointer[
        UnsafePointer[Scalar[Self.dtype], Self.src_origin],
        Self.src_table_origin,
    ]
    var dst: UnsafePointer[DstPtr[Self.dtype], Self.dst_table_origin]
    var tp: Int
    var count: Int
    var chunk: Int
    var rem: Int


@fieldwise_init
struct RuntimeGatherConfig[
    dtype: DType, src_origin: ImmutOrigin, src_table_origin: ImmutOrigin,
]:
    var src: UnsafePointer[
        UnsafePointer[Scalar[Self.dtype], Self.src_origin],
        Self.src_table_origin,
    ]
    var tp: Int
    var shard_count: Int


@fieldwise_init
struct RuntimeReduceStoreKernel[
    dtype: DType,
    src_origin: ImmutOrigin,
    src_table_origin: ImmutOrigin,
    dst_table_origin: ImmutOrigin,
    cfg_origin: ImmutOrigin,
    Accum: DType = DType.float32,
](RangedKernel):
    var config: UnsafePointer[
        RuntimeReduceConfig[
            Self.dtype, Self.src_origin, Self.src_table_origin,
            Self.dst_table_origin,
        ],
        Self.cfg_origin,
    ]
    var rank: Int
    var start: Int
    var end: Int

    def execute(mut self):
        runtime_reduce_store_range[
            Self.dtype, Self.src_origin, Self.src_table_origin,
            Self.dst_table_origin, Self.cfg_origin, Self.Accum,
        ](self.config, self.rank, self.start, self.end)

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.config, self.rank, start, end)


@fieldwise_init
struct RuntimeGatherKernel[
    dtype: DType,
    src_origin: ImmutOrigin,
    src_table_origin: ImmutOrigin,
    dst_table_origin: ImmutOrigin,
    cfg_origin: ImmutOrigin,
](RangedKernel):
    var config: UnsafePointer[
        RuntimeReduceConfig[
            Self.dtype, Self.src_origin, Self.src_table_origin,
            Self.dst_table_origin,
        ],
        Self.cfg_origin,
    ]
    var rank: Int
    var start: Int
    var end: Int

    def execute(mut self):
        runtime_gather_chunks[
            Self.dtype, Self.src_origin, Self.src_table_origin,
            Self.dst_table_origin, Self.cfg_origin,
        ](self.config, self.rank, self.start, self.end)

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.config, self.rank, start, end)


@fieldwise_init
struct RuntimeAllGatherKernel[
    dtype: DType,
    src_origin: ImmutOrigin,
    src_table_origin: ImmutOrigin,
    cfg_origin: ImmutOrigin,
](RangedKernel):
    var config: UnsafePointer[
        RuntimeGatherConfig[Self.dtype, Self.src_origin, Self.src_table_origin],
        Self.cfg_origin,
    ]
    var dst: DstPtr[Self.dtype]
    var start: Int
    var end: Int

    def execute(mut self):
        var cfg = self.config
        var count = self.end - self.start
        for src_rank in range(cfg[].tp):
            memcpy(
                dest=self.dst + src_rank * cfg[].shard_count + self.start,
                src=cfg[].src[src_rank] + self.start,
                count=count,
            )

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.config, self.dst, start, end)


@always_inline
def runtime_rank_chunk_count(tp: Int, chunk: Int, rem: Int, rank: Int) -> Int:
    if rank == tp - 1:
        return chunk + rem
    return chunk


@always_inline
def runtime_reduce_sources_to[
    src_dtype: DType,
    dst_dtype: DType,
    src_origin: ImmutOrigin,
    src_table_origin: ImmutOrigin,
    Accum: DType = DType.float32,
](
    srcs: UnsafePointer[
        UnsafePointer[Scalar[src_dtype], src_origin], src_table_origin,
    ],
    tp: Int,
    dst: DstPtr[dst_dtype],
    start: Int,
    end: Int,
):
    def step[width: Int](idx: Int) {read}:
        var pos = start + idx
        var acc = (srcs[0] + pos).load[width=width]().cast[Accum]()
        for r in range(1, tp):
            acc += (srcs[r] + pos).load[width=width]().cast[Accum]()
        (dst + pos).store(acc.cast[dst_dtype]())

    vectorize[simd_width_of[Accum]()](end - start, step)


@always_inline
def runtime_reduce_store_range[
    dtype: DType,
    src_origin: ImmutOrigin,
    src_table_origin: ImmutOrigin,
    dst_table_origin: ImmutOrigin,
    cfg_origin: ImmutOrigin,
    Accum: DType = DType.float32,
](
    config: UnsafePointer[
        RuntimeReduceConfig[
            dtype, src_origin, src_table_origin, dst_table_origin,
        ],
        cfg_origin,
    ],
    out_rank: Int,
    start: Int,
    end: Int,
):
    runtime_reduce_sources_to[dtype, dtype, src_origin, src_table_origin, Accum](
        config[].src, config[].tp, config[].dst[out_rank], start, end
    )


def runtime_copy_chunk[
    dtype: DType,
    src_origin: ImmutOrigin,
    src_table_origin: ImmutOrigin,
    dst_table_origin: ImmutOrigin,
    cfg_origin: ImmutOrigin,
](
    config: UnsafePointer[
        RuntimeReduceConfig[
            dtype, src_origin, src_table_origin, dst_table_origin,
        ],
        cfg_origin,
    ],
    dst_rank: Int,
    src_rank: Int,
    start: Int,
    end: Int,
):
    if start >= end:
        return
    memcpy(
        dest=config[].dst[dst_rank] + start,
        src=config[].dst[src_rank] + start,
        count=end - start,
    )


def runtime_gather_chunks[
    dtype: DType,
    src_origin: ImmutOrigin,
    src_table_origin: ImmutOrigin,
    dst_table_origin: ImmutOrigin,
    cfg_origin: ImmutOrigin,
](
    config: UnsafePointer[
        RuntimeReduceConfig[
            dtype, src_origin, src_table_origin, dst_table_origin,
        ],
        cfg_origin,
    ],
    dst_rank: Int,
    start: Int,
    end: Int,
):
    for src_rank in range(config[].tp):
        if src_rank == dst_rank:
            continue
        var src_start = config[].chunk * src_rank
        var src_count = runtime_rank_chunk_count(
            config[].tp, config[].chunk, config[].rem, src_rank
        )
        runtime_copy_chunk[
            dtype, src_origin, src_table_origin, dst_table_origin, cfg_origin,
        ](
            config,
            dst_rank,
            src_rank,
            max(start, src_start),
            min(end, src_start + src_count),
        )


def runtime_tile_dispatch[K: RangedKernel, P: BurstThreadPool](
    mut buf: DispatchBuffer[K],
    proto: K,
    mut pool: P,
    total: Int,
    base: Int = 0,
):
    var workers = pool.get_capacity()
    for w in range(workers):
        var wr = worker_range(total, workers, w, base)
        buf.slot()[] = proto.over_range(wr[0], wr[1])
    buf.dispatch(pool)


def join_runtime[P: BurstThreadPool, pool_origin: MutOrigin](
    pools: UnsafePointer[P, pool_origin], tp: Int
):
    for r in range(tp):
        (pools + r)[].join()


def build_runtime_reduce_config[
    dtype: DType,
    src_origin: ImmutOrigin,
    src_table_origin: ImmutOrigin,
    dst_origin: MutOrigin,
    dst_ptr_table_origin: ImmutOrigin,
    dst_any_table_origin: MutOrigin,
](
    src: RuntimeRankBuffers[dtype, src_origin, src_table_origin],
    dst: RuntimeRankBuffers[dtype, dst_origin, dst_ptr_table_origin],
    dst_any_table: UnsafePointer[DstPtr[dtype], dst_any_table_origin],
) -> RuntimeReduceConfig[
    dtype, src_origin, src_table_origin, ImmutOrigin(dst_any_table_origin),
]:
    for r in range(src.tp):
        dst_any_table[r] = dst[r].as_any_origin()
    var chunk = src.count // src.tp
    return RuntimeReduceConfig[
        dtype, src_origin, src_table_origin, ImmutOrigin(dst_any_table_origin),
    ](
        src.ptrs,
        dst_any_table.as_immutable(),
        src.tp,
        src.count,
        chunk,
        src.count - chunk * src.tp,
    )


def runtime_allreduce_no_alloc[
    P: BurstThreadPool,
    pool_origin: MutOrigin,
    dtype: DType,
    src_origin: ImmutOrigin,
    src_table_origin: ImmutOrigin,
    dst_origin: MutOrigin,
    dst_table_origin: MutOrigin,
    cfg_origin: ImmutOrigin,
    Accum: DType = DType.float32,
](
    config: UnsafePointer[
        RuntimeReduceConfig[
            dtype, src_origin, src_table_origin, ImmutOrigin(dst_table_origin),
        ],
        cfg_origin,
    ],
    pools: UnsafePointer[P, pool_origin],
    mut reduce_buf: DispatchBuffer[
        RuntimeReduceStoreKernel[
            dtype, src_origin, src_table_origin, ImmutOrigin(dst_table_origin),
            cfg_origin, Accum,
        ]
    ],
    mut gather_buf: DispatchBuffer[
        RuntimeGatherKernel[
            dtype, src_origin, src_table_origin, ImmutOrigin(dst_table_origin),
            cfg_origin,
        ]
    ],
    inline_max_elements: Int,
):
    if config[].count <= 0:
        return

    if config[].count <= inline_max_elements or config[].tp <= 1:
        runtime_reduce_store_range[
            dtype, src_origin, src_table_origin, ImmutOrigin(dst_table_origin),
            cfg_origin, Accum,
        ](config, 0, 0, config[].count)
        for r in range(1, config[].tp):
            if config[].dst[r] != config[].dst[0]:
                memcpy(
                    dest=config[].dst[r],
                    src=config[].dst[0],
                    count=config[].count,
                )
        return

    for r in range(config[].tp):
        var rank_start = config[].chunk * r
        var rank_count = runtime_rank_chunk_count(
            config[].tp, config[].chunk, config[].rem, r
        )
        runtime_tile_dispatch(
            reduce_buf,
            RuntimeReduceStoreKernel[
                dtype, src_origin, src_table_origin,
                ImmutOrigin(dst_table_origin), cfg_origin, Accum,
            ](config, r, 0, 0),
            (pools + r)[],
            rank_count,
            rank_start,
        )
    join_runtime(pools, config[].tp)

    for r in range(config[].tp):
        runtime_tile_dispatch(
            gather_buf,
            RuntimeGatherKernel[
                dtype, src_origin, src_table_origin,
                ImmutOrigin(dst_table_origin), cfg_origin,
            ](config, r, 0, 0),
            (pools + r)[],
            config[].count,
        )
    join_runtime(pools, config[].tp)


def runtime_allgather_no_alloc[
    P: BurstThreadPool,
    pool_origin: MutOrigin,
    dtype: DType,
    src_origin: ImmutOrigin,
    src_table_origin: ImmutOrigin,
    dst_origin: MutOrigin,
    dst_table_origin: ImmutOrigin,
    cfg_origin: ImmutOrigin,
](
    config: UnsafePointer[
        RuntimeGatherConfig[dtype, src_origin, src_table_origin], cfg_origin,
    ],
    dst: RuntimeRankBuffers[dtype, dst_origin, dst_table_origin],
    pools: UnsafePointer[P, pool_origin],
    mut gather_buf: DispatchBuffer[
        RuntimeAllGatherKernel[dtype, src_origin, src_table_origin, cfg_origin]
    ],
    inline_max_elements: Int,
):
    if config[].shard_count <= 0:
        return

    if config[].tp <= 1:
        if config[].src[0] != dst[0]:
            memcpy(dest=dst[0].as_any_origin(), src=config[].src[0], count=dst.count)
        return

    if config[].shard_count <= inline_max_elements:
        for r in range(config[].tp):
            var d = dst[r].as_any_origin()
            for src_rank in range(config[].tp):
                memcpy(
                    dest=d + src_rank * config[].shard_count,
                    src=config[].src[src_rank],
                    count=config[].shard_count,
                )
        return

    for r in range(config[].tp):
        runtime_tile_dispatch(
            gather_buf,
            RuntimeAllGatherKernel[dtype, src_origin, src_table_origin, cfg_origin](
                config, dst[r].as_any_origin(), 0, 0
            ),
            (pools + r)[],
            config[].shard_count,
        )
    join_runtime(pools, config[].tp)


def check_allreduce_runtime_tp(tp: Int, inline_max_elements: Int):
    comptime N = 8
    comptime dtype = DType.float32
    var buffers = InlineArray[InlineArray[Float32, N], MAX_TP](uninitialized=True)
    var src_table = InlineArray[
        UnsafePointer[Float32, ImmutOrigin(origin_of(buffers))], MAX_TP,
    ](uninitialized=True)
    var dst_table = InlineArray[
        UnsafePointer[Float32, origin_of(buffers)], MAX_TP,
    ](uninitialized=True)
    var dst_any_table = InlineArray[DstPtr[dtype], MAX_TP](uninitialized=True)
    var pools = InlineArray[TestPool, MAX_TP](uninitialized=True)

    for r in range(tp):
        pools[r] = TestPool(2, 0)
        src_table[r] = UnsafePointer(to=buffers[r][0]).as_immutable()
        dst_table[r] = UnsafePointer(to=buffers[r][0])
        for i in range(N):
            buffers[r][i] = Float32(r * 10 + i)

    var src = RuntimeRankBuffers[
        dtype, ImmutOrigin(origin_of(buffers)), ImmutOrigin(origin_of(src_table)),
    ](UnsafePointer(to=src_table[0]).as_immutable(), tp, N)
    var dst = RuntimeRankBuffers[
        dtype, origin_of(buffers), ImmutOrigin(origin_of(dst_table)),
    ](UnsafePointer(to=dst_table[0]).as_immutable(), tp, N)
    var cfg = build_runtime_reduce_config[
        dtype,
        ImmutOrigin(origin_of(buffers)),
        ImmutOrigin(origin_of(src_table)),
        origin_of(buffers),
        ImmutOrigin(origin_of(dst_table)),
        origin_of(dst_any_table),
    ](src, dst, UnsafePointer(to=dst_any_table[0]))
    var config = UnsafePointer(to=cfg).as_immutable()
    comptime cfg_ro = ImmutOrigin(origin_of(cfg))

    var reduce_buf = DispatchBuffer[
        RuntimeReduceStoreKernel[
            dtype, ImmutOrigin(origin_of(buffers)), ImmutOrigin(origin_of(src_table)),
            ImmutOrigin(origin_of(dst_any_table)), cfg_ro,
        ]
    ]()
    var gather_buf = DispatchBuffer[
        RuntimeGatherKernel[
            dtype, ImmutOrigin(origin_of(buffers)), ImmutOrigin(origin_of(src_table)),
            ImmutOrigin(origin_of(dst_any_table)), cfg_ro,
        ]
    ]()

    runtime_allreduce_no_alloc[
        TestPool, origin_of(pools), dtype,
        ImmutOrigin(origin_of(buffers)), ImmutOrigin(origin_of(src_table)),
        origin_of(buffers), origin_of(dst_any_table), cfg_ro,
    ](
        config,
        UnsafePointer(to=pools[0]),
        reduce_buf,
        gather_buf,
        inline_max_elements,
    )

    var rank_sum = Float32(0)
    for r in range(tp):
        rank_sum += Float32(r * 10)
    for r in range(tp):
        for i in range(N):
            debug_assert(
                buffers[r][i] == rank_sum + Float32(tp * i),
                "runtime TP allreduce mismatch",
            )


def check_allgather_runtime_tp(tp: Int, inline_max_elements: Int):
    comptime N = 5
    comptime OUT_N = N * MAX_TP
    comptime dtype = DType.float32
    var inputs = InlineArray[InlineArray[Float32, N], MAX_TP](uninitialized=True)
    var outputs = InlineArray[InlineArray[Float32, OUT_N], MAX_TP](uninitialized=True)
    var src_table = InlineArray[
        UnsafePointer[Float32, ImmutOrigin(origin_of(inputs))], MAX_TP,
    ](uninitialized=True)
    var dst_table = InlineArray[
        UnsafePointer[Float32, origin_of(outputs)], MAX_TP,
    ](uninitialized=True)
    var pools = InlineArray[TestPool, MAX_TP](uninitialized=True)

    for r in range(tp):
        pools[r] = TestPool(2, 0)
        src_table[r] = UnsafePointer(to=inputs[r][0]).as_immutable()
        dst_table[r] = UnsafePointer(to=outputs[r][0])
        for i in range(N):
            inputs[r][i] = Float32(r * 10 + i)
        for i in range(OUT_N):
            outputs[r][i] = Float32(-1)

    var src = RuntimeRankBuffers[
        dtype, ImmutOrigin(origin_of(inputs)), ImmutOrigin(origin_of(src_table)),
    ](UnsafePointer(to=src_table[0]).as_immutable(), tp, N)
    var dst = RuntimeRankBuffers[
        dtype, origin_of(outputs), ImmutOrigin(origin_of(dst_table)),
    ](UnsafePointer(to=dst_table[0]).as_immutable(), tp, N)
    var cfg = RuntimeGatherConfig[
        dtype, ImmutOrigin(origin_of(inputs)), ImmutOrigin(origin_of(src_table)),
    ](src.ptrs, tp, N)
    var config = UnsafePointer(to=cfg).as_immutable()
    comptime cfg_ro = ImmutOrigin(origin_of(cfg))
    var gather_buf = DispatchBuffer[
        RuntimeAllGatherKernel[
            dtype, ImmutOrigin(origin_of(inputs)), ImmutOrigin(origin_of(src_table)),
            cfg_ro,
        ]
    ]()

    runtime_allgather_no_alloc[
        TestPool, origin_of(pools), dtype,
        ImmutOrigin(origin_of(inputs)), ImmutOrigin(origin_of(src_table)),
        origin_of(outputs), ImmutOrigin(origin_of(dst_table)), cfg_ro,
    ](
        config,
        dst,
        UnsafePointer(to=pools[0]),
        gather_buf,
        inline_max_elements,
    )

    for r in range(tp):
        for src_rank in range(tp):
            for i in range(N):
                debug_assert(
                    outputs[r][src_rank * N + i] == Float32(src_rank * 10 + i),
                    "runtime TP allgather mismatch",
                )


def main():
    check_allreduce_runtime_tp(2, 8192)
    check_allreduce_runtime_tp(2, 0)
    check_allreduce_runtime_tp(3, 8192)
    check_allreduce_runtime_tp(3, 0)
    check_allgather_runtime_tp(2, 8192)
    check_allgather_runtime_tp(2, 0)
    check_allgather_runtime_tp(3, 8192)
    check_allgather_runtime_tp(3, 0)
    print("runtime TP static storage prototype ok")
