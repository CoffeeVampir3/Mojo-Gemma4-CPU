from std.algorithm import vectorize
from std.collections import InlineArray
from std.memory import Span, UnsafePointer, alloc, memcpy
from std.sys.info import simd_width_of


comptime DstPtr[dtype: DType] = UnsafePointer[Scalar[dtype], MutAnyOrigin]


# ── Probe 1: reduce_sources_to ──────────────────────────────────────────
#
# The duplicated loop in reduce_store_range / reduce_to_scratch_range
# differs only in store dtype. This function unifies them.
# src_dtype is what we load, dst_dtype is what we store, Accum is the
# accumulation precision. The cast from Accum -> dst_dtype is a no-op
# when they match (scratch case).

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


def test_reduce_sources_to():
    comptime TP = 2
    comptime N = 16

    var a = alloc[Scalar[DType.bfloat16]](N)
    var b = alloc[Scalar[DType.bfloat16]](N)
    var out_bf16 = alloc[Scalar[DType.bfloat16]](N)
    var out_f32 = alloc[Float32](N)

    for i in range(N):
        a[i] = Scalar[DType.bfloat16](Float32(i))
        b[i] = Scalar[DType.bfloat16](Float32(10 + i))

    comptime src_ro = ImmutOrigin(MutExternalOrigin)
    var srcs = InlineArray[
        UnsafePointer[Scalar[DType.bfloat16], src_ro], TP,
    ](uninitialized=True)
    srcs[0] = a.as_immutable()
    srcs[1] = b.as_immutable()

    reduce_sources_to[DType.bfloat16, DType.bfloat16, TP, src_ro](
        srcs, out_bf16.as_any_origin(), 0, N)
    reduce_sources_to[DType.bfloat16, DType.float32, TP, src_ro](
        srcs, out_f32.as_any_origin(), 0, N)

    for i in range(N):
        var expected = Float32(10 + 2 * i)
        debug_assert(Float32(out_bf16[i]) == expected, "bf16 output mismatch")
        debug_assert(out_f32[i] == expected, "f32 scratch mismatch")

    reduce_sources_to[DType.bfloat16, DType.bfloat16, TP, src_ro](
        srcs, out_bf16.as_any_origin(), 4, 12)
    debug_assert(Float32(out_bf16[0]) == Float32(10), "partial should not touch idx 0")
    debug_assert(Float32(out_bf16[4]) == Float32(18), "partial idx 4 mismatch")
    debug_assert(Float32(out_bf16[11]) == Float32(32), "partial idx 11 mismatch")

    print("  reduce_sources_to: ok")


# ── Probe 2: RankBuffers without cursor ─────────────────────────────────
#
# cursor is construction-only state. insert_next forces sequential writes
# but the underlying InlineArray is random-access with comptime tp. Just
# expose ptrs directly — callers already access src.ptrs in config
# builders anyway.

struct RankBuffers[dtype: DType, tp: Int, origin: Origin]:
    var ptrs: InlineArray[UnsafePointer[Scalar[Self.dtype], Self.origin], Self.tp]
    var count: Int

    def __init__(out self, count: Int):
        self.ptrs = InlineArray[
            UnsafePointer[Scalar[Self.dtype], Self.origin], Self.tp,
        ](uninitialized=True)
        self.count = count

    @always_inline
    def __getitem__(self, rank: Int) -> UnsafePointer[Scalar[Self.dtype], Self.origin]:
        return self.ptrs[rank]


def test_rank_buffers_no_cursor():
    comptime TP = 2
    comptime N = 4
    var a = alloc[Float32](N)
    var b = alloc[Float32](N)
    for i in range(N):
        a[i] = Float32(i)
        b[i] = Float32(10 + i)

    comptime src_ro = ImmutOrigin(MutExternalOrigin)
    var src = RankBuffers[DType.float32, TP, src_ro](count=N)
    src.ptrs[0] = a.as_immutable()
    src.ptrs[1] = b.as_immutable()

    debug_assert(src[0][0] == Float32(0), "rank 0 idx 0 mismatch")
    debug_assert(src[1][0] == Float32(10), "rank 1 idx 0 mismatch")
    debug_assert(src.count == N, "count mismatch")

    var dst = RankBuffers[DType.float32, TP, MutExternalOrigin](count=N)
    dst.ptrs[0] = a
    dst.ptrs[1] = b

    dst[0][0] = Float32(99)
    debug_assert(a[0] == Float32(99), "write through dst failed")

    print("  RankBuffers without cursor: ok")


# ── Probe 3: src_origin on copy/gather ──────────────────────────────────
#
# copy_chunk and gather_chunks carry src_origin purely because
# ReduceConfig is parameterized on it. They only touch config[].dst
# (MutAnyOrigin) and config[].chunk/rem (Int). The src_origin is a
# comptime parameter inferred from the caller, zero runtime cost.
#
# Could we factor ReduceConfig to separate chunk metadata from pointer
# arrays? Let's see if a smaller "ChunkPlan" helps.

@fieldwise_init
struct ChunkPlan(Copyable, ImplicitlyCopyable):
    var chunk: Int
    var rem: Int
    var total_elements: Int


@fieldwise_init
struct ReduceLayout[E_DTYPE: DType, tp: Int, src_origin: ImmutOrigin]:
    var src: InlineArray[UnsafePointer[Scalar[Self.E_DTYPE], Self.src_origin], Self.tp]
    var dst: InlineArray[DstPtr[Self.E_DTYPE], Self.tp]
    var plan: ChunkPlan


def rank_chunk_count[tp: Int](chunk: Int, rem: Int, rank: Int) -> Int:
    if rank == tp - 1:
        return chunk + rem
    return chunk


def copy_chunk_from_plan[E_DTYPE: DType](
    dst: InlineArray[DstPtr[E_DTYPE], _],
    plan: ChunkPlan,
    dst_rank: Int, src_rank: Int, start: Int, end: Int,
):
    if start >= end:
        return
    memcpy(dest=dst[dst_rank] + start, src=dst[src_rank] + start, count=end - start)


def gather_from_plan[E_DTYPE: DType, tp: Int](
    dst: InlineArray[DstPtr[E_DTYPE], tp],
    plan: ChunkPlan,
    dst_rank: Int, start: Int, end: Int,
):
    for src_rank in range(tp):
        if src_rank == dst_rank:
            continue
        var src_start = plan.chunk * src_rank
        var src_count = rank_chunk_count[tp](plan.chunk, plan.rem, src_rank)
        copy_chunk_from_plan[E_DTYPE](
            dst, plan, dst_rank, src_rank,
            max(start, src_start), min(end, src_start + src_count))


def test_factored_gather():
    comptime TP = 2
    comptime N = 8
    var a = alloc[Float32](N)
    var b = alloc[Float32](N)

    for i in range(N):
        a[i] = Float32(10 + 2 * i)
        b[i] = Float32(10 + 2 * i)

    a[0] = Float32(100)
    a[1] = Float32(101)
    a[2] = Float32(102)
    a[3] = Float32(103)

    b[4] = Float32(200)
    b[5] = Float32(201)
    b[6] = Float32(202)
    b[7] = Float32(203)

    var dst = InlineArray[DstPtr[DType.float32], TP](uninitialized=True)
    dst[0] = a.as_any_origin()
    dst[1] = b.as_any_origin()

    var plan = ChunkPlan(chunk=N // TP, rem=0, total_elements=N)

    gather_from_plan[DType.float32, TP](dst, plan, 0, 0, N)
    gather_from_plan[DType.float32, TP](dst, plan, 1, 0, N)

    debug_assert(a[4] == Float32(200), "gather rank0 from rank1 failed")
    debug_assert(b[0] == Float32(100), "gather rank1 from rank0 failed")
    debug_assert(a[0] == Float32(100), "gather should not touch own chunk")

    print("  factored gather (no src_origin): ok")


def main():
    test_reduce_sources_to()
    test_rank_buffers_no_cursor()
    test_factored_gather()
    print("cleanup probe ok")
