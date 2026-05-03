from std.algorithm import vectorize
from std.collections import InlineArray
from std.memory import UnsafePointer, alloc, memcpy
from std.sys.info import simd_width_of


comptime DstPtr[dtype: DType] = UnsafePointer[Scalar[dtype], MutAnyOrigin]


@fieldwise_init
struct FrozenConfig:
    var n: Int


def frozen_ptr[T: AnyType, origin: Origin](
    ref[origin] value: T
) -> UnsafePointer[T, ImmutOrigin(origin)]:
    return UnsafePointer(to=value).as_immutable()


@always_inline
def copy_elements[
    dtype: DType, src_origin: Origin, dst_origin: MutOrigin,
](
    dst: UnsafePointer[Scalar[dtype], dst_origin],
    src: UnsafePointer[Scalar[dtype], src_origin],
    count: Int,
):
    if count <= 0:
        return
    memcpy(dest=dst, src=src, count=count)


@always_inline
def reduce_sources_to[
    src_dtype: DType,
    dst_dtype: DType,
    tp: Int,
    src_origin: ImmutOrigin,
    Accum: DType = DType.float32,
](
    srcs: InlineArray[UnsafePointer[Scalar[src_dtype], src_origin], tp],
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


def check_frozen_ptr():
    var cfg = FrozenConfig(7)
    var ptr = frozen_ptr(cfg)
    comptime cfg_ro = ImmutOrigin(origin_of(cfg))
    var same_type: UnsafePointer[FrozenConfig, cfg_ro] = ptr
    debug_assert(same_type[].n == 7, "frozen ptr mismatch")


def check_copy_elements():
    comptime N = 8
    var src = alloc[Float32](N)
    var dst = alloc[Float32](N)
    for i in range(N):
        src[i] = Float32(10 + i)
        dst[i] = Float32(-1)

    copy_elements[DType.float32, ImmutOrigin(MutExternalOrigin), MutExternalOrigin](
        dst, src.as_immutable(), N
    )

    for i in range(N):
        debug_assert(dst[i] == Float32(10 + i), "copy elements mismatch")


def check_reduce_sources_to():
    comptime TP = 2
    comptime N = 8
    comptime SrcT = Scalar[DType.bfloat16]
    var a = alloc[SrcT](N)
    var b = alloc[SrcT](N)
    var out = alloc[SrcT](N)
    var scratch = alloc[Float32](N)

    for i in range(N):
        a[i] = SrcT(Float32(i))
        b[i] = SrcT(Float32(10 + i))
        out[i] = SrcT(Float32(-1))
        scratch[i] = Float32(-1)

    var srcs = InlineArray[
        UnsafePointer[SrcT, ImmutOrigin(MutExternalOrigin)], TP,
    ](uninitialized=True)
    srcs[0] = a.as_immutable()
    srcs[1] = b.as_immutable()

    reduce_sources_to[
        DType.bfloat16, DType.bfloat16, TP, ImmutOrigin(MutExternalOrigin),
    ](srcs, out.as_any_origin(), 0, N)
    reduce_sources_to[
        DType.bfloat16, DType.float32, TP, ImmutOrigin(MutExternalOrigin),
    ](srcs, scratch.as_any_origin(), 0, N)

    for i in range(N):
        var expected = Float32(10 + 2 * i)
        debug_assert(Float32(out[i]) == expected, "reduced output mismatch")
        debug_assert(scratch[i] == expected, "reduced scratch mismatch")


def main():
    check_frozen_ptr()
    check_copy_elements()
    check_reduce_sources_to()
    print("reduction extraction primitives prototype ok")
