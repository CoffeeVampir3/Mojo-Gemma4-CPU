from std.memory import UnsafePointer
from std.sys.info import CompilationTarget

from butterquant.types import BF16Ptr


@always_inline
def has_avx512_bf16() -> Bool:
    return CompilationTarget._has_feature["avx512bf16"]()


@always_inline
def store_bf16[width: Int](
    v: SIMD[DType.float32, width], dst: BF16Ptr,
):
    comptime if has_avx512_bf16():
        dst.store(v.cast[DType.bfloat16]())
    else:
        var bits = v.to_bits().cast[DType.uint32]()
        var rne = (bits + ((bits >> 16) & SIMD[DType.uint32, width](1))
            + SIMD[DType.uint32, width](0x7FFF))
        dst.bitcast[UInt16]().store((rne >> 16).cast[DType.uint16]())


@always_inline
def store_out[Out: DType, width: Int](
    v: SIMD[DType.float32, width],
    dst: UnsafePointer[Scalar[Out], MutAnyOrigin],
):
    comptime if Out == DType.bfloat16:
        store_bf16[width](v, dst.bitcast[Scalar[DType.bfloat16]]())
    else:
        dst.store(v.cast[Out]())
