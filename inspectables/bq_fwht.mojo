from std.collections import InlineArray
from std.sys import argv, llvm_intrinsic
from std.sys.info import simd_width_of
from std.benchmark import keep
from std.memory import UnsafePointer, alloc
from std.utils import IndexList

comptime PtrF32 = UnsafePointer[Float32, MutAnyOrigin]


def is_power_of_two[N: Int]() -> Bool:
    return N > 0 and (N & (N - 1)) == 0


def log2[N: Int]() -> Int:
    comptime if N == 1:
        return 0
    else:
        return 1 + log2[N // 2]()


def butterfly_shuffle[width: Int, stride: Int]() -> IndexList[width]:
    var result = IndexList[width]()
    comptime for i in range(width):
        result[i] = i ^ stride
    return result


@always_inline
def sqrt[dtype: DType, width: Int](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return llvm_intrinsic["llvm.sqrt", SIMD[dtype, width], SIMD[dtype, width]](x)


def fwht_width[T: DType, block: Int]() -> Int:
    comptime hw = simd_width_of[T]()
    comptime if block <= hw:
        return block
    else:
        return hw


@always_inline
def fwht_apply[T: DType, block: Int](
    mut r: InlineArray[
        SIMD[T, fwht_width[T, block]()],
        block // fwht_width[T, block](),
    ],
):
    comptime width = fwht_width[T, block]()
    comptime regs = block // width
    comptime stages = log2[block]()

    comptime for stage in range(stages):
        comptime stride = 1 << stage
        comptime if stride < width:
            comptime mask = butterfly_shuffle[width, stride]()
            var sign_buf = InlineArray[Scalar[T], width](fill=Scalar[T](1.0))
            comptime for k in range(width):
                comptime if (k >> stage) & 1 != 0:
                    sign_buf[k] = Scalar[T](-1.0)
            var sign = UnsafePointer(to=sign_buf).bitcast[Scalar[T]]().load[width=width]()
            comptime for i in range(regs):
                var partner = r[i].shuffle[mask=mask](r[i])
                r[i] = r[i].fma(sign, partner)
        else:
            comptime reg_stride = stride // width
            comptime num_groups = regs // (2 * reg_stride)
            comptime for g in range(num_groups):
                comptime for j in range(reg_stride):
                    comptime a_idx = g * 2 * reg_stride + j
                    comptime b_idx = a_idx + reg_stride
                    var a_val = r[a_idx]
                    var b_val = r[b_idx]
                    r[a_idx] = a_val + b_val
                    r[b_idx] = a_val - b_val

    var sc = Scalar[T](1.0 / Float64(sqrt[T, 1](Scalar[T](block))))
    comptime for i in range(regs):
        r[i] = r[i] * sc


@always_inline
def fwht_block[block: Int](buf: PtrF32):
    comptime width = fwht_width[DType.float32, block]()
    comptime regs = block // width
    var r = InlineArray[SIMD[DType.float32, width], regs](
        fill=SIMD[DType.float32, width](0))
    comptime for i in range(regs):
        r[i] = (buf + i * width).load[width=width]()
    fwht_apply[DType.float32, block](r)
    comptime for i in range(regs):
        (buf + i * width).store(r[i])


@no_inline
@export
def probe_fwht_16(buf: PtrF32):
    fwht_block[16](buf)


@no_inline
@export
def probe_fwht_64(buf: PtrF32):
    fwht_block[64](buf)


@no_inline
@export
def probe_fwht_256(buf: PtrF32):
    fwht_block[256](buf)


def main():
    var seed = len(argv())
    var buf = alloc[Float32](256).as_any_origin()
    for i in range(256):
        buf[i] = Float32(i + seed)
    probe_fwht_16(buf)
    probe_fwht_64(buf)
    probe_fwht_256(buf)
    keep(buf[0])
    keep(buf[seed])
