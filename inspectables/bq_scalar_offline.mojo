from std.collections import InlineArray
from std.sys import argv, llvm_intrinsic
from std.sys.info import simd_width_of
from std.benchmark import keep
from std.memory import UnsafePointer, alloc
from std.utils import IndexList

comptime PtrF32 = UnsafePointer[Float32, MutAnyOrigin]
comptime PtrU8 = UnsafePointer[UInt8, MutAnyOrigin]


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
    mut r: InlineArray[SIMD[T, fwht_width[T, block]()], block // fwht_width[T, block]()],
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
    var r = InlineArray[SIMD[DType.float32, width], regs](fill=SIMD[DType.float32, width](0))
    comptime for i in range(regs):
        r[i] = (buf + i * width).load[width=width]()
    fwht_apply[DType.float32, block](r)
    comptime for i in range(regs):
        (buf + i * width).store(r[i])


# A: source fwht_rotate_columns — scalar strided gather/scatter into scratch
def fwht_rotate_columns_a[head_dim: Int](work: PtrF32, rows: Int, cols: Int):
    var scratch_buf = List[Float32](length=head_dim, fill=Float32(0))
    var scratch = scratch_buf.unsafe_ptr().as_any_origin()
    var num_heads = rows // head_dim
    for h in range(num_heads):
        var base = h * head_dim
        for c in range(cols):
            for r in range(head_dim):
                (scratch + r).store((work + (base + r) * cols + c).load())
            fwht_block[head_dim](scratch)
            for r in range(head_dim):
                (work + (base + r) * cols + c).store((scratch + r).load())
    _ = scratch_buf^


# pack colsum inner loop (vnni.pack_and_colsum_impl core) — should vectorize
@always_inline
def colsum_block[simd_width: Int](
    src_row: PtrU8, scratch_row: PtrU8, base: Int, block_cols: Int,
) -> Float32:
    var acc = SIMD[DType.int32, simd_width](0)
    for k in range(0, block_cols, simd_width):
        var v = (src_row.bitcast[Int8]() + base + k).load[width=simd_width]()
        (scratch_row.bitcast[Int8]() + base + k).store(v)
        acc += v.cast[DType.int32]()
    return Float32(Int(acc.reduce_add()))


@no_inline
@export
def probe_rotate_columns_128(work: PtrF32, rows: Int, cols: Int):
    fwht_rotate_columns_a[128](work, rows, cols)


@no_inline
@export
def probe_colsum_block(src: PtrU8, scr: PtrU8, base: Int, bc: Int) -> Float32:
    return colsum_block[simd_width_of[DType.int8]()](src, scr, base, bc)


def main():
    var seed = len(argv())
    var work = alloc[Float32](128 * 256).as_any_origin()
    var src = alloc[UInt8](512).as_any_origin()
    var scr = alloc[UInt8](512).as_any_origin()
    for i in range(128 * 256):
        work[i] = Float32(i % 19 + seed)
    for i in range(512):
        src[i] = UInt8((i + seed) % 200)
    probe_rotate_columns_128(work, 128, 256)
    keep(probe_colsum_block(src, scr, 0, 128))
    keep(work[seed]); keep(scr[seed])
