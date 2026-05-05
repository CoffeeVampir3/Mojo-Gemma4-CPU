from std.memory import UnsafePointer

from numa import NumaArena, NumaInfo, NumaTopology
from threading import BurstPool
from notstdcollections import HeapMoveArray
from std.collections import InlineArray
from std.sys.info import simd_width_of
from simd_math import pick_port_unroll, tree_reduce_accs
from kernels.gemv import dot_row, gemv, gemv_chained_qkv


comptime ALIGNMENT = 64
comptime HIDDEN = 2816
comptime Q_ROWS = 1024
comptime KV_ROWS = 512

comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


def fill_val(ptr: BF16Ptr, count: Int, val: Float32):
    for i in range(count):
        ptr[i] = Scalar[DType.bfloat16](val)


def fill_identity_row(ptr: BF16Ptr, cols: Int, row: Int):
    for j in range(cols):
        ptr[row * cols + j] = Scalar[DType.bfloat16](Float32(1.0) if j == row else Float32(0.0))


def main():
    var numa = NumaInfo()
    var topo = numa.plan_topology(1)

    comptime ARENA_BYTES = 128 * 1024 * 1024
    var arena = NumaArena[alignment=ALIGNMENT](topo[0], ARENA_BYTES)
    if not arena:
        print("arena alloc failed")
        return

    var input_vec = arena.alloc[Scalar[DType.bfloat16]](HIDDEN).value()
    var weight = arena.alloc[Scalar[DType.bfloat16]](Q_ROWS * HIDDEN).value()
    var output = arena.alloc[Scalar[DType.bfloat16]](Q_ROWS).value()

    fill_val(input_vec, HIDDEN, 0.01)
    fill_val(weight, Q_ROWS * HIDDEN, 0.5)

    comptime FW = simd_width_of[DType.float32]()
    comptime PU = pick_port_unroll[FW, HIDDEN]()
    var accs = InlineArray[SIMD[DType.float32, FW], PU](fill=SIMD[DType.float32, FW](0))
    dot_row[HIDDEN, PU](input_vec, weight, accs)
    var d = tree_reduce_accs(accs)
    print("dot_row (all 0.01 · all 0.5, 2816 elems): " + String(d))

    var pools = HeapMoveArray[BurstPool[]](1)
    pools.push(BurstPool[].for_topology(numa, topo[0]))

    gemv[rows=Q_ROWS, cols=HIDDEN](input_vec, weight, output, pools[0])
    print("matvec output[0]=" + String(output[0].cast[DType.float32]())
          + " output[511]=" + String(output[511].cast[DType.float32]()))

    var q_weight = arena.alloc[Scalar[DType.bfloat16]](Q_ROWS * HIDDEN).value()
    var k_weight = arena.alloc[Scalar[DType.bfloat16]](KV_ROWS * HIDDEN).value()
    var v_weight = arena.alloc[Scalar[DType.bfloat16]](KV_ROWS * HIDDEN).value()
    var q_out = arena.alloc[Scalar[DType.bfloat16]](Q_ROWS).value()
    var k_out = arena.alloc[Scalar[DType.bfloat16]](KV_ROWS).value()
    var v_out = arena.alloc[Scalar[DType.bfloat16]](KV_ROWS).value()

    fill_val(q_weight, Q_ROWS * HIDDEN, 0.5)
    fill_val(k_weight, KV_ROWS * HIDDEN, 0.25)
    fill_val(v_weight, KV_ROWS * HIDDEN, 0.125)

    gemv_chained_qkv[q_rows=Q_ROWS, kv_rows=KV_ROWS, cols=HIDDEN](
        input_vec, q_weight, k_weight, v_weight,
        q_out, k_out, v_out, pools[0])

    print("chained QKV:")
    print("  q_out[0]=" + String(q_out[0].cast[DType.float32]()))
    print("  k_out[0]=" + String(k_out[0].cast[DType.float32]()))
    print("  v_out[0]=" + String(v_out[0].cast[DType.float32]()))
