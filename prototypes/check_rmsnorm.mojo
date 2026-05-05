from kernels.rmsnorm import rms_norm_row, fused_residual_norm_row, rms_norm, rms_normalize_row
from std.memory import UnsafePointer
from simd_math.ops import sqrt

from numa import NumaArena, NumaInfo, NumaTopology
from threading import BurstPool
from notstdcollections import HeapMoveArray


comptime HIDDEN = 2816
comptime SQRT_N = sqrt[DType.float32, 1](HIDDEN)
comptime N_EPS = HIDDEN * 1e-6


def main():
    var numa = NumaInfo()
    var topo = numa.plan_topology(1)

    comptime ARENA_BYTES = 16 * 1024 * 1024
    var arena = NumaArena[alignment=64](topo[0], ARENA_BYTES)
    if not arena:
        print("arena alloc failed")
        return

    var src = arena.alloc[Scalar[DType.bfloat16]](HIDDEN).value()
    var dst = arena.alloc[Scalar[DType.bfloat16]](HIDDEN).value()
    var weight = arena.alloc[Scalar[DType.bfloat16]](HIDDEN).value()

    for i in range(HIDDEN):
        src[i] = Scalar[DType.bfloat16](Float32(i % 64) * 0.01)
        weight[i] = Scalar[DType.bfloat16](Float32(1.0))

    rms_norm_row[HIDDEN, SQRT_N, N_EPS](src, dst, weight)
    print("standalone norm ok, dst[1]=" + String(dst[1].cast[DType.float32]())
          + " dst[100]=" + String(dst[100].cast[DType.float32]()))

    var partial = arena.alloc[Scalar[DType.bfloat16]](HIDDEN).value()
    var residual = arena.alloc[Scalar[DType.bfloat16]](HIDDEN).value()
    var res_dst = arena.alloc[Scalar[DType.bfloat16]](HIDDEN).value()
    var norm_dst = arena.alloc[Scalar[DType.bfloat16]](HIDDEN).value()

    for i in range(HIDDEN):
        partial[i] = Scalar[DType.bfloat16](Float32(i % 32) * 0.01)
        residual[i] = Scalar[DType.bfloat16](Float32(i % 32) * 0.005)

    fused_residual_norm_row[HIDDEN, SQRT_N, N_EPS](
        partial, residual, res_dst, norm_dst, weight)
    print("fused residual+norm ok, norm_dst[1]=" + String(norm_dst[1].cast[DType.float32]())
          + " res_dst[1]=" + String(res_dst[1].cast[DType.float32]()))

    var pools = HeapMoveArray[BurstPool[]](1)
    pools.push(BurstPool[].for_topology(numa, topo[0]))

    var src_big = arena.alloc[Scalar[DType.bfloat16]](32 * HIDDEN).value()
    var dst_big = arena.alloc[Scalar[DType.bfloat16]](32 * HIDDEN).value()
    for i in range(32 * HIDDEN):
        src_big[i] = Scalar[DType.bfloat16](Float32(i % 128) * 0.005)

    rms_norm[hidden=HIDDEN, sqrt_n=SQRT_N, n_eps=N_EPS](
        src_big, dst_big, weight, 32, pools[0])
    print("dispatched norm (32 tokens) ok, dst[0]=" + String(dst_big[0].cast[DType.float32]()))
