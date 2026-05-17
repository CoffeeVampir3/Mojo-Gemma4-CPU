from std.memory import UnsafePointer

from kernels.gemv import dispatch_gemv, dispatch_gemv_softcap
from kernels.helpers import Binding, RankBuffers
from kernels.reductions import dispatch_allreduce, dispatch_broadcast
from kernels.rmsnorm import dispatch_rms_norm, fused_norm_residual_add
from modeling.model_spec import BF16
from notstdcollections import HeapMoveArray
from simd_math.ops import sqrt
from threading.threading_traits import BurstThreadPool


comptime BF16AnyImmut = ImmutOrigin(MutAnyOrigin)
comptime BF16Bind[tp: Int] = Binding[Scalar[DType.bfloat16], tp]
comptime BF16AnyPtr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


struct BF16Ops[degree: Int, max_worker_count: Int = 128]:
    @staticmethod
    def rms[
        P: BurstThreadPool, //, hidden: Int, eps: Float64 = 1e-6,
    ](
        src: BF16Bind[Self.degree],
        dst: BF16Bind[Self.degree],
        weight: BF16Bind[Self.degree],
        seq_len: Int,
        mut pools: HeapMoveArray[P],
    ):
        comptime sqrt_n = sqrt[DType.float32, 1](Float32(hidden))
        comptime n_eps = Scalar[DType.float32](Float32(hidden) * Float32(eps))
        dispatch_rms_norm[
            hidden=hidden, sqrt_n=sqrt_n, n_eps=n_eps,
            tp=Self.degree, max_worker_count=Self.max_worker_count,
        ](src, dst, weight, seq_len, pools)

    @staticmethod
    def fused_residual_norm_add[
        P: BurstThreadPool, //, hidden: Int, eps: Float64 = 1e-6,
    ](
        src: BF16Bind[Self.degree],
        residual: BF16Bind[Self.degree],
        dst: BF16Bind[Self.degree],
        weight: BF16Bind[Self.degree],
        seq_len: Int,
        mut pools: HeapMoveArray[P],
    ):
        comptime sqrt_n = sqrt[DType.float32, 1](Float32(hidden))
        comptime n_eps = Scalar[DType.float32](Float32(hidden) * Float32(eps))
        fused_norm_residual_add[
            hidden=hidden, sqrt_n=sqrt_n, n_eps=n_eps,
            tp=Self.degree, max_worker_count=Self.max_worker_count,
        ](src, residual, dst, weight, seq_len, pools)

    @staticmethod
    def gemv[P: BurstThreadPool, //, rows: Int, cols: Int](
        x: BF16Bind[Self.degree],
        weight: BF16Bind[Self.degree],
        output: BF16Bind[Self.degree],
        mut pools: HeapMoveArray[P],
    ):
        dispatch_gemv[
            rows=rows, cols=cols, tp=Self.degree,
            max_worker_count=Self.max_worker_count,
        ](x, weight, output, pools)

    @staticmethod
    def gemv_softcap[
        P: BurstThreadPool, //, rows: Int, cols: Int, cap: Float64,
    ](
        x: BF16Bind[Self.degree],
        weight: BF16Bind[Self.degree],
        output: BF16Bind[Self.degree],
        mut pools: HeapMoveArray[P],
    ):
        dispatch_gemv_softcap[
            rows=rows, cols=cols, cap=cap, tp=Self.degree,
            max_worker_count=Self.max_worker_count,
        ](x, weight, output, pools)

    @staticmethod
    def allreduce[P: BurstThreadPool](
        values: BF16Bind[Self.degree],
        count: Int,
        mut pools: HeapMoveArray[P],
    ):
        var src = RankBuffers[DType.bfloat16, Self.degree, BF16AnyImmut](
            count=count)
        var dst = RankBuffers[DType.bfloat16, Self.degree, MutAnyOrigin](
            count=count)
        for r in range(Self.degree):
            src.ptrs[r] = values[r].as_immutable()
            dst.ptrs[r] = values[r]
        dispatch_allreduce[
            BF16, Self.degree, max_worker_count=Self.max_worker_count,
        ](src, dst, pools)

    @staticmethod
    def allreduce_into[P: BurstThreadPool](
        src_values: BF16Bind[Self.degree],
        dst_values: BF16Bind[Self.degree],
        count: Int,
        mut pools: HeapMoveArray[P],
    ):
        var src = RankBuffers[DType.bfloat16, Self.degree, BF16AnyImmut](
            count=count)
        var dst = RankBuffers[DType.bfloat16, Self.degree, MutAnyOrigin](
            count=count)
        for r in range(Self.degree):
            src.ptrs[r] = src_values[r].as_immutable()
            dst.ptrs[r] = dst_values[r]
        dispatch_allreduce[
            BF16, Self.degree, max_worker_count=Self.max_worker_count,
        ](src, dst, pools)

    @staticmethod
    def broadcast_from_owner[P: BurstThreadPool](
        source: BF16AnyPtr,
        dst_values: BF16Bind[Self.degree],
        count: Int,
        owner: Int,
        mut pools: HeapMoveArray[P],
    ):
        var src = RankBuffers[DType.bfloat16, Self.degree, BF16AnyImmut](
            count=count)
        var dst = RankBuffers[DType.bfloat16, Self.degree, MutAnyOrigin](
            count=count)
        for r in range(Self.degree):
            src.ptrs[r] = source.as_immutable()
            dst.ptrs[r] = dst_values[r]
        dispatch_broadcast[
            BF16, Self.degree, max_worker_count=Self.max_worker_count,
        ](src, dst, pools, src_rank=owner)


def main():
    print("model call cleanup prototype")
