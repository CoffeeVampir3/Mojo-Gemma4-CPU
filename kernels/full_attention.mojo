from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from simd_math import pick_port_unroll, tree_reduce_accs, fast_exp_softmax_biased
from threading.threading_traits import BurstThreadPool
from .helpers import DispatchBuffer, recommended_workers, worker_range


comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
comptime W = simd_width_of[DType.float32]()
comptime TILE = W

comptime PARTIAL_STRIDE[num_q: Int, head_dim: Int]: Int = (
    (num_q * head_dim + num_q + num_q) * 4 + 63) // 64 * 16


@always_inline
def dot[head_dim: Int](q: BF16Ptr, k: BF16Ptr) -> Scalar[DType.float32]:
    comptime PU = pick_port_unroll[W, head_dim]()
    comptime STRIDE = PU * W
    var accs = InlineArray[SIMD[DType.float32, W], PU](fill=SIMD[DType.float32, W](0))
    for i in range(head_dim // STRIDE):
        comptime for p in range(PU):
            var qv = (q + i * STRIDE + p * W).load[width=W]().cast[DType.float32]()
            var kv = (k + i * STRIDE + p * W).load[width=W]().cast[DType.float32]()
            accs[p] = qv.fma(kv, accs[p])
    return tree_reduce_accs(accs)


@always_inline
def scale_acc[head_dim: Int](acc: F32Ptr, factor: Scalar[DType.float32]):
    comptime PU = pick_port_unroll[W, head_dim]()
    comptime STRIDE = PU * W
    var f = SIMD[DType.float32, W](factor)
    for i in range(head_dim // STRIDE):
        comptime for p in range(PU):
            var a = (acc + i * STRIDE + p * W).load[width=W]()
            (acc + i * STRIDE + p * W).store(a * f)


@always_inline
def weighted_add[head_dim: Int](
    acc: F32Ptr, v_row: BF16Ptr, weight: Scalar[DType.float32],
):
    comptime PU = pick_port_unroll[W, head_dim]()
    comptime STRIDE = PU * W
    var w = SIMD[DType.float32, W](weight)
    for i in range(head_dim // STRIDE):
        comptime for p in range(PU):
            var v = (v_row + i * STRIDE + p * W).load[width=W]().cast[DType.float32]()
            var a = (acc + i * STRIDE + p * W).load[width=W]()
            (acc + i * STRIDE + p * W).store(v.fma(w, a))


@fieldwise_init
struct FullAttentionKernel[
    head_dim: Int, num_q: Int, num_kv: Int, gqa_ratio: Int,
    kv_stride: Int,
](RangedKernel):
    var q: BF16Ptr
    var k_base: BF16Ptr
    var v_base: BF16Ptr
    var partials: F32Ptr
    var partial_stride: Int
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var my_partial = self.partials + self.worker_id * self.partial_stride
        comptime m_off = Self.num_q * Self.head_dim
        comptime l_off = m_off + Self.num_q

        var acc_ptrs = InlineArray[F32Ptr, Self.num_q](uninitialized=True)
        var q_ptrs = InlineArray[BF16Ptr, Self.num_q](uninitialized=True)
        var m = InlineArray[Scalar[DType.float32], Self.num_q](
            fill=Scalar[DType.float32](-1e30))
        var l = InlineArray[Scalar[DType.float32], Self.num_q](
            fill=Scalar[DType.float32](0))

        comptime for h in range(Self.num_q):
            acc_ptrs[h] = my_partial + h * Self.head_dim
            q_ptrs[h] = self.q + h * Self.head_dim
            for j in range(0, Self.head_dim, W):
                (acc_ptrs[h] + j).store(SIMD[DType.float32, W](0))

        var pos = self.start
        while pos < self.end:
            var tile_len = min(TILE, self.end - pos)

            comptime for q_idx in range(Self.num_q):
                comptime kv_h = q_idx // Self.gqa_ratio

                var scores = SIMD[DType.float32, TILE](-1e30)
                for t in range(tile_len):
                    var k_head = self.k_base + (pos + t) * Self.kv_stride + kv_h * Self.head_dim
                    scores[t] = dot[Self.head_dim](q_ptrs[q_idx], k_head)

                var tile_max = scores.reduce_max()
                var m_new = tile_max if tile_max > m[q_idx] else m[q_idx]

                var corr = fast_exp_softmax_biased[1](
                    SIMD[DType.float32, 1](m[q_idx] - m_new))[0]
                var weights = fast_exp_softmax_biased[TILE](
                    scores - SIMD[DType.float32, TILE](m_new))

                scale_acc[Self.head_dim](acc_ptrs[q_idx], corr)
                l[q_idx] = l[q_idx] * corr + weights.reduce_add()
                m[q_idx] = m_new

                for t in range(tile_len):
                    var v_head = self.v_base + (pos + t) * Self.kv_stride + kv_h * Self.head_dim
                    weighted_add[Self.head_dim](acc_ptrs[q_idx], v_head, weights[t])

            pos += TILE

        comptime for h in range(Self.num_q):
            (my_partial + m_off + h)[] = m[h]
            (my_partial + l_off + h)[] = l[h]

    def over_range(self, start: Int, end: Int) -> Self:
        return Self(self.q, self.k_base, self.v_base, self.partials,
            self.partial_stride, self.worker_id, start, end)


def dispatch_full_attention[
    P: BurstThreadPool, //,
    head_dim: Int, num_q: Int, num_kv: Int, gqa_ratio: Int,
    kv_stride: Int,
](
    q: BF16Ptr,
    k_base: BF16Ptr,
    v_base: BF16Ptr,
    worker_partials: F32Ptr,
    valid_len: Int,
    mut pool: P,
) -> Int:
    comptime partial_stride = PARTIAL_STRIDE[num_q, head_dim]

    if valid_len <= 0:
        return 0

    var nw = recommended_workers(valid_len * kv_stride * 2, pool.get_capacity())

    var buf = DispatchBuffer[
        FullAttentionKernel[head_dim, num_q, num_kv, gqa_ratio, kv_stride]]()
    for w in range(nw):
        var wr = worker_range(valid_len, nw, w)
        buf.slot()[] = FullAttentionKernel[
            head_dim, num_q, num_kv, gqa_ratio, kv_stride](
            q, k_base, v_base, worker_partials,
            partial_stride, w, wr[0], wr[1])
    buf.dispatch(pool)
    pool.join()
    return nw
