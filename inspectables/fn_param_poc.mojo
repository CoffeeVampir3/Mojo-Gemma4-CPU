from std.memory import UnsafePointer, alloc
from std.collections import InlineArray
from std.benchmark import keep
from std.sys.info import simd_width_of

from simd_math import pick_port_unroll, fast_exp_softmax_biased


comptime W = simd_width_of[DType.float32]()
comptime F32Ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


trait MergeStrategy:
    @staticmethod
    def accumulate[hd: Int, ns: Int](
        dst: BF16Ptr,
        sources: InlineArray[F32Ptr, ns],
        h: Int,
        global_m: Scalar[DType.float32],
        m_off: Int,
        l_off: Int,
    ): ...


struct Unrolled(MergeStrategy):
    @staticmethod
    def accumulate[hd: Int, ns: Int](
        dst: BF16Ptr,
        sources: InlineArray[F32Ptr, ns],
        h: Int,
        global_m: Scalar[DType.float32],
        m_off: Int,
        l_off: Int,
    ):
        comptime PU = pick_port_unroll[W, hd]()
        comptime STRIDE = PU * W

        var deltas = SIMD[DType.float32, ns](0)
        var ls = SIMD[DType.float32, ns](0)
        comptime for s in range(ns):
            deltas[s] = (sources[s] + m_off + h)[] - global_m
            ls[s] = (sources[s] + l_off + h)[]

        var corrections = fast_exp_softmax_biased[ns](deltas)
        corrections = ls.gt(SIMD[DType.float32, ns](0)).select(
            corrections, SIMD[DType.float32, ns](0))
        var global_l = (ls * corrections).reduce_add()

        if global_l <= 0:
            for i in range(hd // STRIDE):
                comptime for p in range(PU):
                    (dst + i * STRIDE + p * W).store(SIMD[DType.bfloat16, W](0))
            return

        var inv_l = Scalar[DType.float32](1.0) / global_l
        for i in range(hd // STRIDE):
            comptime for p in range(PU):
                var accs = InlineArray[SIMD[DType.float32, W], ns](
                    fill=SIMD[DType.float32, W](0))
                comptime for s in range(ns):
                    var v = (sources[s] + h * hd + i * STRIDE + p * W).load[width=W]()
                    accs[s] = v * SIMD[DType.float32, W](corrections[s])
                var merged = accs[0]
                comptime for s in range(1, ns):
                    merged += accs[s]
                (dst + i * STRIDE + p * W).store(
                    (merged * SIMD[DType.float32, W](inv_l)).cast[DType.bfloat16]())


struct Batched(MergeStrategy):
    @staticmethod
    def accumulate[hd: Int, ns: Int](
        dst: BF16Ptr,
        sources: InlineArray[F32Ptr, ns],
        h: Int,
        global_m: Scalar[DType.float32],
        m_off: Int,
        l_off: Int,
    ):
        comptime PU = pick_port_unroll[W, hd]()
        comptime STRIDE = PU * W

        var global_l = Scalar[DType.float32](0)
        var batch_start = 0
        while batch_start < ns:
            var batch_end = min(batch_start + W, ns)
            var batch_len = batch_end - batch_start
            var deltas = SIMD[DType.float32, W](-1e30)
            var batch_ls = SIMD[DType.float32, W](0)
            for b in range(batch_len):
                deltas[b] = (sources[batch_start + b] + m_off + h)[] - global_m
                batch_ls[b] = (sources[batch_start + b] + l_off + h)[]
            var corrs = fast_exp_softmax_biased[W](deltas)
            corrs = batch_ls.gt(SIMD[DType.float32, W](0)).select(
                corrs, SIMD[DType.float32, W](0))
            global_l += (batch_ls * corrs).reduce_add()
            batch_start += W

        if global_l <= 0:
            for i in range(hd // STRIDE):
                comptime for p in range(PU):
                    (dst + i * STRIDE + p * W).store(SIMD[DType.bfloat16, W](0))
            return

        var inv_l = Scalar[DType.float32](1.0) / global_l
        var dst_f32 = dst.bitcast[Scalar[DType.float32]]().as_any_origin()
        var first = True

        batch_start = 0
        while batch_start < ns:
            var batch_end = min(batch_start + W, ns)
            var batch_len = batch_end - batch_start
            var deltas = SIMD[DType.float32, W](-1e30)
            var batch_ls = SIMD[DType.float32, W](0)
            for b in range(batch_len):
                deltas[b] = (sources[batch_start + b] + m_off + h)[] - global_m
                batch_ls[b] = (sources[batch_start + b] + l_off + h)[]
            var corrs = fast_exp_softmax_biased[W](deltas)
            corrs = batch_ls.gt(SIMD[DType.float32, W](0)).select(
                corrs, SIMD[DType.float32, W](0))

            for b in range(batch_len):
                var c = corrs[b]
                if c <= 0:
                    continue
                var src = sources[batch_start + b] + h * hd
                if first:
                    for i in range(hd // STRIDE):
                        comptime for p in range(PU):
                            (dst_f32 + i * STRIDE + p * W).store(
                                (src + i * STRIDE + p * W).load[width=W]()
                                * SIMD[DType.float32, W](c * inv_l))
                    first = False
                else:
                    for i in range(hd // STRIDE):
                        comptime for p in range(PU):
                            var v = (src + i * STRIDE + p * W).load[width=W]()
                            var a = (dst_f32 + i * STRIDE + p * W).load[width=W]()
                            (dst_f32 + i * STRIDE + p * W).store(
                                v.fma(SIMD[DType.float32, W](c * inv_l), a))
            batch_start += W

        for i in range(hd // STRIDE):
            comptime for p in range(PU):
                var a = (dst_f32 + i * STRIDE + p * W).load[width=W]()
                (dst + i * STRIDE + p * W).store(a.cast[DType.bfloat16]())


def merge_partials[
    head_dim: Int, num_q: Int, num_q_out: Int, num_sources: Int,
    S: MergeStrategy,
](
    output: BF16Ptr,
    sources: InlineArray[F32Ptr, num_sources],
    partial_stride: Int,
    head_offset: Int = 0,
):
    comptime m_off = num_q * head_dim
    comptime l_off = m_off + num_q

    for local_h in range(num_q_out):
        var h = head_offset + local_h

        var global_m = Scalar[DType.float32](-1e30)
        comptime for s in range(num_sources):
            var sm = (sources[s] + m_off + h)[]
            if sm > global_m:
                global_m = sm

        S.accumulate[head_dim, num_sources](
            output + local_h * head_dim, sources, h, global_m, m_off, l_off)


def main():
    comptime HEAD_DIM = 256
    comptime NUM_Q = 2
    comptime NS = 4
    comptime M_OFF = NUM_Q * HEAD_DIM
    comptime L_OFF = M_OFF + NUM_Q
    comptime PARTIAL_ELEMS = L_OFF + NUM_Q

    var sources = InlineArray[F32Ptr, NS](uninitialized=True)
    for s in range(NS):
        sources[s] = alloc[Scalar[DType.float32]](PARTIAL_ELEMS)
        for i in range(M_OFF):
            sources[s][i] = Scalar[DType.float32](Float64(i % 97 - 48) * 0.01)
        for h in range(NUM_Q):
            (sources[s] + M_OFF + h)[] = Scalar[DType.float32](Float64(s * 3 + h) * 0.1)
            (sources[s] + L_OFF + h)[] = Scalar[DType.float32](Float64(h + 1) * 0.5)

    var out_a = alloc[Scalar[DType.bfloat16]](NUM_Q * HEAD_DIM)
    var out_b = alloc[Scalar[DType.bfloat16]](NUM_Q * HEAD_DIM)

    merge_partials[HEAD_DIM, NUM_Q, NUM_Q, NS, Unrolled](
        out_a, sources, PARTIAL_ELEMS)
    merge_partials[HEAD_DIM, NUM_Q, NUM_Q, NS, Batched](
        out_b, sources, PARTIAL_ELEMS)

    print("unrolled[0..3]:",
        out_a[0].cast[DType.float32](),
        out_a[1].cast[DType.float32](),
        out_a[2].cast[DType.float32](),
        out_a[3].cast[DType.float32]())
    print("batched[0..3]:",
        out_b[0].cast[DType.float32](),
        out_b[1].cast[DType.float32](),
        out_b[2].cast[DType.float32](),
        out_b[3].cast[DType.float32]())

    for s in range(NS):
        sources[s].free()
    out_a.free()
    out_b.free()
