from std.memory import UnsafePointer, alloc
from std.collections import InlineArray
from std.benchmark import keep
from std.sys.info import simd_width_of


comptime W = simd_width_of[DType.float32]()
comptime HEAD_DIM = 256
comptime NUM_Q = 2
comptime NUM_KV = 1
comptime GQA_RATIO = 2
comptime KV_STRIDE = 256
comptime WINDOW = 1024
comptime NUM_POSITIONS = 64
comptime TILE = W

comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]


@always_inline
def port_unroll_for[count: Int]() -> Int:
    comptime assert count > 0, "port_unroll_for requires positive count"
    return 8 if count >= 8 else 4 if count >= 4 else 2 if count >= 2 else 1


@always_inline
def pick_port_unroll[width: Int, cols: Int]() -> Int:
    comptime assert cols >= width, "pick_port_unroll requires cols >= width"
    return port_unroll_for[cols // width]()


comptime PU = pick_port_unroll[W, HEAD_DIM]()
comptime STRIDE = PU * W


@always_inline
def tree_reduce_accs[T: DType, width: Int, port_unroll: Int, //](
    mut accs: InlineArray[SIMD[T, width], port_unroll],
) -> Scalar[T]:
    comptime for stride in range(1, port_unroll):
        comptime if (stride & (stride - 1)) == 0:
            comptime for i in range(0, port_unroll, 2 * stride):
                accs[i] += accs[i + stride]
    return accs[0].reduce_add()


@always_inline
def fast_exp_softmax_biased[width: Int](
    x: SIMD[DType.float32, width],
) -> SIMD[DType.float32, width]:
    comptime A_MAGIC = Float32(12102203.16156148)
    comptime BIAS_F = Float32(1059208216.0)
    comptime INV_TWO23 = Float32(1.0) / Float32(1 << 23)
    comptime QC_A = Float32(1.6501418352127075)
    comptime QC_B = Float32(-0.37554836273193359)
    comptime QC_C = Float32(0.38696467876434326)

    var i = (A_MAGIC * x + BIAS_F).cast[DType.int32]()
    var u = i.cast[DType.uint32]()
    var k = SIMD[DType.float32, width](from_bits=u)
    var fbits = u & SIMD[DType.uint32, width](0x7FFFFF)
    var f = fbits.cast[DType.float32]() * INV_TWO23
    return k * (QC_A + f * (QC_B + f * QC_C))


@always_inline
def score_position[head_dim: Int](q: BF16Ptr, k_row: BF16Ptr) -> Float32:
    comptime PU_ = pick_port_unroll[W, head_dim]()
    comptime S_ = PU_ * W
    var accs = InlineArray[SIMD[DType.float32, W], PU_](fill=SIMD[DType.float32, W](0))
    for i in range(head_dim // S_):
        comptime for p in range(PU_):
            var qv = (q + i * S_ + p * W).load[width=W]().cast[DType.float32]()
            var kv = (k_row + i * S_ + p * W).load[width=W]().cast[DType.float32]()
            accs[p] = qv.fma(kv, accs[p])
    return tree_reduce_accs(accs)


@always_inline
def scale_acc[head_dim: Int](acc: F32Ptr, factor: Float32):
    comptime PU_ = pick_port_unroll[W, head_dim]()
    comptime S_ = PU_ * W
    var f = SIMD[DType.float32, W](factor)
    for i in range(head_dim // S_):
        comptime for p in range(PU_):
            var a = (acc + i * S_ + p * W).load[width=W]()
            (acc + i * S_ + p * W).store(a * f)


@always_inline
def weighted_add[head_dim: Int](
    acc: F32Ptr, v_row: BF16Ptr, weight: Float32,
):
    comptime PU_ = pick_port_unroll[W, head_dim]()
    comptime S_ = PU_ * W
    var w = SIMD[DType.float32, W](weight)
    for i in range(head_dim // S_):
        comptime for p in range(PU_):
            var v = (v_row + i * S_ + p * W).load[width=W]().cast[DType.float32]()
            var a = (acc + i * S_ + p * W).load[width=W]()
            (acc + i * S_ + p * W).store(v.fma(w, a))


@always_inline
def accumulate_v_corrected[head_dim: Int](
    v_row: BF16Ptr, weight: Float32,
    correction: Float32, acc: F32Ptr,
):
    comptime PU_ = pick_port_unroll[W, head_dim]()
    comptime S_ = PU_ * W
    var w_vec = SIMD[DType.float32, W](weight)
    var c_vec = SIMD[DType.float32, W](correction)
    for i in range(head_dim // S_):
        comptime for p in range(PU_):
            var v = (v_row + i * S_ + p * W).load[width=W]().cast[DType.float32]()
            var a = (acc + i * S_ + p * W).load[width=W]()
            (acc + i * S_ + p * W).store(a.fma(c_vec, v * w_vec))


@no_inline
def flash_decode_original(
    q: BF16Ptr,
    k_base: BF16Ptr,
    v_base: BF16Ptr,
    partials: F32Ptr,
    start_pos: Int,
    start: Int,
    end: Int,
):
    comptime m_off = NUM_Q * HEAD_DIM
    comptime l_off = m_off + NUM_Q

    var acc_ptrs = InlineArray[F32Ptr, NUM_Q](uninitialized=True)
    var q_ptrs = InlineArray[BF16Ptr, NUM_Q](uninitialized=True)
    var m = InlineArray[Float32, NUM_Q](
        fill=Float32(-1e30))
    var l = InlineArray[Float32, NUM_Q](
        fill=Float32(0))

    comptime for h in range(NUM_Q):
        acc_ptrs[h] = partials + h * HEAD_DIM
        q_ptrs[h] = q + h * HEAD_DIM
        for j in range(0, HEAD_DIM, W):
            (acc_ptrs[h] + j).store(SIMD[DType.float32, W](0))

    for p in range(start, end):
        var cache_slot = (start_pos + p) & (WINDOW - 1)
        var k_row = k_base + cache_slot * KV_STRIDE
        var v_row = v_base + cache_slot * KV_STRIDE

        comptime for q_idx in range(NUM_Q):
            comptime kv_h = q_idx // GQA_RATIO
            var k_head = k_row + kv_h * HEAD_DIM
            var v_head = v_row + kv_h * HEAD_DIM

            var score = score_position[HEAD_DIM](q_ptrs[q_idx], k_head)

            var m_old = m[q_idx]
            var m_new = score if score > m_old else m_old
            var correction = fast_exp_softmax_biased[1](
                SIMD[DType.float32, 1](m_old - m_new))[0]
            var weight = fast_exp_softmax_biased[1](
                SIMD[DType.float32, 1](score - m_new))[0]

            l[q_idx] = l[q_idx] * correction + weight
            m[q_idx] = m_new

            accumulate_v_corrected[HEAD_DIM](
                v_head, weight, correction, acc_ptrs[q_idx])

    comptime for h in range(NUM_Q):
        (partials + m_off + h)[] = m[h]
        (partials + l_off + h)[] = l[h]


@no_inline
def flash_decode_tiled(
    q: BF16Ptr,
    k_base: BF16Ptr,
    v_base: BF16Ptr,
    partials: F32Ptr,
    start_pos: Int,
    start: Int,
    end: Int,
):
    comptime m_off = NUM_Q * HEAD_DIM
    comptime l_off = m_off + NUM_Q

    var acc_ptrs = InlineArray[F32Ptr, NUM_Q](uninitialized=True)
    var q_ptrs = InlineArray[BF16Ptr, NUM_Q](uninitialized=True)
    var m = InlineArray[Float32, NUM_Q](
        fill=Float32(-1e30))
    var l = InlineArray[Float32, NUM_Q](
        fill=Float32(0))

    comptime for h in range(NUM_Q):
        acc_ptrs[h] = partials + h * HEAD_DIM
        q_ptrs[h] = q + h * HEAD_DIM
        for j in range(0, HEAD_DIM, W):
            (acc_ptrs[h] + j).store(SIMD[DType.float32, W](0))

    var pos = start
    while pos < end:
        var tile_len = min(TILE, end - pos)

        comptime for q_idx in range(NUM_Q):
            comptime kv_h = q_idx // GQA_RATIO

            var scores = SIMD[DType.float32, TILE](-1e30)
            for t in range(tile_len):
                var cache_slot = (start_pos + pos + t) & (WINDOW - 1)
                var k_head = k_base + cache_slot * KV_STRIDE + kv_h * HEAD_DIM
                scores[t] = score_position[HEAD_DIM](q_ptrs[q_idx], k_head)

            var tile_max = scores.reduce_max()
            var m_new = tile_max if tile_max > m[q_idx] else m[q_idx]

            var corr = fast_exp_softmax_biased[1](
                SIMD[DType.float32, 1](m[q_idx] - m_new))[0]
            var weights = fast_exp_softmax_biased[TILE](
                scores - SIMD[DType.float32, TILE](m_new))

            scale_acc[HEAD_DIM](acc_ptrs[q_idx], corr)
            l[q_idx] = l[q_idx] * corr + weights.reduce_add()
            m[q_idx] = m_new

            for t in range(tile_len):
                var cache_slot = (start_pos + pos + t) & (WINDOW - 1)
                var v_head = v_base + cache_slot * KV_STRIDE + kv_h * HEAD_DIM
                weighted_add[HEAD_DIM](acc_ptrs[q_idx], v_head, weights[t])

        pos += TILE

    comptime for h in range(NUM_Q):
        (partials + m_off + h)[] = m[h]
        (partials + l_off + h)[] = l[h]


@no_inline
def run_case() -> Int:
    var q = alloc[BFloat16](NUM_Q * HEAD_DIM)
    var k = alloc[BFloat16](WINDOW * KV_STRIDE)
    var v = alloc[BFloat16](WINDOW * KV_STRIDE)
    comptime partial_elems = NUM_Q * HEAD_DIM + NUM_Q + NUM_Q
    var partials_a = alloc[Float32](partial_elems)
    var partials_b = alloc[Float32](partial_elems)

    for i in range(NUM_Q * HEAD_DIM):
        q[i] = BFloat16(Float32(Float64(i % 127 - 63) * 0.01))
    for i in range(WINDOW * KV_STRIDE):
        k[i] = BFloat16(Float32(Float64(i % 97 - 48) * 0.01))
        v[i] = BFloat16(Float32(Float64(i % 83 - 41) * 0.01))

    flash_decode_original(q, k, v, partials_a, 0, 0, NUM_POSITIONS)
    flash_decode_tiled(q, k, v, partials_b, 0, 0, NUM_POSITIONS)

    var checksum = Int(0)
    for i in range(partial_elems):
        checksum += Int(partials_a[i].cast[DType.int32]())
        checksum += Int(partials_b[i].cast[DType.int32]())

    q.free()
    k.free()
    v.free()
    partials_a.free()
    partials_b.free()
    return checksum


def main():
    keep(run_case())
