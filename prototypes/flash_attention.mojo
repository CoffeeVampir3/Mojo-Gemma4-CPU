from std.collections import InlineArray
from std.math import exp
from std.memory import UnsafePointer
from std.sys.info import simd_width_of


comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime F32Ptr  = UnsafePointer[Float32,  MutAnyOrigin]
comptime W = simd_width_of[DType.float32]()


trait KVSlot:
    @staticmethod
    @always_inline
    def slot(start_pos: Int, pos_t: Int) -> Int: ...


struct LinearKV(KVSlot):
    @staticmethod
    @always_inline
    def slot(start_pos: Int, pos_t: Int) -> Int:
        return pos_t


struct RingKV[window: Int](KVSlot):
    @staticmethod
    @always_inline
    def slot(start_pos: Int, pos_t: Int) -> Int:
        return (start_pos + pos_t) & (Self.window - 1)


@always_inline
def score_position[head_dim: Int](q: BF16Ptr, k_row: BF16Ptr) -> Float32:
    var acc = SIMD[DType.float32, W](0)
    for i in range(0, head_dim, W):
        var qv = (q + i).load[width=W]().cast[DType.float32]()
        var kv = (k_row + i).load[width=W]().cast[DType.float32]()
        acc = qv.fma(kv, acc)
    return acc.reduce_add()


@always_inline
def accumulate_scaled[head_dim: Int](
    src: BF16Ptr, weight: Float32, acc: F32Ptr,
):
    var w_vec = SIMD[DType.float32, W](weight)
    for i in range(0, head_dim, W):
        var v = (src + i).load[width=W]().cast[DType.float32]()
        var a = (acc + i).load[width=W]()
        (acc + i).store(v.fma(w_vec, a))


@always_inline
def scale_unrolled[head_dim: Int](acc: F32Ptr, factor: Float32):
    var f = SIMD[DType.float32, W](factor)
    for i in range(0, head_dim, W):
        (acc + i).store((acc + i).load[width=W]() * f)


@always_inline
def flash_partial_stride[num_q: Int, head_dim: Int]() -> Int:
    return ((num_q * head_dim + 2 * num_q) * 4 + 63) // 64 * 16


@fieldwise_init
struct FlashAttentionKernel[
    KV: KVSlot,
    head_dim: Int, num_q: Int, gqa_ratio: Int, kv_stride: Int,
]:
    comptime PARTIAL_STRIDE = flash_partial_stride[Self.num_q, Self.head_dim]()
    comptime TILE = W

    var q: BF16Ptr
    var k_base: BF16Ptr
    var v_base: BF16Ptr
    var partials: F32Ptr
    var worker_id: Int
    var start_pos: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var my_partial = self.partials + self.worker_id * Self.PARTIAL_STRIDE
        comptime m_off = Self.num_q * Self.head_dim
        comptime l_off = m_off + Self.num_q

        var acc_ptrs = InlineArray[F32Ptr, Self.num_q](uninitialized=True)
        var q_ptrs = InlineArray[BF16Ptr, Self.num_q](uninitialized=True)
        var m = InlineArray[Float32, Self.num_q](fill=Float32(-1e30))
        var l = InlineArray[Float32, Self.num_q](fill=Float32(0))

        comptime for h in range(Self.num_q):
            acc_ptrs[h] = my_partial + h * Self.head_dim
            q_ptrs[h] = self.q + h * Self.head_dim
            for j in range(0, Self.head_dim, W):
                (acc_ptrs[h] + j).store(SIMD[DType.float32, W](0))

        var pos = self.start
        while pos < self.end:
            var tile_len = min(Self.TILE, self.end - pos)

            comptime for q_idx in range(Self.num_q):
                comptime kv_h = q_idx // Self.gqa_ratio

                var scores = SIMD[DType.float32, Self.TILE](-1e30)
                for t in range(tile_len):
                    var s_idx = Self.KV.slot(self.start_pos, pos + t)
                    var k_head = self.k_base + s_idx * Self.kv_stride \
                                 + kv_h * Self.head_dim
                    scores[t] = score_position[Self.head_dim](
                        q_ptrs[q_idx], k_head)

                var tile_max = scores.reduce_max()
                var m_new = tile_max if tile_max > m[q_idx] else m[q_idx]

                var corr = exp(m[q_idx] - m_new)
                var weights = SIMD[DType.float32, Self.TILE](0)
                for t in range(tile_len):
                    weights[t] = exp(scores[t] - m_new)

                scale_unrolled[Self.head_dim](acc_ptrs[q_idx], corr)
                l[q_idx] = l[q_idx] * corr + weights.reduce_add()
                m[q_idx] = m_new

                for t in range(tile_len):
                    var s_idx = Self.KV.slot(self.start_pos, pos + t)
                    var v_head = self.v_base + s_idx * Self.kv_stride \
                                 + kv_h * Self.head_dim
                    accumulate_scaled[Self.head_dim](
                        v_head, weights[t], acc_ptrs[q_idx])

            pos += Self.TILE

        comptime for h in range(Self.num_q):
            (my_partial + m_off + h)[] = m[h]
            (my_partial + l_off + h)[] = l[h]

    @always_inline
    def install_worker_range(mut self, worker_id: Int, start: Int, end: Int):
        self.worker_id = worker_id
        self.start = start
        self.end = end


def dispatch_full_attention[
    head_dim: Int, num_q: Int, gqa_ratio: Int, kv_stride: Int,
](
    q: BF16Ptr, k_base: BF16Ptr, v_base: BF16Ptr,
    partials: F32Ptr, valid_len: Int,
):
    var k = FlashAttentionKernel[
        LinearKV, head_dim, num_q, gqa_ratio, kv_stride,
    ](q, k_base, v_base, partials, 0, 0, 0, valid_len)
    k.execute()


def dispatch_sliding_attention[
    head_dim: Int, num_q: Int, gqa_ratio: Int, kv_stride: Int, window: Int,
](
    q: BF16Ptr, k_base: BF16Ptr, v_base: BF16Ptr,
    partials: F32Ptr, pos: Int, valid_len: Int,
):
    var start_pos = pos - valid_len + 1
    var k = FlashAttentionKernel[
        RingKV[window], head_dim, num_q, gqa_ratio, kv_stride,
    ](q, k_base, v_base, partials, 0, start_pos, 0, valid_len)
    k.execute()


def reference_attention[
    head_dim: Int, num_q: Int, gqa_ratio: Int, kv_stride: Int, window: Int,
](
    q: BF16Ptr, k_base: BF16Ptr, v_base: BF16Ptr,
    ref_acc: F32Ptr, ref_m: F32Ptr, ref_l: F32Ptr,
    start_pos: Int, valid_len: Int, use_ring: Bool,
):
    var scores = InlineArray[Float32, 1024](uninitialized=True)
    for q_idx in range(num_q):
        var kv_h = q_idx // gqa_ratio
        var m_max = Float32(-1e30)
        for t in range(valid_len):
            var s_idx = ((start_pos + t) & (window - 1)) if use_ring else t
            var s = Float32(0)
            for i in range(head_dim):
                s += q[q_idx * head_dim + i].cast[DType.float32]() \
                   * k_base[s_idx * kv_stride + kv_h * head_dim + i] \
                       .cast[DType.float32]()
            scores[t] = s
            if s > m_max:
                m_max = s
        var l_sum = Float32(0)
        for j in range(head_dim):
            (ref_acc + q_idx * head_dim + j)[] = Float32(0)
        for t in range(valid_len):
            var w = exp(scores[t] - m_max)
            l_sum += w
            var s_idx = ((start_pos + t) & (window - 1)) if use_ring else t
            for j in range(head_dim):
                var cur = (ref_acc + q_idx * head_dim + j)[]
                (ref_acc + q_idx * head_dim + j)[] = cur \
                    + w * v_base[s_idx * kv_stride + kv_h * head_dim + j] \
                        .cast[DType.float32]()
        ref_m[q_idx] = m_max
        ref_l[q_idx] = l_sum


def fill_buffers(
    q_buf: BF16Ptr, k_buf: BF16Ptr, v_buf: BF16Ptr,
    num_q: Int, head_dim: Int, kv_total_rows: Int, kv_stride: Int,
):
    for i in range(num_q * head_dim):
        q_buf[i] = BFloat16(Float32(i % 7) * Float32(0.1))
    for i in range(kv_total_rows * kv_stride):
        k_buf[i] = BFloat16(Float32(i % 11) * Float32(0.05))
        v_buf[i] = BFloat16(Float32(i % 13) * Float32(0.05))


def compare_outputs(
    unified: F32Ptr, ref_acc: F32Ptr, ref_m: F32Ptr, ref_l: F32Ptr,
    num_q: Int, head_dim: Int,
) -> Tuple[Float32, Float32, Float32]:
    var acc_diff = Float32(0)
    var m_diff = Float32(0)
    var l_diff = Float32(0)
    var m_off = num_q * head_dim
    var l_off = m_off + num_q
    for q_idx in range(num_q):
        var l_unified = unified[l_off + q_idx]
        var l_ref = ref_l[q_idx]
        var dl = abs(l_unified - l_ref) / max(abs(l_ref), Float32(1e-9))
        if dl > l_diff: l_diff = dl
        var dm = abs(unified[m_off + q_idx] - ref_m[q_idx])
        if dm > m_diff: m_diff = dm
        for j in range(head_dim):
            var u = unified[q_idx * head_dim + j] / l_unified
            var r = ref_acc[q_idx * head_dim + j] / l_ref
            var d = abs(u - r)
            if d > acc_diff: acc_diff = d
    return (acc_diff, m_diff, l_diff)


def main():
    comptime head_dim = 32
    comptime num_q = 4
    comptime gqa_ratio = 4
    comptime kv_stride = head_dim
    comptime window = 32
    comptime kv_rows = 64

    var q_buf = InlineArray[BFloat16, num_q * head_dim](uninitialized=True)
    var k_buf = InlineArray[BFloat16, kv_rows * kv_stride](uninitialized=True)
    var v_buf = InlineArray[BFloat16, kv_rows * kv_stride](uninitialized=True)
    var q_ptr = UnsafePointer(to=q_buf[0]).as_any_origin()
    var k_ptr = UnsafePointer(to=k_buf[0]).as_any_origin()
    var v_ptr = UnsafePointer(to=v_buf[0]).as_any_origin()
    fill_buffers(q_ptr, k_ptr, v_ptr, num_q, head_dim, kv_rows, kv_stride)

    comptime stride = flash_partial_stride[num_q, head_dim]()
    var partials = InlineArray[Float32, stride](fill=Float32(0))
    var ref_acc = InlineArray[Float32, num_q * head_dim](fill=Float32(0))
    var ref_m = InlineArray[Float32, num_q](fill=Float32(0))
    var ref_l = InlineArray[Float32, num_q](fill=Float32(0))
    var p_ptr = UnsafePointer(to=partials[0]).as_any_origin()
    var ra_ptr = UnsafePointer(to=ref_acc[0]).as_any_origin()
    var rm_ptr = UnsafePointer(to=ref_m[0]).as_any_origin()
    var rl_ptr = UnsafePointer(to=ref_l[0]).as_any_origin()

    print("=== KVSlot strategy resolves at comptime ===")
    print("  LinearKV.slot(100, 5)       =", LinearKV.slot(100, 5),
          " expected 5")
    print("  RingKV[32].slot(100, 5)     =", RingKV[32].slot(100, 5),
          " expected", (100 + 5) & 31)
    print("  RingKV[1024].slot(2000, 50) =", RingKV[1024].slot(2000, 50),
          " expected", (2000 + 50) & 1023)
    print()

    print("=== unified FlashAttentionKernel ===")
    print("  head_dim=", head_dim, " num_q=", num_q,
          " gqa_ratio=", gqa_ratio, " kv_stride=", kv_stride)
    print()

    var seq_len = 16
    dispatch_full_attention[head_dim, num_q, gqa_ratio, kv_stride](
        q_ptr, k_ptr, v_ptr, p_ptr, seq_len)
    reference_attention[head_dim, num_q, gqa_ratio, kv_stride, 1](
        q_ptr, k_ptr, v_ptr, ra_ptr, rm_ptr, rl_ptr, 0, seq_len, False)
    var d_full = compare_outputs(
        p_ptr, ra_ptr, rm_ptr, rl_ptr, num_q, head_dim)
    print("--- LinearKV (full attention), seq_len=", seq_len, " ---")
    print("  max acc diff (normalized):", d_full[0])
    print("  max m   diff             :", d_full[1])
    print("  max l   diff (rel)       :", d_full[2])

    for i in range(stride): partials[i] = Float32(0)
    for i in range(num_q * head_dim): ref_acc[i] = Float32(0)
    for i in range(num_q):
        ref_m[i] = Float32(0)
        ref_l[i] = Float32(0)

    var pos = 47
    var valid_len = window
    dispatch_sliding_attention[
        head_dim, num_q, gqa_ratio, kv_stride, window,
    ](q_ptr, k_ptr, v_ptr, p_ptr, pos, valid_len)
    reference_attention[head_dim, num_q, gqa_ratio, kv_stride, window](
        q_ptr, k_ptr, v_ptr, ra_ptr, rm_ptr, rl_ptr,
        pos - valid_len + 1, valid_len, True)
    var d_ring = compare_outputs(
        p_ptr, ra_ptr, rm_ptr, rl_ptr, num_q, head_dim)
    print("--- RingKV[", window, "] (sliding), pos=", pos,
          " valid_len=", valid_len, " ---")
    print("  max acc diff (normalized):", d_ring[0])
    print("  max m   diff             :", d_ring[1])
    print("  max l   diff (rel)       :", d_ring[2])

    print()
    var tol = Float32(1e-3)
    var ok = d_full[0] < tol and d_full[1] < tol \
        and d_ring[0] < tol and d_ring[1] < tol
    if ok:
        print("OK: unified kernel matches reference on both paths.")
    else:
        print("FAIL: diffs exceed tolerance.")
