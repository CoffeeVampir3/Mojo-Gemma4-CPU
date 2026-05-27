from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from simd_math.matrixops import pick_port_unroll

from .fwht import fwht_block
from .i8_quantize import absmax_quantize_i8
from .types import F32Ptr, I8Ptr
from .vnni_dot import act_broadcast_vnni, dot_broadcasted
from .vnni_layout import (
    VNNI_BLK, VNNI_K_STEP, VNNI_N_STEP, VNNI_TILE_N, compute_n_block,
)


def fused_w1w3_gemv_row[N: Int, K: Int](
    act_row: I8Ptr,
    w1_packed: I8Ptr,
    w3_packed: I8Ptr,
    act_sc: Float32,
    w1_sc: F32Ptr,
    w1_cs: F32Ptr,
    w3_sc: F32Ptr,
    w3_cs: F32Ptr,
    gate: F32Ptr,
    up: F32Ptr,
):
    """Compute W1 and W3 GEMVs from one activation row in one VNNI walk."""
    debug_assert(K % VNNI_K_STEP == 0,
        "fused_w1w3_gemv_row: K must be a multiple of VNNI_K_STEP")
    debug_assert(N % VNNI_N_STEP == 0,
        "fused_w1w3_gemv_row: N must be a multiple of VNNI_N_STEP")
    comptime width = simd_width_of[DType.int32]()
    comptime passes = VNNI_TILE_N // width
    comptime bytes_per_pass = width * VNNI_BLK
    comptime acc_count = VNNI_N_STEP // width
    comptime dc_count = VNNI_K_STEP // VNNI_BLK
    comptime tile_dc_bytes = VNNI_TILE_N * VNNI_BLK
    comptime tile_ks_bytes = dc_count * tile_dc_bytes

    var n_block = compute_n_block(N, K)
    var packed_off = 0

    for nb in range(0, N, n_block):
        var nb_size = min(n_block, N - nb)
        for ns in range(0, nb_size, VNNI_N_STEP):
            var w1_acc = InlineArray[
                SIMD[DType.int32, width], acc_count](
                fill=SIMD[DType.int32, width](0))
            var w3_acc = InlineArray[
                SIMD[DType.int32, width], acc_count](
                fill=SIMD[DType.int32, width](0))

            for ks in range(0, K, VNNI_K_STEP):
                for dc in range(dc_count):
                    var k_pos = ks + dc * VNNI_BLK
                    var act_bytes = act_broadcast_vnni[width](act_row, k_pos)
                    var t0 = packed_off + dc * tile_dc_bytes
                    var t1 = t0 + tile_ks_bytes

                    comptime for p in range(passes):
                        var off = t0 + p * bytes_per_pass
                        w1_acc[p] = dot_broadcasted[width](
                            w1_acc[p], act_bytes, w1_packed + off)
                        w3_acc[p] = dot_broadcasted[width](
                            w3_acc[p], act_bytes, w3_packed + off)

                    comptime for p in range(passes):
                        var off = t1 + p * bytes_per_pass
                        w1_acc[passes + p] = dot_broadcasted[width](
                            w1_acc[passes + p], act_bytes, w1_packed + off)
                        w3_acc[passes + p] = dot_broadcasted[width](
                            w3_acc[passes + p], act_bytes, w3_packed + off)

                packed_off += 2 * tile_ks_bytes

            comptime for a in range(acc_count):
                var n_base = nb + ns + a * width
                var w1_corr = (
                    w1_acc[a].cast[DType.float32]()
                    - Float32(128) * (w1_cs + n_base).load[width=width]()
                )
                (gate + n_base).store(
                    w1_corr * act_sc * (w1_sc + n_base).load[width=width]())
                var w3_corr = (
                    w3_acc[a].cast[DType.float32]()
                    - Float32(128) * (w3_cs + n_base).load[width=width]()
                )
                (up + n_base).store(
                    w3_corr * act_sc * (w3_sc + n_base).load[width=width]())


@always_inline
def fused_gateup_quant_row[
    fwht_blk: Int, K: Int, fwht: Bool,
    act_fn: def[w: Int](SIMD[DType.float32, w]) thin
        -> SIMD[DType.float32, w],
](
    act_i8: I8Ptr,
    gate_packed: I8Ptr,
    gate_scale: F32Ptr,
    gate_colsum: F32Ptr,
    up_packed: I8Ptr,
    up_scale: F32Ptr,
    up_colsum: F32Ptr,
    act_dequant: Float32,
    qi_row: I8Ptr,
    blk_row: F32Ptr,
    n_off: Int,
    n_count: Int,
):
    """SwiGLU-style gate/up chunk: GEMV, act_fn(gate)*up, optional FWHT, i8."""
    comptime assert fwht_blk >= VNNI_N_STEP,
        "fused_gateup_quant_row: fwht_blk must cover at least one VNNI N-step"
    comptime width = simd_width_of[DType.float32]()
    comptime PU = pick_port_unroll[width, fwht_blk]()
    comptime STRIDE = PU * width

    var local_n = 0
    while local_n < n_count:
        var n_pos = n_off + local_n

        var gate_buf = InlineArray[Float32, fwht_blk](fill=Float32(0))
        var gate = UnsafePointer(to=gate_buf).bitcast[Float32]()
        var up_buf = InlineArray[Float32, fwht_blk](fill=Float32(0))
        var up = UnsafePointer(to=up_buf).bitcast[Float32]()

        fused_w1w3_gemv_row[fwht_blk, K](
            act_i8,
            gate_packed + n_pos * K,
            up_packed + n_pos * K,
            act_dequant,
            gate_scale + n_pos,
            gate_colsum + n_pos,
            up_scale + n_pos,
            up_colsum + n_pos,
            gate,
            up,
        )

        for i in range(fwht_blk // STRIDE):
            comptime for p in range(PU):
                comptime off = p * width
                var idx = i * STRIDE + off
                var g = (gate + idx).load[width=width]()
                var u = (up + idx).load[width=width]()
                (gate + idx).store(act_fn[width](g) * u)

        comptime if fwht:
            fwht_block[fwht_blk](gate)
        blk_row[local_n // fwht_blk] = absmax_quantize_i8[fwht_blk](
            gate, qi_row + local_n)

        local_n += fwht_blk
