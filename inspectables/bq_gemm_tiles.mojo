from std.sys import argv, llvm_intrinsic, CompilationTarget
from std.sys.info import simd_width_of
from std.benchmark import keep
from std.collections import InlineArray
from std.memory import UnsafePointer, alloc

comptime I8Ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]

comptime VNNI_N_STEP = 32
comptime VNNI_K_STEP = 64
comptime VNNI_TILE_N = 16
comptime VNNI_BLK = 4


@always_inline
def vpdpbusd[width: Int](
    acc: SIMD[DType.int32, width],
    a: SIMD[DType.uint8, width * 4],
    b: SIMD[DType.int8, width * 4],
) -> SIMD[DType.int32, width]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd." + String(width * 32),
        SIMD[DType.int32, width],
    ](acc, a, b)


@always_inline
def act_broadcast_vnni[width: Int](act_row: I8Ptr, k_pos: Int) -> SIMD[
    DType.uint8, width * 4,
]:
    var b4 = (act_row + k_pos).bitcast[UInt8]().load[width=4]() ^ SIMD[
        DType.uint8, 4](0x80)
    var out = SIMD[DType.uint8, width * 4]()
    comptime for lane in range(width):
        out = out.insert[offset = lane * 4](b4)
    return out


@always_inline
def dot_loaded[width: Int](
    acc: SIMD[DType.int32, width],
    act_bytes: SIMD[DType.uint8, width * 4],
    weights: SIMD[DType.int8, width * 4],
) -> SIMD[DType.int32, width]:
    return vpdpbusd[width](acc, act_bytes, weights)


# accumulate_tiles with comptime `nt` to A/B the non_temporal weight-load hint
@always_inline
def accumulate_tiles[
    width: Int, PR: Int, nt: Bool, row_ptr: def(Int) capturing [_] -> I8Ptr,
](
    wpacked: I8Ptr,
    packed_base: Int,
    k_base: Int,
    k_len: Int,
    mut acc: InlineArray[SIMD[DType.int32, width], PR * (VNNI_N_STEP // width)],
):
    comptime passes = VNNI_TILE_N // width
    comptime bytes_per_pass = width * VNNI_BLK
    comptime acc_count = VNNI_N_STEP // width
    comptime dc_count = VNNI_K_STEP // VNNI_BLK
    comptime tile_dc_bytes = VNNI_TILE_N * VNNI_BLK
    comptime tile_ks_bytes = dc_count * tile_dc_bytes

    var packed_off = packed_base
    for ks in range(0, k_len, VNNI_K_STEP):
        for dc in range(dc_count):
            var k_pos = k_base + ks + dc * VNNI_BLK
            var ab = InlineArray[SIMD[DType.uint8, width * 4], PR](
                uninitialized=True)
            comptime for r in range(PR):
                ab[r] = act_broadcast_vnni[width](row_ptr(r), k_pos)
            var t0 = packed_off + dc * tile_dc_bytes
            var t1 = t0 + tile_ks_bytes
            comptime for p in range(passes):
                var w0 = (wpacked + t0 + p * bytes_per_pass).load[
                    width = width * 4, non_temporal=nt]()
                comptime for r in range(PR):
                    acc[r * acc_count + p] = dot_loaded[width](
                        acc[r * acc_count + p], ab[r], w0)
            comptime for p in range(passes):
                var w1 = (wpacked + t1 + p * bytes_per_pass).load[
                    width = width * 4, non_temporal=nt]()
                comptime for r in range(PR):
                    acc[r * acc_count + passes + p] = dot_loaded[width](
                        acc[r * acc_count + passes + p], ab[r], w1)
        packed_off += 2 * tile_ks_bytes


@always_inline
def accumulate_n_step[width: Int, PR: Int, K: Int, nt: Bool](
    act: I8Ptr,
    m_panel: Int,
    wpacked: I8Ptr,
    packed_base: Int,
    k_base: Int,
    k_len: Int,
    mut acc: InlineArray[SIMD[DType.int32, width], PR * (VNNI_N_STEP // width)],
):
    @parameter
    def row_ptr(r: Int) -> I8Ptr:
        return act + (m_panel + r) * K

    accumulate_tiles[width, PR, nt, row_ptr](
        wpacked, packed_base, k_base, k_len, acc)


@always_inline
def gemm_i8_per_row_panel[N: Int, K: Int, PR: Int, nt: Bool, Out: DType](
    act: I8Ptr,
    m_panel: Int,
    act_scale: F32Ptr,
    wpacked: I8Ptr,
    wsc: F32Ptr,
    colsum: F32Ptr,
    dst: UnsafePointer[Scalar[Out], MutAnyOrigin],
    ns: Int,
):
    comptime width = simd_width_of[DType.int32]()
    comptime acc_count = VNNI_N_STEP // width
    var iacc = InlineArray[SIMD[DType.int32, width], PR * acc_count](
        fill=SIMD[DType.int32, width](0))
    accumulate_n_step[width, PR, K, nt](
        act, m_panel, wpacked, ns * K, 0, K, iacc)

    var inv127 = Float32(1.0) / Float32(127.0)
    comptime for r in range(PR):
        var ad = act_scale[m_panel + r] * inv127
        comptime for a in range(acc_count):
            var n_base = ns + a * width
            var cs = (colsum + n_base).load[width=width]()
            var corrected = (
                iacc[r * acc_count + a].cast[DType.float32]()
                - Float32(128) * cs)
            var res = corrected * ad * (wsc + n_base).load[width=width]()
            (dst + (m_panel + r) * N + n_base).store(res.cast[Out]())


# vectorized RNE f32->bf16 store (no per-element libcall)
@always_inline
def store_bf16_rne[width: Int](
    v: SIMD[DType.float32, width], dst: UnsafePointer[BFloat16, MutAnyOrigin],
):
    var bits = v.to_bits().cast[DType.uint32]()
    var one = SIMD[DType.uint32, width](1)
    var bias = SIMD[DType.uint32, width](0x7FFF)
    var rounded = bits + ((bits >> 16) & one) + bias
    dst.bitcast[UInt16]().store((rounded >> 16).cast[DType.uint16]())


# project-idiomatic arch gate (mirrors kernels/dot_products.has_avx512_bf16)
@always_inline
def has_avx512_bf16() -> Bool:
    return CompilationTarget._has_feature["avx512bf16"]()


# single code path: hardware convert where available, vectorized RNE otherwise
@always_inline
def store_bf16_gated[width: Int](
    v: SIMD[DType.float32, width], dst: UnsafePointer[BFloat16, MutAnyOrigin],
):
    comptime if has_avx512_bf16():
        dst.store(v.cast[DType.bfloat16]())
    else:
        store_bf16_rne[width](v, dst)


@always_inline
def gemm_i8_per_row_panel_gated[N: Int, K: Int, PR: Int](
    act: I8Ptr, m_panel: Int, act_scale: F32Ptr, wpacked: I8Ptr,
    wsc: F32Ptr, colsum: F32Ptr,
    dst: UnsafePointer[BFloat16, MutAnyOrigin], ns: Int,
):
    comptime width = simd_width_of[DType.int32]()
    comptime acc_count = VNNI_N_STEP // width
    var iacc = InlineArray[SIMD[DType.int32, width], PR * acc_count](
        fill=SIMD[DType.int32, width](0))
    accumulate_n_step[width, PR, K, False](act, m_panel, wpacked, ns * K, 0, K, iacc)
    var inv127 = Float32(1.0) / Float32(127.0)
    comptime for r in range(PR):
        var ad = act_scale[m_panel + r] * inv127
        comptime for a in range(acc_count):
            var n_base = ns + a * width
            var cs = (colsum + n_base).load[width=width]()
            var corrected = (iacc[r * acc_count + a].cast[DType.float32]()
                - Float32(128) * cs)
            var res = corrected * ad * (wsc + n_base).load[width=width]()
            store_bf16_gated[width](res, dst + (m_panel + r) * N + n_base)


# B-epilogue panel: identical accumulate, vectorized bf16 store
@always_inline
def gemm_i8_per_row_panel_bf16[N: Int, K: Int, PR: Int](
    act: I8Ptr, m_panel: Int, act_scale: F32Ptr, wpacked: I8Ptr,
    wsc: F32Ptr, colsum: F32Ptr,
    dst: UnsafePointer[BFloat16, MutAnyOrigin], ns: Int,
):
    comptime width = simd_width_of[DType.int32]()
    comptime acc_count = VNNI_N_STEP // width
    var iacc = InlineArray[SIMD[DType.int32, width], PR * acc_count](
        fill=SIMD[DType.int32, width](0))
    accumulate_n_step[width, PR, K, False](act, m_panel, wpacked, ns * K, 0, K, iacc)
    var inv127 = Float32(1.0) / Float32(127.0)
    comptime for r in range(PR):
        var ad = act_scale[m_panel + r] * inv127
        comptime for a in range(acc_count):
            var n_base = ns + a * width
            var cs = (colsum + n_base).load[width=width]()
            var corrected = (iacc[r * acc_count + a].cast[DType.float32]()
                - Float32(128) * cs)
            var res = corrected * ad * (wsc + n_base).load[width=width]()
            store_bf16_rne[width](res, dst + (m_panel + r) * N + n_base)


comptime N = 4096
comptime K = 512


# A: PR sweep, non_temporal=True (as in source)
@no_inline
@export
def probe_panel_pr1_nt(act: I8Ptr, asc: F32Ptr, w: I8Ptr, wsc: F32Ptr,
                       cs: F32Ptr, dst: UnsafePointer[BFloat16, MutAnyOrigin]):
    gemm_i8_per_row_panel[N, K, 1, True, DType.bfloat16](act, 0, asc, w, wsc, cs, dst, 0)


@no_inline
@export
def probe_panel_pr2_nt(act: I8Ptr, asc: F32Ptr, w: I8Ptr, wsc: F32Ptr,
                       cs: F32Ptr, dst: UnsafePointer[BFloat16, MutAnyOrigin]):
    gemm_i8_per_row_panel[N, K, 2, True, DType.bfloat16](act, 0, asc, w, wsc, cs, dst, 0)


@no_inline
@export
def probe_panel_pr4_nt(act: I8Ptr, asc: F32Ptr, w: I8Ptr, wsc: F32Ptr,
                       cs: F32Ptr, dst: UnsafePointer[BFloat16, MutAnyOrigin]):
    gemm_i8_per_row_panel[N, K, 4, True, DType.bfloat16](act, 0, asc, w, wsc, cs, dst, 0)


# B(ii): same but non_temporal=False
@no_inline
@export
def probe_panel_pr2_reg(act: I8Ptr, asc: F32Ptr, w: I8Ptr, wsc: F32Ptr,
                        cs: F32Ptr, dst: UnsafePointer[BFloat16, MutAnyOrigin]):
    gemm_i8_per_row_panel[N, K, 2, False, DType.bfloat16](act, 0, asc, w, wsc, cs, dst, 0)


@no_inline
@export
def probe_panel_pr4_reg(act: I8Ptr, asc: F32Ptr, w: I8Ptr, wsc: F32Ptr,
                        cs: F32Ptr, dst: UnsafePointer[BFloat16, MutAnyOrigin]):
    gemm_i8_per_row_panel[N, K, 4, False, DType.bfloat16](act, 0, asc, w, wsc, cs, dst, 0)


@no_inline
@export
def probe_panel_pr1_bf16(act: I8Ptr, asc: F32Ptr, w: I8Ptr, wsc: F32Ptr,
                         cs: F32Ptr, dst: UnsafePointer[BFloat16, MutAnyOrigin]):
    gemm_i8_per_row_panel_bf16[N, K, 1](act, 0, asc, w, wsc, cs, dst, 0)


@no_inline
@export
def probe_panel_pr4_bf16(act: I8Ptr, asc: F32Ptr, w: I8Ptr, wsc: F32Ptr,
                         cs: F32Ptr, dst: UnsafePointer[BFloat16, MutAnyOrigin]):
    gemm_i8_per_row_panel_bf16[N, K, 4](act, 0, asc, w, wsc, cs, dst, 0)


@no_inline
@export
def probe_panel_pr4_gated(act: I8Ptr, asc: F32Ptr, w: I8Ptr, wsc: F32Ptr,
                          cs: F32Ptr, dst: UnsafePointer[BFloat16, MutAnyOrigin]):
    gemm_i8_per_row_panel_gated[N, K, 4](act, 0, asc, w, wsc, cs, dst, 0)


def main():
    var seed = len(argv())
    var act = alloc[Int8](N * K).as_any_origin()
    var w = alloc[Int8](N * K).as_any_origin()
    var asc = alloc[Float32](N).as_any_origin()
    var wsc = alloc[Float32](N).as_any_origin()
    var cs = alloc[Float32](N).as_any_origin()
    var dst = alloc[BFloat16](N * N).as_any_origin()
    for i in range(N * K):
        act[i] = Int8((i + seed) % 127)
        w[i] = Int8((i * 7 + seed) % 127)
    for i in range(N):
        asc[i] = Float32(i % 13 + seed)
        wsc[i] = Float32(i % 7 + 1)
        cs[i] = Float32(i % 5)
    probe_panel_pr1_nt(act, asc, w, wsc, cs, dst)
    probe_panel_pr2_nt(act, asc, w, wsc, cs, dst)
    probe_panel_pr4_nt(act, asc, w, wsc, cs, dst)
    probe_panel_pr2_reg(act, asc, w, wsc, cs, dst)
    probe_panel_pr4_reg(act, asc, w, wsc, cs, dst)
    probe_panel_pr1_bf16(act, asc, w, wsc, cs, dst)
    probe_panel_pr4_bf16(act, asc, w, wsc, cs, dst)
    probe_panel_pr4_gated(act, asc, w, wsc, cs, dst)
    keep(dst[0])
    keep(dst[seed])
