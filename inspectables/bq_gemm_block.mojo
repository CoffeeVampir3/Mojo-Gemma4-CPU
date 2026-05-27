from std.sys import argv, llvm_intrinsic
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
    acc: SIMD[DType.int32, width], a: SIMD[DType.uint8, width * 4],
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
    acc: SIMD[DType.int32, width], act_bytes: SIMD[DType.uint8, width * 4],
    weights: SIMD[DType.int8, width * 4],
) -> SIMD[DType.int32, width]:
    return vpdpbusd[width](acc, act_bytes, weights)


@always_inline
def accumulate_tiles[
    width: Int, PR: Int, row_ptr: def(Int) capturing [_] -> I8Ptr,
](
    wpacked: I8Ptr, packed_base: Int, k_base: Int, k_len: Int,
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
            var ab = InlineArray[SIMD[DType.uint8, width * 4], PR](uninitialized=True)
            comptime for r in range(PR):
                ab[r] = act_broadcast_vnni[width](row_ptr(r), k_pos)
            var t0 = packed_off + dc * tile_dc_bytes
            var t1 = t0 + tile_ks_bytes
            comptime for p in range(passes):
                var w0 = (wpacked + t0 + p * bytes_per_pass).load[width = width * 4]()
                comptime for r in range(PR):
                    acc[r * acc_count + p] = dot_loaded[width](acc[r * acc_count + p], ab[r], w0)
            comptime for p in range(passes):
                var w1 = (wpacked + t1 + p * bytes_per_pass).load[width = width * 4]()
                comptime for r in range(PR):
                    acc[r * acc_count + passes + p] = dot_loaded[width](
                        acc[r * acc_count + passes + p], ab[r], w1)
        packed_off += 2 * tile_ks_bytes


@always_inline
def accumulate_n_step[width: Int, PR: Int, K: Int](
    act: I8Ptr, m_panel: Int, wpacked: I8Ptr, packed_base: Int, k_base: Int, k_len: Int,
    mut acc: InlineArray[SIMD[DType.int32, width], PR * (VNNI_N_STEP // width)],
):
    @parameter
    def row_ptr(r: Int) -> I8Ptr:
        return act + (m_panel + r) * K
    accumulate_tiles[width, PR, row_ptr](wpacked, packed_base, k_base, k_len, acc)


# per-block panel (source): facc live across nb, iacc live inside each block
@always_inline
def gemm_i8_per_block_panel[N: Int, K: Int, block: Int, PR: Int, Out: DType](
    act: I8Ptr, m_panel: Int, act_scale: F32Ptr, wpacked: I8Ptr,
    wsc: F32Ptr, colsum: F32Ptr,
    dst: UnsafePointer[Scalar[Out], MutAnyOrigin], ns: Int,
):
    comptime width = simd_width_of[DType.int32]()
    comptime acc_count = VNNI_N_STEP // width
    comptime nb = K // block
    comptime inv127 = Float32(1.0) / Float32(127.0)
    comptime blk_bytes = block * VNNI_N_STEP
    var facc = InlineArray[SIMD[DType.float32, width], PR * acc_count](
        fill=SIMD[DType.float32, width](0))
    for b in range(nb):
        var iacc = InlineArray[SIMD[DType.int32, width], PR * acc_count](
            fill=SIMD[DType.int32, width](0))
        accumulate_n_step[width, PR, K](
            act, m_panel, wpacked, ns * K + b * blk_bytes, b * block, block, iacc)
        comptime for r in range(PR):
            var adv = SIMD[DType.float32, width](act_scale[(m_panel + r) * nb + b] * inv127)
            comptime for a in range(acc_count):
                var n_base = ns + a * width
                var cs = (colsum + b * N + n_base).load[width=width]()
                var corrected = (iacc[r * acc_count + a].cast[DType.float32]()
                    - Float32(128) * cs)
                facc[r * acc_count + a] = corrected.fma(adv, facc[r * acc_count + a])
    comptime for r in range(PR):
        comptime for a in range(acc_count):
            var n_base = ns + a * width
            var res = facc[r * acc_count + a] * (wsc + n_base).load[width=width]()
            (dst + (m_panel + r) * N + n_base).store(res.cast[Out]())


comptime N = 4096
comptime K = 512
comptime BLK = 128


@no_inline
@export
def probe_block_pr1(act: I8Ptr, asc: F32Ptr, w: I8Ptr, wsc: F32Ptr,
                    cs: F32Ptr, dst: UnsafePointer[BFloat16, MutAnyOrigin]):
    gemm_i8_per_block_panel[N, K, BLK, 1, DType.bfloat16](act, 0, asc, w, wsc, cs, dst, 0)


@no_inline
@export
def probe_block_pr2(act: I8Ptr, asc: F32Ptr, w: I8Ptr, wsc: F32Ptr,
                    cs: F32Ptr, dst: UnsafePointer[BFloat16, MutAnyOrigin]):
    gemm_i8_per_block_panel[N, K, BLK, 2, DType.bfloat16](act, 0, asc, w, wsc, cs, dst, 0)


@no_inline
@export
def probe_block_pr4(act: I8Ptr, asc: F32Ptr, w: I8Ptr, wsc: F32Ptr,
                    cs: F32Ptr, dst: UnsafePointer[BFloat16, MutAnyOrigin]):
    gemm_i8_per_block_panel[N, K, BLK, 4, DType.bfloat16](act, 0, asc, w, wsc, cs, dst, 0)


def main():
    var seed = len(argv())
    var act = alloc[Int8](N * K).as_any_origin()
    var w = alloc[Int8](N * K).as_any_origin()
    var asc = alloc[Float32](N * (K // BLK)).as_any_origin()
    var wsc = alloc[Float32](N).as_any_origin()
    var cs = alloc[Float32]((K // BLK) * N).as_any_origin()
    var dst = alloc[BFloat16](N * N).as_any_origin()
    for i in range(N * K):
        act[i] = Int8((i + seed) % 127)
        w[i] = Int8((i * 7 + seed) % 127)
    for i in range(N * (K // BLK)):
        asc[i] = Float32(i % 13 + seed)
    for i in range(N):
        wsc[i] = Float32(i % 7 + 1)
    for i in range((K // BLK) * N):
        cs[i] = Float32(i % 5)
    probe_block_pr1(act, asc, w, wsc, cs, dst)
    probe_block_pr2(act, asc, w, wsc, cs, dst)
    probe_block_pr4(act, asc, w, wsc, cs, dst)
    keep(dst[0])
    keep(dst[seed])
