from std.collections import InlineArray
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

from threading.threading_traits import BurstThreadPool
from kernels.helpers import (
    RangePartitionedKernel, Binding, fanout_dispatch, matmul_workers,
)
from kernels.dispatch_heuristics import GEMV_INLINE_ROWS
from kernels.profiling import Profiler

from butterquant.convert import store_out
from butterquant.vnni import VNNI_N_STEP
from butterquant.types import F32Ptr, I8Ptr, BF16Ptr
from butterquant.weight import (
    ButterquantWeight, ButterquantActivation, ButterquantBlockActivation,
    quant_vnni_packed, quant_k_block,
)
from quant.recipe import QuantRecipe

from prototypes.amx_intrinsics import (
    AMX_TILE_M, AMX_TILE_N, AMX_K_STEP,
    make_224_i8_config, make_partial_config,
    ldtilecfg, tilerelease, tilezero, tileload, tilestore, tdpbssd,
)


comptime I32Ptr = UnsafePointer[Int32, MutAnyOrigin]
comptime WIDTH = simd_width_of[DType.int32]()
comptime INV127 = Float32(1.0) / Float32(127.0)
comptime CTILE = AMX_TILE_M * AMX_TILE_N
comptime HALF_BYTES = AMX_TILE_M * AMX_K_STEP
comptime C_STRIDE = AMX_TILE_N * 4


@always_inline
def amx_b_tile_base[K: Int](t: Int, k_off: Int) -> Int:
    return (t * VNNI_N_STEP) * K + k_off * VNNI_N_STEP


@always_inline
def dequant_fused[
    N: Int,
    write: def(Int, Int, SIMD[DType.float32, WIDTH]) capturing [_] -> None,
](
    c: I32Ptr, row_base: Int, n_base: Int, m_rows: Int,
    act_scale: F32Ptr, wsc: F32Ptr,
):
    for r in range(m_rows):
        var ad = act_scale[row_base + r] * INV127
        var nj = 0
        while nj < AMX_TILE_N:
            var cv = (c + r * AMX_TILE_N + nj).load[width=WIDTH]().cast[
                DType.float32]()
            var ws = (wsc + n_base + nj).load[width=WIDTH]()
            write(row_base + r, n_base + nj, cv * ad * ws)
            nj += WIDTH


@always_inline
def block_accumulate(
    c: I32Ptr, facc: F32Ptr, row_base: Int, b: Int, nb: Int, m_rows: Int,
    act_scale: F32Ptr,
):
    for r in range(m_rows):
        var adv = act_scale[(row_base + r) * nb + b] * INV127
        var nj = 0
        while nj < AMX_TILE_N:
            var slot = facc + r * AMX_TILE_N + nj
            var cv = (c + r * AMX_TILE_N + nj).load[width=WIDTH]().cast[
                DType.float32]()
            slot.store(slot.load[width=WIDTH]() + cv * adv)
            nj += WIDTH


@always_inline
def block_finalize[
    N: Int,
    write: def(Int, Int, SIMD[DType.float32, WIDTH]) capturing [_] -> None,
](
    facc: F32Ptr, row_base: Int, n_base: Int, m_rows: Int, wsc: F32Ptr,
):
    for r in range(m_rows):
        var nj = 0
        while nj < AMX_TILE_N:
            var fv = (facc + r * AMX_TILE_N + nj).load[width=WIDTH]()
            var ws = (wsc + n_base + nj).load[width=WIDTH]()
            write(row_base + r, n_base + nj, fv * ws)
            nj += WIDTH


@always_inline
def amx_panel_2x32[
    N: Int, K: Int, block: Int,
    write: def(Int, Int, SIMD[DType.float32, WIDTH]) capturing [_] -> None,
](
    act: I8Ptr, m_panel: Int, act_scale: F32Ptr, weight: I8Ptr, wsc: F32Ptr,
    t: Int,
):
    comptime nb = K // block
    var c = InlineArray[Int32, 4 * CTILE](uninitialized=True)
    var cp = UnsafePointer(to=c).bitcast[Int32]()
    var c00 = cp
    var c01 = cp + CTILE
    var c10 = cp + 2 * CTILE
    var c11 = cp + 3 * CTILE
    var rb0 = m_panel
    var rb1 = m_panel + AMX_TILE_M
    var n0 = t * VNNI_N_STEP
    var n1 = n0 + AMX_TILE_N

    comptime if nb == 1:
        tilezero[4]()
        tilezero[5]()
        tilezero[6]()
        tilezero[7]()
        for k_off in range(0, K, AMX_K_STEP):
            tileload[0, DType.int8](act + rb0 * K + k_off, K)
            tileload[1, DType.int8](act + rb1 * K + k_off, K)
            var base = amx_b_tile_base[K](t, k_off)
            tileload[2, DType.int8](weight + base, AMX_K_STEP)
            tileload[3, DType.int8](weight + base + HALF_BYTES, AMX_K_STEP)
            tdpbssd[4, 0, 2]()
            tdpbssd[5, 0, 3]()
            tdpbssd[6, 1, 2]()
            tdpbssd[7, 1, 3]()
        tilestore[4, DType.int32](c00, C_STRIDE)
        tilestore[5, DType.int32](c01, C_STRIDE)
        tilestore[6, DType.int32](c10, C_STRIDE)
        tilestore[7, DType.int32](c11, C_STRIDE)
        dequant_fused[N, write](c00, rb0, n0, AMX_TILE_M, act_scale, wsc)
        dequant_fused[N, write](c01, rb0, n1, AMX_TILE_M, act_scale, wsc)
        dequant_fused[N, write](c10, rb1, n0, AMX_TILE_M, act_scale, wsc)
        dequant_fused[N, write](c11, rb1, n1, AMX_TILE_M, act_scale, wsc)
    else:
        var facc = InlineArray[Float32, 4 * CTILE](fill=Float32(0))
        var fp = UnsafePointer(to=facc).bitcast[Float32]()
        var f00 = fp
        var f01 = fp + CTILE
        var f10 = fp + 2 * CTILE
        var f11 = fp + 3 * CTILE
        for b in range(nb):
            tilezero[4]()
            tilezero[5]()
            tilezero[6]()
            tilezero[7]()
            for k_off in range(b * block, (b + 1) * block, AMX_K_STEP):
                tileload[0, DType.int8](act + rb0 * K + k_off, K)
                tileload[1, DType.int8](act + rb1 * K + k_off, K)
                var base = amx_b_tile_base[K](t, k_off)
                tileload[2, DType.int8](weight + base, AMX_K_STEP)
                tileload[3, DType.int8](weight + base + HALF_BYTES, AMX_K_STEP)
                tdpbssd[4, 0, 2]()
                tdpbssd[5, 0, 3]()
                tdpbssd[6, 1, 2]()
                tdpbssd[7, 1, 3]()
            tilestore[4, DType.int32](c00, C_STRIDE)
            tilestore[5, DType.int32](c01, C_STRIDE)
            tilestore[6, DType.int32](c10, C_STRIDE)
            tilestore[7, DType.int32](c11, C_STRIDE)
            block_accumulate(c00, f00, rb0, b, nb, AMX_TILE_M, act_scale)
            block_accumulate(c01, f01, rb0, b, nb, AMX_TILE_M, act_scale)
            block_accumulate(c10, f10, rb1, b, nb, AMX_TILE_M, act_scale)
            block_accumulate(c11, f11, rb1, b, nb, AMX_TILE_M, act_scale)
        block_finalize[N, write](f00, rb0, n0, AMX_TILE_M, wsc)
        block_finalize[N, write](f01, rb0, n1, AMX_TILE_M, wsc)
        block_finalize[N, write](f10, rb1, n0, AMX_TILE_M, wsc)
        block_finalize[N, write](f11, rb1, n1, AMX_TILE_M, wsc)


@always_inline
def amx_panel_1x32[
    N: Int, K: Int, block: Int,
    write: def(Int, Int, SIMD[DType.float32, WIDTH]) capturing [_] -> None,
](
    act: I8Ptr, m_panel: Int, m_rows: Int, act_scale: F32Ptr, weight: I8Ptr,
    wsc: F32Ptr, t: Int,
):
    comptime nb = K // block
    var c = InlineArray[Int32, 2 * CTILE](uninitialized=True)
    var cp = UnsafePointer(to=c).bitcast[Int32]()
    var c0 = cp
    var c1 = cp + CTILE
    var n0 = t * VNNI_N_STEP
    var n1 = n0 + AMX_TILE_N

    comptime if nb == 1:
        tilezero[4]()
        tilezero[5]()
        for k_off in range(0, K, AMX_K_STEP):
            tileload[0, DType.int8](act + m_panel * K + k_off, K)
            var base = amx_b_tile_base[K](t, k_off)
            tileload[2, DType.int8](weight + base, AMX_K_STEP)
            tileload[3, DType.int8](weight + base + HALF_BYTES, AMX_K_STEP)
            tdpbssd[4, 0, 2]()
            tdpbssd[5, 0, 3]()
        tilestore[4, DType.int32](c0, C_STRIDE)
        tilestore[5, DType.int32](c1, C_STRIDE)
        dequant_fused[N, write](c0, m_panel, n0, m_rows, act_scale, wsc)
        dequant_fused[N, write](c1, m_panel, n1, m_rows, act_scale, wsc)
    else:
        var facc = InlineArray[Float32, 2 * CTILE](fill=Float32(0))
        var fp = UnsafePointer(to=facc).bitcast[Float32]()
        var f0 = fp
        var f1 = fp + CTILE
        for b in range(nb):
            tilezero[4]()
            tilezero[5]()
            for k_off in range(b * block, (b + 1) * block, AMX_K_STEP):
                tileload[0, DType.int8](act + m_panel * K + k_off, K)
                var base = amx_b_tile_base[K](t, k_off)
                tileload[2, DType.int8](weight + base, AMX_K_STEP)
                tileload[3, DType.int8](weight + base + HALF_BYTES, AMX_K_STEP)
                tdpbssd[4, 0, 2]()
                tdpbssd[5, 0, 3]()
            tilestore[4, DType.int32](c0, C_STRIDE)
            tilestore[5, DType.int32](c1, C_STRIDE)
            block_accumulate(c0, f0, m_panel, b, nb, m_rows, act_scale)
            block_accumulate(c1, f1, m_panel, b, nb, m_rows, act_scale)
        block_finalize[N, write](f0, m_panel, n0, m_rows, wsc)
        block_finalize[N, write](f1, m_panel, n1, m_rows, wsc)


def amx_gemm[
    N: Int, K: Int, block: Int,
    write: def(Int, Int, SIMD[DType.float32, WIDTH]) capturing [_] -> None,
](
    act: I8Ptr, m: Int, act_scale: F32Ptr, weight: I8Ptr, wsc: F32Ptr,
    start_tile: Int, end_tile: Int,
):
    comptime assert K % AMX_K_STEP == 0, "K must be AMX K-step aligned"
    comptime assert K % block == 0, "block must divide K"
    comptime assert block % AMX_K_STEP == 0, "block must be AMX K-step aligned"
    var full_cfg = make_224_i8_config()
    var m_panel = 0

    if m >= AMX_TILE_M:
        ldtilecfg(UnsafePointer(to=full_cfg))
        while m_panel + 2 * AMX_TILE_M <= m:
            for t in range(start_tile, end_tile):
                amx_panel_2x32[N, K, block, write](
                    act, m_panel, act_scale, weight, wsc, t)
            m_panel += 2 * AMX_TILE_M
        if m_panel + AMX_TILE_M <= m:
            for t in range(start_tile, end_tile):
                amx_panel_1x32[N, K, block, write](
                    act, m_panel, AMX_TILE_M, act_scale, weight, wsc, t)
            m_panel += AMX_TILE_M

    if m_panel < m:
        var rem = m - m_panel
        var part_cfg = make_partial_config(rem)
        ldtilecfg(UnsafePointer(to=part_cfg))
        for t in range(start_tile, end_tile):
            amx_panel_1x32[N, K, block, write](
                act, m_panel, rem, act_scale, weight, wsc, t)

    tilerelease()


@fieldwise_init
struct BqLinearAmxKernel[N: Int, K: Int, MR: Int](RangePartitionedKernel):
    var act: I8Ptr
    var act_scale: F32Ptr
    var weight: I8Ptr
    var wsc: F32Ptr
    var output: BF16Ptr
    var m: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var output = self.output

        @parameter
        def write(row: Int, n_base: Int, res: SIMD[DType.float32, WIDTH]):
            store_out[DType.bfloat16, WIDTH](res, output + row * Self.N + n_base)

        amx_gemm[Self.N, Self.K, Self.K, write](
            self.act, self.m, self.act_scale, self.weight, self.wsc,
            self.start, self.end)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_bq_linear_amx[
    P: BurstThreadPool, quant: QuantRecipe, n: Int, m: Int, tp: Int,
    Profile: Bool, N: Int, //,
    MR: Int = 4, max_worker_count: Int = 128,
](
    act: ButterquantActivation[tp],
    weight: ButterquantWeight[quant, n, m, tp],
    output: Binding[BFloat16, tp],
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime assert quant_vnni_packed[quant](), "bq linear amx consumes a VNNI-packed weight"
    if seq_len <= 0:
        return
    comptime num_tiles = n // VNNI_N_STEP
    comptime Kern = BqLinearAmxKernel[n, m, MR]

    @parameter
    def make(r: Int) -> Kern:
        return Kern(act.data[r], act.scale[r], weight.data[r], weight.scale[r],
                    output[r], seq_len, 0, 0)

    fanout_dispatch[
        tp, make, max_worker_count=max_worker_count,
        worker_policy=matmul_workers, label="bq_linear_amx",
    ](pools, prof, num_tiles, n * m * seq_len,
      inline_threshold_bytes=GEMV_INLINE_ROWS * m)


@fieldwise_init
struct BqBlockLinearAmxKernel[N: Int, K: Int, block: Int, MR: Int](
    RangePartitionedKernel
):
    var act: I8Ptr
    var act_scale: F32Ptr
    var weight: I8Ptr
    var wsc: F32Ptr
    var output: BF16Ptr
    var m: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var output = self.output

        @parameter
        def write(row: Int, n_base: Int, res: SIMD[DType.float32, WIDTH]):
            store_out[DType.bfloat16, WIDTH](res, output + row * Self.N + n_base)

        amx_gemm[Self.N, Self.K, Self.block, write](
            self.act, self.m, self.act_scale, self.weight, self.wsc,
            self.start, self.end)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_bq_block_linear_amx[
    P: BurstThreadPool, quant: QuantRecipe, n: Int, m: Int, tp: Int,
    Profile: Bool, N: Int, //,
    MR: Int = 4, max_worker_count: Int = 128,
](
    act: ButterquantBlockActivation[tp],
    weight: ButterquantWeight[quant, n, m, tp],
    output: Binding[BFloat16, tp],
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime assert quant_vnni_packed[quant](), "bq block linear amx consumes a VNNI-packed weight"
    if seq_len <= 0:
        return
    comptime num_tiles = n // VNNI_N_STEP
    comptime Kern = BqBlockLinearAmxKernel[n, m, quant_k_block[quant](), MR]

    @parameter
    def make(r: Int) -> Kern:
        return Kern(act.data[r], act.scale[r], weight.data[r], weight.scale[r],
                    output[r], seq_len, 0, 0)

    fanout_dispatch[
        tp, make, max_worker_count=max_worker_count,
        worker_policy=matmul_workers, label="bq_block_linear_amx",
    ](pools, prof, num_tiles, n * m * seq_len,
      inline_threshold_bytes=GEMV_INLINE_ROWS * m)
