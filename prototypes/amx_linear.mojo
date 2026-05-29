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
    ButterquantWeight, ButterquantActivation, quant_vnni_packed,
)
from quant.recipe import QuantRecipe

from prototypes.amx_intrinsics import (
    AMX_TILE_M, AMX_TILE_N, AMX_K_STEP,
    TileConfig, make_224_i8_config, make_partial_config,
    ldtilecfg, tilerelease, tilezero, tileload, tilestore, tdpbssd,
)


comptime I32Ptr = UnsafePointer[Int32, MutAnyOrigin]
comptime AMX_HALF_BYTES = AMX_TILE_M * AMX_K_STEP


@always_inline
def amx_b_tile_base[K: Int](t: Int, k_off: Int) -> Int:
    return (t * VNNI_N_STEP) * K + k_off * VNNI_N_STEP


@always_inline
def dequant_c_tile[N: Int, Out: DType](
    c: I32Ptr,
    row_base: Int,
    n_base: Int,
    m_rows: Int,
    act_scale: F32Ptr,
    wsc: F32Ptr,
    dst: UnsafePointer[Scalar[Out], MutAnyOrigin],
):
    comptime width = simd_width_of[DType.int32]()
    comptime inv127 = Float32(1.0) / Float32(127.0)
    for mi in range(m_rows):
        var ad = act_scale[row_base + mi] * inv127
        var nj = 0
        while nj < AMX_TILE_N:
            var cv = (c + mi * AMX_TILE_N + nj).load[width=width]().cast[
                DType.float32]()
            var ws = (wsc + n_base + nj).load[width=width]()
            var res = cv * ad * ws
            store_out[Out, width](res, dst + (row_base + mi) * N + n_base + nj)
            nj += width


@always_inline
def amx_panel_2x32[N: Int, K: Int, Out: DType](
    act: I8Ptr,
    m_panel: Int,
    act_scale: F32Ptr,
    weight: I8Ptr,
    wsc: F32Ptr,
    dst: UnsafePointer[Scalar[Out], MutAnyOrigin],
    t: Int,
):
    var c00 = InlineArray[Int32, AMX_TILE_M * AMX_TILE_N](uninitialized=True)
    var c01 = InlineArray[Int32, AMX_TILE_M * AMX_TILE_N](uninitialized=True)
    var c10 = InlineArray[Int32, AMX_TILE_M * AMX_TILE_N](uninitialized=True)
    var c11 = InlineArray[Int32, AMX_TILE_M * AMX_TILE_N](uninitialized=True)

    tilezero[4]()
    tilezero[5]()
    tilezero[6]()
    tilezero[7]()

    for k_off in range(0, K, AMX_K_STEP):
        tileload[0, DType.int8](act + m_panel * K + k_off, K)
        tileload[1, DType.int8](act + (m_panel + AMX_TILE_M) * K + k_off, K)
        var base = amx_b_tile_base[K](t, k_off)
        tileload[2, DType.int8](weight + base, AMX_K_STEP)
        tileload[3, DType.int8](weight + base + AMX_HALF_BYTES, AMX_K_STEP)
        tdpbssd[4, 0, 2]()
        tdpbssd[5, 0, 3]()
        tdpbssd[6, 1, 2]()
        tdpbssd[7, 1, 3]()

    var c_stride = AMX_TILE_N * 4
    tilestore[4, DType.int32](UnsafePointer(to=c00).bitcast[Int32](), c_stride)
    tilestore[5, DType.int32](UnsafePointer(to=c01).bitcast[Int32](), c_stride)
    tilestore[6, DType.int32](UnsafePointer(to=c10).bitcast[Int32](), c_stride)
    tilestore[7, DType.int32](UnsafePointer(to=c11).bitcast[Int32](), c_stride)

    var n0 = t * VNNI_N_STEP
    var n1 = n0 + AMX_TILE_N
    dequant_c_tile[N, Out](
        UnsafePointer(to=c00).bitcast[Int32](), m_panel, n0, AMX_TILE_M,
        act_scale, wsc, dst)
    dequant_c_tile[N, Out](
        UnsafePointer(to=c01).bitcast[Int32](), m_panel, n1, AMX_TILE_M,
        act_scale, wsc, dst)
    dequant_c_tile[N, Out](
        UnsafePointer(to=c10).bitcast[Int32](), m_panel + AMX_TILE_M, n0,
        AMX_TILE_M, act_scale, wsc, dst)
    dequant_c_tile[N, Out](
        UnsafePointer(to=c11).bitcast[Int32](), m_panel + AMX_TILE_M, n1,
        AMX_TILE_M, act_scale, wsc, dst)


@always_inline
def amx_panel_1x32[N: Int, K: Int, Out: DType](
    act: I8Ptr,
    m_panel: Int,
    m_rows: Int,
    act_scale: F32Ptr,
    weight: I8Ptr,
    wsc: F32Ptr,
    dst: UnsafePointer[Scalar[Out], MutAnyOrigin],
    t: Int,
):
    var c0 = InlineArray[Int32, AMX_TILE_M * AMX_TILE_N](uninitialized=True)
    var c1 = InlineArray[Int32, AMX_TILE_M * AMX_TILE_N](uninitialized=True)

    tilezero[4]()
    tilezero[5]()

    for k_off in range(0, K, AMX_K_STEP):
        tileload[0, DType.int8](act + m_panel * K + k_off, K)
        var base = amx_b_tile_base[K](t, k_off)
        tileload[2, DType.int8](weight + base, AMX_K_STEP)
        tileload[3, DType.int8](weight + base + AMX_HALF_BYTES, AMX_K_STEP)
        tdpbssd[4, 0, 2]()
        tdpbssd[5, 0, 3]()

    var c_stride = AMX_TILE_N * 4
    tilestore[4, DType.int32](UnsafePointer(to=c0).bitcast[Int32](), c_stride)
    tilestore[5, DType.int32](UnsafePointer(to=c1).bitcast[Int32](), c_stride)

    var n0 = t * VNNI_N_STEP
    dequant_c_tile[N, Out](
        UnsafePointer(to=c0).bitcast[Int32](), m_panel, n0, m_rows,
        act_scale, wsc, dst)
    dequant_c_tile[N, Out](
        UnsafePointer(to=c1).bitcast[Int32](), m_panel, n0 + AMX_TILE_N,
        m_rows, act_scale, wsc, dst)


def amx_gemm_i8_per_row[N: Int, K: Int, Out: DType](
    act: I8Ptr,
    m: Int,
    act_scale: F32Ptr,
    weight: I8Ptr,
    wsc: F32Ptr,
    dst: UnsafePointer[Scalar[Out], MutAnyOrigin],
    start_tile: Int,
    end_tile: Int,
):
    comptime assert K % AMX_K_STEP == 0, "K must be AMX K-step aligned"
    var full_cfg = make_224_i8_config()
    var m_panel = 0

    if m >= AMX_TILE_M:
        ldtilecfg(UnsafePointer(to=full_cfg))
        while m_panel + 2 * AMX_TILE_M <= m:
            for t in range(start_tile, end_tile):
                amx_panel_2x32[N, K, Out](
                    act, m_panel, act_scale, weight, wsc, dst, t)
            m_panel += 2 * AMX_TILE_M
        if m_panel + AMX_TILE_M <= m:
            for t in range(start_tile, end_tile):
                amx_panel_1x32[N, K, Out](
                    act, m_panel, AMX_TILE_M, act_scale, weight, wsc, dst, t)
            m_panel += AMX_TILE_M

    if m_panel < m:
        var rem = m - m_panel
        var part_cfg = make_partial_config(rem)
        ldtilecfg(UnsafePointer(to=part_cfg))
        for t in range(start_tile, end_tile):
            amx_panel_1x32[N, K, Out](
                act, m_panel, rem, act_scale, weight, wsc, dst, t)

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
        amx_gemm_i8_per_row[Self.N, Self.K, DType.bfloat16](
            self.act, self.m, self.act_scale, self.weight, self.wsc,
            self.output, self.start, self.end)

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
