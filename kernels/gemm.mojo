from std.collections import InlineArray

from simd_math import pick_port_unroll
from threading.threading_traits import BurstThreadPool
from .helpers import (
    Chain, RangePartitionedKernel,
    fanout_dispatch, saturate_workers,
    Binding, BF16Ptr, BW,
)
from .dispatch_heuristics import GEMV_INLINE_ROWS
from .dot_products import bf16_panel_dot_to_scalars
from .profiling import Profiler


@always_inline
def gemm_row_panel[
    panel: Int, //,
    rows: Int, cols: Int, port_unroll: Int,
](
    x_base: BF16Ptr, weight: BF16Ptr, output: BF16Ptr,
    m_panel: Int, start: Int, end: Int,
):
    var x_rows = InlineArray[BF16Ptr, panel](uninitialized=True)
    comptime for r in range(panel):
        x_rows[r] = x_base + (m_panel + r) * cols

    for n in range(start, end):
        var w_row = weight + n * cols
        var scalars = bf16_panel_dot_to_scalars[
            cols=cols, port_unroll=port_unroll,
        ](w_row, x_rows)
        comptime for r in range(panel):
            (output + (m_panel + r) * rows + n)[] = (
                scalars[r].cast[DType.bfloat16]())


@always_inline
def gemm_range[
    rows: Int, cols: Int, MR: Int,
](
    x: BF16Ptr, weight: BF16Ptr, output: BF16Ptr,
    m: Int, start: Int, end: Int,
):
    comptime PU = pick_port_unroll[BW, cols]()
    var m_panel = 0
    while m_panel + MR <= m:
        gemm_row_panel[
            panel=MR, rows=rows, cols=cols, port_unroll=PU,
        ](x, weight, output, m_panel, start, end)
        m_panel += MR
    while m_panel < m:
        gemm_row_panel[
            panel=1, rows=rows, cols=cols, port_unroll=PU,
        ](x, weight, output, m_panel, start, end)
        m_panel += 1


@fieldwise_init
struct GemmKernel[rows: Int, cols: Int, MR: Int = 4](
    RangePartitionedKernel
):
    """x: [m, cols] bf16 row-major, weight: [rows, cols] bf16 row-major
    (one weight row per output channel), output: [m, rows] bf16 row-major.
    Partition is over the rows axis; each worker streams its weight stripe
    once and walks m in MR-sized panels for register reuse."""
    var x: BF16Ptr
    var weight: BF16Ptr
    var output: BF16Ptr
    var m: Int
    var start: Int
    var end: Int

    def execute(mut self):
        gemm_range[Self.rows, Self.cols, Self.MR](
            self.x, self.weight, self.output, self.m,
            self.start, self.end)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_gemm[
    P: BurstThreadPool, Profile: Bool, N: Int, //,
    rows: Int, cols: Int, tp: Int, MR: Int = 4,
    max_worker_count: Int = 128,
](
    x: Binding[BFloat16, tp],
    weight: Binding[BFloat16, tp],
    output: Binding[BFloat16, tp],
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    if seq_len <= 0:
        return
    comptime K = GemmKernel[rows, cols, MR]

    @parameter
    def make(r: Int) -> K:
        return K(x[r], weight[r], output[r], seq_len, 0, 0)

    fanout_dispatch[tp, make, max_worker_count=max_worker_count, label="gemm"](
        pools, prof, rows,
        seq_len * cols * 2 + rows * cols * 2,
        inline_threshold_bytes=GEMV_INLINE_ROWS * cols * 2)


@fieldwise_init
struct ScaledGemmKernel[
    rows: Int, cols: Int, MR: Int, numer: Int, denom: Int,
](RangePartitionedKernel):
    var x: BF16Ptr
    var weight: BF16Ptr
    var output: BF16Ptr
    var m: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var my_start = self.start * Self.numer // Self.denom
        var my_end = self.end * Self.numer // Self.denom
        gemm_range[Self.rows, Self.cols, Self.MR](
            self.x, self.weight, self.output, self.m,
            my_start, my_end)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


def dispatch_gemm_chained_qkv[
    P: BurstThreadPool, Profile: Bool, N: Int, //,
    q_rows: Int, kv_rows: Int, cols: Int, tp: Int, MR: Int = 4,
    max_worker_count: Int = 128,
](
    x: Binding[BFloat16, tp],
    q_weight: Binding[BFloat16, tp],
    k_weight: Binding[BFloat16, tp],
    v_weight: Binding[BFloat16, tp],
    q_out: Binding[BFloat16, tp],
    k_out: Binding[BFloat16, tp],
    v_out: Binding[BFloat16, tp],
    seq_len: Int,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    if seq_len <= 0:
        return
    comptime total_rows = q_rows + kv_rows + kv_rows
    comptime QKernel = ScaledGemmKernel[q_rows, cols, MR, q_rows, total_rows]
    comptime KKernel = ScaledGemmKernel[kv_rows, cols, MR, kv_rows, total_rows]
    comptime VKernel = ScaledGemmKernel[kv_rows, cols, MR, kv_rows, total_rows]
    comptime QK = Chain[QKernel, KKernel]
    comptime QKV = Chain[QK, VKernel]

    @parameter
    def make(r: Int) -> QKV:
        return QKV(
            QK(
                QKernel(x[r], q_weight[r], q_out[r], seq_len, 0, 0),
                KKernel(x[r], k_weight[r], k_out[r], seq_len, 0, 0),
            ),
            VKernel(x[r], v_weight[r], v_out[r], seq_len, 0, 0),
        )

    fanout_dispatch[
        tp, make,
        max_worker_count=max_worker_count,
        worker_policy=saturate_workers,
        label="gemm_chained_qkv",
    ](pools, prof, total_rows, seq_len * cols * 2 + total_rows * cols * 2)
