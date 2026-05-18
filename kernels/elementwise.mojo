from std.algorithm import vectorize

from notstdcollections import HeapMoveArray
from threading.threading_traits import BurstThreadPool
from simd_math.ops import gelu_tanh_f32
from .helpers import (
    OutputPartitionedKernel, DispatchBuffer, Binding,
    tile_dispatch, recommended_workers, join_all,
    BF16Ptr, W,
)


@always_inline
def gelu_gate_up_row[intermediate: Int](
    gate: BF16Ptr, up: BF16Ptr, dst: BF16Ptr,
):
    def step[width: Int](idx: Int) {read}:
        var g = (gate + idx).load[width=width]().cast[DType.float32]()
        var u = (up + idx).load[width=width]().cast[DType.float32]()
        var v = gelu_tanh_f32[width](g) * u
        (dst + idx).store(v.cast[DType.bfloat16]())

    vectorize[W](intermediate, step)


@fieldwise_init
struct GeluGateUpTokenKernel[intermediate: Int](OutputPartitionedKernel):
    var gate: BF16Ptr
    var up: BF16Ptr
    var dst: BF16Ptr
    var start: Int
    var end: Int

    def execute(mut self):
        for tok in range(self.start, self.end):
            var off = tok * Self.intermediate
            gelu_gate_up_row[Self.intermediate](
                self.gate + off, self.up + off, self.dst + off)

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.start = start
        self.end = end


comptime GELU_GATE_UP_INLINE_TOKENS = 16


def dispatch_gelu_gate_up[
    P: BurstThreadPool, //,
    intermediate: Int, tp: Int, max_worker_count: Int = 128,
](
    gate: Binding[BFloat16, tp],
    up: Binding[BFloat16, tp],
    dst: Binding[BFloat16, tp],
    seq_len: Int,
    mut pools: HeapMoveArray[P],
):
    if seq_len <= GELU_GATE_UP_INLINE_TOKENS:
        for r in range(tp):
            for tok in range(seq_len):
                var off = tok * intermediate
                gelu_gate_up_row[intermediate](
                    gate[r] + off, up[r] + off, dst[r] + off)
        return

    var data_bytes = seq_len * intermediate * 6
    var buf = DispatchBuffer[
        GeluGateUpTokenKernel[intermediate], max_worker_count,
    ]()
    for r in range(tp):
        var nw = recommended_workers(
            data_bytes, min(max_worker_count, pools[r].get_capacity()))
        tile_dispatch(buf,
            GeluGateUpTokenKernel[intermediate](
                gate[r], up[r], dst[r], 0, 0),
            pools[r], seq_len, num_workers=nw)
    join_all[tp](pools)


@always_inline
def scalar_mul_row[hidden: Int](
    src: BF16Ptr, dst: BF16Ptr, scalar: Float32,
):
    def step[width: Int](idx: Int) {read}:
        var x = (src + idx).load[width=width]().cast[DType.float32]()
        var factor = SIMD[DType.float32, width](scalar)
        (dst + idx).store((x * factor).cast[DType.bfloat16]())

    vectorize[W](hidden, step)


@fieldwise_init
struct ScalarMulTokenKernel[hidden: Int](OutputPartitionedKernel):
    var src: BF16Ptr
    var dst: BF16Ptr
    var scalar: Float32
    var start: Int
    var end: Int

    def execute(mut self):
        for tok in range(self.start, self.end):
            var off = tok * Self.hidden
            scalar_mul_row[Self.hidden](
                self.src + off, self.dst + off, self.scalar)

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.start = start
        self.end = end


comptime SCALAR_MUL_INLINE_TOKENS = 16


def dispatch_scalar_mul[
    P: BurstThreadPool, //,
    hidden: Int, tp: Int, max_worker_count: Int = 128,
](
    src: Binding[BFloat16, tp],
    dst: Binding[BFloat16, tp],
    scalar: Float32,
    seq_len: Int,
    mut pools: HeapMoveArray[P],
):
    if seq_len <= SCALAR_MUL_INLINE_TOKENS:
        for r in range(tp):
            for tok in range(seq_len):
                var off = tok * hidden
                scalar_mul_row[hidden](
                    src[r] + off, dst[r] + off, scalar)
        return

    var data_bytes = seq_len * hidden * 4
    var buf = DispatchBuffer[
        ScalarMulTokenKernel[hidden], max_worker_count,
    ]()
    for r in range(tp):
        var nw = recommended_workers(
            data_bytes, min(max_worker_count, pools[r].get_capacity()))
        tile_dispatch(buf,
            ScalarMulTokenKernel[hidden](
                src[r], dst[r], scalar, 0, 0),
            pools[r], seq_len, num_workers=nw)
    join_all[tp](pools)
