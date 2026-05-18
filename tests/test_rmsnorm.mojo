from std.collections import InlineArray
from std.memory import Span, UnsafePointer, alloc
from std.os import abort

from kernels.rmsnorm import (
    rms_norm_row, dispatch_rms_norm,
    norm_residual_add_row, fused_norm_residual_add,
)
from kernels.helpers import Binding, ArenaBases
from simd_math.ops import sqrt
from notstdcollections import HeapMoveArray
from threading.threading_traits import BurstKernel, BurstThreadPool


comptime HIDDEN = 2816
comptime SQRT_N = sqrt[DType.float32, 1](HIDDEN)
comptime N_EPS = HIDDEN * 1e-6

comptime BF16ExtPtr = UnsafePointer[BFloat16, MutExternalOrigin]
comptime BASES = ArenaBases[1].fill(0)


@fieldwise_init
struct TestPool(BurstThreadPool):
    var capacity: Int
    var timestamp: Int

    def get_capacity(self) -> Int:
        return self.capacity

    def dispatch[K: BurstKernel, origin: MutOrigin](
        mut self, kernels: Span[K, origin], num_jobs: Int):
        for i in range(num_jobs):
            var kernel = kernels[i]
            kernel.execute()
        self.timestamp += 1

    def join(mut self):
        pass

    def last_worker_timestamp(self) -> Int:
        return self.timestamp

    def wake(mut self):
        pass

    def sleep(mut self):
        pass


def check(ok: Bool, msg: String):
    if not ok:
        abort("FAIL: " + msg)


def fill_input(ptr: BF16ExtPtr, count: Int):
    for i in range(count):
        ptr[i] = BFloat16(Float32((i % 127) - 63) * 0.01)


def fill_residual(ptr: BF16ExtPtr, count: Int):
    for i in range(count):
        ptr[i] = BFloat16(Float32((i % 97) - 48) * 0.02)


def fill_weight(ptr: BF16ExtPtr, count: Int, seed: Int):
    for i in range(count):
        ptr[i] = BFloat16(
            Float32(1.0) + Float32((i + seed) % 64) * 0.001)


def assert_same(a: BF16ExtPtr, b: BF16ExtPtr, count: Int, label: String):
    for i in range(count):
        var got = Float32(a[i])
        var expected = Float32(b[i])
        check(got == expected,
            label + " [" + String(i) + "] expected="
            + String(expected) + " got=" + String(got))


def check_rms_norm_seq(count: Int):
    var total = count * HIDDEN
    var src = alloc[BFloat16](total)
    var expected = alloc[BFloat16](total)
    var got = alloc[BFloat16](total)
    var weight = alloc[BFloat16](HIDDEN)

    fill_input(src, total)
    fill_weight(weight, HIDDEN, 0)

    var src_any = src.as_any_origin()
    var expected_any = expected.as_any_origin()
    var got_any = got.as_any_origin()
    var weight_any = weight.as_any_origin()

    for tok in range(count):
        var off = tok * HIDDEN
        rms_norm_row[HIDDEN, SQRT_N, N_EPS](
            src_any + off, expected_any + off, weight_any)

    var pools = HeapMoveArray[TestPool](1)
    pools.push(TestPool(4, 0))
    dispatch_rms_norm[hidden=HIDDEN, sqrt_n=SQRT_N, n_eps=N_EPS, tp=1](
        Binding[BFloat16, 1](src_any, BASES),
        Binding[BFloat16, 1](got_any, BASES),
        Binding[BFloat16, 1](weight_any, BASES),
        count, pools)
    assert_same(got, expected, total, "rms dispatch output")

    src.free()
    expected.free()
    got.free()
    weight.free()


def check_fused_norm_residual_add_seq(count: Int):
    var total = count * HIDDEN
    var src = alloc[BFloat16](total)
    var residual = alloc[BFloat16](total)
    var expected = alloc[BFloat16](total)
    var got = alloc[BFloat16](total)
    var weight = alloc[BFloat16](HIDDEN)

    fill_input(src, total)
    fill_residual(residual, total)
    fill_residual(got, total)
    fill_weight(weight, HIDDEN, 31)

    var src_any = src.as_any_origin()
    var residual_any = residual.as_any_origin()
    var expected_any = expected.as_any_origin()
    var got_any = got.as_any_origin()
    var weight_any = weight.as_any_origin()

    for tok in range(count):
        var off = tok * HIDDEN
        norm_residual_add_row[HIDDEN, SQRT_N, N_EPS](
            src_any + off, residual_any + off, expected_any + off,
            weight_any)

    var pools = HeapMoveArray[TestPool](1)
    pools.push(TestPool(4, 0))
    fused_norm_residual_add[
        hidden=HIDDEN, sqrt_n=SQRT_N, n_eps=N_EPS, tp=1,
    ](
        Binding[BFloat16, 1](src_any, BASES),
        Binding[BFloat16, 1](got_any, BASES),
        Binding[BFloat16, 1](got_any, BASES),
        Binding[BFloat16, 1](weight_any, BASES),
        count, pools)

    assert_same(got, expected, total, "fused norm residual add output")

    src.free()
    residual.free()
    expected.free()
    got.free()
    weight.free()


def main():
    check_rms_norm_seq(1)
    check_rms_norm_seq(4)
    check_rms_norm_seq(33)
    check_fused_norm_residual_add_seq(1)
    check_fused_norm_residual_add_seq(4)
    check_fused_norm_residual_add_seq(33)
    print("rmsnorm tests passed")
