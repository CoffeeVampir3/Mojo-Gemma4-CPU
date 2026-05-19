from std.memory import Span, UnsafePointer, alloc
from std.os import abort

from kernels.gemm import dispatch_gemm
from kernels.gemv import dispatch_gemv
from kernels.helpers import Binding, ArenaBases
from notstdcollections import HeapMoveArray
from threading.threading_traits import BurstKernel, BurstThreadPool


comptime BASES = ArenaBases[1].fill(0)
comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]


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


def fill_pattern(p: BF16Ptr, count: Int, off: Int):
    for i in range(count):
        var v = Float32(((i + off) % 31) - 15) * Float32(0.03125)
        p[i] = BFloat16(v)


def reference_gemm[rows: Int, cols: Int](
    x: BF16Ptr, w: BF16Ptr, out_buf: BF16Ptr, m: Int,
):
    for tok in range(m):
        for n in range(rows):
            var acc = Float32(0)
            for k in range(cols):
                acc += Float32(x[tok * cols + k]) * Float32(w[n * cols + k])
            out_buf[tok * rows + n] = BFloat16(acc)


def assert_close(
    got: Float32, exp: Float32, label: String, abs_floor: Float32 = 0.5,
):
    var diff = abs(got - exp)
    var tol = abs(exp) * Float32(0.02)
    if abs_floor > tol:
        tol = abs_floor
    check(diff <= tol,
        label + " got=" + String(got) + " exp=" + String(exp)
        + " diff=" + String(diff) + " tol=" + String(tol))


def run_gemm_case[rows: Int, cols: Int, MR: Int](m: Int, label: String):
    var alloc_m = m if m > 0 else 1
    var x = alloc[BFloat16](alloc_m * cols).as_any_origin()
    var w = alloc[BFloat16](rows * cols).as_any_origin()
    var got = alloc[BFloat16](alloc_m * rows).as_any_origin()
    var expected = alloc[BFloat16](alloc_m * rows).as_any_origin()

    fill_pattern(x, alloc_m * cols, 0)
    fill_pattern(w, rows * cols, 7)
    for i in range(alloc_m * rows):
        got[i] = BFloat16(0.0)
        expected[i] = BFloat16(0.0)

    var pools = HeapMoveArray[TestPool](1)
    pools.push(TestPool(4, 0))

    dispatch_gemm[
        rows=rows, cols=cols, tp=1, MR=MR,
    ](
        Binding[BFloat16, 1](x, BASES),
        Binding[BFloat16, 1](w, BASES),
        Binding[BFloat16, 1](got, BASES),
        m, pools)

    if m == 0:
        for i in range(alloc_m * rows):
            check(Float32(got[i]) == Float32(0.0),
                label + " m=0 must not write outputs")
    else:
        reference_gemm[rows, cols](x, w, expected, m)
        for tok in range(m):
            for n in range(rows):
                assert_close(
                    Float32(got[tok * rows + n]),
                    Float32(expected[tok * rows + n]),
                    label + " tok=" + String(tok) + " n=" + String(n))

    x.free(); w.free(); got.free(); expected.free()


def test_gemm_correctness():
    run_gemm_case[rows=8, cols=256, MR=1](4, "MR=1 m=4")
    run_gemm_case[rows=8, cols=256, MR=2](4, "MR=2 m=4")
    run_gemm_case[rows=8, cols=256, MR=4](4, "MR=4 m=4")
    run_gemm_case[rows=8, cols=256, MR=8](4, "MR=8 m=4")

    run_gemm_case[rows=8, cols=256, MR=4](1, "MR=4 m=1")
    run_gemm_case[rows=8, cols=256, MR=4](17, "MR=4 m=17")
    run_gemm_case[rows=8, cols=256, MR=4](64, "MR=4 m=64")

    run_gemm_case[rows=8, cols=256, MR=4](0, "MR=4 m=0")

    run_gemm_case[rows=16, cols=128, MR=4](4, "wide cols=128")
    run_gemm_case[rows=16, cols=2112, MR=4](4, "FFN down cols=2112")
    run_gemm_case[rows=16, cols=2816, MR=4](4, "FFN gate cols=2816")


def test_gemm_matches_gemv_at_m1():
    comptime rows = 16
    comptime cols = 2816
    var x = alloc[BFloat16](cols).as_any_origin()
    var w = alloc[BFloat16](rows * cols).as_any_origin()
    var out_gemm = alloc[BFloat16](rows).as_any_origin()
    var out_gemv = alloc[BFloat16](rows).as_any_origin()

    fill_pattern(x, cols, 0)
    fill_pattern(w, rows * cols, 7)
    for i in range(rows):
        out_gemm[i] = BFloat16(0.0)
        out_gemv[i] = BFloat16(0.0)

    var pools = HeapMoveArray[TestPool](1)
    pools.push(TestPool(4, 0))

    dispatch_gemm[
        rows=rows, cols=cols, tp=1, MR=4,
    ](
        Binding[BFloat16, 1](x, BASES),
        Binding[BFloat16, 1](w, BASES),
        Binding[BFloat16, 1](out_gemm, BASES),
        1, pools)

    dispatch_gemv[
        rows=rows, cols=cols, tp=1,
    ](
        Binding[BFloat16, 1](x, BASES),
        Binding[BFloat16, 1](w, BASES),
        Binding[BFloat16, 1](out_gemv, BASES),
        pools)

    for n in range(rows):
        var g = Float32(out_gemm[n])
        var v = Float32(out_gemv[n])
        assert_close(g, v, "m=1 gemm-vs-gemv n=" + String(n))

    x.free(); w.free(); out_gemm.free(); out_gemv.free()


def main():
    test_gemm_correctness()
    test_gemm_matches_gemv_at_m1()
    print("gemm tests passed")
