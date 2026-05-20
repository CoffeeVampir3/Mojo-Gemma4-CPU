from std.collections import InlineArray
from std.memory import Span, alloc
from std.os import abort
from std.sys.info import simd_width_of

from kernels.gemv import dispatch_gemv_softcap, softcap_value
from kernels.helpers import Binding, ArenaBases
from notstdcollections import HeapMoveArray
from threading.threading_traits import BurstKernel, BurstThreadPool


comptime W = simd_width_of[DType.float32]()
comptime ROWS = 6


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


def bf16_softcap_expected(raw: Float32) -> Float32:
    var capped = softcap_value[30.0](SIMD[DType.float32, 1](raw))
    return Float32(capped.cast[DType.bfloat16]()[0])


def check_bf16(got: Float32, expected: Float32, msg: String):
    check(got == expected,
        msg + " expected=" + String(expected) + " got=" + String(got))


def test_gemv_softcap():
    var x = alloc[BFloat16](W)
    var weight = alloc[BFloat16](ROWS * W)
    var output = alloc[BFloat16](ROWS)

    for i in range(W):
        x[i] = BFloat16(Float32(0.0))
    x[0] = BFloat16(Float32(1.0))

    var raw = InlineArray[Float32, ROWS](uninitialized=True)
    raw[0] = Float32(0.0)
    raw[1] = Float32(15.0)
    raw[2] = Float32(-15.0)
    raw[3] = Float32(30.0)
    raw[4] = Float32(60.0)
    raw[5] = Float32(-60.0)

    for row in range(ROWS):
        for col in range(W):
            weight[row * W + col] = BFloat16(Float32(0.0))
        weight[row * W] = BFloat16(raw[row])

    var pools = HeapMoveArray[TestPool](1)
    pools.push(TestPool(3, 0))
    var bases = ArenaBases[1].uninitialized()
    bases[0] = 0
    dispatch_gemv_softcap[
        rows=ROWS, cols=W, tp=1, cap=30.0,
    ](
        Binding[BFloat16, 1](x.as_any_origin(), bases),
        Binding[BFloat16, 1](weight.as_any_origin(), bases),
        Binding[BFloat16, 1](output.as_any_origin(), bases),
        pools,
    )

    for row in range(ROWS):
        var got = Float32(output[row])
        var expected = bf16_softcap_expected(raw[row])
        check_bf16(got, expected, "softcap row " + String(row))

    check(
        abs(Float64(Float32(output[4]))) <= Float64(30.0),
        "positive softcap did not bound a large logit",
    )
    check(
        abs(Float64(Float32(output[5]))) <= Float64(30.0),
        "negative softcap did not bound a large logit",
    )
    check(
        Float32(output[3]) < raw[3],
        "softcap should change a logit at the configured cap",
    )

    x.free()
    weight.free()
    output.free()


def main():
    test_gemv_softcap()
    print("gemv tests passed")
