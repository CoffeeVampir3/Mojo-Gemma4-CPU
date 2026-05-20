from std.memory import Span, alloc
from std.os import abort

from kernels.attention_dispatch_kernels import dispatch_sliding_attention
from kernels.attention_ops import flash_partial_stride
from kernels.helpers import ArenaBases, Binding
from notstdcollections import HeapMoveArray
from threading.threading_traits import BurstKernel, BurstThreadPool


comptime HEAD_DIM = 32
comptime NUM_Q = 1
comptime GQA_RATIO = 1
comptime KV_STRIDE = HEAD_DIM
comptime WINDOW = 16
comptime CACHE_SIZE = 16
comptime PSTRIDE = flash_partial_stride[NUM_Q, HEAD_DIM]()


@fieldwise_init
struct TestPool(BurstThreadPool):
    var capacity: Int
    var timestamp: Int

    def get_capacity(self) -> Int:
        return self.capacity

    def dispatch[K: BurstKernel, origin: MutOrigin](
        mut self, kernels: Span[K, origin], num_jobs: Int
    ):
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


def test_sliding_decode_zeros_first_accumulator():
    var q = alloc[BFloat16](NUM_Q * HEAD_DIM).as_any_origin()
    var k = alloc[BFloat16](CACHE_SIZE * KV_STRIDE).as_any_origin()
    var v = alloc[BFloat16](CACHE_SIZE * KV_STRIDE).as_any_origin()
    var out = alloc[BFloat16](NUM_Q * HEAD_DIM).as_any_origin()
    var partials = alloc[Float32](PSTRIDE).as_any_origin()

    for i in range(NUM_Q * HEAD_DIM):
        q[i] = BFloat16(0.0)
        out[i] = BFloat16(0.0)
    for i in range(CACHE_SIZE * KV_STRIDE):
        k[i] = BFloat16(0.0)
        v[i] = BFloat16(0.0)
    for i in range(PSTRIDE):
        partials[i] = Float32(1.0e38)

    var bases = ArenaBases[1].uninitialized()
    bases[0] = 0
    var pools = HeapMoveArray[TestPool](1)
    pools.push(TestPool(1, 0))

    dispatch_sliding_attention[
        head_dim=HEAD_DIM,
        num_q=NUM_Q,
        gqa_ratio=GQA_RATIO,
        kv_stride=KV_STRIDE,
        window=WINDOW,
        cache_size=CACHE_SIZE,
        partial_stride=PSTRIDE,
        tp=1,
    ](
        Binding[BFloat16, 1](q, bases),
        Binding[BFloat16, 1](k, bases),
        Binding[BFloat16, 1](v, bases),
        Binding[BFloat16, 1](out, bases),
        Binding[Float32, 1](partials, bases),
        0,
        1,
        pools,
    )

    for i in range(NUM_Q * HEAD_DIM):
        check(
            Float32(out[i]) == Float32(0.0),
            "poisoned first accumulator leaked into output at " + String(i),
        )

    q.free()
    k.free()
    v.free()
    out.free()
    partials.free()


def main():
    test_sliding_decode_zeros_first_accumulator()
    print("attention accumulator tests passed")
