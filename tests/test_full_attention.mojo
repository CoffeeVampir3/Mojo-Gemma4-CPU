from std.collections import InlineArray
from std.memory import Span, UnsafePointer, alloc
from std.os import abort
from std.sys.info import simd_width_of

from kernels.full_attention import dispatch_full_attention, FullAttentionKernel
from kernels.logsum_merge import dispatch_merge_context_flash_partials
from kernels.helpers import Binding, ArenaBases
from notstdcollections import HeapMoveArray
from threading.threading_traits import BurstKernel, BurstThreadPool


comptime TP = 4
comptime HEAD_DIM = simd_width_of[DType.float32]()
comptime GLOBAL_Q = 16
comptime LOCAL_Q = GLOBAL_Q // TP
comptime NUM_KV = 2
comptime GLOBAL_GQA = GLOBAL_Q // NUM_KV
comptime KV_STRIDE = NUM_KV * HEAD_DIM
comptime PSTRIDE = FullAttentionKernel[
    HEAD_DIM, GLOBAL_Q, GLOBAL_GQA, KV_STRIDE,
].PARTIAL_STRIDE


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


def rank_bases(stride_bytes: Int) -> ArenaBases[TP]:
    var bases = ArenaBases[TP].uninitialized()
    for r in range(TP):
        bases[r] = r * stride_bytes
    return bases


def run_case[
    q_origin: MutOrigin, k_origin: MutOrigin, v_origin: MutOrigin,
    out_origin: MutOrigin, partials_origin: MutOrigin,
](
    q: UnsafePointer[BFloat16, q_origin],
    k: UnsafePointer[BFloat16, k_origin],
    v: UnsafePointer[BFloat16, v_origin],
    output: UnsafePointer[BFloat16, out_origin],
    partials: UnsafePointer[Float32, partials_origin],
    valid: InlineArray[Int, TP],
    expected_kv0: Float32,
    expected_kv1: Float32,
    label: String,
):
    var pools = HeapMoveArray[TestPool](TP)
    for _ in range(TP):
        pools.push(TestPool(1, 0))

    var nws = dispatch_full_attention[
        head_dim=HEAD_DIM, num_q=GLOBAL_Q,
        gqa_ratio=GLOBAL_GQA, kv_stride=KV_STRIDE, tp=TP,
    ](
        Binding[BFloat16, TP](
            q.as_any_origin(), rank_bases(GLOBAL_Q * HEAD_DIM * 2)),
        Binding[BFloat16, TP](
            k.as_any_origin(), rank_bases(KV_STRIDE * 2)),
        Binding[BFloat16, TP](
            v.as_any_origin(), rank_bases(KV_STRIDE * 2)),
        Binding[Float32, TP](
            partials.as_any_origin(), rank_bases(PSTRIDE * 4)),
        valid, pools)

    dispatch_merge_context_flash_partials[
        head_dim=HEAD_DIM, num_q=GLOBAL_Q, local_num_q=LOCAL_Q, tp=TP,
    ](
        Binding[BFloat16, TP](
            output.as_any_origin(), rank_bases(LOCAL_Q * HEAD_DIM * 2)),
        Binding[Float32, TP](
            partials.as_any_origin(), rank_bases(PSTRIDE * 4)),
        nws, pools)

    for r in range(TP):
        var out_base = r * LOCAL_Q * HEAD_DIM
        for local_h in range(LOCAL_Q):
            var global_h = r * LOCAL_Q + local_h
            var expected = expected_kv0 if global_h < GLOBAL_GQA else expected_kv1
            for i in range(HEAD_DIM):
                var got = Float32(output[out_base + local_h * HEAD_DIM + i])
                check(got == expected,
                    label + ": rank " + String(r)
                    + " head " + String(local_h)
                    + " expected=" + String(expected) + " got=" + String(got))


def main():
    var q = alloc[BFloat16](TP * GLOBAL_Q * HEAD_DIM)
    var k = alloc[BFloat16](TP * KV_STRIDE)
    var v = alloc[BFloat16](TP * KV_STRIDE)
    var output = alloc[BFloat16](TP * LOCAL_Q * HEAD_DIM)
    var partials = alloc[Float32](TP * PSTRIDE)

    for r in range(TP):
        var q_base = r * GLOBAL_Q * HEAD_DIM
        for i in range(GLOBAL_Q * HEAD_DIM):
            q[q_base + i] = BFloat16(Float32(r * 10 + i % 7))

        var kv_base = r * KV_STRIDE
        for i in range(KV_STRIDE):
            k[kv_base + i] = BFloat16(0)
        for i in range(HEAD_DIM):
            v[kv_base + i] = BFloat16(Float32(10 + r))
            v[kv_base + HEAD_DIM + i] = BFloat16(Float32(50 + r))

    var valid_first = InlineArray[Int, TP](fill=0)
    valid_first[0] = 1
    run_case(
        q, k, v, output, partials, valid_first,
        10.0, 50.0, "single context owner")

    var valid_all = InlineArray[Int, TP](fill=1)
    run_case(
        q, k, v, output, partials, valid_all,
        11.5, 51.5, "all context owners")

    q.free()
    k.free()
    v.free()
    output.free()
    partials.free()
    print("full attention tests passed")
