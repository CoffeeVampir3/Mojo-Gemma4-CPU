from std.memory import Span, UnsafePointer, alloc
from std.os import abort

from kernels.attention_dispatch_kernels import dispatch_full_attention
from kernels.attention_ops import flash_partial_stride
from kernels.helpers import ArenaBases, Binding
from threading.threading_traits import BurstKernel, BurstThreadPool


comptime HEAD_DIM = 64
comptime NUM_Q = 4
comptime NUM_KV = 2
comptime GQA = NUM_Q // NUM_KV
comptime KV_STRIDE = NUM_KV * HEAD_DIM
comptime SEQ_LEN = 17
comptime WORKERS = 4
comptime PSTRIDE = flash_partial_stride[NUM_Q, HEAD_DIM]()

comptime BF16Ptr = UnsafePointer[BFloat16, MutExternalOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutExternalOrigin]


@fieldwise_init
struct TestPool(BurstThreadPool):
    var capacity: Int
    var timestamp: Int

    def get_capacity(self) -> Int:
        return self.capacity

    def dispatch[K: BurstKernel, origin: MutOrigin](
        mut self, kernels: Span[K, origin], num_jobs: Int,
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


def bases_from_ptrs[tp: Int](ptrs: UnsafePointer[BF16Ptr, MutExternalOrigin]) -> ArenaBases[tp]:
    var bases = ArenaBases[tp].uninitialized()
    for r in range(tp):
        bases[r] = Int(ptrs[r])
    return bases


def fill_q(ptr: BF16Ptr):
    for i in range(NUM_Q * HEAD_DIM):
        ptr[i] = BFloat16(Float32((i * 17 + 3) % 29 - 14) * Float32(0.03125))


def fill_kv_row(k: BF16Ptr, v: BF16Ptr, pos: Int):
    for i in range(KV_STRIDE):
        k[pos * KV_STRIDE + i] = BFloat16(
            Float32((pos * 13 + i * 7) % 31 - 15) * Float32(0.025)
        )
        v[pos * KV_STRIDE + i] = BFloat16(
            Float32((pos * 11 + i * 5 + 1) % 37 - 18) * Float32(0.02)
        )


def fill_full_kv(k: BF16Ptr, v: BF16Ptr):
    for pos in range(SEQ_LEN):
        fill_kv_row(k, v, pos)


def fill_sharded_kv[tp: Int](k_ref: BF16Ptr, v_ref: BF16Ptr, k: Binding[BFloat16, tp], v: Binding[BFloat16, tp]):
    for pos in range(SEQ_LEN):
        var rank = pos % tp
        var local = pos // tp
        for i in range(KV_STRIDE):
            k[rank][local * KV_STRIDE + i] = k_ref[pos * KV_STRIDE + i]
            v[rank][local * KV_STRIDE + i] = v_ref[pos * KV_STRIDE + i]


def run_decode_reference(mut pools: List[TestPool], q: BF16Ptr, k: BF16Ptr, v: BF16Ptr, output: BF16Ptr, partials: F32Ptr):
    var ptrs = alloc[BF16Ptr](1)
    ptrs[0] = q
    var bases = bases_from_ptrs[1](ptrs)
    dispatch_full_attention[
        head_dim=HEAD_DIM, num_q=NUM_Q, local_num_q=NUM_Q,
        gqa_ratio=GQA, kv_stride=KV_STRIDE, partial_stride=PSTRIDE, tp=1,
        max_worker_count=WORKERS,
    ](
        Binding[BFloat16, 1](q, bases),
        Binding[BFloat16, 1](k, bases),
        Binding[BFloat16, 1](v, bases),
        Binding[BFloat16, 1](output, bases),
        Binding[Float32, 1](partials, bases),
        SEQ_LEN - 1, 1, pools,
    )
    ptrs.free()


def run_decode_tp4(mut pools: List[TestPool], q_ptrs: UnsafePointer[BF16Ptr, MutExternalOrigin], k_ptrs: UnsafePointer[BF16Ptr, MutExternalOrigin], v_ptrs: UnsafePointer[BF16Ptr, MutExternalOrigin], out_ptrs: UnsafePointer[BF16Ptr, MutExternalOrigin], partial_ptrs: UnsafePointer[F32Ptr, MutExternalOrigin]):
    var bases = bases_from_ptrs[4](q_ptrs)
    var pbases = ArenaBases[4].uninitialized()
    for r in range(4):
        pbases[r] = Int(partial_ptrs[r])
    dispatch_full_attention[
        head_dim=HEAD_DIM, num_q=NUM_Q, local_num_q=NUM_Q // 4,
        gqa_ratio=GQA, kv_stride=KV_STRIDE, partial_stride=PSTRIDE, tp=4,
        max_worker_count=WORKERS,
    ](
        Binding[BFloat16, 4](q_ptrs[0], bases),
        Binding[BFloat16, 4](k_ptrs[0], bases),
        Binding[BFloat16, 4](v_ptrs[0], bases),
        Binding[BFloat16, 4](out_ptrs[0], bases),
        Binding[Float32, 4](partial_ptrs[0], pbases),
        SEQ_LEN - 1, 1, pools,
    )


def main():
    var pools1 = List[TestPool](capacity=1)
    pools1.append(TestPool(WORKERS, 0))
    var pools4 = List[TestPool](capacity=4)
    for _ in range(4):
        pools4.append(TestPool(WORKERS, 0))

    var q_ref = alloc[BFloat16](NUM_Q * HEAD_DIM)
    var k_ref = alloc[BFloat16](SEQ_LEN * KV_STRIDE)
    var v_ref = alloc[BFloat16](SEQ_LEN * KV_STRIDE)
    var out_ref = alloc[BFloat16](NUM_Q * HEAD_DIM)
    var partial_ref = alloc[Float32](WORKERS * PSTRIDE)
    fill_q(q_ref)
    fill_full_kv(k_ref, v_ref)
    run_decode_reference(pools1, q_ref, k_ref, v_ref, out_ref, partial_ref)

    var q_ptrs = alloc[BF16Ptr](4)
    var k_ptrs = alloc[BF16Ptr](4)
    var v_ptrs = alloc[BF16Ptr](4)
    var out_ptrs = alloc[BF16Ptr](4)
    var partial_ptrs = alloc[F32Ptr](4)
    for r in range(4):
        q_ptrs[r] = alloc[BFloat16](NUM_Q * HEAD_DIM)
        k_ptrs[r] = alloc[BFloat16](SEQ_LEN * KV_STRIDE)
        v_ptrs[r] = alloc[BFloat16](SEQ_LEN * KV_STRIDE)
        out_ptrs[r] = alloc[BFloat16]((NUM_Q // 4) * HEAD_DIM)
        partial_ptrs[r] = alloc[Float32](WORKERS * PSTRIDE)
        fill_q(q_ptrs[r])
    var bases4 = bases_from_ptrs[4](q_ptrs)
    fill_sharded_kv[4](
        k_ref, v_ref,
        Binding[BFloat16, 4](k_ptrs[0], bases4),
        Binding[BFloat16, 4](v_ptrs[0], bases4),
    )
    run_decode_tp4(pools4, q_ptrs, k_ptrs, v_ptrs, out_ptrs, partial_ptrs)

    for h in range(NUM_Q):
        var rank = h // (NUM_Q // 4)
        var local_h = h % (NUM_Q // 4)
        for j in range(HEAD_DIM):
            var got = Float32(out_ptrs[rank][local_h * HEAD_DIM + j])
            var expected = Float32(out_ref[h * HEAD_DIM + j])
            check(abs(got - expected) <= Float32(0.015625), String(t"decode h={h} j={j} expected={expected} got={got}"))

    print("full attention correctness tests passed")
