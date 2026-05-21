from std.collections import InlineArray
from std.memory import Span, UnsafePointer, alloc
from std.os import abort

from kernels.helpers import RankBuffers
from kernels.reductions import (
    dispatch_allreduce,
)
from modeling.model_spec import BF16
from notstdcollections import HeapMoveArray
from threading.threading_traits import BurstKernel, BurstThreadPool


comptime ImmutExt = ImmutOrigin(MutExternalOrigin)
comptime BF16Ptr = UnsafePointer[BFloat16, MutExternalOrigin]


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


@always_inline
def hash_val(i: Int, seed: Int) -> Float32:
    var h = ((i * 7 + seed * 131) * 2654435761) & 0x7FFFFFFF
    return Float32(Int(h % 200) - 100) * Float32(0.1)


def hash_fill(ptr: BF16Ptr, n: Int, seed: Int):
    for i in range(n):
        ptr[i] = BFloat16(hash_val(i, seed))


@always_inline
def bf16_ulp_at(value: Float64) -> Float64:
    var a = abs(value)
    if a < Float64(1e-38):
        return Float64(1e-40)
    return a / Float64(128.0)


def fill_rank_bf16[tp: Int](ptrs: InlineArray[BF16Ptr, tp], n: Int):
    for r in range(tp):
        for i in range(n):
            ptrs[r][i] = BFloat16(Float32(r * 10 + i % 32))


def check_bf16(ptr: BF16Ptr, idx: Int, expected: Float32, msg: StringSlice):
    var got = Float32(ptr[idx])
    check(got == expected, String(t"{msg} [{idx}] expected={expected} got={got}"))


# ---- Exact correctness tests (logic bugs) ----

def test_allreduce_exact[tp: Int](n: Int, label: String, inline_max_bytes: Int):
    var pools = HeapMoveArray[TestPool](tp)
    for _ in range(tp):
        pools.push(TestPool(4, 0))
    var raw = InlineArray[BF16Ptr, tp](uninitialized=True)
    var out = InlineArray[BF16Ptr, tp](uninitialized=True)
    for r in range(tp):
        raw[r] = alloc[BFloat16](n)
        out[r] = alloc[BFloat16](n)
    fill_rank_bf16[tp](raw, n)
    var src = RankBuffers[DType.bfloat16, tp, ImmutExt](count=n)
    var dst = RankBuffers[DType.bfloat16, tp, MutExternalOrigin](count=n)
    for r in range(tp):
        src.ptrs[r] = raw[r].as_immutable()
        dst.ptrs[r] = out[r]
    dispatch_allreduce[BF16, tp](src, dst, pools, inline_max_bytes)
    for i in range(n):
        var expected = Float32(0)
        for r in range(tp):
            expected += Float32(r * 10 + i % 32)
        for r in range(tp):
            check_bf16(out[r], i, expected, String(t"{label} r{r}"))
    for r in range(tp):
        raw[r].free()
        out[r].free()


def run_exact_suite[tp: Int](n: Int, tag: StringSlice):
    var d = String(t"tp={tp} n={n} {tag}")
    test_allreduce_exact[tp](n, String(t"allreduce inline {d}"), 16384)
    test_allreduce_exact[tp](n, String(t"allreduce parallel {d}"), 0)


# ---- Accuracy tests (numerical error measurement) ----

def accuracy_allreduce[tp: Int](
    n: Int, label: String, hypothesis_ulp: Float64,
    inline_max_bytes: Int,
):
    var pools = HeapMoveArray[TestPool](tp)
    for _ in range(tp):
        pools.push(TestPool(4, 0))
    var raw = InlineArray[BF16Ptr, tp](uninitialized=True)
    var out = InlineArray[BF16Ptr, tp](uninitialized=True)
    for r in range(tp):
        raw[r] = alloc[BFloat16](n)
        out[r] = alloc[BFloat16](n)
        hash_fill(raw[r], n, r)
    var src = RankBuffers[DType.bfloat16, tp, ImmutExt](count=n)
    var dst = RankBuffers[DType.bfloat16, tp, MutExternalOrigin](count=n)
    for r in range(tp):
        src.ptrs[r] = raw[r].as_immutable()
        dst.ptrs[r] = out[r]

    dispatch_allreduce[BF16, tp](src, dst, pools, inline_max_bytes)

    var max_ulp = Float64(0)
    var sum_ulp = Float64(0)
    var max_abs = Float64(0)
    var total = 0
    for i in range(n):
        var reference = Float64(0)
        for r in range(tp):
            reference += Float64(Float32(raw[r][i]))
        for r in range(tp):
            var actual = Float64(Float32(out[r][i]))
            var err = abs(actual - reference)
            if err > max_abs:
                max_abs = err
            var ulp = bf16_ulp_at(reference)
            var ulp_err = err / ulp
            if ulp_err > max_ulp:
                max_ulp = ulp_err
            sum_ulp += ulp_err
            total += 1

    var mean_ulp = sum_ulp / Float64(total) if total > 0 else Float64(0)
    var accepted = max_ulp <= hypothesis_ulp
    print(label)
    print(t"  hypothesis: max error <= {hypothesis_ulp} ULP")
    print(t"  measured:   max_ulp={max_ulp}  mean_ulp={mean_ulp}  max_abs={max_abs}")
    print("  ", "ACCEPT" if accepted else "REJECT")
    if not accepted:
        abort(String(t"hypothesis rejected: {label}"))

    for r in range(tp):
        raw[r].free()
        out[r].free()


def accuracy_crosspath[tp: Int](n: Int, label: String):
    var pools_a = HeapMoveArray[TestPool](tp)
    var pools_b = HeapMoveArray[TestPool](tp)
    for _ in range(tp):
        pools_a.push(TestPool(4, 0))
        pools_b.push(TestPool(4, 0))
    var raw = InlineArray[BF16Ptr, tp](uninitialized=True)
    var out_inline = InlineArray[BF16Ptr, tp](uninitialized=True)
    var out_parallel = InlineArray[BF16Ptr, tp](uninitialized=True)
    for r in range(tp):
        raw[r] = alloc[BFloat16](n)
        out_inline[r] = alloc[BFloat16](n)
        out_parallel[r] = alloc[BFloat16](n)
        hash_fill(raw[r], n, r)

    var src = RankBuffers[DType.bfloat16, tp, ImmutExt](count=n)
    var dst_i = RankBuffers[DType.bfloat16, tp, MutExternalOrigin](count=n)
    var dst_p = RankBuffers[DType.bfloat16, tp, MutExternalOrigin](count=n)
    for r in range(tp):
        src.ptrs[r] = raw[r].as_immutable()
        dst_i.ptrs[r] = out_inline[r]
        dst_p.ptrs[r] = out_parallel[r]

    dispatch_allreduce[BF16, tp](src, dst_i, pools_a, 16384)
    dispatch_allreduce[BF16, tp](src, dst_p, pools_b, 0)

    var mismatches = 0
    for r in range(tp):
        for i in range(n):
            if Float32(out_inline[r][i]) != Float32(out_parallel[r][i]):
                mismatches += 1

    var total_compare = n * tp
    print(label)
    print("  hypothesis: inline == parallel (bitwise)")
    print(t"  measured:   mismatches={mismatches} / {total_compare}")
    print("  ", "ACCEPT" if mismatches == 0 else "REJECT")
    if mismatches > 0:
        abort(String(t"crosspath mismatch: {label}"))

    for r in range(tp):
        raw[r].free()
        out_inline[r].free()
        out_parallel[r].free()


def run_accuracy_suite[tp: Int](n: Int, tag: StringSlice):
    var d = String(t"tp={tp} n={n} {tag}")
    accuracy_allreduce[tp](n, String(t"allreduce bf16 inline {d}"), 1.0, 16384)
    accuracy_allreduce[tp](n, String(t"allreduce bf16 parallel {d}"), 1.0, 0)
    accuracy_crosspath[tp](n, String(t"allreduce crosspath {d}"))


def main():
    print("=== Exact correctness ===")
    run_exact_suite[tp=1](8, "trivial")
    run_exact_suite[tp=1](33, "odd")
    run_exact_suite[tp=2](8, "trivial")
    run_exact_suite[tp=2](33, "odd")
    run_exact_suite[tp=2](4096, "large")
    run_exact_suite[tp=4](8, "trivial")
    run_exact_suite[tp=4](33, "odd")
    run_exact_suite[tp=4](256, "medium")
    run_exact_suite[tp=4](4096, "large")
    run_exact_suite[tp=4](9999, "prime-ish")
    print("exact: ok\n")

    print("=== Numerical accuracy ===")
    run_accuracy_suite[tp=2](1024, "medium")
    run_accuracy_suite[tp=2](8192, "large")
    run_accuracy_suite[tp=4](1024, "medium")
    run_accuracy_suite[tp=4](8192, "large")
    run_accuracy_suite[tp=4](9999, "prime-ish")
    print("\nall tests passed")
