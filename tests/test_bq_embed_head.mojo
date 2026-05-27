from std.collections import InlineArray
from std.memory import Span, UnsafePointer, alloc
from std.os import abort

from simd_math.ops import sqrt
from threading.threading_traits import BurstKernel, BurstThreadPool

from kernels.helpers import Binding, ArenaBases
from kernels.rmsnorm import rms_reduce_row
from kernels.gemv import softcap_value
from butterquant import (
    rotate_and_quant,
    ButterquantWeight, ButterquantActivation,
)
from butterquant_kernels import (
    dispatch_bq_embed_lookup, dispatch_bq_head_prep, dispatch_bq_head_gemv,
)
from quant.recipe import PerBlockQuant, NoGamma, SingleSided, NoColsum, RowMajor


comptime HIDDEN = 2816
comptime BLOCK = 128
comptime NB = HIDDEN // BLOCK
comptime VOCAB = 512
comptime CAP = 30.0
comptime SQRT_N = sqrt[DType.float32, 1](HIDDEN)
comptime N_EPS = HIDDEN * 1e-6
comptime EMBED_SCALE = Float64(
    sqrt[DType.float32, 1](HIDDEN).cast[DType.bfloat16]().cast[DType.float32]())
comptime QUANT = PerBlockQuant(BLOCK, NoGamma(), SingleSided(), NoColsum(), RowMajor())

comptime F32ExtPtr = UnsafePointer[Float32, MutAnyOrigin]
comptime I8ExtPtr = UnsafePointer[Int8, MutAnyOrigin]
comptime BF16ExtPtr = UnsafePointer[BFloat16, MutAnyOrigin]


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


def rnd(seed: Int, i: Int) -> Float32:
    var x = UInt64(seed) * 2654435761 + UInt64(i) * 40503 + 12345
    x = x ^ (x >> 13)
    x = x * 1274126177
    x = x ^ (x >> 16)
    return Float32(Int(x & 0xFFFF)) / 32768.0 - Float32(1.0)


def cosine(a: F32ExtPtr, b: F32ExtPtr, n: Int) -> Float32:
    var dot = Float32(0)
    var na = Float32(0)
    var nb = Float32(0)
    for i in range(n):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    return dot / max(sqrt[DType.float32, 1](na * nb), Float32(1e-30))


def rel_l2(a: F32ExtPtr, b: F32ExtPtr, n: Int) -> Float32:
    var num = Float32(0)
    var den = Float32(0)
    for i in range(n):
        var d = a[i] - b[i]
        num += d * d
        den += b[i] * b[i]
    return sqrt[DType.float32, 1](num / max(den, Float32(1e-30)))


def max_abs(a: F32ExtPtr, b: F32ExtPtr, n: Int) -> Float32:
    var m = Float32(0)
    for i in range(n):
        m = max(m, abs(a[i] - b[i]))
    return m


def argmax(a: F32ExtPtr, n: Int) -> Int:
    var best = 0
    var bestv = a[0]
    for i in range(1, n):
        if a[i] > bestv:
            bestv = a[i]
            best = i
    return best


def main():
    var w_orig = alloc[Float32](VOCAB * HIDDEN).as_any_origin()
    var work = alloc[Float32](VOCAB * HIDDEN).as_any_origin()
    var wi8 = alloc[Int8](VOCAB * HIDDEN).as_any_origin()
    var sw = alloc[Float32](VOCAB * NB).as_any_origin()

    for n in range(VOCAB):
        for k in range(HIDDEN):
            var v = rnd(n * 3 + 1, k) * Float32(0.08)
            w_orig[n * HIDDEN + k] = v
            work[n * HIDDEN + k] = v

    rotate_and_quant[True](BLOCK, work, wi8, sw, VOCAB, HIDDEN)

    var bases = ArenaBases[1].uninitialized()
    bases[0] = 0
    var pools = List[TestPool](capacity=1)
    pools.append(TestPool(4, 0))

    var bqw = ButterquantWeight[QUANT, VOCAB, HIDDEN, 1](
        Binding[Int8, 1](wi8, bases),
        Binding[Float32, 1](sw, bases),
        Binding[Float32, 1](sw, bases))

    # --- lookup: reconstruct rows from the shared int8 head encoding ---
    var tok_buf = List[Int32]()
    tok_buf.append(0)
    tok_buf.append(1)
    tok_buf.append(137)
    tok_buf.append(VOCAB - 1)
    var n_tok = len(tok_buf)

    var dst = alloc[BFloat16](n_tok * HIDDEN).as_any_origin()
    dispatch_bq_embed_lookup[scale=EMBED_SCALE, shard_rows=VOCAB](
        Span[Int32, origin_of(tok_buf)](ptr=tok_buf.unsafe_ptr(), length=n_tok),
        bqw,
        Binding[BFloat16, 1](dst, bases),
        n_tok, pools)

    var row_ref = alloc[Float32](HIDDEN).as_any_origin()
    var row_got = alloc[Float32](HIDDEN).as_any_origin()
    for ti in range(n_tok):
        var t = Int(tok_buf[ti])
        for k in range(HIDDEN):
            row_ref[k] = w_orig[t * HIDDEN + k] * Float32(EMBED_SCALE)
            row_got[k] = dst[ti * HIDDEN + k].cast[DType.float32]()
        var cos = cosine(row_got, row_ref, HIDDEN)
        var rl = rel_l2(row_got, row_ref, HIDDEN)
        print(t"lookup token {t}: cosine={cos} rel_l2={rl}")
        check(cos > 0.999, String(t"lookup cosine too low for token {t}: {cos}"))
        check(rl < 0.03, String(t"lookup rel_l2 too high for token {t}: {rl}"))

    # --- head: embed-domain activation -> int8 unembed vs bf16 ---
    var h = alloc[BFloat16](HIDDEN).as_any_origin()
    var gamma = alloc[BFloat16](HIDDEN).as_any_origin()
    for k in range(HIDDEN):
        h[k] = BFloat16(rnd(7, k) * Float32(0.1))
        gamma[k] = BFloat16(Float32(1.0) + rnd(99, k) * Float32(0.05))

    var sum_sq = rms_reduce_row[HIDDEN](h)
    var inv_rms = SQRT_N / sqrt[DType.float32, 1](sum_sq + N_EPS)
    var x_ref = alloc[Float32](HIDDEN).as_any_origin()
    for k in range(HIDDEN):
        x_ref[k] = h[k].cast[DType.float32]() * inv_rms * gamma[k].cast[DType.float32]()

    var logit_ref = alloc[Float32](VOCAB).as_any_origin()
    for n in range(VOCAB):
        var raw = Float32(0)
        for k in range(HIDDEN):
            raw += x_ref[k] * w_orig[n * HIDDEN + k]
        logit_ref[n] = softcap_value[CAP](SIMD[DType.float32, 1](raw))

    var x_i8 = alloc[Int8](HIDDEN).as_any_origin()
    var sa = alloc[Float32](NB).as_any_origin()
    var bq_act = ButterquantActivation[1](
        Binding[Int8, 1](x_i8, bases),
        Binding[Float32, 1](sa, bases))
    dispatch_bq_head_prep[
        hidden=HIDDEN, block=BLOCK, sqrt_n=SQRT_N, n_eps=N_EPS,
    ](
        Binding[BFloat16, 1](h, bases),
        Binding[BFloat16, 1](gamma, bases),
        bq_act)

    var logits_bf16 = alloc[BFloat16](VOCAB).as_any_origin()
    dispatch_bq_head_gemv[cap=CAP](
        bq_act, bqw, Binding[BFloat16, 1](logits_bf16, bases), pools)

    var logit_bq = alloc[Float32](VOCAB).as_any_origin()
    for n in range(VOCAB):
        logit_bq[n] = logits_bf16[n].cast[DType.float32]()

    # independent dequant-identity golden (scalar int accumulation)
    var logit_golden = alloc[Float32](VOCAB).as_any_origin()
    for n in range(VOCAB):
        var acc = Float32(0)
        for b in range(NB):
            var off = b * BLOCK
            var r = 0
            for j in range(BLOCK):
                r += Int(x_i8[off + j]) * Int(wi8[n * HIDDEN + off + j])
            acc += Float32(r) * (sa[b] / Float32(127.0)) * sw[n * NB + b]
        logit_golden[n] = softcap_value[CAP](SIMD[DType.float32, 1](acc))

    var gold_err = max_abs(logit_bq, logit_golden, VOCAB)
    print(t"head golden dequant-identity max_abs={gold_err}")
    check(gold_err < 0.02, String(t"head kernel deviates from identity: {gold_err}"))

    var head_cos = cosine(logit_bq, logit_ref, VOCAB)
    var head_rl = rel_l2(logit_bq, logit_ref, VOCAB)
    var am_ref = argmax(logit_ref, VOCAB)
    var am_bq = argmax(logit_bq, VOCAB)
    print(t"head vs bf16: cosine={head_cos} rel_l2={head_rl} "
          t"argmax ref={am_ref} bq={am_bq}")
    check(head_cos > 0.99, String(t"head cosine too low: {head_cos}"))
    check(am_ref == am_bq, String(t"head argmax mismatch: {am_ref} vs {am_bq}"))

    print("bq embed/head kernel tests passed")
