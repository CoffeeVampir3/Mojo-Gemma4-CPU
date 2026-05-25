from std.collections import List
from std.memory import UnsafePointer
from std.pathlib import Path
from std.time import perf_counter_ns

from numa import NumaArena, NumaTopology
from modeling.model_spec import WeightDesc, DEFAULT_ALIGNMENT
from modeling.loader import discover_shards, load_weights_from_descs
from todo_quant_math import fwht_row, quantize_i8_scalar
from simd_math.ops import sqrt


comptime CHECKPOINT = "checkpoints/gemma-4-26B-A4B"
comptime EMBED_NAME = "model.language_model.embed_tokens.weight"

comptime VOCAB = 262144
comptime HIDDEN = 2816
comptime FWHT_BLOCK = 128
comptime NB = HIDDEN // FWHT_BLOCK
comptime V_TEST = 16384
comptime QEPS = Float32(1e-10)

comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime I8Ptr = UnsafePointer[Int8, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]


def build_int8_encoding(
    wb: BF16Ptr, wi8: I8Ptr, sw: F32Ptr, cs: F32Ptr, rowbuf: F32Ptr,
):
    for n in range(V_TEST):
        var base = n * HIDDEN
        for k in range(HIDDEN):
            rowbuf[k] = wb[base + k].cast[DType.float32]()
        fwht_row[FWHT_BLOCK](rowbuf, HIDDEN)
        for b in range(NB):
            var off = b * FWHT_BLOCK
            var amax = QEPS
            for j in range(FWHT_BLOCK):
                amax = max(amax, abs(rowbuf[off + j]))
            var inv = Float32(127.0) / amax
            var csum = 0
            for j in range(FWHT_BLOCK):
                var q = quantize_i8_scalar(rowbuf[off + j], inv)
                wi8[base + off + j] = q
                csum += Int(q)
            sw[n * NB + b] = amax / Float32(127.0)
            cs[n * NB + b] = Float32(csum)


def lookup_ref(t: Int, wb: BF16Ptr, x_ref: F32Ptr, scale: Float32):
    var base = t * HIDDEN
    for k in range(HIDDEN):
        x_ref[k] = wb[base + k].cast[DType.float32]() * scale


def lookup_int8(t: Int, wi8: I8Ptr, sw: F32Ptr, x_q: F32Ptr, scale: Float32):
    var base = t * HIDDEN
    for b in range(NB):
        var sbar = sw[t * NB + b]
        var off = b * FWHT_BLOCK
        for j in range(FWHT_BLOCK):
            x_q[off + j] = Float32(Int(wi8[base + off + j])) * sbar
    fwht_row[FWHT_BLOCK](x_q, HIDDEN)
    for k in range(HIDDEN):
        x_q[k] = x_q[k] * scale


def unembed_ref(x_ref: F32Ptr, wb: BF16Ptr, logit: F32Ptr):
    for n in range(V_TEST):
        var base = n * HIDDEN
        var acc = Float32(0)
        for k in range(HIDDEN):
            acc += x_ref[k] * wb[base + k].cast[DType.float32]()
        logit[n] = acc


def unembed_int8(
    x_q: F32Ptr, wi8: I8Ptr, sw: F32Ptr, cs: F32Ptr,
    logit: F32Ptr, xbuf: F32Ptr, xi8: I8Ptr, sa: F32Ptr,
):
    for k in range(HIDDEN):
        xbuf[k] = x_q[k]
    fwht_row[FWHT_BLOCK](xbuf, HIDDEN)
    for b in range(NB):
        var off = b * FWHT_BLOCK
        var amax = QEPS
        for j in range(FWHT_BLOCK):
            amax = max(amax, abs(xbuf[off + j]))
        sa[b] = amax
        var inv = Float32(127.0) / amax
        for j in range(FWHT_BLOCK):
            xi8[off + j] = quantize_i8_scalar(xbuf[off + j], inv)

    for n in range(V_TEST):
        var base = n * HIDDEN
        var acc = Float32(0)
        for b in range(NB):
            var off = b * FWHT_BLOCK
            var r = 0
            for j in range(FWHT_BLOCK):
                var k = off + j
                r += (Int(xi8[k]) + 128) * Int(wi8[base + k])
            var deq = Float32(r) - Float32(128.0) * cs[n * NB + b]
            acc += deq * (sa[b] / Float32(127.0)) * sw[n * NB + b]
        logit[n] = acc


def argmax(v: F32Ptr) -> Int:
    var best = 0
    var bestv = v[0]
    for n in range(1, V_TEST):
        if v[n] > bestv:
            bestv = v[n]
            best = n
    return best


def topk_overlap[k: Int](a: F32Ptr, b: F32Ptr) -> Int:
    var idx_a = List[Int]()
    var idx_b = List[Int]()

    @parameter
    def fill(src: F32Ptr, mut out: List[Int]):
        var taken = List[Bool]()
        for _ in range(k):
            var best = -1
            var bestv = Float32(0)
            for n in range(V_TEST):
                var seen = False
                for q in range(len(out)):
                    if out[q] == n:
                        seen = True
                        break
                if seen:
                    continue
                if best < 0 or src[n] > bestv:
                    best = n
                    bestv = src[n]
            out.append(best)

    fill(a, idx_a)
    fill(b, idx_b)
    var hits = 0
    for i in range(len(idx_a)):
        for j in range(len(idx_b)):
            if idx_a[i] == idx_b[j]:
                hits += 1
                break
    return hits


def rel_l2(a: F32Ptr, b: F32Ptr, n: Int) -> Float32:
    var num = Float32(0)
    var den = Float32(0)
    for i in range(n):
        var d = a[i] - b[i]
        num += d * d
        den += b[i] * b[i]
    return sqrt[DType.float32, 1](num / max(den, QEPS))


def cosine(a: F32Ptr, b: F32Ptr, n: Int) -> Float32:
    var dot = Float32(0)
    var na = Float32(0)
    var nb = Float32(0)
    for i in range(n):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    return dot / max(sqrt[DType.float32, 1](na * nb), QEPS)


def max_abs_err(a: F32Ptr, b: F32Ptr, n: Int) -> Float32:
    var m = Float32(0)
    for i in range(n):
        m = max(m, abs(a[i] - b[i]))
    return m


def main():
    var topo = NumaTopology()
    var shards = discover_shards(Path(CHECKPOINT))
    if len(shards) == 0:
        print(t"prototype: no shards in {CHECKPOINT}")
        return

    var embed_elems = V_TEST * HIDDEN
    var arena_bytes = (
        embed_elems * 2          # bf16 weights
        + embed_elems            # int8 weights
        + V_TEST * NB * 4 * 2    # scales + colsums
        + HIDDEN * 4 * 4         # f32 scratch rows
        + V_TEST * 4 * 2         # logit arrays
        + HIDDEN + NB * 4        # int8 act + act scales
        + 64 * 1024 * 1024       # alignment slack
    )
    var arena = NumaArena[alignment=DEFAULT_ALIGNMENT](topo.node(0), arena_bytes)
    if not arena:
        print("prototype: arena alloc failed")
        return
    var base = Int(arena.base.value())

    var wb = arena.alloc[Scalar[DType.bfloat16]](embed_elems).value()
    var wi8 = arena.alloc[Int8](embed_elems).value()
    var sw = arena.alloc[Float32](V_TEST * NB).value()
    var cs = arena.alloc[Float32](V_TEST * NB).value()
    var rowbuf = arena.alloc[Float32](HIDDEN).value()
    var x_ref = arena.alloc[Float32](HIDDEN).value()
    var x_q = arena.alloc[Float32](HIDDEN).value()
    var xbuf = arena.alloc[Float32](HIDDEN).value()
    var xi8 = arena.alloc[Int8](HIDDEN).value()
    var sa = arena.alloc[Float32](NB).value()
    var logit_ref = arena.alloc[Float32](V_TEST).value()
    var logit_bq = arena.alloc[Float32](V_TEST).value()

    var desc = WeightDesc(
        name=EMBED_NAME,
        arena_offset=Int(wb.bitcast[UInt8]()) - base,
        dtype=DType.bfloat16,
        element_bytes=2,
        global_rows=VOCAB, global_cols=HIDDEN,
        local_cols=HIDDEN,
        data_rows=V_TEST, data_cols=HIDDEN,
        target_rank=0,
    )
    var descs = List[WeightDesc]()
    descs.append(desc^)
    var arena_bases = List[Int]()
    arena_bases.append(base)

    print(t"prototype: loading first {V_TEST} embed rows from {CHECKPOINT}")
    var load = load_weights_from_descs(descs, shards, arena_bases, topo)
    if not load:
        print("prototype: load failed")
        return

    var s = sqrt[DType.float32, 1](Float32(HIDDEN))
    var embed_scale = s.cast[DType.bfloat16]().cast[DType.float32]()
    print(t"prototype: embed_scale = {embed_scale} (sqrt({HIDDEN}))")

    var t_enc = perf_counter_ns()
    build_int8_encoding(wb, wi8, sw, cs, rowbuf)
    var enc_ms = (perf_counter_ns() - t_enc) / 1_000_000
    print(t"prototype: built shared int8 encoding "
          t"(W#, per-block S_w + colsum, fwht={FWHT_BLOCK}) in {enc_ms} ms")
    print("")

    var test_tokens = List[Int]()
    test_tokens.append(0)
    test_tokens.append(1)
    test_tokens.append(137)
    test_tokens.append(1024)
    test_tokens.append(8191)
    test_tokens.append(V_TEST - 1)

    print("=== lookup reconstruction (int8 dequant + inverse FWHT) vs bf16 ===")
    for ti in range(len(test_tokens)):
        var t = test_tokens[ti]
        lookup_ref(t, wb, x_ref, embed_scale)
        lookup_int8(t, wi8, sw, x_q, embed_scale)
        var rl = rel_l2(x_q, x_ref, HIDDEN)
        var mx = max_abs_err(x_q, x_ref, HIDDEN)
        var cos = cosine(x_q, x_ref, HIDDEN)
        print(t"  token {t}: rel_l2={rl}  max_abs={mx}  cosine={cos}")
    print("")

    print("=== embed -> unembed logits: bf16 vs full int8 path ===")
    var agree = 0
    for ti in range(len(test_tokens)):
        var t = test_tokens[ti]
        lookup_ref(t, wb, x_ref, embed_scale)
        lookup_int8(t, wi8, sw, x_q, embed_scale)

        unembed_ref(x_ref, wb, logit_ref)
        unembed_int8(x_q, wi8, sw, cs, logit_bq, xbuf, xi8, sa)

        var rl = rel_l2(logit_bq, logit_ref, V_TEST)
        var mx = max_abs_err(logit_bq, logit_ref, V_TEST)
        var cos = cosine(logit_bq, logit_ref, V_TEST)
        var am_ref = argmax(logit_ref)
        var am_bq = argmax(logit_bq)
        var ov = topk_overlap[5](logit_ref, logit_bq)
        var same = "yes" if am_ref == am_bq else "NO"
        if am_ref == am_bq:
            agree += 1
        print(t"  token {t}: rel_l2={rl}  max_abs={mx}  cosine={cos}")
        print(t"            argmax ref={am_ref} bq={am_bq} match={same} "
              t"(self={t})  top5_overlap={ov}/5")

    var nt = len(test_tokens)
    print("")
    print(t"prototype: argmax agreement {agree}/{nt}")
