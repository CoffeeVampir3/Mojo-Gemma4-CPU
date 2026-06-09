from std.memory import UnsafePointer, alloc
from std.sys.info import simd_width_of

from simd_math.ops import exp_f32, log_f32, tanh_f32
from simd_math.sampling_rng import rng_counter, gumbel_noise


comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime SIMDW = simd_width_of[DType.float32]()
comptime NEG_INF = Float32(-1.0e30)
comptime MAX_TOPN = 64


@always_inline
def expf(x: Float32) -> Float32:
    return exp_f32[1](x)


@always_inline
def logf(x: Float32) -> Float32:
    return log_f32[1](x)


@always_inline
def tanhf(x: Float32) -> Float32:
    return tanh_f32[1](x)


@always_inline
def fabs(x: Float32) -> Float32:
    return -x if x < 0.0 else x


@always_inline
def gumbel(seed: UInt64, row: Int, idx: Int) -> Float32:
    return gumbel_noise(seed, rng_counter(row, idx))


@always_inline
def apply_softcap(x: Float32, cap: Float32) -> Float32:
    if cap <= 0.0:
        return x
    return tanhf(x / cap) * cap


@always_inline
def dotf(a: F32Ptr, b: F32Ptr, n: Int) -> Float32:
    var acc = SIMD[DType.float32, SIMDW](0.0)
    for j in range(0, n, SIMDW):
        acc = (a + j).load[width=SIMDW]().fma((b + j).load[width=SIMDW](), acc)
    return acc.reduce_add()


@fieldwise_init
struct SamplingParams(Copyable, Movable):
    """Per-step sampling metadata. Carries only scalars and a bounded
    materialization width `n_keep`; nothing here scales with the vocabulary, so
    the whole spec lives in registers/stack regardless of model size."""
    var temperature: Float32
    var min_p: Float32
    var softcap: Float32
    var n_keep: Int
    var seed: UInt64


struct RowAccum(Copyable, Movable):
    """The fused-epilogue working set for one output row. Holds, in O(n_keep)
    space: a descending top-N of raw logits with their global indices, an
    online log-sum-exp `(lse_max, lse_sum)` for the exact normalizer, and a
    running Gumbel-max `(samp_score, samp_idx)` for the full-support sample.
    Every field merges associatively, which is what lets tiles and shards be
    combined into the monolithic result."""
    var n: Int
    var topn_val: InlineArray[Float32, MAX_TOPN]
    var topn_idx: InlineArray[Int32, MAX_TOPN]
    var lse_max: Float32
    var lse_sum: Float32
    var samp_score: Float32
    var samp_idx: Int32

    def __init__(out self):
        self.n = 0
        self.topn_val = InlineArray[Float32, MAX_TOPN](fill=NEG_INF)
        self.topn_idx = InlineArray[Int32, MAX_TOPN](fill=Int32(-1))
        self.lse_max = NEG_INF
        self.lse_sum = 0.0
        self.samp_score = NEG_INF
        self.samp_idx = Int32(-1)

    @always_inline
    def topn_insert(mut self, y: Float32, idx: Int, cap: Int):
        if self.n < cap:
            var p = self.n
            while p > 0 and self.topn_val[p - 1] < y:
                self.topn_val[p] = self.topn_val[p - 1]
                self.topn_idx[p] = self.topn_idx[p - 1]
                p -= 1
            self.topn_val[p] = y
            self.topn_idx[p] = Int32(idx)
            self.n += 1
        elif y > self.topn_val[self.n - 1]:
            var p = self.n - 1
            while p > 0 and self.topn_val[p - 1] < y:
                self.topn_val[p] = self.topn_val[p - 1]
                self.topn_idx[p] = self.topn_idx[p - 1]
                p -= 1
            self.topn_val[p] = y
            self.topn_idx[p] = Int32(idx)

    @always_inline
    def lse_push(mut self, yt: Float32):
        if self.lse_sum <= 0.0:
            self.lse_max = yt
            self.lse_sum = 1.0
        elif yt > self.lse_max:
            self.lse_sum = self.lse_sum * expf(self.lse_max - yt) + 1.0
            self.lse_max = yt
        else:
            self.lse_sum = self.lse_sum + expf(yt - self.lse_max)

    @always_inline
    def absorb(mut self, y: Float32, idx: Int, inv_t: Float32, g: Float32, cap: Int):
        var yt = y * inv_t
        self.lse_push(yt)
        var score = yt + g
        if score > self.samp_score:
            self.samp_score = score
            self.samp_idx = Int32(idx)
        self.topn_insert(y, idx, cap)

    def merge(mut self, read other: Self, cap: Int):
        if other.lse_sum > 0.0:
            if self.lse_sum <= 0.0:
                self.lse_max = other.lse_max
                self.lse_sum = other.lse_sum
            elif other.lse_max > self.lse_max:
                self.lse_sum = self.lse_sum * expf(
                    self.lse_max - other.lse_max) + other.lse_sum
                self.lse_max = other.lse_max
            else:
                self.lse_sum = self.lse_sum + other.lse_sum * expf(
                    other.lse_max - self.lse_max)
        if other.samp_score > self.samp_score:
            self.samp_score = other.samp_score
            self.samp_idx = other.samp_idx
        for k in range(other.n):
            self.topn_insert(other.topn_val[k], Int(other.topn_idx[k]), cap)

    @always_inline
    def logz(self) -> Float32:
        return self.lse_max + logf(self.lse_sum)


def fused_partitioned(
    h_row: F32Ptr, w: F32Ptr, vocab: Int, dim: Int, row: Int,
    read params: SamplingParams, num_parts: Int,
) -> RowAccum:
    """Run the fused epilogue over `num_parts` contiguous vocabulary
    partitions, each accumulated independently then merged. `num_parts == 1` is
    the monolithic single pass; larger values model on-rank tiling and, at
    coarse granularity, the tensor-parallel vocabulary shards. The result must
    be identical across all partitionings."""
    var inv_t = 1.0 / params.temperature
    var acc = RowAccum()
    var part_len = (vocab + num_parts - 1) // num_parts
    for p in range(num_parts):
        var start = p * part_len
        if start >= vocab:
            break
        var end = min(start + part_len, vocab)
        var sub = RowAccum()
        for i in range(start, end):
            var y = apply_softcap(dotf(h_row, w + i * dim, dim), params.softcap)
            sub.absorb(y, i, inv_t, gumbel(params.seed, row, i), params.n_keep)
        acc.merge(sub, params.n_keep)
    return acc^


def materialize_logits(
    h_row: F32Ptr, w: F32Ptr, vocab: Int, dim: Int, cap: Float32, out_ys: F32Ptr,
):
    for i in range(vocab):
        out_ys[i] = apply_softcap(dotf(h_row, w + i * dim, dim), cap)


def ref_logz(ys: F32Ptr, vocab: Int, inv_t: Float32) -> Float32:
    var m = NEG_INF
    for i in range(vocab):
        var yt = ys[i] * inv_t
        if yt > m:
            m = yt
    var s = Float32(0.0)
    for i in range(vocab):
        s += expf(ys[i] * inv_t - m)
    return m + logf(s)


def ref_sample(ys: F32Ptr, vocab: Int, inv_t: Float32, seed: UInt64, row: Int) -> Int:
    var best = NEG_INF
    var bi = -1
    for i in range(vocab):
        var sc = ys[i] * inv_t + gumbel(seed, row, i)
        if sc > best:
            best = sc
            bi = i
    return bi


def ref_topn_idx(ys: F32Ptr, vocab: Int, n: Int) -> List[Int]:
    var taken = alloc[Float32](vocab).as_any_origin()
    for i in range(vocab):
        taken[i] = ys[i]
    var out = List[Int](capacity=n)
    for _ in range(n):
        var best = NEG_INF
        var bi = -1
        for i in range(vocab):
            if taken[i] > best:
                best = taken[i]
                bi = i
        out.append(bi)
        taken[bi] = NEG_INF
    taken.free()
    return out^


def ref_minp_count(ys: F32Ptr, vocab: Int, temperature: Float32, min_p: Float32) -> Int:
    var y_max = NEG_INF
    for i in range(vocab):
        if ys[i] > y_max:
            y_max = ys[i]
    var thr = y_max + temperature * logf(min_p)
    var count = 0
    for i in range(vocab):
        if ys[i] >= thr:
            count += 1
    return count


def topn_minp_count(read acc: RowAccum, temperature: Float32, min_p: Float32) -> Int:
    var thr = acc.topn_val[0] + temperature * logf(min_p)
    var count = 0
    for k in range(acc.n):
        if acc.topn_val[k] >= thr:
            count += 1
    return count


def topn_matches(read acc: RowAccum, read reference: List[Int], n: Int) -> Bool:
    if acc.n != n:
        return False
    for k in range(n):
        if Int(acc.topn_idx[k]) != reference[k]:
            return False
    return True


def minp_demo(read params: SamplingParams) -> Int:
    """Min-p on a controlled peaked distribution. Builds a decaying logit
    vector whose keep-set is small enough to fit inside the materialized top-N,
    then shows the threshold `l_max + T*log(min_p)` recovers exactly the same
    set the reference full scan does."""
    comptime VP = 256
    var inv_t = 1.0 / params.temperature
    var logits = alloc[Float32](VP).as_any_origin()
    for i in range(VP):
        logits[i] = 8.0 - 0.4 * Float32(i) + Float32((i * 29) % 7 - 3) * 0.05

    var acc = RowAccum()
    for i in range(VP):
        acc.absorb(logits[i], i, inv_t, gumbel(params.seed, 99, i), params.n_keep)

    var ref_count = ref_minp_count(logits, VP, params.temperature, params.min_p)
    var topn_count = topn_minp_count(acc, params.temperature, params.min_p)
    var thr = acc.topn_val[0] + params.temperature * logf(params.min_p)
    var fits = ref_count <= params.n_keep
    var ok = fits and (ref_count == topn_count)
    print(t"  peaked dist (V={VP}): l_max={acc.topn_val[0]} thr={thr}  "
          t"keep-set={ref_count} (ref) vs {topn_count} (from top-N) : "
          t"{'PASS' if ok else 'FAIL'}")
    logits.free()
    return 0 if ok else 1


def empirical_test(read params: SamplingParams) -> Int:
    """Statistical check that the Gumbel-max draw reproduces the categorical
    distribution: sweep the RNG seed, tally argmax frequencies, compare to the
    closed-form softmax."""
    comptime EV = 32
    comptime ETRIALS = 200000
    var inv_t = 1.0 / params.temperature
    var ys = InlineArray[Float32, EV](fill=0.0)
    for i in range(EV):
        ys[i] = Float32((i * 13) % 7 - 3) * 0.5 + Float32(i % 5) * 0.3

    var m = NEG_INF
    for i in range(EV):
        if ys[i] * inv_t > m:
            m = ys[i] * inv_t
    var s = Float32(0.0)
    for i in range(EV):
        s += expf(ys[i] * inv_t - m)

    var counts = InlineArray[Int, EV](fill=0)
    for tr in range(ETRIALS):
        var best = NEG_INF
        var bi = 0
        var seed = params.seed + UInt64(tr) * 2654435761
        for i in range(EV):
            var sc = ys[i] * inv_t + gumbel(seed, 0, i)
            if sc > best:
                best = sc
                bi = i
        counts[bi] += 1

    var max_err = Float32(0.0)
    for i in range(EV):
        var p = expf(ys[i] * inv_t - m) / s
        var f = Float32(counts[i]) / Float32(ETRIALS)
        var e = fabs(f - p)
        if e > max_err:
            max_err = e
    var ok = max_err < 0.01
    print(t"  empirical max|freq - softmax| = {max_err}  over {ETRIALS} draws, "
          t"{EV} categories : {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    comptime B = 4
    comptime D = 256
    comptime V = 4096
    comptime TILE = 256
    comptime SHARDS = 4

    print("FlashSampling viability prototype (fused tile-streamed epilogue)")
    print(t"  B={B} D={D} V={V}  tile={TILE} ({V // TILE} tiles)  shards={SHARDS}")
    print(t"  n_keep={MAX_TOPN}  simd_width={SIMDW}")
    print("")

    var h = alloc[Float32](B * D).as_any_origin()
    var w = alloc[Float32](V * D).as_any_origin()
    for b in range(B):
        for d in range(D):
            h[b * D + d] = Float32((b * 7 + d) % 13 - 6) * 0.08
    for v in range(V):
        for d in range(D):
            w[v * D + d] = Float32((v * 3 + d * 5) % 17 - 8) * 0.05

    var params = SamplingParams(
        temperature=0.8, min_p=0.05, softcap=30.0, n_keep=MAX_TOPN, seed=0x1234ABCD)
    var inv_t = 1.0 / params.temperature

    var ys = alloc[Float32](V).as_any_origin()
    var fails = 0

    print("per-row equivalence: monolithic vs tiled vs sharded vs reference")
    for b in range(B):
        var h_row = h + b * D
        var mono = fused_partitioned(h_row, w, V, D, b, params, 1)
        var tiled = fused_partitioned(h_row, w, V, D, b, params, V // TILE)
        var sharded = fused_partitioned(h_row, w, V, D, b, params, SHARDS)

        materialize_logits(h_row, w, V, D, params.softcap, ys)
        var rlogz = ref_logz(ys, V, inv_t)
        var rsamp = ref_sample(ys, V, inv_t, params.seed, b)
        var rtopn = ref_topn_idx(ys, V, params.n_keep)

        var samp_ok = (Int(mono.samp_idx) == rsamp
            and Int(tiled.samp_idx) == rsamp and Int(sharded.samp_idx) == rsamp)
        var topn_ok = (topn_matches(mono, rtopn, params.n_keep)
            and topn_matches(tiled, rtopn, params.n_keep)
            and topn_matches(sharded, rtopn, params.n_keep))
        var dz_tile = fabs(tiled.logz() - rlogz)
        var dz_shard = fabs(sharded.logz() - rlogz)
        var dz_mono = fabs(mono.logz() - rlogz)
        var logz_ok = dz_tile < 1e-2 and dz_shard < 1e-2 and dz_mono < 1e-2

        if not (samp_ok and topn_ok and logz_ok):
            fails += 1
        var status = "PASS" if (samp_ok and topn_ok and logz_ok) else "FAIL"
        print(t"  row {b}: sample={rsamp} (idx-match {samp_ok})  "
              t"top{params.n_keep}-match {topn_ok}  "
              t"logZ={rlogz} dmax={max(dz_mono, max(dz_tile, dz_shard))}  : {status}")

    print("")
    print("min-p truncation (threshold = l_max + T*log(min_p), logZ-free):")
    fails += minp_demo(params)
    var flat = fused_partitioned(h + 0, w, V, D, 0, params, SHARDS)
    materialize_logits(h + 0, w, V, D, params.softcap, ys)
    var flat_ref = ref_minp_count(ys, V, params.temperature, params.min_p)
    print(t"  flat dist (V={V}): keep-set={flat_ref} exceeds top-N={params.n_keep} "
          t"-> boundary case (larger N or fused-exact masked sample)")

    print("")
    print("distributional correctness of the Gumbel-max sampler:")
    fails += empirical_test(params)

    print("")
    if fails == 0:
        print("RESULT: PASS -- fused top-N + logZ + sample is exact under "
              "tile/shard decomposition; min-p reduces to a logit threshold")
    else:
        print(t"RESULT: FAIL -- {fails} check(s) failed")

    h.free()
    w.free()
    ys.free()
