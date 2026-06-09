from std.memory import UnsafePointer
from std.os import abort

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from simd_math.ops import exp_f32, log_f32
from simd_math.sampling_rng import gumbel_noise, rng_counter

from kernels.dot_products import dot_to_scalar
from simd_math.ops import softcap_value
from kernels.helpers import RankView, Binding
from kernels.flash_sample import (
    SamplingParams, SampleAccum, SampleOutcome, dispatch_flash_sample,
)
from kernels.profiling import Profiler


comptime HIDDEN = 2816
comptime VOCAB_PER_RANK = 1024
comptime CAP = 30.0
comptime MAX_WORKERS = 128
comptime N_MAX = 64
comptime N_KEEP = 32
comptime NUM_ROWS = 2
comptime TEMPERATURE = Float32(0.8)
comptime SEED = UInt64(0x1234ABCD)
comptime MIN_P = Float32(0.05)


@always_inline
def xval(d: Int) -> Float32:
    return Float32((d % 11) - 5) * 0.1


@always_inline
def wval(g: Int, d: Int, g_star: Int) -> Float32:
    if g == g_star:
        return xval(d) * 0.045
    if g == g_star - 1:
        return xval(d) * 0.042
    if g == g_star - 2:
        return xval(d) * 0.039
    return Float32(((g * 131 + d * 17) % 101) - 50) * 0.003


def alloc_all[T: AnyType](
    mut arenas: List[NumaArena[]], count: Int,
) -> UnsafePointer[T, MutAnyOrigin]:
    var first = UnsafePointer[T, MutAnyOrigin].unsafe_dangling()
    for r in range(len(arenas)):
        var got = arenas[r].alloc[T](count)
        if not got:
            abort("alloc failed")
        if r == 0:
            first = got.value()
    return first


def brute_force[o: ImmutOrigin, //](
    x: Binding[BFloat16, o], weight: Binding[BFloat16, o],
    degree: Int, inv_t: Float32,
) -> Tuple[Int, Float32, List[Int]]:
    var total = degree * VOCAB_PER_RANK
    var ys = List[Float32](capacity=total)
    for r in range(degree):
        var wr = weight[r]
        for local in range(VOCAB_PER_RANK):
            var dot = dot_to_scalar[HIDDEN](x[0], wr + local * HIDDEN)
            ys.append(softcap_value[CAP](SIMD[DType.float32, 1](dot)).cast[
                DType.bfloat16]().cast[DType.float32]())

    var token = -1
    var best = Float32(-1.0e30)
    for g in range(total):
        if ys[g] > best:
            best = ys[g]
            token = g

    var m = Float32(-1.0e30)
    for g in range(total):
        if ys[g] * inv_t > m:
            m = ys[g] * inv_t
    var s = Float32(0.0)
    for g in range(total):
        s += exp_f32[1](ys[g] * inv_t - m)
    var logz = m + log_f32[1](s)

    var topn = List[Int](capacity=N_KEEP)
    for _ in range(N_KEEP):
        var bv = Float32(-1.0e30)
        var bi = -1
        for g in range(total):
            if ys[g] > bv:
                bv = ys[g]
                bi = g
        topn.append(bi)
        ys[bi] = Float32(-1.0e30)
    return (token, logz, topn^)


def brute_force_minp[o: ImmutOrigin, //](
    x: Binding[BFloat16, o], weight: Binding[BFloat16, o],
    degree: Int, row: Int, inv_t: Float32, seed: UInt64,
    temperature: Float32, min_p: Float32,
) -> Tuple[Int, Int]:
    var total = degree * VOCAB_PER_RANK
    var ys = List[Float32](capacity=total)
    for r in range(degree):
        var wr = weight[r]
        for local in range(VOCAB_PER_RANK):
            var dot = dot_to_scalar[HIDDEN](x[0] + row * HIDDEN, wr + local * HIDDEN)
            ys.append(softcap_value[CAP](SIMD[DType.float32, 1](dot)).cast[
                DType.bfloat16]().cast[DType.float32]())

    var y_max = Float32(-1.0e30)
    for g in range(total):
        if ys[g] > y_max:
            y_max = ys[g]
    var thr = y_max + temperature * log_f32[1](min_p)

    var best_idx = -1
    var best_score = Float32(-1.0e30)
    var set_size = 0
    for g in range(total):
        if ys[g] >= thr:
            set_size += 1
            var score = ys[g] * inv_t + gumbel_noise(seed, rng_counter(row, g))
            if score > best_score:
                best_score = score
                best_idx = g
    return (best_idx, set_size)


def run_test[P: BurstThreadPool, //](
    read topo: NumaTopology,
    var pools: List[P],
):
    var degree = len(pools)
    var total_vocab = degree * VOCAB_PER_RANK
    var g_star = total_vocab // 2 + 3
    var inv_t = Float32(1.0) / TEMPERATURE

    var arenas = List[NumaArena[]](capacity=degree)
    for r in range(degree):
        arenas.append(NumaArena[](topo.node(r % len(topo)), 256 * 1024 * 1024))
        if not arenas[r]:
            abort("arena allocation failed")

    var x_ptr = alloc_all[BFloat16](arenas, NUM_ROWS * HIDDEN)
    var w_ptr = alloc_all[BFloat16](arenas, VOCAB_PER_RANK * HIDDEN)
    var acc_ptr = alloc_all[SampleAccum[N_MAX]](arenas, MAX_WORKERS * NUM_ROWS)
    var par_ptr = alloc_all[SamplingParams](arenas, NUM_ROWS)
    var out_ptr = alloc_all[SampleOutcome[N_MAX]](arenas, NUM_ROWS)

    var bases = List[Int](capacity=degree)
    for r in range(degree):
        bases.append(Int(arenas[r].base.value()))
    var view = RankView(Span(bases))
    var x = view.bind(x_ptr)
    var weight = view.bind(w_ptr)
    var accums = view.bind(acc_ptr)
    var sampling = view.bind(par_ptr)

    for r in range(degree):
        var xr = x[r]
        for j in range(NUM_ROWS):
            for d in range(HIDDEN):
                xr[j * HIDDEN + d] = BFloat16(xval(d))
        var wr = weight[r]
        for local in range(VOCAB_PER_RANK):
            var g = r * VOCAB_PER_RANK + local
            for d in range(HIDDEN):
                wr[local * HIDDEN + d] = BFloat16(wval(g, d, g_star))

    for r in range(degree):
        _ = arenas[r].prefault(0, arenas[r].used())

    var params = SamplingParams(TEMPERATURE, MIN_P, SEED, N_KEEP, True)
    for r in range(degree):
        for j in range(NUM_ROWS):
            (sampling[r] + j)[] = params
    var prof = Profiler[False]()
    dispatch_flash_sample[cols=HIDDEN, cap=CAP, n_max=N_MAX](
        x, weight, accums, sampling, out_ptr, NUM_ROWS, VOCAB_PER_RANK, pools, prof)

    var reference = brute_force(x, weight, degree, inv_t)
    var rtok = reference[0]
    var rlogz = reference[1]
    var rtopn = reference[2].copy()

    print(t"degree={degree} total_vocab={total_vocab} g_star={g_star} num_rows={NUM_ROWS}")
    print(t"  reference: token={rtok} logZ={rlogz} top1={rtopn[0]} top2={rtopn[1]}")

    var fails = 0
    for j in range(NUM_ROWS):
        var op = out_ptr + j
        var tok = Int(op[].token_id)
        var lz = op[].logz
        var tok_ok = tok == rtok and tok == g_star
        var lz_ok = (lz - rlogz if lz > rlogz else rlogz - lz) < 1e-2
        var topn_ok = op[].n == N_KEEP
        for k in range(N_KEEP):
            if Int(op[].topn_idx[k]) != rtopn[k]:
                topn_ok = False
        var ok = tok_ok and lz_ok and topn_ok
        if not ok:
            fails += 1
        print(t"  row {j}: token={tok} (match {tok_ok})  logZ={lz} (match {lz_ok})  "
              t"top{N_KEEP}-match {topn_ok}  : {'PASS' if ok else 'FAIL'}")

    var sp_sample = SamplingParams(TEMPERATURE, MIN_P, SEED, N_KEEP, False)
    for r in range(degree):
        for j in range(NUM_ROWS):
            (sampling[r] + j)[] = sp_sample
    dispatch_flash_sample[cols=HIDDEN, cap=CAP, n_max=N_MAX](
        x, weight, accums, sampling, out_ptr, NUM_ROWS, VOCAB_PER_RANK, pools, prof)

    for j in range(NUM_ROWS):
        var mp = brute_force_minp(x, weight, degree, j, inv_t, SEED, TEMPERATURE, MIN_P)
        var ref_tok = mp[0]
        var set_size = mp[1]
        var ktok = Int((out_ptr + j)[].token_id)
        var ok = ktok == ref_tok and set_size >= 2
        if not ok:
            fails += 1
        print(t"  min-p row {j}: token={ktok} ref={ref_tok} set={set_size} : "
              t"{'PASS' if ok else 'FAIL'}")

    print("  RESULT:", "PASS" if fails == 0 else "FAIL")

    for r in range(degree):
        _ = arenas[r].used()


def main():
    var topo = NumaTopology()
    print(t"flash_sample full sampler test  ({topo.num_nodes()} NUMA nodes)")

    @parameter
    def dispatch_test[P: BurstThreadPool, //](var selected_pools: List[P]):
        run_test(topo, selected_pools^)

    with_topological_rank_dispatch[
        dispatch=dispatch_test,
    ](topo, "mode: isolated (spin-only)", "mode: cold (spin-backoff)")
