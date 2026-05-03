"""Exp variant benchmark: ns/element + rounded-bf16 ULP distance per variant.

Variants:
  - Mojo std.math.exp at width=1 (scalar reference)
  - exp_simd[N=2]   bf16-oriented current default
  - exp_simd[N=3]   f16-oriented
  - exp_simd[N=6]   f32-oriented
  - Schraudolph C=0           - pure bit trick, ~3 ops, NOT bf16-tenable
  - Schraudolph + quad corr   - bit trick + mantissa extraction + 2 FMAs
"""

from std.benchmark import run, keep, clobber_memory, Unit
from std.math import exp as math_exp
from std.memory import UnsafePointer
from std.memory.unsafe_pointer import alloc
from std.random import seed, random_float64
from std.sys.info import simd_width_of
from simd_math import exp_simd


comptime W = simd_width_of[DType.float32]()
comptime N = 1 << 14

comptime BENCH_MIN_RUNTIME_SECS = 0.1
comptime BENCH_MAX_RUNTIME_SECS = 0.5
comptime BENCH_WARMUP_ITERS = 1
comptime SCALAR_MAX_ITERS = 2048
comptime SIMD_MAX_ITERS = 32768
comptime PASSES_PER_ITER = 32
comptime TIMED_ELEMS_PER_ITER = N * PASSES_PER_ITER

comptime A_MAGIC  = Float32(12102203.16156148)        # 2^23 / ln 2
comptime BIAS_F   = Float32(127) * Float32(1 << 23)   # 127 * 2^23
comptime CLAMP_LO = Float32(-87.0)
comptime CLAMP_HI = Float32(88.7228)
comptime INV_TWO23 = Float32(1.0) / Float32(1 << 23)

# Quadratic correction (study/schraudolph.py, fit_quadratic_correction(C=0)).
comptime QC_A = Float32(0.9999971357998674)
comptime QC_B = Float32(-0.23410922365524586)
comptime QC_C = Float32(0.2354340427504884)


@always_inline
def exp_schraudolph(
    x: SIMD[DType.float32, W],
) -> SIMD[DType.float32, W]:
    var xc = x.clamp(
        SIMD[DType.float32, W](CLAMP_LO),
        SIMD[DType.float32, W](CLAMP_HI),
    )
    var i = (A_MAGIC * xc + BIAS_F).cast[DType.int32]()
    return SIMD[DType.float32, W](from_bits=i.cast[DType.uint32]())


@always_inline
def exp_schraudolph_quad(
    x: SIMD[DType.float32, W],
) -> SIMD[DType.float32, W]:
    var xc = x.clamp(
        SIMD[DType.float32, W](CLAMP_LO),
        SIMD[DType.float32, W](CLAMP_HI),
    )
    var i = (A_MAGIC * xc + BIAS_F).cast[DType.int32]()
    var u = i.cast[DType.uint32]()
    var k = SIMD[DType.float32, W](from_bits=u)
    var fbits = u & SIMD[DType.uint32, W](0x7FFFFF)
    var f = fbits.cast[DType.float32]() * INV_TWO23
    return k * (QC_A + f * (QC_B + f * QC_C))


@always_inline
def bf16_round(x: Float32) -> Float32:
    """RNE round f32 -> bf16, returned as f32 for comparison."""
    var u = x.to_bits()
    var lsb = (u >> 16) & 1
    var rounded = (u + 0x7FFF + lsb) & 0xFFFF0000
    return Float32(from_bits=rounded)


@always_inline
def bf16_ulp_at(x: Float32) -> Float64:
    if x == Float32(0.0):
        return Float64(2.0) ** Float64(-133)
    var exp_bits = abs(x).to_bits() & 0x7F800000
    if exp_bits == 0:
        return Float64(2.0) ** Float64(-133)
    var e = Int((exp_bits >> 23).cast[DType.int32]()) - 127
    return Float64(2.0) ** Float64(e - 7)


@always_inline
def ulp_step(got: Float32, x_in: Float32, mut max_ulp: Float64):
    var truth = math_exp(Float64(x_in))
    var got_bf = bf16_round(got)
    var truth_bf = bf16_round(Float32(truth))
    var diff = abs(Float64(got_bf) - Float64(truth_bf))
    var ulp = diff / bf16_ulp_at(truth_bf)
    if ulp > max_ulp:
        max_ulp = ulp


@always_inline
def bench_scalar_kernel[output_origin: MutOrigin](
    inputs: UnsafePointer[Float32, _],
    outputs: UnsafePointer[Float32, output_origin],
):
    for _ in range(PASSES_PER_ITER):
        for i in range(N):
            outputs[i] = math_exp(inputs[i])
    clobber_memory()
    keep(outputs)


@always_inline
def bench_exp_simd_kernel[
    polynomial_degree: Int,
    output_origin: MutOrigin,
](
    inputs: UnsafePointer[Float32, _],
    outputs: UnsafePointer[Float32, output_origin],
):
    for _ in range(PASSES_PER_ITER):
        var i = 0
        while i < N:
            var xv = (inputs + i).load[width=W]()
            (outputs + i).store(exp_simd[polynomial_degree, W](xv))
            i += W
    clobber_memory()
    keep(outputs)


@always_inline
def bench_schraudolph_kernel[output_origin: MutOrigin](
    inputs: UnsafePointer[Float32, _],
    outputs: UnsafePointer[Float32, output_origin],
):
    for _ in range(PASSES_PER_ITER):
        var i = 0
        while i < N:
            var xv = (inputs + i).load[width=W]()
            (outputs + i).store(exp_schraudolph(xv))
            i += W
    clobber_memory()
    keep(outputs)


@always_inline
def bench_schraudolph_quad_kernel[output_origin: MutOrigin](
    inputs: UnsafePointer[Float32, _],
    outputs: UnsafePointer[Float32, output_origin],
):
    for _ in range(PASSES_PER_ITER):
        var i = 0
        while i < N:
            var xv = (inputs + i).load[width=W]()
            (outputs + i).store(exp_schraudolph_quad(xv))
            i += W
    clobber_memory()
    keep(outputs)


def max_scalar_bf16_ulp(inputs: UnsafePointer[Float32, _]) -> Float64:
    var u = Float64(0.0)
    for i in range(N):
        ulp_step(math_exp(inputs[i]), inputs[i], u)
    return u


def max_exp_simd_bf16_ulp[
    polynomial_degree: Int,
](inputs: UnsafePointer[Float32, _]) -> Float64:
    var u = Float64(0.0)
    var i = 0
    while i < N:
        var xv = (inputs + i).load[width=W]()
        var yv = exp_simd[polynomial_degree, W](xv)
        for k in range(W):
            ulp_step(yv[k], xv[k], u)
        i += W
    return u


def max_schraudolph_bf16_ulp(inputs: UnsafePointer[Float32, _]) -> Float64:
    var u = Float64(0.0)
    var i = 0
    while i < N:
        var xv = (inputs + i).load[width=W]()
        var yv = exp_schraudolph(xv)
        for k in range(W):
            ulp_step(yv[k], xv[k], u)
        i += W
    return u


def max_schraudolph_quad_bf16_ulp(inputs: UnsafePointer[Float32, _]) -> Float64:
    var u = Float64(0.0)
    var i = 0
    while i < N:
        var xv = (inputs + i).load[width=W]()
        var yv = exp_schraudolph_quad(xv)
        for k in range(W):
            ulp_step(yv[k], xv[k], u)
        i += W
    return u


def report_row(name: String, ns_per_elem: Float64, max_bf16_ulp: Float64):
    print("  ", name, "  ", ns_per_elem, " ns/elem",
          "   max rounded-bf16 ULP=", max_bf16_ulp, sep="")


def main() raises:
    comptime assert N % W == 0, "bench_exp expects N to be a multiple of SIMD width"

    seed(0xC0FFEE)
    var inputs = alloc[Float32](N)
    var outputs = alloc[Float32](N)
    for i in range(N):
        inputs[i] = Float32(random_float64(-10.0, 10.0))
    for i in range(N):
        outputs[i] = Float32(0.0)

    print("exp variant benchmark - N=", N, " elements/pass, SIMD width W=", W, sep="")
    print("Inputs: uniform random in [-10, 10] (typical post-shift softmax range).")
    print("Timed passes per benchmark iteration: ", PASSES_PER_ITER, sep="")
    print()
    print("  Variant                                        ns/elem     max rounded-bf16 ULP")
    print("  ", "-" * 80, sep="")

    # ---- Mojo math.exp scalar reference ------------------------------------
    @parameter
    def bench_scalar():
        bench_scalar_kernel(inputs, outputs)

    var r = run[bench_scalar](
        num_warmup_iters=BENCH_WARMUP_ITERS,
        min_runtime_secs=BENCH_MIN_RUNTIME_SECS,
        max_runtime_secs=BENCH_MAX_RUNTIME_SECS,
        max_iters=SCALAR_MAX_ITERS,
    )
    report_row(String("Mojo math.exp (scalar)                    "),
               r.mean(Unit.ns) / Float64(TIMED_ELEMS_PER_ITER), max_scalar_bf16_ulp(inputs))

    # ---- exp_simd[N=2] -----------------------------------------------------
    @parameter
    def bench_n2():
        bench_exp_simd_kernel[2](inputs, outputs)

    r = run[bench_n2](
        num_warmup_iters=BENCH_WARMUP_ITERS,
        min_runtime_secs=BENCH_MIN_RUNTIME_SECS,
        max_runtime_secs=BENCH_MAX_RUNTIME_SECS,
        max_iters=SIMD_MAX_ITERS,
    )
    report_row(String("exp_simd[N=2] (bf16-oriented, current)    "),
               r.mean(Unit.ns) / Float64(TIMED_ELEMS_PER_ITER), max_exp_simd_bf16_ulp[2](inputs))

    # ---- exp_simd[N=3] -----------------------------------------------------
    @parameter
    def bench_n3():
        bench_exp_simd_kernel[3](inputs, outputs)

    r = run[bench_n3](
        num_warmup_iters=BENCH_WARMUP_ITERS,
        min_runtime_secs=BENCH_MIN_RUNTIME_SECS,
        max_runtime_secs=BENCH_MAX_RUNTIME_SECS,
        max_iters=SIMD_MAX_ITERS,
    )
    report_row(String("exp_simd[N=3] (f16-oriented)              "),
               r.mean(Unit.ns) / Float64(TIMED_ELEMS_PER_ITER), max_exp_simd_bf16_ulp[3](inputs))

    # ---- exp_simd[N=6] -----------------------------------------------------
    @parameter
    def bench_n6():
        bench_exp_simd_kernel[6](inputs, outputs)

    r = run[bench_n6](
        num_warmup_iters=BENCH_WARMUP_ITERS,
        min_runtime_secs=BENCH_MIN_RUNTIME_SECS,
        max_runtime_secs=BENCH_MAX_RUNTIME_SECS,
        max_iters=SIMD_MAX_ITERS,
    )
    report_row(String("exp_simd[N=6] (f32-oriented)              "),
               r.mean(Unit.ns) / Float64(TIMED_ELEMS_PER_ITER), max_exp_simd_bf16_ulp[6](inputs))

    # ---- Schraudolph C=0 ---------------------------------------------------
    @parameter
    def bench_schraudolph():
        bench_schraudolph_kernel(inputs, outputs)

    r = run[bench_schraudolph](
        num_warmup_iters=BENCH_WARMUP_ITERS,
        min_runtime_secs=BENCH_MIN_RUNTIME_SECS,
        max_runtime_secs=BENCH_MAX_RUNTIME_SECS,
        max_iters=SIMD_MAX_ITERS,
    )
    report_row(String("Schraudolph C=0 (pure bit trick)          "),
               r.mean(Unit.ns) / Float64(TIMED_ELEMS_PER_ITER), max_schraudolph_bf16_ulp(inputs))

    # ---- Schraudolph + quad correction -------------------------------------
    @parameter
    def bench_schraudolph_quad():
        bench_schraudolph_quad_kernel(inputs, outputs)

    r = run[bench_schraudolph_quad](
        num_warmup_iters=BENCH_WARMUP_ITERS,
        min_runtime_secs=BENCH_MIN_RUNTIME_SECS,
        max_runtime_secs=BENCH_MAX_RUNTIME_SECS,
        max_iters=SIMD_MAX_ITERS,
    )
    report_row(String("Schraudolph + quadratic correction        "),
               r.mean(Unit.ns) / Float64(TIMED_ELEMS_PER_ITER),
               max_schraudolph_quad_bf16_ulp(inputs))

    inputs.free()
    outputs.free()
