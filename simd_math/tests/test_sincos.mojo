from simd_math import sincos_simd
from std.math import sin as libm_sin, cos as libm_cos
from std.os import abort
from std.sys.info import simd_width_of


comptime W = simd_width_of[DType.float64]()

# Format-relative ULP at value 1.0 (sin/cos bounded by 1, so abs == rel here).
comptime F16_ULP = Float64(0.0009765625)
comptime F32_ULP = Float64(1.1920928955078125e-07)
comptime F64_ULP = Float64(2.220446049250313e-16)


@fieldwise_init
struct AbsErrPair(Copyable, Movable, ImplicitlyCopyable):
    var sin_max: Float64
    var cos_max: Float64


def measure_max_abs_err[
    polynomial_degree: Int,
    cody_waite: Bool = False,
](lo: Float64, hi: Float64, n: Int) -> AbsErrPair:
    var sin_max = Float64(0.0)
    var cos_max = Float64(0.0)
    var step = (hi - lo) / Float64(n)
    var i = 0
    while i + W <= n:
        var xv = SIMD[DType.float64, W]()
        var k = 0
        while k < W:
            xv[k] = lo + Float64(i + k) * step
            k += 1
        var got = sincos_simd[polynomial_degree, W, cody_waite=cody_waite](xv)
        var j = 0
        while j < W:
            var x = xv[j]
            var sin_truth = libm_sin(x)
            var cos_truth = libm_cos(x)
            var sin_err = abs(got.sin_val[j] - sin_truth)
            var cos_err = abs(got.cos_val[j] - cos_truth)
            if sin_err > sin_max: sin_max = sin_err
            if cos_err > cos_max: cos_max = cos_err
            j += 1
        i += W
    return AbsErrPair(sin_max=sin_max, cos_max=cos_max)


def report(label: String, p: AbsErrPair):
    var worst = p.sin_max if p.sin_max > p.cos_max else p.cos_max
    print(
        "  ", label,
        "  sin_abs=", p.sin_max,
        "  cos_abs=", p.cos_max,
        "  worst f32_ULPs=", worst / F32_ULP,
        "  worst f64_ULPs=", worst / F64_ULP,
        sep="",
    )


def expect_le(mut failed: Int, label: String, value: Float64, bound: Float64):
    if value > bound:
        print("  FAIL ", label, ": ", value, " > ", bound, sep="")
        failed += 1
    else:
        print("  pass ", label, ": ", value, " ≤ ", bound, sep="")


def main():
    print("sincos_simd error profile — sweep θ ∈ [-π, π], 2^18 samples")
    print()
    var lo = Float64(-3.141592653589793)
    var hi = Float64(3.141592653589793)
    var n = 1 << 18

    var p4 = measure_max_abs_err[4](lo, hi, n)
    var p5 = measure_max_abs_err[5](lo, hi, n)
    var p6 = measure_max_abs_err[6](lo, hi, n)
    var p7 = measure_max_abs_err[7](lo, hi, n)
    var p8 = measure_max_abs_err[8](lo, hi, n)
    var p9 = measure_max_abs_err[9](lo, hi, n)
    var p10 = measure_max_abs_err[10](lo, hi, n)

    report(String("N=4 "), p4)
    report(String("N=5 "), p5)
    report(String("N=6 "), p6)
    report(String("N=7 "), p7)
    report(String("N=8 "), p8)
    report(String("N=9 "), p9)
    report(String("N=10"), p10)

    print()
    print("Assertions (documented bounds in sincos.mojo):")
    var failed = 0
    var w4 = max(p4.sin_max, p4.cos_max)
    var w5 = max(p5.sin_max, p5.cos_max)
    var w6 = max(p6.sin_max, p6.cos_max)
    var w7 = max(p7.sin_max, p7.cos_max)
    var w8 = max(p8.sin_max, p8.cos_max)
    var w9 = max(p9.sin_max, p9.cos_max)
    var w10 = max(p10.sin_max, p10.cos_max)
    # N=4: documented ~10 f32 ULPs (~1.2e-6 abs)
    expect_le(failed, String("N=4 < 12 f32 ULPs"), w4, F32_ULP * 12.0)
    # N=5: ~6.7e-9 abs (well below f32 ULP)
    expect_le(failed, String("N=5 < 1e-8 abs (sub-f32 ULP)"), w5, Float64(1e-8))
    # N=6: ~2.7e-11
    expect_le(failed, String("N=6 < 5e-11 abs"), w6, Float64(5e-11))
    # N=7: ~7.9e-14
    expect_le(failed, String("N=7 < 1e-13 abs"), w7, Float64(1e-13))
    # N=8: ~7.5e-16 (~3.4 f64 ULPs)
    expect_le(failed, String("N=8 < 5 f64 ULPs (loss-free for f64)"), w8, F64_ULP * 5.0)
    # N=9, N=10: same as N=8 (saturated)
    expect_le(failed, String("N=9 < 5 f64 ULPs (saturated)"), w9, F64_ULP * 5.0)
    expect_le(failed, String("N=10 < 5 f64 ULPs (saturated)"), w10, F64_ULP * 5.0)

    print()
    print("Cody-Waite reduction at large angles — sweep θ ∈ [-1e6, 1e6], 2^18 samples")
    print()
    var clo = Float64(-1.0e6)
    var chi = Float64(1.0e6)

    var p8_naive = measure_max_abs_err[8, cody_waite=False](clo, chi, n)
    var p8_cw = measure_max_abs_err[8, cody_waite=True](clo, chi, n)
    report(String("N=8 naive   "), p8_naive)
    report(String("N=8 + CW    "), p8_cw)

    var w8_cw = max(p8_cw.sin_max, p8_cw.cos_max)
    # N=8 + CW at angles up to 1e6 should still be at f64 ULP
    expect_le(failed, String("N=8 + CW @ ±1e6 < 10 f64 ULPs"), w8_cw, F64_ULP * 10.0)
    # Sanity: naive N=8 at this range should be measurably worse than f64 ULP
    var w8_naive = max(p8_naive.sin_max, p8_naive.cos_max)
    if w8_naive < F64_ULP * 100.0:
        print("  WARN: naive reduction error at ±1e6 surprisingly small (", w8_naive, ")", sep="")

    print()
    if failed > 0:
        print("FAILED:", failed, "assertion(s)")
        abort()
    print("All sincos_simd assertions passed.")
