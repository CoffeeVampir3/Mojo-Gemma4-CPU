from simd_math import log_simd
from std.math import log as libm_log, exp as libm_exp
from std.os import abort
from std.sys.info import simd_width_of


comptime W = simd_width_of[DType.float32]()

comptime BF16_ULP = Float64(0.0078125)
comptime F16_ULP  = Float64(0.0009765625)
comptime F32_ULP  = Float64(1.1920928955078125e-07)


def measure_max_rel_err[
    polynomial_degree: Int,
](log_lo: Float64, log_hi: Float64, n: Int) -> Float64:
    # Sweep log-uniform: x = exp(log_lo + i * step)
    var max_err = Float64(0.0)
    var step = (log_hi - log_lo) / Float64(n)
    var i = 0
    while i + W <= n:
        var xv = SIMD[DType.float32, W]()
        var k = 0
        while k < W:
            xv[k] = Float32(libm_exp(log_lo + Float64(i + k) * step))
            k += 1
        var got = log_simd[polynomial_degree, W](xv)
        var j = 0
        while j < W:
            var truth = libm_log(Float64(xv[j]))
            if truth != 0.0:
                var rel = abs(Float64(got[j]) - truth) / abs(truth)
                if rel > max_err: max_err = rel
            j += 1
        i += W
    return max_err


def report(label: String, rel: Float64):
    print(
        "  ", label, "  rel=", rel,
        "  bf16=", rel / BF16_ULP,
        "  f16=", rel / F16_ULP,
        "  f32=", rel / F32_ULP,
        sep="",
    )


def expect_le(mut failed: Int, label: String, value: Float64, bound: Float64):
    if value > bound:
        print("  FAIL ", label, ": ", value, " > ", bound, sep="")
        failed += 1
    else:
        print("  pass ", label, ": ", value, " ≤ ", bound, sep="")


def main():
    print("log_simd error profile — sweep x ∈ [0.1, 10] log-uniform, 2^18 samples")
    print()
    var lo = Float64(-2.302585092994046)  # ln(0.1)
    var hi = Float64(2.302585092994046)   # ln(10)
    var n = 1 << 18

    var e2 = measure_max_rel_err[2](lo, hi, n)
    var e3 = measure_max_rel_err[3](lo, hi, n)
    var e4 = measure_max_rel_err[4](lo, hi, n)
    var e5 = measure_max_rel_err[5](lo, hi, n)
    var e6 = measure_max_rel_err[6](lo, hi, n)
    var e7 = measure_max_rel_err[7](lo, hi, n)

    report(String("N=2"), e2)
    report(String("N=3"), e3)
    report(String("N=4"), e4)
    report(String("N=5"), e5)
    report(String("N=6"), e6)
    report(String("N=7"), e7)

    print()
    print("Assertions (documented bounds in logarithm.mojo):")
    var failed = 0
    # log_simd polynomial converges fast — N=2 already at f16 ULP, N≥3 at f32 floor
    # N=2: documented ~5 f32 ULPs / ~0.0007 f16 ULPs → bound 8 f32 ULPs
    expect_le(failed, String("N=2 < 8 f32 ULPs (loss-free for f16/bf16)"), e2, F32_ULP * 8.0)
    # N=3+: documented ~1.8 f32 ULPs (saturated by f32 evaluation noise)
    expect_le(failed, String("N=3 < 3 f32 ULPs (saturated)"), e3, F32_ULP * 3.0)
    expect_le(failed, String("N=4 < 3 f32 ULPs (saturated)"), e4, F32_ULP * 3.0)
    expect_le(failed, String("N=5 < 3 f32 ULPs (saturated)"), e5, F32_ULP * 3.0)
    expect_le(failed, String("N=6 < 3 f32 ULPs (saturated)"), e6, F32_ULP * 3.0)
    expect_le(failed, String("N=7 < 3 f32 ULPs (saturated)"), e7, F32_ULP * 3.0)

    print()
    if failed > 0:
        print("FAILED:", failed, "assertion(s)")
        abort()
    print("All log_simd assertions passed.")
