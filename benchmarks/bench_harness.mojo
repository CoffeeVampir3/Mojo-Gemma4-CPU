from std.math import sqrt
from std.memory import UnsafePointer, alloc
from std.time import perf_counter_ns
from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray


comptime DEFAULT_SAMPLES = 600


struct SampleBuffer:
    var kernel_ns: UnsafePointer[Int64, MutAnyOrigin]
    var wall_ns: UnsafePointer[Int64, MutAnyOrigin]
    var n: Int
    var cap: Int

    def __init__(out self, cap: Int = DEFAULT_SAMPLES):
        self.kernel_ns = alloc[Int64](cap).as_any_origin()
        self.wall_ns = alloc[Int64](cap).as_any_origin()
        self.n = 0
        self.cap = cap

    def __del__(deinit self):
        self.kernel_ns.free()
        self.wall_ns.free()

    @always_inline
    def push(mut self, kernel_ns: Int, wall_ns: Int):
        if self.n < self.cap:
            self.kernel_ns[self.n] = Int64(kernel_ns)
            self.wall_ns[self.n] = Int64(wall_ns)
            self.n += 1

    @always_inline
    def clear(mut self):
        self.n = 0


@fieldwise_init
struct Stats(Copyable, ImplicitlyCopyable):
    var min: Int64
    var p50: Int64
    var p90: Int64
    var p99: Int64
    var mean: Int64
    var stddev: Int64
    var n: Int


def sort_in_place(p: UnsafePointer[Int64, MutAnyOrigin], n: Int):
    for i in range(1, n):
        var v = p[i]
        var j = i - 1
        while j >= 0 and p[j] > v:
            p[j + 1] = p[j]
            j -= 1
        p[j + 1] = v


def compute_stats(p: UnsafePointer[Int64, MutAnyOrigin], n: Int) -> Stats:
    if n <= 0:
        return Stats(0, 0, 0, 0, 0, 0, 0)
    sort_in_place(p, n)
    var total = Int64(0)
    for i in range(n):
        total += p[i]
    var mean = total // Int64(n)
    var var_sum = Float64(0)
    for i in range(n):
        var d = Float64(p[i] - mean)
        var_sum += d * d
    var stddev = Int64(sqrt(var_sum / Float64(n)))
    var p99_idx = (n * 99) // 100 if n >= 100 else n - 1
    return Stats(
        min=p[0],
        p50=p[n // 2],
        p90=p[(n * 9) // 10],
        p99=p[p99_idx],
        mean=mean,
        stddev=stddev,
        n=n,
    )


def pick_unit(p50: Int64) -> Tuple[Int64, String]:
    if p50 < 1_000:
        return (Int64(1), String("ns"))
    if p50 < 1_000_000:
        return (Int64(1_000), String("us"))
    return (Int64(1_000_000), String("ms"))


def fmt_scaled(ns: Int64, divisor: Int64) -> String:
    var whole = ns // divisor
    var frac = (ns * 10 // divisor) % 10
    return String(whole) + "." + String(frac)


def fmt_stats_line(label: String, s: Stats, divisor: Int64, unit: String) -> String:
    return ("    " + label + " (" + unit + "):  "
        + "min=" + fmt_scaled(s.min, divisor)
        + " p50=" + fmt_scaled(s.p50, divisor)
        + " p90=" + fmt_scaled(s.p90, divisor)
        + " p99=" + fmt_scaled(s.p99, divisor)
        + " mean=" + fmt_scaled(s.mean, divisor)
        + " σ=" + fmt_scaled(s.stddev, divisor))


def print_row(
    row_label: String, kernel: Stats, wall: Stats, payload_bytes: Int = 0,
):
    var unit = pick_unit(kernel.p50)
    var div = unit[0]
    var u = unit[1]
    var bw_suffix = String("")
    if payload_bytes > 0 and wall.p50 > 0:
        var bw_100 = Int64(payload_bytes) * 100 // wall.p50
        bw_suffix = "  | " + String(bw_100 // 100) + "." \
                  + String(bw_100 % 100) + " GB/s"
    print(row_label)
    print(fmt_stats_line("kernel", kernel, div, u) + "  (n=" + String(kernel.n) + ")")
    print(fmt_stats_line("wall  ", wall,   div, u) + bw_suffix)


def max_last_ts[P: BurstThreadPool, //, tp: Int](
    mut pools: HeapMoveArray[P],
) -> Int:
    var hi = 0
    for r in range(tp):
        var ts = pools[r].last_worker_timestamp()
        if ts > hi:
            hi = ts
    return hi


def now_ns() -> Int:
    return Int(perf_counter_ns())
