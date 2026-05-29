from std.time import perf_counter_ns
from threading.threading_traits import BurstThreadPool


comptime PROFILE_RESERVOIR = 1024  # fixed samples kept per metric for p50/p99


def two_dp(ns: Int, unit: Int) -> String:
    var scaled = ns * 100 // unit
    var frac = scaled % 100
    var fs = String(frac)
    if frac < 10:
        fs = String("0") + fs
    return String(scaled // 100) + "." + fs


def human_ns(ns: Int) -> String:
    """Render a nanosecond duration with an adaptive unit."""
    if ns < 1000:
        return String(ns) + "ns"
    if ns < 1_000_000:
        return two_dp(ns, 1000) + "us"
    if ns < 1_000_000_000:
        return two_dp(ns, 1_000_000) + "ms"
    return two_dp(ns, 1_000_000_000) + "s"


def pct_str(part: Int, whole: Int) -> String:
    if whole <= 0:
        return "0.0%"
    var tenths = part * 1000 // whole
    return String(tenths // 10) + "." + String(tenths % 10) + "%"


def pad_left(s: String, w: Int) -> String:
    var pad = String("")
    for _ in range(w - s.byte_length()):
        pad += " "
    return pad + s


def pad_right(s: String, w: Int) -> String:
    var pad = String("")
    for _ in range(w - s.byte_length()):
        pad += " "
    return s + pad


def rule(n: Int) -> String:
    var s = String("")
    for _ in range(n):
        s += "-"
    return s


struct ReservoirMetric[N: Int = PROFILE_RESERVOIR](Copyable, Movable):
    """Bounded per-metric stats. count/sum/min/max are exact running
    values; a fixed-N uniform reservoir (Vitter algorithm R) on the heap
    backs the quantiles, so memory is capped at N regardless of sample
    count and the enclosing value type stays small."""
    var samples: List[Int]
    var count: Int
    var total: Int
    var minv: Int
    var maxv: Int
    var rng: UInt64

    @always_inline
    def __init__(out self):
        self.samples = List[Int]()
        self.count = 0
        self.total = 0
        self.minv = 0
        self.maxv = 0
        self.rng = 0x9E3779B97F4A7C15

    @always_inline
    def next_rand(mut self) -> UInt64:
        var x = self.rng
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        self.rng = x
        return x

    @always_inline
    def add(mut self, raw: Int):
        var v = raw if raw > 0 else 0
        var seen = self.count
        if seen == 0 or v < self.minv:
            self.minv = v
        if seen == 0 or v > self.maxv:
            self.maxv = v
        self.total += v
        if seen < Self.N:
            self.samples.append(v)
        else:
            var j = Int(self.next_rand() % UInt64(seen + 1))
            if j < Self.N:
                self.samples[j] = v
        self.count = seen + 1

    @always_inline
    def mean(self) -> Int:
        if self.count == 0:
            return 0
        return self.total // self.count

    def quantiles(self, q_lo: Float64, q_hi: Float64) -> Tuple[Int, Int]:
        """(p_lo, p_hi) from the reservoir. Returns exact min/max bounds
        when the reservoir is empty/degenerate."""
        var filled = len(self.samples)
        if filled == 0:
            return (0, 0)
        var ordered = self.samples.copy()
        sort(ordered)
        var lo_idx = Int(q_lo * Float64(filled))
        var hi_idx = Int(q_hi * Float64(filled))
        if lo_idx >= filled:
            lo_idx = filled - 1
        if hi_idx >= filled:
            hi_idx = filled - 1
        return (ordered[lo_idx], ordered[hi_idx])

    def fmt_row(self, section: String, name_w: Int, cw: Int, grand: Int) -> String:
        var q = self.quantiles(0.5, 0.99)
        return (pad_right(section, name_w)
            + pad_left(String(self.count), 8)
            + pad_left(human_ns(self.minv), cw)
            + pad_left(human_ns(q[0]), cw)
            + pad_left(human_ns(q[1]), cw)
            + pad_left(human_ns(self.maxv), cw)
            + pad_left(human_ns(self.mean()), cw)
            + pad_left(pct_str(self.total, grand), 8))


struct ProfileRecord(Copyable, Movable):
    var label: StaticString
    var dispatch: ReservoirMetric[]
    var compute: ReservoirMetric[]
    var join: ReservoirMetric[]

    @always_inline
    def __init__(out self, label: StaticString = ""):
        self.label = label
        self.dispatch = ReservoirMetric[]()
        self.compute = ReservoirMetric[]()
        self.join = ReservoirMetric[]()

    @always_inline
    def add(mut self, dispatch_ns: Int, compute_ns: Int, join_ns: Int):
        self.dispatch.add(dispatch_ns)
        self.compute.add(compute_ns)
        self.join.add(join_ns)

struct Profiler[Profile: Bool, N: Int = 64](Copyable, Movable):
    """Per-dispatch timing sink with bounded per-label statistics.

    `Profile=False` => zero-sized, every method a no-op. When on, holds N
    label slots; each tracks exact count/sum/min/max and a fixed-size
    reservoir for p50/p99 on dispatch / compute / join. Touched only on
    the dispatch thread, so no atomics and no cross-NUMA traffic."""
    comptime CAP = Self.N if Self.Profile else 0
    var records: InlineArray[ProfileRecord, Self.CAP]
    var count: Int
    var wall_ns: Int  # true elapsed forward time over the current window

    @always_inline
    def __init__(out self):
        self.records = InlineArray[ProfileRecord, Self.CAP](fill=ProfileRecord())
        self.count = 0
        self.wall_ns = 0

    @always_inline
    def add_wall(mut self, ns: Int):
        comptime if Self.Profile:
            self.wall_ns += ns

    @always_inline
    def record(mut self, label: StaticString,
               dispatch_ns: Int, compute_ns: Int, join_ns: Int):
        comptime if Self.Profile:
            for i in range(self.count):
                if self.records[i].label == label:
                    self.records[i].add(dispatch_ns, compute_ns, join_ns)
                    return
            if self.count < Self.N:
                self.records[self.count] = ProfileRecord(label)
                self.records[self.count].add(dispatch_ns, compute_ns, join_ns)
                self.count += 1

    @always_inline
    def reset(mut self):
        comptime if Self.Profile:
            self.count = 0
            self.wall_ns = 0

    def report(self, title: StaticString = "dispatch profile"):
        comptime if Self.Profile:
            if self.count == 0:
                return
            var name_w = 7  # byte_length("section")
            for i in range(self.count):
                var w = self.records[i].label.byte_length() + 11  # + " / dispatch"
                if w > name_w:
                    name_w = w
            var cw = 11

            # per-label totals, then order labels by cost desc
            var totals = List[Int](capacity=self.count)
            var order = List[Int](capacity=self.count)
            for i in range(self.count):
                ref r = self.records[i]
                var lt = r.dispatch.total + r.compute.total + r.join.total
                totals.append(lt)
                order.append(i)
            for a in range(1, self.count):
                var key = order[a]
                var kt = totals[key]
                var b = a - 1
                while b >= 0 and totals[order[b]] < kt:
                    order[b + 1] = order[b]
                    b -= 1
                order[b + 1] = key

            var header = (pad_right("section", name_w)
                + pad_left("calls", 8)
                + pad_left("min", cw) + pad_left("p50", cw)
                + pad_left("p99", cw) + pad_left("max", cw)
                + pad_left("mean", cw) + pad_left("%wall", 8))
            print()
            print("=== " + String(title) + " : per-dispatch latency ===")
            print(header)
            print(rule(header.byte_length()))
            var td = 0
            var tc = 0
            var tj = 0
            for oi in range(self.count):
                ref r = self.records[order[oi]]
                var lbl = String(r.label)
                print(r.dispatch.fmt_row(lbl + " / dispatch", name_w, cw, self.wall_ns))
                print(r.compute.fmt_row(lbl + " / compute", name_w, cw, self.wall_ns))
                print(r.join.fmt_row(lbl + " / join", name_w, cw, self.wall_ns))
                td += r.dispatch.total
                tc += r.compute.total
                tj += r.join.total
            print(rule(header.byte_length()))
            var accounted = td + tc + tj
            var wall = self.wall_ns
            var dark = wall - accounted
            if dark < 0:
                dark = 0
            print("by phase   dispatch " + human_ns(td) + " (" + pct_str(td, wall) + ")"
                + "   compute " + human_ns(tc) + " (" + pct_str(tc, wall) + ")"
                + "   join " + human_ns(tj) + " (" + pct_str(tj, wall) + ")")
            print("wall " + human_ns(wall)
                + "   accounted " + human_ns(accounted) + " (" + pct_str(accounted, wall) + ")"
                + "   dark " + human_ns(dark) + " (" + pct_str(dark, wall) + ")")


@always_inline
def compute_end_across[
    P: BurstThreadPool, //, tp: Int,
](mut pools: List[P]) -> Int:
    """Max worker completion timestamp across all ranks from the last
    dispatch. Call after join. perf_counter_ns is monotonic and shared
    across cores, so this is comparable to the dispatcher's own clock."""
    var m = 0
    for r in range(tp):
        var ts = pools[r].last_worker_timestamp()
        if ts > m:
            m = ts
    return m


struct DispatchSpan[Profile: Bool](Copyable, ImplicitlyCopyable):
    """Comptime-gated timing scope for one dispatch/compute/join cycle.

    Construct at issue start (captures t0), call `issued()` after the
    dispatch loop (t1), then `finish(...)` after join (t2 + cross-pool
    compute end, clamped to the measured span). `Profile=False` =>
    zero-sized, every method no-op."""
    comptime M = 2 if Self.Profile else 0
    var ts: InlineArray[Int, Self.M]  # [0]=t0 (issue start), [1]=t1 (issue end)

    @always_inline
    def __init__(out self):
        comptime if Self.Profile:
            var now = Int(perf_counter_ns())
            self.ts = InlineArray[Int, Self.M](fill=now)
        else:
            self.ts = InlineArray[Int, Self.M](uninitialized=True)

    @always_inline
    def issued(mut self):
        comptime if Self.Profile:
            self.ts[1] = Int(perf_counter_ns())

    @always_inline
    def finish[
        P: BurstThreadPool, N: Int, //, tp: Int,
    ](self, mut prof: Profiler[Self.Profile, N], mut pools: List[P],
      label: StaticString):
        comptime if Self.Profile:
            var t2 = Int(perf_counter_ns())
            var compute_end = compute_end_across[tp](pools)
            if compute_end < self.ts[1]:
                compute_end = self.ts[1]
            if compute_end > t2:
                compute_end = t2
            var dispatch_ns = self.ts[1] - self.ts[0]
            var compute_ns = compute_end - self.ts[1]
            var join_ns = t2 - compute_end
            prof.record(label, dispatch_ns, compute_ns, join_ns)

    @always_inline
    def finish_inline[
        N: Int, //,
    ](self, mut prof: Profiler[Self.Profile, N], label: StaticString):
        """Record a main-thread inline execution (no dispatch/join)."""
        comptime if Self.Profile:
            var t_end = Int(perf_counter_ns())
            prof.record(label, 0, t_end - self.ts[0], 0)
