from std.collections import InlineArray
from std.memory import Span, UnsafePointer
from std.sys.info import simd_width_of


comptime ALIGN = 64


@always_inline
def align_up(value: Int, alignment: Int = ALIGN) -> Int:
    return ((value + alignment - 1) // alignment) * alignment


# ───────────────────────── 1. view-based binding (no cap) ───────────────────
# The model owns one List[Int] of per-rank arena bases. Every binding carries a
# Span view over it: len is the degree, indexing is bounds-checked, and the
# origin ties the view to the owning list — the lifetime invariant is enforced
# by the compiler, not by manual discipline. The `o` parameter is inferred at
# construction and slots into the `tp` slot the old design carried. The arena
# invariant: every rank's arena is byte-identical, so rank r's pointer =
# ptr + (bases[r] - bases[0]).
@fieldwise_init
struct RankView[o: ImmutOrigin](Copyable, ImplicitlyCopyable):
    var bases: Span[Int, Self.o]

    @always_inline
    def degree(self) -> Int:
        return len(self.bases)

    @always_inline
    def delta(self, rank: Int) -> Int:
        return self.bases[rank] - self.bases[0]

    @always_inline
    def bind[T: AnyType](
        self, ptr: UnsafePointer[T, MutAnyOrigin],
    ) -> Binding[T, Self.o]:
        return Binding[T, Self.o](ptr, self)


@fieldwise_init
struct Binding[T: AnyType, o: ImmutOrigin](Copyable, ImplicitlyCopyable):
    var ptr: UnsafePointer[Self.T, MutAnyOrigin]
    var view: RankView[Self.o]

    @always_inline
    def __getitem__(self, rank: Int) -> UnsafePointer[Self.T, MutAnyOrigin]:
        return UnsafePointer[Self.T, MutAnyOrigin](
            unsafe_from_address=Int(self.ptr) + self.view.delta(rank))

    @always_inline
    def shifted(self, n: Int) -> Self:
        return Self(self.ptr + n, self.view)


# ──────────────────────────── 2. degree-free shape ──────────────────────────
# Logical extent + shard axis stay comptime (intrinsic to the model); the
# physical per-rank extent is a runtime function of a runtime degree.
struct ShardAxis:
    comptime NONE = 0   # replicated  -> fixed region, identical on every rank
    comptime ROW = 1    # output axis -> cheap runtime (outer / partition)
    comptime COL = 2    # contraction -> strip-mine to stay vectorized


struct LogicalShape[rows: Int, cols: Int, axis: Int = ShardAxis.NONE]:
    comptime ROWS = Self.rows
    comptime COLS = Self.cols
    comptime AXIS = Self.axis

    @always_inline
    @staticmethod
    def data_rows(degree: Int) -> Int:
        return Self.rows // degree if Self.axis == ShardAxis.ROW else Self.rows

    @always_inline
    @staticmethod
    def data_cols(degree: Int) -> Int:
        return Self.cols // degree if Self.axis == ShardAxis.COL else Self.cols

    @always_inline
    @staticmethod
    def bytes(degree: Int, elt_bytes: Int) -> Int:
        return Self.data_rows(degree) * Self.data_cols(degree) * elt_bytes


# ───────────────────── 3. two-part region (fixed + runtime) ──────────────────
# Audit confirmed no host-local data: every slot is distributed or replicated,
# so there is no host-append tail. The common region is fixed ++ runtime and is
# byte-identical on every rank — every byte is delta-bindable, no exceptions.
@fieldwise_init
struct CommonRegion(Copyable, ImplicitlyCopyable):
    var fixed_bytes: Int        # replicated — independent of degree
    var runtime_bytes: Int      # sharded — = sum(logical // degree)
    var total_bytes: Int        # identical on every rank


@always_inline
def plan_region(
    degree: Int, read fixed: List[Int], read sharded: List[Int],
) -> CommonRegion:
    var off = 0
    for i in range(len(fixed)):
        off = align_up(off + fixed[i])
    var fixed_bytes = off
    for i in range(len(sharded)):
        off = align_up(off + sharded[i] // degree)
    return CommonRegion(fixed_bytes, off - fixed_bytes, off)


# ──────────────────── 4. runtime scratch planner (the new piece) ─────────────
# The bin-packer at temporal_scratch.mojo:106-144 is already pure runtime
# arithmetic over `sizes[]`; only its *inputs* were comptime. Here it runs at
# load time, fed sizes resolved from a comptime scaling class plus runtime
# (degree, workers). No information is missing: the phase/lifetime structure is
# comptime, and degree + max-workers-across-ranks are both known at load.
struct ScaleClass:
    comptime FIXED = 0
    comptime PER_DEGREE = 1
    comptime PER_WORKER = 2
    comptime PER_WORKER_PER_DEGREE = 3


@fieldwise_init
struct ScratchSpec(Copyable, ImplicitlyCopyable):
    var base_elems: Int   # comptime base unit (degree-1, 1-worker)
    var elt_bytes: Int
    var scale: Int        # ScaleClass
    var first: Int        # phase interval [first, last] — lifetime, not size
    var last: Int


@always_inline
def resolve_bytes(spec: ScratchSpec, degree: Int, workers: Int) -> Int:
    var n = spec.base_elems
    if spec.scale == ScaleClass.PER_DEGREE:
        n = n // degree
    elif spec.scale == ScaleClass.PER_WORKER:
        n = n * workers
    elif spec.scale == ScaleClass.PER_WORKER_PER_DEGREE:
        n = (n // degree) * workers
    return align_up(n * spec.elt_bytes)


def plan_scratch(
    read specs: List[ScratchSpec], degree: Int, workers: Int,
    mut offsets: List[Int],
) -> Int:
    var n = len(specs)
    var sizes = List[Int]()
    var firsts = List[Int]()
    var lasts = List[Int]()
    var order = List[Int]()
    var placed_offsets = List[Int]()
    var placed = List[Bool]()
    while len(offsets) < n:
        offsets.append(0)
    for i in range(n):
        sizes.append(resolve_bytes(specs[i], degree, workers))
        firsts.append(specs[i].first)
        lasts.append(specs[i].last)
        order.append(i)
        placed_offsets.append(0)
        placed.append(False)

    # largest first (greedy) — identical to derive_scratch_plan's sort
    for i in range(n):
        var best = i
        for j in range(i + 1, n):
            if sizes[order[j]] > sizes[order[best]]:
                best = j
        var tmp = order[i]
        order[i] = order[best]
        order[best] = tmp

    var peak = 0
    for k in range(n):
        var idx = order[k]
        var x = 0
        var stable = False
        while not stable:
            stable = True
            for j in range(n):
                if not placed[j]:
                    continue
                if firsts[idx] > lasts[j] or lasts[idx] < firsts[j]:
                    continue  # disjoint lifetimes -> may share bytes
                var jl = placed_offsets[j]
                var jh = placed_offsets[j] + sizes[j]
                if x < jh and jl < x + sizes[idx]:
                    x = jh
                    stable = False
                    break
        placed_offsets[idx] = x
        placed[idx] = True
        offsets[idx] = x
        if x + sizes[idx] > peak:
            peak = x + sizes[idx]
    return peak


def co_live_buffers_overlap(
    read specs: List[ScratchSpec], read offsets: List[Int],
    degree: Int, workers: Int,
) -> Bool:
    """The safety invariant: any two buffers whose lifetimes overlap must hold
    disjoint byte ranges. If this is ever True the plan is unsound."""
    var n = len(specs)
    for i in range(n):
        for j in range(i + 1, n):
            if specs[i].first > specs[j].last or specs[i].last < specs[j].first:
                continue
            var ai = offsets[i]
            var bi = ai + resolve_bytes(specs[i], degree, workers)
            var aj = offsets[j]
            var bj = aj + resolve_bytes(specs[j], degree, workers)
            if ai < bj and aj < bi:
                return True
    return False


# Representative islands, with the real Gemma4 config + scaling classes.
comptime HIDDEN = 2816
comptime INTERMEDIATE = 2112
comptime MOE_INTERMEDIATE = 704
comptime SLIDING_WINDOW = 1024
comptime TOP_K = 8
comptime Q_DIM_SLIDING = 4096
comptime KV_DIM_SLIDING = 2048
comptime HEAD_DIM_SLIDING = 256
comptime MR = 4
comptime TILE_J = 64


def ffn_moe_island() -> List[ScratchSpec]:
    # phases: dense_gate_up=0 router=1 setup=2 phase1=3 phase2=4 dense_down=5
    var s = List[ScratchSpec]()
    s.append(ScratchSpec(SLIDING_WINDOW * INTERMEDIATE, 2, ScaleClass.PER_DEGREE, 0, 5))  # ffn_gate
    s.append(ScratchSpec(SLIDING_WINDOW * INTERMEDIATE, 2, ScaleClass.PER_DEGREE, 0, 0))  # ffn_up
    s.append(ScratchSpec(HIDDEN, 4, ScaleClass.PER_WORKER, 1, 1))                         # router_scaled
    s.append(ScratchSpec(SLIDING_WINDOW * HIDDEN, 2, ScaleClass.FIXED, 2, 3))             # x_normed
    s.append(ScratchSpec(SLIDING_WINDOW * TOP_K * MOE_INTERMEDIATE, 2, ScaleClass.FIXED, 3, 4))  # bucket
    s.append(ScratchSpec(MR * 2 * TILE_J, 4, ScaleClass.PER_WORKER, 3, 3))                # gate_scratch
    s.append(ScratchSpec(SLIDING_WINDOW * HIDDEN, 4, ScaleClass.FIXED, 4, 4))             # accum
    s.append(ScratchSpec(SLIDING_WINDOW * HIDDEN, 2, ScaleClass.FIXED, 5, 5))             # dense_out
    return s^


def sliding_island() -> List[ScratchSpec]:
    # phases: qkv=0 attention=1 o_proj=2
    var per_head = HEAD_DIM_SLIDING + 2
    var heads_full = Q_DIM_SLIDING // HEAD_DIM_SLIDING
    var s = List[ScratchSpec]()
    s.append(ScratchSpec(SLIDING_WINDOW * Q_DIM_SLIDING, 2, ScaleClass.PER_DEGREE, 0, 2))       # q
    s.append(ScratchSpec(SLIDING_WINDOW * KV_DIM_SLIDING * 2, 2, ScaleClass.PER_DEGREE, 0, 1))  # kv
    s.append(ScratchSpec(heads_full * per_head, 4, ScaleClass.PER_WORKER_PER_DEGREE, 1, 1))     # partials
    return s^


def run_scratch_config(degree: Int, workers: Int) -> Int:
    var ffn = ffn_moe_island()
    var sld = sliding_island()
    var ffn_off = List[Int]()
    var sld_off = List[Int]()
    var ffn_peak = plan_scratch(ffn, degree, workers, ffn_off)
    var sld_peak = plan_scratch(sld, degree, workers, sld_off)
    var agg = max(ffn_peak, sld_peak)
    debug_assert(
        not co_live_buffers_overlap(ffn, ffn_off, degree, workers),
        "ffn co-live buffers overlap")
    debug_assert(
        not co_live_buffers_overlap(sld, sld_off, degree, workers),
        "sliding co-live buffers overlap")
    print("  degree", degree, "| workers", workers,
          "| ffn peak", ffn_peak, "| sliding peak", sld_peak,
          "| aggregate", agg, "| (co-live disjoint: OK)")
    return agg


# ──────────────── 5. strip-mined contraction (comptime tile, runtime count) ──
@always_inline
def dot_block[BLOCK: Int, width: Int](
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    base: Int,
) -> Float32:
    var acc = SIMD[DType.float32, width](0)
    comptime for k in range(BLOCK // width):
        var off = base + k * width
        acc = (a + off).load[width=width]().fma(
            (b + off).load[width=width](), acc)
    return acc.reduce_add()


@always_inline
def strip_mined_dot[BLOCK: Int](
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    contraction: Int,
) -> Float32:
    comptime width = simd_width_of[DType.float32]()
    var total: Float32 = 0
    for blk in range(contraction // BLOCK):
        total += dot_block[BLOCK, width](a, b, blk * BLOCK)
    return total


def main():
    print("=== runtime-degree layout prototype ===")

    # (1) view-based binding: [r] is total for every rank, no cap ------------
    var bases = List[Int]()
    bases.append(0x10000)
    bases.append(0x90000)
    bases.append(0x110000)
    var view = RankView(Span(bases))
    var slot_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=bases[0] + 4096)
    var b = view.bind(slot_ptr)
    for r in range(view.degree()):
        debug_assert(Int(b[r]) == bases[r] + 4096, "binding delta wrong")
    print("  view-based binding OK for degree", view.degree(), "(no MAX_RANKS)")

    # (2) degree-free shapes --------------------------------------------------
    comptime VOCAB = 262144
    comptime GateUp = LogicalShape[INTERMEDIATE, HIDDEN, ShardAxis.ROW]
    comptime Down = LogicalShape[HIDDEN, INTERMEDIATE, ShardAxis.COL]
    comptime FullQ = LogicalShape[8192, HIDDEN, ShardAxis.NONE]
    comptime Embed = LogicalShape[VOCAB, HIDDEN, ShardAxis.ROW]
    for degree in [1, 2, 4]:
        print("  degree", degree,
              "| gate_up rows/rank", GateUp.data_rows(degree),
              "| down contraction/rank", Down.data_cols(degree),
              "| fullQ rows (fixed)", FullQ.data_rows(degree),
              "| vocab/rank", Embed.data_rows(degree))

    # (3) two-part region: fixed stable, runtime shrinks, all delta-bindable --
    var fixed = List[Int]()
    fixed.append(FullQ.bytes(1, 2))
    fixed.append(HIDDEN * 2)
    for degree in [1, 2, 4]:
        var sharded = List[Int]()
        sharded.append(GateUp.ROWS * GateUp.COLS * 2)
        sharded.append(Down.ROWS * Down.COLS * 2)
        sharded.append(Embed.ROWS * Embed.COLS * 2)
        var region = plan_region(degree, fixed, sharded)
        print("  degree", degree, "| fixed", region.fixed_bytes,
              "| runtime", region.runtime_bytes, "| total", region.total_bytes)
        debug_assert(
            region.fixed_bytes == align_up(align_up(FullQ.bytes(1, 2)) + HIDDEN * 2),
            "fixed region must not move with degree")

    # (4) runtime scratch planner — exact + co-live-disjoint at any config ----
    var agg_lo = run_scratch_config(1, 64)
    var agg_hi = run_scratch_config(4, 128)
    debug_assert(agg_lo > 0 and agg_hi > 0, "scratch peaks must be positive")

    # (5) planner output feeds binding: place + bind a scratch buffer ---------
    var ffn = ffn_moe_island()
    var ffn_off = List[Int]()
    _ = plan_scratch(ffn, 4, 128, ffn_off)
    var scratch_off = align_up(plan_region(4, fixed, List[Int]()).total_bytes)
    var gate_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=bases[0] + scratch_off + ffn_off[0])
    var gate = view.bind(gate_ptr)
    for r in range(view.degree()):
        debug_assert(
            Int(gate[r]) == bases[r] + scratch_off + ffn_off[0],
            "scratch binding delta wrong")
    print("  planned scratch offset", ffn_off[0], "binds across all ranks: OK")

    # (6) strip-mined contraction stays correct as runtime extent shrinks -----
    comptime N = 512
    var av = InlineArray[Float32, N](uninitialized=True)
    var bv = InlineArray[Float32, N](uninitialized=True)
    for i in range(N):
        av[i] = Float32(i % 7) - 3.0
        bv[i] = Float32((i * 3) % 5) - 2.0
    var ap = UnsafePointer(to=av[0])
    var bp = UnsafePointer(to=bv[0])
    for degree in [1, 2, 4]:
        var contraction = N // degree
        var got = strip_mined_dot[64](ap, bp, contraction)
        var expected_dot: Float32 = 0
        for i in range(contraction):
            expected_dot += av[i] * bv[i]
        debug_assert(abs(got - expected_dot) < 1e-2, "strip-mined dot mismatch")
    print("  strip-mined contraction correct across degrees")

    print("=== all invariants held ===")
