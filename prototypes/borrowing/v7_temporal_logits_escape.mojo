from std.reflection import reflect
from std.memory import UnsafePointer, alloc, free, Layout
from std.collections import InlineArray
from std.sys.info import size_of
from std.builtin.rebind import downcast


comptime ALIGN = 64
comptime MAX_BUFS = 16


@always_inline
def aligned64(b: Int) -> Int:
    return ((b + ALIGN - 1) // ALIGN) * ALIGN


trait BufferLike:
    """Scratch buffer carrier.

    Like v6, a buffer inherits its lifetime from the most recent PhaseGroup
    marker in declaration order.
    """
    comptime Element: AnyType
    comptime SIZE: Int


@fieldwise_init
struct Buf[T: AnyType, count: Int](BufferLike, Copyable, ImplicitlyCopyable):
    comptime Element = Self.T
    comptime SIZE = aligned64(Self.count * size_of[Self.T]())


trait PhaseRange:
    comptime FIRST: Int
    comptime LAST: Int


@fieldwise_init
struct PhaseGroup[first: Int, last: Int](
    PhaseRange, Copyable, ImplicitlyCopyable
):
    comptime FIRST = Self.first
    comptime LAST = Self.last


@fieldwise_init
struct IslandPlan(Copyable, ImplicitlyCopyable):
    var offsets: InlineArray[Int, MAX_BUFS]
    var peak: Int
    var count: Int


def derive_plan[T: AnyType]() -> IslandPlan:
    var sizes = InlineArray[Int, MAX_BUFS](fill=0)
    var firsts = InlineArray[Int, MAX_BUFS](fill=0)
    var lasts = InlineArray[Int, MAX_BUFS](fill=0)
    var n = 0
    var cur_first = -1
    var cur_last = -1

    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, PhaseRange):
            cur_first = FT.FIRST
            cur_last = FT.LAST
        comptime if conforms_to(FT, BufferLike):
            if cur_first < 0 or cur_last < cur_first:
                print("scratch buffer declared without a valid PhaseGroup")
            sizes[n] = FT.SIZE
            firsts[n] = cur_first
            lasts[n] = cur_last
            n += 1

    var order = InlineArray[Int, MAX_BUFS](fill=0)
    for k in range(n):
        order[k] = k
    for i in range(n):
        var best = i
        for j in range(i + 1, n):
            if sizes[order[j]] > sizes[order[best]]:
                best = j
        var tmp = order[i]
        order[i] = order[best]
        order[best] = tmp

    var offsets = InlineArray[Int, MAX_BUFS](fill=0)
    var placed = InlineArray[Bool, MAX_BUFS](fill=False)
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
                    continue
                var jl = offsets[j]
                var jh = offsets[j] + sizes[j]
                if x < jh and jl < x + sizes[idx]:
                    x = jh
                    stable = False
                    break
        offsets[idx] = x
        placed[idx] = True
        if x + sizes[idx] > peak:
            peak = x + sizes[idx]

    return IslandPlan(offsets=offsets, peak=peak, count=n)


trait Island:
    comptime PLAN: IslandPlan = derive_plan[Self]()


def buffer_index_for[T: AnyType, name: StringLiteral]() -> Int:
    """Field name -> position among BufferLike fields only."""
    comptime target = reflect[T].field_index[name]()
    var n = 0
    comptime for i in range(reflect[T].field_count()):
        if i == target:
            return n
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, BufferLike):
            n += 1
    return -1


comptime F32 = Scalar[DType.float32]
comptime HIDDEN = 8
comptime VOCAB = 16


@fieldwise_init
struct TinyForwardScratch(Island, Copyable, ImplicitlyCopyable):
    comptime input_norm = 1
    comptime lm_head = 2
    comptime returned_to_caller = 3

    var hidden_band: PhaseGroup[Self.input_norm, Self.lm_head]
    var hidden: Buf[F32, HIDDEN]

    # This is intentionally modeled through the return edge. It is still just
    # a scratch slot, but callers receive a temporal view and must consume it
    # before the next forward overwrites the same address.
    var logits_band: PhaseGroup[Self.lm_head, Self.returned_to_caller]
    var logits: Buf[F32, VOCAB]


@explicit_destroy
struct TemporalLogitsView[vocab: Int, dtype: DType = DType.float32](Movable):
    """Linear, non-owning view over a temporal logits scratch slot.

    The pointer is deliberately untracked (`MutAnyOrigin`) so the view does
    not borrow the model/pool. The API contract is temporal: callers consume
    the view before calling forward again, because the next forward may
    overwrite the same scratch slot.
    """
    comptime DTYPE = Self.dtype
    comptime VOCAB = Self.vocab

    var ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]

    def __init__(
        out self,
        ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
    ):
        self.ptr = ptr

    @always_inline
    def load_f32[width: Int](self, offset: Int) -> SIMD[DType.float32, width]:
        return (self.ptr + offset).load[width=width]().cast[DType.float32]()

    def release(deinit self):
        # Semantic release only. Positional scratch slots are reused by the
        # next forward call rather than returned to a LIFO allocator.
        pass


struct PositionalScratchPool[size: Int](Movable):
    var base: UnsafePointer[UInt8, MutAnyOrigin]

    def __init__(out self, base: UnsafePointer[UInt8, MutAnyOrigin]):
        self.base = base

    @always_inline
    def slot[
        I: Island, name: StringLiteral,
    ](self) -> UnsafePointer[
        downcast[reflect[I].field_type[name].T, BufferLike].Element,
        MutAnyOrigin,
    ]:
        comptime idx = buffer_index_for[I, name]()
        comptime off = I.PLAN.offsets[idx]
        return UnsafePointer[
            downcast[reflect[I].field_type[name].T, BufferLike].Element,
            MutAnyOrigin,
        ](unsafe_from_address=Int(self.base) + off)

    @always_inline
    def logits_view[
        I: Island, name: StringLiteral, vocab: Int,
    ](mut self) -> TemporalLogitsView[vocab]:
        var logits_ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(self.slot[I, name]()))
        return TemporalLogitsView[vocab](logits_ptr)


comptime POOL_SIZE = TinyForwardScratch.PLAN.peak


def fake_forward(
    mut pool: PositionalScratchPool[POOL_SIZE],
) -> TemporalLogitsView[VOCAB]:
    var hidden = pool.slot[TinyForwardScratch, "hidden"]()
    for i in range(HIDDEN):
        hidden[i] = Float32(i)

    var logits = pool.slot[TinyForwardScratch, "logits"]()
    for i in range(VOCAB):
        logits[i] = Float32(i)

    return pool.logits_view[TinyForwardScratch, "logits", VOCAB]()


def main():
    print("== v7_temporal_logits_escape ==")
    print("TinyForwardScratch.PEAK =", TinyForwardScratch.PLAN.peak)
    print("POOL_SIZE               =", POOL_SIZE)
    print(
        "hidden/logits offsets   =",
        TinyForwardScratch.PLAN.offsets[
            buffer_index_for[TinyForwardScratch, "hidden"]()],
        TinyForwardScratch.PLAN.offsets[
            buffer_index_for[TinyForwardScratch, "logits"]()],
    )

    var ly = Layout[UInt8](count=POOL_SIZE)
    var raw = alloc(ly)
    var pool = PositionalScratchPool[POOL_SIZE](raw)

    var logits = fake_forward(pool)
    print("first logits[7]         =", logits.load_f32[1](7)[0])
    logits^.release()

    var fresh = fake_forward(pool)
    print("fresh logits[7]         =", fresh.load_f32[1](7)[0])
    fresh^.release()

    free(raw, ly)
