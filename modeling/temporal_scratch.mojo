from std.builtin.rebind import downcast
from std.collections import InlineArray
from std.memory import UnsafePointer
from std.reflection import reflect
from std.sys.info import size_of

from kernels.helpers import ArenaBases, Binding
from modeling.slot import BindContext


comptime SCRATCH_ALIGNMENT = 64
comptime MAX_SCRATCH_SLOTS = 64


@always_inline
def aligned_scratch_bytes[nbytes: Int]() -> Int:
    return ((nbytes + SCRATCH_ALIGNMENT - 1) // SCRATCH_ALIGNMENT) * SCRATCH_ALIGNMENT


trait ScratchBufferLike:
    comptime Element: AnyType
    comptime SIZE: Int


@fieldwise_init
struct ScratchBuffer[T: AnyType, count: Int](
    ScratchBufferLike, Copyable, ImplicitlyCopyable
):
    comptime Element = Self.T
    comptime SIZE = aligned_scratch_bytes[Self.count * size_of[Self.T]()]()


trait ScratchPhaseOrderLike:
    @staticmethod
    def index[name: StaticString]() -> Int: ...


struct ScratchPhaseOrder[*names: StaticString](ScratchPhaseOrderLike):
    @staticmethod
    def index[name: StaticString]() -> Int:
        comptime for i in range(len(Self.names)):
            comptime if Self.names[i] == name:
                return i
        return -1


trait ScratchPhaseRange:
    comptime FIRST_NAME: StaticString
    comptime LAST_NAME: StaticString


@fieldwise_init
struct ScratchPhase[first: StaticString, last: StaticString](
    ScratchPhaseRange, Copyable, ImplicitlyCopyable
):
    comptime FIRST_NAME = Self.first
    comptime LAST_NAME = Self.last


@fieldwise_init
struct ScratchPlan(Copyable, ImplicitlyCopyable):
    var offsets: InlineArray[Int, MAX_SCRATCH_SLOTS]
    var peak: Int
    var count: Int


trait ScratchPhaseSchema:
    comptime PHASES: ScratchPhaseOrderLike

    @staticmethod
    def phase_index[name: StaticString]() -> Int:
        return Self.PHASES.index[name]()


def derive_scratch_plan[T: ScratchPhaseSchema]() -> ScratchPlan:
    var sizes = InlineArray[Int, MAX_SCRATCH_SLOTS](fill=0)
    var firsts = InlineArray[Int, MAX_SCRATCH_SLOTS](fill=0)
    var lasts = InlineArray[Int, MAX_SCRATCH_SLOTS](fill=0)
    var n = 0
    var cur_first = -1
    var cur_last = -1

    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, ScratchPhaseRange):
            comptime first = T.phase_index[FT.FIRST_NAME]()
            comptime last = T.phase_index[FT.LAST_NAME]()
            comptime assert first >= 0, "scratch phase start is not declared in PHASES"
            comptime assert last >= 0, "scratch phase end is not declared in PHASES"
            comptime assert last >= first, "scratch phase end precedes start"
            cur_first = first
            cur_last = last
        comptime if conforms_to(FT, ScratchBufferLike):
            if cur_first < 0 or cur_last < cur_first:
                print("scratch buffer declared without a valid phase")
            sizes[n] = FT.SIZE
            firsts[n] = cur_first
            lasts[n] = cur_last
            n += 1

    var order = InlineArray[Int, MAX_SCRATCH_SLOTS](fill=0)
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

    var offsets = InlineArray[Int, MAX_SCRATCH_SLOTS](fill=0)
    var placed = InlineArray[Bool, MAX_SCRATCH_SLOTS](fill=False)
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

    return ScratchPlan(offsets=offsets, peak=peak, count=n)


trait ScratchIsland(ScratchPhaseSchema):
    comptime PLAN: ScratchPlan = derive_scratch_plan[Self]()


def aggregate_scratch_peak[T: AnyType]() -> Int:
    var m = 0
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, ScratchIsland):
            if FT.PLAN.peak > m:
                m = FT.PLAN.peak
    return m


def scratch_slot_index[T: AnyType, name: StringLiteral]() -> Int:
    comptime target = reflect[T].field_index[name]()
    var n = 0
    comptime for i in range(reflect[T].field_count()):
        if i == target:
            return n
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, ScratchBufferLike):
            n += 1
    return -1


struct TemporalScratchPool[size: Int](Movable):
    var base: UnsafePointer[UInt8, MutAnyOrigin]

    def __init__(out self, base: Int):
        self.base = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=base)

    @always_inline
    def slot[
        I: ScratchIsland, name: StringLiteral,
    ](self) -> UnsafePointer[
        downcast[reflect[I].field_type[name].T, ScratchBufferLike].Element,
        MutAnyOrigin,
    ]:
        comptime idx = scratch_slot_index[I, name]()
        comptime off = I.PLAN.offsets[idx]
        return UnsafePointer[
            downcast[reflect[I].field_type[name].T, ScratchBufferLike].Element,
            MutAnyOrigin,
        ](unsafe_from_address=Int(self.base) + off)

    @always_inline
    def binding[
        I: ScratchIsland, name: StringLiteral, tp: Int,
    ](self, bases: ArenaBases[tp]) -> Binding[
        downcast[reflect[I].field_type[name].T, ScratchBufferLike].Element,
        tp,
    ]:
        return bases.bind(self.slot[I, name]())

    @always_inline
    def binding[
        I: ScratchIsland, name: StringLiteral, tp: Int,
    ](self, ctx: BindContext[tp]) -> Binding[
        downcast[reflect[I].field_type[name].T, ScratchBufferLike].Element,
        tp,
    ]:
        return self.binding[I, name](ctx.arena_bases)


@explicit_destroy
struct TemporalLogitsView[
    vocab: Int, degree: Int, dtype: DType = DType.bfloat16,
](Movable):
    comptime DTYPE = Self.dtype
    comptime VOCAB = Self.vocab
    comptime VOCAB_PER_RANK = Self.vocab // Self.degree

    var ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    var bases: ArenaBases[Self.degree]

    def __init__(
        out self,
        ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        bases: ArenaBases[Self.degree],
    ):
        self.ptr = ptr
        self.bases = bases

    @always_inline
    def local_ptr(self, offset: Int) -> UnsafePointer[
        Scalar[Self.dtype], MutAnyOrigin,
    ]:
        var rank = offset // Self.VOCAB_PER_RANK
        var local = offset - rank * Self.VOCAB_PER_RANK
        return UnsafePointer[Scalar[Self.dtype], MutAnyOrigin](
            unsafe_from_address=Int(self.ptr) + self.bases[rank]
                - self.bases[0]) + local

    @always_inline
    def load_f32[width: Int](self, offset: Int) -> SIMD[DType.float32, width]:
        debug_assert(
            offset % Self.VOCAB_PER_RANK + width <= Self.VOCAB_PER_RANK,
            "TemporalLogitsView load crosses a rank shard",
        )
        return self.local_ptr(offset).load[width=width]().cast[DType.float32]()

    def release(deinit self):
        pass
