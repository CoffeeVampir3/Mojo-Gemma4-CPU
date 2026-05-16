from std.collections import InlineArray
from std.memory import UnsafePointer
from std.reflection import reflect

from kernels.helpers import NumaPointerArray
from modeling.model_spec import (
    Encoding, ShapeLike, StaticView, WeightDesc,
    DISTRIBUTED, align_up,
)
from modeling.utilities import FieldwiseDefault


trait SlotLike:
    comptime ENCODING: Encoding
    comptime SHAPE: ShapeLike
    comptime NAME: StaticString
    comptime TARGET_RANK: Int

    @always_inline
    def set_offset(mut self, off: Int): ...

    @always_inline
    def get_offset(self) -> Int: ...


trait SlotGroup(FieldwiseDefault):
    pass


@fieldwise_init
struct BindContext[degree: Int](Copyable, ImplicitlyCopyable):
    """Per-call binding context. `layer_base` is the current layer's absolute
    arena offset for weight-slot resolution; for state slots whose offsets are
    absolute, callers pass `arena_bases[0]` as the base explicitly."""
    var arena_bases: InlineArray[Int, Self.degree]
    var layer_base: Int

    @always_inline
    def with_layer(self, lb: Int) -> Self:
        var c = self
        c.layer_base = lb
        return c


struct Slot[
    encoding: Encoding, shape: ShapeLike, name: StaticString = "",
    target_rank: Int = DISTRIBUTED,
](SlotLike, Defaultable, Copyable, ImplicitlyCopyable):
    comptime ENCODING = Self.encoding
    comptime SHAPE = Self.shape
    comptime NAME = Self.name
    comptime TARGET_RANK = Self.target_rank

    var offset: Int

    def __init__(out self):
        self.offset = -1

    @implicit
    def __init__(out self, offset: Int):
        self.offset = offset

    @always_inline
    def set_offset(mut self, off: Int):
        self.offset = off

    @always_inline
    def get_offset(self) -> Int:
        return self.offset

    @always_inline
    def bound(self, base: Int) -> StaticView[Self.ENCODING, Self.SHAPE]:
        return StaticView[Self.ENCODING, Self.SHAPE](
            UnsafePointer[Scalar[Self.ENCODING.DTYPE], MutAnyOrigin](
                unsafe_from_address=base + self.offset))

    @always_inline
    def ranks[degree: Int](
        self, base: Int, bases: InlineArray[Int, degree],
    ) -> NumaPointerArray[Self.ENCODING.DTYPE, degree]:
        return NumaPointerArray[Self.ENCODING.DTYPE, degree](
            UnsafePointer[Scalar[Self.ENCODING.DTYPE], MutAnyOrigin](
                unsafe_from_address=base + self.offset),
            bases)

    @always_inline
    def ranks[degree: Int](
        self, ctx: BindContext[degree],
    ) -> NumaPointerArray[Self.ENCODING.DTYPE, degree]:
        return self.ranks(ctx.layer_base, ctx.arena_bases)

    @always_inline
    def state_ranks[degree: Int](
        self, ctx: BindContext[degree],
    ) -> NumaPointerArray[Self.ENCODING.DTYPE, degree]:
        return self.ranks(ctx.arena_bases[0], ctx.arena_bases)


def stamp_offsets[T: AnyType](mut t: T, off_in: Int = 0) -> Int:
    """Walk T (recursing into SlotGroup fields), stamping each Slot's
    within-region byte offset. Returns total bytes consumed. Does NOT
    emit any loader records — that's emit_descs's job."""
    var off = off_in
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, SlotLike):
            ref slot = reflect[T].field_ref[i](t)
            slot.set_offset(off)
            off = align_up(off + FT.SHAPE.bytes[FT.ENCODING]())
        comptime if conforms_to(FT, SlotGroup):
            ref nested = reflect[T].field_ref[i](t)
            off = stamp_offsets(nested, off)
    return off


def emit_descs[T: AnyType](
    prefix: String,
    region_base: Int,
    mut ops: List[WeightDesc],
    off_in: Int = 0,
) -> Int:
    """Walk T comptime, emitting WeightDescs for each named Slot at
    region_base + within-region offset. Recurses into SlotGroup fields.
    Returns total bytes (must match stamp_offsets's return for the same T)."""
    var off = off_in
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, SlotLike):
            comptime if FT.NAME != "":
                ops.append(WeightDesc(
                    name=prefix + String(FT.NAME),
                    arena_offset=region_base + off,
                    dtype=FT.ENCODING.DTYPE,
                    element_bytes=FT.ENCODING.ELEMENT_BYTES,
                    global_rows=FT.SHAPE.GLOBAL_N,
                    global_cols=FT.SHAPE.GLOBAL_M,
                    local_cols=FT.SHAPE.M,
                    data_rows=FT.SHAPE.DATA_N,
                    data_cols=FT.SHAPE.DATA_M,
                    target_rank=FT.TARGET_RANK,
                ))
            off = align_up(off + FT.SHAPE.bytes[FT.ENCODING]())
        comptime if conforms_to(FT, SlotGroup):
            off = emit_descs[FT](prefix, region_base, ops, off)
    return off
