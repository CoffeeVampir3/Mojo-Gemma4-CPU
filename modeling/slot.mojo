from std.memory import UnsafePointer
from std.reflection import reflect

from kernels.helpers import ArenaBases, Binding
from modeling.model_spec import (
    Encoding, ShapeLike, WeightDesc,
    DISTRIBUTED, align_up,
)
from modeling.utilities import FieldwiseDefault
from quant.recipe import QuantRecipe, Passthrough


trait SlotLike:
    comptime ENCODING: Encoding
    comptime SHAPE: ShapeLike
    comptime NAME: StaticString
    comptime TARGET_RANK: Int
    comptime QUANT: QuantRecipe

    @always_inline
    def set_offset(mut self, off: Int): ...

    @always_inline
    def get_offset(self) -> Int: ...


trait SlotGroup(FieldwiseDefault):
    pass


@fieldwise_init
struct BindContext[degree: Int](Copyable, ImplicitlyCopyable):
    """Per-call binding context. `layer_base` is the current layer's absolute
    arena offset for weight-slot resolution; state slots produce bindings
    anchored at `arena_bases[0]` via `state_binding`."""
    var arena_bases: ArenaBases[Self.degree]
    var layer_base: Int

    @always_inline
    def with_layer(self, lb: Int) -> Self:
        var c = self
        c.layer_base = lb
        return c

    @always_inline
    def bind[T: AnyType](
        self, ptr: UnsafePointer[T, MutAnyOrigin],
    ) -> Binding[T, Self.degree]:
        return self.arena_bases.bind(ptr)


struct Slot[
    encoding: Encoding, shape: ShapeLike, name: StaticString = "",
    quant: QuantRecipe = Passthrough(),
    target_rank: Int = DISTRIBUTED,
](SlotLike, Defaultable, Copyable, ImplicitlyCopyable):
    comptime ENCODING = Self.encoding
    comptime SHAPE = Self.shape
    comptime NAME = Self.name
    comptime TARGET_RANK = Self.target_rank
    comptime QUANT = Self.quant

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
    def at(self, base: Int) -> UnsafePointer[Scalar[Self.ENCODING.DTYPE], MutAnyOrigin]:
        return UnsafePointer[Scalar[Self.ENCODING.DTYPE], MutAnyOrigin](
            unsafe_from_address=base + self.offset)

    @always_inline
    def binding[degree: Int](
        self, base: Int, bases: ArenaBases[degree],
    ) -> Binding[Scalar[Self.ENCODING.DTYPE], degree]:
        return bases.bind(self.at(base))

    @always_inline
    def binding[degree: Int](
        self, ctx: BindContext[degree],
    ) -> Binding[Scalar[Self.ENCODING.DTYPE], degree]:
        return self.binding(ctx.layer_base, ctx.arena_bases)

    @always_inline
    def state_binding[degree: Int](
        self, ctx: BindContext[degree],
    ) -> Binding[Scalar[Self.ENCODING.DTYPE], degree]:
        return self.binding(ctx.arena_bases[0], ctx.arena_bases)


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
