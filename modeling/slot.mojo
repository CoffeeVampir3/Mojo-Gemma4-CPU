from std.memory import UnsafePointer
from std.reflection import reflect

from modeling.model_spec import (
    Encoding, BF16, ShapeLike, StaticView, WeightDesc,
    DEFAULT_ALIGNMENT, DISTRIBUTED, align_up,
)
from modeling.utilities import FieldwiseDefault


trait SlotLike:
    comptime E: Encoding
    comptime S: ShapeLike
    comptime NAME: StaticString
    comptime TARGET_RANK: Int

    @always_inline
    def set_offset(mut self, off: Int): ...

    @always_inline
    def get_offset(self) -> Int: ...


trait SlotGroup(FieldwiseDefault):
    pass


struct Slot[
    E_: Encoding, S_: ShapeLike, name_: StaticString = "",
    target_rank_: Int = DISTRIBUTED,
](SlotLike, Defaultable, Copyable, ImplicitlyCopyable):
    comptime E = Self.E_
    comptime S = Self.S_
    comptime NAME = Self.name_
    comptime TARGET_RANK = Self.target_rank_

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
    def bound(self, base: Int) -> StaticView[Self.E_, Self.S_]:
        return StaticView[Self.E_, Self.S_](
            UnsafePointer[Scalar[Self.E_.DTYPE], MutAnyOrigin](
                unsafe_from_address=base + self.offset))


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
            off = align_up(off + FT.S.bytes[FT.E]())
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
                    dtype=FT.E.DTYPE,
                    element_bytes=FT.E.ELEMENT_BYTES,
                    global_rows=FT.S.GLOBAL_N,
                    global_cols=FT.S.GLOBAL_M,
                    local_cols=FT.S.M,
                    data_rows=FT.S.DATA_N,
                    data_cols=FT.S.DATA_M,
                    target_rank=FT.TARGET_RANK,
                ))
            off = align_up(off + FT.S.bytes[FT.E]())
        comptime if conforms_to(FT, SlotGroup):
            off = emit_descs[FT](prefix, region_base, ops, off)
    return off


def slot_layout_bytes[T: AnyType]() -> Int:
    """Pure comptime layout sizing (no instance, no descs). Useful for
    SectionBuilder-style stride calculation."""
    var off = 0
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, SlotLike):
            off = align_up(off + FT.S.bytes[FT.E]())
        comptime if conforms_to(FT, SlotGroup):
            off = align_up(off + slot_layout_bytes[FT]())
    return off
