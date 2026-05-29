from std.memory import UnsafePointer
from std.reflection import reflect

from kernels.helpers import RankView, Binding
from modeling.model_spec import (
    Encoding, ShapeLike, WeightDesc,
    DISTRIBUTED, align_up,
)
from modeling.utilities import FieldwiseDefault
from quant.recipe import QuantRecipe, Passthrough
from quant.manifest import (
    quant_manifest, manifest_arena_bytes, QuantRole,
)


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
struct BindContext[o: ImmutOrigin](Copyable, ImplicitlyCopyable):
    """Per-call binding context. `layer_base` is the current layer's absolute
    arena offset for weight-slot resolution; state slots anchor at the rank-0
    arena base via `state_binding`. The `RankView` carries the runtime
    tensor-parallel degree (`len` of the borrowed bases)."""
    var view: RankView[Self.o]
    var layer_base: Int

    @always_inline
    def degree(self) -> Int:
        return self.view.degree()

    @always_inline
    def with_layer(self, lb: Int) -> Self:
        var c = self
        c.layer_base = lb
        return c

    @always_inline
    def bind[T: AnyType](
        self, ptr: UnsafePointer[T, MutAnyOrigin],
    ) -> Binding[T, Self.o]:
        return self.view.bind(ptr)


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
    def binding[o: ImmutOrigin](
        self, base: Int, view: RankView[o],
    ) -> Binding[Scalar[Self.ENCODING.DTYPE], o]:
        return view.bind(self.at(base))

    @always_inline
    def binding[o: ImmutOrigin](
        self, ctx: BindContext[o],
    ) -> Binding[Scalar[Self.ENCODING.DTYPE], o]:
        return self.binding(ctx.layer_base, ctx.view)

    @always_inline
    def state_binding[o: ImmutOrigin](
        self, ctx: BindContext[o],
    ) -> Binding[Scalar[Self.ENCODING.DTYPE], o]:
        return self.binding(ctx.view.bases[0], ctx.view)


@always_inline
def slot_arena_bytes[
    encoding: Encoding, shape: ShapeLike, quant: QuantRecipe,
](degree: Int) -> Int:
    """Per-rank arena bytes for a slot at runtime `degree`: weight + every
    sidecar implied by `quant`, summed from the shared manifest."""
    return manifest_arena_bytes[encoding, shape, quant](degree)


@always_inline
def emit_quant_descs[
    encoding: Encoding, shape: ShapeLike, quant: QuantRecipe,
    name: StaticString, target_rank: Int,
](
    prefix: String, slot_arena_off: Int, degree: Int, mut ops: List[WeightDesc],
):
    """Emit one WeightDesc per physical tensor in this slot's encoding, driven
    by the shared manifest at runtime `degree`: the weight at slot_arena_off,
    then sidecars packed tightly after at their manifest rel_offs. The colsum
    member is reserved in the arena but computed at model init, so it is never
    read from the checkpoint and emits no loader desc."""
    var full = prefix + String(name)
    var manifest = quant_manifest[encoding, shape, quant](degree)
    for i in range(manifest.count):
        var member = manifest.members[i]
        if member.role != QuantRole.COLSUM:
            ops.append(WeightDesc(
                name=full + String(member.suffix),
                arena_offset=slot_arena_off + member.rel_off,
                dtype=member.dtype, element_bytes=member.element_bytes,
                global_rows=member.global_rows, global_cols=member.global_cols,
                local_cols=member.local_cols,
                data_rows=member.data_rows, data_cols=member.data_cols,
                target_rank=target_rank,
            ))


def stamp_offsets[T: AnyType](mut t: T, degree: Int, off_in: Int = 0) -> Int:
    """Walk T (recursing into SlotGroup fields), stamping each Slot's
    within-region byte offset for the runtime `degree`. Returns total bytes
    consumed. Each slot's byte footprint comes from `slot_arena_bytes`, which is
    recipe- and degree-aware. Does NOT emit any loader records."""
    var off = off_in
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, SlotLike):
            ref slot = reflect[T].field_ref[i](t)
            slot.set_offset(off)
            off = align_up(off + slot_arena_bytes[
                FT.ENCODING, FT.SHAPE, FT.QUANT,
            ](degree))
        comptime if conforms_to(FT, SlotGroup):
            ref nested = reflect[T].field_ref[i](t)
            off = stamp_offsets(nested, degree, off)
    return off


def emit_descs[T: AnyType](
    prefix: String,
    region_base: Int,
    degree: Int,
    mut ops: List[WeightDesc],
    off_in: Int = 0,
) -> Int:
    """Walk T comptime, emitting one or more WeightDescs per named Slot at
    region_base + within-region offset for the runtime `degree`. Recurses into
    SlotGroup fields. Returns total bytes (must match stamp_offsets for the same
    T and degree)."""
    var off = off_in
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, SlotLike):
            comptime if FT.NAME != StaticString(""):
                emit_quant_descs[
                    FT.ENCODING, FT.SHAPE, FT.QUANT,
                    FT.NAME, FT.TARGET_RANK,
                ](prefix, region_base + off, degree, ops)
            off = align_up(off + slot_arena_bytes[
                FT.ENCODING, FT.SHAPE, FT.QUANT,
            ](degree))
        comptime if conforms_to(FT, SlotGroup):
            off = emit_descs[FT](prefix, region_base, degree, ops, off)
    return off
