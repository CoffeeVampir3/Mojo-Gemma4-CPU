from std.memory import UnsafePointer
from std.reflection import reflect

from kernels.helpers import ArenaBases, Binding
from modeling.model_spec import (
    Encoding, ShapeLike, WeightDesc,
    DISTRIBUTED, align_up,
)
from modeling.utilities import FieldwiseDefault
from quant.recipe import QuantRecipe, Passthrough, RouterCenter, SoftmaxRouterCenter
from quant.manifest import (
    quant_manifest, manifest_arena_bytes, member_rel_off, has_role, QuantRole,
)
from butterquant.weight import (
    ButterquantWeight, ButterquantRouter,
    quant_vnni_packed, quant_colsum_per_block, quant_k_block,
)
from butterquant.pack import PackColsumTask


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

    @always_inline
    def bq_weight[degree: Int](
        self, ctx: BindContext[degree],
    ) -> ButterquantWeight[
        Self.QUANT, Self.SHAPE.DATA_N, Self.SHAPE.DATA_M, degree,
    ]:
        """Bind the int8 weight + scale + colsum sidecars of a quantized slot
        from the same per-slot offsets `emit_quant_descs` wrote them to. The
        colsum binding points at the weight base when the recipe declares none;
        `colsum_checked` gates access at comptime."""
        comptime assert has_role[
            Self.ENCODING, Self.SHAPE, Self.QUANT, QuantRole.SCALE,
        ](), "Slot.bq_weight requires a quantized slot with a scale sidecar."
        comptime SCALE_OFF = member_rel_off[
            Self.ENCODING, Self.SHAPE, Self.QUANT, QuantRole.SCALE]()
        comptime CS_OFF = member_rel_off[
            Self.ENCODING, Self.SHAPE, Self.QUANT, QuantRole.COLSUM]()
        var base = ctx.layer_base + self.offset
        var data = UnsafePointer[Int8, MutAnyOrigin](unsafe_from_address=base)
        var scale = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=base + SCALE_OFF)
        var colsum = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=base + CS_OFF)
        return ButterquantWeight[
            Self.QUANT, Self.SHAPE.DATA_N, Self.SHAPE.DATA_M, degree,
        ](ctx.bind(data), ctx.bind(scale), ctx.bind(colsum))

    @always_inline
    def bq_router[degree: Int](
        self, ctx: BindContext[degree],
    ) -> ButterquantRouter[
        Self.QUANT, Self.SHAPE.DATA_N, Self.SHAPE.DATA_M, degree,
    ]:
        """Bind a router-centered slot. The centered bf16 weight is always
        present; the gauge (§13.2 pivot) and bias are bound only when the recipe
        stores them, so a §13.5 SoftmaxRouterCenter slot binds neither."""
        comptime assert (
            Self.QUANT.isa[RouterCenter]() or Self.QUANT.isa[SoftmaxRouterCenter]()
        ), "Slot.bq_router requires a router-centered slot."
        var base = ctx.layer_base + self.offset
        var centered = ctx.bind(UnsafePointer[BFloat16, MutAnyOrigin](
            unsafe_from_address=base))
        var gauge = Optional[Binding[BFloat16, degree]](None)
        var bias = Optional[Binding[Float32, degree]](None)
        comptime if has_role[
            Self.ENCODING, Self.SHAPE, Self.QUANT, QuantRole.GAUGE,
        ]():
            comptime GAUGE_OFF = member_rel_off[
                Self.ENCODING, Self.SHAPE, Self.QUANT, QuantRole.GAUGE]()
            gauge = Optional[Binding[BFloat16, degree]](
                ctx.bind(UnsafePointer[BFloat16, MutAnyOrigin](
                    unsafe_from_address=base + GAUGE_OFF)))
        comptime if has_role[
            Self.ENCODING, Self.SHAPE, Self.QUANT, QuantRole.BIAS,
        ]():
            comptime BIAS_OFF = member_rel_off[
                Self.ENCODING, Self.SHAPE, Self.QUANT, QuantRole.BIAS]()
            bias = Optional[Binding[Float32, degree]](
                ctx.bind(UnsafePointer[Float32, MutAnyOrigin](
                    unsafe_from_address=base + BIAS_OFF)))
        return ButterquantRouter[
            Self.QUANT, Self.SHAPE.DATA_N, Self.SHAPE.DATA_M, degree,
        ](centered, gauge, bias)


@always_inline
def slot_arena_bytes[
    encoding: Encoding, shape: ShapeLike, quant: QuantRecipe,
]() -> Int:
    """Per-rank arena bytes for a slot: weight + every sidecar implied by
    `quant`, summed from the shared manifest (`quant/manifest.mojo`)."""
    return manifest_arena_bytes[encoding, shape, quant]()


@always_inline
def emit_quant_descs[
    encoding: Encoding, shape: ShapeLike, quant: QuantRecipe,
    name: StaticString, target_rank: Int,
](
    prefix: String, slot_arena_off: Int, mut ops: List[WeightDesc],
):
    """Emit one WeightDesc per physical tensor in this slot's encoding, driven
    by the shared manifest: the weight at slot_arena_off, then sidecars packed
    tightly after at their manifest rel_offs. Member shapes carry the loader's
    row-shard / col-shard / replicated parameters directly. The colsum member is
    reserved in the arena but computed at model init during the VNNI pack, so it
    is never read from the checkpoint and emits no loader desc."""
    var full = prefix + String(name)
    comptime MANIFEST = quant_manifest[encoding, shape, quant]()
    comptime for i in range(MANIFEST.count):
        comptime MEMBER = MANIFEST.members[i]
        comptime if MEMBER.role != QuantRole.COLSUM:
            ops.append(WeightDesc(
                name=full + String(MEMBER.suffix),
                arena_offset=slot_arena_off + MEMBER.rel_off,
                dtype=MEMBER.dtype, element_bytes=MEMBER.element_bytes,
                global_rows=MEMBER.global_rows, global_cols=MEMBER.global_cols,
                local_cols=MEMBER.local_cols,
                data_rows=MEMBER.data_rows, data_cols=MEMBER.data_cols,
                target_rank=target_rank,
            ))


def stamp_offsets[T: AnyType](mut t: T, off_in: Int = 0) -> Int:
    """Walk T (recursing into SlotGroup fields), stamping each Slot's
    within-region byte offset. Returns total bytes consumed. Each slot's
    byte footprint comes from `slot_arena_bytes`, which is recipe-aware
    (passthrough is source bytes; quantized recipes include sidecars).
    Does NOT emit any loader records — that's emit_descs's job."""
    var off = off_in
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, SlotLike):
            ref slot = reflect[T].field_ref[i](t)
            slot.set_offset(off)
            off = align_up(off + slot_arena_bytes[
                FT.ENCODING, FT.SHAPE, FT.QUANT,
            ]())
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
    """Walk T comptime, emitting one or more WeightDescs per named Slot
    at region_base + within-region offset (one per physical tensor in the
    slot's quant encoding). Recurses into SlotGroup fields. Returns total
    bytes (must match stamp_offsets's return for the same T)."""
    var off = off_in
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, SlotLike):
            comptime if FT.NAME != StaticString(""):
                emit_quant_descs[
                    FT.ENCODING, FT.SHAPE, FT.QUANT,
                    FT.NAME, FT.TARGET_RANK,
                ](prefix, region_base + off, ops)
            off = align_up(off + slot_arena_bytes[
                FT.ENCODING, FT.SHAPE, FT.QUANT,
            ]())
        comptime if conforms_to(FT, SlotGroup):
            off = emit_descs[FT](prefix, region_base, ops, off)
    return off


def emit_pack_tasks[T: AnyType](
    region_base: Int,
    mut tasks: List[PackColsumTask],
    off_in: Int = 0,
) -> Int:
    """Walk T comptime, emitting one VNNI pack task per VnniPacked weight slot
    (row-major / passthrough / router slots declare no pack and are skipped) at
    region_base + within-region offset. Offset accumulation matches emit_descs so
    weight/colsum offsets land on the same arena bytes the loader wrote. Returns
    total bytes consumed."""
    var off = off_in
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, SlotLike):
            comptime if quant_vnni_packed[FT.QUANT]():
                comptime per_block = quant_colsum_per_block[FT.QUANT]()
                comptime block_cols = (
                    quant_k_block[FT.QUANT]() if per_block else FT.SHAPE.DATA_M)
                comptime cs_off = member_rel_off[
                    FT.ENCODING, FT.SHAPE, FT.QUANT, QuantRole.COLSUM]()
                tasks.append(PackColsumTask(
                    weight_off=region_base + off,
                    colsum_off=region_base + off + cs_off,
                    rows=FT.SHAPE.DATA_N,
                    cols=FT.SHAPE.DATA_M,
                    block_cols=block_cols,
                    colsum_row_major=not per_block,
                ))
            off = align_up(off + slot_arena_bytes[
                FT.ENCODING, FT.SHAPE, FT.QUANT,
            ]())
        comptime if conforms_to(FT, SlotGroup):
            off = emit_pack_tasks[FT](region_base, tasks, off)
    return off
