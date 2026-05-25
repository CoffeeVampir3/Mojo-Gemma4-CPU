from std.memory import UnsafePointer
from std.reflection import reflect

from kernels.helpers import ArenaBases, Binding
from modeling.model_spec import (
    Encoding, ShapeLike, WeightDesc,
    DISTRIBUTED, align_up,
)
from modeling.utilities import FieldwiseDefault
from quant.recipe import (
    QuantRecipe, Passthrough, PerRowQuant, PerBlockQuant, RouterCenter,
    NoColsum, PerRowCs, PerBlockCs,
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
def slot_arena_bytes[
    encoding: Encoding, shape: ShapeLike, quant: QuantRecipe,
]() -> Int:
    """Per-rank arena bytes for a slot: weight + every sidecar implied
    by `quant`. Passthrough returns the source-dtype aligned byte size;
    PerRow / PerBlock add an int8 weight, an f32 scale, and (when the
    recipe asks) an f32 colsum; RouterCenter adds a bf16 centered weight,
    a bf16 gauge over weight cols, and (when bias_name is set) an f32 bias."""
    comptime if quant.isa[Passthrough]():
        return shape.bytes[encoding]()

    comptime if quant.isa[PerRowQuant]():
        comptime QT = quant[PerRowQuant]
        var total = shape.DATA_N * shape.DATA_M
        total += shape.DATA_N * 4
        comptime if QT.colsum.isa[PerRowCs]():
            total += shape.DATA_N * 4
        comptime if QT.colsum.isa[PerBlockCs]():
            total += shape.DATA_N * (shape.DATA_M // QT.fwht_block) * 4
        return total

    comptime if quant.isa[PerBlockQuant]():
        comptime QT = quant[PerBlockQuant]
        comptime nb_local = shape.DATA_M // QT.fwht_block
        var total = shape.DATA_N * shape.DATA_M
        total += shape.DATA_N * nb_local * 4
        comptime if QT.colsum.isa[PerBlockCs]():
            total += shape.DATA_N * nb_local * 4
        return total

    comptime if quant.isa[RouterCenter]():
        comptime QT = quant[RouterCenter]
        var total = shape.DATA_N * shape.DATA_M * 2
        total += shape.DATA_M * 2
        comptime if QT.bias_name != StaticString(""):
            total += shape.DATA_N * 4
        return total

    return shape.bytes[encoding]()


@always_inline
def emit_quant_descs[
    encoding: Encoding, shape: ShapeLike, quant: QuantRecipe,
    name: StaticString, target_rank: Int,
](
    prefix: String, slot_arena_off: Int, mut ops: List[WeightDesc],
):
    """Emit one WeightDesc per physical tensor in this slot's encoding:
    the weight at slot_arena_off, then sidecars packed tightly after.
    The sidecar shapes inherit the weight's row/col sharding through
    SHAPE.DATA_N / DATA_M so the loader's emit_reads picks the same
    row-shard / col-shard / replicated path for them."""
    var full = prefix + String(name)
    var off = slot_arena_off

    comptime if quant.isa[Passthrough]():
        ops.append(WeightDesc(
            name=full, arena_offset=off,
            dtype=encoding.DTYPE, element_bytes=encoding.ELEMENT_BYTES,
            global_rows=shape.GLOBAL_N, global_cols=shape.GLOBAL_M,
            local_cols=shape.M,
            data_rows=shape.DATA_N, data_cols=shape.DATA_M,
            target_rank=target_rank,
        ))
        return

    comptime if quant.isa[PerRowQuant]():
        comptime QT = quant[PerRowQuant]
        ops.append(WeightDesc(
            name=full, arena_offset=off,
            dtype=DType.int8, element_bytes=1,
            global_rows=shape.GLOBAL_N, global_cols=shape.GLOBAL_M,
            local_cols=shape.M,
            data_rows=shape.DATA_N, data_cols=shape.DATA_M,
            target_rank=target_rank,
        ))
        off += shape.DATA_N * shape.DATA_M
        ops.append(WeightDesc(
            name=full + String(".scale"), arena_offset=off,
            dtype=DType.float32, element_bytes=4,
            global_rows=shape.GLOBAL_N, global_cols=1,
            local_cols=1, data_rows=shape.DATA_N, data_cols=1,
            target_rank=target_rank,
        ))
        off += shape.DATA_N * 4
        comptime if QT.colsum.isa[PerRowCs]():
            ops.append(WeightDesc(
                name=full + String(".colsum"), arena_offset=off,
                dtype=DType.float32, element_bytes=4,
                global_rows=shape.GLOBAL_N, global_cols=1,
                local_cols=1, data_rows=shape.DATA_N, data_cols=1,
                target_rank=target_rank,
            ))
        comptime if QT.colsum.isa[PerBlockCs]():
            comptime nb_global = shape.GLOBAL_M // QT.fwht_block
            comptime nb_local = shape.DATA_M // QT.fwht_block
            ops.append(WeightDesc(
                name=full + String(".colsum"), arena_offset=off,
                dtype=DType.float32, element_bytes=4,
                global_rows=shape.GLOBAL_N, global_cols=nb_global,
                local_cols=nb_local,
                data_rows=shape.DATA_N, data_cols=nb_local,
                target_rank=target_rank,
            ))
        return

    comptime if quant.isa[PerBlockQuant]():
        comptime QT = quant[PerBlockQuant]
        comptime nb_global = shape.GLOBAL_M // QT.fwht_block
        comptime nb_local = shape.DATA_M // QT.fwht_block
        ops.append(WeightDesc(
            name=full, arena_offset=off,
            dtype=DType.int8, element_bytes=1,
            global_rows=shape.GLOBAL_N, global_cols=shape.GLOBAL_M,
            local_cols=shape.M,
            data_rows=shape.DATA_N, data_cols=shape.DATA_M,
            target_rank=target_rank,
        ))
        off += shape.DATA_N * shape.DATA_M
        ops.append(WeightDesc(
            name=full + String(".scale"), arena_offset=off,
            dtype=DType.float32, element_bytes=4,
            global_rows=shape.GLOBAL_N, global_cols=nb_global,
            local_cols=nb_local,
            data_rows=shape.DATA_N, data_cols=nb_local,
            target_rank=target_rank,
        ))
        off += shape.DATA_N * nb_local * 4
        comptime if QT.colsum.isa[PerBlockCs]():
            ops.append(WeightDesc(
                name=full + String(".colsum"), arena_offset=off,
                dtype=DType.float32, element_bytes=4,
                global_rows=shape.GLOBAL_N, global_cols=nb_global,
                local_cols=nb_local,
                data_rows=shape.DATA_N, data_cols=nb_local,
                target_rank=target_rank,
            ))
        return

    comptime if quant.isa[RouterCenter]():
        comptime QT = quant[RouterCenter]
        ops.append(WeightDesc(
            name=full, arena_offset=off,
            dtype=DType.bfloat16, element_bytes=2,
            global_rows=shape.GLOBAL_N, global_cols=shape.GLOBAL_M,
            local_cols=shape.M,
            data_rows=shape.DATA_N, data_cols=shape.DATA_M,
            target_rank=target_rank,
        ))
        off += shape.DATA_N * shape.DATA_M * 2
        ops.append(WeightDesc(
            name=full + String(".gauge"), arena_offset=off,
            dtype=DType.bfloat16, element_bytes=2,
            global_rows=shape.GLOBAL_M, global_cols=1,
            local_cols=1, data_rows=shape.DATA_M, data_cols=1,
            target_rank=target_rank,
        ))
        off += shape.DATA_M * 2
        comptime if QT.bias_name != StaticString(""):
            ops.append(WeightDesc(
                name=prefix + String(QT.bias_name), arena_offset=off,
                dtype=DType.float32, element_bytes=4,
                global_rows=shape.GLOBAL_N, global_cols=1,
                local_cols=1, data_rows=shape.DATA_N, data_cols=1,
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
