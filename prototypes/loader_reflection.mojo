from std.collections import InlineArray
from std.memory import UnsafePointer
from std.reflection import reflect


trait Encoding:
    comptime DTYPE: DType
    comptime ELEMENT_BYTES: Int


struct BF16(Encoding):
    comptime DTYPE = DType.bfloat16
    comptime ELEMENT_BYTES = 2


struct F32(Encoding):
    comptime DTYPE = DType.float32
    comptime ELEMENT_BYTES = 4


trait ShapeLike:
    comptime GLOBAL_N: Int
    comptime GLOBAL_M: Int
    comptime DATA_N: Int
    comptime DATA_M: Int

    @staticmethod
    def bytes[E: Encoding]() -> Int: ...


struct Shape[
    global_n: Int, global_m: Int,
    shard_n: Bool = False, shard_m: Bool = False,
    degree: Int = 1,
](ShapeLike, Copyable, ImplicitlyCopyable):
    comptime GLOBAL_N = Self.global_n
    comptime GLOBAL_M = Self.global_m
    comptime DATA_N = Self.global_n // Self.degree if Self.shard_n else Self.global_n
    comptime DATA_M = Self.global_m // Self.degree if Self.shard_m else Self.global_m

    @staticmethod
    def bytes[E: Encoding]() -> Int:
        return Self.DATA_N * Self.DATA_M * E.ELEMENT_BYTES


comptime DISTRIBUTED = -1
comptime DEFAULT_ALIGN = 64


@always_inline
def align_up(v: Int, a: Int = DEFAULT_ALIGN) -> Int:
    return ((v + a - 1) // a) * a


trait SlotLike:
    comptime E: Encoding
    comptime S: ShapeLike
    comptime NAME: StaticString
    comptime TARGET_RANK: Int

    @always_inline
    def set_offset(mut self, off: Int): ...

    @always_inline
    def get_offset(self) -> Int: ...


@fieldwise_init
struct Slot[
    E_: Encoding, S_: ShapeLike, name_: StaticString,
    target_rank_: Int = DISTRIBUTED,
](SlotLike, Copyable, ImplicitlyCopyable):
    comptime E = Self.E_
    comptime S = Self.S_
    comptime NAME = Self.name_
    comptime TARGET_RANK = Self.target_rank_

    var offset: Int

    @always_inline
    def set_offset(mut self, off: Int):
        self.offset = off

    @always_inline
    def get_offset(self) -> Int:
        return self.offset


@fieldwise_init
struct WeightDesc(Copyable):
    var name: String
    var arena_offset: Int
    var dtype: DType
    var element_bytes: Int
    var data_rows: Int
    var data_cols: Int
    var global_rows: Int
    var global_cols: Int
    var target_rank: Int


@fieldwise_init
struct InitResult(Copyable, Movable):
    var bytes_used: Int
    var slot_count: Int


def init_slot_struct[T: AnyType](
    mut t: T, prefix: String, region_offset: Int, mut ops: List[WeightDesc],
) -> InitResult:
    """Walk T's fields. For each SlotLike field:
        - stamp the slot's runtime offset (region_offset + running cursor)
        - append a WeightDesc to `ops` (the loader's I/O queue)
    Returns the bytes consumed and the number of slots populated. The
    loader and the runner both end up looking at the same single source
    of truth: T's field declarations, in declaration order."""
    var off = 0
    var n = 0
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, SlotLike):
            ref slot = reflect[T].field_ref[i](t)
            slot.set_offset(off)
            ops.append(WeightDesc(
                name=prefix + String(FT.NAME),
                arena_offset=region_offset + off,
                dtype=FT.E.DTYPE,
                element_bytes=FT.E.ELEMENT_BYTES,
                data_rows=FT.S.DATA_N,
                data_cols=FT.S.DATA_M,
                global_rows=FT.S.GLOBAL_N,
                global_cols=FT.S.GLOBAL_M,
                target_rank=FT.TARGET_RANK,
            ))
            off = align_up(off + FT.S.bytes[FT.E]())
            n += 1
    return InitResult(bytes_used=off, slot_count=n)


@fieldwise_init
struct BindContext[degree: Int](Copyable, ImplicitlyCopyable):
    var arena_bases: InlineArray[Int, Self.degree]
    var layer_base: Int

    @always_inline
    def with_layer(self, lb: Int) -> Self:
        var c = self
        c.layer_base = lb
        return c


@fieldwise_init
struct NumaPointerArray[dtype: DType, degree: Int](Copyable, ImplicitlyCopyable):
    var ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    var bases: InlineArray[Int, Self.degree]

    @always_inline
    def __getitem__(self, rank: Int) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        return UnsafePointer[Scalar[Self.dtype], MutAnyOrigin](
            unsafe_from_address=Int(self.ptr) + self.bases[rank] - self.bases[0])


@always_inline
def slot_ranks[
    E: Encoding, S: ShapeLike, name: StaticString, target_rank: Int,
    degree: Int,
](slot: Slot[E, S, name, target_rank], ctx: BindContext[degree],
) -> NumaPointerArray[E.DTYPE, degree]:
    var addr = ctx.layer_base + slot.offset
    return NumaPointerArray[E.DTYPE, degree](
        UnsafePointer[Scalar[E.DTYPE], MutAnyOrigin](unsafe_from_address=addr),
        ctx.arena_bases)


comptime HIDDEN = 2816
comptime INTERMEDIATE = 2112
comptime NUM_EXPERTS = 128
comptime Q_DIM_SLIDING = 4096
comptime KV_DIM_SLIDING = 2048


comptime W[S: ShapeLike, name: StaticString] = Slot[BF16, S, name]


@fieldwise_init
struct AttentionWeights[degree: Int](Copyable, ImplicitlyCopyable):
    var input_norm: W[Shape[HIDDEN, 1], "input_layernorm.weight"]
    var q_proj: W[
        Shape[Q_DIM_SLIDING, HIDDEN, shard_n=True, degree=Self.degree],
        "self_attn.q_proj.weight",
    ]
    var k_proj: W[
        Shape[KV_DIM_SLIDING, HIDDEN, shard_n=True, degree=Self.degree],
        "self_attn.k_proj.weight",
    ]
    var v_proj: W[
        Shape[KV_DIM_SLIDING, HIDDEN, shard_n=True, degree=Self.degree],
        "self_attn.v_proj.weight",
    ]
    var o_proj: W[
        Shape[HIDDEN, Q_DIM_SLIDING, shard_m=True, degree=Self.degree],
        "self_attn.o_proj.weight",
    ]
    var q_norm: W[Shape[256, 1], "self_attn.q_norm.weight"]
    var k_norm: W[Shape[256, 1], "self_attn.k_norm.weight"]


@fieldwise_init
struct MoEWeights[degree: Int](Copyable, ImplicitlyCopyable):
    var pre_ffn_norm: W[Shape[HIDDEN, 1], "pre_feedforward_layernorm.weight"]
    var router_proj: W[
        Shape[NUM_EXPERTS, HIDDEN], "router.proj.weight",
    ]
    var router_scale: W[Shape[HIDDEN, 1], "router.scale"]
    var post_ffn_norm: W[Shape[HIDDEN, 1], "post_feedforward_layernorm.weight"]


def make_uninit_attention[degree: Int]() -> AttentionWeights[degree]:
    return AttentionWeights[degree](
        input_norm = W[Shape[HIDDEN, 1], "input_layernorm.weight"](-1),
        q_proj = W[Shape[Q_DIM_SLIDING, HIDDEN, shard_n=True, degree=degree],
                   "self_attn.q_proj.weight"](-1),
        k_proj = W[Shape[KV_DIM_SLIDING, HIDDEN, shard_n=True, degree=degree],
                   "self_attn.k_proj.weight"](-1),
        v_proj = W[Shape[KV_DIM_SLIDING, HIDDEN, shard_n=True, degree=degree],
                   "self_attn.v_proj.weight"](-1),
        o_proj = W[Shape[HIDDEN, Q_DIM_SLIDING, shard_m=True, degree=degree],
                   "self_attn.o_proj.weight"](-1),
        q_norm = W[Shape[256, 1], "self_attn.q_norm.weight"](-1),
        k_norm = W[Shape[256, 1], "self_attn.k_norm.weight"](-1),
    )


def make_uninit_moe[degree: Int]() -> MoEWeights[degree]:
    return MoEWeights[degree](
        pre_ffn_norm = W[Shape[HIDDEN, 1], "pre_feedforward_layernorm.weight"](-1),
        router_proj = W[Shape[NUM_EXPERTS, HIDDEN], "router.proj.weight"](-1),
        router_scale = W[Shape[HIDDEN, 1], "router.scale"](-1),
        post_ffn_norm = W[Shape[HIDDEN, 1], "post_feedforward_layernorm.weight"](-1),
    )


def demo_dispatch_attention[degree: Int](
    weights: AttentionWeights[degree], ctx: BindContext[degree],
):
    var input_norm = slot_ranks(weights.input_norm, ctx)
    var q = slot_ranks(weights.q_proj, ctx)
    var k = slot_ranks(weights.k_proj, ctx)
    var v = slot_ranks(weights.v_proj, ctx)
    var o = slot_ranks(weights.o_proj, ctx)
    print("  input_norm rank0:", Int(input_norm[0]))
    print("  q_proj    rank0:", Int(q[0]),  " rank1:", Int(q[1]))
    print("  k_proj    rank0:", Int(k[0]),  " rank1:", Int(k[1]))
    print("  v_proj    rank0:", Int(v[0]))
    print("  o_proj    rank0:", Int(o[0]))


def main():
    print("=== loader walks reflection, stamps offsets, emits ops ===")

    var attn = make_uninit_attention[2]()
    var moe = make_uninit_moe[2]()

    var ops = List[WeightDesc]()
    var attn_layer_base = 0
    var attn_init = init_slot_struct(
        attn, prefix="model.layers.0.", region_offset=attn_layer_base, ops=ops)
    var moe_layer_base = align_up(attn_init.bytes_used)
    var moe_init = init_slot_struct(
        moe, prefix="model.layers.0.", region_offset=moe_layer_base, ops=ops)

    print("attn slots:", attn_init.slot_count, " bytes:", attn_init.bytes_used)
    print("moe  slots:", moe_init.slot_count,  " bytes:", moe_init.bytes_used)
    print()
    print("--- generated WeightDesc list (single source of truth: the structs above) ---")
    for i in range(len(ops)):
        var d = ops[i].copy()
        print(" ", d.name, "@", d.arena_offset,
              " dtype=", d.dtype,
              " shape=[", d.global_rows, ",", d.global_cols, "]",
              " local=[", d.data_rows, ",", d.data_cols, "]")

    print()
    print("=== runner uses the same struct via natural field access ===")
    var bases = InlineArray[Int, 2](uninitialized=True)
    bases[0] = 0x10000000
    bases[1] = 0x20000000

    var ctx = BindContext[2](
        arena_bases=bases,
        layer_base=bases[0] + attn_layer_base,
    )
    demo_dispatch_attention(attn, ctx)

    print()
    print("=== sanity: per-slot offset matches the loader's arena_offset ===")
    print("  attn.q_proj.offset =", attn.q_proj.offset,
          " (loader recorded arena_offset =", ops[1].arena_offset, ")")
    print("  attn.o_proj.offset =", attn.o_proj.offset,
          " (loader recorded arena_offset =", ops[4].arena_offset, ")")
    print("  moe.router_proj.offset =", moe.router_proj.offset,
          " (region-local; loader arena_offset =", ops[7].arena_offset, ")")
