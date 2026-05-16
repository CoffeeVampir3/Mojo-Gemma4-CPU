from std.collections import InlineArray
from std.memory import UnsafePointer


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


@fieldwise_init
struct StaticView[E: Encoding, S: ShapeLike](Copyable, ImplicitlyCopyable):
    var ptr: UnsafePointer[Scalar[Self.E.DTYPE], MutAnyOrigin]

    @always_inline
    def as_ptr(self) -> UnsafePointer[Scalar[Self.E.DTYPE], MutAnyOrigin]:
        return self.ptr


@fieldwise_init
struct NumaPointerArray[dtype: DType, degree: Int](Copyable, ImplicitlyCopyable):
    var ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    var bases: InlineArray[Int, Self.degree]

    @always_inline
    def __getitem__(self, rank: Int) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        return UnsafePointer[Scalar[Self.dtype], MutAnyOrigin](
            unsafe_from_address=Int(self.ptr) + self.bases[rank] - self.bases[0])


@fieldwise_init
struct BindContext[degree: Int](Copyable, ImplicitlyCopyable):
    var arena_bases: InlineArray[Int, Self.degree]
    var layer_base: Int
    var scratch_base: Int

    @always_inline
    def with_layer(self, lb: Int) -> Self:
        var c = self
        c.layer_base = lb
        return c

    @always_inline
    def with_scratch(self, sb: Int) -> Self:
        var c = self
        c.scratch_base = sb
        return c


trait SlotOrigin(Copyable, ImplicitlyCopyable):
    @always_inline
    def address[degree: Int](self, ctx: BindContext[degree]) -> Int: ...


@fieldwise_init
struct WeightOrigin(SlotOrigin):
    var layer_offset: Int

    @always_inline
    def address[degree: Int](self, ctx: BindContext[degree]) -> Int:
        return ctx.layer_base + self.layer_offset


@fieldwise_init
struct StateOrigin(SlotOrigin):
    var arena_offset: Int

    @always_inline
    def address[degree: Int](self, ctx: BindContext[degree]) -> Int:
        return ctx.arena_bases[0] + self.arena_offset


@fieldwise_init
struct ScratchOrigin[scratch_offset: Int](SlotOrigin):
    @always_inline
    def address[degree: Int](self, ctx: BindContext[degree]) -> Int:
        return ctx.scratch_base + Self.scratch_offset


@fieldwise_init
struct Slot[E: Encoding, S: ShapeLike, O: SlotOrigin](
    Copyable, ImplicitlyCopyable
):
    var origin: Self.O

    @always_inline
    def bound[degree: Int](
        self, ctx: BindContext[degree],
    ) -> StaticView[Self.E, Self.S]:
        var addr = self.origin.address(ctx)
        return StaticView[Self.E, Self.S](
            UnsafePointer[Scalar[Self.E.DTYPE], MutAnyOrigin](
                unsafe_from_address=addr))

    @always_inline
    def ranks[degree: Int](
        self, ctx: BindContext[degree],
    ) -> NumaPointerArray[Self.E.DTYPE, degree]:
        var addr = self.origin.address(ctx)
        return NumaPointerArray[Self.E.DTYPE, degree](
            UnsafePointer[Scalar[Self.E.DTYPE], MutAnyOrigin](
                unsafe_from_address=addr),
            ctx.arena_bases)


comptime W[S: ShapeLike] = Slot[BF16, S, WeightOrigin]
comptime St[S: ShapeLike] = Slot[BF16, S, StateOrigin]
comptime Sc[E: Encoding, S: ShapeLike, off: Int] = Slot[E, S, ScratchOrigin[off]]


comptime HIDDEN = 2816
comptime INTERMEDIATE = 2112
comptime NUM_EXPERTS = 128
comptime Q_DIM_SLIDING = 4096
comptime KV_DIM_SLIDING = 2048
comptime MAX_SEQ_LEN = 4096
comptime SLIDING_WINDOW = 1024


@fieldwise_init
struct DemoBody[degree: Int](Copyable, ImplicitlyCopyable):
    var input_norm:  W[Shape[HIDDEN, 1]]
    var q_proj:      W[Shape[Q_DIM_SLIDING, HIDDEN, shard_n=True, degree=Self.degree]]
    var k_proj:      W[Shape[KV_DIM_SLIDING, HIDDEN, shard_n=True, degree=Self.degree]]
    var v_proj:      W[Shape[KV_DIM_SLIDING, HIDDEN, shard_n=True, degree=Self.degree]]
    var gate_proj:   W[Shape[INTERMEDIATE, HIDDEN, shard_n=True, degree=Self.degree]]
    var router_proj: W[Shape[NUM_EXPERTS, HIDDEN]]


@fieldwise_init
struct DemoState[degree: Int](Copyable, ImplicitlyCopyable):
    var x_main:     St[Shape[MAX_SEQ_LEN, HIDDEN]]
    var x_residual: St[Shape[MAX_SEQ_LEN, HIDDEN]]
    var sliding_k:  St[Shape[SLIDING_WINDOW, KV_DIM_SLIDING, shard_m=True, degree=Self.degree]]
    var sliding_v:  St[Shape[SLIDING_WINDOW, KV_DIM_SLIDING, shard_m=True, degree=Self.degree]]


def demo_attention_dispatch[degree: Int](
    body: DemoBody[degree], state: DemoState[degree], ctx: BindContext[degree],
):
    var x = state.x_residual.ranks(ctx)
    var input_norm_w = body.input_norm.ranks(ctx)
    var q_w = body.q_proj.ranks(ctx)
    var k_w = body.k_proj.ranks(ctx)
    var v_w = body.v_proj.ranks(ctx)
    var router_w = body.router_proj.ranks(ctx)

    print("  x_residual rank0 ptr:    ", Int(x[0]))
    print("  x_residual rank1 ptr:    ", Int(x[1]))
    print("  input_norm  rank0 ptr:   ", Int(input_norm_w[0]))
    print("  q_proj      rank0 ptr:   ", Int(q_w[0]))
    print("  q_proj      rank1 ptr:   ", Int(q_w[1]))
    print("  k_proj      rank0 ptr:   ", Int(k_w[0]))
    print("  v_proj      rank0 ptr:   ", Int(v_w[0]))
    print("  router_proj rank0 ptr:   ", Int(router_w[0]))


def demo_scratch_dispatch[degree: Int](ctx: BindContext[degree]):
    var q_buf = Sc[BF16, Shape[MAX_SEQ_LEN, Q_DIM_SLIDING // degree], 0](
        ScratchOrigin[0]())
    var kv_buf = Sc[BF16, Shape[MAX_SEQ_LEN, 2 * KV_DIM_SLIDING // degree], 256](
        ScratchOrigin[256]())

    var q = q_buf.ranks(ctx)
    var kv = kv_buf.ranks(ctx)
    print("  scratch q  rank0:        ", Int(q[0]))
    print("  scratch kv rank0:        ", Int(kv[0]))


def main():
    var bases = InlineArray[Int, 2](uninitialized=True)
    bases[0] = 0x10000000
    bases[1] = 0x20000000

    var ctx = BindContext[2](
        arena_bases=bases,
        layer_base=0x10001000,
        scratch_base=0x10800000,
    )

    var body = DemoBody[2](
        input_norm  = W[Shape[HIDDEN, 1]](WeightOrigin(0)),
        q_proj      = W[Shape[Q_DIM_SLIDING, HIDDEN, shard_n=True, degree=2]](
                          WeightOrigin(0x100)),
        k_proj      = W[Shape[KV_DIM_SLIDING, HIDDEN, shard_n=True, degree=2]](
                          WeightOrigin(0x200)),
        v_proj      = W[Shape[KV_DIM_SLIDING, HIDDEN, shard_n=True, degree=2]](
                          WeightOrigin(0x300)),
        gate_proj   = W[Shape[INTERMEDIATE, HIDDEN, shard_n=True, degree=2]](
                          WeightOrigin(0x400)),
        router_proj = W[Shape[NUM_EXPERTS, HIDDEN]](WeightOrigin(0x500)),
    )

    var state = DemoState[2](
        x_main     = St[Shape[MAX_SEQ_LEN, HIDDEN]](StateOrigin(0x600000)),
        x_residual = St[Shape[MAX_SEQ_LEN, HIDDEN]](StateOrigin(0x600000 + MAX_SEQ_LEN * HIDDEN * 2)),
        sliding_k  = St[Shape[SLIDING_WINDOW, KV_DIM_SLIDING, shard_m=True, degree=2]](
                          StateOrigin(0x700000)),
        sliding_v  = St[Shape[SLIDING_WINDOW, KV_DIM_SLIDING, shard_m=True, degree=2]](
                          StateOrigin(0x700000 + SLIDING_WINDOW * (KV_DIM_SLIDING // 2) * 2)),
    )

    print("=== weight & state slots ===")
    demo_attention_dispatch(body, state, ctx)

    print("=== scratch slots ===")
    demo_scratch_dispatch(ctx)

    print("=== shape facts (still on the type, not restated at the call site) ===")
    print("  q_proj DATA_N (per rank):", Shape[Q_DIM_SLIDING, HIDDEN, shard_n=True, degree=2].DATA_N)
    print("  q_proj bytes  (per rank):", Shape[Q_DIM_SLIDING, HIDDEN, shard_n=True, degree=2].bytes[BF16]())
    print("  router_proj DATA_N:      ", Shape[NUM_EXPERTS, HIDDEN].DATA_N)
