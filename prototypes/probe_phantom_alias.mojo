from std.collections import InlineArray


trait Encoding:
    comptime DTYPE: DType
    comptime ELEMENT_BYTES: Int


struct BF16(Encoding):
    comptime DTYPE = DType.bfloat16
    comptime ELEMENT_BYTES = 2


trait ShapeLike:
    comptime DATA_N: Int
    comptime DATA_M: Int


struct Shape[n: Int, m: Int](ShapeLike, Copyable, ImplicitlyCopyable):
    comptime DATA_N = Self.n
    comptime DATA_M = Self.m


@fieldwise_init
struct Ctx[degree: Int](Copyable, ImplicitlyCopyable):
    var bases: InlineArray[Int, Self.degree]
    var layer_base: Int
    var scratch_base: Int


trait Origin(Copyable, ImplicitlyCopyable):
    @always_inline
    def address[degree: Int](self, ctx: Ctx[degree]) -> Int: ...


@fieldwise_init
struct WeightO(Origin):
    var off: Int

    @always_inline
    def address[degree: Int](self, ctx: Ctx[degree]) -> Int:
        return ctx.layer_base + self.off


@fieldwise_init
struct ScratchO[off: Int](Origin):
    @always_inline
    def address[degree: Int](self, ctx: Ctx[degree]) -> Int:
        return ctx.scratch_base + Self.off


@fieldwise_init
struct Slot[E: Encoding, S: ShapeLike, O: Origin](Copyable, ImplicitlyCopyable):
    var origin: Self.O

    @always_inline
    def at[degree: Int](self, ctx: Ctx[degree]) -> Int:
        return self.origin.address(ctx)


comptime W[S: ShapeLike] = Slot[BF16, S, WeightO]


@fieldwise_init
struct DemoBody(Copyable, ImplicitlyCopyable):
    var input_norm: W[Shape[2816, 1]]
    var gate_proj:  W[Shape[2112, 2816]]


def main():
    var bases = InlineArray[Int, 2](uninitialized=True)
    bases[0] = 1000
    bases[1] = 2000
    var ctx = Ctx[2](bases=bases, layer_base=500, scratch_base=10000)

    var body = DemoBody(
        input_norm=W[Shape[2816, 1]](WeightO(off=0)),
        gate_proj=W[Shape[2112, 2816]](WeightO(off=64)),
    )
    print("input_norm at:", body.input_norm.at(ctx))     # 500 + 0
    print("gate_proj at:", body.gate_proj.at(ctx))       # 500 + 64

    # phantom scratch slot — no runtime field
    var sx = Slot[BF16, Shape[1024, 1], ScratchO[256]](ScratchO[256]())
    print("scratch at:", sx.at(ctx))                     # 10000 + 256
