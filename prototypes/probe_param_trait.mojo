from std.collections import InlineArray


@fieldwise_init
struct Ctx[degree: Int](Copyable, ImplicitlyCopyable):
    var bases: InlineArray[Int, Self.degree]
    var layer_base: Int


trait Origin:
    @always_inline
    def address[degree: Int](self, ctx: Ctx[degree]) -> Int: ...


@fieldwise_init
struct WeightO(Origin, Copyable, ImplicitlyCopyable):
    var off: Int

    @always_inline
    def address[degree: Int](self, ctx: Ctx[degree]) -> Int:
        return ctx.layer_base + self.off


@fieldwise_init
struct StateO(Origin, Copyable, ImplicitlyCopyable):
    var off: Int

    @always_inline
    def address[degree: Int](self, ctx: Ctx[degree]) -> Int:
        return ctx.bases[0] + self.off


@fieldwise_init
struct Slot[O: Origin & Copyable & ImplicitlyCopyable](Copyable, ImplicitlyCopyable):
    var origin: Self.O

    @always_inline
    def at[degree: Int](self, ctx: Ctx[degree]) -> Int:
        return self.origin.address(ctx)


def main():
    var bases = InlineArray[Int, 2](uninitialized=True)
    bases[0] = 100
    bases[1] = 200
    var ctx = Ctx[2](bases=bases, layer_base=10)

    var w = Slot[WeightO](WeightO(off=5))
    var s = Slot[StateO](StateO(off=7))
    print("weight slot at:", w.at(ctx))   # 10 + 5 = 15
    print("state slot at:", s.at(ctx))    # 100 + 7 = 107
