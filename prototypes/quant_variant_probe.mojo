from std.utils.variant import Variant


@fieldwise_init
struct NoQuant(Copyable, Movable, ImplicitlyCopyable):
    pass


@fieldwise_init
struct PerRowConcrete(Copyable, Movable, ImplicitlyCopyable):
    var fwht_block: Int
    var gamma_name: String


@fieldwise_init
struct PerBlockConcrete(Copyable, Movable, ImplicitlyCopyable):
    var fwht_block: Int
    var scale_block: Int


@fieldwise_init
struct RouterCenter(Copyable, Movable, ImplicitlyCopyable):
    pass


comptime QuantSpec = Variant[NoQuant, PerRowConcrete, PerBlockConcrete, RouterCenter]


def main():
    var v1: QuantSpec = PerRowConcrete(128, "input_norm")
    print(t"isa NoQuant: {v1.isa[NoQuant]()}")
    print(t"isa PerRow: {v1.isa[PerRowConcrete]()}")
    print(t"isa PerBlock: {v1.isa[PerBlockConcrete]()}")
    var v2: QuantSpec = RouterCenter()
    print(t"v2 isa Router: {v2.isa[RouterCenter]()}")
