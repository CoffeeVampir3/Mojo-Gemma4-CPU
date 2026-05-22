from std.utils.variant import Variant


@fieldwise_init
struct NoQuant(Copyable, Movable, ImplicitlyCopyable):
    pass


@fieldwise_init
struct PerRow(Copyable, Movable, ImplicitlyCopyable):
    var fwht_block: Int
    var gamma_name: StaticString


@fieldwise_init
struct PerBlock(Copyable, Movable, ImplicitlyCopyable):
    var fwht_block: Int
    var scale_block: Int


@fieldwise_init
struct RouterCenter(Copyable, Movable, ImplicitlyCopyable):
    pass


comptime QuantSpec = Variant[NoQuant, PerRow, PerBlock, RouterCenter]


struct Slot[name: StaticString, quant: QuantSpec = NoQuant()]:
    comptime NAME = Self.name
    comptime QUANT = Self.quant


@fieldwise_init
struct DemoModel:
    var q_proj:     Slot["q_proj",     PerRow(128, "input_norm")]
    var lm_head:    Slot["lm_head",    PerBlock(128, 32)]
    var router:     Slot["router",     RouterCenter()]
    var input_norm: Slot["input_norm"]


def main():
    print("constructed")
