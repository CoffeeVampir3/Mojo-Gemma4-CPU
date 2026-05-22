from std.utils.variant import Variant


@fieldwise_init
struct Passthrough(Copyable, Movable, ImplicitlyCopyable):
    pass


@fieldwise_init
struct PerRow(Copyable, Movable, ImplicitlyCopyable):
    var fwht_block: Int
    var gamma_name: StaticString
    var m_block: Int
    var want_cs: Bool


comptime QuantRecipe = Variant[Passthrough, PerRow]


comptime QkvProj[gamma: StaticString]: QuantRecipe = PerRow(128, gamma, 0, True)
comptime OProj[head_dim: Int]: QuantRecipe = PerRow(head_dim, "", 0, True)


struct Slot[name: StaticString, quant: QuantRecipe = Passthrough()]:
    pass


@fieldwise_init
struct Model:
    var q: Slot["q_proj", QkvProj["input_layernorm.weight"]]
    var o: Slot["o_proj", OProj[256]]


def main():
    print("ok")
