from kernels.helpers import Binding


@fieldwise_init
struct ButterquantEncoding(Copyable, Movable, ImplicitlyCopyable):
    var n: Int
    var m: Int
    var k_block: Int
    var per_block_scale: Bool
    var has_colsum: Bool


@fieldwise_init
struct ButterquantWeight[encoding: ButterquantEncoding, tp: Int](
    Copyable, ImplicitlyCopyable,
):
    var data: Binding[Int8, Self.tp]
    var scale: Binding[Float32, Self.tp]
    var colsum: Binding[Float32, Self.tp]

    @always_inline
    def colsum_checked(self) -> Binding[Float32, Self.tp]:
        comptime assert Self.encoding.has_colsum, (
            "ButterquantWeight encoding declared no colsum but tried to access one.")
        return self.colsum


@fieldwise_init
struct ButterquantActivation[tp: Int](Copyable, ImplicitlyCopyable):
    var data: Binding[Int8, Self.tp]
    var scale: Binding[Float32, Self.tp]
