from kernels.helpers import Binding
from quant.recipe import (
    QuantRecipe, PerRowQuant, PerBlockQuant, RouterCenter,
    PerRowCs, PerBlockCs,
)


@always_inline
def quant_k_block[q: QuantRecipe]() -> Int:
    comptime if q.isa[PerRowQuant]():
        return q[PerRowQuant].fwht_block
    comptime if q.isa[PerBlockQuant]():
        return q[PerBlockQuant].fwht_block
    return 0


@always_inline
def quant_per_block[q: QuantRecipe]() -> Bool:
    return q.isa[PerBlockQuant]()


@always_inline
def quant_has_colsum[q: QuantRecipe]() -> Bool:
    comptime if q.isa[PerRowQuant]():
        comptime QT = q[PerRowQuant]
        return QT.colsum.isa[PerRowCs]() or QT.colsum.isa[PerBlockCs]()
    comptime if q.isa[PerBlockQuant]():
        return q[PerBlockQuant].colsum.isa[PerBlockCs]()
    return False


@always_inline
def router_has_bias[q: QuantRecipe]() -> Bool:
    comptime if q.isa[RouterCenter]():
        return q[RouterCenter].bias_name != StaticString("")
    return False


@fieldwise_init
struct ButterquantWeight[quant: QuantRecipe, n: Int, m: Int, tp: Int](
    Copyable, ImplicitlyCopyable,
):
    comptime K_BLOCK = quant_k_block[Self.quant]()
    comptime PER_BLOCK = quant_per_block[Self.quant]()
    comptime HAS_COLSUM = quant_has_colsum[Self.quant]()

    var data: Binding[Int8, Self.tp]
    var scale: Binding[Float32, Self.tp]
    var colsum: Binding[Float32, Self.tp]

    @always_inline
    def colsum_checked(self) -> Binding[Float32, Self.tp]:
        comptime assert Self.HAS_COLSUM, (
            "ButterquantWeight recipe declared no colsum but tried to access one.")
        return self.colsum


@fieldwise_init
struct ButterquantRouter[quant: QuantRecipe, n: Int, m: Int, tp: Int](
    Copyable, ImplicitlyCopyable,
):
    comptime HAS_BIAS = router_has_bias[Self.quant]()

    var centered: Binding[BFloat16, Self.tp]
    var gauge: Binding[BFloat16, Self.tp]
    var bias: Binding[Float32, Self.tp]

    @always_inline
    def bias_checked(self) -> Binding[Float32, Self.tp]:
        comptime assert Self.HAS_BIAS, (
            "ButterquantRouter recipe declared no bias but tried to access one.")
        return self.bias


@fieldwise_init
struct ButterquantActivation[tp: Int](Copyable, ImplicitlyCopyable):
    var data: Binding[Int8, Self.tp]
    var scale: Binding[Float32, Self.tp]
