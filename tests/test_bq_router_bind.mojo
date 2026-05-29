from std.memory import alloc
from std.os import abort

from kernels.helpers import ArenaBases
from modeling.model_spec import BF16, Shape
from modeling.slot import Slot, BindContext
from quant.recipe import RouterCenter, SoftmaxRouterCenter
from quant.manifest import member_rel_off, QuantRole


comptime EXPERTS = 32
comptime HIDDEN = 64
comptime SHAPE = Shape[EXPERTS, HIDDEN]

comptime SOFTMAX = SoftmaxRouterCenter()
comptime CENTER_NOBIAS = RouterCenter("")
comptime CENTER_BIAS = RouterCenter("router.bias")


def check(ok: Bool, msg: String):
    if not ok:
        abort("FAIL: " + msg)


def main():
    var bytes = EXPERTS * HIDDEN * 2 + HIDDEN * 2 + EXPERTS * 4
    var arena = alloc[UInt8](bytes).as_any_origin()
    var base = Int(arena)

    var bases = ArenaBases[1].uninitialized()
    bases[0] = base
    var ctx = BindContext[1](arena_bases=bases, layer_base=base)

    # §13.5 shift-invariant softmax router: centered weight only.
    var s_slot = Slot[BF16, SHAPE, "router.proj.weight", SOFTMAX](0)
    var sb = s_slot.bq_router(ctx)
    check(Int(sb.centered[0]) == base, "softmax centered must point at slot base")
    check(not sb.gauge, "softmax router must not bind a gauge")
    check(not sb.bias, "softmax router must not bind a bias")

    # §13.2 centered router, no bias: centered + gauge.
    var n_slot = Slot[BF16, SHAPE, "router.proj.weight", CENTER_NOBIAS](0)
    var nb = n_slot.bq_router(ctx)
    check(Int(nb.centered[0]) == base, "centered must point at slot base")
    check(Bool(nb.gauge), "centered router must bind a gauge")
    comptime G_OFF = member_rel_off[BF16, SHAPE, CENTER_NOBIAS, QuantRole.GAUGE]()
    check(Int(nb.gauge.value()[0]) == base + G_OFF, "gauge offset mismatch")
    check(not nb.bias, "no-bias router must not bind a bias")

    # §13.2 centered router with bias: centered + gauge + bias.
    var b_slot = Slot[BF16, SHAPE, "router.proj.weight", CENTER_BIAS](0)
    var bb = b_slot.bq_router(ctx)
    check(Bool(bb.gauge), "biased router must bind a gauge")
    check(Bool(bb.bias), "biased router must bind a bias")
    comptime B_OFF = member_rel_off[BF16, SHAPE, CENTER_BIAS, QuantRole.BIAS]()
    check(Int(bb.bias.value()[0]) == base + B_OFF, "bias offset mismatch")

    arena.free()
    print("bq router bind tests passed")
