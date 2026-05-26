from std.collections import List

from modeling.model_spec import (
    BF16, Encoding, Shape, ShapeLike, TensorRowSharded,
    DistributionDegree, WeightDesc,
)
from modeling.slot import Slot, SlotGroup, stamp_offsets, emit_descs
from quant.recipe import (
    QuantRecipe, Passthrough, PerRowQuant,
    SplitGamma, NoGamma, SingleSided, PerRowCs, PerBlockCs,
)


trait WeightEncodingPolicy:
    @staticmethod
    def materialize[requested: QuantRecipe]() -> QuantRecipe: ...


struct SourceWeights(WeightEncodingPolicy):
    @staticmethod
    def materialize[requested: QuantRecipe]() -> QuantRecipe:
        return Passthrough()


struct ButterQuantWeights(WeightEncodingPolicy):
    @staticmethod
    def materialize[requested: QuantRecipe]() -> QuantRecipe:
        return requested


comptime WeightSlot[
    Policy: WeightEncodingPolicy,
    encoding: Encoding,
    shape: ShapeLike,
    name: StaticString = "",
    requested: QuantRecipe = Passthrough(),
] = Slot[
    encoding, shape, name,
    Policy.materialize[requested](),
]


comptime SplitGainPerRowCs[fwht: Int, gamma: StaticString]: QuantRecipe = PerRowQuant(
    fwht, SplitGamma(gamma), SingleSided(), PerRowCs(),
)


comptime PlainPerRowBlockCs[fwht: Int]: QuantRecipe = PerRowQuant(
    fwht, NoGamma(), SingleSided(), PerBlockCs(),
)


struct ToyRefs[
    degree: Int, Policy: WeightEncodingPolicy,
](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime D = DistributionDegree[Self.degree]
    comptime Hidden = Shape[16, 1]
    comptime Gate = TensorRowSharded[32, 16, Self.D]
    comptime Down = Shape[16, 32, shard_m=True, degree=Self.degree]

    var input_norm: WeightSlot[
        Self.Policy, BF16, Self.Hidden, "input_layernorm.weight",
    ]
    var gate_proj: WeightSlot[
        Self.Policy, BF16, Self.Gate, "mlp.gate_proj.weight",
        SplitGainPerRowCs[16, "input_layernorm.weight"],
    ]
    var down_proj: WeightSlot[
        Self.Policy, BF16, Self.Down, "mlp.down_proj.weight",
        PlainPerRowBlockCs[16],
    ]


comptime SourceGate = WeightSlot[
    SourceWeights, BF16, Shape[32, 16, shard_n=True, degree=2],
    "mlp.gate_proj.weight", SplitGainPerRowCs[16, "input_layernorm.weight"],
]
comptime BqGate = WeightSlot[
    ButterQuantWeights, BF16, Shape[32, 16, shard_n=True, degree=2],
    "mlp.gate_proj.weight", SplitGainPerRowCs[16, "input_layernorm.weight"],
]


def print_descs(ref descs: List[WeightDesc]):
    for i in range(len(descs)):
        ref d = descs[i]
        print(
            t"  {i}: {d.name} dtype={d.dtype} "
            t"global={d.global_rows}x{d.global_cols} "
            t"data={d.data_rows}x{d.data_cols} off={d.arena_offset}"
        )


def summarize[Policy: WeightEncodingPolicy](label: String):
    var refs = ToyRefs[2, Policy]()
    var bytes = stamp_offsets(refs)
    var descs = List[WeightDesc]()
    _ = emit_descs[ToyRefs[2, Policy]]("", 0, descs)
    print(t"{label}: arena_bytes={bytes} descs={len(descs)}")
    print_descs(descs)


def main():
    comptime assert SourceGate.QUANT.isa[Passthrough](), (
        "source model policy should erase requested quant recipes"
    )
    comptime assert BqGate.QUANT.isa[PerRowQuant](), (
        "bq model policy should preserve requested quant recipes"
    )
    summarize[SourceWeights]("source")
    print("")
    summarize[ButterQuantWeights]("butterquant")
