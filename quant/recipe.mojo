from std.utils.variant import Variant


@fieldwise_init
struct NoGamma(Copyable, Movable, ImplicitlyCopyable):
    pass


@fieldwise_init
struct SplitGamma(Copyable, Movable, ImplicitlyCopyable):
    var name: StaticString


@fieldwise_init
struct AbsorbedGamma(Copyable, Movable, ImplicitlyCopyable):
    var name: StaticString


comptime GammaMode = Variant[NoGamma, SplitGamma, AbsorbedGamma]


@fieldwise_init
struct SingleSided(Copyable, Movable, ImplicitlyCopyable):
    pass


@fieldwise_init
struct TwoSided(Copyable, Movable, ImplicitlyCopyable):
    var m_block: Int


comptime RotationMode = Variant[SingleSided, TwoSided]


@fieldwise_init
struct NoColsum(Copyable, Movable, ImplicitlyCopyable):
    pass


@fieldwise_init
struct PerRowCs(Copyable, Movable, ImplicitlyCopyable):
    pass


@fieldwise_init
struct PerBlockCs(Copyable, Movable, ImplicitlyCopyable):
    pass


comptime ColsumMode = Variant[NoColsum, PerRowCs, PerBlockCs]
comptime PerBlockColsumMode = Variant[NoColsum, PerBlockCs]


@fieldwise_init
struct PerRowQuant(Copyable, Movable, ImplicitlyCopyable):
    var fwht_block: Int
    var gamma: GammaMode
    var rotation: RotationMode
    var colsum: ColsumMode


@fieldwise_init
struct PerBlockQuant(Copyable, Movable, ImplicitlyCopyable):
    var fwht_block: Int
    var gamma: GammaMode
    var rotation: RotationMode
    var colsum: PerBlockColsumMode


@fieldwise_init
struct RouterCenter(Copyable, Movable, ImplicitlyCopyable):
    """§13.2 centered-bf16 router. `bias_name=""` skips the bias output
    (algebraically valid; §13.2 minus the additive bias term). Non-empty
    `bias_name` reads the named source tensor and writes it as f32 per §13.4."""
    var bias_name: StaticString


@fieldwise_init
struct Passthrough(Copyable, Movable, ImplicitlyCopyable):
    pass


comptime QuantRecipe = Variant[
    Passthrough, PerRowQuant, PerBlockQuant, RouterCenter,
]
