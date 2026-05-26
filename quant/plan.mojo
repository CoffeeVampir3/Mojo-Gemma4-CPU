from std.utils.variant import Variant


struct ColsumKind:
    comptime NONE = 0
    comptime PER_ROW = 1
    comptime PER_BLOCK = 2


@fieldwise_init
struct SlotIdentity(Copyable, Movable):
    """`name` is the full safetensors key (e.g.
    `model.language_model.layers.5.self_attn.q_proj.weight`). `local_name`
    is the slot's own annotation (e.g. `self_attn.q_proj.weight`), useful
    when logging without the layer prefix noise. `layer_idx` is -1 for
    layer-agnostic slots (e.g. `model.language_model.norm.weight`)."""
    var name: String
    var local_name: String
    var layer_idx: Int
    var shard: Int
    var src_offset: Int
    var src_dtype: DType
    var rows: Int
    var cols: Int
    var weight_off: Int


@fieldwise_init
struct GammaRef(Copyable, Movable):
    """Empty `name` signals no gamma; `absorbed` selects between the
    sqrt-abs split factor and the raw absorbed gain."""
    var name: String
    var absorbed: Bool

    @staticmethod
    def none() -> Self:
        return Self(String(""), False)

    def is_present(self) -> Bool:
        return self.name.byte_length() > 0


@fieldwise_init
struct PassthroughPlan(Copyable, Movable):
    var id: SlotIdentity
    var byte_count: Int


@fieldwise_init
struct QuantPlan(Copyable, Movable):
    var id: SlotIdentity
    var per_block: Bool
    var fwht_block: Int
    var two_sided_m: Int
    var gamma: GammaRef
    var colsum_kind: Int
    var scale_off: Int
    var cs_off: Int


@fieldwise_init
struct RouterPlan(Copyable, Movable):
    """Empty `bias_name` skips bias emission (§13.2 minus the bias term);
    a non-empty `bias_name` reads the source tensor and writes it as f32
    per §13.4."""
    var id: SlotIdentity
    var gauge_off: Int
    var bias_name: String
    var bias_off: Int


comptime SlotPlan = Variant[PassthroughPlan, QuantPlan, RouterPlan]


@fieldwise_init
struct ScratchCapacity(TrivialRegisterPassable):
    """Worst-case staging sizes computed during plan walk and consumed
    by the per-worker scratch allocator. Trivially register-passable so it
    can ride in a BurstKernel mailbox."""
    var max_panel_rows: Int
    var max_cols: Int
    var max_src_bytes_per: Int
    var max_scale_per_row: Int
    var max_cs_per_row: Int

    @staticmethod
    def zero(max_panel_rows: Int) -> Self:
        return Self(max_panel_rows, 0, 0, 1, 0)

    @always_inline
    def absorb_quant(mut self, p: QuantPlan, src_bytes_per: Int):
        if p.id.cols > self.max_cols:
            self.max_cols = p.id.cols
        if src_bytes_per > self.max_src_bytes_per:
            self.max_src_bytes_per = src_bytes_per
        var spr = (p.id.cols // p.fwht_block) if p.per_block else 1
        if spr > self.max_scale_per_row:
            self.max_scale_per_row = spr
        if p.colsum_kind == ColsumKind.PER_ROW:
            if 1 > self.max_cs_per_row:
                self.max_cs_per_row = 1
        elif p.colsum_kind == ColsumKind.PER_BLOCK:
            var cpr = p.id.cols // p.fwht_block
            if cpr > self.max_cs_per_row:
                self.max_cs_per_row = cpr
