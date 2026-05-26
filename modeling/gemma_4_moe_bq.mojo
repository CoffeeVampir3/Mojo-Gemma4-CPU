from std.pathlib import Path
from std.memory import Span, UnsafePointer

from numa import NumaArena, NumaTopology
from threading import BurstPool
from threading.threading_traits import BurstThreadPool
from simd_math.ops import sqrt
from kernels.helpers import ArenaBases
from kernels.moe_router import RouterCandidate, SparseRoute
from kernels.attention_ops import flash_partial_stride
from kernels.reductions import dispatch_allreduce_inplace
from butterquant_kernels import dispatch_bq_embed_lookup
from modeling.temporal_scratch import (
    ScratchBuffer, ScratchIsland, ScratchPhase, ScratchPhaseOrder,
    TemporalScratchPool, aggregate_scratch_peak,
)

from modeling.model_spec import (
    BF16, F32,
    Shape, WeightDesc,
    DistributionDegree,
    Replicated,
    TensorRowSharded, TensorColumnSharded,
    ContextRowSharded, ExpertRowBlockSharded, VocabularyRowSharded,
    DEFAULT_ALIGNMENT,
    align_up,
)
from modeling.gemma4_common import (
    Gemma4BaseConfig, LAYER_SCHEDULE, LayerKind,
)
from modeling.modeling_common import (
    Repeated, ArenaLayout,
)
from modeling.slot import (
    Slot, SlotGroup, BindContext, stamp_offsets, emit_descs,
)
from quant.recipe import (
    QuantRecipe, PerRowQuant, PerBlockQuant, RouterCenter,
    SplitGamma, NoGamma, SingleSided, PerRowCs, PerBlockCs,
)
from quant.quantizer import Quantizer
from modeling.loader import discover_shards, load_weights_from_descs


comptime C = Gemma4BaseConfig


comptime SlidingAttentionContract[degree: Int]: Bool = (
    degree > 0
    and C.NUM_HEADS % degree == 0
    and C.NUM_KV_HEADS_SLIDING % degree == 0
    and C.Q_DIM_SLIDING % degree == 0
    and C.KV_DIM_SLIDING % degree == 0
)


comptime FullAttentionContract[degree: Int, max_seq_len: Int]: Bool = (
    degree > 0
    and C.NUM_HEADS % degree == 0
    and C.Q_DIM_FULL % degree == 0
    and max_seq_len % degree == 0
)


comptime DenseMlpContract[degree: Int]: Bool = (
    degree > 0
    and C.INTERMEDIATE % degree == 0
)


comptime MoeContract[degree: Int]: Bool = (
    degree > 0
    and C.NUM_EXPERTS % degree == 0
)


comptime LmHeadContract[degree: Int]: Bool = (
    degree > 0
    and C.VOCAB_SIZE % degree == 0
)


struct Gemma4Shapes[degree: Int]:
    comptime D = DistributionDegree[Self.degree]
    comptime GateUp      = TensorRowSharded[C.INTERMEDIATE, C.HIDDEN, Self.D]
    comptime Down        = TensorColumnSharded[C.HIDDEN, C.INTERMEDIATE, Self.D]
    comptime SlidingQ    = TensorRowSharded[C.Q_DIM_SLIDING, C.HIDDEN, Self.D]
    comptime SlidingKV   = TensorRowSharded[C.KV_DIM_SLIDING, C.HIDDEN, Self.D]
    comptime SlidingO    = TensorColumnSharded[C.HIDDEN, C.Q_DIM_SLIDING, Self.D]
    comptime FullQ       = Replicated[C.Q_DIM_FULL, C.HIDDEN]
    comptime FullK       = Replicated[C.KV_DIM_FULL, C.HIDDEN]
    comptime FullO       = TensorColumnSharded[C.HIDDEN, C.Q_DIM_FULL, Self.D]
    comptime RouterProj  = ExpertRowBlockSharded[
        C.NUM_EXPERTS, 1, C.HIDDEN, Self.D,
    ]
    comptime ExpertsGateUp = ExpertRowBlockSharded[
        C.NUM_EXPERTS, C.MOE_GATE_UP_FUSED, C.HIDDEN, Self.D,
    ]
    comptime ExpertsDown = ExpertRowBlockSharded[
        C.NUM_EXPERTS, C.HIDDEN, C.MOE_INTERMEDIATE, Self.D,
    ]


struct Gemma4StateShapes[degree: Int, max_seq_len: Int]:
    """Distribution choices for per-token state tensors.

    `D` shards a tensor across the `degree` ranks (one slice per rank).
    `Local` keeps the whole tensor on every rank (replicated). The
    project's NUMA principle says: data should live closest to its
    most-frequent reader. Both rope tables are read by every attention
    worker on every token, so both are replicated per rank to keep reads
    NUMA-local and to let prefill rotate Q identically on every rank
    without a cross-rank gather. The sliding KV ring is sized at
    `2 * SLIDING_WINDOW` so a prefill chunk of W tokens can land in one
    half while the prior chunk stays addressable in the other half.
    """
    comptime D     = DistributionDegree[Self.degree]
    comptime Local = DistributionDegree[1]
    comptime SLIDING_CACHE = 2 * C.SLIDING_WINDOW
    comptime SlidingKV   = TensorColumnSharded[Self.SLIDING_CACHE, C.KV_DIM_SLIDING, Self.D]
    comptime FullKV      = ContextRowSharded[Self.max_seq_len, C.KV_DIM_FULL, Self.D]
    comptime SlidingRope = ContextRowSharded[Self.max_seq_len, C.ROPE_HALF_SLIDING, Self.Local]
    comptime FullRope    = ContextRowSharded[Self.max_seq_len, C.ROPE_HALF_FULL, Self.Local]


struct Gemma4TailShapes[degree: Int]:
    comptime D = DistributionDegree[Self.degree]
    comptime FinalNorm = Replicated[C.HIDDEN, 1]
    comptime Embed = VocabularyRowSharded[C.VOCAB_SIZE, C.HIDDEN, Self.D]


comptime SplitGainPerRowCs[fwht: Int, gamma: StaticString]: QuantRecipe = PerRowQuant(
    fwht, SplitGamma(gamma), SingleSided(), PerRowCs(),
)


comptime PlainPerRowBlockCs[fwht: Int]: QuantRecipe = PerRowQuant(
    fwht, NoGamma(), SingleSided(), PerBlockCs(),
)


comptime TiedHeadEmbed[fwht: Int]: QuantRecipe = PerBlockQuant(
    fwht, NoGamma(), SingleSided(), PerBlockCs(),
)


struct SlidingAttnRefs[degree: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4Shapes[Self.degree]
    var q_proj: Slot[BF16, Self.S.SlidingQ,  "self_attn.q_proj.weight",
        SplitGainPerRowCs[128, "input_layernorm.weight"]]
    var k_proj: Slot[BF16, Self.S.SlidingKV, "self_attn.k_proj.weight",
        SplitGainPerRowCs[128, "input_layernorm.weight"]]
    var v_proj: Slot[BF16, Self.S.SlidingKV, "self_attn.v_proj.weight",
        SplitGainPerRowCs[128, "input_layernorm.weight"]]
    var o_proj: Slot[BF16, Self.S.SlidingO,  "self_attn.o_proj.weight",
        PlainPerRowBlockCs[C.HEAD_DIM_SLIDING]]
    var q_norm: Slot[BF16, Shape[C.HEAD_DIM_SLIDING, 1], "self_attn.q_norm.weight"]
    var k_norm: Slot[BF16, Shape[C.HEAD_DIM_SLIDING, 1], "self_attn.k_norm.weight"]


struct FullAttnRefs[degree: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4Shapes[Self.degree]
    var q_proj: Slot[BF16, Self.S.FullQ, "self_attn.q_proj.weight",
        SplitGainPerRowCs[128, "input_layernorm.weight"]]
    var k_proj: Slot[BF16, Self.S.FullK, "self_attn.k_proj.weight",
        SplitGainPerRowCs[128, "input_layernorm.weight"]]
    var o_proj: Slot[BF16, Self.S.FullO, "self_attn.o_proj.weight",
        PlainPerRowBlockCs[C.HEAD_DIM_FULL]]
    var q_norm: Slot[BF16, Shape[C.HEAD_DIM_FULL, 1], "self_attn.q_norm.weight"]
    var k_norm: Slot[BF16, Shape[C.HEAD_DIM_FULL, 1], "self_attn.k_norm.weight"]


struct BodyRefs[degree: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4Shapes[Self.degree]
    var input_norm:      Slot[BF16, Shape[C.HIDDEN, 1],         "input_layernorm.weight"]
    var post_attn_norm:  Slot[BF16, Shape[C.HIDDEN, 1],         "post_attention_layernorm.weight"]
    var pre_ffn_norm:    Slot[BF16, Shape[C.HIDDEN, 1],         "pre_feedforward_layernorm.weight"]
    var pre_ffn_norm_2:  Slot[BF16, Shape[C.HIDDEN, 1],         "pre_feedforward_layernorm_2.weight"]
    var post_ffn_norm_1: Slot[BF16, Shape[C.HIDDEN, 1],         "post_feedforward_layernorm_1.weight"]
    var post_ffn_norm_2: Slot[BF16, Shape[C.HIDDEN, 1],         "post_feedforward_layernorm_2.weight"]
    var post_ffn_norm:   Slot[BF16, Shape[C.HIDDEN, 1],         "post_feedforward_layernorm.weight"]
    var gate_proj:       Slot[BF16, Self.S.GateUp,              "mlp.gate_proj.weight",
        SplitGainPerRowCs[128, "pre_feedforward_layernorm.weight"]]
    var up_proj:         Slot[BF16, Self.S.GateUp,              "mlp.up_proj.weight",
        SplitGainPerRowCs[128, "pre_feedforward_layernorm.weight"]]
    var down_proj:       Slot[BF16, Self.S.Down,                "mlp.down_proj.weight",
        PlainPerRowBlockCs[64]]
    var router_proj:     Slot[BF16, Self.S.RouterProj,          "router.proj.weight",
        RouterCenter("")]
    var router_scale:    Slot[BF16, Shape[C.HIDDEN, 1],         "router.scale"]
    var router_pes:      Slot[BF16, Shape[C.NUM_EXPERTS, 1],    "router.per_expert_scale"]
    var experts_gate_up: Slot[BF16, Self.S.ExpertsGateUp,       "experts.gate_up_proj",
        SplitGainPerRowCs[128, "pre_feedforward_layernorm_2.weight"]]
    var experts_down:    Slot[BF16, Self.S.ExpertsDown,         "experts.down_proj",
        PlainPerRowBlockCs[64]]
    var layer_scalar:    Slot[BF16, Shape[1, 1],                "layer_scalar"]


struct SlidingLayerRefs[degree: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    var attn: SlidingAttnRefs[Self.degree]
    var body: BodyRefs[Self.degree]


struct FullLayerRefs[degree: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    var attn: FullAttnRefs[Self.degree]
    var body: BodyRefs[Self.degree]


struct SlidingKVSlots[degree: Int, max_seq_len: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4StateShapes[Self.degree, Self.max_seq_len]
    var k: Slot[BF16, Self.S.SlidingKV]
    var v: Slot[BF16, Self.S.SlidingKV]


struct FullKVSlots[degree: Int, max_seq_len: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4StateShapes[Self.degree, Self.max_seq_len]
    var k: Slot[BF16, Self.S.FullKV]
    var v: Slot[BF16, Self.S.FullKV]


struct RopeSlots[half: Int, max_seq_len: Int, degree: Int = 1](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime D = DistributionDegree[Self.degree]
    var cos: Slot[F32, ContextRowSharded[Self.max_seq_len, Self.half, Self.D]]
    var sin: Slot[F32, ContextRowSharded[Self.max_seq_len, Self.half, Self.D]]


struct ActivationSlots(Copyable, ImplicitlyCopyable, SlotGroup):
    var x_main:     Slot[BF16, Shape[C.SLIDING_WINDOW, C.HIDDEN]]
    var x_residual: Slot[BF16, Shape[C.SLIDING_WINDOW, C.HIDDEN]]


struct TailRefs[degree: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4TailShapes[Self.degree]
    var final_norm: Slot[BF16, Self.S.FinalNorm, "model.language_model.norm.weight"]
    var embed:      Slot[BF16, Self.S.Embed, "model.language_model.embed_tokens.weight",
        TiedHeadEmbed[128]]


@fieldwise_init
struct Gemma4Layout[degree: Int, max_seq_len: Int](Copyable, ImplicitlyCopyable):
    var arena: ArenaLayout
    var sliding: Repeated[SlidingLayerRefs[Self.degree]]
    var full: Repeated[FullLayerRefs[Self.degree]]

    var sliding_kv: Repeated[SlidingKVSlots[Self.degree, Self.max_seq_len]]
    var full_kv: Repeated[FullKVSlots[Self.degree, Self.max_seq_len]]
    var activations: ActivationSlots
    var sliding_rope: RopeSlots[C.ROPE_HALF_SLIDING, Self.max_seq_len]
    var full_rope: RopeSlots[C.ROPE_HALF_FULL, Self.max_seq_len]

    var tail: Repeated[TailRefs[Self.degree]]

    @always_inline
    def bind(self, base: Int) -> Self:
        var t = self
        t.arena = t.arena.bind(base)
        return t




@fieldwise_init
struct Gemma4SlidingScratch[degree: Int, max_worker_count: Int = 128](
    ScratchIsland, Copyable, ImplicitlyCopyable
):
    comptime S = Gemma4Shapes[Self.degree]
    comptime q_rows = Self.S.SlidingQ.DATA_N
    comptime kv_rows = Self.S.SlidingKV.DATA_N
    comptime head_dim = C.HEAD_DIM_SLIDING
    comptime num_kv_heads = Self.kv_rows // Self.head_dim
    comptime num_q_heads = Self.q_rows // Self.head_dim
    comptime cache_size = 2 * C.SLIDING_WINDOW
    comptime PARTIAL_STRIDE = flash_partial_stride[
        Self.num_q_heads, Self.head_dim,
    ]()

    comptime PHASES = ScratchPhaseOrder[
        "gemv_qkv", "rms_norm_qkv", "rope_cache_write",
        "flash", "merge_partials", "o_proj",
    ]

    var q_band: ScratchPhase["gemv_qkv", "o_proj"]
    var q: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * Self.q_rows,
    ]

    var kv_band: ScratchPhase["gemv_qkv", "rope_cache_write"]
    var kv: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * Self.kv_rows * 2,
    ]

    var partials_band: ScratchPhase["flash", "merge_partials"]
    var partials: ScratchBuffer[
        Float32, Self.max_worker_count * Self.PARTIAL_STRIDE,
    ]


@fieldwise_init
struct Gemma4FullScratch[degree: Int, max_worker_count: Int = 128](
    ScratchIsland, Copyable, ImplicitlyCopyable
):
    comptime S = Gemma4Shapes[Self.degree]
    comptime q_rows = Self.S.FullQ.DATA_N
    comptime k_rows = Self.S.FullK.DATA_N
    comptime local_q_rows = Self.S.FullO.DATA_M
    comptime head_dim = C.HEAD_DIM_FULL
    comptime num_q_heads = Self.q_rows // Self.head_dim
    comptime PARTIAL_STRIDE = flash_partial_stride[
        Self.num_q_heads, Self.head_dim,
    ]()
    comptime PARTIAL_SLOTS = (
        Self.max_worker_count
        if Self.max_worker_count >= C.SLIDING_WINDOW
        else C.SLIDING_WINDOW
    )

    comptime PHASES = ScratchPhaseOrder[
        "gemv_q", "gemv_kv", "rms_norm_qkv", "rope_cache_write",
        "flash", "merge_partials", "o_proj",
    ]

    var q_band: ScratchPhase["gemv_q", "flash"]
    var q: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * Self.q_rows,
    ]

    var kv_band: ScratchPhase["gemv_kv", "rope_cache_write"]
    var kv: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * Self.k_rows * 2,
    ]

    var partials_band: ScratchPhase["flash", "merge_partials"]
    var partials: ScratchBuffer[
        Float32, Self.PARTIAL_SLOTS * Self.PARTIAL_STRIDE,
    ]

    var q_local_band: ScratchPhase["merge_partials", "o_proj"]
    var q_local: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * Self.local_q_rows,
    ]


@fieldwise_init
struct Gemma4FfnMoeScratch[degree: Int, max_worker_count: Int = 128](
    ScratchIsland, Copyable, ImplicitlyCopyable
):
    comptime S = Gemma4Shapes[Self.degree]
    comptime intermediate_per_rank = Self.S.GateUp.DATA_N
    comptime experts_per_rank = C.NUM_EXPERTS // Self.degree
    comptime PHASE1_TILE_J = 64
    comptime PHASE1_MR = 4

    comptime PHASES = ScratchPhaseOrder[
        "ffn_rms_norm", "gemv_gate", "gemv_up", "gelu_gate_up",
        "router_sharded", "merge_cands", "moe_rms_norm",
        "build_schedules", "phase1_gate_up", "phase2_down",
        "moe_allreduce", "gemv_dense", "allreduce_dense",
        "post_norm_1", "post_norm_2", "post_norm_3",
    ]

    var ffn_gate_band: ScratchPhase["gemv_gate", "gemv_dense"]
    var ffn_gate: ScratchBuffer[
        BFloat16,
        C.SLIDING_WINDOW * Self.intermediate_per_rank,
    ]

    var ffn_up_band: ScratchPhase["gemv_up", "gelu_gate_up"]
    var ffn_up: ScratchBuffer[
        BFloat16,
        C.SLIDING_WINDOW * Self.intermediate_per_rank,
    ]

    var router_workspace: ScratchPhase[
        "router_sharded", "router_sharded",
    ]
    var moe_router_scaled: ScratchBuffer[
        Float32, Self.max_worker_count * C.HIDDEN,
    ]

    var router_cands: ScratchPhase["router_sharded", "merge_cands"]
    var moe_cands: ScratchBuffer[RouterCandidate, C.SLIDING_WINDOW * C.TOP_K]

    var router_products: ScratchPhase["merge_cands", "build_schedules"]
    var moe_route_idx: ScratchBuffer[
        Int32, C.SLIDING_WINDOW * C.TOP_K,
    ]
    var moe_route_w: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * C.TOP_K,
    ]

    var expert_input: ScratchPhase["moe_rms_norm", "phase1_gate_up"]
    var moe_x_normed: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.HIDDEN,
    ]

    var schedule_products: ScratchPhase[
        "build_schedules", "phase2_down",
    ]
    var moe_expert_offset: ScratchBuffer[
        Int32, Self.experts_per_rank + 1,
    ]
    var moe_routes: ScratchBuffer[SparseRoute, C.SLIDING_WINDOW * C.TOP_K]

    var hidden_bucket: ScratchPhase["phase1_gate_up", "phase2_down"]
    var moe_hidden_bucket: ScratchBuffer[
        BFloat16,
        C.SLIDING_WINDOW * C.TOP_K * C.MOE_INTERMEDIATE,
    ]

    var phase1_workspace: ScratchPhase[
        "phase1_gate_up", "phase1_gate_up",
    ]
    var moe_gate_scratch: ScratchBuffer[
        Float32,
        Self.max_worker_count * Self.PHASE1_MR * 2 * Self.PHASE1_TILE_J,
    ]

    var phase2_accum: ScratchPhase["phase2_down", "phase2_down"]
    var moe_accum: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * C.HIDDEN,
    ]

    var dense_band: ScratchPhase["gemv_dense", "post_norm_3"]
    var ffn_dense_out: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.HIDDEN,
    ]


@fieldwise_init
struct Gemma4HeadScratch[degree: Int](
    ScratchIsland, Copyable, ImplicitlyCopyable
):
    comptime PHASES = ScratchPhaseOrder[
        "final_norm", "gemv_logits", "returned_to_caller",
    ]
    comptime vocab_per_rank = C.VOCAB_SIZE // Self.degree

    var logits_band: ScratchPhase["gemv_logits", "returned_to_caller"]
    var logits: ScratchBuffer[
        BFloat16, Self.vocab_per_rank,
    ]


@fieldwise_init
struct Gemma4ForwardScratch[
    degree: Int, max_worker_count: Int = 128,
](Copyable, ImplicitlyCopyable):
    var sliding: Gemma4SlidingScratch[Self.degree, Self.max_worker_count]
    var full: Gemma4FullScratch[Self.degree, Self.max_worker_count]
    var ffn: Gemma4FfnMoeScratch[Self.degree, Self.max_worker_count]
    var head: Gemma4HeadScratch[Self.degree]


def calculate_peak_scratch[degree: Int, max_worker_count: Int = 128]() -> Int:
    return aggregate_scratch_peak[
        Gemma4ForwardScratch[degree, max_worker_count],
    ]()


comptime Gemma4ScratchPool[
    degree: Int, max_worker_count: Int = 128,
] = TemporalScratchPool[
    calculate_peak_scratch[degree, max_worker_count](),
]


def build_gemma4_plan[
    degree: Int, max_seq_len: Int, max_worker_count: Int = 128,
](mut descs: List[WeightDesc]) -> Gemma4Layout[degree, max_seq_len]:
    comptime assert SlidingAttentionContract[degree], "sliding attention distribution contract failed"
    comptime assert FullAttentionContract[degree, max_seq_len], "full attention distribution contract failed"
    comptime assert DenseMlpContract[degree], "dense MLP distribution contract failed"
    comptime assert MoeContract[degree], "MoE distribution contract failed"
    comptime assert LmHeadContract[degree], "LM head distribution contract failed"
    comptime assert max_worker_count > 0, "max_worker_count must be positive"

    var sl_proto = SlidingLayerRefs[degree]()
    var sl_stride = stamp_offsets(sl_proto)
    var fl_proto = FullLayerRefs[degree]()
    var fl_stride = stamp_offsets(fl_proto)

    var sl_off = 0
    var fl_off = sl_off + C.NUM_SLIDING_LAYERS * sl_stride
    var distributed = fl_off + C.NUM_FULL_LAYERS * fl_stride

    for i in range(C.NUM_LAYERS):
        var entry = LAYER_SCHEDULE[i]
        var prefix = String(t"model.language_model.layers.{entry.idx}.")
        if entry.kind == LayerKind.FULL:
            _ = emit_descs[FullLayerRefs[degree]](
                prefix, fl_off + entry.local_idx * fl_stride, descs)
        else:
            _ = emit_descs[SlidingLayerRefs[degree]](
                prefix, sl_off + entry.local_idx * sl_stride, descs)

    var tail_proto = TailRefs[degree]()
    var tail_bytes = stamp_offsets(tail_proto)
    _ = emit_descs[TailRefs[degree]]("", distributed, descs)
    var tail = Repeated[TailRefs[degree]](tail_proto, distributed, tail_bytes, 1)
    distributed += tail_bytes

    var state_cursor = distributed

    var skv_proto = SlidingKVSlots[degree, max_seq_len]()
    var skv_stride = stamp_offsets(skv_proto)
    var sliding_kv = Repeated[SlidingKVSlots[degree, max_seq_len]](
        skv_proto, state_cursor, skv_stride, C.NUM_SLIDING_LAYERS)
    state_cursor = align_up(state_cursor + C.NUM_SLIDING_LAYERS * skv_stride)

    var fkv_proto = FullKVSlots[degree, max_seq_len]()
    var fkv_stride = stamp_offsets(fkv_proto)
    var full_kv = Repeated[FullKVSlots[degree, max_seq_len]](
        fkv_proto, state_cursor, fkv_stride, C.NUM_FULL_LAYERS)
    state_cursor = align_up(state_cursor + C.NUM_FULL_LAYERS * fkv_stride)

    var activations = ActivationSlots()
    state_cursor = stamp_offsets(activations, state_cursor)

    var scratch_cap = calculate_peak_scratch[degree, max_worker_count]()
    state_cursor = align_up(state_cursor)
    var scratch_off = state_cursor
    state_cursor = align_up(state_cursor + scratch_cap)

    var sliding_rope = RopeSlots[C.ROPE_HALF_SLIDING, max_seq_len]()
    state_cursor = stamp_offsets(sliding_rope, state_cursor)
    var full_rope = RopeSlots[C.ROPE_HALF_FULL, max_seq_len]()
    state_cursor = stamp_offsets(full_rope, state_cursor)

    var arena = ArenaLayout(
        base=0,
        distributed_bytes=distributed,
        state_bytes=state_cursor - distributed,
        host_bytes=align_up(state_cursor),
        scratch_off=scratch_off,
    )
    return Gemma4Layout[degree, max_seq_len](
        arena=arena,
        sliding=Repeated[SlidingLayerRefs[degree]](sl_proto, sl_off, sl_stride, C.NUM_SLIDING_LAYERS),
        full=Repeated[FullLayerRefs[degree]](fl_proto, fl_off, fl_stride, C.NUM_FULL_LAYERS),
        sliding_kv=sliding_kv, full_kv=full_kv,
        activations=activations,
        sliding_rope=sliding_rope, full_rope=full_rope,
        tail=tail)


struct Gemma4[
    degree: Int, max_seq_len: Int = 8192,
    max_worker_count: Int = 128,
    Pool: BurstThreadPool = BurstPool[],
](Movable):
    """BQ variant of the Gemma 4 model. The runtime forward path lives
    in `gemma_4_moe.mojo`; this file only exists to drive the quantizer
    (recipes on every weight slot) and to load and inspect the resulting
    int8 checkpoint until ButterQuant kernels arrive."""
    var arenas: List[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var pools: List[Self.Pool]
    var layout: Gemma4Layout[Self.degree, Self.max_seq_len]
    var scratch: Gemma4ScratchPool[Self.degree, Self.max_worker_count]
    var arena_bases: ArenaBases[Self.degree]
    var descs: List[WeightDesc]

    def __init__(out self,
        var arenas: List[NumaArena[alignment=DEFAULT_ALIGNMENT]],
        var pools: List[Self.Pool],
        layout: Gemma4Layout[Self.degree, Self.max_seq_len],
        var descs: List[WeightDesc],
    ):
        self.arena_bases = ArenaBases[Self.degree].uninitialized()
        for r in range(Self.degree):
            self.arena_bases[r] = Int(arenas[r].base.value())
        self.layout = layout.bind(self.arena_bases[0])
        self.arenas = arenas^
        self.pools = pools^
        self.descs = descs^
        self.scratch = Gemma4ScratchPool[
            Self.degree, Self.max_worker_count,
        ](
            self.layout.arena.scratch_base())

    def dump_tensors(self):
        """One line per loaded tensor: full name, dtype, global/data shape,
        per-rank arena offset, plus the first four values from rank 0. The
        line count is high (one per WeightDesc, including every sidecar)
        but is the cheapest way to verify the int8 / scale / colsum /
        gauge layout end-to-end without a runtime kernel."""
        var n = len(self.descs)
        print(t"--- {n} loaded tensors ---")
        for i in range(n):
            ref d = self.descs[i]
            var addr = self.arena_bases[0] + d.arena_offset
            var name = d.name
            var dt = d.dtype
            var gr = d.global_rows
            var gc = d.global_cols
            var dr = d.data_rows
            var dc = d.data_cols
            var off = d.arena_offset
            var tr = d.target_rank
            print(
                t"{name} :: {dt} global=[{gr},{gc}] data=[{dr},{dc}] "
                t"target={tr} arena_off={off}"
            )
            var count = dr * dc
            if count > 4: count = 4
            if dt == DType.int8:
                var p = UnsafePointer[Int8, MutAnyOrigin](
                    unsafe_from_address=addr)
                for k in range(count):
                    var v = Int(p[k])
                    print(t"    [{k}] = {v}")
            elif dt == DType.float32:
                var p = UnsafePointer[Float32, MutAnyOrigin](
                    unsafe_from_address=addr)
                for k in range(count):
                    var v = p[k]
                    print(t"    [{k}] = {v}")
            elif dt == DType.bfloat16:
                var p = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin](
                    unsafe_from_address=addr)
                for k in range(count):
                    var v = p[k].cast[DType.float32]()
                    print(t"    [{k}] = {v}")

    def forward[
        tok_origin: ImmutOrigin, //,
    ](
        mut self,
        token_ids: Span[Int32, tok_origin],
        base_pos: Int,
    ):
        """Runs the bq model up to and including token embedding: resolves the
        int8 embedding through the bq_weight bridge, dispatches the butterquant
        embed-lookup into the per-token activation state, replicates across
        ranks, and logs the embedded hidden values. The remaining layers are
        not yet implemented."""
        ref layout = self.layout
        comptime shard_rows = Gemma4TailShapes[Self.degree].Embed.DATA_N
        comptime embed_scale = Float64(sqrt[DType.float32, 1](C.HIDDEN)
            .cast[DType.bfloat16]().cast[DType.float32]())

        var total_len = len(token_ids)
        debug_assert(total_len > 0, "forward called with empty token_ids")
        var chunk_len = total_len if total_len < C.SLIDING_WINDOW else C.SLIDING_WINDOW

        var ctx = BindContext[Self.degree](
            arena_bases=self.arena_bases, layer_base=0)
        var tail_ctx = ctx.with_layer(
            layout.tail.base(self.arena_bases[0], 0))

        var x_main_ranks = layout.activations.x_main.state_binding(ctx)

        var chunk = Span[Int32, tok_origin](
            ptr=token_ids.unsafe_ptr(), length=chunk_len)

        dispatch_bq_embed_lookup[
            scale=embed_scale, shard_rows=shard_rows,
            max_worker_count=Self.max_worker_count,
        ](chunk,
          layout.tail.proto.embed.bq_weight(tail_ctx),
          x_main_ranks, chunk_len, self.pools)
        dispatch_allreduce_inplace[
            BF16, Self.degree, max_worker_count=Self.max_worker_count,
        ](x_main_ranks, chunk_len * C.HIDDEN, self.pools)

        var x0 = x_main_ranks[0]
        var h0 = x0[0].cast[DType.float32]()
        var h1 = x0[1].cast[DType.float32]()
        var h2 = x0[2].cast[DType.float32]()
        var h3 = x0[3].cast[DType.float32]()
        var tid0 = Int(token_ids[0])
        print(
            t"bq forward: embedded {chunk_len} tokens; "
            t"token0(id={tid0}) hidden[0..3] = {h0}, {h1}, {h2}, {h3}"
        )

    @staticmethod
    def load(
        dir_path: Path,
        topo: NumaTopology,
        var pools: List[Self.Pool],
    ) -> Optional[Self]:
        var shards = discover_shards(dir_path)
        if len(shards) == 0:
            print(t"no safetensors shards found in {dir_path}")
            return None
        var n_shards = len(shards)
        print(t"found {n_shards} shard(s)")

        var descs = List[WeightDesc]()
        var layout = build_gemma4_plan[
            Self.degree, Self.max_seq_len, Self.max_worker_count,
        ](descs)

        var size = layout.arena.host_arena_bytes()
        var size_mb = size // (1024 * 1024)
        var weights_mb = layout.arena.distributed_bytes // (1024 * 1024)
        var state_mb = layout.arena.state_bytes // (1024 * 1024)
        print(
            t"allocating {size_mb} MB x {Self.degree} rank(s) "
            t"({weights_mb} MB weights + {state_mb} MB state each)"
        )

        var arenas = List[NumaArena[alignment=DEFAULT_ALIGNMENT]](capacity=Self.degree)
        var arena_bases = List[Int]()
        for rank in range(Self.degree):
            arenas.append(NumaArena[alignment=DEFAULT_ALIGNMENT](topo.node(rank), size))
            if not arenas[rank]:
                var node = topo.node(rank)
                print(t"arena allocation failed on node {node}")
                return None
            arena_bases.append(Int(arenas[rank].base.value()))

        var load_result = load_weights_from_descs(descs, shards, arena_bases, topo)
        if not load_result:
            print("weight loading failed")
            return None
        var loaded = load_result.take()
        var loaded_mb = loaded.bytes_loaded // (1024 * 1024)
        print(t"loaded {loaded_mb} MB in {loaded.num_ops} ops")

        for rank in range(Self.degree):
            _ = arenas[rank].prefault(layout.arena.distributed_bytes, layout.arena.state_bytes)

        var model = Self(arenas^, pools^, layout, descs^)
        return model^

    @staticmethod
    def quantize(
        source_dir: Path, output_path: Path,
        topo: NumaTopology, var pools: List[Self.Pool],
    ) -> Bool:
        """Quantize the source checkpoint to `output_path`. Slots are
        dispatched as one job per pool, so caller-controlled topology
        (typically via `with_topological_rank_dispatch`) determines how
        much parallelism is applied."""
        var q = Quantizer(source_dir, output_path)
        if not q:
            return False
        for i in range(C.NUM_LAYERS):
            var entry = LAYER_SCHEDULE[i]
            var prefix = String(t"model.language_model.layers.{entry.idx}.")
            if entry.kind == LayerKind.FULL:
                if not q.plan_walk[FullLayerRefs[1]](prefix, entry.idx):
                    return False
            else:
                if not q.plan_walk[SlidingLayerRefs[1]](prefix, entry.idx):
                    return False
        if not q.plan_walk[TailRefs[1]](String(""), -1):
            return False
        if not q.write_header():
            return False
        return q.execute(topo, pools^)
