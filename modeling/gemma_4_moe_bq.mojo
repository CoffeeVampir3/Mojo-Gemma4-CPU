from std.collections import InlineArray
from std.pathlib import Path
from std.memory import Span, UnsafePointer
from std.time import perf_counter_ns

from numa import NumaArena, NumaTopology
from threading import BurstPool
from threading.threading_traits import BurstThreadPool
from std.sys.info import simd_width_of
from simd_math.ops import sqrt
from simd_math import has_amx_int8
from butterquant.amx_tiles import prime_amx_environment
from kernels.helpers import ArenaBases, Binding, prime_fp_environment
from kernels.profiling import Profiler
from kernels.moe_router import (
    RouterCandidate, SparseRoute,
    dispatch_router_expert, merge_router_candidates_expert, build_expert_schedules,
)
from kernels.attention_ops import flash_partial_stride
from kernels.reductions import dispatch_allreduce_inplace
from kernels.rmsnorm import dispatch_rms_norm, fused_norm_residual_add
from kernels.elementwise import dispatch_gelu_gate_up, dispatch_scalar_mul
from butterquant_kernels import (
    dispatch_bq_embed_lookup, dispatch_bq_norm_quant, dispatch_bq_qkv,
    dispatch_bq_linear, dispatch_bq_attn_prep,
    dispatch_bq_sliding_attention, dispatch_bq_full_attention,
    dispatch_bq_block_quant, dispatch_bq_block_linear,
    dispatch_bq_phase1_gate_up, dispatch_bq_phase2_down,
    dispatch_bq_head_prep, dispatch_bq_head_gemv,
)
from butterquant import (
    VnniPackable, PackColsumTask, dispatch_pack_colsum,
    bake_split_gain_in_place, ButterquantActivation, ButterquantBlockActivation,
)
from modeling.temporal_scratch import (
    ScratchBuffer, ScratchIsland, ScratchPhase, ScratchPhaseOrder,
    TemporalScratchPool, TemporalLogitsView, aggregate_scratch_peak,
)

from modeling.model_spec import (
    BF16, F32, I8,
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
    Slot, SlotGroup, BindContext, stamp_offsets, emit_descs, emit_pack_tasks,
)
from quant.recipe import (
    QuantRecipe, PerRowQuant, PerBlockQuant, SoftmaxRouterCenter,
    SplitGamma, NoGamma, SingleSided, PerRowCs, PerBlockCs, NoColsum,
    VnniPacked, RowMajor,
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
    comptime GateUp      = TensorRowSharded[C.INTERMEDIATE, C.HIDDEN, Self.D, block=C.DOWN_FWHT_BLOCK]
    comptime Down        = TensorColumnSharded[C.HIDDEN, C.INTERMEDIATE, Self.D, block=C.DOWN_FWHT_BLOCK]
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


comptime VnniPackContract[degree: Int]: Bool = (
    VnniPackable[Gemma4Shapes[degree].SlidingQ]
    and VnniPackable[Gemma4Shapes[degree].SlidingKV]
    and VnniPackable[Gemma4Shapes[degree].SlidingO]
    and VnniPackable[Gemma4Shapes[degree].FullQ]
    and VnniPackable[Gemma4Shapes[degree].FullK]
    and VnniPackable[Gemma4Shapes[degree].FullO]
    and VnniPackable[Gemma4Shapes[degree].GateUp]
    and VnniPackable[Gemma4Shapes[degree].Down]
    and VnniPackable[Gemma4Shapes[degree].ExpertsGateUp]
    and VnniPackable[Gemma4Shapes[degree].ExpertsDown]
)


struct Gemma4StateShapes[degree: Int, max_seq_len: Int]:
    comptime D     = DistributionDegree[Self.degree]
    comptime Local = DistributionDegree[1]
    comptime SLIDING_CACHE = 2 * C.SLIDING_WINDOW
    comptime SlidingKV   = TensorColumnSharded[Self.SLIDING_CACHE, C.KV_DIM_SLIDING, Self.D]
    comptime FullKV      = ContextRowSharded[Self.max_seq_len, C.KV_DIM_FULL, Self.D]
    comptime SlidingKVScale = TensorColumnSharded[Self.SLIDING_CACHE, C.NUM_KV_HEADS_SLIDING, Self.D]
    comptime FullKVScale = ContextRowSharded[Self.max_seq_len, C.NUM_KV_HEADS_FULL, Self.D]
    comptime SlidingRope = ContextRowSharded[Self.max_seq_len, C.ROPE_HALF_SLIDING, Self.Local]
    comptime FullRope    = ContextRowSharded[Self.max_seq_len, C.ROPE_HALF_FULL, Self.Local]


struct Gemma4TailShapes[degree: Int]:
    comptime D = DistributionDegree[Self.degree]
    comptime FinalNorm = Replicated[C.HIDDEN, 1]
    comptime Embed = VocabularyRowSharded[C.VOCAB_SIZE, C.HIDDEN, Self.D]


comptime SplitGainPerRowCs[fwht: Int, gamma: StaticString]: QuantRecipe = PerRowQuant(
    fwht, SplitGamma(gamma), SingleSided(), PerRowCs(), VnniPacked(),
)


comptime PlainPerRowBlockCs[fwht: Int]: QuantRecipe = PerRowQuant(
    fwht, NoGamma(), SingleSided(), PerBlockCs(), VnniPacked(),
)


comptime TiedHeadEmbed[fwht: Int]: QuantRecipe = PerBlockQuant(
    fwht, NoGamma(), SingleSided(), NoColsum(), RowMajor(),
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
        PlainPerRowBlockCs[C.DOWN_FWHT_BLOCK]]
    var router_proj:     Slot[BF16, Self.S.RouterProj,          "router.proj.weight",
        SoftmaxRouterCenter()]
    var router_scale:    Slot[BF16, Shape[C.HIDDEN, 1],         "router.scale"]
    var router_pes:      Slot[BF16, Shape[C.NUM_EXPERTS, 1],    "router.per_expert_scale"]
    var experts_gate_up: Slot[BF16, Self.S.ExpertsGateUp,       "experts.gate_up_proj",
        SplitGainPerRowCs[128, "pre_feedforward_layernorm_2.weight"]]
    var experts_down:    Slot[BF16, Self.S.ExpertsDown,         "experts.down_proj",
        PlainPerRowBlockCs[C.DOWN_FWHT_BLOCK]]
    var layer_scalar:    Slot[BF16, Shape[1, 1],                "layer_scalar"]


struct SlidingLayerRefs[degree: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    var attn: SlidingAttnRefs[Self.degree]
    var body: BodyRefs[Self.degree]


struct FullLayerRefs[degree: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    var attn: FullAttnRefs[Self.degree]
    var body: BodyRefs[Self.degree]


struct SlidingKVSlots[degree: Int, max_seq_len: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4StateShapes[Self.degree, Self.max_seq_len]
    var k:       Slot[I8,  Self.S.SlidingKV]
    var k_scale: Slot[F32, Self.S.SlidingKVScale]
    var v:       Slot[I8,  Self.S.SlidingKV]
    var v_scale: Slot[F32, Self.S.SlidingKVScale]


struct FullKVSlots[degree: Int, max_seq_len: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4StateShapes[Self.degree, Self.max_seq_len]
    var k:       Slot[I8,  Self.S.FullKV]
    var k_scale: Slot[F32, Self.S.FullKVScale]
    var v:       Slot[I8,  Self.S.FullKV]
    var v_scale: Slot[F32, Self.S.FullKVScale]


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
    comptime PARTIAL_STRIDE = flash_partial_stride[
        Self.num_q_heads, Self.head_dim,
    ]()

    comptime PHASES = ScratchPhaseOrder[
        "norm_quant", "qkv", "attn_prep", "attention", "o_prep",
    ]

    var x_i8_band: ScratchPhase["norm_quant", "qkv"]
    var x_i8: ScratchBuffer[
        Int8, C.SLIDING_WINDOW * C.HIDDEN,
    ]
    var x_sa: ScratchBuffer[
        Float32, C.SLIDING_WINDOW,
    ]
    var x_row_workspace_band: ScratchPhase["norm_quant", "norm_quant"]
    var x_row_workspace: ScratchBuffer[
        Float32, Self.max_worker_count * C.HIDDEN,
    ]

    var q_band: ScratchPhase["qkv", "o_prep"]
    var q: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * Self.q_rows,
    ]

    var kv_band: ScratchPhase["qkv", "attn_prep"]
    var kv: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * Self.kv_rows * 2,
    ]

    var qprep_band: ScratchPhase["attn_prep", "attention"]
    var q_i8: ScratchBuffer[
        Int8, C.SLIDING_WINDOW * Self.q_rows,
    ]
    var qi_bias: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * Self.num_q_heads,
    ]
    var f_q: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * Self.num_q_heads,
    ]

    var partials_band: ScratchPhase["attention", "attention"]
    var partials: ScratchBuffer[
        Float32, Self.max_worker_count * Self.PARTIAL_STRIDE,
    ]

    var o_band: ScratchPhase["o_prep", "o_prep"]
    var o_i8: ScratchBuffer[
        Int8, C.SLIDING_WINDOW * Self.q_rows,
    ]
    var o_sa: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * Self.num_q_heads,
    ]
    var o_row_workspace: ScratchBuffer[
        Float32, Self.max_worker_count * Self.q_rows,
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
    comptime local_num_q_heads = Self.local_q_rows // Self.head_dim
    comptime PARTIAL_STRIDE = flash_partial_stride[
        Self.num_q_heads, Self.head_dim,
    ]()
    comptime PARTIAL_SLOTS = (
        Self.max_worker_count
        if Self.max_worker_count >= C.SLIDING_WINDOW
        else C.SLIDING_WINDOW
    )

    comptime PHASES = ScratchPhaseOrder[
        "norm_quant", "qkv", "attn_prep", "flash", "merge", "o_prep",
    ]

    var x_i8_band: ScratchPhase["norm_quant", "qkv"]
    var x_i8: ScratchBuffer[
        Int8, C.SLIDING_WINDOW * C.HIDDEN,
    ]
    var x_sa: ScratchBuffer[
        Float32, C.SLIDING_WINDOW,
    ]
    var x_row_workspace_band: ScratchPhase["norm_quant", "norm_quant"]
    var x_row_workspace: ScratchBuffer[
        Float32, Self.max_worker_count * C.HIDDEN,
    ]

    var q_band: ScratchPhase["qkv", "attn_prep"]
    var q: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * Self.q_rows,
    ]

    var kv_band: ScratchPhase["qkv", "attn_prep"]
    var kv: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * Self.k_rows * 2,
    ]

    var qprep_band: ScratchPhase["attn_prep", "flash"]
    var q_i8: ScratchBuffer[
        Int8, C.SLIDING_WINDOW * Self.q_rows,
    ]
    var qi_bias: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * Self.num_q_heads,
    ]
    var f_q: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * Self.num_q_heads,
    ]

    var partials_band: ScratchPhase["flash", "merge"]
    var partials: ScratchBuffer[
        Float32, Self.PARTIAL_SLOTS * Self.PARTIAL_STRIDE,
    ]

    var q_local_band: ScratchPhase["merge", "o_prep"]
    var q_local: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * Self.local_q_rows,
    ]

    var o_band: ScratchPhase["o_prep", "o_prep"]
    var o_i8: ScratchBuffer[
        Int8, C.SLIDING_WINDOW * Self.local_q_rows,
    ]
    var o_sa: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * Self.local_num_q_heads,
    ]
    var o_row_workspace: ScratchBuffer[
        Float32, Self.max_worker_count * Self.local_q_rows,
    ]


@fieldwise_init
struct Gemma4FfnMoeScratch[degree: Int, max_worker_count: Int = 128](
    ScratchIsland, Copyable, ImplicitlyCopyable
):
    comptime S = Gemma4Shapes[Self.degree]
    comptime intermediate_per_rank = Self.S.GateUp.DATA_N
    comptime experts_per_rank = C.NUM_EXPERTS // Self.degree
    comptime nb_down_dense = Self.intermediate_per_rank // C.DOWN_FWHT_BLOCK
    comptime nb_down_moe = C.MOE_INTERMEDIATE // C.DOWN_FWHT_BLOCK
    comptime num_routes = C.SLIDING_WINDOW * C.TOP_K

    comptime PHASES = ScratchPhaseOrder[
        "dense_norm", "dense_gate_up", "dense_down_quant",
        "router_select", "moe_norm", "build_schedules", "phase1",
        "bucket_quant", "phase2", "dense_down_post",
    ]

    var dense_x_band: ScratchPhase["dense_norm", "dense_gate_up"]
    var dense_x_i8: ScratchBuffer[Int8, C.SLIDING_WINDOW * C.HIDDEN]
    var dense_x_sa: ScratchBuffer[Float32, C.SLIDING_WINDOW]
    var dense_x_row_workspace_band: ScratchPhase[
        "dense_norm", "dense_norm",
    ]
    var dense_x_row_workspace: ScratchBuffer[
        Float32, Self.max_worker_count * C.HIDDEN,
    ]

    var ffn_gate_band: ScratchPhase["dense_gate_up", "dense_down_quant"]
    var ffn_gate: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * Self.intermediate_per_rank,
    ]

    var ffn_up_band: ScratchPhase["dense_gate_up", "dense_gate_up"]
    var ffn_up: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * Self.intermediate_per_rank,
    ]

    var dense_gate_band: ScratchPhase["dense_down_quant", "dense_down_post"]
    var dense_gate_i8: ScratchBuffer[
        Int8, C.SLIDING_WINDOW * Self.intermediate_per_rank,
    ]
    var dense_gate_sa: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * Self.nb_down_dense,
    ]
    var dense_gate_row_workspace_band: ScratchPhase[
        "dense_down_quant", "dense_down_quant",
    ]
    var dense_gate_row_workspace: ScratchBuffer[
        Float32, Self.max_worker_count * Self.intermediate_per_rank,
    ]

    var router_workspace: ScratchPhase["router_select", "router_select"]
    var moe_router_scaled: ScratchBuffer[
        Float32, Self.max_worker_count * C.HIDDEN,
    ]

    var router_cands: ScratchPhase["router_select", "router_select"]
    var moe_cands: ScratchBuffer[
        RouterCandidate,
        min(Self.max_worker_count, Self.experts_per_rank)
        * C.SLIDING_WINDOW * C.TOP_K,
    ]

    var router_products: ScratchPhase["router_select", "build_schedules"]
    var moe_route_idx: ScratchBuffer[Int32, C.SLIDING_WINDOW * C.TOP_K]
    var moe_route_w: ScratchBuffer[Float32, C.SLIDING_WINDOW * C.TOP_K]

    var expert_input: ScratchPhase["moe_norm", "phase1"]
    var moe_x_i8: ScratchBuffer[Int8, C.SLIDING_WINDOW * C.HIDDEN]
    var moe_x_sa: ScratchBuffer[Float32, C.SLIDING_WINDOW]
    var moe_x_row_workspace_band: ScratchPhase[
        "moe_norm", "moe_norm",
    ]
    var moe_x_row_workspace: ScratchBuffer[
        Float32, Self.max_worker_count * C.HIDDEN,
    ]

    var schedule_products: ScratchPhase["build_schedules", "phase2"]
    var moe_expert_offset: ScratchBuffer[Int32, Self.experts_per_rank + 1]
    var moe_routes: ScratchBuffer[SparseRoute, C.SLIDING_WINDOW * C.TOP_K]

    var hidden_bucket: ScratchPhase["phase1", "bucket_quant"]
    var moe_hidden_bucket: ScratchBuffer[
        BFloat16, Self.num_routes * C.MOE_INTERMEDIATE,
    ]

    var bucket_i8_band: ScratchPhase["bucket_quant", "phase2"]
    var moe_bucket_i8: ScratchBuffer[
        Int8, Self.num_routes * C.MOE_INTERMEDIATE,
    ]
    var moe_bucket_sa: ScratchBuffer[
        Float32, Self.num_routes * Self.nb_down_moe,
    ]
    var bucket_row_workspace_band: ScratchPhase[
        "bucket_quant", "bucket_quant",
    ]
    var bucket_row_workspace: ScratchBuffer[
        Float32, Self.max_worker_count * C.MOE_INTERMEDIATE,
    ]

    var phase2_accum: ScratchPhase["phase2", "phase2"]
    var moe_accum: ScratchBuffer[Float32, C.SLIDING_WINDOW * C.HIDDEN]

    var dense_band: ScratchPhase["dense_down_post", "dense_down_post"]
    var ffn_dense_out: ScratchBuffer[BFloat16, C.SLIDING_WINDOW * C.HIDDEN]


@fieldwise_init
struct Gemma4HeadScratch[degree: Int, max_worker_count: Int = 128](
    ScratchIsland, Copyable, ImplicitlyCopyable
):
    comptime PHASES = ScratchPhaseOrder[
        "embed", "logits",
    ]
    comptime vocab_per_rank = C.VOCAB_SIZE // Self.degree
    comptime head_nb = C.HIDDEN // 128

    var embed_row_workspace_band: ScratchPhase[
        "embed", "embed",
    ]
    var embed_row_workspace: ScratchBuffer[
        Float32, Self.max_worker_count * C.HIDDEN,
    ]

    var head_prep_band: ScratchPhase["logits", "logits"]
    var head_x_i8: ScratchBuffer[Int8, C.HIDDEN]
    var head_x_sa: ScratchBuffer[Float32, Self.head_nb]
    var head_row_workspace: ScratchBuffer[Float32, C.HIDDEN]

    var logits_band: ScratchPhase["logits", "logits"]
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
    var head: Gemma4HeadScratch[Self.degree, Self.max_worker_count]


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
    comptime assert VnniPackContract[degree], "VNNI pack alignment invariant failed: a packed weight's per-rank DATA_N must be a multiple of 32 and DATA_M a multiple of 64"
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


def dispatch_bq_sliding_attention_qkv[
    P: BurstThreadPool, Profile: Bool, N: Int, //, degree: Int, max_seq_len: Int,
    max_worker_count: Int = 128,
](
    layout: Gemma4Layout[degree, max_seq_len],
    ctx: BindContext[degree],
    act: ButterquantActivation[degree],
    base_pos: Int,
    seq_len: Int,
    layer_idx: Int,
    mut scratch: Gemma4ScratchPool[degree, max_worker_count],
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime S = Gemma4Shapes[degree]
    comptime q_rows = S.SlidingQ.DATA_N
    comptime kv_rows = S.SlidingKV.DATA_N
    comptime head_dim = C.HEAD_DIM_SLIDING
    comptime num_q_heads = q_rows // head_dim
    comptime num_kv_heads = kv_rows // head_dim
    comptime sqrt_hd = sqrt[DType.float32, 1](head_dim)
    comptime hd_eps = Float32(head_dim) * Float32(C.RMS_NORM_EPS)
    comptime rope_half = C.ROPE_HALF_SLIDING
    comptime cache_size = Gemma4StateShapes[degree, max_seq_len].SLIDING_CACHE
    comptime Island = Gemma4SlidingScratch[degree, max_worker_count]

    var attn_ctx = ctx.with_layer(
        layout.sliding.base(ctx.arena_bases[0], layer_idx))
    var attn = layout.sliding.proto.attn

    var q_outs = scratch.binding[Island, "q"](ctx)
    var k_outs = scratch.binding[Island, "kv"](ctx)
    var v_outs = k_outs.shifted(seq_len * kv_rows)

    dispatch_bq_qkv[max_worker_count=max_worker_count](
        act,
        attn.q_proj.bq_weight(attn_ctx),
        attn.k_proj.bq_weight(attn_ctx),
        attn.v_proj.bq_weight(attn_ctx),
        q_outs, k_outs, v_outs, seq_len, pools, prof)

    var q_i8 = scratch.binding[Island, "q_i8"](ctx)
    var qi_bias = scratch.binding[Island, "qi_bias"](ctx)
    var f_q = scratch.binding[Island, "f_q"](ctx)

    var kv_lb = layout.sliding_kv.base(ctx.arena_bases[0], layer_idx)
    var k_cache = layout.sliding_kv.proto.k.binding(kv_lb, ctx.arena_bases)
    var k_scale = layout.sliding_kv.proto.k_scale.binding(kv_lb, ctx.arena_bases)
    var v_cache = layout.sliding_kv.proto.v.binding(kv_lb, ctx.arena_bases)
    var v_scale = layout.sliding_kv.proto.v_scale.binding(kv_lb, ctx.arena_bases)

    dispatch_bq_attn_prep[
        head_dim=head_dim, num_q=num_q_heads, num_kv=num_kv_heads,
        rope_half=rope_half, pair_stride=head_dim // 2,
        slot_mask=cache_size - 1, cache_degree=1,
        sqrt_n=sqrt_hd, n_eps=hd_eps, tp=degree,
        max_worker_count=max_worker_count,
    ](q_outs, k_outs, v_outs,
      attn.q_norm.binding(attn_ctx), attn.k_norm.binding(attn_ctx),
      q_i8, qi_bias, f_q, k_cache, k_scale, v_cache, v_scale,
      layout.sliding_rope.cos.state_binding(ctx),
      layout.sliding_rope.sin.state_binding(ctx),
      base_pos, seq_len, pools, prof)

    var partials = scratch.binding[Island, "partials"](ctx)

    dispatch_bq_sliding_attention[
        head_dim=head_dim, num_q=num_q_heads, num_kv=num_kv_heads,
        gqa_ratio=num_q_heads // num_kv_heads, kv_stride=kv_rows,
        window=C.SLIDING_WINDOW, cache_size=cache_size,
        partial_stride=Island.PARTIAL_STRIDE, tp=degree,
        max_worker_count=max_worker_count,
    ](q_i8, qi_bias, f_q, k_cache, k_scale, v_cache, v_scale,
      q_outs, partials, base_pos, seq_len, pools, prof)

    var o_i8 = scratch.binding[Island, "o_i8"](ctx)
    var o_sa = scratch.binding[Island, "o_sa"](ctx)
    var o_row_workspace = scratch.binding[Island, "o_row_workspace"](ctx)

    dispatch_bq_block_quant[
        cols=q_rows, block=head_dim, apply_fwht=False, tp=degree,
        max_worker_count=max_worker_count,
    ](q_outs, o_i8, o_sa, o_row_workspace, seq_len, pools, prof)

    var o_act = ButterquantBlockActivation[degree](o_i8, o_sa)
    var xs = layout.activations.x_residual.state_binding(ctx)

    dispatch_bq_block_linear[max_worker_count=max_worker_count](
        o_act, attn.o_proj.bq_weight(attn_ctx), xs, seq_len, pools, prof)


def dispatch_bq_full_attention_qkv[
    P: BurstThreadPool, Profile: Bool, N: Int, //, degree: Int, max_seq_len: Int,
    max_worker_count: Int = 128,
](
    layout: Gemma4Layout[degree, max_seq_len],
    ctx: BindContext[degree],
    act: ButterquantActivation[degree],
    base_pos: Int,
    seq_len: Int,
    layer_idx: Int,
    mut scratch: Gemma4ScratchPool[degree, max_worker_count],
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime S = Gemma4Shapes[degree]
    comptime q_rows = S.FullQ.DATA_N
    comptime k_rows = S.FullK.DATA_N
    comptime local_q_rows = S.FullO.DATA_M
    comptime head_dim = C.HEAD_DIM_FULL
    comptime num_q_heads = q_rows // head_dim
    comptime local_num_q_heads = local_q_rows // head_dim
    comptime num_kv_heads = k_rows // head_dim
    comptime sqrt_hd = sqrt[DType.float32, 1](head_dim)
    comptime hd_eps = Float32(head_dim) * Float32(C.RMS_NORM_EPS)
    comptime rope_half = C.ROPE_HALF_FULL
    comptime Island = Gemma4FullScratch[degree, max_worker_count]

    var attn_ctx = ctx.with_layer(
        layout.full.base(ctx.arena_bases[0], layer_idx))
    var attn = layout.full.proto.attn

    var q_outs = scratch.binding[Island, "q"](ctx)
    var k_outs = scratch.binding[Island, "kv"](ctx)

    dispatch_bq_linear[max_worker_count=max_worker_count](
        act, attn.q_proj.bq_weight(attn_ctx), q_outs, seq_len, pools, prof)
    dispatch_bq_linear[max_worker_count=max_worker_count](
        act, attn.k_proj.bq_weight(attn_ctx), k_outs, seq_len, pools, prof)

    var q_i8 = scratch.binding[Island, "q_i8"](ctx)
    var qi_bias = scratch.binding[Island, "qi_bias"](ctx)
    var f_q = scratch.binding[Island, "f_q"](ctx)

    var kv_lb = layout.full_kv.base(ctx.arena_bases[0], layer_idx)
    var k_cache = layout.full_kv.proto.k.binding(kv_lb, ctx.arena_bases)
    var k_scale = layout.full_kv.proto.k_scale.binding(kv_lb, ctx.arena_bases)
    var v_cache = layout.full_kv.proto.v.binding(kv_lb, ctx.arena_bases)
    var v_scale = layout.full_kv.proto.v_scale.binding(kv_lb, ctx.arena_bases)

    dispatch_bq_attn_prep[
        head_dim=head_dim, num_q=num_q_heads, num_kv=num_kv_heads,
        rope_half=rope_half, pair_stride=head_dim // 2,
        slot_mask=-1, cache_degree=degree,
        sqrt_n=sqrt_hd, n_eps=hd_eps, tp=degree,
        max_worker_count=max_worker_count,
    ](q_outs, k_outs, k_outs,
      attn.q_norm.binding(attn_ctx), attn.k_norm.binding(attn_ctx),
      q_i8, qi_bias, f_q, k_cache, k_scale, v_cache, v_scale,
      layout.full_rope.cos.state_binding(ctx),
      layout.full_rope.sin.state_binding(ctx),
      base_pos, seq_len, pools, prof)

    var q_local = scratch.binding[Island, "q_local"](ctx)
    var partials = scratch.binding[Island, "partials"](ctx)

    dispatch_bq_full_attention[
        head_dim=head_dim, num_q=num_q_heads, num_kv=num_kv_heads,
        local_num_q=local_num_q_heads,
        gqa_ratio=num_q_heads // num_kv_heads, kv_stride=k_rows,
        partial_stride=Island.PARTIAL_STRIDE, tp=degree,
        max_worker_count=max_worker_count,
    ](q_i8, qi_bias, f_q, k_cache, k_scale, v_cache, v_scale,
      q_local, partials, base_pos, seq_len, pools, prof)

    var o_i8 = scratch.binding[Island, "o_i8"](ctx)
    var o_sa = scratch.binding[Island, "o_sa"](ctx)
    var o_row_workspace = scratch.binding[Island, "o_row_workspace"](ctx)

    dispatch_bq_block_quant[
        cols=local_q_rows, block=head_dim, apply_fwht=False, tp=degree,
        max_worker_count=max_worker_count,
    ](q_local, o_i8, o_sa, o_row_workspace, seq_len, pools, prof)

    var o_act = ButterquantBlockActivation[degree](o_i8, o_sa)
    var xs = layout.activations.x_residual.state_binding(ctx)

    dispatch_bq_block_linear[max_worker_count=max_worker_count](
        o_act, attn.o_proj.bq_weight(attn_ctx), xs, seq_len, pools, prof)


def dispatch_bq_moe[
    P: BurstThreadPool, Profile: Bool, N: Int, //, degree: Int, max_worker_count: Int = 128,
](
    body: BodyRefs[degree],
    ctx: BindContext[degree],
    x_input: Binding[BFloat16, degree],
    moe_out: Binding[BFloat16, degree],
    seq_len: Int,
    mut scratch: Gemma4ScratchPool[degree, max_worker_count],
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime experts_per_rank = C.NUM_EXPERTS // degree
    comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
    comptime n_eps = Float32(C.HIDDEN) * Float32(C.RMS_NORM_EPS)
    comptime Ffn = Gemma4FfnMoeScratch[degree, max_worker_count]

    var per_expert_scale_ptr = body.router_pes.at(ctx.layer_base)
    var router_scaled = scratch.binding[Ffn, "moe_router_scaled"](ctx)
    var cands = scratch.binding[Ffn, "moe_cands"](ctx)
    var route_idx = scratch.binding[Ffn, "moe_route_idx"](ctx)
    var route_w = scratch.binding[Ffn, "moe_route_w"](ctx)
    var expert_offset = scratch.binding[Ffn, "moe_expert_offset"](ctx)
    var routes = scratch.binding[Ffn, "moe_routes"](ctx)
    var moe_x_i8 = scratch.binding[Ffn, "moe_x_i8"](ctx)
    var moe_x_sa = scratch.binding[Ffn, "moe_x_sa"](ctx)
    var moe_x_row_workspace = scratch.binding[
        Ffn, "moe_x_row_workspace",
    ](ctx)
    var bucket = scratch.binding[Ffn, "moe_hidden_bucket"](ctx)
    var bucket_i8 = scratch.binding[Ffn, "moe_bucket_i8"](ctx)
    var bucket_sa = scratch.binding[Ffn, "moe_bucket_sa"](ctx)
    var bucket_row_workspace = scratch.binding[
        Ffn, "bucket_row_workspace",
    ](ctx)
    var moe_accum = scratch.binding[Ffn, "moe_accum"](ctx)

    var nws = dispatch_router_expert[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
        experts_per_rank=experts_per_rank, top_k=C.TOP_K, tp=degree,
        max_worker_count=max_worker_count,
    ](x_input, body.router_proj.bq_router(ctx).centered,
      body.router_scale.binding(ctx), router_scaled, cands, seq_len, pools, prof)

    merge_router_candidates_expert[degree, C.TOP_K](
        cands, nws, seq_len, per_expert_scale_ptr, route_idx, route_w)

    dispatch_bq_norm_quant[
        hidden=C.HIDDEN, block=128, sqrt_n=sqrt_n, n_eps=n_eps, tp=degree,
        max_worker_count=max_worker_count,
    ](x_input, body.pre_ffn_norm_2.binding(ctx),
      moe_x_i8, moe_x_sa, moe_x_row_workspace, seq_len, pools, prof)

    build_expert_schedules[degree, experts_per_rank, C.TOP_K](
        route_idx, route_w, expert_offset, routes, seq_len)

    var moe_act = ButterquantActivation[degree](moe_x_i8, moe_x_sa)
    dispatch_bq_phase1_gate_up[
        hidden=C.HIDDEN, gate_up=C.MOE_GATE_UP_FUSED,
        inter=C.MOE_INTERMEDIATE, experts_per_rank=experts_per_rank,
        max_worker_count=max_worker_count,
    ](moe_act, expert_offset, routes,
      body.experts_gate_up.bq_weight(ctx), bucket, pools, prof)

    var num_routes = seq_len * C.TOP_K
    dispatch_bq_block_quant[
        cols=C.MOE_INTERMEDIATE, block=C.DOWN_FWHT_BLOCK, apply_fwht=True,
        tp=degree, max_worker_count=max_worker_count,
    ](
        bucket, bucket_i8, bucket_sa, bucket_row_workspace, num_routes,
        pools, prof)

    var bucket_act = ButterquantBlockActivation[degree](bucket_i8, bucket_sa)
    dispatch_bq_phase2_down[
        hidden=C.HIDDEN, inter=C.MOE_INTERMEDIATE,
        experts_per_rank=experts_per_rank,
        max_worker_count=max_worker_count,
    ](bucket_act, expert_offset, routes,
      body.experts_down.bq_weight(ctx), moe_accum, moe_out, seq_len, pools, prof)

    dispatch_allreduce_inplace[
        BF16, degree, max_worker_count=max_worker_count,
    ](moe_out, seq_len * C.HIDDEN, pools, prof)


def dispatch_bq_ffn[
    P: BurstThreadPool, Profile: Bool, N: Int, //, degree: Int, max_worker_count: Int = 128,
](
    body: BodyRefs[degree],
    ctx: BindContext[degree],
    x_main: Binding[BFloat16, degree],
    x_residual: Binding[BFloat16, degree],
    seq_len: Int,
    mut scratch: Gemma4ScratchPool[degree, max_worker_count],
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
    comptime n_eps = Float32(C.HIDDEN) * Float32(C.RMS_NORM_EPS)
    comptime intermediate_per_rank = Gemma4Shapes[degree].GateUp.DATA_N
    comptime Ffn = Gemma4FfnMoeScratch[degree, max_worker_count]

    var layer_scalar_ptr = body.layer_scalar.at(ctx.layer_base)
    var dense_x_i8 = scratch.binding[Ffn, "dense_x_i8"](ctx)
    var dense_x_sa = scratch.binding[Ffn, "dense_x_sa"](ctx)
    var dense_x_row_workspace = scratch.binding[
        Ffn, "dense_x_row_workspace",
    ](ctx)
    var gate = scratch.binding[Ffn, "ffn_gate"](ctx)
    var up = scratch.binding[Ffn, "ffn_up"](ctx)
    var dense_gate_i8 = scratch.binding[Ffn, "dense_gate_i8"](ctx)
    var dense_gate_sa = scratch.binding[Ffn, "dense_gate_sa"](ctx)
    var dense_gate_row_workspace = scratch.binding[
        Ffn, "dense_gate_row_workspace",
    ](ctx)
    var dense_out = scratch.binding[Ffn, "ffn_dense_out"](ctx)

    dispatch_bq_norm_quant[
        hidden=C.HIDDEN, block=128, sqrt_n=sqrt_n, n_eps=n_eps, tp=degree,
        max_worker_count=max_worker_count,
    ](x_main, body.pre_ffn_norm.binding(ctx),
      dense_x_i8, dense_x_sa, dense_x_row_workspace, seq_len, pools, prof)

    var dense_act = ButterquantActivation[degree](dense_x_i8, dense_x_sa)
    dispatch_bq_linear[max_worker_count=max_worker_count](
        dense_act, body.gate_proj.bq_weight(ctx), gate, seq_len, pools, prof)
    dispatch_bq_linear[max_worker_count=max_worker_count](
        dense_act, body.up_proj.bq_weight(ctx), up, seq_len, pools, prof)

    dispatch_gelu_gate_up[
        intermediate=intermediate_per_rank, tp=degree,
        max_worker_count=max_worker_count,
    ](gate, up, gate, seq_len, pools, prof)

    dispatch_bq_block_quant[
        cols=intermediate_per_rank, block=C.DOWN_FWHT_BLOCK, apply_fwht=True,
        tp=degree, max_worker_count=max_worker_count,
    ](
        gate, dense_gate_i8, dense_gate_sa, dense_gate_row_workspace,
        seq_len, pools, prof)

    dispatch_bq_moe[degree=degree, max_worker_count=max_worker_count](
        body, ctx, x_main, x_residual, seq_len, scratch, pools, prof)

    var dense_gate_act = ButterquantBlockActivation[degree](
        dense_gate_i8, dense_gate_sa)
    dispatch_bq_block_linear[max_worker_count=max_worker_count](
        dense_gate_act, body.down_proj.bq_weight(ctx), dense_out, seq_len, pools, prof)

    dispatch_allreduce_inplace[
        BF16, degree, max_worker_count=max_worker_count,
    ](dense_out, seq_len * C.HIDDEN, pools, prof)

    dispatch_rms_norm[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps, tp=degree,
        max_worker_count=max_worker_count,
    ](dense_out, dense_out, body.post_ffn_norm_1.binding(ctx), seq_len, pools, prof)

    fused_norm_residual_add[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps, tp=degree,
        max_worker_count=max_worker_count,
    ](x_residual, dense_out, dense_out,
      body.post_ffn_norm_2.binding(ctx), seq_len, pools, prof)

    fused_norm_residual_add[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps, tp=degree,
        max_worker_count=max_worker_count,
    ](dense_out, x_main, x_main,
      body.post_ffn_norm.binding(ctx), seq_len, pools, prof)

    var ls_value = layer_scalar_ptr[0].cast[DType.float32]()
    dispatch_scalar_mul[
        hidden=C.HIDDEN, tp=degree, max_worker_count=max_worker_count,
    ](x_main, x_main, ls_value, seq_len, pools, prof)


struct Gemma4[
    degree: Int, max_seq_len: Int = 8192,
    max_worker_count: Int = 128,
    Pool: BurstThreadPool = BurstPool[],
    profile: Bool = False, profile_slots: Int = 64,
](Movable):
    var arenas: List[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var pools: List[Self.Pool]
    var layout: Gemma4Layout[Self.degree, Self.max_seq_len]
    var scratch: Gemma4ScratchPool[Self.degree, Self.max_worker_count]
    var arena_bases: ArenaBases[Self.degree]
    var profiler: Profiler[Self.profile, Self.profile_slots]

    def __init__(out self,
        var arenas: List[NumaArena[alignment=DEFAULT_ALIGNMENT]],
        var pools: List[Self.Pool],
        layout: Gemma4Layout[Self.degree, Self.max_seq_len],
    ):
        self.arena_bases = ArenaBases[Self.degree].uninitialized()
        for r in range(Self.degree):
            self.arena_bases[r] = Int(arenas[r].base.value())
        self.layout = layout.bind(self.arena_bases[0])
        self.arenas = arenas^
        self.pools = pools^
        self.scratch = Gemma4ScratchPool[
            Self.degree, Self.max_worker_count,
    ](
            self.layout.arena.scratch_base())
        self.profiler = Profiler[Self.profile, Self.profile_slots]()

    def init_state(mut self):
        var tasks = List[PackColsumTask]()
        for i in range(C.NUM_LAYERS):
            var entry = LAYER_SCHEDULE[i]
            if entry.kind == LayerKind.FULL:
                _ = emit_pack_tasks[FullLayerRefs[Self.degree]](
                    self.layout.full.off
                    + entry.local_idx * self.layout.full.stride,
                    tasks)
            else:
                _ = emit_pack_tasks[SlidingLayerRefs[Self.degree]](
                    self.layout.sliding.off
                    + entry.local_idx * self.layout.sliding.stride,
                    tasks)
        _ = emit_pack_tasks[TailRefs[Self.degree]](self.layout.tail.off, tasks)
        var nodes = InlineArray[Int, Self.degree](uninitialized=True)
        for r in range(Self.degree):
            nodes[r] = self.arenas[r].node
        var noprof = Profiler[False]()  # one-time setup, not part of forward profiling
        dispatch_pack_colsum[
            Self.degree, max_worker_count=Self.max_worker_count,
        ](self.pools, noprof, self.arena_bases, nodes, tasks)

    def model_init(mut self):
        ref layout = self.layout
        comptime width = simd_width_of[DType.float32]()
        comptime inv_sqrt_hidden = 1.0 / sqrt[DType.float32, 1](C.HIDDEN)

        prime_fp_environment[Self.degree, Self.max_worker_count](self.pools)

        comptime if has_amx_int8():
            prime_amx_environment[Self.degree, Self.max_worker_count](self.pools)

        @parameter
        def bake_router_scale(p: UnsafePointer[BFloat16, MutAnyOrigin]):
            for j in range(0, C.HIDDEN, width):
                var lane = p + j
                var v = lane.load[width=width]().cast[DType.float32]()
                lane.store((v * inv_sqrt_hidden).cast[DType.bfloat16]())

        for rank in range(Self.degree):
            var base = self.arena_bases[rank]
            for i in range(C.NUM_LAYERS):
                var entry = LAYER_SCHEDULE[i]
                if entry.kind == LayerKind.FULL:
                    var lb = layout.full.base(base, entry.local_idx)
                    ref fbody = layout.full.proto.body
                    bake_split_gain_in_place(fbody.input_norm.at(lb), C.HIDDEN)
                    bake_split_gain_in_place(fbody.pre_ffn_norm.at(lb), C.HIDDEN)
                    bake_split_gain_in_place(fbody.pre_ffn_norm_2.at(lb), C.HIDDEN)
                    bake_router_scale(fbody.router_scale.at(lb))
                else:
                    var lb = layout.sliding.base(base, entry.local_idx)
                    ref sbody = layout.sliding.proto.body
                    bake_split_gain_in_place(sbody.input_norm.at(lb), C.HIDDEN)
                    bake_split_gain_in_place(sbody.pre_ffn_norm.at(lb), C.HIDDEN)
                    bake_split_gain_in_place(sbody.pre_ffn_norm_2.at(lb), C.HIDDEN)
                    bake_router_scale(sbody.router_scale.at(lb))

        from kernels.rope import (
            init_rope_table, init_rope_table_partial_strided,
        )
        for rank in range(Self.degree):
            var base = self.arena_bases[rank]
            var sl_cos = layout.sliding_rope.cos.at(base)
            var sl_sin = layout.sliding_rope.sin.at(base)
            init_rope_table[C.ROPE_HALF_SLIDING, Self.max_seq_len](
                sl_cos, sl_sin, 10000.0)
            var fl_cos = layout.full_rope.cos.at(base)
            var fl_sin = layout.full_rope.sin.at(base)
            init_rope_table_partial_strided[
                C.ROPE_HALF_FULL, Self.max_seq_len,
            ](fl_cos, fl_sin, 1000000.0, C.HEAD_DIM_FULL, 0, 1)

    def forward[
        tok_origin: ImmutOrigin, //,
    ](
        mut self,
        token_ids: Span[Int32, tok_origin],
        base_pos: Int = 0,
    ) -> TemporalLogitsView[C.VOCAB_SIZE, Self.degree]:
        ref layout = self.layout
        comptime shard_rows = Gemma4TailShapes[Self.degree].Embed.DATA_N
        comptime embed_scale = Float64(sqrt[DType.float32, 1](C.HIDDEN)
            .cast[DType.bfloat16]().cast[DType.float32]())
        comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
        comptime n_eps = Float32(C.HIDDEN) * Float32(C.RMS_NORM_EPS)
        comptime SS = Gemma4SlidingScratch[Self.degree, Self.max_worker_count]
        comptime FS = Gemma4FullScratch[Self.degree, Self.max_worker_count]
        comptime HS = Gemma4HeadScratch[Self.degree, Self.max_worker_count]

        var wall_t0 = perf_counter_ns()
        var total_len = len(token_ids)
        debug_assert(total_len > 0, "forward called with empty token_ids")
        debug_assert(
            base_pos + total_len <= Self.max_seq_len,
            "forward exceeds max_seq_len")

        var ctx = BindContext[Self.degree](
            arena_bases=self.arena_bases, layer_base=0)
        var tail_ctx = ctx.with_layer(layout.tail.base(self.arena_bases[0], 0))

        var x_main_ranks = layout.activations.x_main.state_binding(ctx)
        var xs = layout.activations.x_residual.state_binding(ctx)
        var logits = self.scratch.binding[HS, "logits"](ctx)

        var consumed = 0
        var pos = base_pos
        while consumed < total_len:
            var remaining = total_len - consumed
            var chunk_len = remaining if remaining < C.SLIDING_WINDOW else C.SLIDING_WINDOW
            var is_last = (consumed + chunk_len == total_len)

            var chunk = Span[Int32, tok_origin](
                ptr=token_ids.unsafe_ptr() + consumed, length=chunk_len)

            var embed_row_workspace = self.scratch.binding[
                HS, "embed_row_workspace",
            ](ctx)
            dispatch_bq_embed_lookup[
                scale=embed_scale, shard_rows=shard_rows,
                max_worker_count=Self.max_worker_count,
            ](chunk, layout.tail.proto.embed.bq_weight(tail_ctx),
              x_main_ranks, embed_row_workspace,
              chunk_len, self.pools, self.profiler)
            dispatch_allreduce_inplace[
                BF16, Self.degree, max_worker_count=Self.max_worker_count,
            ](x_main_ranks, chunk_len * C.HIDDEN, self.pools, self.profiler)

            for i in range(C.NUM_LAYERS):
                var entry = LAYER_SCHEDULE[i]
                var body: BodyRefs[Self.degree]
                var layer_ctx: BindContext[Self.degree]
                if entry.kind == LayerKind.FULL:
                    layer_ctx = ctx.with_layer(
                        layout.full.base(self.arena_bases[0], entry.local_idx))
                    body = layout.full.proto.body
                    var fx_i8 = self.scratch.binding[FS, "x_i8"](ctx)
                    var fx_sa = self.scratch.binding[FS, "x_sa"](ctx)
                    var fx_row_workspace = self.scratch.binding[
                        FS, "x_row_workspace",
                    ](ctx)
                    dispatch_bq_norm_quant[
                        hidden=C.HIDDEN, block=128, sqrt_n=sqrt_n, n_eps=n_eps,
                        tp=Self.degree, max_worker_count=Self.max_worker_count,
                    ](x_main_ranks, body.input_norm.binding(layer_ctx),
                      fx_i8, fx_sa, fx_row_workspace,
                      chunk_len, self.pools, self.profiler)
                    var full_act = ButterquantActivation[Self.degree](
                        fx_i8, fx_sa)
                    dispatch_bq_full_attention_qkv[
                        degree=Self.degree, max_seq_len=Self.max_seq_len,
                        max_worker_count=Self.max_worker_count,
                    ](layout, ctx, full_act, pos, chunk_len, entry.local_idx,
                      self.scratch, self.pools, self.profiler)
                else:
                    layer_ctx = ctx.with_layer(
                        layout.sliding.base(
                            self.arena_bases[0], entry.local_idx))
                    body = layout.sliding.proto.body
                    var sx_i8 = self.scratch.binding[SS, "x_i8"](ctx)
                    var sx_sa = self.scratch.binding[SS, "x_sa"](ctx)
                    var sx_row_workspace = self.scratch.binding[
                        SS, "x_row_workspace",
                    ](ctx)
                    dispatch_bq_norm_quant[
                        hidden=C.HIDDEN, block=128, sqrt_n=sqrt_n, n_eps=n_eps,
                        tp=Self.degree, max_worker_count=Self.max_worker_count,
                    ](x_main_ranks, body.input_norm.binding(layer_ctx),
                      sx_i8, sx_sa, sx_row_workspace,
                      chunk_len, self.pools, self.profiler)
                    var sl_act = ButterquantActivation[Self.degree](
                        sx_i8, sx_sa)
                    dispatch_bq_sliding_attention_qkv[
                        degree=Self.degree, max_seq_len=Self.max_seq_len,
                        max_worker_count=Self.max_worker_count,
                    ](layout, ctx, sl_act, pos, chunk_len, entry.local_idx,
                      self.scratch, self.pools, self.profiler)

                dispatch_allreduce_inplace[
                    BF16, Self.degree, max_worker_count=Self.max_worker_count,
                ](xs, chunk_len * C.HIDDEN, self.pools, self.profiler)

                fused_norm_residual_add[
                    hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps, tp=Self.degree,
                    max_worker_count=Self.max_worker_count,
                ](xs, x_main_ranks, x_main_ranks,
                  body.post_attn_norm.binding(layer_ctx), chunk_len, self.pools, self.profiler)

                dispatch_bq_ffn[
                    degree=Self.degree, max_worker_count=Self.max_worker_count,
                ](body, layer_ctx, x_main_ranks, xs, chunk_len,
                  self.scratch, self.pools, self.profiler)

            if is_last:
                var x_last = x_main_ranks.shifted((chunk_len - 1) * C.HIDDEN)
                var head_x_i8 = self.scratch.binding[HS, "head_x_i8"](ctx)
                var head_x_sa = self.scratch.binding[HS, "head_x_sa"](ctx)
                var head_act = ButterquantActivation[Self.degree](
                    head_x_i8, head_x_sa)
                var head_row_workspace = self.scratch.binding[
                    HS, "head_row_workspace",
                ](ctx)

                dispatch_bq_head_prep[
                    hidden=C.HIDDEN, block=128, sqrt_n=sqrt_n, n_eps=n_eps,
                ](x_last, layout.tail.proto.final_norm.binding(tail_ctx),
                  head_act, head_row_workspace)

                dispatch_bq_head_gemv[
                    cap=C.LOGIT_SOFTCAP,
                    max_worker_count=Self.max_worker_count,
                ](head_act, layout.tail.proto.embed.bq_weight(tail_ctx),
                  logits, self.pools, self.profiler)

            consumed += chunk_len
            pos += chunk_len

        self.profiler.add_wall(Int(perf_counter_ns() - wall_t0))
        return TemporalLogitsView[C.VOCAB_SIZE, Self.degree](
            logits.ptr, self.arena_bases)

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

        var model = Self(arenas^, pools^, layout)
        model.init_state()
        model.model_init()
        return model^

    @staticmethod
    def quantize(
        source_dir: Path, output_path: Path,
        topo: NumaTopology, var pools: List[Self.Pool],
    ) -> Bool:
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
