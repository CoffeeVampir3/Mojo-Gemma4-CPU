from std.os import abort
from std.pathlib import Path
from std.memory import Span, UnsafePointer
from std.sys.info import simd_width_of
from std.time import perf_counter_ns
from simd_math.ops import sqrt

from numa import NumaArena, NumaTopology
from threading import BurstPool
from threading.threading_traits import BurstThreadPool
from kernels.helpers import RankView, Binding, prime_fp_environment
from kernels.profiling import Profiler
from kernels.reductions import dispatch_allreduce_inplace
from kernels.rmsnorm import dispatch_rms_norm, fused_norm_residual_add
from kernels.elementwise import dispatch_gelu_gate_up, dispatch_scalar_mul
from kernels.attention_ops import flash_partial_stride
from kernels.logsum_merge import MergeSegment
from kernels.moe_router import (
    RouterCandidate, SparseRoute,
    dispatch_router_expert, dispatch_merge_router_candidates,
    dispatch_build_expert_schedules,
)
from butterquant_kernels import (
    dispatch_bq_embed_lookup, dispatch_bq_norm_quant, dispatch_bq_qkv,
    dispatch_bq_linear, dispatch_bq_attn_prep,
    dispatch_bq_sliding_attention, dispatch_bq_full_attention,
    dispatch_bq_block_quant, dispatch_bq_block_linear,
    dispatch_bq_phase1_gate_up, dispatch_bq_phase2_down,
    dispatch_bq_head_prep, dispatch_bq_head_gemv,
)
from butterquant import (
    PackColsumTask, dispatch_pack_colsum,
    bake_split_gain_in_place, ButterquantActivation, ButterquantBlockActivation,
)
from butterquant.vnni import VNNI_N_STEP, VNNI_K_STEP
from butterquant.amx_tiles import prime_amx_environment
from modeling.temporal_scratch import (
    ScratchBuffer, ScratchIsland, ScratchPhase, ScratchPhaseOrder, ScaleClass,
    TemporalScratchPool, TemporalLogitsView, ScratchPlan,
    derive_scratch_plan, aggregate_scratch_peak, co_live_buffers_overlap,
)

from modeling.model_spec import (
    BF16, F32, I8,
    Shape, WeightDesc,
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
    vnni_pack_contract_ok,
)
from quant.recipe import (
    QuantRecipe, PerRowQuant, PerBlockQuant, SoftmaxRouterCenter,
    SplitGamma, NoGamma, SingleSided, PerRowCs, PerBlockCs, NoColsum,
    VnniPacked, RowMajor,
)
from quant.quantizer import Quantizer
from modeling.loader import discover_shards, load_weights_from_descs


comptime C = Gemma4BaseConfig
comptime MAX_WORKERS = 128


@always_inline
def degree_contracts_ok(degree: Int) -> Bool:
    return (
        degree > 0
        and C.NUM_HEADS % degree == 0
        and C.NUM_KV_HEADS_SLIDING % degree == 0
        and C.Q_DIM_SLIDING % degree == 0
        and C.KV_DIM_SLIDING % degree == 0
        and C.Q_DIM_FULL % degree == 0
        and C.INTERMEDIATE % degree == 0
        and C.NUM_EXPERTS % degree == 0
        and C.VOCAB_SIZE % degree == 0
    )


struct Gemma4Shapes:
    comptime GateUp      = TensorRowSharded[C.INTERMEDIATE, C.HIDDEN, block=C.DOWN_FWHT_BLOCK]
    comptime Down        = TensorColumnSharded[C.HIDDEN, C.INTERMEDIATE, block=C.DOWN_FWHT_BLOCK]
    comptime SlidingQ    = TensorRowSharded[C.Q_DIM_SLIDING, C.HIDDEN]
    comptime SlidingKV   = TensorRowSharded[C.KV_DIM_SLIDING, C.HIDDEN]
    comptime SlidingO    = TensorColumnSharded[C.HIDDEN, C.Q_DIM_SLIDING]
    comptime FullQ       = Replicated[C.Q_DIM_FULL, C.HIDDEN]
    comptime FullK       = Replicated[C.KV_DIM_FULL, C.HIDDEN]
    comptime FullO       = TensorColumnSharded[C.HIDDEN, C.Q_DIM_FULL]
    comptime RouterProj  = ExpertRowBlockSharded[C.NUM_EXPERTS, 1, C.HIDDEN]
    comptime ExpertsGateUp = ExpertRowBlockSharded[
        C.NUM_EXPERTS, C.MOE_GATE_UP_FUSED, C.HIDDEN,
    ]
    comptime ExpertsDown = ExpertRowBlockSharded[
        C.NUM_EXPERTS, C.HIDDEN, C.MOE_INTERMEDIATE,
    ]


struct Gemma4StateShapes[max_seq_len: Int]:
    comptime SLIDING_CACHE = 2 * C.SLIDING_WINDOW
    comptime SlidingKV      = TensorColumnSharded[Self.SLIDING_CACHE, C.KV_DIM_SLIDING]
    comptime FullKV         = ContextRowSharded[Self.max_seq_len, C.KV_DIM_FULL]
    comptime SlidingKVScale = TensorColumnSharded[Self.SLIDING_CACHE, C.NUM_KV_HEADS_SLIDING]
    comptime FullKVScale    = ContextRowSharded[Self.max_seq_len, C.NUM_KV_HEADS_FULL]


struct Gemma4TailShapes:
    comptime FinalNorm = Replicated[C.HIDDEN, 1]
    comptime Embed = VocabularyRowSharded[C.VOCAB_SIZE, C.HIDDEN]


comptime SplitGainPerRowCs[fwht: Int, gamma: StaticString]: QuantRecipe = PerRowQuant(
    fwht, SplitGamma(gamma), SingleSided(), PerRowCs(), VnniPacked(),
)


comptime PlainPerRowBlockCs[fwht: Int]: QuantRecipe = PerRowQuant(
    fwht, NoGamma(), SingleSided(), PerBlockCs(), VnniPacked(),
)


comptime TiedHeadEmbed[fwht: Int]: QuantRecipe = PerBlockQuant(
    fwht, NoGamma(), SingleSided(), NoColsum(), RowMajor(),
)


struct SlidingAttnRefs(Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4Shapes
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


struct FullAttnRefs(Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4Shapes
    var q_proj: Slot[BF16, Self.S.FullQ, "self_attn.q_proj.weight",
        SplitGainPerRowCs[128, "input_layernorm.weight"]]
    var k_proj: Slot[BF16, Self.S.FullK, "self_attn.k_proj.weight",
        SplitGainPerRowCs[128, "input_layernorm.weight"]]
    var o_proj: Slot[BF16, Self.S.FullO, "self_attn.o_proj.weight",
        PlainPerRowBlockCs[C.HEAD_DIM_FULL]]
    var q_norm: Slot[BF16, Shape[C.HEAD_DIM_FULL, 1], "self_attn.q_norm.weight"]
    var k_norm: Slot[BF16, Shape[C.HEAD_DIM_FULL, 1], "self_attn.k_norm.weight"]


struct BodyRefs(Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4Shapes
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


struct SlidingLayerRefs(Copyable, ImplicitlyCopyable, SlotGroup):
    var attn: SlidingAttnRefs
    var body: BodyRefs


struct FullLayerRefs(Copyable, ImplicitlyCopyable, SlotGroup):
    var attn: FullAttnRefs
    var body: BodyRefs


struct SlidingKVSlots[max_seq_len: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4StateShapes[Self.max_seq_len]
    var k:       Slot[I8,  Self.S.SlidingKV]
    var k_scale: Slot[F32, Self.S.SlidingKVScale]
    var v:       Slot[I8,  Self.S.SlidingKV]
    var v_scale: Slot[F32, Self.S.SlidingKVScale]


struct FullKVSlots[max_seq_len: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4StateShapes[Self.max_seq_len]
    var k:       Slot[I8,  Self.S.FullKV]
    var k_scale: Slot[F32, Self.S.FullKVScale]
    var v:       Slot[I8,  Self.S.FullKV]
    var v_scale: Slot[F32, Self.S.FullKVScale]


struct RopeSlots[half: Int, max_seq_len: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    var cos: Slot[F32, Replicated[Self.max_seq_len, Self.half]]
    var sin: Slot[F32, Replicated[Self.max_seq_len, Self.half]]


struct ActivationSlots(Copyable, ImplicitlyCopyable, SlotGroup):
    var x_main:     Slot[BF16, Shape[C.SLIDING_WINDOW, C.HIDDEN]]
    var x_residual: Slot[BF16, Shape[C.SLIDING_WINDOW, C.HIDDEN]]


struct TailRefs(Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4TailShapes
    var final_norm: Slot[BF16, Self.S.FinalNorm, "model.language_model.norm.weight"]
    var embed:      Slot[BF16, Self.S.Embed, "model.language_model.embed_tokens.weight",
        TiedHeadEmbed[128]]


@fieldwise_init
struct Gemma4Layout[max_seq_len: Int](Copyable, ImplicitlyCopyable):
    var arena: ArenaLayout
    var sliding: Repeated[SlidingLayerRefs]
    var full: Repeated[FullLayerRefs]

    var sliding_kv: Repeated[SlidingKVSlots[Self.max_seq_len]]
    var full_kv: Repeated[FullKVSlots[Self.max_seq_len]]
    var activations: ActivationSlots
    var sliding_rope: RopeSlots[C.ROPE_HALF_SLIDING, Self.max_seq_len]
    var full_rope: RopeSlots[C.ROPE_HALF_FULL, Self.max_seq_len]

    var tail: Repeated[TailRefs]

    @always_inline
    def bind(self, base: Int) -> Self:
        var t = self
        t.arena = t.arena.bind(base)
        return t


comptime SLIDING_NUM_Q_MAX = C.Q_DIM_SLIDING // C.HEAD_DIM_SLIDING
comptime SLIDING_PARTIAL_STRIDE_MAX = flash_partial_stride(
    SLIDING_NUM_Q_MAX, C.HEAD_DIM_SLIDING)
comptime FULL_NUM_Q = C.Q_DIM_FULL // C.HEAD_DIM_FULL
comptime FULL_PARTIAL_STRIDE = flash_partial_stride(FULL_NUM_Q, C.HEAD_DIM_FULL)
comptime SLIDING_NB_DOWN = C.INTERMEDIATE // C.DOWN_FWHT_BLOCK
comptime MOE_NB_DOWN = C.MOE_INTERMEDIATE // C.DOWN_FWHT_BLOCK
comptime HEAD_NB = C.HIDDEN // 128


@fieldwise_init
struct Gemma4SlidingScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
    comptime PHASES = ScratchPhaseOrder[
        "norm_quant", "qkv", "attn_prep", "attention", "o_prep",
    ]

    var x_i8_band: ScratchPhase["norm_quant", "qkv"]
    var x_i8: ScratchBuffer[Int8, C.SLIDING_WINDOW * C.HIDDEN]
    var x_sa: ScratchBuffer[Float32, C.SLIDING_WINDOW]
    var x_row_workspace_band: ScratchPhase["norm_quant", "norm_quant"]
    var x_row_workspace: ScratchBuffer[Float32, C.HIDDEN, ScaleClass.PER_WORKER]

    var q_band: ScratchPhase["qkv", "o_prep"]
    var q: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.Q_DIM_SLIDING, ScaleClass.PER_DEGREE,
    ]

    var kv_band: ScratchPhase["qkv", "attn_prep"]
    var kv: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.KV_DIM_SLIDING * 2, ScaleClass.PER_DEGREE,
    ]

    var qprep_band: ScratchPhase["attn_prep", "attention"]
    var q_i8: ScratchBuffer[
        Int8, C.SLIDING_WINDOW * C.Q_DIM_SLIDING, ScaleClass.PER_DEGREE,
    ]
    var qi_bias: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * SLIDING_NUM_Q_MAX, ScaleClass.PER_DEGREE,
    ]
    var f_q: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * SLIDING_NUM_Q_MAX, ScaleClass.PER_DEGREE,
    ]

    var partials_band: ScratchPhase["attention", "attention"]
    var partials: ScratchBuffer[
        Float32, SLIDING_PARTIAL_STRIDE_MAX, ScaleClass.PER_WORKER,
    ]

    var o_band: ScratchPhase["o_prep", "o_prep"]
    var o_i8: ScratchBuffer[
        Int8, C.SLIDING_WINDOW * C.Q_DIM_SLIDING, ScaleClass.PER_DEGREE,
    ]
    var o_sa: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * SLIDING_NUM_Q_MAX, ScaleClass.PER_DEGREE,
    ]
    var o_row_workspace: ScratchBuffer[
        Float32, C.Q_DIM_SLIDING, ScaleClass.PER_WORKER,
    ]


@fieldwise_init
struct Gemma4FullScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
    comptime PHASES = ScratchPhaseOrder[
        "norm_quant", "qkv", "attn_prep", "flash", "merge", "o_prep",
    ]

    var x_i8_band: ScratchPhase["norm_quant", "qkv"]
    var x_i8: ScratchBuffer[Int8, C.SLIDING_WINDOW * C.HIDDEN]
    var x_sa: ScratchBuffer[Float32, C.SLIDING_WINDOW]
    var x_row_workspace_band: ScratchPhase["norm_quant", "norm_quant"]
    var x_row_workspace: ScratchBuffer[Float32, C.HIDDEN, ScaleClass.PER_WORKER]

    var q_band: ScratchPhase["qkv", "attn_prep"]
    var q: ScratchBuffer[BFloat16, C.SLIDING_WINDOW * C.Q_DIM_FULL]

    var kv_band: ScratchPhase["qkv", "attn_prep"]
    var kv: ScratchBuffer[BFloat16, C.SLIDING_WINDOW * C.KV_DIM_FULL * 2]

    var qprep_band: ScratchPhase["attn_prep", "flash"]
    var q_i8: ScratchBuffer[Int8, C.SLIDING_WINDOW * C.Q_DIM_FULL]
    var qi_bias: ScratchBuffer[Float32, C.SLIDING_WINDOW * FULL_NUM_Q]
    var f_q: ScratchBuffer[Float32, C.SLIDING_WINDOW * FULL_NUM_Q]

    var partials_band: ScratchPhase["flash", "merge"]
    var partials: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * FULL_PARTIAL_STRIDE,
    ]

    var merge_segments_band: ScratchPhase["merge", "merge"]
    var merge_segments: ScratchBuffer[
        MergeSegment, 1, ScaleClass.PER_WORKER_PER_DEGREE,
    ]

    var q_local_band: ScratchPhase["merge", "o_prep"]
    var q_local: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.Q_DIM_FULL, ScaleClass.PER_DEGREE,
    ]

    var o_band: ScratchPhase["o_prep", "o_prep"]
    var o_i8: ScratchBuffer[
        Int8, C.SLIDING_WINDOW * C.Q_DIM_FULL, ScaleClass.PER_DEGREE,
    ]
    var o_sa: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * FULL_NUM_Q, ScaleClass.PER_DEGREE,
    ]
    var o_row_workspace: ScratchBuffer[
        Float32, C.Q_DIM_FULL, ScaleClass.PER_WORKER,
    ]


@fieldwise_init
struct Gemma4FfnMoeScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
    comptime PHASES = ScratchPhaseOrder[
        "dense_norm", "dense_gate_up", "dense_down_quant",
        "router_select", "moe_norm", "build_schedules", "phase1",
        "bucket_quant", "phase2", "dense_down_post",
    ]

    var dense_x_band: ScratchPhase["dense_norm", "dense_gate_up"]
    var dense_x_i8: ScratchBuffer[Int8, C.SLIDING_WINDOW * C.HIDDEN]
    var dense_x_sa: ScratchBuffer[Float32, C.SLIDING_WINDOW]
    var dense_x_row_workspace_band: ScratchPhase["dense_norm", "dense_norm"]
    var dense_x_row_workspace: ScratchBuffer[
        Float32, C.HIDDEN, ScaleClass.PER_WORKER,
    ]

    var ffn_gate_band: ScratchPhase["dense_gate_up", "dense_down_quant"]
    var ffn_gate: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.INTERMEDIATE, ScaleClass.PER_DEGREE,
    ]

    var ffn_up_band: ScratchPhase["dense_gate_up", "dense_gate_up"]
    var ffn_up: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.INTERMEDIATE, ScaleClass.PER_DEGREE,
    ]

    var dense_gate_band: ScratchPhase["dense_down_quant", "dense_down_post"]
    var dense_gate_i8: ScratchBuffer[
        Int8, C.SLIDING_WINDOW * C.INTERMEDIATE, ScaleClass.PER_DEGREE,
    ]
    var dense_gate_sa: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * SLIDING_NB_DOWN, ScaleClass.PER_DEGREE,
    ]
    var dense_gate_row_workspace_band: ScratchPhase[
        "dense_down_quant", "dense_down_quant",
    ]
    var dense_gate_row_workspace: ScratchBuffer[
        Float32, C.INTERMEDIATE, ScaleClass.PER_WORKER,
    ]

    var router_workspace: ScratchPhase["router_select", "router_select"]
    var moe_router_scaled: ScratchBuffer[Float32, C.HIDDEN, ScaleClass.PER_WORKER]

    var router_cands: ScratchPhase["router_select", "router_select"]
    var moe_cands: ScratchBuffer[
        RouterCandidate, C.SLIDING_WINDOW * C.TOP_K, ScaleClass.PER_WORKER,
    ]

    var router_products: ScratchPhase["router_select", "build_schedules"]
    var moe_route_idx: ScratchBuffer[Int32, C.SLIDING_WINDOW * C.TOP_K]
    var moe_route_w: ScratchBuffer[Float32, C.SLIDING_WINDOW * C.TOP_K]

    var expert_input: ScratchPhase["moe_norm", "phase1"]
    var moe_x_i8: ScratchBuffer[Int8, C.SLIDING_WINDOW * C.HIDDEN]
    var moe_x_sa: ScratchBuffer[Float32, C.SLIDING_WINDOW]
    var moe_x_row_workspace_band: ScratchPhase["moe_norm", "moe_norm"]
    var moe_x_row_workspace: ScratchBuffer[
        Float32, C.HIDDEN, ScaleClass.PER_WORKER,
    ]

    var schedule_products: ScratchPhase["build_schedules", "phase2"]
    var moe_expert_offset: ScratchBuffer[Int32, C.NUM_EXPERTS + 1]
    var moe_routes: ScratchBuffer[SparseRoute, C.SLIDING_WINDOW * C.TOP_K]

    var hidden_bucket: ScratchPhase["phase1", "bucket_quant"]
    var moe_hidden_bucket: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.TOP_K * C.MOE_INTERMEDIATE,
    ]

    var bucket_i8_band: ScratchPhase["bucket_quant", "phase2"]
    var moe_bucket_i8: ScratchBuffer[
        Int8, C.SLIDING_WINDOW * C.TOP_K * C.MOE_INTERMEDIATE,
    ]
    var moe_bucket_sa: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * C.TOP_K * MOE_NB_DOWN,
    ]
    var bucket_row_workspace_band: ScratchPhase["bucket_quant", "bucket_quant"]
    var bucket_row_workspace: ScratchBuffer[
        Float32, C.MOE_INTERMEDIATE, ScaleClass.PER_WORKER,
    ]

    var phase2_accum: ScratchPhase["phase2", "phase2"]
    var moe_accum: ScratchBuffer[Float32, C.SLIDING_WINDOW * C.HIDDEN]

    var dense_band: ScratchPhase["dense_down_post", "dense_down_post"]
    var ffn_dense_out: ScratchBuffer[BFloat16, C.SLIDING_WINDOW * C.HIDDEN]


@fieldwise_init
struct Gemma4HeadScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
    comptime PHASES = ScratchPhaseOrder["embed", "logits"]

    var embed_row_workspace_band: ScratchPhase["embed", "embed"]
    var embed_row_workspace: ScratchBuffer[
        Float32, C.HIDDEN, ScaleClass.PER_WORKER,
    ]

    var head_prep_band: ScratchPhase["logits", "logits"]
    var head_x_i8: ScratchBuffer[Int8, C.HIDDEN]
    var head_x_sa: ScratchBuffer[Float32, HEAD_NB]
    var head_row_workspace: ScratchBuffer[Float32, C.HIDDEN]

    var logits_band: ScratchPhase["logits", "logits"]
    var logits: ScratchBuffer[BFloat16, C.VOCAB_SIZE, ScaleClass.PER_DEGREE]


@fieldwise_init
struct Gemma4ForwardScratch(Copyable, ImplicitlyCopyable):
    var sliding: Gemma4SlidingScratch
    var full: Gemma4FullScratch
    var ffn: Gemma4FfnMoeScratch
    var head: Gemma4HeadScratch


def calculate_peak_scratch(degree: Int, max_workers: Int) -> Int:
    return aggregate_scratch_peak[Gemma4ForwardScratch](degree, max_workers)


def build_gemma4_plan[
    max_seq_len: Int,
](degree: Int, max_workers: Int, mut descs: List[WeightDesc]) -> Gemma4Layout[max_seq_len]:
    if not degree_contracts_ok(degree):
        abort(t"gemma4: degree {degree} does not divide the model dimensions")
    if not (
        vnni_pack_contract_ok[SlidingLayerRefs](degree)
        and vnni_pack_contract_ok[FullLayerRefs](degree)
        and vnni_pack_contract_ok[TailRefs](degree)
    ):
        abort(
            t"gemma4: degree {degree} breaks VNNI pack alignment; every packed "
            t"weight's per-rank rows must be a multiple of {VNNI_N_STEP} and "
            t"cols a multiple of {VNNI_K_STEP}")
    if max_workers <= 0:
        abort(t"gemma4: max_workers must be positive, got {max_workers}")
    if max_workers > C.SLIDING_WINDOW:
        abort(
            t"gemma4: full-attention partials require max_workers <= "
            t"SLIDING_WINDOW ({C.SLIDING_WINDOW}), got {max_workers}")

    var sl_proto = SlidingLayerRefs()
    var sl_stride = stamp_offsets(sl_proto, degree)
    var fl_proto = FullLayerRefs()
    var fl_stride = stamp_offsets(fl_proto, degree)

    var sl_off = 0
    var fl_off = sl_off + C.NUM_SLIDING_LAYERS * sl_stride
    var distributed = fl_off + C.NUM_FULL_LAYERS * fl_stride

    for i in range(C.NUM_LAYERS):
        var entry = LAYER_SCHEDULE[i]
        var prefix = String(t"model.language_model.layers.{entry.idx}.")
        if entry.kind == LayerKind.FULL:
            _ = emit_descs[FullLayerRefs](
                prefix, fl_off + entry.local_idx * fl_stride, degree, descs)
        else:
            _ = emit_descs[SlidingLayerRefs](
                prefix, sl_off + entry.local_idx * sl_stride, degree, descs)

    var tail_proto = TailRefs()
    var tail_bytes = stamp_offsets(tail_proto, degree)
    _ = emit_descs[TailRefs]("", distributed, degree, descs)
    var tail = Repeated[TailRefs](tail_proto, distributed, tail_bytes, 1)
    distributed += tail_bytes

    var state_cursor = distributed

    var skv_proto = SlidingKVSlots[max_seq_len]()
    var skv_stride = stamp_offsets(skv_proto, degree)
    var sliding_kv = Repeated[SlidingKVSlots[max_seq_len]](
        skv_proto, state_cursor, skv_stride, C.NUM_SLIDING_LAYERS)
    state_cursor = align_up(state_cursor + C.NUM_SLIDING_LAYERS * skv_stride)

    var fkv_proto = FullKVSlots[max_seq_len]()
    var fkv_stride = stamp_offsets(fkv_proto, degree)
    var full_kv = Repeated[FullKVSlots[max_seq_len]](
        fkv_proto, state_cursor, fkv_stride, C.NUM_FULL_LAYERS)
    state_cursor = align_up(state_cursor + C.NUM_FULL_LAYERS * fkv_stride)

    var activations = ActivationSlots()
    state_cursor = stamp_offsets(activations, degree, state_cursor)

    var scratch_cap = calculate_peak_scratch(degree, max_workers)
    state_cursor = align_up(state_cursor)
    var scratch_off = state_cursor
    state_cursor = align_up(state_cursor + scratch_cap)

    var sliding_rope = RopeSlots[C.ROPE_HALF_SLIDING, max_seq_len]()
    state_cursor = stamp_offsets(sliding_rope, degree, state_cursor)
    var full_rope = RopeSlots[C.ROPE_HALF_FULL, max_seq_len]()
    state_cursor = stamp_offsets(full_rope, degree, state_cursor)

    var arena = ArenaLayout(
        base=0,
        distributed_bytes=distributed,
        state_bytes=state_cursor - distributed,
        host_bytes=align_up(state_cursor),
        scratch_off=scratch_off,
    )
    return Gemma4Layout[max_seq_len](
        arena=arena,
        sliding=Repeated[SlidingLayerRefs](sl_proto, sl_off, sl_stride, C.NUM_SLIDING_LAYERS),
        full=Repeated[FullLayerRefs](fl_proto, fl_off, fl_stride, C.NUM_FULL_LAYERS),
        sliding_kv=sliding_kv, full_kv=full_kv,
        activations=activations,
        sliding_rope=sliding_rope, full_rope=full_rope,
        tail=tail)


def dispatch_bq_sliding_attention_qkv[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_seq_len: Int, max_worker_count: Int = 128,
](
    layout: Gemma4Layout[max_seq_len],
    ctx: BindContext[o],
    act: ButterquantActivation[o],
    base_pos: Int,
    seq_len: Int,
    layer_idx: Int,
    scratch: TemporalScratchPool,
    plan: ScratchPlan,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    var degree = ctx.degree()
    comptime head_dim = C.HEAD_DIM_SLIDING
    comptime gqa_ratio = C.NUM_HEADS // C.NUM_KV_HEADS_SLIDING
    comptime sqrt_hd = sqrt[DType.float32, 1](head_dim)
    comptime hd_eps = Float32(head_dim) * Float32(C.RMS_NORM_EPS)
    comptime rope_half = C.ROPE_HALF_SLIDING
    comptime cache_size = Gemma4StateShapes[max_seq_len].SLIDING_CACHE
    comptime max_q = SLIDING_NUM_Q_MAX
    var q_rows = Gemma4Shapes.SlidingQ.data_n(degree)
    var kv_rows = Gemma4Shapes.SlidingKV.data_n(degree)
    var num_q_heads = q_rows // head_dim
    var num_kv_heads = kv_rows // head_dim
    var partial_stride = flash_partial_stride(num_q_heads, head_dim)

    debug_assert(
        seq_len <= C.SLIDING_WINDOW,
        "sliding attention chunk exceeds SLIDING_WINDOW",
    )

    var attn_ctx = ctx.with_layer(
        layout.sliding.base(ctx.view.bases[0], layer_idx))
    var attn = layout.sliding.proto.attn

    var q_outs = scratch.binding[Gemma4SlidingScratch, "q"](ctx, plan)
    var k_outs = scratch.binding[Gemma4SlidingScratch, "kv"](ctx, plan)
    var v_outs = k_outs.shifted(seq_len * kv_rows)

    dispatch_bq_qkv[
        hidden=C.HIDDEN, qn_full=C.Q_DIM_SLIDING, kvn_full=C.KV_DIM_SLIDING,
        max_worker_count=max_worker_count,
    ](
        act,
        attn.q_proj.bq_weight(attn_ctx),
        attn.k_proj.bq_weight(attn_ctx),
        attn.v_proj.bq_weight(attn_ctx),
        q_outs, k_outs, v_outs, q_rows, kv_rows, seq_len, pools, prof)

    var q_i8 = scratch.binding[Gemma4SlidingScratch, "q_i8"](ctx, plan)
    var qi_bias = scratch.binding[Gemma4SlidingScratch, "qi_bias"](ctx, plan)
    var f_q = scratch.binding[Gemma4SlidingScratch, "f_q"](ctx, plan)

    var kv_lb = layout.sliding_kv.base(ctx.view.bases[0], layer_idx)
    var k_cache = layout.sliding_kv.proto.k.binding(kv_lb, ctx.view)
    var k_scale = layout.sliding_kv.proto.k_scale.binding(kv_lb, ctx.view)
    var v_cache = layout.sliding_kv.proto.v.binding(kv_lb, ctx.view)
    var v_scale = layout.sliding_kv.proto.v_scale.binding(kv_lb, ctx.view)

    dispatch_bq_attn_prep[
        head_dim=head_dim, rope_half=rope_half, pair_stride=head_dim // 2,
        slot_mask=cache_size - 1, sqrt_n=sqrt_hd, n_eps=hd_eps,
        max_worker_count=max_worker_count,
    ](q_outs, k_outs, v_outs,
      attn.q_norm.binding(attn_ctx), attn.k_norm.binding(attn_ctx),
      q_i8, qi_bias, f_q, k_cache, k_scale, v_cache, v_scale,
      layout.sliding_rope.cos.state_binding(ctx),
      layout.sliding_rope.sin.state_binding(ctx),
      num_q_heads, num_kv_heads, 1, base_pos, seq_len, pools, prof)

    var partials = scratch.binding[Gemma4SlidingScratch, "partials"](ctx, plan)

    dispatch_bq_sliding_attention[
        head_dim=head_dim, max_q=max_q, gqa_ratio=gqa_ratio,
        window=C.SLIDING_WINDOW, cache_size=cache_size,
        max_worker_count=max_worker_count,
    ](q_i8, qi_bias, f_q, k_cache, k_scale, v_cache, v_scale,
      q_outs, partials, num_q_heads, num_kv_heads, partial_stride, kv_rows,
      base_pos, seq_len, pools, prof)

    var o_i8 = scratch.binding[Gemma4SlidingScratch, "o_i8"](ctx, plan)
    var o_sa = scratch.binding[Gemma4SlidingScratch, "o_sa"](ctx, plan)
    var o_row_workspace = scratch.binding[
        Gemma4SlidingScratch, "o_row_workspace"](ctx, plan)

    dispatch_bq_block_quant[
        block=head_dim, apply_fwht=False, max_worker_count=max_worker_count,
    ](q_outs, o_i8, o_sa, o_row_workspace, q_rows, seq_len, pools, prof)

    var o_act = ButterquantBlockActivation(o_i8, o_sa)
    var xs = layout.activations.x_residual.state_binding(ctx)

    dispatch_bq_block_linear[
        n_rows=C.HIDDEN, max_worker_count=max_worker_count,
    ](o_act, attn.o_proj.bq_weight(attn_ctx), xs, q_rows, seq_len, pools, prof)


def dispatch_bq_full_attention_qkv[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_seq_len: Int, max_worker_count: Int = 128,
](
    layout: Gemma4Layout[max_seq_len],
    ctx: BindContext[o],
    act: ButterquantActivation[o],
    base_pos: Int,
    seq_len: Int,
    layer_idx: Int,
    scratch: TemporalScratchPool,
    plan: ScratchPlan,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    var degree = ctx.degree()
    comptime head_dim = C.HEAD_DIM_FULL
    comptime q_rows = C.Q_DIM_FULL
    comptime k_rows = C.KV_DIM_FULL
    comptime num_q_heads = q_rows // head_dim
    comptime num_kv_heads = k_rows // head_dim
    comptime gqa_ratio = C.NUM_HEADS // C.NUM_KV_HEADS_FULL
    comptime sqrt_hd = sqrt[DType.float32, 1](head_dim)
    comptime hd_eps = Float32(head_dim) * Float32(C.RMS_NORM_EPS)
    comptime rope_half = C.ROPE_HALF_FULL
    comptime pair_stride = head_dim // 2
    comptime partial_stride = FULL_PARTIAL_STRIDE
    var local_q_rows = Gemma4Shapes.FullO.data_m(degree)
    var local_num_q_heads = local_q_rows // head_dim

    var attn_ctx = ctx.with_layer(
        layout.full.base(ctx.view.bases[0], layer_idx))
    var attn = layout.full.proto.attn

    var q_outs = scratch.binding[Gemma4FullScratch, "q"](ctx, plan)
    var k_outs = scratch.binding[Gemma4FullScratch, "kv"](ctx, plan)

    dispatch_bq_linear[hidden=C.HIDDEN, max_worker_count=max_worker_count](
        act, attn.q_proj.bq_weight(attn_ctx), q_outs, q_rows, seq_len, pools, prof)
    dispatch_bq_linear[hidden=C.HIDDEN, max_worker_count=max_worker_count](
        act, attn.k_proj.bq_weight(attn_ctx), k_outs, k_rows, seq_len, pools, prof)

    var q_i8 = scratch.binding[Gemma4FullScratch, "q_i8"](ctx, plan)
    var qi_bias = scratch.binding[Gemma4FullScratch, "qi_bias"](ctx, plan)
    var f_q = scratch.binding[Gemma4FullScratch, "f_q"](ctx, plan)

    var kv_lb = layout.full_kv.base(ctx.view.bases[0], layer_idx)
    var k_cache = layout.full_kv.proto.k.binding(kv_lb, ctx.view)
    var k_scale = layout.full_kv.proto.k_scale.binding(kv_lb, ctx.view)
    var v_cache = layout.full_kv.proto.v.binding(kv_lb, ctx.view)
    var v_scale = layout.full_kv.proto.v_scale.binding(kv_lb, ctx.view)

    dispatch_bq_attn_prep[
        head_dim=head_dim, rope_half=rope_half, pair_stride=pair_stride,
        slot_mask=-1, sqrt_n=sqrt_hd, n_eps=hd_eps,
        max_worker_count=max_worker_count,
    ](q_outs, k_outs, k_outs,
      attn.q_norm.binding(attn_ctx), attn.k_norm.binding(attn_ctx),
      q_i8, qi_bias, f_q, k_cache, k_scale, v_cache, v_scale,
      layout.full_rope.cos.state_binding(ctx),
      layout.full_rope.sin.state_binding(ctx),
      num_q_heads, num_kv_heads, degree, base_pos, seq_len, pools, prof)

    var q_local = scratch.binding[Gemma4FullScratch, "q_local"](ctx, plan)
    var partials = scratch.binding[Gemma4FullScratch, "partials"](ctx, plan)
    var merge_segments = scratch.binding[
        Gemma4FullScratch, "merge_segments"](ctx, plan)

    dispatch_bq_full_attention[
        head_dim=head_dim, num_q=num_q_heads, num_kv=num_kv_heads,
        gqa_ratio=gqa_ratio, kv_stride=k_rows, partial_stride=partial_stride,
        max_worker_count=max_worker_count,
    ](q_i8, qi_bias, f_q, k_cache, k_scale, v_cache, v_scale,
      q_local, partials, merge_segments, local_num_q_heads,
      base_pos, seq_len, pools, prof)

    var o_i8 = scratch.binding[Gemma4FullScratch, "o_i8"](ctx, plan)
    var o_sa = scratch.binding[Gemma4FullScratch, "o_sa"](ctx, plan)
    var o_row_workspace = scratch.binding[
        Gemma4FullScratch, "o_row_workspace"](ctx, plan)

    dispatch_bq_block_quant[
        block=head_dim, apply_fwht=False, max_worker_count=max_worker_count,
    ](q_local, o_i8, o_sa, o_row_workspace, local_q_rows, seq_len, pools, prof)

    var o_act = ButterquantBlockActivation(o_i8, o_sa)
    var xs = layout.activations.x_residual.state_binding(ctx)

    dispatch_bq_block_linear[
        n_rows=C.HIDDEN, max_worker_count=max_worker_count,
    ](o_act, attn.o_proj.bq_weight(attn_ctx), xs, local_q_rows, seq_len, pools, prof)


def dispatch_bq_moe[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_worker_count: Int = 128,
](
    body: BodyRefs,
    ctx: BindContext[o],
    x_input: Binding[BFloat16, o],
    moe_out: Binding[BFloat16, o],
    seq_len: Int,
    scratch: TemporalScratchPool,
    plan: ScratchPlan,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    var degree = ctx.degree()
    var experts_per_rank = C.NUM_EXPERTS // degree
    comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
    comptime n_eps = Float32(C.HIDDEN) * Float32(C.RMS_NORM_EPS)

    var router_scaled = scratch.binding[Gemma4FfnMoeScratch, "moe_router_scaled"](ctx, plan)
    var cands = scratch.binding[Gemma4FfnMoeScratch, "moe_cands"](ctx, plan)
    var route_idx = scratch.binding[Gemma4FfnMoeScratch, "moe_route_idx"](ctx, plan)
    var route_w = scratch.binding[Gemma4FfnMoeScratch, "moe_route_w"](ctx, plan)
    var expert_offset = scratch.binding[Gemma4FfnMoeScratch, "moe_expert_offset"](ctx, plan)
    var routes = scratch.binding[Gemma4FfnMoeScratch, "moe_routes"](ctx, plan)
    var moe_x_i8 = scratch.binding[Gemma4FfnMoeScratch, "moe_x_i8"](ctx, plan)
    var moe_x_sa = scratch.binding[Gemma4FfnMoeScratch, "moe_x_sa"](ctx, plan)
    var moe_x_row_workspace = scratch.binding[
        Gemma4FfnMoeScratch, "moe_x_row_workspace"](ctx, plan)
    var bucket = scratch.binding[Gemma4FfnMoeScratch, "moe_hidden_bucket"](ctx, plan)
    var bucket_i8 = scratch.binding[Gemma4FfnMoeScratch, "moe_bucket_i8"](ctx, plan)
    var bucket_sa = scratch.binding[Gemma4FfnMoeScratch, "moe_bucket_sa"](ctx, plan)
    var bucket_row_workspace = scratch.binding[
        Gemma4FfnMoeScratch, "bucket_row_workspace"](ctx, plan)
    var moe_accum = scratch.binding[Gemma4FfnMoeScratch, "moe_accum"](ctx, plan)

    var nws = dispatch_router_expert[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
        top_k=C.TOP_K, max_worker_count=max_worker_count,
    ](x_input, body.router_proj.bq_router(ctx).centered,
      body.router_scale.binding(ctx), router_scaled, cands,
      experts_per_rank, seq_len, pools, prof)

    dispatch_merge_router_candidates[
        C.TOP_K, max_worker_count=max_worker_count,
    ](cands, nws, body.router_pes.binding(ctx), route_idx, route_w,
      seq_len, pools, prof)

    dispatch_bq_norm_quant[
        hidden=C.HIDDEN, block=128, sqrt_n=sqrt_n, n_eps=n_eps,
        max_worker_count=max_worker_count,
    ](x_input, body.pre_ffn_norm_2.binding(ctx),
      moe_x_i8, moe_x_sa, moe_x_row_workspace, seq_len, pools, prof)

    dispatch_build_expert_schedules[
        C.NUM_EXPERTS, C.TOP_K, max_worker_count=max_worker_count,
    ](route_idx, route_w, expert_offset, routes,
      experts_per_rank, seq_len, pools, prof)

    var moe_act = ButterquantActivation(moe_x_i8, moe_x_sa)
    dispatch_bq_phase1_gate_up[
        hidden=C.HIDDEN, gate_up=C.MOE_GATE_UP_FUSED,
        inter=C.MOE_INTERMEDIATE, max_worker_count=max_worker_count,
    ](moe_act, expert_offset, routes,
      body.experts_gate_up.bq_weight(ctx), bucket, experts_per_rank, pools, prof)

    var num_routes = seq_len * C.TOP_K
    dispatch_bq_block_quant[
        block=C.DOWN_FWHT_BLOCK, apply_fwht=True,
        max_worker_count=max_worker_count,
    ](bucket, bucket_i8, bucket_sa, bucket_row_workspace,
      C.MOE_INTERMEDIATE, num_routes, pools, prof)

    var bucket_act = ButterquantBlockActivation(bucket_i8, bucket_sa)
    dispatch_bq_phase2_down[
        hidden=C.HIDDEN, inter=C.MOE_INTERMEDIATE,
        max_worker_count=max_worker_count,
    ](bucket_act, expert_offset, routes,
      body.experts_down.bq_weight(ctx), moe_accum, moe_out,
      experts_per_rank, seq_len, pools, prof)

    dispatch_allreduce_inplace[
        BF16, max_worker_count=max_worker_count,
    ](moe_out, seq_len * C.HIDDEN, pools, prof)


def dispatch_bq_ffn[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_worker_count: Int = 128,
](
    body: BodyRefs,
    ctx: BindContext[o],
    x_main: Binding[BFloat16, o],
    x_residual: Binding[BFloat16, o],
    seq_len: Int,
    scratch: TemporalScratchPool,
    plan: ScratchPlan,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    var degree = ctx.degree()
    comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
    comptime n_eps = Float32(C.HIDDEN) * Float32(C.RMS_NORM_EPS)
    var intermediate_per_rank = Gemma4Shapes.GateUp.data_n(degree)

    var layer_scalar_ptr = body.layer_scalar.at(ctx.layer_base)

    var dense_x_i8 = scratch.binding[Gemma4FfnMoeScratch, "dense_x_i8"](ctx, plan)
    var dense_x_sa = scratch.binding[Gemma4FfnMoeScratch, "dense_x_sa"](ctx, plan)
    var dense_x_row_workspace = scratch.binding[
        Gemma4FfnMoeScratch, "dense_x_row_workspace"](ctx, plan)
    var gate = scratch.binding[Gemma4FfnMoeScratch, "ffn_gate"](ctx, plan)
    var up = scratch.binding[Gemma4FfnMoeScratch, "ffn_up"](ctx, plan)
    var dense_gate_i8 = scratch.binding[Gemma4FfnMoeScratch, "dense_gate_i8"](ctx, plan)
    var dense_gate_sa = scratch.binding[Gemma4FfnMoeScratch, "dense_gate_sa"](ctx, plan)
    var dense_gate_row_workspace = scratch.binding[
        Gemma4FfnMoeScratch, "dense_gate_row_workspace"](ctx, plan)
    var dense_out = scratch.binding[Gemma4FfnMoeScratch, "ffn_dense_out"](ctx, plan)

    dispatch_bq_norm_quant[
        hidden=C.HIDDEN, block=128, sqrt_n=sqrt_n, n_eps=n_eps,
        max_worker_count=max_worker_count,
    ](x_main, body.pre_ffn_norm.binding(ctx),
      dense_x_i8, dense_x_sa, dense_x_row_workspace, seq_len, pools, prof)

    var dense_act = ButterquantActivation(dense_x_i8, dense_x_sa)
    dispatch_bq_linear[hidden=C.HIDDEN, max_worker_count=max_worker_count](
        dense_act, body.gate_proj.bq_weight(ctx), gate,
        intermediate_per_rank, seq_len, pools, prof)
    dispatch_bq_linear[hidden=C.HIDDEN, max_worker_count=max_worker_count](
        dense_act, body.up_proj.bq_weight(ctx), up,
        intermediate_per_rank, seq_len, pools, prof)

    dispatch_gelu_gate_up[max_worker_count=max_worker_count](
        gate, up, gate, intermediate_per_rank, seq_len, pools, prof)

    dispatch_bq_block_quant[
        block=C.DOWN_FWHT_BLOCK, apply_fwht=True,
        max_worker_count=max_worker_count,
    ](gate, dense_gate_i8, dense_gate_sa, dense_gate_row_workspace,
      intermediate_per_rank, seq_len, pools, prof)

    dispatch_bq_moe[max_worker_count=max_worker_count](
        body, ctx, x_main, x_residual, seq_len, scratch, plan, pools, prof)

    var dense_gate_act = ButterquantBlockActivation(dense_gate_i8, dense_gate_sa)
    dispatch_bq_block_linear[
        n_rows=C.HIDDEN, max_worker_count=max_worker_count,
    ](dense_gate_act, body.down_proj.bq_weight(ctx), dense_out,
      intermediate_per_rank, seq_len, pools, prof)

    dispatch_allreduce_inplace[
        BF16, max_worker_count=max_worker_count,
    ](dense_out, seq_len * C.HIDDEN, pools, prof)

    dispatch_rms_norm[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
        max_worker_count=max_worker_count,
    ](dense_out, dense_out, body.post_ffn_norm_1.binding(ctx), seq_len, pools, prof)

    fused_norm_residual_add[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
        max_worker_count=max_worker_count,
    ](x_residual, dense_out, dense_out,
      body.post_ffn_norm_2.binding(ctx), seq_len, pools, prof)

    fused_norm_residual_add[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
        max_worker_count=max_worker_count,
    ](dense_out, x_main, x_main,
      body.post_ffn_norm.binding(ctx), seq_len, pools, prof)

    var ls_value = layer_scalar_ptr[0].cast[DType.float32]()
    dispatch_scalar_mul[
        hidden=C.HIDDEN, max_worker_count=max_worker_count,
    ](x_main, x_main, ls_value, seq_len, pools, prof)


struct Gemma4[
    max_seq_len: Int = 8192,
    Pool: BurstThreadPool = BurstPool[],
    profile: Bool = False, profile_slots: Int = 64,
](Movable):
    var arenas: List[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var pools: List[Self.Pool]
    var layout: Gemma4Layout[Self.max_seq_len]
    var scratch: TemporalScratchPool
    var arena_bases: List[Int]
    var degree: Int
    var sliding_plan: ScratchPlan
    var full_plan: ScratchPlan
    var ffn_plan: ScratchPlan
    var head_plan: ScratchPlan
    var profiler: Profiler[Self.profile, Self.profile_slots]

    def __init__(out self,
        var arenas: List[NumaArena[alignment=DEFAULT_ALIGNMENT]],
        var pools: List[Self.Pool],
        layout: Gemma4Layout[Self.max_seq_len],
        degree: Int,
        max_workers: Int,
    ):
        self.degree = degree
        self.arena_bases = List[Int]()
        for r in range(degree):
            self.arena_bases.append(Int(arenas[r].base.value()))
        self.layout = layout.bind(self.arena_bases[0])
        self.arenas = arenas^
        self.pools = pools^
        self.scratch = TemporalScratchPool(self.layout.arena.scratch_base())
        self.sliding_plan = derive_scratch_plan[Gemma4SlidingScratch](degree, max_workers)
        self.full_plan = derive_scratch_plan[Gemma4FullScratch](degree, max_workers)
        self.ffn_plan = derive_scratch_plan[Gemma4FfnMoeScratch](degree, max_workers)
        self.head_plan = derive_scratch_plan[Gemma4HeadScratch](degree, max_workers)
        debug_assert(
            not co_live_buffers_overlap[Gemma4SlidingScratch](
                self.sliding_plan, degree, max_workers),
            "sliding scratch plan overlaps co-live buffers",
        )
        debug_assert(
            not co_live_buffers_overlap[Gemma4FullScratch](
                self.full_plan, degree, max_workers),
            "full scratch plan overlaps co-live buffers",
        )
        debug_assert(
            not co_live_buffers_overlap[Gemma4FfnMoeScratch](
                self.ffn_plan, degree, max_workers),
            "ffn scratch plan overlaps co-live buffers",
        )
        debug_assert(
            not co_live_buffers_overlap[Gemma4HeadScratch](
                self.head_plan, degree, max_workers),
            "head scratch plan overlaps co-live buffers",
        )
        self.profiler = Profiler[Self.profile, Self.profile_slots]()

    def init_state(mut self):
        var tasks = List[PackColsumTask]()
        for i in range(C.NUM_LAYERS):
            var entry = LAYER_SCHEDULE[i]
            if entry.kind == LayerKind.FULL:
                _ = emit_pack_tasks[FullLayerRefs](
                    self.layout.full.off
                    + entry.local_idx * self.layout.full.stride,
                    self.degree, tasks)
            else:
                _ = emit_pack_tasks[SlidingLayerRefs](
                    self.layout.sliding.off
                    + entry.local_idx * self.layout.sliding.stride,
                    self.degree, tasks)
        _ = emit_pack_tasks[TailRefs](self.layout.tail.off, self.degree, tasks)

        var nodes = List[Int]()
        for r in range(self.degree):
            nodes.append(self.arenas[r].node)
        var noprof = Profiler[False]()
        dispatch_pack_colsum[max_worker_count=MAX_WORKERS](
            self.pools, noprof, self.arena_bases, nodes, tasks)

    def model_init(mut self):
        ref layout = self.layout
        comptime width = simd_width_of[DType.float32]()
        comptime inv_sqrt_hidden = 1.0 / sqrt[DType.float32, 1](C.HIDDEN)

        prime_fp_environment(self.pools)
        prime_amx_environment(self.pools)

        @parameter
        def bake_router_scale(p: UnsafePointer[BFloat16, MutAnyOrigin]):
            for j in range(0, C.HIDDEN, width):
                var lane = p + j
                var v = lane.load[width=width]().cast[DType.float32]()
                lane.store((v * inv_sqrt_hidden).cast[DType.bfloat16]())

        for rank in range(self.degree):
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
        print("  router constants baked")

        from kernels.rope import init_rope_table, init_rope_table_partial_strided
        for rank in range(self.degree):
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
        print("  rope tables initialized")

    def forward[
        tok_origin: ImmutOrigin, //,
    ](
        mut self,
        token_ids: Span[Int32, tok_origin],
        base_pos: Int = 0,
    ) -> TemporalLogitsView[C.VOCAB_SIZE]:
        ref layout = self.layout
        var degree = self.degree
        comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
        comptime n_eps = Float32(C.HIDDEN) * Float32(C.RMS_NORM_EPS)
        comptime embed_scale = Float64(sqrt[DType.float32, 1](C.HIDDEN)
            .cast[DType.bfloat16]().cast[DType.float32]())
        var shard_rows = Gemma4TailShapes.Embed.data_n(degree)
        var vocab_per_rank = C.VOCAB_SIZE // degree

        var wall_t0 = perf_counter_ns()
        var total_len = len(token_ids)
        debug_assert(total_len > 0, "forward called with empty token_ids")
        debug_assert(
            base_pos + total_len <= Self.max_seq_len,
            "forward exceeds max_seq_len",
        )

        var ctx = BindContext(RankView(Span(self.arena_bases)), 0)
        var tail_ctx = ctx.with_layer(
            layout.tail.base(self.arena_bases[0], 0))

        var x_main_ranks = layout.activations.x_main.state_binding(ctx)
        var xs = layout.activations.x_residual.state_binding(ctx)
        var logits = self.scratch.binding[
            Gemma4HeadScratch, "logits",
        ](ctx, self.head_plan)

        var consumed = 0
        var pos = base_pos
        while consumed < total_len:
            var remaining = total_len - consumed
            var chunk_len = remaining if remaining < C.SLIDING_WINDOW else C.SLIDING_WINDOW
            var is_last = (consumed + chunk_len == total_len)

            var chunk = Span[Int32, tok_origin](
                ptr=token_ids.unsafe_ptr() + consumed,
                length=chunk_len)

            var embed_row_workspace = self.scratch.binding[
                Gemma4HeadScratch, "embed_row_workspace",
            ](ctx, self.head_plan)
            dispatch_bq_embed_lookup[
                hidden=C.HIDDEN, scale=embed_scale,
                max_worker_count=MAX_WORKERS,
            ](chunk, layout.tail.proto.embed.bq_weight(tail_ctx),
              x_main_ranks, embed_row_workspace, shard_rows,
              chunk_len, self.pools, self.profiler)
            dispatch_allreduce_inplace[BF16](
                x_main_ranks, chunk_len * C.HIDDEN, self.pools, self.profiler)

            for i in range(C.NUM_LAYERS):
                var entry = LAYER_SCHEDULE[i]
                var lb: Int
                var body: BodyRefs
                if entry.kind == LayerKind.FULL:
                    lb = layout.full.base(self.arena_bases[0], entry.local_idx)
                    body = layout.full.proto.body
                else:
                    lb = layout.sliding.base(
                        self.arena_bases[0], entry.local_idx)
                    body = layout.sliding.proto.body
                var layer_ctx = ctx.with_layer(lb)

                if entry.kind == LayerKind.FULL:
                    var fx_i8 = self.scratch.binding[Gemma4FullScratch, "x_i8"](ctx, self.full_plan)
                    var fx_sa = self.scratch.binding[Gemma4FullScratch, "x_sa"](ctx, self.full_plan)
                    var fx_row_workspace = self.scratch.binding[
                        Gemma4FullScratch, "x_row_workspace",
                    ](ctx, self.full_plan)
                    dispatch_bq_norm_quant[
                        hidden=C.HIDDEN, block=128, sqrt_n=sqrt_n, n_eps=n_eps,
                        max_worker_count=MAX_WORKERS,
                    ](x_main_ranks, body.input_norm.binding(layer_ctx),
                      fx_i8, fx_sa, fx_row_workspace,
                      chunk_len, self.pools, self.profiler)
                    var full_act = ButterquantActivation(fx_i8, fx_sa)
                    dispatch_bq_full_attention_qkv[
                        max_seq_len=Self.max_seq_len,
                        max_worker_count=MAX_WORKERS,
                    ](layout, ctx, full_act, pos, chunk_len, entry.local_idx,
                      self.scratch, self.full_plan, self.pools, self.profiler)
                else:
                    var sx_i8 = self.scratch.binding[Gemma4SlidingScratch, "x_i8"](ctx, self.sliding_plan)
                    var sx_sa = self.scratch.binding[Gemma4SlidingScratch, "x_sa"](ctx, self.sliding_plan)
                    var sx_row_workspace = self.scratch.binding[
                        Gemma4SlidingScratch, "x_row_workspace",
                    ](ctx, self.sliding_plan)
                    dispatch_bq_norm_quant[
                        hidden=C.HIDDEN, block=128, sqrt_n=sqrt_n, n_eps=n_eps,
                        max_worker_count=MAX_WORKERS,
                    ](x_main_ranks, body.input_norm.binding(layer_ctx),
                      sx_i8, sx_sa, sx_row_workspace,
                      chunk_len, self.pools, self.profiler)
                    var sl_act = ButterquantActivation(sx_i8, sx_sa)
                    dispatch_bq_sliding_attention_qkv[
                        max_seq_len=Self.max_seq_len,
                        max_worker_count=MAX_WORKERS,
                    ](layout, ctx, sl_act, pos, chunk_len, entry.local_idx,
                      self.scratch, self.sliding_plan, self.pools, self.profiler)

                dispatch_allreduce_inplace[BF16](
                    xs, chunk_len * C.HIDDEN, self.pools, self.profiler)

                fused_norm_residual_add[
                    hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
                ](xs, x_main_ranks, x_main_ranks,
                  body.post_attn_norm.binding(layer_ctx),
                  chunk_len, self.pools, self.profiler)

                dispatch_bq_ffn[max_worker_count=MAX_WORKERS](
                    body, layer_ctx, x_main_ranks, xs, chunk_len,
                    self.scratch, self.ffn_plan, self.pools, self.profiler)

            if is_last:
                var x_last = x_main_ranks.shifted((chunk_len - 1) * C.HIDDEN)
                var head_x_i8 = self.scratch.binding[Gemma4HeadScratch, "head_x_i8"](ctx, self.head_plan)
                var head_x_sa = self.scratch.binding[Gemma4HeadScratch, "head_x_sa"](ctx, self.head_plan)
                var head_row_workspace = self.scratch.binding[
                    Gemma4HeadScratch, "head_row_workspace",
                ](ctx, self.head_plan)
                var head_act = ButterquantActivation(head_x_i8, head_x_sa)

                dispatch_bq_head_prep[
                    hidden=C.HIDDEN, block=128, sqrt_n=sqrt_n, n_eps=n_eps,
                ](x_last, layout.tail.proto.final_norm.binding(tail_ctx),
                  head_act, head_row_workspace)

                dispatch_bq_head_gemv[
                    hidden=C.HIDDEN, cap=C.LOGIT_SOFTCAP,
                    max_worker_count=MAX_WORKERS,
                ](head_act, layout.tail.proto.embed.bq_weight(tail_ctx),
                  logits, vocab_per_rank, self.pools, self.profiler)

            consumed += chunk_len
            pos += chunk_len

        self.profiler.add_wall(Int(perf_counter_ns() - wall_t0))
        return TemporalLogitsView[C.VOCAB_SIZE](
            logits.ptr, self.arena_bases.copy())

    @staticmethod
    def load(
        dir_path: Path,
        topo: NumaTopology,
        var pools: List[Self.Pool],
    ) -> Optional[Self]:
        var degree = len(pools)
        var max_workers = 0
        for r in range(degree):
            var cap = min(MAX_WORKERS, pools[r].get_capacity())
            if cap > max_workers:
                max_workers = cap

        var shards = discover_shards(dir_path)
        if len(shards) == 0:
            print(t"no safetensors shards found in {dir_path}")
            return None
        var n_shards = len(shards)
        print(t"found {n_shards} shard(s)")

        var descs = List[WeightDesc]()
        var layout = build_gemma4_plan[Self.max_seq_len](degree, max_workers, descs)

        var size = layout.arena.host_arena_bytes()
        var size_mb = size // (1024 * 1024)
        var weights_mb = layout.arena.distributed_bytes // (1024 * 1024)
        var state_mb = layout.arena.state_bytes // (1024 * 1024)
        print(
            t"allocating {size_mb} MB x {degree} rank(s) "
            t"({weights_mb} MB weights + {state_mb} MB state each)"
        )

        var arenas = List[NumaArena[alignment=DEFAULT_ALIGNMENT]](capacity=degree)
        var arena_bases = List[Int]()
        for rank in range(degree):
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

        for rank in range(degree):
            _ = arenas[rank].prefault(layout.arena.distributed_bytes, layout.arena.state_bytes)

        var model = Self(arenas^, pools^, layout, degree, max_workers)
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
                if not q.plan_walk[FullLayerRefs](prefix, entry.idx):
                    return False
            else:
                if not q.plan_walk[SlidingLayerRefs](prefix, entry.idx):
                    return False
        if not q.plan_walk[TailRefs](String(""), -1):
            return False
        if not q.write_header():
            return False
        return q.execute(topo, pools^)
