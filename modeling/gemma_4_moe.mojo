from std.collections import InlineArray
from std.pathlib import Path
from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from simd_math.ops import sqrt

from numa import NumaArena, NumaTopology
from threading import BurstPool
from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from kernels.helpers import Binding, ArenaBases
from kernels.reductions import (
    dispatch_allreduce_inplace, dispatch_broadcast_from_ptr,
)
from kernels.rmsnorm import dispatch_rms_norm, dispatch_rms_norm_qkv_heads
from kernels.rmsnorm import fused_norm_residual_add
from kernels.gemv import (
    dispatch_gemv_chained_qkv, dispatch_gemv, dispatch_gemv_softcap,
)
from kernels.rope import dispatch_rope_cache_write
from kernels.kv_tiled_attention import (
    dispatch_sliding_attention, FlashDecodeKernel,
)
from kernels.full_attention import (
    dispatch_full_attention, FullAttentionKernel,
)
from kernels.logsum_merge import (
    dispatch_merge_flash_partials, dispatch_merge_context_flash_partials,
)
from kernels.moe_router import (
    RouterCandidate, SparseRoute,
    dispatch_router_sharded, merge_router_candidates, build_expert_schedules,
)
from kernels.moe_experts import (
    dispatch_phase1_gate_up, dispatch_phase2_down,
    Phase1GateUpKernel,
)
from kernels.elementwise import dispatch_gelu_gate_up, dispatch_scalar_mul
from modeling.temporal_scratch import (
    ScratchBuffer, ScratchIsland, ScratchPhase, ScratchPhaseOrder,
    TemporalLogitsView, TemporalScratchPool, aggregate_scratch_peak,
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
    Repeated, ArenaLayout, BF16Bind,
)
from modeling.slot import (
    Slot, SlotGroup, BindContext, stamp_offsets, emit_descs,
)
from modeling.loader import discover_shards, load_weights_from_descs


comptime C = Gemma4BaseConfig


comptime SlidingAttentionContract[degree: Int]: Bool = (
    degree > 0
    and C.NUM_HEADS % degree == 0
    and C.NUM_KV_HEADS_SLIDING % degree == 0
    and C.Q_DIM_SLIDING % degree == 0
    and C.KV_DIM_SLIDING % degree == 0
)


comptime FullAttentionContract[degree: Int]: Bool = (
    degree > 0
    and C.NUM_HEADS % degree == 0
    and C.Q_DIM_FULL % degree == 0
    and C.MAX_SEQ_LEN % degree == 0
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


struct Gemma4StateShapes[degree: Int]:
    """Distribution choices for per-token state tensors.

    `D` shards a tensor across the `degree` ranks (one slice per rank).
    `Local` keeps the whole tensor on every rank (replicated). The
    project's NUMA principle says: data should live closest to its
    most-frequent reader. The sliding rope table is small and read by
    every attention worker on every token — replicating it per rank
    keeps reads NUMA-local. The full rope table is larger and accessed
    less uniformly, so it's sharded.
    """
    comptime D     = DistributionDegree[Self.degree]
    comptime Local = DistributionDegree[1]
    comptime SlidingKV   = TensorColumnSharded[C.SLIDING_WINDOW, C.KV_DIM_SLIDING, Self.D]
    comptime FullKV      = ContextRowSharded[C.MAX_SEQ_LEN, C.KV_DIM_FULL, Self.D]
    comptime SlidingRope = ContextRowSharded[C.MAX_SEQ_LEN, C.ROPE_HALF_SLIDING, Self.Local]
    comptime FullRope    = ContextRowSharded[C.MAX_SEQ_LEN, C.ROPE_HALF_FULL, Self.D]


struct Gemma4TailShapes[degree: Int]:
    comptime D = DistributionDegree[Self.degree]
    comptime FinalNorm = Replicated[C.HIDDEN, 1]
    comptime Embed = VocabularyRowSharded[C.VOCAB_SIZE, C.HIDDEN, Self.D]


struct SlidingAttnRefs[degree: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4Shapes[Self.degree]
    var q_proj: Slot[BF16, Self.S.SlidingQ,  "self_attn.q_proj.weight"]
    var k_proj: Slot[BF16, Self.S.SlidingKV, "self_attn.k_proj.weight"]
    var v_proj: Slot[BF16, Self.S.SlidingKV, "self_attn.v_proj.weight"]
    var o_proj: Slot[BF16, Self.S.SlidingO,  "self_attn.o_proj.weight"]
    var q_norm: Slot[BF16, Shape[C.HEAD_DIM_SLIDING, 1], "self_attn.q_norm.weight"]
    var k_norm: Slot[BF16, Shape[C.HEAD_DIM_SLIDING, 1], "self_attn.k_norm.weight"]


struct FullAttnRefs[degree: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4Shapes[Self.degree]
    var q_proj: Slot[BF16, Self.S.FullQ, "self_attn.q_proj.weight"]
    var k_proj: Slot[BF16, Self.S.FullK, "self_attn.k_proj.weight"]
    var o_proj: Slot[BF16, Self.S.FullO, "self_attn.o_proj.weight"]
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
    var gate_proj:       Slot[BF16, Self.S.GateUp,              "mlp.gate_proj.weight"]
    var up_proj:         Slot[BF16, Self.S.GateUp,              "mlp.up_proj.weight"]
    var down_proj:       Slot[BF16, Self.S.Down,                "mlp.down_proj.weight"]
    var router_proj:     Slot[BF16, Self.S.RouterProj,          "router.proj.weight"]
    var router_scale:    Slot[BF16, Shape[C.HIDDEN, 1],         "router.scale"]
    var router_pes:      Slot[BF16, Shape[C.NUM_EXPERTS, 1],    "router.per_expert_scale"]
    var experts_gate_up: Slot[BF16, Self.S.ExpertsGateUp,       "experts.gate_up_proj"]
    var experts_down:    Slot[BF16, Self.S.ExpertsDown,         "experts.down_proj"]
    var layer_scalar:    Slot[BF16, Shape[1, 1],                "layer_scalar"]


struct SlidingLayerRefs[degree: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    var attn: SlidingAttnRefs[Self.degree]
    var body: BodyRefs[Self.degree]


struct FullLayerRefs[degree: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    var attn: FullAttnRefs[Self.degree]
    var body: BodyRefs[Self.degree]


struct SlidingKVSlots[degree: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4StateShapes[Self.degree]
    var k: Slot[BF16, Self.S.SlidingKV]
    var v: Slot[BF16, Self.S.SlidingKV]


struct FullKVSlots[degree: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4StateShapes[Self.degree]
    var k: Slot[BF16, Self.S.FullKV]
    var v: Slot[BF16, Self.S.FullKV]


struct RopeSlots[half: Int, degree: Int = 1](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime D = DistributionDegree[Self.degree]
    var cos: Slot[F32, ContextRowSharded[C.MAX_SEQ_LEN, Self.half, Self.D]]
    var sin: Slot[F32, ContextRowSharded[C.MAX_SEQ_LEN, Self.half, Self.D]]


struct ActivationSlots(Copyable, ImplicitlyCopyable, SlotGroup):
    var x_main:     Slot[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]]
    var x_residual: Slot[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]]


struct TailRefs[degree: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4TailShapes[Self.degree]
    var final_norm: Slot[BF16, Self.S.FinalNorm, "model.language_model.norm.weight"]
    var embed:      Slot[BF16, Self.S.Embed, "model.language_model.embed_tokens.weight"]


@fieldwise_init
struct Gemma4Layout[degree: Int](Copyable, ImplicitlyCopyable):
    var arena: ArenaLayout
    var sliding: Repeated[SlidingLayerRefs[Self.degree]]
    var full: Repeated[FullLayerRefs[Self.degree]]

    var sliding_kv: Repeated[SlidingKVSlots[Self.degree]]
    var full_kv: Repeated[FullKVSlots[Self.degree]]
    var activations: ActivationSlots
    var sliding_rope: RopeSlots[C.ROPE_HALF_SLIDING]
    var full_rope: RopeSlots[C.ROPE_HALF_FULL, Self.degree]

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
    # Per-worker flash partial stride is owned by the FlashDecodeKernel.
    comptime FlashK = FlashDecodeKernel[
        Self.head_dim, Self.num_q_heads,
        Self.num_q_heads // Self.num_kv_heads, Self.kv_rows,
        C.SLIDING_WINDOW,
    ]

    comptime PHASES = ScratchPhaseOrder[
        "gemv_qkv", "rms_norm_qkv", "rope_cache_write",
        "flash", "merge_partials", "o_proj",
    ]

    var q_band: ScratchPhase["gemv_qkv", "o_proj"]
    var q: ScratchBuffer[
        Scalar[DType.bfloat16], C.MAX_SEQ_LEN * Self.q_rows,
    ]

    var kv_band: ScratchPhase["gemv_qkv", "rope_cache_write"]
    var kv: ScratchBuffer[
        Scalar[DType.bfloat16], C.MAX_SEQ_LEN * Self.kv_rows * 2,
    ]

    var partials_band: ScratchPhase["flash", "merge_partials"]
    var partials: ScratchBuffer[
        Scalar[DType.float32], Self.max_worker_count * Self.FlashK.PARTIAL_STRIDE,
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
    # Per-worker flash partial stride is owned by the FullAttentionKernel.
    comptime FullK = FullAttentionKernel[
        Self.head_dim, Self.num_q_heads,
        C.NUM_HEADS // C.NUM_KV_HEADS_FULL, Self.k_rows,
    ]

    comptime PHASES = ScratchPhaseOrder[
        "gemv_q", "gemv_kv", "rms_norm_qkv", "rope_cache_write",
        "flash", "merge_partials", "o_proj",
    ]

    var q_band: ScratchPhase["gemv_q", "flash"]
    var q: ScratchBuffer[
        Scalar[DType.bfloat16], C.MAX_SEQ_LEN * Self.q_rows,
    ]

    var kv_band: ScratchPhase["gemv_kv", "rope_cache_write"]
    var kv: ScratchBuffer[
        Scalar[DType.bfloat16], C.MAX_SEQ_LEN * Self.k_rows * 2,
    ]

    var partials_band: ScratchPhase["flash", "merge_partials"]
    var partials: ScratchBuffer[
        Scalar[DType.float32], Self.max_worker_count * Self.FullK.PARTIAL_STRIDE,
    ]

    var q_local_band: ScratchPhase["merge_partials", "o_proj"]
    var q_local: ScratchBuffer[
        Scalar[DType.bfloat16], C.MAX_SEQ_LEN * Self.local_q_rows,
    ]


@fieldwise_init
struct Gemma4FfnMoeScratch[degree: Int, max_worker_count: Int = 128](
    ScratchIsland, Copyable, ImplicitlyCopyable
):
    comptime S = Gemma4Shapes[Self.degree]
    comptime intermediate_per_rank = Self.S.GateUp.DATA_N
    comptime experts_per_rank = C.NUM_EXPERTS // Self.degree

    comptime PHASES = ScratchPhaseOrder[
        "ffn_rms_norm", "gemv_gate", "gemv_up", "gelu_gate_up",
        "router_sharded", "merge_cands", "moe_rms_norm",
        "build_schedules", "phase1_gate_up", "phase2_down",
        "moe_allreduce", "gemv_dense", "allreduce_dense",
        "post_norm_1", "post_norm_2", "post_norm_3",
    ]

    var ffn_gate_band: ScratchPhase["gemv_gate", "gemv_dense"]
    var ffn_gate: ScratchBuffer[
        Scalar[DType.bfloat16],
        C.MAX_SEQ_LEN * Self.intermediate_per_rank,
    ]

    var ffn_up_band: ScratchPhase["gemv_up", "gelu_gate_up"]
    var ffn_up: ScratchBuffer[
        Scalar[DType.bfloat16],
        C.MAX_SEQ_LEN * Self.intermediate_per_rank,
    ]

    var router_workspace: ScratchPhase[
        "router_sharded", "router_sharded",
    ]
    var moe_router_scaled: ScratchBuffer[
        Scalar[DType.float32], Self.max_worker_count * C.HIDDEN,
    ]

    var router_cands: ScratchPhase["router_sharded", "merge_cands"]
    var moe_cands: ScratchBuffer[RouterCandidate, C.MAX_SEQ_LEN * C.TOP_K]

    var router_products: ScratchPhase["merge_cands", "build_schedules"]
    var moe_route_idx: ScratchBuffer[
        Scalar[DType.int32], C.MAX_SEQ_LEN * C.TOP_K,
    ]
    var moe_route_w: ScratchBuffer[
        Scalar[DType.float32], C.MAX_SEQ_LEN * C.TOP_K,
    ]

    var expert_input: ScratchPhase["moe_rms_norm", "phase1_gate_up"]
    var moe_x_normed: ScratchBuffer[
        Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.HIDDEN,
    ]

    var schedule_products: ScratchPhase[
        "build_schedules", "phase2_down",
    ]
    var moe_expert_offset: ScratchBuffer[
        Scalar[DType.int32], Self.experts_per_rank + 1,
    ]
    var moe_routes: ScratchBuffer[SparseRoute, C.MAX_SEQ_LEN * C.TOP_K]

    var hidden_bucket: ScratchPhase["phase1_gate_up", "phase2_down"]
    var moe_hidden_bucket: ScratchBuffer[
        Scalar[DType.bfloat16],
        C.MAX_SEQ_LEN * C.TOP_K * C.MOE_INTERMEDIATE,
    ]

    var phase1_workspace: ScratchPhase[
        "phase1_gate_up", "phase1_gate_up",
    ]
    var moe_gate_scratch: ScratchBuffer[
        Scalar[DType.float32],
        Self.max_worker_count * Phase1GateUpKernel[
            C.HIDDEN, C.MOE_GATE_UP_FUSED, C.MOE_INTERMEDIATE,
            Self.experts_per_rank,
        ].WORKER_SCRATCH_ELEMS,
    ]

    var phase2_accum: ScratchPhase["phase2_down", "phase2_down"]
    var moe_accum: ScratchBuffer[
        Scalar[DType.float32], C.MAX_SEQ_LEN * C.HIDDEN,
    ]

    var dense_band: ScratchPhase["gemv_dense", "post_norm_3"]
    var ffn_dense_out: ScratchBuffer[
        Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.HIDDEN,
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
        Scalar[DType.bfloat16], Self.vocab_per_rank,
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
    degree: Int, max_worker_count: Int = 128,
](mut descs: List[WeightDesc]) -> Gemma4Layout[degree]:
    comptime assert SlidingAttentionContract[degree], "sliding attention distribution contract failed"
    comptime assert FullAttentionContract[degree], "full attention distribution contract failed"
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
        var prefix = "model.language_model.layers." + String(entry.idx) + "."
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

    var skv_proto = SlidingKVSlots[degree]()
    var skv_stride = stamp_offsets(skv_proto)
    var sliding_kv = Repeated[SlidingKVSlots[degree]](
        skv_proto, state_cursor, skv_stride, C.NUM_SLIDING_LAYERS)
    state_cursor = align_up(state_cursor + C.NUM_SLIDING_LAYERS * skv_stride)

    var fkv_proto = FullKVSlots[degree]()
    var fkv_stride = stamp_offsets(fkv_proto)
    var full_kv = Repeated[FullKVSlots[degree]](
        fkv_proto, state_cursor, fkv_stride, C.NUM_FULL_LAYERS)
    state_cursor = align_up(state_cursor + C.NUM_FULL_LAYERS * fkv_stride)

    var activations = ActivationSlots()
    state_cursor = stamp_offsets(activations, state_cursor)

    var scratch_cap = calculate_peak_scratch[degree, max_worker_count]()
    state_cursor = align_up(state_cursor)
    var scratch_off = state_cursor
    state_cursor = align_up(state_cursor + scratch_cap)

    var sliding_rope = RopeSlots[C.ROPE_HALF_SLIDING]()
    state_cursor = stamp_offsets(sliding_rope, state_cursor)
    var full_rope = RopeSlots[C.ROPE_HALF_FULL, degree]()
    state_cursor = stamp_offsets(full_rope, state_cursor)

    var arena = ArenaLayout(
        base=0,
        distributed_bytes=distributed,
        state_bytes=state_cursor - distributed,
        host_bytes=align_up(state_cursor),
        scratch_off=scratch_off,
    )
    return Gemma4Layout[degree](
        arena=arena,
        sliding=Repeated[SlidingLayerRefs[degree]](sl_proto, sl_off, sl_stride, C.NUM_SLIDING_LAYERS),
        full=Repeated[FullLayerRefs[degree]](fl_proto, fl_off, fl_stride, C.NUM_FULL_LAYERS),
        sliding_kv=sliding_kv, full_kv=full_kv,
        activations=activations,
        sliding_rope=sliding_rope, full_rope=full_rope,
        tail=tail)


@always_inline
def sliding_valid_len(pos: Int) -> Int:
    if pos + 1 >= C.SLIDING_WINDOW:
        return C.SLIDING_WINDOW
    return pos + 1


@always_inline
def full_valid_count(rank: Int, pos: Int, degree: Int) -> Int:
    if pos < 0:
        return 0
    if rank <= pos % degree:
        return pos // degree + 1
    return pos // degree


def dispatch_sliding_attention_qkv[
    P: BurstThreadPool, //, degree: Int, max_worker_count: Int = 128,
](
    layout: Gemma4Layout[degree],
    ctx: BindContext[degree],
    pos: Int,
    layer_idx: Int,
    mut scratch: Gemma4ScratchPool[degree, max_worker_count],
    mut pools: HeapMoveArray[P],
):
    comptime S = Gemma4Shapes[degree]
    comptime q_rows = S.SlidingQ.DATA_N
    comptime kv_rows = S.SlidingKV.DATA_N
    comptime head_dim = C.HEAD_DIM_SLIDING
    comptime num_q_heads = q_rows // head_dim
    comptime num_kv_heads = kv_rows // head_dim
    comptime sqrt_hd = sqrt[DType.float32, 1](head_dim)
    comptime hd_eps = head_dim * C.RMS_NORM_EPS
    comptime rope_half = C.ROPE_HALF_SLIDING
    comptime kv_cols = kv_rows
    comptime Island = Gemma4SlidingScratch[degree, max_worker_count]

    var attn_ctx = ctx.with_layer(layout.sliding.base(ctx.arena_bases[0], layer_idx))
    var attn = layout.sliding.proto.attn

    var q_outs = scratch.binding[Island, "q"](ctx)
    var k_outs = scratch.binding[Island, "kv"](ctx)
    var v_outs = k_outs.shifted(kv_rows)
    var xs = layout.activations.x_residual.state_binding(ctx)

    dispatch_gemv_chained_qkv[
        q_rows=q_rows, kv_rows=kv_rows, cols=C.HIDDEN, tp=degree,
        max_worker_count=max_worker_count,
    ](xs,
      attn.q_proj.binding(attn_ctx),
      attn.k_proj.binding(attn_ctx),
      attn.v_proj.binding(attn_ctx),
      q_outs, k_outs, v_outs, pools)

    dispatch_rms_norm_qkv_heads[
        head_dim=head_dim, sqrt_n=sqrt_hd, n_eps=hd_eps,
        num_q=num_q_heads, num_kv=num_kv_heads, tp=degree,
        max_worker_count=max_worker_count,
    ](q_outs, q_outs, k_outs, k_outs, v_outs, v_outs,
      attn.q_norm.binding(attn_ctx),
      attn.k_norm.binding(attn_ctx),
      pools)

    var kv_lb = layout.sliding_kv.base(ctx.arena_bases[0], layer_idx)
    var k_kv = layout.sliding_kv.proto.k.binding(kv_lb, ctx.arena_bases)
    var v_kv = layout.sliding_kv.proto.v.binding(kv_lb, ctx.arena_bases)

    dispatch_rope_cache_write[
        half=rope_half, pair_stride=head_dim // 2,
        num_q=num_q_heads, num_kv=num_kv_heads,
        head_dim=head_dim, kv_cache_stride=kv_cols,
        slot_mask=C.SLIDING_WINDOW - 1, cache_degree=1, tp=degree,
        max_worker_count=max_worker_count,
    ](q_outs, k_outs, v_outs,
      k_kv, v_kv,
      layout.sliding_rope.cos.state_binding(ctx),
      layout.sliding_rope.sin.state_binding(ctx),
      pos, 1, pools)

    var partials = scratch.binding[Island, "partials"](ctx)

    var nws = dispatch_sliding_attention[
        head_dim=head_dim, num_q=num_q_heads,
        gqa_ratio=num_q_heads // num_kv_heads, kv_stride=kv_cols,
        window=C.SLIDING_WINDOW, tp=degree,
        max_worker_count=max_worker_count,
    ](q_outs, k_kv, v_kv, partials, pos, sliding_valid_len(pos), pools)

    dispatch_merge_flash_partials[
        head_dim, num_q_heads, tp=degree,
        max_worker_count=max_worker_count,
    ](
        q_outs, partials, nws, pools)

    dispatch_gemv[
        rows=C.HIDDEN, cols=q_rows, tp=degree,
        max_worker_count=max_worker_count,
    ](
        q_outs,
        attn.o_proj.binding(attn_ctx),
        xs, pools)


def dispatch_full_attention_qkv[
    P: BurstThreadPool, //, degree: Int, max_worker_count: Int = 128,
](
    layout: Gemma4Layout[degree],
    ctx: BindContext[degree],
    pos: Int,
    layer_idx: Int,
    mut scratch: Gemma4ScratchPool[degree, max_worker_count],
    mut pools: HeapMoveArray[P],
):
    comptime S = Gemma4Shapes[degree]
    comptime q_rows = S.FullQ.DATA_N
    comptime local_q_rows = S.FullO.DATA_M
    comptime k_rows = S.FullK.DATA_N
    comptime head_dim = C.HEAD_DIM_FULL
    comptime num_q_heads = q_rows // head_dim
    comptime local_num_q_heads = local_q_rows // head_dim
    comptime num_kv_heads = k_rows // head_dim
    comptime sqrt_hd = sqrt[DType.float32, 1](head_dim)
    comptime hd_eps = head_dim * C.RMS_NORM_EPS
    comptime rope_half = C.ROPE_HALF_FULL
    comptime pair_stride = head_dim // 2
    comptime kv_cols = k_rows
    comptime Island = Gemma4FullScratch[degree, max_worker_count]

    var attn_ctx = ctx.with_layer(layout.full.base(ctx.arena_bases[0], layer_idx))
    var attn = layout.full.proto.attn

    var q_outs = scratch.binding[Island, "q"](ctx)
    var k_outs = scratch.binding[Island, "kv"](ctx)
    var v_outs = k_outs.shifted(k_rows)
    var xs = layout.activations.x_residual.state_binding(ctx)

    dispatch_gemv[
        rows=q_rows, cols=C.HIDDEN, tp=degree,
        max_worker_count=max_worker_count,
    ](
        xs, attn.q_proj.binding(attn_ctx), q_outs, pools)
    dispatch_gemv[
        rows=k_rows, cols=C.HIDDEN, tp=degree,
        max_worker_count=max_worker_count,
    ](
        xs, attn.k_proj.binding(attn_ctx), k_outs, pools)

    dispatch_rms_norm_qkv_heads[
        head_dim=head_dim, sqrt_n=sqrt_hd, n_eps=hd_eps,
        num_q=num_q_heads, num_kv=num_kv_heads, tp=degree,
        max_worker_count=max_worker_count,
    ](q_outs, q_outs, k_outs, k_outs, k_outs, v_outs,
      attn.q_norm.binding(attn_ctx),
      attn.k_norm.binding(attn_ctx),
      pools)

    var owner_bases = ArenaBases[degree].fill(ctx.arena_bases[pos % degree])
    var rope_owner_ctx = BindContext[degree](
        arena_bases=owner_bases, layer_base=owner_bases[0])

    var kv_lb = layout.full_kv.base(ctx.arena_bases[0], layer_idx)
    var k_kv = layout.full_kv.proto.k.binding(kv_lb, ctx.arena_bases)
    var v_kv = layout.full_kv.proto.v.binding(kv_lb, ctx.arena_bases)

    dispatch_rope_cache_write[
        half=rope_half, pair_stride=pair_stride,
        num_q=num_q_heads, num_kv=num_kv_heads,
        head_dim=head_dim, kv_cache_stride=kv_cols,
        slot_mask=-1, cache_degree=degree, tp=degree,
        max_worker_count=max_worker_count,
    ](q_outs, k_outs, v_outs,
      k_kv, v_kv,
      layout.full_rope.cos.state_binding(rope_owner_ctx),
      layout.full_rope.sin.state_binding(rope_owner_ctx),
      pos, 1, pools)

    var q_local_outs = scratch.binding[Island, "q_local"](ctx)
    var partials = scratch.binding[Island, "partials"](ctx)

    var valid_lens = InlineArray[Int, degree](uninitialized=True)
    for rank in range(degree):
        valid_lens[rank] = full_valid_count(rank, pos, degree)

    var nws = dispatch_full_attention[
        head_dim=head_dim, num_q=num_q_heads,
        gqa_ratio=C.NUM_HEADS // C.NUM_KV_HEADS_FULL, kv_stride=kv_cols,
        tp=degree, max_worker_count=max_worker_count,
    ](q_outs, k_kv, v_kv, partials, valid_lens, pools)

    dispatch_merge_context_flash_partials[
        head_dim=head_dim, num_q=num_q_heads,
        local_num_q=local_num_q_heads, tp=degree,
        max_worker_count=max_worker_count,
    ](q_local_outs, partials, nws, pools)

    dispatch_gemv[
        rows=C.HIDDEN, cols=local_q_rows, tp=degree,
        max_worker_count=max_worker_count,
    ](
        q_local_outs,
        attn.o_proj.binding(attn_ctx),
        xs, pools)


def dispatch_moe[
    P: BurstThreadPool, //, degree: Int, max_worker_count: Int = 128,
](
    body: BodyRefs[degree],
    ctx: BindContext[degree],
    x_input: BF16Bind[degree],
    moe_out: BF16Bind[degree],
    seq_len: Int,
    mut scratch: Gemma4ScratchPool[degree, max_worker_count],
    mut pools: HeapMoveArray[P],
):
    comptime experts_per_rank = C.NUM_EXPERTS // degree
    comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
    comptime n_eps = C.HIDDEN * C.RMS_NORM_EPS
    comptime rms_eps = Scalar[DType.float32](C.RMS_NORM_EPS)
    comptime Ffn = Gemma4FfnMoeScratch[degree, max_worker_count]

    var per_expert_scale_ptr = body.router_pes.at(ctx.layer_base)

    var x_normed = scratch.binding[Ffn, "moe_x_normed"](ctx)
    var cands = scratch.binding[Ffn, "moe_cands"](ctx)
    var router_scaled = scratch.binding[Ffn, "moe_router_scaled"](ctx)
    var route_idx = scratch.binding[Ffn, "moe_route_idx"](ctx)
    var route_w = scratch.binding[Ffn, "moe_route_w"](ctx)
    var expert_offset = scratch.binding[Ffn, "moe_expert_offset"](ctx)
    var routes = scratch.binding[Ffn, "moe_routes"](ctx)
    var hidden_bucket = scratch.binding[Ffn, "moe_hidden_bucket"](ctx)
    var moe_accum = scratch.binding[Ffn, "moe_accum"](ctx)
    var gate_scratch = scratch.binding[Ffn, "moe_gate_scratch"](ctx)

    dispatch_router_sharded[
        hidden=C.HIDDEN, experts_per_rank=experts_per_rank,
        top_k=C.TOP_K, tp=degree, rms_eps=rms_eps,
        max_worker_count=max_worker_count,
    ](x_input,
      body.router_proj.binding(ctx),
      body.router_scale.binding(ctx),
      router_scaled, cands, seq_len, pools)

    merge_router_candidates[degree, C.TOP_K](
        cands, per_expert_scale_ptr, route_idx, route_w, seq_len)

    dispatch_rms_norm[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps, tp=degree,
        max_worker_count=max_worker_count,
    ](x_input, x_normed,
      body.pre_ffn_norm_2.binding(ctx), seq_len, pools)

    build_expert_schedules[degree, experts_per_rank, C.TOP_K](
        route_idx, route_w, expert_offset, routes, seq_len)

    dispatch_phase1_gate_up[
        hidden=C.HIDDEN, gate_up_fused=C.MOE_GATE_UP_FUSED,
        intermediate=C.MOE_INTERMEDIATE,
        experts_per_rank=experts_per_rank, tp=degree,
        max_worker_count=max_worker_count,
    ](x_normed, expert_offset, routes,
      body.experts_gate_up.binding(ctx),
      gate_scratch, hidden_bucket, pools)

    dispatch_phase2_down[
        hidden=C.HIDDEN, intermediate=C.MOE_INTERMEDIATE,
        experts_per_rank=experts_per_rank, tp=degree,
        max_worker_count=max_worker_count,
    ](expert_offset, routes, hidden_bucket,
      body.experts_down.binding(ctx),
      moe_accum, moe_out, seq_len, pools)

    dispatch_allreduce_inplace[
        BF16, degree, max_worker_count=max_worker_count,
    ](moe_out, seq_len * C.HIDDEN, pools)


def dispatch_ffn[
    P: BurstThreadPool, //, degree: Int, max_worker_count: Int = 128,
](
    body: BodyRefs[degree],
    ctx: BindContext[degree],
    x_main: BF16Bind[degree],
    x_residual: BF16Bind[degree],
    seq_len: Int,
    mut scratch: Gemma4ScratchPool[degree, max_worker_count],
    mut pools: HeapMoveArray[P],
):
    comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
    comptime n_eps = C.HIDDEN * C.RMS_NORM_EPS
    comptime intermediate_per_rank = Gemma4Shapes[degree].GateUp.DATA_N
    comptime Ffn = Gemma4FfnMoeScratch[degree, max_worker_count]

    var layer_scalar_ptr = body.layer_scalar.at(ctx.layer_base)

    var gate = scratch.binding[Ffn, "ffn_gate"](ctx)
    var up = scratch.binding[Ffn, "ffn_up"](ctx)
    var dense_out = scratch.binding[Ffn, "ffn_dense_out"](ctx)

    dispatch_rms_norm[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps, tp=degree,
        max_worker_count=max_worker_count,
    ](x_main, x_residual,
      body.pre_ffn_norm.binding(ctx), seq_len, pools)

    dispatch_gemv[
        rows=intermediate_per_rank, cols=C.HIDDEN, tp=degree,
        max_worker_count=max_worker_count,
    ](x_residual, body.gate_proj.binding(ctx), gate, pools)

    dispatch_gemv[
        rows=intermediate_per_rank, cols=C.HIDDEN, tp=degree,
        max_worker_count=max_worker_count,
    ](x_residual, body.up_proj.binding(ctx), up, pools)

    dispatch_gelu_gate_up[
        intermediate=intermediate_per_rank, tp=degree,
        max_worker_count=max_worker_count,
    ](gate, up, gate, seq_len, pools)

    dispatch_moe[degree=degree, max_worker_count=max_worker_count](
        body, ctx, x_main, x_residual, seq_len, scratch, pools)

    dispatch_gemv[
        rows=C.HIDDEN, cols=intermediate_per_rank, tp=degree,
        max_worker_count=max_worker_count,
    ](gate, body.down_proj.binding(ctx), dense_out, pools)

    dispatch_allreduce_inplace[
        BF16, degree, max_worker_count=max_worker_count,
    ](dense_out, seq_len * C.HIDDEN, pools)

    dispatch_rms_norm[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps, tp=degree,
        max_worker_count=max_worker_count,
    ](dense_out, dense_out,
      body.post_ffn_norm_1.binding(ctx), seq_len, pools)

    fused_norm_residual_add[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps, tp=degree,
        max_worker_count=max_worker_count,
    ](x_residual, dense_out, dense_out,
      body.post_ffn_norm_2.binding(ctx), seq_len, pools)

    fused_norm_residual_add[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps, tp=degree,
        max_worker_count=max_worker_count,
    ](dense_out, x_main, x_main,
      body.post_ffn_norm.binding(ctx), seq_len, pools)

    var ls_value = layer_scalar_ptr[0].cast[DType.float32]()
    dispatch_scalar_mul[
        hidden=C.HIDDEN, tp=degree, max_worker_count=max_worker_count,
    ](x_main, x_main, ls_value, seq_len, pools)


struct Gemma4[
    degree: Int, max_worker_count: Int = 128,
    Pool: BurstThreadPool = BurstPool[],
](Movable):
    var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var pools: HeapMoveArray[Self.Pool]
    var layout: Gemma4Layout[Self.degree]
    var scratch: Gemma4ScratchPool[Self.degree, Self.max_worker_count]
    var arena_bases: ArenaBases[Self.degree]

    def __init__(out self,
        var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]],
        var pools: HeapMoveArray[Self.Pool],
        layout: Gemma4Layout[Self.degree],
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

    def model_init(mut self):
        ref layout = self.layout
        comptime width = simd_width_of[DType.float32]()

        comptime inv_sqrt_hidden = 1.0 / sqrt[DType.float32, 1](C.HIDDEN)
        for rank in range(Self.degree):
            var arena_base = self.arena_bases[rank]
            for i in range(C.NUM_LAYERS):
                var entry = LAYER_SCHEDULE[i]
                var p: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
                if entry.kind == LayerKind.FULL:
                    var lb = layout.full.base(arena_base, entry.local_idx)
                    p = layout.full.proto.body.router_scale.at(lb)
                else:
                    var lb = layout.sliding.base(arena_base, entry.local_idx)
                    p = layout.sliding.proto.body.router_scale.at(lb)
                for j in range(0, C.HIDDEN, width):
                    var lane = p + j
                    var v = lane.load[width=width]().cast[DType.float32]()
                    lane.store((v * inv_sqrt_hidden).cast[DType.bfloat16]())
        print("  router constants baked")

        from kernels.rope import init_rope_table, init_rope_table_partial_strided
        for rank in range(Self.degree):
            var base = self.arena_bases[rank]
            var sl_cos = layout.sliding_rope.cos.at(base)
            var sl_sin = layout.sliding_rope.sin.at(base)
            init_rope_table[C.ROPE_HALF_SLIDING, C.MAX_SEQ_LEN](
                sl_cos, sl_sin, 10000.0)
            var fl_cos = layout.full_rope.cos.at(base)
            var fl_sin = layout.full_rope.sin.at(base)
            init_rope_table_partial_strided[
                C.ROPE_HALF_FULL, C.MAX_SEQ_LEN // Self.degree,
            ](fl_cos, fl_sin, 1000000.0, C.HEAD_DIM_FULL, rank, Self.degree)
        print("  rope tables initialized")

    def forward(
        mut self, token_id: Int, pos: Int,
    ) -> TemporalLogitsView[C.VOCAB_SIZE, Self.degree]:
        ref layout = self.layout
        comptime shard_rows = Gemma4TailShapes[Self.degree].Embed.DATA_N
        comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
        comptime n_eps = C.HIDDEN * C.RMS_NORM_EPS

        var owner = token_id // shard_rows
        var local_row = token_id % shard_rows

        var ctx = BindContext[Self.degree](
            arena_bases=self.arena_bases, layer_base=0)

        var x_main_ranks = layout.activations.x_main.state_binding(ctx)
        var x_res_ranks = layout.activations.x_residual.state_binding(ctx)

        var tail_base_owner = layout.tail.base(self.arena_bases[owner], 0)
        var embed_row = layout.tail.proto.embed.at(tail_base_owner) + local_row * C.HIDDEN

        dispatch_broadcast_from_ptr[
            BF16, Self.degree, max_worker_count=Self.max_worker_count,
        ](embed_row.as_immutable(), x_main_ranks, C.HIDDEN,
          self.pools, src_rank=owner)

        comptime embed_scale = sqrt[DType.float32, 1](C.HIDDEN).cast[DType.bfloat16]().cast[DType.float32]()
        dispatch_scalar_mul[
            hidden=C.HIDDEN, tp=Self.degree,
            max_worker_count=Self.max_worker_count,
        ](x_main_ranks, x_main_ranks, embed_scale, 1, self.pools)

        for i in range(C.NUM_LAYERS):
            var entry = LAYER_SCHEDULE[i]
            var body: BodyRefs[Self.degree]
            var layer_ctx: BindContext[Self.degree]
            if entry.kind == LayerKind.FULL:
                layer_ctx = ctx.with_layer(layout.full.base(self.arena_bases[0], entry.local_idx))
                body = layout.full.proto.body
            else:
                layer_ctx = ctx.with_layer(layout.sliding.base(self.arena_bases[0], entry.local_idx))
                body = layout.sliding.proto.body

            dispatch_rms_norm[
                hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
                tp=Self.degree, max_worker_count=Self.max_worker_count,
            ](x_main_ranks, x_res_ranks,
              body.input_norm.binding(layer_ctx),
              1, self.pools)

            if entry.kind == LayerKind.FULL:
                dispatch_full_attention_qkv[
                    degree=Self.degree,
                    max_worker_count=Self.max_worker_count,
                ](
                    layout, ctx, pos, entry.local_idx, self.scratch, self.pools)
            else:
                dispatch_sliding_attention_qkv[
                    degree=Self.degree,
                    max_worker_count=Self.max_worker_count,
                ](
                    layout, ctx, pos, entry.local_idx, self.scratch, self.pools)

            dispatch_allreduce_inplace[
                BF16, Self.degree, max_worker_count=Self.max_worker_count,
            ](x_res_ranks, C.HIDDEN, self.pools)

            fused_norm_residual_add[
                hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
                tp=Self.degree, max_worker_count=Self.max_worker_count,
            ](x_res_ranks, x_main_ranks, x_main_ranks,
              body.post_attn_norm.binding(layer_ctx),
              1, self.pools)

            dispatch_ffn[
                degree=Self.degree, max_worker_count=Self.max_worker_count,
            ](
                body, layer_ctx, x_main_ranks, x_res_ranks, 1,
                self.scratch, self.pools)

        var tail_ctx = ctx.with_layer(layout.tail.base(self.arena_bases[0], 0))
        dispatch_rms_norm[
            hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
            tp=Self.degree, max_worker_count=Self.max_worker_count,
        ](x_main_ranks, x_main_ranks,
          layout.tail.proto.final_norm.binding(tail_ctx),
          1, self.pools)

        comptime vocab_per_rank = C.VOCAB_SIZE // Self.degree
        var logits = self.scratch.binding[
            Gemma4HeadScratch[Self.degree], "logits",
        ](ctx)
        dispatch_gemv_softcap[
            rows=vocab_per_rank, cols=C.HIDDEN, tp=Self.degree,
            cap=C.LOGIT_SOFTCAP,
            max_worker_count=Self.max_worker_count,
        ](x_main_ranks, layout.tail.proto.embed.binding(tail_ctx), logits, self.pools)

        return TemporalLogitsView[C.VOCAB_SIZE, Self.degree](
            logits.ptr, self.arena_bases)

    @staticmethod
    def load(
        dir_path: Path,
        topo: NumaTopology,
        var pools: HeapMoveArray[Self.Pool],
    ) -> Optional[Self]:
        var shards = discover_shards(dir_path)
        if len(shards) == 0:
            print("no safetensors shards found in", String(dir_path))
            return None
        print("found", len(shards), "shard(s)")

        var descs = List[WeightDesc]()
        var layout = build_gemma4_plan[
            Self.degree, Self.max_worker_count,
        ](descs)

        var size = layout.arena.host_arena_bytes()
        print("allocating", size // (1024 * 1024), "MB x " + String(Self.degree) + " rank(s) (" +
              String(layout.arena.distributed_bytes // (1024 * 1024)) + " MB weights + " +
              String(layout.arena.state_bytes // (1024 * 1024)) + " MB state each)")

        var arenas = HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]](Self.degree)
        var arena_bases = List[Int]()
        for rank in range(Self.degree):
            arenas.push(NumaArena[alignment=DEFAULT_ALIGNMENT](topo.node(rank), size))
            if not arenas[rank]:
                print("arena allocation failed on node", topo.node(rank))
                return None
            arena_bases.append(Int(arenas[rank].base.value()))

        var load_result = load_weights_from_descs(descs, shards, arena_bases, topo)
        if not load_result:
            print("weight loading failed")
            return None
        var loaded = load_result.take()
        print("loaded", loaded.bytes_loaded // (1024 * 1024), "MB in", loaded.num_ops, "ops")

        for rank in range(Self.degree):
            _ = arenas[rank].prefault(layout.arena.distributed_bytes, layout.arena.state_bytes)

        var model = Self(arenas^, pools^, layout)
        model.model_init()
        return model^
