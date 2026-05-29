from std.collections import InlineArray
from std.pathlib import Path
from std.memory import Span, UnsafePointer
from std.sys.info import simd_width_of
from std.time import perf_counter_ns
from simd_math.ops import sqrt

from numa import NumaArena, NumaTopology
from threading import BurstPool
from threading.threading_traits import BurstThreadPool
from kernels.helpers import Binding, ArenaBases, prime_fp_environment
from kernels.reductions import dispatch_allreduce_inplace
from kernels.embedding import dispatch_embed_lookup
from kernels.rmsnorm import dispatch_rms_norm, dispatch_rms_norm_qkv_heads
from kernels.rmsnorm import fused_norm_residual_add
from kernels.gemv import dispatch_gemv_softcap
from kernels.gemm import dispatch_gemm, dispatch_gemm_chained_qkv
from kernels.rope import dispatch_rope_cache_write
from kernels.attention_ops import flash_partial_stride
from kernels.attention_dispatch_kernels import (
    dispatch_sliding_attention, dispatch_full_attention,
)
from kernels.moe_router import (
    RouterCandidate, SparseRoute,
    dispatch_router_expert, merge_router_candidates_expert, build_expert_schedules,
)
from kernels.moe_experts import (
    dispatch_phase1_gate_up, dispatch_phase2_down,
)
from kernels.elementwise import dispatch_gelu_gate_up, dispatch_scalar_mul
from kernels.profiling import Profiler
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
    var embed:      Slot[BF16, Self.S.Embed, "model.language_model.embed_tokens.weight"]


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
        "qkv", "attention", "o_proj",
    ]

    var q_band: ScratchPhase["qkv", "o_proj"]
    var q: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * Self.q_rows,
    ]

    var kv_band: ScratchPhase["qkv", "attention"]
    var kv: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * Self.kv_rows * 2,
    ]

    var partials_band: ScratchPhase["attention", "attention"]
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
        "prep", "flash", "merge",
    ]

    var q_band: ScratchPhase["prep", "flash"]
    var q: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * Self.q_rows,
    ]

    var kv_band: ScratchPhase["prep", "prep"]
    var kv: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * Self.k_rows * 2,
    ]

    var partials_band: ScratchPhase["flash", "merge"]
    var partials: ScratchBuffer[
        Float32, Self.PARTIAL_SLOTS * Self.PARTIAL_STRIDE,
    ]

    var q_local_band: ScratchPhase["merge", "merge"]
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
        "dense_gate_up", "router_select", "moe_setup",
        "phase1", "phase2", "dense_down_post",
    ]

    var ffn_gate_band: ScratchPhase["dense_gate_up", "dense_down_post"]
    var ffn_gate: ScratchBuffer[
        BFloat16,
        C.SLIDING_WINDOW * Self.intermediate_per_rank,
    ]

    var ffn_up_band: ScratchPhase["dense_gate_up", "dense_gate_up"]
    var ffn_up: ScratchBuffer[
        BFloat16,
        C.SLIDING_WINDOW * Self.intermediate_per_rank,
    ]

    var router_workspace: ScratchPhase[
        "router_select", "router_select",
    ]
    var moe_router_scaled: ScratchBuffer[
        Float32, Self.max_worker_count * C.HIDDEN,
    ]

    var router_cands: ScratchPhase["router_select", "router_select"]
    var moe_cands: ScratchBuffer[
        RouterCandidate,
        min(Self.max_worker_count, Self.experts_per_rank)
        * C.SLIDING_WINDOW * C.TOP_K,
    ]

    var router_products: ScratchPhase["router_select", "moe_setup"]
    var moe_route_idx: ScratchBuffer[
        Int32, C.SLIDING_WINDOW * C.TOP_K,
    ]
    var moe_route_w: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * C.TOP_K,
    ]

    var expert_input: ScratchPhase["moe_setup", "phase1"]
    var moe_x_normed: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.HIDDEN,
    ]

    var schedule_products: ScratchPhase[
        "moe_setup", "phase2",
    ]
    var moe_expert_offset: ScratchBuffer[
        Int32, Self.experts_per_rank + 1,
    ]
    var moe_routes: ScratchBuffer[SparseRoute, C.SLIDING_WINDOW * C.TOP_K]

    var hidden_bucket: ScratchPhase["phase1", "phase2"]
    var moe_hidden_bucket: ScratchBuffer[
        BFloat16,
        C.SLIDING_WINDOW * C.TOP_K * C.MOE_INTERMEDIATE,
    ]

    var phase1_workspace: ScratchPhase[
        "phase1", "phase1",
    ]
    var moe_gate_scratch: ScratchBuffer[
        Float32,
        Self.max_worker_count * Self.PHASE1_MR * 2 * Self.PHASE1_TILE_J,
    ]

    var phase2_accum: ScratchPhase["phase2", "phase2"]
    var moe_accum: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * C.HIDDEN,
    ]

    var dense_band: ScratchPhase["dense_down_post", "dense_down_post"]
    var ffn_dense_out: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.HIDDEN,
    ]


@fieldwise_init
struct Gemma4HeadScratch[degree: Int](
    ScratchIsland, Copyable, ImplicitlyCopyable
):
    comptime PHASES = ScratchPhaseOrder[
        "logits",
    ]
    comptime vocab_per_rank = C.VOCAB_SIZE // Self.degree

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


def dispatch_sliding_attention_qkv[
    P: BurstThreadPool, Profile: Bool, N: Int, //, degree: Int, max_seq_len: Int, max_worker_count: Int = 128,
](
    layout: Gemma4Layout[degree, max_seq_len],
    ctx: BindContext[degree],
    base_pos: Int,
    seq_len: Int,
    layer_idx: Int,
    mut scratch: Gemma4ScratchPool[degree, max_worker_count],
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    """Sliding-window attention for `seq_len` Q-tokens starting at
    `base_pos`. `seq_len == 1` uses kv-range-parallel decode flash plus
    cross-worker merge; `seq_len > 1` uses Q-range-parallel prefill
    flash that writes finalized output directly (no merge). Caller must
    keep `seq_len <= SLIDING_WINDOW` so the chunk's KV writes land in
    one half of the 2W ring and the prior chunk stays addressable."""
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
    comptime cache_size = Gemma4StateShapes[degree, max_seq_len].SLIDING_CACHE
    comptime Island = Gemma4SlidingScratch[degree, max_worker_count]

    debug_assert(
        seq_len <= C.SLIDING_WINDOW,
        "sliding attention chunk exceeds SLIDING_WINDOW; the 2W ring "
        "only protects W tokens of prior context.",
    )

    var attn_ctx = ctx.with_layer(layout.sliding.base(ctx.arena_bases[0], layer_idx))
    var attn = layout.sliding.proto.attn

    var q_outs = scratch.binding[Island, "q"](ctx)
    var k_outs = scratch.binding[Island, "kv"](ctx)
    var v_outs = k_outs.shifted(seq_len * kv_rows)
    var xs = layout.activations.x_residual.state_binding(ctx)

    dispatch_gemm_chained_qkv[
        q_rows=q_rows, kv_rows=kv_rows, cols=C.HIDDEN, tp=degree,
        max_worker_count=max_worker_count,
    ](xs,
      attn.q_proj.binding(attn_ctx),
      attn.k_proj.binding(attn_ctx),
      attn.v_proj.binding(attn_ctx),
      q_outs, k_outs, v_outs, seq_len, pools, prof)

    dispatch_rms_norm_qkv_heads[
        head_dim=head_dim, sqrt_n=sqrt_hd, n_eps=hd_eps,
        num_q=num_q_heads, num_kv=num_kv_heads, tp=degree,
        max_worker_count=max_worker_count,
    ](q_outs, q_outs, k_outs, k_outs, v_outs, v_outs,
      attn.q_norm.binding(attn_ctx),
      attn.k_norm.binding(attn_ctx),
      seq_len, pools, prof)

    var kv_lb = layout.sliding_kv.base(ctx.arena_bases[0], layer_idx)
    var k_kv = layout.sliding_kv.proto.k.binding(kv_lb, ctx.arena_bases)
    var v_kv = layout.sliding_kv.proto.v.binding(kv_lb, ctx.arena_bases)

    dispatch_rope_cache_write[
        half=rope_half, pair_stride=head_dim // 2,
        num_q=num_q_heads, num_kv=num_kv_heads,
        head_dim=head_dim, kv_cache_stride=kv_cols,
        slot_mask=cache_size - 1, cache_degree=1, tp=degree,
        max_worker_count=max_worker_count,
    ](q_outs, k_outs, v_outs,
      k_kv, v_kv,
      layout.sliding_rope.cos.state_binding(ctx),
      layout.sliding_rope.sin.state_binding(ctx),
      base_pos, seq_len, pools, prof)

    var partials = scratch.binding[Island, "partials"](ctx)

    dispatch_sliding_attention[
        head_dim=head_dim, num_q=num_q_heads,
        gqa_ratio=num_q_heads // num_kv_heads,
        kv_stride=kv_cols, window=C.SLIDING_WINDOW,
        cache_size=cache_size,
        partial_stride=Island.PARTIAL_STRIDE, tp=degree,
        max_worker_count=max_worker_count,
    ](q_outs, k_kv, v_kv, q_outs, partials,
      base_pos, seq_len, pools, prof)

    dispatch_gemm[
        rows=C.HIDDEN, cols=q_rows, tp=degree,
        max_worker_count=max_worker_count,
    ](
        q_outs,
        attn.o_proj.binding(attn_ctx),
        xs, seq_len, pools, prof)


def dispatch_full_attention_qkv[
    P: BurstThreadPool, Profile: Bool, N: Int, //, degree: Int, max_seq_len: Int, max_worker_count: Int = 128,
](
    layout: Gemma4Layout[degree, max_seq_len],
    ctx: BindContext[degree],
    base_pos: Int,
    seq_len: Int,
    layer_idx: Int,
    mut scratch: Gemma4ScratchPool[degree, max_worker_count],
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    """Full attention for `seq_len` Q-tokens starting at `base_pos`. The
    unified attention dispatcher routes decode vs prefill internally
    based on seq_len; this orchestrator does not branch."""
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
    var v_outs = k_outs.shifted(seq_len * k_rows)
    var xs = layout.activations.x_residual.state_binding(ctx)

    dispatch_gemm[
        rows=q_rows, cols=C.HIDDEN, tp=degree,
        max_worker_count=max_worker_count,
    ](
        xs, attn.q_proj.binding(attn_ctx), q_outs, seq_len, pools, prof)
    dispatch_gemm[
        rows=k_rows, cols=C.HIDDEN, tp=degree,
        max_worker_count=max_worker_count,
    ](
        xs, attn.k_proj.binding(attn_ctx), k_outs, seq_len, pools, prof)

    # Full attention has no v_proj; v reads from k's pre-norm buffer (chain runs V→Q→K).
    dispatch_rms_norm_qkv_heads[
        head_dim=head_dim, sqrt_n=sqrt_hd, n_eps=hd_eps,
        num_q=num_q_heads, num_kv=num_kv_heads, tp=degree,
        max_worker_count=max_worker_count,
    ](q_outs, q_outs, k_outs, k_outs, k_outs, v_outs,
      attn.q_norm.binding(attn_ctx),
      attn.k_norm.binding(attn_ctx),
      seq_len, pools, prof)

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
      layout.full_rope.cos.state_binding(ctx),
      layout.full_rope.sin.state_binding(ctx),
      base_pos, seq_len, pools, prof)

    var q_local_outs = scratch.binding[Island, "q_local"](ctx)
    var partials = scratch.binding[Island, "partials"](ctx)

    dispatch_full_attention[
        head_dim=head_dim, num_q=num_q_heads,
        local_num_q=local_num_q_heads,
        gqa_ratio=C.NUM_HEADS // C.NUM_KV_HEADS_FULL,
        kv_stride=kv_cols,
        partial_stride=Island.PARTIAL_STRIDE, tp=degree,
        max_worker_count=max_worker_count,
    ](q_outs, k_kv, v_kv, q_local_outs, partials,
      base_pos, seq_len, pools, prof)

    dispatch_gemm[
        rows=C.HIDDEN, cols=local_q_rows, tp=degree,
        max_worker_count=max_worker_count,
    ](
        q_local_outs,
        attn.o_proj.binding(attn_ctx),
        xs, seq_len, pools, prof)


def dispatch_moe[
    P: BurstThreadPool, Profile: Bool, N: Int, //, degree: Int, max_worker_count: Int = 128,
](
    body: BodyRefs[degree],
    ctx: BindContext[degree],
    x_input: BF16Bind[degree],
    moe_out: BF16Bind[degree],
    seq_len: Int,
    mut scratch: Gemma4ScratchPool[degree, max_worker_count],
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    comptime experts_per_rank = C.NUM_EXPERTS // degree
    comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
    comptime n_eps = C.HIDDEN * C.RMS_NORM_EPS
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

    var nws = dispatch_router_expert[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
        experts_per_rank=experts_per_rank,
        top_k=C.TOP_K, tp=degree,
        max_worker_count=max_worker_count,
    ](x_input,
      body.router_proj.binding(ctx),
      body.router_scale.binding(ctx),
      router_scaled, cands, seq_len, pools, prof)

    merge_router_candidates_expert[degree, C.TOP_K](
        cands, nws, seq_len, per_expert_scale_ptr, route_idx, route_w)

    dispatch_rms_norm[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps, tp=degree,
        max_worker_count=max_worker_count,
    ](x_input, x_normed,
      body.pre_ffn_norm_2.binding(ctx), seq_len, pools, prof)

    build_expert_schedules[degree, experts_per_rank, C.TOP_K](
        route_idx, route_w, expert_offset, routes, seq_len)

    dispatch_phase1_gate_up[
        hidden=C.HIDDEN, gate_up_fused=C.MOE_GATE_UP_FUSED,
        intermediate=C.MOE_INTERMEDIATE,
        experts_per_rank=experts_per_rank, tp=degree,
        tile_j=Ffn.PHASE1_TILE_J, MR=Ffn.PHASE1_MR,
        max_worker_count=max_worker_count,
    ](x_normed, expert_offset, routes,
      body.experts_gate_up.binding(ctx),
      gate_scratch, hidden_bucket, pools, prof)

    dispatch_phase2_down[
        hidden=C.HIDDEN, intermediate=C.MOE_INTERMEDIATE,
        experts_per_rank=experts_per_rank, tp=degree,
        max_worker_count=max_worker_count,
    ](expert_offset, routes, hidden_bucket,
      body.experts_down.binding(ctx),
      moe_accum, moe_out, seq_len, pools, prof)

    dispatch_allreduce_inplace[
        BF16, degree, max_worker_count=max_worker_count,
    ](moe_out, seq_len * C.HIDDEN, pools, prof)


def dispatch_ffn[
    P: BurstThreadPool, Profile: Bool, N: Int, //, degree: Int, max_worker_count: Int = 128,
](
    body: BodyRefs[degree],
    ctx: BindContext[degree],
    x_main: BF16Bind[degree],
    x_residual: BF16Bind[degree],
    seq_len: Int,
    mut scratch: Gemma4ScratchPool[degree, max_worker_count],
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
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
      body.pre_ffn_norm.binding(ctx), seq_len, pools, prof)

    dispatch_gemm[
        rows=intermediate_per_rank, cols=C.HIDDEN, tp=degree,
        max_worker_count=max_worker_count,
    ](x_residual, body.gate_proj.binding(ctx), gate, seq_len, pools, prof)

    dispatch_gemm[
        rows=intermediate_per_rank, cols=C.HIDDEN, tp=degree,
        max_worker_count=max_worker_count,
    ](x_residual, body.up_proj.binding(ctx), up, seq_len, pools, prof)

    dispatch_gelu_gate_up[
        intermediate=intermediate_per_rank, tp=degree,
        max_worker_count=max_worker_count,
    ](gate, up, gate, seq_len, pools, prof)

    dispatch_moe[degree=degree, max_worker_count=max_worker_count](
        body, ctx, x_main, x_residual, seq_len, scratch, pools, prof)

    dispatch_gemm[
        rows=C.HIDDEN, cols=intermediate_per_rank, tp=degree,
        max_worker_count=max_worker_count,
    ](gate, body.down_proj.binding(ctx), dense_out, seq_len, pools, prof)

    dispatch_allreduce_inplace[
        BF16, degree, max_worker_count=max_worker_count,
    ](dense_out, seq_len * C.HIDDEN, pools, prof)

    dispatch_rms_norm[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps, tp=degree,
        max_worker_count=max_worker_count,
    ](dense_out, dense_out,
      body.post_ffn_norm_1.binding(ctx), seq_len, pools, prof)

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

    def model_init(mut self):
        ref layout = self.layout
        comptime width = simd_width_of[DType.float32]()

        prime_fp_environment[Self.degree, Self.max_worker_count](self.pools)

        comptime inv_sqrt_hidden = 1.0 / sqrt[DType.float32, 1](C.HIDDEN)
        for rank in range(Self.degree):
            var arena_base = self.arena_bases[rank]
            for i in range(C.NUM_LAYERS):
                var entry = LAYER_SCHEDULE[i]
                var p: UnsafePointer[BFloat16, MutAnyOrigin]
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
        base_pos: Int,
    ) -> TemporalLogitsView[C.VOCAB_SIZE, Self.degree]:
        ref layout = self.layout
        comptime shard_rows = Gemma4TailShapes[Self.degree].Embed.DATA_N
        comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
        comptime n_eps = C.HIDDEN * C.RMS_NORM_EPS
        comptime embed_scale = Float64(sqrt[DType.float32, 1](C.HIDDEN)
            .cast[DType.bfloat16]().cast[DType.float32]())
        comptime vocab_per_rank = C.VOCAB_SIZE // Self.degree

        var wall_t0 = perf_counter_ns()
        var total_len = len(token_ids)
        debug_assert(total_len > 0, "forward called with empty token_ids")
        debug_assert(
            base_pos + total_len <= Self.max_seq_len,
            "forward exceeds max_seq_len",
        )

        var ctx = BindContext[Self.degree](
            arena_bases=self.arena_bases, layer_base=0)
        var tail_ctx = ctx.with_layer(
            layout.tail.base(self.arena_bases[0], 0))

        var x_main_ranks = layout.activations.x_main.state_binding(ctx)
        var x_res_ranks = layout.activations.x_residual.state_binding(ctx)
        var logits = self.scratch.binding[
            Gemma4HeadScratch[Self.degree], "logits",
        ](ctx)

        var consumed = 0
        var pos = base_pos
        while consumed < total_len:
            var remaining = total_len - consumed
            var chunk_len = remaining if remaining < C.SLIDING_WINDOW else C.SLIDING_WINDOW
            var is_last = (consumed + chunk_len == total_len)

            var chunk = Span[Int32, tok_origin](
                ptr=token_ids.unsafe_ptr() + consumed,
                length=chunk_len)

            dispatch_embed_lookup[
                hidden=C.HIDDEN, scale=embed_scale, shard_rows=shard_rows,
                tp=Self.degree, max_worker_count=Self.max_worker_count,
            ](chunk,
              layout.tail.proto.embed.binding(tail_ctx),
              x_main_ranks, chunk_len, self.pools, self.profiler)
            dispatch_allreduce_inplace[
                BF16, Self.degree, max_worker_count=Self.max_worker_count,
            ](x_main_ranks, chunk_len * C.HIDDEN, self.pools, self.profiler)

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
                  chunk_len, self.pools, self.profiler)

                if entry.kind == LayerKind.FULL:
                    dispatch_full_attention_qkv[
                        degree=Self.degree,
                        max_seq_len=Self.max_seq_len,
                        max_worker_count=Self.max_worker_count,
                    ](
                        layout, ctx, pos, chunk_len, entry.local_idx,
                        self.scratch, self.pools, self.profiler)
                else:
                    dispatch_sliding_attention_qkv[
                        degree=Self.degree,
                        max_seq_len=Self.max_seq_len,
                        max_worker_count=Self.max_worker_count,
                    ](
                        layout, ctx, pos, chunk_len, entry.local_idx,
                        self.scratch, self.pools, self.profiler)

                dispatch_allreduce_inplace[
                    BF16, Self.degree, max_worker_count=Self.max_worker_count,
                ](x_res_ranks, chunk_len * C.HIDDEN, self.pools, self.profiler)

                fused_norm_residual_add[
                    hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
                    tp=Self.degree, max_worker_count=Self.max_worker_count,
                ](x_res_ranks, x_main_ranks, x_main_ranks,
                  body.post_attn_norm.binding(layer_ctx),
                  chunk_len, self.pools, self.profiler)

                dispatch_ffn[
                    degree=Self.degree, max_worker_count=Self.max_worker_count,
                ](
                    body, layer_ctx, x_main_ranks, x_res_ranks, chunk_len,
                    self.scratch, self.pools, self.profiler)

            if is_last:
                var x_last = x_main_ranks.shifted((chunk_len - 1) * C.HIDDEN)

                dispatch_rms_norm[
                    hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
                    tp=Self.degree, max_worker_count=Self.max_worker_count,
                ](x_last, x_last,
                  layout.tail.proto.final_norm.binding(tail_ctx),
                  1, self.pools, self.profiler)

                dispatch_gemv_softcap[
                    rows=vocab_per_rank, cols=C.HIDDEN, tp=Self.degree,
                    cap=C.LOGIT_SOFTCAP,
                    max_worker_count=Self.max_worker_count,
                ](x_last, layout.tail.proto.embed.binding(tail_ctx), logits, self.pools, self.profiler)

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
        model.model_init()
        return model^
