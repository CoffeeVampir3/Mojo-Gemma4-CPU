from std.pathlib import Path
from std.memory import Span, UnsafePointer
from std.sys.info import simd_width_of
from std.time import perf_counter_ns
from simd_math.ops import sqrt

from numa import NumaArena, NumaTopology
from threading import BurstPool
from threading.threading_traits import BurstThreadPool
from kernels.helpers import RankView, Binding, prime_fp_environment
from kernels.reductions import dispatch_allreduce_inplace
from kernels.embedding import dispatch_embed_lookup
from kernels.rmsnorm import dispatch_rms_norm, dispatch_rms_norm_qkv_heads
from kernels.rmsnorm import fused_norm_residual_add
from kernels.gemv import dispatch_gemv_softcap
from kernels.gemm import dispatch_gemm, dispatch_gemm_cols, dispatch_gemm_chained_qkv
from kernels.rope import dispatch_rope_cache_write
from kernels.attention_ops import flash_partial_stride
from kernels.attention_dispatch_kernels import (
    dispatch_sliding_attention, dispatch_full_attention,
)
from kernels.logsum_merge import MergeSegment
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
    ScratchBuffer, ScratchIsland, ScratchPhase, ScratchPhaseOrder, ScaleClass,
    TemporalLogitsView, TemporalScratchPool, ScratchPlan,
    derive_scratch_plan, aggregate_scratch_peak,
)

from modeling.model_spec import (
    BF16, F32,
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
    Repeated, ArenaLayout, BF16Bind,
)
from modeling.slot import (
    Slot, SlotGroup, BindContext, stamp_offsets, emit_descs,
)
from modeling.loader import discover_shards, load_weights_from_descs


comptime C = Gemma4BaseConfig
comptime MAX_WORKERS = 128


@always_inline
def degree_contracts_ok(degree: Int) -> Bool:
    """Runtime divisibility checks; the model is allocated once at the deployed
    degree (= NUMA-domain count)."""
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
    comptime GateUp      = TensorRowSharded[C.INTERMEDIATE, C.HIDDEN]
    comptime Down        = TensorColumnSharded[C.HIDDEN, C.INTERMEDIATE]
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
    """Per-token state tensors. KV caches shard with the runtime degree; rope
    tables are replicated per rank (most-frequent reader is every attention
    worker on every token). The sliding KV ring is sized at 2 * SLIDING_WINDOW."""
    comptime SLIDING_CACHE = 2 * C.SLIDING_WINDOW
    comptime SlidingKV   = TensorColumnSharded[Self.SLIDING_CACHE, C.KV_DIM_SLIDING]
    comptime FullKV      = ContextRowSharded[Self.max_seq_len, C.KV_DIM_FULL]
    comptime SlidingRope = Replicated[Self.max_seq_len, C.ROPE_HALF_SLIDING]
    comptime FullRope    = Replicated[Self.max_seq_len, C.ROPE_HALF_FULL]


struct Gemma4TailShapes:
    comptime FinalNorm = Replicated[C.HIDDEN, 1]
    comptime Embed = VocabularyRowSharded[C.VOCAB_SIZE, C.HIDDEN]


struct SlidingAttnRefs(Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4Shapes
    var q_proj: Slot[BF16, Self.S.SlidingQ,  "self_attn.q_proj.weight"]
    var k_proj: Slot[BF16, Self.S.SlidingKV, "self_attn.k_proj.weight"]
    var v_proj: Slot[BF16, Self.S.SlidingKV, "self_attn.v_proj.weight"]
    var o_proj: Slot[BF16, Self.S.SlidingO,  "self_attn.o_proj.weight"]
    var q_norm: Slot[BF16, Shape[C.HEAD_DIM_SLIDING, 1], "self_attn.q_norm.weight"]
    var k_norm: Slot[BF16, Shape[C.HEAD_DIM_SLIDING, 1], "self_attn.k_norm.weight"]


struct FullAttnRefs(Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4Shapes
    var q_proj: Slot[BF16, Self.S.FullQ, "self_attn.q_proj.weight"]
    var k_proj: Slot[BF16, Self.S.FullK, "self_attn.k_proj.weight"]
    var o_proj: Slot[BF16, Self.S.FullO, "self_attn.o_proj.weight"]
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
    var gate_proj:       Slot[BF16, Self.S.GateUp,              "mlp.gate_proj.weight"]
    var up_proj:         Slot[BF16, Self.S.GateUp,              "mlp.up_proj.weight"]
    var down_proj:       Slot[BF16, Self.S.Down,                "mlp.down_proj.weight"]
    var router_proj:     Slot[BF16, Self.S.RouterProj,          "router.proj.weight"]
    var router_scale:    Slot[BF16, Shape[C.HIDDEN, 1],         "router.scale"]
    var router_pes:      Slot[BF16, Shape[C.NUM_EXPERTS, 1],    "router.per_expert_scale"]
    var experts_gate_up: Slot[BF16, Self.S.ExpertsGateUp,       "experts.gate_up_proj"]
    var experts_down:    Slot[BF16, Self.S.ExpertsDown,         "experts.down_proj"]
    var layer_scalar:    Slot[BF16, Shape[1, 1],                "layer_scalar"]


struct SlidingLayerRefs(Copyable, ImplicitlyCopyable, SlotGroup):
    var attn: SlidingAttnRefs
    var body: BodyRefs


struct FullLayerRefs(Copyable, ImplicitlyCopyable, SlotGroup):
    var attn: FullAttnRefs
    var body: BodyRefs


struct SlidingKVSlots[max_seq_len: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4StateShapes[Self.max_seq_len]
    var k: Slot[BF16, Self.S.SlidingKV]
    var v: Slot[BF16, Self.S.SlidingKV]


struct FullKVSlots[max_seq_len: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4StateShapes[Self.max_seq_len]
    var k: Slot[BF16, Self.S.FullKV]
    var v: Slot[BF16, Self.S.FullKV]


struct RopeSlots[half: Int, max_seq_len: Int](Copyable, ImplicitlyCopyable, SlotGroup):
    var cos: Slot[F32, Replicated[Self.max_seq_len, Self.half]]
    var sin: Slot[F32, Replicated[Self.max_seq_len, Self.half]]


struct ActivationSlots(Copyable, ImplicitlyCopyable, SlotGroup):
    var x_main:     Slot[BF16, Shape[C.SLIDING_WINDOW, C.HIDDEN]]
    var x_residual: Slot[BF16, Shape[C.SLIDING_WINDOW, C.HIDDEN]]


struct TailRefs(Copyable, ImplicitlyCopyable, SlotGroup):
    comptime S = Gemma4TailShapes
    var final_norm: Slot[BF16, Self.S.FinalNorm, "model.language_model.norm.weight"]
    var embed:      Slot[BF16, Self.S.Embed, "model.language_model.embed_tokens.weight"]


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


# ───────────────────────────── scratch islands (degree-free) ────────────────
comptime SLIDING_NUM_Q_MAX = C.Q_DIM_SLIDING // C.HEAD_DIM_SLIDING
comptime SLIDING_PARTIAL_STRIDE_MAX = flash_partial_stride(
    SLIDING_NUM_Q_MAX, C.HEAD_DIM_SLIDING)
comptime FULL_NUM_Q = C.Q_DIM_FULL // C.HEAD_DIM_FULL
comptime FULL_PARTIAL_STRIDE = flash_partial_stride(FULL_NUM_Q, C.HEAD_DIM_FULL)


@fieldwise_init
struct Gemma4SlidingScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
    comptime PHASES = ScratchPhaseOrder["qkv", "attention", "o_proj"]

    var q_band: ScratchPhase["qkv", "o_proj"]
    var q: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.Q_DIM_SLIDING, ScaleClass.PER_DEGREE,
    ]

    var kv_band: ScratchPhase["qkv", "attention"]
    var kv: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.KV_DIM_SLIDING * 2, ScaleClass.PER_DEGREE,
    ]

    var partials_band: ScratchPhase["attention", "attention"]
    var partials: ScratchBuffer[
        Float32, SLIDING_PARTIAL_STRIDE_MAX, ScaleClass.PER_WORKER,
    ]


@fieldwise_init
struct Gemma4FullScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
    comptime PHASES = ScratchPhaseOrder["prep", "flash", "merge"]

    var q_band: ScratchPhase["prep", "flash"]
    var q: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.Q_DIM_FULL, ScaleClass.FIXED,
    ]

    var kv_band: ScratchPhase["prep", "prep"]
    var kv: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.KV_DIM_FULL * 2, ScaleClass.FIXED,
    ]

    var partials_band: ScratchPhase["flash", "merge"]
    var partials: ScratchBuffer[
        Float32, C.SLIDING_WINDOW * FULL_PARTIAL_STRIDE, ScaleClass.FIXED,
    ]

    var q_local_band: ScratchPhase["merge", "merge"]
    var q_local: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.Q_DIM_FULL, ScaleClass.PER_DEGREE,
    ]

    var merge_segments_band: ScratchPhase["merge", "merge"]
    var merge_segments: ScratchBuffer[
        MergeSegment, 1, ScaleClass.PER_WORKER_PER_DEGREE,
    ]


@fieldwise_init
struct Gemma4FfnMoeScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
    comptime PHASE1_TILE_J = 64
    comptime PHASE1_MR = 4

    comptime PHASES = ScratchPhaseOrder[
        "dense_gate_up", "router_select", "moe_setup",
        "phase1", "phase2", "dense_down_post",
    ]

    var ffn_gate_band: ScratchPhase["dense_gate_up", "dense_down_post"]
    var ffn_gate: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.INTERMEDIATE, ScaleClass.PER_DEGREE,
    ]

    var ffn_up_band: ScratchPhase["dense_gate_up", "dense_gate_up"]
    var ffn_up: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.INTERMEDIATE, ScaleClass.PER_DEGREE,
    ]

    var router_workspace: ScratchPhase["router_select", "router_select"]
    var moe_router_scaled: ScratchBuffer[Float32, C.HIDDEN, ScaleClass.PER_WORKER]

    var router_cands: ScratchPhase["router_select", "router_select"]
    var moe_cands: ScratchBuffer[
        RouterCandidate, C.SLIDING_WINDOW * C.TOP_K, ScaleClass.PER_WORKER,
    ]

    var router_products: ScratchPhase["router_select", "moe_setup"]
    var moe_route_idx: ScratchBuffer[Int32, C.SLIDING_WINDOW * C.TOP_K, ScaleClass.FIXED]
    var moe_route_w: ScratchBuffer[Float32, C.SLIDING_WINDOW * C.TOP_K, ScaleClass.FIXED]

    var expert_input: ScratchPhase["moe_setup", "phase1"]
    var moe_x_normed: ScratchBuffer[BFloat16, C.SLIDING_WINDOW * C.HIDDEN, ScaleClass.FIXED]

    var schedule_products: ScratchPhase["moe_setup", "phase2"]
    var moe_expert_offset: ScratchBuffer[Int32, C.NUM_EXPERTS + 1, ScaleClass.FIXED]
    var moe_routes: ScratchBuffer[SparseRoute, C.SLIDING_WINDOW * C.TOP_K, ScaleClass.FIXED]

    var hidden_bucket: ScratchPhase["phase1", "phase2"]
    var moe_hidden_bucket: ScratchBuffer[
        BFloat16, C.SLIDING_WINDOW * C.TOP_K * C.MOE_INTERMEDIATE, ScaleClass.FIXED,
    ]

    var phase1_workspace: ScratchPhase["phase1", "phase1"]
    var moe_gate_scratch: ScratchBuffer[
        Float32, Self.PHASE1_MR * 2 * Self.PHASE1_TILE_J, ScaleClass.PER_WORKER,
    ]

    var phase2_accum: ScratchPhase["phase2", "phase2"]
    var moe_accum: ScratchBuffer[Float32, C.SLIDING_WINDOW * C.HIDDEN, ScaleClass.FIXED]

    var dense_band: ScratchPhase["dense_down_post", "dense_down_post"]
    var ffn_dense_out: ScratchBuffer[BFloat16, C.SLIDING_WINDOW * C.HIDDEN, ScaleClass.FIXED]


@fieldwise_init
struct Gemma4HeadScratch(ScratchIsland, Copyable, ImplicitlyCopyable):
    comptime PHASES = ScratchPhaseOrder["logits"]

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
    debug_assert(degree_contracts_ok(degree), "degree does not divide model dims")
    debug_assert(max_workers > 0, "max_workers must be positive")
    debug_assert(
        max_workers <= C.SLIDING_WINDOW,
        "full-attention partials are sized for max_workers <= SLIDING_WINDOW",
    )

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


def dispatch_sliding_attention_qkv[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_seq_len: Int, max_worker_count: Int = 128,
](
    layout: Gemma4Layout[max_seq_len],
    ctx: BindContext[o],
    base_pos: Int,
    seq_len: Int,
    layer_idx: Int,
    scratch: TemporalScratchPool,
    plan: ScratchPlan,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    """Sliding-window attention for `seq_len` Q-tokens starting at `base_pos`.
    Caller keeps `seq_len <= SLIDING_WINDOW` so the chunk's KV writes land in
    one half of the 2W ring and the prior chunk stays addressable."""
    var degree = ctx.degree()
    comptime head_dim = C.HEAD_DIM_SLIDING
    comptime gqa_ratio = C.NUM_HEADS // C.NUM_KV_HEADS_SLIDING
    comptime sqrt_hd = sqrt[DType.float32, 1](head_dim)
    comptime hd_eps = head_dim * C.RMS_NORM_EPS
    comptime rope_half = C.ROPE_HALF_SLIDING
    comptime cache_size = 2 * C.SLIDING_WINDOW
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

    var attn_ctx = ctx.with_layer(layout.sliding.base(ctx.view.bases[0], layer_idx))
    var attn = layout.sliding.proto.attn

    var q_outs = scratch.binding[Gemma4SlidingScratch, "q"](ctx, plan)
    var k_outs = scratch.binding[Gemma4SlidingScratch, "kv"](ctx, plan)
    var v_outs = k_outs.shifted(seq_len * kv_rows)
    var xs = layout.activations.x_residual.state_binding(ctx)

    dispatch_gemm_chained_qkv[
        cols=C.HIDDEN, max_worker_count=max_worker_count,
    ](xs,
      attn.q_proj.binding(attn_ctx),
      attn.k_proj.binding(attn_ctx),
      attn.v_proj.binding(attn_ctx),
      q_outs, k_outs, v_outs, q_rows, kv_rows, seq_len, pools, prof)

    dispatch_rms_norm_qkv_heads[
        head_dim=head_dim, sqrt_n=sqrt_hd, n_eps=hd_eps,
        max_worker_count=max_worker_count,
    ](q_outs, q_outs, k_outs, k_outs, v_outs, v_outs,
      attn.q_norm.binding(attn_ctx),
      attn.k_norm.binding(attn_ctx),
      num_q_heads, num_kv_heads, seq_len, pools, prof)

    var kv_lb = layout.sliding_kv.base(ctx.view.bases[0], layer_idx)
    var k_kv = layout.sliding_kv.proto.k.binding(kv_lb, ctx.view)
    var v_kv = layout.sliding_kv.proto.v.binding(kv_lb, ctx.view)

    dispatch_rope_cache_write[
        half=rope_half, pair_stride=head_dim // 2,
        head_dim=head_dim, slot_mask=cache_size - 1,
        max_worker_count=max_worker_count,
    ](q_outs, k_outs, v_outs,
      k_kv, v_kv,
      layout.sliding_rope.cos.state_binding(ctx),
      layout.sliding_rope.sin.state_binding(ctx),
      num_q_heads, num_kv_heads, 1, base_pos, seq_len, pools, prof)

    var partials = scratch.binding[Gemma4SlidingScratch, "partials"](ctx, plan)

    dispatch_sliding_attention[
        head_dim=head_dim, max_q=max_q, gqa_ratio=gqa_ratio,
        window=C.SLIDING_WINDOW, cache_size=cache_size,
        max_worker_count=max_worker_count,
    ](q_outs, k_kv, v_kv, q_outs, partials,
      num_q_heads, partial_stride, kv_rows, base_pos, seq_len, pools, prof)

    dispatch_gemm_cols[
        rows=C.HIDDEN, max_worker_count=max_worker_count,
    ](
        q_outs,
        attn.o_proj.binding(attn_ctx),
        xs, q_rows, seq_len, pools, prof)


def dispatch_full_attention_qkv[
    P: BurstThreadPool, Profile: Bool, N: Int, o: ImmutOrigin, //,
    max_seq_len: Int, max_worker_count: Int = 128,
](
    layout: Gemma4Layout[max_seq_len],
    ctx: BindContext[o],
    base_pos: Int,
    seq_len: Int,
    layer_idx: Int,
    scratch: TemporalScratchPool,
    plan: ScratchPlan,
    mut pools: List[P],
    mut prof: Profiler[Profile, N],
):
    """Full attention. Q/K are replicated (head counts comptime); the context
    shard `degree` and the column-sharded o_proj are runtime."""
    var degree = ctx.degree()
    comptime head_dim = C.HEAD_DIM_FULL
    comptime q_rows = C.Q_DIM_FULL
    comptime k_rows = C.KV_DIM_FULL
    comptime num_q_heads = q_rows // head_dim
    comptime num_kv_heads = k_rows // head_dim
    comptime gqa_ratio = C.NUM_HEADS // C.NUM_KV_HEADS_FULL
    comptime sqrt_hd = sqrt[DType.float32, 1](head_dim)
    comptime hd_eps = head_dim * C.RMS_NORM_EPS
    comptime rope_half = C.ROPE_HALF_FULL
    comptime pair_stride = head_dim // 2
    comptime partial_stride = FULL_PARTIAL_STRIDE
    var local_q_rows = Gemma4Shapes.FullO.data_m(degree)
    var local_num_q_heads = local_q_rows // head_dim

    var attn_ctx = ctx.with_layer(layout.full.base(ctx.view.bases[0], layer_idx))
    var attn = layout.full.proto.attn

    var q_outs = scratch.binding[Gemma4FullScratch, "q"](ctx, plan)
    var k_outs = scratch.binding[Gemma4FullScratch, "kv"](ctx, plan)
    var v_outs = k_outs.shifted(seq_len * k_rows)
    var xs = layout.activations.x_residual.state_binding(ctx)

    dispatch_gemm[
        cols=C.HIDDEN, max_worker_count=max_worker_count,
    ](xs, attn.q_proj.binding(attn_ctx), q_outs, q_rows, seq_len, pools, prof)
    dispatch_gemm[
        cols=C.HIDDEN, max_worker_count=max_worker_count,
    ](xs, attn.k_proj.binding(attn_ctx), k_outs, k_rows, seq_len, pools, prof)

    # Full attention has no v_proj; v reads from k's pre-norm buffer (chain V→Q→K).
    dispatch_rms_norm_qkv_heads[
        head_dim=head_dim, sqrt_n=sqrt_hd, n_eps=hd_eps,
        max_worker_count=max_worker_count,
    ](q_outs, q_outs, k_outs, k_outs, k_outs, v_outs,
      attn.q_norm.binding(attn_ctx),
      attn.k_norm.binding(attn_ctx),
      num_q_heads, num_kv_heads, seq_len, pools, prof)

    var kv_lb = layout.full_kv.base(ctx.view.bases[0], layer_idx)
    var k_kv = layout.full_kv.proto.k.binding(kv_lb, ctx.view)
    var v_kv = layout.full_kv.proto.v.binding(kv_lb, ctx.view)

    dispatch_rope_cache_write[
        half=rope_half, pair_stride=pair_stride,
        head_dim=head_dim, slot_mask=-1,
        max_worker_count=max_worker_count,
    ](q_outs, k_outs, v_outs,
      k_kv, v_kv,
      layout.full_rope.cos.state_binding(ctx),
      layout.full_rope.sin.state_binding(ctx),
      num_q_heads, num_kv_heads, degree, base_pos, seq_len, pools, prof)

    var q_local_outs = scratch.binding[Gemma4FullScratch, "q_local"](ctx, plan)
    var partials = scratch.binding[Gemma4FullScratch, "partials"](ctx, plan)
    var merge_segments = scratch.binding[
        Gemma4FullScratch, "merge_segments",
    ](ctx, plan)

    dispatch_full_attention[
        head_dim=head_dim, num_q=num_q_heads, gqa_ratio=gqa_ratio,
        kv_stride=k_rows, partial_stride=partial_stride,
        max_worker_count=max_worker_count,
    ](q_outs, k_kv, v_kv, q_local_outs, partials,
      merge_segments, local_num_q_heads, base_pos, seq_len, pools, prof)

    dispatch_gemm_cols[
        rows=C.HIDDEN, max_worker_count=max_worker_count,
    ](
        q_local_outs,
        attn.o_proj.binding(attn_ctx),
        xs, local_q_rows, seq_len, pools, prof)


def dispatch_moe[
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
    comptime n_eps = C.HIDDEN * C.RMS_NORM_EPS

    var per_expert_scale_ptr = body.router_pes.at(ctx.layer_base)

    var x_normed = scratch.binding[Gemma4FfnMoeScratch, "moe_x_normed"](ctx, plan)
    var cands = scratch.binding[Gemma4FfnMoeScratch, "moe_cands"](ctx, plan)
    var router_scaled = scratch.binding[Gemma4FfnMoeScratch, "moe_router_scaled"](ctx, plan)
    var route_idx = scratch.binding[Gemma4FfnMoeScratch, "moe_route_idx"](ctx, plan)
    var route_w = scratch.binding[Gemma4FfnMoeScratch, "moe_route_w"](ctx, plan)
    var expert_offset = scratch.binding[Gemma4FfnMoeScratch, "moe_expert_offset"](ctx, plan)
    var routes = scratch.binding[Gemma4FfnMoeScratch, "moe_routes"](ctx, plan)
    var hidden_bucket = scratch.binding[Gemma4FfnMoeScratch, "moe_hidden_bucket"](ctx, plan)
    var moe_accum = scratch.binding[Gemma4FfnMoeScratch, "moe_accum"](ctx, plan)
    var gate_scratch = scratch.binding[Gemma4FfnMoeScratch, "moe_gate_scratch"](ctx, plan)

    var nws = dispatch_router_expert[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
        top_k=C.TOP_K,
        max_worker_count=max_worker_count,
    ](x_input,
      body.router_proj.binding(ctx),
      body.router_scale.binding(ctx),
      router_scaled, cands, experts_per_rank, seq_len, pools, prof)

    merge_router_candidates_expert[C.TOP_K](
        cands, nws, seq_len, per_expert_scale_ptr, route_idx, route_w)

    dispatch_rms_norm[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
        max_worker_count=max_worker_count,
    ](x_input, x_normed,
      body.pre_ffn_norm_2.binding(ctx), seq_len, pools, prof)

    build_expert_schedules[C.NUM_EXPERTS, C.TOP_K](
        route_idx, route_w, expert_offset, routes, experts_per_rank, degree, seq_len)

    dispatch_phase1_gate_up[
        hidden=C.HIDDEN, gate_up_fused=C.MOE_GATE_UP_FUSED,
        intermediate=C.MOE_INTERMEDIATE,
        tile_j=Gemma4FfnMoeScratch.PHASE1_TILE_J, MR=Gemma4FfnMoeScratch.PHASE1_MR,
        max_worker_count=max_worker_count,
    ](x_normed, expert_offset, routes,
      body.experts_gate_up.binding(ctx),
      gate_scratch, hidden_bucket, experts_per_rank, pools, prof)

    dispatch_phase2_down[
        hidden=C.HIDDEN, intermediate=C.MOE_INTERMEDIATE,
        max_worker_count=max_worker_count,
    ](expert_offset, routes, hidden_bucket,
      body.experts_down.binding(ctx),
      moe_accum, moe_out, experts_per_rank, seq_len, pools, prof)

    dispatch_allreduce_inplace[
        BF16, max_worker_count=max_worker_count,
    ](moe_out, seq_len * C.HIDDEN, pools, prof)


def dispatch_ffn[
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
    comptime n_eps = C.HIDDEN * C.RMS_NORM_EPS
    var intermediate_per_rank = Gemma4Shapes.GateUp.data_n(degree)

    var layer_scalar_ptr = body.layer_scalar.at(ctx.layer_base)

    var gate = scratch.binding[Gemma4FfnMoeScratch, "ffn_gate"](ctx, plan)
    var up = scratch.binding[Gemma4FfnMoeScratch, "ffn_up"](ctx, plan)
    var dense_out = scratch.binding[Gemma4FfnMoeScratch, "ffn_dense_out"](ctx, plan)

    dispatch_rms_norm[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
        max_worker_count=max_worker_count,
    ](x_main, x_residual,
      body.pre_ffn_norm.binding(ctx), seq_len, pools, prof)

    dispatch_gemm[
        cols=C.HIDDEN, max_worker_count=max_worker_count,
    ](x_residual, body.gate_proj.binding(ctx), gate, intermediate_per_rank, seq_len, pools, prof)

    dispatch_gemm[
        cols=C.HIDDEN, max_worker_count=max_worker_count,
    ](x_residual, body.up_proj.binding(ctx), up, intermediate_per_rank, seq_len, pools, prof)

    dispatch_gelu_gate_up[
        max_worker_count=max_worker_count,
    ](gate, up, gate, intermediate_per_rank, seq_len, pools, prof)

    dispatch_moe[max_worker_count=max_worker_count](
        body, ctx, x_main, x_residual, seq_len, scratch, plan, pools, prof)

    dispatch_gemm_cols[
        rows=C.HIDDEN, max_worker_count=max_worker_count,
    ](gate, body.down_proj.binding(ctx), dense_out, intermediate_per_rank, seq_len, pools, prof)

    dispatch_allreduce_inplace[
        BF16, max_worker_count=max_worker_count,
    ](dense_out, seq_len * C.HIDDEN, pools, prof)

    dispatch_rms_norm[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
        max_worker_count=max_worker_count,
    ](dense_out, dense_out,
      body.post_ffn_norm_1.binding(ctx), seq_len, pools, prof)

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
    var max_workers: Int
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
        self.max_workers = max_workers
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
        self.profiler = Profiler[Self.profile, Self.profile_slots]()

    def model_init(mut self):
        ref layout = self.layout
        comptime width = simd_width_of[DType.float32]()

        prime_fp_environment(self.pools)

        comptime inv_sqrt_hidden = 1.0 / sqrt[DType.float32, 1](C.HIDDEN)
        for rank in range(self.degree):
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
        base_pos: Int,
    ) -> TemporalLogitsView[C.VOCAB_SIZE]:
        ref layout = self.layout
        var degree = self.degree
        comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
        comptime n_eps = C.HIDDEN * C.RMS_NORM_EPS
        comptime embed_scale = Float64(sqrt[DType.float32, 1](C.HIDDEN)
            .cast[DType.bfloat16]().cast[DType.float32]())
        var vocab_per_rank = C.VOCAB_SIZE // degree
        var shard_rows = Gemma4TailShapes.Embed.data_n(degree)

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
        var x_res_ranks = layout.activations.x_residual.state_binding(ctx)
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

            dispatch_embed_lookup[
                hidden=C.HIDDEN, scale=embed_scale,
            ](chunk,
              layout.tail.proto.embed.binding(tail_ctx),
              x_main_ranks, shard_rows, chunk_len, self.pools, self.profiler)
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
                    lb = layout.sliding.base(self.arena_bases[0], entry.local_idx)
                    body = layout.sliding.proto.body
                var layer_ctx = ctx.with_layer(lb)

                dispatch_rms_norm[
                    hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
                ](x_main_ranks, x_res_ranks,
                  body.input_norm.binding(layer_ctx),
                  chunk_len, self.pools, self.profiler)

                if entry.kind == LayerKind.FULL:
                    dispatch_full_attention_qkv[
                        max_seq_len=Self.max_seq_len,
                    ](
                        layout, ctx, pos, chunk_len, entry.local_idx,
                        self.scratch, self.full_plan, self.pools, self.profiler)
                else:
                    dispatch_sliding_attention_qkv[
                        max_seq_len=Self.max_seq_len,
                    ](
                        layout, ctx, pos, chunk_len, entry.local_idx,
                        self.scratch, self.sliding_plan, self.pools, self.profiler)

                dispatch_allreduce_inplace[BF16](
                    x_res_ranks, chunk_len * C.HIDDEN, self.pools, self.profiler)

                fused_norm_residual_add[
                    hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
                ](x_res_ranks, x_main_ranks, x_main_ranks,
                  body.post_attn_norm.binding(layer_ctx),
                  chunk_len, self.pools, self.profiler)

                dispatch_ffn(
                    body, layer_ctx, x_main_ranks, x_res_ranks, chunk_len,
                    self.scratch, self.ffn_plan, self.pools, self.profiler)

            if is_last:
                var x_last = x_main_ranks.shifted((chunk_len - 1) * C.HIDDEN)

                dispatch_rms_norm[
                    hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps,
                ](x_last, x_last,
                  layout.tail.proto.final_norm.binding(tail_ctx),
                  1, self.pools, self.profiler)

                dispatch_gemv_softcap[
                    cols=C.HIDDEN, cap=C.LOGIT_SOFTCAP,
                ](x_last, layout.tail.proto.embed.binding(tail_ctx), logits,
                  vocab_per_rank, self.pools, self.profiler)

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
        model.model_init()
        return model^
