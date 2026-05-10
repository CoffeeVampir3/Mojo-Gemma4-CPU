from std.collections import InlineArray
from std.pathlib import Path
from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from simd_math.ops import sqrt

from numa import NumaArena, NumaInfo, NumaTopology
from threading import BurstPool
from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from kernels.helpers import (
    RankBuffers, NumaPointerArray, NumaTypedPointerArray, MAX_WORKERS,
)
from kernels.reductions import dispatch_broadcast, dispatch_allreduce
from kernels.rmsnorm import dispatch_rms_norm, dispatch_rms_norm_qkv_heads
from kernels.rmsnorm import fused_norm_residual_add
from kernels.gemv import dispatch_gemv_chained_qkv, dispatch_gemv
from kernels.rope import dispatch_rope_cache_write
from kernels.kv_tiled_attention import dispatch_sliding_attention, FLASH_PARTIAL_STRIDE
from kernels.full_attention import dispatch_full_attention, PARTIAL_STRIDE
from kernels.logsum_merge import (
    dispatch_merge_flash_partials, dispatch_merge_context_flash_partials,
)
from kernels.moe_router import (
    RouterCandidate, SparseRoute,
    dispatch_router_sharded, merge_router_candidates, build_expert_schedules,
)
from kernels.moe_experts import (
    dispatch_phase1_gate_up, dispatch_phase2_down,
    PHASE1_TILE_J, PHASE1_MR,
)
from kernels.elementwise import dispatch_gelu_gate_up, dispatch_scalar_mul
from modeling.linear_borrow_pool import ScratchPool
from modeling.linear_borrow_pool import scratch_block_bytes
from modeling.kv_cache import Gemma4KV, Gemma4KVSliding, Gemma4KVGlobal

from modeling.model_spec import (
    BF16, F32,
    Shape, WeightDesc,
    DistributionDegree,
    TensorParallelRows, TensorParallelColumns,
    ContextParallelRows, ExpertParallelRows, VocabularyParallelRows,
    Replicated,
    TensorRowSharded, TensorColumnSharded,
    ContextRowSharded, ExpertRowBlockSharded, VocabularyRowSharded,
    DEFAULT_ALIGNMENT,
    align_up,
)
from modeling.gemma4_common import (
    Gemma4BaseConfig, is_full_layer,
)
from modeling.modeling_common import (
    TensorRef, Repeated, SectionBuilder,
    LayerBuilder, ArenaLayout,
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
    comptime D = DistributionDegree[Self.degree]
    comptime LocalD = DistributionDegree[1]
    comptime SlidingKV = TensorColumnSharded[C.SLIDING_WINDOW, C.KV_DIM_SLIDING, Self.D]
    comptime FullKV = ContextRowSharded[C.MAX_SEQ_LEN, C.KV_DIM_FULL, Self.D]
    comptime SlidingRope = ContextRowSharded[C.MAX_SEQ_LEN, C.ROPE_HALF_SLIDING, Self.LocalD]
    comptime FullRope = ContextRowSharded[C.MAX_SEQ_LEN, C.ROPE_HALF_FULL, Self.D]


struct Gemma4TailShapes[degree: Int]:
    comptime D = DistributionDegree[Self.degree]
    comptime FinalNorm = Replicated[C.HIDDEN, 1]
    comptime Embed = VocabularyRowSharded[C.VOCAB_SIZE, C.HIDDEN, Self.D]


comptime SlidingAttentionShapeContract[degree: Int]: Bool = (
    SlidingAttentionContract[degree]
    and Gemma4Shapes[degree].SlidingQ.DATA_N
        == TensorParallelRows[C.Q_DIM_SLIDING, Gemma4Shapes[degree].D]
    and Gemma4Shapes[degree].SlidingKV.DATA_N
        == TensorParallelRows[C.KV_DIM_SLIDING, Gemma4Shapes[degree].D]
    and Gemma4Shapes[degree].SlidingO.DATA_M
        == TensorParallelColumns[C.Q_DIM_SLIDING, Gemma4Shapes[degree].D]
    and Gemma4StateShapes[degree].SlidingKV.DATA_N == C.SLIDING_WINDOW
    and Gemma4StateShapes[degree].SlidingKV.DATA_M
        == TensorParallelColumns[C.KV_DIM_SLIDING, Gemma4StateShapes[degree].D]
)


comptime FullAttentionShapeContract[degree: Int]: Bool = (
    FullAttentionContract[degree]
    and Gemma4Shapes[degree].FullQ.DATA_N == C.Q_DIM_FULL
    and Gemma4Shapes[degree].FullQ.DATA_M == C.HIDDEN
    and Gemma4Shapes[degree].FullK.DATA_N == C.KV_DIM_FULL
    and Gemma4Shapes[degree].FullK.DATA_M == C.HIDDEN
    and Gemma4Shapes[degree].FullO.DATA_M
        == TensorParallelColumns[C.Q_DIM_FULL, Gemma4Shapes[degree].D]
    and Gemma4StateShapes[degree].FullKV.DATA_N
        == ContextParallelRows[C.MAX_SEQ_LEN, Gemma4StateShapes[degree].D]
    and Gemma4StateShapes[degree].FullKV.DATA_M == C.KV_DIM_FULL
    and Gemma4StateShapes[degree].FullRope.DATA_N
        == ContextParallelRows[C.MAX_SEQ_LEN, Gemma4StateShapes[degree].D]
    and Gemma4StateShapes[degree].FullRope.DATA_M == C.ROPE_HALF_FULL
)


comptime DenseMlpShapeContract[degree: Int]: Bool = (
    DenseMlpContract[degree]
    and Gemma4Shapes[degree].GateUp.DATA_N
        == TensorParallelRows[C.INTERMEDIATE, Gemma4Shapes[degree].D]
    and Gemma4Shapes[degree].Down.DATA_M
        == TensorParallelColumns[C.INTERMEDIATE, Gemma4Shapes[degree].D]
)


comptime MoeShapeContract[degree: Int]: Bool = (
    MoeContract[degree]
    and Gemma4Shapes[degree].RouterProj.DATA_N
        == ExpertParallelRows[
            C.NUM_EXPERTS, 1, Gemma4Shapes[degree].D,
        ]
    and Gemma4Shapes[degree].RouterProj.DATA_M == C.HIDDEN
    and Gemma4Shapes[degree].ExpertsGateUp.DATA_N
        == ExpertParallelRows[
            C.NUM_EXPERTS, C.MOE_GATE_UP_FUSED, Gemma4Shapes[degree].D,
        ]
    and Gemma4Shapes[degree].ExpertsGateUp.DATA_M == C.HIDDEN
    and Gemma4Shapes[degree].ExpertsDown.DATA_N
        == ExpertParallelRows[
            C.NUM_EXPERTS, C.HIDDEN, Gemma4Shapes[degree].D,
        ]
    and Gemma4Shapes[degree].ExpertsDown.DATA_M == C.MOE_INTERMEDIATE
)


comptime LmHeadShapeContract[degree: Int]: Bool = (
    LmHeadContract[degree]
    and Gemma4TailShapes[degree].FinalNorm.DATA_N == C.HIDDEN
    and Gemma4TailShapes[degree].FinalNorm.DATA_M == 1
    and Gemma4TailShapes[degree].Embed.DATA_N
        == VocabularyParallelRows[C.VOCAB_SIZE, Gemma4TailShapes[degree].D]
    and Gemma4TailShapes[degree].Embed.DATA_M == C.HIDDEN
)


@fieldwise_init
struct SlidingAttnRefs[degree: Int](Copyable, ImplicitlyCopyable):
    comptime S = Gemma4Shapes[Self.degree]
    var q_proj: TensorRef[BF16, Self.S.SlidingQ]
    var k_proj: TensorRef[BF16, Self.S.SlidingKV]
    var v_proj: TensorRef[BF16, Self.S.SlidingKV]
    var o_proj: TensorRef[BF16, Self.S.SlidingO]
    var q_norm: TensorRef[BF16, Shape[C.HEAD_DIM_SLIDING, 1]]
    var k_norm: TensorRef[BF16, Shape[C.HEAD_DIM_SLIDING, 1]]


@fieldwise_init
struct FullAttnRefs[degree: Int](Copyable, ImplicitlyCopyable):
    comptime S = Gemma4Shapes[Self.degree]
    var q_proj: TensorRef[BF16, Self.S.FullQ]
    var k_proj: TensorRef[BF16, Self.S.FullK]
    var o_proj: TensorRef[BF16, Self.S.FullO]
    var q_norm: TensorRef[BF16, Shape[C.HEAD_DIM_FULL, 1]]
    var k_norm: TensorRef[BF16, Shape[C.HEAD_DIM_FULL, 1]]


@fieldwise_init
struct BodyRefs[degree: Int](Copyable, ImplicitlyCopyable):
    comptime S = Gemma4Shapes[Self.degree]
    var input_norm:      TensorRef[BF16, Shape[C.HIDDEN, 1]]
    var post_attn_norm:  TensorRef[BF16, Shape[C.HIDDEN, 1]]
    var pre_ffn_norm:    TensorRef[BF16, Shape[C.HIDDEN, 1]]
    var pre_ffn_norm_2:  TensorRef[BF16, Shape[C.HIDDEN, 1]]
    var post_ffn_norm_1: TensorRef[BF16, Shape[C.HIDDEN, 1]]
    var post_ffn_norm_2: TensorRef[BF16, Shape[C.HIDDEN, 1]]
    var post_ffn_norm:   TensorRef[BF16, Shape[C.HIDDEN, 1]]
    var gate_proj:       TensorRef[BF16, Self.S.GateUp]
    var up_proj:         TensorRef[BF16, Self.S.GateUp]
    var down_proj:       TensorRef[BF16, Self.S.Down]
    var router_proj:     TensorRef[BF16, Self.S.RouterProj]
    var router_scale:    TensorRef[BF16, Shape[C.HIDDEN, 1]]
    var router_pes:      TensorRef[BF16, Shape[C.NUM_EXPERTS, 1]]
    var experts_gate_up: TensorRef[BF16, Self.S.ExpertsGateUp]
    var experts_down:    TensorRef[BF16, Self.S.ExpertsDown]
    var layer_scalar:    TensorRef[BF16, Shape[1, 1]]


@fieldwise_init
struct SlidingLayerRefs[degree: Int](Copyable, ImplicitlyCopyable):
    var attn: SlidingAttnRefs[Self.degree]
    var body: BodyRefs[Self.degree]


@fieldwise_init
struct FullLayerRefs[degree: Int](Copyable, ImplicitlyCopyable):
    var attn: FullAttnRefs[Self.degree]
    var body: BodyRefs[Self.degree]


@fieldwise_init
struct SlidingKVSlots[degree: Int](Copyable, ImplicitlyCopyable):
    comptime S = Gemma4StateShapes[Self.degree]
    var k: TensorRef[BF16, Self.S.SlidingKV]
    var v: TensorRef[BF16, Self.S.SlidingKV]


@fieldwise_init
struct FullKVSlots[degree: Int](Copyable, ImplicitlyCopyable):
    comptime S = Gemma4StateShapes[Self.degree]
    var k: TensorRef[BF16, Self.S.FullKV]
    var v: TensorRef[BF16, Self.S.FullKV]


@fieldwise_init
struct RopeSlots[half: Int, degree: Int = 1](Copyable, ImplicitlyCopyable):
    comptime D = DistributionDegree[Self.degree]
    var cos: TensorRef[F32, ContextRowSharded[C.MAX_SEQ_LEN, Self.half, Self.D]]
    var sin: TensorRef[F32, ContextRowSharded[C.MAX_SEQ_LEN, Self.half, Self.D]]


@fieldwise_init
struct ActivationSlots(Copyable, ImplicitlyCopyable):
    var x_main:     TensorRef[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]]
    var x_residual: TensorRef[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]]


@fieldwise_init
struct TailRefs[degree: Int](Copyable, ImplicitlyCopyable):
    comptime S = Gemma4TailShapes[Self.degree]
    var final_norm: TensorRef[BF16, Self.S.FinalNorm]
    var embed:      TensorRef[BF16, Self.S.Embed]


@fieldwise_init
struct Gemma4Topology[degree: Int](Copyable, ImplicitlyCopyable):
    var arena: ArenaLayout
    var sliding: Repeated[SlidingLayerRefs[Self.degree]]
    var full: Repeated[FullLayerRefs[Self.degree]]

    var sliding_kv: Repeated[SlidingKVSlots[Self.degree]]
    var full_kv: Repeated[FullKVSlots[Self.degree]]
    var activations: ActivationSlots
    var sliding_rope: RopeSlots[C.ROPE_HALF_SLIDING]
    var full_rope: RopeSlots[C.ROPE_HALF_FULL, Self.degree]

    var tail: TailRefs[Self.degree]

    @always_inline
    def bind(self, base: Int) -> Self:
        var t = self
        t.arena = t.arena.bind(base)
        return t


def emit_body[degree: Int](mut b: LayerBuilder, mut e: List[WeightDesc]) -> BodyRefs[degree]:
    comptime S = Gemma4Shapes[degree]
    comptime H = C.HIDDEN
    comptime NE = C.NUM_EXPERTS
    return BodyRefs[degree](
        input_norm      = b.bfs[Shape[H, 1]](e, "input_layernorm.weight"),
        post_attn_norm  = b.bfs[Shape[H, 1]](e, "post_attention_layernorm.weight"),
        pre_ffn_norm    = b.bfs[Shape[H, 1]](e, "pre_feedforward_layernorm.weight"),
        pre_ffn_norm_2  = b.bfs[Shape[H, 1]](e, "pre_feedforward_layernorm_2.weight"),
        post_ffn_norm_1 = b.bfs[Shape[H, 1]](e, "post_feedforward_layernorm_1.weight"),
        post_ffn_norm_2 = b.bfs[Shape[H, 1]](e, "post_feedforward_layernorm_2.weight"),
        post_ffn_norm   = b.bfs[Shape[H, 1]](e, "post_feedforward_layernorm.weight"),
        gate_proj       = b.bfs[S.GateUp](e, "mlp.gate_proj.weight"),
        up_proj         = b.bfs[S.GateUp](e, "mlp.up_proj.weight"),
        down_proj       = b.bfs[S.Down](e, "mlp.down_proj.weight"),
        router_proj     = b.bfs[S.RouterProj](e, "router.proj.weight"),
        router_scale    = b.bfs[Shape[H, 1]](e, "router.scale"),
        router_pes      = b.bfs[Shape[NE, 1]](e, "router.per_expert_scale"),
        experts_gate_up = b.bfs[S.ExpertsGateUp](e, "experts.gate_up_proj"),
        experts_down    = b.bfs[S.ExpertsDown](e, "experts.down_proj"),
        layer_scalar    = b.bfs[Shape[1, 1]](e, "layer_scalar"),
    )


def emit_sliding[degree: Int](
    prefix: String, layer_base: Int, mut e: List[WeightDesc],
) -> Tuple[SlidingLayerRefs[degree], Int]:
    var b = LayerBuilder(prefix, layer_base)
    comptime S = Gemma4Shapes[degree]
    var attn = SlidingAttnRefs[degree](
        q_proj = b.bfs[S.SlidingQ](e, "self_attn.q_proj.weight"),
        k_proj = b.bfs[S.SlidingKV](e, "self_attn.k_proj.weight"),
        v_proj = b.bfs[S.SlidingKV](e, "self_attn.v_proj.weight"),
        o_proj = b.bfs[S.SlidingO](e, "self_attn.o_proj.weight"),
        q_norm = b.bfs[Shape[C.HEAD_DIM_SLIDING, 1]](e, "self_attn.q_norm.weight"),
        k_norm = b.bfs[Shape[C.HEAD_DIM_SLIDING, 1]](e, "self_attn.k_norm.weight"),
    )
    return (SlidingLayerRefs[degree](attn=attn, body=emit_body[degree](b, e)), b.cursor)


def emit_full[degree: Int](
    prefix: String, layer_base: Int, mut e: List[WeightDesc],
) -> Tuple[FullLayerRefs[degree], Int]:
    var b = LayerBuilder(prefix, layer_base)
    comptime S = Gemma4Shapes[degree]
    var attn = FullAttnRefs[degree](
        q_proj = b.bfs[S.FullQ](e, "self_attn.q_proj.weight"),
        k_proj = b.bfs[S.FullK](e, "self_attn.k_proj.weight"),
        o_proj = b.bfs[S.FullO](e, "self_attn.o_proj.weight"),
        q_norm = b.bfs[Shape[C.HEAD_DIM_FULL, 1]](e, "self_attn.q_norm.weight"),
        k_norm = b.bfs[Shape[C.HEAD_DIM_FULL, 1]](e, "self_attn.k_norm.weight"),
    )
    return (FullLayerRefs[degree](attn=attn, body=emit_body[degree](b, e)), b.cursor)


def calculate_peak_scratch[degree: Int]() -> Int:
    comptime bf16 = BF16.ELEMENT_BYTES
    comptime f32 = F32.ELEMENT_BYTES
    comptime i32 = 4
    comptime cand_bytes = 8
    comptime route_bytes = 8
    comptime seq = C.MAX_SEQ_LEN
    comptime S = Gemma4Shapes[degree]
    comptime experts_per_rank = C.NUM_EXPERTS // degree

    comptime full_q = scratch_block_bytes[seq * S.FullQ.N * bf16]()
    comptime full_kv = scratch_block_bytes[seq * S.FullK.N * bf16]()
    comptime full_attn = max(full_q + 2 * full_kv, 2 * full_q)

    comptime sliding_q = scratch_block_bytes[seq * S.SlidingQ.N * bf16]()
    comptime sliding_kv = scratch_block_bytes[seq * S.SlidingKV.N * bf16]()
    comptime sliding_attn = max(sliding_q + 2 * sliding_kv, 2 * sliding_q)

    comptime ffn_gate = scratch_block_bytes[seq * S.GateUp.N * bf16]()
    comptime ffn_up = scratch_block_bytes[seq * S.GateUp.N * bf16]()
    comptime ffn_dense_out = scratch_block_bytes[seq * C.HIDDEN * bf16]()
    comptime ffn_outer = ffn_gate + ffn_up + ffn_dense_out

    comptime moe_x_normed = scratch_block_bytes[seq * C.HIDDEN * bf16]()
    comptime moe_hidden_bucket = scratch_block_bytes[
        seq * C.TOP_K * C.MOE_INTERMEDIATE * bf16]()
    comptime moe_accum = scratch_block_bytes[seq * C.HIDDEN * f32]()
    comptime moe_route_idx = scratch_block_bytes[seq * C.TOP_K * i32]()
    comptime moe_route_w = scratch_block_bytes[seq * C.TOP_K * f32]()
    comptime moe_cands = scratch_block_bytes[seq * C.TOP_K * cand_bytes]()
    comptime moe_routes = scratch_block_bytes[seq * C.TOP_K * route_bytes]()
    comptime moe_expert_offset = scratch_block_bytes[
        (experts_per_rank + 1) * i32]()
    comptime moe_gate_scratch = scratch_block_bytes[
        MAX_WORKERS * PHASE1_MR * 2 * PHASE1_TILE_J * f32]()
    comptime moe_router_scaled = scratch_block_bytes[
        MAX_WORKERS * C.HIDDEN * f32]()
    comptime ffn_moe_inner = (
        moe_x_normed + moe_hidden_bucket + moe_accum
        + moe_route_idx + moe_route_w + moe_cands + moe_routes
        + moe_expert_offset + moe_gate_scratch + moe_router_scaled
    )

    comptime ffn_peak = ffn_outer + ffn_moe_inner

    comptime vocab_per_rank = C.VOCAB_SIZE // degree
    comptime lm_head_logits = scratch_block_bytes[vocab_per_rank * bf16]()

    return max(
        max(ffn_peak, max(full_attn, sliding_attn)),
        lm_head_logits,
    )


@fieldwise_init
struct Gemma4LoadPlan[degree: Int](Movable):
    var topology: Gemma4Topology[Self.degree]
    var descs: List[WeightDesc]


def build_gemma4_plan[degree: Int]() -> Gemma4LoadPlan[degree]:
    comptime assert SlidingAttentionShapeContract[degree], "sliding attention distribution contract failed"
    comptime assert FullAttentionShapeContract[degree], "full attention distribution contract failed"
    comptime assert DenseMlpShapeContract[degree], "dense MLP distribution contract failed"
    comptime assert MoeShapeContract[degree], "MoE distribution contract failed"
    comptime assert LmHeadShapeContract[degree], "LM head distribution contract failed"
    comptime StateS = Gemma4StateShapes[degree]
    comptime TailS = Gemma4TailShapes[degree]
    var descs = List[WeightDesc]()

    var probe = List[WeightDesc]()
    var sl_r = emit_sliding[degree]("", 0, probe)
    var fl_r = emit_full[degree]("", 0, probe)
    var sl_proto = sl_r[0]
    var sl_stride = sl_r[1]
    var fl_proto = fl_r[0]
    var fl_stride = fl_r[1]

    var sl_off = 0
    var fl_off = sl_off + C.NUM_SLIDING_LAYERS * sl_stride
    var distributed = fl_off + C.NUM_FULL_LAYERS * fl_stride

    var si = 0
    var fi = 0
    for i in range(C.NUM_LAYERS):
        var prefix = "model.language_model.layers." + String(i) + "."
        if is_full_layer(i):
            _ = emit_full[degree](prefix, fl_off + fi * fl_stride, descs)
            fi += 1
        else:
            _ = emit_sliding[degree](prefix, sl_off + si * sl_stride, descs)
            si += 1

    var tb = LayerBuilder("", 0, start_at=distributed)
    var tail = TailRefs[degree](
        final_norm=tb.bfs[TailS.FinalNorm](descs, "model.language_model.norm.weight"),
        embed=tb.bfs[TailS.Embed](descs, "model.language_model.embed_tokens.weight"))
    distributed = tb.cursor

    var state = SectionBuilder()
    state.cursor = distributed

    var skv_sb = SectionBuilder()
    var skv_proto = SlidingKVSlots[degree](
        k=skv_sb.reserve[BF16, StateS.SlidingKV](),
        v=skv_sb.reserve[BF16, StateS.SlidingKV]())
    var sliding_kv = Repeated[SlidingKVSlots[degree]](
        skv_proto, state.cursor, skv_sb.bytes(), C.NUM_SLIDING_LAYERS)
    state.advance_bytes(C.NUM_SLIDING_LAYERS * skv_sb.bytes())

    var fkv_sb = SectionBuilder()
    var fkv_proto = FullKVSlots[degree](
        k=fkv_sb.reserve[BF16, StateS.FullKV](),
        v=fkv_sb.reserve[BF16, StateS.FullKV]())
    var full_kv = Repeated[FullKVSlots[degree]](
        fkv_proto, state.cursor, fkv_sb.bytes(), C.NUM_FULL_LAYERS)
    state.advance_bytes(C.NUM_FULL_LAYERS * fkv_sb.bytes())

    var activations = ActivationSlots(
        x_main=state.reserve[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]](),
        x_residual=state.reserve[BF16, Shape[C.MAX_SEQ_LEN, C.HIDDEN]]())

    var scratch_cap = calculate_peak_scratch[degree]()
    var scratch_off = state.reserve_bytes(scratch_cap)

    var sliding_rope = RopeSlots[C.ROPE_HALF_SLIDING](
        cos=state.reserve[F32, StateS.SlidingRope](),
        sin=state.reserve[F32, StateS.SlidingRope]())
    var full_rope = RopeSlots[C.ROPE_HALF_FULL, degree](
        cos=state.reserve[F32, StateS.FullRope](),
        sin=state.reserve[F32, StateS.FullRope]())

    var arena = ArenaLayout(
        base=0,
        distributed_bytes=distributed,
        state_bytes=state.bytes() - distributed,
        host_bytes=align_up(state.bytes()),
        scratch_off=scratch_off,
    )
    var topo = Gemma4Topology[degree](
        arena=arena,
        sliding=Repeated[SlidingLayerRefs[degree]](sl_proto, sl_off, sl_stride, C.NUM_SLIDING_LAYERS),
        full=Repeated[FullLayerRefs[degree]](fl_proto, fl_off, fl_stride, C.NUM_FULL_LAYERS),
        sliding_kv=sliding_kv, full_kv=full_kv,
        activations=activations,
        sliding_rope=sliding_rope, full_rope=full_rope,
        tail=tail)
    return Gemma4LoadPlan[degree](topo, descs^)


comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


def dispatch_sliding_attention_qkv[
    P: BurstThreadPool, //, degree: Int,
](
    topo: Gemma4Topology[degree],
    arena_bases: InlineArray[Int, degree],
    pos: Int,
    layer_idx: Int,
    ref kv: Gemma4KV[degree],
    mut scratch: ScratchPool,
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
    comptime flash_stride = FLASH_PARTIAL_STRIDE[num_q_heads, head_dim]

    var lb = topo.sliding.base(arena_bases[0], layer_idx)
    var attn = topo.sliding.proto.attn
    var x = topo.activations.x_residual.bound(arena_bases[0]).as_ptr()

    var q_lease = scratch.borrow[Scalar[DType.bfloat16], q_rows]()
    var kv_lease = scratch.borrow[Scalar[DType.bfloat16], 2 * kv_rows]()
    var q_out = q_lease.as_ptr[Scalar[DType.bfloat16]]()
    var k_out = kv_lease.as_ptr[Scalar[DType.bfloat16]]()
    var v_out = kv_lease.as_ptr[Scalar[DType.bfloat16]](element_offset=kv_rows)

    var q_outs = NumaPointerArray[DType.bfloat16, degree](q_out, arena_bases)
    var k_outs = NumaPointerArray[DType.bfloat16, degree](k_out, arena_bases)
    var v_outs = NumaPointerArray[DType.bfloat16, degree](v_out, arena_bases)
    var xs = NumaPointerArray[DType.bfloat16, degree](x, arena_bases)

    dispatch_gemv_chained_qkv[
        q_rows=q_rows, kv_rows=kv_rows, cols=C.HIDDEN, tp=degree,
    ](xs,
      NumaPointerArray[DType.bfloat16, degree](attn.q_proj.bound(lb).as_ptr(), arena_bases),
      NumaPointerArray[DType.bfloat16, degree](attn.k_proj.bound(lb).as_ptr(), arena_bases),
      NumaPointerArray[DType.bfloat16, degree](attn.v_proj.bound(lb).as_ptr(), arena_bases),
      q_outs, k_outs, v_outs, pools)

    dispatch_rms_norm_qkv_heads[
        head_dim=head_dim, sqrt_n=sqrt_hd, n_eps=hd_eps,
        num_q=num_q_heads, num_kv=num_kv_heads, tp=degree,
    ](q_outs, q_outs, k_outs, k_outs, v_outs, v_outs,
      NumaPointerArray[DType.bfloat16, degree](attn.q_norm.bound(lb).as_ptr(), arena_bases),
      NumaPointerArray[DType.bfloat16, degree](attn.k_norm.bound(lb).as_ptr(), arena_bases),
      pools)

    dispatch_rope_cache_write[
        half=rope_half, pair_stride=head_dim // 2,
        num_q=num_q_heads, num_kv=num_kv_heads,
        head_dim=head_dim, kv_cache_stride=kv_cols,
        slot_mask=C.SLIDING_WINDOW - 1, cache_degree=1, tp=degree,
    ](q_outs, k_outs, v_outs,
      NumaPointerArray[DType.bfloat16, degree](kv.sliding.k(layer_idx, 0), arena_bases),
      NumaPointerArray[DType.bfloat16, degree](kv.sliding.v(layer_idx, 0), arena_bases),
      NumaPointerArray[DType.float32, degree](
          topo.sliding_rope.cos.bound(arena_bases[0]).as_ptr(), arena_bases),
      NumaPointerArray[DType.float32, degree](
          topo.sliding_rope.sin.bound(arena_bases[0]).as_ptr(), arena_bases),
      pos, 1, pools)

    kv_lease^.release()

    var p_lease = scratch.borrow[Scalar[DType.float32], 128 * flash_stride]()
    var partials_ptr = p_lease.as_ptr[Scalar[DType.float32]]()

    var valid_len = Gemma4KVSliding[degree].valid_len(pos)

    var nws = dispatch_sliding_attention[
        head_dim=head_dim, num_q=num_q_heads,
        gqa_ratio=num_q_heads // num_kv_heads, kv_stride=kv_cols,
        window=C.SLIDING_WINDOW, tp=degree,
    ](q_outs,
      NumaPointerArray[DType.bfloat16, degree](kv.sliding.k(layer_idx, 0), arena_bases),
      NumaPointerArray[DType.bfloat16, degree](kv.sliding.v(layer_idx, 0), arena_bases),
      NumaPointerArray[DType.float32, degree](partials_ptr, arena_bases),
      pos, valid_len, pools)

    dispatch_merge_flash_partials[head_dim, num_q_heads, tp=degree](
        q_outs, NumaPointerArray[DType.float32, degree](partials_ptr, arena_bases),
        flash_stride, nws, pools)

    p_lease^.release()

    dispatch_gemv[rows=C.HIDDEN, cols=q_rows, tp=degree](
        q_outs,
        NumaPointerArray[DType.bfloat16, degree](attn.o_proj.bound(lb).as_ptr(), arena_bases),
        xs, pools)

    q_lease^.release()


def dispatch_full_attention_qkv[
    P: BurstThreadPool, //, degree: Int,
](
    topo: Gemma4Topology[degree],
    arena_bases: InlineArray[Int, degree],
    pos: Int,
    layer_idx: Int,
    ref kv: Gemma4KV[degree],
    mut scratch: ScratchPool,
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
    comptime partial_stride = PARTIAL_STRIDE[num_q_heads, head_dim]

    var lb = topo.full.base(arena_bases[0], layer_idx)
    var attn = topo.full.proto.attn
    var x = topo.activations.x_residual.bound(arena_bases[0]).as_ptr()

    var q_lease = scratch.borrow[Scalar[DType.bfloat16], q_rows]()
    var kv_lease = scratch.borrow[Scalar[DType.bfloat16], 2 * k_rows]()
    var q_out = q_lease.as_ptr[Scalar[DType.bfloat16]]()
    var k_out = kv_lease.as_ptr[Scalar[DType.bfloat16]]()
    var v_out = kv_lease.as_ptr[Scalar[DType.bfloat16]](element_offset=k_rows)

    var q_outs = NumaPointerArray[DType.bfloat16, degree](q_out, arena_bases)
    var k_outs = NumaPointerArray[DType.bfloat16, degree](k_out, arena_bases)
    var v_outs = NumaPointerArray[DType.bfloat16, degree](v_out, arena_bases)
    var xs = NumaPointerArray[DType.bfloat16, degree](x, arena_bases)
    var q_norm_ws = NumaPointerArray[DType.bfloat16, degree](
        attn.q_norm.bound(lb).as_ptr(), arena_bases)
    var k_norm_ws = NumaPointerArray[DType.bfloat16, degree](
        attn.k_norm.bound(lb).as_ptr(), arena_bases)

    dispatch_gemv[rows=q_rows, cols=C.HIDDEN, tp=degree](
        xs, NumaPointerArray[DType.bfloat16, degree](attn.q_proj.bound(lb).as_ptr(), arena_bases),
        q_outs, pools)
    dispatch_gemv[rows=k_rows, cols=C.HIDDEN, tp=degree](
        xs, NumaPointerArray[DType.bfloat16, degree](attn.k_proj.bound(lb).as_ptr(), arena_bases),
        k_outs, pools)

    dispatch_rms_norm_qkv_heads[
        head_dim=head_dim, sqrt_n=sqrt_hd, n_eps=hd_eps,
        num_q=num_q_heads, num_kv=num_kv_heads, tp=degree,
    ](q_outs, q_outs, k_outs, k_outs, k_outs, v_outs,
      q_norm_ws, k_norm_ws, pools)

    var owner_bases = InlineArray[Int, degree](fill=arena_bases[pos % degree])
    dispatch_rope_cache_write[
        half=rope_half, pair_stride=pair_stride,
        num_q=num_q_heads, num_kv=num_kv_heads,
        head_dim=head_dim, kv_cache_stride=kv_cols,
        slot_mask=-1, cache_degree=degree, tp=degree,
    ](q_outs, k_outs, v_outs,
      NumaPointerArray[DType.bfloat16, degree](kv.full.k(layer_idx), arena_bases),
      NumaPointerArray[DType.bfloat16, degree](kv.full.v(layer_idx), arena_bases),
      NumaPointerArray[DType.float32, degree](
          topo.full_rope.cos.bound(owner_bases[0]).as_ptr(), owner_bases),
      NumaPointerArray[DType.float32, degree](
          topo.full_rope.sin.bound(owner_bases[0]).as_ptr(), owner_bases),
      pos, 1, pools)

    kv_lease^.release()

    var q_local_lease = scratch.borrow[Scalar[DType.bfloat16], local_q_rows]()
    var q_local = q_local_lease.as_ptr[Scalar[DType.bfloat16]]()
    var q_local_outs = NumaPointerArray[DType.bfloat16, degree](q_local, arena_bases)

    var p_lease = scratch.borrow[Scalar[DType.float32], 128 * partial_stride]()
    var partials_ptr = p_lease.as_ptr[Scalar[DType.float32]]()
    var partials_ptrs = NumaPointerArray[DType.float32, degree](partials_ptr, arena_bases)

    var valid_lens = InlineArray[Int, degree](uninitialized=True)
    for rank in range(degree):
        valid_lens[rank] = Gemma4KVGlobal[degree].valid_count(rank, pos)

    var nws = dispatch_full_attention[
        head_dim=head_dim, num_q=num_q_heads,
        gqa_ratio=C.NUM_HEADS // C.NUM_KV_HEADS_FULL, kv_stride=kv_cols, tp=degree,
    ](q_outs,
      NumaPointerArray[DType.bfloat16, degree](kv.full.k(layer_idx), arena_bases),
      NumaPointerArray[DType.bfloat16, degree](kv.full.v(layer_idx), arena_bases),
      partials_ptrs, valid_lens, pools)

    dispatch_merge_context_flash_partials[
        head_dim=head_dim, num_q=num_q_heads,
        local_num_q=local_num_q_heads, tp=degree,
    ](
        q_local_outs, partials_ptrs, partial_stride, nws, pools)

    p_lease^.release()

    dispatch_gemv[rows=C.HIDDEN, cols=local_q_rows, tp=degree](
        q_local_outs,
        NumaPointerArray[DType.bfloat16, degree](attn.o_proj.bound(lb).as_ptr(), arena_bases),
        xs, pools)

    q_local_lease^.release()
    q_lease^.release()


def dispatch_moe[
    P: BurstThreadPool, //, degree: Int,
](
    body: BodyRefs[degree],
    layer_base: Int,
    arena_bases: InlineArray[Int, degree],
    x_input: BF16Ptr,
    moe_out: BF16Ptr,
    seq_len: Int,
    mut scratch: ScratchPool,
    mut pools: HeapMoveArray[P],
):
    comptime experts_per_rank = C.NUM_EXPERTS // degree
    comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
    comptime n_eps = C.HIDDEN * C.RMS_NORM_EPS
    comptime rms_eps = Scalar[DType.float32](C.RMS_NORM_EPS)
    comptime immut = ImmutOrigin(MutAnyOrigin)

    var router_proj_ptr = body.router_proj.bound(layer_base).as_ptr()
    var router_scale_ptr = body.router_scale.bound(layer_base).as_ptr()
    var per_expert_scale_ptr = body.router_pes.bound(layer_base).as_ptr()
    var pre_ffn_norm_2_ptr = body.pre_ffn_norm_2.bound(layer_base).as_ptr()
    var experts_gate_up_ptr = body.experts_gate_up.bound(layer_base).as_ptr()
    var experts_down_ptr = body.experts_down.bound(layer_base).as_ptr()

    var x_normed_lease = scratch.borrow[
        Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.HIDDEN]()
    var cands_lease = scratch.borrow[
        RouterCandidate, C.MAX_SEQ_LEN * C.TOP_K]()
    var router_scaled_lease = scratch.borrow[
        Scalar[DType.float32], MAX_WORKERS * C.HIDDEN]()
    var route_idx_lease = scratch.borrow[
        Scalar[DType.int32], C.MAX_SEQ_LEN * C.TOP_K]()
    var route_w_lease = scratch.borrow[
        Scalar[DType.float32], C.MAX_SEQ_LEN * C.TOP_K]()
    var expert_offset_lease = scratch.borrow[
        Scalar[DType.int32], experts_per_rank + 1]()
    var routes_lease = scratch.borrow[
        SparseRoute, C.MAX_SEQ_LEN * C.TOP_K]()
    var hidden_bucket_lease = scratch.borrow[
        Scalar[DType.bfloat16],
        C.MAX_SEQ_LEN * C.TOP_K * C.MOE_INTERMEDIATE]()
    var moe_accum_lease = scratch.borrow[
        Scalar[DType.float32], C.MAX_SEQ_LEN * C.HIDDEN]()
    var gate_scratch_lease = scratch.borrow[
        Scalar[DType.float32],
        MAX_WORKERS * PHASE1_MR * 2 * PHASE1_TILE_J]()

    var x_input_ranks = NumaPointerArray[DType.bfloat16, degree](
        x_input, arena_bases)
    var moe_out_ranks = NumaPointerArray[DType.bfloat16, degree](
        moe_out, arena_bases)
    var router_proj_ranks = NumaPointerArray[DType.bfloat16, degree](
        router_proj_ptr, arena_bases)
    var router_scale_ranks = NumaPointerArray[DType.bfloat16, degree](
        router_scale_ptr, arena_bases)
    var pre_ffn_norm_2_ranks = NumaPointerArray[DType.bfloat16, degree](
        pre_ffn_norm_2_ptr, arena_bases)
    var experts_gate_up_ranks = NumaPointerArray[DType.bfloat16, degree](
        experts_gate_up_ptr, arena_bases)
    var experts_down_ranks = NumaPointerArray[DType.bfloat16, degree](
        experts_down_ptr, arena_bases)

    var x_normed_ranks = NumaPointerArray[DType.bfloat16, degree](
        x_normed_lease.as_ptr[Scalar[DType.bfloat16]](), arena_bases)
    var cands_ranks = NumaTypedPointerArray[RouterCandidate, degree](
        cands_lease.as_ptr[RouterCandidate](), arena_bases)
    var router_scaled_ranks = NumaPointerArray[DType.float32, degree](
        router_scaled_lease.as_ptr[Scalar[DType.float32]](), arena_bases)
    var route_idx_ranks = NumaPointerArray[DType.int32, degree](
        route_idx_lease.as_ptr[Scalar[DType.int32]](), arena_bases)
    var route_w_ranks = NumaPointerArray[DType.float32, degree](
        route_w_lease.as_ptr[Scalar[DType.float32]](), arena_bases)
    var expert_offset_ranks = NumaPointerArray[DType.int32, degree](
        expert_offset_lease.as_ptr[Scalar[DType.int32]](), arena_bases)
    var routes_ranks = NumaTypedPointerArray[SparseRoute, degree](
        routes_lease.as_ptr[SparseRoute](), arena_bases)
    var hidden_bucket_ranks = NumaPointerArray[DType.bfloat16, degree](
        hidden_bucket_lease.as_ptr[Scalar[DType.bfloat16]](), arena_bases)
    var moe_accum_ranks = NumaPointerArray[DType.float32, degree](
        moe_accum_lease.as_ptr[Scalar[DType.float32]](), arena_bases)
    var gate_scratch_ranks = NumaPointerArray[DType.float32, degree](
        gate_scratch_lease.as_ptr[Scalar[DType.float32]](), arena_bases)

    dispatch_router_sharded[
        hidden=C.HIDDEN, experts_per_rank=experts_per_rank,
        top_k=C.TOP_K, tp=degree, rms_eps=rms_eps,
    ](x_input_ranks, router_proj_ranks, router_scale_ranks,
      router_scaled_ranks, cands_ranks, seq_len, pools)

    merge_router_candidates[degree, C.TOP_K](
        cands_ranks, per_expert_scale_ptr,
        route_idx_ranks, route_w_ranks, seq_len)

    dispatch_rms_norm[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps, tp=degree,
    ](x_input_ranks, x_normed_ranks, pre_ffn_norm_2_ranks, seq_len, pools)

    build_expert_schedules[degree, experts_per_rank, C.TOP_K](
        route_idx_ranks, route_w_ranks,
        expert_offset_ranks, routes_ranks, seq_len)

    dispatch_phase1_gate_up[
        hidden=C.HIDDEN, gate_up_fused=C.MOE_GATE_UP_FUSED,
        intermediate=C.MOE_INTERMEDIATE,
        experts_per_rank=experts_per_rank, tp=degree,
    ](x_normed_ranks, expert_offset_ranks, routes_ranks,
      experts_gate_up_ranks, gate_scratch_ranks, hidden_bucket_ranks, pools)

    dispatch_phase2_down[
        hidden=C.HIDDEN, intermediate=C.MOE_INTERMEDIATE,
        experts_per_rank=experts_per_rank, tp=degree,
    ](expert_offset_ranks, routes_ranks, hidden_bucket_ranks,
      experts_down_ranks, moe_accum_ranks, moe_out_ranks, seq_len, pools)

    var ar_src = RankBuffers[DType.bfloat16, degree, immut](
        count=seq_len * C.HIDDEN)
    var ar_dst = RankBuffers[DType.bfloat16, degree, MutAnyOrigin](
        count=seq_len * C.HIDDEN)
    for r in range(degree):
        ar_src.ptrs[r] = moe_out_ranks[r].as_immutable()
        ar_dst.ptrs[r] = moe_out_ranks[r]
    dispatch_allreduce[BF16, degree](ar_src, ar_dst, pools)

    gate_scratch_lease^.release()
    moe_accum_lease^.release()
    hidden_bucket_lease^.release()
    routes_lease^.release()
    expert_offset_lease^.release()
    route_w_lease^.release()
    route_idx_lease^.release()
    router_scaled_lease^.release()
    cands_lease^.release()
    x_normed_lease^.release()


def dispatch_ffn[
    P: BurstThreadPool, //, degree: Int,
](
    body: BodyRefs[degree],
    layer_base: Int,
    arena_bases: InlineArray[Int, degree],
    x_main: BF16Ptr,
    x_residual: BF16Ptr,
    seq_len: Int,
    mut scratch: ScratchPool,
    mut pools: HeapMoveArray[P],
):
    comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
    comptime n_eps = C.HIDDEN * C.RMS_NORM_EPS
    comptime intermediate_per_rank = Gemma4Shapes[degree].GateUp.DATA_N
    comptime immut = ImmutOrigin(MutAnyOrigin)

    var pre_ffn_norm_ptr = body.pre_ffn_norm.bound(layer_base).as_ptr()
    var gate_proj_ptr = body.gate_proj.bound(layer_base).as_ptr()
    var up_proj_ptr = body.up_proj.bound(layer_base).as_ptr()
    var down_proj_ptr = body.down_proj.bound(layer_base).as_ptr()
    var post_ffn_norm_1_ptr = body.post_ffn_norm_1.bound(layer_base).as_ptr()
    var post_ffn_norm_2_ptr = body.post_ffn_norm_2.bound(layer_base).as_ptr()
    var post_ffn_norm_ptr = body.post_ffn_norm.bound(layer_base).as_ptr()
    var layer_scalar_ptr = body.layer_scalar.bound(layer_base).as_ptr()

    var gate_lease = scratch.borrow[
        Scalar[DType.bfloat16], C.MAX_SEQ_LEN * intermediate_per_rank]()
    var up_lease = scratch.borrow[
        Scalar[DType.bfloat16], C.MAX_SEQ_LEN * intermediate_per_rank]()
    var dense_out_lease = scratch.borrow[
        Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.HIDDEN]()

    var gate_ranks = NumaPointerArray[DType.bfloat16, degree](
        gate_lease.as_ptr[Scalar[DType.bfloat16]](), arena_bases)
    var up_ranks = NumaPointerArray[DType.bfloat16, degree](
        up_lease.as_ptr[Scalar[DType.bfloat16]](), arena_bases)
    var dense_out_ranks = NumaPointerArray[DType.bfloat16, degree](
        dense_out_lease.as_ptr[Scalar[DType.bfloat16]](), arena_bases)
    var x_main_ranks = NumaPointerArray[DType.bfloat16, degree](
        x_main, arena_bases)
    var x_res_ranks = NumaPointerArray[DType.bfloat16, degree](
        x_residual, arena_bases)
    var pre_ffn_norm_ranks = NumaPointerArray[DType.bfloat16, degree](
        pre_ffn_norm_ptr, arena_bases)
    var gate_proj_ranks = NumaPointerArray[DType.bfloat16, degree](
        gate_proj_ptr, arena_bases)
    var up_proj_ranks = NumaPointerArray[DType.bfloat16, degree](
        up_proj_ptr, arena_bases)
    var down_proj_ranks = NumaPointerArray[DType.bfloat16, degree](
        down_proj_ptr, arena_bases)
    var post_ffn_norm_1_ranks = NumaPointerArray[DType.bfloat16, degree](
        post_ffn_norm_1_ptr, arena_bases)
    var post_ffn_norm_2_ranks = NumaPointerArray[DType.bfloat16, degree](
        post_ffn_norm_2_ptr, arena_bases)
    var post_ffn_norm_ranks = NumaPointerArray[DType.bfloat16, degree](
        post_ffn_norm_ptr, arena_bases)

    dispatch_rms_norm[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps, tp=degree,
    ](x_main_ranks, x_res_ranks, pre_ffn_norm_ranks, seq_len, pools)

    dispatch_gemv[
        rows=intermediate_per_rank, cols=C.HIDDEN, tp=degree,
    ](x_res_ranks, gate_proj_ranks, gate_ranks, pools)

    dispatch_gemv[
        rows=intermediate_per_rank, cols=C.HIDDEN, tp=degree,
    ](x_res_ranks, up_proj_ranks, up_ranks, pools)

    dispatch_gelu_gate_up[
        intermediate=intermediate_per_rank, tp=degree,
    ](gate_ranks, up_ranks, gate_ranks, seq_len, pools)

    dispatch_gemv[
        rows=C.HIDDEN, cols=intermediate_per_rank, tp=degree,
    ](gate_ranks, down_proj_ranks, dense_out_ranks, pools)

    var dense_ar_src = RankBuffers[DType.bfloat16, degree, immut](
        count=seq_len * C.HIDDEN)
    var dense_ar_dst = RankBuffers[DType.bfloat16, degree, MutAnyOrigin](
        count=seq_len * C.HIDDEN)
    for r in range(degree):
        dense_ar_src.ptrs[r] = dense_out_ranks[r].as_immutable()
        dense_ar_dst.ptrs[r] = dense_out_ranks[r]
    dispatch_allreduce[BF16, degree](dense_ar_src, dense_ar_dst, pools)

    dispatch_moe[degree=degree](
        body, layer_base, arena_bases,
        x_main, x_residual, seq_len, scratch, pools)

    dispatch_rms_norm[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps, tp=degree,
    ](dense_out_ranks, dense_out_ranks, post_ffn_norm_1_ranks, seq_len, pools)

    fused_norm_residual_add[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps, tp=degree,
    ](x_res_ranks, dense_out_ranks, dense_out_ranks,
      post_ffn_norm_2_ranks, seq_len, pools)

    fused_norm_residual_add[
        hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps, tp=degree,
    ](dense_out_ranks, x_main_ranks, x_main_ranks,
      post_ffn_norm_ranks, seq_len, pools)

    var ls_value = layer_scalar_ptr[0].cast[DType.float32]()
    dispatch_scalar_mul[
        hidden=C.HIDDEN, tp=degree,
    ](x_main_ranks, x_main_ranks, ls_value, seq_len, pools)

    dense_out_lease^.release()
    up_lease^.release()
    gate_lease^.release()


struct Gemma4[degree: Int, Pool: BurstThreadPool = BurstPool[]](Movable):
    var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var pools: HeapMoveArray[Self.Pool]
    var topology: Gemma4Topology[Self.degree]
    var scratch: ScratchPool
    var arena_bases: InlineArray[Int, Self.degree]

    def __init__(out self,
        var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]],
        var pools: HeapMoveArray[Self.Pool],
        topology: Gemma4Topology[Self.degree],
    ):
        self.arena_bases = InlineArray[Int, Self.degree](uninitialized=True)
        for r in range(Self.degree):
            self.arena_bases[r] = Int(arenas[r].base.value())
        self.topology = topology.bind(self.arena_bases[0])
        self.arenas = arenas^
        self.pools = pools^
        self.scratch = ScratchPool(
            self.topology.arena.scratch_base(),
            calculate_peak_scratch[Self.degree]())

    def model_init(mut self):
        ref topo = self.topology
        comptime width = simd_width_of[DType.float32]()

        comptime inv_sqrt_hidden = 1.0 / sqrt[DType.float32, 1](C.HIDDEN)
        for rank in range(Self.degree):
            var arena_base = self.arena_base(rank)
            var si = 0
            var fi = 0
            for i in range(C.NUM_LAYERS):
                var p: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
                if is_full_layer(i):
                    var lb = topo.full.base(arena_base, fi)
                    p = topo.full.proto.body.router_scale.bound(lb).as_ptr()
                    fi += 1
                else:
                    var lb = topo.sliding.base(arena_base, si)
                    p = topo.sliding.proto.body.router_scale.bound(lb).as_ptr()
                    si += 1
                for j in range(0, C.HIDDEN, width):
                    var lane = p + j
                    var v = lane.load[width=width]().cast[DType.float32]()
                    lane.store((v * inv_sqrt_hidden).cast[DType.bfloat16]())
        print("  router constants baked")

        from kernels.rope import init_rope_table, init_rope_table_partial_strided
        for rank in range(Self.degree):
            var base = self.arena_bases[rank]
            var sl_cos = topo.sliding_rope.cos.bound(base).as_ptr()
            var sl_sin = topo.sliding_rope.sin.bound(base).as_ptr()
            init_rope_table[C.ROPE_HALF_SLIDING, C.MAX_SEQ_LEN](
                sl_cos, sl_sin, 10000.0)
            var fl_cos = topo.full_rope.cos.bound(base).as_ptr()
            var fl_sin = topo.full_rope.sin.bound(base).as_ptr()
            init_rope_table_partial_strided[
                C.ROPE_HALF_FULL, C.MAX_SEQ_LEN // Self.degree,
            ](fl_cos, fl_sin, 1000000.0, C.HEAD_DIM_FULL, rank, Self.degree)
        print("  rope tables initialized")

    def new_kv_cache(self) -> Gemma4KV[Self.degree]:
        ref topo = self.topology
        var kv = Gemma4KV[Self.degree]()
        for layer in range(C.NUM_SLIDING_LAYERS):
            for rank in range(Self.degree):
                var lb = topo.sliding_kv.base(self.arena_base(rank), layer)
                kv.bind_sliding(layer, rank,
                    topo.sliding_kv.proto.k.bound(lb).as_ptr(),
                    topo.sliding_kv.proto.v.bound(lb).as_ptr())
        for layer in range(C.NUM_FULL_LAYERS):
            var lb = topo.full_kv.base(self.arena_bases[0], layer)
            kv.bind_global(layer,
                topo.full_kv.proto.k.bound(lb).as_ptr(),
                topo.full_kv.proto.v.bound(lb).as_ptr())
        return kv^

    def forward(mut self, token_id: Int, pos: Int, mut kv: Gemma4KV[Self.degree]) -> Int:
        # Forward (one step). `pos` = absolute sequence position of the token being processed.
        #
        # Embed:
        #   x ← embed[token_id] · √HIDDEN              # vocab-row sharded tied embedding.
        #                                              # token row owner materializes HIDDEN-wide x.
        #
        # For each layer i in [0, NUM_LAYERS):
        #   residual ← x                               # x is HIDDEN-wide on every rank.
        #   x ← input_norm[i](x)
        #
        #   # ---- Attention. Sliding for !is_full_layer(i), full otherwise. ----
        #   q ← q_proj[i](x)                           # sliding: this rank's Q-head slice.
        #                                              # full: replicated full Q on every rank.
        #   k ← k_proj[i](x)                           # sliding: this rank's KV-head slice (head-sharded).
        #                                              # full:    full KV-dim (K_proj is replicated).
        #   v ← v_proj[i](x)  |  k                     # sliding has its own V proj; full reuses K (K=V).
        #
        #   q, k, v ← per-head RMS norms               # q_norm, k_norm carry learnable scale.
        #                                              # v_norm has no scale (compute only).
        #   q, k    ← rope[i](q, k, pos)               # sliding: full 256-dim rotation, replicated table.
        #                                              # full:    partial 64-of-512 rotation, context table.
        #
        #   cache.write(k, v, pos)                     # sliding: rank-local write into circular slot
        #                                              #          at (pos % SLIDING_WINDOW); every rank
        #                                              #          writes its head-slice.
        #                                              # full:    only the seq-owner of `pos` writes;
        #                                              #          others have already stored prior tokens
        #                                              #          in their context slice.
        #
        #   x ← attn(q, cache_for_layer_i)             # sliding: rank-local Q · K^T over local window,
        #                                              #          softmax, attn · V — fully self-contained.
        #                                              # full:    each rank computes partial attn against
        #                                              #          its context slice; partials combine externally.
        #   x ← o_proj[i](x)                           # row-parallel input → HIDDEN-wide output.
        #
        #   x ← post_attn_norm[i](x)
        #   x ← residual + x
        #   residual ← x                               # snapshot for the dense+MoE parallel branches.
        #
        #   # ---- FFN: dense and MoE in parallel, both fed from `residual`. ----
        #   pre_dense ← pre_ffn_norm[i](residual)
        #   x_dense   ← down[i](gelu(gate[i](pre_dense)) · up[i](pre_dense))
        #                                              # gate/up tensor-sharded on intermediate dim;
        #                                              # down row-parallel back to HIDDEN.
        #
        #   weights, indices ← router[i](residual)     # router_proj replicated; softmax → top-8 of 128,
        #                                              # renormalize, multiply by per_expert_scale.
        #   pre_moe   ← pre_ffn_norm_2[i](residual)
        #   x_moe     ← experts[i](pre_moe, indices, weights)
        #                                              # experts block-sharded across ranks; tokens
        #                                              # dispatch to the rank holding each chosen expert,
        #                                              # results return to all ranks for combination.
        #
        #   x ← post_ffn_norm_1[i](x_dense) + post_ffn_norm_2[i](x_moe)
        #   x ← post_ffn_norm[i](x)
        #   x ← residual + x
        #   x ← x · layer_scalar[i]                    # per-layer learned scalar.
        #
        # Tail:
        #   x      ← final_norm(x)                     # final_norm is replicated.
        #   logits ← x · embed_local.T                 # tied LM head over this rank's vocab slice.
        #   logits ← tanh(logits / LOGIT_SOFTCAP) · LOGIT_SOFTCAP    # cap applied here only.
        #   token  ← distributed argmax | sample(logits)

        ref topo = self.topology
        comptime shard_rows = Gemma4TailShapes[Self.degree].Embed.DATA_N
        comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
        comptime n_eps = C.HIDDEN * C.RMS_NORM_EPS

        var owner = token_id // shard_rows
        var local_row = token_id % shard_rows

        comptime immut = ImmutOrigin(MutAnyOrigin)
        var src = RankBuffers[DType.bfloat16, Self.degree, immut](count=C.HIDDEN)
        var dst = RankBuffers[DType.bfloat16, Self.degree, MutAnyOrigin](count=C.HIDDEN)
        var embed_row = topo.tail.embed.bound(self.arena_bases[owner]).as_ptr()
            + local_row * C.HIDDEN
        for r in range(Self.degree):
            src.ptrs[r] = embed_row.as_immutable()
            dst.ptrs[r] = topo.activations.x_main.bound(self.arena_bases[r]).as_ptr()

        dispatch_broadcast[BF16, Self.degree](src, dst, self.pools, src_rank=owner)

        var x_main = topo.activations.x_main.bound(self.arena_bases[0]).as_ptr()
        var x_residual = topo.activations.x_residual.bound(self.arena_bases[0]).as_ptr()
        var x_main_ranks = NumaPointerArray[DType.bfloat16, Self.degree](
            x_main, self.arena_bases)
        var x_res_ranks = NumaPointerArray[DType.bfloat16, Self.degree](
            x_residual, self.arena_bases)

        comptime embed_scale = sqrt[DType.float32, 1](C.HIDDEN).cast[DType.bfloat16]().cast[DType.float32]()
        dispatch_scalar_mul[
            hidden=C.HIDDEN, tp=Self.degree,
        ](x_main_ranks, x_main_ranks, embed_scale, 1, self.pools)

        comptime immut_ar = ImmutOrigin(MutAnyOrigin)
        var ar_src = RankBuffers[DType.bfloat16, Self.degree, immut_ar](count=C.HIDDEN)
        var ar_dst = RankBuffers[DType.bfloat16, Self.degree, MutAnyOrigin](count=C.HIDDEN)
        for r in range(Self.degree):
            ar_src.ptrs[r] = x_res_ranks[r].as_immutable()
            ar_dst.ptrs[r] = x_res_ranks[r]

        var si = 0
        var fi = 0
        for i in range(C.NUM_LAYERS):
            var body: BodyRefs[Self.degree]
            var lb: Int
            if is_full_layer(i):
                lb = topo.full.base(self.arena_bases[0], fi)
                body = topo.full.proto.body
            else:
                lb = topo.sliding.base(self.arena_bases[0], si)
                body = topo.sliding.proto.body

            var input_norm_w = body.input_norm.bound(lb).as_ptr()
            dispatch_rms_norm[
                hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps, tp=Self.degree,
            ](x_main_ranks, x_res_ranks,
              NumaPointerArray[DType.bfloat16, Self.degree](input_norm_w, self.arena_bases),
              1, self.pools)

            if is_full_layer(i):
                dispatch_full_attention_qkv[degree=Self.degree](
                    topo, self.arena_bases, pos, fi, kv,
                    self.scratch, self.pools)
            else:
                dispatch_sliding_attention_qkv[degree=Self.degree](
                    topo, self.arena_bases, pos, si, kv,
                    self.scratch, self.pools)

            dispatch_allreduce[BF16, Self.degree](ar_src, ar_dst, self.pools)

            var post_attn_w = body.post_attn_norm.bound(lb).as_ptr()
            fused_norm_residual_add[
                hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps, tp=Self.degree,
            ](x_res_ranks, x_main_ranks, x_main_ranks,
              NumaPointerArray[DType.bfloat16, Self.degree](post_attn_w, self.arena_bases),
              1, self.pools)

            dispatch_ffn[degree=Self.degree](
                body, lb, self.arena_bases,
                x_main, x_residual, 1, self.scratch, self.pools)

            if is_full_layer(i):
                fi += 1
            else:
                si += 1

        var final_norm_w = topo.tail.final_norm.bound(self.arena_bases[0]).as_ptr()
        dispatch_rms_norm[
            hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps, tp=Self.degree,
        ](x_main_ranks, x_main_ranks,
          NumaPointerArray[DType.bfloat16, Self.degree](final_norm_w, self.arena_bases),
          1, self.pools)

        comptime vocab_per_rank = C.VOCAB_SIZE // Self.degree
        var logits_lease = self.scratch.borrow[Scalar[DType.bfloat16], vocab_per_rank]()
        var logits_p = logits_lease.as_ptr[Scalar[DType.bfloat16]]()
        var logits_ranks = NumaPointerArray[DType.bfloat16, Self.degree](
            logits_p, self.arena_bases)
        var embed_ptr = topo.tail.embed.bound(self.arena_bases[0]).as_ptr()
        dispatch_gemv[
            rows=vocab_per_rank, cols=C.HIDDEN, tp=Self.degree,
        ](
            x_main_ranks,
            NumaPointerArray[DType.bfloat16, Self.degree](embed_ptr, self.arena_bases),
            logits_ranks, self.pools,
        )

        var best_val = Float32(-1.0e30)
        var best_idx = 0
        for r in range(Self.degree):
            var rank_logits = logits_ranks[r]
            for v in range(vocab_per_rank):
                var val = rank_logits[v].cast[DType.float32]()
                if val > best_val:
                    best_val = val
                    best_idx = r * vocab_per_rank + v

        logits_lease^.release()
        return best_idx

    @staticmethod
    def load(
        dir_path: Path,
        numa: NumaInfo,
        numa_topo: NumaTopology,
        var pools: HeapMoveArray[Self.Pool],
    ) -> Optional[Self]:
        var shards = discover_shards(dir_path)
        if len(shards) == 0:
            print("no safetensors shards found in", String(dir_path))
            return None
        print("found", len(shards), "shard(s)")

        var plan = build_gemma4_plan[Self.degree]()
        var topo = plan.topology

        var size = topo.arena.host_arena_bytes()
        print("allocating", size // (1024 * 1024), "MB x " + String(Self.degree) + " rank(s) (" +
              String(topo.arena.distributed_bytes // (1024 * 1024)) + " MB weights + " +
              String(topo.arena.state_bytes // (1024 * 1024)) + " MB state each)")

        var arenas = HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]](Self.degree)
        var arena_bases = List[Int]()
        for rank in range(Self.degree):
            arenas.push(NumaArena[alignment=DEFAULT_ALIGNMENT](numa_topo[rank], size))
            if not arenas[rank]:
                print("arena allocation failed on node", numa_topo[rank])
                return None
            arena_bases.append(Int(arenas[rank].base.value()))

        var load_result = load_weights_from_descs(plan.descs, shards, arena_bases, numa, numa_topo)
        if not load_result:
            print("weight loading failed")
            return None
        var loaded = load_result.take()
        print("loaded", loaded.bytes_loaded // (1024 * 1024), "MB in", loaded.num_ops, "ops")

        for rank in range(Self.degree):
            _ = arenas[rank].prefault(topo.arena.distributed_bytes, topo.arena.state_bytes)

        var model = Self(arenas^, pools^, topo)
        model.model_init()
        return model^

    def arena_base(self, rank: Int = 0) -> Int:
        return Int(self.arenas[rank].base.value())

    def token_buffer(self) -> UnsafePointer[Scalar[DType.int32], MutAnyOrigin]:
        return UnsafePointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=self.arena_base() + self.topology.arena.scratch_off)
