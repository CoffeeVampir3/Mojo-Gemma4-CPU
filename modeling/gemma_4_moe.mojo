from std.collections import InlineArray
from std.pathlib import Path
from std.memory import UnsafePointer
from std.sys.info import simd_width_of
from simd_math.ops import sqrt

from numa import NumaArena, NumaInfo, NumaTopology
from threading import BurstPool
from threading.threading_traits import BurstThreadPool
from notstdcollections import HeapMoveArray
from kernels.helpers import RankBuffers
from kernels.reductions import broadcast, allreduce
from kernels.rmsnorm import rms_norm_row, rms_norm, rms_norm_qkv_heads, fused_residual_norm_row
from kernels.gemv import gemv_chained_qkv, gemv
from kernels.rope import rope_token, RopeTable
from kernels.kv_tiled_attention import sliding_attention_single_pass, FLASH_PARTIAL_STRIDE
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
from modeling.utilities import dump_bf16, ensure_dir


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
    comptime FullQ       = TensorRowSharded[C.Q_DIM_FULL, C.HIDDEN, Self.D]
    comptime FullK       = Replicated[C.KV_DIM_FULL, C.HIDDEN]
    comptime FullO       = TensorColumnSharded[C.HIDDEN, C.Q_DIM_FULL, Self.D]
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
    and Gemma4Shapes[degree].FullQ.DATA_N
        == TensorParallelRows[C.Q_DIM_FULL, Gemma4Shapes[degree].D]
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
    var router_proj:     TensorRef[BF16, Shape[C.NUM_EXPERTS, C.HIDDEN]]
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
        router_proj     = b.bfs[Shape[NE, H]](e, "router.proj.weight"),
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
    comptime seq = C.MAX_SEQ_LEN
    comptime S = Gemma4Shapes[degree]

    comptime full_q = scratch_block_bytes[seq * S.FullQ.N * bf16]()
    comptime full_kv = scratch_block_bytes[seq * S.FullK.N * bf16]()
    comptime full_attn = max(full_q + 2 * full_kv, 2 * full_q)

    comptime sliding_q = scratch_block_bytes[seq * S.SlidingQ.N * bf16]()
    comptime sliding_kv = scratch_block_bytes[seq * S.SlidingKV.N * bf16]()
    comptime sliding_attn = max(sliding_q + 2 * sliding_kv, 2 * sliding_q)

    comptime gate_up_block = scratch_block_bytes[seq * S.GateUp.N * bf16]()
    comptime hidden_block = scratch_block_bytes[seq * C.HIDDEN * bf16]()
    comptime topk_block = scratch_block_bytes[C.TOP_K * C.HIDDEN * bf16]()
    comptime ffn_dense = 2 * gate_up_block
    comptime ffn_moe = 2 * hidden_block + topk_block

    return max(max(ffn_dense, ffn_moe), max(full_attn, sliding_attn))


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


def sliding_attention_qkv[
    P: BurstThreadPool, //, degree: Int,
](
    x: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    attn: SlidingAttnRefs[degree],
    layer_base: Int,
    pos: Int,
    rank: Int,
    layer_idx: Int,
    rope_table: RopeTable[C.ROPE_HALF_SLIDING],
    ref kv: Gemma4KV[degree],
    mut scratch: ScratchPool,
    mut pool: P,
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

    var q_lease = scratch.borrow[Scalar[DType.bfloat16], q_rows]()
    var k_lease = scratch.borrow[Scalar[DType.bfloat16], kv_rows]()
    var v_lease = scratch.borrow[Scalar[DType.bfloat16], kv_rows]()

    var q_weight = attn.q_proj.bound(layer_base).as_ptr()
    var k_weight = attn.k_proj.bound(layer_base).as_ptr()
    var v_weight = attn.v_proj.bound(layer_base).as_ptr()

    var q_out = q_lease.as_ptr[Scalar[DType.bfloat16]]()
    var k_out = k_lease.as_ptr[Scalar[DType.bfloat16]]()
    var v_out = v_lease.as_ptr[Scalar[DType.bfloat16]]()

    gemv_chained_qkv[q_rows=q_rows, kv_rows=kv_rows, cols=C.HIDDEN](
        x, q_weight, k_weight, v_weight, q_out, k_out, v_out, pool)

    var q_norm_w = attn.q_norm.bound(layer_base).as_ptr()
    var k_norm_w = attn.k_norm.bound(layer_base).as_ptr()

    rms_norm_qkv_heads[
        head_dim=head_dim, sqrt_n=sqrt_hd, n_eps=hd_eps,
        num_q=num_q_heads, num_kv=num_kv_heads,
    ](q_out, k_out, v_out, q_norm_w, k_norm_w, pool)

    var cos_row = rope_table.cos_row(pos)
    var sin_row = rope_table.sin_row(pos)
    rope_token[rope_half, rope_half, num_q_heads, head_dim](q_out, cos_row, sin_row)
    rope_token[rope_half, rope_half, num_kv_heads, head_dim](k_out, cos_row, sin_row)

    var cache_k_dst = kv.sliding.k_row(layer_idx, rank, pos)
    var cache_v_dst = kv.sliding.v_row(layer_idx, rank, pos)
    for i in range(kv_rows):
        cache_k_dst[i] = k_out[i]
        cache_v_dst[i] = v_out[i]

    v_lease^.release()
    k_lease^.release()

    var valid_len = Gemma4KVSliding[degree].valid_len(pos)
    var k_cache_base = kv.sliding.k(layer_idx, rank)
    var v_cache_base = kv.sliding.v(layer_idx, rank)
    comptime kv_cols = kv_rows
    comptime flash_stride = FLASH_PARTIAL_STRIDE[num_q_heads, head_dim]

    var partials_lease = scratch.borrow[Scalar[DType.float32],
        128 * flash_stride]()
    var partials_ptr = partials_lease.as_ptr[Scalar[DType.float32]]()

    sliding_attention_single_pass[
        head_dim=head_dim, num_q=num_q_heads, num_kv=num_kv_heads,
        gqa_ratio=num_q_heads // num_kv_heads, kv_stride=kv_cols,
        window=C.SLIDING_WINDOW](
        q_out, k_cache_base, v_cache_base, q_out,
        partials_ptr, pos, valid_len, pool)

    partials_lease^.release()

    var o_weight = attn.o_proj.bound(layer_base).as_ptr()
    gemv[rows=C.HIDDEN, cols=q_rows](q_out, o_weight, x, pool)

    q_lease^.release()


def full_attention_qkv[
    P: BurstThreadPool, //, degree: Int,
](
    x: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    attn: FullAttnRefs[degree],
    layer_base: Int,
    pos: Int,
    rank: Int,
    layer_idx: Int,
    rope_table: RopeTable[C.ROPE_HALF_FULL],
    ref kv: Gemma4KV[degree],
    mut scratch: ScratchPool,
    mut pool: P,
):
    comptime S = Gemma4Shapes[degree]
    comptime q_rows = S.FullQ.DATA_N
    comptime k_rows = S.FullK.DATA_N
    comptime head_dim = C.HEAD_DIM_FULL
    comptime num_q_heads = q_rows // head_dim
    comptime num_kv_heads = k_rows // head_dim
    comptime sqrt_hd = sqrt[DType.float32, 1](head_dim)
    comptime hd_eps = head_dim * C.RMS_NORM_EPS
    comptime rope_half = C.ROPE_HALF_FULL
    comptime pair_stride = head_dim // 2

    var q_lease = scratch.borrow[Scalar[DType.bfloat16], q_rows]()
    var k_lease = scratch.borrow[Scalar[DType.bfloat16], k_rows]()
    var v_lease = scratch.borrow[Scalar[DType.bfloat16], k_rows]()

    var q_weight = attn.q_proj.bound(layer_base).as_ptr()
    var k_weight = attn.k_proj.bound(layer_base).as_ptr()

    var q_out = q_lease.as_ptr[Scalar[DType.bfloat16]]()
    var k_out = k_lease.as_ptr[Scalar[DType.bfloat16]]()
    var v_out = v_lease.as_ptr[Scalar[DType.bfloat16]]()

    gemv[rows=q_rows, cols=C.HIDDEN](x, q_weight, q_out, pool)
    gemv[rows=k_rows, cols=C.HIDDEN](x, k_weight, k_out, pool)

    var q_norm_w = attn.q_norm.bound(layer_base).as_ptr()
    var k_norm_w = attn.k_norm.bound(layer_base).as_ptr()

    rms_norm[hidden=head_dim, sqrt_n=sqrt_hd, n_eps=hd_eps, scaled=False](
        k_out, v_out, k_norm_w, num_kv_heads, pool)
    rms_norm[hidden=head_dim, sqrt_n=sqrt_hd, n_eps=hd_eps](
        q_out, q_out, q_norm_w, num_q_heads, pool)
    rms_norm[hidden=head_dim, sqrt_n=sqrt_hd, n_eps=hd_eps](
        k_out, k_out, k_norm_w, num_kv_heads, pool)

    var cos_row = rope_table.cos_row(pos)
    var sin_row = rope_table.sin_row(pos)
    rope_token[rope_half, pair_stride, num_q_heads, head_dim](q_out, cos_row, sin_row)
    rope_token[rope_half, pair_stride, num_kv_heads, head_dim](k_out, cos_row, sin_row)

    if Gemma4KVGlobal[degree].owner(pos) == rank:
        var cache_k_dst = kv.full.k_row(layer_idx, rank, pos)
        var cache_v_dst = kv.full.v_row(layer_idx, rank, pos)
        for i in range(k_rows):
            cache_k_dst[i] = k_out[i]
            cache_v_dst[i] = v_out[i]

    # TODO: full attention + o_proj

    v_lease^.release()
    k_lease^.release()
    q_lease^.release()


struct Gemma4[degree: Int, Pool: BurstThreadPool = BurstPool[]](Movable):
    var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var pools: HeapMoveArray[Self.Pool]
    var topology: Gemma4Topology[Self.degree]
    var sliding_rope: RopeTable[C.ROPE_HALF_SLIDING]
    var full_rope: RopeTable[C.ROPE_HALF_FULL]

    def __init__(out self,
        var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]],
        var pools: HeapMoveArray[Self.Pool],
        topology: Gemma4Topology[Self.degree],
    ):
        self.topology = topology.bind(Int(arenas[0].base.value()))
        self.arenas = arenas^
        self.pools = pools^
        self.sliding_rope = RopeTable[C.ROPE_HALF_SLIDING](
            UnsafePointer[Scalar[DType.float32], MutAnyOrigin].unsafe_dangling(),
            UnsafePointer[Scalar[DType.float32], MutAnyOrigin].unsafe_dangling(), 0)
        self.full_rope = RopeTable[C.ROPE_HALF_FULL](
            UnsafePointer[Scalar[DType.float32], MutAnyOrigin].unsafe_dangling(),
            UnsafePointer[Scalar[DType.float32], MutAnyOrigin].unsafe_dangling(), 0)

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

        comptime embed_scale = sqrt[DType.float32, 1](C.HIDDEN).cast[DType.bfloat16]().cast[DType.float32]()
        comptime shard_rows = Gemma4TailShapes[Self.degree].Embed.DATA_N
        for rank in range(Self.degree):
            var p = topo.tail.embed.bound(self.arena_base(rank)).as_ptr()
            for row in range(shard_rows):
                var row_ptr = p + row * C.HIDDEN
                for j in range(0, C.HIDDEN, width):
                    var lane = row_ptr + j
                    var v = lane.load[width=width]().cast[DType.float32]()
                    lane.store((v * embed_scale).cast[DType.bfloat16]())
        print("  embedding scale baked")

        from kernels.rope import init_rope_table, init_rope_table_partial
        var base = self.arena_base(0)
        var sl_cos = topo.sliding_rope.cos.bound(base).as_ptr()
        var sl_sin = topo.sliding_rope.sin.bound(base).as_ptr()
        init_rope_table[C.ROPE_HALF_SLIDING, C.MAX_SEQ_LEN](sl_cos, sl_sin, 10000.0)
        self.sliding_rope = RopeTable[C.ROPE_HALF_SLIDING](sl_cos, sl_sin, C.MAX_SEQ_LEN)

        var fl_cos = topo.full_rope.cos.bound(base).as_ptr()
        var fl_sin = topo.full_rope.sin.bound(base).as_ptr()
        init_rope_table_partial[C.ROPE_HALF_FULL, C.MAX_SEQ_LEN](
            fl_cos, fl_sin, 1000000.0, C.HEAD_DIM_FULL)
        self.full_rope = RopeTable[C.ROPE_HALF_FULL](fl_cos, fl_sin, C.MAX_SEQ_LEN)
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
            for rank in range(Self.degree):
                var lb = topo.full_kv.base(self.arena_base(rank), layer)
                kv.bind_global(layer, rank,
                    topo.full_kv.proto.k.bound(lb).as_ptr(),
                    topo.full_kv.proto.v.bound(lb).as_ptr())
        return kv^

    def forward(mut self, token_id: Int, pos: Int, mut kv: Gemma4KV[Self.degree], dump_dir: Optional[Path] = None):
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
        #   q ← q_proj[i](x)                           # this rank's Q-head slice.
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
        var embed_row = topo.tail.embed.bound(self.arena_base(owner)).as_ptr()
            + local_row * C.HIDDEN
        for r in range(Self.degree):
            src.ptrs[r] = embed_row.as_immutable()
            dst.ptrs[r] = topo.activations.x_main.bound(self.arena_base(r)).as_ptr()

        broadcast[BF16, Self.degree](src, dst, self.pools, src_rank=owner)

        var si = 0
        var fi = 0
        for i in range(C.NUM_LAYERS):
            for rank in range(Self.degree):
                var base = self.arena_base(rank)
                var x_main = topo.activations.x_main.bound(base).as_ptr()
                var x_residual = topo.activations.x_residual.bound(base).as_ptr()

                var norm_weight: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
                if is_full_layer(i):
                    var lb = topo.full.base(base, fi)
                    norm_weight = topo.full.proto.body.input_norm.bound(lb).as_ptr()
                else:
                    var lb = topo.sliding.base(base, si)
                    norm_weight = topo.sliding.proto.body.input_norm.bound(lb).as_ptr()

                rms_norm_row[C.HIDDEN, sqrt_n, n_eps](x_main, x_residual, norm_weight)

            for rank in range(Self.degree):
                var base = self.arena_base(rank)
                var x_residual = topo.activations.x_residual.bound(base).as_ptr()
                var scratch = ScratchPool(
                    topo.arena.scratch_base() + (base - topo.arena.base),
                    calculate_peak_scratch[Self.degree]())

                if is_full_layer(i):
                    var lb = topo.full.base(base, fi)
                    full_attention_qkv[degree=Self.degree](
                        x_residual, topo.full.proto.attn, lb,
                        pos, rank, fi, self.full_rope, kv,
                        scratch, self.pools[rank])
                else:
                    var lb = topo.sliding.base(base, si)
                    sliding_attention_qkv[degree=Self.degree](
                        x_residual, topo.sliding.proto.attn, lb,
                        pos, rank, si, self.sliding_rope, kv,
                        scratch, self.pools[rank])

            # Allreduce O-proj partials across ranks (x_residual holds partials)
            comptime immut_ar = ImmutOrigin(MutAnyOrigin)
            var ar_src = RankBuffers[DType.bfloat16, Self.degree, immut_ar](count=C.HIDDEN)
            var ar_dst = RankBuffers[DType.bfloat16, Self.degree, MutAnyOrigin](count=C.HIDDEN)
            for r in range(Self.degree):
                var base = self.arena_base(r)
                var p = topo.activations.x_residual.bound(base).as_ptr()
                ar_src.ptrs[r] = p.as_immutable()
                ar_dst.ptrs[r] = p
            allreduce[BF16, Self.degree](ar_src, ar_dst, self.pools)

            # Post-attention: fused residual add + norm for next stage
            # x_residual now holds allreduced attention output
            # x_main holds the residual from before input_norm
            # Result: x_main = residual + attn_out, x_residual = norm(x_main)
            for rank in range(Self.degree):
                var base = self.arena_base(rank)
                var x_main = topo.activations.x_main.bound(base).as_ptr()
                var x_residual = topo.activations.x_residual.bound(base).as_ptr()

                var post_attn_w: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
                if is_full_layer(i):
                    var lb = topo.full.base(base, fi)
                    post_attn_w = topo.full.proto.body.post_attn_norm.bound(lb).as_ptr()
                else:
                    var lb = topo.sliding.base(base, si)
                    post_attn_w = topo.sliding.proto.body.post_attn_norm.bound(lb).as_ptr()

                fused_residual_norm_row[C.HIDDEN, sqrt_n, n_eps](
                    x_residual, x_main, x_main, x_residual, post_attn_w)

            # After this: x_main = residual + attn_out (new residual)
            #             x_residual = post_attn_norm(x_main) (ready for FFN)
            # TODO: FFN (dense MLP + MoE)

            if is_full_layer(i):
                fi += 1
            else:
                si += 1

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
