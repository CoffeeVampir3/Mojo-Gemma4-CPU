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
from modeling.temporal_scratch import (
    ScratchBuffer, ScratchIsland, ScratchPhase,
    TemporalLogitsView, TemporalScratchPool, aggregate_scratch_peak,
)
from modeling.kv_cache import Gemma4KV, Gemma4KVSliding, Gemma4KVGlobal

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
    Gemma4BaseConfig, is_full_layer,
)
from modeling.modeling_common import (
    Repeated, ArenaLayout, BF16Ptr,
)
from modeling.slot import (
    Slot, SlotGroup, stamp_offsets, emit_descs,
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
struct Gemma4Topology[degree: Int](Copyable, ImplicitlyCopyable):
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
struct Gemma4SlidingScratch[degree: Int](
    ScratchIsland, Copyable, ImplicitlyCopyable
):
    comptime S = Gemma4Shapes[Self.degree]
    comptime q_rows = Self.S.SlidingQ.DATA_N
    comptime kv_rows = Self.S.SlidingKV.DATA_N
    comptime head_dim = C.HEAD_DIM_SLIDING
    comptime num_q_heads = Self.q_rows // Self.head_dim
    comptime flash_stride = FLASH_PARTIAL_STRIDE[
        Self.num_q_heads, Self.head_dim,
    ]

    comptime gemv_qkv = 1
    comptime rms_norm_qkv = 2
    comptime rope_cache_write = 3
    comptime flash = 4
    comptime merge_partials = 5
    comptime o_proj = 6

    var q_band: ScratchPhase[Self.gemv_qkv, Self.o_proj]
    var q: ScratchBuffer[
        Scalar[DType.bfloat16], C.MAX_SEQ_LEN * Self.q_rows,
    ]

    var kv_band: ScratchPhase[Self.gemv_qkv, Self.rope_cache_write]
    var kv: ScratchBuffer[
        Scalar[DType.bfloat16], C.MAX_SEQ_LEN * Self.kv_rows * 2,
    ]

    var partials_band: ScratchPhase[Self.flash, Self.merge_partials]
    var partials: ScratchBuffer[
        Scalar[DType.float32], 128 * Self.flash_stride,
    ]


@fieldwise_init
struct Gemma4FullScratch[degree: Int](
    ScratchIsland, Copyable, ImplicitlyCopyable
):
    comptime S = Gemma4Shapes[Self.degree]
    comptime q_rows = Self.S.FullQ.DATA_N
    comptime k_rows = Self.S.FullK.DATA_N
    comptime local_q_rows = Self.S.FullO.DATA_M
    comptime head_dim = C.HEAD_DIM_FULL
    comptime num_q_heads = Self.q_rows // Self.head_dim
    comptime partial_stride = PARTIAL_STRIDE[
        Self.num_q_heads, Self.head_dim,
    ]

    comptime gemv_q = 1
    comptime gemv_kv = 2
    comptime rms_norm_qkv = 3
    comptime rope_cache_write = 4
    comptime flash = 5
    comptime merge_partials = 6
    comptime o_proj = 7

    var q_band: ScratchPhase[Self.gemv_q, Self.flash]
    var q: ScratchBuffer[
        Scalar[DType.bfloat16], C.MAX_SEQ_LEN * Self.q_rows,
    ]

    var kv_band: ScratchPhase[Self.gemv_kv, Self.rope_cache_write]
    var kv: ScratchBuffer[
        Scalar[DType.bfloat16], C.MAX_SEQ_LEN * Self.k_rows * 2,
    ]

    var partials_band: ScratchPhase[Self.flash, Self.merge_partials]
    var partials: ScratchBuffer[
        Scalar[DType.float32], 128 * Self.partial_stride,
    ]

    var q_local_band: ScratchPhase[Self.merge_partials, Self.o_proj]
    var q_local: ScratchBuffer[
        Scalar[DType.bfloat16], C.MAX_SEQ_LEN * Self.local_q_rows,
    ]


@fieldwise_init
struct Gemma4FfnMoeScratch[degree: Int](
    ScratchIsland, Copyable, ImplicitlyCopyable
):
    comptime S = Gemma4Shapes[Self.degree]
    comptime intermediate_per_rank = Self.S.GateUp.DATA_N
    comptime experts_per_rank = C.NUM_EXPERTS // Self.degree

    comptime ffn_rms_norm = 1
    comptime gemv_gate = 2
    comptime gemv_up = 3
    comptime gelu_gate_up = 4
    comptime router_sharded = 5
    comptime merge_cands = 6
    comptime moe_rms_norm = 7
    comptime build_schedules = 8
    comptime phase1_gate_up = 9
    comptime phase2_down = 10
    comptime moe_allreduce = 11
    comptime gemv_dense = 12
    comptime allreduce_dense = 13
    comptime post_norm_1 = 14
    comptime post_norm_2 = 15
    comptime post_norm_3 = 16

    var ffn_gate_band: ScratchPhase[Self.gemv_gate, Self.gemv_dense]
    var ffn_gate: ScratchBuffer[
        Scalar[DType.bfloat16],
        C.MAX_SEQ_LEN * Self.intermediate_per_rank,
    ]

    var ffn_up_band: ScratchPhase[Self.gemv_up, Self.gelu_gate_up]
    var ffn_up: ScratchBuffer[
        Scalar[DType.bfloat16],
        C.MAX_SEQ_LEN * Self.intermediate_per_rank,
    ]

    var router_workspace: ScratchPhase[
        Self.router_sharded, Self.router_sharded,
    ]
    var moe_router_scaled: ScratchBuffer[
        Scalar[DType.float32], MAX_WORKERS * C.HIDDEN,
    ]

    var router_cands: ScratchPhase[Self.router_sharded, Self.merge_cands]
    var moe_cands: ScratchBuffer[RouterCandidate, C.MAX_SEQ_LEN * C.TOP_K]

    var router_products: ScratchPhase[Self.merge_cands, Self.build_schedules]
    var moe_route_idx: ScratchBuffer[
        Scalar[DType.int32], C.MAX_SEQ_LEN * C.TOP_K,
    ]
    var moe_route_w: ScratchBuffer[
        Scalar[DType.float32], C.MAX_SEQ_LEN * C.TOP_K,
    ]

    var expert_input: ScratchPhase[Self.moe_rms_norm, Self.phase1_gate_up]
    var moe_x_normed: ScratchBuffer[
        Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.HIDDEN,
    ]

    var schedule_products: ScratchPhase[
        Self.build_schedules, Self.phase2_down,
    ]
    var moe_expert_offset: ScratchBuffer[
        Scalar[DType.int32], Self.experts_per_rank + 1,
    ]
    var moe_routes: ScratchBuffer[SparseRoute, C.MAX_SEQ_LEN * C.TOP_K]

    var hidden_bucket: ScratchPhase[Self.phase1_gate_up, Self.phase2_down]
    var moe_hidden_bucket: ScratchBuffer[
        Scalar[DType.bfloat16],
        C.MAX_SEQ_LEN * C.TOP_K * C.MOE_INTERMEDIATE,
    ]

    var phase1_workspace: ScratchPhase[
        Self.phase1_gate_up, Self.phase1_gate_up,
    ]
    var moe_gate_scratch: ScratchBuffer[
        Scalar[DType.float32],
        MAX_WORKERS * PHASE1_MR * 2 * PHASE1_TILE_J,
    ]

    var phase2_accum: ScratchPhase[Self.phase2_down, Self.phase2_down]
    var moe_accum: ScratchBuffer[
        Scalar[DType.float32], C.MAX_SEQ_LEN * C.HIDDEN,
    ]

    var dense_band: ScratchPhase[Self.gemv_dense, Self.post_norm_3]
    var ffn_dense_out: ScratchBuffer[
        Scalar[DType.bfloat16], C.MAX_SEQ_LEN * C.HIDDEN,
    ]


@fieldwise_init
struct Gemma4HeadScratch[degree: Int](
    ScratchIsland, Copyable, ImplicitlyCopyable
):
    comptime final_norm = 1
    comptime gemv_logits = 2
    comptime returned_to_caller = 3
    comptime vocab_per_rank = C.VOCAB_SIZE // Self.degree

    var logits_band: ScratchPhase[Self.gemv_logits, Self.returned_to_caller]
    var logits: ScratchBuffer[
        Scalar[DType.bfloat16], Self.vocab_per_rank,
    ]


@fieldwise_init
struct Gemma4ForwardScratch[degree: Int](Copyable, ImplicitlyCopyable):
    var sliding: Gemma4SlidingScratch[Self.degree]
    var full: Gemma4FullScratch[Self.degree]
    var ffn: Gemma4FfnMoeScratch[Self.degree]
    var head: Gemma4HeadScratch[Self.degree]


def calculate_peak_scratch[degree: Int]() -> Int:
    return aggregate_scratch_peak[Gemma4ForwardScratch[degree]]()


comptime Gemma4ScratchPool[degree: Int] = TemporalScratchPool[
    calculate_peak_scratch[degree](),
]


def build_gemma4_plan[degree: Int](mut descs: List[WeightDesc]) -> Gemma4Topology[degree]:
    comptime assert SlidingAttentionContract[degree], "sliding attention distribution contract failed"
    comptime assert FullAttentionContract[degree], "full attention distribution contract failed"
    comptime assert DenseMlpContract[degree], "dense MLP distribution contract failed"
    comptime assert MoeContract[degree], "MoE distribution contract failed"
    comptime assert LmHeadContract[degree], "LM head distribution contract failed"

    var sl_proto = SlidingLayerRefs[degree]()
    var sl_stride = stamp_offsets(sl_proto)
    var fl_proto = FullLayerRefs[degree]()
    var fl_stride = stamp_offsets(fl_proto)

    var sl_off = 0
    var fl_off = sl_off + C.NUM_SLIDING_LAYERS * sl_stride
    var distributed = fl_off + C.NUM_FULL_LAYERS * fl_stride

    var si = 0
    var fi = 0
    for i in range(C.NUM_LAYERS):
        var prefix = "model.language_model.layers." + String(i) + "."
        if is_full_layer(i):
            _ = emit_descs[FullLayerRefs[degree]](
                prefix, fl_off + fi * fl_stride, descs)
            fi += 1
        else:
            _ = emit_descs[SlidingLayerRefs[degree]](
                prefix, sl_off + si * sl_stride, descs)
            si += 1

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

    var scratch_cap = calculate_peak_scratch[degree]()
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
    return Gemma4Topology[degree](
        arena=arena,
        sliding=Repeated[SlidingLayerRefs[degree]](sl_proto, sl_off, sl_stride, C.NUM_SLIDING_LAYERS),
        full=Repeated[FullLayerRefs[degree]](fl_proto, fl_off, fl_stride, C.NUM_FULL_LAYERS),
        sliding_kv=sliding_kv, full_kv=full_kv,
        activations=activations,
        sliding_rope=sliding_rope, full_rope=full_rope,
        tail=tail)


def dispatch_sliding_attention_qkv[
    P: BurstThreadPool, //, degree: Int,
](
    topo: Gemma4Topology[degree],
    arena_bases: InlineArray[Int, degree],
    pos: Int,
    layer_idx: Int,
    ref kv: Gemma4KV[degree],
    mut scratch: Gemma4ScratchPool[degree],
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

    var q_out = scratch.slot[Gemma4SlidingScratch[degree], "q"]()
    var k_out = scratch.slot[Gemma4SlidingScratch[degree], "kv"]()
    var v_out = k_out + kv_rows

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

    var partials_ptr = scratch.slot[
        Gemma4SlidingScratch[degree], "partials",
    ]()

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

    dispatch_gemv[rows=C.HIDDEN, cols=q_rows, tp=degree](
        q_outs,
        NumaPointerArray[DType.bfloat16, degree](attn.o_proj.bound(lb).as_ptr(), arena_bases),
        xs, pools)


def dispatch_full_attention_qkv[
    P: BurstThreadPool, //, degree: Int,
](
    topo: Gemma4Topology[degree],
    arena_bases: InlineArray[Int, degree],
    pos: Int,
    layer_idx: Int,
    ref kv: Gemma4KV[degree],
    mut scratch: Gemma4ScratchPool[degree],
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

    var q_out = scratch.slot[Gemma4FullScratch[degree], "q"]()
    var k_out = scratch.slot[Gemma4FullScratch[degree], "kv"]()
    var v_out = k_out + k_rows

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

    var q_local = scratch.slot[Gemma4FullScratch[degree], "q_local"]()
    var q_local_outs = NumaPointerArray[DType.bfloat16, degree](q_local, arena_bases)

    var partials_ptr = scratch.slot[Gemma4FullScratch[degree], "partials"]()
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

    dispatch_gemv[rows=C.HIDDEN, cols=local_q_rows, tp=degree](
        q_local_outs,
        NumaPointerArray[DType.bfloat16, degree](attn.o_proj.bound(lb).as_ptr(), arena_bases),
        xs, pools)


def dispatch_moe[
    P: BurstThreadPool, //, degree: Int,
](
    body: BodyRefs[degree],
    layer_base: Int,
    arena_bases: InlineArray[Int, degree],
    x_input: BF16Ptr,
    moe_out: BF16Ptr,
    seq_len: Int,
    mut scratch: Gemma4ScratchPool[degree],
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

    var x_normed = scratch.slot[Gemma4FfnMoeScratch[degree], "moe_x_normed"]()
    var cands = scratch.slot[Gemma4FfnMoeScratch[degree], "moe_cands"]()
    var router_scaled = scratch.slot[
        Gemma4FfnMoeScratch[degree], "moe_router_scaled",
    ]()
    var route_idx = scratch.slot[Gemma4FfnMoeScratch[degree], "moe_route_idx"]()
    var route_w = scratch.slot[Gemma4FfnMoeScratch[degree], "moe_route_w"]()
    var expert_offset = scratch.slot[
        Gemma4FfnMoeScratch[degree], "moe_expert_offset",
    ]()
    var routes = scratch.slot[Gemma4FfnMoeScratch[degree], "moe_routes"]()
    var hidden_bucket = scratch.slot[
        Gemma4FfnMoeScratch[degree], "moe_hidden_bucket",
    ]()
    var moe_accum = scratch.slot[Gemma4FfnMoeScratch[degree], "moe_accum"]()
    var gate_scratch = scratch.slot[
        Gemma4FfnMoeScratch[degree], "moe_gate_scratch",
    ]()

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
        x_normed, arena_bases)
    var cands_ranks = NumaTypedPointerArray[RouterCandidate, degree](
        cands, arena_bases)
    var router_scaled_ranks = NumaPointerArray[DType.float32, degree](
        router_scaled, arena_bases)
    var route_idx_ranks = NumaPointerArray[DType.int32, degree](
        route_idx, arena_bases)
    var route_w_ranks = NumaPointerArray[DType.float32, degree](
        route_w, arena_bases)
    var expert_offset_ranks = NumaPointerArray[DType.int32, degree](
        expert_offset, arena_bases)
    var routes_ranks = NumaTypedPointerArray[SparseRoute, degree](
        routes, arena_bases)
    var hidden_bucket_ranks = NumaPointerArray[DType.bfloat16, degree](
        hidden_bucket, arena_bases)
    var moe_accum_ranks = NumaPointerArray[DType.float32, degree](
        moe_accum, arena_bases)
    var gate_scratch_ranks = NumaPointerArray[DType.float32, degree](
        gate_scratch, arena_bases)

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


def dispatch_ffn[
    P: BurstThreadPool, //, degree: Int,
](
    body: BodyRefs[degree],
    layer_base: Int,
    arena_bases: InlineArray[Int, degree],
    x_main: BF16Ptr,
    x_residual: BF16Ptr,
    seq_len: Int,
    mut scratch: Gemma4ScratchPool[degree],
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

    var gate = scratch.slot[Gemma4FfnMoeScratch[degree], "ffn_gate"]()
    var up = scratch.slot[Gemma4FfnMoeScratch[degree], "ffn_up"]()
    var dense_out = scratch.slot[
        Gemma4FfnMoeScratch[degree], "ffn_dense_out",
    ]()

    var gate_ranks = NumaPointerArray[DType.bfloat16, degree](
        gate, arena_bases)
    var up_ranks = NumaPointerArray[DType.bfloat16, degree](
        up, arena_bases)
    var dense_out_ranks = NumaPointerArray[DType.bfloat16, degree](
        dense_out, arena_bases)
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

    dispatch_moe[degree=degree](
        body, layer_base, arena_bases,
        x_main, x_residual, seq_len, scratch, pools)

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


struct Gemma4[degree: Int, Pool: BurstThreadPool = BurstPool[]](Movable):
    var arenas: HeapMoveArray[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var pools: HeapMoveArray[Self.Pool]
    var topology: Gemma4Topology[Self.degree]
    var scratch: Gemma4ScratchPool[Self.degree]
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
        self.scratch = Gemma4ScratchPool[Self.degree](
            self.topology.arena.scratch_base())

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

    def forward(
        mut self, token_id: Int, pos: Int, mut kv: Gemma4KV[Self.degree],
    ) -> TemporalLogitsView[C.VOCAB_SIZE, Self.degree]:
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
        var tail_base_owner = topo.tail.base(self.arena_bases[owner], 0)
        var embed_row = topo.tail.proto.embed.bound(tail_base_owner).as_ptr()
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

        var tail_base = topo.tail.base(self.arena_bases[0], 0)
        var final_norm_w = topo.tail.proto.final_norm.bound(tail_base).as_ptr()
        dispatch_rms_norm[
            hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps, tp=Self.degree,
        ](x_main_ranks, x_main_ranks,
          NumaPointerArray[DType.bfloat16, Self.degree](final_norm_w, self.arena_bases),
          1, self.pools)

        comptime vocab_per_rank = C.VOCAB_SIZE // Self.degree
        var logits_p = self.scratch.slot[
            Gemma4HeadScratch[Self.degree], "logits",
        ]()
        var logits_ranks = NumaPointerArray[DType.bfloat16, Self.degree](
            logits_p, self.arena_bases)
        var embed_ptr = topo.tail.proto.embed.bound(tail_base).as_ptr()
        dispatch_gemv[
            rows=vocab_per_rank, cols=C.HIDDEN, tp=Self.degree,
        ](
            x_main_ranks,
            NumaPointerArray[DType.bfloat16, Self.degree](embed_ptr, self.arena_bases),
            logits_ranks, self.pools,
        )

        return TemporalLogitsView[C.VOCAB_SIZE, Self.degree](
            logits_p, self.arena_bases)

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

        var descs = List[WeightDesc]()
        var topo = build_gemma4_plan[Self.degree](descs)

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

        var load_result = load_weights_from_descs(descs, shards, arena_bases, numa, numa_topo)
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
