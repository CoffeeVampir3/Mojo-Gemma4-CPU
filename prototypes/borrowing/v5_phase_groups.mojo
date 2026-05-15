from std.reflection import reflect
from std.memory import alloc, free, Layout
from std.collections import InlineArray
from std.sys.info import size_of
from std.builtin.rebind import downcast


comptime ALIGN = 64
comptime MAX_BUFS = 32


@always_inline
def aligned64(b: Int) -> Int:
    return ((b + ALIGN - 1) // ALIGN) * ALIGN


trait Lifetime:
    """A lifetime window over phase ordinals. PhaseGroup is the canonical
    impl, but any type carrying FIRST/LAST satisfies."""
    comptime FIRST: Int
    comptime LAST:  Int


@fieldwise_init
struct PhaseGroup[first: Int, last: Int](
    Lifetime, Copyable, ImplicitlyCopyable
):
    """A named lifetime band. Multiple buffers can share one PhaseGroup
    when they coexist in the same window; singleton groups are valid."""
    comptime FIRST = Self.first
    comptime LAST  = Self.last


trait BufferLike:
    comptime Element: AnyType
    comptime SIZE:    Int
    comptime FIRST:   Int
    comptime LAST:    Int


@fieldwise_init
struct Buf[T: AnyType, count: Int, during: Lifetime](
    BufferLike, Copyable, ImplicitlyCopyable
):
    comptime Element = Self.T
    comptime SIZE    = aligned64(Self.count * size_of[Self.T]())
    comptime FIRST   = Self.during.FIRST
    comptime LAST    = Self.during.LAST


@fieldwise_init
struct IslandPlan(Copyable, ImplicitlyCopyable):
    var offsets: InlineArray[Int, MAX_BUFS]
    var peak: Int
    var count: Int


def derive_plan[T: AnyType]() -> IslandPlan:
    var sizes  = InlineArray[Int, MAX_BUFS](fill=0)
    var firsts = InlineArray[Int, MAX_BUFS](fill=0)
    var lasts  = InlineArray[Int, MAX_BUFS](fill=0)
    var n = 0

    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, BufferLike):
            sizes[n]  = FT.SIZE
            firsts[n] = FT.FIRST
            lasts[n]  = FT.LAST
            n += 1

    var order = InlineArray[Int, MAX_BUFS](fill=0)
    for k in range(n):
        order[k] = k
    for i in range(n):
        var best = i
        for j in range(i + 1, n):
            if sizes[order[j]] > sizes[order[best]]:
                best = j
        var tmp = order[i]
        order[i] = order[best]
        order[best] = tmp

    var offsets = InlineArray[Int, MAX_BUFS](fill=0)
    var placed  = InlineArray[Bool, MAX_BUFS](fill=False)
    var peak = 0
    for k in range(n):
        var idx = order[k]
        var x = 0
        var stable = False
        while not stable:
            stable = True
            for j in range(n):
                if not placed[j]:
                    continue
                if firsts[idx] > lasts[j] or lasts[idx] < firsts[j]:
                    continue
                var jl = offsets[j]
                var jh = offsets[j] + sizes[j]
                if x < jh and jl < x + sizes[idx]:
                    x = jh
                    stable = False
                    break
        offsets[idx] = x
        placed[idx] = True
        if x + sizes[idx] > peak:
            peak = x + sizes[idx]

    return IslandPlan(offsets=offsets, peak=peak, count=n)


trait Island:
    comptime PLAN: IslandPlan = derive_plan[Self]()


def aggregate_peak[T: AnyType]() -> Int:
    var m = 0
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, Island):
            if FT.PLAN.peak > m:
                m = FT.PLAN.peak
    return m


comptime HIDDEN = 2816
comptime SEQ = 4096
comptime DEGREE = 4
comptime TOP_K = 8
comptime VOCAB_SIZE = 262144
comptime MAX_WORKERS = 128
comptime PHASE1_MR = 4
comptime PHASE1_TILE_J = 64
comptime EXPERTS_PER_RANK = 32
comptime MOE_INTERMEDIATE = 704
comptime SLIDING_Q_N  = 1024
comptime SLIDING_KV_N = 512
comptime FULL_Q_N     = 8192
comptime FULL_K_N     = 1024
comptime FULL_O_M     = 2048
comptime GATE_UP_N    = 528
comptime VOCAB_PER_RANK = VOCAB_SIZE // DEGREE


def flash_stride(num_q: Int, head_dim: Int) -> Int:
    return ((num_q * head_dim + num_q + num_q) * 4 + 63) // 64 * 16


comptime BF16 = Scalar[DType.bfloat16]
comptime F32  = Scalar[DType.float32]
comptime I32  = Scalar[DType.int32]
comptime U8   = UInt8


@fieldwise_init
struct SlidingAttention(Island, Copyable, ImplicitlyCopyable):
    comptime gemv_qkv         = 1
    comptime rms_norm_qkv     = 2
    comptime rope_cache_write = 3
    comptime flash            = 4
    comptime merge_partials   = 5
    comptime o_proj           = 6

    comptime q_lifetime  = PhaseGroup[Self.gemv_qkv, Self.o_proj]
    comptime kv_lifetime = PhaseGroup[Self.gemv_qkv, Self.rope_cache_write]
    comptime p_lifetime  = PhaseGroup[Self.flash,    Self.merge_partials]

    var q:  Buf[BF16, SEQ * SLIDING_Q_N,          during=Self.q_lifetime]
    var kv: Buf[BF16, SEQ * SLIDING_KV_N * 2,     during=Self.kv_lifetime]
    var p:  Buf[F32,  128 * flash_stride(4, 256), during=Self.p_lifetime]


@fieldwise_init
struct FullAttention(Island, Copyable, ImplicitlyCopyable):
    comptime gemv_q           = 1
    comptime gemv_kv          = 2
    comptime rms_norm_qkv     = 3
    comptime rope_cache_write = 4
    comptime flash            = 5
    comptime merge_partials   = 6
    comptime o_proj           = 7

    comptime q_lifetime       = PhaseGroup[Self.gemv_q,         Self.flash]
    comptime kv_lifetime      = PhaseGroup[Self.gemv_kv,        Self.rope_cache_write]
    comptime p_lifetime       = PhaseGroup[Self.flash,          Self.merge_partials]
    comptime q_local_lifetime = PhaseGroup[Self.merge_partials, Self.o_proj]

    var q:       Buf[BF16, SEQ * FULL_Q_N,             during=Self.q_lifetime]
    var kv:      Buf[BF16, SEQ * FULL_K_N * 2,         during=Self.kv_lifetime]
    var p:       Buf[F32,  128 * flash_stride(16, 512), during=Self.p_lifetime]
    var q_local: Buf[BF16, SEQ * FULL_O_M,             during=Self.q_local_lifetime]


@fieldwise_init
struct FfnMoe(Island, Copyable, ImplicitlyCopyable):
    comptime ffn_rms_norm     = 1
    comptime gemv_gate        = 2
    comptime gemv_up          = 3
    comptime gelu_gate_up     = 4
    comptime router_sharded   = 5
    comptime merge_cands      = 6
    comptime moe_rms_norm     = 7
    comptime build_schedules  = 8
    comptime phase1_gate_up   = 9
    comptime phase2_down      = 10
    comptime moe_allreduce    = 11
    comptime gemv_dense       = 12
    comptime allreduce_dense  = 13
    comptime post_norm_1      = 14
    comptime post_norm_2      = 15
    comptime post_norm_3      = 16

    # Lifetime windows. Buffers sharing a window share one declaration —
    # editing the window updates every buffer that lives in it.
    comptime gate_band         = PhaseGroup[Self.gemv_gate,       Self.gemv_dense]
    comptime up_band           = PhaseGroup[Self.gemv_up,         Self.gelu_gate_up]
    comptime router_workspace  = PhaseGroup[Self.router_sharded,  Self.router_sharded]
    comptime cands_band        = PhaseGroup[Self.router_sharded,  Self.merge_cands]
    comptime router_products   = PhaseGroup[Self.merge_cands,     Self.build_schedules]
    comptime expert_input      = PhaseGroup[Self.moe_rms_norm,    Self.phase1_gate_up]
    comptime schedule_products = PhaseGroup[Self.build_schedules, Self.phase2_down]
    comptime hidden_bucket     = PhaseGroup[Self.phase1_gate_up,  Self.phase2_down]
    comptime gate_scratch      = PhaseGroup[Self.phase1_gate_up,  Self.phase1_gate_up]
    comptime accum_band        = PhaseGroup[Self.phase2_down,     Self.phase2_down]
    comptime dense_band        = PhaseGroup[Self.gemv_dense,      Self.post_norm_3]

    var ffn_gate:          Buf[BF16, SEQ * GATE_UP_N,                       during=Self.gate_band]
    var ffn_up:            Buf[BF16, SEQ * GATE_UP_N,                       during=Self.up_band]
    var moe_router_scaled: Buf[F32,  MAX_WORKERS * HIDDEN,                  during=Self.router_workspace]
    var moe_cands:         Buf[U8,   SEQ * TOP_K * 8,                       during=Self.cands_band]
    var moe_route_idx:     Buf[I32,  SEQ * TOP_K,                           during=Self.router_products]
    var moe_route_w:       Buf[F32,  SEQ * TOP_K,                           during=Self.router_products]
    var moe_x_normed:      Buf[BF16, SEQ * HIDDEN,                          during=Self.expert_input]
    var moe_expert_offset: Buf[I32,  EXPERTS_PER_RANK + 1,                  during=Self.schedule_products]
    var moe_routes:        Buf[U8,   SEQ * TOP_K * 8,                       during=Self.schedule_products]
    var moe_hidden_bucket: Buf[BF16, SEQ * TOP_K * MOE_INTERMEDIATE,        during=Self.hidden_bucket]
    var moe_gate_scratch:  Buf[F32,  MAX_WORKERS * PHASE1_MR * 2 * PHASE1_TILE_J, during=Self.gate_scratch]
    var moe_accum:         Buf[F32,  SEQ * HIDDEN,                          during=Self.accum_band]
    var ffn_dense_out:     Buf[BF16, SEQ * HIDDEN,                          during=Self.dense_band]


@fieldwise_init
struct LmHead(Island, Copyable, ImplicitlyCopyable):
    comptime final_norm  = 1
    comptime gemv_logits = 2
    comptime argmax      = 3

    comptime logits_lifetime = PhaseGroup[Self.gemv_logits, Self.argmax]

    var logits: Buf[BF16, VOCAB_PER_RANK, during=Self.logits_lifetime]


@fieldwise_init
struct Forward(Copyable, ImplicitlyCopyable):
    var sliding: SlidingAttention
    var full:    FullAttention
    var ffn:     FfnMoe
    var head:    LmHead


comptime POOL_SIZE = aggregate_peak[Forward]()


struct Pool[size: Int](Movable):
    var base: UnsafePointer[UInt8, MutAnyOrigin]

    def __init__(out self, base: UnsafePointer[UInt8, MutAnyOrigin]):
        self.base = base

    @always_inline
    def slot[
        I: Island, name: StringLiteral,
    ](self) -> UnsafePointer[
        downcast[reflect[I].field_type[name].T, BufferLike].Element,
        MutAnyOrigin,
    ]:
        comptime idx = reflect[I].field_index[name]()
        comptime off = I.PLAN.offsets[idx]
        return UnsafePointer[
            downcast[reflect[I].field_type[name].T, BufferLike].Element,
            MutAnyOrigin,
        ](unsafe_from_address=Int(self.base) + off)


def main():
    print("== v5_phase_groups: lifetime as named annotations ==\n")
    print("SlidingAttention.PEAK =", SlidingAttention.PLAN.peak)
    print("FullAttention.PEAK    =", FullAttention.PLAN.peak)
    print("FfnMoe.PEAK           =", FfnMoe.PLAN.peak)
    print("LmHead.PEAK           =", LmHead.PLAN.peak)
    print("POOL_SIZE             =", POOL_SIZE)
