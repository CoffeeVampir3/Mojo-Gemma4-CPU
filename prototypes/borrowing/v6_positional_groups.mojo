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


trait BufferLike:
    """Buffer carrier — only size info; lifetime is inherited from the
    most-recently-declared PhaseGroup marker that precedes it in the
    island's field order."""
    comptime Element: AnyType
    comptime SIZE:    Int


@fieldwise_init
struct Buf[T: AnyType, count: Int](
    BufferLike, Copyable, ImplicitlyCopyable
):
    comptime Element = Self.T
    comptime SIZE    = aligned64(Self.count * size_of[Self.T]())


trait PhaseRange:
    """Marker trait for fields that announce a lifetime band. Picked up
    by reflection in declaration order; subsequent BufferLike fields
    inherit the most recent PhaseRange's FIRST/LAST."""
    comptime FIRST: Int
    comptime LAST:  Int


@fieldwise_init
struct PhaseGroup[first: Int, last: Int](
    PhaseRange, Copyable, ImplicitlyCopyable
):
    comptime FIRST = Self.first
    comptime LAST  = Self.last


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
    var cur_first = 0
    var cur_last  = 0

    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, PhaseRange):
            cur_first = FT.FIRST
            cur_last  = FT.LAST
        comptime if conforms_to(FT, BufferLike):
            sizes[n]  = FT.SIZE
            firsts[n] = cur_first
            lasts[n]  = cur_last
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


def buffer_index_for[T: AnyType, name: StringLiteral]() -> Int:
    """Field name → position among BufferLike fields only.
    Skips PhaseGroup markers so PLAN.offsets stays buffer-keyed."""
    comptime target = reflect[T].field_index[name]()
    var n = 0
    comptime for i in range(reflect[T].field_count()):
        if i == target:
            return n
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, BufferLike):
            n += 1
    return -1


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

    var q_band:  PhaseGroup[Self.gemv_qkv, Self.o_proj]
    var q:       Buf[BF16, SEQ * SLIDING_Q_N]

    var kv_band: PhaseGroup[Self.gemv_qkv, Self.rope_cache_write]
    var kv:      Buf[BF16, SEQ * SLIDING_KV_N * 2]

    var p_band:  PhaseGroup[Self.flash, Self.merge_partials]
    var p:       Buf[F32, 128 * flash_stride(4, 256)]


@fieldwise_init
struct FullAttention(Island, Copyable, ImplicitlyCopyable):
    comptime gemv_q           = 1
    comptime gemv_kv          = 2
    comptime rms_norm_qkv     = 3
    comptime rope_cache_write = 4
    comptime flash            = 5
    comptime merge_partials   = 6
    comptime o_proj           = 7

    var q_band:       PhaseGroup[Self.gemv_q, Self.flash]
    var q:            Buf[BF16, SEQ * FULL_Q_N]

    var kv_band:      PhaseGroup[Self.gemv_kv, Self.rope_cache_write]
    var kv:           Buf[BF16, SEQ * FULL_K_N * 2]

    var p_band:       PhaseGroup[Self.flash, Self.merge_partials]
    var p:            Buf[F32, 128 * flash_stride(16, 512)]

    var q_local_band: PhaseGroup[Self.merge_partials, Self.o_proj]
    var q_local:      Buf[BF16, SEQ * FULL_O_M]


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

    # FFN gate band — wide, spans across MoE; only ffn_gate lives here.
    var ffn_gate_band:        PhaseGroup[Self.gemv_gate, Self.gemv_dense]
    var ffn_gate:             Buf[BF16, SEQ * GATE_UP_N]

    # FFN up — short-lived right before fusion.
    var ffn_up_band:          PhaseGroup[Self.gemv_up, Self.gelu_gate_up]
    var ffn_up:               Buf[BF16, SEQ * GATE_UP_N]

    # MoE router intermediates — workspace + candidates, single-op lifetimes.
    var router_workspace:     PhaseGroup[Self.router_sharded, Self.router_sharded]
    var moe_router_scaled:    Buf[F32, MAX_WORKERS * HIDDEN]

    var router_cands:         PhaseGroup[Self.router_sharded, Self.merge_cands]
    var moe_cands:            Buf[U8,  SEQ * TOP_K * 8]

    # Router products — TWO buffers share this lifetime.
    var router_products:      PhaseGroup[Self.merge_cands, Self.build_schedules]
    var moe_route_idx:        Buf[I32, SEQ * TOP_K]
    var moe_route_w:          Buf[F32, SEQ * TOP_K]

    # Expert input — single buffer.
    var expert_input:         PhaseGroup[Self.moe_rms_norm, Self.phase1_gate_up]
    var moe_x_normed:         Buf[BF16, SEQ * HIDDEN]

    # Schedule products — TWO buffers share this.
    var schedule_products:    PhaseGroup[Self.build_schedules, Self.phase2_down]
    var moe_expert_offset:    Buf[I32, EXPERTS_PER_RANK + 1]
    var moe_routes:           Buf[U8,  SEQ * TOP_K * 8]

    # Hidden bucket — one buffer, two-phase lifetime.
    var hidden_bucket:        PhaseGroup[Self.phase1_gate_up, Self.phase2_down]
    var moe_hidden_bucket:    Buf[BF16, SEQ * TOP_K * MOE_INTERMEDIATE]

    # Phase1 gate scratch — one phase only.
    var phase1_scratch:       PhaseGroup[Self.phase1_gate_up, Self.phase1_gate_up]
    var moe_gate_scratch:     Buf[F32, MAX_WORKERS * PHASE1_MR * 2 * PHASE1_TILE_J]

    # Phase2 accumulator — one phase only.
    var phase2_accum:         PhaseGroup[Self.phase2_down, Self.phase2_down]
    var moe_accum:            Buf[F32, SEQ * HIDDEN]

    # Dense out — spans the post-MoE tail.
    var dense_band:           PhaseGroup[Self.gemv_dense, Self.post_norm_3]
    var ffn_dense_out:        Buf[BF16, SEQ * HIDDEN]


@fieldwise_init
struct LmHead(Island, Copyable, ImplicitlyCopyable):
    comptime final_norm  = 1
    comptime gemv_logits = 2
    comptime argmax      = 3

    var logits_band: PhaseGroup[Self.gemv_logits, Self.argmax]
    var logits:      Buf[BF16, VOCAB_PER_RANK]


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
        comptime idx = buffer_index_for[I, name]()
        comptime off = I.PLAN.offsets[idx]
        return UnsafePointer[
            downcast[reflect[I].field_type[name].T, BufferLike].Element,
            MutAnyOrigin,
        ](unsafe_from_address=Int(self.base) + off)


def main():
    print("== v6_positional_groups: PhaseGroup as positional marker ==\n")
    print("SlidingAttention.PEAK =", SlidingAttention.PLAN.peak)
    print("FullAttention.PEAK    =", FullAttention.PLAN.peak)
    print("FfnMoe.PEAK           =", FfnMoe.PLAN.peak)
    print("LmHead.PEAK           =", LmHead.PLAN.peak)
    print("POOL_SIZE             =", POOL_SIZE)
