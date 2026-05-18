from std.collections import InlineArray


struct Gemma4BaseConfig:
    comptime HIDDEN = 2816
    comptime NUM_LAYERS = 30
    comptime NUM_HEADS = 16

    comptime HEAD_DIM_SLIDING = 256
    comptime NUM_KV_HEADS_SLIDING = 8
    comptime Q_DIM_SLIDING = 4096
    comptime KV_DIM_SLIDING = 2048
    comptime ROPE_HALF_SLIDING = Self.HEAD_DIM_SLIDING // 2

    comptime HEAD_DIM_FULL = 512
    comptime NUM_KV_HEADS_FULL = 2
    comptime Q_DIM_FULL = 8192
    comptime KV_DIM_FULL = 1024
    # Partial rotary: 0.25 * HEAD_DIM_FULL / 2 = 64 dims rotate, rest pass through.
    comptime ROPE_HALF_FULL = 64

    comptime INTERMEDIATE = 2112
    comptime MOE_GATE_UP_FUSED = 1408
    comptime MOE_INTERMEDIATE = 704
    comptime NUM_EXPERTS = 128
    comptime TOP_K = 8

    comptime VOCAB_SIZE = 262144
    comptime NUM_SLIDING_LAYERS = 25
    comptime NUM_FULL_LAYERS = 5
    comptime MAX_SEQ_LEN = 4096
    comptime SLIDING_WINDOW = 1024
    comptime RMS_NORM_EPS = 1e-6
    comptime LOGIT_SOFTCAP = 30.0

    # Comptime ceiling on chunked-attention fan-out for full-attention layers.
    # Sizes dispatcher/merge stack arrays + cross-chunk partials buffer.
    # Must be >= any pool_capacity we will ever see at runtime.
    comptime FULL_ATTN_MAX_CHUNKS = 32


struct LayerKind:
    """Tag for a layer's attention flavor in `LAYER_SCHEDULE`."""
    comptime FULL = 0
    comptime SLIDING = 1


@fieldwise_init
struct LayerEntry(Copyable, ImplicitlyCopyable):
    """A single entry in the layer schedule.

    `idx`       — the global layer index (0..NUM_LAYERS).
    `kind`      — LayerKind.FULL or LayerKind.SLIDING.
    `local_idx` — index within its kind: 0..NUM_FULL_LAYERS-1 for full
                  layers, 0..NUM_SLIDING_LAYERS-1 for sliding layers.
                  This replaces the open-coded `si`/`fi` counters that
                  used to live in build_gemma4_plan / model_init /
                  forward.
    """
    var idx: Int
    var kind: Int
    var local_idx: Int


@always_inline
def build_layer_schedule() -> InlineArray[
    LayerEntry, Gemma4BaseConfig.NUM_LAYERS,
]:
    var out = InlineArray[LayerEntry, Gemma4BaseConfig.NUM_LAYERS](
        uninitialized=True,
    )
    var si = 0
    var fi = 0
    for i in range(Gemma4BaseConfig.NUM_LAYERS):
        # Gemma 4 places a full-attention layer at every sixth position
        # (indices 5, 11, 17, ...); the rest are sliding.
        if (i + 1) % 6 == 0:
            out[i] = LayerEntry(idx=i, kind=LayerKind.FULL, local_idx=fi)
            fi += 1
        else:
            out[i] = LayerEntry(
                idx=i, kind=LayerKind.SLIDING, local_idx=si,
            )
            si += 1
    return out


comptime LAYER_SCHEDULE = build_layer_schedule()
