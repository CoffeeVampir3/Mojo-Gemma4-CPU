from std.memory import UnsafePointer

from modeling.model_spec import (
    Encoding, BF16,
    ShapeLike, StaticView,
    DEFAULT_ALIGNMENT, WeightDesc, DISTRIBUTED,
    align_up,
)


comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


@fieldwise_init
struct ArenaLayout(Copyable, ImplicitlyCopyable):
    """Common arena metadata shared by every model topology.

    `base` is the per-rank arena start address. The sizing fields describe
    the layout the loader/runtime expects: distributed (weights) +
    state (activations, KV cache, rope, scratch) form the main arena.
    `host_bytes` is the allocation ceiling for that arena plus any optional
    rank-targeted tensors a model topology chooses to append.
    """
    var base: Int
    var distributed_bytes: Int
    var state_bytes: Int
    var host_bytes: Int
    var scratch_off: Int

    def bind(self, new_base: Int) -> Self:
        var t = self
        t.base = new_base
        return t

    def arena_bytes(self) -> Int:
        return self.distributed_bytes + self.state_bytes

    def host_arena_bytes(self) -> Int:
        return self.host_bytes

    @always_inline
    def scratch_base(self) -> Int:
        return self.base + self.scratch_off


@fieldwise_init
struct TensorRef[E: Encoding, S: ShapeLike](Copyable, ImplicitlyCopyable):
    """Typed offset for one materialized tensor family."""
    var offset: Int

    @always_inline
    def bound(self, base: Int) -> StaticView[Self.E, Self.S]:
        return StaticView[Self.E, Self.S](
            UnsafePointer[Scalar[Self.E.DTYPE], MutAnyOrigin](
                unsafe_from_address=base + self.offset))


@fieldwise_init
struct Repeated[T: ImplicitlyCopyable](Copyable, ImplicitlyCopyable):
    """Repeated topology: proto identity + off/stride/count placement."""
    var proto: Self.T
    var off: Int
    var stride: Int
    var count: Int

    @always_inline
    def base(self, arena_base: Int, idx: Int) -> Int:
        return arena_base + self.off + idx * self.stride


struct SectionBuilder:
    """Typed cursor allocator for state and persistent aux sections."""
    var cursor: Int

    def __init__(out self):
        self.cursor = 0

    @always_inline
    def align(mut self, alignment: Int = DEFAULT_ALIGNMENT):
        self.cursor = align_up(self.cursor, alignment)

    @always_inline
    def reserve[E: Encoding, S: ShapeLike](mut self) -> TensorRef[E, S]:
        self.align()
        comptime size = S.bytes[E]()
        var off = self.cursor
        self.cursor += size
        return TensorRef[E, S](off)

    @always_inline
    def reserve_bytes(mut self, nbytes: Int, alignment: Int = DEFAULT_ALIGNMENT) -> Int:
        self.align(alignment)
        var off = self.cursor
        self.cursor += nbytes
        return off

    @always_inline
    def advance_bytes(mut self, nbytes: Int, alignment: Int = DEFAULT_ALIGNMENT):
        self.align(alignment)
        self.cursor += nbytes

    @always_inline
    def bytes(self) -> Int:
        return self.cursor


@fieldwise_init
struct LayerBuilder(Movable):
    var cursor: Int
    var layer_prefix: String
    var layer_base: Int

    def __init__(out self, prefix: String, layer_base: Int, *, start_at: Int = 0):
        self.cursor = start_at
        self.layer_prefix = prefix
        self.layer_base = layer_base

    @always_inline
    def emit_shape[E: Encoding, S: ShapeLike](mut self,
            mut entries: List[WeightDesc],
            suffix: String,
            target_rank: Int = DISTRIBUTED) -> TensorRef[E, S]:
        comptime alloc = S.bytes[E]()
        var off = align_up(self.cursor)
        self.cursor = off + alloc
        entries.append(WeightDesc(
            name=self.layer_prefix + suffix,
            arena_offset=self.layer_base + off,
            dtype=E.DTYPE, element_bytes=E.ELEMENT_BYTES,
            global_rows=S.GLOBAL_N, global_cols=S.GLOBAL_M,
            local_cols=S.M,
            data_rows=S.DATA_N, data_cols=S.DATA_M,
            target_rank=target_rank,
        ))
        return TensorRef[E, S](off)

    @always_inline
    def bfs[S: ShapeLike](mut self, mut entries: List[WeightDesc], suffix: String,
                         target_rank: Int = DISTRIBUTED) -> TensorRef[BF16, S]:
        return self.emit_shape[BF16, S](entries, suffix, target_rank)
