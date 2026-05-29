from std.memory import UnsafePointer

from kernels.helpers import Binding


comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime BF16Bind[o: ImmutOrigin] = Binding[BFloat16, o]


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

    def host_arena_bytes(self) -> Int:
        return self.host_bytes

    @always_inline
    def scratch_base(self) -> Int:
        return self.base + self.scratch_off


@fieldwise_init
struct Repeated[T: ImplicitlyCopyable & ImplicitlyDestructible](Copyable, ImplicitlyCopyable):
    var proto: Self.T
    var off: Int
    var stride: Int
    var count: Int

    @always_inline
    def base(self, arena_base: Int, idx: Int) -> Int:
        return arena_base + self.off + idx * self.stride
