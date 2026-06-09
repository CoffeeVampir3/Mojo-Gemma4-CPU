@fieldwise_init
struct ArenaLayout(Copyable, ImplicitlyCopyable):
    """Common arena metadata shared by every model topology.

    All fields are arena-relative; the layout holds no absolute addresses.
    The sizing fields describe the layout the loader/runtime expects:
    distributed (weights) + state (activations, KV cache, rope, scratch)
    form the main arena. `host_bytes` is the allocation ceiling for that
    arena plus any optional rank-targeted tensors a model topology chooses
    to append.
    """
    var distributed_bytes: Int
    var state_bytes: Int
    var host_bytes: Int
    var scratch_off: Int

    def host_arena_bytes(self) -> Int:
        return self.host_bytes


@fieldwise_init
struct Repeated[T: ImplicitlyCopyable & ImplicitlyDestructible](Copyable, ImplicitlyCopyable):
    var proto: Self.T
    var off: Int
    var stride: Int
    var count: Int

    @always_inline
    def base(self, idx: Int) -> Int:
        return self.off + idx * self.stride
