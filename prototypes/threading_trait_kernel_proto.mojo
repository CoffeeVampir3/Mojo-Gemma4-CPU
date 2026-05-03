from std.sys.info import size_of
from std.memory import Span, UnsafePointer
from threading.threading_shared import (
    MAILBOX_DATA_BYTES,
)


trait BurstKernel(TrivialRegisterPassable):
    """Trivial mailbox payload that knows how to execute itself."""
    def execute(mut self): ...


def kernel_trampoline[K: BurstKernel](data_ptr: Int):
    UnsafePointer[K, MutAnyOrigin](unsafe_from_address=data_ptr)[].execute()


struct PoolStub:
    var capacity: Int
    var active_jobs: Int

    def __init__(out self):
        self.capacity = 0
        self.active_jobs = 0

    def dispatch_kernel[K: BurstKernel, origin: MutOrigin](
        mut self, kernels: Span[K, origin], num_jobs: Int = -1):
        """Synchronous stand-in for real mailbox dispatch.

        The real pool copies kernels[i] into worker-local mailbox storage and
        calls kernel_trampoline on that mailbox address. The stub keeps the
        same copy-then-execute semantics with a stack mailbox copy.
        """
        comptime assert size_of[K]() <= MAILBOX_DATA_BYTES, \
            "kernel struct exceeds mailbox capacity"

        var jobs = num_jobs if num_jobs >= 0 else len(kernels)
        if jobs <= 0:
            return
        debug_assert(jobs <= len(kernels), "num_jobs exceeds kernel span")
        if self.capacity > 0:
            debug_assert(jobs <= self.capacity, "num_jobs exceeds pool capacity")

        var tramp = kernel_trampoline[K]
        for i in range(jobs):
            var mailbox = kernels[i]
            tramp(Int(UnsafePointer(to=mailbox)))
        self.active_jobs = jobs


@fieldwise_init
struct StoreJob(BurstKernel):
    var out_addr: UnsafePointer[Int, MutAnyOrigin]
    var value: Int

    def execute(mut self):
        self.out_addr[] = self.value


comptime I8Ptr = UnsafePointer[Int8, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]


@fieldwise_init
struct MlaCompressedDecodeJob[
    kv_lora_rank: Int, qk_rope_head_dim: Int, max_seq: Int,
](BurstKernel):
    var q_absorbed_i8_base: I8Ptr
    var q_pe_i8_base: I8Ptr
    var q_absorbed_b_base: F32Ptr
    var q_absorbed_f_base: F32Ptr
    var q_pe_b_base: F32Ptr
    var q_pe_f_base: F32Ptr
    var cache_base: Int
    var v_collected_out: F32Ptr
    var head_start: Int
    var head_count: Int
    var t_count: Int
    var q_pos: Int

    def execute(mut self):
        pass


trait NumaAwareKernel(BurstKernel):
    def preferred_node(self) -> Int: ...


trait CostedKernel(BurstKernel):
    def estimated_cost(self) -> Int: ...


comptime SchedulableKernel = NumaAwareKernel & CostedKernel


@fieldwise_init
struct ScheduledKernel[K: BurstKernel](NumaAwareKernel, CostedKernel):
    """Decorator for scheduler metadata.

    This keeps scheduling policy orthogonal to the executable payload: any
    BurstKernel can be pinned and costed without changing the payload type.
    """
    var inner: Self.K
    var node: Int
    var cost: Int

    def execute(mut self):
        self.inner.execute()

    def preferred_node(self) -> Int:
        return self.node

    def estimated_cost(self) -> Int:
        return self.cost
