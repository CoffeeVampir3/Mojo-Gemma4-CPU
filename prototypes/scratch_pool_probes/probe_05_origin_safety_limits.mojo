from std.sys.info import size_of
from std.os import abort
from std.memory import UnsafePointer, Span, alloc
from std.collections import InlineArray

from threading.threading_traits import BurstKernel, BurstThreadPool


comptime SCRATCH_ALIGNMENT = 64


@always_inline
def aligned_bytes[nbytes: Int]() -> Int:
    return ((nbytes + SCRATCH_ALIGNMENT - 1) // SCRATCH_ALIGNMENT) * SCRATCH_ALIGNMENT


@fieldwise_init
struct TestPool(BurstThreadPool):
    var capacity: Int
    var ts: Int

    def get_capacity(self) -> Int:
        return self.capacity

    def dispatch[K: BurstKernel, origin: MutOrigin](
        mut self, kernels: Span[K, origin], num_jobs: Int):
        for i in range(num_jobs):
            var k = kernels[i]
            k.execute()

    def join(mut self):
        pass

    def last_worker_timestamp(self) -> Int:
        return self.ts

    def wake(mut self):
        pass

    def sleep(mut self):
        pass


@explicit_destroy
struct TrackedLease(Movable):
    var addr: Int
    var byte_size: Int
    var released: UnsafePointer[Bool, MutAnyOrigin]

    def __init__(out self, addr: Int, byte_size: Int,
                 released: UnsafePointer[Bool, MutAnyOrigin]):
        self.addr = addr
        self.byte_size = byte_size
        self.released = released

    def release(deinit self):
        self.released[] = True

    @always_inline
    def as_ptr[
        o: MutOrigin, //, T: AnyType,
    ](ref [o] self) -> UnsafePointer[T, o]:
        return UnsafePointer[T, o](unsafe_from_address=self.addr)


@fieldwise_init
struct WriteKernelExact[dst_origin: ImmutOrigin](BurstKernel):
    var dst_token: UnsafePointer[Float32, Self.dst_origin]
    var dst_addr: Int
    var value: Float32

    def execute(mut self):
        var dst = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.dst_addr)
        dst[] = self.value
        _ = self.dst_token


@fieldwise_init
struct WriteKernelErased(BurstKernel):
    var dst_addr: Int
    var value: Float32

    def execute(mut self):
        var dst = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.dst_addr)
        dst[] = self.value


def test_lease_origin_ties_to_ref():
    var backing = alloc[UInt8](256)
    var released = alloc[Bool](1)
    released[] = False

    var lease = TrackedLease(Int(backing), 256, released.as_any_origin())
    var ptr = lease.as_ptr[Float32]()
    ptr[0] = Float32(42)
    debug_assert(ptr[0] == Float32(42), "write through lease ok")

    lease^.release()
    debug_assert(released[], "release tracked")

    print("  lease origin ties to ref: ok")


def test_exact_origin_token_in_kernel():
    var target = alloc[Float32](1)
    target[] = Float32(-1)

    var backing = alloc[UInt8](256)
    var released = alloc[Bool](1)
    released[] = False

    var lease = TrackedLease(Int(backing), 256, released.as_any_origin())

    var pool = TestPool(1, 0)

    var jobs = InlineArray[WriteKernelErased, 1](uninitialized=True)
    jobs[0] = WriteKernelErased(Int(target), Float32(77))
    pool.dispatch(
        Span(ptr=UnsafePointer(to=jobs[0]), length=1), 1)
    pool.join()

    debug_assert(target[] == Float32(77), "erased kernel wrote")

    lease^.release()
    print("  erased kernel dispatch: ok")


def test_origin_token_with_keepalive():
    var target = alloc[Float32](1)
    target[] = Float32(-1)

    var backing = alloc[UInt8](256)
    var released = alloc[Bool](1)
    released[] = False

    var lease = TrackedLease(Int(backing), 256, released.as_any_origin())

    var pool = TestPool(1, 0)

    comptime lease_immut = ImmutOrigin(MutAnyOrigin)
    var token = UnsafePointer[Float32, lease_immut](
        unsafe_from_address=lease.addr)

    var jobs = InlineArray[WriteKernelExact[lease_immut], 1](uninitialized=True)
    jobs[0] = WriteKernelExact[lease_immut](
        token, Int(target), Float32(88))
    pool.dispatch(
        Span(ptr=UnsafePointer(to=jobs[0]), length=1), 1)
    pool.join()

    debug_assert(target[] == Float32(88), "token kernel wrote")

    lease^.release()
    print("  origin token keepalive kernel: ok")


def test_watermark_vs_lease_dispatch_compatibility():
    var backing = alloc[UInt8](1024)
    var base = Int(backing)

    var pool = TestPool(1, 0)

    var offset = 0
    comptime slot_a_size = aligned_bytes[4 * size_of[Float32]()]()
    var slot_a_addr = base + offset
    offset += slot_a_size

    comptime slot_b_size = aligned_bytes[4 * size_of[Float32]()]()
    var slot_b_addr = base + offset
    offset += slot_b_size

    var pa = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=slot_a_addr)
    var pb = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=slot_b_addr)
    pa[0] = Float32(1)
    pb[0] = Float32(2)

    var jobs = InlineArray[WriteKernelErased, 2](uninitialized=True)
    jobs[0] = WriteKernelErased(slot_a_addr, Float32(10))
    jobs[1] = WriteKernelErased(slot_b_addr, Float32(20))
    pool.dispatch(
        Span(ptr=UnsafePointer(to=jobs[0]), length=2), 2)
    pool.join()

    debug_assert(pa[] == Float32(10), "dispatched write a")
    debug_assert(pb[] == Float32(20), "dispatched write b")

    print("  watermark + dispatch compatibility: ok")


def main():
    print("probe 05: origin safety limits with BurstKernel dispatch")
    test_lease_origin_ties_to_ref()
    test_exact_origin_token_in_kernel()
    test_origin_token_with_keepalive()
    test_watermark_vs_lease_dispatch_compatibility()
    print("probe 05 ok")
