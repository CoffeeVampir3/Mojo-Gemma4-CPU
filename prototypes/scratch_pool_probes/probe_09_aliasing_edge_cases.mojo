from std.sys.info import size_of
from std.os import abort
from std.memory import UnsafePointer, Span, alloc
from std.collections import InlineArray

from threading.threading_traits import BurstKernel, BurstThreadPool


comptime SCRATCH_ALIGNMENT = 64

@always_inline
def aligned_bytes[nbytes: Int]() -> Int:
    return ((nbytes + SCRATCH_ALIGNMENT - 1) // SCRATCH_ALIGNMENT) * SCRATCH_ALIGNMENT

@always_inline
def typed_bytes[T: AnyType, count: Int]() -> Int:
    return aligned_bytes[count * size_of[T]()]()


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
struct Lease(Movable):
    var addr: Int
    var byte_size: Int
    var pool_offset_ptr: UnsafePointer[Int, MutAnyOrigin]

    def __init__(out self, addr: Int, byte_size: Int,
                 pool_offset_ptr: UnsafePointer[Int, MutAnyOrigin]):
        self.addr = addr
        self.byte_size = byte_size
        self.pool_offset_ptr = pool_offset_ptr

    def release(deinit self):
        self.pool_offset_ptr[] -= self.byte_size

    @always_inline
    def as_ptr[
        o: MutOrigin, //, T: AnyType,
    ](ref [o] self) -> UnsafePointer[T, o]:
        return UnsafePointer[T, o](unsafe_from_address=self.addr)


struct Pool:
    var base: Int
    var capacity: Int
    var offset: Int

    def __init__(out self, base: Int, capacity: Int):
        self.base = base
        self.capacity = capacity
        self.offset = 0

    def borrow[T: AnyType, count: Int](mut self) -> Lease:
        comptime byte_size = typed_bytes[T, count]()
        var addr = self.base + self.offset
        self.offset += byte_size
        if self.offset > self.capacity:
            abort("Pool: exceeded capacity")
        return Lease(addr, byte_size,
            UnsafePointer[Int, MutAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=self.offset))))


@fieldwise_init
struct TwoBufferKernel[
    src_origin: ImmutOrigin,
](BurstKernel):
    var src: UnsafePointer[Float32, Self.src_origin]
    var dst_addr: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var dst = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.dst_addr)
        for i in range(self.start, self.end):
            dst[i] = self.src[i] * Float32(2)


def test_lease_as_kernel_source():
    var backing = alloc[UInt8](4096)
    var pool = Pool(Int(backing), 4096)

    var src_lease = pool.borrow[Float32, 8]()
    var dst_lease = pool.borrow[Float32, 8]()

    var src_ptr = src_lease.as_ptr[Float32]()
    for i in range(8):
        src_ptr[i] = Float32(i)

    comptime src_immut = ImmutOrigin(MutAnyOrigin)
    var test_pool = TestPool(1, 0)
    var jobs = InlineArray[TwoBufferKernel[src_immut], 1](uninitialized=True)
    jobs[0] = TwoBufferKernel[src_immut](
        UnsafePointer[Float32, src_immut](unsafe_from_address=src_lease.addr),
        dst_lease.addr,
        0, 8)
    test_pool.dispatch(
        Span(ptr=UnsafePointer(to=jobs[0]), length=1), 1)
    test_pool.join()

    var dst_ptr = dst_lease.as_ptr[Float32]()
    for i in range(8):
        debug_assert(dst_ptr[i] == Float32(i * 2), "kernel src->dst mismatch")

    dst_lease^.release()
    src_lease^.release()
    debug_assert(pool.offset == 0, "pool clean")
    print("  lease as kernel source: ok")


@fieldwise_init
struct AccumKernel(BurstKernel):
    var src_addr: Int
    var dst_addr: Int
    var count: Int

    def execute(mut self):
        var src = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.src_addr)
        var dst = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.dst_addr)
        for i in range(self.count):
            dst[i] += src[i]


def test_lease_survives_temp_dispatch():
    var backing = alloc[UInt8](4096)
    var pool = Pool(Int(backing), 4096)

    var persistent = pool.borrow[Float32, 8]()
    persistent.as_ptr[Float32]()[0] = Float32(999)

    var temp = pool.borrow[Float32, 8]()
    var temp_src = temp.as_ptr[Float32]()
    for i in range(8):
        temp_src[i] = Float32(i + 100)

    var test_pool = TestPool(1, 0)

    var jobs = InlineArray[AccumKernel, 1](uninitialized=True)
    jobs[0] = AccumKernel(temp.addr, persistent.addr, 1)
    test_pool.dispatch(
        Span(ptr=UnsafePointer(to=jobs[0]), length=1), 1)
    test_pool.join()

    debug_assert(persistent.as_ptr[Float32]()[0] == Float32(1099),
        "persistent accumulated")

    temp^.release()

    debug_assert(persistent.as_ptr[Float32]()[0] == Float32(1099),
        "persistent survived temp release")

    persistent^.release()
    debug_assert(pool.offset == 0, "pool clean")
    print("  lease survives temp dispatch: ok")


def test_phase_reuse_pattern():
    var backing = alloc[UInt8](4096)
    var pool = Pool(Int(backing), 4096)

    var phase1_a = pool.borrow[Float32, 16]()
    var phase1_b = pool.borrow[Float32, 16]()
    phase1_a.as_ptr[Float32]()[0] = Float32(1)
    phase1_b.as_ptr[Float32]()[0] = Float32(2)
    var offset_after_phase1 = pool.offset

    phase1_b^.release()
    phase1_a^.release()

    var phase2_a = pool.borrow[Float32, 16]()
    var phase2_b = pool.borrow[Float32, 16]()
    phase2_a.as_ptr[Float32]()[0] = Float32(10)
    phase2_b.as_ptr[Float32]()[0] = Float32(20)

    debug_assert(pool.offset == offset_after_phase1, "same peak usage")

    phase2_b^.release()
    phase2_a^.release()
    debug_assert(pool.offset == 0, "fully released")
    print("  phase reuse pattern: ok")


def test_mixed_persistent_and_temp():
    var backing = alloc[UInt8](8192)
    var pool = Pool(Int(backing), 8192)

    var activation = pool.borrow[Float32, 64]()
    activation.as_ptr[Float32]()[0] = Float32(42)
    activation.as_ptr[Float32]()[63] = Float32(99)

    for phase in range(3):
        var scratch = pool.borrow[Float32, 32]()
        scratch.as_ptr[Float32]()[0] = Float32(phase)
        scratch^.release()
        debug_assert(activation.as_ptr[Float32]()[0] == Float32(42),
            "activation stable across phases")

    activation^.release()
    debug_assert(pool.offset == 0, "clean")
    print("  mixed persistent + temp: ok")


def main():
    print("probe 09: aliasing edge cases in real usage patterns")
    test_lease_as_kernel_source()
    test_lease_survives_temp_dispatch()
    test_phase_reuse_pattern()
    test_mixed_persistent_and_temp()
    print("probe 09 ok")
