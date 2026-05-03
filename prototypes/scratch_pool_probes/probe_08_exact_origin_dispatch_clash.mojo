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
struct ExactLease[pool_origin: MutOrigin](Movable):
    var addr: Int
    var byte_size: Int
    var pool: UnsafePointer[ScratchPool, Self.pool_origin]

    def __init__(out self, addr: Int, byte_size: Int,
                 pool: UnsafePointer[ScratchPool, Self.pool_origin]):
        self.addr = addr
        self.byte_size = byte_size
        self.pool = pool

    def release(deinit self):
        self.pool[].offset -= self.byte_size

    @always_inline
    def as_ptr[
        o: MutOrigin, //, T: AnyType,
    ](ref [o] self) -> UnsafePointer[T, o]:
        return UnsafePointer[T, o](unsafe_from_address=self.addr)


struct ScratchPool:
    var base: Int
    var capacity: Int
    var offset: Int

    def __init__(out self, base: Int, capacity: Int):
        self.base = base
        self.capacity = capacity
        self.offset = 0

    def borrow[T: AnyType, count: Int](mut self) -> ExactLease[origin_of(self)]:
        comptime byte_size = typed_bytes[T, count]()
        var addr = self.base + self.offset
        self.offset += byte_size
        if self.offset > self.capacity:
            abort("ScratchPool: exceeded capacity")
        return ExactLease[origin_of(self)](
            addr, byte_size, UnsafePointer(to=self))


@explicit_destroy
struct ErasedLease(Movable):
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


struct ErasedPool:
    var base: Int
    var capacity: Int
    var offset: Int

    def __init__(out self, base: Int, capacity: Int):
        self.base = base
        self.capacity = capacity
        self.offset = 0

    def borrow[T: AnyType, count: Int](mut self) -> ErasedLease:
        comptime byte_size = typed_bytes[T, count]()
        var addr = self.base + self.offset
        self.offset += byte_size
        if self.offset > self.capacity:
            abort("ErasedPool: exceeded capacity")
        return ErasedLease(addr, byte_size,
            UnsafePointer[Int, MutAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=self.offset))))


@fieldwise_init
struct WriteKernel(BurstKernel):
    var dst: Int
    var value: Float32

    def execute(mut self):
        UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.dst)[] = self.value


def test_exact_origin_lease_basic():
    var backing = alloc[UInt8](4096)
    var pool = ScratchPool(Int(backing), 4096)

    var lease = pool.borrow[Float32, 4]()
    var ptr = lease.as_ptr[Float32]()
    ptr[0] = Float32(42)
    debug_assert(ptr[0] == Float32(42), "exact write ok")

    lease^.release()
    debug_assert(pool.offset == 0, "released")
    print("  exact origin lease basic: ok")


def test_erased_pool_dispatch():
    var backing = alloc[UInt8](4096)
    var pool = ErasedPool(Int(backing), 4096)

    var lease = pool.borrow[Float32, 4]()
    var target = lease.addr

    var test_pool = TestPool(1, 0)
    var jobs = InlineArray[WriteKernel, 1](uninitialized=True)
    jobs[0] = WriteKernel(target, Float32(77))
    test_pool.dispatch(
        Span(ptr=UnsafePointer(to=jobs[0]), length=1), 1)
    test_pool.join()

    debug_assert(lease.as_ptr[Float32]()[0] == Float32(77), "dispatched write")

    lease^.release()
    debug_assert(pool.offset == 0, "released after dispatch")
    print("  erased pool + dispatch: ok")


def test_exact_pool_dispatch():
    var backing = alloc[UInt8](4096)
    var pool = ScratchPool(Int(backing), 4096)

    var lease = pool.borrow[Float32, 4]()
    var target = lease.addr

    var test_pool = TestPool(1, 0)
    var jobs = InlineArray[WriteKernel, 1](uninitialized=True)
    jobs[0] = WriteKernel(target, Float32(88))
    test_pool.dispatch(
        Span(ptr=UnsafePointer(to=jobs[0]), length=1), 1)
    test_pool.join()

    debug_assert(lease.as_ptr[Float32]()[0] == Float32(88), "exact dispatched write")

    lease^.release()
    debug_assert(pool.offset == 0, "released after exact dispatch")
    print("  exact pool + dispatch: ok")


def test_multi_lease_with_dispatch():
    var backing = alloc[UInt8](4096)
    var pool = ErasedPool(Int(backing), 4096)

    var lease_a = pool.borrow[Float32, 4]()
    var lease_b = pool.borrow[Float32, 4]()

    var test_pool = TestPool(1, 0)
    var jobs = InlineArray[WriteKernel, 2](uninitialized=True)
    jobs[0] = WriteKernel(lease_a.addr, Float32(10))
    jobs[1] = WriteKernel(lease_b.addr, Float32(20))
    test_pool.dispatch(
        Span(ptr=UnsafePointer(to=jobs[0]), length=2), 2)
    test_pool.join()

    debug_assert(lease_a.as_ptr[Float32]()[0] == Float32(10), "multi a")
    debug_assert(lease_b.as_ptr[Float32]()[0] == Float32(20), "multi b")

    lease_b^.release()
    lease_a^.release()
    debug_assert(pool.offset == 0, "multi released")
    print("  multi lease + dispatch: ok")


def test_phase_pattern_erased():
    var backing = alloc[UInt8](4096)
    var pool = ErasedPool(Int(backing), 4096)

    var persistent = pool.borrow[Float32, 16]()
    persistent.as_ptr[Float32]()[0] = Float32(999)

    var temp = pool.borrow[Float32, 32]()
    temp.as_ptr[Float32]()[0] = Float32(111)
    temp^.release()

    debug_assert(persistent.as_ptr[Float32]()[0] == Float32(999), "persistent survived")

    var temp2 = pool.borrow[Float32, 32]()
    temp2.as_ptr[Float32]()[0] = Float32(222)
    temp2^.release()

    persistent^.release()
    debug_assert(pool.offset == 0, "phase pattern done")
    print("  phase pattern erased: ok")


def main():
    print("probe 08: exact vs erased origin with BurstKernel dispatch")
    test_exact_origin_lease_basic()
    test_erased_pool_dispatch()
    test_exact_pool_dispatch()
    test_multi_lease_with_dispatch()
    test_phase_pattern_erased()
    print("probe 08 ok")
