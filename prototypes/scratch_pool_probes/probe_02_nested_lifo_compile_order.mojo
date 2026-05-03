from std.sys.info import size_of
from std.os import abort
from std.memory import UnsafePointer, alloc


comptime SCRATCH_ALIGNMENT = 64


@always_inline
def aligned_bytes[nbytes: Int]() -> Int:
    return ((nbytes + SCRATCH_ALIGNMENT - 1) // SCRATCH_ALIGNMENT) * SCRATCH_ALIGNMENT


@always_inline
def typed_bytes[T: AnyType, count: Int]() -> Int:
    return aligned_bytes[count * size_of[T]()]()


@explicit_destroy
struct Lease(Movable):
    var addr: Int
    var byte_size: Int
    var pool_offset: UnsafePointer[Int, MutAnyOrigin]

    def __init__(out self, addr: Int, byte_size: Int,
                 pool_offset: UnsafePointer[Int, MutAnyOrigin]):
        self.addr = addr
        self.byte_size = byte_size
        self.pool_offset = pool_offset

    def release(deinit self):
        self.pool_offset[] -= self.byte_size

    @always_inline
    def as_ptr[
        o: MutOrigin, //, T: AnyType,
    ](ref [o] self) -> UnsafePointer[T, o]:
        return UnsafePointer[T, o](unsafe_from_address=self.addr)


struct PoolState:
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
            abort("PoolState: exceeded capacity")
        return Lease(addr, byte_size,
            UnsafePointer[Int, MutAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=self.offset))))


def test_nested_lifo_natural_scoping():
    var backing = alloc[UInt8](4096)
    var pool = PoolState(Int(backing), 4096)

    var outer = pool.borrow[Float32, 16]()
    var outer_ptr = outer.as_ptr[Float32]()
    outer_ptr[0] = Float32(1)

    var inner = pool.borrow[Float32, 8]()
    var inner_ptr = inner.as_ptr[Float32]()
    inner_ptr[0] = Float32(2)

    debug_assert(inner_ptr[0] == Float32(2), "inner read")
    debug_assert(outer_ptr[0] == Float32(1), "outer read")

    inner^.release()
    outer^.release()

    debug_assert(pool.offset == 0, "pool empty")
    print("  nested LIFO natural scoping: ok")


def test_lease_ptr_origin_from_ref():
    var backing = alloc[UInt8](4096)
    var pool = PoolState(Int(backing), 4096)

    var lease = pool.borrow[Float32, 4]()
    var ptr = lease.as_ptr[Float32]()
    ptr[0] = Float32(42)
    ptr[1] = Float32(43)

    debug_assert(ptr[0] == Float32(42), "ptr 0")
    debug_assert(ptr[1] == Float32(43), "ptr 1")

    lease^.release()
    debug_assert(pool.offset == 0, "pool empty")
    print("  lease ptr origin from ref: ok")


def test_three_deep_lifo():
    var backing = alloc[UInt8](4096)
    var pool = PoolState(Int(backing), 4096)

    var a = pool.borrow[Float32, 4]()
    var b = pool.borrow[Float32, 4]()
    var c = pool.borrow[Float32, 4]()

    a.as_ptr[Float32]()[0] = Float32(1)
    b.as_ptr[Float32]()[0] = Float32(2)
    c.as_ptr[Float32]()[0] = Float32(3)

    c^.release()
    b^.release()
    a^.release()

    debug_assert(pool.offset == 0, "three deep restored")
    print("  three deep LIFO: ok")


def main():
    print("probe 02: nested LIFO ordering")
    test_nested_lifo_natural_scoping()
    test_lease_ptr_origin_from_ref()
    test_three_deep_lifo()
    print("probe 02 ok")
