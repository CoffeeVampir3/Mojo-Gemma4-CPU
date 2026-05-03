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
struct StackLease[pool_origin: MutOrigin](Movable):
    var addr: Int
    var byte_size: Int
    var pool: UnsafePointer[StackPool, Self.pool_origin]

    def __init__(out self, addr: Int, byte_size: Int,
                 pool: UnsafePointer[StackPool, Self.pool_origin]):
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


struct StackPool:
    var base: Int
    var capacity: Int
    var offset: Int

    def __init__(out self, base: Int, capacity: Int):
        self.base = base
        self.capacity = capacity
        self.offset = 0

    def borrow[T: AnyType, count: Int](mut self) -> StackLease[origin_of(self)]:
        comptime byte_size = typed_bytes[T, count]()
        var addr = self.base + self.offset
        self.offset += byte_size
        if self.offset > self.capacity:
            abort("StackPool: exceeded capacity")
        return StackLease[origin_of(self)](
            addr, byte_size, UnsafePointer(to=self))


def test_exact_origin_borrow():
    var backing = alloc[UInt8](4096)
    var pool = StackPool(Int(backing), 4096)

    var lease = pool.borrow[Float32, 8]()
    var ptr = lease.as_ptr[Float32]()
    ptr[0] = Float32(42)
    debug_assert(ptr[0] == Float32(42), "exact origin borrow write")

    lease^.release()
    debug_assert(pool.offset == 0, "exact origin release")
    print("  exact origin borrow: ok")


def test_nested_exact_origin():
    var backing = alloc[UInt8](4096)
    var pool = StackPool(Int(backing), 4096)

    var a = pool.borrow[Float32, 4]()
    var b = pool.borrow[Float32, 4]()

    a.as_ptr[Float32]()[0] = Float32(1)
    b.as_ptr[Float32]()[0] = Float32(2)

    b^.release()
    a^.release()
    debug_assert(pool.offset == 0, "nested exact origin release")
    print("  nested exact origin: ok")


def main():
    print("probe 07: compile-time LIFO via exact pool origin")
    test_exact_origin_borrow()
    test_nested_exact_origin()
    print("probe 07 ok")
