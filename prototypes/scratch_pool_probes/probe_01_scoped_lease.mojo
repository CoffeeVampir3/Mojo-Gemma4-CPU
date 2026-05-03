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


struct ScratchArena:
    var base: Int
    var capacity: Int
    var offset: Int
    var high_water: Int

    def __init__(out self, base: Int, capacity: Int):
        self.base = base
        self.capacity = capacity
        self.offset = 0
        self.high_water = 0

    def borrow[T: AnyType, count: Int](mut self) -> ScopedLease:
        comptime byte_size = typed_bytes[T, count]()
        var lease_offset = self.offset
        self.offset += byte_size
        if self.offset > self.capacity:
            abort("ScratchArena: exceeded capacity")
        if self.offset > self.high_water:
            self.high_water = self.offset
        return ScopedLease(
            addr=self.base + lease_offset,
            byte_size=byte_size,
            pool_offset_ptr=UnsafePointer[Int, MutAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=self.offset))),
        )


@explicit_destroy
struct ScopedLease(Movable):
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

    @always_inline
    def as_immut_ptr[
        o: ImmutOrigin, //, T: AnyType,
    ](ref [o] self) -> UnsafePointer[T, o]:
        return UnsafePointer[T, o](unsafe_from_address=self.addr)


def test_scoped_lease_carries_base():
    var backing = alloc[UInt8](4096)
    var arena = ScratchArena(Int(backing), 4096)

    var lease_a = arena.borrow[Float32, 16]()
    var ptr_a = lease_a.as_ptr[Float32]()
    ptr_a[0] = Float32(42)
    debug_assert(ptr_a[0] == Float32(42), "write through scoped lease failed")

    var lease_b = arena.borrow[Float32, 8]()
    var ptr_b = lease_b.as_ptr[Float32]()
    ptr_b[0] = Float32(99)
    debug_assert(ptr_b[0] == Float32(99), "second lease write failed")
    debug_assert(ptr_a[0] == Float32(42), "first lease untouched")

    lease_b^.release()
    lease_a^.release()

    debug_assert(arena.offset == 0, "arena should be empty after releases")
    print("  scoped lease carries base: ok")


def test_origin_tied_to_lease():
    var backing = alloc[UInt8](4096)
    var arena = ScratchArena(Int(backing), 4096)

    var lease = arena.borrow[Float32, 4]()
    var ptr = lease.as_ptr[Float32]()
    ptr[0] = Float32(7)
    ptr[1] = Float32(8)
    debug_assert(ptr[0] == Float32(7), "origin tie read failed")
    lease^.release()

    print("  origin tied to lease ref: ok")


def main():
    print("probe 01: scoped lease (base baked in)")
    test_scoped_lease_carries_base()
    test_origin_tied_to_lease()
    print("probe 01 ok")
