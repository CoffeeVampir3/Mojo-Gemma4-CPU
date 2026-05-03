from std.sys.info import size_of
from std.os import abort

comptime SCRATCH_LEASE_ALIGNMENT = 64


@always_inline
def scratch_block_bytes[nbytes: Int]() -> Int:
    return ((nbytes + SCRATCH_LEASE_ALIGNMENT - 1) // SCRATCH_LEASE_ALIGNMENT) * SCRATCH_LEASE_ALIGNMENT


@always_inline
def scratch_lease_bytes[T: AnyType, count: Int]() -> Int:
    comptime raw_byte_size = count * size_of[T]()
    return scratch_block_bytes[raw_byte_size]()


@explicit_destroy
struct ScratchLease(Movable):
    var addr: Int
    var byte_size: Int
    var pool_end_offset: Int
    var pool_offset_ptr: UnsafePointer[Int, MutAnyOrigin]

    def __init__(out self, addr: Int, byte_size: Int,
                 pool_end_offset: Int,
                 pool_offset_ptr: UnsafePointer[Int, MutAnyOrigin]):
        self.addr = addr
        self.byte_size = byte_size
        self.pool_end_offset = pool_end_offset
        self.pool_offset_ptr = pool_offset_ptr

    def release(deinit self):
        var top = self.pool_offset_ptr[]
        if self.pool_end_offset != top:
            print("ScratchPool: non-LIFO release detected. pool_end_offset=",
                  self.pool_end_offset, " byte_size=", self.byte_size,
                  " actual_top=", top)
            abort("ScratchPool: non-LIFO release")
        self.pool_offset_ptr[] -= self.byte_size

    @always_inline
    def as_ptr[
        o: MutOrigin, //,
        T: AnyType,
    ](
        ref [o] self, element_offset: Int = 0,
    ) -> UnsafePointer[T, o]:
        return UnsafePointer[T, o](
            unsafe_from_address=self.addr
                + element_offset * size_of[T]())


struct ScratchPool(Movable):
    var base: Int
    var capacity: Int
    var offset: Int
    var high_water: Int

    def __init__(out self, base: Int, capacity: Int):
        self.base = base
        self.capacity = capacity
        self.offset = 0
        self.high_water = 0

    def borrow[T: AnyType, count: Int](mut self) -> ScratchLease:
        comptime byte_size = scratch_lease_bytes[T, count]()
        var lease_offset = self.offset
        self.offset += byte_size
        if self.offset > self.capacity:
            abort("ScratchPool: cumulative borrows exceed capacity")
        if self.offset > self.high_water:
            self.high_water = self.offset
            print("scratch: new peak " + String(self.high_water)
                  + " / " + String(self.capacity) + " bytes")
        return ScratchLease(
            addr=self.base + lease_offset,
            byte_size=byte_size,
            pool_end_offset=self.offset,
            pool_offset_ptr=UnsafePointer[Int, MutAnyOrigin](
                unsafe_from_address=Int(UnsafePointer(to=self.offset))
            ),
        )
