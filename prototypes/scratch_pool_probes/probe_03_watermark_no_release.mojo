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


struct WatermarkPool:
    var base: Int
    var capacity: Int
    var offset: Int

    def __init__(out self, base: Int, capacity: Int):
        self.base = base
        self.capacity = capacity
        self.offset = 0

    def save(self) -> Int:
        return self.offset

    def restore(mut self, watermark: Int):
        self.offset = watermark

    def borrow[T: AnyType, count: Int](mut self) -> RegionPtr:
        comptime byte_size = typed_bytes[T, count]()
        var addr = self.base + self.offset
        self.offset += byte_size
        if self.offset > self.capacity:
            abort("WatermarkPool: exceeded capacity")
        return RegionPtr(addr)


struct RegionPtr:
    var addr: Int

    def __init__(out self, addr: Int):
        self.addr = addr

    @always_inline
    def ptr[T: AnyType](self) -> UnsafePointer[T, MutAnyOrigin]:
        return UnsafePointer[T, MutAnyOrigin](unsafe_from_address=self.addr)


def test_watermark_phase_pattern():
    var backing = alloc[UInt8](4096)
    var pool = WatermarkPool(Int(backing), 4096)

    var phase1_mark = pool.save()

    var a = pool.borrow[Float32, 16]()
    var b = pool.borrow[Float32, 8]()
    a.ptr[Float32]()[0] = Float32(10)
    b.ptr[Float32]()[0] = Float32(20)

    debug_assert(a.ptr[Float32]()[0] == Float32(10), "phase1 a")
    debug_assert(b.ptr[Float32]()[0] == Float32(20), "phase1 b")

    pool.restore(phase1_mark)

    var c = pool.borrow[Float32, 16]()
    c.ptr[Float32]()[0] = Float32(30)
    debug_assert(c.ptr[Float32]()[0] == Float32(30), "phase2 c")

    pool.restore(phase1_mark)
    debug_assert(pool.offset == 0, "back to start")
    print("  watermark phase pattern: ok")


def test_watermark_nested_phases():
    var backing = alloc[UInt8](8192)
    var pool = WatermarkPool(Int(backing), 8192)

    var outer_mark = pool.save()

    var persistent = pool.borrow[Float32, 32]()
    persistent.ptr[Float32]()[0] = Float32(100)

    var inner_mark = pool.save()
    var temp = pool.borrow[Float32, 64]()
    temp.ptr[Float32]()[0] = Float32(200)
    pool.restore(inner_mark)

    debug_assert(persistent.ptr[Float32]()[0] == Float32(100), "persistent survived")

    var inner_mark2 = pool.save()
    var temp2 = pool.borrow[Float32, 64]()
    temp2.ptr[Float32]()[0] = Float32(300)
    pool.restore(inner_mark2)

    debug_assert(persistent.ptr[Float32]()[0] == Float32(100), "persistent still ok")

    pool.restore(outer_mark)
    debug_assert(pool.offset == 0, "fully restored")
    print("  watermark nested phases: ok")


def test_watermark_vs_current_tradeoffs():
    var backing = alloc[UInt8](4096)
    var pool = WatermarkPool(Int(backing), 4096)

    var m = pool.save()
    var r1 = pool.borrow[Float32, 4]()
    var r2 = pool.borrow[Float32, 4]()
    var r3 = pool.borrow[Float32, 4]()

    r1.ptr[Float32]()[0] = Float32(1)
    r2.ptr[Float32]()[0] = Float32(2)
    r3.ptr[Float32]()[0] = Float32(3)

    pool.restore(m)

    debug_assert(pool.offset == 0, "batch restore")
    print("  watermark batch restore: ok")


def main():
    print("probe 03: watermark (no per-lease release)")
    test_watermark_phase_pattern()
    test_watermark_nested_phases()
    test_watermark_vs_current_tradeoffs()
    print("probe 03 ok")
