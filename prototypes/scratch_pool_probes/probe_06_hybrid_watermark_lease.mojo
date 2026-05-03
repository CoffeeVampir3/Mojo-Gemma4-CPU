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


struct PhasePool:
    var base: Int
    var capacity: Int
    var offset: Int
    var high_water: Int

    def __init__(out self, base: Int, capacity: Int):
        self.base = base
        self.capacity = capacity
        self.offset = 0
        self.high_water = 0

    def mark(self) -> Int:
        return self.offset

    def restore(mut self, watermark: Int):
        self.offset = watermark

    def bump[T: AnyType, count: Int](mut self) -> Int:
        comptime byte_size = typed_bytes[T, count]()
        var addr = self.base + self.offset
        self.offset += byte_size
        if self.offset > self.capacity:
            abort("PhasePool: exceeded capacity")
        if self.offset > self.high_water:
            self.high_water = self.offset
        return addr


@explicit_destroy
struct PhaseLease(Movable):
    var watermark: Int
    var pool_ptr: UnsafePointer[PhasePool, MutAnyOrigin]

    def __init__(out self, watermark: Int,
                 pool_ptr: UnsafePointer[PhasePool, MutAnyOrigin]):
        self.watermark = watermark
        self.pool_ptr = pool_ptr

    def release(deinit self):
        self.pool_ptr[].restore(self.watermark)

    @always_inline
    def bump[
        o: MutOrigin, //,
        T: AnyType, count: Int,
    ](ref [o] self) -> UnsafePointer[T, o]:
        var addr = self.pool_ptr[].bump[T, count]()
        return UnsafePointer[T, o](unsafe_from_address=addr)


def begin_phase(mut pool: PhasePool) -> PhaseLease:
    var wm = pool.mark()
    return PhaseLease(wm,
        UnsafePointer[PhasePool, MutAnyOrigin](
            unsafe_from_address=Int(UnsafePointer(to=pool))))


def test_phase_lease_basic():
    var backing = alloc[UInt8](4096)
    var pool = PhasePool(Int(backing), 4096)

    var phase = begin_phase(pool)
    var a = phase.bump[Float32, 16]()
    var b = phase.bump[Float32, 8]()

    a[0] = Float32(1)
    b[0] = Float32(2)
    debug_assert(a[0] == Float32(1), "phase a")
    debug_assert(b[0] == Float32(2), "phase b")

    phase^.release()
    debug_assert(pool.offset == 0, "restored to 0")
    print("  phase lease basic: ok")


def test_phase_nesting():
    var backing = alloc[UInt8](8192)
    var pool = PhasePool(Int(backing), 8192)

    var persistent_addr = pool.bump[Float32, 32]()
    var persistent = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=persistent_addr)
    persistent[0] = Float32(999)

    var phase1 = begin_phase(pool)
    var temp1 = phase1.bump[Float32, 64]()
    temp1[0] = Float32(100)
    debug_assert(persistent[0] == Float32(999), "persistent during phase1")
    phase1^.release()

    var phase2 = begin_phase(pool)
    var temp2 = phase2.bump[Float32, 32]()
    temp2[0] = Float32(200)
    debug_assert(persistent[0] == Float32(999), "persistent during phase2")
    phase2^.release()

    debug_assert(persistent[0] == Float32(999), "persistent survived both phases")
    print("  phase nesting: ok")


def test_phase_multi_buffer():
    var backing = alloc[UInt8](4096)
    var pool = PhasePool(Int(backing), 4096)

    var phase = begin_phase(pool)

    var q = phase.bump[Float32, 32]()
    var k = phase.bump[Float32, 16]()
    var v = phase.bump[Float32, 16]()

    q[0] = Float32(1)
    k[0] = Float32(2)
    v[0] = Float32(3)

    debug_assert(q[0] == Float32(1), "q")
    debug_assert(k[0] == Float32(2), "k")
    debug_assert(v[0] == Float32(3), "v")

    phase^.release()

    var phase2 = begin_phase(pool)
    var gate = phase2.bump[Float32, 32]()
    var up = phase2.bump[Float32, 32]()
    gate[0] = Float32(10)
    up[0] = Float32(20)
    debug_assert(gate[0] == Float32(10), "gate reused space")
    phase2^.release()

    debug_assert(pool.offset == 0, "fully restored")
    print("  phase multi-buffer: ok")


def test_high_water_tracking():
    var backing = alloc[UInt8](4096)
    var pool = PhasePool(Int(backing), 4096)

    var phase1 = begin_phase(pool)
    _ = phase1.bump[Float32, 64]()
    var hw1 = pool.high_water
    phase1^.release()

    var phase2 = begin_phase(pool)
    _ = phase2.bump[Float32, 32]()
    var hw2 = pool.high_water
    phase2^.release()

    debug_assert(hw2 == hw1, "high water only grows")
    debug_assert(pool.offset == 0, "restored")
    print("  high water tracking: ok")


def main():
    print("probe 06: hybrid watermark + lease")
    test_phase_lease_basic()
    test_phase_nesting()
    test_phase_multi_buffer()
    test_high_water_tracking()
    print("probe 06 ok")
