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


struct StaticScratch[total_bytes: Int]:
    var base: Int

    def __init__(out self, base: Int):
        self.base = base

    @always_inline
    def ptr_at[T: AnyType, byte_offset: Int](self) -> UnsafePointer[T, MutAnyOrigin]:
        comptime assert byte_offset >= 0, "negative offset"
        comptime assert byte_offset + size_of[T]() <= Self.total_bytes, "exceeds scratch capacity"
        return UnsafePointer[T, MutAnyOrigin](
            unsafe_from_address=self.base + byte_offset)


def test_static_scratch_basic():
    var backing = alloc[UInt8](1024)
    var scratch = StaticScratch[1024](Int(backing))

    var p0 = scratch.ptr_at[Float32, 0]()
    p0[0] = Float32(42)
    debug_assert(p0[0] == Float32(42), "static slot 0")

    comptime SLOT1 = aligned_bytes[16 * size_of[Float32]()]()
    var p1 = scratch.ptr_at[Float32, SLOT1]()
    p1[0] = Float32(99)
    debug_assert(p1[0] == Float32(99), "static slot 1")
    debug_assert(p0[0] == Float32(42), "slot 0 untouched")

    print("  static scratch basic: ok")


def test_static_scratch_phase_reuse():
    comptime BUF_A = 0
    comptime BUF_B = aligned_bytes[32 * size_of[Float32]()]()
    comptime TOTAL = BUF_B + aligned_bytes[16 * size_of[Float32]()]()

    var backing = alloc[UInt8](TOTAL)
    var scratch = StaticScratch[TOTAL](Int(backing))

    var a = scratch.ptr_at[Float32, BUF_A]()
    var b = scratch.ptr_at[Float32, BUF_B]()

    a[0] = Float32(1)
    b[0] = Float32(2)
    debug_assert(a[0] == Float32(1), "phase reuse a")
    debug_assert(b[0] == Float32(2), "phase reuse b")

    a[0] = Float32(10)
    debug_assert(a[0] == Float32(10), "a reused")
    debug_assert(b[0] == Float32(2), "b untouched")

    print("  static scratch phase reuse: ok")


def test_static_overlap_alias():
    comptime TOTAL = 256

    var backing = alloc[UInt8](TOTAL)
    var scratch = StaticScratch[TOTAL](Int(backing))

    var phase1_buf = scratch.ptr_at[Float32, 0]()
    phase1_buf[0] = Float32(111)

    var phase2_buf = scratch.ptr_at[Float32, 0]()
    debug_assert(phase2_buf[0] == Float32(111), "sees phase1 data (aliased)")

    phase2_buf[0] = Float32(222)
    debug_assert(phase1_buf[0] == Float32(222), "both see the write (aliased)")

    print("  static overlap alias: ok (expected - no isolation)")


def main():
    print("probe 04: comptime layout (no allocator)")
    test_static_scratch_basic()
    test_static_scratch_phase_reuse()
    test_static_overlap_alias()
    print("probe 04 ok")
