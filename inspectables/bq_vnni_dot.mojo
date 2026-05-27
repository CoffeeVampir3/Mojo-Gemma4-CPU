from std.sys import argv, llvm_intrinsic
from std.sys.info import simd_width_of, CompilationTarget
from std.benchmark import keep
from std.memory import UnsafePointer, alloc

comptime I8Ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime WI = simd_width_of[DType.int32]()


@always_inline
def vpdpbusd[width: Int](
    acc: SIMD[DType.int32, width],
    a: SIMD[DType.uint8, width * 4],
    b: SIMD[DType.int8, width * 4],
) -> SIMD[DType.int32, width]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd." + String(width * 32),
        SIMD[DType.int32, width],
    ](acc, a, b)


# ---- act_broadcast_vnni: A (insert chain) vs B (vpbroadcastd) ----

@always_inline
def act_broadcast_vnni_a[width: Int](act_row: I8Ptr, k_pos: Int) -> SIMD[
    DType.uint8, width * 4,
]:
    var b4 = (act_row + k_pos).bitcast[UInt8]().load[width=4]() ^ SIMD[
        DType.uint8, 4](0x80)
    var out = SIMD[DType.uint8, width * 4]()
    comptime for lane in range(width):
        out = out.insert[offset = lane * 4](b4)
    return out


@always_inline
def act_broadcast_vnni_b[width: Int](act_row: I8Ptr, k_pos: Int) -> SIMD[
    DType.uint8, width * 4,
]:
    var b4 = (act_row + k_pos).bitcast[UInt8]().load[width=4]() ^ SIMD[
        DType.uint8, 4](0x80)
    var b8 = b4.join(b4)
    var b16 = b8.join(b8)
    comptime if width == 8:
        return rebind[SIMD[DType.uint8, width * 4]](b16.join(b16))
    else:
        var b32 = b16.join(b16)
        return rebind[SIMD[DType.uint8, width * 4]](b32.join(b32))


@always_inline
def dot_loaded[width: Int](
    acc: SIMD[DType.int32, width],
    act_bytes: SIMD[DType.uint8, width * 4],
    weights: SIMD[DType.int8, width * 4],
) -> SIMD[DType.int32, width]:
    return vpdpbusd[width](acc, act_bytes, weights)


# ---- vnni_shifted_dot: A (per-iter reduce_add) vs B (vector accumulate) ----

@always_inline
def vnni_shifted_dot_a[block: Int, emit_rhs_sum: Bool](
    a: I8Ptr, b: I8Ptr,
) -> Tuple[SIMD[DType.int32, WI], Int32]:
    comptime bytes = WI * 4
    var acc = SIMD[DType.int32, WI](0)
    var rhs_sum = Int32(0)
    for k in range(0, block, bytes):
        var av = (a + k).bitcast[UInt8]().load[width=bytes]() ^ SIMD[
            DType.uint8, bytes](0x80)
        var bv = (b + k).load[width=bytes]()
        acc = vpdpbusd[WI](acc, av, bv)
        comptime if emit_rhs_sum:
            rhs_sum += bv.cast[DType.int32]().reduce_add()
    return (acc, rhs_sum)


@always_inline
def vnni_shifted_dot_b[block: Int, emit_rhs_sum: Bool](
    a: I8Ptr, b: I8Ptr,
) -> Tuple[SIMD[DType.int32, WI], Int32]:
    comptime bytes = WI * 4
    var acc = SIMD[DType.int32, WI](0)
    var sum_vec = SIMD[DType.int32, WI](0)
    var ones = SIMD[DType.uint8, bytes](1)
    for k in range(0, block, bytes):
        var av = (a + k).bitcast[UInt8]().load[width=bytes]() ^ SIMD[
            DType.uint8, bytes](0x80)
        var bv = (b + k).load[width=bytes]()
        acc = vpdpbusd[WI](acc, av, bv)
        comptime if emit_rhs_sum:
            sum_vec = vpdpbusd[WI](sum_vec, ones, bv)
    var rhs_sum = sum_vec.reduce_add() if emit_rhs_sum else Int32(0)
    return (acc, rhs_sum)


# ---- exported, non-inlined probes for objdump ----

@no_inline
@export
def probe_vpdpbusd(
    acc: SIMD[DType.int32, WI],
    a: SIMD[DType.uint8, WI * 4],
    b: SIMD[DType.int8, WI * 4],
) -> SIMD[DType.int32, WI]:
    return vpdpbusd[WI](acc, a, b)


@no_inline
@export
def probe_dot_loaded(
    acc: SIMD[DType.int32, WI],
    a: SIMD[DType.uint8, WI * 4],
    b: SIMD[DType.int8, WI * 4],
) -> SIMD[DType.int32, WI]:
    return dot_loaded[WI](acc, a, b)


@no_inline
@export
def probe_act_broadcast_a(act_row: I8Ptr, k_pos: Int) -> SIMD[DType.uint8, WI * 4]:
    return act_broadcast_vnni_a[WI](act_row, k_pos)


@no_inline
@export
def probe_act_broadcast_b(act_row: I8Ptr, k_pos: Int) -> SIMD[DType.uint8, WI * 4]:
    return act_broadcast_vnni_b[WI](act_row, k_pos)


@no_inline
@export
def probe_shifted_dot_a(a: I8Ptr, b: I8Ptr) -> Int32:
    var r = vnni_shifted_dot_a[256, True](a, b)
    return r[0].reduce_add() - Int32(128) * r[1]


@no_inline
@export
def probe_shifted_dot_b(a: I8Ptr, b: I8Ptr) -> Int32:
    var r = vnni_shifted_dot_b[256, True](a, b)
    return r[0].reduce_add() - Int32(128) * r[1]


def main():
    print("=== has_vnni:", CompilationTarget.has_vnni(), " WI:", WI, "===")
    var seed = Int(len(argv()))
    var a = alloc[Int8](256).as_any_origin()
    var b = alloc[Int8](256).as_any_origin()
    for i in range(256):
        a[i] = Int8((i + seed) % 127)
        b[i] = Int8((i * 3 + seed) % 127)

    var acc = SIMD[DType.int32, WI](Int32(seed))
    var av = SIMD[DType.uint8, WI * 4](UInt8(seed))
    var bv = SIMD[DType.int8, WI * 4](Int8(seed))
    keep(probe_vpdpbusd(acc, av, bv))
    keep(probe_dot_loaded(acc, av, bv))
    keep(probe_act_broadcast_a(a, seed))
    keep(probe_act_broadcast_b(a, seed))
    keep(probe_shifted_dot_a(a, b))
    keep(probe_shifted_dot_b(a, b))
    a.free()
    b.free()
