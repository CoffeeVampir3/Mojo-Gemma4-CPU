from std.memory import UnsafePointer, alloc
from std.os import abort

from kernels.rope import init_rope_table_partial_strided


comptime HALF = 64
comptime MAX_POS = 16
comptime DEGREE = 4
comptime ROWS = MAX_POS // DEGREE
comptime HEAD_DIM = 512

comptime F32Ptr = UnsafePointer[Float32, MutExternalOrigin]


def check(ok: Bool, msg: StringSlice):
    if not ok:
        abort(String(t"FAIL: {msg}"))


def check_same(a: F32Ptr, b: F32Ptr, count: Int, label: StringSlice):
    for i in range(count):
        var av = Float32(a[i])
        var bv = Float32(b[i])
        check(av == bv, String(t"{label} [{i}] expected={av} got={bv}"))


def main():
    var direct_cos = alloc[Float32](MAX_POS * HALF)
    var direct_sin = alloc[Float32](MAX_POS * HALF)
    var sharded_cos = alloc[Float32](MAX_POS * HALF)
    var sharded_sin = alloc[Float32](MAX_POS * HALF)

    init_rope_table_partial_strided[HALF, MAX_POS](
        direct_cos, direct_sin, 1000000.0, HEAD_DIM, 0, 1)

    for rank in range(DEGREE):
        var off = rank * ROWS * HALF
        init_rope_table_partial_strided[HALF, ROWS](
            sharded_cos + off, sharded_sin + off,
            1000000.0, HEAD_DIM, rank, DEGREE)

    for pos in range(MAX_POS):
        var rank = pos % DEGREE
        var row = pos // DEGREE
        var sharded_off = (rank * ROWS + row) * HALF
        check_same(
            direct_cos + pos * HALF, sharded_cos + sharded_off,
            HALF, String(t"full rope cos pos={pos}"))
        check_same(
            direct_sin + pos * HALF, sharded_sin + sharded_off,
            HALF, String(t"full rope sin pos={pos}"))

    direct_cos.free()
    direct_sin.free()
    sharded_cos.free()
    sharded_sin.free()
    print("rope tests passed")
