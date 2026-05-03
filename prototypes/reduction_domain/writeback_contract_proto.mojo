from std.memory import UnsafePointer, alloc


trait ReductionOp:
    @staticmethod
    def combine(lhs: Float32, rhs: Float32) -> Float32: ...


struct Sum(ReductionOp):
    @staticmethod
    @always_inline
    def combine(lhs: Float32, rhs: Float32) -> Float32:
        return lhs + rhs


struct Product(ReductionOp):
    @staticmethod
    @always_inline
    def combine(lhs: Float32, rhs: Float32) -> Float32:
        return lhs * rhs


trait Writeback:
    comptime WRITES_SEPARATE_DEST: Bool
    comptime READS_OLD_DEST: Bool

    @staticmethod
    def apply(reduced: Float32, old_dest: Float32) -> Float32: ...


struct StoreReduced(Writeback):
    comptime WRITES_SEPARATE_DEST = False
    comptime READS_OLD_DEST = False

    @staticmethod
    @always_inline
    def apply(reduced: Float32, old_dest: Float32) -> Float32:
        return reduced


struct AddIntoDest(Writeback):
    comptime WRITES_SEPARATE_DEST = True
    comptime READS_OLD_DEST = True

    @staticmethod
    @always_inline
    def apply(reduced: Float32, old_dest: Float32) -> Float32:
        return old_dest + reduced


def reduce_two_ranks[
    Op: ReductionOp, W: Writeback,
](
    rank0: UnsafePointer[Float32, MutAnyOrigin],
    rank1: UnsafePointer[Float32, MutAnyOrigin],
    dest0: UnsafePointer[Float32, MutAnyOrigin],
    dest1: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
):
    for i in range(n):
        var reduced = Op.combine(rank0[i], rank1[i])
        var old0 = Float32(0)
        var old1 = Float32(0)
        comptime if W.READS_OLD_DEST:
            old0 = dest0[i]
            old1 = dest1[i]
        var out0 = W.apply(reduced, old0)
        var out1 = W.apply(reduced, old1)
        comptime if W.WRITES_SEPARATE_DEST:
            dest0[i] = out0
            dest1[i] = out1
        else:
            rank0[i] = out0
            rank1[i] = out1


def seed_pair(
    rank0: UnsafePointer[Float32, MutAnyOrigin],
    rank1: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
):
    for i in range(n):
        rank0[i] = Float32(i + 1)
        rank1[i] = Float32(10 + i)


def main():
    comptime N = 4
    var rank0 = alloc[Float32](N)
    var rank1 = alloc[Float32](N)
    var dest0 = alloc[Float32](N)
    var dest1 = alloc[Float32](N)

    seed_pair(rank0, rank1, N)
    reduce_two_ranks[Sum, StoreReduced](rank0, rank1, dest0, dest1, N)
    for i in range(N):
        var expected = Float32(11 + 2 * i)
        debug_assert(rank0[i] == expected, "sum store rank0 mismatch")
        debug_assert(rank1[i] == expected, "sum store rank1 mismatch")

    seed_pair(rank0, rank1, N)
    for i in range(N):
        dest0[i] = Float32(100 + i)
        dest1[i] = Float32(200 + i)
    reduce_two_ranks[Product, AddIntoDest](rank0, rank1, dest0, dest1, N)
    for i in range(N):
        var product = Float32((i + 1) * (10 + i))
        debug_assert(dest0[i] == Float32(100 + i) + product, "product add dest0 mismatch")
        debug_assert(dest1[i] == Float32(200 + i) + product, "product add dest1 mismatch")

    print("writeback contract prototype ok")
