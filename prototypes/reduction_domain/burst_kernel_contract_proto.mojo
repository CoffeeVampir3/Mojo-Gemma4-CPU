from std.memory import UnsafePointer, alloc


trait BurstKernel(TrivialRegisterPassable):
    def execute(mut self): ...


trait ReductionOp:
    @staticmethod
    def combine(lhs: Float32, rhs: Float32) -> Float32: ...


struct Sum(ReductionOp):
    @staticmethod
    @always_inline
    def combine(lhs: Float32, rhs: Float32) -> Float32:
        return lhs + rhs


trait Writeback:
    comptime READS_OLD_DEST: Bool

    @staticmethod
    def apply(reduced: Float32, old_dest: Float32) -> Float32: ...


struct AddIntoDest(Writeback):
    comptime READS_OLD_DEST = True

    @staticmethod
    @always_inline
    def apply(reduced: Float32, old_dest: Float32) -> Float32:
        return old_dest + reduced


@fieldwise_init
struct ReductionKernel[
    Op: ReductionOp, W: Writeback,
](BurstKernel):
    var lhs: UnsafePointer[Float32, MutAnyOrigin]
    var rhs: UnsafePointer[Float32, MutAnyOrigin]
    var dest: UnsafePointer[Float32, MutAnyOrigin]
    var n: Int

    def execute(mut self):
        for i in range(self.n):
            var reduced = Self.Op.combine(self.lhs[i], self.rhs[i])
            var old = Float32(0)
            comptime if Self.W.READS_OLD_DEST:
                old = self.dest[i]
            self.dest[i] = Self.W.apply(reduced, old)


def main():
    comptime N = 4
    var lhs = alloc[Float32](N)
    var rhs = alloc[Float32](N)
    var dest = alloc[Float32](N)

    for i in range(N):
        lhs[i] = Float32(i)
        rhs[i] = Float32(10 + i)
        dest[i] = Float32(100 + i)

    var kernel = ReductionKernel[Sum, AddIntoDest](lhs, rhs, dest, N)
    kernel.execute()

    for i in range(N):
        debug_assert(dest[i] == Float32(110 + 3 * i), "burst kernel contract mismatch")

    print("burst kernel contract prototype ok")
