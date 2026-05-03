from std.benchmark import keep
from std.collections import InlineArray
from std.memory import UnsafePointer, alloc


trait BurstKernel(TrivialRegisterPassable):
    def execute(mut self): ...


@parameter
@always_inline
def sum_values(lhs: Float32, rhs: Float32) -> Float32:
    return lhs + rhs


@parameter
@always_inline
def product_values(lhs: Float32, rhs: Float32) -> Float32:
    return lhs * rhs


@parameter
@always_inline
def store_value(reduced: Float32, old_dest: Float32) -> Float32:
    return reduced


@parameter
@always_inline
def add_into_dest(reduced: Float32, old_dest: Float32) -> Float32:
    return old_dest + reduced


@fieldwise_init
struct ReductionDesc(ImplicitlyCopyable, Movable):
    var lhs_addr: Int
    var rhs_addr: Int
    var dest_addr: Int
    var n: Int


trait DispatchGroup:
    @staticmethod
    def run(desc_base_addr: Int) capturing: ...


struct FunctionGroupPlan[
    combine: def(lhs: Float32, rhs: Float32) capturing -> Float32,
    writeback: def(reduced: Float32, old_dest: Float32) capturing -> Float32,
    reads_old_dest: Bool,
    start: Int,
    count: Int,
](DispatchGroup):
    @staticmethod
    def run(desc_base_addr: Int):
        var descs = UnsafePointer[ReductionDesc, MutAnyOrigin](
            unsafe_from_address=desc_base_addr)
        for d in range(Self.start, Self.start + Self.count):
            var desc = descs[d]
            var lhs = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=desc.lhs_addr)
            var rhs = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=desc.rhs_addr)
            var dest = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=desc.dest_addr)
            for i in range(desc.n):
                var reduced = Self.combine(lhs[i], rhs[i])
                var old = Float32(0)
                comptime if Self.reads_old_dest:
                    old = dest[i]
                dest[i] = Self.writeback(reduced, old)


@fieldwise_init
struct CompositeFunctionKernel[
    *Groups: DispatchGroup,
](BurstKernel):
    var desc_base_addr: Int

    def execute(mut self):
        comptime for i in range(len(Self.Groups)):
            Self.Groups[i].run(self.desc_base_addr)


struct PoolStub:
    var dispatches: Int

    def __init__(out self):
        self.dispatches = 0

    def dispatch[K: BurstKernel](mut self, mut kernel: K):
        self.dispatches += 1
        kernel.execute()


def fill(ptr: UnsafePointer[Float32, MutAnyOrigin], base: Int, n: Int):
    for i in range(n):
        ptr[i] = Float32(base + i)


def desc(
    lhs: UnsafePointer[Float32, MutAnyOrigin],
    rhs: UnsafePointer[Float32, MutAnyOrigin],
    dest: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
) -> ReductionDesc:
    return ReductionDesc(Int(lhs), Int(rhs), Int(dest), n)


def main():
    comptime N = 4
    comptime SumAdd2 = FunctionGroupPlan[
        sum_values, add_into_dest, True, 0, 2,
    ]
    comptime ProductStore1 = FunctionGroupPlan[
        product_values, store_value, False, 2, 1,
    ]

    var a0 = alloc[Float32](N)
    var a1 = alloc[Float32](N)
    var ao = alloc[Float32](N)
    var b0 = alloc[Float32](N)
    var b1 = alloc[Float32](N)
    var bo = alloc[Float32](N)
    var c0 = alloc[Float32](N)
    var c1 = alloc[Float32](N)
    var co = alloc[Float32](N)

    fill(a0, 0, N)
    fill(a1, 10, N)
    fill(ao, 100, N)
    fill(b0, 20, N)
    fill(b1, 30, N)
    fill(bo, 200, N)
    fill(c0, 2, N)
    fill(c1, 5, N)
    fill(co, 0, N)

    var descs = InlineArray[ReductionDesc, 3](fill=ReductionDesc(0, 0, 0, 0))
    descs[0] = desc(a0, a1, ao, N)
    descs[1] = desc(b0, b1, bo, N)
    descs[2] = desc(c0, c1, co, N)

    var pool = PoolStub()
    var kernel = CompositeFunctionKernel[SumAdd2, ProductStore1](
        Int(UnsafePointer(to=descs[0])))
    pool.dispatch(kernel)
    keep(descs[0].n)

    debug_assert(pool.dispatches == 1, "function groups should use one dispatch")
    for i in range(N):
        debug_assert(ao[i] == Float32(110 + 3 * i), "first sum-add mismatch")
        debug_assert(bo[i] == Float32(250 + 3 * i), "second sum-add mismatch")
        debug_assert(co[i] == Float32((2 + i) * (5 + i)), "product-store mismatch")

    print("function group plan prototype ok")
