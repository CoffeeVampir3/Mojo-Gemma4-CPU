from std.benchmark import keep
from std.collections import InlineArray
from std.memory import UnsafePointer, alloc


trait BurstKernel(TrivialRegisterPassable):
    def execute(mut self): ...


@fieldwise_init
struct WorkDesc(ImplicitlyCopyable, Movable):
    var lhs_addr: Int
    var rhs_addr: Int
    var dest_addr: Int
    var n: Int


@parameter
def sum_add_desc(desc_base_addr: Int, index: Int):
    var descs = UnsafePointer[WorkDesc, MutAnyOrigin](
        unsafe_from_address=desc_base_addr)
    var desc = descs[index]
    var lhs = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=desc.lhs_addr)
    var rhs = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=desc.rhs_addr)
    var dest = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=desc.dest_addr)
    for i in range(desc.n):
        dest[i] = dest[i] + lhs[i] + rhs[i]


@parameter
def product_store_desc(desc_base_addr: Int, index: Int):
    var descs = UnsafePointer[WorkDesc, MutAnyOrigin](
        unsafe_from_address=desc_base_addr)
    var desc = descs[index]
    var lhs = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=desc.lhs_addr)
    var rhs = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=desc.rhs_addr)
    var dest = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=desc.dest_addr)
    for i in range(desc.n):
        dest[i] = lhs[i] * rhs[i]


trait DispatchStep:
    @staticmethod
    def run(desc_base_addr: Int) capturing: ...


struct FunctionStep[
    body: def(desc_base_addr: Int, index: Int) capturing -> None,
    start: Int,
    count: Int,
](DispatchStep):
    @staticmethod
    def run(desc_base_addr: Int) capturing:
        for i in range(Self.start, Self.start + Self.count):
            Self.body(desc_base_addr, i)


@fieldwise_init
struct CompositeWorkKernel[
    *Steps: DispatchStep,
](BurstKernel):
    var desc_base_addr: Int

    def execute(mut self):
        comptime for i in range(len(Self.Steps)):
            Self.Steps[i].run(self.desc_base_addr)


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
) -> WorkDesc:
    return WorkDesc(Int(lhs), Int(rhs), Int(dest), n)


def main():
    comptime N = 4
    comptime SumAddFirstTwo = FunctionStep[sum_add_desc, 0, 2]
    comptime ProductStoreLast = FunctionStep[product_store_desc, 2, 1]

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

    var descs = InlineArray[WorkDesc, 3](fill=WorkDesc(0, 0, 0, 0))
    descs[0] = desc(a0, a1, ao, N)
    descs[1] = desc(b0, b1, bo, N)
    descs[2] = desc(c0, c1, co, N)

    var pool = PoolStub()
    var kernel = CompositeWorkKernel[SumAddFirstTwo, ProductStoreLast](
        Int(UnsafePointer(to=descs[0])))
    pool.dispatch(kernel)
    keep(descs[0].n)

    debug_assert(pool.dispatches == 1, "composite work should use one dispatch")
    for i in range(N):
        debug_assert(ao[i] == Float32(110 + 3 * i), "first function step mismatch")
        debug_assert(bo[i] == Float32(250 + 3 * i), "second function step mismatch")
        debug_assert(co[i] == Float32((2 + i) * (5 + i)), "third function step mismatch")

    print("function work step prototype ok")
