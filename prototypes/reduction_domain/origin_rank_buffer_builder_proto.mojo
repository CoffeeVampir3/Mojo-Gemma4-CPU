from std.collections import InlineArray
from std.memory import UnsafePointer


struct ReadRankBuffers[tp: Int, origin: ImmutOrigin]:
    var ptrs: InlineArray[UnsafePointer[Float32, Self.origin], Self.tp]
    var count: Int
    var cursor: Int

    def __init__(out self, count: Int):
        self.ptrs = InlineArray[
            UnsafePointer[Float32, Self.origin], Self.tp
        ](uninitialized=True)
        self.count = count
        self.cursor = 0

    def insert_next(mut self, ptr: UnsafePointer[Float32, Self.origin]):
        self.ptrs[self.cursor] = ptr
        self.cursor += 1

    def __getitem__(self, rank: Int) -> UnsafePointer[Float32, Self.origin]:
        return self.ptrs[rank]


struct WriteRankBuffers[tp: Int, origin: MutOrigin]:
    var ptrs: InlineArray[UnsafePointer[Float32, Self.origin], Self.tp]
    var count: Int
    var cursor: Int

    def __init__(out self, count: Int):
        self.ptrs = InlineArray[
            UnsafePointer[Float32, Self.origin], Self.tp
        ](uninitialized=True)
        self.count = count
        self.cursor = 0

    def __getitem__(self, rank: Int) -> UnsafePointer[Float32, Self.origin]:
        return self.ptrs[rank]


def check_read_builder():
    comptime N = 4
    var storage = InlineArray[Float32, N * 2](uninitialized=True)
    for i in range(N):
        storage[i] = Float32(i)
        storage[N + i] = Float32(10 + i)

    var buffers = ReadRankBuffers[2, ImmutOrigin(origin_of(storage))](count=N)
    buffers.insert_next(UnsafePointer(to=storage[0]).as_immutable())
    buffers.insert_next(UnsafePointer(to=storage[N]).as_immutable())

    for i in range(N):
        debug_assert(
            buffers[0][i] + buffers[1][i] == Float32(10 + 2 * i),
            "read rank buffer mismatch",
        )


def check_write_direct_storage():
    comptime N = 4
    var storage = InlineArray[Float32, N](uninitialized=True)
    for i in range(N):
        storage[i] = Float32(-1)

    var buffers = WriteRankBuffers[1, origin_of(storage)](count=N)
    buffers.ptrs[0] = UnsafePointer(to=storage[0])
    buffers.cursor = 1
    buffers[0][0] = Float32(9)

    debug_assert(storage[0] == Float32(9), "write rank buffer mismatch")


def main():
    check_read_builder()
    check_write_direct_storage()
    print("origin rank buffer builder prototype ok")
