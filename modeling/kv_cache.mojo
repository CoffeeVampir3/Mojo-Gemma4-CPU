from std.collections import InlineArray
from std.memory import UnsafePointer

from modeling.gemma4_common import Gemma4BaseConfig

comptime C = Gemma4BaseConfig
comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]


struct Gemma4KVSliding[degree: Int](Movable):
    comptime WINDOW = C.SLIDING_WINDOW
    comptime KV_COLS = C.KV_DIM_SLIDING // Self.degree
    comptime ROW_BYTES = Self.KV_COLS * 2
    comptime NUM_LAYERS = C.NUM_SLIDING_LAYERS
    comptime NUM_ENTRIES = Self.NUM_LAYERS * Self.degree

    var k_ptrs: InlineArray[BF16Ptr, Self.NUM_ENTRIES]
    var v_ptrs: InlineArray[BF16Ptr, Self.NUM_ENTRIES]

    def __init__(out self):
        self.k_ptrs = InlineArray[BF16Ptr, Self.NUM_ENTRIES](
            uninitialized=True)
        self.v_ptrs = InlineArray[BF16Ptr, Self.NUM_ENTRIES](
            uninitialized=True)

    @always_inline
    def k(self, layer: Int, rank: Int) -> BF16Ptr:
        return self.k_ptrs[layer * Self.degree + rank]

    @always_inline
    def v(self, layer: Int, rank: Int) -> BF16Ptr:
        return self.v_ptrs[layer * Self.degree + rank]

    @always_inline
    def k_row(self, layer: Int, rank: Int, pos: Int) -> BF16Ptr:
        return self.k(layer, rank) + (pos % Self.WINDOW) * Self.KV_COLS

    @always_inline
    def v_row(self, layer: Int, rank: Int, pos: Int) -> BF16Ptr:
        return self.v(layer, rank) + (pos % Self.WINDOW) * Self.KV_COLS

    @staticmethod
    @always_inline
    def valid_len(pos: Int) -> Int:
        if pos + 1 >= Self.WINDOW:
            return Self.WINDOW
        return pos + 1


struct Gemma4KVGlobal[degree: Int](Movable):
    comptime ROWS_PER_RANK = C.MAX_SEQ_LEN // Self.degree
    comptime KV_COLS = C.KV_DIM_FULL
    comptime ROW_BYTES = Self.KV_COLS * 2
    comptime NUM_LAYERS = C.NUM_FULL_LAYERS
    comptime NUM_ENTRIES = Self.NUM_LAYERS * Self.degree

    var k_ptrs: InlineArray[BF16Ptr, Self.NUM_ENTRIES]
    var v_ptrs: InlineArray[BF16Ptr, Self.NUM_ENTRIES]

    def __init__(out self):
        self.k_ptrs = InlineArray[BF16Ptr, Self.NUM_ENTRIES](
            uninitialized=True)
        self.v_ptrs = InlineArray[BF16Ptr, Self.NUM_ENTRIES](
            uninitialized=True)

    @always_inline
    def k(self, layer: Int, rank: Int) -> BF16Ptr:
        return self.k_ptrs[layer * Self.degree + rank]

    @always_inline
    def v(self, layer: Int, rank: Int) -> BF16Ptr:
        return self.v_ptrs[layer * Self.degree + rank]

    @always_inline
    def k_row(self, layer: Int, rank: Int, pos: Int) -> BF16Ptr:
        return self.k(layer, rank) + (pos // Self.degree) * Self.KV_COLS

    @always_inline
    def v_row(self, layer: Int, rank: Int, pos: Int) -> BF16Ptr:
        return self.v(layer, rank) + (pos // Self.degree) * Self.KV_COLS

    @staticmethod
    @always_inline
    def owner(pos: Int) -> Int:
        return pos % Self.degree

    @staticmethod
    @always_inline
    def local_index(pos: Int) -> Int:
        return pos // Self.degree

    @staticmethod
    @always_inline
    def valid_count(rank: Int, pos: Int) -> Int:
        if pos < 0:
            return 0
        if rank <= pos % Self.degree:
            return pos // Self.degree + 1
        return pos // Self.degree


struct Gemma4KV[degree: Int](Movable):
    var sliding: Gemma4KVSliding[Self.degree]
    var full: Gemma4KVGlobal[Self.degree]
    var pos: Int
    var sliding_valid: Bool

    def __init__(out self):
        self.sliding = Gemma4KVSliding[Self.degree]()
        self.full = Gemma4KVGlobal[Self.degree]()
        self.pos = -1
        self.sliding_valid = False

    def bind_sliding(mut self, layer: Int, rank: Int, k: BF16Ptr, v: BF16Ptr):
        self.sliding.k_ptrs[layer * Self.degree + rank] = k
        self.sliding.v_ptrs[layer * Self.degree + rank] = v

    def bind_global(mut self, layer: Int, rank: Int, k: BF16Ptr, v: BF16Ptr):
        self.full.k_ptrs[layer * Self.degree + rank] = k
        self.full.v_ptrs[layer * Self.degree + rank] = v

    def advance(mut self):
        self.pos += 1

    def rewind(mut self, new_pos: Int):
        if new_pos < self.pos:
            self.sliding_valid = False
        self.pos = new_pos

    def reset(mut self):
        self.pos = -1
        self.sliding_valid = False

    def mark_sliding_valid(mut self):
        self.sliding_valid = True
