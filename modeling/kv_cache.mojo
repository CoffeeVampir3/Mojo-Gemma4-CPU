from std.collections import InlineArray
from std.memory import UnsafePointer

from modeling.gemma4_common import Gemma4BaseConfig

comptime C = Gemma4BaseConfig
comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]


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

    var k_bases: InlineArray[BF16Ptr, Self.NUM_LAYERS]
    var v_bases: InlineArray[BF16Ptr, Self.NUM_LAYERS]

    def __init__(out self):
        self.k_bases = InlineArray[BF16Ptr, Self.NUM_LAYERS](
            uninitialized=True)
        self.v_bases = InlineArray[BF16Ptr, Self.NUM_LAYERS](
            uninitialized=True)

    @always_inline
    def k(self, layer: Int) -> BF16Ptr:
        return self.k_bases[layer]

    @always_inline
    def v(self, layer: Int) -> BF16Ptr:
        return self.v_bases[layer]

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

    def __init__(out self):
        self.sliding = Gemma4KVSliding[Self.degree]()
        self.full = Gemma4KVGlobal[Self.degree]()

    def bind_sliding(mut self, layer: Int, rank: Int, k: BF16Ptr, v: BF16Ptr):
        self.sliding.k_ptrs[layer * Self.degree + rank] = k
        self.sliding.v_ptrs[layer * Self.degree + rank] = v

    def bind_global(mut self, layer: Int, k: BF16Ptr, v: BF16Ptr):
        self.full.k_bases[layer] = k
        self.full.v_bases[layer] = v
