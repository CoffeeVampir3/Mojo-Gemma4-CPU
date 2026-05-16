from std.reflection import reflect


trait Encoding:
    comptime DTYPE: DType
    comptime ELEMENT_BYTES: Int


struct BF16(Encoding):
    comptime DTYPE = DType.bfloat16
    comptime ELEMENT_BYTES = 2


trait ShapeLike:
    comptime DATA_N: Int
    comptime DATA_M: Int

    @staticmethod
    def bytes[E: Encoding]() -> Int: ...


struct Shape[n: Int, m: Int](ShapeLike, Copyable, ImplicitlyCopyable):
    comptime DATA_N = Self.n
    comptime DATA_M = Self.m

    @staticmethod
    def bytes[E: Encoding]() -> Int:
        return Self.n * Self.m * E.ELEMENT_BYTES


trait SlotLike:
    comptime E: Encoding
    comptime S: ShapeLike
    comptime NAME: StaticString

    @always_inline
    def set_offset(mut self, off: Int): ...

    @always_inline
    def get_offset(self) -> Int: ...


@fieldwise_init
struct WeightSlot[
    E_: Encoding, S_: ShapeLike, name_: StaticString,
](SlotLike, Copyable, ImplicitlyCopyable):
    comptime E = Self.E_
    comptime S = Self.S_
    comptime NAME = Self.name_
    var offset: Int

    @always_inline
    def set_offset(mut self, off: Int):
        self.offset = off

    @always_inline
    def get_offset(self) -> Int:
        return self.offset


def align_up(v: Int) -> Int:
    return ((v + 63) // 64) * 64


@fieldwise_init
struct Body(Copyable, ImplicitlyCopyable):
    var input_norm: WeightSlot[BF16, Shape[2816, 1], "input_layernorm.weight"]
    var q_proj: WeightSlot[BF16, Shape[4096, 2816], "self_attn.q_proj.weight"]
    var k_proj: WeightSlot[BF16, Shape[2048, 2816], "self_attn.k_proj.weight"]


def init_offsets[T: AnyType](mut t: T) -> Int:
    var off = 0
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, SlotLike):
            ref slot = reflect[T].field_ref[i](t)
            slot.set_offset(off)
            off = align_up(off + FT.S.bytes[FT.E]())
    return off


def main():
    var body = Body(
        input_norm = WeightSlot[BF16, Shape[2816, 1], "input_layernorm.weight"](offset=-1),
        q_proj = WeightSlot[BF16, Shape[4096, 2816], "self_attn.q_proj.weight"](offset=-1),
        k_proj = WeightSlot[BF16, Shape[2048, 2816], "self_attn.k_proj.weight"](offset=-1),
    )
    var layer_bytes = init_offsets(body)
    print("layer bytes total:", layer_bytes)
    print("input_norm.offset:", body.input_norm.offset)
    print("q_proj.offset:    ", body.q_proj.offset)
    print("k_proj.offset:    ", body.k_proj.offset)
