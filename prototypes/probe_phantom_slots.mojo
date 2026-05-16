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
    comptime NAME: StringLiteral


@fieldwise_init
struct WeightSlot[
    E_: Encoding, S_: ShapeLike, name_: StringLiteral,
](SlotLike, Copyable, ImplicitlyCopyable):
    comptime E = Self.E_
    comptime S = Self.S_
    comptime NAME = Self.name_


@fieldwise_init
struct Body(Copyable, ImplicitlyCopyable):
    var input_norm: WeightSlot[BF16, Shape[2816, 1], "input_layernorm.weight"]
    var q_proj: WeightSlot[BF16, Shape[4096, 2816], "self_attn.q_proj.weight"]
    var k_proj: WeightSlot[BF16, Shape[2048, 2816], "self_attn.k_proj.weight"]


def main():
    # construct empty body — all slots are phantom (zero fields)
    var body = Body(
        input_norm = WeightSlot[BF16, Shape[2816, 1], "input_layernorm.weight"](),
        q_proj = WeightSlot[BF16, Shape[4096, 2816], "self_attn.q_proj.weight"](),
        k_proj = WeightSlot[BF16, Shape[2048, 2816], "self_attn.k_proj.weight"](),
    )
    _ = body

    # enumerate via reflection
    print("field_count:", reflect[Body].field_count())
    comptime for i in range(reflect[Body].field_count()):
        comptime FT = reflect[Body].field_types()[i]
        comptime if conforms_to(FT, SlotLike):
            print("  slot name:", FT.NAME, "dtype-elt-bytes:", FT.E.ELEMENT_BYTES,
                  "rows:", FT.S.DATA_N, "cols:", FT.S.DATA_M,
                  "bytes:", FT.S.bytes[FT.E]())
