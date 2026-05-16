from std.reflection import reflect


trait Encoding:
    comptime DTYPE: DType


struct BF16(Encoding):
    comptime DTYPE = DType.bfloat16


trait ShapeLike:
    comptime DATA_N: Int
    comptime DATA_M: Int


struct Shape[n: Int, m: Int](ShapeLike, Copyable, ImplicitlyCopyable):
    comptime DATA_N = Self.n
    comptime DATA_M = Self.m


# Probe 1: StaticString as a trait member (instead of StringLiteral)
trait SlotLikeA:
    comptime E: Encoding
    comptime S: ShapeLike
    comptime NAME: StaticString


@fieldwise_init
struct WeightSlotA[
    E_: Encoding, S_: ShapeLike, name_: StaticString,
](SlotLikeA, Copyable, ImplicitlyCopyable):
    comptime E = Self.E_
    comptime S = Self.S_
    comptime NAME = Self.name_


# Probe 2: parametric helper that pulls NAME off the concrete WeightSlot type
def slot_name[T: SlotLikeA]() -> StaticString:
    return T.NAME


@fieldwise_init
struct BodyA(Copyable, ImplicitlyCopyable):
    var input_norm: WeightSlotA[BF16, Shape[2816, 1], "input_layernorm.weight"]
    var q_proj: WeightSlotA[BF16, Shape[4096, 2816], "self_attn.q_proj.weight"]


def main():
    print("=== Probe: StaticString as trait comptime member ===")
    print("field_count:", reflect[BodyA].field_count())
    comptime for i in range(reflect[BodyA].field_count()):
        comptime FT = reflect[BodyA].field_types()[i]
        comptime if conforms_to(FT, SlotLikeA):
            print("  slot:", FT.NAME, "rows:", FT.S.DATA_N, "cols:", FT.S.DATA_M,
                  "dtype:", FT.E.DTYPE)
