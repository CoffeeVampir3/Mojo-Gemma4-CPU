from std.pathlib import Path
from std.memory import UnsafePointer, Span
from std.os import makedirs
from std.reflection import reflect


def _default_value[T: Defaultable & ImplicitlyDestructible]() -> T:
    return T()


trait FieldwiseDefault(Defaultable):
    def __init__(out self):
        comptime for i in range(reflect[Self].field_count()):
            comptime FT = reflect[Self].field_types()[i]
            comptime if conforms_to(FT, Defaultable & ImplicitlyDestructible):
                reflect[Self].field_ref[i](self) = _default_value[FT]()
            else:
                comptime assert False, "field is not Defaultable"


def dump_bf16(
    path: Path,
    ptr: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    count: Int,
):
    var byte_count = count * 2
    var byte_ptr = ptr.bitcast[UInt8]()
    try:
        path.write_bytes(Span(ptr=byte_ptr, length=byte_count))
    except e:
        print("dump_bf16: write failed for", String(path), "-", String(e))


def ensure_dir(path: Path):
    try:
        makedirs(String(path), exist_ok=True)
    except e:
        print("ensure_dir: failed for", String(path), "-", String(e))
