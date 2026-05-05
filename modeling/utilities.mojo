from std.pathlib import Path
from std.memory import UnsafePointer, Span
from std.os import makedirs


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
