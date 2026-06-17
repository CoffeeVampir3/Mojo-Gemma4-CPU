from std.os import SEEK_SET
from std.memory import Span, UnsafePointer, memcpy


def probe_rw_overwrite() raises:
    var path = String("prototypes/probe.bin")
    with open(path, "w") as f:
        var init = List[UInt8](length=16, fill=0xAA)
        f.write_all(Span(init))

    with open(path, "rw") as f:
        var pos = f.seek(4)
        print(t"  seek(4) -> {pos}")
        var patch = List[UInt8](length=4, fill=0xBB)
        f.write_all(Span(patch))

    var data: List[UInt8]
    with open(path, "r") as f:
        data = f.read_bytes()
    print(t"  read-back len = {len(data)} (expect 16)")
    var ok = len(data) == 16
    if ok:
        for i in range(16):
            var want: Int = 0xBB if (i >= 4 and i < 8) else 0xAA
            if Int(data[i]) != want:
                ok = False
                print(t"  byte {i} = {Int(data[i])} want {want}")
    print(t"  rw-overwrite preserves length + patches in place: {ok}")


def probe_chunked_copy() raises:
    var src = String("prototypes/probe_src.bin")
    var dst = String("prototypes/probe_dst.bin")
    var n = 1000
    with open(src, "w") as f:
        var buf = List[UInt8](capacity=n)
        for i in range(n):
            buf.append(UInt8(i % 251))
        f.write_all(Span(buf))

    var chunk = 256
    with open(src, "r") as fin:
        with open(dst, "w") as fout:
            while True:
                var part = fin.read_bytes(chunk)
                if len(part) == 0:
                    break
                fout.write_all(Span(part))

    var a: List[UInt8]
    var b: List[UInt8]
    with open(src, "r") as f:
        a = f.read_bytes()
    with open(dst, "r") as f:
        b = f.read_bytes()
    var ok = len(a) == len(b) and len(a) == n
    if ok:
        for i in range(n):
            if a[i] != b[i]:
                ok = False
                break
    print(t"  chunked copy identical ({len(a)} bytes): {ok}")


def probe_memcpy_gather() raises:
    # Simulate a 2-rank column-sharded gather into a row-major buffer.
    # global: 3 rows x 4 cols (bytes), each rank holds 2 cols (local_cols=2).
    var global_rows = 3
    var global_cols = 4
    var data_cols = 2
    var eb = 1
    var rank0 = List[UInt8](length=global_rows * data_cols, fill=0)
    var rank1 = List[UInt8](length=global_rows * data_cols, fill=0)
    for r in range(global_rows):
        for c in range(data_cols):
            rank0[r * data_cols + c] = UInt8(r * 10 + c)
            rank1[r * data_cols + c] = UInt8(100 + r * 10 + c)

    var out = List[UInt8](length=global_rows * global_cols, fill=0)
    var op = out.unsafe_ptr()
    for rank in range(2):
        var col_start = rank * data_cols
        var sp = rank0.unsafe_ptr() if rank == 0 else rank1.unsafe_ptr()
        for row in range(global_rows):
            memcpy(
                dest=op + (row * global_cols + col_start) * eb,
                src=sp + row * data_cols * eb,
                count=data_cols * eb,
            )
    print("  gathered row-major:")
    for r in range(global_rows):
        var line = String("   ")
        for c in range(global_cols):
            line += String(t" {Int(out[r * global_cols + c])}")
        print(line)


def main() raises:
    print("probe: rw overwrite")
    probe_rw_overwrite()
    print("probe: chunked copy")
    probe_chunked_copy()
    print("probe: memcpy column gather")
    probe_memcpy_gather()
