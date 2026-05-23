from std.collections import Dict
from std.math import max, min
from std.memory import Span, UnsafePointer, alloc
from std.pathlib import Path
from std.reflection import reflect
from std.sys.info import simd_width_of, size_of
from std.os import abort

from linux.io_uring import IoRing, ReadOp, WriteOp, ReadMode, WriteMode
import linux.sys as linux

from safetensors.parser import (
    SafetensorsHeader, TensorMeta, parse_safetensors_header,
    dtype_tag, dtype_byte_size,
)
from modeling.loader import discover_shards
from modeling.slot import SlotLike, SlotGroup

from butterquant.kernels import (
    apply_gamma_in_place, gamma_sqrt_abs_in_place,
    rotate_and_quant, router_center,
    colsum_per_row, colsum_per_block,
)

from quant.recipe import (
    QuantRecipe, Passthrough, PerRowQuant, PerBlockQuant, RouterCenter,
    NoGamma, SplitGamma, AbsorbedGamma,
    SingleSided, TwoSided,
    NoColsum, PerRowCs, PerBlockCs,
)


comptime PtrU8 = UnsafePointer[UInt8, MutAnyOrigin]
comptime PtrF32 = UnsafePointer[Float32, MutAnyOrigin]
comptime PtrBF16 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime PtrI8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime W = simd_width_of[DType.float32]()

comptime PANEL_ROWS = 2048
comptime COPY_CHUNK = 16 * 1024 * 1024
comptime QD = 64

comptime KIND_PASSTHROUGH = 0
comptime KIND_PER_ROW = 1
comptime KIND_PER_BLOCK = 2
comptime KIND_ROUTER_CENTER = 3

comptime CS_NONE = 0
comptime CS_PER_ROW = 1
comptime CS_PER_BLOCK = 2


@fieldwise_init
struct OutputEntry(Copyable, Movable):
    var name: String
    var dtype: DType
    var shape0: Int
    var shape1: Int
    var data_start: Int
    var data_end: Int


def build_header(ref entries: List[OutputEntry]) -> List[UInt8]:
    var json = String("{")
    for i in range(len(entries)):
        if i > 0:
            json += ","
        ref e = entries[i]
        json += '"' + e.name + '":{"dtype":"' + String(dtype_tag(e.dtype))
        json += '","shape":[' + String(e.shape0)
        if e.shape1 > 0:
            json += "," + String(e.shape1)
        json += '],"data_offsets":[' + String(e.data_start)
        json += "," + String(e.data_end) + "]}"
    json += "}"

    while json.byte_length() % 8 != 0:
        json += " "

    var json_bytes = json.as_bytes()
    var json_len = len(json_bytes)
    var buf = List[UInt8](capacity=8 + json_len)
    var header_len = UInt64(json_len)
    for i in range(8):
        buf.append(UInt8((header_len >> UInt64(i * 8)) & 0xFF))
    for i in range(json_len):
        buf.append(json_bytes[i])
    return buf^


@fieldwise_init
struct LocatedTensor(Copyable):
    var shard: Int
    var data_start: Int
    var byte_size: Int
    var dtype: DType
    var rows: Int
    var cols: Int


def fold_shape(ref shape: List[Int]) -> Tuple[Int, Int]:
    if len(shape) == 0:
        return (1, 1)
    if len(shape) == 1:
        return (shape[0], 1)
    var rows = 1
    for i in range(len(shape) - 1):
        rows *= shape[i]
    return (rows, shape[len(shape) - 1])


def find_tensor(
    name: String, ref headers: List[SafetensorsHeader],
) -> Optional[LocatedTensor]:
    for i in range(len(headers)):
        ref h = headers[i]
        var meta_opt = h.tensors.get(name)
        if meta_opt:
            ref m = meta_opt.value()
            var rc = fold_shape(m.shape)
            return LocatedTensor(
                shard=i,
                data_start=h.data_offset + m.start,
                byte_size=m.byte_size(),
                dtype=m.dtype,
                rows=rc[0],
                cols=rc[1],
            )
    return None


@fieldwise_init
struct SlotMeta(Copyable, Movable):
    """One record per named slot, recipe-flattened into runtime fields.
    Produced by the comptime plan walk, consumed by the runtime executor."""
    var name: String
    var shard: Int
    var src_offset: Int
    var src_dtype: DType
    var rows: Int
    var cols: Int
    var kind: Int
    var fwht_block: Int
    var two_sided_m: Int
    var gamma_name: String
    var gamma_absorbed: Bool
    var colsum_kind: Int
    var bias_name: String
    var weight_off: Int
    var scale_off: Int
    var cs_off: Int
    var gauge_off: Int
    var bias_off: Int


@always_inline
def decode_bf16(src: PtrU8, dst: PtrF32, count: Int):
    var p = src.bitcast[Scalar[DType.bfloat16]]()
    var k = 0
    while k + W <= count:
        (dst + k).store((p + k).load[width=W]().cast[DType.float32]())
        k += W
    while k < count:
        dst[k] = p[k].cast[DType.float32]()
        k += 1


@always_inline
def decode_f32(src: PtrU8, dst: PtrF32, count: Int):
    var p = src.bitcast[Float32]()
    var k = 0
    while k + W <= count:
        (dst + k).store((p + k).load[width=W]())
        k += W
    while k < count:
        dst[k] = p[k]
        k += 1


@always_inline
def decode_f16(src: PtrU8, dst: PtrF32, count: Int):
    var p = src.bitcast[Scalar[DType.float16]]()
    var k = 0
    while k + W <= count:
        (dst + k).store((p + k).load[width=W]().cast[DType.float32]())
        k += W
    while k < count:
        dst[k] = p[k].cast[DType.float32]()
        k += 1


def decode_to_f32(dt: DType, src: PtrU8, dst: PtrF32, count: Int) -> Bool:
    if dt == DType.bfloat16:
        decode_bf16(src, dst, count)
        return True
    if dt == DType.float32:
        decode_f32(src, dst, count)
        return True
    if dt == DType.float16:
        decode_f16(src, dst, count)
        return True
    return False


def read_sync(
    mut ring: IoRing[QD], fd_idx: Int, offset: Int, length: Int, dest: PtrU8,
) -> Bool:
    try:
        _ = ring.submit_one(ReadOp(
            file_idx=fd_idx, offset=offset, length=length, dest=dest, id=0))
        var c = ring.drain_one()
        if Int(c.result) != length:
            var got = Int(c.result)
            print(t"quant: short read at {offset}: got {got}/{length}")
            return False
        return True
    except err:
        print(t"quant: read failed: {err}")
        return False


def write_sync(
    mut ring: IoRing[QD], fd_idx: Int, offset: Int, length: Int, src: PtrU8,
) -> Bool:
    try:
        _ = ring.submit_one(WriteOp(
            file_idx=fd_idx, offset=offset, length=length, src=src, id=0))
        var c = ring.drain_one()
        if Int(c.result) != length:
            var got = Int(c.result)
            print(t"quant: short write at {offset}: got {got}/{length}")
            return False
        return True
    except err:
        print(t"quant: write failed: {err}")
        return False


struct Quantizer(Movable):
    var ring: IoRing[QD]
    var headers: List[SafetensorsHeader]
    var shard_paths: List[Path]
    var output_path: Path
    var fds: List[Int32]
    var output_fd_idx: Int
    var data_start: Int
    var data_cursor: Int
    var entries: List[OutputEntry]
    var slots: List[SlotMeta]
    var gamma_cache: Dict[String, PtrF32]
    var ready: Bool

    def __init__(out self, source_dir: Path, output_path: Path):
        self.ring = IoRing[QD]()
        self.headers = List[SafetensorsHeader]()
        self.shard_paths = List[Path]()
        self.output_path = output_path
        self.fds = List[Int32]()
        self.output_fd_idx = -1
        self.data_start = -1
        self.data_cursor = 0
        self.entries = List[OutputEntry]()
        self.slots = List[SlotMeta]()
        self.gamma_cache = Dict[String, PtrF32]()
        self.ready = False

        if not self.ring:
            print("quant: io_uring setup failed")
            return

        self.shard_paths = discover_shards(source_dir)
        if len(self.shard_paths) == 0:
            print(t"quant: no shards in {source_dir}")
            return
        for i in range(len(self.shard_paths)):
            var h_opt = parse_safetensors_header(self.shard_paths[i])
            if not h_opt:
                ref p = self.shard_paths[i]
                print(t"quant: failed to parse header {p}")
                return
            self.headers.append(h_opt.take())

        var sys = linux.linux_sys()
        for i in range(len(self.shard_paths)):
            var path_str = String(self.shard_paths[i])
            var fd = sys.sys_openat(linux.AT_FDCWD, path_str,
                ReadMode.OPEN_FLAGS, ReadMode.CREATE_MODE)
            if fd < 0:
                print(t"quant: open failed for {path_str}: errno {fd}")
                return
            self.fds.append(Int32(fd))

        var out_path_str = String(output_path)
        var ofd = sys.sys_openat(linux.AT_FDCWD, out_path_str,
            WriteMode.OPEN_FLAGS, WriteMode.CREATE_MODE)
        if ofd < 0:
            print(t"quant: open failed for {out_path_str}: errno {ofd}")
            return
        self.fds.append(Int32(ofd))
        self.output_fd_idx = len(self.fds) - 1

        try:
            _ = self.ring.register_fds(Span(self.fds))
        except err:
            print(t"quant: register_fds failed: {err.error_message()}")
            return

        self.ready = True

    def __del__(deinit self):
        for entry in self.gamma_cache.items():
            entry.value.free()
        var sys = linux.linux_sys()
        for fd in self.fds:
            if fd >= 0:
                _ = sys.sys_close(Int(fd))

    def __bool__(self) -> Bool:
        return self.ready

    def plan_walk[T: AnyType](mut self, prefix: String) -> Bool:
        comptime for i in range(reflect[T].field_count()):
            comptime FT = reflect[T].field_types()[i]
            comptime if conforms_to(FT, SlotLike):
                comptime if FT.NAME != StaticString(""):
                    if not self.plan_slot[FT](prefix):
                        return False
            comptime if conforms_to(FT, SlotGroup):
                if not self.plan_walk[FT](prefix):
                    return False
        return True

    def plan_slot[FT: SlotLike](mut self, prefix: String) -> Bool:
        comptime ROWS = FT.SHAPE.GLOBAL_N
        comptime COLS = FT.SHAPE.GLOBAL_M
        comptime SRC = FT.ENCODING.DTYPE
        comptime QV = FT.QUANT

        var full = prefix + String(FT.NAME)
        var loc_opt = find_tensor(full, self.headers)
        if not loc_opt:
            print(t"quant plan: missing {full}")
            return False
        var loc = loc_opt.take()

        if loc.dtype != SRC:
            print(t"quant plan: dtype mismatch for {full}: got {loc.dtype} expected {SRC}")
            return False
        if loc.rows != ROWS or loc.cols != COLS:
            var lr = loc.rows
            var lc = loc.cols
            print(t"quant plan: shape mismatch for {full}: got {lr}x{lc} expected {ROWS}x{COLS}")
            return False

        var meta = SlotMeta(
            name=full, shard=loc.shard, src_offset=loc.data_start,
            src_dtype=SRC, rows=ROWS, cols=COLS,
            kind=KIND_PASSTHROUGH, fwht_block=0, two_sided_m=0,
            gamma_name=String(""), gamma_absorbed=False,
            colsum_kind=CS_NONE, bias_name=String(""),
            weight_off=0, scale_off=0, cs_off=0, gauge_off=0, bias_off=0,
        )

        comptime if QV.isa[Passthrough]():
            meta.kind = KIND_PASSTHROUGH
            var size = ROWS * COLS * size_of[Scalar[SRC]]()
            meta.weight_off = self.data_cursor
            self.add_entry(full, SRC, ROWS, COLS, size)

        comptime if QV.isa[PerRowQuant]():
            comptime QT = QV[PerRowQuant]
            meta.kind = KIND_PER_ROW
            meta.fwht_block = QT.fwht_block
            comptime if QT.rotation.isa[TwoSided]():
                meta.two_sided_m = QT.rotation[TwoSided].m_block
            comptime if QT.gamma.isa[SplitGamma]():
                meta.gamma_name = prefix + String(QT.gamma[SplitGamma].name)
                meta.gamma_absorbed = False
            comptime if QT.gamma.isa[AbsorbedGamma]():
                meta.gamma_name = prefix + String(QT.gamma[AbsorbedGamma].name)
                meta.gamma_absorbed = True
            comptime if QT.colsum.isa[PerRowCs]():
                meta.colsum_kind = CS_PER_ROW
            comptime if QT.colsum.isa[PerBlockCs]():
                meta.colsum_kind = CS_PER_BLOCK

            meta.weight_off = self.data_cursor
            self.add_entry(full, DType.int8, ROWS, COLS, ROWS * COLS)
            meta.scale_off = self.data_cursor
            self.add_entry(full + ".scale", DType.float32, ROWS, 0, ROWS * 4)
            if meta.colsum_kind == CS_PER_ROW:
                meta.cs_off = self.data_cursor
                self.add_entry(full + ".colsum", DType.float32, ROWS, 0, ROWS * 4)
            elif meta.colsum_kind == CS_PER_BLOCK:
                comptime NB = COLS // QT.fwht_block
                meta.cs_off = self.data_cursor
                self.add_entry(full + ".colsum", DType.float32, ROWS, NB, ROWS * NB * 4)

        comptime if QV.isa[PerBlockQuant]():
            comptime QT = QV[PerBlockQuant]
            comptime NB = COLS // QT.fwht_block
            meta.kind = KIND_PER_BLOCK
            meta.fwht_block = QT.fwht_block
            comptime if QT.rotation.isa[TwoSided]():
                meta.two_sided_m = QT.rotation[TwoSided].m_block
            comptime if QT.gamma.isa[SplitGamma]():
                meta.gamma_name = prefix + String(QT.gamma[SplitGamma].name)
                meta.gamma_absorbed = False
            comptime if QT.gamma.isa[AbsorbedGamma]():
                meta.gamma_name = prefix + String(QT.gamma[AbsorbedGamma].name)
                meta.gamma_absorbed = True
            comptime if QT.colsum.isa[PerBlockCs]():
                meta.colsum_kind = CS_PER_BLOCK

            meta.weight_off = self.data_cursor
            self.add_entry(full, DType.int8, ROWS, COLS, ROWS * COLS)
            meta.scale_off = self.data_cursor
            self.add_entry(full + ".scale", DType.float32, ROWS, NB, ROWS * NB * 4)
            if meta.colsum_kind == CS_PER_BLOCK:
                meta.cs_off = self.data_cursor
                self.add_entry(full + ".colsum", DType.float32, ROWS, NB, ROWS * NB * 4)

        comptime if QV.isa[RouterCenter]():
            comptime QT = QV[RouterCenter]
            meta.kind = KIND_ROUTER_CENTER

            meta.weight_off = self.data_cursor
            self.add_entry(full, DType.bfloat16, ROWS, COLS, ROWS * COLS * 2)
            meta.gauge_off = self.data_cursor
            self.add_entry(full + ".gauge", DType.bfloat16, COLS, 0, COLS * 2)

            comptime if QT.bias_name != StaticString(""):
                var bias_full = prefix + String(QT.bias_name)
                var bloc = find_tensor(bias_full, self.headers)
                if not bloc:
                    print(t"quant plan: missing router bias {bias_full}")
                    return False
                meta.bias_name = bias_full
                meta.bias_off = self.data_cursor
                self.add_entry(full + ".bias", DType.float32, ROWS, 0, ROWS * 4)

        self.slots.append(meta^)
        return True

    def add_entry(
        mut self, name: String, dt: DType, s0: Int, s1: Int, size: Int,
    ):
        self.entries.append(OutputEntry(
            name=name, dtype=dt, shape0=s0, shape1=s1,
            data_start=self.data_cursor, data_end=self.data_cursor + size,
        ))
        self.data_cursor += size

    def write_header(mut self) -> Bool:
        var header = build_header(self.entries)
        var header_size = len(header)
        var hp = PtrU8(unsafe_from_address=Int(header.unsafe_ptr()))
        if not write_sync(self.ring, self.output_fd_idx, 0, header_size, hp):
            return False
        self.data_start = header_size
        _ = header^
        return True

    def execute(mut self) -> Bool:
        for i in range(len(self.slots)):
            var meta = self.slots[i].copy()
            if not self.execute_slot(meta):
                ref n = meta.name
                print(t"quant: failed at {n}")
                return False
        return True

    def execute_slot(mut self, ref meta: SlotMeta) -> Bool:
        if meta.kind == KIND_PASSTHROUGH:
            return self.do_passthrough(meta)
        if meta.kind == KIND_PER_ROW:
            return self.do_per_row(meta)
        if meta.kind == KIND_PER_BLOCK:
            return self.do_per_block(meta)
        if meta.kind == KIND_ROUTER_CENTER:
            return self.do_router_center(meta)
        return False

    def get_gamma(
        mut self, gamma_name: String, expected_cols: Int, absorbed: Bool,
    ) -> Optional[PtrF32]:
        var key = gamma_name + (":a" if absorbed else ":s")
        var existing = self.gamma_cache.get(key)
        if existing:
            return existing.value()

        var loc_opt = find_tensor(gamma_name, self.headers)
        if not loc_opt:
            print(t"quant: missing gamma {gamma_name}")
            return None
        var loc = loc_opt.take()
        var cols = loc.rows * loc.cols
        if cols != expected_cols:
            print(t"quant: gamma {gamma_name} cols {cols} != expected {expected_cols}")
            return None

        var raw = alloc[UInt8](loc.byte_size).as_any_origin()
        if not read_sync(self.ring, loc.shard, loc.data_start, loc.byte_size, raw):
            raw.free()
            return None

        var gamma = alloc[Float32](cols).as_any_origin()
        if not decode_to_f32(loc.dtype, raw, gamma, cols):
            print(t"quant: unsupported gamma dtype for {gamma_name}: {loc.dtype}")
            raw.free()
            gamma.free()
            return None
        raw.free()

        if not absorbed:
            gamma_sqrt_abs_in_place(gamma, cols)

        self.gamma_cache[key^] = gamma
        return gamma

    def do_passthrough(mut self, ref meta: SlotMeta) -> Bool:
        var total = meta.rows * meta.cols * dtype_byte_size(meta.src_dtype)
        var chunk_cap = min(COPY_CHUNK, total)
        var buf = alloc[UInt8](chunk_cap).as_any_origin()
        var copied = 0
        while copied < total:
            var n = min(COPY_CHUNK, total - copied)
            if not read_sync(self.ring, meta.shard, meta.src_offset + copied, n, buf):
                buf.free()
                return False
            if not write_sync(self.ring, self.output_fd_idx,
                    self.data_start + meta.weight_off + copied, n, buf):
                buf.free()
                return False
            copied += n
        buf.free()
        return True

    def do_per_row(mut self, ref meta: SlotMeta) -> Bool:
        return self.do_quant(meta, per_block=False)

    def do_per_block(mut self, ref meta: SlotMeta) -> Bool:
        return self.do_quant(meta, per_block=True)

    def do_quant(mut self, ref meta: SlotMeta, per_block: Bool) -> Bool:
        var rows = meta.rows
        var cols = meta.cols
        var fwht_block = meta.fwht_block
        var src_bytes_per = dtype_byte_size(meta.src_dtype)
        if src_bytes_per <= 0:
            print(t"quant: bad src dtype {meta.src_dtype}")
            return False
        if meta.src_dtype != DType.bfloat16 and meta.src_dtype != DType.float32 and meta.src_dtype != DType.float16:
            print(t"quant: unsupported source dtype for quant: {meta.src_dtype}")
            return False

        var gamma = PtrF32.unsafe_dangling()
        var has_gamma = False
        if meta.gamma_name.byte_length() > 0:
            var g = self.get_gamma(meta.gamma_name, cols, meta.gamma_absorbed)
            if not g:
                return False
            gamma = g.value()
            has_gamma = True

        var nb = cols // fwht_block if per_block else 1
        var scale_per_row = nb if per_block else 1
        var cs_per_row: Int
        if meta.colsum_kind == CS_PER_ROW:
            cs_per_row = 1
        elif meta.colsum_kind == CS_PER_BLOCK:
            cs_per_row = cols // fwht_block
        else:
            cs_per_row = 0

        var pr = min(PANEL_ROWS, rows)
        var src_buf = alloc[UInt8](pr * cols * src_bytes_per).as_any_origin()
        var work = alloc[Float32](pr * cols).as_any_origin()
        var qi = alloc[Scalar[DType.int8]](pr * cols).as_any_origin()
        var scales = alloc[Float32](pr * scale_per_row).as_any_origin()
        var cs = alloc[Float32](max(1, pr * cs_per_row)).as_any_origin()

        var row_off = 0
        var ok = True
        while row_off < rows and ok:
            var panel_rows = min(pr, rows - row_off)
            var panel_bytes = panel_rows * cols * src_bytes_per
            var src_off = meta.src_offset + row_off * cols * src_bytes_per
            if not read_sync(self.ring, meta.shard, src_off, panel_bytes, src_buf):
                ok = False
                break

            if not decode_to_f32(meta.src_dtype, src_buf, work, panel_rows * cols):
                ok = False
                break

            if has_gamma:
                for r in range(panel_rows):
                    apply_gamma_in_place(work + r * cols, gamma, cols)

            if per_block:
                rotate_and_quant[True](
                    fwht_block, work, qi, scales, panel_rows, cols, meta.two_sided_m)
            else:
                rotate_and_quant[False](
                    fwht_block, work, qi, scales, panel_rows, cols, meta.two_sided_m)

            if meta.colsum_kind == CS_PER_ROW:
                colsum_per_row(qi, cs, panel_rows, cols)
            elif meta.colsum_kind == CS_PER_BLOCK:
                colsum_per_block(qi, cs, fwht_block, panel_rows, cols)

            var w_off = self.data_start + meta.weight_off + row_off * cols
            if not write_sync(self.ring, self.output_fd_idx, w_off,
                    panel_rows * cols, qi.bitcast[UInt8]()):
                ok = False
                break
            var s_off = self.data_start + meta.scale_off + row_off * scale_per_row * 4
            if not write_sync(self.ring, self.output_fd_idx, s_off,
                    panel_rows * scale_per_row * 4, scales.bitcast[UInt8]()):
                ok = False
                break
            if cs_per_row > 0:
                var c_off = self.data_start + meta.cs_off + row_off * cs_per_row * 4
                if not write_sync(self.ring, self.output_fd_idx, c_off,
                        panel_rows * cs_per_row * 4, cs.bitcast[UInt8]()):
                    ok = False
                    break

            row_off += panel_rows

        src_buf.free()
        work.free()
        qi.free()
        scales.free()
        cs.free()
        return ok

    def do_router_center(mut self, ref meta: SlotMeta) -> Bool:
        var rows = meta.rows
        var cols = meta.cols
        if meta.src_dtype != DType.bfloat16 and meta.src_dtype != DType.float32:
            print(t"quant: router_center only supports bf16/f32 source; got {meta.src_dtype}")
            return False
        var src_bytes_per = dtype_byte_size(meta.src_dtype)

        var src_buf = alloc[UInt8](rows * cols * src_bytes_per).as_any_origin()
        var gauge_f32 = alloc[Float32](cols).as_any_origin()
        var centered = alloc[Scalar[DType.bfloat16]](rows * cols).as_any_origin()
        var gauge_bf16 = alloc[Scalar[DType.bfloat16]](cols).as_any_origin()

        var ok = True
        if not read_sync(self.ring, meta.shard, meta.src_offset,
                rows * cols * src_bytes_per, src_buf):
            ok = False

        if ok:
            if meta.src_dtype == DType.bfloat16:
                router_center[DType.bfloat16](
                    src_buf.bitcast[Scalar[DType.bfloat16]](),
                    gauge_f32, centered, gauge_bf16, rows, cols)
            else:
                router_center[DType.float32](
                    src_buf.bitcast[Float32](),
                    gauge_f32, centered, gauge_bf16, rows, cols)

            if not write_sync(self.ring, self.output_fd_idx,
                    self.data_start + meta.weight_off, rows * cols * 2,
                    centered.bitcast[UInt8]()):
                ok = False

        if ok and not write_sync(self.ring, self.output_fd_idx,
                self.data_start + meta.gauge_off, cols * 2,
                gauge_bf16.bitcast[UInt8]()):
            ok = False

        if ok and meta.bias_name.byte_length() > 0:
            ok = self.write_router_bias(meta)

        src_buf.free()
        gauge_f32.free()
        centered.free()
        gauge_bf16.free()
        return ok

    def write_router_bias(mut self, ref meta: SlotMeta) -> Bool:
        var loc_opt = find_tensor(meta.bias_name, self.headers)
        if not loc_opt:
            print(t"quant: missing router bias {meta.bias_name}")
            return False
        var loc = loc_opt.take()
        var rows = meta.rows
        if loc.rows * loc.cols != rows:
            print(t"quant: router bias {meta.bias_name} size mismatch")
            return False
        var raw = alloc[UInt8](loc.byte_size).as_any_origin()
        if not read_sync(self.ring, loc.shard, loc.data_start, loc.byte_size, raw):
            raw.free()
            return False
        var f32 = alloc[Float32](rows).as_any_origin()
        if not decode_to_f32(loc.dtype, raw, f32, rows):
            print(t"quant: unsupported router bias dtype for {meta.bias_name}: {loc.dtype}")
            raw.free()
            f32.free()
            return False
        raw.free()
        var ok = write_sync(self.ring, self.output_fd_idx,
            self.data_start + meta.bias_off, rows * 4,
            f32.bitcast[UInt8]())
        f32.free()
        return ok
