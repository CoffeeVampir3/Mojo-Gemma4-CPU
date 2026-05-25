from std.collections import Dict, InlineArray
from std.math import max, min
from std.memory import Span, UnsafePointer, alloc
from std.pathlib import Path
from std.reflection import reflect
from std.sys.info import simd_width_of

from linux.io_uring import IoRing, ReadOp, WriteOp, ReadMode, WriteMode
import linux.sys as linux
from numa import NumaTopology
from threading.threading_traits import BurstKernel, BurstThreadPool

from safetensors.parser import (
    SafetensorsHeader, parse_safetensors_header,
    dtype_tag, dtype_byte_size,
)
from modeling.loader import discover_shards
from modeling.slot import SlotLike, SlotGroup

from butterquant.kernels import (
    apply_gamma_in_place, gamma_sqrt_abs_in_place,
    rotate_and_quant, router_center,
    colsum_per_row, colsum_per_block,
)
from butterquant.constants import is_supported_fwht_block

from quant.recipe import (
    QuantRecipe, Passthrough, PerRowQuant, PerBlockQuant, RouterCenter,
    NoGamma, SplitGamma, AbsorbedGamma,
    SingleSided, TwoSided,
    NoColsum, PerRowCs, PerBlockCs,
)
from quant.plan import (
    SlotIdentity, GammaRef, PassthroughPlan, QuantPlan, RouterPlan, SlotPlan,
    ColsumKind, ScratchCapacity,
    SCALE_SUFFIX, COLSUM_SUFFIX, GAUGE_SUFFIX, BIAS_SUFFIX,
)


comptime PtrU8 = UnsafePointer[UInt8, MutAnyOrigin]
comptime PtrF32 = UnsafePointer[Float32, MutAnyOrigin]
comptime PtrBF16 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime PtrI8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime W = simd_width_of[DType.float32]()

comptime PANEL_ROWS = 2048
comptime COPY_CHUNK = 16 * 1024 * 1024
comptime QD = 64


@fieldwise_init
struct OutputEntry(Copyable, Movable):
    var name: String
    var dtype: DType
    var shape0: Int
    var shape1: Int
    var data_start: Int
    var data_end: Int


@always_inline
def append_static(mut buf: List[UInt8], s: StaticString):
    var bytes = s.as_bytes()
    for i in range(len(bytes)):
        buf.append(bytes[i])


@always_inline
def append_string(mut buf: List[UInt8], s: String):
    var bytes = s.as_bytes()
    for i in range(len(bytes)):
        buf.append(bytes[i])


@always_inline
def append_int(mut buf: List[UInt8], v: Int):
    var s = String(v)
    var bytes = s.as_bytes()
    for i in range(len(bytes)):
        buf.append(bytes[i])


def build_header(ref entries: List[OutputEntry]) -> List[UInt8]:
    """Emit the safetensors header preceded by its 8-byte little-endian length.
    The header bytes start at offset 8 of the returned buffer; the leading 8
    bytes are patched in after the JSON length is known."""
    var buf = List[UInt8](capacity=128 * 1024)
    for _ in range(8):
        buf.append(0)
    comptime JSON_START = 8

    buf.append(0x7B)  # '{'
    for i in range(len(entries)):
        if i > 0:
            buf.append(0x2C)  # ','
        ref e = entries[i]
        buf.append(0x22)  # '"'
        append_string(buf, e.name)
        append_static(buf, '":{"dtype":"')
        append_static(buf, dtype_tag(e.dtype))
        append_static(buf, '","shape":[')
        append_int(buf, e.shape0)
        if e.shape1 > 0:
            buf.append(0x2C)
            append_int(buf, e.shape1)
        append_static(buf, '],"data_offsets":[')
        append_int(buf, e.data_start)
        buf.append(0x2C)
        append_int(buf, e.data_end)
        append_static(buf, "]}")
    buf.append(0x7D)  # '}'

    while (len(buf) - JSON_START) % 8 != 0:
        buf.append(0x20)  # ' '

    var json_len = UInt64(len(buf) - JSON_START)
    for i in range(8):
        buf[i] = UInt8((json_len >> UInt64(i * 8)) & 0xFF)
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
    name: String, headers: Span[SafetensorsHeader, MutAnyOrigin],
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


@always_inline
def supports_decode_to_f32(dt: DType) -> Bool:
    return (
        dt == DType.bfloat16
        or dt == DType.float32
        or dt == DType.float16
    )


@always_inline
def supports_router_source(dt: DType) -> Bool:
    return dt == DType.bfloat16 or dt == DType.float32


def validate_gamma_tensor(
    name: String,
    headers: Span[SafetensorsHeader, MutAnyOrigin],
    expected_cols: Int,
) -> Bool:
    var loc_opt = find_tensor(name, headers)
    if not loc_opt:
        print(t"quant plan: missing gamma {name}")
        return False
    var loc = loc_opt.take()
    var cols = loc.rows * loc.cols
    if cols != expected_cols:
        print(t"quant plan: gamma {name} cols {cols} != expected {expected_cols}")
        return False
    if not supports_decode_to_f32(loc.dtype):
        print(t"quant plan: unsupported gamma dtype for {name}: {loc.dtype}")
        return False
    return True


def validate_router_bias_tensor(
    name: String,
    headers: Span[SafetensorsHeader, MutAnyOrigin],
    expected_rows: Int,
) -> Bool:
    var loc_opt = find_tensor(name, headers)
    if not loc_opt:
        print(t"quant plan: missing router bias {name}")
        return False
    var loc = loc_opt.take()
    var rows = loc.rows * loc.cols
    if rows != expected_rows:
        print(t"quant plan: router bias {name} rows {rows} != expected {expected_rows}")
        return False
    if not supports_decode_to_f32(loc.dtype):
        print(t"quant plan: unsupported router bias dtype for {name}: {loc.dtype}")
        return False
    return True


@always_inline
def headers_span(
    ref headers: List[SafetensorsHeader],
) -> Span[SafetensorsHeader, MutAnyOrigin]:
    """Strip origin tracking from `headers` for storage in worker structs.
    Caller guarantees `headers` outlives the returned span."""
    return Span[SafetensorsHeader, MutAnyOrigin](
        ptr=UnsafePointer[SafetensorsHeader, MutAnyOrigin](
            unsafe_from_address=Int(headers.unsafe_ptr())),
        length=len(headers),
    )


@always_inline
def fds_span(
    ref fds: List[Int32],
) -> Span[Int32, MutAnyOrigin]:
    return Span[Int32, MutAnyOrigin](
        ptr=UnsafePointer[Int32, MutAnyOrigin](
            unsafe_from_address=Int(fds.unsafe_ptr())),
        length=len(fds),
    )


@always_inline
def slots_span(
    ref slots: List[SlotPlan],
) -> Span[SlotPlan, MutAnyOrigin]:
    return Span[SlotPlan, MutAnyOrigin](
        ptr=UnsafePointer[SlotPlan, MutAnyOrigin](
            unsafe_from_address=Int(slots.unsafe_ptr())),
        length=len(slots),
    )


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
        print(t"quant: read failed: {err.error_message()}")
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
        print(t"quant: write failed: {err.error_message()}")
        return False


def write_sync_many(mut ring: IoRing[QD], ops: Span[WriteOp[], MutAnyOrigin]) -> Bool:
    """Submit `ops` in one or more batches and drain all completions.
    Assigns each op's `id` to its index in the span so completion validation
    is O(1). Caller guarantees the span lives until this returns."""
    var n = len(ops)
    if n == 0:
        return True
    for i in range(n):
        ops[i].id = i
    try:
        var submitted = 0
        while submitted < n:
            var got = ring.submit_many[WriteOp[]](ops, submitted)
            if got == 0:
                var c = ring.drain_one()
                if Int(c.result) != ops[c.id].length:
                    var got_b = Int(c.result)
                    var expected = ops[c.id].length
                    print(t"quant: short write op {c.id}: {got_b}/{expected}")
                    return False
            submitted += got
        var to_drain = n
        while to_drain > 0:
            var c = ring.drain_one()
            if Int(c.result) != ops[c.id].length:
                var got_b = Int(c.result)
                var expected = ops[c.id].length
                print(t"quant: short write op {c.id}: {got_b}/{expected}")
                return False
            to_drain -= 1
    except err:
        print(t"quant: batched write failed: {err.error_message()}")
        return False
    return True


struct QuantScratch(Movable):
    """Reusable per-worker staging buffers sized to the worst-case slot.
    Owned and freed by the QuantScratch lifetime; never reallocated mid-run."""
    var src_buf: PtrU8
    var work: PtrF32
    var qi: PtrI8
    var scales: PtrF32
    var cs: PtrF32
    var ready: Bool

    def __init__(out self, cap: ScratchCapacity):
        self.ready = False
        self.src_buf = PtrU8.unsafe_dangling()
        self.work = PtrF32.unsafe_dangling()
        self.qi = PtrI8.unsafe_dangling()
        self.scales = PtrF32.unsafe_dangling()
        self.cs = PtrF32.unsafe_dangling()
        var pr = cap.max_panel_rows
        var cols = cap.max_cols
        if pr <= 0 or cols <= 0:
            return
        var src_bytes_per = cap.max_src_bytes_per if cap.max_src_bytes_per > 0 else 1
        var scale_per = cap.max_scale_per_row if cap.max_scale_per_row > 0 else 1
        var cs_per = cap.max_cs_per_row if cap.max_cs_per_row > 0 else 1
        self.src_buf = alloc[UInt8](pr * cols * src_bytes_per).as_any_origin()
        self.work = alloc[Float32](pr * cols).as_any_origin()
        self.qi = alloc[Scalar[DType.int8]](pr * cols).as_any_origin()
        self.scales = alloc[Float32](pr * scale_per).as_any_origin()
        self.cs = alloc[Float32](pr * cs_per).as_any_origin()
        self.ready = True

    def __del__(deinit self):
        if not self.ready:
            return
        self.src_buf.free()
        self.work.free()
        self.qi.free()
        self.scales.free()
        self.cs.free()

    def __bool__(self) -> Bool:
        return self.ready


struct Quantizer(Movable):
    """Coordinator: discovers shards, parses headers, opens fds, walks slot
    plans, writes the safetensors header, then dispatches the per-slot work
    to one or more QuantWorker instances. The coordinator's ring is used only
    for the header write; workers each set up their own ring."""

    var ring: IoRing[QD]
    var headers: List[SafetensorsHeader]
    var shard_paths: List[Path]
    var output_path: Path
    var fds: List[Int32]
    var output_fd_idx: Int
    var data_start: Int
    var data_cursor: Int
    var entries: List[OutputEntry]
    var slots: List[SlotPlan]
    var scratch_cap: ScratchCapacity
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
        self.slots = List[SlotPlan]()
        self.scratch_cap = ScratchCapacity.zero(PANEL_ROWS)
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
        var sys = linux.linux_sys()
        for fd in self.fds:
            if fd >= 0:
                _ = sys.sys_close(Int(fd))

    def __bool__(self) -> Bool:
        return self.ready

    def plan_walk[T: AnyType](
        mut self, prefix: String, layer_idx: Int = -1,
    ) -> Bool:
        """Walk `T` reflectively, emitting plans for every named SlotLike
        field and recursing into SlotGroup fields. `layer_idx` is stamped
        onto each emitted plan and propagates through recursion — set it to
        the layer index for per-layer plan walks and leave it -1 for
        layer-agnostic walks (tail tensors)."""
        comptime for i in range(reflect[T].field_count()):
            comptime FT = reflect[T].field_types()[i]
            comptime if conforms_to(FT, SlotLike):
                comptime if FT.NAME != StaticString(""):
                    if not self.plan_slot[FT](prefix, layer_idx):
                        return False
            comptime if conforms_to(FT, SlotGroup):
                if not self.plan_walk[FT](prefix, layer_idx):
                    return False
        return True

    def plan_slot[FT: SlotLike](mut self, prefix: String, layer_idx: Int) -> Bool:
        comptime ROWS = FT.SHAPE.GLOBAL_N
        comptime COLS = FT.SHAPE.GLOBAL_M
        comptime SRC = FT.ENCODING.DTYPE
        comptime QV = FT.QUANT

        var full = prefix + String(FT.NAME)
        var loc_opt = find_tensor(full, headers_span(self.headers))
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

        var local = String(FT.NAME)

        comptime if QV.isa[Passthrough]():
            self.plan_passthrough(full, local, layer_idx, loc)

        comptime if QV.isa[PerRowQuant]():
            comptime QT = QV[PerRowQuant]
            var tsm = 0
            comptime if QT.rotation.isa[TwoSided]():
                tsm = QT.rotation[TwoSided].m_block
            var gamma = GammaRef.none()
            comptime if QT.gamma.isa[SplitGamma]():
                gamma = GammaRef(prefix + String(QT.gamma[SplitGamma].name), False)
            comptime if QT.gamma.isa[AbsorbedGamma]():
                gamma = GammaRef(prefix + String(QT.gamma[AbsorbedGamma].name), True)
            var ck = ColsumKind.NONE
            comptime if QT.colsum.isa[PerRowCs]():
                ck = ColsumKind.PER_ROW
            comptime if QT.colsum.isa[PerBlockCs]():
                ck = ColsumKind.PER_BLOCK
            if not self.plan_quant(full, local, layer_idx, loc,
                per_block=False, fwht=QT.fwht_block,
                two_sided_m=tsm, gamma=gamma, colsum_kind=ck):
                return False

        comptime if QV.isa[PerBlockQuant]():
            comptime QT = QV[PerBlockQuant]
            var tsm = 0
            comptime if QT.rotation.isa[TwoSided]():
                tsm = QT.rotation[TwoSided].m_block
            var gamma = GammaRef.none()
            comptime if QT.gamma.isa[SplitGamma]():
                gamma = GammaRef(prefix + String(QT.gamma[SplitGamma].name), False)
            comptime if QT.gamma.isa[AbsorbedGamma]():
                gamma = GammaRef(prefix + String(QT.gamma[AbsorbedGamma].name), True)
            var ck = ColsumKind.NONE
            comptime if QT.colsum.isa[PerBlockCs]():
                ck = ColsumKind.PER_BLOCK
            if not self.plan_quant(full, local, layer_idx, loc,
                per_block=True, fwht=QT.fwht_block,
                two_sided_m=tsm, gamma=gamma, colsum_kind=ck):
                return False

        comptime if QV.isa[RouterCenter]():
            comptime QT = QV[RouterCenter]
            var bias_name = String("")
            comptime if QT.bias_name != StaticString(""):
                bias_name = prefix + String(QT.bias_name)
            if not self.plan_router(full, local, layer_idx, loc, bias_name):
                return False

        return True

    def plan_passthrough(
        mut self, name: String, local: String, layer_idx: Int,
        loc: LocatedTensor,
    ):
        var bytes = loc.rows * loc.cols * size_of_dtype(loc.dtype)
        var weight_off = self.add_entry(name, loc.dtype, loc.rows, loc.cols, bytes)
        var id = SlotIdentity(
            name=name, local_name=local, layer_idx=layer_idx,
            shard=loc.shard, src_offset=loc.data_start,
            src_dtype=loc.dtype, rows=loc.rows, cols=loc.cols,
            weight_off=weight_off,
        )
        self.slots.append(PassthroughPlan(id^, bytes))

    def plan_quant(
        mut self, name: String, local: String, layer_idx: Int,
        loc: LocatedTensor, per_block: Bool,
        fwht: Int, two_sided_m: Int, gamma: GammaRef, colsum_kind: Int,
    ) -> Bool:
        if loc.rows <= 0 or loc.cols <= 0:
            print(t"quant plan: invalid quant shape for {name}: {loc.rows}x{loc.cols}")
            return False
        if not supports_decode_to_f32(loc.dtype):
            print(t"quant plan: unsupported source dtype for quant {name}: {loc.dtype}")
            return False
        if not is_supported_fwht_block(fwht):
            print(t"quant plan: unsupported K-axis FWHT block for {name}: {fwht}")
            return False
        if loc.cols % fwht != 0:
            print(t"quant plan: cols {loc.cols} not divisible by K-axis FWHT block {fwht} for {name}")
            return False
        if two_sided_m != 0:
            if not is_supported_fwht_block(two_sided_m):
                print(t"quant plan: unsupported M-axis FWHT block for {name}: {two_sided_m}")
                return False
            if loc.rows % two_sided_m != 0:
                print(t"quant plan: rows {loc.rows} not divisible by M-axis FWHT block {two_sided_m} for {name}")
                return False
        if (colsum_kind != ColsumKind.NONE
                and colsum_kind != ColsumKind.PER_ROW
                and colsum_kind != ColsumKind.PER_BLOCK):
            print(t"quant plan: unsupported colsum kind {colsum_kind} for {name}")
            return False
        if gamma.is_present():
            if not validate_gamma_tensor(gamma.name, headers_span(self.headers), loc.cols):
                return False

        var nb = loc.cols // fwht if per_block else 1
        var weight_off = self.add_entry(
            name, DType.int8, loc.rows, loc.cols, loc.rows * loc.cols)
        var scale_off: Int
        if per_block:
            scale_off = self.add_entry(
                name + SCALE_SUFFIX, DType.float32, loc.rows, nb,
                loc.rows * nb * 4)
        else:
            scale_off = self.add_entry(
                name + SCALE_SUFFIX, DType.float32, loc.rows, 0,
                loc.rows * 4)
        var cs_off = -1
        if colsum_kind == ColsumKind.PER_ROW:
            cs_off = self.add_entry(
                name + COLSUM_SUFFIX, DType.float32, loc.rows, 0,
                loc.rows * 4)
        elif colsum_kind == ColsumKind.PER_BLOCK:
            var cs_nb = loc.cols // fwht
            cs_off = self.add_entry(
                name + COLSUM_SUFFIX, DType.float32, loc.rows, cs_nb,
                loc.rows * cs_nb * 4)

        var id = SlotIdentity(
            name=name, local_name=local, layer_idx=layer_idx,
            shard=loc.shard, src_offset=loc.data_start,
            src_dtype=loc.dtype, rows=loc.rows, cols=loc.cols,
            weight_off=weight_off,
        )
        var plan = QuantPlan(
            id^, per_block, fwht, two_sided_m, gamma.copy(),
            colsum_kind, scale_off, cs_off,
        )
        self.scratch_cap.absorb_quant(plan, size_of_dtype(loc.dtype))
        self.slots.append(plan^)
        return True

    def plan_router(
        mut self, name: String, local: String, layer_idx: Int,
        loc: LocatedTensor, bias_name: String,
    ) -> Bool:
        if loc.rows <= 0 or loc.cols <= 0:
            print(t"quant plan: invalid router shape for {name}: {loc.rows}x{loc.cols}")
            return False
        if not supports_router_source(loc.dtype):
            print(t"quant plan: router_center only supports bf16/f32 source for {name}: {loc.dtype}")
            return False
        if loc.cols % W != 0:
            print(t"quant plan: router cols {loc.cols} not divisible by SIMD width {W} for {name}")
            return False
        if bias_name.byte_length() > 0:
            if not validate_router_bias_tensor(
                    bias_name, headers_span(self.headers), loc.rows):
                return False

        var weight_off = self.add_entry(
            name, DType.bfloat16, loc.rows, loc.cols, loc.rows * loc.cols * 2)
        var gauge_off = self.add_entry(
            name + GAUGE_SUFFIX, DType.bfloat16, loc.cols, 0, loc.cols * 2)
        var bias_off = 0
        if bias_name.byte_length() > 0:
            bias_off = self.add_entry(
                name + BIAS_SUFFIX, DType.float32, loc.rows, 0, loc.rows * 4)
        var id = SlotIdentity(
            name=name, local_name=local, layer_idx=layer_idx,
            shard=loc.shard, src_offset=loc.data_start,
            src_dtype=loc.dtype, rows=loc.rows, cols=loc.cols,
            weight_off=weight_off,
        )
        self.slots.append(RouterPlan(id^, gauge_off, bias_name, bias_off))
        return True

    def add_entry(
        mut self, name: String, dt: DType, s0: Int, s1: Int, size: Int,
    ) -> Int:
        var off = self.data_cursor
        self.entries.append(OutputEntry(
            name=name, dtype=dt, shape0=s0, shape1=s1,
            data_start=off, data_end=off + size,
        ))
        self.data_cursor += size
        return off

    def write_header(mut self) -> Bool:
        var header = build_header(self.entries)
        var header_size = len(header)
        var hp = PtrU8(unsafe_from_address=Int(header.unsafe_ptr()))
        if not write_sync(self.ring, self.output_fd_idx, 0, header_size, hp):
            return False
        self.data_start = header_size
        _ = header^
        return True

    def execute[P: BurstThreadPool, //](
        mut self, topo: NumaTopology, var pools: List[P],
    ) -> Bool:
        """Run the plan across the caller's `pools`, one job per pool. The
        caller is responsible for constructing pools — typically via
        `with_topological_rank_dispatch` so isolation mode and pinning match
        the rest of the system. Slots are bin-packed across pools; each pool
        receives one QuantShardKernel that opens its own io_uring on the
        worker thread and writes to preassigned, disjoint regions of the
        output file."""
        var n = len(pools)
        if n <= 0:
            print("quant: empty pool list")
            return False
        if n > len(topo):
            var rl = len(topo)
            print(t"quant: pool count {n} exceeds topology rank length {rl}")
            return False

        var buckets = partition_slots(self.slots, n)

        var kernels = List[QuantShardKernel](capacity=n)
        for w in range(n):
            kernels.append(QuantShardKernel(
                fds=fds_span(self.fds),
                output_fd_idx=self.output_fd_idx,
                headers=headers_span(self.headers),
                slots=slots_span(buckets[w]),
                data_start=self.data_start,
                scratch_cap=self.scratch_cap,
                rank=w,
            ))

        var pool_base = pools.unsafe_ptr()
        for w in range(n):
            var kernel_span = Span[QuantShardKernel, MutAnyOrigin](
                ptr=UnsafePointer[QuantShardKernel, MutAnyOrigin](
                    unsafe_from_address=Int(UnsafePointer(to=kernels[w]))),
                length=1)
            (pool_base + w)[].dispatch(kernel_span, 1)
        for w in range(n):
            (pool_base + w)[].join()

        _ = kernels^
        _ = buckets
        _ = pools^
        return True


@always_inline
def size_of_dtype(dt: DType) -> Int:
    return dtype_byte_size(dt)


@always_inline
def colsum_label(kind: Int) -> StaticString:
    if kind == ColsumKind.PER_ROW: return "row"
    if kind == ColsumKind.PER_BLOCK: return "block"
    return "none"


@always_inline
def gamma_label(ref g: GammaRef) -> StaticString:
    if not g.is_present(): return "none"
    if g.absorbed: return "absorbed"
    return "split"


struct QuantWorker(Movable):
    """One thread's worth of quantization state: its own io_uring, staging
    scratch, and gamma cache. Fields holding spans into shared, read-only
    storage (fds / headers) carry origins stripped to MutAnyOrigin so that
    QuantWorker is safe to construct on a worker thread; the caller is
    responsible for keeping the backing storage alive."""

    var ring: IoRing[QD]
    var scratch: QuantScratch
    var fds: Span[Int32, MutAnyOrigin]
    var output_fd_idx: Int
    var headers: Span[SafetensorsHeader, MutAnyOrigin]
    var data_start: Int
    var rank: Int
    var worker_idx: Int
    var gamma_split: Dict[String, PtrF32]
    var gamma_absorbed: Dict[String, PtrF32]
    var ready: Bool

    def __init__(
        out self,
        fds: Span[Int32, MutAnyOrigin],
        output_fd_idx: Int,
        headers: Span[SafetensorsHeader, MutAnyOrigin],
        data_start: Int,
        cap: ScratchCapacity,
        rank: Int = 0,
        worker_idx: Int = 0,
    ):
        self.fds = fds
        self.output_fd_idx = output_fd_idx
        self.headers = headers
        self.data_start = data_start
        self.rank = rank
        self.worker_idx = worker_idx
        self.ring = IoRing[QD]()
        self.scratch = QuantScratch(cap)
        self.gamma_split = Dict[String, PtrF32]()
        self.gamma_absorbed = Dict[String, PtrF32]()
        self.ready = False
        if not self.ring:
            print("quant worker: io_uring setup failed")
            return
        if not self.scratch:
            print("quant worker: scratch alloc failed")
            return
        try:
            _ = self.ring.register_fds(self.fds)
        except err:
            print(t"quant worker: register_fds failed: {err.error_message()}")
            return
        self.ready = True

    def __del__(deinit self):
        for entry in self.gamma_split.items():
            entry.value.free()
        for entry in self.gamma_absorbed.items():
            entry.value.free()

    def __bool__(self) -> Bool:
        return self.ready

    def run(mut self, slots: Span[SlotPlan, MutAnyOrigin]) -> Bool:
        for i in range(len(slots)):
            var plan = slots[i].copy()
            if not self.execute_slot(plan):
                self.report_slot_failure(plan)
                return False
        return True

    def execute_slot(mut self, plan: SlotPlan) -> Bool:
        self.log_slot(plan)
        if plan.isa[PassthroughPlan]():
            return self.do_passthrough(plan[PassthroughPlan])
        if plan.isa[QuantPlan]():
            return self.do_quant(plan[QuantPlan])
        if plan.isa[RouterPlan]():
            return self.do_router(plan[RouterPlan].copy())
        return False

    def log_slot(self, plan: SlotPlan):
        """One line per slot: rank/worker tag, layer marker, slot's local
        name, then a variant-specific description. Streamed via t-string
        so each line is one print call with no String allocation."""
        if plan.isa[PassthroughPlan]():
            ref pp = plan[PassthroughPlan]
            self.log_line(pp.id,
                t"passthrough {dtype_tag(pp.id.src_dtype)} {pp.byte_count}B")
        elif plan.isa[QuantPlan]():
            var qp = plan[QuantPlan].copy()
            var shape = StaticString("per-block") if qp.per_block else StaticString("per-row")
            var cs = colsum_label(qp.colsum_kind)
            var g = gamma_label(qp.gamma)
            if qp.two_sided_m != 0:
                self.log_line(qp.id,
                    t"{shape} fwht={qp.fwht_block} 2x={qp.two_sided_m} cs={cs} γ={g}")
            else:
                self.log_line(qp.id,
                    t"{shape} fwht={qp.fwht_block} cs={cs} γ={g}")
        elif plan.isa[RouterPlan]():
            var rp = plan[RouterPlan].copy()
            if rp.bias_name.byte_length() > 0:
                self.log_line(rp.id,
                    t"router-center +bias({rp.bias_name})")
            else:
                self.log_line(rp.id, t"router-center")

    def log_line[W: Writable](self, id: SlotIdentity, desc: W):
        if id.layer_idx >= 0:
            print(t"[r{self.rank}/w{self.worker_idx} L{id.layer_idx}] {id.local_name} :: {desc}")
        else:
            print(t"[r{self.rank}/w{self.worker_idx} tail ] {id.local_name} :: {desc}")

    def report_slot_failure(self, plan: SlotPlan):
        var name: String
        if plan.isa[PassthroughPlan]():
            name = plan[PassthroughPlan].id.name
        elif plan.isa[QuantPlan]():
            name = plan[QuantPlan].id.name
        elif plan.isa[RouterPlan]():
            name = plan[RouterPlan].copy().id.name
        else:
            name = String("<unknown>")
        print(t"quant: failed at {name}")

    def get_gamma(
        mut self, ref g: GammaRef, expected_cols: Int,
    ) -> Optional[PtrF32]:
        var existing = self.gamma_absorbed.get(g.name) if g.absorbed else self.gamma_split.get(g.name)
        if existing:
            return existing.value()

        var loc_opt = find_tensor(g.name, self.headers)
        if not loc_opt:
            print(t"quant: missing gamma {g.name}")
            return None
        var loc = loc_opt.take()
        var cols = loc.rows * loc.cols
        if cols != expected_cols:
            print(t"quant: gamma {g.name} cols {cols} != expected {expected_cols}")
            return None

        var raw = alloc[UInt8](loc.byte_size).as_any_origin()
        if not read_sync(self.ring, loc.shard, loc.data_start, loc.byte_size, raw):
            raw.free()
            return None

        var buf = alloc[Float32](cols).as_any_origin()
        if not decode_to_f32(loc.dtype, raw, buf, cols):
            print(t"quant: unsupported gamma dtype for {g.name}: {loc.dtype}")
            raw.free()
            buf.free()
            return None
        raw.free()

        if not g.absorbed:
            gamma_sqrt_abs_in_place(buf, cols)
            self.gamma_split[g.name] = buf
        else:
            self.gamma_absorbed[g.name] = buf
        return buf

    def do_passthrough(mut self, p: PassthroughPlan) -> Bool:
        var chunk_cap = min(COPY_CHUNK, p.byte_count)
        var buf = alloc[UInt8](chunk_cap).as_any_origin()
        var copied = 0
        while copied < p.byte_count:
            var n = min(COPY_CHUNK, p.byte_count - copied)
            if not read_sync(self.ring, p.id.shard,
                    p.id.src_offset + copied, n, buf):
                buf.free()
                return False
            if not write_sync(self.ring, self.output_fd_idx,
                    self.data_start + p.id.weight_off + copied, n, buf):
                buf.free()
                return False
            copied += n
        buf.free()
        return True

    def do_quant(mut self, p: QuantPlan) -> Bool:
        var src_bytes_per = size_of_dtype(p.id.src_dtype)
        if src_bytes_per <= 0:
            print(t"quant: bad src dtype {p.id.src_dtype}")
            return False
        if (p.id.src_dtype != DType.bfloat16
                and p.id.src_dtype != DType.float32
                and p.id.src_dtype != DType.float16):
            print(t"quant: unsupported source dtype for quant: {p.id.src_dtype}")
            return False

        var gamma_ptr = PtrF32.unsafe_dangling()
        var has_gamma = p.gamma.is_present()
        if has_gamma:
            var g = self.get_gamma(p.gamma, p.id.cols)
            if not g:
                return False
            gamma_ptr = g.value()

        var nb = p.id.cols // p.fwht_block if p.per_block else 1
        var scale_per_row = nb if p.per_block else 1
        var cs_per_row: Int
        if p.colsum_kind == ColsumKind.PER_ROW:
            cs_per_row = 1
        elif p.colsum_kind == ColsumKind.PER_BLOCK:
            cs_per_row = p.id.cols // p.fwht_block
        else:
            cs_per_row = 0

        var pr = min(PANEL_ROWS, p.id.rows)
        var row_off = 0
        while row_off < p.id.rows:
            var panel_rows = min(pr, p.id.rows - row_off)
            var panel_bytes = panel_rows * p.id.cols * src_bytes_per
            var src_off = p.id.src_offset + row_off * p.id.cols * src_bytes_per
            if not read_sync(self.ring, p.id.shard, src_off, panel_bytes,
                    self.scratch.src_buf):
                return False

            if not decode_to_f32(p.id.src_dtype, self.scratch.src_buf,
                    self.scratch.work, panel_rows * p.id.cols):
                return False

            if has_gamma:
                for r in range(panel_rows):
                    apply_gamma_in_place(
                        self.scratch.work + r * p.id.cols, gamma_ptr, p.id.cols)

            if p.per_block:
                rotate_and_quant[True](
                    p.fwht_block, self.scratch.work, self.scratch.qi,
                    self.scratch.scales, panel_rows, p.id.cols, p.two_sided_m)
            else:
                rotate_and_quant[False](
                    p.fwht_block, self.scratch.work, self.scratch.qi,
                    self.scratch.scales, panel_rows, p.id.cols, p.two_sided_m)

            if p.colsum_kind == ColsumKind.PER_ROW:
                colsum_per_row(self.scratch.qi, self.scratch.cs,
                    panel_rows, p.id.cols)
            elif p.colsum_kind == ColsumKind.PER_BLOCK:
                colsum_per_block(self.scratch.qi, self.scratch.cs,
                    p.fwht_block, panel_rows, p.id.cols)

            var w_off = self.data_start + p.id.weight_off + row_off * p.id.cols
            var s_off = (self.data_start + p.scale_off
                + row_off * scale_per_row * 4)
            var c_off = (self.data_start + p.cs_off
                + row_off * cs_per_row * 4) if cs_per_row > 0 else 0

            var ops = InlineArray[WriteOp[], 3](uninitialized=True)
            ops[0] = WriteOp(
                file_idx=self.output_fd_idx, offset=w_off,
                length=panel_rows * p.id.cols,
                src=self.scratch.qi.bitcast[UInt8](), id=0)
            ops[1] = WriteOp(
                file_idx=self.output_fd_idx, offset=s_off,
                length=panel_rows * scale_per_row * 4,
                src=self.scratch.scales.bitcast[UInt8](), id=1)
            var n_ops = 2
            if cs_per_row > 0:
                ops[2] = WriteOp(
                    file_idx=self.output_fd_idx, offset=c_off,
                    length=panel_rows * cs_per_row * 4,
                    src=self.scratch.cs.bitcast[UInt8](), id=2)
                n_ops = 3

            var ops_span = Span[WriteOp[], MutAnyOrigin](
                ptr=UnsafePointer[WriteOp[], MutAnyOrigin](
                    unsafe_from_address=Int(UnsafePointer(to=ops[0]))),
                length=n_ops)
            if not write_sync_many(self.ring, ops_span):
                return False

            row_off += panel_rows
        return True

    def do_router(mut self, var p: RouterPlan) -> Bool:
        var rows = p.id.rows
        var cols = p.id.cols
        if p.id.src_dtype != DType.bfloat16 and p.id.src_dtype != DType.float32:
            print(t"quant: router_center only supports bf16/f32 source; got {p.id.src_dtype}")
            return False
        var src_bytes_per = size_of_dtype(p.id.src_dtype)

        var src_buf = alloc[UInt8](rows * cols * src_bytes_per).as_any_origin()
        var gauge_f32 = alloc[Float32](cols).as_any_origin()
        var centered = alloc[Scalar[DType.bfloat16]](rows * cols).as_any_origin()
        var gauge_bf16 = alloc[Scalar[DType.bfloat16]](cols).as_any_origin()

        var ok = read_sync(self.ring, p.id.shard, p.id.src_offset,
            rows * cols * src_bytes_per, src_buf)

        if ok:
            if p.id.src_dtype == DType.bfloat16:
                router_center[DType.bfloat16](
                    src_buf.bitcast[Scalar[DType.bfloat16]](),
                    gauge_f32, centered, gauge_bf16, rows, cols)
            else:
                router_center[DType.float32](
                    src_buf.bitcast[Float32](),
                    gauge_f32, centered, gauge_bf16, rows, cols)

            ok = write_sync(self.ring, self.output_fd_idx,
                self.data_start + p.id.weight_off, rows * cols * 2,
                centered.bitcast[UInt8]())

        if ok:
            ok = write_sync(self.ring, self.output_fd_idx,
                self.data_start + p.gauge_off, cols * 2,
                gauge_bf16.bitcast[UInt8]())

        if ok and p.bias_name.byte_length() > 0:
            ok = self.write_router_bias(p)

        src_buf.free()
        gauge_f32.free()
        centered.free()
        gauge_bf16.free()
        return ok

    def write_router_bias(mut self, ref p: RouterPlan) -> Bool:
        var loc_opt = find_tensor(p.bias_name, self.headers)
        if not loc_opt:
            print(t"quant: missing router bias {p.bias_name}")
            return False
        var loc = loc_opt.take()
        var rows = p.id.rows
        if loc.rows * loc.cols != rows:
            print(t"quant: router bias {p.bias_name} size mismatch")
            return False
        var raw = alloc[UInt8](loc.byte_size).as_any_origin()
        if not read_sync(self.ring, loc.shard, loc.data_start, loc.byte_size, raw):
            raw.free()
            return False
        var f32 = alloc[Float32](rows).as_any_origin()
        if not decode_to_f32(loc.dtype, raw, f32, rows):
            print(t"quant: unsupported router bias dtype for {p.bias_name}: {loc.dtype}")
            raw.free()
            f32.free()
            return False
        raw.free()
        var ok = write_sync(self.ring, self.output_fd_idx,
            self.data_start + p.bias_off, rows * 4,
            f32.bitcast[UInt8]())
        f32.free()
        return ok


def estimate_slot_bytes(plan: SlotPlan) -> Int:
    """Rough I/O+compute weight for bin-packing slots across workers.
    Counts source-read bytes plus output bytes; gamma reads are amortized
    across the slots that share a gamma and are dropped here."""
    if plan.isa[PassthroughPlan]():
        return plan[PassthroughPlan].byte_count
    if plan.isa[QuantPlan]():
        var q = plan[QuantPlan].copy()
        var rows = q.id.rows
        var cols = q.id.cols
        var src = rows * cols * dtype_byte_size(q.id.src_dtype)
        var weight = rows * cols
        var nb = (cols // q.fwht_block) if q.per_block else 1
        var scale = rows * nb * 4
        var cs = 0
        if q.colsum_kind == ColsumKind.PER_ROW:
            cs = rows * 4
        elif q.colsum_kind == ColsumKind.PER_BLOCK:
            cs = rows * (cols // q.fwht_block) * 4
        return src + weight + scale + cs
    if plan.isa[RouterPlan]():
        var r = plan[RouterPlan].copy()
        return r.id.rows * r.id.cols * 3
    return 0


def partition_slots(
    ref slots: List[SlotPlan], num_workers: Int,
) -> List[List[SlotPlan]]:
    """Greedy descending bin-pack: sort slots by estimated bytes desc and
    assign each to the worker with least current load. Selection sort over
    indices is O(n²) but n~1500 and this runs once per quantize."""
    var n = len(slots)
    var sizes = List[Int](capacity=n)
    var order = List[Int](capacity=n)
    for i in range(n):
        sizes.append(estimate_slot_bytes(slots[i]))
        order.append(i)
    for i in range(n - 1):
        var max_idx = i
        for j in range(i + 1, n):
            if sizes[order[j]] > sizes[order[max_idx]]:
                max_idx = j
        if max_idx != i:
            var tmp = order[i]
            order[i] = order[max_idx]
            order[max_idx] = tmp

    var buckets = List[List[SlotPlan]](capacity=num_workers)
    var loads = List[Int](capacity=num_workers)
    for _ in range(num_workers):
        buckets.append(List[SlotPlan]())
        loads.append(0)

    for k in range(n):
        var idx = order[k]
        var sz = sizes[idx]
        var min_w = 0
        for w in range(1, num_workers):
            if loads[w] < loads[min_w]:
                min_w = w
        buckets[min_w].append(slots[idx].copy())
        loads[min_w] += sz
    return buckets^


@fieldwise_init
struct QuantShardKernel(BurstKernel):
    """Per-worker payload dispatched through a BurstPool mailbox. Owns no
    resources — every field is POD with caller-managed backing storage.
    Mirrors `LoadShardKernel` in `linux/io_uring.mojo:601`."""
    var fds: Span[Int32, MutAnyOrigin]
    var output_fd_idx: Int
    var headers: Span[SafetensorsHeader, MutAnyOrigin]
    var slots: Span[SlotPlan, MutAnyOrigin]
    var data_start: Int
    var scratch_cap: ScratchCapacity
    var rank: Int

    def execute(mut self):
        var sys = linux.linux_sys()
        var worker = QuantWorker(
            fds=self.fds, output_fd_idx=self.output_fd_idx,
            headers=self.headers, data_start=self.data_start,
            cap=self.scratch_cap, rank=self.rank, worker_idx=0,
        )
        if not worker:
            print("quant worker: setup failed")
            sys.sys_exit_group(1)
            return
        if not worker.run(self.slots):
            print("quant worker: run failed")
            sys.sys_exit_group(1)
            return
