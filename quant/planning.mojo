from std.memory import UnsafePointer, alloc
from std.pathlib import Path
from std.reflection import reflect

from safetensors.parser import (
    dtype_byte_size, dtype_tag, parse_safetensors_header,
    SafetensorsHeader, TensorMeta,
)
from modeling.slot import SlotLike, SlotGroup
from quant.recipe import (
    QuantRecipe, Passthrough, PerRowQuant, PerBlockQuant, RouterCenter,
    NoGamma, SplitGamma, AbsorbedGamma,
    SingleSided, TwoSided,
    NoColsum, PerRowCs, PerBlockCs,
)
from quant.source_format import (
    Converter, Raw, Fp8E4M3Block,
    Bf16Converter, F32Converter, F16Converter, Fp8E4M3Block128Converter,
)


@fieldwise_init
struct LocatedTensor(Copyable, Movable):
    var shard: Int
    var dtype: DType
    var data_start: Int
    var byte_size: Int
    var shape: List[Int]


def find_tensor(
    name: String, headers: List[SafetensorsHeader],
) -> Optional[LocatedTensor]:
    for i in range(len(headers)):
        var m = headers[i].tensors.get(name)
        if m:
            ref tm = m.value()
            return LocatedTensor(
                shard=i,
                dtype=tm.dtype,
                data_start=headers[i].data_offset + tm.start,
                byte_size=tm.end - tm.start,
                shape=tm.shape.copy(),
            )
    return None


def discover_quant_shards(path: Path) -> List[Path]:
    var shards = List[Path]()
    if String(path).endswith(".safetensors"):
        shards.append(path)
        return shards^
    var names = List[String]()
    try:
        for entry in path.listdir():
            var name = String(entry)
            if name.endswith(".safetensors"):
                shards.append(path / name)
                names.append(name^)
    except:
        pass
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if names[j] < names[i]:
                var tmp_name = names[i]
                names[i] = names[j]
                names[j] = tmp_name
                var tmp_shard = shards[i]
                shards[i] = shards[j]
                shards[j] = tmp_shard
    return shards^


def parse_source_headers(
    paths: List[Path],
) -> Optional[List[SafetensorsHeader]]:
    var headers = List[SafetensorsHeader]()
    for i in range(len(paths)):
        var h = parse_safetensors_header(paths[i])
        if not h:
            ref p = paths[i]
            print(t"quant: failed to parse {p}")
            return None
        headers.append(h.take())
    return headers^


@fieldwise_init
struct OutputEntry(Copyable, Movable):
    var name: String
    var dtype: DType
    var shape: List[Int]
    var data_start: Int
    var data_end: Int

    def byte_size(self) -> Int:
        return self.data_end - self.data_start


struct HeaderBuffer(Movable):
    var data: UnsafePointer[UInt8, MutExternalOrigin]
    var size: Int

    def __init__(
        out self,
        var data: UnsafePointer[UInt8, MutExternalOrigin],
        size: Int,
    ):
        self.data = data
        self.size = size

    def __del__(deinit self):
        self.data.free()

    @always_inline
    def any_ptr(self) -> UnsafePointer[UInt8, MutAnyOrigin]:
        return self.data.as_any_origin()


def build_header(entries: List[OutputEntry]) -> HeaderBuffer:
    var json = String("{")
    for i in range(len(entries)):
        if i > 0:
            json += ","
        ref e = entries[i]
        json += '"' + e.name + '":{"dtype":"' + String(dtype_tag(e.dtype))
        json += '","shape":['
        for d in range(len(e.shape)):
            if d > 0:
                json += ","
            json += String(e.shape[d])
        json += '],"data_offsets":[' + String(e.data_start)
        json += "," + String(e.data_end) + "]}"
    json += "}"
    var bytes = json.as_bytes()
    var json_len = len(bytes)
    var total = 8 + json_len
    var data = alloc[UInt8](total)
    var hl = UInt64(json_len)
    for i in range(8):
        data[i] = UInt8((hl >> UInt64(i * 8)) & 0xFF)
    for i in range(json_len):
        data[8 + i] = bytes[i]
    return HeaderBuffer(data, total)


def folds_to_shape(src: List[Int], rows: Int, cols: Int) -> Bool:
    if len(src) == 1:
        return cols == 1 and src[0] == rows
    if len(src) == 2:
        return src[0] == rows and src[1] == cols
    var r = 1
    for d in range(len(src) - 1):
        r *= src[d]
    return r == rows and src[len(src) - 1] == cols


def emit_per_row_entries(
    mut entries: List[OutputEntry],
    name: String, src_shape: List[Int],
    rows: Int, cols: Int, cs_blocks: Int, off_in: Int,
) -> Int:
    var off = off_in
    var wb = rows * cols
    entries.append(OutputEntry(
        name=name, dtype=DType.int8, shape=src_shape.copy(),
        data_start=off, data_end=off + wb))
    off += wb
    var sb = rows * 4
    var scale_shape = List[Int]()
    scale_shape.append(rows)
    entries.append(OutputEntry(
        name=name + "_scale", dtype=DType.float32, shape=scale_shape^,
        data_start=off, data_end=off + sb))
    off += sb
    if cs_blocks > 0:
        var cb = rows * cs_blocks * 4
        var cs_shape = List[Int]()
        cs_shape.append(rows)
        if cs_blocks > 1:
            cs_shape.append(cs_blocks)
        entries.append(OutputEntry(
            name=name + "_cs", dtype=DType.float32, shape=cs_shape^,
            data_start=off, data_end=off + cb))
        off += cb
    return off


def emit_per_block_entries(
    mut entries: List[OutputEntry],
    name: String, src_shape: List[Int],
    rows: Int, cols: Int, fwht_block: Int, cs_blocks: Int, off_in: Int,
) -> Int:
    var off = off_in
    var nb = cols // fwht_block
    var wb = rows * cols
    entries.append(OutputEntry(
        name=name, dtype=DType.int8, shape=src_shape.copy(),
        data_start=off, data_end=off + wb))
    off += wb
    var sb = rows * nb * 4
    var scale_shape = List[Int]()
    scale_shape.append(rows)
    scale_shape.append(nb)
    entries.append(OutputEntry(
        name=name + "_scale", dtype=DType.float32, shape=scale_shape^,
        data_start=off, data_end=off + sb))
    off += sb
    if cs_blocks > 0:
        var cb = rows * cs_blocks * 4
        var cs_shape = List[Int]()
        cs_shape.append(rows)
        cs_shape.append(cs_blocks)
        entries.append(OutputEntry(
            name=name + "_cs", dtype=DType.float32, shape=cs_shape^,
            data_start=off, data_end=off + cb))
        off += cb
    return off


def emit_router_center_entries(
    mut entries: List[OutputEntry],
    name: String, bias_name: String, src_shape: List[Int],
    rows: Int, cols: Int, off_in: Int,
) -> Int:
    var off = off_in
    var wb = rows * cols * 2
    entries.append(OutputEntry(
        name=name, dtype=DType.bfloat16, shape=src_shape.copy(),
        data_start=off, data_end=off + wb))
    off += wb
    var gb = cols * 2
    var gauge_shape = List[Int]()
    gauge_shape.append(cols)
    entries.append(OutputEntry(
        name=name + "_gauge", dtype=DType.bfloat16, shape=gauge_shape^,
        data_start=off, data_end=off + gb))
    off += gb
    if bias_name.byte_length() > 0:
        var bb = rows * 4
        var bias_shape = List[Int]()
        bias_shape.append(rows)
        entries.append(OutputEntry(
            name=bias_name, dtype=DType.float32, shape=bias_shape^,
            data_start=off, data_end=off + bb))
        off += bb
    return off


def emit_passthrough_entry(
    mut entries: List[OutputEntry],
    name: String, src_shape: List[Int], dtype: DType, bytes: Int, off_in: Int,
) -> Int:
    entries.append(OutputEntry(
        name=name, dtype=dtype, shape=src_shape.copy(),
        data_start=off_in, data_end=off_in + bytes))
    return off_in + bytes


@always_inline
def validate_aux_companion[C: Converter](
    weight_name: String, headers: List[SafetensorsHeader],
    rows: Int, cols: Int,
) -> Bool:
    comptime if C.AUX_SUFFIX == StaticString(""):
        return True
    var aux_name = weight_name + String(C.AUX_SUFFIX)
    var aux_opt = find_tensor(aux_name, headers)
    if not aux_opt:
        print(t"quant: missing aux companion {aux_name}")
        return False
    var aux = aux_opt.take()
    if aux.dtype != C.AUX_DTYPE:
        print(t"quant: aux dtype mismatch for {aux_name}")
        return False
    var expected = C.aux_bytes_for(rows, cols)
    if aux.byte_size != expected:
        print(t"quant: aux byte-size mismatch for {aux_name}: expected {expected} got {aux.byte_size}")
        return False
    return True


def emit_quant_plan[T: AnyType](
    prefix: String,
    headers: List[SafetensorsHeader],
    mut entries: List[OutputEntry],
    off_in: Int = 0,
) -> Optional[Int]:
    var off = off_in
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, SlotLike):
            comptime if FT.NAME != StaticString(""):
                var full = prefix + String(FT.NAME)
                var loc_opt = find_tensor(full, headers)
                if not loc_opt:
                    print(t"quant: missing source tensor {full}")
                    return None
                var loc = loc_opt.take()
                comptime expected = FT.ENCODING.DTYPE
                if loc.dtype != expected:
                    print(t"quant: dtype mismatch for {full}: expected {expected} got {loc.dtype}")
                    return None
                comptime ROWS = FT.SHAPE.GLOBAL_N
                comptime COLS = FT.SHAPE.GLOBAL_M
                if not folds_to_shape(loc.shape, ROWS, COLS):
                    print(t"quant: shape mismatch for {full}: expected rows={ROWS} cols={COLS}")
                    return None
                comptime QV = FT.QUANT
                comptime if QV.isa[PerRowQuant]():
                    comptime QT = QV[PerRowQuant]
                    if COLS % QT.fwht_block != 0:
                        print(t"quant: {full} cols={COLS} not divisible by fwht_block={QT.fwht_block}")
                        return None
                    comptime if expected == DType.float8_e4m3fn:
                        if not validate_aux_companion[Fp8E4M3Block128Converter](
                                full, headers, ROWS, COLS):
                            return None
                    comptime CS_BLOCKS = (
                        0 if QT.colsum.isa[NoColsum]()
                        else (1 if QT.colsum.isa[PerRowCs]()
                              else COLS // QT.fwht_block)
                    )
                    off = emit_per_row_entries(
                        entries, full, loc.shape, ROWS, COLS, CS_BLOCKS, off)
                comptime if QV.isa[PerBlockQuant]():
                    comptime QT = QV[PerBlockQuant]
                    if COLS % QT.fwht_block != 0:
                        print(t"quant: {full} cols={COLS} not divisible by fwht_block={QT.fwht_block}")
                        return None
                    comptime if expected == DType.float8_e4m3fn:
                        if not validate_aux_companion[Fp8E4M3Block128Converter](
                                full, headers, ROWS, COLS):
                            return None
                    comptime CS_BLOCKS = (
                        0 if QT.colsum.isa[NoColsum]()
                        else COLS // QT.fwht_block
                    )
                    off = emit_per_block_entries(
                        entries, full, loc.shape, ROWS, COLS,
                        QT.fwht_block, CS_BLOCKS, off)
                comptime if QV.isa[RouterCenter]():
                    comptime QT = QV[RouterCenter]
                    off = emit_router_center_entries(
                        entries, full, String(QT.bias_name),
                        loc.shape, ROWS, COLS, off)
                comptime if QV.isa[Passthrough]():
                    off = emit_passthrough_entry(
                        entries, full, loc.shape, loc.dtype, loc.byte_size, off)
        comptime if conforms_to(FT, SlotGroup):
            var nested = emit_quant_plan[FT](prefix, headers, entries, off)
            if not nested:
                return None
            off = nested.value()
    return off
