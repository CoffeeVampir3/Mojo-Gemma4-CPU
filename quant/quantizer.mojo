from std.algorithm import vectorize
from std.collections import Dict, InlineArray
from std.math import max, min, align_up
from std.memory import Span, UnsafePointer, alloc
from std.pathlib import Path
from std.reflection import reflect
from std.sys.info import simd_width_of, size_of

from numa import NumaTopology, NumaArena
from threading.burst_threading import BurstPool
from threading.threading_traits import BurstKernel
from linux.io_uring import (
    IoRing, ReadOp, WriteOp, Completion,
    ReadMode, WriteMode, open_files_for_ring, close_fds,
)
from safetensors.parser import SafetensorsHeader
from butterquant.kernels import (
    apply_gamma_in_place, gamma_sqrt_abs_in_place,
    rotate_and_quant, router_center,
    colsum_per_row, colsum_per_block,
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
from quant.planning import (
    LocatedTensor, OutputEntry, HeaderBuffer,
    find_tensor, build_header, emit_quant_plan,
)


comptime PtrU8 = UnsafePointer[UInt8, MutAnyOrigin]
comptime PtrF32 = UnsafePointer[Float32, MutAnyOrigin]
comptime PtrBF16 = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime PtrI8 = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
comptime SrcPtr[dtype: DType] = UnsafePointer[Scalar[dtype], MutAnyOrigin]

comptime DEFAULT_QUEUE_DEPTH = 256
comptime DEFAULT_PANEL_ROWS = 2048
comptime DEFAULT_COPY_CHUNK = 16 * 1024 * 1024
comptime DEFAULT_IN_FLIGHT = 4
comptime DEFAULT_MASK_SIZE = 128

comptime SLOT_FREE = 0
comptime SLOT_READING = 1
comptime SLOT_MATH = 2
comptime SLOT_WRITING = 3

comptime MAX_WRITES_PER_PANEL = 3

comptime W = simd_width_of[DType.float32]()


@always_inline
def panel_rows_for[rows: Int]() -> Int:
    comptime if rows < DEFAULT_PANEL_ROWS:
        return rows
    return DEFAULT_PANEL_ROWS


@always_inline
def num_panels_for[rows: Int]() -> Int:
    comptime pr = panel_rows_for[rows]()
    comptime if pr == 0:
        return 0
    return (rows + pr - 1) // pr


@always_inline
def last_panel_rows_for[rows: Int]() -> Int:
    comptime pr = panel_rows_for[rows]()
    comptime if pr == 0:
        return 0
    comptime rem = rows - (rows // pr) * pr
    comptime if rem == 0:
        return pr
    return rem


@always_inline
def source_aux_row_block[dt: DType]() -> Int:
    comptime if dt == DType.float8_e4m3fn:
        return 128
    return 0


@always_inline
def source_aux_bytes[dt: DType](rows: Int, cols: Int) -> Int:
    comptime if dt == DType.float8_e4m3fn:
        return Fp8E4M3Block128Converter.aux_bytes_for(rows, cols)
    return 0


@always_inline
def source_aux_suffix[dt: DType]() -> StaticString:
    comptime if dt == DType.float8_e4m3fn:
        return "_scale_inv"
    return ""


def per_row_slot_bytes[FT: SlotLike]() -> Int:
    comptime SRC_BYTES = size_of[Scalar[FT.ENCODING.DTYPE]]()
    comptime COLS = FT.SHAPE.GLOBAL_M
    comptime PR = panel_rows_for[FT.SHAPE.GLOBAL_N]()
    comptime QT = FT.QUANT[PerRowQuant]
    comptime FW = QT.fwht_block
    comptime CS_BLOCKS = (
        0 if QT.colsum.isa[NoColsum]()
        else (1 if QT.colsum.isa[PerRowCs]() else COLS // FW)
    )
    var total = align_up(PR * COLS * SRC_BYTES, 64)
    total += align_up(PR * COLS * 4, 64)      # work f32
    total += align_up(PR * COLS, 64)          # qi int8
    total += align_up(PR * 4, 64)             # per-row scale f32
    total += align_up(PR * max(1, CS_BLOCKS) * 4, 64)  # cs (unused if no colsum)
    return total


def per_block_slot_bytes[FT: SlotLike]() -> Int:
    comptime SRC_BYTES = size_of[Scalar[FT.ENCODING.DTYPE]]()
    comptime COLS = FT.SHAPE.GLOBAL_M
    comptime PR = panel_rows_for[FT.SHAPE.GLOBAL_N]()
    comptime QT = FT.QUANT[PerBlockQuant]
    comptime NB = COLS // QT.fwht_block
    comptime CS_BLOCKS = 0 if QT.colsum.isa[NoColsum]() else NB
    var total = align_up(PR * COLS * SRC_BYTES, 64)
    total += align_up(PR * COLS * 4, 64)
    total += align_up(PR * COLS, 64)
    total += align_up(PR * NB * 4, 64)
    total += align_up(PR * max(1, CS_BLOCKS) * 4, 64)
    return total


def router_slot_bytes[FT: SlotLike]() -> Int:
    comptime SRC_BYTES = size_of[Scalar[FT.ENCODING.DTYPE]]()
    comptime ROWS = FT.SHAPE.GLOBAL_N
    comptime COLS = FT.SHAPE.GLOBAL_M
    var total = align_up(ROWS * COLS * SRC_BYTES, 64)
    total += align_up(COLS * 4, 64)
    total += align_up(ROWS * COLS * 2, 64)
    total += align_up(COLS * 2, 64)
    return total


def passthrough_slot_bytes() -> Int:
    return DEFAULT_COPY_CHUNK


def slot_bytes_for[FT: SlotLike]() -> Int:
    comptime QV = FT.QUANT
    comptime if QV.isa[PerRowQuant]():
        return per_row_slot_bytes[FT]()
    comptime if QV.isa[PerBlockQuant]():
        return per_block_slot_bytes[FT]()
    comptime if QV.isa[RouterCenter]():
        return router_slot_bytes[FT]()
    return passthrough_slot_bytes()


def max_slot_bytes_for_rank[T: AnyType](
    rank: Int, tp: Int, mut slot_idx: Int, in_max: Int,
) -> Int:
    var current = in_max
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, SlotLike):
            comptime if FT.NAME != StaticString(""):
                if slot_idx % tp == rank:
                    comptime n = slot_bytes_for[FT]()
                    if n > current:
                        current = n
                slot_idx += 1
        comptime if conforms_to(FT, SlotGroup):
            current = max_slot_bytes_for_rank[FT](rank, tp, slot_idx, current)
    return current


@always_inline
def decode_with[C: Converter](
    src_bytes: PtrU8, aux_ptr: PtrU8, work: PtrF32,
    panel_rows: Int, cols: Int,
):
    """Reinterpret `src_bytes` (raw disk bytes loaded into a slot) as a
    typed buffer of `C.SOURCE_DTYPE` and decode to f32 via `C.convert`.
    Aux bytes are reinterpreted to `C.AUX_DTYPE` when the converter
    declares one."""
    var src = src_bytes.bitcast[Scalar[C.SOURCE_DTYPE]]()
    var aux = aux_ptr.bitcast[Scalar[C.AUX_DTYPE]]()
    C.convert(src, aux, work, panel_rows, cols)


@always_inline
def decode_panel[FT: SlotLike](
    src_bytes: PtrU8, aux_ptr: PtrU8, work: PtrF32, panel_rows: Int,
):
    """Dispatch by `FT.ENCODING.DTYPE` to the matching `Converter`.
    Branches are mutually exclusive at comptime; only one expands."""
    comptime SRC = FT.ENCODING.DTYPE
    comptime COLS = FT.SHAPE.GLOBAL_M
    comptime if SRC == DType.bfloat16:
        decode_with[Bf16Converter](src_bytes, aux_ptr, work, panel_rows, COLS)
    comptime if SRC == DType.float32:
        decode_with[F32Converter](src_bytes, aux_ptr, work, panel_rows, COLS)
    comptime if SRC == DType.float16:
        decode_with[F16Converter](src_bytes, aux_ptr, work, panel_rows, COLS)
    comptime if SRC == DType.float8_e4m3fn:
        decode_with[Fp8E4M3Block128Converter](
            src_bytes, aux_ptr, work, panel_rows, COLS)


comptime MathFn = def(PtrU8, Int, PtrF32, PtrU8) thin -> None


@always_inline
def carve_per_row[FT: SlotLike](
    base: PtrU8,
) -> Tuple[PtrU8, PtrF32, PtrI8, PtrF32, PtrF32]:
    comptime SRC_BYTES = size_of[Scalar[FT.ENCODING.DTYPE]]()
    comptime COLS = FT.SHAPE.GLOBAL_M
    comptime PR = panel_rows_for[FT.SHAPE.GLOBAL_N]()
    comptime SRC_END = align_up(PR * COLS * SRC_BYTES, 64)
    comptime WORK_END = SRC_END + align_up(PR * COLS * 4, 64)
    comptime QI_END = WORK_END + align_up(PR * COLS, 64)
    comptime SCALES_END = QI_END + align_up(PR * 4, 64)
    return (
        base,
        (base + SRC_END).bitcast[Float32](),
        (base + WORK_END).bitcast[Scalar[DType.int8]](),
        (base + QI_END).bitcast[Float32](),
        (base + SCALES_END).bitcast[Float32](),
    )


@always_inline
def carve_per_block[FT: SlotLike](
    base: PtrU8,
) -> Tuple[PtrU8, PtrF32, PtrI8, PtrF32, PtrF32]:
    comptime SRC_BYTES = size_of[Scalar[FT.ENCODING.DTYPE]]()
    comptime COLS = FT.SHAPE.GLOBAL_M
    comptime PR = panel_rows_for[FT.SHAPE.GLOBAL_N]()
    comptime QT = FT.QUANT[PerBlockQuant]
    comptime NB = COLS // QT.fwht_block
    comptime SRC_END = align_up(PR * COLS * SRC_BYTES, 64)
    comptime WORK_END = SRC_END + align_up(PR * COLS * 4, 64)
    comptime QI_END = WORK_END + align_up(PR * COLS, 64)
    comptime SCALES_END = QI_END + align_up(PR * NB * 4, 64)
    return (
        base,
        (base + SRC_END).bitcast[Float32](),
        (base + WORK_END).bitcast[Scalar[DType.int8]](),
        (base + QI_END).bitcast[Float32](),
        (base + SCALES_END).bitcast[Float32](),
    )


@always_inline
def carve_router[FT: SlotLike](
    base: PtrU8,
) -> Tuple[PtrU8, PtrF32, PtrBF16, PtrBF16]:
    comptime SRC_BYTES = size_of[Scalar[FT.ENCODING.DTYPE]]()
    comptime ROWS = FT.SHAPE.GLOBAL_N
    comptime COLS = FT.SHAPE.GLOBAL_M
    comptime SRC_END = align_up(ROWS * COLS * SRC_BYTES, 64)
    comptime GAUGE_END = SRC_END + align_up(COLS * 4, 64)
    comptime CENTERED_END = GAUGE_END + align_up(ROWS * COLS * 2, 64)
    return (
        base,
        (base + SRC_END).bitcast[Float32](),
        (base + GAUGE_END).bitcast[Scalar[DType.bfloat16]](),
        (base + CENTERED_END).bitcast[Scalar[DType.bfloat16]](),
    )


@always_inline
def apply_rotation[QT: AnyType](
    work: PtrF32, qi: PtrI8, scales: PtrF32,
    panel_rows: Int, cols: Int, per_block: Bool,
):
    comptime RT = QT.rotation
    comptime m_block = 0 if RT.isa[SingleSided]() else RT[TwoSided].m_block
    rotate_and_quant[per_block](
        QT.fwht_block, work, qi, scales, panel_rows, cols, m_block)


def math_body[FT: SlotLike](
    slot_base: PtrU8, panel_rows: Int, gamma_ptr: PtrF32, aux_ptr: PtrU8,
):
    comptime SRC = FT.ENCODING.DTYPE
    comptime COLS = FT.SHAPE.GLOBAL_M
    comptime QV = FT.QUANT

    comptime if QV.isa[PerRowQuant]():
        comptime QT = QV[PerRowQuant]
        comptime RT = QT.rotation
        comptime m_block = 0 if RT.isa[SingleSided]() else RT[TwoSided].m_block
        var sp = carve_per_row[FT](slot_base)
        var src_bytes = sp[0]
        var work = sp[1]
        var qi = sp[2]
        var scales = sp[3]
        var cs = sp[4]
        decode_panel[FT](src_bytes, aux_ptr, work, panel_rows)
        comptime if QT.gamma.isa[SplitGamma]() or QT.gamma.isa[AbsorbedGamma]():
            for r in range(panel_rows):
                apply_gamma_in_place(work + r * COLS, gamma_ptr, COLS)
        rotate_and_quant[False](
            QT.fwht_block, work, qi, scales, panel_rows, COLS, m_block)
        comptime if QT.colsum.isa[PerRowCs]():
            colsum_per_row(qi, cs, panel_rows, COLS)
        comptime if QT.colsum.isa[PerBlockCs]():
            colsum_per_block(qi, cs, QT.fwht_block, panel_rows, COLS)
        return

    comptime if QV.isa[PerBlockQuant]():
        comptime QT = QV[PerBlockQuant]
        comptime RT = QT.rotation
        comptime m_block = 0 if RT.isa[SingleSided]() else RT[TwoSided].m_block
        var sp = carve_per_block[FT](slot_base)
        var src_bytes = sp[0]
        var work = sp[1]
        var qi = sp[2]
        var scales = sp[3]
        var cs = sp[4]
        decode_panel[FT](src_bytes, aux_ptr, work, panel_rows)
        comptime if QT.gamma.isa[SplitGamma]() or QT.gamma.isa[AbsorbedGamma]():
            for r in range(panel_rows):
                apply_gamma_in_place(work + r * COLS, gamma_ptr, COLS)
        rotate_and_quant[True](
            QT.fwht_block, work, qi, scales, panel_rows, COLS, m_block)
        comptime if QT.colsum.isa[PerBlockCs]():
            colsum_per_block(qi, cs, QT.fwht_block, panel_rows, COLS)
        return

    comptime if QV.isa[RouterCenter]():
        var sp = carve_router[FT](slot_base)
        var src_ptr = sp[0].bitcast[Scalar[SRC]]()
        var gauge = sp[1]
        var centered = sp[2]
        var gauge_bf16 = sp[3]
        router_center[SRC](
            src_ptr, gauge, centered, gauge_bf16,
            FT.SHAPE.GLOBAL_N, FT.SHAPE.GLOBAL_M)
        return

    # Passthrough: identity. The slot's src bytes are the bytes we write back.
    return


comptime MAX_WRITE_STREAMS = MAX_WRITES_PER_PANEL


@fieldwise_init
struct WriteSpec(Copyable, Movable, ImplicitlyCopyable):
    var stream_base_offset: Int      # absolute file offset of this stream's base
    var slot_buf_offset: Int          # bytes within slot.base where this stream's data starts
    var bytes_per_panel_row: Int      # bytes per panel-row this stream emits
    var full_panel_bytes: Int         # bytes per full panel (=rows_per_panel*bytes_per_panel_row)


@fieldwise_init
struct TaskDesc(Copyable, Movable):
    var name: String
    var src_shard: Int
    var src_data_start: Int
    var src_row_bytes: Int            # bytes-per-row in source layout
    var rows_per_panel: Int           # full panel size
    var last_panel_rows: Int
    var num_panels: Int
    var panels_submitted: Int
    var panels_completed: Int
    var n_writes: Int
    var write0: WriteSpec
    var write1: WriteSpec
    var write2: WriteSpec
    var gamma_ptr: PtrF32
    var has_gamma: Bool
    var aux_base: PtrU8               # whole-tensor aux (FP8 _scale_inv); null if none
    var aux_row_block: Int            # source rows per aux row; 0 if no aux
    var aux_row_stride: Int           # bytes per aux row
    var math_fn: MathFn

    @always_inline
    def panel_rows(self, idx: Int) -> Int:
        return self.last_panel_rows if idx == self.num_panels - 1 else self.rows_per_panel

    @always_inline
    def src_offset(self, idx: Int) -> Int:
        return self.src_data_start + idx * self.rows_per_panel * self.src_row_bytes

    @always_inline
    def src_bytes(self, idx: Int) -> Int:
        return self.panel_rows(idx) * self.src_row_bytes

    @always_inline
    def aux_ptr_for_panel(self, idx: Int) -> PtrU8:
        if self.aux_row_block == 0:
            return self.aux_base
        var first_row = idx * self.rows_per_panel
        var aux_row = first_row // self.aux_row_block
        return self.aux_base + aux_row * self.aux_row_stride


@fieldwise_init
struct SlotState(Copyable, Movable, ImplicitlyCopyable):
    var stage: Int
    var task_idx: Int
    var panel_idx: Int
    var writes_remaining: Int


@always_inline
def encode_id(slot_idx: Int, kind: Int) -> Int:
    return slot_idx * (MAX_WRITES_PER_PANEL + 1) + kind


@always_inline
def decode_id(id: Int) -> Tuple[Int, Int]:
    var k = id % (MAX_WRITES_PER_PANEL + 1)
    var s = id // (MAX_WRITES_PER_PANEL + 1)
    return (s, k)


def per_row_writes[FT: SlotLike](
    out_base_weight: Int, out_base_scale: Int, out_base_cs: Int,
    rows_per_panel: Int,
) -> Tuple[Int, WriteSpec, WriteSpec, WriteSpec]:
    comptime SRC_BYTES = size_of[Scalar[FT.ENCODING.DTYPE]]()
    comptime COLS = FT.SHAPE.GLOBAL_M
    comptime PR = panel_rows_for[FT.SHAPE.GLOBAL_N]()
    comptime QT = FT.QUANT[PerRowQuant]
    comptime FW = QT.fwht_block
    comptime CS_BPR = (
        0 if QT.colsum.isa[NoColsum]()
        else (4 if QT.colsum.isa[PerRowCs]() else (COLS // FW) * 4)
    )
    comptime SRC_END = align_up(PR * COLS * SRC_BYTES, 64)
    comptime WORK_END = SRC_END + align_up(PR * COLS * 4, 64)
    comptime QI_END = WORK_END + align_up(PR * COLS, 64)
    comptime SCALES_END = QI_END + align_up(PR * 4, 64)
    var w0 = WriteSpec(
        stream_base_offset=out_base_weight,
        slot_buf_offset=WORK_END,
        bytes_per_panel_row=COLS,
        full_panel_bytes=PR * COLS,
    )
    var w1 = WriteSpec(
        stream_base_offset=out_base_scale,
        slot_buf_offset=QI_END,
        bytes_per_panel_row=4,
        full_panel_bytes=PR * 4,
    )
    var w2 = WriteSpec(
        stream_base_offset=out_base_cs,
        slot_buf_offset=SCALES_END,
        bytes_per_panel_row=CS_BPR,
        full_panel_bytes=PR * CS_BPR,
    )
    var n = 3 if CS_BPR > 0 else 2
    return (n, w0, w1, w2)


def per_block_writes[FT: SlotLike](
    out_base_weight: Int, out_base_scale: Int, out_base_cs: Int,
    rows_per_panel: Int,
) -> Tuple[Int, WriteSpec, WriteSpec, WriteSpec]:
    comptime SRC_BYTES = size_of[Scalar[FT.ENCODING.DTYPE]]()
    comptime COLS = FT.SHAPE.GLOBAL_M
    comptime PR = panel_rows_for[FT.SHAPE.GLOBAL_N]()
    comptime QT = FT.QUANT[PerBlockQuant]
    comptime NB = COLS // QT.fwht_block
    comptime CS_BPR = 0 if QT.colsum.isa[NoColsum]() else NB * 4
    comptime SRC_END = align_up(PR * COLS * SRC_BYTES, 64)
    comptime WORK_END = SRC_END + align_up(PR * COLS * 4, 64)
    comptime QI_END = WORK_END + align_up(PR * COLS, 64)
    comptime SCALES_END = QI_END + align_up(PR * NB * 4, 64)
    var w0 = WriteSpec(
        stream_base_offset=out_base_weight,
        slot_buf_offset=WORK_END,
        bytes_per_panel_row=COLS,
        full_panel_bytes=PR * COLS,
    )
    var w1 = WriteSpec(
        stream_base_offset=out_base_scale,
        slot_buf_offset=QI_END,
        bytes_per_panel_row=NB * 4,
        full_panel_bytes=PR * NB * 4,
    )
    var w2 = WriteSpec(
        stream_base_offset=out_base_cs,
        slot_buf_offset=SCALES_END,
        bytes_per_panel_row=CS_BPR,
        full_panel_bytes=PR * CS_BPR,
    )
    var n = 3 if CS_BPR > 0 else 2
    return (n, w0, w1, w2)


def router_writes[FT: SlotLike](
    out_base_centered: Int, out_base_gauge: Int,
) -> Tuple[Int, WriteSpec, WriteSpec, WriteSpec]:
    comptime SRC_BYTES = size_of[Scalar[FT.ENCODING.DTYPE]]()
    comptime ROWS = FT.SHAPE.GLOBAL_N
    comptime COLS = FT.SHAPE.GLOBAL_M
    comptime SRC_END = align_up(ROWS * COLS * SRC_BYTES, 64)
    comptime GAUGE_END = SRC_END + align_up(COLS * 4, 64)
    comptime CENTERED_END = GAUGE_END + align_up(ROWS * COLS * 2, 64)
    var w0 = WriteSpec(
        stream_base_offset=out_base_centered,
        slot_buf_offset=GAUGE_END,
        bytes_per_panel_row=COLS * 2,
        full_panel_bytes=ROWS * COLS * 2,
    )
    var w1 = WriteSpec(
        stream_base_offset=out_base_gauge,
        slot_buf_offset=CENTERED_END,
        bytes_per_panel_row=COLS * 2,
        full_panel_bytes=COLS * 2,
    )
    var dummy = WriteSpec(0, 0, 0, 0)
    return (2, w0, w1, dummy)


def passthrough_writes(out_base: Int) -> Tuple[Int, WriteSpec, WriteSpec, WriteSpec]:
    var w0 = WriteSpec(
        stream_base_offset=out_base,
        slot_buf_offset=0,
        bytes_per_panel_row=1,
        full_panel_bytes=DEFAULT_COPY_CHUNK,
    )
    var dummy = WriteSpec(0, 0, 0, 0)
    return (1, w0, dummy, dummy)


def load_gamma_as[gdt: DType, qd: Int](
    mut ring: IoRing[qd],
    loc: LocatedTensor,
    name: String,
    cols: Int,
    absorbed: Bool,
    mut arena: NumaArena,
) -> Optional[PtrF32]:
    var expected_bytes = cols * size_of[Scalar[gdt]]()
    if loc.byte_size != expected_bytes:
        print(t"quant: gamma byte-size mismatch for {name}: expected {expected_bytes} got {loc.byte_size}")
        return None
    var raw_opt = arena.alloc[Scalar[gdt]](cols)
    if not raw_opt:
        return None
    var raw = raw_opt.value()
    var f32_opt = arena.alloc[Float32](cols)
    if not f32_opt:
        return None
    var f32 = f32_opt.value()
    try:
        _ = ring.submit_one(ReadOp(
            file_idx=loc.shard, offset=loc.data_start, length=loc.byte_size,
            dest=raw.bitcast[UInt8](), id=encode_id(0, 0)))
        var c = ring.drain_one()
        if Int(c.result) != loc.byte_size:
            return None
    except err:
        print(t"quant: gamma read failed: {err}")
        return None

    var k = 0
    while k + W <= cols:
        (f32 + k).store((raw + k).load[width=W]().cast[DType.float32]())
        k += W
    while k < cols:
        (f32 + k).store((raw + k).load[width=1]().cast[DType.float32]())
        k += 1
    if not absorbed:
        gamma_sqrt_abs_in_place(f32, cols)
    return f32


def ensure_gamma[qd: Int](
    mut ring: IoRing[qd],
    headers: List[SafetensorsHeader],
    mut gamma_cache: Dict[String, PtrF32],
    name: String,
    expected_cols: Int,
    absorbed: Bool,
    mut arena: NumaArena,
) -> Optional[PtrF32]:
    """Load and prep a γ vector. Gamma dtype is independent of weight dtype."""
    var key = name + (":a:" if absorbed else ":s:") + String(expected_cols)
    var existing = gamma_cache.get(key)
    if existing:
        return existing.value()
    var loc_opt = find_tensor(name, headers)
    if not loc_opt:
        print(t"quant: gamma missing: {name}")
        return None
    var loc = loc_opt.take()
    var cols = 1
    for d in loc.shape:
        cols *= d
    if cols != expected_cols:
        print(t"quant: gamma shape mismatch for {name}: expected {expected_cols} values got {cols}")
        return None

    var out = Optional[PtrF32]()
    if loc.dtype == DType.bfloat16:
        out = load_gamma_as[DType.bfloat16, qd](
            ring, loc, name, cols, absorbed, arena)
    elif loc.dtype == DType.float32:
        out = load_gamma_as[DType.float32, qd](
            ring, loc, name, cols, absorbed, arena)
    elif loc.dtype == DType.float16:
        out = load_gamma_as[DType.float16, qd](
            ring, loc, name, cols, absorbed, arena)
    elif loc.dtype == DType.float64:
        out = load_gamma_as[DType.float64, qd](
            ring, loc, name, cols, absorbed, arena)
    else:
        print(t"quant: unsupported gamma dtype for {name}: {loc.dtype}")
        return None
    if not out:
        return None
    gamma_cache[key^] = out.value()
    return out


def ensure_aux[qd: Int](
    mut ring: IoRing[qd],
    headers: List[SafetensorsHeader],
    weight_name: String, aux_suffix: StaticString, expected_aux_dtype: DType,
    expected_bytes: Int,
    mut arena: NumaArena,
) -> Optional[PtrU8]:
    """Synchronously load an aux companion (e.g. FP8 _scale_inv) into the
    worker's arena. Returns the base pointer; one allocation per task."""
    var aux_name = weight_name + String(aux_suffix)
    var loc_opt = find_tensor(aux_name, headers)
    if not loc_opt:
        print(t"quant: missing aux companion {aux_name}")
        return None
    var loc = loc_opt.take()
    if loc.dtype != expected_aux_dtype:
        print(t"quant: aux dtype mismatch for {aux_name}")
        return None
    if loc.byte_size != expected_bytes:
        print(t"quant: aux size mismatch for {aux_name}: expected {expected_bytes} got {loc.byte_size}")
        return None
    var buf_opt = arena.alloc[UInt8](loc.byte_size)
    if not buf_opt:
        return None
    var buf = buf_opt.value()
    try:
        _ = ring.submit_one(ReadOp(
            file_idx=loc.shard, offset=loc.data_start, length=loc.byte_size,
            dest=buf, id=encode_id(0, 0)))
        var c = ring.drain_one()
        if Int(c.result) != loc.byte_size:
            return None
    except err:
        print(t"quant: aux read failed: {err}")
        return None
    return buf


def write_bias_f32_sync[qd: Int](
    mut ring: IoRing[qd], output_file_idx: Int,
    headers: List[SafetensorsHeader],
    bias_name: String, dest_offset: Int, rows: Int,
    mut arena: NumaArena,
) -> Bool:
    """Read the source bias tensor (bf16 or f32), promote to f32, write
    rows*4 bytes to the output. Synchronous, performed during emit_tasks
    for RouterCenter recipes."""
    var loc_opt = find_tensor(bias_name, headers)
    if not loc_opt:
        print(t"quant: missing router bias {bias_name}")
        return False
    var loc = loc_opt.take()
    if loc.dtype != DType.bfloat16 and loc.dtype != DType.float32:
        print(t"quant: router bias {bias_name} must be BF16 or F32, got {loc.dtype}")
        return False
    if loc.dtype == DType.bfloat16 and loc.byte_size != rows * 2:
        print(t"quant: router bias {bias_name} bf16 size mismatch")
        return False
    if loc.dtype == DType.float32 and loc.byte_size != rows * 4:
        print(t"quant: router bias {bias_name} f32 size mismatch")
        return False

    var f32_opt = arena.alloc[Float32](rows)
    if not f32_opt:
        return False
    var f32 = f32_opt.value()
    if loc.dtype == DType.float32:
        try:
            _ = ring.submit_one(ReadOp(
                file_idx=loc.shard, offset=loc.data_start, length=rows * 4,
                dest=f32.bitcast[UInt8](), id=encode_id(0, 0)))
            var c = ring.drain_one()
            if Int(c.result) != rows * 4:
                return False
        except err:
            print(t"quant: router bias read failed: {err}")
            return False
    else:
        var bf16_opt = arena.alloc[Scalar[DType.bfloat16]](rows)
        if not bf16_opt:
            return False
        var bf16 = bf16_opt.value()
        try:
            _ = ring.submit_one(ReadOp(
                file_idx=loc.shard, offset=loc.data_start, length=rows * 2,
                dest=bf16.bitcast[UInt8](), id=encode_id(0, 0)))
            var c = ring.drain_one()
            if Int(c.result) != rows * 2:
                return False
        except err:
            print(t"quant: router bias read failed: {err}")
            return False
        var k = 0
        while k + W <= rows:
            (f32 + k).store((bf16 + k).load[width=W]().cast[DType.float32]())
            k += W
        while k < rows:
            (f32 + k).store(
                (bf16 + k).load[width=1]().cast[DType.float32]())
            k += 1

    try:
        _ = ring.submit_one(WriteOp(
            file_idx=output_file_idx, offset=dest_offset, length=rows * 4,
            src=f32.bitcast[UInt8](), id=encode_id(0, 0)))
        var c = ring.drain_one()
        if Int(c.result) != rows * 4:
            return False
    except err:
        print(t"quant: router bias write failed: {err}")
        return False
    return True


def emit_tasks[T: AnyType, qd: Int](
    prefix: String,
    headers: List[SafetensorsHeader],
    rank: Int, tp: Int,
    output_file_idx: Int, data_start: Int,
    mut ring: IoRing[qd],
    mut arena: NumaArena,
    mut gamma_cache: Dict[String, PtrF32],
    mut tasks: List[TaskDesc],
    mut global_slot_idx: Int,
    mut out_off: Int,
) -> Bool:
    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, SlotLike):
            comptime if FT.NAME != StaticString(""):
                var full = prefix + String(FT.NAME)
                var loc_opt = find_tensor(full, headers)
                if not loc_opt:
                    print(t"quant: missing source {full}")
                    return False
                var loc = loc_opt.take()
                comptime ROWS = FT.SHAPE.GLOBAL_N
                comptime COLS = FT.SHAPE.GLOBAL_M
                comptime SRC = FT.ENCODING.DTYPE
                comptime SRC_BYTES = size_of[Scalar[SRC]]()
                comptime QV = FT.QUANT

                comptime if QV.isa[PerRowQuant]():
                    comptime QT = QV[PerRowQuant]
                    comptime FW = QT.fwht_block
                    comptime CS_BLOCKS = (
                        0 if QT.colsum.isa[NoColsum]()
                        else (1 if QT.colsum.isa[PerRowCs]() else COLS // FW)
                    )
                    var weight_off = data_start + out_off
                    var scale_off = weight_off + ROWS * COLS
                    var cs_off = scale_off + ROWS * 4
                    var advance = ROWS * COLS + ROWS * 4
                    if CS_BLOCKS > 0:
                        advance += ROWS * CS_BLOCKS * 4
                    if global_slot_idx % tp == rank:
                        var gptr = PtrF32.unsafe_dangling()
                        var has_g = False
                        comptime if QT.gamma.isa[SplitGamma]() or QT.gamma.isa[AbsorbedGamma]():
                            comptime gname = (
                                QT.gamma[SplitGamma].name if QT.gamma.isa[SplitGamma]()
                                else QT.gamma[AbsorbedGamma].name
                            )
                            comptime absorbed = QT.gamma.isa[AbsorbedGamma]()
                            var gopt = ensure_gamma[qd](
                                ring, headers, gamma_cache,
                                String(gname), COLS, absorbed, arena)
                            if not gopt:
                                return False
                            gptr = gopt.value()
                            has_g = True
                        var aux_base = PtrU8.unsafe_dangling()
                        var aux_row_block = 0
                        var aux_row_stride = 0
                        comptime if SRC == DType.float8_e4m3fn:
                            var aopt = ensure_aux[qd](
                                ring, headers, full,
                                source_aux_suffix[SRC](), DType.float32,
                                source_aux_bytes[SRC](ROWS, COLS), arena)
                            if not aopt:
                                return False
                            aux_base = aopt.value()
                            aux_row_block = source_aux_row_block[SRC]()
                            aux_row_stride = (COLS // aux_row_block) * 4
                        var pr = panel_rows_for[ROWS]()
                        var nps = num_panels_for[ROWS]()
                        var lpr = last_panel_rows_for[ROWS]()
                        var wt = per_row_writes[FT](
                            weight_off, scale_off, cs_off, pr)
                        tasks.append(TaskDesc(
                            name=full,
                            src_shard=loc.shard,
                            src_data_start=loc.data_start,
                            src_row_bytes=COLS * SRC_BYTES,
                            rows_per_panel=pr,
                            last_panel_rows=lpr,
                            num_panels=nps,
                            panels_submitted=0,
                            panels_completed=0,
                            n_writes=wt[0],
                            write0=wt[1], write1=wt[2], write2=wt[3],
                            gamma_ptr=gptr, has_gamma=has_g,
                            aux_base=aux_base,
                            aux_row_block=aux_row_block,
                            aux_row_stride=aux_row_stride,
                            math_fn=math_body[FT],
                        ))
                    out_off += advance
                    global_slot_idx += 1

                comptime if QV.isa[PerBlockQuant]():
                    comptime QT = QV[PerBlockQuant]
                    comptime NB = COLS // QT.fwht_block
                    comptime CS_BLOCKS = 0 if QT.colsum.isa[NoColsum]() else NB
                    var weight_off = data_start + out_off
                    var scale_off = weight_off + ROWS * COLS
                    var cs_off = scale_off + ROWS * NB * 4
                    var advance = ROWS * COLS + ROWS * NB * 4
                    if CS_BLOCKS > 0:
                        advance += ROWS * NB * 4
                    if global_slot_idx % tp == rank:
                        var gptr = PtrF32.unsafe_dangling()
                        var has_g = False
                        comptime if QT.gamma.isa[SplitGamma]() or QT.gamma.isa[AbsorbedGamma]():
                            comptime gname = (
                                QT.gamma[SplitGamma].name if QT.gamma.isa[SplitGamma]()
                                else QT.gamma[AbsorbedGamma].name
                            )
                            comptime absorbed = QT.gamma.isa[AbsorbedGamma]()
                            var gopt = ensure_gamma[qd](
                                ring, headers, gamma_cache,
                                String(gname), COLS, absorbed, arena)
                            if not gopt:
                                return False
                            gptr = gopt.value()
                            has_g = True
                        var aux_base = PtrU8.unsafe_dangling()
                        var aux_row_block = 0
                        var aux_row_stride = 0
                        comptime if SRC == DType.float8_e4m3fn:
                            var aopt = ensure_aux[qd](
                                ring, headers, full,
                                source_aux_suffix[SRC](), DType.float32,
                                source_aux_bytes[SRC](ROWS, COLS), arena)
                            if not aopt:
                                return False
                            aux_base = aopt.value()
                            aux_row_block = source_aux_row_block[SRC]()
                            aux_row_stride = (COLS // aux_row_block) * 4
                        var pr = panel_rows_for[ROWS]()
                        var nps = num_panels_for[ROWS]()
                        var lpr = last_panel_rows_for[ROWS]()
                        var wt = per_block_writes[FT](
                            weight_off, scale_off, cs_off, pr)
                        tasks.append(TaskDesc(
                            name=full,
                            src_shard=loc.shard,
                            src_data_start=loc.data_start,
                            src_row_bytes=COLS * SRC_BYTES,
                            rows_per_panel=pr,
                            last_panel_rows=lpr,
                            num_panels=nps,
                            panels_submitted=0,
                            panels_completed=0,
                            n_writes=wt[0],
                            write0=wt[1], write1=wt[2], write2=wt[3],
                            gamma_ptr=gptr, has_gamma=has_g,
                            aux_base=aux_base,
                            aux_row_block=aux_row_block,
                            aux_row_stride=aux_row_stride,
                            math_fn=math_body[FT],
                        ))
                    out_off += advance
                    global_slot_idx += 1

                comptime if QV.isa[RouterCenter]():
                    comptime QT = QV[RouterCenter]
                    var centered_off = data_start + out_off
                    var gauge_off = centered_off + ROWS * COLS * 2
                    var bias_off = gauge_off + COLS * 2
                    var advance = ROWS * COLS * 2 + COLS * 2
                    comptime has_bias = QT.bias_name != StaticString("")
                    comptime if has_bias:
                        advance += ROWS * 4
                    if global_slot_idx % tp == rank:
                        var wt = router_writes[FT](centered_off, gauge_off)
                        tasks.append(TaskDesc(
                            name=full,
                            src_shard=loc.shard,
                            src_data_start=loc.data_start,
                            src_row_bytes=COLS * SRC_BYTES,
                            rows_per_panel=ROWS,
                            last_panel_rows=ROWS,
                            num_panels=1,
                            panels_submitted=0,
                            panels_completed=0,
                            n_writes=wt[0],
                            write0=wt[1], write1=wt[2], write2=wt[3],
                            gamma_ptr=PtrF32.unsafe_dangling(),
                            has_gamma=False,
                            aux_base=PtrU8.unsafe_dangling(),
                            aux_row_block=0,
                            aux_row_stride=0,
                            math_fn=math_body[FT],
                        ))
                        comptime if has_bias:
                            if not write_bias_f32_sync[qd](
                                    ring, output_file_idx, headers,
                                    String(QT.bias_name), bias_off, ROWS, arena):
                                return False
                    out_off += advance
                    global_slot_idx += 1

                comptime if QV.isa[Passthrough]():
                    if global_slot_idx % tp == rank:
                        var wt = passthrough_writes(data_start + out_off)
                        var nps = (loc.byte_size + DEFAULT_COPY_CHUNK - 1) // DEFAULT_COPY_CHUNK
                        var lpr = loc.byte_size - (nps - 1) * DEFAULT_COPY_CHUNK
                        tasks.append(TaskDesc(
                            name=full,
                            src_shard=loc.shard,
                            src_data_start=loc.data_start,
                            src_row_bytes=1,
                            rows_per_panel=DEFAULT_COPY_CHUNK,
                            last_panel_rows=lpr,
                            num_panels=nps,
                            panels_submitted=0,
                            panels_completed=0,
                            n_writes=wt[0],
                            write0=wt[1], write1=wt[2], write2=wt[3],
                            gamma_ptr=PtrF32.unsafe_dangling(),
                            has_gamma=False,
                            aux_base=PtrU8.unsafe_dangling(),
                            aux_row_block=0,
                            aux_row_stride=0,
                            math_fn=math_body[FT],
                        ))
                    out_off += loc.byte_size
                    global_slot_idx += 1
        comptime if conforms_to(FT, SlotGroup):
            if not emit_tasks[FT, qd](
                prefix, headers, rank, tp,
                output_file_idx, data_start, ring, arena,
                gamma_cache, tasks, global_slot_idx, out_off):
                return False
    return True


def write_spec_at(task: TaskDesc, k: Int) -> WriteSpec:
    if k == 0: return task.write0
    if k == 1: return task.write1
    return task.write2


def run_scheduler[qd: Int](
    mut tasks: List[TaskDesc],
    mut ring: IoRing[qd],
    output_file_idx: Int,
    slot_bases: List[PtrU8],
    in_flight: Int,
) -> Bool:
    var slot_states = List[SlotState](
        length=in_flight,
        fill=SlotState(SLOT_FREE, -1, 0, 0))
    var next_task_for_read = 0
    var num_tasks = len(tasks)
    var total_panels = 0
    for i in range(num_tasks):
        total_panels += tasks[i].num_panels
    var panels_completed_total = 0

    @parameter
    def visit(c: Completion):
        var dec = decode_id(c.id)
        var slot_idx = dec[0]
        var kind = dec[1]
        if kind == 0:
            slot_states[slot_idx].stage = SLOT_MATH
        else:
            slot_states[slot_idx].writes_remaining -= 1
            if slot_states[slot_idx].writes_remaining == 0:
                slot_states[slot_idx].stage = SLOT_FREE
                tasks[slot_states[slot_idx].task_idx].panels_completed += 1
                panels_completed_total += 1

    while panels_completed_total < total_panels:
        while ring.sq_free() > 0:
            var free_slot = -1
            for s in range(in_flight):
                if slot_states[s].stage == SLOT_FREE:
                    free_slot = s
                    break
            if free_slot < 0:
                break
            var picked = -1
            for offset in range(num_tasks):
                var t = (next_task_for_read + offset) % num_tasks
                if tasks[t].panels_submitted < tasks[t].num_panels:
                    picked = t
                    break
            if picked < 0:
                break
            next_task_for_read = (picked + 1) % num_tasks
            ref task = tasks[picked]
            var panel_idx = task.panels_submitted
            try:
                _ = ring.submit_one(ReadOp(
                    file_idx=task.src_shard,
                    offset=task.src_offset(panel_idx),
                    length=task.src_bytes(panel_idx),
                    dest=slot_bases[free_slot],
                    id=encode_id(free_slot, 0)))
            except err:
                print(t"quant: read submit failed: {err}")
                return False
            slot_states[free_slot].stage = SLOT_READING
            slot_states[free_slot].task_idx = picked
            slot_states[free_slot].panel_idx = panel_idx
            task.panels_submitted += 1

        try:
            _ = ring.drain[visit](1)
        except err:
            print(t"quant: drain failed: {err}")
            return False

        for s in range(in_flight):
            if slot_states[s].stage != SLOT_MATH:
                continue
            var ti = slot_states[s].task_idx
            var pi = slot_states[s].panel_idx
            ref task = tasks[ti]
            var pr = task.panel_rows(pi)
            task.math_fn(
                slot_bases[s], pr, task.gamma_ptr,
                task.aux_ptr_for_panel(pi))
            slot_states[s].writes_remaining = task.n_writes
            for k in range(task.n_writes):
                var spec = write_spec_at(task, k)
                try:
                    _ = ring.submit_one(WriteOp(
                        file_idx=output_file_idx,
                        offset=spec.stream_base_offset
                            + pi * spec.full_panel_bytes,
                        length=pr * spec.bytes_per_panel_row
                            if task.src_row_bytes != 1
                            else task.panel_rows(pi),
                        src=slot_bases[s] + spec.slot_buf_offset
                            if task.src_row_bytes != 1
                            else slot_bases[s],
                        id=encode_id(s, k + 1)))
                except err:
                    print(t"quant: write submit failed: {err}")
                    return False
            slot_states[s].stage = SLOT_WRITING

    return True


@fieldwise_init
struct QuantWorker[T: AnyType, qd: Int](BurstKernel):
    var rank: Int
    var tp: Int
    var node: Int
    var arena_size: Int
    var in_flight: Int
    var fds: Span[Int32, MutAnyOrigin]
    var output_file_idx: Int
    var data_start: Int
    var prefix: StaticString
    var headers: Span[SafetensorsHeader, MutAnyOrigin]
    var off_in: Int
    var off_out: UnsafePointer[Int, MutAnyOrigin]
    var ok_out: UnsafePointer[Int32, MutAnyOrigin]

    def execute(mut self):
        var arena = NumaArena[](self.node, self.arena_size)
        if not arena:
            print(t"quant: rank {self.rank} arena alloc failed")
            self.ok_out[] = 0
            return

        var ring = IoRing[Self.qd]()
        if not ring:
            print(t"quant: rank {self.rank} ring init failed")
            self.ok_out[] = 0
            return
        try:
            _ = ring.register_fds(self.fds)
        except err:
            print(t"quant: rank {self.rank} register_fds failed: {err}")
            self.ok_out[] = 0
            return

        var slot_size = 0
        var slot_idx = 0
        slot_size = max_slot_bytes_for_rank[Self.T](
            self.rank, self.tp, slot_idx, slot_size)
        if slot_size == 0:
            self.off_out[] = self.off_in
            self.ok_out[] = 1
            return

        var slot_bases = List[PtrU8](capacity=self.in_flight)
        for _ in range(self.in_flight):
            var opt = arena.alloc[UInt8](slot_size)
            if not opt:
                print(t"quant: rank {self.rank} slot alloc failed")
                self.ok_out[] = 0
                return
            slot_bases.append(opt.value())

        var headers_local = List[SafetensorsHeader]()
        for i in range(len(self.headers)):
            var h = self.headers[i].copy()
            headers_local.append(h^)

        var gamma_cache = Dict[String, PtrF32]()
        var tasks = List[TaskDesc]()
        var global_slot_idx = 0
        var out_off = self.off_in
        if not emit_tasks[Self.T, Self.qd](
            String(self.prefix), headers_local,
            self.rank, self.tp,
            self.output_file_idx, self.data_start,
            ring, arena, gamma_cache, tasks,
            global_slot_idx, out_off):
            self.ok_out[] = 0
            return
        self.off_out[] = out_off

        var ok = run_scheduler[Self.qd](
            tasks, ring, self.output_file_idx, slot_bases, self.in_flight)
        self.ok_out[] = 1 if ok else 0
        var n_tasks = len(tasks)
        print(t"quant: rank {self.rank} node {self.node} tasks {n_tasks} ok={ok}")


def make_worker[T: AnyType, qd: Int](
    rank: Int, tp: Int, node: Int, arena_size: Int, in_flight: Int,
    fds: Span[Int32, _], output_file_idx: Int, data_start: Int,
    prefix: StaticString,
    headers: Span[SafetensorsHeader, _],
    off_in: Int,
    off_out: UnsafePointer[Int, MutAnyOrigin],
    ok_out: UnsafePointer[Int32, MutAnyOrigin],
) -> QuantWorker[T, qd]:
    var fds_wild = UnsafePointer[Int32, MutAnyOrigin](
        unsafe_from_address=Int(fds.unsafe_ptr()))
    var headers_wild = UnsafePointer[SafetensorsHeader, MutAnyOrigin](
        unsafe_from_address=Int(headers.unsafe_ptr()))
    return QuantWorker[T, qd](
        rank=rank, tp=tp, node=node,
        arena_size=arena_size, in_flight=in_flight,
        fds=Span[Int32, MutAnyOrigin](ptr=fds_wild, length=len(fds)),
        output_file_idx=output_file_idx, data_start=data_start,
        prefix=prefix,
        headers=Span[SafetensorsHeader, MutAnyOrigin](
            ptr=headers_wild, length=len(headers)),
        off_in=off_in, off_out=off_out, ok_out=ok_out,
    )


def estimate_arena_size[T: AnyType](
    rank: Int, tp: Int, in_flight: Int,
) -> Int:
    var slot_idx = 0
    var max_slot = max_slot_bytes_for_rank[T](rank, tp, slot_idx, 0)
    # extra room for gamma cache + aux companions + bias scratch + slop
    return in_flight * max_slot + 64 * 1024 * 1024


def run_quantizer_template[
    LayerT: AnyType,
    qd: Int = DEFAULT_QUEUE_DEPTH,
    mask_size: Int = DEFAULT_MASK_SIZE,
](
    prefix: StaticString,
    source_paths: List[Path],
    output_fd: Int32,
    output_file_idx: Int,
    data_start: Int,
    headers: List[SafetensorsHeader],
    topo: NumaTopology,
    fds: List[Int32],
    off_in: Int,
    in_flight: Int = DEFAULT_IN_FLIGHT,
) -> Optional[Int]:
    var tp = len(topo)

    var off_results = List[Int](length=tp, fill=off_in)
    var ok_results = List[Int32](length=tp, fill=Int32(0))
    var off_base = UnsafePointer[Int, MutAnyOrigin](
        unsafe_from_address=Int(off_results.unsafe_ptr()))
    var ok_base = UnsafePointer[Int32, MutAnyOrigin](
        unsafe_from_address=Int(ok_results.unsafe_ptr()))

    var pools = List[BurstPool[mask_size]](capacity=tp)
    for r in range(tp):
        var mask = topo.mask[mask_size](r)
        pools.append(BurstPool[mask_size](
            capacity=1, cpu_mask=mask, numa_node=topo.node(r)))
        if not pools[r]:
            print(t"quant: pool setup failed for rank {r}")
            return None

    var kernels = List[QuantWorker[LayerT, qd]](capacity=tp)
    for r in range(tp):
        var arena_size = estimate_arena_size[LayerT](r, tp, in_flight)
        kernels.append(make_worker[LayerT, qd](
            rank=r, tp=tp, node=topo.node(r),
            arena_size=arena_size, in_flight=in_flight,
            fds=Span(fds),
            output_file_idx=output_file_idx, data_start=data_start,
            prefix=prefix,
            headers=Span(headers),
            off_in=off_in,
            off_out=off_base + r, ok_out=ok_base + r,
        ))

    var pool_base = pools.unsafe_ptr()
    for r in range(tp):
        var span = Span[QuantWorker[LayerT, qd], MutAnyOrigin](
            ptr=UnsafePointer(to=kernels[r]), length=1)
        (pool_base + r)[].dispatch(span, 1)
    for r in range(tp):
        (pool_base + r)[].join()

    _ = kernels^

    var all_ok = True
    var off_max = off_in
    for r in range(tp):
        if ok_results[r] == 0:
            all_ok = False
        if off_results[r] > off_max:
            off_max = off_results[r]
    if not all_ok:
        return None
    return off_max


def write_header_sync[qd: Int](
    mut ring: IoRing[qd], header: HeaderBuffer, output_file_idx: Int,
) -> Bool:
    try:
        _ = ring.submit_one(WriteOp(
            file_idx=output_file_idx, offset=0, length=header.size,
            src=header.any_ptr(), id=0))
        var c = ring.drain_one()
        return Int(c.result) == header.size
    except err:
        print(t"quant: header write failed: {err}")
        return False
