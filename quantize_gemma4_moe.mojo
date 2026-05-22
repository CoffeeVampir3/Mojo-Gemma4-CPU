from std.memory import Span
from std.pathlib import Path

from linux.io_uring import (
    IoRing, ReadMode, WriteMode, open_files_for_ring, close_fds,
)
from numa import NumaTopology
from modeling.gemma4_common import Gemma4BaseConfig, LAYER_SCHEDULE, LayerKind
from modeling.gemma_4_moe import (
    SlidingLayerRefs, FullLayerRefs, TailRefs,
)
from safetensors.parser import SafetensorsHeader
from quant.planning import (
    OutputEntry, discover_quant_shards, parse_source_headers,
    build_header, emit_quant_plan,
)
from quant.quantizer import run_quantizer_template, write_header_sync


comptime C = Gemma4BaseConfig
comptime SOURCE_DIR = "checkpoints/gemma-4-26B-A4B"
comptime OUTPUT_PATH = "checkpoints/gemma-4-26B-A4B.butterquant.safetensors"
comptime QD = 256
comptime MASK_SIZE = 128
comptime IN_FLIGHT = 4


def plan_gemma4_quant[degree: Int](
    headers: List[SafetensorsHeader],
    mut entries: List[OutputEntry],
) -> Optional[Int]:
    var off = 0
    for i in range(C.NUM_LAYERS):
        var entry = LAYER_SCHEDULE[i]
        var prefix = String(t"model.language_model.layers.{entry.idx}.")
        var next = Optional[Int]()
        if entry.kind == LayerKind.FULL:
            next = emit_quant_plan[FullLayerRefs[degree]](
                prefix, headers, entries, off)
        else:
            next = emit_quant_plan[SlidingLayerRefs[degree]](
                prefix, headers, entries, off)
        if not next:
            return None
        off = next.value()
    return emit_quant_plan[TailRefs[degree]]("", headers, entries, off)


def quantize_one[
    LayerT: AnyType, prefix: StaticString,
    qd: Int, mask_size: Int,
](
    source_paths: List[Path],
    output_fd: Int32,
    output_file_idx: Int,
    data_start: Int,
    headers: List[SafetensorsHeader],
    topo: NumaTopology,
    fds: List[Int32],
    off: Int,
    in_flight: Int,
) -> Optional[Int]:
    print(t"quant: {prefix}")
    return run_quantizer_template[LayerT, qd, mask_size](
        prefix, source_paths, output_fd, output_file_idx, data_start,
        headers, topo, fds, off, in_flight=in_flight)


def run_gemma4_quant_layers[
    degree: Int, qd: Int, mask_size: Int,
](
    source_paths: List[Path],
    output_fd: Int32,
    output_file_idx: Int,
    data_start: Int,
    headers: List[SafetensorsHeader],
    topo: NumaTopology,
    fds: List[Int32],
    off_in: Int,
    in_flight: Int,
) -> Optional[Int]:
    var off = off_in

    var next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.0.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.1.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.2.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.3.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.4.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        FullLayerRefs[degree], "model.language_model.layers.5.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.6.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.7.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.8.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.9.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.10.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        FullLayerRefs[degree], "model.language_model.layers.11.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.12.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.13.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.14.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.15.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.16.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        FullLayerRefs[degree], "model.language_model.layers.17.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.18.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.19.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.20.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.21.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.22.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        FullLayerRefs[degree], "model.language_model.layers.23.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.24.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.25.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.26.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.27.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        SlidingLayerRefs[degree], "model.language_model.layers.28.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    next = quantize_one[
        FullLayerRefs[degree], "model.language_model.layers.29.",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)
    if not next: return None
    off = next.value()

    return quantize_one[
        TailRefs[degree], "",
        qd, mask_size,
    ](source_paths, output_fd, output_file_idx, data_start,
      headers, topo, fds, off, in_flight)


def run_for_topology[qd: Int, mask_size: Int](
    source_paths: List[Path],
    output_fd: Int32,
    output_file_idx: Int,
    data_start: Int,
    headers: List[SafetensorsHeader],
    topo: NumaTopology,
    fds: List[Int32],
    in_flight: Int,
) -> Optional[Int]:
    var tp = len(topo)
    if tp == 1:
        return run_gemma4_quant_layers[1, qd, mask_size](
            source_paths, output_fd, output_file_idx, data_start,
            headers, topo, fds, 0, in_flight)
    if tp == 2:
        return run_gemma4_quant_layers[2, qd, mask_size](
            source_paths, output_fd, output_file_idx, data_start,
            headers, topo, fds, 0, in_flight)
    if tp == 4:
        return run_gemma4_quant_layers[4, qd, mask_size](
            source_paths, output_fd, output_file_idx, data_start,
            headers, topo, fds, 0, in_flight)
    if tp == 8:
        return run_gemma4_quant_layers[8, qd, mask_size](
            source_paths, output_fd, output_file_idx, data_start,
            headers, topo, fds, 0, in_flight)
    print(t"quant: unsupported topology degree {tp}")
    return None


def plan_for_topology(
    headers: List[SafetensorsHeader],
    mut entries: List[OutputEntry],
    topo: NumaTopology,
) -> Optional[Int]:
    var tp = len(topo)
    if tp == 1:
        return plan_gemma4_quant[1](headers, entries)
    if tp == 2:
        return plan_gemma4_quant[2](headers, entries)
    if tp == 4:
        return plan_gemma4_quant[4](headers, entries)
    if tp == 8:
        return plan_gemma4_quant[8](headers, entries)
    print(t"quant: unsupported topology degree {tp}")
    return None


def main():
    var topo = NumaTopology()
    var shards = discover_quant_shards(Path(SOURCE_DIR))
    if len(shards) == 0:
        print(t"quant: no safetensors shards under {SOURCE_DIR}")
        return

    var headers_opt = parse_source_headers(shards)
    if not headers_opt:
        return
    var headers = headers_opt.take()

    var entries = List[OutputEntry]()
    var total_opt = plan_for_topology(headers, entries, topo)
    if not total_opt:
        return
    var total_bytes = total_opt.value()
    var header = build_header(entries)
    print(t"quant: planned {len(entries)} entries, {total_bytes} payload bytes")

    var read_fds: List[Int32]
    try:
        read_fds = open_files_for_ring[ReadMode](Span(shards))
    except err:
        print(t"quant: open source failed: {err.error_message()}")
        return

    var output_paths = List[Path]()
    output_paths.append(Path(OUTPUT_PATH))
    var write_fds: List[Int32]
    try:
        write_fds = open_files_for_ring[WriteMode](Span(output_paths))
    except err:
        print(t"quant: open output failed: {err.error_message()}")
        close_fds(read_fds^)
        return

    var fds = List[Int32]()
    for fd in read_fds:
        fds.append(fd)
    for fd in write_fds:
        fds.append(fd)

    var output_file_idx = len(read_fds)
    var output_fd = write_fds[0]
    var ring = IoRing[QD]()
    if not ring:
        print("quant: header ring init failed")
        close_fds(fds^)
        return
    try:
        _ = ring.register_fds(Span(fds))
    except err:
        print(t"quant: header ring register_fds failed: {err.error_message()}")
        close_fds(fds^)
        return
    if not write_header_sync[QD](ring, header, output_file_idx):
        close_fds(fds^)
        return

    var final_off = run_for_topology[QD, MASK_SIZE](
        shards, output_fd, output_file_idx, header.size,
        headers, topo, fds, IN_FLIGHT)
    close_fds(fds^)
    if not final_off:
        print("quant: FAILED")
        return
    print(t"quant: wrote {final_off.value()} payload bytes to {OUTPUT_PATH}")
