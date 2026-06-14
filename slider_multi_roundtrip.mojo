from std.memory import Span
from std.pathlib import Path

from jsontools.parser import Parser
from safetensors.parser import parse_safetensors_header
from modeling.gemma4_common import Gemma4BaseConfig
from modeling.slider_pack import (
    write_pack, SliderConfig, SliderCalibration, parse_manifest,
)


comptime C = Gemma4BaseConfig
comptime BASE = "/tmp/sliders_multi/ocean"


def main():
    var hidden = C.HIDDEN
    var names = List[String]()
    names.append("openness")
    names.append("extraversion")
    names.append("neuroticism")
    var layers = List[Int]()
    layers.append(10)
    layers.append(12)
    layers.append(18)
    var amins = List[Float32]()
    amins.append(Float32(8))
    amins.append(Float32(16))
    amins.append(Float32(4))
    var amaxs = List[Float32]()
    amaxs.append(Float32(48))
    amaxs.append(Float32(64))
    amaxs.append(Float32(32))

    var n = len(names)
    var directions = List[BFloat16]()
    for s in range(n):
        for i in range(hidden):
            directions.append(BFloat16(Float32((i + s * 13) % 11) - Float32(5)))

    var configs = List[SliderConfig]()
    var cals = List[SliderCalibration]()
    for s in range(n):
        configs.append(SliderConfig(s, layers[s], amins[s], amaxs[s], True))
        cals.append(SliderCalibration(
            Float64(0.1 + s), Float64(3 + s), Float64(70 + s), 24, Float64(2)))

    if not write_pack(
            BASE, names, directions, configs, cals, hidden, C.NUM_LAYERS,
            "gemma-4-26B-A4B-it"):
        print("write_pack failed")
        return
    print(t"wrote {n}-slider pack")

    var ok = True
    try:
        var data = Path(BASE + ".json").read_bytes()
        var p = Parser(Span(data))
        var m = parse_manifest(p)
        print(t"manifest sliders: {len(m.sliders)}")
        if len(m.sliders) != n:
            ok = False
        for s in range(len(m.sliders)):
            ref e = m.sliders[s]
            print(t"  {e.trait_name}: layer {e.layer} [{e.alpha_min}, {e.alpha_max}]")
            if e.trait_name != names[s] or e.layer != layers[s]:
                ok = False
            if e.alpha_min != amins[s] or e.alpha_max != amaxs[s]:
                ok = False
    except e:
        print(t"manifest parse error: {e}")
        ok = False

    var hdr_opt = parse_safetensors_header(Path(BASE + ".safetensors"))
    if not hdr_opt:
        print("header read failed")
        return
    var hdr = hdr_opt.take()
    var total_mism = 0
    for s in range(n):
        var meta_opt = hdr.tensors.get(names[s] + ".vector")
        if not meta_opt:
            print(t"tensor {names[s]}.vector missing")
            ok = False
            continue
        ref meta = meta_opt.value()
        if meta.shape[0] != hidden or meta.dtype != DType.bfloat16:
            ok = False
        var nbytes = meta.end - meta.start
        var raw: List[Byte]
        try:
            with open(Path(BASE + ".safetensors"), "r") as f:
                _ = f.seek(UInt64(hdr.data_offset + meta.start), 0)
                raw = f.read_bytes(size=nbytes)
        except e:
            print(t"read failed: {e}")
            return
        var vp = raw.unsafe_ptr().bitcast[Scalar[DType.bfloat16]]()
        for i in range(hidden):
            if vp[i] != directions[s * hidden + i]:
                total_mism += 1
    print(t"vector byte mismatches across all sliders: {total_mism}")
    if total_mism != 0:
        ok = False

    if ok:
        print("MULTI ROUNDTRIP CHECK: PASS")
    else:
        print("MULTI ROUNDTRIP CHECK: FAIL")
