from std.pathlib import Path
from std.time import perf_counter_ns

from modeling.gemma_4_moe import Gemma4

comptime SOURCE = "checkpoints/gemma-4-26B-A4B"
comptime OUTPUT = "checkpoints/gemma-4-26B-A4B-bq/model.safetensors"


def main():
    var t0 = perf_counter_ns()
    var ok = Gemma4[degree=1].quantize(Path(SOURCE), Path(OUTPUT))
    var elapsed_s = (perf_counter_ns() - t0) / 1_000_000_000
    if ok:
        print(t"quantize ok in {elapsed_s} s")
    else:
        print(t"quantize failed after {elapsed_s} s")
