from std.sys import is_defined
from std.sys.info import num_physical_cores, num_logical_cores

from numa import NumaTopology


comptime COMPILE_TIME_DEGREE: Int = (
    8 if is_defined["TP8"]()
    else 4 if is_defined["TP4"]()
    else 2 if is_defined["TP2"]()
    else 1
)


comptime COMPILE_TIME_MAX_WORKERS: Int = (
    192 if is_defined["W192"]()
    else 96 if is_defined["W96"]()
    else 48 if is_defined["W48"]()
    else 24 if is_defined["W24"]()
    else 128
)


def main():
    var topo = NumaTopology()
    var detected_degree = len(topo)
    var detected_workers = topo.worker_count(0) if detected_degree > 0 else 0

    print("compile-time degree       =", COMPILE_TIME_DEGREE)
    print("compile-time max_workers  =", COMPILE_TIME_MAX_WORKERS)
    print("runtime detected nodes    =", detected_degree)
    print("runtime workers on rank 0 =", detected_workers)
    print("runtime physical cores    =", num_physical_cores())
    print("runtime logical cores     =", num_logical_cores())

    if detected_degree != COMPILE_TIME_DEGREE:
        print(
            "MISMATCH: binary specialized for tp=", COMPILE_TIME_DEGREE,
            "but host has tp=", detected_degree,
            "- rebuild with the matching -D flag (e.g. -D TP2 / -D TP4).",
        )
    else:
        print("OK: compile-time degree matches detected topology.")
