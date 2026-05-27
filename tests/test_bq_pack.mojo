from std.collections import InlineArray
from std.memory import Span, UnsafePointer, alloc
from std.os import abort

from threading.threading_traits import BurstKernel, BurstThreadPool
from kernels.helpers import ArenaBases
from butterquant import (
    PackColsumTask, dispatch_pack_colsum,
)


comptime VNNI_N_STEP = 32
comptime VNNI_K_STEP = 64
comptime VNNI_TILE_N = 16


@fieldwise_init
struct TestPool(BurstThreadPool):
    var capacity: Int
    var timestamp: Int

    def get_capacity(self) -> Int:
        return self.capacity

    def dispatch[K: BurstKernel, origin: MutOrigin](
        mut self, kernels: Span[K, origin], num_jobs: Int):
        for i in range(num_jobs):
            var kernel = kernels[i]
            kernel.execute()
        self.timestamp += 1

    def join(mut self):
        pass

    def last_worker_timestamp(self) -> Int:
        return self.timestamp

    def wake(mut self):
        pass

    def sleep(mut self):
        pass


def check(ok: Bool, msg: String):
    if not ok:
        abort("FAIL: " + msg)


def src_byte(n: Int, k: Int, cols: Int) -> Int8:
    return Int8(((n * 7 + k * 3) % 251) - 125)


def ref_pack(
    src: UnsafePointer[Int8, MutAnyOrigin],
    dst: UnsafePointer[Int8, MutAnyOrigin],
    rows: Int, cols: Int,
):
    """Scalar VNNI tile pack for rows <= one n_block: each 32x64 tile is two
    16x64 dword-transposed halves laid out as [k/4][n][4]."""
    for n_begin in range(0, rows, VNNI_N_STEP):
        for k_begin in range(0, cols, VNNI_K_STEP):
            var tile_base = n_begin * cols + k_begin * VNNI_N_STEP
            for half in range(2):
                var base_n = n_begin + half * VNNI_TILE_N
                var half_base = tile_base + half * VNNI_TILE_N * VNNI_K_STEP
                for kg in range(VNNI_K_STEP // 4):
                    for n in range(VNNI_TILE_N):
                        for sub in range(4):
                            dst[half_base + (kg * VNNI_TILE_N + n) * 4 + sub] = (
                                src[(base_n + n) * cols + k_begin + kg * 4 + sub])


def run_case(rows: Int, cols: Int, block_cols: Int, colsum_row_major: Bool):
    var nb = cols // block_cols
    var w_bytes = rows * cols
    var cs_off = ((w_bytes + 63) // 64) * 64
    var arena = alloc[UInt8](cs_off + rows * nb * 4).as_any_origin()
    var wptr = arena.bitcast[Int8]()
    var csptr = (arena + cs_off).bitcast[Float32]()

    var srccopy = alloc[Int8](w_bytes).as_any_origin()
    for n in range(rows):
        for k in range(cols):
            var v = src_byte(n, k, cols)
            wptr[n * cols + k] = v
            srccopy[n * cols + k] = v

    var refpack = alloc[Int8](w_bytes).as_any_origin()
    ref_pack(srccopy, refpack, rows, cols)

    var pools = List[TestPool](capacity=1)
    pools.append(TestPool(1, 0))
    var bases = ArenaBases[1].uninitialized()
    bases[0] = Int(arena)
    var nodes = InlineArray[Int, 1](fill=0)

    var tasks = List[PackColsumTask]()
    tasks.append(PackColsumTask(
        weight_off=0, colsum_off=cs_off,
        rows=rows, cols=cols, block_cols=block_cols,
        colsum_row_major=colsum_row_major))

    dispatch_pack_colsum[1](pools, bases, nodes, tasks)

    var pack_ok = True
    for i in range(w_bytes):
        if wptr[i] != refpack[i]:
            pack_ok = False
    check(pack_ok, "packed bytes mismatch")

    var cs_ok = True
    for n in range(rows):
        for b in range(nb):
            var acc = 0
            for k in range(block_cols):
                acc += Int(srccopy[n * cols + b * block_cols + k])
            var idx = (n * nb + b) if colsum_row_major else (b * rows + n)
            if csptr[idx] != Float32(acc):
                cs_ok = False
    check(cs_ok, "colsum mismatch")

    arena.free()
    srccopy.free()
    refpack.free()


def main():
    run_case(64, 128, 128, True)
    run_case(64, 128, 64, False)
    run_case(96, 256, 64, False)
    print("bq pack dispatch tests passed")
