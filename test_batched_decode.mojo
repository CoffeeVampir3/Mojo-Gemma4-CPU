from std.collections import InlineArray
from std.memory import Span, UnsafePointer

from numa import NumaArena, NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch
from kernels.attention_ops import KVRun, KVRunTable, flash_partial_stride
from kernels.attention_dispatch_kernels import (
    dispatch_sliding_attention, dispatch_full_attention,
)
from kernels.helpers import Binding, RankView
from kernels.logsum_merge import MergeSegment
from kernels.profiling import Profiler


comptime ALIGNMENT = 64
comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]
comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime MAX_WORKERS = 128
comptime PASS_ATOL = Float32(5.0e-2)

# Sliding geometry: 2-page flip-flop ring, page_len = window.
comptime S_HEAD_DIM = 256
comptime S_NUM_Q = 4
comptime S_GQA = 2
comptime S_KV_STRIDE = 512          # 2 kv heads * 256
comptime S_WINDOW = 64
comptime S_PAGE = 64
comptime S_CACHE = 128              # two pages
comptime S_SLAB = 128              # cache rows reserved per sequence
comptime S_QSTRIDE = S_NUM_Q * S_HEAD_DIM
comptime S_PSTRIDE = flash_partial_stride(S_NUM_Q, S_HEAD_DIM)

# Full geometry: growing page table, single page per sequence (base_pos < page).
comptime F_HEAD_DIM = 256
comptime F_NUM_Q = 8               # global (replicated)
comptime F_GQA = 4
comptime F_KV_STRIDE = 512         # 2 kv heads * 256
comptime F_PAGE = 128
comptime F_SLAB = 128             # cache rows reserved per sequence
comptime F_QSTRIDE = F_NUM_Q * F_HEAD_DIM
comptime F_PSTRIDE = flash_partial_stride(F_NUM_Q, F_HEAD_DIM)


def arena_bases(
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
) -> List[Int]:
    var bases = List[Int](capacity=len(arenas))
    for r in range(len(arenas)):
        bases.append(Int(arenas[r].base.value()))
    return bases^


def arena_alloc_all[T: AnyType](
    mut arenas: List[NumaArena[alignment=ALIGNMENT]], count: Int,
) -> UnsafePointer[T, MutAnyOrigin]:
    var first = UnsafePointer[T, MutAnyOrigin].unsafe_dangling()
    for r in range(len(arenas)):
        var ptr = arenas[r].alloc[T](count)
        if not ptr:
            print("arena alloc failed for", count, "elements")
            return UnsafePointer[T, MutAnyOrigin].unsafe_dangling()
        if r == 0:
            first = ptr.value()
    return first


def fill_pattern(ptr: BF16Ptr, count: Int, seed: Int):
    for i in range(count):
        var v = ((i * 131 + seed * 17) % 251) - 125
        ptr[i] = BFloat16(Float32(v) * 0.008)


def fill_pattern_all[o: ImmutOrigin](
    ptrs: Binding[BFloat16, o], count: Int, seed: Int,
):
    for r in range(ptrs.degree()):
        fill_pattern(ptrs[r], count, seed)


def compare_all[o: ImmutOrigin](
    a: Binding[BFloat16, o], b: Binding[BFloat16, o], count: Int,
) -> Tuple[Float32, Int]:
    var max_abs = Float32(0)
    var mismatches = 0
    for r in range(a.degree()):
        var ap = a[r]
        var bp = b[r]
        for i in range(count):
            var av = ap[i].cast[DType.float32]()
            var bv = bp[i].cast[DType.float32]()
            var d = av - bv
            if d < 0:
                d = -d
            if d > max_abs:
                max_abs = d
            if d > PASS_ATOL:
                mismatches += 1
    return (max_abs, mismatches)


def make_positions() -> List[Int]:
    var pos = List[Int]()
    pos.append(10)   # short: valid_len < window
    pos.append(63)   # exactly window-1 (sliding valid_len == window)
    pos.append(64)   # crosses the sliding ring page seam
    pos.append(100)  # beyond window (sliding ring wrap); deep full context
    pos.append(1)    # tiny: full attention has zero-KV ranks at degree > 1
    return pos^


def run_sliding[P: BurstThreadPool, //](
    mut pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
    view: RankView,
) -> Int:
    print("\n=== sliding batched decode vs per-run reference ===")
    var positions = make_positions()
    var B = len(positions)

    var q_ptr = arena_alloc_all[BFloat16](arenas, B * S_QSTRIDE)
    var k_ptr = arena_alloc_all[BFloat16](arenas, B * S_SLAB * S_KV_STRIDE)
    var v_ptr = arena_alloc_all[BFloat16](arenas, B * S_SLAB * S_KV_STRIDE)
    var ref_ptr = arena_alloc_all[BFloat16](arenas, B * S_QSTRIDE)
    var bat_ptr = arena_alloc_all[BFloat16](arenas, B * S_QSTRIDE)
    var part_ptr = arena_alloc_all[Float32](arenas, MAX_WORKERS * S_PSTRIDE)

    var q = view.bind(q_ptr)
    var k = view.bind(k_ptr)
    var v = view.bind(v_ptr)
    var ref_out = view.bind(ref_ptr)
    var bat_out = view.bind(bat_ptr)
    var partials = view.bind(part_ptr)

    fill_pattern_all(q, B * S_QSTRIDE, 1)
    fill_pattern_all(k, B * S_SLAB * S_KV_STRIDE, 2)
    fill_pattern_all(v, B * S_SLAB * S_KV_STRIDE, 3)
    for r in range(len(arenas)):
        _ = arenas[r].prefault(0, arenas[r].used())

    var prof = Profiler[False]()

    # Reference: each run alone through the single-run (seq_len == 1) path.
    for i in range(B):
        var rt = KVRunTable()
        var run = KVRun(0, positions[i])
        run.base_rows.append(Int32(i * S_SLAB))
        run.base_rows.append(Int32(i * S_SLAB + S_PAGE))
        rt.runs.append(run^)
        var runs = UnsafePointer(to=rt)
        var q_i = view.bind(q_ptr + i * S_QSTRIDE)
        var out_i = view.bind(ref_ptr + i * S_QSTRIDE)
        dispatch_sliding_attention[
            head_dim=S_HEAD_DIM, max_q=S_NUM_Q, gqa_ratio=S_GQA,
            window=S_WINDOW, cache_size=S_CACHE, page_len=S_PAGE,
        ](q_i, k, v, out_i, partials, runs,
          S_NUM_Q, S_PSTRIDE, S_KV_STRIDE, 1, pools, prof)

    # Under test: all runs packed, the batched-split path.
    var bt = KVRunTable()
    for i in range(B):
        var run = KVRun(i, positions[i])
        run.base_rows.append(Int32(i * S_SLAB))
        run.base_rows.append(Int32(i * S_SLAB + S_PAGE))
        bt.runs.append(run^)
    var bruns = UnsafePointer(to=bt)
    dispatch_sliding_attention[
        head_dim=S_HEAD_DIM, max_q=S_NUM_Q, gqa_ratio=S_GQA,
        window=S_WINDOW, cache_size=S_CACHE, page_len=S_PAGE,
    ](q, k, v, bat_out, partials, bruns,
      S_NUM_Q, S_PSTRIDE, S_KV_STRIDE, B, pools, prof)

    var cmp = compare_all(ref_out, bat_out, B * S_QSTRIDE)
    print(t"  max abs diff {cmp[0]}   mismatches {cmp[1]}")
    return cmp[1]


def run_full[P: BurstThreadPool, //](
    mut pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
    view: RankView,
) -> Int:
    print("\n=== full batched decode vs per-run reference ===")
    var tp = len(pools)
    var local_num_q = F_NUM_Q // tp
    var out_stride = local_num_q * F_HEAD_DIM
    var positions = make_positions()
    var B = len(positions)

    var q_ptr = arena_alloc_all[BFloat16](arenas, B * F_QSTRIDE)
    var k_ptr = arena_alloc_all[BFloat16](arenas, B * F_SLAB * F_KV_STRIDE)
    var v_ptr = arena_alloc_all[BFloat16](arenas, B * F_SLAB * F_KV_STRIDE)
    var ref_ptr = arena_alloc_all[BFloat16](arenas, B * out_stride)
    var bat_ptr = arena_alloc_all[BFloat16](arenas, B * out_stride)
    var part_ptr = arena_alloc_all[Float32](arenas, MAX_WORKERS * F_PSTRIDE)
    var seg_ptr = arena_alloc_all[MergeSegment](arenas, MAX_WORKERS * tp)

    var q = view.bind(q_ptr)
    var k = view.bind(k_ptr)
    var v = view.bind(v_ptr)
    var ref_out = view.bind(ref_ptr)
    var bat_out = view.bind(bat_ptr)
    var partials = view.bind(part_ptr)
    var segs = view.bind(seg_ptr)

    fill_pattern_all(q, B * F_QSTRIDE, 4)
    fill_pattern_all(k, B * F_SLAB * F_KV_STRIDE, 5)
    fill_pattern_all(v, B * F_SLAB * F_KV_STRIDE, 6)
    for r in range(len(arenas)):
        _ = arenas[r].prefault(0, arenas[r].used())

    var prof = Profiler[False]()

    for i in range(B):
        var rt = KVRunTable()
        var run = KVRun(0, positions[i])
        run.base_rows.append(Int32(i * F_SLAB))
        rt.runs.append(run^)
        var runs = UnsafePointer(to=rt)
        var q_i = view.bind(q_ptr + i * F_QSTRIDE)
        var out_i = view.bind(ref_ptr + i * out_stride)
        dispatch_full_attention[
            head_dim=F_HEAD_DIM, num_q=F_NUM_Q, gqa_ratio=F_GQA,
            kv_stride=F_KV_STRIDE, partial_stride=F_PSTRIDE, page_len=F_PAGE,
        ](q_i, k, v, out_i, partials, segs, runs,
          local_num_q, 1, pools, prof)

    var bt = KVRunTable()
    for i in range(B):
        var run = KVRun(i, positions[i])
        run.base_rows.append(Int32(i * F_SLAB))
        bt.runs.append(run^)
    var bruns = UnsafePointer(to=bt)
    dispatch_full_attention[
        head_dim=F_HEAD_DIM, num_q=F_NUM_Q, gqa_ratio=F_GQA,
        kv_stride=F_KV_STRIDE, partial_stride=F_PSTRIDE, page_len=F_PAGE,
    ](q, k, v, bat_out, partials, segs, bruns,
      local_num_q, B, pools, prof)

    var cmp = compare_all(ref_out, bat_out, B * out_stride)
    print(t"  max abs diff {cmp[0]}   mismatches {cmp[1]}")
    return cmp[1]


def run_all[P: BurstThreadPool, //](
    mut pools: List[P],
    mut arenas: List[NumaArena[alignment=ALIGNMENT]],
):
    var tp = len(pools)
    var bases = arena_bases(arenas)
    var view = RankView(Span(bases))
    var cap = pools[0].get_capacity()
    var num_seqs = len(make_positions())
    print(t"degree {tp}, pool capacity {cap}, B {num_seqs}")

    var failures = 0
    failures += run_sliding(pools, arenas, view)
    failures += run_full(pools, arenas, view)

    if failures == 0:
        print("\nRESULT: PASS -- batched decode matches per-run reference")
    else:
        print(t"\nRESULT: FAIL -- {failures} element mismatches")


def main():
    var topo = NumaTopology()
    var tp = len(topo)
    print("Batched decode differential test")
    var iso = len(topo.isolated_cpus)
    print(t"{tp} NUMA node(s), {iso} isolated cpus")

    comptime ARENA_BYTES = 128 * 1024 * 1024
    var arenas = List[NumaArena[alignment=ALIGNMENT]](capacity=tp)
    for i in range(tp):
        arenas.append(NumaArena[alignment=ALIGNMENT](topo[i], ARENA_BYTES))
        if not arenas[i]:
            print("arena alloc failed on node", topo[i])
            return

    @parameter
    def dispatch_tp[P: BurstThreadPool, //](var selected_pools: List[P]):
        run_all(selected_pools, arenas)

    with_topological_rank_dispatch[
        dispatch=dispatch_tp,
    ](topo, "mode: isolated", "mode: spin-backoff")
