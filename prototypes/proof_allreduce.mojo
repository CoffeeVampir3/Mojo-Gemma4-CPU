from std.collections import InlineArray
from std.memory import Span, UnsafePointer, alloc, memcpy
from std.os import abort

# Use the REAL dispatch machinery so the exclusivity check is the real one.
from kernels.helpers import RangePartitionedKernel, DispatchBuffer, tile_dispatch
from threading.threading_traits import BurstKernel, BurstThreadPool


comptime SrcPtrT = UnsafePointer[Float32, ImmutExternalOrigin]  # tracked-immutable source
comptime DstPtrT = UnsafePointer[Float32, MutAnyOrigin]         # erased, parallel-written


# Synchronous test pool — exclusivity is a COMPILE-TIME check, so running kernels
# on dispatch is sufficient to prove the design type-checks through tile_dispatch.
@fieldwise_init
struct TestPool(BurstThreadPool):
    var capacity: Int
    var ts: Int
    def get_capacity(self) -> Int:
        return self.capacity
    def dispatch[K: BurstKernel, origin: MutOrigin](
        mut self, kernels: Span[K, origin], num_jobs: Int):
        for i in range(num_jobs):
            var k = kernels[i]
            k.execute()
        self.ts += 1
    def join(mut self): pass
    def last_worker_timestamp(self) -> Int: return self.ts
    def wake(mut self): pass
    def sleep(mut self): pass


# The value-type config: NON-owning Span views, single union origin `buffers_origin`.
#   src: immutable span of tracked-immutable pointees  -> no aliasing (read-only)
#   dst: immutable span of MutAnyOrigin pointees        -> table read-only; writes go
#        through erased pointees (unchecked), like every other kernel in the codebase
@fieldwise_init
struct ReduceCfg[buffers_origin: ImmutOrigin](TrivialRegisterPassable):
    var src: Span[SrcPtrT, Self.buffers_origin]
    var dst: Span[DstPtrT, Self.buffers_origin]
    var chunk: Int
    var rem: Int


@fieldwise_init
struct ReduceK[buffers_origin: ImmutOrigin](RangePartitionedKernel):
    var cfg: ReduceCfg[Self.buffers_origin]      # carried BY VALUE — no UnsafePointer(to=local)
    var rank: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var tp = len(self.cfg.src)
        var out = self.cfg.dst[self.rank]          # erased MutAnyOrigin write target
        for pos in range(self.start, self.end):
            var acc = Float32(0)
            for r in range(tp):
                acc += (self.cfg.src[r] + pos)[]    # immutable reads
            (out + pos)[] = acc

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


@fieldwise_init
struct GatherK[buffers_origin: ImmutOrigin](RangePartitionedKernel):
    var cfg: ReduceCfg[Self.buffers_origin]
    var rank: Int
    var start: Int
    var end: Int

    def execute(mut self):
        # Gather pattern: read dst[other], write dst[self] — full dst table needed.
        var tp = len(self.cfg.dst)
        for src_rank in range(tp):
            if src_rank == self.rank:
                continue
            var lo = max(self.start, self.cfg.chunk * src_rank)
            var hi = min(self.end, self.cfg.chunk * src_rank
                         + (self.cfg.chunk + (self.cfg.rem if src_rank == tp - 1 else 0)))
            if lo < hi:
                memcpy(dest=self.cfg.dst[self.rank] + lo,
                       src=self.cfg.dst[src_rank] + lo, count=hi - lo)

    @always_inline
    def install_range(mut self, start: Int, end: Int):
        self.start = start
        self.end = end


# The dispatcher takes the two pointer tables as BORROWED lists; their union origin
# is the config origin. No local owned List is created; the spans view the caller's
# tables, which outlive the whole call (borrowed args are immune to ASAP).
def dispatch_allreduce[max_worker_count: Int](
    read src_tbl: List[SrcPtrT],
    read dst_tbl: List[DstPtrT],
    n: Int,
    mut pools: List[TestPool],
):
    var tp = len(pools)
    var chunk = n // tp
    comptime buffers_origin = origin_of(src_tbl, dst_tbl)
    var cfg = ReduceCfg[buffers_origin](
        Span[SrcPtrT, buffers_origin](src_tbl), Span[DstPtrT, buffers_origin](dst_tbl),
        chunk, n - chunk * tp)

    # Reduce phase: each rank reduces its chunk into dst[rank].
    var rbuf = DispatchBuffer[ReduceK[buffers_origin], max_worker_count]()
    for r in range(tp):
        var lo = chunk * r
        var hi = lo + chunk + (n - chunk * tp if r == tp - 1 else 0)
        _ = tile_dispatch(rbuf, ReduceK[buffers_origin](cfg, r, 0, 0), pools[r],
                          hi - lo, lo, 1)
    for r in range(tp):
        pools[r].join()

    # Gather phase: each rank copies the other ranks' reduced chunks into its dst.
    var gbuf = DispatchBuffer[GatherK[buffers_origin], max_worker_count]()
    for r in range(tp):
        _ = tile_dispatch(gbuf, GatherK[buffers_origin](cfg, r, 0, 0), pools[r], n, 0, 1)
    for r in range(tp):
        pools[r].join()


def check(ok: Bool, msg: String):
    if not ok:
        abort("FAIL: " + msg)
    print("  ok:", msg)


def main():
    print("=== proof: value-type span config through REAL tile_dispatch ===\n")
    comptime TP = 4
    comptime N = 100

    # Keep the owning (allocated) pointers for free(); the tables hold views.
    var own_s = InlineArray[UnsafePointer[Float32, MutExternalOrigin], TP](
        uninitialized=True)
    var own_d = InlineArray[UnsafePointer[Float32, MutExternalOrigin], TP](
        uninitialized=True)
    var src_tbl = List[SrcPtrT]()
    var dst_tbl = List[DstPtrT]()
    for r in range(TP):
        var s = alloc[Float32](N)
        var d = alloc[Float32](N)
        for i in range(N):
            s[i] = Float32(r * 1000 + i)
            d[i] = Float32(0)
        own_s[r] = s
        own_d[r] = d
        src_tbl.append(s.as_immutable())     # ImmutExternalOrigin
        dst_tbl.append(d.as_any_origin())     # MutAnyOrigin (erased)

    var pools = List[TestPool]()
    for _ in range(TP):
        pools.append(TestPool(4, 0))

    dispatch_allreduce[max_worker_count=8](src_tbl, dst_tbl, N, pools)

    var ok = True
    for i in range(N):
        var expect = Float32(0)
        for r in range(TP):
            expect += Float32(r * 1000 + i)
        for r in range(TP):
            if own_d[r][i] != expect:
                ok = False
    check(ok, "all ranks hold the reduced sum (reduce+gather via real tile_dispatch)")

    for r in range(TP):
        own_s[r].free()
        own_d[r].free()
    print("\n=== DESIGN VIABLE: value-type config compiled through real tile_dispatch ===")
