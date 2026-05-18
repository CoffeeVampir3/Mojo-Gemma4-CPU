"""
Prototype B (final): dispatch via widened `Partitioned` trait.

Premise
-------
The burst pool calls `K.execute(mut self)` with no args, so every work
item must carry its own per-worker state inside the struct. K is a
(params + per_worker_state + start + end) trivial struct.

Today the codebase has two patterns for materializing those structs:

  1. `tile_dispatch` + `over_range(start, end)` — used by kernels whose
     only per-worker state is the (start, end) pair (gemv, rmsnorm,
     elementwise, rope, simple merges).

  2. Inline construction in a per-worker loop — used by kernels that
     also need a `worker_id` field (FullAttention, FlashDecode, Router,
     Phase1, Phase2). These kernels opt out of `tile_dispatch` entirely
     because its (start, end) interface isn't enough.

This is a real cost: 5 of the largest kernels carry their own dispatch
boilerplate because the abstraction doesn't fit. We unify by widening
the trait to pass *all* per-worker indices:

    trait Partitioned:
        @always_inline
        def set_partition(mut self, worker_id: Int, start: Int, end: Int): ...

Kernels without a `worker_id` field simply ignore the parameter (one
line of overhead). Kernels with one assign it. Same `tile_dispatch`
serves both. The 5 inline-dispatched kernels go through the same path.

Properties
----------
- Contract is on the type. Forgetting `set_partition` is a trait-
  conformance error, not a name-lookup failure.
- One dispatcher shape (`tile_dispatch`) for all kernels. The
  inline-build pattern disappears.
- Per-kernel cost: one `set_partition` method, 1-3 lines.
- Per-call-site cost: one `proto` line + one `tile_dispatch` line.
- Chain composition: `Chain.set_partition` forwards to both.

Demerits
--------
- Simple kernels ignore the `worker_id` parameter. One unused line.
- The proto-with-placeholder-zeros idiom survives (proto is built with
  worker_id=0, start=0, end=0; the dispatcher overwrites them).
"""

from std.collections import InlineArray
from std.memory import UnsafePointer


trait MiniKernel(Copyable, ImplicitlyCopyable, Movable, ImplicitlyDestructible):
    def execute(mut self): ...


trait Partitioned:
    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int): ...


# --- Kernel without worker_id: ignores the parameter. ---
@fieldwise_init
struct SumKernel(MiniKernel, Partitioned):
    var acc: UnsafePointer[Int, MutAnyOrigin]
    var start: Int
    var end: Int

    def execute(mut self):
        var local = self.acc[]
        for i in range(self.start, self.end):
            local += i
        self.acc[] = local

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.start = start
        self.end = end


# --- Kernel with worker_id: writes its result to a per-worker slot.
# This mirrors FullAttentionKernel/FlashDecodeKernel/Phase1/etc., which
# index into a per-worker partials buffer by worker_id.
@fieldwise_init
struct WorkerSumKernel(MiniKernel, Partitioned):
    var per_worker_acc: UnsafePointer[Int, MutAnyOrigin]
    var worker_id: Int
    var start: Int
    var end: Int

    def execute(mut self):
        var local = 0
        for i in range(self.start, self.end):
            local += i
        (self.per_worker_acc + self.worker_id)[] = local

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.worker_id = worker_id
        self.start = start
        self.end = end


@fieldwise_init
struct Chain[
    A: MiniKernel & Partitioned, B: MiniKernel & Partitioned,
](MiniKernel, Partitioned):
    var a: Self.A
    var b: Self.B

    def execute(mut self):
        self.a.execute()
        self.b.execute()

    @always_inline
    def set_partition(mut self, worker_id: Int, start: Int, end: Int):
        self.a.set_partition(worker_id, start, end)
        self.b.set_partition(worker_id, start, end)


@always_inline
def worker_range(total: Int, n_workers: Int, w: Int) -> Tuple[Int, Int]:
    var per = (total + n_workers - 1) // n_workers
    var s = w * per
    var e = min(s + per, total)
    return (s, e)


def tile_dispatch[
    K: MiniKernel & Partitioned, //, max_workers: Int,
](
    mut buf: InlineArray[K, max_workers],
    mut count: Int,
    proto: K,
    total: Int,
    n_workers: Int,
):
    var w = 0
    while w < n_workers and count < max_workers:
        var wr = worker_range(total, n_workers, w)
        var item = proto
        item.set_partition(w, wr[0], wr[1])
        buf[count] = item^
        count += 1
        w += 1


def main():
    print("=== widened Partitioned-trait dispatch ===")

    # --- (1) Simple kernel without worker_id ---
    var acc = 0
    var acc_ptr = UnsafePointer(to=acc).as_any_origin()

    var buf = InlineArray[SumKernel, 8](uninitialized=True)
    var count = 0

    var proto = SumKernel(acc_ptr, 0, 0)
    tile_dispatch[max_workers=8](buf, count, proto, 100, 4)

    for i in range(count):
        buf[i].execute()

    print("  sum [0..100), 4 workers, no worker_id:", acc, "expected 4950")

    # --- (2) Kernel WITH worker_id (FlashDecode-shaped) ---
    var per_worker = InlineArray[Int, 8](fill=0)
    var pw_ptr = UnsafePointer(to=per_worker[0]).as_any_origin()

    var wbuf = InlineArray[WorkerSumKernel, 8](uninitialized=True)
    var wcount = 0

    var wproto = WorkerSumKernel(pw_ptr, 0, 0, 0)
    tile_dispatch[max_workers=8](wbuf, wcount, wproto, 100, 4)

    for i in range(wcount):
        wbuf[i].execute()

    # Expected: each worker writes its partial sum of its [start..end)
    # slice into slot worker_id. With total=100, n_workers=4:
    # worker 0: [0,25)   sum=300
    # worker 1: [25,50)  sum=925
    # worker 2: [50,75)  sum=1550
    # worker 3: [75,100) sum=2175
    print("  per-worker partials:",
        per_worker[0], per_worker[1], per_worker[2], per_worker[3],
        "expected 300 925 1550 2175")

    var verify_sum = per_worker[0] + per_worker[1] + per_worker[2] + per_worker[3]
    print("  sum of partials:", verify_sum, "expected 4950")

    # --- (3) Chain composition (both sides Partitioned) ---
    var acc_a = 0
    var acc_b = 0
    var pa = UnsafePointer(to=acc_a).as_any_origin()
    var pb = UnsafePointer(to=acc_b).as_any_origin()

    var cbuf = InlineArray[Chain[SumKernel, SumKernel], 8](uninitialized=True)
    var ccount = 0

    var cproto = Chain[SumKernel, SumKernel](
        SumKernel(pa, 0, 0), SumKernel(pb, 0, 0),
    )
    tile_dispatch[max_workers=8](cbuf, ccount, cproto, 100, 4)

    for i in range(ccount):
        cbuf[i].execute()

    print("  chain a:", acc_a, "b:", acc_b, "expected both 4950")

    # --- (4) Chain mixing a worker_id kernel and a simple kernel ---
    # This is the most interesting case: worker_id propagates only to the
    # side that uses it. The simple side ignores it.
    var pw2 = InlineArray[Int, 8](fill=0)
    var pw2_ptr = UnsafePointer(to=pw2[0]).as_any_origin()
    var acc_simple = 0
    var ps = UnsafePointer(to=acc_simple).as_any_origin()

    var mbuf = InlineArray[Chain[WorkerSumKernel, SumKernel], 8](uninitialized=True)
    var mcount = 0

    var mproto = Chain[WorkerSumKernel, SumKernel](
        WorkerSumKernel(pw2_ptr, 0, 0, 0), SumKernel(ps, 0, 0),
    )
    tile_dispatch[max_workers=8](mbuf, mcount, mproto, 100, 4)

    for i in range(mcount):
        mbuf[i].execute()

    print("  mixed chain partials:",
        pw2[0], pw2[1], pw2[2], pw2[3], "expected 300 925 1550 2175")
    print("  mixed chain simple side:", acc_simple, "expected 4950")
