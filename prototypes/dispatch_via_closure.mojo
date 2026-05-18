"""
Prototype A: dispatch via build-closure.

Premise
-------
The burst pool calls `K.execute(mut self)` with no args. So every work
item in the dispatch buffer must already carry its own range as struct
fields. K is a (params + start + end) trivial struct — there is no way
to avoid that as long as the pool's contract is `execute(mut self)`.

What this prototype changes is the *materialization* step. Today every
kernel writes an `over_range(self, start, end) -> Self` method that
restates every field. The dispatcher calls `proto.over_range(s, e)` per
worker. The "proto" carries placeholder zeros for start/end and a real
value for everything else.

Here, the dispatcher takes a builder closure `def(Int, Int) -> K`
instead of a proto. The closure captures the per-rank pointers from the
caller's stack. No template, no placeholders, no `over_range` method.

NOTE: Mojo requires capturing closures to be passed as *comptime
parameters* (the closure cannot be materialized as a runtime value, see
the "capturing closures cannot be materialized as runtime values"
error). So `build` goes in the bracket parameter list, not the args.
Each call site monomorphizes a fresh `tile_dispatch` — same shape as
`with_topological_rank_dispatch` in threading/topological_dispatch.mojo.

Properties
----------
- K has no method requirements beyond `execute(mut self)`. The
  `OutputPartitionedKernel` trait disappears.
- No reflection. The contract — "give me a way to build a K from a
  range" — is the closure's signature, and the compiler enforces it.
- Chain composition is one-shot: the caller's closure constructs both
  sides of the chain with the same (s, e).

Demerits
--------
- Each call site uses `@parameter def build(...)` + a tile_dispatch
  call that names the closure as a comptime arg. ~4-5 extra lines per
  dispatch vs the trait approach.
- Each call site monomorphizes `tile_dispatch`. Negligible for a
  bounded number of kernel-dispatch sites.
"""

from std.collections import InlineArray
from std.memory import UnsafePointer


trait MiniKernel(Copyable, Movable, ImplicitlyDestructible):
    def execute(mut self): ...


@fieldwise_init
struct SumKernel(MiniKernel):
    var acc: UnsafePointer[Int, MutAnyOrigin]
    var start: Int
    var end: Int

    def execute(mut self):
        var local = self.acc[]
        for i in range(self.start, self.end):
            local += i
        self.acc[] = local


@fieldwise_init
struct Chain[A: MiniKernel, B: MiniKernel](MiniKernel):
    var a: Self.A
    var b: Self.B

    def execute(mut self):
        self.a.execute()
        self.b.execute()


@always_inline
def worker_range(total: Int, n_workers: Int, w: Int) -> Tuple[Int, Int]:
    var per = (total + n_workers - 1) // n_workers
    var s = w * per
    var e = min(s + per, total)
    return (s, e)


def tile_dispatch[
    K: MiniKernel, //, max_workers: Int,
    build: def(Int, Int) capturing [_] -> K,
](
    mut buf: InlineArray[K, max_workers],
    mut count: Int,
    total: Int,
    n_workers: Int,
):
    var w = 0
    while w < n_workers and count < max_workers:
        var wr = worker_range(total, n_workers, w)
        buf[count] = build(wr[0], wr[1])
        count += 1
        w += 1


def main():
    print("=== build-closure dispatch ===")

    # --- Simple kernel ---
    var acc = 0
    var acc_ptr = UnsafePointer(to=acc).as_any_origin()

    var buf = InlineArray[SumKernel, 8](uninitialized=True)
    var count = 0

    @parameter
    def build_sum(s: Int, e: Int) -> SumKernel:
        return SumKernel(acc_ptr, s, e)

    tile_dispatch[max_workers=8, build=build_sum](buf, count, 100, 4)

    # In the real system the pool runs these; here we run inline.
    for i in range(count):
        buf[i].execute()

    print("  sum [0..100) over 4 workers:", acc, "expected 4950")

    # --- Chain composition ---
    var acc_a = 0
    var acc_b = 0
    var pa = UnsafePointer(to=acc_a).as_any_origin()
    var pb = UnsafePointer(to=acc_b).as_any_origin()

    var cbuf = InlineArray[Chain[SumKernel, SumKernel], 8](uninitialized=True)
    var ccount = 0

    @parameter
    def build_chain(s: Int, e: Int) -> Chain[SumKernel, SumKernel]:
        return Chain[SumKernel, SumKernel](
            SumKernel(pa, s, e), SumKernel(pb, s, e),
        )

    tile_dispatch[max_workers=8, build=build_chain](cbuf, ccount, 100, 4)

    for i in range(ccount):
        cbuf[i].execute()

    print("  chain a:", acc_a, "b:", acc_b, "expected both 4950")
