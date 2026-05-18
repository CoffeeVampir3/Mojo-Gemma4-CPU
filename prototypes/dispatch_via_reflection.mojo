"""
Prototype C: dispatch via reflection (`with_range`).

This prototype was meant to show a zero-per-kernel-surface design:
each kernel just declares `start: Int` and `end: Int` fields, and a
shared `with_range[K]` helper uses `reflect[K].field_index["start"]`
+ `field_ref[i]` to write the runtime range into those fields.

The prototype does not compile. The failure is the point.

The wall
--------
`reflect[K].field_ref[i: Int](ref s: K)` returns a reference typed as
the abstract field type — i.e. `AnyType` until narrowed. Mojo's type
system has no path from "ref to AnyType" to "assignable from Int":

    ref s = reflect[K].field_ref[start_idx](out)
    s = start          # error: cannot convert Int to <kgen.struct_field_types ...>

The slot.mojo / temporal_scratch.mojo patterns get past this *only*
because they narrow with `comptime if conforms_to(FT, SlotLike)`, then
call `slot.set_offset(off)` — a method on a trait the field's struct
implements. The narrowing yields a SlotLike-bound view; the method
gives the type system something to dispatch through.

For a bare `Int` field, there is no equivalent. `Intable` defines
`__int__` (a getter), not `__assign_int__`. No builtin trait says
"this scalar is writable from an Int". Direct assignment through a
reflected ref simply does not typecheck.

Two workarounds
---------------
(1) Use byte offsets and bypass the type system:

        comptime start_off = reflect[K].field_offset[name="start"]()
        var p = UnsafePointer(to=out).bitcast[Int]()
        (p + start_off // sizeof[Int]())[] = start

    Unsafe, fragile to field-order changes, and obviously not what
    the language wants you to do.

(2) Give each kernel a `set_start(mut self, v: Int)` /
    `set_end(mut self, v: Int)` method, or equivalently a single
    `set_range(mut self, start: Int, end: Int)` method, and reach
    them through `conforms_to`.

Option (2) IS the Partitioned trait from Prototype B, with extra
reflection ceremony layered on top. Strictly worse than just having
the trait.

Conclusion
----------
The reflection approach was an attempt to avoid declaring a trait. It
turns out it cannot avoid declaring one — Mojo's type system forces
the contract into either (a) trait methods (which is the explicit-
trait design) or (b) raw byte writes (which we don't want). The
"reflection as implicit trait" framing is exactly right: it's a trait
hidden behind a name-lookup, and the type system won't let it work
without either a real trait or an unsafe escape.

This is good evidence that the explicit-trait or build-closure designs
are the right ones to pick from.
"""

from std.collections import InlineArray
from std.memory import UnsafePointer
from std.reflection import reflect


trait MiniKernel(Copyable, ImplicitlyCopyable, Movable, ImplicitlyDestructible):
    def execute(mut self): ...


# The following body is the cleanest expression of the intent. It does
# not compile — see the docstring for why. Left here as evidence of
# the dead end.
#
# @always_inline
# def with_range[K: MiniKernel, //](
#     read proto: K, start: Int, end: Int,
# ) -> K:
#     comptime start_idx = reflect[K].field_index["start"]()
#     comptime end_idx = reflect[K].field_index["end"]()
#     var out = proto
#     ref s = reflect[K].field_ref[start_idx](out)
#     s = start                      # <-- TYPE ERROR: AnyType not Int-assignable
#     ref e = reflect[K].field_ref[end_idx](out)
#     e = end                        # <-- TYPE ERROR: AnyType not Int-assignable
#     return out^


def main():
    print("=== reflection-based dispatch ===")
    print("  (does not compile — see file docstring for analysis)")
