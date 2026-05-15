from std.reflection import reflect
from std.memory import alloc, free, Layout
from std.collections import InlineArray


trait BorrowTag:
    comptime SIZE:  Int
    comptime FIRST: Int
    comptime LAST:  Int
    comptime UNION: Int
    comptime IDX:   Int


struct AttnQ(BorrowTag, Copyable):
    comptime SIZE  = 16_000_000
    comptime FIRST = 1
    comptime LAST  = 5
    comptime UNION = 2
    comptime IDX   = 0


struct AttnKv(BorrowTag, Copyable):
    comptime SIZE  = 4_000_000
    comptime FIRST = 1
    comptime LAST  = 3
    comptime UNION = 2
    comptime IDX   = 1


struct FfnGate(BorrowTag, Copyable):
    comptime SIZE  = 4_300_000
    comptime FIRST = 7
    comptime LAST  = 9
    comptime UNION = 0
    comptime IDX   = 2


struct FfnDense(BorrowTag, Copyable):
    comptime SIZE  = 23_000_000
    comptime FIRST = 10
    comptime LAST  = 14
    comptime UNION = 0
    comptime IDX   = 3


@fieldwise_init
struct SampleModel(Copyable):
    var attn_q:    AttnQ
    var attn_kv:   AttnKv
    var ffn_gate:  FfnGate
    var ffn_dense: FfnDense


def section_1_handle():
    """
    The reflect handle.

    `reflect[T]` is a comptime alias for `Reflected[T]`. It is a handle
    type, not a function: spell it `reflect[T].method()` with no parens
    after `[T]`. The handle carries `T` as its only parameter; every
    introspection method is a `@staticmethod` driven by that parameter.

    Members come in two shapes:

    - @staticmethod call sites (parens required) return runtime/comptime
      VALUES: `field_count()`, `field_names()`, `field_types()`,
      `name()`, `base_name()`, `is_struct()`, `field_index[lit]()`,
      `field_offset[name=lit]()`, `field_offset[index=N]()`,
      `field_ref[i](s)`.
    - parametric comptime ALIASES (no parens) return TYPES:
      `field_type[lit]` returns `Reflected[FieldT]`.

    Pattern: assign the result of any value-returning method to a
    `comptime` local. That forces evaluation at compile time and lets
    the rest of the body see the result as a comptime constant.
    """
    print("name:           ", reflect[SampleModel].name())
    print("base_name:      ", reflect[SampleModel].base_name())
    print("is_struct:      ", reflect[SampleModel].is_struct())
    print("field_count:    ", reflect[SampleModel].field_count())

    comptime names = reflect[SampleModel].field_names()
    print("field_names[0]: ", names[0])
    print("field_names[3]: ", names[3])


def section_2_literal_name_access():
    """
    Single-field access by hardcoded literal name.

    `reflect[T].field_type[name]` is a parametric comptime member alias.
    Its `name` parameter is typed `StringLiteral[value: !kgen.string]` —
    the literal is encoded into the type at the call site. Two
    consequences:

    1. A string LITERAL at the call site works: `field_type["attn_q"]`.
    2. A `comptime` variable holding a `StaticString` value does NOT
       work as a substitute. `StaticString` and `StringLiteral` are
       distinct types — `StaticString = StringSlice[StaticConstantOrigin]`
       carries no literal value in its type, only at the value level.
       There is no documented public bridge.

    The returned `Reflected[FieldT]` is composable: chain `.T` to get
    the underlying field type, then access comptime members on it.
    """
    comptime q_handle = reflect[SampleModel].field_type["attn_q"]
    print("field_type[\"attn_q\"].T.SIZE: ", q_handle.T.SIZE)
    print("field_type[\"attn_q\"].name(): ", q_handle.name())

    comptime kv_handle = reflect[SampleModel].field_type["attn_kv"]
    print("field_type[\"attn_kv\"].T.LAST:", kv_handle.T.LAST)


def section_3_index_paths():
    """
    Index-based access — partial.

    `field_offset` has TWO overloads, by-name and by-index:
        field_offset[name=lit]()    # StringLiteral parameter
        field_offset[index=N]()     # Int parameter

    `field_type` has only the by-name form. There is no
    `field_type[index=N]`. There is no positional-int form either —
    `field_type[0]` is rejected because the slot is typed
    `StringLiteral[!kgen.string]`, not `Int`.

    `field_ref[idx: Int](ref s: T)` does accept an Int parameter, but
    it returns a REFERENCE into an instance — not a TYPE. Use it for
    runtime field reads/writes, not for compile-time introspection.

    The asymmetry matters for the iteration problem in section_4.
    """
    print("field_offset[name=\"attn_q\"]:    ", reflect[SampleModel].field_offset[name="attn_q"]())
    print("field_offset[index=0]:           ", reflect[SampleModel].field_offset[index=0]())
    print("field_offset[name=\"ffn_dense\"]: ", reflect[SampleModel].field_offset[name="ffn_dense"]())
    print("field_offset[index=3]:           ", reflect[SampleModel].field_offset[index=3]())


def section_4_iteration_problem():
    """
    Why naive iteration over field_types() looks broken.

    `field_types()` returns a `TypeList` whose elements are tracked at
    the MLIR level as `!kgen.param_list<AnyType>`. The trait bound is
    AnyType — the strongest trait every type satisfies, and the
    weakest by what it lets you do.

    Subscripting with a comptime constant from a `comptime for` loop
    falls back to value-level access. The result is typed AnyType,
    which exposes no user members. Specifically:

        comptime for i in range(reflect[T].field_count()):
            comptime FT = reflect[T].field_types()[i]
            total += FT.SIZE                   # ERROR: AnyType has no SIZE

    A literal at the call site fully specializes and gives the
    concrete type:

        comptime FT = reflect[T].field_types()[0]
        total += FT.SIZE                       # works

    But a literal-only path can't drive a generic solver. Section 5
    resolves this.
    """
    print("field_count:", reflect[SampleModel].field_count())
    comptime types = reflect[SampleModel].field_types()
    print("types[0].SIZE (literal index works):", types[0].SIZE)
    print("types[3].SIZE (literal index works):", types[3].SIZE)


def section_5_conforms_to_unlock():
    """
    The unlock: comptime if conforms_to(FT, Trait).

    Inside a `comptime for`, the loop variable `i` is a comptime
    constant. `field_types()[i]` produces a comptime type value typed
    AnyType. Wrapping access in `comptime if conforms_to(FT, Trait)`
    causes the compiler to narrow `FT` inside the branch to a
    Trait-conforming view. Within that branch, `FT.MEMBER` resolves
    to the trait member — including comptime members like SIZE that
    are declared on the trait.

    This is the documented downcast pattern (`conforms_to` /
    `trait_downcast`) applied at the TYPE level. `trait_downcast`
    operates on a VALUE and yields a `T(Trait)` view. The
    type-level analog inside a `comptime if conforms_to(...)` branch
    yields the same trait-bound view at the type level — sufficient
    to read comptime members.

    Two practical patterns work:

    1. Direct: `comptime if conforms_to(FT, T): use FT.MEMBER`
    2. Dispatch: `get[FT]()` where `get` is `def get[T: Trait]() -> X`

    Both produce the same comptime constant. Section 6 shows the
    dispatch form.
    """
    var total = 0
    comptime for i in range(reflect[SampleModel].field_count()):
        comptime FT = reflect[SampleModel].field_types()[i]
        comptime if conforms_to(FT, BorrowTag):
            total += FT.SIZE
    print("naive sum via reflection:", total)

    var peak = 0
    comptime for phase in range(1, 25):
        var alive = 0
        comptime for i in range(reflect[SampleModel].field_count()):
            comptime FT = reflect[SampleModel].field_types()[i]
            comptime if conforms_to(FT, BorrowTag):
                if FT.FIRST <= phase and phase <= FT.LAST:
                    alive += FT.SIZE
        if alive > peak:
            peak = alive
    print("max alive bytes (entropy bound):", peak)


def tag_size[T: BorrowTag]() -> Int:
    return T.SIZE


def tag_window[T: BorrowTag]() -> Int:
    return T.LAST - T.FIRST + 1


def section_6_parametric_helper():
    """
    Passing reflected field types as type parameters.

    A `comptime FT = reflect[T].field_types()[i]` inside a
    `conforms_to(FT, Trait)` guard is bindable to a parametric
    helper whose own parameter requires `Trait`. Inside the
    helper, the parameter `T` is Trait-conforming, so `T.MEMBER`
    works directly without further guards.

    This pattern is useful when the per-tag logic is non-trivial.
    Each helper isolates one operation; the caller stays a simple
    loop over fields.

    Restrictions:
    - The helper's parameter must use `[T: Trait]`. Plain
      `[T: AnyType]` won't grant access to Trait members.
    - The call site `helper[FT]()` only compiles inside the
      narrowed branch.
    """
    var s = 0
    var w = 0
    comptime for i in range(reflect[SampleModel].field_count()):
        comptime FT = reflect[SampleModel].field_types()[i]
        comptime if conforms_to(FT, BorrowTag):
            s += tag_size[FT]()
            w += tag_window[FT]()
    print("sum via helper dispatch:", s)
    print("total window span:      ", w)


def section_7_trait_downcast_for_values():
    """
    trait_downcast — same idea, value level.

    Counterpart to the type-level `conforms_to` narrowing. Operates
    on a VALUE whose type is too loose, rebinds it to a trait-bound
    view that preserves the concrete type. Documented signatures:

        trait_downcast[T: AnyType, //, Trait](ref src: T) -> ref[src] T(Trait)
        trait_downcast[T: TrivialRegisterPassable, //, Trait](var src: T) -> T(Trait)

    Combined with `reflect[T].field_ref[i: Int]`, it would let
    runtime code iterate fields and call trait methods on each.
    But for our scratch-pool solver every metadata field is a
    comptime member, and Section 5's type-level path is sufficient
    — we never need a runtime instance.

    Recorded here for completeness so the relationship between the
    type-level and value-level forms is explicit.

    Mojo treats `comptime SIZE: Int` on a trait as a static member
    requirement: it lives at the type, not on instances. Type-level
    access via the `conforms_to` narrowing reaches it directly. A
    `trait_downcast` of a value gives a view that still exposes the
    static member via `type_of(view).SIZE`, but that's strictly
    longer than the type-level form for the same answer.
    """
    print("value-level downcast: not needed for our use case;")
    print("type-level via conforms_to suffices for comptime members")


comptime MAX_TAGS = 16


@fieldwise_init
struct Solved(Copyable, ImplicitlyCopyable):
    var offsets: InlineArray[Int, MAX_TAGS]
    var total: Int
    var n_slots: Int


def solve_layout[T: AnyType]() -> Solved:
    """
    Greedy interval coloring over a reflected model.

    Walks `reflect[T].field_types()`. For each field whose type
    conforms to BorrowTag, places the tag in the first existing
    slot whose interval (FIRST..LAST) does not overlap in time —
    or sits in a different non-zero UNION group. If no slot fits,
    opens a new slot.

    Yields per-tag byte offsets indexed by `IDX`, and the total
    pool size = sum of slot maxima. This is exactly the entropy
    bound when intervals are on a line.
    """
    var slot_sizes  = InlineArray[Int, MAX_TAGS](fill=0)
    var slot_first  = InlineArray[Int, MAX_TAGS](fill=0)
    var slot_last   = InlineArray[Int, MAX_TAGS](fill=0)
    var slot_union  = InlineArray[Int, MAX_TAGS](fill=0)
    var tag_to_slot = InlineArray[Int, MAX_TAGS](fill=-1)
    var n_slots = 0

    comptime for i in range(reflect[T].field_count()):
        comptime FT = reflect[T].field_types()[i]
        comptime if conforms_to(FT, BorrowTag):
            var placed = False
            comptime for s in range(MAX_TAGS):
                if not placed and s < n_slots:
                    var overlap = not (FT.LAST < slot_first[s] or slot_last[s] < FT.FIRST)
                    var diff_un = (slot_union[s] != 0 and FT.UNION != 0
                                   and slot_union[s] != FT.UNION)
                    if not overlap or diff_un:
                        tag_to_slot[FT.IDX] = s
                        if FT.SIZE > slot_sizes[s]: slot_sizes[s] = FT.SIZE
                        if FT.FIRST < slot_first[s]: slot_first[s] = FT.FIRST
                        if FT.LAST > slot_last[s]:   slot_last[s]  = FT.LAST
                        if slot_union[s] == 0:       slot_union[s] = FT.UNION
                        placed = True
            if not placed:
                tag_to_slot[FT.IDX]    = n_slots
                slot_sizes[n_slots]    = FT.SIZE
                slot_first[n_slots]    = FT.FIRST
                slot_last[n_slots]     = FT.LAST
                slot_union[n_slots]    = FT.UNION
                n_slots += 1

    var slot_offsets = InlineArray[Int, MAX_TAGS](fill=0)
    var acc = 0
    comptime for s in range(MAX_TAGS):
        if s < n_slots:
            slot_offsets[s] = acc
            acc += slot_sizes[s]

    var per_tag = InlineArray[Int, MAX_TAGS](fill=0)
    comptime for k in range(MAX_TAGS):
        if tag_to_slot[k] >= 0:
            per_tag[k] = slot_offsets[tag_to_slot[k]]

    return Solved(offsets=per_tag, total=acc, n_slots=n_slots)


comptime LAYOUT = solve_layout[SampleModel]()


struct Pool[total: Int](Movable):
    var base: Int

    def __init__(out self, base: Int):
        self.base = base

    def slot[Tag: BorrowTag](self) -> UnsafePointer[UInt8, MutAnyOrigin]:
        comptime off = LAYOUT.offsets[Tag.IDX]
        return UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.base + off)


def section_8_layout_alloc_and_pool():
    """
    End-to-end binding.

    `comptime LAYOUT = solve_layout[SampleModel]()` runs the solver
    at compile time. Its `total` becomes the `Pool[total: Int]`
    parameter — fully comptime-known.

    `memory.alloc(Layout[T](count=N))` is the layout-aware
    allocator. `Layout[T]` co-locates element type and count with
    the matching `free`. The pool stores only `base: Int`; every
    `slot[Tag]` returns `base + comptime offset`. Zero allocator
    cost.

    Properties:
    - Pool size = entropy bound, not declaration-order sum.
    - Slot offsets fold into pointer arithmetic; no runtime branch.
    - The struct's field types are the single source of truth.
      Adding or removing a tag is a one-line struct edit.
    """
    print("LAYOUT.total:   ", LAYOUT.total)
    print("LAYOUT.n_slots: ", LAYOUT.n_slots)
    print("  AttnQ off:    ", LAYOUT.offsets[AttnQ.IDX])
    print("  AttnKv off:   ", LAYOUT.offsets[AttnKv.IDX])
    print("  FfnGate off:  ", LAYOUT.offsets[FfnGate.IDX])
    print("  FfnDense off: ", LAYOUT.offsets[FfnDense.IDX])

    var ly = Layout[UInt8](count=LAYOUT.total)
    var ptr = alloc(ly)
    var pool = Pool[LAYOUT.total](Int(ptr))

    var q = pool.slot[AttnQ]()
    var kv = pool.slot[AttnKv]()
    var gate = pool.slot[FfnGate]()
    var dense = pool.slot[FfnDense]()
    print("AttnQ ptr  - base:", Int(q)     - pool.base)
    print("AttnKv ptr - base:", Int(kv)    - pool.base)
    print("Gate ptr   - base:", Int(gate)  - pool.base)
    print("Dense ptr  - base:", Int(dense) - pool.base)

    free(ptr, ly)


def section_9_constraints_summary():
    """
    Summary of typological relationships and constraints.

    Type-pair relationships:
    - StringLiteral[v: !kgen.string] vs StaticString:
      StringLiteral carries the literal value in its type parameter.
      StaticString is `StringSlice[StaticConstantOrigin]`; the value
      is at the value level only. No documented public bridge.
    - TypeList[Trait, *Ts] vs reflect's field_types():
      User-declared TypeList admits a trait parameter; reflection's
      TypeList is fixed at `!kgen.param_list<AnyType>`. There is no
      documented refinement from AnyType to a stricter user trait.
    - conforms_to(T, Trait) at type level <-> trait_downcast at
      value level: both perform the same compile-time narrowing,
      applied to a type vs a value.
    - comptime member access on a Trait declaration: members declared
      `comptime X: Int` on a trait are static type members, not
      instance fields. They are reachable through any trait-bound
      view (type or value).

    Parameter-vs-value, the rule of thumb:
    - A parameter slot typed `StringLiteral` requires a literal at
      the call site. A `comptime` variable of type `StaticString`
      will not coerce — different parameter type categories.
    - A parameter slot typed `Int` accepts any comptime Int,
      including a `comptime for` loop variable in some contexts.
      `field_offset[index=N]` and `field_ref[idx=N]` both accept
      a comptime Int. `field_type` has no Int-parameter form.
    - Type values from `field_types()` are usable as type
      parameters to generic helpers (e.g. `helper[FT]()`).
    """
    print("see docstring for the summary table")


def main():
    print("=== 1. The reflect handle ===")
    section_1_handle()
    print("")

    print("=== 2. Single-field access by literal name ===")
    section_2_literal_name_access()
    print("")

    print("=== 3. Index-based access (partial) ===")
    section_3_index_paths()
    print("")

    print("=== 4. The iteration problem (and what fails) ===")
    section_4_iteration_problem()
    print("")

    print("=== 5. The conforms_to unlock ===")
    section_5_conforms_to_unlock()
    print("")

    print("=== 6. Parametric helper dispatch ===")
    section_6_parametric_helper()
    print("")

    print("=== 7. trait_downcast for value-level access ===")
    section_7_trait_downcast_for_values()
    print("")

    print("=== 8. End-to-end: layout solver + pool ===")
    section_8_layout_alloc_and_pool()
    print("")

    print("=== 9. Constraints summary ===")
    section_9_constraints_summary()
