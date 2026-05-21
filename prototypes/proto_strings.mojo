from std.collections.string import Codepoint
from std.collections.string.string_slice import StaticString, StringSlice
from std.format.tstring import TString
from std.builtin.string_literal import StringLiteral


@fieldwise_init
struct Person(Copyable, Writable):
    var name: String
    var age: Int

    def write_to(self, mut writer: Some[Writer]):
        t"{self.name} ({self.age})".write_to(writer)


@fieldwise_init
struct Tensor(Copyable, Writable):
    var name: String
    var shape: List[Int]
    var dtype: String

    def write_to(self, mut writer: Some[Writer]):
        t"Tensor({self.name}, shape=".write_to(writer)
        "[".write_to(writer)
        for i in range(len(self.shape)):
            if i:
                ", ".write_to(writer)
            String(self.shape[i]).write_to(writer)
        "]".write_to(writer)
        t", dtype={self.dtype})".write_to(writer)


def section_1_the_four_string_types():
    """The Mojo string ecosystem has four primary types.

    They are distinguished by who owns the bytes and when those bytes
    are decided.

    Type           | Owns bytes? | Mutable? | When known?  | Allocates?
    ---------------+-------------+----------+--------------+-----------
    StringLiteral  | n/a         | no       | comptime     | no
    StaticString   | no (views)  | no       | comptime     | no
    StringSlice    | no (views)  | param    | runtime      | no
    String         | yes         | yes      | runtime      | maybe (SSO)

    StringLiteral is a comptime-only carrier. Its `value` is an MLIR
    `!kgen.string` parameter, so the bytes ARE the type. Two literals
    with different bytes are different types. It is marked
    `@__nonmaterializable(String)`, meaning the moment a literal flows
    into a runtime variable, Mojo materializes it to `String` (or to
    `StaticString` if context demands).

    StaticString is an alias: `StringSlice[StaticConstantOrigin]`. It
    is a runtime value (a pointer plus a length) whose origin promises
    the referent lives forever, so it can be stored long-term and
    passed across thread boundaries without lifetime ceremony.

    StringSlice is the general view: `StringSlice[mut, origin]`. It is
    register-passable and ABI-compatible with `llvm::StringRef`. The
    origin parameter is what makes a slice safe - it ties the view to
    the lifetime of the owner.

    String owns its bytes. It is three words wide (24 bytes on 64-bit)
    and uses the top byte of the capacity field for flags. There are
    three storage forms encoded in those flags:

    - inline (SSO): up to 23 bytes live inside the String struct itself.
    - static: pointer into static data; mutation triggers a copy.
    - ref-counted: heap allocation with an atomic refcount prepended
      to the data; copy-on-write via `_is_unique()`.
    """
    var literal_var = "hello"                  # StringLiteral materializes to String
    var static_view: StaticString = "world"    # literal -> StaticString
    var owned = String("abc")                  # heap or inline String
    var view = StringSlice(owned)              # explicit view of owned

    print("literal_var:", literal_var)
    print("static_view:", static_view)
    print("owned:      ", owned)
    print("view:       ", view)


def section_2_string_storage_modes():
    """A String picks its storage strategy lazily and tries hard not to allocate.

    The decision tree:

    - Constructed from StringLiteral / StaticString: stores a pointer
      to the static bytes. `capacity_or_data` is 0 (or just the
      nul-term flag). Zero allocations until mutated.
    - First mutation:
        - if total length will fit in 23 bytes, inline it on the stack
          (no allocation, the bytes clobber the struct's own fields).
        - otherwise, allocate a refcounted buffer with a leading atomic
          refcount, set FLAG_IS_REF_COUNTED.
    - Copy of a refcounted String: bumps the refcount with RELAXED
      ordering. Modifying a non-unique buffer triggers a real copy.

    INLINE_CAPACITY = `Int.BITWIDTH // 8 * 3 - 1`, i.e. 23 bytes on a
    64-bit system. The top byte of `capacity_or_data` doubles as a
    5-bit length field plus three flag bits, which is why 23 (not 24)
    bytes fit inline.

    Practical NUMA-relevant consequence: short strings touch one cache
    line and no shared atomic. Long strings touch the refcount, which
    is a shared write site - prefer to construct long strings on the
    NUMA node where they will be read most often.
    """
    var short = String("12345")
    var long = String("a string definitely longer than twenty-three bytes inline")

    print("short capacity:", short.capacity(), "len:", short.byte_length())
    print("long  capacity:", long.capacity(),  "len:", long.byte_length())


def section_3_writable_and_writer():
    """The Writable trait is the spine of all string formatting in Mojo.

    A type is Writable when it has `write_to(self, mut writer: T)`.
    Anywhere you would reach for `__str__` in Python, you implement
    `write_to` in Mojo - and you can write directly into the target
    sink instead of allocating an intermediate String.

    This is why `print()` is variadic over `*Ts: Writable` rather than
    over `*Ts: String`. The contract is "I will hand you a writer; you
    push bytes." It pairs with String's two-phase init: first walk the
    pack with `_TotalWritableBytes()` to size the buffer, then write
    once into a pre-sized String.

    String itself is BOTH Writable (it can be written into another
    writer) and Writer (other Writables can write into it). This is
    what lets `String(t"hello {x}")` work without an intermediate
    buffer.
    """
    var alice = Person("Alice", 30)

    print(alice)                          # print sees a Writable directly
    print(String(alice))                  # explicit collapse to a String
    print(String("met ", alice, sep=""))  # variadic Writable construction


def section_4_byte_codepoint_grapheme():
    """UTF-8 strings have THREE length notions and they almost never agree.

    This is the most common source of bugs when porting Python or
    JavaScript string code.

    - byte_length():        O(1), counts UTF-8 bytes.
    - count_codepoints():   O(n), counts Unicode scalar values.
    - count_graphemes():    O(n), counts user-perceived characters
                            per UAX #29 (handles ZWJ sequences,
                            flag emoji, combining marks).

    Cafe with a precomposed acute (U+00E9): 5 bytes, 4 codepoints,
                                            4 graphemes.
    Cafe with a combining acute   (U+0065 U+0301):
                                            5 bytes, 5 codepoints,
                                            4 graphemes.

    Iteration follows the same three flavors:
    - `s.codepoints()`        yields Codepoint values.
    - `s.codepoint_slices()`  yields one-codepoint StringSlices.
    - `s.graphemes()`         yields one-grapheme StringSlices.
    - `s.grapheme_indices()`  yields (byte_offset, grapheme) pairs.

    Slicing also comes in three index spaces:
        s[byte=0:5]
        s[codepoint=0:3]
        s[grapheme=0:3]
    `byte=` slicing aborts if the boundary is mid-codepoint.
    Codepoint and grapheme slicing have to forward-scan, so they are
    O(n) in the byte length.
    """

    def show(label: StaticString, s: StringSlice):
        print(
            label,
            "bytes=",      s.byte_length(),
            "codepoints=", s.count_codepoints(),
            "graphemes=",  s.count_graphemes(),
        )

    show("family    ", "👨‍👩‍👧‍👦")
    show("flag      ", "🇺🇸")
    show("wave      ", "👋🏽")
    show("namaste   ", "नमस्ते")
    show("ascii     ", "hello")


def section_5_origins_keep_slices_safe():
    """Every StringSlice carries an `origin` parameter; the borrow checker uses it to refuse to let the view outlive the owner.

    `StringSlice[mut, origin]` parameters:
    - `mut`: whether bytes may be modified through this view.
    - `origin`: where the bytes live. Special origin
      `StaticConstantOrigin` means "forever," which is what makes
      StaticString safe to store.

    `StaticString = StringSlice[StaticConstantOrigin]`.

    When you write a function parameter as `StringSlice`, Mojo infers
    the origin from the caller. This is what allows the same function
    to accept literals, owned strings, and slices of slices without
    copying.

    A practical NUMA pattern: pass StringSlice into read-only helpers.
    A slice is a pointer plus a length (16 bytes, register-passable);
    it carries no atomic to bump on a remote read.
    """
    var owned = String("hello world")

    def first_word(s: StringSlice) -> StringSlice[s.origin]:
        var i = s.find(" ")
        if i < 0:
            return s
        return s[byte=0:i]

    var head = first_word(owned)        # origin = owned's origin
    var literal_head = first_word("compile time greeting")

    print("head:        ", head)
    print("literal_head:", literal_head)


def section_6_format_vs_tstring() raises:
    """Mojo ships two string-templating mechanisms.

    1) `.format(...)` on a runtime template, e.g.
         "{} + {} = {}".format(1, 2, 3)
       The template is parsed every call (or precompiled into a
       static entries table for literal templates). Supports manual
       and automatic indexing and `!r`/`!s` conversion flags. It
       still allocates a String to return.

    2) T-strings: a literal syntax `t\"...\"` that is parsed at COMPILE
       TIME into a `TString[format_string, *Ts]`. The struct stores a
       VariadicPack of references to the interpolated values; the
       format template is encoded by the compiler into a NUL-separated
       byte sequence. `{}` is the only replacement field; arbitrary
       expressions are allowed inside the braces.

    T-strings beat .format() on three axes:
    - safety: the format string is checked at compile time.
    - speed: no runtime format parsing, no allocation if the t-string
      is just written to an existing writer (print, log, file).
    - flexibility: replacement fields are real expressions, so
      `t\"{list[0] + list[1]}\"` works and `\"{}\".format(...)` cannot.

    T-strings are LAZY. `var template = t\"x={x}\"` constructs no
    bytes; it just captures references. Bytes are materialized when
    you (a) print it, (b) explicitly `String(template)`, or (c) hand
    it to any other Writer.
    """
    var count = 3
    var items = "apples"
    var template = t"Give me {count} {items}."

    print(template)                   # no allocation - writes to stdout
    var collapsed = String(template)  # allocation here
    print(collapsed)

    print("{0} {1} {0}".format("Mojo", 1.125))   # classic .format
    print(t"{1 + 1} squared is {(1 + 1) ** 2}")  # arbitrary exprs


def section_7_tstring_internals():
    """The compiler lowers a t-string into a TString[format, *Ts] value.

    Lowering of `t\"a={x} b={y}\"` becomes a call to
    `__make_tstring[format=\"a={} b={}\"](x, y)`, which returns

        TString[
            origins=ImmutOrigin(args.origin),
            format_string=StaticString(\"a={} b={}\"),
            T0, T1,
        ]

    The struct has exactly one field: a VariadicPack of immutable
    references to the captured values. Compile-time encoding of the
    format string produces a flat byte sequence (see
    `_encode_format_string` in tstring.mojo) where each `{}` is
    replaced by a single NUL byte and the template ends with a
    trailing NUL. So `\"result: {} + {} = {}\"` becomes

        result: \\0 + \\0 = \\0\\0

    `{{` and `}}` are resolved to literal `{` `}` during this
    encoding, and any unmatched or non-empty replacement field is a
    compile-time error.

    At write time, `write_to` walks the NUL-separated literal
    segments and emits each one as a StringSlice, calling
    `self._values[i].write_to(writer)` between them. The result: no
    runtime format parsing, no temporary String, just N+1 literal
    writes interleaved with N value writes.

    Implementation consequence worth knowing: TString implements
    Writable but NOT Copyable, NOT ImplicitlyCopyable. It also pins
    to the origins of its captured values. So a TString is
    fundamentally a short-lived expression, not a storeable value.
    If you want to keep the formatted result, collapse it with
    `String(template)`.

    The encoded byte sequence is itself materialized into a comptime
    `List[Byte]`, then exposed at runtime via
    `_comptime_list_to_span`, which calls `global_constant[array]()`.
    That lands the bytes in the static read-only segment - no
    per-call heap activity for the template itself.
    """
    var x = 41
    var y = 1
    var t = t"{x} + {y} = {x + y}"
    print(t)
    print("repr-style debug:")
    var buffer = String()
    t.write_repr_to(buffer)
    print(buffer)


def section_8_tstring_lazy_no_alloc():
    """Demonstrate that a TString does NOT allocate when streamed.

    The Writable contract gives us a hook: print(), log sinks, and
    file writers all accept Writables. A TString streamed into any
    of those produces exactly the bytes you'd expect with zero
    intermediate buffer.

    Contrast with .format(): even when called with a literal
    template, it always returns a freshly allocated String, because
    that is its signature.

    Rule of thumb: if you're about to immediately print/log/write
    the string, hand the t-string straight to the sink. Only
    collapse to String when you need to STORE the result.
    """
    var name = "Nate"
    var template = t"Hello, {name}!"

    print(template)
    print(template)
    print(template)

    var collapsed = String(template)
    print("collapsed kept:", collapsed)


def section_9_tstring_nested_and_raw():
    """Two features t-strings inherit from being a literal syntax.

    NESTED T-STRINGS. The expression inside `{}` can itself be a
    t-string. The compiler allows up to 20 levels of nesting. Each
    inner t-string is a Writable, so it streams into the outer one
    without producing an intermediate String.

    RAW T-STRINGS. Prefix `rt\"...\"` (or `tr\"...\"`, `rT\"...\"`,
    etc.) creates a raw t-string: backslashes are literal,
    interpolation still works. Useful for paths, regex, and code
    generation.

    Curly-brace escapes: `{{` and `}}` write literal `{` and `}`.

    Triple-quoted t-strings exist (the t-prefixed triple-quoted
    form) and behave the same way - multi-line, with interpolation.
    """
    var name = "world"

    var nested = t"Hello, {t"dear {name}"}!"
    print(nested)

    var base = "/home/grail"
    print(rt"Path: {base}\subdir\file.txt")

    print(t"Use {{braces}} to print a literal pair around {name}")


def section_10_tstring_as_writable_field():
    """Use t-strings inside another type's write_to to compose without allocating.

    Because TString is Writable, you can use t-strings inside
    another type's `write_to` method to compose representations
    without ever constructing intermediate Strings.

    This is the idiomatic way to give a struct a printable form: do
    not allocate inside write_to, just stream.

    The pattern below produces zero allocations on the print() path
    even for deeply structured output - everything funnels through
    the same writer the caller provided.
    """
    var t1 = Tensor("Q", [32, 128, 64], "bf16")
    print(t1)


def section_11_codepoints_and_chr_ord():
    """Codepoint is a validated Unicode scalar value.

    Codepoint stores values in 0..D7FF or E000..10FFFF (excluding
    surrogates). Construction surfaces:

    - `Codepoint.ord(s: StringSlice)`: from a single-char slice.
    - `Codepoint.from_u32(u: UInt32)`: returns Optional, None for
      invalid scalars.
    - `Codepoint(b: UInt8)`: a single byte (always a valid codepoint).

    Free functions `ord()` and `chr()` are in the prelude and
    mirror Python:
    - ord(s) -> Int            single-char to codepoint number.
    - chr(c: Int) -> String    codepoint number to a String.

    `chr()` aborts for invalid scalar values (e.g. surrogates,
    > 0x10FFFF). When a fallible API is needed prefer
    `Codepoint.from_u32`.

    Escape forms inside a string literal:
        \\xHH        1-byte hex
        \\uHHHH      4-digit hex up to U+FFFF
        \\UHHHHHHHH  8-digit hex up to U+10FFFF
    Surrogate codepoints (U+D800..U+DFFF) are rejected; for
    above-BMP characters use \\U with the full scalar value (not a
    UTF-16 surrogate pair).
    """
    print("ord('A')         =", ord("A"))
    print("chr(0x1F44B)     =", chr(0x1F44B))   # wave hand
    print("Codepoint.ord(.) =", Int(Codepoint.ord("€")))
    print("escape u20AC     =", "€")
    print("escape U0001F44B =", "\U0001F44B")


def section_12_parsing_and_atox() raises:
    """Parsing helpers atol and atof are in the prelude.

    - `atol(s, base=10)`: parses an int. base=0 enables 0b/0o/0x
      prefix sniffing. Underscores between digits are allowed.
      Leading and trailing ASCII whitespace is trimmed.
    - `atof(s)`: parses a float64.

    Both raise on malformed input. They take StringSlice, so they
    work on String, StaticString, and any other slice without
    copying.

    For the reverse direction, all primitive numeric types are
    Writable, so you stream them directly into a String or a
    t-string:

        var n: Int = 42
        print(t\"answer={n}\")
        var s = String(n)
    """
    print("atol('  255 '):       ", atol("  255 "))
    print("atol('0xff', base=0): ", atol("0xff", base=0))
    print("atol('1_000_000'):    ", atol("1_000_000"))
    print("atof('3.14'):         ", atof("3.14"))
    print("atof('1e-9'):         ", atof("1e-9"))


def section_13_teaching_summary():
    """A short instructional summary - what to tell someone new to Mojo strings.

    1. Pick the cheapest type. If the bytes are known at compile
       time use StringLiteral / StaticString; if you only need to
       look at bytes someone else owns, use StringSlice; only reach
       for String when you actually own the bytes or need to mutate
       them. The compiler will materialize literals to whatever the
       context demands.

    2. Build output through Writable, not through String
       concatenation. Implement write_to for your types, accept
       writers in helpers, and let print() / loggers / file sinks
       consume Writables directly. This is the single biggest
       performance lever in Mojo string code.

    3. Prefer t-strings over .format(). T-strings are checked at
       compile time, contain real expressions (`t\"{a + b}\"`),
       nest cleanly, and don't allocate when you stream them.
       Reach for .format() only when the template itself must be
       chosen at runtime.

    4. Use the right length. `byte_length()` is O(1) and is what
       you want for buffer math. `count_codepoints()` and
       `count_graphemes()` are O(n); `count_graphemes()` is what
       agrees with human intuition for \"how many characters\".

    5. Respect origins. A StringSlice is tied to its source by an
       origin parameter. Return slices whose origin matches the
       input (`StringSlice[s.origin]`) so the borrow checker can
       protect downstream callers.

    6. For NUMA-aware code: short Strings are inline (no heap, no
       atomic refcount) - they live on the calling thread's stack
       and travel with it. Long Strings carry a shared atomic
       refcount; copy them across NUMA domains the same way you'd
       copy any other shared-atomic structure - sparingly, and
       prefer to construct them where they will be read.
    """
    pass


def main() raises:
    print("=== section 1: the four string types ===")
    section_1_the_four_string_types()
    print()

    print("=== section 2: String storage modes ===")
    section_2_string_storage_modes()
    print()

    print("=== section 3: Writable and Writer ===")
    section_3_writable_and_writer()
    print()

    print("=== section 4: bytes vs codepoints vs graphemes ===")
    section_4_byte_codepoint_grapheme()
    print()

    print("=== section 5: origins keep slices safe ===")
    section_5_origins_keep_slices_safe()
    print()

    print("=== section 6: format vs t-string ===")
    section_6_format_vs_tstring()
    print()

    print("=== section 7: t-string internals ===")
    section_7_tstring_internals()
    print()

    print("=== section 8: t-string is lazy / no-alloc ===")
    section_8_tstring_lazy_no_alloc()
    print()

    print("=== section 9: nested and raw t-strings ===")
    section_9_tstring_nested_and_raw()
    print()

    print("=== section 10: t-string in write_to ===")
    section_10_tstring_as_writable_field()
    print()

    print("=== section 11: Codepoint, chr, ord ===")
    section_11_codepoints_and_chr_ord()
    print()

    print("=== section 12: atol and atof ===")
    section_12_parsing_and_atox()
    print()

    print("=== section 13: teaching summary ===")
    section_13_teaching_summary()
