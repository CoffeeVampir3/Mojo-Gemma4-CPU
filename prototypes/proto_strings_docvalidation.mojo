from std.collections.string import Codepoint
from std.collections.string.string_slice import StaticString, StringSlice
from std.ffi import CStringSlice


@fieldwise_init
struct Dog(Copyable, Writable):
    var name: String
    var age: Int

    def write_to(self, mut writer: Some[Writer]):
        t"Dog({self.name}, {self.age})".write_to(writer)

    def write_repr_to(self, mut writer: Some[Writer]):
        t"Dog(name={self.name}, age={self.age})".write_to(writer)


@fieldwise_init
struct Tensor(Copyable, Writable):
    var name: String
    var shape: List[Int]

    def write_to(self, mut writer: Some[Writer]):
        t"Tensor({self.name}, shape=[".write_to(writer)
        for i in range(len(self.shape)):
            if i:
                ", ".write_to(writer)
            String(self.shape[i]).write_to(writer)
        t"])".write_to(writer)


def first_word(s: StringSlice) -> StringSlice[s.origin]:
    var i = s.find(" ")
    return s if i < 0 else s[byte=0:i]


def validate_string_constructors() raises:
    var a = String()
    var b = String(capacity=128)
    var c = String(unsafe_uninit_length=4)
    var d = String("a", 1, ", ", 3.14, sep="")
    var e = String(t"hello {a}")
    var f = String(copy=d)
    var g = String.write(1, " ", 2.0, sep="", end="\n")
    var h = String.write(42)
    var bytes: List[Byte] = [72, 105]
    var i_valid = String(from_utf8=Span(bytes))
    var i_lossy = String(from_utf8_lossy=Span(bytes))
    var i_unsafe = String(unsafe_from_utf8=Span(bytes))
    print(
        "constructors OK:",
        d.byte_length(),
        e.byte_length(),
        g.byte_length(),
        i_valid,
        i_lossy,
        i_unsafe,
    )
    _ = a
    _ = b
    _ = c
    _ = f
    _ = h


def validate_indexing_and_lengths():
    var s = String("Hello, 世界")
    print("byte_length =", s.byte_length())
    print("count_codepoints =", s.count_codepoints())
    print("count_graphemes =", s.count_graphemes())
    print("len(codepoints) =", len(s.codepoints()))
    print("b[0]  =", s[byte=0])
    print("b[0:5] =", s[byte=0:5])

    var slc = StringSlice(s)
    print("is_codepoint_boundary(0) =", slc.is_codepoint_boundary(0))
    print("is_codepoint_boundary(7) =", slc.is_codepoint_boundary(7))
    print("cp[0]  =", slc[codepoint=0])
    print("gp[0:2] =", slc[grapheme=0:2])


def validate_iteration():
    var s = String("café")
    for cp in s.codepoints():
        print("cp:", Int(cp))
    for sl in s.codepoint_slices():
        print("slc:", sl)
    for sl in s.codepoint_slices_reversed():
        print("rev slc:", sl)
    for g in s.graphemes():
        print("gr:", g)
    for g in s.graphemes_reversed():
        print("rev gr:", g)
    for off, g in s.grapheme_indices():
        print("at", off, ":", g)
    var fourth = s.nth_grapheme(3)
    if fourth:
        print("4th grapheme:", fourth.value())


def validate_split_join_strip():
    var s = String("  hello,  world  ")
    var parts = s.split(",")
    print("split:", parts[0], "|", parts[1])
    var ws = s.split()
    print("ws split count:", len(ws))
    var stripped = s.strip()
    print("strip:", stripped)
    var joined = ", ".join(parts)
    print("joined:", joined)
    var lines = String("a\nb\r\nc").splitlines()
    print("lines:", len(lines))


def validate_search_and_replace():
    var s = String("Hello World")
    print("find:", s.find("World"))
    print("rfind:", s.rfind("l"))
    print("count:", s.count("l"))
    print("contains:", "World" in s)
    print("starts:", s.startswith("Hello"))
    print("ends:", s.endswith("World"))
    print("removeprefix:", s.removeprefix("Hello "))
    print("replace:", s.replace("World", "Mojo"))


def validate_writable_struct():
    var d = Dog("Rex", 5)
    print(d)
    print(repr(d))
    var t = Tensor("Q", [32, 128, 64])
    print(t)

    # Stream into a String through Writer
    var buf = String()
    t.write_to(buf)
    buf.write(" ", d)
    print(buf)


def validate_tstring():
    var x = 41
    var y = 1
    var nums = [10, 20, 30]
    print(t"answer = {x + y}")
    print(t"{nums[0] + nums[1]}")
    print(t"Use {{braces}} around {x}")

    var name = "world"
    print(t"Hello, {t"dear {name}"}!")

    var base = "/home/user"
    print(rt"Path: {base}\subdir\file")

    var collapsed = String(t"snapshot: x={x}, y={y}")
    print(collapsed)

    # Triple-quoted t-string
    print(t"""
multi
line t-string with {name}
""")


def validate_codepoint() raises:
    var cp = Codepoint.ord("a")
    print("ord('a'):", Int(cp))
    var maybe = Codepoint.from_u32(0x1F44B)
    if maybe:
        var c = maybe.value()
        print("0x1F44B utf8 width:", c.utf8_byte_length())
    var b = Codepoint(UInt8(65))
    print("byte 65:", Int(b))
    print("is_ascii:", cp.is_ascii(), "is_digit:", cp.is_ascii_digit())


def validate_cstring_slice() raises:
    var s = String("hello")
    var cs = s.as_c_string_slice()
    print("cs len:", len(cs))
    var view = StringSlice(unsafe_from_utf8_ptr=cs.unsafe_ptr())
    print("via StringSlice:", view)


def validate_static_and_literal():
    var sl: StaticString = "compile-time bytes"
    print(sl)
    var head = first_word("the quick brown fox")
    print("head:", head)
    var owned = String("hello world")
    var head2 = first_word(owned)
    print("head2:", head2)

    var fmt = "{0} {1} {0}".format("Mojo", 1.125)
    print(fmt)


def validate_prelude_helpers() raises:
    print("atol:", atol("  255 "))
    print("atol 0x:", atol("0xff", base=0))
    print("atof:", atof("3.14"))
    print("chr:", chr(0x1F44B))
    print("ord:", ord("Z"))


def main() raises:
    validate_string_constructors()
    print("---")
    validate_indexing_and_lengths()
    print("---")
    validate_iteration()
    print("---")
    validate_split_join_strip()
    print("---")
    validate_search_and_replace()
    print("---")
    validate_writable_struct()
    print("---")
    validate_tstring()
    print("---")
    validate_codepoint()
    print("---")
    validate_cstring_slice()
    print("---")
    validate_static_and_literal()
    print("---")
    validate_prelude_helpers()
