from std.reflection import reflect


struct Phantom[name_: StaticString](Copyable, ImplicitlyCopyable):
    var offset: Int

    @implicit
    def __init__(out self, offset: Int):
        self.offset = offset


@fieldwise_init
struct Inner(Copyable, ImplicitlyCopyable):
    var a: Phantom["inner.a"]
    var b: Phantom["inner.b"]


@fieldwise_init
struct Outer(Copyable, ImplicitlyCopyable):
    var inner: Inner
    var c: Phantom["outer.c"]


def main():
    var t = Outer(Inner(-1, -1), -1)
    print("inner.a =", t.inner.a.offset, " inner.b =", t.inner.b.offset, " c =", t.c.offset)
