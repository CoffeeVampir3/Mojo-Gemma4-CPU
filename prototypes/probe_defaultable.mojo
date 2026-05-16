from std.reflection import reflect


struct Phantom[name_: StaticString](Copyable, ImplicitlyCopyable):
    def __init__(out self):
        pass


# Probe: reflection over comptime members
struct Inner(Copyable, ImplicitlyCopyable):
    comptime a = Phantom["a"]()
    comptime b = Phantom["b"]()
    comptime c = Phantom["c"]()

    def __init__(out self):
        pass


def main():
    print("field_count:", reflect[Inner].field_count())

    comptime for i in range(reflect[Inner].field_count()):
        print("  field i =", i, "name =", reflect[Inner].field_names()[i])
