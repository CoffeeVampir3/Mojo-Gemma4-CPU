from std.reflection import reflect


trait ArgLike(Copyable, Movable, Defaultable, ImplicitlyDestructible):
    @staticmethod
    def parse(token: String) -> Optional[Self]: ...

    @staticmethod
    def usage() -> String: ...


trait Command(Copyable, Movable, Defaultable, ImplicitlyDestructible):
    @staticmethod
    def keys() -> List[String]: ...

    def run(self): ...


struct PositiveFloat(ArgLike, Copyable, Movable):
    var value: Float32

    def __init__(out self):
        self.value = 0.0

    def __init__(out self, value: Float32):
        self.value = value

    @staticmethod
    def parse(token: String) -> Optional[Self]:
        try:
            var v = Float32(atof(token))
            if v > 0.0:
                return Self(v)
        except:
            pass
        print(t"  '{token}' must be a number > 0")
        return None

    @staticmethod
    def usage() -> String:
        return String("<float > 0>")


struct UnitFloat(ArgLike, Copyable, Movable):
    var value: Float32

    def __init__(out self):
        self.value = 0.0

    def __init__(out self, value: Float32):
        self.value = value

    @staticmethod
    def parse(token: String) -> Optional[Self]:
        try:
            var v = Float32(atof(token))
            if v >= 0.0 and v < 1.0:
                return Self(v)
        except:
            pass
        print(t"  '{token}' must be in [0, 1)")
        return None

    @staticmethod
    def usage() -> String:
        return String("<float in [0, 1)>")


struct BoundedInt[hi: Int](ArgLike, Copyable, Movable):
    var value: Int

    def __init__(out self):
        self.value = 0

    def __init__(out self, value: Int):
        self.value = value

    @staticmethod
    def parse(token: String) -> Optional[Self]:
        try:
            var v = atol(token)
            if v >= 0 and v <= Self.hi:
                return Self(v)
        except:
            pass
        print(t"  '{token}' must be an int in [0, {Self.hi}]")
        return None

    @staticmethod
    def usage() -> String:
        return String(t"<int 0..{Self.hi}>")


struct GreedyCmd(Command, Copyable, Movable):
    def __init__(out self):
        pass

    @staticmethod
    def keys() -> List[String]:
        return [String("/greedy")]

    def run(self):
        print("  sampler -> greedy")


struct TempCmd(Command, Copyable, Movable):
    var temp: PositiveFloat

    def __init__(out self):
        self.temp = PositiveFloat()

    @staticmethod
    def keys() -> List[String]:
        return [String("/temp")]

    def run(self):
        print(t"  temperature -> {self.temp.value}")


struct MinPCmd(Command, Copyable, Movable):
    var min_p: UnitFloat

    def __init__(out self):
        self.min_p = UnitFloat()

    @staticmethod
    def keys() -> List[String]:
        return [String("/min-p"), String("/min_p")]

    def run(self):
        print(t"  min_p -> {self.min_p.value}")


struct TopKCmd(Command, Copyable, Movable):
    var k: BoundedInt[256]

    def __init__(out self):
        self.k = BoundedInt[256]()

    @staticmethod
    def keys() -> List[String]:
        return [String("/top-k"), String("/top_k")]

    def run(self):
        print(t"  top_k -> {self.k.value}")


struct Registry:
    var greedy: GreedyCmd
    var temp: TempCmd
    var min_p: MinPCmd
    var top_k: TopKCmd


def has_key[C: Command](cmd: String) -> Bool:
    for k in C.keys():
        if k == cmd:
            return True
    return False


def fill[C: Command](read parts: List[String]) -> Optional[C]:
    var cmd = C()
    comptime r = reflect[C]
    comptime types = r.field_types()
    var pos = 1
    var ok = True
    comptime for i in range(r.field_count()):
        comptime AT = types[i]
        comptime if conforms_to(AT, ArgLike):
            if ok and pos < len(parts):
                var got = AT.parse(String(parts[pos]))
                pos += 1
                if got:
                    r.field_ref[i](cmd) = got.value().copy()
                else:
                    ok = False
            elif ok:
                print(t"  missing arg: {AT.usage()}")
                ok = False
    if not ok:
        return None
    return cmd^


def dispatch[Reg: AnyType](read parts: List[String]) -> Bool:
    if len(parts) == 0:
        return False
    comptime r = reflect[Reg]
    comptime types = r.field_types()
    var handled = False
    comptime for i in range(r.field_count()):
        comptime CT = types[i]
        comptime if conforms_to(CT, Command):
            if not handled and has_key[CT](parts[0]):
                var c = fill[CT](parts)
                if c:
                    c.value().run()
                handled = True
    return handled


def main():
    print(dispatch[Registry]([String("/temp"), String("0.7")]))
    print(dispatch[Registry]([String("/min-p"), String("0.05")]))
    print(dispatch[Registry]([String("/top-k"), String("100")]))
    print(dispatch[Registry]([String("/top_k"), String("999")]))
    print(dispatch[Registry]([String("/greedy")]))
    print(dispatch[Registry]([String("/temp")]))
    print(dispatch[Registry]([String("/nope")]))
