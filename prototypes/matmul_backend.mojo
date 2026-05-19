from std.sys import CompilationTarget


trait MatmulBackend:
    @staticmethod
    def supported() -> Bool: ...


trait AvxBackend(MatmulBackend): pass
trait AmxBackend(MatmulBackend): pass
trait VnniBackend(MatmulBackend): pass


struct Avx(AvxBackend):
    @staticmethod
    def supported() -> Bool:
        return True


struct Amx(AmxBackend):
    @staticmethod
    def supported() -> Bool:
        return (CompilationTarget._has_feature["amx-bf16"]()
                or CompilationTarget._has_feature["amx-int8"]())


struct Vnni(VnniBackend):
    @staticmethod
    def supported() -> Bool:
        return (CompilationTarget._has_feature["avx512vnni"]()
                or CompilationTarget._has_feature["avxvnni"]())


def dispatch_matmul[B: AvxBackend](m: Int, n: Int, k: Int):
    print("  avx path: m=", m, "n=", n, "k=", k)


def dispatch_matmul[B: AmxBackend](m: Int, n: Int, k: Int):
    print("  amx path: m=", m, "n=", n, "k=", k)


def dispatch_matmul[B: VnniBackend](m: Int, n: Int, k: Int):
    print("  vnni path: m=", m, "n=", n, "k=", k)


def main():
    print("Avx.supported():", Avx.supported())
    print("Amx.supported():", Amx.supported())
    print("Vnni.supported():", Vnni.supported())
    print()
    dispatch_matmul[Avx](16, 64, 128)
    dispatch_matmul[Amx](16, 64, 128)
    dispatch_matmul[Vnni](16, 64, 128)
