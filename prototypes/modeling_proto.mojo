from std.sys.info import CompilationTarget
from experimental3.amx import init_intel_amx


trait Backend:
    @staticmethod
    def is_available() -> Bool: ...


trait AMXBackend(Backend):
    pass


trait VNNIBackend(Backend):
    pass


struct AMX(AMXBackend):
    @staticmethod
    def is_available() -> Bool:
        return CompilationTarget.has_intel_amx()

struct VNNI(VNNIBackend):
    @staticmethod
    def is_available() -> Bool:
        return CompilationTarget.has_vnni()


def dispatch_kernel[B: AMXBackend]():
    print("hello world from amx")


def dispatch_kernel[B: VNNIBackend]():
    print("hello world from vnni")


def main():
    print("amx available:", AMX.is_available())
    print("vnni available:", VNNI.is_available())
    dispatch_kernel[AMX]()
    dispatch_kernel[VNNI]()
