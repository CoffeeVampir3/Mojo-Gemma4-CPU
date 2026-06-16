from std.memory import UnsafePointer

from numa import NumaArena, NumaTopology
from modeling.model_spec import DEFAULT_ALIGNMENT


@fieldwise_init
struct AbliterationParameters(Copyable, Movable, ImplicitlyCopyable):
    var max_weight: Float32
    var max_weight_position: Float32
    var min_weight: Float32
    var min_weight_distance: Float32

    @always_inline
    def strength(self, layer: Int) -> Float32:
        var distance = abs(Float32(layer) - self.max_weight_position)
        if distance > self.min_weight_distance:
            return Float32(0)
        if self.min_weight_distance <= Float32(0):
            return self.max_weight
        return self.max_weight + (distance / self.min_weight_distance) * (
            self.min_weight - self.max_weight)


struct AbliterateWorkspace(Movable):
    """Per-rank f32 scratch for the norm-preserving edit, NUMA-bound to mirror
    the model's per-rank arenas. Each rank's block holds `v`, `m`, `a`
    (each `rows`) followed by `p` (`pmax`) at identical intra-block offsets, so
    a `RankView` over `bases` resolves rank r by constant per-rank delta and the
    existing allreduce reduces `m`/`a` across ranks. The block is independent of
    the model arena: directions are an external artifact and this scratch is
    transient edit workspace, reused across optimizer trials."""
    var arenas: List[NumaArena[alignment=DEFAULT_ALIGNMENT]]
    var bases: List[Int]
    var rows: Int
    var pmax: Int

    def __init__(out self, topo: NumaTopology, degree: Int, rows: Int, pmax: Int):
        self.rows = rows
        self.pmax = pmax
        self.arenas = List[NumaArena[alignment=DEFAULT_ALIGNMENT]]()
        self.bases = List[Int]()
        var count = 3 * rows + pmax
        var size = count * 4
        for r in range(degree):
            var arena = NumaArena[alignment=DEFAULT_ALIGNMENT](topo.node(r), size)
            var blk = arena.alloc[Float32](count)
            if not blk:
                self.bases.append(0)
            else:
                self.bases.append(Int(blk.value()))
            _ = arena.prefault()
            self.arenas.append(arena^)

    @always_inline
    def ok(self) -> Bool:
        for r in range(len(self.bases)):
            if self.bases[r] == 0:
                return False
        return len(self.bases) > 0

    @always_inline
    def v_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.bases[0])

    @always_inline
    def m_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.bases[0] + self.rows * 4)

    @always_inline
    def a_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.bases[0] + 2 * self.rows * 4)

    @always_inline
    def p_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=self.bases[0] + 3 * self.rows * 4)


trait AbliterableModel:
    def restore_abliterated_layers(mut self): ...

    def abliterate_layers(
        mut self,
        read directions: List[BFloat16],
        read attn: AbliterationParameters,
        read down: AbliterationParameters,
        mut ws: AbliterateWorkspace,
    ): ...
