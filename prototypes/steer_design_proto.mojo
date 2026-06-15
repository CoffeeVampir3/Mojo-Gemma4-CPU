comptime HIDDEN = 4
comptime DEFAULT_STEER_VECTORS = 16


@fieldwise_init
struct InjectOp(Copyable, Movable, ImplicitlyCopyable):
    var layer: Int
    var alpha: Int


struct SteerState(Movable):
    var armed: Bool
    var ops: List[InjectOp]

    def __init__(out self):
        self.armed = False
        self.ops = List[InjectOp]()

    def set_inject(mut self, var ops: List[InjectOp]):
        self.ops = ops^
        self.armed = True

    def disarm(mut self):
        self.armed = False


# Part A: trait interface for steering upload. STEER_VECTORS exposes the
# model's compile-time slab capacity so consumers can validate fit.
trait Steerable:
    comptime STEER_VECTORS: Int
    def set_steer_vector(mut self, idx: Int, val: Int): ...
    def set_inject_ops(mut self, var ops: List[InjectOp]): ...
    def disarm_steer(mut self): ...


# Part B: conditional-size slab threaded through a layout type.
@fieldwise_init
struct SlabSlots[n_vectors: Int](Copyable, Movable):
    var elems: Int

    def __init__(out self):
        self.elems = Self.n_vectors * HIDDEN


@fieldwise_init
struct Layout[steer_vectors: Int](Copyable, Movable):
    var steer: SlabSlots[Self.steer_vectors]


def build_layout[steer_vectors: Int]() -> Layout[steer_vectors]:
    return Layout[steer_vectors](SlabSlots[steer_vectors]())


# Infer-only param threads the steer size with no call-site churn.
def steer_bytes[steer_vectors: Int, //](read layout: Layout[steer_vectors]) -> Int:
    return layout.steer.elems


struct Model[steer_vectors: Int = 0](Steerable):
    comptime STEER_VECTORS = Self.steer_vectors
    var layout: Layout[Self.steer_vectors]
    var steer: SteerState
    var slab: List[Int]

    def __init__(out self):
        self.layout = build_layout[Self.steer_vectors]()
        self.steer = SteerState()
        self.slab = List[Int](length=Self.steer_vectors * HIDDEN, fill=0)

    def set_steer_vector(mut self, idx: Int, val: Int):
        comptime if Self.steer_vectors > 0:
            self.slab[idx * HIDDEN] = val

    def set_inject_ops(mut self, var ops: List[InjectOp]):
        self.steer.set_inject(ops^)

    def disarm_steer(mut self):
        self.steer.disarm()

    def run(mut self) -> Int:
        comptime if Self.steer_vectors > 0:
            if self.steer.armed:
                return 1
        return 0


# A structurally different model that also implements the interface.
struct OtherModel(Steerable):
    comptime STEER_VECTORS = 0
    var steer: SteerState

    def __init__(out self):
        self.steer = SteerState()

    def set_steer_vector(mut self, idx: Int, val: Int):
        pass

    def set_inject_ops(mut self, var ops: List[InjectOp]):
        self.steer.set_inject(ops^)

    def disarm_steer(mut self):
        self.steer.disarm()


# Generic consumer over the trait — validates the pack fits the model's
# compile-time capacity, then uploads. Never names a concrete model.
def upload[M: Steerable, //](mut model: M, count: Int) -> Bool:
    if count > M.STEER_VECTORS:
        return False
    for i in range(count):
        model.set_steer_vector(i, 7)
    var ops = List[InjectOp]()
    ops.append(InjectOp(0, 3))
    model.set_inject_ops(ops^)
    return True


def main():
    var on = Model[steer_vectors=DEFAULT_STEER_VECTORS]()
    var ok = upload(on, 3)
    print("on    bytes:", steer_bytes(on.layout), "cap:", on.STEER_VECTORS,
          "ok:", ok, "armed:", on.steer.armed, "run:", on.run())

    var small = Model[steer_vectors=4]()
    var overflow = upload(small, 9)
    print("small bytes:", steer_bytes(small.layout), "cap:", small.STEER_VECTORS,
          "overflow_rejected:", not overflow)

    var off = Model[steer_vectors=0]()
    print("off   bytes:", steer_bytes(off.layout), "run:", off.run())

    var other = OtherModel()
    _ = upload(other, 0)
    print("other armed:", other.steer.armed)
