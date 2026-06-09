from kernels.flash_sample import SamplingParams, SampleOutcome

from .paging import KVPageAccountant, BatchGeometry


comptime MAXIMUM_SAMPLING_LOGITS = 64


@fieldwise_init
struct BatchSlot(Copyable, Movable, ImplicitlyCopyable):
    var seq_id: Int
    var base_pos: Int
    var n_tokens: Int
    var emit: Bool
    var sampling: SamplingParams


struct Schedule(Movable):
    var slots: List[BatchSlot]
    var tokens: List[Int32]

    def __init__(out self):
        self.slots = List[BatchSlot]()
        self.tokens = List[Int32]()

    def clear(mut self):
        self.slots.clear()
        self.tokens.clear()


trait ScheduledModel:
    comptime POSITIONS_PER_PAGE: Int

    def batch_geometry(self) -> BatchGeometry: ...

    def execute(
        mut self,
        read schedule: Schedule,
        read pages: KVPageAccountant,
    ) -> List[SampleOutcome[MAXIMUM_SAMPLING_LOGITS]]: ...
