from std.memory import Span
from std.pathlib import Path

from numa import NumaTopology
from threading.threading_traits import BurstThreadPool, SleepableThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from tokenizer import load_tokenizer, BPETokenizer, AutoPreTokenizer, AutoByteTransform
from modeling.gemma_4_moe import Gemma4
from modeling.gemma4_common import Gemma4BaseConfig
from modeling.slider_pack import load_pack, SliderBank
from kernels.flash_sample import SamplingParams
from continuous_batching.schedule import MAXIMUM_SAMPLING_LOGITS
from continuous_batching.scheduler import ContinuousBatchScheduler


comptime C = Gemma4BaseConfig
comptime TOKENIZER_PATH = "checkpoints/gemma-4-26B-A4B-it/tokenizer.json"
comptime MODEL_DIR = "checkpoints/gemma-4-26B-A4B-it"
comptime PACK_PATH = "sliders/ocean.json"
comptime BOS_TOKEN_ID = 2
comptime TURN_START_TOKEN_ID = 105
comptime TURN_END_TOKEN_ID = 106
comptime STEP_BUDGET = Gemma4BaseConfig.SLIDING_WINDOW
comptime MAX_REPLY_TOKENS = 256


struct StdinReader(Movable):
    var buf: List[Byte]
    var pos: Int
    var eof: Bool

    def __init__(out self):
        self.buf = List[Byte]()
        self.pos = 0
        self.eof = False

    def slice_from(self, start: Int, end: Int) -> String:
        return String(unsafe_from_utf8=Span(self.buf).unsafe_subspan(
            offset=start, length=end - start))

    def read_line(mut self) -> Optional[String]:
        while True:
            for i in range(self.pos, len(self.buf)):
                if self.buf[i] == Byte(10):
                    var line = self.slice_from(self.pos, i)
                    self.pos = i + 1
                    return line^
            if self.eof:
                if self.pos < len(self.buf):
                    var line = self.slice_from(self.pos, len(self.buf))
                    self.pos = len(self.buf)
                    return line^
                return None
            if self.pos > 0:
                var rem = List[Byte]()
                for i in range(self.pos, len(self.buf)):
                    rem.append(self.buf[i])
                self.buf = rem^
                self.pos = 0
            var chunk = List[Byte](length=4096, fill=0)
            var got: Int
            try:
                var fd = FileDescriptor(0)
                got = Int(fd.read_bytes(Span(chunk)))
            except:
                got = 0
            if got == 0:
                self.eof = True
            else:
                for i in range(got):
                    self.buf.append(chunk[i])


def append_encoded(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    mut history: List[Int32],
    read text: String,
):
    var enc = tok.encode(text)
    for i in range(len(enc)):
        history.append(Int32(enc[i]))


def append_user_turn(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    mut history: List[Int32],
    read message: String,
):
    history.append(Int32(TURN_START_TOKEN_ID))
    append_encoded(tok, history, "user\n" + message)
    history.append(Int32(TURN_END_TOKEN_ID))
    append_encoded(tok, history, "\n")
    history.append(Int32(TURN_START_TOKEN_ID))
    append_encoded(tok, history, "model\n")


def decode_response(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read gen: List[Int32],
) -> String:
    var ids = List[Int]()
    for i in range(len(gen)):
        if i == len(gen) - 1 and gen[i] == Int32(TURN_END_TOKEN_ID):
            continue
        ids.append(Int(gen[i]))
    return tok.decode(ids)


def print_help():
    print("commands:")
    print("  /<trait> <value>   set a trait, value in [-1, 1] (0 = off)")
    print("  /sliders           list traits and current doses")
    print("  /reset             clear the conversation context")
    print("  /help              show this help")
    print("  /quit              exit")


def print_sliders(read bank: SliderBank):
    if bank.count() == 0:
        print("  (no sliders loaded)")
        return
    for i in range(bank.count()):
        var cfg = bank.configs[i]
        print(t"  {bank.names[i]}: alpha {bank.alphas[i]} "
              t"(layer {cfg.layer}, corridor [{cfg.alpha_min}, "
              t"{cfg.alpha_max}])")


def park[T: SleepableThreadPool, //](mut pool: T):
    pool.sleep()


def unpark[T: SleepableThreadPool, //](mut pool: T):
    pool.wake()


def run[
    P: BurstThreadPool, //,
](
    topo: NumaTopology,
    var pools: List[P],
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
):
    var model_opt = Gemma4[Pool=P].load(Path(MODEL_DIR), topo, pools^)
    if not model_opt:
        print("model load failed")
        return
    var model = model_opt.take()
    print(t"loaded (degree {model.degree})")

    var bank_opt = load_pack(model, PACK_PATH)
    var bank: SliderBank
    if bank_opt:
        bank = bank_opt.take()
        print(t"loaded {bank.count()} slider(s) from {PACK_PATH}")
    else:
        bank = SliderBank()
        print(t"no pack at {PACK_PATH} (chat unsteered)")
    bank.apply(model)

    var greedy = SamplingParams(
        Float32(1.0), Float32(0.0), 0, MAXIMUM_SAMPLING_LOGITS, True)
    var sched = ContinuousBatchScheduler[
        Gemma4[Pool=P].POSITIONS_PER_PAGE,
    ](model.batch_geometry(), STEP_BUDGET, Int32(TURN_END_TOKEN_ID))

    print()
    print_help()
    print_sliders(bank)

    var history = List[Int32]()
    history.append(Int32(BOS_TOKEN_ID))
    var stdin = StdinReader()

    while True:
        for i in range(len(model.pools)):
            park(model.pools[i])
        print("\nyou> ", end="", flush=True)
        var line_opt = stdin.read_line()
        if not line_opt:
            print()
            break
        for i in range(len(model.pools)):
            unpark(model.pools[i])
        var s = String(line_opt.value().strip())
        if s.byte_length() == 0:
            continue

        if s.startswith("/"):
            var parts = s.split()
            var cmd = String(parts[0])
            if cmd == "/quit" or cmd == "/exit" or cmd == "/q":
                break
            elif cmd == "/help" or cmd == "/h":
                print_help()
            elif cmd == "/reset":
                history = List[Int32]()
                history.append(Int32(BOS_TOKEN_ID))
                print("  context cleared")
            elif cmd == "/sliders" or cmd == "/list":
                print_sliders(bank)
            else:
                var matched = -1
                for i in range(bank.count()):
                    if cmd == String("/") + bank.names[i]:
                        matched = i
                if matched < 0:
                    print(t"  unknown command {cmd} (try /help)")
                elif len(parts) < 2:
                    print("  usage: /<trait> <value in [-1, 1]>")
                else:
                    var val = Float32(0)
                    var bad = False
                    try:
                        val = Float32(atof(String(parts[1])))
                    except:
                        bad = True
                    if bad:
                        print(t"  '{String(parts[1])}' is not a number")
                    else:
                        bank.set_position(matched, val)
                        bank.apply(model)
                        print(t"  {bank.names[matched]} -> position {val}, "
                              t"alpha {bank.alphas[matched]} at layer "
                              t"{bank.configs[matched].layer}")
            continue

        var pre_len = len(history)
        append_user_turn(tok, history, s)
        var rid_opt = sched.submit(history.copy(), greedy, MAX_REPLY_TOKENS)
        if not rid_opt:
            print("  (context full — use /reset to clear)")
            while len(history) > pre_len:
                _ = history.pop()
            continue
        var rid = rid_opt.value()

        var guard = 0
        var stalled = False
        while not sched.requests[rid].done:
            guard += 1
            if guard > 8 * MAX_REPLY_TOKENS:
                print("  (generation stalled)")
                stalled = True
                break
            if sched.step(model) == 0:
                print("  (scheduler stalled)")
                stalled = True
                break

        if stalled:
            _ = sched.retire(rid)
            while len(history) > pre_len:
                _ = history.pop()
            continue

        ref gen = sched.requests[rid].generated
        print(t"model> {decode_response(tok, gen)}")
        for i in range(len(gen)):
            history.append(gen[i])
        append_encoded(tok, history, "\n")
        _ = sched.retire(rid)

    model.steer.disarm()
    print("bye")


def main():
    print("gemma personality chat")
    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print(t"failed to load tokenizer from {TOKENIZER_PATH}")
        return
    var tok = tok_opt.take()

    var topo = NumaTopology()
    var nodes = topo.num_nodes()
    print(t"{nodes} NUMA nodes")

    @parameter
    def dispatch_tp[
        P: BurstThreadPool, //,
    ](var selected_pools: List[P]):
        run(topo, selected_pools^, tok)

    with_topological_rank_dispatch[
        dispatch=dispatch_tp,
    ](
        topo, "mode: isolated (spin-only)", "mode: cold (spin-backoff)")
