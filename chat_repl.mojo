from std.memory import Span
from std.pathlib import Path

from numa import NumaTopology
from threading.threading_traits import BurstThreadPool, SleepableThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from tokenizer import (
    load_tokenizer, BPETokenizer, AutoPreTokenizer, AutoByteTransform,
    StreamDetokenizer,
)
from modeling_config import (
    Model, TOKENIZER_PATH, MODEL_DIR, stop_tokens,
    BOS_TOKEN_ID, TURN_START_TOKEN_ID, TURN_END_TOKEN_ID,
)
from modeling.gemma4_common import Gemma4BaseConfig
from modeling.slider_pack import load_pack, SliderBank
from kernels.flash_sample import SamplingParams
from continuous_batching.schedule import MAXIMUM_SAMPLING_LOGITS
from continuous_batching.scheduler import ContinuousBatchScheduler


comptime PACK_PATH = "sliders/ocean.json"
comptime STEP_BUDGET = Gemma4BaseConfig.SLIDING_WINDOW
comptime MAX_CONTEXT = 65536
comptime CLEAR_SCREEN = "\x1b[2J\x1b[3J\x1b[H"
comptime YOU_PROMPT = "\n\n\x1b[32myou> \x1b[0m"
comptime MODEL_PROMPT = "\x1b[38;5;211mmodel> \x1b[0m"
comptime CYAN = "\x1b[36m"
comptime RESET = "\x1b[0m"


@fieldwise_init
struct ChatTurn(Copyable, Movable):
    var user: String
    var reply: String


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


def print_help():
    print("commands:")
    print("  /<trait> <value>   set a trait; [-1, 1] is the safe range, "
          "|value| > 1 overdrives past the corridor (0 = off)")
    print("  /sliders           list traits and current doses")
    print("  /reset             clear the conversation context")
    print("  /rewind            undo the last turn (you + model)")
    print("  /retry             undo the last turn without clearing the screen")
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


def render_chat(read transcript: List[ChatTurn]):
    print(CLEAR_SCREEN, end="", flush=True)
    print("gemma personality chat")
    for i in range(len(transcript)):
        print(YOU_PROMPT + transcript[i].user)
        print(MODEL_PROMPT + transcript[i].reply)


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
    var model_opt = Model[steer_vectors=16, max_seq_len=MAX_CONTEXT, batching_seq_len=MAX_CONTEXT, Pool=P].load(Path(MODEL_DIR), topo, pools^)
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
        Model[steer_vectors=16, max_seq_len=MAX_CONTEXT, batching_seq_len=MAX_CONTEXT, Pool=P].POSITIONS_PER_PAGE,
    ](model.batch_geometry(), STEP_BUDGET, stop_tokens())

    print()
    print_help()
    print_sliders(bank)

    var history = List[Int32]()
    history.append(Int32(BOS_TOKEN_ID))
    var turn_starts = List[Int]()
    var transcript = List[ChatTurn]()
    var stdin = StdinReader()

    while True:
        for i in range(len(model.pools)):
            park(model.pools[i])
        print(YOU_PROMPT, end="", flush=True)
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
                turn_starts = List[Int]()
                transcript = List[ChatTurn]()
                render_chat(transcript)
                print("  context cleared")
            elif cmd == "/rewind" or cmd == "/undo":
                if len(turn_starts) == 0:
                    print("  nothing to rewind")
                else:
                    var mark = turn_starts.pop()
                    while len(history) > mark:
                        _ = history.pop()
                    _ = transcript.pop()
                    render_chat(transcript)
                    print(t"  rewound last turn ({len(turn_starts)} left)")
            elif cmd == "/retry" or cmd == "/regen":
                if len(turn_starts) == 0:
                    print("  nothing to retry")
                else:
                    var mark = turn_starts.pop()
                    while len(history) > mark:
                        _ = history.pop()
                    _ = transcript.pop()
                    print(t"  rewound last turn ({len(turn_starts)} left)")
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
                    print("  usage: /<trait> <value>  ([-1, 1] safe, "
                          "beyond overdrives)")
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
                        print(t"{CYAN}  {bank.names[matched]} -> position {val}, "
                              t"alpha {bank.alphas[matched]} at layer "
                              t"{bank.configs[matched].layer}{RESET}")
            continue

        var pre_len = len(history)
        append_user_turn(tok, history, s)
        var reply_budget = MAX_CONTEXT - len(history)
        if reply_budget < 1:
            print("  (context full — use /reset to clear)")
            while len(history) > pre_len:
                _ = history.pop()
            continue
        var rid_opt = sched.submit(history.copy(), greedy, reply_budget)
        if not rid_opt:
            print("  (context full — use /reset to clear)")
            while len(history) > pre_len:
                _ = history.pop()
            continue
        var rid = rid_opt.value()

        var detok = StreamDetokenizer()
        var consumed = 0
        var guard = 0
        var stalled = False
        var reply = String("")
        print(MODEL_PROMPT, end="", flush=True)
        while not sched.requests[rid].done:
            guard += 1
            if guard > 8 * reply_budget:
                print("\n  (generation stalled)")
                stalled = True
                break
            if sched.step(model) == 0:
                print("\n  (scheduler stalled)")
                stalled = True
                break
            ref cur_gen = sched.requests[rid].generated
            while consumed < len(cur_gen):
                var tid = cur_gen[consumed]
                consumed += 1
                if sched.is_stop_token(tid):
                    continue
                var piece = detok.push(tok, tid)
                if piece.byte_length() > 0:
                    reply += piece
                    print(piece, end="", flush=True)

        if stalled:
            _ = sched.retire(rid)
            while len(history) > pre_len:
                _ = history.pop()
            continue

        var tail = detok.flush()
        if tail.byte_length() > 0:
            reply += tail
            print(tail, end="", flush=True)
        print()

        ref gen = sched.requests[rid].generated
        for i in range(len(gen)):
            history.append(gen[i])
        append_encoded(tok, history, "\n")
        _ = sched.retire(rid)
        turn_starts.append(pre_len)
        transcript.append(ChatTurn(s.copy(), reply^))

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
