from std.pathlib import Path
from std.memory import Span
from std.random import Random
from std.time import perf_counter_ns

from numa import NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from tokenizer import load_tokenizer, BPETokenizer, AutoPreTokenizer, AutoByteTransform
from modeling.gemma_4_moe import Gemma4
from modeling.gemma4_common import Gemma4BaseConfig
from modeling.steer import InjectOp
from modeling.probe import (
    ContrastSet, ProbeResult, build_probe, mean_row_norm, projection_stats,
)
from modeling.slider_pack import write_pack, SliderConfig, SliderCalibration
from kernels.helpers import BF16Ptr, F32Ptr
from kernels.dot_products import dot_to_scalar
from kernels.flash_sample import SamplingParams
from continuous_batching.schedule import MAXIMUM_SAMPLING_LOGITS
from continuous_batching.scheduler import ContinuousBatchScheduler


comptime C = Gemma4BaseConfig
comptime TOKENIZER_PATH = "checkpoints/gemma-4-26B-A4B-it/tokenizer.json"
comptime MODEL_DIR = "checkpoints/gemma-4-26B-A4B-it"
comptime DATA_DIR = "steering_data"
comptime BOS_TOKEN_ID = 2
comptime TURN_START_TOKEN_ID = 105
comptime TURN_END_TOKEN_ID = 106
comptime STEP_BUDGET = Gemma4BaseConfig.SLIDING_WINDOW
comptime FIRST_TAP_LAYER = 5
comptime NUM_TAP_LAYERS = 20
comptime CB_RESIDENT = 5
comptime CB_BATCH_LEN = 8192
comptime WAVE_SIZE = CB_RESIDENT
comptime WAVE_STEP_GUARD = 64
comptime USER_CAP = 80
comptime MODEL_CAP = 112
comptime SHIFT_SIGMA_MIN = 1.0
comptime STABILITY_FRACTION = 0.6
comptime HELDOUT_SIGMA_MIN = 1.0
comptime SAS_SEED = UInt64(0x5A5C0FFEE)
comptime MEASURE_K = NUM_TAP_LAYERS - 1


def read_lines(path: Path) -> List[String]:
    var data: List[Byte]
    try:
        data = path.read_bytes()
    except:
        return List[String]()
    var lines = List[String]()
    var start = 0
    for i in range(len(data)):
        if data[i] == Byte(10):
            if i > start:
                lines.append(String(
                    unsafe_from_utf8=Span(data).unsafe_subspan(
                        offset=start, length=i - start)))
            start = i + 1
    if start < len(data):
        lines.append(String(
            unsafe_from_utf8=Span(data).unsafe_subspan(
                offset=start, length=len(data) - start)))
    return lines^


def append_enc(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    mut ids: List[Int32],
    read text: String,
    cap: Int,
):
    var enc = tok.encode(text)
    var n = min(len(enc), cap)
    for i in range(n):
        ids.append(Int32(enc[i]))


def encode_turn(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read user_text: String,
    read model_text: String,
) -> List[Int32]:
    var ids = List[Int32]()
    ids.append(Int32(BOS_TOKEN_ID))
    ids.append(Int32(TURN_START_TOKEN_ID))
    append_enc(tok, ids, "user\n" + user_text, USER_CAP)
    ids.append(Int32(TURN_END_TOKEN_ID))
    append_enc(tok, ids, "\n", 4)
    ids.append(Int32(TURN_START_TOKEN_ID))
    append_enc(tok, ids, "model\n" + model_text, MODEL_CAP)
    return ids^


struct EvalCaptures(Movable):
    var rows: List[BFloat16]
    var count: Int

    def __init__(out self, capacity: Int):
        self.rows = List[BFloat16](
            length=capacity * C.HIDDEN, fill=BFloat16(0))
        self.count = 0

    @always_inline
    def row_ptr(mut self, i: Int) -> BF16Ptr:
        return self.rows.unsafe_ptr() + i * C.HIDDEN


def extract_train[
    P: BurstThreadPool, //,
](
    mut model: Gemma4[batching_seq_len=CB_BATCH_LEN, max_resident_seqs=CB_RESIDENT, Pool=P],
    mut sched: ContinuousBatchScheduler[Gemma4[batching_seq_len=CB_BATCH_LEN, max_resident_seqs=CB_RESIDENT, Pool=P].POSITIONS_PER_PAGE],
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read inputs: List[String],
    read outputs: List[String],
    is_high: Bool,
    greedy: SamplingParams,
    mut dataset: List[ContrastSet[C.HIDDEN]],
) -> Bool:
    var cursor = 0
    while cursor < len(inputs):
        var count = min(WAVE_SIZE, len(inputs) - cursor)
        var wave_rids = List[Int]()
        var harvested = List[Bool]()
        for j in range(cursor, cursor + count):
            var rid = sched.submit(
                encode_turn(tok, inputs[j], outputs[j]), greedy, 1,
                no_share=True).value()
            wave_rids.append(rid)
            harvested.append(False)

        var guard = 0
        while True:
            var all_done = True
            for w in range(len(wave_rids)):
                if not sched.requests[wave_rids[w]].done:
                    all_done = False
            if all_done:
                break
            guard += 1
            if guard > WAVE_STEP_GUARD:
                return False
            if sched.step(model) == 0:
                return False
            for s in range(model.steer.last_num_slots):
                var rid = model.steer.last_step_requests[s]
                for w in range(len(wave_rids)):
                    if (
                        wave_rids[w] == rid
                        and not harvested[w]
                        and len(sched.requests[rid].generated) > 0
                    ):
                        for k in range(NUM_TAP_LAYERS):
                            dataset[k].add_row(
                                is_high, model.steer.captured_ptr(k, s))
                        harvested[w] = True

        for w in range(len(wave_rids)):
            if not harvested[w]:
                return False
            _ = sched.retire(wave_rids[w])
        cursor += count
    return True


def extract_eval[
    P: BurstThreadPool, //,
](
    mut model: Gemma4[batching_seq_len=CB_BATCH_LEN, max_resident_seqs=CB_RESIDENT, Pool=P],
    mut sched: ContinuousBatchScheduler[Gemma4[batching_seq_len=CB_BATCH_LEN, max_resident_seqs=CB_RESIDENT, Pool=P].POSITIONS_PER_PAGE],
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read inputs: List[String],
    read outputs: List[String],
    greedy: SamplingParams,
    tap_pos: Int,
    mut sink: EvalCaptures,
) -> Bool:
    sink.count = 0
    var cursor = 0
    while cursor < len(inputs):
        var count = min(WAVE_SIZE, len(inputs) - cursor)
        var wave_rids = List[Int]()
        var harvested = List[Bool]()
        for j in range(cursor, cursor + count):
            var rid = sched.submit(
                encode_turn(tok, inputs[j], outputs[j]), greedy, 1,
                no_share=True).value()
            wave_rids.append(rid)
            harvested.append(False)

        var guard = 0
        while True:
            var all_done = True
            for w in range(len(wave_rids)):
                if not sched.requests[wave_rids[w]].done:
                    all_done = False
            if all_done:
                break
            guard += 1
            if guard > WAVE_STEP_GUARD:
                return False
            if sched.step(model) == 0:
                return False
            for s in range(model.steer.last_num_slots):
                var rid = model.steer.last_step_requests[s]
                for w in range(len(wave_rids)):
                    if (
                        wave_rids[w] == rid
                        and not harvested[w]
                        and len(sched.requests[rid].generated) > 0
                    ):
                        var dst = sink.row_ptr(sink.count)
                        var src = model.steer.captured_ptr(tap_pos, s)
                        for i in range(C.HIDDEN):
                            dst[i] = src[i]
                        sink.count += 1
                        harvested[w] = True

        for w in range(len(wave_rids)):
            if not harvested[w]:
                return False
            _ = sched.retire(wave_rids[w])
        cursor += count
    return True


def mean_projection(mut sink: EvalCaptures, direction: BF16Ptr) -> Float64:
    if sink.count == 0:
        return Float64(0)
    var s = Float64(0)
    for i in range(sink.count):
        s += Float64(dot_to_scalar[C.HIDDEN](sink.row_ptr(i), direction))
    return s / Float64(sink.count)


def sample_inject_ops(
    read predecessors: List[SliderConfig], mut rng: Random[rounds=10],
) -> List[InjectOp]:
    var ops = List[InjectOp]()
    for i in range(len(predecessors)):
        var c = predecessors[i]
        var lanes = rng.step_uniform()
        var mag = c.alpha_min + lanes[0] * (c.alpha_max - c.alpha_min)
        var alpha = mag
        if c.bidirectional and lanes[1] < Float32(0.5):
            alpha = -mag
        ops.append(InjectOp(c.layer, c.vec_idx, alpha))
    return ops^


def extract_shifted[
    P: BurstThreadPool, //,
](
    mut model: Gemma4[batching_seq_len=CB_BATCH_LEN, max_resident_seqs=CB_RESIDENT, Pool=P],
    mut sched: ContinuousBatchScheduler[Gemma4[batching_seq_len=CB_BATCH_LEN, max_resident_seqs=CB_RESIDENT, Pool=P].POSITIONS_PER_PAGE],
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read inputs: List[String],
    read outputs: List[String],
    is_high: Bool,
    greedy: SamplingParams,
    tap_idx: Int,
    mut keep: ContrastSet[C.HIDDEN],
    read predecessors: List[SliderConfig],
    mut rng: Random[rounds=10],
) -> Bool:
    var cursor = 0
    while cursor < len(inputs):
        var count = min(WAVE_SIZE, len(inputs) - cursor)
        var wave_rids = List[Int]()
        var harvested = List[Bool]()
        for j in range(cursor, cursor + count):
            var rid = sched.submit(
                encode_turn(tok, inputs[j], outputs[j]), greedy, 1,
                no_share=True).value()
            model.steer.set_request_inject(
                rid, sample_inject_ops(predecessors, rng))
            wave_rids.append(rid)
            harvested.append(False)

        var guard = 0
        while True:
            var all_done = True
            for w in range(len(wave_rids)):
                if not sched.requests[wave_rids[w]].done:
                    all_done = False
            if all_done:
                break
            guard += 1
            if guard > WAVE_STEP_GUARD:
                return False
            if sched.step(model) == 0:
                return False
            for s in range(model.steer.last_num_slots):
                var rid = model.steer.last_step_requests[s]
                for w in range(len(wave_rids)):
                    if (
                        wave_rids[w] == rid
                        and not harvested[w]
                        and len(sched.requests[rid].generated) > 0
                    ):
                        keep.add_row(is_high, model.steer.captured_ptr(tap_idx, s))
                        harvested[w] = True

        for w in range(len(wave_rids)):
            if not harvested[w]:
                return False
            _ = sched.retire(wave_rids[w])
        cursor += count
    return True


struct TraitOutcome(Movable):
    var name: String
    var passed: Bool
    var direction: List[BFloat16]
    var layer: Int
    var alpha_min: Float32
    var alpha_max: Float32
    var fisher_ratio: Float64
    var holdout_sigma: Float64
    var steer_norm: Float64
    var measure_layer: Int
    var measure_std: Float64

    def __init__(out self, read name: String):
        self.name = name
        self.passed = False
        self.direction = List[BFloat16]()
        self.layer = -1
        self.alpha_min = Float32(0)
        self.alpha_max = Float32(0)
        self.fisher_ratio = Float64(0)
        self.holdout_sigma = Float64(0)
        self.steer_norm = Float64(0)
        self.measure_layer = -1
        self.measure_std = Float64(0)


struct TraitRecord(Movable):
    var name: String
    var valid: Bool
    var best_k: Int
    var layer: Int
    var steer_norm: Float64
    var measure_layer: Int
    var measure_std: Float64
    var fisher_ratio: Float64
    var measure_dir: List[BFloat16]
    var data: ContrastSet[C.HIDDEN]

    def __init__(out self, read name: String):
        self.name = name
        self.valid = False
        self.best_k = -1
        self.layer = -1
        self.steer_norm = Float64(0)
        self.measure_layer = -1
        self.measure_std = Float64(0)
        self.fisher_ratio = Float64(0)
        self.measure_dir = List[BFloat16]()
        self.data = ContrastSet[C.HIDDEN](1)


def select_layer[
    P: BurstThreadPool, //,
](
    mut model: Gemma4[batching_seq_len=CB_BATCH_LEN, max_resident_seqs=CB_RESIDENT, Pool=P],
    mut sched: ContinuousBatchScheduler[Gemma4[batching_seq_len=CB_BATCH_LEN, max_resident_seqs=CB_RESIDENT, Pool=P].POSITIONS_PER_PAGE],
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read trait_name: String,
    greedy: SamplingParams,
) -> TraitRecord:
    var rec = TraitRecord(trait_name)
    var prefix = String(DATA_DIR) + "/" + trait_name
    var hi_train_in = read_lines(Path(prefix + "_high_train_in.txt"))
    var hi_train_out = read_lines(Path(prefix + "_high_train_out.txt"))
    var lo_train_in = read_lines(Path(prefix + "_low_train_in.txt"))
    var lo_train_out = read_lines(Path(prefix + "_low_train_out.txt"))
    if len(hi_train_in) == 0 or len(lo_train_in) == 0:
        print(t"  missing data for {trait_name}")
        return rec^
    print(t"  data: {len(hi_train_in)} high / {len(lo_train_in)} low train")

    var cap = max(len(hi_train_in), len(lo_train_in))
    var dataset = List[ContrastSet[C.HIDDEN]](capacity=NUM_TAP_LAYERS)
    for _ in range(NUM_TAP_LAYERS):
        dataset.append(ContrastSet[C.HIDDEN](cap))

    model.steer.clear_inject()
    if not extract_train(
            model, sched, tok, hi_train_in, hi_train_out, True, greedy, dataset):
        print("  high-class extraction failed")
        return rec^
    if not extract_train(
            model, sched, tok, lo_train_in, lo_train_out, False, greedy, dataset):
        print("  low-class extraction failed")
        return rec^

    var mean_high = List[Float32](length=C.HIDDEN, fill=Float32(0))
    var mean_low = List[Float32](length=C.HIDDEN, fill=Float32(0))
    var mh_ptr: F32Ptr = mean_high.unsafe_ptr()
    var ml_ptr: F32Ptr = mean_low.unsafe_ptr()
    var directions = List[BFloat16](
        length=NUM_TAP_LAYERS * C.HIDDEN, fill=BFloat16(0))
    var results = List[ProbeResult]()
    var best_k = -1
    var best_fr = Float64(0)
    for k in range(NUM_TAP_LAYERS):
        var dir_ptr: BF16Ptr = directions.unsafe_ptr() + k * C.HIDDEN
        var r = build_probe[C.HIDDEN](
            dataset[k], model.steer.tap_layers[k], mh_ptr, ml_ptr, dir_ptr)
        results.append(r)
        if r.fr > best_fr:
            best_fr = r.fr
            best_k = k
    if best_k < 0:
        print("  no valid probe")
        return rec^

    var keep = ContrastSet[C.HIDDEN](2 * cap)
    for i in range(dataset[best_k].n_high):
        keep.add_row(True, dataset[best_k].high_ptr(i))
    for i in range(dataset[best_k].n_low):
        keep.add_row(False, dataset[best_k].low_ptr(i))

    var mdir = List[BFloat16](length=C.HIDDEN, fill=BFloat16(0))
    for i in range(C.HIDDEN):
        mdir[i] = directions[MEASURE_K * C.HIDDEN + i]

    rec.best_k = best_k
    rec.layer = results[best_k].layer
    rec.steer_norm = mean_row_norm[C.HIDDEN](dataset[best_k])
    rec.measure_layer = results[MEASURE_K].layer
    rec.measure_std = results[MEASURE_K].pooled_std()
    rec.fisher_ratio = best_fr
    rec.measure_dir = mdir^
    rec.data = keep^
    rec.valid = True
    print(t"  fisher {best_fr} | steer layer {rec.layer}")
    return rec^


def finalize_trait[
    P: BurstThreadPool, //,
](
    mut model: Gemma4[batching_seq_len=CB_BATCH_LEN, max_resident_seqs=CB_RESIDENT, Pool=P],
    mut sched: ContinuousBatchScheduler[Gemma4[batching_seq_len=CB_BATCH_LEN, max_resident_seqs=CB_RESIDENT, Pool=P].POSITIONS_PER_PAGE],
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    mut rec: TraitRecord,
    slot: Int,
    read predecessors: List[SliderConfig],
    greedy: SamplingParams,
    mut rng: Random[rounds=10],
) -> TraitOutcome:
    var outcome = TraitOutcome(rec.name)
    var prefix = String(DATA_DIR) + "/" + rec.name

    if len(predecessors) > 0:
        var hi_train_in = read_lines(Path(prefix + "_high_train_in.txt"))
        var hi_train_out = read_lines(Path(prefix + "_high_train_out.txt"))
        var lo_train_in = read_lines(Path(prefix + "_low_train_in.txt"))
        var lo_train_out = read_lines(Path(prefix + "_low_train_out.txt"))
        if not extract_shifted(
                model, sched, tok, hi_train_in, hi_train_out, True, greedy,
                rec.best_k, rec.data, predecessors, rng):
            print("  shifted high extraction failed")
            model.steer.clear_inject()
            return outcome^
        if not extract_shifted(
                model, sched, tok, lo_train_in, lo_train_out, False, greedy,
                rec.best_k, rec.data, predecessors, rng):
            print("  shifted low extraction failed")
            model.steer.clear_inject()
            return outcome^
        model.steer.clear_inject()

    var mean_high = List[Float32](length=C.HIDDEN, fill=Float32(0))
    var mean_low = List[Float32](length=C.HIDDEN, fill=Float32(0))
    var composite = List[BFloat16](length=C.HIDDEN, fill=BFloat16(0))
    _ = build_probe[C.HIDDEN](
        rec.data, rec.layer, mean_high.unsafe_ptr(), mean_low.unsafe_ptr(),
        composite.unsafe_ptr())
    var composite_ptr: BF16Ptr = composite.unsafe_ptr()

    model.set_steer_vector(slot, composite)
    model.steer.clear_inject()

    var hi_eval_in = read_lines(Path(prefix + "_high_eval_in.txt"))
    var hi_eval_out = read_lines(Path(prefix + "_high_eval_out.txt"))
    var lo_eval_in = read_lines(Path(prefix + "_low_eval_in.txt"))
    var lo_eval_out = read_lines(Path(prefix + "_low_eval_out.txt"))

    var eval_high = EvalCaptures(len(hi_eval_in))
    var eval_low = EvalCaptures(len(lo_eval_in))
    if not extract_eval(
            model, sched, tok, hi_eval_in, hi_eval_out, greedy,
            rec.best_k, eval_high):
        print("  held-out high extraction failed")
        return outcome^
    if not extract_eval(
            model, sched, tok, lo_eval_in, lo_eval_out, greedy,
            rec.best_k, eval_low):
        print("  held-out low extraction failed")
        return outcome^
    var hi_stats = projection_stats[C.HIDDEN](
        eval_high.row_ptr(0), eval_high.count, composite_ptr)
    var lo_stats = projection_stats[C.HIDDEN](
        eval_low.row_ptr(0), eval_low.count, composite_ptr)
    var hd = ProbeResult(
        rec.layer, Float64(0), Float64(0), hi_stats[0], lo_stats[0],
        hi_stats[1], lo_stats[1])
    var holdout_std = hd.pooled_std()
    var holdout_effect = (
        (hi_stats[0] - lo_stats[0]) / holdout_std
        if holdout_std > Float64(0) else Float64(0))

    var measure_ptr: BF16Ptr = rec.measure_dir.unsafe_ptr()
    var base_low = EvalCaptures(len(lo_eval_in))
    if not extract_eval(
            model, sched, tok, lo_eval_in, lo_eval_out, greedy,
            MEASURE_K, base_low):
        print("  baseline measure extraction failed")
        return outcome^
    var base_proj = mean_projection(base_low, measure_ptr)

    var alpha_max = Float32(rec.steer_norm * Float64(STABILITY_FRACTION))
    var probe_ops = List[InjectOp]()
    probe_ops.append(InjectOp(rec.layer, slot, alpha_max))
    model.steer.set_inject(probe_ops^)
    var steered = EvalCaptures(len(lo_eval_in))
    if not extract_eval(
            model, sched, tok, lo_eval_in, lo_eval_out, greedy,
            MEASURE_K, steered):
        print("  steered measure extraction failed")
        model.steer.clear_inject()
        return outcome^
    model.steer.clear_inject()
    var shift_max = (
        (mean_projection(steered, measure_ptr) - base_proj) / rec.measure_std
        if rec.measure_std > Float64(0) else Float64(0))

    var alpha_min = Float32(-1)
    if shift_max >= Float64(SHIFT_SIGMA_MIN) and alpha_max > Float32(0):
        var amin = Float64(SHIFT_SIGMA_MIN) * Float64(alpha_max) / shift_max
        alpha_min = Float32(amin)
        if alpha_min > alpha_max:
            alpha_min = alpha_max

    print(t"  holdout {holdout_effect} sigma | corridor [{alpha_min}, {alpha_max}]")
    if holdout_effect > Float64(HELDOUT_SIGMA_MIN):
        outcome.passed = True
        outcome.direction = composite^
        outcome.layer = rec.layer
        outcome.alpha_min = alpha_min
        outcome.alpha_max = alpha_max
        outcome.fisher_ratio = rec.fisher_ratio
        outcome.holdout_sigma = holdout_effect
        outcome.steer_norm = rec.steer_norm
        outcome.measure_layer = rec.measure_layer
        outcome.measure_std = rec.measure_std
    return outcome^


def run[
    P: BurstThreadPool, //,
](
    topo: NumaTopology,
    var pools: List[P],
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
):
    var model_opt = Gemma4[batching_seq_len=CB_BATCH_LEN, max_resident_seqs=CB_RESIDENT, Pool=P].load(Path(MODEL_DIR), topo, pools^)
    if not model_opt:
        print("model load failed")
        return
    var model = model_opt.take()
    print(t"loaded (degree {model.degree})")

    var tap = List[Int](capacity=NUM_TAP_LAYERS)
    for k in range(NUM_TAP_LAYERS):
        tap.append(FIRST_TAP_LAYER + k)
    model.steer.arm(tap^)

    var greedy = SamplingParams(
        Float32(1.0), Float32(0.0), 0, MAXIMUM_SAMPLING_LOGITS, True)
    var sched = ContinuousBatchScheduler[
        Gemma4[batching_seq_len=CB_BATCH_LEN, max_resident_seqs=CB_RESIDENT, Pool=P].POSITIONS_PER_PAGE,
    ](model.batch_geometry(), STEP_BUDGET, Int32(TURN_END_TOKEN_ID))
    var rng = Random[rounds=10](
        seed=SAS_SEED, subsequence=UInt64(0), offset=UInt64(0))

    var traits = List[String]()
    traits.append("openness")
    traits.append("conscientiousness")
    traits.append("extraversion")
    traits.append("agreeableness")
    traits.append("neuroticism")

    var tok_total = List[Int](length=len(traits), fill=0)
    var ns_total = List[Int](length=len(traits), fill=0)
    var train_t0 = perf_counter_ns()

    print()
    print("--- phase A: layer selection (clean activations) ---")
    var records = List[TraitRecord]()
    for ti in range(len(traits)):
        print(t"=== {traits[ti]} ({ti + 1}/{len(traits)}) ===")
        var k0 = model.tokens_processed
        var t0 = perf_counter_ns()
        records.append(select_layer(model, sched, tok, traits[ti], greedy))
        tok_total[ti] += model.tokens_processed - k0
        ns_total[ti] += Int(perf_counter_ns() - t0)

    var order = List[Int]()
    for i in range(len(records)):
        if records[i].valid:
            order.append(i)
    for a in range(len(order)):
        var best = a
        for b in range(a + 1, len(order)):
            if records[order[b]].layer < records[order[best]].layer:
                best = b
        if best != a:
            var tmp = order[a]
            order[a] = order[best]
            order[best] = tmp

    print()
    print("--- phase B: sequential adaptive steering (ascending layer) ---")
    var names = List[String]()
    var directions = List[BFloat16]()
    var configs = List[SliderConfig]()
    var cals = List[SliderCalibration]()
    var predecessors = List[SliderConfig]()

    for oi in range(len(order)):
        var idx = order[oi]
        var slot = len(predecessors)
        print(t"=== {records[idx].name} | layer {records[idx].layer} | "
              t"{slot} predecessor(s) ({oi + 1}/{len(order)}) ===")
        var k0 = model.tokens_processed
        var t0 = perf_counter_ns()
        var outcome = finalize_trait(
            model, sched, tok, records[idx], slot, predecessors, greedy, rng)
        tok_total[idx] += model.tokens_processed - k0
        ns_total[idx] += Int(perf_counter_ns() - t0)
        if outcome.passed:
            names.append(outcome.name)
            for i in range(C.HIDDEN):
                directions.append(outcome.direction[i])
            var cfg = SliderConfig(
                slot, outcome.layer, outcome.alpha_min, outcome.alpha_max, True)
            configs.append(cfg)
            cals.append(SliderCalibration(
                outcome.fisher_ratio, outcome.holdout_sigma, outcome.steer_norm,
                outcome.measure_layer, outcome.measure_std))
            predecessors.append(cfg)
            print(t"  {outcome.name}: PASS (slot {slot})")
        else:
            print(t"  {records[idx].name}: FAIL (excluded)")

    model.steer.disarm()
    var train_ns = Int(perf_counter_ns() - train_t0)

    var ns_per_s = Float64(1_000_000_000)
    print()
    print("=== training throughput ===")
    var grand_tok = 0
    for ti in range(len(traits)):
        grand_tok += tok_total[ti]
        var secs = Float64(ns_total[ti]) / ns_per_s
        var tps = (
            Float64(tok_total[ti]) / secs if secs > Float64(0) else Float64(0))
        print(t"  {traits[ti]}: {tok_total[ti]} tok | {secs} s | "
              t"{Int(tps)} tok/s")
    var total_secs = Float64(train_ns) / ns_per_s
    var overall_tps = (
        Float64(grand_tok) / total_secs if total_secs > Float64(0)
        else Float64(0))
    print(t"  TOTAL: {grand_tok} tok | {total_secs} s | "
          t"{Int(overall_tps)} tok/s")

    print()
    if len(names) == 0:
        print("no traits passed calibration; no pack written")
        return
    if write_pack(
            "sliders/ocean", names, directions, configs, cals,
            C.HIDDEN, C.NUM_LAYERS, "gemma-4-26B-A4B-it"):
        print(t"saved {len(names)}-slider pack: sliders/ocean.safetensors + .json")
    else:
        print("pack save failed")


def main():
    print("personality calibration (Big Five)")
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
