from std.memory import Span, UnsafePointer, alloc
from std.pathlib import Path

from numa import NumaTopology
from threading.burst_threading import BurstPool
from threading.threading_traits import BurstThreadPool
from simd_math.ops import sqrt

from tokenizer import (
    load_tokenizer, BPETokenizer, AutoPreTokenizer, AutoByteTransform,
)
from modeling.gemma4_common import Gemma4BaseConfig
from modeling.model_spec import BF16
from modeling.slot import BindContext
from kernels.embedding import dispatch_embed_lookup
from kernels.reductions import dispatch_allreduce_inplace
from kernels.rmsnorm import dispatch_rms_norm
from kernels.gemm import dispatch_gemm_chained_qkv
from butterquant_kernels import dispatch_bq_norm_quant, dispatch_bq_qkv
from butterquant import ButterquantActivation

from modeling.gemma_4_moe import (
    Gemma4 as Gemma4Ref, dispatch_sliding_attention_qkv,
    Gemma4SlidingScratch as Gemma4SlidingScratchRef,
)
from modeling.gemma_4_moe_bq import (
    Gemma4 as Gemma4Bq, dispatch_bq_sliding_attention_qkv,
    Gemma4SlidingScratch as Gemma4SlidingScratchBq,
)


comptime C = Gemma4BaseConfig
comptime DEGREE = 1
comptime MAX_SEQ = 8192
comptime TOKENIZER_PATH = "checkpoints/gemma-4-26B-A4B/tokenizer.json"
comptime REF_DIR = "checkpoints/gemma-4-26B-A4B"
comptime BQ_DIR = "checkpoints/gemma-4-26B-A4B-bq"

comptime BF16Ptr = UnsafePointer[BFloat16, MutAnyOrigin]


@fieldwise_init
struct ErrorReport(Copyable, ImplicitlyCopyable):
    var ref_rms: Float64
    var err_rms: Float64
    var rel: Float64
    var max_abs: Float64
    var cosine: Float64
    var n: Int


def compare_bf16(reference: BF16Ptr, got: BF16Ptr, n: Int) -> ErrorReport:
    """F32-accumulated error of `got` against the bf16 `reference`."""
    var ssq_ref = Float64(0)
    var ssq_err = Float64(0)
    var dot = Float64(0)
    var ssq_got = Float64(0)
    var max_abs = Float64(0)
    for i in range(n):
        var r = Float64(reference[i].cast[DType.float32]())
        var g = Float64(got[i].cast[DType.float32]())
        var d = g - r
        ssq_ref += r * r
        ssq_got += g * g
        ssq_err += d * d
        dot += r * g
        var ad = -d if d < 0 else d
        if ad > max_abs:
            max_abs = ad
    var ref_rms = sqrt[DType.float64, 1](ssq_ref / Float64(n))[0]
    var err_rms = sqrt[DType.float64, 1](ssq_err / Float64(n))[0]
    var rel = err_rms / ref_rms if ref_rms > 0 else Float64(0)
    var denom = sqrt[DType.float64, 1](ssq_ref * ssq_got)[0]
    var cosine = dot / denom if denom > 0 else Float64(0)
    return ErrorReport(ref_rms, err_rms, rel, max_abs, cosine, n)


def embed_ref[P: BurstThreadPool, //](
    mut model: Gemma4Ref[degree=DEGREE, Pool=P],
    read tok_buf: List[Int32],
    chunk_len: Int,
):
    """Run the reference embedding + allreduce, leaving the bf16 hidden state in
    `x_main` exactly as the real forward would for the first (sliding) layer."""
    comptime shard_rows = C.VOCAB_SIZE // DEGREE
    comptime embed_scale = Float64(sqrt[DType.float32, 1](C.HIDDEN)
        .cast[DType.bfloat16]().cast[DType.float32]())

    ref layout = model.layout
    var ctx = BindContext[DEGREE](
        arena_bases=model.arena_bases, layer_base=0)
    var tail_ctx = ctx.with_layer(layout.tail.base(model.arena_bases[0], 0))
    var x_main = layout.activations.x_main.state_binding(ctx)

    var chunk = Span[Int32, origin_of(tok_buf)](
        ptr=tok_buf.unsafe_ptr(), length=chunk_len)

    dispatch_embed_lookup[
        hidden=C.HIDDEN, scale=embed_scale, shard_rows=shard_rows,
        tp=DEGREE,
    ](chunk, layout.tail.proto.embed.binding(tail_ctx),
      x_main, chunk_len, model.pools)
    dispatch_allreduce_inplace[BF16, DEGREE](
        x_main, chunk_len * C.HIDDEN, model.pools)


def ref_sliding_attention[P: BurstThreadPool, //](
    mut model: Gemma4Ref[degree=DEGREE, Pool=P], chunk_len: Int,
):
    """Reference layer-0 sliding attention: input_norm then the full bf16
    qkv→heads→rope→flash→o_proj block, leaving the output in `x_residual`."""
    comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
    comptime n_eps = Float32(C.HIDDEN) * Float32(C.RMS_NORM_EPS)
    ref layout = model.layout
    var ctx = BindContext[DEGREE](
        arena_bases=model.arena_bases, layer_base=0)
    var sl_ctx = ctx.with_layer(layout.sliding.base(model.arena_bases[0], 0))
    var body = layout.sliding.proto.body

    var x_main = layout.activations.x_main.state_binding(ctx)
    var x_res = layout.activations.x_residual.state_binding(ctx)

    dispatch_rms_norm[hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps, tp=DEGREE](
        x_main, x_res, body.input_norm.binding(sl_ctx), chunk_len, model.pools)

    dispatch_sliding_attention_qkv[
        degree=DEGREE, max_seq_len=MAX_SEQ,
    ](layout, ctx, 0, chunk_len, 0, model.scratch, model.pools)


def bq_sliding_attention[P: BurstThreadPool, //](
    mut model: Gemma4Bq[degree=DEGREE, Pool=P], chunk_len: Int,
):
    """ButterQuant layer-0 sliding attention: §12.1 norm-quant then the int8
    qkv→head-prep→flash→§7.2 o_proj block, output in `x_residual`."""
    comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
    comptime n_eps = Float32(C.HIDDEN) * Float32(C.RMS_NORM_EPS)
    comptime SS = Gemma4SlidingScratchBq[DEGREE, 128]

    ref layout = model.layout
    var ctx = BindContext[DEGREE](
        arena_bases=model.arena_bases, layer_base=0)
    var sl_ctx = ctx.with_layer(layout.sliding.base(model.arena_bases[0], 0))
    var body = layout.sliding.proto.body

    var x_main = layout.activations.x_main.state_binding(ctx)
    var x_i8 = model.scratch.binding[SS, "x_i8"](ctx)
    var x_sa = model.scratch.binding[SS, "x_sa"](ctx)

    dispatch_bq_norm_quant[
        hidden=C.HIDDEN, block=128, sqrt_n=sqrt_n, n_eps=n_eps, tp=DEGREE,
    ](x_main, body.input_norm.binding(sl_ctx), x_i8, x_sa, chunk_len,
      model.pools)

    var act = ButterquantActivation[DEGREE](x_i8, x_sa)
    dispatch_bq_sliding_attention_qkv[
        degree=DEGREE, max_seq_len=MAX_SEQ,
    ](layout, ctx, act, 0, chunk_len, 0, model.scratch, model.pools)


def ref_qkv[P: BurstThreadPool, //](
    mut model: Gemma4Ref[degree=DEGREE, Pool=P], chunk_len: Int,
):
    """Reference: input_norm then the chained bf16 q/k/v projection, leaving the
    raw projections in the `q`/`kv` scratch (true basis, pre head-norm/rope)."""
    comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
    comptime n_eps = Float32(C.HIDDEN) * Float32(C.RMS_NORM_EPS)
    comptime q_rows = C.Q_DIM_SLIDING // DEGREE
    comptime kv_rows = C.KV_DIM_SLIDING // DEGREE
    comptime SSRef = Gemma4SlidingScratchRef[DEGREE, 128]
    ref layout = model.layout
    var ctx = BindContext[DEGREE](
        arena_bases=model.arena_bases, layer_base=0)
    var sl_ctx = ctx.with_layer(layout.sliding.base(model.arena_bases[0], 0))
    var attn = layout.sliding.proto.attn
    var body = layout.sliding.proto.body

    var x_main = layout.activations.x_main.state_binding(ctx)
    var x_res = layout.activations.x_residual.state_binding(ctx)
    dispatch_rms_norm[hidden=C.HIDDEN, sqrt_n=sqrt_n, n_eps=n_eps, tp=DEGREE](
        x_main, x_res, body.input_norm.binding(sl_ctx), chunk_len, model.pools)

    var q_outs = model.scratch.binding[SSRef, "q"](ctx)
    var k_outs = model.scratch.binding[SSRef, "kv"](ctx)
    var v_outs = k_outs.shifted(chunk_len * kv_rows)
    dispatch_gemm_chained_qkv[
        q_rows=q_rows, kv_rows=kv_rows, cols=C.HIDDEN, tp=DEGREE,
    ](x_res, attn.q_proj.binding(sl_ctx), attn.k_proj.binding(sl_ctx),
      attn.v_proj.binding(sl_ctx), q_outs, k_outs, v_outs, chunk_len,
      model.pools)


def bq_qkv[P: BurstThreadPool, //](
    mut model: Gemma4Bq[degree=DEGREE, Pool=P], chunk_len: Int,
):
    """ButterQuant: §12.1 norm-quant then the chained int8 §6.2 q/k/v
    projection, leaving the dequantized projections in `q`/`kv` (true basis)."""
    comptime sqrt_n = sqrt[DType.float32, 1](C.HIDDEN)
    comptime n_eps = Float32(C.HIDDEN) * Float32(C.RMS_NORM_EPS)
    comptime kv_rows = C.KV_DIM_SLIDING // DEGREE
    comptime SSBq = Gemma4SlidingScratchBq[DEGREE, 128]
    ref layout = model.layout
    var ctx = BindContext[DEGREE](
        arena_bases=model.arena_bases, layer_base=0)
    var sl_ctx = ctx.with_layer(layout.sliding.base(model.arena_bases[0], 0))
    var attn = layout.sliding.proto.attn
    var body = layout.sliding.proto.body

    var x_main = layout.activations.x_main.state_binding(ctx)
    var x_i8 = model.scratch.binding[SSBq, "x_i8"](ctx)
    var x_sa = model.scratch.binding[SSBq, "x_sa"](ctx)
    dispatch_bq_norm_quant[
        hidden=C.HIDDEN, block=128, sqrt_n=sqrt_n, n_eps=n_eps, tp=DEGREE,
    ](x_main, body.input_norm.binding(sl_ctx), x_i8, x_sa, chunk_len,
      model.pools)

    var q_outs = model.scratch.binding[SSBq, "q"](ctx)
    var k_outs = model.scratch.binding[SSBq, "kv"](ctx)
    var v_outs = k_outs.shifted(chunk_len * kv_rows)
    var act = ButterquantActivation[DEGREE](x_i8, x_sa)
    dispatch_bq_qkv(
        act, attn.q_proj.bq_weight(sl_ctx), attn.k_proj.bq_weight(sl_ctx),
        attn.v_proj.bq_weight(sl_ctx), q_outs, k_outs, v_outs, chunk_len,
        model.pools)


def report(label: StringSlice, rep: ErrorReport):
    var rel_pct = rep.rel * 100.0
    print(t"  [{label}] n={rep.n} ref_rms={rep.ref_rms} err_rms={rep.err_rms} "
          t"rel={rel_pct}% max_abs={rep.max_abs} cos={rep.cosine}")


def main():
    var topo = NumaTopology()
    var tp = len(topo)
    print(t"validate_attention: {tp} NUMA node(s), running at degree={DEGREE}")

    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print("failed to load tokenizer")
        return
    var tok = tok_opt.take()

    var prompt = """The quick brown fox jumps over the lazy dog near the river."""
    var token_ids = List[Int]()
    token_ids.append(2)  # <bos>
    var encoded = tok.encode(prompt)
    for i in range(len(encoded)):
        token_ids.append(encoded[i])
    var chunk_len = len(token_ids)
    if chunk_len > C.SLIDING_WINDOW:
        chunk_len = C.SLIDING_WINDOW
    print(t"prompt tokens: {chunk_len}")

    var tok_buf = List[Int32](capacity=chunk_len)
    for i in range(chunk_len):
        tok_buf.append(Int32(token_ids[i]))

    var pools_ref = List[BurstPool[]](capacity=tp)
    var pools_bq = List[BurstPool[]](capacity=tp)
    for i in range(tp):
        pools_ref.append(BurstPool[].for_rank(topo, i))
        pools_bq.append(BurstPool[].for_rank(topo, i))

    print("loading bf16 reference model ...")
    var ref_opt = Gemma4Ref[degree=DEGREE].load(Path(REF_DIR), topo, pools_ref^)
    if not ref_opt:
        print("bf16 model load failed")
        return
    var ref_model = ref_opt.take()

    print("loading butterquant model ...")
    var bq_opt = Gemma4Bq[degree=DEGREE].load(Path(BQ_DIR), topo, pools_bq^)
    if not bq_opt:
        print("bq model load failed")
        return
    var bq_model = bq_opt.take()

    var ctx_ref = BindContext[DEGREE](
        arena_bases=ref_model.arena_bases, layer_base=0)
    var ctx_bq = BindContext[DEGREE](
        arena_bases=bq_model.arena_bases, layer_base=0)

    comptime q_rows = C.Q_DIM_SLIDING // DEGREE
    comptime kv_rows = C.KV_DIM_SLIDING // DEGREE
    comptime SSRef = Gemma4SlidingScratchRef[DEGREE, 128]
    comptime SSBq = Gemma4SlidingScratchBq[DEGREE, 128]

    # Identical input: embed with the reference, copy the hidden state into bq.
    embed_ref(ref_model, tok_buf, chunk_len)
    var ref_xmain = ref_model.layout.activations.x_main.state_binding(
        ctx_ref)[0]
    var bq_xmain = bq_model.layout.activations.x_main.state_binding(ctx_bq)[0]
    for i in range(chunk_len * C.HIDDEN):
        bq_xmain[i] = ref_xmain[i]

    # Stage 1: q/k/v projections (same true basis on both paths).
    ref_qkv(ref_model, chunk_len)
    bq_qkv(bq_model, chunk_len)
    var ref_q = ref_model.scratch.binding[SSRef, "q"](ctx_ref)[0]
    var ref_k = ref_model.scratch.binding[SSRef, "kv"](ctx_ref)[0]
    var bq_q = bq_model.scratch.binding[SSBq, "q"](ctx_bq)[0]
    var bq_k = bq_model.scratch.binding[SSBq, "kv"](ctx_bq)[0]
    print("\n=== Stage 1: QKV projection (true basis, directly comparable) ===")
    report("q", compare_bf16(ref_q, bq_q, chunk_len * q_rows))
    report("k", compare_bf16(ref_k, bq_k, chunk_len * kv_rows))
    report("v", compare_bf16(
        ref_k + chunk_len * kv_rows, bq_k + chunk_len * kv_rows,
        chunk_len * kv_rows))

    # Stage 2: full sliding-attention block output (post o_proj, true basis).
    ref_sliding_attention(ref_model, chunk_len)
    bq_sliding_attention(bq_model, chunk_len)
    var ref_out = ref_model.layout.activations.x_residual.state_binding(
        ctx_ref)[0]
    var bq_out = bq_model.layout.activations.x_residual.state_binding(
        ctx_bq)[0]
    print("\n=== Stage 2: sliding attention block (layer 0, post o_proj) ===")
    report("attn_out", compare_bf16(ref_out, bq_out, chunk_len * C.HIDDEN))
