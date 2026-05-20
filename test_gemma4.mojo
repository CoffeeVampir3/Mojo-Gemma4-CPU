from std.memory import Span, UnsafePointer
from std.sys.info import simd_width_of
from std.pathlib import Path
from std.time import perf_counter_ns

from numa import NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from notstdcollections import HeapMoveArray
from tokenizer import load_tokenizer, BPETokenizer, AutoPreTokenizer, AutoByteTransform
from modeling.gemma4_common import Gemma4BaseConfig
from modeling.gemma_4_moe import Gemma4
from modeling.temporal_scratch import TemporalLogitsView


comptime TOKENIZER_PATH = "checkpoints/gemma-4-26B-A4B/tokenizer.json"
comptime MODEL_DIR = "checkpoints/gemma-4-26B-A4B"
comptime VOCAB = Gemma4BaseConfig.VOCAB_SIZE
comptime MAX_NEW_TOKENS = 128


def greedy_argmax[degree: Int](
    read view: TemporalLogitsView[VOCAB, degree],
) -> Tuple[Int, Float32]:
    comptime width = simd_width_of[DType.float32]()
    var best_val = Float32(-1e30)
    var best_idx = 0

    for j in range(0, VOCAB, width):
        var v = view.load_f32[width](j)
        for k in range(width):
            if v[k] > best_val:
                best_val = v[k]
                best_idx = j + k

    return (best_idx, best_val)


def load_and_run[
    P: BurstThreadPool, //, degree: Int,
](
    topo: NumaTopology,
    var pools: HeapMoveArray[P],
    read tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read token_ids: List[Int],
):
    var t0 = perf_counter_ns()
    var model_opt = Gemma4[degree=degree, Pool=P].load(
        Path(MODEL_DIR), topo, pools^)
    if not model_opt:
        return
    var model = model_opt.take()
    var load_ms = (perf_counter_ns() - t0) / 1_000_000
    print("model loaded in", load_ms, "ms")
    print()

    var prompt_len = len(token_ids)

    var tok_buf = List[Int32](capacity=prompt_len)
    for i in range(prompt_len):
        tok_buf.append(Int32(token_ids[i]))

    var t1 = perf_counter_ns()
    var logits = model.forward(
        Span[Int32, origin_of(tok_buf)](
            ptr=tok_buf.unsafe_ptr(), length=prompt_len),
        0)
    var prefill_ms = (perf_counter_ns() - t1) / 1_000_000

    var top_vals = InlineArray[Float32, 5](fill=Float32(-1e30))
    var top_ids = InlineArray[Int, 5](fill=0)
    for j in range(VOCAB):
        var v = logits.load_f32[1](j)
        if v[0] > top_vals[4]:
            top_vals[4] = v[0]
            top_ids[4] = j
            for k in range(3, -1, -1):
                if top_vals[k + 1] > top_vals[k]:
                    var tv = top_vals[k]; top_vals[k] = top_vals[k + 1]; top_vals[k + 1] = tv
                    var ti = top_ids[k]; top_ids[k] = top_ids[k + 1]; top_ids[k + 1] = ti
    print("top-5 logits after prompt:")
    for i in range(5):
        var id_list = List[Int]()
        id_list.append(top_ids[i])
        print(" ", i, "id=", top_ids[i], "val=", top_vals[i], "tok=", repr(tok.decode(id_list)))

    var result = greedy_argmax[degree](logits)
    var next_id = result[0]
    logits^.release()

    var generated = List[Int]()
    generated.append(next_id)

    var prefill_tps = Float64(prompt_len) / (Float64(prefill_ms) / 1000.0)
    print(
        "prompt  |", prompt_len, "tokens |",
        prefill_ms, "ms |",
        Int(prefill_tps), "t/s",
    )

    var pos = prompt_len
    var decode_start = perf_counter_ns()

    while len(generated) < MAX_NEW_TOKENS:
        var step_id = Int32(next_id)
        var step_logits = model.forward(
            Span[Int32, origin_of(step_id)](
                ptr=UnsafePointer(to=step_id), length=1),
            pos)
        result = greedy_argmax[degree](step_logits)
        next_id = result[0]
        step_logits^.release()
        generated.append(next_id)
        pos += 1

        if next_id == 1:
            break

    var decode_elapsed_ms = (perf_counter_ns() - decode_start) / 1_000_000
    var decode_tokens = len(generated) - 1
    var decode_tps = Float64(decode_tokens) / (Float64(decode_elapsed_ms) / 1000.0)
    print(
        "decode  |", decode_tokens, "tokens |",
        decode_elapsed_ms, "ms |",
        Int(decode_tps), "t/s",
    )

    var all_ids = List[Int]()
    for i in range(len(token_ids)):
        all_ids.append(token_ids[i])
    for i in range(len(generated)):
        all_ids.append(generated[i])

    var full_text = tok.decode(all_ids)
    print()
    print("=== generated", len(generated), "tokens ===")
    print(full_text)


def main():
    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print("failed to load tokenizer from", TOKENIZER_PATH)
        return
    var tok = tok_opt.take()

    var prompt = """ImagePhaseCongruency

This package provides a collection of image processing functions that exploit the importance of phase information in our perception of images. The functions form two main groups:

    Functions that detect specific patterns of local phase for the purpose of feature detection.

    Functions that enhance an image in a way that does not corrupt the local phase so that our perception of important features are not disrupted.

Installation

pkg> add ImagePhaseCongruency
Feature detection via phase congruency
.	
	

Rather than assume a feature is a point of maximal intensity gradient, the Local Energy Model postulates that features are perceived at points in an image where the Fourier components are maximally in phase. (See the Fourier Series logo of this page). This model was developed by Morrone et al. [1986] and Morrone and Owens [1987]. Kovesi [1997, 1999, 2002] subsequently developed methods of computing phase congruency from quadrature pairs of log-Gabor wavelets.

Phase congruency is an illumination and contrast invariant measure of feature significance. Unlike gradient based feature detectors, which can only detect step features, phase congruency correctly detects features at all kind of phase angle, and not just step features having a phase angle of 0 or 180 degrees. Another key attribute is that phase congruency is a dimensionless quantity ranging from 0 to 1, making it contrast invariant. This allows fixed threshold values to be used over large classes of images.

    phasecongmono() Phase congruency of an image using monogenic filters.
    phasecong3() Computes edge and corner phase congruency in an image via log-Gabor filters.
    Example of using phasecongmono() and phasecong3().

Phase symmetry
.	
	

A point of local symmetry in an image corresponds to a point where the local frequency components are at either the minimum or maximum points in their cycles, that is, where all the frequency components are at the most symmetric points in their cycles. Like phase congruency, phase symmetry is a dimensionless quantity.

    phasesym() Compute phase symmetry on an image via log-Gabor filters.
    phasesymmono() Phase symmetry of an image using monogenic filters.
    Example of using phasesymmono().

Phase preserving denoising
.	
	

This is a wavelet denoising scheme that uses non-orthogonal, complex valued, log-Gabor wavelets, rather than the more usual orthogonal or bi-orthogonal wavelets. Thresholding of wavelet responses in the complex domain allows one to ensure that perceptually important phase information in the image is not corrupted. It is also allows threshold values can be determined automatically from the statistics of the wavelet responses to the image.

    ppdenoise() Phase preserving wavelet image denoising.
    Example of using ppdenoise().

Phase preserving dynamic range compression
.	
	

A common method for displaying images with a high dynamic range is to use some variant of histogram equalization. The problem with histogram equalization is that the contrast amplification of a feature depends on how commonly its data value occurs and this can lead to some undesirable distortions in the relative amplitudes of features. Phase Preserving Dynamic Range Compression allows subtle features in images to be revealed without these distortions. It also allows the scale of analysis to be controlled. Perceptually important phase information is preserved and the contrast amplification of structures in the signal is purely a function of their amplitude.

    ppdrc() Phase Preserving Dynamic Range Compression.
    Example of using ppdrc().

Supporting filtering functions

    gaborconvolve() Convolve an image with a bank of log-Gabor filters.
    monofilt() Apply monogenic filters to an image to obtain 2D analytic signal.
    highpassmonogenic() Compute phase and amplitude in highpass images via monogenic filters.

Test images and functions for manipulating image phase

    step2line() A phase congruent test image that interpolates from a step to a line.
    circsine() Generate a phase congruent circular sine wave grating.
    starsine() Generate a phase congruent star shaped sine wave grating.
    noiseonf() Create $ 1/f^p $ spectrum noise images.
    nophase() Randomize image phase leaving amplitude spectrum unchanged.
    quantizephase() Quantize phase values in an image.
    swapphase() Demonstrates phase - amplitude swapping between images.

Utility functions for construction of filters in the frequency domain

    filtergrids() Generate grids for constructing frequency domain filters.
    filtergrid() Generate grid for constructing frequency domain filters.
    gridangles() Generate arrays of filter grid angles.
    monogenicfilters() Generate monogenic filter grids.
    packedmonogenicfilters() Monogenic filter where both filters are packed in the one Complex grid.
    lowpassfilter() Construct a low-pass Butterworth filter.
    highpassfilter() Construct a high-pass Butterworth filter.
    bandpassfilter() Construct a band-pass Butterworth filter.
    highboostfilter() Construct a high-boost Butterworth filter.
    loggabor() The logarithmic Gabor function in the frequency domain.
    cosineangularfilter() Orientation selective filter with cosine windowing function.
    gaussianangularfilter() Orientation selective filter with Gaussian windowing function.
    perfft2() 2D Fourier transform of Moisan's periodic image component.
    geoseries() Generate geometric series.

Misc functions

    fillnan Fill NaN values in an image with closest non NaN value.
    replacenan Replace NaNs in an array with a specified value.
    hysthresh Hysteresis thresholding of an image.

References

M. C. Morrone and R. A. Owens. "Feature detection from local energy". Pattern Recognition Letters, 6:303-313, 1987.

M. C. Morrone, J. R. Ross, D. C. Burr, and R. A. Owens. " Mach bands are phase dependent". Nature, 324(6094):250-253, November 1986.

Peter Kovesi, "Symmetry and Asymmetry From Local Phase". AI'97, Tenth Australian Joint Conference on Artificial Intelligence. 2 - 4 December 1997. Proceedings - Poster Papers. pp 185-190. preprint

Peter Kovesi, "Image Features From Phase Congruency". Videre: A Journal of Computer Vision Research. MIT Press. Volume 1, Number 3, Summer 1999. paper

Peter Kovesi, "Edges Are Not Just Steps". Proceedings of ACCV2002 The Fifth Asian Conference on Computer Vision, Melbourne Jan 22-25, 2002. pp 822-827. preprint

Peter Kovesi, "Phase Preserving Denoising of Images". The Australian Pattern Recognition Society Conference: DICTA'99. December 1999. Perth WA. pp 212-217. preprint

Peter Kovesi, "Phase Preserving Tone Mapping of Non-Photographic High Dynamic Range Images". Proceedings: The Australian Pattern Recognition Society Conference: Digital Image Computing: Techniques and Applications DICTA 2012. preprint"""
    var token_ids = List[Int]()
    token_ids.append(2)  # <bos>
    var encoded = tok.encode(prompt)
    for i in range(len(encoded)):
        token_ids.append(encoded[i])
    print("prompt:", repr(prompt))
    print("tokens:", len(token_ids), "ids:", end="")
    for i in range(len(token_ids)):
        print("", token_ids[i], end="")
    print()

    var topo = NumaTopology()

    print(String(topo.num_nodes()) + " NUMA nodes, "
        + String(len(topo.isolated_cpus)) + " isolated cpus")

    @parameter
    def dispatch_gemma4_tp[
        P: BurstThreadPool, //, degree: Int,
    ](var selected_pools: HeapMoveArray[P]):
        load_and_run[degree=degree](topo, selected_pools^, tok, token_ids)

    with_topological_rank_dispatch[
        power_of_two_unrolling=3,
        dispatch=dispatch_gemma4_tp,
    ](
        topo, "mode: isolated (spin-only)", "mode: cold (spin-backoff)")
