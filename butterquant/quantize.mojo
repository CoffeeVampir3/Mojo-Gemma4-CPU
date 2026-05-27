from butterquant.kernels import row_absmax, quantize_inv
from butterquant.types import F32Ptr, I8Ptr, WF


@always_inline
def quantize_activation_per_row(
    work: F32Ptr, qi: I8Ptr, sa: F32Ptr, cols: Int,
):
    var amax = row_absmax(work, cols)
    sa[0] = amax
    var inv = Float32(127.0) / amax if amax > Float32(0) else Float32(0)
    quantize_inv(work, qi, inv, cols)


@always_inline
def quantize_activation_per_block[block: Int](
    work: F32Ptr, qi: I8Ptr, sa: F32Ptr, cols: Int,
):
    comptime assert block % WF == 0, (
        "FWHT block must be a multiple of the f32 SIMD width")
    debug_assert(cols % block == 0,
        "quantize_activation_per_block: cols must be block-aligned")
    var nb = cols // block
    for b in range(nb):
        var off = b * block
        var amax = row_absmax(work + off, block)
        sa[b] = amax
        var inv = Float32(127.0) / amax if amax > Float32(0) else Float32(0)
        quantize_inv(work + off, qi + off, inv, block)
