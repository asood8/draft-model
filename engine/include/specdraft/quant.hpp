#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

// Number formats for the engine. Weights are quantized offline by the Python export
// script; activations are quantized at every matmul input while decoding. The Python
// module specdraft.quant mirrors all of this and the tests require byte-identical
// output, so any change here has to be made in both places.
//
// Block layouts (32 values per block, one scale per block):
//   BlockQ4  4-bit weights:     fp16 scale + 16 packed bytes  = 18 bytes (4.5 bits/weight)
//   BlockQ8  8-bit weights:     fp16 scale + 32 int8          = 34 bytes (8.5 bits/weight)
//   BlockA8  8-bit activations: fp32 scale + 32 int8          = 36 bytes
//
// In BlockQ4, byte i holds weight i in its low nibble and weight i + 16 in its high
// nibble, and a stored nibble q means the value (q - 8) * scale. Quantized values are
// kept in [-127, 127] so that negating them for the AVX-VNNI sign trick cannot
// overflow int8.

namespace specdraft {

constexpr int QK = 32;  // values per block

struct BlockQ4 {
    uint16_t scale;
    uint8_t q[QK / 2];
};

struct BlockQ8 {
    uint16_t scale;
    int8_t q[QK];
};

struct BlockA8 {
    float scale;
    int8_t q[QK];
};

static_assert(sizeof(BlockQ4) == 18, "BlockQ4 must be 18 bytes");
static_assert(sizeof(BlockQ8) == 34, "BlockQ8 must be 34 bytes");
static_assert(sizeof(BlockA8) == 36, "BlockA8 must be 36 bytes");

// Defined here rather than in the .cpp so the dot kernels inline them: they run once per
// 32-weight block, and a real call per block costs more than the arithmetic it performs.
// The scalar _cvtsh_ss / _cvtss_sh intrinsics are a GCC/Clang extension MSVC lacks, so these
// use the F16C vector forms, which every supported compiler provides.
inline float fp16_to_fp32(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(h))));
}

// Rounding comes from the immediate (nearest, ties to even), not MXCSR, so it matches
// NumPy's and torch's float32 -> float16 conversion.
inline uint16_t fp32_to_fp16(float f) {
    const __m128i h = _mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return static_cast<uint16_t>(_mm_extract_epi16(h, 0));
}

// Quantization. n must be a multiple of QK; the output holds n / QK blocks.
//
// Rounding rules, which the Python mirror follows exactly:
//   Q4: scale = fp16(v_max / -8), where v_max is the value of largest magnitude in the
//       block, sign kept; q = clamp(floor(x / scale + 8.5), 0, 15).
//   Q8: scale = fp16(max|x| / 127); q = clamp(rint(x / scale), -127, 127).
//   A8: scale = max|x| / 127 in fp32; q = clamp(rint(x / scale), -127, 127).
// Q4 and Q8 divide by the fp16-rounded scale, the same value dequantization uses.
void quantize_q4(const float* x, int n, BlockQ4* out);
void quantize_q8(const float* x, int n, BlockQ8* out);
void quantize_a8(const float* x, int n, BlockA8* out);

void dequantize_q4(const BlockQ4* blocks, int n, float* out);
void dequantize_q8(const BlockQ8* blocks, int n, float* out);
void dequantize_a8(const BlockA8* blocks, int n, float* out);

// Dot products of one quantized weight row with one quantized activation vector.
// The *_scalar versions are the reference the SIMD kernels are tested against; the
// unsuffixed ones dispatch on the CPU's features.
float dot_q4_a8_scalar(const BlockQ4* w, const BlockA8* x, int nblocks);
float dot_q8_a8_scalar(const BlockQ8* w, const BlockA8* x, int nblocks);
float dot_q4_a8(const BlockQ4* w, const BlockA8* x, int nblocks);
float dot_q8_a8(const BlockQ8* w, const BlockA8* x, int nblocks);

// One weight row against k activation vectors, unpacking each weight block once and reusing
// it for every token.
//
// **No longer the engine's path.** It was, and the project's v(k) figures up to 2026-10-01 come from
// it, so it stays as the measured baseline the split-layout kernels below are compared against --
// `scripts/bench_kernel.py` reports both. It is also a second implementation of the same arithmetic,
// which is worth having when the other one is the one in use.
//
// Activations are laid out as k consecutive vectors of `nblocks` blocks each; `out` receives
// one float per token. Each token accumulates in exactly the order the single-token kernel
// uses, so results are bit-identical to calling that kernel k times — the property greedy
// speculative decoding depends on.
void dot_q4_a8_multi(const BlockQ4* w, const BlockA8* x, int nblocks, int k, float* out);
void dot_q8_a8_multi(const BlockQ8* w, const BlockA8* x, int nblocks, int k, float* out);

// ------------------------------------------------------------------- the layout the engine uses
//
// The kernels above keep each block's scale next to its quantized bytes, which is how the file
// stores them and how ggml lays out q4_0. It is the wrong shape for the processor. Those kernels are
// limited by the ports that issue dpbusd, and measurement put each block at about five such
// operations to perform one multiply-accumulate instruction: two for the sign trick, three to
// convert the fp16 scale, multiply it by the activation scale and broadcast the result. Both costs
// disappear once the scales and the quantized bytes live in separate arrays:
//
//   * eight fp16 scales convert in one vcvtph2ps instead of eight scalar conversions, and the eight
//     per-block products apply with one multiply and one fmadd, after reducing eight blocks worth
//     of int32 accumulators into a single vector in block order;
//   * the sign trick gives way to the identity sum((q - z) * x) = sum(q * x) - z * sum(x), for the
//     format's zero point z, which lets the raw bytes be dpbusd's unsigned operand. The correction
//     costs nothing in the loop, because the accumulator starts at a precomputed per-lane bias
//     instead of at zero.
//
// Measured at 1.3x to 1.8x the interleaved kernels, and slightly more accurate, since grouping eight
// blocks shortens the float accumulation chain. Model files from version 4 on store this layout, so
// the engine maps it and never repacks; the quantization itself is unchanged, which is why every
// byte-exactness test against the Python mirror still applies to both.
//
// A row of `nblocks` blocks is a pair of arrays: `scales[nblocks]`, and `qs` holding 16 bytes a
// block for q4 or 32 for q8. Activations are the same plus a bias: `x_bias[nblocks * 8]` holds
// -z times the sum of the four activations in each of dpbusd's eight int32 lanes.

// Quantize one activation vector straight into that layout. `zero_point` is the weight format's:
// 8 for q4 nibbles, 128 for q8 bytes. It belongs to the weights rather than the activations, but
// the bias is a property of the activations, so the caller passes the one its matmul needs.
void quantize_a8_soa(const float* x, int n, int zero_point, float* scales, int8_t* qs,
                     int32_t* bias);

// Rearrange one interleaved row into the split layout. The exporter does this in Python; these
// exist so tests can drive the kernels from the same blobs the other bindings take.
void repack_q4_soa(const BlockQ4* blocks, int nblocks, uint16_t* scales, uint8_t* qs);
void repack_q8_soa(const BlockQ8* blocks, int nblocks, uint16_t* scales, int8_t* qs);
void repack_a8_soa(const BlockA8* blocks, int nblocks, int zero_point, float* scales, int8_t* qs,
                   int32_t* bias);

void dequantize_q4_soa(const uint16_t* scales, const uint8_t* qs, int n, float* out);
void dequantize_q8_soa(const uint16_t* scales, const int8_t* qs, int n, float* out);

// One weight row against k activation vectors, in the split layout. Tokens are computed one after
// another over a row that a single pass has already brought into L1, so the weights are read from
// memory once however large k is -- and a k-token pass is bit-identical to k single-token passes by
// construction rather than by agreement between two kernels.
void dot_q4_soa_multi(const uint16_t* w_scales, const uint8_t* w_qs, const float* x_scales,
                      const int8_t* x_qs, const int32_t* x_bias, int nblocks, int k, float* out);
void dot_q8_soa_multi(const uint16_t* w_scales, const int8_t* w_qs, const float* x_scales,
                      const int8_t* x_qs, const int32_t* x_bias, int nblocks, int k, float* out);

// The reference the SIMD versions are tested against.
float dot_q4_soa_scalar(const uint16_t* w_scales, const uint8_t* w_qs, const float* x_scales,
                        const int8_t* x_qs, const int32_t* x_bias, int nblocks);
float dot_q8_soa_scalar(const uint16_t* w_scales, const int8_t* w_qs, const float* x_scales,
                        const int8_t* x_qs, const int32_t* x_bias, int nblocks);

}  // namespace specdraft
