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
// it for every token. This is the kernel that decides v(k): the cost of verifying γ+1
// guesses relative to producing one token. Reading the weights is what a single token's step
// is made of, so k tokens that share those reads cost far less than k separate steps.
//
// Activations are laid out as k consecutive vectors of `nblocks` blocks each; `out` receives
// one float per token. Each token accumulates in exactly the order the single-token kernel
// uses, so results are bit-identical to calling that kernel k times — the property greedy
// speculative decoding depends on.
void dot_q4_a8_multi(const BlockQ4* w, const BlockA8* x, int nblocks, int k, float* out);
void dot_q8_a8_multi(const BlockQ8* w, const BlockA8* x, int nblocks, int k, float* out);

// ---------------------------------------------------------------------- the SoA prototype
//
// A measurement, not yet the engine's path. The kernels above are limited by the ports that issue
// dpbusd, and each block spends about five such operations to perform one multiply-accumulate
// instruction: the sign trick costs two, and converting and applying the block scale costs three.
// Both are avoidable, but only if the block scales and the quantized bytes live in separate arrays:
//
//   * eight fp16 scales convert in one vcvtph2ps instead of eight scalar conversions, and the eight
//     per-block products apply with one multiply and one fmadd, after reducing eight blocks worth
//     of int32 accumulators into a single vector in block order;
//   * the sign trick gives way to the identity sum((q - 8) * x) = sum(q * x) - 8 * sum(x), which
//     lets the raw nibbles be the unsigned operand. The correction costs nothing in the loop: the
//     accumulator is seeded with a precomputed per-lane bias instead of with zero.
//
// Laid out as w_scales[nblocks], w_qs[nblocks * 16], x_scales[tokens * nblocks],
// x_qs[tokens * nblocks * 32] and x_bias[tokens * nblocks * 8], where x_bias holds -8 times the sum
// of the four activations in each of dpbusd's eight int32 lanes.
//
// If this wins, the model file gains a version with this layout. The quantization itself is
// unchanged either way, so every byte-exactness test against the Python mirror still applies.
void repack_q4_soa(const BlockQ4* blocks, int nblocks, uint16_t* scales, uint8_t* qs);
void repack_a8_soa(const BlockA8* blocks, int nblocks, float* scales, int8_t* qs, int32_t* bias);
void dot_q4_a8_soa(const uint16_t* w_scales, const uint8_t* w_qs, const float* x_scales,
                   const int8_t* x_qs, const int32_t* x_bias, int nblocks, int tokens, float* out);

}  // namespace specdraft
