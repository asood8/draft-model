#include "specdraft/quant.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "specdraft/cpu.hpp"
#include "specdraft/simd.hpp"

// Clang needs per-function target features for AVX-VNNI, since the baseline flags only
// enable AVX2/FMA/F16C. MSVC lets any intrinsic be used anywhere.
#if defined(__clang__) || defined(__GNUC__)
#define SD_TARGET_VNNI __attribute__((target("avx2,avxvnni")))
#else
#define SD_TARGET_VNNI
#endif

namespace specdraft {
namespace {

inline int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// One accumulator, with an FMA per block. This serializes on the FMA latency, which two
// accumulators would hide -- but the multi-token kernel has to accumulate in exactly this order for
// a k-token pass to stay bit-identical to k single-token passes, and it cannot afford two
// accumulators per token across a tile of eight. Correctness decides the shape; the cost of that
// choice is measured rather than assumed.
SD_TARGET_VNNI float dot_q4_a8_vnni(const BlockQ4* w, const BlockA8* x, int nblocks) {
    const __m256i low_nibble = _mm256_set1_epi8(0x0F);
    const __m256i eight = _mm256_set1_epi8(8);
    __m256 sum = _mm256_setzero_ps();
    for (int b = 0; b < nblocks; ++b) {
        // 16 bytes hold 32 nibbles: low nibbles are weights 0..15, high nibbles 16..31.
        __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w[b].q));
        __m256i both = _mm256_set_m128i(_mm_srli_epi16(packed, 4), packed);
        __m256i wq = _mm256_sub_epi8(_mm256_and_si256(both, low_nibble), eight);
        __m256i xq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[b].q));
        // dpbusd multiplies unsigned by signed bytes, so take |w| and move w's sign onto x.
        __m256i acc = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(wq, wq),
                                              _mm256_sign_epi8(xq, wq));
        const float d = fp16_to_fp32(w[b].scale) * x[b].scale;
        sum = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(acc), sum);
    }
    return hsum256(sum);
}

SD_TARGET_VNNI float dot_q8_a8_vnni(const BlockQ8* w, const BlockA8* x, int nblocks) {
    __m256 sum = _mm256_setzero_ps();
    for (int b = 0; b < nblocks; ++b) {
        __m256i wq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w[b].q));
        __m256i xq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[b].q));
        __m256i acc = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(wq, wq),
                                              _mm256_sign_epi8(xq, wq));
        const float d = fp16_to_fp32(w[b].scale) * x[b].scale;
        sum = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(acc), sum);
    }
    return hsum256(sum);
}

// How many tokens share one pass over the weights. Every extra tile is another pass, and
// measurement showed that boundary dominating v(k): with a tile of four, eight tokens cost twice
// what four did, giving the curve a sawtooth at each multiple of four.
//
// Eight tokens with one accumulator each fit the sixteen vector registers, where eight tokens with
// two each would spill. One accumulator per token is enough here precisely because there are
// several tokens: their chains are independent, so the processor has plenty to overlap. The
// single-token case is different -- one chain, nothing to overlap -- so it keeps its own kernel
// above, which unrolls across blocks instead.
constexpr int kTile = 8;

SD_TARGET_VNNI void dot_q4_tile(const BlockQ4* w, const BlockA8* x, int nblocks, int tokens,
                                float* out) {
    const __m256i low_nibble = _mm256_set1_epi8(0x0F);
    const __m256i eight = _mm256_set1_epi8(8);
    __m256 sums[kTile];
    for (int t = 0; t < tokens; ++t) {
        sums[t] = _mm256_setzero_ps();
    }

    for (int b = 0; b < nblocks; ++b) {
        // Unpacked once, then used by every token in the tile.
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w[b].q));
        const __m256i both = _mm256_set_m128i(_mm_srli_epi16(packed, 4), packed);
        const __m256i wq = _mm256_sub_epi8(_mm256_and_si256(both, low_nibble), eight);
        const __m256i magnitude = _mm256_sign_epi8(wq, wq);
        const float weight_scale = fp16_to_fp32(w[b].scale);

        for (int t = 0; t < tokens; ++t) {
            const BlockA8& block = x[static_cast<size_t>(t) * nblocks + b];
            const __m256i xq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block.q));
            const __m256i acc = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), magnitude,
                                                       _mm256_sign_epi8(xq, wq));
            sums[t] = _mm256_fmadd_ps(_mm256_set1_ps(weight_scale * block.scale),
                                      _mm256_cvtepi32_ps(acc), sums[t]);
        }
    }

    for (int t = 0; t < tokens; ++t) {
        out[t] = hsum256(sums[t]);
    }
}

SD_TARGET_VNNI void dot_q8_tile(const BlockQ8* w, const BlockA8* x, int nblocks, int tokens,
                                float* out) {
    __m256 sums[kTile];
    for (int t = 0; t < tokens; ++t) {
        sums[t] = _mm256_setzero_ps();
    }

    for (int b = 0; b < nblocks; ++b) {
        const __m256i wq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w[b].q));
        const __m256i magnitude = _mm256_sign_epi8(wq, wq);
        const float weight_scale = fp16_to_fp32(w[b].scale);

        for (int t = 0; t < tokens; ++t) {
            const BlockA8& block = x[static_cast<size_t>(t) * nblocks + b];
            const __m256i xq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block.q));
            const __m256i acc = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), magnitude,
                                                       _mm256_sign_epi8(xq, wq));
            sums[t] = _mm256_fmadd_ps(_mm256_set1_ps(weight_scale * block.scale),
                                      _mm256_cvtepi32_ps(acc), sums[t]);
        }
    }

    for (int t = 0; t < tokens; ++t) {
        out[t] = hsum256(sums[t]);
    }
}

}  // namespace

void dot_q4_a8_multi(const BlockQ4* w, const BlockA8* x, int nblocks, int k, float* out) {
    const CpuFeatures& f = cpu_features();
    if (!(f.avx2 && f.avx_vnni) || force_scalar()) {
        for (int t = 0; t < k; ++t) {
            out[t] = dot_q4_a8_scalar(w, x + static_cast<size_t>(t) * nblocks, nblocks);
        }
        return;
    }
    for (int t = 0; t < k; t += kTile) {
        const int tokens = std::min(kTile, k - t);
        if (tokens == 1) {
            out[t] = dot_q4_a8_vnni(w, x + static_cast<size_t>(t) * nblocks, nblocks);
        } else {
            dot_q4_tile(w, x + static_cast<size_t>(t) * nblocks, nblocks, tokens, out + t);
        }
    }
}

void dot_q8_a8_multi(const BlockQ8* w, const BlockA8* x, int nblocks, int k, float* out) {
    const CpuFeatures& f = cpu_features();
    if (!(f.avx2 && f.avx_vnni) || force_scalar()) {
        for (int t = 0; t < k; ++t) {
            out[t] = dot_q8_a8_scalar(w, x + static_cast<size_t>(t) * nblocks, nblocks);
        }
        return;
    }
    for (int t = 0; t < k; t += kTile) {
        const int tokens = std::min(kTile, k - t);
        if (tokens == 1) {
            out[t] = dot_q8_a8_vnni(w, x + static_cast<size_t>(t) * nblocks, nblocks);
        } else {
            dot_q8_tile(w, x + static_cast<size_t>(t) * nblocks, nblocks, tokens, out + t);
        }
    }
}

void quantize_q4(const float* x, int n, BlockQ4* out) {
    const int nblocks = n / QK;
    for (int b = 0; b < nblocks; ++b) {
        const float* xb = x + b * QK;
        float amax = 0.0f, vmax = 0.0f;  // largest magnitude, and that value with its sign
        for (int i = 0; i < QK; ++i) {
            const float a = std::fabs(xb[i]);
            if (a > amax) {
                amax = a;
                vmax = xb[i];
            }
        }
        const uint16_t scale_h = fp32_to_fp16(vmax / -8.0f);
        const float scale = fp16_to_fp32(scale_h);
        const float inv = (scale != 0.0f) ? 1.0f / scale : 0.0f;
        out[b].scale = scale_h;
        for (int i = 0; i < QK / 2; ++i) {
            const int lo = clamp_int(static_cast<int>(std::floor(xb[i] * inv + 8.5f)), 0, 15);
            const int hi = clamp_int(static_cast<int>(std::floor(xb[i + 16] * inv + 8.5f)), 0, 15);
            out[b].q[i] = static_cast<uint8_t>(lo | (hi << 4));
        }
    }
}

void quantize_q8(const float* x, int n, BlockQ8* out) {
    const int nblocks = n / QK;
    for (int b = 0; b < nblocks; ++b) {
        const float* xb = x + b * QK;
        float amax = 0.0f;
        for (int i = 0; i < QK; ++i) {
            amax = std::fmax(amax, std::fabs(xb[i]));
        }
        const uint16_t scale_h = fp32_to_fp16(amax / 127.0f);
        const float scale = fp16_to_fp32(scale_h);
        const float inv = (scale != 0.0f) ? 1.0f / scale : 0.0f;
        out[b].scale = scale_h;
        for (int i = 0; i < QK; ++i) {
            out[b].q[i] =
                static_cast<int8_t>(clamp_int(static_cast<int>(std::rint(xb[i] * inv)), -127, 127));
        }
    }
}

void quantize_a8(const float* x, int n, BlockA8* out) {
    const int nblocks = n / QK;
    for (int b = 0; b < nblocks; ++b) {
        const float* xb = x + b * QK;
        float amax = 0.0f;
        for (int i = 0; i < QK; ++i) {
            amax = std::fmax(amax, std::fabs(xb[i]));
        }
        const float scale = amax / 127.0f;
        const float inv = (scale != 0.0f) ? 1.0f / scale : 0.0f;
        out[b].scale = scale;
        for (int i = 0; i < QK; ++i) {
            out[b].q[i] =
                static_cast<int8_t>(clamp_int(static_cast<int>(std::rint(xb[i] * inv)), -127, 127));
        }
    }
}

void dequantize_q4(const BlockQ4* blocks, int n, float* out) {
    const int nblocks = n / QK;
    for (int b = 0; b < nblocks; ++b) {
        const float scale = fp16_to_fp32(blocks[b].scale);
        float* ob = out + b * QK;
        for (int i = 0; i < QK / 2; ++i) {
            ob[i] = static_cast<float>((blocks[b].q[i] & 0x0F) - 8) * scale;
            ob[i + 16] = static_cast<float>((blocks[b].q[i] >> 4) - 8) * scale;
        }
    }
}

void dequantize_q8(const BlockQ8* blocks, int n, float* out) {
    const int nblocks = n / QK;
    for (int b = 0; b < nblocks; ++b) {
        const float scale = fp16_to_fp32(blocks[b].scale);
        for (int i = 0; i < QK; ++i) {
            out[b * QK + i] = static_cast<float>(blocks[b].q[i]) * scale;
        }
    }
}

void dequantize_a8(const BlockA8* blocks, int n, float* out) {
    const int nblocks = n / QK;
    for (int b = 0; b < nblocks; ++b) {
        for (int i = 0; i < QK; ++i) {
            out[b * QK + i] = static_cast<float>(blocks[b].q[i]) * blocks[b].scale;
        }
    }
}

float dot_q4_a8_scalar(const BlockQ4* w, const BlockA8* x, int nblocks) {
    float sum = 0.0f;
    for (int b = 0; b < nblocks; ++b) {
        int32_t acc = 0;
        for (int i = 0; i < QK / 2; ++i) {
            const int lo = (w[b].q[i] & 0x0F) - 8;  // weight i
            const int hi = (w[b].q[i] >> 4) - 8;    // weight i + 16
            acc += lo * x[b].q[i] + hi * x[b].q[i + 16];
        }
        sum += fp16_to_fp32(w[b].scale) * x[b].scale * static_cast<float>(acc);
    }
    return sum;
}

float dot_q8_a8_scalar(const BlockQ8* w, const BlockA8* x, int nblocks) {
    float sum = 0.0f;
    for (int b = 0; b < nblocks; ++b) {
        int32_t acc = 0;
        for (int i = 0; i < QK; ++i) {
            acc += static_cast<int>(w[b].q[i]) * static_cast<int>(x[b].q[i]);
        }
        sum += fp16_to_fp32(w[b].scale) * x[b].scale * static_cast<float>(acc);
    }
    return sum;
}

float dot_q4_a8(const BlockQ4* w, const BlockA8* x, int nblocks) {
    const CpuFeatures& f = cpu_features();
    if (f.avx2 && f.avx_vnni && !force_scalar()) {
        return dot_q4_a8_vnni(w, x, nblocks);
    }
    return dot_q4_a8_scalar(w, x, nblocks);
}

float dot_q8_a8(const BlockQ8* w, const BlockA8* x, int nblocks) {
    const CpuFeatures& f = cpu_features();
    if (f.avx2 && f.avx_vnni && !force_scalar()) {
        return dot_q8_a8_vnni(w, x, nblocks);
    }
    return dot_q8_a8_scalar(w, x, nblocks);
}

// ---------------------------------------------------------------------- the SoA prototype

void repack_q4_soa(const BlockQ4* blocks, int nblocks, uint16_t* scales, uint8_t* qs) {
    for (int b = 0; b < nblocks; ++b) {
        scales[b] = blocks[b].scale;
        std::memcpy(qs + static_cast<size_t>(b) * (QK / 2), blocks[b].q, QK / 2);
    }
}

void repack_a8_soa(const BlockA8* blocks, int nblocks, float* scales, int8_t* qs, int32_t* bias) {
    for (int b = 0; b < nblocks; ++b) {
        scales[b] = blocks[b].scale;
        std::memcpy(qs + static_cast<size_t>(b) * QK, blocks[b].q, QK);
        // dpbusd sums four byte products into each of its eight int32 lanes, so the correction for
        // the stored nibbles each being eight too large has to be split the same way.
        for (int lane = 0; lane < 8; ++lane) {
            int sum = 0;
            for (int i = 0; i < 4; ++i) {
                sum += blocks[b].q[lane * 4 + i];
            }
            bias[static_cast<size_t>(b) * 8 + lane] = -8 * sum;
        }
    }
}

namespace {

constexpr int kGroup = 8;  // blocks sharing one scale application; also the lanes of one vector

// Eight accumulators reduced to one vector holding one block per lane, in block order. hadd works
// within each 128-bit half, so three levels of it leave every lane mixing two blocks; the two
// vperm2i128s put the halves back together, making lane j exactly block j total. Nine instructions
// for eight blocks, against the twenty-four float operations they replace.
SD_TARGET_VNNI inline __m256i reduce8(const __m256i* acc) {
    const __m256i t0 = _mm256_hadd_epi32(acc[0], acc[1]);
    const __m256i t1 = _mm256_hadd_epi32(acc[2], acc[3]);
    const __m256i t2 = _mm256_hadd_epi32(acc[4], acc[5]);
    const __m256i t3 = _mm256_hadd_epi32(acc[6], acc[7]);
    const __m256i u0 = _mm256_hadd_epi32(t0, t1);  // lower lanes: blocks 0-3 over source lanes 0-3
    const __m256i u1 = _mm256_hadd_epi32(t2, t3);  // and the upper half over source lanes 4-7
    return _mm256_add_epi32(_mm256_permute2x128_si256(u0, u1, 0x20),
                            _mm256_permute2x128_si256(u0, u1, 0x31));
}

SD_TARGET_VNNI void dot_q4_a8_soa_one(const uint16_t* w_scales, const uint8_t* w_qs,
                                      const float* x_scales, const int8_t* x_qs,
                                      const int32_t* x_bias, int nblocks, float* out) {
    const __m256i low_nibble = _mm256_set1_epi8(0x0F);
    __m256 sum = _mm256_setzero_ps();

    int b = 0;
    for (; b + kGroup <= nblocks; b += kGroup) {
        __m256i acc[kGroup];
        for (int j = 0; j < kGroup; ++j) {
            const size_t block = static_cast<size_t>(b + j);
            // The nibbles go in unsigned and uncorrected; the bias carries the -8 per lane.
            const __m128i packed =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_qs + block * (QK / 2)));
            const __m256i both = _mm256_set_m128i(_mm_srli_epi16(packed, 4), packed);
            const __m256i wq = _mm256_and_si256(both, low_nibble);
            const __m256i xq =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_qs + block * QK));
            const __m256i bias =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_bias + block * 8));
            acc[j] = _mm256_dpbusd_avx_epi32(bias, wq, xq);
        }
        // Eight fp16 weight scales in one instruction, times the eight activation scales.
        const __m256 ws =
            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w_scales + b)));
        const __m256 xs = _mm256_loadu_ps(x_scales + b);
        sum = _mm256_fmadd_ps(_mm256_mul_ps(ws, xs), _mm256_cvtepi32_ps(reduce8(acc)), sum);
    }

    // The tail, for a row whose block count is not a multiple of eight: the same arithmetic one
    // block at a time, so a short row comes out correct rather than fast.
    for (; b < nblocks; ++b) {
        const size_t block = static_cast<size_t>(b);
        const __m128i packed =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_qs + block * (QK / 2)));
        const __m256i both = _mm256_set_m128i(_mm_srli_epi16(packed, 4), packed);
        const __m256i wq = _mm256_and_si256(both, low_nibble);
        const __m256i xq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_qs + block * QK));
        const __m256i bias =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_bias + block * 8));
        const __m256i acc = _mm256_dpbusd_avx_epi32(bias, wq, xq);
        const float d = fp16_to_fp32(w_scales[b]) * x_scales[b];
        sum = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(acc), sum);
    }
    *out = hsum256(sum);
}

}  // namespace

void dot_q4_a8_soa(const uint16_t* w_scales, const uint8_t* w_qs, const float* x_scales,
                   const int8_t* x_qs, const int32_t* x_bias, int nblocks, int tokens, float* out) {
    // One token at a time. Eight accumulators and the unpack temporaries already fill the sixteen
    // vector registers, so there is no room left to share the weight unpacking across a tile of
    // tokens. Whether that trade pays is exactly what this prototype exists to measure -- the
    // sweep in PLAN.md section 10.1 put the value of sharing the unpack at 24%.
    for (int t = 0; t < tokens; ++t) {
        const size_t offset = static_cast<size_t>(t) * nblocks;
        dot_q4_a8_soa_one(w_scales, w_qs, x_scales + offset, x_qs + offset * QK, x_bias + offset * 8,
                          nblocks, out + t);
    }
}

}  // namespace specdraft
