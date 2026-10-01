#include "specdraft/quant.hpp"

#include <immintrin.h>

#include <cmath>

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

// Two accumulators, because one would serialize the loop: each block's FMA would wait on the
// previous block's, and an FMA takes several cycles while a block is only 18 bytes.
SD_TARGET_VNNI float dot_q4_a8_vnni(const BlockQ4* w, const BlockA8* x, int nblocks) {
    const __m256i low_nibble = _mm256_set1_epi8(0x0F);
    const __m256i eight = _mm256_set1_epi8(8);
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
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
        __m256 scaled = _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(acc));
        if ((b & 1) == 0) {
            sum0 = _mm256_add_ps(sum0, scaled);
        } else {
            sum1 = _mm256_add_ps(sum1, scaled);
        }
    }
    return hsum256(_mm256_add_ps(sum0, sum1));
}

SD_TARGET_VNNI float dot_q8_a8_vnni(const BlockQ8* w, const BlockA8* x, int nblocks) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    for (int b = 0; b < nblocks; ++b) {
        __m256i wq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w[b].q));
        __m256i xq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[b].q));
        __m256i acc = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(wq, wq),
                                              _mm256_sign_epi8(xq, wq));
        const float d = fp16_to_fp32(w[b].scale) * x[b].scale;
        __m256 scaled = _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(acc));
        if ((b & 1) == 0) {
            sum0 = _mm256_add_ps(sum0, scaled);
        } else {
            sum1 = _mm256_add_ps(sum1, scaled);
        }
    }
    return hsum256(_mm256_add_ps(sum0, sum1));
}

}  // namespace

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
    if (f.avx2 && f.avx_vnni) {
        return dot_q4_a8_vnni(w, x, nblocks);
    }
    return dot_q4_a8_scalar(w, x, nblocks);
}

float dot_q8_a8(const BlockQ8* w, const BlockA8* x, int nblocks) {
    const CpuFeatures& f = cpu_features();
    if (f.avx2 && f.avx_vnni) {
        return dot_q8_a8_vnni(w, x, nblocks);
    }
    return dot_q8_a8_scalar(w, x, nblocks);
}

}  // namespace specdraft
