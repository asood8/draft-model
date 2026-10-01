#pragma once

#include <immintrin.h>

#include <cstdint>

// Small vectorized helpers for the float parts of the forward pass: attention over the fp16
// KV cache, the norms, and the cache writes. The quantized matrix multiplies have their own
// kernels in quant.hpp; everything here works on plain float32 and fp16.
//
// AVX2, FMA and F16C are the compile-time baseline, so these need no runtime dispatch. Two
// accumulators are used where a reduction would otherwise serialize on FMA latency.

namespace specdraft {

inline float hsum256(__m256 v) {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
    return _mm_cvtss_f32(sum);
}

// fp16 -> fp32, eight at a time. Converting the KV cache one element at a time was 16x
// slower than its memory traffic would suggest.
inline void fp16_to_fp32_many(const uint16_t* src, float* dst, uint32_t n) {
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i,
                         _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i))));
    }
    for (; i < n; ++i) {
        dst[i] = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(src[i]))));
    }
}

// fp32 -> fp16, eight at a time, rounding to nearest with ties to even.
inline void fp32_to_fp16_many(const float* src, uint16_t* dst, uint32_t n) {
    constexpr int kRound = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
                         _mm256_cvtps_ph(_mm256_loadu_ps(src + i), kRound));
    }
    for (; i < n; ++i) {
        dst[i] = static_cast<uint16_t>(
            _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(src[i]), kRound), 0));
    }
}

inline float dot_f32(const float* a, const float* b, uint32_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    for (; i + 8 <= n; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    }
    float sum = hsum256(_mm256_add_ps(acc0, acc1));
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// out += weight * values
inline void accumulate_scaled(float* out, const float* values, float weight, uint32_t n) {
    const __m256 broadcast = _mm256_set1_ps(weight);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_fmadd_ps(broadcast, _mm256_loadu_ps(values + i),
                                                  _mm256_loadu_ps(out + i)));
    }
    for (; i < n; ++i) {
        out[i] += weight * values[i];
    }
}

inline float sum_of_squares(const float* x, uint32_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256 a = _mm256_loadu_ps(x + i);
        const __m256 b = _mm256_loadu_ps(x + i + 8);
        acc0 = _mm256_fmadd_ps(a, a, acc0);
        acc1 = _mm256_fmadd_ps(b, b, acc1);
    }
    for (; i + 8 <= n; i += 8) {
        const __m256 a = _mm256_loadu_ps(x + i);
        acc0 = _mm256_fmadd_ps(a, a, acc0);
    }
    float sum = hsum256(_mm256_add_ps(acc0, acc1));
    for (; i < n; ++i) {
        sum += x[i] * x[i];
    }
    return sum;
}

// out[i] = x[i] * scale * weight[i]
inline void scale_and_weight(const float* x, const float* weight, float scale, uint32_t n,
                             float* out) {
    const __m256 broadcast = _mm256_set1_ps(scale);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i), broadcast),
                                                _mm256_loadu_ps(weight + i)));
    }
    for (; i < n; ++i) {
        out[i] = x[i] * scale * weight[i];
    }
}

inline void add_in_place(float* out, const float* other, uint32_t n) {
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(out + i), _mm256_loadu_ps(other + i)));
    }
    for (; i < n; ++i) {
        out[i] += other[i];
    }
}

}  // namespace specdraft
