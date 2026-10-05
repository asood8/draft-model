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

// The same two operations against an fp16 source, converting eight at a time in registers instead
// of through a scratch buffer.
//
// Attention reads the KV cache this way: the old shape converted a 128-element row into a buffer and
// then read the buffer back, which costs a store and a load per element and, worse, makes every
// position's conversion wait on the previous position's reads of the same buffer. The accumulator
// structure is identical to the fp32 versions below, so results are bit-for-bit what they were.
inline float dot_f16_f32(const uint16_t* a, const float* b, uint32_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_ps(
            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i))),
            _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(
            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i + 8))),
            _mm256_loadu_ps(b + i + 8), acc1);
    }
    for (; i + 8 <= n; i += 8) {
        acc0 = _mm256_fmadd_ps(
            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i))),
            _mm256_loadu_ps(b + i), acc0);
    }
    float sum = hsum256(_mm256_add_ps(acc0, acc1));
    for (; i < n; ++i) {
        sum += _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(a[i])))) * b[i];
    }
    return sum;
}

// out += weight * values, with the values in fp16.
inline void accumulate_scaled_f16(float* out, const uint16_t* values, float weight, uint32_t n) {
    const __m256 broadcast = _mm256_set1_ps(weight);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(
            out + i,
            _mm256_fmadd_ps(broadcast,
                            _mm256_cvtph_ps(_mm_loadu_si128(
                                reinterpret_cast<const __m128i*>(values + i))),
                            _mm256_loadu_ps(out + i)));
    }
    for (; i < n; ++i) {
        out[i] += weight * _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(values[i]))));
    }
}

// One cached row against a whole group of query heads, converting it once.
//
// This is the shape attention actually wants. A key row is shared by the `group` query heads that
// map to its kv head, so converting it per head -- which both the scratch-buffer version and the
// fused one above end up doing -- is `group` times the conversion work for one row's worth of data.
// Here each eight-element chunk is converted once and multiplied into every head's accumulators.
//
// Two accumulators per head in sixteen-element steps, exactly as `dot_f32` sums, so a row scored
// this way is bit-for-bit what the old path produced. `group` is at most 8: beyond that the
// accumulators stop fitting in registers and this gets slower, not faster.
template <int Group>
inline void dot_f16_f32_group_n(const uint16_t* keys, const float* queries, uint32_t q_stride,
                                uint32_t n, float* out) {
    __m256 acc0[Group];
    __m256 acc1[Group];
    for (int g = 0; g < Group; ++g) {
        acc0[g] = _mm256_setzero_ps();
        acc1[g] = _mm256_setzero_ps();
    }
    uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256 low =
            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(keys + i)));
        const __m256 high =
            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(keys + i + 8)));
        for (int g = 0; g < Group; ++g) {
            const float* q = queries + static_cast<size_t>(g) * q_stride + i;
            acc0[g] = _mm256_fmadd_ps(low, _mm256_loadu_ps(q), acc0[g]);
            acc1[g] = _mm256_fmadd_ps(high, _mm256_loadu_ps(q + 8), acc1[g]);
        }
    }
    for (int g = 0; g < Group; ++g) {
        out[g] = hsum256(_mm256_add_ps(acc0[g], acc1[g]));
    }
    for (; i < n; ++i) {
        const float converted =
            _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(keys[i]))));
        for (int g = 0; g < Group; ++g) {
            out[g] += converted * queries[static_cast<size_t>(g) * q_stride + i];
        }
    }
}

// The group size is a template parameter because it has to be: with a runtime `group` the
// accumulator array cannot live in registers, and measured that way this shape was no faster than
// converting each row once per head -- 8.61 ms a token against 8.43 -- despite doing a quarter of
// the conversions. Dispatching to a fixed size is what makes it worth anything.
inline void dot_f16_f32_group(const uint16_t* keys, const float* queries, uint32_t q_stride,
                              int group, uint32_t n, float* out) {
    switch (group) {
        case 1: dot_f16_f32_group_n<1>(keys, queries, q_stride, n, out); return;
        case 2: dot_f16_f32_group_n<2>(keys, queries, q_stride, n, out); return;
        case 4: dot_f16_f32_group_n<4>(keys, queries, q_stride, n, out); return;
        case 8: dot_f16_f32_group_n<8>(keys, queries, q_stride, n, out); return;
        default:
            for (int g = 0; g < group; ++g) {
                out[g] = dot_f16_f32(keys, queries + static_cast<size_t>(g) * q_stride, n);
            }
            return;
    }
}

template <int Group>
inline void accumulate_scaled_f16_group_n(float* out, uint32_t out_stride, const uint16_t* values,
                                          const float* weights, uint32_t n) {
    __m256 scaled[Group];
    for (int g = 0; g < Group; ++g) {
        scaled[g] = _mm256_set1_ps(weights[g]);
    }
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 converted =
            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(values + i)));
        for (int g = 0; g < Group; ++g) {
            float* row = out + static_cast<size_t>(g) * out_stride + i;
            _mm256_storeu_ps(row, _mm256_fmadd_ps(scaled[g], converted, _mm256_loadu_ps(row)));
        }
    }
    for (; i < n; ++i) {
        const float converted =
            _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(values[i]))));
        for (int g = 0; g < Group; ++g) {
            out[static_cast<size_t>(g) * out_stride + i] += weights[g] * converted;
        }
    }
}

inline void accumulate_scaled_f16_group(float* out, uint32_t out_stride, const uint16_t* values,
                                        const float* weights, int group, uint32_t n) {
    switch (group) {
        case 1: accumulate_scaled_f16_group_n<1>(out, out_stride, values, weights, n); return;
        case 2: accumulate_scaled_f16_group_n<2>(out, out_stride, values, weights, n); return;
        case 4: accumulate_scaled_f16_group_n<4>(out, out_stride, values, weights, n); return;
        case 8: accumulate_scaled_f16_group_n<8>(out, out_stride, values, weights, n); return;
        default:
            for (int g = 0; g < group; ++g) {
                accumulate_scaled_f16(out + static_cast<size_t>(g) * out_stride, values, weights[g],
                                      n);
            }
            return;
    }
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
