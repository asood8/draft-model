#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cmath>
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

// e^x for eight floats at once.
//
// Range reduction to x = n*ln2 + r with |r| <= ln2/2, a degree-7 Taylor series for e^r, and 2^n
// built by hand out of the exponent bits. The error is under 1e-8 relative, where the tolerances
// this has to meet are 1e-6 against the Python sampling oracle and 1e-5 against the PyTorch twin.
//
// Worth having because both softmaxes in the engine are over large arrays: the vocabulary is 151,936
// entries, and attention at context 2048 runs 32 heads over 2048 positions in each of 36 layers,
// which is 2.4 million exponentials a token.
//
// Anything at or below -88 returns exactly zero, so a masked-out logit of -inf stays exactly zero
// through the softmax and top-k still leaves exactly k entries carrying mass. Just above the clamp,
// down to about -87.7, the result is zero too: 2^n is assembled from the exponent field, and n
// rounds to -127 there, which that field cannot hold. Both are terms 1e-38 the size of the largest
// in any softmax this is used for.
inline __m256 exp256_ps(__m256 x) {
    const __m256 lowest = _mm256_set1_ps(-88.0f);
    const __m256 vanished = _mm256_cmp_ps(x, lowest, _CMP_LE_OQ);
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));
    x = _mm256_max_ps(x, lowest);

    // n = round(x / ln2), r = x - n*ln2 with ln2 split in two so the product is exact.
    const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);

    __m256 p = _mm256_set1_ps(1.0f / 5040.0f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 720.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 120.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 24.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 6.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));

    // 2^n, assembled from the exponent field.
    const __m256i exponent = _mm256_slli_epi32(
        _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    const __m256 scaled = _mm256_mul_ps(p, _mm256_castsi256_ps(exponent));
    return _mm256_andnot_ps(vanished, scaled);
}

inline float hmax256(__m256 v) {
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 best = _mm_max_ps(_mm256_castps256_ps128(v), high);
    best = _mm_max_ps(best, _mm_movehl_ps(best, best));
    best = _mm_max_ss(best, _mm_shuffle_ps(best, best, 1));
    return _mm_cvtss_f32(best);
}

// Softmax in place: the largest value subtracted for stability, then exponentials, then one
// reciprocal applied to all of them. Shared by attention and by the sampling warps, which had a
// scalar copy each.
inline void softmax_in_place(float* values, uint32_t n) {
    if (n == 0) {
        return;
    }
    uint32_t i = 0;
    float largest = values[0];
    if (n >= 8) {
        __m256 best = _mm256_loadu_ps(values);
        for (i = 8; i + 8 <= n; i += 8) {
            best = _mm256_max_ps(best, _mm256_loadu_ps(values + i));
        }
        largest = hmax256(best);
    }
    for (; i < n; ++i) {
        largest = std::max(largest, values[i]);
    }

    // The total is accumulated in double. Summing 151,936 float32 exponentials into float32 lanes
    // left it at 1.00003 rather than 1, which is a 3e-5 bias on every probability in the row for the
    // sake of an addition the exponentials dwarf. The scalar loop this replaced was worse: it summed
    // all of them into one float.
    const __m256 offset = _mm256_set1_ps(largest);
    __m256d totals_low = _mm256_setzero_pd();
    __m256d totals_high = _mm256_setzero_pd();
    double total = 0.0;
    for (i = 0; i + 8 <= n; i += 8) {
        const __m256 e = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(values + i), offset));
        _mm256_storeu_ps(values + i, e);
        totals_low = _mm256_add_pd(totals_low, _mm256_cvtps_pd(_mm256_castps256_ps128(e)));
        totals_high = _mm256_add_pd(totals_high, _mm256_cvtps_pd(_mm256_extractf128_ps(e, 1)));
    }
    if (n >= 8) {
        alignas(32) double lanes[4];
        _mm256_store_pd(lanes, _mm256_add_pd(totals_low, totals_high));
        total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    }
    for (; i < n; ++i) {
        values[i] = std::exp(values[i] - largest);
        total += values[i];
    }

    const float inverse = static_cast<float>(1.0 / total);
    const __m256 broadcast = _mm256_set1_ps(inverse);
    for (i = 0; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(values + i, _mm256_mul_ps(_mm256_loadu_ps(values + i), broadcast));
    }
    for (; i < n; ++i) {
        values[i] *= inverse;
    }
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
    // The same eight-element step dot_f32 has, which is what makes the two agree exactly when n is
    // not a multiple of sixteen. Attention's head_dim always is, so this was easy to leave out and
    // the test for it was what noticed.
    for (; i + 8 <= n; i += 8) {
        const __m256 low =
            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(keys + i)));
        for (int g = 0; g < Group; ++g) {
            acc0[g] = _mm256_fmadd_ps(
                low, _mm256_loadu_ps(queries + static_cast<size_t>(g) * q_stride + i), acc0[g]);
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
