#include "ops/moe_experts/gguf_expert_cpu.h"

#include <immintrin.h>

#include <cstring>

namespace ninfer::ops::gguf_cpu_detail {
namespace {

inline float half_at(const std::uint8_t* p) {
    std::uint16_t bits;
    std::memcpy(&bits, p, 2);
    // The vector F16C form: MSVC does not declare the scalar _cvtsh_ss.
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)));
}

inline float reduce(__m256 v) {
    const __m128 sum = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 pair = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    return _mm_cvtss_f32(_mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 1)));
}

// Unsigned codes times signed activations, summed to 8 int32 lanes.
inline __m256i dot_u8s8(__m256i codes, __m256i a) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(codes, a), _mm256_set1_epi16(1));
}

} // namespace

void rows_avx2(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
               float* y, std::size_t y_stride) {
    const int tokens = x.tokens();
    if (tokens > 16) { return rows_scalar(w, x, row_begin, row_end, y, y_stride); }
    __m256 acc[16];
    float correction[16];
    if (w.format == QType::GGUF_Q2_0) {
        const int blocks    = w.k / 64;
        const __m128i three = _mm_set1_epi8(3);
        for (int row = row_begin; row < row_end; ++row) {
            const std::uint8_t* r = w.data + std::size_t(row) * w.row_bytes;
            for (int t = 0; t < tokens; ++t) {
                acc[t]        = _mm256_setzero_ps();
                correction[t] = 0;
            }
            for (int b = 0; b < blocks; ++b) {
                const std::uint8_t* block = r + std::size_t(b) * 18;
                const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 2));
                const __m128i c0 = _mm_and_si128(q, three);
                const __m128i c1 = _mm_and_si128(_mm_srli_epi16(q, 2), three);
                const __m128i c2 = _mm_and_si128(_mm_srli_epi16(q, 4), three);
                const __m128i c3 = _mm_and_si128(_mm_srli_epi16(q, 6), three);
                const __m256i low  = _mm256_set_m128i(c1, c0);
                const __m256i high = _mm256_set_m128i(c3, c2);
                const float d      = half_at(block);
                for (int t = 0; t < tokens; ++t) {
                    const std::int8_t* a = x.codes(t) + b * 64;
                    const __m256i dot    = _mm256_add_epi32(
                        dot_u8s8(low, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a))),
                        dot_u8s8(high, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 32))));
                    const float scale = d * x.scales(t)[b];
                    acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot), _mm256_set1_ps(scale), acc[t]);
                    correction[t] += scale * float(x.sums(t)[b]);
                }
            }
            for (int t = 0; t < tokens; ++t) {
                y[std::size_t(t) * y_stride + row] = reduce(acc[t]) - correction[t];
            }
        }
        return;
    }
    const int blocks = w.k / 32;
    for (int row = row_begin; row < row_end; ++row) {
        const std::uint8_t* r = w.data + std::size_t(row) * w.row_bytes;
        for (int t = 0; t < tokens; ++t) { acc[t] = _mm256_setzero_ps(); }
        for (int b = 0; b < blocks; ++b) {
            const std::uint8_t* block = r + std::size_t(b) * 34;
            const __m256i codes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block + 2));
            const __m256i magnitude = _mm256_sign_epi8(codes, codes);
            const float d           = half_at(block);
            for (int t = 0; t < tokens; ++t) {
                const __m256i a =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.codes(t) + b * 32));
                const __m256i dot = dot_u8s8(magnitude, _mm256_sign_epi8(a, codes));
                acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot),
                                         _mm256_set1_ps(d * x.scales(t)[b]), acc[t]);
            }
        }
        for (int t = 0; t < tokens; ++t) { y[std::size_t(t) * y_stride + row] = reduce(acc[t]); }
    }
}

} // namespace ninfer::ops::gguf_cpu_detail
