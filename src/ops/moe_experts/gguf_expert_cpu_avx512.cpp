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

// Q2_0: one 16-byte code load per 64 weights, expanded once and reused by every token. The four
// 128-bit lanes take bit pairs 0, 2, 4 and 6 of the same bytes, the lane order the activations
// were stored in. Codes c in 0..3 stand for c - 1: sum (c - 1) a = dpbusd(c, a) - sum a.
template <int T>
void q2_rows(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
             float* y, std::size_t y_stride, int tokens) {
    const int blocks    = w.k / 64;
    const __m512i shift = _mm512_set_epi16(6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2,
                                           2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m512i three = _mm512_set1_epi8(3);
    const int count     = T > 0 ? T : tokens;
    for (int row = row_begin; row < row_end; ++row) {
        const std::uint8_t* r = w.data + std::size_t(row) * w.row_bytes;
        __m512 acc[T > 0 ? T : 16];
        float correction[T > 0 ? T : 16];
        for (int t = 0; t < count; ++t) {
            acc[t]        = _mm512_setzero_ps();
            correction[t] = 0;
        }
        for (int b = 0; b < blocks; ++b) {
            const std::uint8_t* block = r + std::size_t(b) * 18;
            const __m512i bytes =
                _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block + 2)));
            const __m512i codes = _mm512_and_si512(_mm512_srlv_epi16(bytes, shift), three);
            const float d       = half_at(block);
            for (int t = 0; t < count; ++t) {
                const __m512i a   = _mm512_loadu_si512(x.codes(t) + b * 64);
                const __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), codes, a);
                const float scale = d * x.scales(t)[b];
                acc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot), _mm512_set1_ps(scale), acc[t]);
                correction[t] += scale * float(x.sums(t)[b]);
            }
        }
        for (int t = 0; t < count; ++t) {
            y[std::size_t(t) * y_stride + row] = _mm512_reduce_add_ps(acc[t]) - correction[t];
        }
    }
}

// Q8_0: two 32-code blocks per vector. Signed codes become unsigned by flipping their sign bit
// (c + 128), so sum c a = dpbusd(c + 128, a) - 128 sum a, block by block.
template <int T>
void q8_rows(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
             float* y, std::size_t y_stride, int tokens) {
    const int blocks   = w.k / 32;
    const __m512i flip = _mm512_set1_epi8(static_cast<char>(0x80));
    const int count    = T > 0 ? T : tokens;
    for (int row = row_begin; row < row_end; ++row) {
        const std::uint8_t* r = w.data + std::size_t(row) * w.row_bytes;
        __m512 acc[T > 0 ? T : 16];
        float correction[T > 0 ? T : 16];
        for (int t = 0; t < count; ++t) {
            acc[t]        = _mm512_setzero_ps();
            correction[t] = 0;
        }
        for (int b = 0; b < blocks; b += 2) {
            const std::uint8_t* first  = r + std::size_t(b) * 34;
            const std::uint8_t* second = first + 34;
            const __m512i codes        = _mm512_xor_si512(
                _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256(
                                       reinterpret_cast<const __m256i*>(first + 2))),
                                   _mm256_loadu_si256(reinterpret_cast<const __m256i*>(second + 2)),
                                   1),
                flip);
            const float d0 = half_at(first), d1 = half_at(second);
            for (int t = 0; t < count; ++t) {
                const __m512i a   = _mm512_loadu_si512(x.codes(t) + b * 32);
                const __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), codes, a);
                const float s0 = d0 * x.scales(t)[b], s1 = d1 * x.scales(t)[b + 1];
                const __m512 scale =
                    _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(s0)),
                                       _mm256_set1_ps(s1), 1);
                acc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot), scale, acc[t]);
                correction[t] += 128.0F * (s0 * float(x.sums(t)[b]) + s1 * float(x.sums(t)[b + 1]));
            }
        }
        for (int t = 0; t < count; ++t) {
            y[std::size_t(t) * y_stride + row] = _mm512_reduce_add_ps(acc[t]) - correction[t];
        }
    }
}

} // namespace

void rows_avx512(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
                 float* y, std::size_t y_stride) {
    const int tokens = x.tokens();
    if (w.format == QType::GGUF_Q2_0) {
        switch (tokens) {
        case 1: return q2_rows<1>(w, x, row_begin, row_end, y, y_stride, tokens);
        case 2: return q2_rows<2>(w, x, row_begin, row_end, y, y_stride, tokens);
        case 4: return q2_rows<4>(w, x, row_begin, row_end, y, y_stride, tokens);
        default:
            if (tokens <= 16) { return q2_rows<0>(w, x, row_begin, row_end, y, y_stride, tokens); }
            return rows_scalar(w, x, row_begin, row_end, y, y_stride);
        }
    }
    if (w.k % 64 != 0) {
        return rows_scalar(w, x, row_begin, row_end, y, y_stride);
    }
    switch (tokens) {
    case 1: return q8_rows<1>(w, x, row_begin, row_end, y, y_stride, tokens);
    case 2: return q8_rows<2>(w, x, row_begin, row_end, y, y_stride, tokens);
    case 4: return q8_rows<4>(w, x, row_begin, row_end, y, y_stride, tokens);
    default:
        if (tokens <= 16) { return q8_rows<0>(w, x, row_begin, row_end, y, y_stride, tokens); }
        return rows_scalar(w, x, row_begin, row_end, y, y_stride);
    }
}

} // namespace ninfer::ops::gguf_cpu_detail
