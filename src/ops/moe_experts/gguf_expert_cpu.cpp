#include "ops/moe_experts/gguf_expert_cpu.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>

#if defined(NINFER_CPU_EXPERT_X86) && (defined(__GNUC__) || defined(__clang__))
#    include <cpuid.h>
#endif
#if defined(_MSC_VER) && defined(NINFER_CPU_EXPERT_X86)
#    include <intrin.h>
#endif

namespace ninfer::ops {
namespace gguf_cpu_detail {

float half_to_float(std::uint16_t bits) noexcept {
    unsigned exponent = (bits >> 10) & 31U, mantissa = bits & 1023U;
    const auto sign   = std::uint32_t(bits & 0x8000U) << 16;
    if (exponent == 0) {
        if (mantissa == 0) { return std::bit_cast<float>(sign); }
        int shift = 0;
        while ((mantissa & 1024U) == 0) {
            mantissa <<= 1;
            ++shift;
        }
        return std::bit_cast<float>(sign | (std::uint32_t(113 - shift) << 23) |
                                    ((mantissa & 1023U) << 13));
    }
    return std::bit_cast<float>(sign | ((exponent == 31 ? 255U : exponent + 112U) << 23) |
                                (mantissa << 13));
}

namespace {
std::uint16_t load16(const std::uint8_t* p) {
    std::uint16_t value;
    std::memcpy(&value, p, 2);
    return value;
}
} // namespace

void rows_scalar(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
                 float* y, std::size_t y_stride) {
    const int tokens = x.tokens();
    if (w.format == QType::GGUF_Q2_0) {
        const int blocks = w.k / 64;
        for (int row = row_begin; row < row_end; ++row) {
            const std::uint8_t* r = w.data + std::size_t(row) * w.row_bytes;
            for (int t = 0; t < tokens; ++t) {
                const std::int8_t* a = x.codes(t);
                float sum            = 0;
                for (int b = 0; b < blocks; ++b) {
                    const std::uint8_t* block = r + std::size_t(b) * 18;
                    const float d             = half_to_float(load16(block));
                    std::int32_t dot          = 0;
                    for (int j = 0; j < 64; ++j) {
                        const int code = (block[2 + j / 4] >> (2 * (j % 4))) & 3;
                        dot += code * a[b * 64 + q2_lane_position(j)];
                    }
                    sum += d * x.scales(t)[b] * float(dot - x.sums(t)[b]);
                }
                y[std::size_t(t) * y_stride + row] = sum;
            }
        }
        return;
    }
    // Q8_0: a binary16 d then 32 signed codes.
    const int blocks = w.k / 32;
    for (int row = row_begin; row < row_end; ++row) {
        const std::uint8_t* r = w.data + std::size_t(row) * w.row_bytes;
        for (int t = 0; t < tokens; ++t) {
            const std::int8_t* a = x.codes(t);
            float sum            = 0;
            for (int b = 0; b < blocks; ++b) {
                const std::uint8_t* block = r + std::size_t(b) * 34;
                const float d             = half_to_float(load16(block));
                std::int32_t dot          = 0;
                for (int j = 0; j < 32; ++j) {
                    dot += int(static_cast<std::int8_t>(block[2 + j])) * a[b * 32 + j];
                }
                sum += d * x.scales(t)[b] * float(dot);
            }
            y[std::size_t(t) * y_stride + row] = sum;
        }
    }
}

} // namespace gguf_cpu_detail

namespace {
#if defined(NINFER_CPU_EXPERT_X86) && defined(_MSC_VER)
// MSVC has no __builtin_cpu_supports: CPUID leaves 1 and 7 and the register state the OS saves
// (XCR0) answer the same questions.
struct CpuFeatures {
    unsigned leaf1_ecx = 0, leaf7_ebx = 0, leaf7_ecx = 0;
    unsigned long long xcr0 = 0;
};

CpuFeatures cpu_features() noexcept {
    CpuFeatures out;
    int regs[4];
    __cpuid(regs, 0);
    if (regs[0] < 7) { return out; }
    __cpuid(regs, 1);
    out.leaf1_ecx = static_cast<unsigned>(regs[2]);
    if ((out.leaf1_ecx & (1U << 27)) == 0) { return out; } // no OSXSAVE: no AVX state
    out.xcr0 = _xgetbv(0);
    __cpuidex(regs, 7, 0);
    out.leaf7_ebx = static_cast<unsigned>(regs[1]);
    out.leaf7_ecx = static_cast<unsigned>(regs[2]);
    return out;
}

constexpr bool all_bits(unsigned value, unsigned mask) noexcept { return (value & mask) == mask; }
#endif

bool cpu_has_avx2() noexcept {
#if defined(NINFER_CPU_EXPERT_X86) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
           __builtin_cpu_supports("f16c");
#elif defined(NINFER_CPU_EXPERT_X86) && defined(_MSC_VER)
    const CpuFeatures cpu = cpu_features();
    // XCR0 SSE and AVX state; leaf 1 FMA, AVX, F16C; leaf 7 AVX2.
    return (cpu.xcr0 & 0x6U) == 0x6U &&
           all_bits(cpu.leaf1_ecx, (1U << 12) | (1U << 28) | (1U << 29)) &&
           all_bits(cpu.leaf7_ebx, 1U << 5);
#else
    return false;
#endif
}

bool cpu_has_avx512_vnni() noexcept {
#if defined(NINFER_CPU_EXPERT_X86) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("f16c");
#elif defined(NINFER_CPU_EXPERT_X86) && defined(_MSC_VER)
    const CpuFeatures cpu = cpu_features();
    // XCR0 SSE, AVX, opmask and both ZMM halves; leaf 1 F16C; leaf 7 AVX512F, DQ, BW, VL and VNNI.
    return (cpu.xcr0 & 0xe6U) == 0xe6U && all_bits(cpu.leaf1_ecx, 1U << 29) &&
           all_bits(cpu.leaf7_ebx, (1U << 16) | (1U << 17) | (1U << 30) | (1U << 31)) &&
           all_bits(cpu.leaf7_ecx, 1U << 11);
#else
    return false;
#endif
}

int group_of(QType format) {
    if (format == QType::GGUF_Q2_0) { return 64; }
    if (format == QType::GGUF_Q8_0) { return 32; }
    throw std::invalid_argument("GGUF CPU expert: unsupported block format");
}
} // namespace

bool gguf_cpu_supported(QType format) noexcept {
    return format == QType::GGUF_Q2_0 || format == QType::GGUF_Q8_0;
}

bool gguf_cpu_backend_available(GgufCpuBackend backend) noexcept {
    switch (backend) {
    case GgufCpuBackend::Automatic:
    case GgufCpuBackend::Scalar: return true;
    case GgufCpuBackend::Avx2: return cpu_has_avx2();
    case GgufCpuBackend::Avx512Vnni: return cpu_has_avx512_vnni();
    }
    return false;
}

const char* gguf_cpu_backend_name(GgufCpuBackend backend) noexcept {
    switch (gguf_cpu_backend_selected(backend)) {
    case GgufCpuBackend::Avx512Vnni: return "avx512-vnni";
    case GgufCpuBackend::Avx2: return "avx2";
    default: return "scalar";
    }
}

GgufCpuBackend gguf_cpu_backend_selected(GgufCpuBackend requested) noexcept {
    if (requested != GgufCpuBackend::Automatic) { return requested; }
    if (cpu_has_avx512_vnni()) { return GgufCpuBackend::Avx512Vnni; }
    if (cpu_has_avx2()) { return GgufCpuBackend::Avx2; }
    return GgufCpuBackend::Scalar;
}

GgufCpuActivation::GgufCpuActivation(int token_capacity, int k)
    : capacity_(token_capacity), k_(k), groups_(k / 32),
      codes_(std::size_t(token_capacity) * k), scales_(std::size_t(token_capacity) * (k / 32)),
      scratch_(std::size_t(k)), sums_(std::size_t(token_capacity) * (k / 32)) {
    if (token_capacity <= 0 || k <= 0 || k % 64 != 0) {
        throw std::invalid_argument("GGUF CPU activation: k must be a positive multiple of 64");
    }
}

void GgufCpuActivation::quantize(QType format, int token, const float* values) {
    const int group = group_of(format);
    std::int8_t* codes = codes_.data() + std::size_t(token) * k_;
    float* scales      = scales_.data() + std::size_t(token) * groups_;
    std::int32_t* sums = sums_.data() + std::size_t(token) * groups_;
    for (int start = 0, g = 0; start < k_; start += group, ++g) {
        float maximum = 0;
        for (int i = 0; i < group; ++i) {
            const float v = values[start + i];
            if (!std::isfinite(v)) { throw std::runtime_error("GGUF CPU expert: nonfinite activation"); }
            maximum = std::max(maximum, std::abs(v));
        }
        const float scale = maximum / 127.0F;
        const float inv   = scale == 0 ? 0.0F : 1.0F / scale;
        scales[g]         = scale;
        std::int32_t sum  = 0;
        for (int i = 0; i < group; ++i) {
            const auto code = static_cast<std::int8_t>(
                std::clamp(std::nearbyint(values[start + i] * inv), -127.0F, 127.0F));
            sum += code;
            const int at = group == 64 ? gguf_cpu_detail::q2_lane_position(i) : i;
            codes[start + at] = code;
        }
        sums[g] = sum;
    }
}

void GgufCpuActivation::prepare(QType format, const float* values, std::size_t stride, int tokens) {
    if (tokens <= 0 || tokens > capacity_) {
        throw std::invalid_argument("GGUF CPU activation: token count exceeds capacity");
    }
    format_ = format;
    tokens_ = tokens;
    for (int t = 0; t < tokens; ++t) { quantize(format, t, values + std::size_t(t) * stride); }
}

void GgufCpuActivation::prepare_bf16(QType format, const std::uint16_t* values, std::size_t stride,
                                     int tokens) {
    if (tokens <= 0 || tokens > capacity_) {
        throw std::invalid_argument("GGUF CPU activation: token count exceeds capacity");
    }
    format_ = format;
    tokens_ = tokens;
    for (int t = 0; t < tokens; ++t) {
        const std::uint16_t* column = values + std::size_t(t) * stride;
        for (int i = 0; i < k_; ++i) {
            scratch_[i] = std::bit_cast<float>(std::uint32_t(column[i]) << 16);
        }
        quantize(format, t, scratch_.data());
    }
}

void gguf_cpu_rows(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
                   float* y, std::size_t y_stride, GgufCpuBackend backend) {
    if (!gguf_cpu_supported(w.format) || w.format != x.format() || w.k != x.k() || !w.data ||
        row_begin < 0 || row_end > w.rows || row_begin > row_end) {
        throw std::invalid_argument("GGUF CPU expert: mismatched matrix, activation or rows");
    }
    switch (gguf_cpu_backend_selected(backend)) {
#if defined(NINFER_CPU_EXPERT_X86)
    case GgufCpuBackend::Avx512Vnni:
        return gguf_cpu_detail::rows_avx512(w, x, row_begin, row_end, y, y_stride);
    case GgufCpuBackend::Avx2: return gguf_cpu_detail::rows_avx2(w, x, row_begin, row_end, y, y_stride);
#endif
    default: return gguf_cpu_detail::rows_scalar(w, x, row_begin, row_end, y, y_stride);
    }
}

} // namespace ninfer::ops
