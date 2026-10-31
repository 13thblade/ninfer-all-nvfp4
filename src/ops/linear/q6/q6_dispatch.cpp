#include "ops/linear/q6/q6_dispatch.h"
#include "ops/linear/common/route_table.h"
#include "ops/linear/q6/q6_shapes.h"
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
struct ShapeEntry {
    std::int32_t n, k;
    Q6Launch (*select)(std::int32_t);
    // The unified-template table; shapes without one keep the legacy routes everywhere.
    Q6Launch (*unified)(std::int32_t) = nullptr;
    // A device-profile route that, where it names a schedule, takes precedence over both tables.
    // Its launches also write FP32 out.
    Q6Launch (*routed)(std::int32_t) = nullptr;
    // The two tables writing FP32; shapes without them take the runtime-shape route for FP32 out.
    Q6Launch (*fp32)(std::int32_t)         = nullptr;
    Q6Launch (*unified_fp32)(std::int32_t) = nullptr;
};

constexpr std::array kShapes{
    ShapeEntry{248320, 5120, select_q6_n248320_k5120, select_q6_n248320_k5120_unified,
               routed_q6_n248320_k5120, select_q6_n248320_k5120_fp32,
               select_q6_n248320_k5120_unified_fp32},
    ShapeEntry{34816, 5120, select_q6_n34816_k5120},
    ShapeEntry{248320, 2048, select_q6_n248320_k2048, select_q6_n248320_k2048_unified},
    ShapeEntry{1152, 1536, select_q6_n1152_k1536, select_q6_n1152_k1536_unified},
};
} // namespace

Q6Launch select_q6_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("q6 linear: T must be positive");
    for (const auto& entry : kShapes) {
        if (entry.n != n || entry.k != k) continue;
        if (entry.routed != nullptr) {
            if (const Q6Launch launch = entry.routed(t)) { return launch; }
        }
        if (entry.unified != nullptr &&
            linear_route_table(LinearRouteFamily::Q6, t) == LinearRouteTable::Unified) {
            return entry.unified(t);
        }
        return entry.select(t);
    }
    return select_q6_general_launch(t, DType::BF16);
}

Q6Launch select_q6_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q6 linear: unsupported policy");
    return select_q6_a16_launch(n, k, t);
}

// FP32 out chooses as BF16 does among a shape's FP32 tables, else takes the runtime-shape route.
Q6Launch select_q6_fp32_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    for (const auto& entry : kShapes) {
        if (entry.n != n || entry.k != k || entry.fp32 == nullptr) continue;
        if (entry.routed != nullptr) {
            if (const Q6Launch launch = entry.routed(t)) { return launch; }
        }
        if (entry.unified_fp32 != nullptr &&
            linear_route_table(LinearRouteFamily::Q6, t) == LinearRouteTable::Unified) {
            return entry.unified_fp32(t);
        }
        return entry.fp32(t);
    }
    return select_q6_general_launch(t, DType::FP32);
}

void q6_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    const Q6Launch launch = out.dtype == DType::FP32
                                ? select_q6_fp32_launch(weight.n, weight.k, x.ne[1])
                                : select_q6_launch(weight.n, weight.k, x.ne[1], policy);
    launch(x, weight, out, stream);
}
} // namespace ninfer::ops::detail
