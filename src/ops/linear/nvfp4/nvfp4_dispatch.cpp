#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/common/route_table.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear/nvfp4/nvfp4_dispatch_unified.h"
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include <array>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {
const std::array kShapes{&kNvfp4N14336K5120, &kNvfp4N16384K5120, &kNvfp4N34816K5120,
                         &kNvfp4N5120K6144, &kNvfp4N5120K17408};

const Nvfp4LinearShape& resolve_shape(std::int32_t n, std::int32_t k, LinearPolicy policy) {
    if (!valid_linear_policy(policy))
        throw std::invalid_argument("nvfp4 linear: unsupported policy");
    for (const auto* shape : kShapes)
        if (shape->n == n && shape->k == k) return *shape;
    throw std::invalid_argument("nvfp4 linear: unsupported shape");
}
} // namespace

bool nvfp4_registered(std::int32_t n, std::int32_t k) {
    for (const auto* shape : kShapes)
        if (shape->n == n && shape->k == k) return true;
    return false;
}

static std::size_t nvfp4_linear_workspace_capacity_bytes_own(std::int32_t n, std::int32_t k,
                                                             LinearPolicy policy,
                                                             std::int32_t min_tokens,
                                                             std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("nvfp4 linear workspace: invalid token interval");
    const auto& shape = resolve_shape(n, k, policy);
    return allows_a4(policy) && shape.uses_a4(min_tokens, max_tokens)
               ? nvfp4_w4a4_workspace_capacity_bytes(max_tokens, k)
               : 0;
}

std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k,
                                                  LinearPolicy policy, std::int32_t min_tokens,
                                                  std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("nvfp4 linear workspace: invalid token interval");
    if (!valid_linear_policy(policy))
        throw std::invalid_argument("nvfp4 linear workspace: unsupported policy");
    const bool registered = nvfp4_registered(n, k);
    const bool int8       = allows_nvfp4_int8(policy);
    return capacity_by_table(
        min_tokens, max_tokens,
        [](std::int32_t width) { return linear_route_table(LinearRouteFamily::Nvfp4, width); },
        [&](LinearRouteTable table, std::int32_t first, std::int32_t last) -> std::size_t {
            // The plain int8 family is registered on the unified table alone: its rows there, and
            // its alignment admission for a geometry with no row. An own-table width has no plain
            // int8 route at all (the dispatcher says so), so its envelope is the own table's.
            if (int8) {
                return table == LinearRouteTable::Unified
                           ? unified::nvfp4_linear_workspace_capacity_bytes(n, k, policy, first,
                                                                            last)
                           : nvfp4_linear_workspace_capacity_bytes_own(n, k, policy, first, last);
            }
            // A geometry no table names runs the general runtime-shape route whenever the policy
            // allows one: A4 above its crossover (a tensor-parallel share's rows), A16 below it
            // and for every A16 request, which needs no scratch. This holds on either table --
            // the runtime shape is the route a geometry with no row answers on.
            if (!registered) {
                return allows_a4(policy) && nvfp4_general_uses_a4(k, last)
                           ? unified::nvfp4_a4_workspace_capacity_bytes(last, k)
                           : 0;
            }
            return table == LinearRouteTable::Unified
                       ? unified::nvfp4_linear_workspace_capacity_bytes(n, k, policy, first, last)
                       : nvfp4_linear_workspace_capacity_bytes_own(n, k, policy, first, last);
        });
}

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    WorkspaceArena* workspace, cudaStream_t stream) {
    validate_nvfp4_weight(weight, "nvfp4 linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("nvfp4 linear: T must be positive");
    if (!valid_linear_policy(policy))
        throw std::invalid_argument("nvfp4 linear: unsupported policy");
    const bool registered = nvfp4_registered(weight.n, weight.k);
    const bool unified_table =
        linear_route_table(LinearRouteFamily::Nvfp4, x.ne[1]) == LinearRouteTable::Unified;
    // The int8 family must never be answered by the general A16/A4 route: an "int8" run that never
    // touched an int8 kernel is the hollow-green failure this policy exists to avoid. A width the
    // unified table serves runs the unified dispatcher -- its rows, and the alignment admission for
    // a geometry with no row (the fork's extra rows live only there). An own-table width has no
    // plain-Linear int8 route, and the message names the knob that moves it.
    if (allows_nvfp4_int8(policy)) {
        if (!unified_table) {
            throw std::invalid_argument(
                "nvfp4 linear: T=" + std::to_string(x.ne[1]) +
                " resolves to the own (non-unified) shape table, which registers no plain-Linear "
                "int8 route. Run with NINFER_LINEAR_ROUTES=unified, or widen the device profile's "
                "nvfp4 width band to cover it");
        }
        unified::nvfp4_dispatch(x, weight, out, policy, workspace, stream);
        return;
    }
    // A tensor-parallel rank's partial sum is FP32, and the int8 family writes BF16, so the general
    // route answers it -- with A4 for a share above its crossover, A16 otherwise and for a
    // registered row a table still covers.
    if (out.dtype == DType::FP32) {
        if (!registered && allows_a4(policy) && nvfp4_general_uses_a4(weight.k, x.ne[1])) {
            if (workspace == nullptr)
                throw std::invalid_argument("nvfp4 A4 linear requires caller workspace");
            return nvfp4_general_a4(x, weight, out, *workspace, stream);
        }
        return nvfp4_general_a16(x, weight, out, stream);
    }
    // A geometry no table names runs the general runtime-shape route whatever table its width
    // resolves to: a tensor-parallel share's rows, whole runtime-shape tiles, A4 above the
    // crossover and A16 below it. That route is the rowless geometry's composition, so it stands
    // ahead of either table's row lookup.
    if (!registered) {
        if (allows_a4(policy) && nvfp4_general_uses_a4(weight.k, x.ne[1])) {
            if (workspace == nullptr)
                throw std::invalid_argument("nvfp4 A4 linear requires caller workspace");
            return nvfp4_general_a4(x, weight, out, *workspace, stream);
        }
        return nvfp4_general_a16(x, weight, out, stream);
    }
    // A registered row whose width resolves to the unified table is the unified dispatcher's.
    if (unified_table) {
        unified::nvfp4_dispatch(x, weight, out, policy, workspace, stream);
        return;
    }
    const auto& shape = resolve_shape(weight.n, weight.k, policy);
    if (!allows_a4(policy) || !shape.uses_a4(x.ne[1], x.ne[1]))
        return shape.a16(x, weight, out, stream);
    if (workspace == nullptr)
        throw std::invalid_argument("nvfp4 A4 linear requires caller workspace");
    auto scope         = workspace->scope();
    const auto scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    shape.a4(x, weight, out, scratch, stream);
}
} // namespace ninfer::ops::detail
