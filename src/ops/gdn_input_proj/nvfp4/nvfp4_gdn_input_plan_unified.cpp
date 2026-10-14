#include "core/weight.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan_unified.h"

#include "ops/linear/nvfp4/nvfp4_int8_grouped_host.h"
#include "ops/linear/nvfp4/nvfp4_layout.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail::unified {
namespace {

enum class Nvfp4GdnInputRoute : std::uint8_t {
    A16,
    Int8Grouped,
    A4,
};

Nvfp4GdnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 gdn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) {
        return Nvfp4GdnInputRoute::A16;
    }
    if (policy == LinearPolicy::AllowNvfp4Int8) {
        // The named route decides the width, not the tile count: a forced grouped route
        // takes every width and pads its last token tile, so this is the route's own
        // admission. Under `default` only whole 128-token tiles get here.
        return nvfp4_int8_route_admits(tokens) ? Nvfp4GdnInputRoute::Int8Grouped
       : Nvfp4GdnInputRoute::A16;
    }
    if (allows_a4(policy)) { return Nvfp4GdnInputRoute::A4; }
    throw std::invalid_argument("nvfp4 gdn_input_proj: unsupported policy");
}


} // namespace

std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 gdn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    const Nvfp4GdnInputRoute widest = resolve_route(policy, max_tokens);
    if (widest == Nvfp4GdnInputRoute::A4) {
        return nvfp4_a4_workspace_capacity_bytes(max_tokens, Nvfp4N16384K5120::kInputRows);
    }
    if (widest == Nvfp4GdnInputRoute::Int8Grouped) {
        // Sized for the padded width a launch addresses -- nvfp4_int8_grouped_workspace_bytes
        // rounds up to whole tiles itself, so a single T and an interval ask the same question.
        return nvfp4_int8_grouped_workspace_bytes(max_tokens, Nvfp4N16384K5120::kInputRows);
    }
    return 0;
}

void nvfp4_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                              LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    switch (resolve_route(policy, x.ne[1])) {
    case Nvfp4GdnInputRoute::A16:
        nvfp4_gdn_input_a16_launch(x, weight, qkv, z, stream);
        return;
    case Nvfp4GdnInputRoute::Int8Grouped:
        if (workspace == nullptr) {
            throw std::invalid_argument("nvfp4 int8 gdn_input_proj requires caller workspace");
        }
        nvfp4_gdn_input_int8_launch(x, weight, qkv, z, *workspace, stream);
        return;
    case Nvfp4GdnInputRoute::A4:
        break;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 A4 gdn_input_proj requires caller workspace");
    }
    auto scope                     = workspace->scope();
    const Nvfp4A4Workspace scratch = allocate_nvfp4_a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_gdn_input_a4_launch(x, weight, qkv, z, scratch, stream);
}

} // namespace ninfer::ops::detail::unified
