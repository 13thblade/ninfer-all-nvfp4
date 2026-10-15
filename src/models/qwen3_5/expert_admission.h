#pragma once

#include <cstdint>
#include <optional>

namespace ninfer::models::qwen3_5 {

// Explicit cache capacity gets priority above the minimum legal runtime when KV is automatic.
// Otherwise KV gets its requested/resolved shape first. In particular auto KV + auto cache can
// leave no persistent cache; transient slots still deliver every route.
[[nodiscard]] inline std::uint64_t native_expert_runtime_reservation(
    bool automatic_kv, std::optional<std::uint64_t> cache_bytes,
    std::uint64_t resolved_runtime, std::uint64_t minimum_runtime) noexcept {
    return automatic_kv && cache_bytes && *cache_bytes != 0 ? minimum_runtime : resolved_runtime;
}

} // namespace ninfer::models::qwen3_5
