#pragma once

// Adapted from Infernix a3edb450 src/models/qwen4_exp/program/expert_cache/expert_state.h (Apache-2.0).
// Modified for NInfer-3090: NInfer's namespace.

// The expert cache's saved state (design §19.3.5 S4b, warm start): every expert's LFRU count and the
// resident experts, highest score first, written at stop and between requests and read at the next
// start to fill the VRAM frames before the first request. A file is used only when its identity
// (the artifact and the key count) matches; anything else is ignored with a reason.

#include "models/qwen4_exp/program/expert_cache/expert_cache.h"

#include <filesystem>
#include <optional>
#include <string>
#include <cstdint>
#include <string_view>

namespace ninfer::models::qwen4_exp::expert_cache {

struct ExpertStateLoad {
    std::optional<SavedState> state;
    std::string message; // why nothing was loaded
};

// Writes `state` to `path` through `path.tmp` and a rename; returns an error message, empty on success.
[[nodiscard]] std::string save_expert_state(const std::filesystem::path& path, std::string_view identity,
                                            const SavedState& state);

// Reads a state saved for `identity` with `keys` keys.
[[nodiscard]] ExpertStateLoad load_expert_state(const std::filesystem::path& path, std::string_view identity,
                                                std::uint32_t keys);

} // namespace ninfer::models::qwen4_exp::expert_cache
