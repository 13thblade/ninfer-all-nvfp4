#pragma once

#include "models/qwen3_5/config.h"

namespace ninfer::models::qwen3_5 {

// The artifact's `vision` component (target text) as the Qwen3.5 vision tower's config. Qwen4Exp
// carries the same tower and binds it through this config.
[[nodiscard]] VisionConfig parse_vision_component(const artifact::Directory& directory);

} // namespace ninfer::models::qwen3_5
