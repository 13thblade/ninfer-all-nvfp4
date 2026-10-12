#pragma once

// Adapted from Infernix a3edb450 src/models/qwen4_exp/program/vision_program.cpp (vision_step) and
// vision_window.h (Apache-2.0).
// Modified for NInfer-3090: the pass runs this repo's Qwen3.5 tower (VisionContext::encode) one
// item at a time in one window, the tower streamed from the pinned host block through Qwen3.5's
// VisionWeightStream (overlay residency), each item's output copied to its columns of the handoff;
// no layer-major multi-item pass or stepped placement (one lane: nothing else waits on the encode).
//
// The vision encode of a Qwen4Exp prompt: every media item through the vision tower, the merged
// embeddings landing token-ordered in one handoff BF16 [out, tokens] that the prefill calls scatter
// over their image tokens.

#include "core/device.h"
#include "core/tensor.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/vision_control.h"
#include "models/qwen3_5/weights.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::models::qwen4_exp::execution {

// One item to encode: its processor patches (BF16 [patches, patch width], row-major) and control.
struct VisionEncodeItem {
    std::span<const std::uint16_t> patches;
    const qwen3_5::VisionItemControl* control = nullptr;
};

// The device memory of one encode pass.
struct VisionPassPlan {
    qwen3_5::detail::VisionWorkspacePlan workspace; // one item's encode, sized for the largest item
    std::size_t staging_bytes = 0;                  // the streamed tower's staging (256-aligned)
    std::size_t window_bytes  = 0;                  // staging then the encode workspace
    std::size_t handoff_bytes = 0;                  // BF16 [out, tokens]
    std::uint32_t tokens      = 0;                  // merged tokens of every item
    std::int32_t out          = 0;                  // the tower's output width (the text hidden)
};

// The window and handoff an encode of items with these merged-token counts needs.
[[nodiscard]] VisionPassPlan plan_vision_pass(const qwen3_5::VisionConfig& config,
                                              const qwen3_5::execution::VisionParameters& parameters,
                                              const qwen3_5::VisionOverlayLayout& layout,
                                              std::span<const std::size_t> merged_counts);

// Encodes `items` in order on `stream`: the tower is streamed from `pinned_block` (where
// `parameters` point) into the window's staging, each item runs in the window's workspace, and its
// merged embeddings BF16 [out, merged] are copied to the next columns of `handoff` [out, tokens].
// `window` holds plan.window_bytes and must not be touched by other work until the stream has
// passed this call. Returns the tower bytes uploaded.
std::size_t encode_vision(DeviceContext& device, const qwen3_5::VisionConfig& config,
                          const qwen3_5::execution::VisionParameters& parameters,
                          const qwen3_5::VisionOverlayLayout& layout, std::span<const std::byte> pinned_block,
                          std::span<const VisionEncodeItem> items, const VisionPassPlan& plan, DeviceSpan window,
                          Tensor& handoff, cudaStream_t stream);

} // namespace ninfer::models::qwen4_exp::execution
