// Adapted from Infernix a3edb450 src/models/qwen4_exp/program/vision_program.cpp (Apache-2.0).
// Modified for NInfer-3090: see vision.h.

#include "models/qwen4_exp/execution/vision.h"

#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/execution/vision_overlay.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen4_exp::execution {
namespace {

constexpr std::size_t kAlignment = 256;

std::size_t align(std::size_t bytes) { return (bytes + kAlignment - 1) / kAlignment * kAlignment; }

} // namespace

VisionPassPlan plan_vision_pass(const qwen3_5::VisionConfig& config,
                                const qwen3_5::execution::VisionParameters& parameters,
                                const qwen3_5::VisionOverlayLayout& layout,
                                std::span<const std::size_t> merged_counts) {
    if (merged_counts.empty()) { throw std::invalid_argument("Qwen4Exp vision pass has no items"); }
    std::size_t largest = 0, tokens = 0;
    for (const std::size_t merged : merged_counts) {
        if (merged == 0) { throw std::invalid_argument("Qwen4Exp vision item has no merged tokens"); }
        largest = std::max(largest, merged);
        tokens += merged;
    }
    if (tokens > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        largest > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error("Qwen4Exp vision tokens exceed int32");
    }
    using qwen3_5::execution::VisionContext;
    const auto max_tokens = static_cast<std::uint32_t>(largest);
    // A general region sized by the encode itself puts the item's output right after its scratch
    // (Qwen3.5's plan_vision_window_workspace).
    const auto compact = VisionContext::plan_workspace(config, parameters, max_tokens, 1);
    const std::size_t patches =
        largest * static_cast<std::size_t>(config.spatial_merge_size) * config.spatial_merge_size;
    const std::size_t encode = VisionContext::workspace_bytes(config, parameters, patches, largest, compact);
    VisionPassPlan out;
    out.workspace     = VisionContext::plan_workspace(config, parameters, max_tokens, encode);
    out.staging_bytes = align(layout.staging_bytes);
    out.window_bytes  = out.staging_bytes + align(out.workspace.capacity_bytes);
    out.out           = out.workspace.output_hidden;
    out.tokens        = static_cast<std::uint32_t>(tokens);
    out.handoff_bytes = 2ULL * static_cast<std::size_t>(out.out) * tokens;
    return out;
}

std::size_t encode_vision(DeviceContext& device, const qwen3_5::VisionConfig& config,
                          const qwen3_5::execution::VisionParameters& parameters,
                          const qwen3_5::VisionOverlayLayout& layout, std::span<const std::byte> pinned_block,
                          std::span<const VisionEncodeItem> items, const VisionPassPlan& plan, DeviceSpan window,
                          Tensor& handoff, cudaStream_t stream) {
    if (window.data == nullptr || window.bytes < plan.window_bytes) {
        throw std::invalid_argument("Qwen4Exp vision window is smaller than its plan");
    }
    if (handoff.dtype != DType::BF16 || handoff.ne[0] != plan.out || handoff.ne[1] != static_cast<std::int32_t>(plan.tokens) ||
        !handoff.is_contiguous() || handoff.data == nullptr) {
        throw std::invalid_argument("Qwen4Exp vision handoff must be contiguous BF16 [out, tokens]");
    }
    auto* const base = static_cast<std::byte*>(window.data);
    const DeviceSpan backing{base + plan.staging_bytes, window.bytes - plan.staging_bytes};
    // The tower's prelude and merger upload at construction, its layers one slot ahead of compute.
    qwen3_5::execution::VisionWeightStream weights(device, layout, pinned_block, base);
    const qwen3_5::execution::VisionParameters staged = weights.window_parameters(parameters);
    const qwen3_5::execution::VisionContext context(device, config, staged, stream);
    std::size_t column = 0;
    for (const VisionEncodeItem& item : items) {
        if (item.control == nullptr) { throw std::invalid_argument("Qwen4Exp vision item has no control"); }
        const std::size_t merged = item.control->merged_count;
        if (column + merged > plan.tokens) { throw std::invalid_argument("Qwen4Exp vision items exceed the plan"); }
        weights.reset(stream);
        Tensor output = qwen3_5::execution::VisionContext::bind_output(backing, plan.workspace, merged);
        context.encode(qwen3_5::execution::VisionItemView{item.patches, item.control}, output, backing, plan.workspace,
                       &weights);
        const std::size_t column_bytes = 2ULL * static_cast<std::size_t>(plan.out);
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(handoff.data) + column_bytes * column, output.data,
                                   output.bytes(), cudaMemcpyDeviceToDevice, stream));
        column += merged;
    }
    if (column != plan.tokens) { throw std::invalid_argument("Qwen4Exp vision items do not fill the handoff"); }
    // Every upload into the window was awaited by `stream` before its use, so once `stream` passes
    // here nothing else writes the window; the stream's events are idle before they are destroyed.
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    return weights.uploaded_bytes();
}

} // namespace ninfer::models::qwen4_exp::execution
