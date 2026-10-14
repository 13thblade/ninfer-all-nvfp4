#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_int8_grouped.cuh"

#include "core/device.h"
#include "core/weight.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_int8_grouped_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_mmq_launch.cuh"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail::unified {
namespace {

// The registered 27B gate_up profile this route serves.
constexpr std::int32_t kRows = 34816;
constexpr std::int32_t kOut  = kRows / 2; // gate rows [0,17408) then up rows [17408,34816)
constexpr std::int32_t kCols = 5120;

// Row bytes of the E2M1 code plane and the row offset of the up half's block in the
// BlockScaleK16M128x4 scale plane: both halves' rows start on a 128-row tile boundary, so the
// up weight is the same single-source plane seen from row kOut.
constexpr std::size_t kHalfCodeBytes = std::size_t(kOut) * (kCols / 2);
constexpr std::size_t kHalfScaleBytes = std::size_t(kOut / 128) * (kCols / 64) * 512;

constexpr std::size_t round256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

// Activation planes (shared by both projections) followed by the gate projection's FP32 plane.
// Sized from the PADDED token count: the mainloop addresses whole 128-token tiles, so a ragged or
// forced width writes and reads up to the tile end, and the gate plane is indexed by that same
// padded column.
struct WorkspaceParts {
    std::size_t codes;
    std::size_t scales;
    std::size_t gate_plane;
};

constexpr WorkspaceParts workspace_parts(std::int32_t tokens) {
    const std::size_t t = static_cast<std::size_t>(nvfp4_int8_padded_tokens(tokens));
    const std::size_t codes    = round256(t * std::size_t(kCols));
    const std::size_t scales   = round256(t * (kCols / 32) * sizeof(float));
    const std::size_t gate_f32 = round256(std::size_t(kOut) * t * sizeof(float));
    return {codes, scales, gate_f32};
}

} // namespace

bool nvfp4_linear_swiglu_int8_tokens_supported(std::int32_t tokens) {
    return nvfp4_int8_route_admits(tokens);
}

bool nvfp4_linear_swiglu_int8_grouped_supported(const Weight& gate_up, std::int32_t tokens) {
    return gate_up.qtype == QType::NVFP4 && gate_up.layout == QuantLayout::BlockScaleK16M128x4 &&
           gate_up.group == 16 && gate_up.n == kRows && gate_up.k == kCols &&
           gate_up.qdata != nullptr && gate_up.scales != nullptr &&
           nvfp4_int8_route_admits(tokens);
}

std::size_t nvfp4_linear_swiglu_int8_workspace_capacity_bytes(std::int32_t min_tokens,
                                                              std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_swiglu int8 workspace: invalid token interval");
    }
    if (min_tokens == max_tokens) {
        // A single-T query reports the route that T will actually run: under `default` only a whole
        // 128-token tile takes this route and every other width stays on the workspace-free A16
        // routes, while a forced grouped route takes all of them.
        if (!nvfp4_int8_route_admits(min_tokens)) { return 0; }
        const WorkspaceParts parts = workspace_parts(min_tokens);
        return parts.codes + parts.scales + parts.gate_plane;
    }
    // An interval reserves for the widest width in it that the route can reach, rounded up to whole
    // tiles by workspace_parts. Under a forced route that is max_tokens itself; under the historical
    // cascade the largest whole tile at or below max_tokens, which admission guarantees is at or
    // above min_tokens. Rounding the interval DOWN here would under-reserve a padded launch.
    if (!nvfp4_int8_route_interval_admits(min_tokens, max_tokens)) { return 0; }
    const WorkspaceParts parts = workspace_parts(max_tokens);
    return parts.codes + parts.scales + parts.gate_plane;
}

void nvfp4_linear_swiglu_int8_grouped_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                             WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    const std::int32_t padded = nvfp4_int8_padded_tokens(tokens);
    if (!nvfp4_linear_swiglu_int8_grouped_supported(weight, tokens)) {
        throw std::invalid_argument("nvfp4 linear_swiglu int8: unsupported profile");
    }
    (void)validate_nvfp4_weight(weight, "nvfp4 linear_swiglu int8");

    auto scope = workspace.scope();
    // Two projections read the same activations, so the plane is quantised once: interleaved
    // (four FP32 scales then 128 s8 codes per token per 128-column window) for the MMQ mainloop,
    // two separated planes for the grouped one. Both cost K + K/8 bytes per token.
    const bool mmq_route = mmq::select(kCols, tokens);
    DeviceSpan codes;
    DeviceSpan scales;
    if (mmq_route) {
        codes = workspace.alloc_bytes(mmq::activation_bytes(tokens, kCols));
    } else {
        codes  = workspace.alloc_bytes(static_cast<std::size_t>(padded) * kCols);
        scales = workspace.alloc_bytes(static_cast<std::size_t>(padded) * (kCols / 32) *
                                       sizeof(float));
    }
    const DeviceSpan gate_plane =
        workspace.alloc_bytes(std::size_t(kOut) * padded * sizeof(float));

    if (mmq_route) {
        mmq::quantize_activations(x.data, tokens, kCols, codes.data, stream);
    } else {
        quantize_nvfp4_int8_grouped<<<padded, 128, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data), tokens, kCols,
            reinterpret_cast<std::int8_t*>(codes.data), reinterpret_cast<float*>(scales.data));
    }

    // alpha = 1/weight_scale_divisor: the E2M1 1/2 already folded into the E4M3 scale decode,
    // where llama.cpp folds it too, like the A16 route's global divisor. Both
    // projections share it -- the gate plane holds the rescaled gate, so silu(gate)*up is the
    // same FP32 expression the retired fused kernel evaluated from its paired accumulators.
    const float alpha = 1.0F / weight.weight_scale_divisor;
    const auto* weight_codes  = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* weight_scales = static_cast<const std::uint8_t*>(weight.scales);
    const auto* x_codes       = reinterpret_cast<const std::int8_t*>(codes.data);
    const auto* x_scales      = reinterpret_cast<const float*>(scales.data);
    const dim3 grid(kOut / 128, padded / 128);

    const LinearF32Output gate_output{reinterpret_cast<float*>(gate_plane.data), kOut};
    if (mmq_route) {
        mmq::launch<128, LinearF32Output>(weight_codes, weight_scales, x_codes, alpha, gate_output,
                                          kCols, tokens, kOut, stream);
        const LinearBf16GateSiluOutput up_output{static_cast<__nv_bfloat16*>(out.data), kOut,
                                                reinterpret_cast<const float*>(gate_plane.data),
                                                kOut};
        mmq::launch<128, LinearBf16GateSiluOutput>(weight_codes + kHalfCodeBytes,
                                                   weight_scales + kHalfScaleBytes, x_codes, alpha,
                                                   up_output, kCols, tokens, kOut, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    nvfp4_int8_grouped_kernel<LinearF32Output>
        <<<grid, detail::kInt8Threads, nvfp4_int8_grouped_smem_bytes(), stream>>>(
            weight_codes, weight_scales, x_codes, x_scales, alpha, gate_output, kCols, tokens);

    const LinearBf16GateSiluOutput up_output{static_cast<__nv_bfloat16*>(out.data), kOut,
                                            reinterpret_cast<const float*>(gate_plane.data), kOut};
    nvfp4_int8_grouped_kernel<LinearBf16GateSiluOutput>
        <<<grid, detail::kInt8Threads, nvfp4_int8_grouped_smem_bytes(), stream>>>(
            weight_codes + kHalfCodeBytes, weight_scales + kHalfScaleBytes, x_codes, x_scales,
            alpha, up_output, kCols, tokens);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::unified
