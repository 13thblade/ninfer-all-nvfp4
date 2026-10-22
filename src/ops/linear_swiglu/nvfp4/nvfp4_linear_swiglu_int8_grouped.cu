#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_int8_grouped.cuh"

#include "core/device.h"
#include "core/weight.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_int8_fused_norm_quant.h"
#include "ops/linear/nvfp4/nvfp4_int8_grouped_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_int8_narrow_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_mmq_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_mmvq_launch.cuh"

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
// Sized from the width the launch ADDRESSES, which the table decides: the tile-padded chains round a
// ragged or forced width up and write and read to the tile end, while the narrow band's width is the
// call's. The gate plane is indexed by that same column count, which is why the query and the launch
// have to ask one question rather than two -- the up half's SiLU reads columns the gate half wrote.
struct WorkspaceParts {
    std::size_t codes;
    std::size_t scales;
    std::size_t gate_plane;
};

constexpr WorkspaceParts workspace_parts(std::int32_t addressed_tokens) {
    const std::size_t t = static_cast<std::size_t>(addressed_tokens);
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
                                                              std::int32_t max_tokens,
                                                              Nvfp4Int8Mainloop table) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_swiglu int8 workspace: invalid token interval");
    }
    if (min_tokens == max_tokens) {
        // A single-T query reports the route that T will actually run: under `default` only a whole
        // 128-token tile takes this route and every other width stays on the workspace-free A16
        // routes, while a forced route takes all of them -- and which planes it takes is the table's
        // answer as well, because the narrow band sizes to the call and the port to the tile.
        if (!nvfp4_int8_route_admits(min_tokens)) { return 0; }
        const WorkspaceParts parts =
            workspace_parts(narrow::addressed_tokens(min_tokens, table));
        return parts.codes + parts.scales + parts.gate_plane;
    }
    // An interval reserves for the widest width in it that the route can reach, rounded up to whole
    // tiles by workspace_parts where the table pads. Both the narrow and the padded sizing grow
    // monotonically with the width and meet at eight, so the widest point of the interval is also
    // its largest reservation; rounding the interval DOWN here would under-reserve a padded launch.
    if (!nvfp4_int8_route_interval_admits(min_tokens, max_tokens)) { return 0; }
    const WorkspaceParts parts = workspace_parts(narrow::addressed_tokens(max_tokens, table));
    return parts.codes + parts.scales + parts.gate_plane;
}

namespace {

// The two-projection mainloop half, identical whichever way the activation planes were filled.
// The grouped mainloop is the exception: it pairs a gate row with its up partner inside the block
// (Nvfp4Int8SwiGluRowPolicy), so it needs no gate plane and no second launch.
void launch_from_planes(const Weight& weight, const void* codes, const void* scales,
                        const void* gate_plane, Tensor& out, bool narrow_route, bool mmq_route,
                        bool mmvq_route, std::int32_t tokens, cudaStream_t stream) {
    // alpha = 1/weight_scale_divisor: the E2M1 1/2 already folded into the E4M3 scale decode,
    // where llama.cpp folds it too, like the A16 route's global divisor. Both
    // projections share it -- the gate plane holds the rescaled gate, so silu(gate)*up is the
    // same FP32 expression the paired grouped kernel evaluates from its paired accumulators.
    const float alpha = 1.0F / weight.weight_scale_divisor;
    const auto* weight_codes  = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* weight_scales = static_cast<const std::uint8_t*>(weight.scales);
    const auto* x_codes       = reinterpret_cast<const std::int8_t*>(codes);
    const auto* x_scales      = reinterpret_cast<const float*>(scales);

    if (mmvq_route || mmq_route) {
        const LinearF32Output gate_output{reinterpret_cast<float*>(const_cast<void*>(gate_plane)),
                                          kOut};
        const LinearBf16GateSiluOutput up_output{
            static_cast<__nv_bfloat16*>(out.data), kOut,
            reinterpret_cast<const float*>(gate_plane), kOut};
        if (mmvq_route) {
            if (narrow_route) {
                narrow::launch<LinearF32Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                alpha, gate_output, kCols, tokens, kOut, stream);
                narrow::launch<LinearBf16GateSiluOutput>(
                    weight_codes + kHalfCodeBytes, weight_scales + kHalfScaleBytes, x_codes,
                    x_scales, alpha, up_output, kCols, tokens, kOut, stream);
            } else {
                mmvq::launch<LinearF32Output>(weight_codes, weight_scales, x_codes, x_scales, alpha,
                                              gate_output, kCols, tokens, kOut, stream);
                mmvq::launch<LinearBf16GateSiluOutput>(
                    weight_codes + kHalfCodeBytes, weight_scales + kHalfScaleBytes, x_codes,
                    x_scales, alpha, up_output, kCols, tokens, kOut, stream);
            }
        } else {
            mmq::launch<LinearF32Output>(weight_codes, weight_scales, x_codes, alpha, gate_output,
                                         kCols, tokens, kOut, stream);
            mmq::launch<LinearBf16GateSiluOutput>(weight_codes + kHalfCodeBytes,
                                                  weight_scales + kHalfScaleBytes, x_codes, alpha,
                                                  up_output, kCols, tokens, kOut, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const LinearBf16Output paired_output{static_cast<__nv_bfloat16*>(out.data), kOut};
    nvfp4_int8_grouped_swiglu_launch<LinearBf16Output>(
        weight_codes, weight_scales, x_codes, x_scales, alpha, paired_output, kCols, tokens, kRows,
        stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void nvfp4_linear_swiglu_int8_grouped_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                             WorkspaceArena& workspace, Nvfp4Int8Mainloop table,
                                             cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    // The unified table answers this width with the narrow kernel, whose columns are the call's width,
    // so the planes are the call's rows. Allocation, the quantiser's grid and the gate plane this call
    // hands from its gate projection to its up projection all follow that one number: a gate column
    // the first kernel never wrote is a SiLU argument the second one still reads.
    const bool narrow_route = narrow::select(tokens, table);
    const std::int32_t padded = narrow::addressed_tokens(tokens, table);
    if (!nvfp4_linear_swiglu_int8_grouped_supported(weight, tokens)) {
        throw std::invalid_argument("nvfp4 linear_swiglu int8: unsupported profile");
    }
    (void)validate_nvfp4_weight(weight, "nvfp4 linear_swiglu int8");

    auto scope = workspace.scope();
    // Two projections read the same activations, so the plane is quantised once: interleaved
    // (four FP32 scales then 128 s8 codes per token per 128-column window) for the MMQ mainloop,
    // two separated planes for the grouped one. Both cost K + K/8 bytes per token.
    const bool mmq_route  = mmq::select(kCols, tokens, table);
    // Below the crossover the MMVQ-shaped mainloop reads these same two planes, so it needs no
    // second quantisation pass and none of the interleaved form's routing.
    const bool mmvq_route = mmvq::select(tokens);
    DeviceSpan codes;
    DeviceSpan scales;
    if (mmq_route) {
        codes = workspace.alloc_bytes(mmq::activation_bytes(tokens, kCols));
    } else {
        codes  = workspace.alloc_bytes(static_cast<std::size_t>(padded) * kCols);
        scales = workspace.alloc_bytes(static_cast<std::size_t>(padded) * (kCols / 32) *
                                       sizeof(float));
    }
    // The grouped mainloop pairs gate and up rows inside the block, so only the two-projection
    // routes below the crossover need the FP32 gate plane.
    const bool two_projection = mmq_route || mmvq_route;
    const DeviceSpan gate_plane =
        two_projection ? workspace.alloc_bytes(std::size_t(kOut) * padded * sizeof(float))
                       : DeviceSpan{};

    if (mmq_route) {
        mmq::quantize_activations(x.data, tokens, kCols, codes.data, stream);
    } else {
        quantize_nvfp4_int8_grouped<<<padded, 128, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data), tokens, kCols,
            reinterpret_cast<std::int8_t*>(codes.data), reinterpret_cast<float*>(scales.data));
    }

    launch_from_planes(weight, codes.data, scales.data, gate_plane.data, out, narrow_route,
                       mmq_route, mmvq_route, tokens, stream);
}

// The fused-edge entry for the gate_up pair: the RMSNorm that produced these activations folds into
// the single quantise pass, so the normed row never round-trips through global BF16 and the model
// layer skips one launch. Both projections still read the one plane pair, and the mainloop half is
// the shared launch_from_planes above. The unified table only: the interleaved MMQ plane belongs to
// the own chain, whose files are frozen.
void nvfp4_linear_swiglu_int8_fused_norm_launch(const Tensor& residual, const Tensor& norm_weight,
                                                float eps, const Weight& weight, Tensor& out,
                                                WorkspaceArena& workspace, Nvfp4Int8Mainloop table,
                                                cudaStream_t stream) {
    const std::int32_t tokens = residual.ne[1];
    const bool narrow_route   = narrow::select(tokens, table);
    const std::int32_t padded = narrow::addressed_tokens(tokens, table);
    if (!nvfp4_linear_swiglu_int8_grouped_supported(weight, tokens)) {
        throw std::invalid_argument("nvfp4 linear_swiglu int8 fused: unsupported profile");
    }
    (void)validate_nvfp4_weight(weight, "nvfp4 linear_swiglu int8 fused");
    if (residual.ne[0] != kCols) {
        throw std::invalid_argument("nvfp4 linear_swiglu int8 fused: residual width must be 5120");
    }

    auto scope = workspace.scope();
    const bool mmq_route  = mmq::select(kCols, tokens, table);
    const bool mmvq_route = mmvq::select(tokens);
    if (mmq_route) {
        throw std::invalid_argument(
            "nvfp4 linear_swiglu int8 fused: the interleaved MMQ plane is the own chain's");
    }
    const DeviceSpan codes  = workspace.alloc_bytes(static_cast<std::size_t>(padded) * kCols);
    const DeviceSpan scales = workspace.alloc_bytes(static_cast<std::size_t>(padded) *
                                                    (kCols / 32) * sizeof(float));
    const bool two_projection = mmq_route || mmvq_route;
    const DeviceSpan gate_plane =
        two_projection ? workspace.alloc_bytes(std::size_t(kOut) * padded * sizeof(float))
                       : DeviceSpan{};

    launch_rmsnorm_int8_quantize(residual, norm_weight, eps,
                                 static_cast<std::int8_t*>(codes.data),
                                 static_cast<float*>(scales.data), padded, stream);
    launch_from_planes(weight, codes.data, scales.data, gate_plane.data, out, narrow_route,
                       mmq_route, mmvq_route, tokens, stream);
}
} // namespace ninfer::ops::detail::unified
