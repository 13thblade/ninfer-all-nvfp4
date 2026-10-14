// The plain Linear int8-grouped route: identity rows over any registered NVFP4 geometry. A whole
// 128-token tile is what the mainloop addresses, so a width that is not one is served by padding the
// last tile and dropping its surplus columns -- the route answers for every positive token count.
// Shares the mainloop with the linear_add, attn/gdn input-projection and SwiGLU routes via
// nvfp4_int8_grouped_mma.cuh; this file adds the instantiation and the shape-checked launcher.

#include "core/device.h"
#include "core/weight.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_int8_grouped_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_mmq_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_shapes_unified.h"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail::unified {

bool nvfp4_linear_int8_tokens_supported(std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) { return false; }
    if (min_tokens == max_tokens) { return nvfp4_int8_route_admits(min_tokens); }
    return nvfp4_int8_route_interval_admits(min_tokens, max_tokens);
}

void nvfp4_linear_int8_grouped_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                      WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    const std::int32_t padded = nvfp4_int8_padded_tokens(tokens);
    if (weight.qtype != QType::NVFP4 || weight.layout != QuantLayout::BlockScaleK16M128x4 ||
        weight.group != 16 || weight.n <= 0 || weight.n % 128 || weight.k <= 0 || weight.k % 64 ||
        weight.qdata == nullptr || weight.scales == nullptr ||
        !nvfp4_int8_route_admits(tokens)) {
        throw std::invalid_argument("nvfp4 linear int8: unsupported profile");
    }
    (void)validate_nvfp4_weight(weight, "nvfp4 linear int8");

    auto scope = workspace.scope();
    // The two routes quantise the same activations but lay the plane out differently: the MMQ
    // mainloop consumes one interleaved plane (four FP32 scales then 128 s8 codes per token per
    // 128-column window), the grouped mainloop the two separated planes. Both cost K + K/8 bytes
    // per token, so the workspace the plan declared covers either. The grouped planes are sized for
    // the padded width because that is the width the mainloop addresses.
    const bool mmq_route = mmq::select(weight.k, tokens);
    DeviceSpan codes;
    DeviceSpan scales;
    if (mmq_route) {
        codes = workspace.alloc_bytes(mmq::activation_bytes(tokens, weight.k));
    } else {
        codes  = workspace.alloc_bytes(static_cast<std::size_t>(padded) * weight.k);
        scales = workspace.alloc_bytes(static_cast<std::size_t>(padded) * (weight.k / 32) *
                                       sizeof(float));
    }
    if (mmq_route) {
        mmq::quantize_activations(x.data, tokens, weight.k, codes.data, stream);
    } else {
        quantize_nvfp4_int8_grouped<<<padded, 128, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data), tokens, weight.k,
            reinterpret_cast<std::int8_t*>(codes.data), reinterpret_cast<float*>(scales.data));
    }

    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), weight.n};
    if (mmq_route) {
        mmq::launch<128, LinearBf16Output>(
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales),
            reinterpret_cast<const std::int8_t*>(codes.data),
            1.0F / weight.weight_scale_divisor, output, weight.k, tokens, weight.n, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const dim3 grid(weight.n / 128, padded / 128);
    // alpha = 1/weight_scale_divisor: the E2M1 1/2 already folded into the E4M3 scale decode,
    // where llama.cpp folds it too -- the same for every route on this mainloop. The kernel is given
    // the real token count, not the padded one, and predicates its stores on it.
    nvfp4_int8_grouped_kernel<LinearBf16Output>
        <<<grid, detail::kInt8Threads, nvfp4_int8_grouped_smem_bytes(), stream>>>(
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales),
            reinterpret_cast<const std::int8_t*>(codes.data),
            reinterpret_cast<const float*>(scales.data), 1.0F / weight.weight_scale_divisor,
            output, weight.k, tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::unified
