#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t output_rows,
                                                                std::int32_t input_rows,
                                                                LinearPolicy policy,
                                                                std::int32_t min_tokens,
                                                                std::int32_t max_tokens);

// Whether a table names [N,K]. Every other shape, and every FP32 result, runs the runtime-shape A16
// route (nvfp4_general.cu): whole tiles of `kNvfp4ShareUnits`, the policy's other routes unused.
[[nodiscard]] bool nvfp4_registered(std::int32_t output_rows, std::int32_t input_rows);
inline constexpr LinearShareUnits kNvfp4ShareUnits{128, 256};
void nvfp4_general_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
