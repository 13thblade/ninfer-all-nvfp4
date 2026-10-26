#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              LinearPolicy policy,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

// Whether a table names [N,K]. Every other shape, and every FP32 result, runs the runtime-shape A16
// route (fp8_general.cu): whole tiles of `kFp8ShareUnits`, the policy's other routes unused.
[[nodiscard]] bool fp8_registered(std::int32_t output_rows, std::int32_t input_rows);
inline constexpr LinearShareUnits kFp8ShareUnits{64, 512};
void fp8_general_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
