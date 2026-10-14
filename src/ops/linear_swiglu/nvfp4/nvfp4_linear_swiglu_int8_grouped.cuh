#pragma once

// Integer-activation route for the NVFP4 27B mlp/gate_up LinearSwiGLU on sm_86.
//
// Reads the artifact's native NVFP4 weight with no repack: the E2M1 code plane is row-major
// [n][k/2] and the E4M3 scale plane is the BlockScaleK16M128x4 tiling, exactly what the A16 routes
// consume. The E2M1 codes widen losslessly to s8 (v = 2*e2m1) and feed mma.m16n8k16.s8 with the
// per-16 E4M3 group scales rescaling in FP32; activations are quantised to s8 with one FP32 scale
// per (token, 32 columns) -- the same outlier confinement the q4a8 route chose, and the reason a
// per-token scalar is not used (a massive-activation channel would starve its whole token).
//
// gate_up is two ordinary GEMMs over the one stored [gate; up] plane, not a fused paired-row
// kernel: the gate projection writes an FP32 plane and the up projection's store reads it back
// and applies silu(gate) * up in FP32 with a single BF16 rounding -- the same assembly ggml's
// GateProduct carries (ggml_bridge_internal.cuh), and what removes the paired-row / N=2*intermediate
// constraint the fused kernel carried.
//
// This adds activation quantisation error the A16 route does not have: ~1.2-1.4% relative L2 on
// realistic distributions (measured against real artifact weight rows), held to the A8Int
// allowance. It is opt-in through LinearPolicy::AllowNvfp4Int8 for that reason; which widths then
// take it is named by NINFER_NVFP4_GEMM, since the mainloop pads its last 128-token tile rather
// than refusing a width it does not divide exactly.

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::unified {

// True when this route can serve the call: the registered 27B gate_up profile as a single-source
// NVFP4 plane, and a token count the 128-wide tiling covers exactly.
[[nodiscard]] bool nvfp4_linear_swiglu_int8_grouped_supported(const Weight& gate_up,
                                                              std::int32_t tokens);

// Token half of the admission rule, for the workspace query, which sees the profile but not the
// Weight. A single-T capacity query must report the route that will actually run.
[[nodiscard]] bool nvfp4_linear_swiglu_int8_tokens_supported(std::int32_t tokens);

// Transient bytes this route needs for the quantised activation planes over [min,max] tokens.
// Exact when min==max, an upper bound over the admitted widths otherwise.
[[nodiscard]] std::size_t
nvfp4_linear_swiglu_int8_workspace_capacity_bytes(std::int32_t min_tokens, std::int32_t max_tokens);

// x is contiguous BF16 [k, tokens]; out is contiguous BF16 [n/2, tokens]. Requires the exact
// workspace capacity above to be reserved in `workspace`.
void nvfp4_linear_swiglu_int8_grouped_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                             WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail::unified
