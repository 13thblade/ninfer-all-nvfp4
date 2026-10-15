#pragma once

// Launch front end for the MMQ-structured NVFP4 mainloop (nvfp4_mmq_mma.cuh). One header so every
// projection route -- Linear, LinearAdd, both halves of gate_up, attn/gdn input projections --
// switches with the same three lines and the same predicate.
//
// The route is named by NINFER_NVFP4_GEMM, which now selects between whole paths rather than
// switching a nested flag inside the int8 branch -- see nvfp4_int8_route.h for the four values and
// why an unrecognised one is an error instead of a fallback. Note the change of meaning for the
// value `mmq`: it used to mean "prefer MMQ where the int8 branch was already entered", and it now
// names the path outright, at every token count, padding the last J-tile. The historical nesting is
// what `default` does. `select` is the entry point a route calls; it refuses on K rather than
// falling back, because a weight plane cannot be padded the way an activation tile can.
//
// Both routes quantise the same activations and the two layouts cost the same bytes per token
// (K + K/8), so a switch costs no extra scratch; only the quantiser that fills the plane differs.

#include "ops/linear/nvfp4/nvfp4_int8_route.h"
#include "ops/linear/nvfp4/nvfp4_mmq_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_mmvq_host.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::mmq {

// Whether this call runs the MMQ mainloop. A named route serves every token count **above the
// crossover** by padding its last J-tile, so only K can refuse it -- and K has nothing to pad
// against, so a named route that fails on K is an error rather than a quiet fallback to the grouped
// kernel. The A16 route never reaches either int8 mainloop.
//
// `mmvq` reaches MMQ too, but only above the crossover: below it the MMVQ-shaped kernel takes the
// call on the two-plane activation form instead, so those widths answer false here -- and that holds
// for every int8 route now, not just the one named `mmvq`. The refusal goes through `mmvq::applies`
// rather than a copy of its test, because a width claimed by both would make the site allocate the
// interleaved plane for a kernel that reads two.
inline bool select(std::int32_t k, std::int32_t tokens) {
    const Nvfp4GemmRoute route = nvfp4_gemm_route();
    if (route != Nvfp4GemmRoute::Mmq && route != Nvfp4GemmRoute::Mmvq) { return false; }
    if (mmvq::applies(tokens)) { return false; }
    if (!k_admitted(k)) {
        throw std::invalid_argument(
            "nvfp4 mmq route is named, but K=" + std::to_string(k) +
            " is not a whole ITER_K=256 partition; the grouped mainloop has no such requirement and "
            "a weight plane cannot be padded the way an activation tile can");
    }
    return true;
}

// Quantise the activations into this route's plane: per token, per 128-column window, four FP32
// scales then 128 s8 codes (see quantize_nvfp4_int8_mmq). Callers that fail applies() must keep the
// two-plane form and the grouped mainloop.
inline void quantize_activations(const void* x, std::int32_t tokens, std::int32_t k, void* tile,
                                 cudaStream_t stream) {
    // One block per token of the PADDED width: the surplus rows are zeroed, not skipped, because
    // the mainloop reads every row of the tile it is given.
    quantize_nvfp4_int8_mmq<<<unified::nvfp4_int8_padded_tokens(tokens), 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x), tokens, unified::nvfp4_int8_padded_tokens(tokens), k,
        static_cast<std::int8_t*>(tile));
}

// One block per (128 output rows, J tokens). J is no longer 128 by construction: it is the stage of
// the token ladder the width selects, and both the grid and the plane's stride follow it -- which is
// the difference between "small J buys less tensor-core work" and "small J re-reads the weights once
// per tile column". Only a simultaneous change to the pad basis buys the former; see
// nvfp4_int8_select_j.
template <int J, class Output>
inline void launch_j(const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                     const std::int8_t* x_tile, float alpha_half, Output output, std::int32_t k,
                     std::int32_t tokens, std::int32_t rows, cudaStream_t stream) {
    static const bool smem_opted_in = [] {
        cudaFuncSetAttribute(reinterpret_cast<const void*>(&nvfp4_mmq_kernel<J, Output>),
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_bytes(J)));
        return true;
    }();
    (void)smem_opted_in;
    // The grid and the plane layout run over the padded width; the kernel stores only the tokens
    // the caller asked for. Passing one number for both is what would make a narrow forced run
    // either write out of range or silently drop columns.
    const std::int32_t tile_tokens = unified::nvfp4_int8_padded_tokens(tokens);
    const dim3 grid(rows / kRows, tile_tokens / J);
    nvfp4_mmq_kernel<J, Output><<<grid, kThreads, smem_bytes(J), stream>>>(
        weight_codes, weight_scales, x_tile, alpha_half, output, k, tile_tokens, tokens);
}

// The entry point a route calls: same ladder, same widths, same result, whichever stage the width
// selects.
template <class Output>
inline void launch(const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                   const std::int8_t* x_tile, float alpha_half, Output output, std::int32_t k,
                   std::int32_t tokens, std::int32_t rows, cudaStream_t stream) {
#define NINFER_MMQ_LAUNCH_STAGE(STAGE)                                                    \
    case STAGE:                                                                           \
        launch_j<STAGE, Output>(weight_codes, weight_scales, x_tile, alpha_half, output, k, \
                                tokens, rows, stream);                                    \
        return;
    switch (unified::nvfp4_int8_select_j(tokens)) {
        NINFER_MMQ_LAUNCH_STAGE(8)
        NINFER_MMQ_LAUNCH_STAGE(16)
        NINFER_MMQ_LAUNCH_STAGE(24)
        NINFER_MMQ_LAUNCH_STAGE(32)
        NINFER_MMQ_LAUNCH_STAGE(40)
        NINFER_MMQ_LAUNCH_STAGE(48)
        NINFER_MMQ_LAUNCH_STAGE(64)
        NINFER_MMQ_LAUNCH_STAGE(80)
        NINFER_MMQ_LAUNCH_STAGE(96)
        NINFER_MMQ_LAUNCH_STAGE(112)
        NINFER_MMQ_LAUNCH_STAGE(128)
        default:
            break;
    }
#undef NINFER_MMQ_LAUNCH_STAGE
    throw std::invalid_argument("nvfp4 mmq: no admissible J tile for T=" + std::to_string(tokens));
}

} // namespace ninfer::ops::detail::mmq
