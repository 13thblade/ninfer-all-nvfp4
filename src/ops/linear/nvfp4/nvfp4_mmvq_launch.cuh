#pragma once

// Launch front end for the MMVQ-shaped NVFP4 mainloop (nvfp4_mmvq_mma.cuh). Width, and only width,
// decides whether a call comes here -- llama.cpp's own crossover: `ggml_cuda_should_use_mmvq` answers
// for ncols_dst <= 8 and everything wider is MMQ's business. The named route `mmvq` therefore spans
// both shapes, one launch helper deciding which half answers, which is also how its CMS maintainer
// would read a benchmark of it: not a third mainloop, but the narrow half of the width ladder given
// its own name so the wide half and the narrow half can be measured apart.
//
// The activation planes this route reads are the grouped route's two -- s8 codes [T][K] plus one FP32
// scale per (token, 32 columns) -- so unlike mmq it needs no separate quantisation pass and no
// separate workspace accounting: whatever the grouped mainloop would have been handed is what this
// kernel takes.

#include "ops/linear/nvfp4/nvfp4_int8_route.h"
#include "ops/linear/nvfp4/nvfp4_mmvq_host.h"
#include "ops/linear/nvfp4/nvfp4_mmvq_mma.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::mmvq {

/// Whether this call answered by the MMVQ-shaped kernel. Route name and width, nothing else: the k
/// constraints are this route's existing 64-column tile and are enforced upstream by the same
/// alignment gate every NVFP4 route goes through.
inline bool select(std::int32_t tokens) { return applies(tokens); }

template <int NC, class Output>
inline void launch_nc(const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                      const std::int8_t* x_codes, const float* x_scales, float alpha_half,
                      Output output, std::int32_t k, std::int32_t tokens, std::int32_t rows,
                      cudaStream_t stream) {
    nvfp4_mmvq_kernel<NC, Output><<<rows, kThreads, 0, stream>>>(
        weight_codes, weight_scales, x_codes, x_scales, alpha_half, output, k, tokens);
}

/// One block per output row, NC the next power of two at or above the caller's width. The stage set
/// is deliberately only the powers of two: NC indexes the accumulator array, so every additional
/// stage is another full instantiation of the mainloop per projection route.
template <class Output>
inline void launch(const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                   const std::int8_t* x_codes, const float* x_scales, float alpha_half,
                   Output output, std::int32_t k, std::int32_t tokens, std::int32_t rows,
                   cudaStream_t stream) {
    switch (padded_columns(tokens)) {
        case 1:
            launch_nc<1, Output>(weight_codes, weight_scales, x_codes, x_scales, alpha_half, output,
                                 k, tokens, rows, stream);
            return;
        case 2:
            launch_nc<2, Output>(weight_codes, weight_scales, x_codes, x_scales, alpha_half, output,
                                 k, tokens, rows, stream);
            return;
        case 4:
            launch_nc<4, Output>(weight_codes, weight_scales, x_codes, x_scales, alpha_half, output,
                                 k, tokens, rows, stream);
            return;
        case 8:
            launch_nc<8, Output>(weight_codes, weight_scales, x_codes, x_scales, alpha_half, output,
                                 k, tokens, rows, stream);
            return;
        default:
            break;
    }
    throw std::invalid_argument("nvfp4 mmvq: no column stage for T=" + std::to_string(tokens));
}

} // namespace ninfer::ops::detail::mmvq
