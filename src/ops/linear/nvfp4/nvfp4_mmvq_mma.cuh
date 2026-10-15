#pragma once

// The MMVQ-shaped NVFP4 mainloop (see nvfp4_mmvq_host.h for what is ported, from where, and which
// three things had to change for ninfer's planes and epilogue contract).
//
// Arithmetic destination is deliberately the one the other two int8 routes already hit -- E2M1 codes
// widened to the exact integer grid s8 can hold (v = 2*e2m1), one NVFP4 G16 group rescaling the
// partial sum in FP32 through the E4M3 LUT, one per-(token, 32 columns) FP32 scale per group, FP32
// accumulation, a single BF16 rounding at the store -- so a value produced here and one produced by
// nvfp4_int8_grouped_kernel agree bit for bit, not merely to within a tolerance. Three things make
// that true rather than merely plausible:
//
//   * The 16-column group is the same unit both accumulation paths sum over. One `mma.m16n8k16.s8`
//     covers sixteen k at a time, which is exactly one NVFP4 group, and so does one trip here --
//     four `dp4a` folding sixteen products into one int32. The integers therefore add in the same
//     width, over the same terms, in the same order.
//   * The rescale is one product per (group, column): the weight scale is E4M3 already halved by the
//     LUT (nvfp4_e4m3_scale.h), the activation scale is the single 32-block this group sits inside,
//     and they are applied with one fmaf against the int32 sum -- the same expression the grouped
//     mainloop's `w0 * xa0` feeds to its own fmaf.
//   * Nothing between here and the store rounds: the accumulator is FP32, the cross-warp reduction
//     is FP32, and the adapter owns the one conversion.

#include "ops/linear/common/output.cuh"
#include "ops/linear/nvfp4/nvfp4_codec_unified.cuh"
#include "ops/linear/nvfp4/nvfp4_int8_grouped_mma.cuh" // e2m1_word, the E4M3 LUT
#include "ops/linear/nvfp4/nvfp4_mmvq_host.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail::mmvq {

namespace g = ninfer::ops::detail::unified;

// One CUDA block owns one output row outright. `NC` columns are retired together, so the row's weight
// bytes are fetched once and reused across every column -- the whole reason this shape beats a token
// tile at the widths that reach it, where a tile would pay for columns the caller never asked for.
//
// `tokens` is the caller's real width and the only columns stored; NC >= tokens pads the instruction
// stream, not any plane, so the store is the only thing that needs a guard. A width above kMaxTokens
// never gets here (see nvfp4_mmvq_launch.cuh), which is llama.cpp's crossover rather than a
// limitation of this kernel.
template <int NC, class Output>
__global__ __launch_bounds__(kThreads) void nvfp4_mmvq_kernel(
    const std::uint8_t* __restrict__ weight_codes, const std::uint8_t* __restrict__ weight_scales,
    const std::int8_t* __restrict__ x_codes, const float* __restrict__ x_scales, float alpha_half,
    Output output, std::int32_t k, std::int32_t tokens) {
    static_assert(NC >= 1 && NC <= kMaxTokens, "NC is one of the instantiated width stages");

    const std::int32_t row  = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t tid  = static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t lane = tid & (kWarpSize - 1);
    const std::int32_t warp = tid / kWarpSize;

    const std::int32_t groups   = k / kGroupCodes; // one E4M3 scale each
    const std::int32_t x_groups = k / 32;          // activation scale stride

    const std::uint8_t* w_row =
        weight_codes + static_cast<std::size_t>(row) * static_cast<std::size_t>(k / 2);

    float acc[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) { acc[j] = 0.0F; }

    for (std::int32_t g = tid; g < groups; g += kThreads) {
        // An eight-byte read is exactly what a sixteen-column NVFP4 group is. K % 64 == 0 is already
        // this route's alignment gate, so every group of every row starts on an eight-byte boundary
        // and the device-side weight planes are read for no other reason.
        const uint2 packed = *reinterpret_cast<const uint2*>(w_row + static_cast<std::size_t>(g) * kGroupBytes);

        unsigned q[4];
        // e2m1_word emits consecutive codes, so these four words are the group's sixteen operands in
        // k order: q[c] holds k = g*16 + 4c .. 4c+3.
        g::e2m1_word(packed.x, q[0], q[1]);
        g::e2m1_word(packed.y, q[2], q[3]);

        // One E4M3 byte per group through the shared LUT -- the trailing /2 and NaN -> 0 are llama's
        // ue4m3 semantics and live in the table, exactly as they do for the other two routes.
        const unsigned raw =
            weight_scales[static_cast<std::size_t>(g::nvfp4_scale_byte_offset(row, g, k))];
        const float w_scale = g::kNvfp4E4M3ScaleLut[raw];

        const std::int32_t kbase = g * kGroupCodes;
        // Which 32-column activation scale this group sits in: a group never straddles two, because
        // 32 is a whole number of 16s.
        const int x_block = kbase >> 5;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const uint4 xw = *reinterpret_cast<const uint4*>(
                x_codes + static_cast<std::size_t>(j) * k + kbase);
            int sum = 0;
            sum = __dp4a(static_cast<int>(q[0]), static_cast<int>(xw.x), sum);
            sum = __dp4a(static_cast<int>(q[1]), static_cast<int>(xw.y), sum);
            sum = __dp4a(static_cast<int>(q[2]), static_cast<int>(xw.z), sum);
            sum = __dp4a(static_cast<int>(q[3]), static_cast<int>(xw.w), sum);
            const float x_scale = x_scales[static_cast<std::size_t>(j) * x_groups + x_block];
            // One product per (group, column): the int32 sum is exact, both scales are exact, and
            // this fmaf is the same operation the grouped mainloop spends on the same sixteen terms.
            acc[j] = fmaf(w_scale * x_scale, static_cast<float>(sum), acc[j]);
        }
    }

    // llama's reduction, minus what it needed and we do not: every warp reduces by shuffle and parks
    // its column partials in shared memory for warp 0 to absorb. There is no rows_per_cuda_block
    // here -- one block is one row -- so there is no second axis to share along.
    __shared__ float reduced[kWarps][NC > 1 ? NC : 1];
#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            acc[j] += __shfl_xor_sync(0xFFFFFFFFu, acc[j], off);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < NC; ++j) { reduced[warp][j] = acc[j]; }
    }
    __syncthreads();

    if (tid >= NC) { return; }
    float total = reduced[0][tid];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) { total += reduced[w][tid]; }
    if (tid < tokens) {
        const auto epilogue = linear_output_tile<1>(output, row);
        epilogue.store(row, tid, total * alpha_half);
    }
}

} // namespace ninfer::ops::detail::mmvq
