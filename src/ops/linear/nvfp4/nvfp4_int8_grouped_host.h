#pragma once

// Host-side arithmetic for the NVFP4 Tier-2 int8-grouped route (see nvfp4_int8_grouped_mma.cuh
// for the device side and the performance model). Kept separate from the .cuh so plan and
// dispatch translation units -- which the MSVC host compiler builds -- can query admission and
// workspace sizes without pulling in device intrinsics.
//
// Width is no longer an admission question: nvfp4_int8_route_admits answers for the route named in
// the environment, and both int8 mainloops serve every positive width by padding their last token
// tile (nvfp4_int8_padded_tokens) instead of refusing it. The old whole-128-token-tile predicate is
// gone with the hybrid cascade that used it.

#include "ops/linear/nvfp4/nvfp4_int8_route.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::unified {

// Shared launch geometry. BM=BN=128, BK=64 (exactly four G16 weight groups per k-tile), 16 bytes
// of smem row padding to keep every lds128 word-aligned without a 128-multiple stride. The CTA is
// 256 threads / 8 warps (ggml-MMQ shaped): 256 * 255 regs stays under the SM's 64K register file
// with room to hoist the A operands for the whole k-tile, which the old 512-thread CTA (512 * 128
// = the whole file) could not.
namespace detail {

constexpr int kInt8TileRows    = 128; // BM
constexpr int kInt8TileTokens  = 128; // BN
constexpr int kInt8TileK       = 64;  // BK, exactly four G16 groups
constexpr int kInt8SRow        = kInt8TileK + 16;
constexpr int kInt8Threads     = 256;
constexpr int kInt8ScaleGroups = 4; // weight group scales per row per k-tile
constexpr int kInt8XScales     = 2; // per-32 activation scales per row per k-tile

} // namespace detail

constexpr int nvfp4_int8_grouped_smem_bytes() {
    return detail::kInt8TileRows * detail::kInt8SRow * 2 +                      // sa + sb
           detail::kInt8TileRows * detail::kInt8ScaleGroups * sizeof(float) +   // sws
           detail::kInt8TileRows * detail::kInt8XScales * sizeof(float);        // sxs
}

// The token count a padded launch actually runs. The mainloop has no predicated tail -- its grid is
// one CTA per whole 128-token tile and every one of its 128 activation rows is read -- so a width
// that is not a whole tile is served by running the tile full and discarding the surplus: the
// quantiser zeroes the padding rows' planes instead of reading activations that do not exist, and
// the epilogue stores only the tokens the caller asked for.
constexpr std::int32_t nvfp4_int8_padded_tokens(std::int32_t tokens) {
    return tokens <= 0
               ? 0
               : ((tokens + detail::kInt8TileTokens - 1) / detail::kInt8TileTokens) *
                     detail::kInt8TileTokens;
}

// Width admission of the named route. The int8 mainloops serve every positive width -- they pad
// rather than refuse -- and A16 runs neither of them, which is what an unset environment selects.
// There is no longer a width-derived answer to this question; that was the hybrid cascade.
inline bool nvfp4_int8_route_admits(std::int32_t tokens) {
    return tokens >= 1 && nvfp4_gemm_route() != Nvfp4GemmRoute::A16;
}

// The same question for a [min,max] planning interval: can any width in it reach the int8 kernels?
inline bool nvfp4_int8_route_interval_admits(std::int32_t min_tokens, std::int32_t max_tokens) {
    return min_tokens >= 1 && max_tokens >= min_tokens &&
           nvfp4_gemm_route() != Nvfp4GemmRoute::A16;
}

// s8 codes [T][K] plus the FP32 per-(token,32) scale plane [T][K/32], each 256-aligned by the
// arena. Sized for the PADDED token count, because that is what the launch addresses: a caller that
// passes an exact whole-tile width is unaffected, and one that passes a ragged width no longer
// under-reserves by up to 127 tokens' worth of plane.
//
// The MMQ mainloop reads the same quantisation interleaved instead -- per token, per 128-column
// window, four FP32 scales then 128 s8 codes, 144 bytes -- which is the same K + K/8 bytes per
// token as one contiguous span rather than two 256-aligned ones, so it never exceeds this envelope.
// mmq::activation_bytes() is what that route allocates, and it is checked against this function
// for every T in nvfp4_int8's workspace-envelope test.
constexpr std::size_t nvfp4_int8_grouped_workspace_bytes(std::int32_t tokens, std::int32_t k) {
    const std::size_t t = static_cast<std::size_t>(nvfp4_int8_padded_tokens(tokens));
    const std::size_t groups = static_cast<std::size_t>(k) / 32;
    return ((t * static_cast<std::size_t>(k) + 255) / 256) * 256 +
           ((t * groups * sizeof(float) + 255) / 256) * 256;
}

} // namespace ninfer::ops::detail::unified
