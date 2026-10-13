#pragma once

#include "ops/common/memory.cuh"
#include "ops/linear/q8/q8_geometry.h"

#include <cstdint>

namespace ninfer::ops::detail {

enum class Q8KSplitScaleAccess : std::uint8_t {
    Direct,
    Shared,
};

enum class Q8KSplitActivationStage : std::uint8_t {
    // Stage the compile-time column extent; tiled calls zero-fill its inactive columns.
    ActiveOnly,
    // Stage the full MMA tile, including padding beyond the compile-time column extent.
    PaddedZero,
    // Stage only live columns. Inactive MMA columns are discarded by the store epilogue.
    RuntimeActive,
};

template <int KWarps, int TileTokens, int MinBlocksPerSm, Q8KSplitScaleAccess ScaleAccess,
          Cache ActivationCache = Cache::ca, Cache WeightCache = Cache::cg,
          Q8KSplitActivationStage ActivationStage = Q8KSplitActivationStage::ActiveOnly>
struct Q8KSplitSchedule {
    static_assert(KWarps == 4 || KWarps == 8 || KWarps == 16);
    static_assert(TileTokens == 8 || TileTokens == 16 || TileTokens == 24 || TileTokens == 32 ||
                  TileTokens == 40 || TileTokens == 48 || TileTokens == 56 || TileTokens == 64 ||
                  TileTokens == 72 || TileTokens == 80 || TileTokens == 88);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kKWarps            = KWarps;
    static constexpr int kTileTokens        = TileTokens;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr auto kScaleAccess      = ScaleAccess;
    static constexpr auto kActivationCache  = ActivationCache;
    static constexpr auto kWeightCache      = WeightCache;
    static constexpr auto kActivationStage  = ActivationStage;
    static constexpr int kThreads           = KWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = KWarps * kTileKPerWarp;
    static constexpr int kRowsPerCta        = 16;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / KWarps;
    static constexpr int kScaleBytesPerRow  = kGroupK / 16;
};

template <int TileTokens, int ActiveTokens>
using Q8KSplitDefaultSchedule = Q8KSplitSchedule<
#if defined(NINFER_SM8X_COMPAT)
    4, TileTokens, 2,
#else
    8, TileTokens, TileTokens == 8 ? 5 : (TileTokens == 16 ? 4 : (TileTokens == 24 ? 3 : 2)),
#endif
    (ActiveTokens > 4 ? Q8KSplitScaleAccess::Shared : Q8KSplitScaleAccess::Direct)>;

// sm_86/sm_89 cap a kernel's static shared memory at 49,152 bytes. Q8KSplitSharedStorage stages
// KWarps * TileTokens * 128 bytes of activations beside 1,024 * KWarps bytes of codes, so sixteen
// K warps overflow it from a 16-column tile and eight K warps from a 40-column tile; those are
// compile errors there, not slow paths. Shapes whose SM120 tables use such tiles declare them
// through this alias, which keeps the SM120 schedule and on sm_8x falls back to the four-K-warp,
// two-block schedule the W8 routes used on this card.
template <int KWarps, int TileTokens, int MinBlocksPerSm, Q8KSplitScaleAccess ScaleAccess,
          Cache ActivationCache = Cache::ca, Cache WeightCache = Cache::cg,
          Q8KSplitActivationStage ActivationStage = Q8KSplitActivationStage::ActiveOnly>
using Q8KSplitSm8xFourWarpSchedule =
#if defined(NINFER_SM8X_COMPAT)
    Q8KSplitSchedule<4, TileTokens, 2, ScaleAccess, ActivationCache, WeightCache, ActivationStage>;
#else
    Q8KSplitSchedule<KWarps, TileTokens, MinBlocksPerSm, ScaleAccess, ActivationCache, WeightCache,
                     ActivationStage>;
#endif

// The schedule a route's K-split launch runs on this build. On Turing (sm_75) an 8-column tile of
// a matrix with at least 6144 rows streams its codes at 420-600 GB/s with four K warps on the
// TITAN RTX and at 230-330 GB/s with eight or sixteen, so a `75` build runs those tiles with four.
// Wider tiles keep the route's schedule (four K warps need 190 registers for a 16-column tile and
// run it at half the speed of eight), and so do smaller matrices: at 5120 rows four K warps are
// 15-25 % slower than eight. A four-K-warp 8-column tile is bounded to 128 registers, four CTAs
// an SM: the register prefetch of q8_ksplit_mma would otherwise take the fused SwiGLU tile to 160
// and three CTAs, 2-5 % slower with FP16 activations and 11-13 % when some tiles fall back to the
// BF16 emulation (docs/performance/titan-rtx.md). Other builds run the route's schedule as written.
#if defined(NINFER_SM75)
template <class Geometry, class Schedule>
struct Q8KSplitTuringSchedule {
    static constexpr bool kNarrow = Schedule::kTileTokens == 8;
    static constexpr int kKWarps =
        kNarrow && Schedule::kKWarps > 4 && Geometry::kOutputRows >= 6144 ? 4 : Schedule::kKWarps;
    static constexpr int kMinBlocksPerSm =
        kNarrow && kKWarps == 4 && Schedule::kMinBlocksPerSm < 4 ? 4 : Schedule::kMinBlocksPerSm;
    using type = Q8KSplitSchedule<kKWarps, Schedule::kTileTokens, kMinBlocksPerSm,
                                  Schedule::kScaleAccess, Schedule::kActivationCache,
                                  Schedule::kWeightCache, Schedule::kActivationStage>;
};

template <class Geometry, class Schedule>
using Q8KSplitDeviceSchedule = typename Q8KSplitTuringSchedule<Geometry, Schedule>::type;
#else
template <class Geometry, class Schedule>
using Q8KSplitDeviceSchedule = Schedule;
#endif

} // namespace ninfer::ops::detail
