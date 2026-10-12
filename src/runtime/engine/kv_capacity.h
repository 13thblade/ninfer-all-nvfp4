#pragma once

#include "runtime/contract/resources.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>

namespace ninfer::runtime {

// Device bytes an explicit capacity of `tokens` reserves on the primary device under `curve`.
[[nodiscard]] std::size_t explicit_reservation_bytes(std::uint32_t tokens,
                                                     const SequenceCapacityCurve& curve);

// The first device, in rank order (0 is the primary), whose available bytes an explicit capacity of
// `tokens` exceeds, with what it needs there; nullopt when every device holds it. A capacity
// outside the curve's page range is held to the nearest end of the range, so this never throws for
// it; explicit_capacity_fits still reports such a capacity as not fitting.
struct CapacityShortfall {
    std::size_t rank            = 0;
    std::size_t need_bytes      = 0;
    std::size_t available_bytes = 0;
};
[[nodiscard]] std::optional<CapacityShortfall>
explicit_capacity_shortfall(std::uint32_t tokens, const SequenceCapacityCurve& curve,
                            std::size_t available_runtime_bytes,
                            std::span<const std::size_t> extra_rank_available_bytes = {});

// Whether an explicit capacity of `tokens` fits `available_runtime_bytes` on the primary device and
// each further device's entry of `extra_rank_available_bytes`. Answers instead of throwing, for
// callers that search for a capacity that fits.
[[nodiscard]] bool explicit_capacity_fits(std::uint32_t tokens, const SequenceCapacityCurve& curve,
                                          std::size_t available_runtime_bytes,
                                          std::span<const std::size_t> extra_rank_available_bytes = {});

// The largest context in [floor, ceiling] that `fits`: the ceiling itself, else the largest
// multiple of `step` below it, else the floor; nullopt when not even the floor fits. `fits` must be
// monotone -- every context below one that fits also fits -- and is called O(log) times.
[[nodiscard]] std::optional<std::uint32_t>
fit_context(std::uint32_t floor, std::uint32_t ceiling, std::uint32_t step,
            const std::function<bool(std::uint32_t)>& fits);

// `available_runtime_bytes` is the primary device's free memory after weights;
// `extra_rank_available_bytes` is the same for each further device, in the order of
// `curve.extra_ranks`. The capacity is the largest that fits every device, and a failure names the
// device that could not hold it.
[[nodiscard]] KvCapacityResolution
resolve_kv_capacity(const KvCapacityPolicy& policy, const SequenceCapacityCurve& curve,
                    std::size_t available_runtime_bytes,
                    std::span<const std::size_t> extra_rank_available_bytes = {});

} // namespace ninfer::runtime
