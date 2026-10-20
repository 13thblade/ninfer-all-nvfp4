#pragma once

#include "models/expert_profile.h"
#include "models/qwen3_5/model.h"

#include <filesystem>

namespace ninfer::models::qwen3_5::execution {

// The routing profile that fills the expert caches once warm-up is done, and the file that
// receives the counts the engine's decode and verification calls record when it stops.
struct NativeExpertProfile {
    std::optional<ExpertRouteCounts> counts;
    std::filesystem::path record;
    artifact::ArtifactId artifact{};
};

// Program-owned service. Immutable sources are borrowed from Model. No service or slot is shared
// between Programs, even when their Parameters refer to the same Model.
class NativeExpertRuntime {
public:
    NativeExpertRuntime(const Model& model, DeviceContext& device,
                         std::span<const std::uint64_t> keep_free_by_rank,
                         std::optional<std::uint64_t> cache_bytes = {},
                         NativeExpertProfile profile = {});
    ~NativeExpertRuntime();
    NativeExpertRuntime(const NativeExpertRuntime&) = delete;
    NativeExpertRuntime& operator=(const NativeExpertRuntime&) = delete;
    // The routed experts of each of the model's expert layers.
    [[nodiscard]] static std::vector<std::size_t> widths(const Model& model);
    // Once warm-up is done: empties every rank's cache, forgets the warm-up's routes and fills
    // the cache from the profile.
    void prime();
    // Every route the decode and verification calls took through each expert layer since prime().
    [[nodiscard]] ExpertRouteCounts totals() const;
    [[nodiscard]] ExpertDelivery& layer(std::size_t index) const;
    [[nodiscard]] std::size_t local_layer(std::size_t index) const;
    [[nodiscard]] ExpertDeliveryMemory memory(std::size_t rank) const noexcept;
    [[nodiscard]] ExpertDeliveryStats stats() const;
    void check() const;
    // Waits for every rank's delivery work, prefetches included.
    void drain();
    void release_device_backing();
    void restore_device_backing();
    [[nodiscard]] std::uint64_t device_backing_bytes() const noexcept;

private:
    struct Handle { std::size_t rank, local; };
    std::vector<Handle> handles_;
    std::vector<std::unique_ptr<ExpertDelivery>> ranks_;
    NativeExpertProfile profile_;
};

} // namespace ninfer::models::qwen3_5::execution
