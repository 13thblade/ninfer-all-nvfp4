#pragma once

#include "core/device.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer {

// A blob is copied without decoding or repacking. File extents concatenate in order.
struct ExpertExtent {
    std::size_t file = 0;
    std::uint64_t offset = 0, bytes = 0;
};
struct ExpertBlob {
    const void* pinned = nullptr;
    std::vector<ExpertExtent> extents;
    std::uint64_t bytes = 0;
    std::uint64_t zero_tail = 0;
};
struct ExpertAlias {
    std::size_t blob = 0;
    std::uint64_t offset = 0;
};
struct ExpertDeliveryLayer {
    // [expert][blob]. Aliases describe the projection-major device pointer table.
    std::vector<std::vector<ExpertBlob>> experts;
    std::vector<ExpertAlias> aliases;
};
struct ExpertDeliveryOptions {
    std::uint32_t pair_capacity = 0;
    // Maximum distinct experts per call. Zero selects the largest layer.
    std::uint32_t active_capacity = 0;
    // Persistent cache only. Transient slots cover active_capacity independently. Both explicit
    // and automatic cache must preserve keep_free_bytes after physical allocation rounding.
    std::optional<std::uint64_t> cache_bytes;
    std::uint64_t keep_free_bytes = 512ULL << 20;
    std::uint64_t staging_bytes = 32ULL << 20; // disk only, two pinned halves
    bool suspendable = false; // existing VmmRegion allocations, fixed virtual addresses
};
struct ExpertDeliveryStats {
    std::uint64_t requests = 0, routes = 0, hits = 0, misses = 0, evictions = 0;
    std::uint64_t pairs = 0, pair_hits = 0, admissions = 0;
    std::uint64_t read_bytes = 0, copied_bytes = 0;
    double service_seconds = 0;
};
struct ExpertDeliveryMemory {
    std::uint64_t cache_bytes = 0, transient_bytes = 0, metadata_bytes = 0, pinned_bytes = 0;
    std::uint32_t cache_slots = 0;
};

// One Program, rank and serial compute stream. Sources outlive this object. All device addresses
// stay fixed until destruction. The caller drains graph users before destroying their graphs.
class ExpertDelivery {
public:
    ExpertDelivery(const RankContext& rank, std::vector<std::filesystem::path> files,
                   std::vector<ExpertDeliveryLayer> layers, ExpertDeliveryOptions options);
    ~ExpertDelivery();
    ExpertDelivery(const ExpertDelivery&) = delete;
    ExpertDelivery& operator=(const ExpertDelivery&) = delete;

    struct Call {
        const std::int32_t* cached_ids = nullptr;
        const std::int32_t* missing_ids = nullptr;
    };
    // Capturable. The next begin must follow every consumer of the preceding call on stream.
    // Cached pairs can run immediately. Missing pairs run only after await(). Neither set drops
    // or renormalizes routes. -1 denotes a pair absent from that set.
    [[nodiscard]] Call begin(std::size_t layer, const std::int32_t* ids, std::uint32_t pairs,
                             cudaStream_t stream);
    void await(cudaStream_t stream);
    // Projection-major [aliases][experts], including aliases into the same blob.
    [[nodiscard]] const void* const* tables(std::size_t layer) const;
    void check() const;
    // Drain before suspend. Invalidate preserves allocations and graph addresses for restore.
    void drain();
    void invalidate();
    void release_device_backing(); // drain first, then release every transport device allocation
    void restore_device_backing(); // map at the same addresses and invalidate residency
    [[nodiscard]] std::uint64_t state_backing_bytes() const noexcept;
    [[nodiscard]] ExpertDeliveryStats stats() const;
    [[nodiscard]] ExpertDeliveryMemory memory() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
