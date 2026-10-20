#pragma once

// A device cache of the routed experts of host-resident banks. Every expert of a layer is reached
// through device tables of base pointers (one per projection) that the expert kernels read; a
// cached expert's entries point at its copy in a device slot, every other entry at its bytes in the
// pinned host block. The kernels never see the difference, so no pass waits on the cache.
//
// Which experts the slots hold follows the shared expert tier (core/expert_tier.h), one pool per
// layer: decode and verification routes raise decayed counts, and every few decode passes a round
// swaps the most routed missing experts into the slots of the least routed cached ones, copied on
// a side stream while the next passes run; a candidate's table entries move to its slot once its
// copy has landed. The expert misses may also place a miss they copy anyway into a slot at once,
// by the same rule (admit). A recorded routing profile fills the slots at startup and, through the
// ranking of its counts over every layer, sizes each layer's share of them.

#include "core/arena.h"
#include "core/device.h"
#include "core/expert_tier.h"
#include "core/weight.h"

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace ninfer::models::qwen4_exp {

struct ExpertBank {
    std::int64_t expert_bytes = 0; // one expert's rows
    std::vector<const void*> host; // each expert's rows in the pinned host block
    void* table = nullptr;         // the device table the kernels read
};

struct ExpertCacheLayer {
    std::size_t rank    = 0;
    cudaStream_t stream = nullptr; // the rank's compute stream, which reads the tables
    ExpertBank gate, up, down;
};

struct ExpertCacheStats {
    std::uint64_t routes       = 0; // (token, expert) pairs observed
    std::uint64_t hits         = 0; // of them, served from a slot
    std::uint64_t admitted     = 0; // experts copied into slots
    std::uint64_t copied_bytes = 0;
    std::uint32_t slots        = 0;
    std::uint64_t cpu_routes   = 0; // native hybrid: routed (token, expert) pairs computed on CPU
    std::uint64_t dma_routes   = 0; // native hybrid: uncached pairs staged to GPU slots
};

class ExpertCache {
public:
    // `bytes_by_rank` is the device memory each rank lends its layers' slots. `profile` (optional,
    // [layer][expert] route counts of the same layers) ranks every (layer, expert) pair: each text
    // layer's slots are its pairs among the top of that ranking, the last layer (the MTP block's,
    // when `last_is_mtp`) keeps an even share.
    ExpertCache(DeviceContext& device, std::vector<ExpertCacheLayer> layers,
                std::span<const std::uint64_t> bytes_by_rank,
                const std::vector<std::vector<std::uint64_t>>* profile = nullptr,
                bool last_is_mtp = false, ExpertTierOptions tier = {});
    ~ExpertCache();
    ExpertCache(const ExpertCache&)            = delete;
    ExpertCache& operator=(const ExpertCache&) = delete;

    // The routes one pass took through `layer`: `ids` holds every (token, slot) pair's expert.
    // `learn`: a decode or verification pass, whose routes the tier counts (a prompt chunk's only
    // feed the statistics and the recorded totals).
    void observe(std::size_t layer, std::span<const std::int32_t> ids, bool learn);
    // Ends a decode pass: tables of candidates whose copies have landed move to their slots, and
    // every few passes a round of the tier evicts its victims and starts its candidates' copies.
    // Between passes only; the next pass on each layer's stream sees every table change.
    void adapt();
    // Projection k (0 gate, 1 up, 2 down) of `expert` in its device slot, or null when the expert
    // is not cached. Slots are zeroed when allocated and 256 zero bytes follow the last, as the
    // expert matrix kernel needs (see moe_experts_gguf).
    [[nodiscard]] const void* cached(std::size_t layer, int k, std::int32_t expert) const;
    [[nodiscard]] ExpertCacheStats stats() const noexcept;
    // Fills every layer's slots with its most routed experts of `counts` (a recorded profile of
    // the same layers), before the first pass.
    void seed(const std::vector<std::vector<std::uint64_t>>& counts);
    // Every route each layer's experts took since startup.
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> totals() const;
    // The device memory of `layer`'s slots, in up to two blocks (the second empty until grow):
    // a table entry inside either is a cached expert.
    using Ranges = std::array<std::pair<std::uintptr_t, std::uintptr_t>, 2>;
    [[nodiscard]] Ranges storage(std::size_t layer) const;
    // Once startup's allocations are done: more slots for every layer that is not full, from
    // `bytes_by_rank`, as free slots, shared as at construction. Returns the bytes each rank gave.
    // At most once.
    std::vector<std::uint64_t> grow(std::span<const std::uint64_t> bytes_by_rank);

    // Slot admission by the expert misses, whose service thread calls it while the layer's call
    // runs: from now on the misses patch `layer`'s device tables of the experts they admit.
    void hand_over(std::size_t layer);
    struct Admitted {
        std::array<void*, 3> slot{};              // the expert's projections in its slot
        std::int32_t victim = -1;                 // the expert that left the slot, or -1
        std::array<const void*, 3> victim_host{}; // the victim's projections in host memory
    };
    // Takes a slot of a handed-over `layer` for the uncached `expert`, which the call copies to the
    // GPU anyway and routes `pairs` times: a free one, else the tier's victim among the experts
    // `in_use` (a flag per expert) does not mark. False when none qualifies. The caller copies the
    // expert in and updates the device tables before the layer's next call; a route of the expert
    // counts as a miss until the next observe.
    bool admit(std::size_t layer, std::int32_t expert, int pairs,
               std::span<const unsigned char> in_use, Admitted& out);
    // The device tables (gate, up, down) of `layer`.
    [[nodiscard]] std::array<void*, 3> tables(std::size_t layer) const;

private:
    struct Layer;
    void upload(Layer& layer, cudaStream_t stream);
    DeviceContext& device_;
    std::vector<Layer> layers_;
    std::unique_ptr<PinnedHostBuffer> staging_;
    std::unique_ptr<ExpertTier> tier_;
    // One side stream per rank for a round's copies, and the event the rank's compute stream
    // records once a round's victims have left its tables.
    std::vector<cudaStream_t> side_;
    std::vector<cudaEvent_t> evicted_;
    struct Copy {
        std::size_t layer  = 0;
        std::int32_t expert = -1;
        std::int32_t slot   = -1;
        cudaEvent_t done    = nullptr;
    };
    std::vector<Copy> copies_;          // a round's copies in flight
    std::vector<cudaEvent_t> spare_;    // events of landed copies, reused
    ExpertCacheStats stats_;
    // Guards the bookkeeping, which the misses' service thread changes through admit.
    mutable std::mutex mutex_;
};

} // namespace ninfer::models::qwen4_exp
