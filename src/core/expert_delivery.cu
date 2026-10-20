#include "core/expert_delivery.h"

#include "core/arena.h"
#include "core/cpu_spin_pool.h"
#include "core/cuda_host_handshake.h"
#include "core/expert_split.h"
#include "core/file_read_queue.h"
#include "core/growable_region.h"
#include "core/vmm.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>

namespace ninfer {
namespace {

// A round's copies a call's post work issues at most: experts, and bytes read from files.
constexpr std::size_t kRoundExperts = 8;
constexpr std::uint64_t kRoundFileBytes = 8ULL << 20;

// needed[e] of a call: its pairs routed to expert e, and whether the table entry the split read
// lies in the persistent cache.
constexpr std::uint32_t kCountMask = (1u << 30) - 1;
// Request flag: begin() received the family's CPU inputs, so the call may leave misses to the CPU.
constexpr std::uint32_t kCpuCall = 1;
constexpr std::uint32_t kHit       = 1u << 30;
constexpr std::uint32_t kMiss      = 1u << 31;

std::uint64_t aligned(std::uint64_t n) {
    if (n > std::numeric_limits<std::uint64_t>::max() - 255) {
        throw std::invalid_argument("expert delivery: byte extent overflow");
    }
    return (n + 255) / 256 * 256;
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::invalid_argument("expert delivery: byte count overflow");
    }
    return a + b;
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::invalid_argument("expert delivery: byte count overflow");
    }
    return a * b;
}

// Use the core's existing fixed-address allocation when the Program supports model suspend. An
// elastic cache is a growable region: addresses for every expert, `bytes` of them in slots.
struct Storage {
    DeviceBuffer owned;
    VmmRegion region;
    GrowableRegion growable;
    void* p = nullptr;
    std::size_t bytes = 0;
    Storage() = default;
    Storage(std::size_t n, int device, bool suspendable) : bytes(n) {
        if (suspendable) {
            region = VmmRegion(n, {.device = device});
            p = region.span().data;
        } else {
            owned = DeviceBuffer(n);
            p = owned.p;
        }
    }
    void fill() {
        CUDA_CHECK(cudaMemset(p, 0, bytes));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
    [[nodiscard]] std::uint64_t backing_bytes() const noexcept {
        return growable ? growable.mapped_bytes() : region ? region.reserved_bytes() : bytes;
    }
    void release_backing() {
        if (growable) { growable.release_backing(); }
        else { region.release_backing(); }
    }
    void restore_backing() {
        if (growable) { growable.restore_backing(); }
        else { region.restore_backing(); }
    }
};

// One block: a shared histogram of the call's experts, then each pair's cached/missing id by the
// table entry the expert has now. The mask is copied before publish(), on the same stream. Tables
// change only at missing entries or at experts absent from this call, so cached consumers can
// read them while the host prepares the rest. An entry in a lent bank (`lent`: [bank][low, high),
// empty when not lent) hits in a narrow call only: a wide call, which routes half the layer's
// experts or more, may take that bank back while its cached pairs run.
__global__ void split_routes(const std::int32_t* ids, std::uint32_t pairs, int experts,
                             const void* const* tables, std::uintptr_t low, std::uintptr_t high,
                             const std::uintptr_t* lent, std::int32_t* cached,
                             std::int32_t* missing, std::uint32_t* needed) {
    extern __shared__ std::uint32_t counts[];
    __shared__ std::uint32_t routes;
    if (threadIdx.x == 0) { routes = 0; }
    for (int e = threadIdx.x; e < experts; e += blockDim.x) { counts[e] = 0; }
    __syncthreads();
    for (std::uint32_t p = threadIdx.x; p < pairs; p += blockDim.x) {
        const int e = ids[p];
        if (e >= 0 && e < experts) { atomicAdd(counts + e, 1u); }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        if (counts[e]) { atomicAdd(&routes, 1u); }
    }
    __syncthreads();
    const bool wide = 2 * routes >= std::uint32_t(experts);
    const std::uintptr_t lent_low[2]{lent[0], lent[2]}, lent_high[2]{lent[1], lent[3]};
    const auto hit = [&](const void* entry) {
        const auto address = reinterpret_cast<std::uintptr_t>(entry);
        if (address >= low && address < high) { return true; }
        return !wide && ((address >= lent_low[0] && address < lent_high[0]) ||
                         (address >= lent_low[1] && address < lent_high[1]));
    };
    for (std::uint32_t p = threadIdx.x; p < pairs; p += blockDim.x) {
        const int e       = ids[p];
        const bool valid  = e >= 0 && e < experts;
        const bool cached_pair = valid && hit(tables[e]);
        cached[p]  = cached_pair ? e : -1;
        missing[p] = valid && !cached_pair ? e : -1;
    }
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        const std::uint32_t count = counts[e];
        needed[e] = count == 0 ? 0u : (count & kCountMask) | (hit(tables[e]) ? kHit : kMiss);
    }
}

} // namespace

struct ExpertDelivery::Impl {
    struct Layer {
        ExpertDeliveryLayer source;
        Storage tables;
        std::unique_ptr<PinnedHostBuffer> mirror;
        // Whole-table uploads alternate between two pinned copies, so the mirror can change while
        // an earlier upload still reads its copy.
        std::unique_ptr<PinnedHostBuffer> upload[2];
        cudaEvent_t uploaded[2]{};
        int next_upload = 0;
        bool dirty      = false;
        std::vector<std::int32_t> slot_of;
        std::uint32_t cached = 0; // slots holding this layer's experts
        // Bank stride of each blob: the blob's bytes when every expert's are equal (the resident
        // layout, one copy per run of experts), else its aligned maximum.
        std::vector<std::uint64_t> stride;
        std::int64_t successor = -1;
        bool successor_wide    = false;
        std::vector<std::uint64_t> total; // the narrow calls' routes of each expert
        bool cpu = false;                 // the family computes its experts on the CPU
        // Where each expert's entries point besides a cache slot: the lent slot holding it, and
        // the bank whose layout they address (-1: none).
        std::vector<std::int32_t> lent_of;
        std::vector<std::int8_t> bank_of;
    };
    struct Bank {
        std::int64_t layer = -1;
        std::vector<unsigned char> valid;
    };
    struct Slot {
        std::int32_t layer = -1, expert = -1;
        // Its table entries still point elsewhere: a promotion's are patched at the next call, a
        // round's once its copy has landed.
        bool pending = false;
        std::vector<std::uint64_t> written;
    };
    // A slot of a lent bank: bank k holds slots [k * per_bank, (k + 1) * per_bank). `used`: the
    // last call that routed its expert.
    struct LentSlot {
        std::int32_t layer = -1, expert = -1;
        std::uint64_t used = 0;
        std::vector<std::uint64_t> written;
    };
    struct Deferred {
        std::size_t layer;
        std::int32_t expert, slot;
        cudaEvent_t landed = nullptr; // a round's copy; null: a promotion, ordered on `copies`
    };
    // A host-to-device copy of contiguous bytes from one pinned range or a run of file extents.
    struct Piece {
        std::byte* target = nullptr;
        const std::byte* pinned = nullptr;
        std::uintptr_t region = 0;
        std::vector<ExpertExtent> extents;
        std::uint64_t bytes = 0;
    };

    RankContext rank;
    ExpertDeliveryOptions options;
    std::vector<Layer> layers;
    std::uint32_t max_experts = 0;
    // Per blob: the slot offset/size (aligned for the largest blob and tail of any layer), the
    // bank region offset/size and the zero tail kept after a bank run.
    std::vector<std::uint64_t> slot_offset, slot_size, bank_offset, bank_size, tail;
    std::uint64_t slot_bytes = 0, bank_bytes = 0;
    std::uint64_t expert_count = 0;
    Storage cache, cached_ids, missing_ids, needed_device;
    // The addresses a table entry lies in when it points into the cache: an elastic cache's whole
    // reservation, so graphs captured before it grew still find its later slots.
    std::uint64_t cache_span = 0;
    std::vector<Storage> banks;
    std::vector<Bank> bank_state;
    int last_bank = 0;
    // Lent banks (ExpertDeliveryOptions::lend_banks): their slots, the slots one bank holds (0:
    // banks are never lent), which banks are lent, the device copy of their address ranges the
    // split reads and its pinned source, the narrow passes since the last wide call, and the
    // served calls' count.
    std::vector<LentSlot> lent_slots;
    std::uint32_t per_bank = 0;
    std::vector<unsigned char> bank_lent;
    Storage lent_ranges;
    std::unique_ptr<PinnedHostBuffer> host_ranges;
    bool ranges_dirty         = false;
    std::uint32_t quiet_passes = 0;
    std::uint64_t stamp        = 0;
    std::unique_ptr<PinnedHostBuffer> needed_host, staging;
    std::vector<Slot> slots;
    std::vector<std::int32_t> free_slots;
    std::vector<Deferred> deferred;
    // The tier over every slot (one pool), the last narrow call's layer (a lower one starts a
    // pass), and a round's candidates whose copies post work has not issued yet.
    std::unique_ptr<ExpertTier> tier;
    std::int64_t last_narrow = -1;
    std::vector<Deferred> queued;
    cudaStream_t rounds = nullptr;
    std::vector<cudaEvent_t> spare_events;
    std::unique_ptr<PinnedHostBuffer> round_staging; // disk experts: a call's round reads
    cudaEvent_t round_staged = nullptr;               // the staging's last copies
    // A CPU share: the pool, the split, the call's inputs and ids on the host, the CPU's
    // products and flag, the corrected missing ids, and disk experts' bytes read for the CPU.
    std::unique_ptr<CpuSpinPool> pool;
    ExpertSplit split;
    std::unique_ptr<PinnedHostBuffer> host_inputs, host_ids, host_products, host_flag, host_missing;
    Storage products, cpu_flag;
    std::vector<std::byte> cpu_bytes;
    cudaEvent_t copies_begun = nullptr, copies_done = nullptr; // a call's copies, timed
    bool spinning = false;
    CudaHostHandshake handshake;
    cudaStream_t copies = nullptr;
    cudaEvent_t copied[2]{};
    std::uint64_t half_bytes = 0;
    int half = 0;
    std::uint64_t staging_cursor = 0;
    std::vector<QueuedFileRead> batch_reads;
    struct StagedCopy { std::byte* target; const std::uint8_t* source; std::size_t bytes; };
    std::vector<StagedCopy> batch_copies;
    std::vector<std::ifstream> files;
    std::unique_ptr<FileReadQueue> read_queue;
    std::atomic<bool> stopping{false};
    std::atomic<std::uint32_t> finished{0}; // the last request whose service, post work included, ended
    std::thread service;
    mutable std::mutex mutex;
    std::exception_ptr failure;
    ExpertDeliveryStats counters;
    ExpertDeliveryMemory allocation;
    bool released = false;
    // The call being served and its post work: layer, bank, whether wide, and its routed experts.
    // Between a call's ready signal and the next request the GPU may read any layer's table (the
    // next split, then that call's cached pairs), so post work writes only free slots. Cached
    // experts are replaced only while a request is served, never one it routes.
    std::int64_t previous_layer = -1;
    std::size_t call_layer      = 0;
    int call_bank               = 0;
    bool call_wide              = false;
    std::vector<unsigned char> call_routed, call_cpu;
    std::vector<std::uint32_t> call_count;
    std::vector<Piece> pieces;
    std::vector<std::pair<std::byte*, std::uint64_t>> zeroes;

    Impl(const RankContext& r, std::vector<std::filesystem::path> paths,
         std::vector<ExpertDeliveryLayer> specs, ExpertDeliveryOptions o)
        : rank(r), options(o), handshake(r) {
        DeviceBinding bind(rank.device);
        if (specs.empty() || options.pair_capacity == 0) {
            throw std::invalid_argument("expert delivery: layers and pair capacity are required");
        }
        if (options.banks < 1 || options.banks > 2) {
            throw std::invalid_argument("expert delivery: one or two layer banks");
        }
        bool disk = false;
        std::vector<std::uint64_t> largest, largest_tail;
        for (const auto& layer : specs) {
            expert_count += layer.experts.size();
            if (layer.experts.empty() || layer.aliases.empty()) {
                throw std::invalid_argument("expert delivery: empty expert or alias table");
            }
            if (layer.experts.size() > 8192) {
                throw std::invalid_argument("expert delivery: at most 8192 experts per layer");
            }
            max_experts = std::max(max_experts, std::uint32_t(layer.experts.size()));
            const std::size_t blobs = layer.experts.front().size();
            for (const auto& expert : layer.experts) {
                if (expert.empty() || expert.size() != blobs) {
                    throw std::invalid_argument("expert delivery: every expert of a layer needs its blobs");
                }
                largest.resize(std::max(largest.size(), expert.size()), 0);
                largest_tail.resize(largest.size(), 0);
                for (std::size_t b = 0; b < expert.size(); ++b) {
                    const auto& blob = expert[b];
                    if (!blob.bytes || (blob.pinned != nullptr) == !blob.extents.empty()) {
                        throw std::invalid_argument("expert delivery: choose pinned bytes or extents");
                    }
                    std::uint64_t extent_bytes = 0;
                    for (const auto& extent : blob.extents) {
                        if (extent.file >= paths.size() || !extent.bytes ||
                            extent.bytes > blob.bytes - std::min(blob.bytes, extent_bytes)) {
                            throw std::invalid_argument("expert delivery: invalid file extent");
                        }
                        extent_bytes += extent.bytes;
                    }
                    if (!blob.pinned && extent_bytes != blob.bytes) {
                        throw std::invalid_argument("expert delivery: extents do not cover blob");
                    }
                    if (blob.zero_tail > std::numeric_limits<std::uint64_t>::max() - blob.bytes) {
                        throw std::invalid_argument("expert delivery: blob tail overflow");
                    }
                    largest[b]      = std::max(largest[b], blob.bytes);
                    largest_tail[b] = std::max(largest_tail[b], blob.zero_tail);
                    disk |= !blob.extents.empty();
                }
                for (const auto& alias : layer.aliases) {
                    if (alias.blob >= expert.size() || alias.offset >= expert[alias.blob].bytes) {
                        throw std::invalid_argument("expert delivery: alias outside blob");
                    }
                }
            }
        }
        for (std::size_t b = 0; b < largest.size(); ++b) {
            slot_offset.push_back(slot_bytes);
            slot_size.push_back(aligned(checked_add(largest[b], largest_tail[b])));
            slot_bytes = checked_add(slot_bytes, slot_size.back());
        }
        tail = largest_tail;
        bank_size.assign(largest.size(), 0);
        if (options.cpu.products != nullptr &&
            (options.cpu.product_bytes == 0 || !(options.cpu.share >= 0 && options.cpu.share <= 1))) {
            throw std::invalid_argument("expert delivery: a CPU share needs its products and a share in [0, 1]");
        }
        std::uint64_t largest_expert = 0;
        for (std::size_t index = 0; index < specs.size(); ++index) {
            auto& spec = specs[index];
            Layer layer;
            for (const auto& expert : spec.experts) {
                std::uint64_t bytes = 0;
                for (const auto& blob : expert) { bytes += blob.bytes; }
                largest_expert = std::max(largest_expert, bytes);
            }
            layer.total.assign(spec.experts.size(), 0);
            layer.cpu = options.cpu.products != nullptr && options.cpu.products->supports(index);
            const auto count = spec.experts.size();
            const auto blobs = spec.experts.front().size();
            for (std::size_t b = 0; b < blobs; ++b) {
                const auto first = spec.experts.front()[b].bytes;
                const bool equal = std::all_of(spec.experts.begin(), spec.experts.end(),
                    [&](const auto& expert) { return expert[b].bytes == first; });
                // Kernels read these planes as the resident bank: every expert at its index.
                const bool resident = equal && first % 4 == 0;
                layer.stride.push_back(resident ? first : aligned(largest[b]));
                bank_size[b] = std::max(bank_size[b],
                                        checked_add(checked_mul(count, layer.stride.back()), tail[b]));
            }
            layer.slot_of.assign(count, -1);
            layer.lent_of.assign(count, -1);
            layer.bank_of.assign(count, -1);
            const auto bytes = count * spec.aliases.size() * sizeof(void*);
            layer.mirror = std::make_unique<PinnedHostBuffer>(bytes);
            std::memset(layer.mirror->data(), 0, bytes);
            for (int u = 0; u < 2; ++u) {
                layer.upload[u] = std::make_unique<PinnedHostBuffer>(bytes);
                CUDA_CHECK(cudaEventCreateWithFlags(&layer.uploaded[u], cudaEventDisableTiming));
                allocation.pinned_bytes += bytes;
            }
            layer.source = std::move(spec);
            allocation.pinned_bytes += bytes;
            layers.push_back(std::move(layer));
        }
        for (const auto size : bank_size) {
            bank_offset.push_back(bank_bytes);
            bank_bytes = checked_add(bank_bytes, aligned(size));
        }
        cached_ids = Storage(std::size_t(options.pair_capacity) * 4, rank.device, options.suspendable);
        missing_ids = Storage(std::size_t(options.pair_capacity) * 4, rank.device, options.suspendable);
        needed_device = Storage(std::size_t(max_experts) * 4, rank.device, options.suspendable);
        needed_host = std::make_unique<PinnedHostBuffer>(std::size_t(max_experts) * 4);
        std::memset(needed_host->data(), 0, needed_host->size());
        allocation.metadata_bytes = cached_ids.backing_bytes() + missing_ids.backing_bytes() +
                                     needed_device.backing_bytes();
        allocation.pinned_bytes += needed_host->size() + handshake.pinned_bytes();
        for (auto& layer : layers) {
            const auto bytes = layer.mirror->size();
            layer.tables = Storage(bytes, rank.device, options.suspendable);
            layer.tables.fill();
            allocation.metadata_bytes += layer.tables.backing_bytes();
        }
        if (disk) {
            half_bytes = aligned(std::max<std::uint64_t>(options.staging_bytes / 2, 256));
            staging = std::make_unique<PinnedHostBuffer>(2 * half_bytes);
            allocation.pinned_bytes += staging->size();
            for (const auto& path : paths) {
                files.emplace_back(path, std::ios::binary);
                if (!files.back()) {
                    throw std::runtime_error("expert delivery: cannot open " + path.string());
                }
            }
            read_queue = FileReadQueue::open(paths, false, 16);
        }
        const std::uint64_t granule = options.suspendable ? vmm::granularity(rank.device) : 1;
        if (!granule) {
            throw std::invalid_argument("expert delivery: suspend needs VMM allocation granularity");
        }
        const std::uint64_t cache_granule = options.elastic ? vmm::granularity(rank.device) : granule;
        if (!cache_granule) {
            throw std::invalid_argument("expert delivery: an elastic cache needs VMM allocation");
        }
        const auto physical = [&](std::uint64_t bytes) {
            return checked_add(bytes, granule - 1) / granule * granule;
        };
        const auto available = [&] {
            std::size_t free = 0, total = 0;
            CUDA_CHECK(cudaMemGetInfo(&free, &total));
            return free > options.keep_free_bytes ? free - options.keep_free_bytes : 0;
        };
        // Banks before the cache: every miss lands in one. A second bank is dropped before the
        // last one, and an explicit cache keeps its size by dropping the second bank first.
        std::uint32_t bank_count = options.banks;
        const std::uint64_t explicit_cache =
            options.cache_bytes ? physical((*options.cache_bytes / slot_bytes) * slot_bytes) : 0;
        while (bank_count > 1 && bank_count * physical(bank_bytes) + explicit_cache > available()) {
            --bank_count;
        }
        if (physical(bank_bytes) > available()) {
            throw std::invalid_argument(
                "expert delivery: a layer bank needs " + std::to_string(physical(bank_bytes)) +
                " device bytes, but only " + std::to_string(available()) +
                " remain after keep_free_bytes=" + std::to_string(options.keep_free_bytes));
        }
        banks.reserve(bank_count);
        for (std::uint32_t b = 0; b < bank_count; ++b) {
            banks.emplace_back(bank_bytes, rank.device, options.suspendable);
            banks.back().fill();
            bank_state.push_back({-1, std::vector<unsigned char>(max_experts, 0)});
            allocation.bank_bytes += banks.back().backing_bytes();
        }
        allocation.banks = bank_count;
        // A narrow call routes fewer than half a layer's experts, so one lent bank whose slots
        // number at least that many always has a slot for a miss that its call does not route.
        bank_lent.assign(bank_count, 0);
        if (options.lend_banks && 2 * (bank_bytes / slot_bytes) >= max_experts) {
            per_bank = std::uint32_t(bank_bytes / slot_bytes);
            lent_slots.resize(std::size_t(per_bank) * bank_count);
            for (auto& slot : lent_slots) { slot.written.assign(slot_offset.size(), 0); }
        }
        lent_ranges = Storage(4 * sizeof(std::uintptr_t), rank.device, options.suspendable);
        lent_ranges.fill();
        host_ranges = std::make_unique<PinnedHostBuffer>(4 * sizeof(std::uintptr_t));
        std::memset(host_ranges->data(), 0, host_ranges->size());
        allocation.metadata_bytes += lent_ranges.backing_bytes();
        allocation.pinned_bytes += host_ranges->size();
        const std::uint64_t remaining = available();
        // Automatic slots may use only whole physical granules inside the remaining budget.
        // An explicit request must preserve the same reserve, including VMM rounding.
        const auto budget = options.cache_bytes.value_or(remaining / cache_granule * cache_granule);
        const auto count = std::min<std::uint64_t>(budget / slot_bytes, expert_count);
        const auto cache_bytes = count * slot_bytes;
        const auto backing_bytes = checked_add(cache_bytes, cache_granule - 1) / cache_granule * cache_granule;
        if (backing_bytes > remaining) {
            throw std::invalid_argument(
                "expert delivery: cache needs " + std::to_string(backing_bytes) +
                " device bytes, but only " + std::to_string(remaining) +
                " remain after keep_free_bytes=" + std::to_string(options.keep_free_bytes));
        }
        if (options.elastic) {
            cache.growable = GrowableRegion(checked_mul(expert_count, slot_bytes), {.device = rank.device});
            cache.p = cache.growable.data();
            if (!cache.growable.map(0, cache_bytes)) {
                throw std::invalid_argument("expert delivery: the device cannot back the cache");
            }
            cache.bytes = cache_bytes;
        }
        if (count) {
            if (!options.elastic) { cache = Storage(cache_bytes, rank.device, options.suspendable); }
            cache.fill();
            slots.resize(count);
            for (auto& slot : slots) { slot.written.assign(slot_offset.size(), 0); }
            for (std::int64_t s = std::int64_t(count) - 1; s >= 0; --s) {
                free_slots.push_back(std::int32_t(s));
            }
        }
        cache_span = options.elastic ? cache.growable.bytes() : cache.bytes;
        allocation.cache_bytes = cache.backing_bytes();
        allocation.cache_slots = std::uint32_t(slots.size());
        call_routed.assign(max_experts, 0);
        call_cpu.assign(max_experts, 0);
        call_count.assign(max_experts, 0);
        std::vector<std::uint32_t> widths;
        for (const auto& layer : layers) { widths.push_back(std::uint32_t(layer.source.experts.size())); }
        tier = std::make_unique<ExpertTier>(widths, std::vector<std::uint32_t>(widths.size(), 0),
                                            options.tier);
        tier->set_free(0, std::uint32_t(free_slots.size()));
        if (disk) {
            round_staging = std::make_unique<PinnedHostBuffer>(
                std::size_t(std::max(kRoundFileBytes, largest_expert)));
            allocation.pinned_bytes += round_staging->size();
        }
        if (options.cpu.products != nullptr) {
            const auto pinned = [&](std::size_t bytes) {
                auto buffer = std::make_unique<PinnedHostBuffer>(std::max<std::size_t>(bytes, 64));
                std::memset(buffer->data(), 0, buffer->size());
                allocation.pinned_bytes += buffer->size();
                return buffer;
            };
            host_inputs   = pinned(options.cpu.input_bytes);
            host_ids      = pinned(std::size_t(options.pair_capacity) * 4);
            host_missing  = pinned(std::size_t(options.pair_capacity) * 4);
            host_products = pinned(options.cpu.product_bytes);
            host_flag     = pinned(4);
            products = Storage(options.cpu.product_bytes, rank.device, options.suspendable);
            cpu_flag = Storage(4, rank.device, options.suspendable);
            products.fill();
            cpu_flag.fill();
            allocation.metadata_bytes += products.backing_bytes() + cpu_flag.backing_bytes();
            pool = std::make_unique<CpuSpinPool>(
                options.cpu.threads != 0 ? options.cpu.threads : pool_cpu_count(), true);
        }
        CUDA_CHECK(cudaFuncSetAttribute(split_routes, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        int(max_experts * 4)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&copies, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&rounds, cudaStreamNonBlocking));
        try {
            for (auto& event : copied) { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
            CUDA_CHECK(cudaEventCreateWithFlags(&round_staged, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreate(&copies_begun));
            CUDA_CHECK(cudaEventCreate(&copies_done));
            service = std::thread([this] { run(); });
        } catch (...) {
            for (auto event : copied) { if (event) { cudaEventDestroy(event); } }
            for (auto event : {round_staged, copies_begun, copies_done}) {
                if (event) { cudaEventDestroy(event); }
            }
            cudaStreamDestroy(copies);
            cudaStreamDestroy(rounds);
            throw;
        }
    }

    ~Impl() {
        DeviceBinding bind(rank.device);
        // Keep the worker alive until the last graph's waits and consumers finish.
        cudaStreamSynchronize(rank.stream);
        stopping.store(true, std::memory_order_release);
        if (service.joinable()) { service.join(); }
        cudaStreamSynchronize(copies);
        cudaStreamSynchronize(rounds);
        for (auto event : copied) { cudaEventDestroy(event); }
        for (auto& layer : layers) {
            for (auto event : layer.uploaded) { if (event) { cudaEventDestroy(event); } }
        }
        for (const auto& d : deferred) {
            if (d.landed && std::find(spare_events.begin(), spare_events.end(), d.landed) ==
                                spare_events.end()) {
                spare_events.push_back(d.landed);
            }
        }
        for (auto event : spare_events) { cudaEventDestroy(event); }
        for (auto event : {round_staged, copies_begun, copies_done}) { cudaEventDestroy(event); }
        cudaStreamDestroy(copies);
        cudaStreamDestroy(rounds);
    }

    void check() const {
        std::lock_guard lock(mutex);
        if (failure) { std::rethrow_exception(failure); }
    }

    template <typename Function> void each_storage(Function&& function) {
        for (Storage* storage : {&cache, &cached_ids, &missing_ids, &needed_device, &products, &cpu_flag,
                                 &lent_ranges}) {
            if (storage->bytes) { function(*storage); }
        }
        for (auto& bank : banks) { function(bank); }
        for (auto& layer : layers) { function(layer.tables); }
    }

    [[nodiscard]] std::byte* slot_base(std::int32_t slot) const {
        return static_cast<std::byte*>(cache.p) + std::uint64_t(slot) * slot_bytes;
    }
    [[nodiscard]] std::byte* bank_blob(int bank, std::size_t li, std::int32_t e, std::size_t b) const {
        return static_cast<std::byte*>(banks[std::size_t(bank)].p) + bank_offset[b] +
               std::uint64_t(e) * layers[li].stride[b];
    }

    // The table entries of (li, e) point at base(b) for each alias's blob, or at nothing.
    template <typename Base> void point(std::size_t li, std::int32_t e, Base&& base) {
        auto& layer = layers[li];
        const auto n = layer.source.experts.size();
        auto** mirror = static_cast<const void**>(layer.mirror->data());
        for (std::size_t a = 0; a < layer.source.aliases.size(); ++a) {
            const auto alias = layer.source.aliases[a];
            std::byte* blob = base(alias.blob);
            mirror[a * n + std::size_t(e)] = blob ? blob + alias.offset : nullptr;
        }
        layer.dirty = true;
    }
    // Before (li, e)'s entries point elsewhere: the lent slot holding it is free again.
    void leave(std::size_t li, std::int32_t e) {
        auto& layer = layers[li];
        layer.bank_of[std::size_t(e)] = -1;
        const auto l = layer.lent_of[std::size_t(e)];
        if (l < 0) { return; }
        layer.lent_of[std::size_t(e)] = -1;
        lent_slots[std::size_t(l)].layer = lent_slots[std::size_t(l)].expert = -1;
    }
    void point_slot(std::size_t li, std::int32_t e, std::int32_t slot) {
        leave(li, e);
        std::byte* base = slot_base(slot);
        point(li, e, [&](std::size_t b) { return base + slot_offset[b]; });
    }
    void point_bank(std::size_t li, std::int32_t e, int bank) {
        leave(li, e);
        point(li, e, [&](std::size_t b) { return bank_blob(bank, li, e, b); });
        layers[li].bank_of[std::size_t(e)] = std::int8_t(bank);
    }
    void point_nowhere(std::size_t li, std::int32_t e) {
        leave(li, e);
        point(li, e, [](std::size_t) { return static_cast<std::byte*>(nullptr); });
    }
    [[nodiscard]] std::byte* lent_base(std::int32_t l) const {
        const auto bank = std::uint32_t(l) / per_bank;
        return static_cast<std::byte*>(banks[bank].p) + std::uint64_t(std::uint32_t(l) % per_bank) * slot_bytes;
    }
    void point_lent(std::size_t li, std::int32_t e, std::int32_t l) {
        leave(li, e);
        std::byte* base = lent_base(l);
        point(li, e, [&](std::size_t b) { return base + slot_offset[b]; });
        layers[li].lent_of[std::size_t(e)] = l;
        auto& slot = lent_slots[std::size_t(l)];
        slot.layer = std::int32_t(li);
        slot.expert = e;
        slot.used = stamp;
    }

    // Lends bank k: nothing points into its layout any more, and its zeroed bytes hold free slots.
    // The served call's split read every such entry as a miss, and earlier calls' consumers are done.
    void lend_bank(int k) {
        for (std::size_t li = 0; li < layers.size(); ++li) {
            for (std::size_t e = 0; e < layers[li].bank_of.size(); ++e) {
                if (layers[li].bank_of[e] == k) { point_nowhere(li, std::int32_t(e)); }
            }
        }
        bank_state[std::size_t(k)].layer = -1;
        std::fill(bank_state[std::size_t(k)].valid.begin(), bank_state[std::size_t(k)].valid.end(), 0);
        CUDA_CHECK(cudaMemsetAsync(banks[std::size_t(k)].p, 0, banks[std::size_t(k)].bytes, copies));
        for (std::uint32_t i = 0; i < per_bank; ++i) {
            auto& slot = lent_slots[std::size_t(k) * per_bank + i];
            slot.layer = slot.expert = -1;
            std::fill(slot.written.begin(), slot.written.end(), 0);
        }
        bank_lent[std::size_t(k)] = 1;
        auto* ranges = static_cast<std::uintptr_t*>(host_ranges->data());
        ranges[2 * k]     = reinterpret_cast<std::uintptr_t>(banks[std::size_t(k)].p);
        ranges[2 * k + 1] = ranges[2 * k] + std::uint64_t(per_bank) * slot_bytes;
        ranges_dirty = true;
        set_lent_count();
    }

    // A wide call takes bank k back: the experts of its slots leave (the call read their entries
    // as misses).
    void reclaim_bank(int k, ExpertDeliveryStats& delta) {
        for (std::uint32_t i = 0; i < per_bank; ++i) {
            const auto& slot = lent_slots[std::size_t(k) * per_bank + i];
            if (slot.layer < 0) { continue; }
            point_nowhere(std::size_t(slot.layer), slot.expert);
            ++delta.lent_evictions;
        }
        bank_lent[std::size_t(k)] = 0;
        auto* ranges = static_cast<std::uintptr_t*>(host_ranges->data());
        ranges[2 * k] = ranges[2 * k + 1] = 0;
        ranges_dirty = true;
        set_lent_count();
    }

    void set_lent_count() {
        std::uint32_t lent = 0;
        for (const auto flag : bank_lent) { lent += flag ? per_bank : 0; }
        std::lock_guard lock(mutex);
        allocation.lent_slots = lent;
    }

    [[nodiscard]] bool any_lent() const {
        return std::any_of(bank_lent.begin(), bank_lent.end(), [](unsigned char flag) { return flag != 0; });
    }

    // The lent slots a narrow call of layer li may fill, best first: free ones, then the least
    // routed experts (the tier's counts, then the oldest use) that the call does not route.
    std::vector<std::int32_t> lent_candidates(std::size_t li) const {
        std::vector<std::int32_t> out;
        for (std::size_t k = 0; k < bank_lent.size(); ++k) {
            if (!bank_lent[k]) { continue; }
            for (std::uint32_t i = 0; i < per_bank; ++i) {
                const auto l = std::int32_t(k * per_bank + i);
                const auto& slot = lent_slots[std::size_t(l)];
                if (slot.layer == std::int32_t(li) && call_routed[std::size_t(slot.expert)]) { continue; }
                out.push_back(l);
            }
        }
        const auto key = [&](std::int32_t l) {
            const auto& slot = lent_slots[std::size_t(l)];
            return slot.layer < 0 ? std::tuple{0, 0.0F, std::uint64_t(0)}
                                  : std::tuple{1, tier->count(std::size_t(slot.layer), slot.expert), slot.used};
        };
        std::stable_sort(out.begin(), out.end(), [&](std::int32_t a, std::int32_t b) { return key(a) < key(b); });
        return out;
    }

    // Copies a narrow call's missing expert e of layer li into lent slot l.
    void admit_lent(std::int32_t l, std::size_t li, std::int32_t e, ExpertDeliveryStats& delta) {
        const auto& slot = lent_slots[std::size_t(l)];
        if (slot.layer >= 0) {
            point_nowhere(std::size_t(slot.layer), slot.expert);
            ++delta.lent_evictions;
        }
        auto& written = lent_slots[std::size_t(l)].written;
        std::byte* base = lent_base(l);
        for (std::size_t b = 0; b < layers[li].source.experts[std::size_t(e)].size(); ++b) {
            const auto& blob = layers[li].source.experts[std::size_t(e)][b];
            add_piece(base + slot_offset[b], blob);
            if (written[b] > blob.bytes) {
                zeroes.push_back({base + slot_offset[b] + blob.bytes, written[b] - blob.bytes});
            }
            written[b] = blob.bytes;
        }
        point_lent(li, e, l);
        ++delta.lent_admissions;
    }

    // One copy per changed table, from a pinned copy no earlier upload still reads.
    void upload_tables() {
        for (auto& layer : layers) {
            if (!layer.dirty) { continue; }
            const int u = layer.next_upload;
            layer.next_upload = 1 - u;
            CUDA_CHECK(cudaEventSynchronize(layer.uploaded[u]));
            std::memcpy(layer.upload[u]->data(), layer.mirror->data(), layer.mirror->size());
            CUDA_CHECK(cudaMemcpyAsync(layer.tables.p, layer.upload[u]->data(), layer.mirror->size(),
                                       cudaMemcpyHostToDevice, copies));
            CUDA_CHECK(cudaEventRecord(layer.uploaded[u], copies));
            layer.dirty = false;
        }
    }

    // Queue a host-to-device copy, extending the previous piece when both its source and its
    // target continue it.
    void add_piece(std::byte* target, const ExpertBlob& blob) {
        if (!pieces.empty()) {
            auto& last = pieces.back();
            const bool target_follows = last.target + last.bytes == target;
            if (target_follows && blob.pinned && last.pinned && blob.pinned_region != 0 &&
                blob.pinned_region == last.region &&
                last.pinned + last.bytes == static_cast<const std::byte*>(blob.pinned)) {
                last.bytes += blob.bytes;
                return;
            }
            if (target_follows && !blob.pinned && !last.pinned) {
                for (const auto& extent : blob.extents) {
                    auto& end = last.extents.back();
                    if (end.file == extent.file && end.offset + end.bytes == extent.offset) {
                        end.bytes += extent.bytes;
                    } else {
                        last.extents.push_back(extent);
                    }
                }
                last.bytes += blob.bytes;
                return;
            }
        }
        Piece piece;
        piece.target = target;
        piece.pinned = static_cast<const std::byte*>(blob.pinned);
        piece.region = blob.pinned_region;
        piece.extents = blob.extents;
        piece.bytes = blob.bytes;
        pieces.push_back(std::move(piece));
    }

    // Several reads share one bounded half. FileReadQueue can submit their extents together.
    // Its fallback reads the same ranges through the OS page cache.
    void read_now(std::span<const QueuedFileRead> reads) {
        if (read_queue) {
            read_queue->read(reads);
            return;
        }
        for (const auto& read : reads) {
            auto& file = files.at(read.file);
            file.clear();
            file.seekg(std::streamoff(read.offset));
            file.read(reinterpret_cast<char*>(read.destination.data()),
                      std::streamsize(read.destination.size()));
            if (!file) { throw std::runtime_error("expert delivery: incomplete file read"); }
        }
    }

    void flush_reads() {
        if (batch_reads.empty()) { return; }
        CUDA_CHECK(cudaEventSynchronize(copied[half]));
        read_now(batch_reads);
        for (const auto& copy : batch_copies) {
            CUDA_CHECK(cudaMemcpyAsync(copy.target, copy.source, copy.bytes,
                                       cudaMemcpyHostToDevice, copies));
        }
        CUDA_CHECK(cudaEventRecord(copied[half], copies));
        batch_reads.clear();
        batch_copies.clear();
        staging_cursor = 0;
        half = 1 - half;
    }

    void stage(const Piece& piece, ExpertDeliveryStats& delta) {
        std::uint64_t destination = 0;
        for (const auto& extent : piece.extents) {
            std::uint64_t within = 0;
            while (within < extent.bytes) {
                if (staging_cursor == half_bytes) { flush_reads(); }
                auto* buffer = static_cast<std::uint8_t*>(staging->data()) +
                               half * half_bytes + staging_cursor;
                const auto bytes = std::min(extent.bytes - within, half_bytes - staging_cursor);
                batch_reads.push_back({extent.file, extent.offset + within,
                                       {buffer, std::size_t(bytes)}});
                batch_copies.push_back({piece.target + destination, buffer, std::size_t(bytes)});
                staging_cursor += bytes;
                destination += bytes;
                within += bytes;
                delta.read_bytes += bytes;
            }
        }
    }

    void flush_pieces(ExpertDeliveryStats& delta) {
        for (const auto& piece : pieces) {
            if (piece.pinned) {
                CUDA_CHECK(cudaMemcpyAsync(piece.target, piece.pinned, piece.bytes,
                                           cudaMemcpyHostToDevice, copies));
            } else {
                stage(piece, delta);
            }
            delta.copied_bytes += piece.bytes;
        }
        flush_reads();
        for (const auto& [target, bytes] : zeroes) {
            CUDA_CHECK(cudaMemsetAsync(target, 0, bytes, copies));
        }
        pieces.clear();
        zeroes.clear();
    }

    // Bytes of blob b written into slot s: zero what a larger earlier blob left past them, so the
    // slot's tail stays zero.
    void slot_written(std::int32_t s, std::size_t b, std::uint64_t bytes) {
        auto& written = slots[std::size_t(s)].written[b];
        if (written > bytes) { zeroes.push_back({slot_base(s) + slot_offset[b] + bytes, written - bytes}); }
        written = bytes;
    }

    void evict(std::int32_t s, ExpertDeliveryStats& delta) {
        auto& slot = slots[std::size_t(s)];
        auto& layer = layers[std::size_t(slot.layer)];
        layer.slot_of[std::size_t(slot.expert)] = -1;
        --layer.cached;
        point_nowhere(std::size_t(slot.layer), slot.expert);
        tier->set(std::size_t(slot.layer), slot.expert, ExpertTier::State::Out);
        slot.layer = slot.expert = -1;
        slot.pending = false;
        ++delta.evictions;
    }

    // A free slot for the served call's missing expert e of layer li, routed `pairs` times, else
    // the slot of the coldest expert the call does not route when the tier admits e. -1: none.
    std::int32_t admit_slot(std::size_t li, std::int32_t e, std::uint32_t pairs,
                            ExpertDeliveryStats& delta) {
        if (!free_slots.empty()) {
            const auto s = free_slots.back();
            free_slots.pop_back();
            return s;
        }
        const auto victim = tier->admission_victim(li, e, float(pairs), call_routed);
        if (victim.expert < 0) { return -1; }
        const auto s = layers[victim.layer].slot_of[std::size_t(victim.expert)];
        if (s < 0 || slots[std::size_t(s)].pending) { return -1; }
        evict(s, delta);
        return s;
    }

    void occupy(std::int32_t s, std::size_t li, std::int32_t e, bool pending) {
        auto& slot = slots[std::size_t(s)];
        slot.layer = std::int32_t(li);
        slot.expert = e;
        slot.pending = pending;
        layers[li].slot_of[std::size_t(e)] = s;
        ++layers[li].cached;
        tier->set(li, e, pending ? ExpertTier::State::Pending : ExpertTier::State::Held);
    }

    cudaEvent_t take_event() {
        if (!spare_events.empty()) {
            const auto event = spare_events.back();
            spare_events.pop_back();
            return event;
        }
        cudaEvent_t event = nullptr;
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        return event;
    }

    // Promotions of an earlier wide call (their copies precede the next table upload on
    // `copies`) and round candidates whose copies have landed: their entries move to their slots.
    void land() {
        std::vector<cudaEvent_t> done;
        for (auto it = deferred.begin(); it != deferred.end();) {
            if (it->landed) {
                const cudaError_t status = cudaEventQuery(it->landed);
                if (status == cudaErrorNotReady) {
                    ++it;
                    continue;
                }
                CUDA_CHECK(status);
                if (std::find(done.begin(), done.end(), it->landed) == done.end()) {
                    done.push_back(it->landed);
                }
            }
            auto& slot = slots[std::size_t(it->slot)];
            if (slot.layer == std::int32_t(it->layer) && slot.expert == it->expert && slot.pending) {
                slot.pending = false;
                point_slot(it->layer, it->expert, it->slot);
                tier->set(it->layer, it->expert, ExpertTier::State::Held);
            }
            it = deferred.erase(it);
        }
        for (const auto event : done) {
            const bool used = std::any_of(deferred.begin(), deferred.end(),
                                          [&](const Deferred& d) { return d.landed == event; });
            if (!used) { spare_events.push_back(event); }
        }
    }

    // A pass has ended: the tier's round. Victims leave now (never an expert the served call
    // routes); candidates hold their slots pending until post work has copied them in.
    void round(ExpertDeliveryStats& delta) {
        tier->set_free(0, std::uint32_t(free_slots.size()));
        for (const auto& swap : tier->end_pass()) {
            auto& layer = layers[swap.layer];
            if (layer.slot_of[std::size_t(swap.in)] >= 0) { continue; }
            std::int32_t s = -1;
            if (swap.out < 0) {
                if (free_slots.empty()) { continue; }
                s = free_slots.back();
                free_slots.pop_back();
            } else {
                s = layers[swap.out_layer].slot_of[std::size_t(swap.out)];
                if (s < 0 || slots[std::size_t(s)].pending ||
                    (swap.out_layer == call_layer && call_routed[std::size_t(swap.out)])) {
                    continue;
                }
                evict(s, delta);
            }
            occupy(s, swap.layer, swap.in, true);
            queued.push_back({swap.layer, swap.in, s, nullptr});
            ++delta.admissions;
        }
    }

    // Post work of a narrow call: copies of the oldest queued candidates on the side stream, a few
    // experts and file bytes at a time, so the next call's service is not delayed.
    void issue_rounds(ExpertDeliveryStats& delta) {
        if (queued.empty()) { return; }
        const cudaEvent_t done = take_event();
        std::uint64_t staged = 0;
        std::size_t issued = 0;
        std::vector<QueuedFileRead> reads;
        std::vector<StagedCopy> staged_copies;
        auto* staging = round_staging ? static_cast<std::uint8_t*>(round_staging->data()) : nullptr;
        for (; issued < queued.size() && issued < kRoundExperts; ++issued) {
            const auto& d = queued[issued];
            const auto& expert = layers[d.layer].source.experts[std::size_t(d.expert)];
            std::uint64_t file_bytes = 0;
            for (const auto& blob : expert) { file_bytes += blob.pinned ? 0 : blob.bytes; }
            if (file_bytes != 0 && staged != 0 &&
                (staged + file_bytes > round_staging->size() || staged + file_bytes > kRoundFileBytes)) {
                break;
            }
            if (file_bytes != 0 && staged == 0) { CUDA_CHECK(cudaEventSynchronize(round_staged)); }
            for (std::size_t b = 0; b < expert.size(); ++b) {
                const auto& blob = expert[b];
                std::byte* target = slot_base(d.slot) + slot_offset[b];
                if (blob.pinned) {
                    CUDA_CHECK(cudaMemcpyAsync(target, blob.pinned, blob.bytes, cudaMemcpyHostToDevice,
                                               rounds));
                } else {
                    std::uint64_t at = 0;
                    for (const auto& extent : blob.extents) {
                        reads.push_back({extent.file, extent.offset,
                                         {staging + staged + at, std::size_t(extent.bytes)}});
                        at += extent.bytes;
                    }
                    staged_copies.push_back({target, staging + staged, std::size_t(blob.bytes)});
                    staged += blob.bytes;
                    delta.read_bytes += blob.bytes;
                }
                auto& written = slots[std::size_t(d.slot)].written[b];
                if (written > blob.bytes) {
                    CUDA_CHECK(cudaMemsetAsync(target + blob.bytes, 0, written - blob.bytes, rounds));
                }
                written = blob.bytes;
                delta.copied_bytes += blob.bytes;
            }
        }
        if (!reads.empty()) {
            read_now(reads);
            for (const auto& copy : staged_copies) {
                CUDA_CHECK(cudaMemcpyAsync(copy.target, copy.source, copy.bytes,
                                           cudaMemcpyHostToDevice, rounds));
            }
            CUDA_CHECK(cudaEventRecord(round_staged, rounds));
        }
        CUDA_CHECK(cudaEventRecord(done, rounds));
        for (std::size_t i = 0; i < issued; ++i) {
            auto d = queued[i];
            d.landed = done;
            deferred.push_back(d);
        }
        queued.erase(queued.begin(), queued.begin() + std::ptrdiff_t(issued));
        if (issued == 0) { spare_events.push_back(done); }
    }

    [[nodiscard]] int bank_holding(std::size_t li) const {
        for (std::size_t b = 0; b < bank_state.size(); ++b) {
            if (bank_state[b].layer == std::int64_t(li)) { return int(b); }
        }
        return -1;
    }

    // A served call may write any bank: the preceding call's consumers finished before it was
    // published. It takes the bank the last bank-using call did not use, which keeps that layer
    // for an immediate repeat (a second prefill slice, the next draft of the same layer).
    [[nodiscard]] int pick_bank(int avoid) const {
        return bank_state.size() == 1 ? 0 : avoid >= 0 ? 1 - avoid : 1 - last_bank;
    }
    void claim_bank(int k, std::size_t li) {
        auto& bank = bank_state[std::size_t(k)];
        bank.layer = std::int64_t(li);
        std::fill(bank.valid.begin(), bank.valid.end(), 0);
    }

    // Copy experts of layer li into bank k at their indices, one piece per contiguous run, and keep
    // zeros after the last expert of each run that the bank does not already continue.
    void fill_bank(int k, std::size_t li, const std::vector<std::int32_t>& experts) {
        auto& layer = layers[li];
        auto& bank = bank_state[std::size_t(k)];
        const auto blobs = layer.source.experts.front().size();
        for (std::size_t b = 0; b < blobs; ++b) {
            for (const auto e : experts) {
                add_piece(bank_blob(k, li, e, b), layer.source.experts[std::size_t(e)][b]);
            }
        }
        for (const auto e : experts) { bank.valid[std::size_t(e)] = 1; }
        for (std::size_t b = 0; b < blobs; ++b) {
            if (!tail[b]) { continue; }
            for (const auto e : experts) {
                const auto next = std::size_t(e) + 1;
                if (next < layer.source.experts.size() && bank.valid[next]) { continue; }
                const auto end = std::uint64_t(e) * layer.stride[b] +
                                 layer.source.experts[std::size_t(e)][b].bytes;
                const auto room = bank_size[b] - std::min(bank_size[b], end);
                if (room) {
                    zeroes.push_back({static_cast<std::byte*>(banks[std::size_t(k)].p) +
                                          bank_offset[b] + end,
                                      std::min(tail[b], room)});
                }
            }
        }
    }

    // Device copies of bank expert e into slot s.
    void promote(std::int32_t s, std::size_t li, std::int32_t e, int bank, ExpertDeliveryStats& delta) {
        for (std::size_t b = 0; b < layers[li].source.experts[std::size_t(e)].size(); ++b) {
            const auto bytes = layers[li].source.experts[std::size_t(e)][b].bytes;
            CUDA_CHECK(cudaMemcpyAsync(slot_base(s) + slot_offset[b], bank_blob(bank, li, e, b),
                                       bytes, cudaMemcpyDeviceToDevice, copies));
            slot_written(s, b, bytes);
            delta.promoted_bytes += bytes;
        }
        ++delta.admissions;
    }

    // The CPU's share of a narrow call's misses that neither a slot nor the layer's bank holds:
    // the experts with the fewest pairs (a copy serves all of an expert's pairs at once, the CPU's
    // work grows with each). Marks them in call_cpu.
    std::vector<std::int32_t> cpu_share(std::size_t li, const std::uint32_t* needed, int held) {
        const auto& layer = layers[li];
        std::vector<std::pair<std::uint32_t, std::int32_t>> order;
        for (int e = 0; e < int(layer.source.experts.size()); ++e) {
            if (!(needed[e] & kMiss)) { continue; }
            const auto s = layer.slot_of[std::size_t(e)];
            if (s >= 0 && !slots[std::size_t(s)].pending) { continue; }
            if (held >= 0 && bank_state[std::size_t(held)].valid[std::size_t(e)]) { continue; }
            order.emplace_back(needed[e] & kCountMask, e);
        }
        std::stable_sort(order.begin(), order.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        std::size_t count = 0;
        if (options.cpu.balance) {
            std::vector<int> pairs;
            for (const auto& [n, e] : order) { pairs.push_back(int(n)); }
            count = split.cpu_count(pairs, megabytes(li));
        } else {
            count = std::min(order.size(),
                             std::size_t(std::lround(double(order.size()) * options.cpu.share)));
        }
        std::vector<std::int32_t> out;
        for (std::size_t i = order.size() - count; i < order.size(); ++i) {
            out.push_back(order[i].second);
            call_cpu[std::size_t(order[i].second)] = 1;
        }
        return out;
    }

    [[nodiscard]] double megabytes(std::size_t li) const {
        double bytes = 0;
        for (const auto& blob : layers[li].source.experts.front()) { bytes += double(blob.bytes); }
        return bytes / 1e6;
    }

    // The CPU's products of `cpu` for the served call of `pairs` pairs, overlapping the call's
    // copies, then its missing ids without the CPU's pairs, its products and flag for the device.
    void compute_cpu(std::size_t li, const std::vector<std::int32_t>& cpu, std::uint32_t pairs,
                     const std::uint32_t* needed, ExpertDeliveryStats& delta) {
        auto* flag = static_cast<std::int32_t*>(host_flag->data());
        if (cpu.empty()) {
            if (*flag != 0) {
                *flag = 0;
                CUDA_CHECK(cudaMemcpyAsync(cpu_flag.p, flag, 4, cudaMemcpyHostToDevice, copies));
            }
            return;
        }
        const auto started = std::chrono::steady_clock::now();
        const auto& layer = layers[li];
        std::vector<std::vector<const std::byte*>> blobs(cpu.size());
        std::vector<QueuedFileRead> reads;
        std::uint64_t file_bytes = 0;
        for (const auto e : cpu) {
            for (const auto& blob : layer.source.experts[std::size_t(e)]) {
                file_bytes += blob.pinned ? 0 : blob.bytes;
            }
        }
        cpu_bytes.resize(std::size_t(file_bytes));
        std::uint64_t at = 0;
        for (std::size_t i = 0; i < cpu.size(); ++i) {
            for (const auto& blob : layer.source.experts[std::size_t(cpu[i])]) {
                if (blob.pinned) {
                    blobs[i].push_back(static_cast<const std::byte*>(blob.pinned));
                    continue;
                }
                blobs[i].push_back(cpu_bytes.data() + at);
                for (const auto& extent : blob.extents) {
                    reads.push_back({extent.file, extent.offset,
                                     {reinterpret_cast<std::uint8_t*>(cpu_bytes.data() + at),
                                      std::size_t(extent.bytes)}});
                    at += extent.bytes;
                }
            }
        }
        if (!reads.empty()) { read_now(reads); }
        const auto* ids = static_cast<const std::int32_t*>(host_ids->data());
        ExpertCpuProducts::Work work;
        work.layer    = li;
        work.experts  = cpu;
        work.blobs    = blobs;
        work.ids      = {ids, pairs};
        work.inputs   = static_cast<const std::byte*>(host_inputs->data());
        work.products = static_cast<std::byte*>(host_products->data());
        work.pool     = pool.get();
        const std::size_t used = options.cpu.products->compute(work);
        if (used > options.cpu.product_bytes) {
            throw std::logic_error("expert delivery: CPU products exceed their buffer");
        }
        auto* missing = static_cast<std::int32_t*>(host_missing->data());
        const int experts = int(layer.source.experts.size());
        for (std::uint32_t p = 0; p < pairs; ++p) {
            const auto e = ids[p];
            missing[p] = e >= 0 && e < experts && (needed[e] & kMiss) && !call_cpu[std::size_t(e)] ? e : -1;
        }
        CUDA_CHECK(cudaMemcpyAsync(missing_ids.p, missing, std::size_t(pairs) * 4, cudaMemcpyHostToDevice,
                                   copies));
        if (used != 0) {
            CUDA_CHECK(cudaMemcpyAsync(products.p, host_products->data(), used, cudaMemcpyHostToDevice,
                                       copies));
        }
        *flag = 1;
        CUDA_CHECK(cudaMemcpyAsync(cpu_flag.p, flag, 4, cudaMemcpyHostToDevice, copies));
        double work_done = 0;
        for (const auto e : cpu) {
            const int n = int(needed[e] & kCountMask);
            delta.cpu_pairs += std::uint64_t(n);
            work_done += ExpertSplit::cpu_work(megabytes(li), n);
        }
        delta.cpu_experts += cpu.size();
        delta.read_bytes += file_bytes;
        if (options.cpu.balance) {
            split.computed(work_done, 1e6 * std::chrono::duration<double>(
                                                std::chrono::steady_clock::now() - started).count());
        }
    }

    void serve(const CudaHostHandshake::Request& request) {
        const auto started = std::chrono::steady_clock::now();
        ExpertDeliveryStats delta;
        land();
        const std::size_t li = request.tag;
        auto& layer = layers.at(li);
        call_layer = li;
        const auto* needed = static_cast<const std::uint32_t*>(needed_host->data());
        const int experts = int(layer.source.experts.size());
        delta.requests = 1;
        std::uint32_t routes = 0;
        std::fill(call_routed.begin(), call_routed.end(), 0);
        std::fill(call_cpu.begin(), call_cpu.end(), 0);
        for (int e = 0; e < experts; ++e) {
            const std::uint32_t count = needed[e] & kCountMask;
            call_count[std::size_t(e)] = count;
            if (!count) { continue; }
            ++routes;
            call_routed[std::size_t(e)] = 1;
            delta.pairs += count;
            if (needed[e] & kHit) {
                ++delta.hits;
                delta.pair_hits += count;
            }
        }
        delta.routes = routes;
        const bool wide = 2 * routes >= std::uint32_t(experts);
        ++stamp;
        for (int e = 0; e < experts; ++e) {
            const auto l = layer.lent_of[std::size_t(e)];
            if (call_routed[std::size_t(e)] && l >= 0) { lent_slots[std::size_t(l)].used = stamp; }
        }
        if (previous_layer >= 0) {
            layers[std::size_t(previous_layer)].successor = std::int64_t(li);
            layers[std::size_t(previous_layer)].successor_wide = wide;
        }
        previous_layer = std::int64_t(li);
        const bool new_pass = !wide && last_narrow >= 0 && std::int64_t(li) < last_narrow;
        if (wide) { quiet_passes = 0; }
        if (per_bank && new_pass && ++quiet_passes >= 2) {
            // Only narrow calls ran for two passes: the banks become cache slots.
            for (std::size_t b = 0; b < banks.size(); ++b) {
                if (!bank_lent[b]) { lend_bank(int(b)); }
            }
        }
        const int held = bank_holding(li);
        int k = held;
        if (wide && per_bank) {
            // The wide call's bank: a lent one is taken back first.
            if (k < 0) { k = pick_bank(-1); }
            if (bank_lent[std::size_t(k)]) { reclaim_bank(k, delta); }
        }
        const bool lent = !wide && any_lent();
        std::vector<std::int32_t> lent_order;
        std::size_t lent_next = 0;
        const auto cpu = !wide && layer.cpu && (request.flags & kCpuCall) ? cpu_share(li, needed, held)
                                                                          : std::vector<std::int32_t>{};
        std::vector<std::int32_t> to_bank;
        for (int e = 0; e < experts; ++e) {
            if (!(needed[e] & kMiss)) { continue; }
            ++delta.misses;
            if (call_cpu[std::size_t(e)]) { continue; }
            const auto s = layer.slot_of[std::size_t(e)];
            if (s >= 0 && !slots[std::size_t(s)].pending) {
                // Promoted or landed after the split read its entry.
                point_slot(li, e, s);
                continue;
            }
            if (held >= 0 && bank_state[std::size_t(held)].valid[std::size_t(e)]) {
                ++delta.bank_reuses;
                point_bank(li, e, held);
                continue;
            }
            // A wide call reads a lent slot's entry as a miss; the slot keeps its bytes until a
            // later call takes its bank back.
            if (layer.lent_of[std::size_t(e)] >= 0) { continue; }
            // A round's candidate whose copy is still in flight is served from the bank.
            if (!wide && s < 0) {
                const auto slot = admit_slot(li, e, needed[e] & kCountMask, delta);
                if (slot >= 0) {
                    occupy(slot, li, e, false);
                    for (std::size_t b = 0; b < layer.source.experts[std::size_t(e)].size(); ++b) {
                        const auto& blob = layer.source.experts[std::size_t(e)][b];
                        add_piece(slot_base(slot) + slot_offset[b], blob);
                        slot_written(slot, b, blob.bytes);
                    }
                    ++delta.admissions;
                    point_slot(li, e, slot);
                    continue;
                }
            }
            if (lent) {
                // Lent banks take every miss no cache slot took, so none needs a bank.
                if (lent_order.empty()) { lent_order = lent_candidates(li); }
                if (lent_next == lent_order.size()) {
                    throw std::logic_error("expert delivery: no lent slot for a narrow call's miss");
                }
                admit_lent(lent_order[lent_next++], li, e, delta);
                continue;
            }
            to_bank.push_back(e);
        }
        if (!wide) {
            // The tier learns from decode and verification only; a call below the last narrow
            // one starts a pass (a draft layer's repeated calls do not).
            if (new_pass) { round(delta); }
            last_narrow = std::int64_t(li);
            std::lock_guard lock(mutex);
            for (int e = 0; e < experts; ++e) {
                const auto count = call_count[std::size_t(e)];
                if (!count) { continue; }
                tier->route(li, e, float(count));
                layer.total[std::size_t(e)] += count;
            }
        }
        if (!to_bank.empty()) {
            if (k < 0) { k = pick_bank(-1); }
            if (k != held) { claim_bank(k, li); }
        }
        for (const auto e : to_bank) { point_bank(li, e, k); }
        if (k >= 0 && (held >= 0 || !to_bank.empty())) { last_bank = k; }
        call_bank = (held >= 0 || !to_bank.empty()) ? k : -1;
        call_wide = wide;
        if (!to_bank.empty()) { fill_bank(k, li, to_bank); }
        const bool timed = options.cpu.balance && !cpu.empty() && !pieces.empty();
        const auto copied_before = delta.copied_bytes;
        if (timed) { CUDA_CHECK(cudaEventRecord(copies_begun, copies)); }
        flush_pieces(delta);
        if (timed) { CUDA_CHECK(cudaEventRecord(copies_done, copies)); }
        if (layer.cpu && host_flag) { compute_cpu(li, cpu, request.width, needed, delta); }
        if (ranges_dirty) {
            CUDA_CHECK(cudaMemcpyAsync(lent_ranges.p, host_ranges->data(), host_ranges->size(),
                                       cudaMemcpyHostToDevice, copies));
            ranges_dirty = false;
        }
        upload_tables();
        CUDA_CHECK(cudaStreamSynchronize(copies));
        if (timed) {
            float milliseconds = 0;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, copies_begun, copies_done));
            split.copied(double(delta.copied_bytes - copied_before) / 1e6, 1e3 * double(milliseconds));
        }
        delta.service_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        add(delta);
    }

    // After the call's experts are ready: a wide call promotes its most routed bank experts into
    // free cache slots by device copies, up to the layer's share of the cache, so a first prompt
    // fills the cache across every layer and evicts nothing. It also starts the bank of the layer
    // that followed it last time.
    void post() {
        const auto started = std::chrono::steady_clock::now();
        ExpertDeliveryStats delta;
        if (!call_wide) {
            if (queued.empty()) { return; }
            issue_rounds(delta);
            delta.service_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            add(delta);
            return;
        }
        const std::size_t li = call_layer;
        auto& layer = layers[li];
        if (!slots.empty() && call_bank >= 0) {
            const auto& bank = bank_state[std::size_t(call_bank)];
            std::vector<std::int32_t> candidates;
            for (std::size_t e = 0; e < layer.source.experts.size(); ++e) {
                if (call_routed[e] && bank.valid[e] && layer.slot_of[e] < 0) {
                    candidates.push_back(std::int32_t(e));
                }
            }
            std::sort(candidates.begin(), candidates.end(), [&](std::int32_t a, std::int32_t b) {
                const auto ca = call_count[std::size_t(a)], cb = call_count[std::size_t(b)];
                return ca != cb ? ca > cb : a < b;
            });
            const auto share = std::max<std::size_t>(1, slots.size() / layers.size());
            for (const auto e : candidates) {
                if (free_slots.empty() || layer.cached >= share) { break; }
                const auto s = free_slots.back();
                free_slots.pop_back();
                occupy(s, li, e, true);
                promote(s, li, e, call_bank, delta);
                deferred.push_back({li, e, s});
            }
        }
        if (bank_state.size() == 2 && layer.successor >= 0 && layer.successor_wide) {
            const auto next = std::size_t(layer.successor);
            const bool held = std::any_of(bank_state.begin(), bank_state.end(),
                                          [&](const Bank& b) { return b.layer == std::int64_t(next); });
            std::vector<std::int32_t> experts;
            for (std::size_t e = 0; e < layers[next].source.experts.size() && !held; ++e) {
                if (layers[next].slot_of[e] < 0) { experts.push_back(std::int32_t(e)); }
            }
            // Never the bank this call's consumers are reading, nor a lent one: the next call may
            // be narrow and read its slots.
            const int k = pick_bank(call_bank);
            if (!experts.empty() && !bank_lent[std::size_t(k)]) {
                claim_bank(k, next);
                fill_bank(k, next, experts);
                ++delta.prefetched_layers;
            }
        }
        flush_pieces(delta); // tables are unchanged: promoted entries are patched at the next request
        delta.service_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        add(delta);
    }

    void add(const ExpertDeliveryStats& d) {
        std::lock_guard lock(mutex);
        counters = counters + d;
    }

    void fail(const CudaHostHandshake::Request* request) {
        {
            std::lock_guard lock(mutex);
            if (!failure) { failure = std::current_exception(); }
        }
        batch_reads.clear();
        batch_copies.clear();
        pieces.clear();
        zeroes.clear();
        staging_cursor = 0;
        if (request) {
            // Captured decode can finish safely; the caller checks before publishing output.
            cudaMemsetAsync(missing_ids.p, 0xff, std::size_t(request->width) * 4, copies);
            if (cpu_flag.bytes) {
                cudaMemsetAsync(cpu_flag.p, 0, 4, copies);
                *static_cast<std::int32_t*>(host_flag->data()) = 0;
            }
        }
        cudaStreamSynchronize(copies);
    }

    void run() {
        DeviceBinding bind(rank.device);
        std::uint32_t served = 0;
        auto active_until = std::chrono::steady_clock::now();
        while (!stopping.load(std::memory_order_acquire)) {
            const auto request = handshake.pending(served);
            if (!request) {
                if (std::chrono::steady_clock::now() < active_until) { std::this_thread::yield(); }
                else {
                    // Idle: the CPU workers sleep until the next request.
                    if (spinning) {
                        pool->spin(false);
                        spinning = false;
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(20));
                }
                continue;
            }
            if (pool && !spinning) {
                pool->spin(true);
                spinning = true;
            }
            bool served_ok = false;
            try { check(); serve(*request); served_ok = true; }
            catch (...) { fail(&*request); }
            handshake.prepared(request->sequence);
            served = request->sequence;
            if (served_ok) {
                try { post(); }
                catch (...) { fail(nullptr); }
            }
            finished.store(served, std::memory_order_release);
            active_until = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        }
    }

    // After the compute stream is idle: the worker has finished every published request,
    // post work included, and its copies are done.
    void quiesce() {
        DeviceBinding bind(rank.device);
        CUDA_CHECK(cudaStreamSynchronize(rank.stream));
        while (const auto request = handshake.pending(finished.load(std::memory_order_acquire))) {
            (void)request;
            std::this_thread::yield();
        }
        CUDA_CHECK(cudaStreamSynchronize(copies));
        CUDA_CHECK(cudaStreamSynchronize(rounds));
    }

    // Free slots popped lowest first.
    void order_free_slots() {
        std::sort(free_slots.begin(), free_slots.end(), std::greater<>());
    }

    // An elastic cache, quiesced and locked: keeps the slots inside the granules that stay mapped
    // once `bytes` are given back. The most routed experts it holds (the tier's decayed counts, then
    // the routes since prime()) stay, moved into those slots when they sat past them; the others
    // leave, and so do a round's candidates whose copies were not issued yet.
    std::uint64_t shrink(std::uint64_t bytes) {
        const std::uint64_t before = cache.growable.mapped_bytes();
        if (slots.empty() || bytes == 0) { return 0; }
        const std::uint64_t g      = cache.growable.granule();
        const std::uint64_t drop   = (bytes + g - 1) / g;
        const std::uint64_t mapped = before / g; // the mapped granules are a prefix
        const auto keep = std::int32_t(std::min<std::uint64_t>(
            slots.size(), (mapped > drop ? mapped - drop : 0) * g / slot_bytes));
        ExpertDeliveryStats delta;
        land();
        for (const auto& d : queued) {
            if (slots[std::size_t(d.slot)].layer >= 0) {
                evict(d.slot, delta);
                free_slots.push_back(d.slot);
            }
        }
        queued.clear();
        std::vector<std::int32_t> held;
        for (std::int32_t s = 0; s < std::int32_t(slots.size()); ++s) {
            if (slots[std::size_t(s)].layer >= 0) { held.push_back(s); }
        }
        const auto rank_of = [&](std::int32_t s) {
            const auto& slot = slots[std::size_t(s)];
            return std::pair{tier->count(std::size_t(slot.layer), slot.expert),
                             layers[std::size_t(slot.layer)].total[std::size_t(slot.expert)]};
        };
        std::stable_sort(held.begin(), held.end(),
                         [&](std::int32_t a, std::int32_t b) { return rank_of(a) > rank_of(b); });
        for (std::size_t i = std::size_t(keep); i < held.size(); ++i) { evict(held[i], delta); }
        held.resize(std::min(held.size(), std::size_t(keep)));
        std::vector<std::int32_t> inside;
        for (std::int32_t s = keep - 1; s >= 0; --s) {
            if (slots[std::size_t(s)].layer < 0) { inside.push_back(s); }
        }
        for (const auto from : held) {
            if (from < keep) { continue; }
            const auto to = inside.back();
            inside.pop_back();
            CUDA_CHECK(cudaMemcpyAsync(slot_base(to), slot_base(from), slot_bytes,
                                       cudaMemcpyDeviceToDevice, copies));
            auto& source = slots[std::size_t(from)];
            auto& target = slots[std::size_t(to)];
            target.layer   = source.layer;
            target.expert  = source.expert;
            target.pending = false;
            target.written = source.written;
            layers[std::size_t(source.layer)].slot_of[std::size_t(source.expert)] = to;
            point_slot(std::size_t(source.layer), source.expert, to);
            source.layer = source.expert = -1;
        }
        free_slots = std::move(inside);
        slots.resize(std::size_t(keep));
        upload_tables();
        CUDA_CHECK(cudaStreamSynchronize(copies));
        const std::uint64_t end = std::uint64_t(keep) * slot_bytes;
        cache.growable.unmap(end, cache.growable.bytes() - end);
        cache.bytes = end;
        tier->set_free(0, std::uint32_t(free_slots.size()));
        allocation.cache_bytes = cache.backing_bytes();
        allocation.cache_slots = std::uint32_t(slots.size());
        counters = counters + delta;
        return before - cache.growable.mapped_bytes();
    }

    // An elastic cache, quiesced and locked: empty slots in the free device memory beyond
    // keep_free_bytes, up to every expert (or an explicit cache size), as much of it as the device
    // supplies.
    std::uint64_t grow() {
        const std::uint64_t before = cache.growable.mapped_bytes();
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        const std::uint64_t g    = cache.growable.granule();
        const std::uint64_t room = free > options.keep_free_bytes
                                       ? (free - options.keep_free_bytes) / g * g : 0;
        std::uint64_t most = expert_count;
        if (options.cache_bytes) { most = std::min<std::uint64_t>(most, *options.cache_bytes / slot_bytes); }
        const auto old = std::uint64_t(slots.size());
        // The slots that fit in the granules mapped now and in `room` more.
        auto target = std::min<std::uint64_t>(most, (before + room) / slot_bytes);
        while (target > old && !cache.growable.map(old * slot_bytes, (target - old) * slot_bytes)) {
            target = old + (target - old) / 2;
        }
        if (target <= old) { return 0; }
        CUDA_CHECK(cudaMemsetAsync(slot_base(std::int32_t(old)), 0, (target - old) * slot_bytes, copies));
        CUDA_CHECK(cudaStreamSynchronize(copies));
        slots.resize(std::size_t(target));
        for (auto s = old; s < target; ++s) {
            slots[std::size_t(s)].written.assign(slot_offset.size(), 0);
            free_slots.push_back(std::int32_t(s));
        }
        order_free_slots();
        cache.bytes = target * slot_bytes;
        tier->set_free(0, std::uint32_t(free_slots.size()));
        allocation.cache_bytes = cache.backing_bytes();
        allocation.cache_slots = std::uint32_t(slots.size());
        return cache.growable.mapped_bytes() - before;
    }
};

ExpertDelivery::ExpertDelivery(const RankContext& rank, std::vector<std::filesystem::path> files,
                               std::vector<ExpertDeliveryLayer> layers, ExpertDeliveryOptions options)
    : impl_(std::make_unique<Impl>(rank, std::move(files), std::move(layers), options)) {}
ExpertDelivery::~ExpertDelivery() = default;

ExpertDelivery::Call ExpertDelivery::begin(std::size_t index, const std::int32_t* ids,
                                          std::uint32_t pairs, cudaStream_t stream,
                                          std::span<const Input> inputs) {
    auto& p = *impl_;
    p.check();
    if (p.released) { throw std::logic_error("expert delivery: device backing is released"); }
    const auto& layer = p.layers.at(index);
    if (!ids || !pairs || pairs > p.options.pair_capacity || stream != p.rank.stream) {
        throw std::invalid_argument("expert delivery: invalid call or compute stream");
    }
    const auto low = reinterpret_cast<std::uintptr_t>(p.cache.p);
    const int experts = int(layer.source.experts.size());
    split_routes<<<1, 1024, std::size_t(experts) * 4, stream>>>(
        ids, pairs, experts, static_cast<const void* const*>(layer.tables.p), low,
        low + p.cache_span, static_cast<const std::uintptr_t*>(p.lent_ranges.p),
        static_cast<std::int32_t*>(p.cached_ids.p),
        static_cast<std::int32_t*>(p.missing_ids.p), static_cast<std::uint32_t*>(p.needed_device.p));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(p.needed_host->data(), p.needed_device.p, std::size_t(experts) * 4,
                               cudaMemcpyDeviceToHost, stream));
    Call call{static_cast<const std::int32_t*>(p.cached_ids.p),
              static_cast<const std::int32_t*>(p.missing_ids.p)};
    const bool cpu = p.host_flag && layer.cpu && !inputs.empty();
    if (cpu) {
        // The CPU's operands: the pairs' experts and the family's inputs.
        CUDA_CHECK(cudaMemcpyAsync(p.host_ids->data(), ids, std::size_t(pairs) * 4,
                                   cudaMemcpyDeviceToHost, stream));
        std::size_t at = 0;
        for (const auto& input : inputs) {
            if (input.bytes > p.options.cpu.input_bytes - std::min(at, p.options.cpu.input_bytes)) {
                throw std::invalid_argument("expert delivery: CPU inputs exceed their buffer");
            }
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(p.host_inputs->data()) + at, input.data,
                                       input.bytes, cudaMemcpyDeviceToHost, stream));
            at += input.bytes;
        }
        call.products = p.products.p;
        call.cpu = static_cast<const std::int32_t*>(p.cpu_flag.p);
    }
    p.handshake.publish(std::uint32_t(index), pairs, cpu ? kCpuCall : 0, stream);
    return call;
}
void ExpertDelivery::await(cudaStream_t stream) {
    impl_->handshake.wait_prepared(stream);
    impl_->check();
}
const void* const* ExpertDelivery::tables(std::size_t layer) const {
    return static_cast<const void* const*>(impl_->layers.at(layer).tables.p);
}
void ExpertDelivery::check() const { impl_->check(); }
void ExpertDelivery::drain() {
    impl_->quiesce();
    impl_->check();
}
void ExpertDelivery::invalidate() {
    auto& p = *impl_;
    if (p.released) { throw std::logic_error("expert delivery: restore backing before invalidation"); }
    drain();
    for (auto& layer : p.layers) {
        layer.tables.fill();
        std::memset(layer.mirror->data(), 0, layer.mirror->size());
        std::fill(layer.slot_of.begin(), layer.slot_of.end(), -1);
        layer.cached = 0;
        layer.dirty = false;
    }
    p.free_slots.clear();
    for (std::int64_t s = std::int64_t(p.slots.size()) - 1; s >= 0; --s) {
        auto& slot = p.slots[std::size_t(s)];
        slot.layer = slot.expert = -1;
        slot.pending = false;
        p.free_slots.push_back(std::int32_t(s));
    }
    for (auto& bank : p.bank_state) {
        bank.layer = -1;
        std::fill(bank.valid.begin(), bank.valid.end(), 0);
    }
    // No bank stays lent: the split reads an empty range for each.
    for (auto& layer : p.layers) {
        std::fill(layer.lent_of.begin(), layer.lent_of.end(), -1);
        std::fill(layer.bank_of.begin(), layer.bank_of.end(), std::int8_t(-1));
    }
    for (auto& slot : p.lent_slots) { slot.layer = slot.expert = -1; }
    std::fill(p.bank_lent.begin(), p.bank_lent.end(), 0);
    std::memset(p.host_ranges->data(), 0, p.host_ranges->size());
    p.lent_ranges.fill();
    p.ranges_dirty = false;
    p.quiet_passes = 0;
    p.set_lent_count();
    for (const auto& d : p.deferred) {
        if (d.landed && std::find(p.spare_events.begin(), p.spare_events.end(), d.landed) ==
                            p.spare_events.end()) {
            p.spare_events.push_back(d.landed);
        }
    }
    p.deferred.clear();
    p.queued.clear();
    for (std::size_t li = 0; li < p.layers.size(); ++li) {
        for (std::size_t e = 0; e < p.layers[li].source.experts.size(); ++e) {
            p.tier->set(li, std::int32_t(e), ExpertTier::State::Out);
        }
    }
    p.tier->set_free(0, std::uint32_t(p.free_slots.size()));
    p.previous_layer = -1;
    p.last_narrow = -1;
    if (p.host_flag) {
        *static_cast<std::int32_t*>(p.host_flag->data()) = 0;
        p.cpu_flag.fill();
    }
}
void ExpertDelivery::release_device_backing() {
    auto& p = *impl_;
    if (!p.options.suspendable) { throw std::logic_error("expert delivery: suspend was not enabled"); }
    if (p.released) { return; }
    drain();
    DeviceBinding bind(p.rank.device);
    p.each_storage([](Storage& storage) { storage.release_backing(); });
    p.released = true;
}
void ExpertDelivery::restore_device_backing() {
    auto& p = *impl_;
    if (!p.options.suspendable) { throw std::logic_error("expert delivery: suspend was not enabled"); }
    if (!p.released) { return; }
    DeviceBinding bind(p.rank.device);
    try {
        p.each_storage([](Storage& storage) { storage.restore_backing(); });
        p.each_storage([](Storage& storage) { storage.fill(); });
        for (auto& slot : p.slots) { std::fill(slot.written.begin(), slot.written.end(), 0); }
        p.released = false;
        invalidate();
    } catch (...) {
        p.each_storage([](Storage& storage) { storage.release_backing(); });
        p.released = true;
        throw;
    }
}
std::uint64_t ExpertDelivery::state_backing_bytes() const noexcept {
    const auto memory = this->memory();
    return memory.cache_bytes + memory.bank_bytes + memory.metadata_bytes;
}
ExpertDeliveryStats ExpertDelivery::stats() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->counters;
}
ExpertDeliveryMemory ExpertDelivery::memory() const noexcept {
    std::lock_guard lock(impl_->mutex);
    return impl_->allocation;
}

void ExpertDelivery::prime(const std::vector<std::vector<std::uint64_t>>* profile) {
    auto& p = *impl_;
    invalidate();
    p.tier->reset();
    {
        std::lock_guard lock(p.mutex);
        for (auto& layer : p.layers) { std::fill(layer.total.begin(), layer.total.end(), 0); }
    }
    p.tier->set_free(0, std::uint32_t(p.free_slots.size()));
    if (profile == nullptr) { return; }
    const auto& counts = *profile;
    if (counts.size() != p.layers.size()) {
        throw std::invalid_argument("expert delivery: the profile describes other layers");
    }
    // Most routed first over every layer; ties keep layer and expert order.
    std::vector<std::tuple<std::uint64_t, std::size_t, std::int32_t>> ranked;
    for (std::size_t li = 0; li < p.layers.size(); ++li) {
        if (counts[li].size() != p.layers[li].source.experts.size()) {
            throw std::invalid_argument("expert delivery: the profile's bank width differs");
        }
        for (std::size_t e = 0; e < counts[li].size(); ++e) {
            if (counts[li][e] != 0 && p.layers[li].slot_of[e] < 0) {
                ranked.emplace_back(counts[li][e], li, std::int32_t(e));
            }
        }
    }
    std::stable_sort(ranked.begin(), ranked.end(),
                     [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
    DeviceBinding bind(p.rank.device);
    // The service is idle before the first call; the lock orders these changes before it serves.
    std::lock_guard lock(p.mutex);
    ExpertDeliveryStats delta;
    for (const auto& [count, li, e] : ranked) {
        if (p.free_slots.empty()) { break; }
        const auto s = p.free_slots.back();
        p.free_slots.pop_back();
        p.occupy(s, li, e, false);
        for (std::size_t b = 0; b < p.layers[li].source.experts[std::size_t(e)].size(); ++b) {
            const auto& blob = p.layers[li].source.experts[std::size_t(e)][b];
            p.add_piece(p.slot_base(s) + p.slot_offset[b], blob);
            p.slot_written(s, b, blob.bytes);
        }
        p.point_slot(li, e, s);
        ++delta.admissions;
    }
    p.flush_pieces(delta);
    p.upload_tables();
    CUDA_CHECK(cudaStreamSynchronize(p.copies));
    p.tier->set_free(0, std::uint32_t(p.free_slots.size()));
    p.counters = p.counters + delta;
}

std::vector<std::vector<std::uint64_t>> ExpertDelivery::totals() const {
    std::lock_guard lock(impl_->mutex);
    std::vector<std::vector<std::uint64_t>> out;
    for (const auto& layer : impl_->layers) { out.push_back(layer.total); }
    return out;
}

std::uint64_t ExpertDelivery::shrink_cache(std::uint64_t bytes) {
    auto& p = *impl_;
    if (!p.options.elastic) { throw std::logic_error("expert delivery: the cache is not elastic"); }
    if (p.released) { throw std::logic_error("expert delivery: device backing is released"); }
    drain();
    DeviceBinding bind(p.rank.device);
    std::lock_guard lock(p.mutex);
    return p.shrink(bytes);
}

std::uint64_t ExpertDelivery::grow_cache() {
    auto& p = *impl_;
    if (!p.options.elastic) { throw std::logic_error("expert delivery: the cache is not elastic"); }
    if (p.released) { throw std::logic_error("expert delivery: device backing is released"); }
    drain();
    DeviceBinding bind(p.rank.device);
    std::lock_guard lock(p.mutex);
    return p.grow();
}

void ExpertDelivery::set_keep_free_bytes(std::uint64_t bytes) {
    std::lock_guard lock(impl_->mutex);
    impl_->options.keep_free_bytes = bytes;
}

std::string ExpertDelivery::cpu_description() const {
    if (!impl_->pool) { return {}; }
    return std::to_string(impl_->pool->threads()) +
           (impl_->pool->pinned() ? " pinned threads" : " threads");
}

} // namespace ninfer
