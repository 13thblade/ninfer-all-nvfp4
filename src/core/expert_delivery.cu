#include "core/expert_delivery.h"

#include "core/arena.h"
#include "core/cuda_host_handshake.h"
#include "core/file_read_queue.h"
#include "core/vmm.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace ninfer {
namespace {

std::uint64_t aligned(std::uint64_t n) {
    if (n > std::numeric_limits<std::uint64_t>::max() - 255) {
        throw std::invalid_argument("expert delivery: byte extent overflow");
    }
    return (n + 255) / 256 * 256;
}

// Use the core's existing fixed-address allocation when the Program supports model suspend.
// The transport does not create a second virtual-memory or remapping mechanism.
struct Storage {
    DeviceBuffer owned;
    VmmRegion region;
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
        return region ? region.reserved_bytes() : bytes;
    }
};

// The mask is copied before publish(), on the same stream. Tables change only at missing
// entries or at victims absent from this call, so cached consumers can read them during DMA.
__global__ void split_routes(const std::int32_t* ids, std::uint32_t pairs, int experts,
                             const void* const* tables, std::uintptr_t low, std::uintptr_t high,
                             std::int32_t* cached, std::int32_t* missing, std::uint32_t* needed) {
    for (int e = threadIdx.x; e < experts; e += blockDim.x) { needed[e] = 0; }
    if (threadIdx.x < 2) { needed[experts + threadIdx.x] = 0; }
    __syncthreads();
    for (std::uint32_t p = threadIdx.x; p < pairs; p += blockDim.x) {
        const int e = ids[p];
        bool hit = false;
        if (e >= 0 && e < experts) {
            const auto address = reinterpret_cast<std::uintptr_t>(tables[e]);
            hit = address >= low && address < high;
            atomicOr(needed + e, hit ? 1u : 2u);
            atomicAdd(needed + experts, 1u);
            if (hit) { atomicAdd(needed + experts + 1, 1u); }
        }
        cached[p] = hit ? e : -1;
        missing[p] = !hit && e >= 0 && e < experts ? e : -1;
    }
}

} // namespace

struct ExpertDelivery::Impl {
    struct Layer {
        ExpertDeliveryLayer source;
        Storage tables;
        std::unique_ptr<PinnedHostBuffer> mirror;
        std::vector<int> slot_of;
    };
    struct Slot {
        int layer = -1, expert = -1;
        std::uint64_t stamp = 0;
    };
    RankContext rank;
    ExpertDeliveryOptions options;
    std::vector<Layer> layers;
    std::vector<std::uint64_t> offsets, sizes;
    std::uint64_t slot_bytes = 0;
    std::uint32_t max_experts = 0, active_capacity = 0;
    Storage cache, transient, cached_ids, missing_ids, needed_device;
    std::unique_ptr<PinnedHostBuffer> needed_host, staging;
    std::vector<Slot> slots;
    CudaHostHandshake handshake;
    cudaStream_t copies = nullptr;
    cudaEvent_t copied[2]{};
    std::uint64_t half_bytes = 0, stamp = 0;
    int half = 0;
    std::uint64_t staging_cursor = 0;
    std::vector<QueuedFileRead> batch_reads;
    struct StagedCopy { std::byte* target; const std::uint8_t* source; std::size_t bytes; };
    std::vector<StagedCopy> batch_copies;
    std::vector<std::ifstream> files;
    std::unique_ptr<FileReadQueue> read_queue;
    std::atomic<bool> stopping{false};
    std::thread service;
    mutable std::mutex mutex;
    std::exception_ptr failure;
    ExpertDeliveryStats counters;
    ExpertDeliveryMemory allocation;
    bool released = false;

    Impl(const RankContext& r, std::vector<std::filesystem::path> paths,
         std::vector<ExpertDeliveryLayer> specs, ExpertDeliveryOptions o)
        : rank(r), options(o), handshake(r) {
        DeviceBinding bind(rank.device);
        if (specs.empty() || options.pair_capacity == 0) {
            throw std::invalid_argument("expert delivery: layers and pair capacity are required");
        }
        std::uint64_t expert_count = 0;
        bool disk = false;
        for (const auto& layer : specs) {
            expert_count += layer.experts.size();
            if (layer.experts.empty() || layer.aliases.empty()) {
                throw std::invalid_argument("expert delivery: empty expert or alias table");
            }
            if (layer.experts.size() > 8192) {
                throw std::invalid_argument("expert delivery: at most 8192 experts per layer");
            }
            max_experts = std::max(max_experts, std::uint32_t(layer.experts.size()));
            for (const auto& expert : layer.experts) {
                if (expert.empty()) { throw std::invalid_argument("expert delivery: empty expert"); }
                sizes.resize(std::max(sizes.size(), expert.size()), 0);
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
                    sizes[b] = std::max(sizes[b], aligned(blob.bytes + blob.zero_tail));
                    disk |= !blob.extents.empty();
                }
                for (const auto& alias : layer.aliases) {
                    if (alias.blob >= expert.size() || alias.offset >= expert[alias.blob].bytes) {
                        throw std::invalid_argument("expert delivery: alias outside blob");
                    }
                }
            }
        }
        active_capacity = options.active_capacity ? options.active_capacity : max_experts;
        if (!active_capacity || active_capacity > max_experts) {
            throw std::invalid_argument("expert delivery: invalid active capacity");
        }
        for (auto size : sizes) { offsets.push_back(slot_bytes); slot_bytes += size; }
        if (slot_bytes > std::numeric_limits<std::size_t>::max() / active_capacity) {
            throw std::invalid_argument("expert delivery: transient capacity overflow");
        }
        transient = Storage(slot_bytes * active_capacity, rank.device, options.suspendable);
        transient.fill();
        allocation.transient_bytes = transient.backing_bytes();
        cached_ids = Storage(std::size_t(options.pair_capacity) * 4, rank.device, options.suspendable);
        missing_ids = Storage(std::size_t(options.pair_capacity) * 4, rank.device, options.suspendable);
        needed_device = Storage(std::size_t(max_experts + 2) * 4, rank.device, options.suspendable);
        needed_host = std::make_unique<PinnedHostBuffer>(std::size_t(max_experts + 2) * 4);
        std::memset(needed_host->data(), 0, needed_host->size());
        allocation.metadata_bytes = cached_ids.backing_bytes() + missing_ids.backing_bytes() +
                                     needed_device.backing_bytes();
        allocation.pinned_bytes = needed_host->size() + handshake.pinned_bytes();
        for (auto& spec : specs) {
            Layer layer;
            const auto count = spec.experts.size();
            layer.slot_of.assign(count, -1);
            const auto bytes = count * spec.aliases.size() * sizeof(void*);
            layer.tables = Storage(bytes, rank.device, options.suspendable);
            layer.tables.fill();
            layer.mirror = std::make_unique<PinnedHostBuffer>(bytes);
            std::memset(layer.mirror->data(), 0, bytes);
            layer.source = std::move(spec);
            allocation.metadata_bytes += layer.tables.backing_bytes();
            allocation.pinned_bytes += bytes;
            layers.push_back(std::move(layer));
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
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        const std::uint64_t available = free > options.keep_free_bytes
                                            ? free - options.keep_free_bytes : 0;
        const std::uint64_t granule = options.suspendable ? vmm::granularity(rank.device) : 1;
        if (!granule) {
            throw std::invalid_argument("expert delivery: suspend needs VMM allocation granularity");
        }
        // Automatic slots may use only whole physical granules inside the remaining budget.
        // An explicit request must preserve the same reserve, including VMM rounding.
        const auto budget = options.cache_bytes.value_or(available / granule * granule);
        const auto count = std::min<std::uint64_t>(budget / slot_bytes, expert_count);
        const auto cache_bytes = count * slot_bytes;
        if (cache_bytes > std::numeric_limits<std::uint64_t>::max() - (granule - 1)) {
            throw std::invalid_argument("expert delivery: cache backing size overflow");
        }
        const auto backing_bytes = (cache_bytes + granule - 1) / granule * granule;
        if (backing_bytes > available) {
            throw std::invalid_argument(
                "expert delivery: cache needs " + std::to_string(backing_bytes) +
                " device bytes, but only " + std::to_string(available) +
                " remain after keep_free_bytes=" + std::to_string(options.keep_free_bytes));
        }
        if (count) {
            cache = Storage(cache_bytes, rank.device, options.suspendable);
            cache.fill();
            slots.resize(count);
        }
        allocation.cache_bytes = cache.backing_bytes();
        allocation.cache_slots = std::uint32_t(slots.size());
        CUDA_CHECK(cudaStreamCreateWithFlags(&copies, cudaStreamNonBlocking));
        try {
            for (auto& event : copied) { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
            service = std::thread([this] { run(); });
        } catch (...) {
            for (auto event : copied) { if (event) { cudaEventDestroy(event); } }
            cudaStreamDestroy(copies);
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
        for (auto event : copied) { cudaEventDestroy(event); }
        cudaStreamDestroy(copies);
    }

    void check() const {
        std::lock_guard lock(mutex);
        if (failure) { std::rethrow_exception(failure); }
    }

    template <typename Function> void each_storage(Function&& function) {
        for (Storage* storage : {&cache, &transient, &cached_ids, &missing_ids, &needed_device}) {
            if (storage->bytes) { function(*storage); }
        }
        for (auto& layer : layers) { function(layer.tables); }
    }

    void patch(std::size_t li, int e, std::byte* base) {
        auto& layer = layers[li];
        const auto n = layer.source.experts.size();
        auto** mirror = static_cast<const void**>(layer.mirror->data());
        for (std::size_t a = 0; a < layer.source.aliases.size(); ++a) {
            const auto alias = layer.source.aliases[a];
            const auto index = a * n + std::size_t(e);
            mirror[index] = base ? base + offsets[alias.blob] + alias.offset : nullptr;
            CUDA_CHECK(cudaMemcpyAsync(static_cast<const void**>(layer.tables.p) + index,
                                       mirror + index, sizeof(void*), cudaMemcpyHostToDevice, copies));
        }
    }

    // Several expert reads share one bounded half. FileReadQueue can submit their extents
    // together. Its fallback reads the same ranges through the OS page cache.
    void flush_reads() {
        if (batch_reads.empty()) { return; }
        CUDA_CHECK(cudaEventSynchronize(copied[half]));
        if (read_queue) { read_queue->read(batch_reads); }
        else {
            for (auto& read : batch_reads) {
                auto& file = files.at(read.file);
                file.clear();
                file.seekg(std::streamoff(read.offset));
                file.read(reinterpret_cast<char*>(read.destination.data()),
                          std::streamsize(read.destination.size()));
                if (!file) { throw std::runtime_error("expert delivery: incomplete file read"); }
            }
        }
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

    void upload_disk(const ExpertBlob& blob, std::byte* target, ExpertDeliveryStats& delta) {
        std::uint64_t destination = 0;
        for (const auto& extent : blob.extents) {
            std::uint64_t within = 0;
            while (within < extent.bytes) {
                if (staging_cursor == half_bytes) { flush_reads(); }
                auto* buffer = static_cast<std::uint8_t*>(staging->data()) +
                               half * half_bytes + staging_cursor;
                const auto bytes = std::min(extent.bytes - within, half_bytes - staging_cursor);
                batch_reads.push_back({extent.file, extent.offset + within,
                                       {buffer, std::size_t(bytes)}});
                batch_copies.push_back({target + destination, buffer, std::size_t(bytes)});
                staging_cursor += bytes;
                destination += bytes;
                within += bytes;
                delta.read_bytes += bytes;
            }
        }
    }

    void serve(const CudaHostHandshake::Request& request) {
        const auto started = std::chrono::steady_clock::now();
        auto& layer = layers.at(request.tag);
        const auto* needed = static_cast<const std::uint32_t*>(needed_host->data());
        const int experts = int(layer.source.experts.size());
        ExpertDeliveryStats delta;
        delta.requests = 1;
        delta.pairs = needed[experts];
        delta.pair_hits = needed[experts + 1];
        for (int e = 0; e < experts; ++e) {
            delta.routes += needed[e] != 0;
            delta.hits += needed[e] == 1;
            if (needed[e] && layer.slot_of[e] >= 0) {
                slots[layer.slot_of[e]].stamp = ++stamp;
            }
        }
        if (delta.routes > active_capacity) {
            throw std::runtime_error("expert delivery: call exceeds distinct expert capacity");
        }
        std::uint32_t transient_next = 0;
        for (int e = 0; e < experts; ++e) {
            if (needed[e] != 2) { continue; }
            ++delta.misses;
            int chosen = -1;
            for (std::size_t s = 0; s < slots.size(); ++s) {
                const auto& slot = slots[s];
                if (slot.layer == int(request.tag) && needed[slot.expert]) { continue; }
                if (chosen < 0 || slot.stamp < slots[chosen].stamp) { chosen = int(s); }
                if (slot.layer < 0) { break; }
            }
            std::byte* target = nullptr;
            if (chosen >= 0) {
                auto& slot = slots[chosen];
                if (slot.layer >= 0) {
                    layers[slot.layer].slot_of[slot.expert] = -1;
                    patch(slot.layer, slot.expert, nullptr);
                    ++delta.evictions;
                }
                slot = {int(request.tag), e, ++stamp};
                layer.slot_of[e] = chosen;
                ++delta.admissions;
                target = static_cast<std::byte*>(cache.p) + std::uint64_t(chosen) * slot_bytes;
            } else {
                target = static_cast<std::byte*>(transient.p) + transient_next++ * slot_bytes;
            }
            const auto& expert = layer.source.experts[e];
            for (std::size_t b = 0; b < expert.size(); ++b) {
                const auto& blob = expert[b];
                auto* out = target + offsets[b];
                if (blob.pinned) {
                    CUDA_CHECK(cudaMemcpyAsync(out, blob.pinned, blob.bytes,
                                               cudaMemcpyHostToDevice, copies));
                } else { upload_disk(blob, out, delta); }
                // A larger expert or a different layer may have occupied this slot.
                CUDA_CHECK(cudaMemsetAsync(out + blob.bytes, 0, sizes[b] - blob.bytes, copies));
                delta.copied_bytes += blob.bytes;
            }
            patch(request.tag, e, target);
        }
        flush_reads();
        CUDA_CHECK(cudaStreamSynchronize(copies));
        delta.service_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        std::lock_guard lock(mutex);
        counters.requests += delta.requests;
        counters.routes += delta.routes;
        counters.hits += delta.hits;
        counters.misses += delta.misses;
        counters.evictions += delta.evictions;
        counters.pairs += delta.pairs;
        counters.pair_hits += delta.pair_hits;
        counters.admissions += delta.admissions;
        counters.read_bytes += delta.read_bytes;
        counters.copied_bytes += delta.copied_bytes;
        counters.service_seconds += delta.service_seconds;
    }

    void run() {
        DeviceBinding bind(rank.device);
        std::uint32_t served = 0;
        auto active_until = std::chrono::steady_clock::now();
        while (!stopping.load(std::memory_order_acquire)) {
            const auto request = handshake.pending(served);
            if (!request) {
                if (std::chrono::steady_clock::now() < active_until) { std::this_thread::yield(); }
                else { std::this_thread::sleep_for(std::chrono::microseconds(20)); }
                continue;
            }
            try { check(); serve(*request); }
            catch (...) {
                {
                    std::lock_guard lock(mutex);
                    if (!failure) { failure = std::current_exception(); }
                }
                batch_reads.clear();
                batch_copies.clear();
                staging_cursor = 0;
                // Captured decode can finish safely; the caller checks before publishing output.
                cudaMemsetAsync(missing_ids.p, 0xff, std::size_t(request->width) * 4, copies);
                cudaStreamSynchronize(copies);
            }
            handshake.prepared(request->sequence);
            served = request->sequence;
            active_until = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        }
    }
};

ExpertDelivery::ExpertDelivery(const RankContext& rank, std::vector<std::filesystem::path> files,
                               std::vector<ExpertDeliveryLayer> layers, ExpertDeliveryOptions options)
    : impl_(std::make_unique<Impl>(rank, std::move(files), std::move(layers), options)) {}
ExpertDelivery::~ExpertDelivery() = default;

ExpertDelivery::Call ExpertDelivery::begin(std::size_t index, const std::int32_t* ids,
                                          std::uint32_t pairs, cudaStream_t stream) {
    auto& p = *impl_;
    p.check();
    if (p.released) { throw std::logic_error("expert delivery: device backing is released"); }
    const auto& layer = p.layers.at(index);
    if (!ids || !pairs || pairs > p.options.pair_capacity || stream != p.rank.stream) {
        throw std::invalid_argument("expert delivery: invalid call or compute stream");
    }
    const auto low = reinterpret_cast<std::uintptr_t>(p.cache.p);
    split_routes<<<1, 256, 0, stream>>>(ids, pairs, int(layer.source.experts.size()),
        static_cast<const void* const*>(layer.tables.p), low, low + p.cache.bytes,
        static_cast<std::int32_t*>(p.cached_ids.p), static_cast<std::int32_t*>(p.missing_ids.p),
        static_cast<std::uint32_t*>(p.needed_device.p));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(p.needed_host->data(), p.needed_device.p,
        (layer.source.experts.size() + 2) * 4, cudaMemcpyDeviceToHost, stream));
    p.handshake.publish(std::uint32_t(index), pairs, 0, stream);
    return {static_cast<const std::int32_t*>(p.cached_ids.p),
            static_cast<const std::int32_t*>(p.missing_ids.p)};
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
    DeviceBinding bind(impl_->rank.device);
    CUDA_CHECK(cudaStreamSynchronize(impl_->rank.stream));
    CUDA_CHECK(cudaStreamSynchronize(impl_->copies));
    impl_->check();
}
void ExpertDelivery::invalidate() {
    if (impl_->released) { throw std::logic_error("expert delivery: restore backing before invalidation"); }
    drain();
    for (auto& layer : impl_->layers) {
        layer.tables.fill();
        std::memset(layer.mirror->data(), 0, layer.mirror->size());
        std::fill(layer.slot_of.begin(), layer.slot_of.end(), -1);
    }
    std::fill(impl_->slots.begin(), impl_->slots.end(), Impl::Slot{});
}
void ExpertDelivery::release_device_backing() {
    auto& p = *impl_;
    if (!p.options.suspendable) { throw std::logic_error("expert delivery: suspend was not enabled"); }
    if (p.released) { return; }
    drain();
    DeviceBinding bind(p.rank.device);
    p.each_storage([](Storage& storage) { storage.region.release_backing(); });
    p.released = true;
}
void ExpertDelivery::restore_device_backing() {
    auto& p = *impl_;
    if (!p.options.suspendable) { throw std::logic_error("expert delivery: suspend was not enabled"); }
    if (!p.released) { return; }
    DeviceBinding bind(p.rank.device);
    try {
        p.each_storage([](Storage& storage) { storage.region.restore_backing(); });
        p.each_storage([](Storage& storage) { storage.fill(); });
        p.released = false;
        invalidate();
    } catch (...) {
        p.each_storage([](Storage& storage) { storage.region.release_backing(); });
        p.released = true;
        throw;
    }
}
std::uint64_t ExpertDelivery::state_backing_bytes() const noexcept {
    const auto memory = impl_->allocation;
    return memory.cache_bytes + memory.transient_bytes + memory.metadata_bytes;
}
ExpertDeliveryStats ExpertDelivery::stats() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->counters;
}
ExpertDeliveryMemory ExpertDelivery::memory() const noexcept { return impl_->allocation; }

} // namespace ninfer
