#include "core/tensor_parallel.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

void check(cudaError_t err, const char* what) {
    if (err == cudaSuccess) { return; }
    throw std::runtime_error(std::string("TensorParallelExchange ") + what + ": " +
                             cudaGetErrorName(err) + ": " + cudaGetErrorString(err));
}

struct PartialSet {
    const float* part[kMaxRanks];
    int count;
};

// One value's sum into the output, as TensorParallelSum says.
template <TensorParallelSum Sum>
__device__ __forceinline__ void store_sum(void* out, std::int64_t i, float sum) {
    if constexpr (Sum == TensorParallelSum::AddBf16) {
        auto* residual = static_cast<__nv_bfloat16*>(out);
        residual[i]    = __float2bfloat16(__bfloat162float(residual[i]) + sum);
    } else if constexpr (Sum == TensorParallelSum::StoreBf16) {
        static_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16(sum);
    } else {
        static_cast<float*>(out)[i] = sum;
    }
}

// The adds run in rank order.
template <TensorParallelSum Sum>
__global__ void tensor_parallel_reduce_kernel(void* __restrict__ out, PartialSet parts,
                                              std::int64_t values) {
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < values; i += stride) {
        float sum = parts.part[0][i];
        for (int r = 1; r < parts.count; ++r) { sum += parts.part[r][i]; }
        store_sum<Sum>(out, i, sum);
    }
}

struct LowLatencyArgs {
    void* out;
    const float* mine;
    const unsigned long long* rows[kMaxRanks];
    std::uint32_t* epoch;
    std::int64_t values;
    std::int64_t capacity;
    int ranks;
    int rank;
    int contributors;
};

__device__ __forceinline__ unsigned long long low_latency_clock() {
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}

// The value of a word once it carries `epoch`.
__device__ __forceinline__ std::uint32_t await_word(const unsigned long long* word,
                                                    std::uint32_t epoch,
                                                    unsigned long long begun) {
    unsigned long long seen = *reinterpret_cast<const volatile unsigned long long*>(word);
    for (std::uint32_t spins = 0; static_cast<std::uint32_t>(seen >> 32U) != epoch;) {
        if ((++spins & 1023U) == 0 &&
            low_latency_clock() - begun > TensorParallelExchange::kLowLatencyTimeoutNs) {
            printf("ninfer: a tensor-parallel exchange waited %llu s for epoch %u\n",
                   TensorParallelExchange::kLowLatencyTimeoutNs / 1'000'000'000ULL, epoch);
            __trap();
        }
        seen = *reinterpret_cast<const volatile unsigned long long*>(word);
    }
    return static_cast<std::uint32_t>(seen);
}

template <TensorParallelSum Sum>
__global__ void tensor_parallel_low_latency_kernel(LowLatencyArgs a) {
    const std::uint32_t epoch = *reinterpret_cast<const volatile std::uint32_t*>(a.epoch) + 1U;
    const std::int64_t base   = static_cast<std::int64_t>(epoch & 1U) * a.capacity;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t first  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool contributes    = a.rank < a.contributors;
    auto* own = const_cast<volatile unsigned long long*>(a.rows[a.rank]) + base;
    const unsigned long long tag = static_cast<unsigned long long>(epoch) << 32U;
    if (contributes) {
        for (std::int64_t i = first; i < a.values; i += stride) {
            own[i] = tag | __float_as_uint(a.mine[i]);
        }
    } else if (first == 0) {
        own[0] = tag;
    }
    const unsigned long long begun = low_latency_clock();
    for (std::int64_t i = first; i < a.values; i += stride) {
        float sum = 0.0F;
        for (int r = 0; r < a.contributors; ++r) {
            const float part = r == a.rank
                                   ? a.mine[i]
                                   : __uint_as_float(await_word(a.rows[r] + base + i, epoch, begun));
            sum = r == 0 ? part : sum + part;
        }
        store_sum<Sum>(a.out, i, sum);
    }
    if (first == 0) {
        for (int r = a.contributors; r < a.ranks; ++r) {
            if (r != a.rank) { (void)await_word(a.rows[r] + base, epoch, begun); }
        }
    }
    // Every block read the count before it finished, so the last one to finish advances it.
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(a.epoch + 1, 1U) + 1U == gridDim.x) {
            a.epoch[1] = 0;
            a.epoch[0] = epoch;
        }
    }
}

} // namespace

namespace detail {

void tensor_parallel_reduce(void* out, const float* const* parts, std::size_t count,
                            std::int64_t values, cudaStream_t stream, TensorParallelSum sum) {
    if (count == 0 || count > kMaxRanks || values < 0) {
        throw std::invalid_argument("tensor_parallel_reduce: invalid rank or value count");
    }
    if (values == 0) { return; }
    PartialSet set{};
    for (std::size_t r = 0; r < count; ++r) { set.part[r] = parts[r]; }
    set.count             = static_cast<int>(count);
    constexpr int kBlock  = 256;
    const std::int64_t blocks = std::min<std::int64_t>((values + kBlock - 1) / kBlock, 4096);
    const auto grid = static_cast<unsigned>(blocks);
    switch (sum) {
    case TensorParallelSum::AddBf16:
        tensor_parallel_reduce_kernel<TensorParallelSum::AddBf16>
            <<<grid, kBlock, 0, stream>>>(out, set, values);
        break;
    case TensorParallelSum::StoreBf16:
        tensor_parallel_reduce_kernel<TensorParallelSum::StoreBf16>
            <<<grid, kBlock, 0, stream>>>(out, set, values);
        break;
    case TensorParallelSum::StoreFp32:
        tensor_parallel_reduce_kernel<TensorParallelSum::StoreFp32>
            <<<grid, kBlock, 0, stream>>>(out, set, values);
        break;
    }
    check(cudaGetLastError(), "reduce launch");
}

void tensor_parallel_low_latency(void* out, const float* mine,
                                 const unsigned long long* const* rows, std::size_t ranks,
                                 std::size_t rank, std::size_t contributors, std::int64_t values,
                                 std::size_t capacity, std::uint32_t* epoch, cudaStream_t stream,
                                 TensorParallelSum sum) {
    if (ranks < 2 || ranks > kMaxRanks || rank >= ranks || contributors == 0 ||
        contributors > ranks || values <= 0 || static_cast<std::size_t>(values) > capacity) {
        throw std::invalid_argument("tensor_parallel_low_latency: invalid ranks or value count");
    }
    LowLatencyArgs args{};
    args.out      = out;
    args.mine     = mine;
    for (std::size_t r = 0; r < ranks; ++r) { args.rows[r] = rows[r]; }
    args.epoch        = epoch;
    args.values       = values;
    args.capacity     = static_cast<std::int64_t>(capacity);
    args.ranks        = static_cast<int>(ranks);
    args.rank         = static_cast<int>(rank);
    args.contributors = static_cast<int>(contributors);
    // About two values per thread: a poll is a bus round trip, so the waits overlap across blocks.
    constexpr int kBlock = 256;
    const std::int64_t blocks = std::clamp<std::int64_t>(values / (2 * kBlock), 1, 64);
    const auto grid = static_cast<unsigned>(blocks);
    switch (sum) {
    case TensorParallelSum::AddBf16:
        tensor_parallel_low_latency_kernel<TensorParallelSum::AddBf16>
            <<<grid, kBlock, 0, stream>>>(args);
        break;
    case TensorParallelSum::StoreBf16:
        tensor_parallel_low_latency_kernel<TensorParallelSum::StoreBf16>
            <<<grid, kBlock, 0, stream>>>(args);
        break;
    case TensorParallelSum::StoreFp32:
        tensor_parallel_low_latency_kernel<TensorParallelSum::StoreFp32>
            <<<grid, kBlock, 0, stream>>>(args);
        break;
    }
    check(cudaGetLastError(), "low-latency exchange launch");
}

} // namespace detail

std::vector<std::uint64_t> rank_free_bytes(const DeviceContext& context) {
    std::vector<std::uint64_t> out(context.size(), 0);
    for (std::size_t rank = 0; rank < context.size(); ++rank) {
        std::size_t sharing = 0;
        for (std::size_t other = 0; other < context.size(); ++other) {
            if (context.same_physical_device(rank, other)) { ++sharing; }
        }
        DeviceBinding bind(context.rank(rank).device);
        std::size_t free = 0, total = 0;
        check(cudaMemGetInfo(&free, &total), "free device memory");
        out[rank] = free / sharing;
    }
    return out;
}

std::vector<double> rank_copy_bandwidth(const DeviceContext& context) {
    // Large enough to stream past the L2 of every supported card, small enough to fit beside a
    // CUDA context on any of them.
    constexpr std::size_t kBytes  = 128ULL << 20;
    constexpr int kTimedCopies    = 3;
    std::vector<double> out(context.size(), 0.0);
    for (std::size_t rank = 0; rank < context.size(); ++rank) {
        std::size_t first = rank;
        for (std::size_t other = 0; other < rank; ++other) {
            if (context.same_physical_device(rank, other)) {
                first = other;
                break;
            }
        }
        if (first != rank) {
            out[rank] = out[first];
            continue;
        }
        DeviceBinding bind(context.rank(rank).device);
        void* source      = nullptr;
        void* destination = nullptr;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop  = nullptr;
        cudaStream_t stream = nullptr;
        const auto release = [&] {
            if (stream != nullptr) { (void)cudaStreamDestroy(stream); }
            if (start != nullptr) { (void)cudaEventDestroy(start); }
            if (stop != nullptr) { (void)cudaEventDestroy(stop); }
            if (source != nullptr) { (void)cudaFree(source); }
            if (destination != nullptr) { (void)cudaFree(destination); }
        };
        try {
            check(cudaMalloc(&source, kBytes), "bandwidth cudaMalloc");
            check(cudaMalloc(&destination, kBytes), "bandwidth cudaMalloc");
            check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "bandwidth stream");
            check(cudaEventCreate(&start), "bandwidth event");
            check(cudaEventCreate(&stop), "bandwidth event");
            check(cudaMemsetAsync(source, 0, kBytes, stream), "bandwidth memset");
            check(cudaMemcpyAsync(destination, source, kBytes, cudaMemcpyDeviceToDevice, stream),
                  "bandwidth warm-up");
            check(cudaEventRecord(start, stream), "bandwidth record");
            for (int copy = 0; copy < kTimedCopies; ++copy) {
                check(cudaMemcpyAsync(destination, source, kBytes, cudaMemcpyDeviceToDevice, stream),
                      "bandwidth copy");
            }
            check(cudaEventRecord(stop, stream), "bandwidth record");
            check(cudaEventSynchronize(stop), "bandwidth synchronize");
            float milliseconds = 0.0F;
            check(cudaEventElapsedTime(&milliseconds, start, stop), "bandwidth elapsed");
            // A copy reads and writes every byte.
            const double moved = 2.0 * static_cast<double>(kBytes) * kTimedCopies;
            out[rank] = moved / (std::max(static_cast<double>(milliseconds), 1e-3) * 1e-3);
        } catch (...) {
            release();
            throw;
        }
        release();
    }
    for (std::size_t rank = 0; rank < context.size(); ++rank) {
        std::size_t sharing = 0;
        for (std::size_t other = 0; other < context.size(); ++other) {
            if (context.same_physical_device(rank, other)) { ++sharing; }
        }
        out[rank] /= static_cast<double>(sharing);
    }
    return out;
}

TensorParallelExchange::TensorParallelExchange(const DeviceContext& context,
                                               TensorParallelExchangeOptions options)
    : slot_bytes_(options.slot_bytes) {
    if (context.size() < 2) {
        throw std::invalid_argument("a tensor-parallel exchange needs at least two ranks");
    }
    if (slot_bytes_ == 0 || slot_bytes_ % sizeof(float) != 0) {
        throw std::invalid_argument("a tensor-parallel slot holds a positive number of floats");
    }
    bool peer = !options.force_staged;
    for (std::size_t a = 0; a < context.size() && peer; ++a) {
        for (std::size_t b = 0; b < context.size(); ++b) {
            if (a != b && !context.same_physical_device(a, b) && !context.peer_access(b, a)) {
                peer = false;
                break;
            }
        }
    }
    transport_ = peer ? TensorParallelTransport::Peer : TensorParallelTransport::StagedHost;
    if (transport_ == TensorParallelTransport::StagedHost && context.size() <= kMaxRanks) {
        low_latency_bytes_ = std::min(options.low_latency_bytes, slot_bytes_) / sizeof(float) *
                             sizeof(float);
        for (std::size_t a = 0; a < context.size(); ++a) {
            for (std::size_t b = a + 1; b < context.size(); ++b) {
                if (context.same_physical_device(a, b)) { low_latency_bytes_ = 0; }
            }
        }
    }
    ranks_.resize(context.size());
    try {
        for (std::size_t r = 0; r < ranks_.size(); ++r) {
            Rank& rank  = ranks_[r];
            rank.device = context.rank(r).device;
            DeviceBinding bind(rank.device);
            for (std::size_t slot = 0; slot < kSlots; ++slot) {
                check(cudaMalloc(reinterpret_cast<void**>(&rank.device_slot[slot]), slot_bytes_),
                      "cudaMalloc");
                if (transport_ == TensorParallelTransport::StagedHost) {
                    // Mapped, so a reduce on any rank reads it in place; portable, so every
                    // rank's context treats it as pinned.
                    check(cudaHostAlloc(reinterpret_cast<void**>(&rank.host_slot[slot]),
                                        slot_bytes_, cudaHostAllocMapped | cudaHostAllocPortable),
                          "cudaHostAlloc");
                }
                // Events cannot be created while a stream captures, so all exist up front. A
                // rank records its own events.
                check(cudaEventCreateWithFlags(&rank.published[slot], cudaEventDisableTiming),
                      "cudaEventCreate");
            }
            if (low_latency_bytes_ != 0) {
                const std::size_t words = kSlots * (low_latency_bytes_ / sizeof(float));
                check(cudaHostAlloc(reinterpret_cast<void**>(&rank.words),
                                    words * sizeof(unsigned long long),
                                    cudaHostAllocMapped | cudaHostAllocPortable),
                      "cudaHostAlloc");
                std::fill_n(rank.words, words, 0ULL);
                check(cudaMalloc(reinterpret_cast<void**>(&rank.epoch), 2 * sizeof(std::uint32_t)),
                      "cudaMalloc");
                check(cudaMemset(rank.epoch, 0, 2 * sizeof(std::uint32_t)), "cudaMemset");
                // Loaded now, so no first launch waits on module loading while a peer spins.
                cudaFuncAttributes attributes{};
                for (const auto kernel :
                     {tensor_parallel_low_latency_kernel<TensorParallelSum::AddBf16>,
                      tensor_parallel_low_latency_kernel<TensorParallelSum::StoreBf16>,
                      tensor_parallel_low_latency_kernel<TensorParallelSum::StoreFp32>}) {
                    check(cudaFuncGetAttributes(&attributes, kernel), "low-latency kernel");
                }
            }
            check(cudaStreamCreateWithFlags(&rank.side, cudaStreamNonBlocking), "side stream");
            for (std::size_t piece = 0; piece < kPieces; ++piece) {
                check(cudaEventCreateWithFlags(&rank.ready[piece], cudaEventDisableTiming),
                      "cudaEventCreate");
                check(cudaEventCreateWithFlags(&rank.done[piece], cudaEventDisableTiming),
                      "cudaEventCreate");
            }
        }
    } catch (...) {
        release();
        throw;
    }
}

TensorParallelExchange::~TensorParallelExchange() { release(); }

void TensorParallelExchange::release() noexcept {
    for (Rank& rank : ranks_) {
        DeviceBinding bind(rank.device);
        for (std::size_t slot = 0; slot < kSlots; ++slot) {
            if (rank.published[slot] != nullptr) { (void)cudaEventDestroy(rank.published[slot]); }
            if (rank.device_slot[slot] != nullptr) { (void)cudaFree(rank.device_slot[slot]); }
            if (rank.host_slot[slot] != nullptr) { (void)cudaFreeHost(rank.host_slot[slot]); }
            rank.published[slot]   = nullptr;
            rank.device_slot[slot] = nullptr;
            rank.host_slot[slot]   = nullptr;
        }
        for (std::size_t piece = 0; piece < kPieces; ++piece) {
            if (rank.ready[piece] != nullptr) { (void)cudaEventDestroy(rank.ready[piece]); }
            if (rank.done[piece] != nullptr) { (void)cudaEventDestroy(rank.done[piece]); }
        }
        if (rank.side != nullptr) { (void)cudaStreamDestroy(rank.side); }
        if (rank.words != nullptr) { (void)cudaFreeHost(rank.words); }
        if (rank.epoch != nullptr) { (void)cudaFree(rank.epoch); }
    }
    ranks_.clear();
}

void TensorParallelExchange::check_slot(std::size_t rank, std::size_t slot, std::size_t bytes,
                                        std::size_t offset_bytes) const {
    if (rank >= ranks_.size() || slot >= kSlots) {
        throw std::out_of_range("tensor-parallel rank or slot is out of range");
    }
    if (offset_bytes > slot_bytes_ || bytes > slot_bytes_ - offset_bytes) {
        throw std::invalid_argument("tensor-parallel partial exceeds its slot");
    }
}

void TensorParallelExchange::check_piece(std::size_t rank, std::size_t piece) const {
    if (rank >= ranks_.size() || piece >= kPieces) {
        throw std::out_of_range("tensor-parallel rank or piece is out of range");
    }
}

cudaStream_t TensorParallelExchange::side_stream(std::size_t rank) const {
    check_piece(rank, 0);
    return ranks_[rank].side;
}

cudaEvent_t TensorParallelExchange::piece_ready(std::size_t rank, std::size_t piece) const {
    check_piece(rank, piece);
    return ranks_[rank].ready[piece];
}

cudaEvent_t TensorParallelExchange::piece_done(std::size_t rank, std::size_t piece) const {
    check_piece(rank, piece);
    return ranks_[rank].done[piece];
}

float* TensorParallelExchange::partial(std::size_t rank, std::size_t slot) const {
    check_slot(rank, slot, 0);
    return ranks_[rank].device_slot[slot];
}

void TensorParallelExchange::publish(std::size_t rank, std::size_t slot, std::size_t bytes,
                                     cudaStream_t stream, std::size_t offset) {
    check_slot(rank, slot, bytes, offset * sizeof(float));
    Rank& self = ranks_[rank];
    DeviceBinding bind(self.device);
    // Low-latency words leave the slot where it is: the reduce's kernel stores them itself.
    if (transport_ == TensorParallelTransport::StagedHost && bytes != 0 &&
        !(offset == 0 && low_latency(bytes))) {
        check(cudaMemcpyAsync(self.host_slot[slot] + offset, self.device_slot[slot] + offset,
                              bytes, cudaMemcpyDeviceToHost, stream),
              "publish copy");
    }
    check(cudaEventRecord(self.published[slot], stream), "publish fence");
}

void TensorParallelExchange::publish_whole(std::size_t rank, std::size_t slot, std::size_t bytes,
                                           cudaStream_t stream) {
    check_slot(rank, slot, bytes);
    Rank& self = ranks_[rank];
    DeviceBinding bind(self.device);
    if (transport_ == TensorParallelTransport::StagedHost && bytes != 0) {
        check(cudaMemcpyAsync(self.host_slot[slot], self.device_slot[slot], bytes,
                              cudaMemcpyDeviceToHost, stream),
              "publish copy");
    }
    check(cudaEventRecord(self.published[slot], stream), "publish fence");
}

void TensorParallelExchange::gather(std::size_t rank, std::size_t slot,
                                    std::span<const float*> parts, cudaStream_t stream) {
    check_slot(rank, slot, 0);
    if (parts.size() != ranks_.size()) {
        throw std::invalid_argument("tensor-parallel gather needs one pointer per rank");
    }
    DeviceBinding bind(ranks_[rank].device);
    for (std::size_t r = 0; r < ranks_.size(); ++r) {
        if (r == rank) {
            parts[r] = ranks_[r].device_slot[slot];
            continue;
        }
        check(cudaStreamWaitEvent(stream, ranks_[r].published[slot], 0), "gather wait");
        parts[r] = transport_ == TensorParallelTransport::Peer ? ranks_[r].device_slot[slot]
                                                              : ranks_[r].host_slot[slot];
    }
}

void TensorParallelExchange::reduce(std::size_t rank, std::size_t slot, void* out,
                                    std::int64_t count, std::size_t contributors,
                                    cudaStream_t stream, std::size_t offset,
                                    TensorParallelSum sum) {
    check_slot(rank, slot, static_cast<std::size_t>(count) * sizeof(float),
               offset * sizeof(float));
    if (contributors == 0 || contributors > ranks_.size()) {
        throw std::invalid_argument("tensor-parallel reduce needs one to all ranks as contributors");
    }
    DeviceBinding bind(ranks_[rank].device);
    if (offset == 0 && low_latency(static_cast<std::size_t>(count) * sizeof(float))) {
        std::array<const unsigned long long*, kMaxRanks> rows{};
        for (std::size_t r = 0; r < ranks_.size(); ++r) { rows[r] = ranks_[r].words; }
        detail::tensor_parallel_low_latency(out, ranks_[rank].device_slot[slot], rows.data(),
                                            ranks_.size(), rank, contributors, count,
                                            low_latency_bytes_ / sizeof(float),
                                            ranks_[rank].epoch, stream, sum);
        return;
    }
    std::array<const float*, kMaxRanks> parts{};
    for (std::size_t r = 0; r < ranks_.size(); ++r) {
        if (r == rank) {
            parts[r] = ranks_[r].device_slot[slot] + offset;
            continue;
        }
        check(cudaStreamWaitEvent(stream, ranks_[r].published[slot], 0), "reduce wait");
        parts[r] = (transport_ == TensorParallelTransport::Peer ? ranks_[r].device_slot[slot]
                                                                : ranks_[r].host_slot[slot]) +
                   offset;
    }
    detail::tensor_parallel_reduce(out, parts.data(), contributors, count, stream, sum);
}

} // namespace ninfer
