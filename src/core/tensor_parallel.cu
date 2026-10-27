#include "core/tensor_parallel.h"

#include <algorithm>
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

// Four values per thread where the residual allows it; the adds run in rank order either way.
__global__ void tensor_parallel_reduce_kernel(__nv_bfloat16* __restrict__ residual,
                                              PartialSet parts, std::int64_t values) {
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < values; i += stride) {
        float sum = parts.part[0][i];
        for (int r = 1; r < parts.count; ++r) { sum += parts.part[r][i]; }
        residual[i] = __float2bfloat16(__bfloat162float(residual[i]) + sum);
    }
}

} // namespace

namespace detail {

void tensor_parallel_reduce(__nv_bfloat16* residual, const float* const* parts, std::size_t count,
                            std::int64_t values, cudaStream_t stream) {
    if (count == 0 || count > kMaxRanks || values < 0) {
        throw std::invalid_argument("tensor_parallel_reduce: invalid rank or value count");
    }
    if (values == 0) { return; }
    PartialSet set{};
    for (std::size_t r = 0; r < count; ++r) { set.part[r] = parts[r]; }
    set.count             = static_cast<int>(count);
    constexpr int kBlock  = 256;
    const std::int64_t blocks = std::min<std::int64_t>((values + kBlock - 1) / kBlock, 4096);
    tensor_parallel_reduce_kernel<<<static_cast<unsigned>(blocks), kBlock, 0, stream>>>(
        residual, set, values);
    check(cudaGetLastError(), "reduce launch");
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
    if (transport_ == TensorParallelTransport::StagedHost && bytes != 0) {
        check(cudaMemcpyAsync(self.host_slot[slot] + offset, self.device_slot[slot] + offset,
                              bytes, cudaMemcpyDeviceToHost, stream),
              "publish copy");
    }
    check(cudaEventRecord(self.published[slot], stream), "publish fence");
}

void TensorParallelExchange::reduce(std::size_t rank, std::size_t slot, __nv_bfloat16* residual,
                                    std::int64_t count, std::size_t contributors,
                                    cudaStream_t stream, std::size_t offset) {
    check_slot(rank, slot, static_cast<std::size_t>(count) * sizeof(float),
               offset * sizeof(float));
    if (contributors == 0 || contributors > ranks_.size()) {
        throw std::invalid_argument("tensor-parallel reduce needs one to all ranks as contributors");
    }
    std::array<const float*, kMaxRanks> parts{};
    DeviceBinding bind(ranks_[rank].device);
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
    detail::tensor_parallel_reduce(residual, parts.data(), contributors, count, stream);
}

} // namespace ninfer
