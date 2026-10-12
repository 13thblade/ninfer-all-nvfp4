// Adapted from Infernix a3edb450 src/core/copy_batch.cpp (Apache-2.0).
// Modified for NInfer-3090: NInfer's namespace; CUDA 12.8's cudaMemcpyBatchAsync form, with one-by-one copies when the driver refuses it.

#include "core/copy_batch.h"

#include "core/device.h"

#include <iterator>

namespace ninfer {

void CopyBatch::add(void* destination, const void* source, std::size_t bytes) {
    if (bytes == 0) { return; }
    const auto begin = reinterpret_cast<std::uintptr_t>(destination), end = begin + bytes;
    // Overlap: the first pending range starting after `begin` starts before `end`, or the last one
    // starting at or before `begin` ends after it.
    auto next = ranges_.upper_bound(begin);
    const bool overlaps = (next != ranges_.end() && next->first < end) ||
                          (next != ranges_.begin() && std::prev(next)->second > begin);
    if (overlaps) { flush(); }
    destinations_.push_back(destination);
    sources_.push_back(source);
    sizes_.push_back(bytes);
    ranges_.emplace(begin, end);
}

void CopyBatch::flush() {
    if (sizes_.empty()) { return; }
    // CUDA 12.8's form (non-const sources and a failure index); a driver that refuses the batched
    // call gets the same copies one by one, in the same stream order.
    cudaMemcpyAttributes attributes{};
    attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    std::size_t first         = 0;
    std::size_t failed        = 0;
    std::vector<void*> sources(sources_.size());
    for (std::size_t i = 0; i < sources_.size(); ++i) { sources[i] = const_cast<void*>(sources_[i]); }
    if (!batch_refused_ &&
        cudaMemcpyBatchAsync(destinations_.data(), sources.data(), sizes_.data(), sizes_.size(), &attributes, &first, 1,
                             &failed, stream_) != cudaSuccess) {
        (void)cudaGetLastError();
        batch_refused_ = true;
    }
    if (batch_refused_) {
        for (std::size_t i = 0; i < sizes_.size(); ++i) {
            CUDA_CHECK(cudaMemcpyAsync(destinations_[i], sources_[i], sizes_[i], cudaMemcpyDefault, stream_));
        }
    }
    destinations_.clear();
    sources_.clear();
    sizes_.clear();
    ranges_.clear();
}

} // namespace ninfer
