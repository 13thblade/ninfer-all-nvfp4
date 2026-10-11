// Adapted from Infernix a3edb450 src/core/device.cu (upload_pinned) (Apache-2.0).
// Modified for NInfer-3090: its own translation unit and NInfer's namespace.

#include "core/pinned_upload.h"

#include "core/device.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer {
namespace {

__global__ void upload_pinned_kernel(void* __restrict__ dst, const void* __restrict__ src, std::size_t bytes) {
    const std::size_t thread = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const bool vectors = (reinterpret_cast<std::uintptr_t>(dst) | reinterpret_cast<std::uintptr_t>(src)) % 16 == 0;
    const std::size_t head = vectors ? bytes / 16 : 0;
    for (std::size_t i = thread; i < head; i += stride) {
        static_cast<uint4*>(dst)[i] = static_cast<const uint4*>(src)[i];
    }
    for (std::size_t i = head * 16 + thread; i < bytes; i += stride) {
        static_cast<unsigned char*>(dst)[i] = static_cast<const unsigned char*>(src)[i];
    }
}

} // namespace

void upload_pinned(void* device_dst, const void* pinned_src, std::size_t bytes, cudaStream_t stream) {
    if (bytes == 0) { return; }
    static const bool unified = [] {
        int device = 0, value = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrUnifiedAddressing, device));
        return value != 0;
    }();
    if (!unified) { throw std::runtime_error("upload_pinned needs unified addressing"); }
    constexpr int kThreads = 256;
    // One 16-byte load per thread: every load of a call is in flight at once.
    const std::size_t blocks = std::min<std::size_t>((bytes / 16 + kThreads) / kThreads + 1, 64);
    upload_pinned_kernel<<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(device_dst, pinned_src, bytes);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer
