#pragma once

// Adapted from Infernix a3edb450 src/core/device.h (upload_pinned) (Apache-2.0).
// Modified for NInfer-3090: its own header, so existing users of core/device.h do not rebuild.

#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer {

// Copies `bytes` from pinned host memory to device memory in stream order, read by the SMs
// through the host mapping (pinned allocations are device-accessible under UVA) rather than by a
// copy engine. Small per-round inputs then never queue behind bulk host-to-device transfers of
// other streams (for example expert promotions), which share the copy engine in FIFO order.
void upload_pinned(void* device_dst, const void* pinned_src, std::size_t bytes, cudaStream_t stream);

} // namespace ninfer
