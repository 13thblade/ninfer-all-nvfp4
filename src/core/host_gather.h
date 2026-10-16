#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer {

// One 2D copy for gather_copies: `height` rows of `width` bytes. `source` is page-locked Host memory
// (cudaHostAlloc, mapped under unified addressing) or Device memory.
struct GatherCopy {
    void* destination               = nullptr;
    const void* source              = nullptr;
    std::uint64_t destination_pitch = 0;
    std::uint64_t source_pitch      = 0;
    std::uint64_t width             = 0;
    std::uint64_t height            = 0;
};

// Runs a list of 2D copies as one kernel that reads Host pages over PCIe, instead of copy-engine
// work. A GeForce card under WDDM exposes one copy engine (asyncEngineCount = 1), so a bulk H2D
// there holds every later H2D from any stream behind it -- including the few bytes of inputs a
// forward pass uploads. A gather leaves the engine free and reaches the PCIe rate with a few SMs.
// `descriptors` lives in page-locked Host memory (cudaHostAlloc) and must stay valid and unchanged
// until the kernel has completed on `stream`.
void gather_copies(std::span<const GatherCopy> descriptors, cudaStream_t stream);

// Splits copies by rows into pieces of at most `piece_bytes` (at least one row), so the kernel's
// blocks, which take whole pieces, share a long copy.
void split_gather_copies(std::vector<GatherCopy>& copies, std::size_t piece_bytes);

} // namespace ninfer
