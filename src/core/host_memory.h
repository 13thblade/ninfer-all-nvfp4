#pragma once

// Adapted from Infernix a3edb450 src/core/host_memory.h (Apache-2.0).
// Modified for NInfer-3090: NInfer's namespace; the snapshot and working-set reserve only (the
// available-memory query of the Host context cache stays in runtime/engine/host_cache.h).

#include <cstdint>
#include <optional>
#include <string>

namespace ninfer {

// One reading of the Host memory a startup ledger plans against.
struct HostMemorySnapshot {
    std::uint64_t total_physical = 0;
    // Physical Host memory the OS can hand out now without paging: free plus reclaimable file
    // cache (Windows ullAvailPhys, Linux MemAvailable), lowered on Linux to what the process's
    // cgroup still allows.
    std::uint64_t available_physical = 0;
    // Memory the OS can still commit (Windows ullAvailPageFile: the commit limit less what is
    // committed; Linux: MemAvailable plus free swap, as Linux commits lazily).
    std::uint64_t available_commit = 0;
    // This process's private committed bytes (Windows PrivateUsage, Linux VmRSS).
    std::uint64_t process_private = 0;
    // Linux RLIMIT_MEMLOCK when limited; Windows has no such limit for CUDA pinned memory.
    std::optional<std::uint64_t> lock_limit;
};

[[nodiscard]] HostMemorySnapshot host_memory_snapshot();

// Keeps at least `bytes` of this process resident (Windows: a hard minimum working set), so a small
// RAM reserve trims other programs before NInfer's own pageable memory. Empty on success and where
// the OS has no such control (Linux); otherwise the OS's reason (non-fatal: the caller logs it).
[[nodiscard]] std::string reserve_process_working_set(std::uint64_t bytes);

} // namespace ninfer
