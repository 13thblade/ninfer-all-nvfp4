#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>

namespace ninfer {

// What the memory cgroup of this process still allows, from the contents of `/proc/self/cgroup`
// and the cgroup mount `cgroup_root` (normally /sys/fs/cgroup). Each level from the process's own
// cgroup up to the root is read and the tightest remainder wins, because a limit may sit on any
// ancestor, not at the mount root. Handles cgroup v2 and v1 (and hybrid); reclaimable page cache
// is not counted as use. Empty when no level sets a limit.
[[nodiscard]] std::optional<std::uint64_t>
cgroup_remaining_bytes(const std::filesystem::path& cgroup_root, std::string_view proc_self_cgroup);

// The tightest memory limit (not the remainder) that any level above the process sets.
[[nodiscard]] std::optional<std::uint64_t>
cgroup_limit_bytes(const std::filesystem::path& cgroup_root, std::string_view proc_self_cgroup);

// The tightest swap remainder (memory.swap.max less memory.swap.current) over the levels above
// the process, cgroup v2 only. Empty when no level limits swap.
[[nodiscard]] std::optional<std::uint64_t>
cgroup_swap_remaining_bytes(const std::filesystem::path& cgroup_root, std::string_view proc_self_cgroup);

} // namespace ninfer
