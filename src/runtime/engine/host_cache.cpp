#include "runtime/engine/host_cache.h"

#include "core/cgroup_memory.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace ninfer::runtime {
namespace {

#if defined(__linux__)
std::optional<std::uint64_t> meminfo_bytes(std::string_view wanted) {
    std::ifstream file("/proc/meminfo");
    std::string key;
    std::uint64_t kib = 0;
    std::string unit;
    while (file >> key >> kib) {
        std::getline(file, unit);
        if (key == wanted) { return kib * 1024ULL; }
    }
    return std::nullopt;
}
#endif

} // namespace

std::optional<std::uint64_t> cgroup_remaining_bytes(const std::filesystem::path& cgroup_root,
                                                    std::string_view proc_self_cgroup) {
    return ninfer::cgroup_remaining_bytes(cgroup_root, proc_self_cgroup);
}

std::optional<std::uint64_t> cgroup_limit_bytes(const std::filesystem::path& cgroup_root,
                                                std::string_view proc_self_cgroup) {
    return ninfer::cgroup_limit_bytes(cgroup_root, proc_self_cgroup);
}

std::optional<std::uint64_t> available_host_memory_bytes() noexcept {
    try {
#if defined(_WIN32)
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (!GlobalMemoryStatusEx(&status)) { return std::nullopt; }
        return static_cast<std::uint64_t>(status.ullAvailPhys);
#elif defined(__linux__)
        const std::optional<std::uint64_t> system = meminfo_bytes("MemAvailable:");
        std::ifstream self("/proc/self/cgroup");
        std::ostringstream self_text;
        self_text << self.rdbuf();
        const std::optional<std::uint64_t> cgroup =
            cgroup_remaining_bytes("/sys/fs/cgroup", self_text.str());
        if (system && cgroup) { return std::min(*system, *cgroup); }
        return system ? system : cgroup;
#else
        return std::nullopt;
#endif
    } catch (...) { return std::nullopt; }
}

std::optional<std::uint64_t> total_host_memory_bytes() noexcept {
    try {
#if defined(_WIN32)
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (!GlobalMemoryStatusEx(&status)) { return std::nullopt; }
        return static_cast<std::uint64_t>(status.ullTotalPhys);
#elif defined(__linux__)
        const std::optional<std::uint64_t> system = meminfo_bytes("MemTotal:");
        std::ifstream self("/proc/self/cgroup");
        std::ostringstream self_text;
        self_text << self.rdbuf();
        const std::optional<std::uint64_t> cgroup =
            cgroup_limit_bytes("/sys/fs/cgroup", self_text.str());
        if (system && cgroup) { return std::min(*system, *cgroup); }
        return system ? system : cgroup;
#else
        return std::nullopt;
#endif
    } catch (...) { return std::nullopt; }
}

std::size_t resolve_host_capacity_bytes(const ContextCacheOptions& requested,
                                        std::uint64_t available_host_bytes,
                                        const HostCacheEnvironment& environment) {
    if (!requested.auto_host_cache) {
        throw std::logic_error("host cache sizing was requested without auto_host_cache");
    }
    if (requested.host_capacity_bytes) {
        throw std::invalid_argument(
            "auto_host_cache sizes the Host context capacity; leave host_capacity_bytes unset");
    }

    // The machine serves only this process, so everything but a fixed reserve is spent. The reserve
    // covers what grows after this point: request buffers, the response store, graph
    // instantiation and module loads. Pinned pages cannot be reclaimed, so it is not a fraction.
    // Saturating add: both terms are independently accepted, and a wrapped sum would leave
    // almost nothing reserved.
    const std::uint64_t extra_reserve = environment.extra_reserve_bytes;
    const std::uint64_t reserve =
        requested.host_cache_reserve_bytes > std::numeric_limits<std::uint64_t>::max() - extra_reserve
            ? std::numeric_limits<std::uint64_t>::max()
            : requested.host_cache_reserve_bytes + extra_reserve;
    std::uint64_t budget = available_host_bytes > reserve ? available_host_bytes - reserve : 0;
    // The reserve leaves memory for what still grows; a cap leaves memory for the machine's other
    // users, which `available_host_bytes` cannot know will want it.
    if (requested.host_cache_max_bytes) {
        budget = std::min<std::uint64_t>(budget, *requested.host_cache_max_bytes);
    }
    if (requested.host_cache_percent) {
        if (*requested.host_cache_percent == 0 || *requested.host_cache_percent > 100) {
            throw std::invalid_argument("host cache percent must be in [1,100]");
        }
        if (!environment.total_host_bytes) {
            throw std::invalid_argument("host cache percent needs the machine's total memory");
        }
        // total * percent / 100 without overflow, split so the remainder is not lost to rounding.
        const std::uint64_t total   = *environment.total_host_bytes;
        const std::uint64_t percent = *requested.host_cache_percent;
        budget = std::min<std::uint64_t>(budget, total / 100 * percent + total % 100 * percent / 100);
    }
    if (budget > std::numeric_limits<std::size_t>::max()) {
        budget = std::numeric_limits<std::size_t>::max();
    }
    return static_cast<std::size_t>(budget);
}

} // namespace ninfer::runtime
