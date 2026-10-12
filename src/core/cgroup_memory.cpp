#include "core/cgroup_memory.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// The cgroup hierarchy walk shared by the host-cache sizing (runtime/engine/host_cache.h) and the
// startup memory ledger (core/host_memory.h), so both plan against the same effective limit.

namespace ninfer {
namespace {

std::optional<std::uint64_t> read_u64_file(const std::filesystem::path& path) {
    std::ifstream file(path);
    std::string text;
    if (!file || !(file >> text) || text == "max") { return std::nullopt; }
    try {
        return std::stoull(text);
    } catch (...) { return std::nullopt; }
}

// File-backed page cache charged to a cgroup (active and inactive lists) is reclaimable, so it does
// not count against what remains. Counting only the inactive list would undercount what is free
// right after a large artifact was read, because pages read more than once sit on the active list.
std::uint64_t reclaimable_file_bytes(const std::filesystem::path& path, const char* active_key,
                                     const char* inactive_key) {
    std::ifstream file(path);
    std::string name;
    std::uint64_t value = 0;
    std::uint64_t total = 0;
    while (file >> name >> value) {
        if (name == active_key || name == inactive_key) { total += value; }
    }
    return total;
}

// What one cgroup directory still allows: its limit less its non-reclaimable use. Empty when the
// directory sets no limit (or does not exist), which leaves the decision to its ancestors.
std::optional<std::uint64_t> limit_under(const std::filesystem::path& dir, bool v2) {
    const auto limit = read_u64_file(dir / (v2 ? "memory.max" : "memory.limit_in_bytes"));
    // cgroup v1 reports "no limit" as a value near the top of the page-aligned 63-bit range.
    if (!limit || (!v2 && *limit >= (1ULL << 60))) { return std::nullopt; }
    return limit;
}

enum class CgroupMeasure { Remaining, Limit, SwapRemaining };

std::optional<std::uint64_t> measure_under(const std::filesystem::path& dir, bool v2,
                                           CgroupMeasure measure) {
    if (measure == CgroupMeasure::SwapRemaining) {
        // Only cgroup v2 limits swap on its own (v1 limits memory+swap together, which this
        // does not model); an unset memory.swap.max is "max", i.e. no limit at this level.
        if (!v2) { return std::nullopt; }
        const auto swap_limit = read_u64_file(dir / "memory.swap.max");
        if (!swap_limit) { return std::nullopt; }
        const std::uint64_t swap_used = read_u64_file(dir / "memory.swap.current").value_or(0);
        return *swap_limit > swap_used ? *swap_limit - swap_used : 0;
    }
    const auto limit = limit_under(dir, v2);
    if (!limit || measure == CgroupMeasure::Limit) { return limit; }
    const auto used = read_u64_file(dir / (v2 ? "memory.current" : "memory.usage_in_bytes"));
    if (!used) { return std::nullopt; }
    const std::uint64_t reclaimable = reclaimable_file_bytes(
        dir / "memory.stat", v2 ? "active_file" : "total_active_file",
        v2 ? "inactive_file" : "total_inactive_file");
    const std::uint64_t charged = *used > reclaimable ? *used - reclaimable : 0;
    return *limit > charged ? *limit - charged : 0;
}

std::vector<std::string> path_components(std::string path) {
    if (const auto deleted = path.find(" (deleted)"); deleted != std::string::npos) {
        path.erase(deleted);
    }
    std::vector<std::string> parts;
    std::string part;
    std::istringstream stream(path);
    while (std::getline(stream, part, '/')) {
        if (!part.empty()) { parts.push_back(part); }
    }
    return parts;
}

// The tightest remainder (or limit) over `base/<path>` and every ancestor up to `base`.
std::optional<std::uint64_t> tightest(const std::filesystem::path& base,
                                      const std::vector<std::string>& parts, bool v2,
                                      CgroupMeasure measure) {
    std::optional<std::uint64_t> best;
    for (std::size_t depth = parts.size() + 1; depth-- > 0;) {
        std::filesystem::path dir = base;
        for (std::size_t index = 0; index < depth; ++index) { dir /= parts[index]; }
        if (const auto remaining = measure_under(dir, v2, measure)) {
            best = best ? std::min(*best, *remaining) : *remaining;
        }
    }
    return best;
}


std::optional<std::uint64_t> cgroup_tightest(const std::filesystem::path& cgroup_root,
                                             std::string_view proc_self_cgroup,
                                             CgroupMeasure measure) {
    std::optional<std::uint64_t> best;
    const auto take = [&](std::optional<std::uint64_t> remaining) {
        if (remaining) { best = best ? std::min(*best, *remaining) : *remaining; }
    };
    std::istringstream lines{std::string(proc_self_cgroup)};
    std::string line;
    while (std::getline(lines, line)) {
        // "<hierarchy>:<controllers>:<path>"; the path may itself contain colons.
        const std::size_t first = line.find(':');
        const std::size_t second = first == std::string::npos ? first : line.find(':', first + 1);
        if (second == std::string::npos) { continue; }
        const std::string controllers = line.substr(first + 1, second - first - 1);
        const std::vector<std::string> parts = path_components(line.substr(second + 1));
        if (line.compare(0, first, "0") == 0 && controllers.empty()) {
            take(tightest(cgroup_root, parts, true, measure));
        } else if ((',' + controllers + ',').find(",memory,") != std::string::npos) {
            take(tightest(cgroup_root / "memory", parts, false, measure));
        }
    }
    return best;
}

} // namespace

std::optional<std::uint64_t> cgroup_remaining_bytes(const std::filesystem::path& cgroup_root,
                                                    std::string_view proc_self_cgroup) {
    return cgroup_tightest(cgroup_root, proc_self_cgroup, CgroupMeasure::Remaining);
}

std::optional<std::uint64_t> cgroup_limit_bytes(const std::filesystem::path& cgroup_root,
                                                std::string_view proc_self_cgroup) {
    return cgroup_tightest(cgroup_root, proc_self_cgroup, CgroupMeasure::Limit);
}

std::optional<std::uint64_t> cgroup_swap_remaining_bytes(const std::filesystem::path& cgroup_root,
                                                         std::string_view proc_self_cgroup) {
    return cgroup_tightest(cgroup_root, proc_self_cgroup, CgroupMeasure::SwapRemaining);
}

} // namespace ninfer
