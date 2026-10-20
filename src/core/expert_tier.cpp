#include "core/expert_tier.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ninfer {

ExpertTier::ExpertTier(std::vector<std::uint32_t> experts, std::vector<std::uint32_t> pools,
                       ExpertTierOptions options)
    : options_(options), experts_(std::move(experts)), pools_(std::move(pools)) {
    if (experts_.empty() || pools_.size() != experts_.size() ||
        !std::isfinite(options_.decay) || options_.decay < 0 || options_.decay > 1 ||
        !std::isfinite(options_.min_gain) || options_.min_gain < 0 ||
        !std::isfinite(options_.min_count) || options_.min_count < 0) {
        throw std::invalid_argument("expert tier: invalid layers, pools or options");
    }
    std::uint32_t pool_count = 0;
    for (std::size_t l = 0; l < experts_.size(); ++l) {
        if (experts_[l] == 0) { throw std::invalid_argument("expert tier: a layer has no experts"); }
        offset_.push_back(entries_.size());
        entries_.resize(entries_.size() + experts_[l]);
        layer_of_.resize(entries_.size(), std::uint32_t(l));
        pool_count = std::max(pool_count, pools_[l] + 1);
    }
    free_.assign(pool_count, 0);
    cold_.resize(pool_count);
    cold_next_.assign(pool_count, 0);
}

std::size_t ExpertTier::index(std::size_t layer, std::int32_t expert) const {
    if (layer >= experts_.size() || expert < 0 || std::uint32_t(expert) >= experts_[layer]) {
        throw std::out_of_range("expert tier: no such expert");
    }
    return offset_[layer] + std::size_t(expert);
}

void ExpertTier::route(std::size_t layer, std::span<const std::int32_t> ids) {
    if (layer >= experts_.size()) { throw std::out_of_range("expert tier: no such layer"); }
    for (const std::int32_t id : ids) {
        if (id >= 0 && std::uint32_t(id) < experts_[layer]) {
            entries_[offset_[layer] + std::size_t(id)].count += 1.0F;
        }
    }
}

void ExpertTier::route(std::size_t layer, std::int32_t expert, float pairs) {
    entries_[index(layer, expert)].count += pairs;
}

void ExpertTier::set(std::size_t layer, std::int32_t expert, State state) {
    entries_[index(layer, expert)].state = state;
}

ExpertTier::State ExpertTier::state(std::size_t layer, std::int32_t expert) const {
    return entries_[index(layer, expert)].state;
}

float ExpertTier::count(std::size_t layer, std::int32_t expert) const {
    return entries_[index(layer, expert)].count;
}

void ExpertTier::set_free(std::uint32_t pool, std::uint32_t slots) { free_.at(pool) = slots; }

void ExpertTier::reset() {
    std::fill(entries_.begin(), entries_.end(), Entry{});
    std::fill(free_.begin(), free_.end(), 0);
    for (auto& list : cold_) { list.clear(); }
    std::fill(cold_next_.begin(), cold_next_.end(), 0);
    passes_ = 0;
}

void ExpertTier::refresh_cold() {
    for (auto& list : cold_) { list.clear(); }
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i].state == State::Held) { cold_[pools_[layer_of_[i]]].push_back(i); }
    }
    for (auto& list : cold_) {
        std::stable_sort(list.begin(), list.end(), [&](std::size_t a, std::size_t b) {
            return entries_[a].count < entries_[b].count;
        });
    }
    std::fill(cold_next_.begin(), cold_next_.end(), 0);
}

ExpertTier::Victim ExpertTier::admission_victim(std::size_t layer, std::int32_t expert,
                                                float pairs,
                                                std::span<const unsigned char> busy) {
    const float candidate = entries_[index(layer, expert)].count + pairs;
    const std::uint32_t pool = pools_[layer];
    const auto& list         = cold_[pool];
    std::size_t& first       = cold_next_[pool];
    while (first < list.size() && entries_[list[first]].state != State::Held) { ++first; }
    // Counts grow within a pass, so the order is a guide: the first beatable expert wins.
    std::size_t held = 0;
    for (std::size_t at = first; at < list.size() && held < kScan; ++at) {
        const std::size_t i   = list[at];
        const Entry& victim   = entries_[i];
        const std::uint32_t l = layer_of_[i];
        const auto e          = std::int32_t(i - offset_[l]);
        if (victim.state != State::Held) { continue; }
        ++held;
        if (l == layer && std::size_t(e) < busy.size() && busy[std::size_t(e)] != 0) { continue; }
        if (candidate >= victim.count + options_.min_gain) { return {l, e}; }
    }
    return {};
}

std::vector<ExpertTier::Swap> ExpertTier::end_pass() {
    std::vector<Swap> swaps;
    if (options_.every != 0 && ++passes_ >= options_.every) {
        passes_ = 0;
        std::vector<std::vector<std::size_t>> candidates(cold_.size()), victims(cold_.size());
        for (std::size_t i = 0; i < entries_.size(); ++i) {
            const std::uint32_t pool = pools_[layer_of_[i]];
            if (entries_[i].state == State::Held) {
                victims[pool].push_back(i);
            } else if (entries_[i].state == State::Out && entries_[i].count >= options_.min_count) {
                candidates[pool].push_back(i);
            }
        }
        const auto split = [&](std::size_t i) {
            const std::uint32_t l = layer_of_[i];
            return std::pair{l, std::int32_t(i - offset_[l])};
        };
        for (std::size_t pool = 0; pool < cold_.size(); ++pool) {
            auto& in  = candidates[pool];
            auto& out = victims[pool];
            std::stable_sort(in.begin(), in.end(), [&](std::size_t a, std::size_t b) {
                return entries_[a].count > entries_[b].count;
            });
            std::stable_sort(out.begin(), out.end(), [&](std::size_t a, std::size_t b) {
                return entries_[a].count < entries_[b].count;
            });
            std::size_t next = 0;
            for (std::size_t c = 0; c < in.size(); ++c) {
                const auto [layer, expert] = split(in[c]);
                if (c < free_[pool]) {
                    swaps.push_back({layer, expert, 0, -1, entries_[in[c]].count});
                    continue;
                }
                if (next == out.size()) { break; }
                const float gain = entries_[in[c]].count - entries_[out[next]].count;
                if (gain < options_.min_gain) { break; }
                const auto [victim_layer, victim] = split(out[next++]);
                swaps.push_back({layer, expert, victim_layer, victim, gain});
            }
        }
        std::stable_sort(swaps.begin(), swaps.end(),
                         [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
        if (swaps.size() > options_.swaps) { swaps.resize(options_.swaps); }
        for (auto& entry : entries_) { entry.count *= options_.decay; }
    }
    refresh_cold();
    return swaps;
}

} // namespace ninfer
