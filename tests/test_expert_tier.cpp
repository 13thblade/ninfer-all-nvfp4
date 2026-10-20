// The host side of every MoE family's host and disk experts: the expert tier must fill free slots
// first, swap a candidate in only when its count beats the victim's by the gain, pair the hottest
// candidates with the coldest victims of their own pool, order a round by gain over every pool,
// cap it, decay the counts, and admit a candidate over a held expert its count beats, never one
// its call routes. The split must give the CPU more of the misses when copies turn out slow and fewer when
// its products do, and probe a side it left unused. The pinned pool must run every index of a job
// exactly once, run after run, and rethrow a job's failure.
#include "core/cpu_spin_pool.h"
#include "core/expert_split.h"
#include "core/expert_tier.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

bool near(float a, float b) { return std::fabs(a - b) < 1e-4F; }

void tier_rounds() {
    using State = ExpertTier::State;
    // Two layers of eight experts, each with its own pool; a round every second pass.
    ExpertTier tier({8, 8}, {0, 1}, {.every = 2});
    tier.set_free(0, 2);
    for (std::int32_t e = 0; e < 4; ++e) { tier.set(1, e, State::Held); }
    tier.route(0, std::vector<std::int32_t>{1, 1, 1, 2, 2, 3, -1, 8});
    tier.route(1, std::vector<std::int32_t>{5, 5, 5, 5, 6, 6, 6, 0});
    require(near(tier.count(0, 1), 3) && near(tier.count(0, 3), 1), "routes count once per pair");
    require(tier.end_pass().empty(), "no round before `every` passes");
    const auto swaps = tier.end_pass();
    // Layer 0 fills its free slots with experts 1 and 2 (3 is below min_count); layer 1 swaps 5
    // and 6 for its coldest held experts 1 and 2 (0 was routed once), best gain first.
    require(swaps.size() == 4, "a round's swaps: " + std::to_string(swaps.size()));
    require(swaps[0].layer == 1 && swaps[0].in == 5 && swaps[0].out_layer == 1 && swaps[0].out == 1 &&
                near(swaps[0].gain, 4),
            "the best gain comes first");
    require(swaps[1].layer == 0 && swaps[1].in == 1 && swaps[1].out == -1, "a free slot needs no victim");
    require(swaps[2].layer == 1 && swaps[2].in == 6 && swaps[2].out == 2, "the next coldest leaves next");
    require(swaps[3].layer == 0 && swaps[3].in == 2 && swaps[3].out == -1, "the second free slot");
    require(near(tier.count(1, 5), 4 * 0.7F), "a round decays every count");

    // A gain below min_gain swaps nothing; the cap keeps the best.
    ExpertTier close({4}, {0}, {.every = 1});
    close.set(0, 0, State::Held);
    close.route(0, std::vector<std::int32_t>{0, 0, 1, 1, 1});
    require(close.end_pass().empty(), "a candidate one route ahead does not swap");
    ExpertTier capped({4, 4}, {0, 1}, {.every = 1, .swaps = 1});
    capped.set(0, 0, State::Held);
    capped.set(1, 0, State::Held);
    capped.route(0, std::vector<std::int32_t>{1, 1, 1});
    capped.route(1, std::vector<std::int32_t>{1, 1, 1, 1, 1});
    const auto best = capped.end_pass();
    require(best.size() == 1 && best[0].layer == 1 && best[0].in == 1, "the cap keeps the best swap");
    // Pending experts are neither candidates nor victims.
    ExpertTier pending({4}, {0}, {.every = 1});
    pending.set(0, 0, State::Pending);
    pending.set(0, 1, State::Held);
    pending.route(0, std::vector<std::int32_t>{0, 0, 0, 0, 0});
    require(pending.end_pass().empty(), "a pending expert is no candidate");
}

void tier_admission() {
    using State = ExpertTier::State;
    ExpertTier tier({8}, {0}, {.every = 0});
    for (std::int32_t e = 0; e < 4; ++e) { tier.set(0, e, State::Held); }
    tier.route(0, std::vector<std::int32_t>{0, 0, 0, 1, 2, 2});
    require(tier.end_pass().empty(), "every = 0 runs no rounds");
    std::vector<unsigned char> busy(8, 0);
    // Coldest first: 3 (0 routes), 1 (1), 2 (2), 0 (3).
    require(tier.admission_victim(0, 7, 1, busy).expert < 0, "one pair does not beat 0 + 1.5");
    busy[3] = 1;
    const auto first = tier.admission_victim(0, 7, 3, busy);
    require(first.expert == 1, "the coldest expert the call does not route leaves");
    tier.set(0, 1, State::Out);
    const auto second = tier.admission_victim(0, 6, 5, busy);
    require(second.expert == 2, "an expert that left is not offered again");
    tier.set(0, 2, State::Out);
    require(tier.admission_victim(0, 5, 2, busy).expert < 0, "a cold candidate beats no warmer expert");
    require(tier.end_pass().empty(), "a pass ends");
    std::fill(busy.begin(), busy.end(), 0);
    // Routed since the pass began, expert 3 is no longer the one to beat.
    tier.route(0, std::vector<std::int32_t>(6, 3));
    require(tier.admission_victim(0, 7, 4, busy).expert < 0, "no held expert is cold enough");
    require(tier.admission_victim(0, 7, 5, busy).expert == 0, "the scan passes the warmed expert");

    // One pool for two layers: the victim may be another layer's.
    ExpertTier shared({4, 4}, {0, 0}, {.every = 0});
    shared.set(1, 2, State::Held);
    shared.set(0, 1, State::Held);
    shared.route(0, std::vector<std::int32_t>{1, 1, 1});
    require(shared.end_pass().empty(), "a pass ends");
    const auto other = shared.admission_victim(0, 3, 2, std::vector<unsigned char>(4, 0));
    require(other.layer == 1 && other.expert == 2, "a shared pool's coldest expert may be another layer's");
}

void split() {
    // Ten missing experts of 2 MB, one pair each.
    const std::vector<int> pairs(10, 1);
    ExpertSplit slow_copies;
    for (int i = 0; i < 400; ++i) {
        slow_copies.copied(10.0, 10000.0);    // 1 GB/s
        slow_copies.computed(10.0, 500.0);    // 20 GB/s of work
    }
    ExpertSplit slow_cpu;
    for (int i = 0; i < 400; ++i) {
        slow_cpu.copied(10.0, 400.0);         // 25 GB/s
        slow_cpu.computed(10.0, 20000.0);     // 0.5 GB/s of work
    }
    const auto fast_cpu_share = slow_copies.cpu_count(pairs, 2.0);
    const auto slow_cpu_share = slow_cpu.cpu_count(pairs, 2.0);
    require(fast_cpu_share > slow_cpu_share,
            "slow copies give the CPU more: " + std::to_string(fast_cpu_share) + " vs " +
                std::to_string(slow_cpu_share));
    require(fast_cpu_share >= 8 && slow_cpu_share == 0, "the faster side takes nearly all");
    // A side left out for eight calls takes one expert.
    std::size_t probes = 0;
    for (int call = 0; call < 9; ++call) { probes += slow_cpu.cpu_count(pairs, 2.0); }
    require(probes == 1, "the CPU is probed once in nine calls: " + std::to_string(probes));
    require(ExpertSplit::cpu_work(2.0, 5) > ExpertSplit::cpu_work(2.0, 1), "more pairs, more work");
}

void pool() {
    for (const bool pin : {false, true}) {
        CpuSpinPool workers(4, pin);
        require(workers.threads() == 4 && workers.pinned() == pin, "the pool's threads");
        for (int run = 0; run < 200; ++run) {
            const std::size_t count = std::size_t(1 + run * 7 % 97);
            std::vector<std::atomic<int>> hits(count);
            if (run == 50) { workers.spin(true); }
            if (run == 150) { workers.spin(false); }
            workers.run(count, [&](std::size_t i) { hits[i].fetch_add(1); });
            for (std::size_t i = 0; i < count; ++i) {
                require(hits[i].load() == 1, "every index runs once");
            }
        }
        bool thrown = false;
        try {
            workers.run(16, [](std::size_t i) {
                if (i == 5) { throw std::runtime_error("job failure"); }
            });
        } catch (const std::runtime_error&) { thrown = true; }
        require(thrown, "a job's failure reaches the caller");
        std::atomic<int> after{0};
        workers.run(8, [&](std::size_t) { after.fetch_add(1); });
        require(after.load() == 8, "the pool runs after a failure");
    }
    require(pool_cpu_count() >= 1 && pool_cpu_count() <= std::max(1U, usable_cpus() / 2),
            "the default pool leaves half the CPUs to the rest of the process");
    require(!physical_core_cpus().empty(), "every machine has a core");
}

} // namespace

int main() {
    try {
        tier_rounds();
        tier_admission();
        split();
        pool();
        std::cout << "PASS expert tier, split and pool\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
