#include "runtime/engine/kv_capacity.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    const ninfer::runtime::SequenceCapacityCurve curve{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 6,
        .minimum_device_reservation_bytes     = 1000,
        .bytes_per_additional_main_page_group = 128,
    };

    const auto automatic =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 1360);
    failures +=
        check(automatic.main_page_groups == 4 && automatic.resolved_tokens == 256 &&
                  automatic.runtime_reservation_bytes == 1256 &&
                  automatic.automatic_headroom_bytes == 50 && automatic.planned_slack_bytes == 104,
              "automatic KV capacity did not select the largest fitting page count");

    const auto capped =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 10000);
    failures += check(capped.main_page_groups == 6 && capped.resolved_tokens == 384,
                      "automatic KV capacity exceeded or missed the target maximum");

    const auto explicit_capacity = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(129), curve, 1200);
    failures +=
        check(explicit_capacity.main_page_groups == 3 && explicit_capacity.resolved_tokens == 192 &&
                  explicit_capacity.runtime_reservation_bytes == 1128,
              "explicit KV capacity did not use page-aligned token semantics");

    // Pages held for good by installed context are granted on top of an explicit token capacity.
    ninfer::runtime::SequenceCapacityCurve pinned = curve;
    pinned.minimum_main_page_groups += 1;
    pinned.maximum_main_page_groups += 1;
    pinned.pinned_main_page_groups = 1;
    const auto pinned_explicit = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(129), pinned, 1200);
    failures += check(pinned_explicit.main_page_groups == 4 &&
                          pinned_explicit.runtime_reservation_bytes == 1128,
                      "explicit KV capacity was not granted on top of pinned pages");

    bool insufficient_rejected = false;
    try {
        (void)ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve,
                                                   1049);
    } catch (const std::invalid_argument&) { insufficient_rejected = true; }
    failures += check(insufficient_rejected,
                      "automatic KV capacity accepted less than the minimum reservation");

    // A model split across devices: each holds its own layers' KV, so a page group costs each a
    // different amount, and the capacity is the smallest any one allows.
    ninfer::runtime::SequenceCapacityCurve split{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 20,
        .minimum_device_reservation_bytes     = 1000,
        .bytes_per_additional_main_page_group = 100,
        .extra_ranks                          = {{.minimum_device_reservation_bytes     = 500,
                                                  .bytes_per_additional_main_page_group = 300}},
    };
    const std::size_t split_available[] = {1500};
    // Device 0 alone would allow 11 pages; device 1 allows 2 + (1500 - 50 - 500) / 300 = 5.
    const auto both = ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50),
                                                           split, 2000, split_available);
    failures += check(both.main_page_groups == 5 && both.binding_rank == 1 &&
                          both.extra_rank_reservation_bytes.size() == 1 &&
                          both.extra_rank_reservation_bytes[0] == 500 + 3 * 300,
                      "the capacity was not limited by the tighter device");
    failures += check(split.extra_rank_reservation_bytes(0, 5) == 1400,
                      "a further device's reservation does not follow its own curve");

    // A device holding no KV does not limit the count, so the primary device decides.
    split.extra_ranks[0].bytes_per_additional_main_page_group = 0;
    const auto free_rank = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(50), split, 2000, split_available);
    failures += check(free_rank.main_page_groups == 11 && free_rank.binding_rank == 0,
                      "a device without KV constrained the capacity");
    split.extra_ranks[0].bytes_per_additional_main_page_group = 300;

    // A failure names the device that could not hold the plan.
    const auto names_device = [&](auto&& call, const char* device) {
        try {
            call();
        } catch (const std::invalid_argument& error) {
            return std::string(error.what()).find(device) != std::string::npos;
        }
        return false;
    };
    const std::size_t starved[] = {520};
    failures += check(names_device(
                          [&] {
                              (void)ninfer::runtime::resolve_kv_capacity(
                                  ninfer::KvCapacityPolicy::automatic(50), split, 2000, starved);
                          },
                          "device 1"),
                      "a device below its minimum reservation was not named");
    failures += check(names_device(
                          [&] {
                              (void)ninfer::runtime::resolve_kv_capacity(
                                  ninfer::KvCapacityPolicy::explicit_capacity(64 * 19), split, 100000,
                                  split_available);
                          },
                          "device 1"),
                      "an explicit capacity too large for one device was not named");
    failures += check(names_device(
                          [&] {
                              (void)ninfer::runtime::resolve_kv_capacity(
                                  ninfer::KvCapacityPolicy::automatic(50), split, 2000);
                          },
                          "every device"),
                      "missing free memory for a device was accepted");

    // Context fit. An explicit capacity fits when its whole reservation does, on every device.
    failures += check(ninfer::runtime::explicit_capacity_fits(129, curve, 1128) &&
                          !ninfer::runtime::explicit_capacity_fits(129, curve, 1127) &&
                          ninfer::runtime::explicit_reservation_bytes(129, curve) == 1128,
                      "explicit capacity fit disagrees with its reservation");
    failures += check(!ninfer::runtime::explicit_capacity_fits(64 * 19, split, 100000, split_available),
                      "an explicit capacity too large for a further device was said to fit");
    failures += check(!ninfer::runtime::explicit_capacity_fits(64 * 7, curve, 100000),
                      "an explicit capacity beyond the curve was said to fit");

    // The fit returns the ceiling when it fits, otherwise the largest step multiple below it, the
    // floor when only the floor fits, and nothing when not even the floor does -- in O(log) probes.
    int probes      = 0;
    const auto upto = [&](std::uint32_t limit) {
        return [&probes, limit](std::uint32_t context) {
            ++probes;
            return context <= limit;
        };
    };
    failures += check(ninfer::runtime::fit_context(32768, 188416, 1024, upto(188416)) == 188416,
                      "a ceiling that fits was lowered");
    probes = 0;
    failures += check(ninfer::runtime::fit_context(32768, 188416, 1024, upto(175000)) == 174080,
                      "the fit missed the largest step multiple that fits");
    failures += check(probes <= 2 + 9, "the fit probed more than logarithmically");
    failures += check(ninfer::runtime::fit_context(32768, 188416, 1024, upto(33000)) == 32768,
                      "the fit did not fall back to the floor");
    failures += check(!ninfer::runtime::fit_context(32768, 188416, 1024, upto(30000)).has_value(),
                      "the fit started below its floor");
    failures += check(ninfer::runtime::fit_context(1000, 1500, 1024, upto(1499)) == 1024,
                      "the fit did not take the only step multiple between floor and ceiling");
    bool bad_range_rejected = false;
    try {
        (void)ninfer::runtime::fit_context(4096, 2048, 1024, upto(0));
    } catch (const std::invalid_argument&) { bad_range_rejected = true; }
    failures += check(bad_range_rejected, "a floor above the ceiling was accepted");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
