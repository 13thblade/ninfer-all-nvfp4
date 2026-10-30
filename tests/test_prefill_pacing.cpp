#include "runtime/engine/prefill_pacing.h"

#include <cstdint>
#include <iostream>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

using ninfer::runtime::ContextMachineCostModel;
using ninfer::runtime::PrefillPacing;
using ninfer::runtime::predicted_prefill_step_ns;

constexpr std::uint64_t kQ32 = std::uint64_t{1} << 32;

// 1 ms per step plus 100 us per token, and 2 ns per attention pair: a 2048-token step from token
// zero is predicted at 1 + 204.8 + 4.2 ms.
ContextMachineCostModel costs() {
    ContextMachineCostModel model;
    model.prefill.chunk_ns              = 1'000'000;
    model.prefill.token_ns_q32          = 100'000 * kQ32;
    model.prefill.attention_pair_ns_q32 = 2 * kQ32;
    return model;
}

} // namespace

int main() {
    int failures = 0;
    const ContextMachineCostModel model = costs();

    failures += check(predicted_prefill_step_ns(model, 0, 0) == 0, "an empty step costs nothing");
    failures += check(predicted_prefill_step_ns(model, 0, 2048) ==
                          1'000'000 + 2048 * 100'000 + 2 * (2048 * 2049 / 2),
                      "a step is one chunk of its tokens and their attention pairs");

    // Off: every step runs as planned.
    failures += check(PrefillPacing().step_cap(0, 4096) == 0 && !PrefillPacing().enabled(),
                      "no budget must leave the step planned");

    // 60 ms fits 1 + 0.1 t + ~0 ms up to 590 tokens: the largest multiple of 128 is 512.
    PrefillPacing pacing(60, model);
    failures += check(pacing.step_cap(0, 2048) == 512, "a step is cut to the budget in 128s");
    // A planned step that fits, and one no larger than the minimum, stay as planned.
    failures += check(pacing.step_cap(0, 384) == 0, "a short step must not be cut");
    failures += check(PrefillPacing(1000, model).step_cap(0, 2048) == 0,
                      "a step inside the budget must not be cut");
    // Nothing goes below 512 tokens, even when 512 itself overruns the budget.
    failures += check(PrefillPacing(1, model).step_cap(0, 2048) == 512,
                      "a step is never cut below the minimum");

    // 120 ms: 1152 tokens from token zero; after 100K tokens each token also attends to the
    // prefix (0.2 ms more per token), so the step shrinks to 384 by time, and to the minimum.
    PrefillPacing deep(120, model);
    const std::uint32_t shallow = deep.step_cap(0, 4096);
    const std::uint32_t far     = deep.step_cap(100'000, 4096);
    failures += check(shallow == 1152, "the cap is the largest affordable multiple of 128");
    failures += check(far == 512 && far < shallow, "a deeper prefix must be cut harder");

    // Steps observed at twice the prediction halve what fits: the correction converges on 2.
    PrefillPacing observed(120, model);
    for (int step = 0; step < 32; ++step) {
        observed.step_done(0, 1024, 2 * predicted_prefill_step_ns(model, 0, 1024));
    }
    failures += check(observed.correction() > 1.99 && observed.correction() < 2.01,
                      "the correction must follow the observed step times");
    failures += check(observed.step_cap(0, 4096) == 512,
                      "a slower card must cut the step to what fits at its speed");
    // One outlier moves the correction a quarter of the way in log space, bounded at 64x.
    PrefillPacing outlier(120, model);
    outlier.step_done(0, 1024, 1'000 * predicted_prefill_step_ns(model, 0, 1024));
    failures += check(outlier.correction() > 2.82 && outlier.correction() < 2.83,
                      "one outlier step must move the correction by 64^(1/4)");
    // An empty or unmeasured step changes nothing.
    outlier.step_done(0, 0, 5);
    outlier.step_done(0, 1024, 0);
    failures += check(outlier.correction() > 2.82 && outlier.correction() < 2.83,
                      "an empty or unmeasured step must not move the correction");

    if (failures == 0) { std::cout << "prefill pacing tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
