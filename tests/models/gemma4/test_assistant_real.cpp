// Gemma 4's assistant against a real artifact (skipped without one):
//   NINFER_GEMMA4_ARTIFACT  a .ninfer converted with --components text,mtp
//   NINFER_GEMMA4_TOKENS    a JSON list of prompt token ids, <bos> first
//   NINFER_GEMMA4_DRAFTS    optional: tools/reference/gemma4.py --assistant output for the same ids
//                           ({"anchor", "drafts", "margins"}), the FP64 oracle of the chain
// It prefills the prompt, takes the greedy anchor, drafts three tokens and checks them against the
// oracle where the oracle's top-2 margin is clear of rounding (0.5 logits). It then verifies the
// anchor and the drafts in one pass on that sequence and decodes the same tokens one at a time on
// another, and checks that both give each position the same greedy token and logits within BF16
// rounding; and that commit() keeps the given prefix and leaves the sequence ready to draft.

#include "artifact/reader.h"
#include "core/device.h"
#include "models/gemma4/executor.h"
#include "models/gemma4/model.h"

#include <cuda_bf16.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gemma4 = ninfer::models::gemma4;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

nlohmann::json read_json(const char* path) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error(std::string("cannot read ") + path); }
    return nlohmann::json::parse(in);
}

std::vector<float> host_logits(gemma4::Executor& executor, std::uint32_t rows) {
    const Tensor logits = executor.logits(rows);
    std::vector<__nv_bfloat16> raw(std::size_t(logits.ne[0]) * rows);
    CUDA_CHECK(cudaStreamSynchronize(executor.head_stream()));
    CUDA_CHECK(cudaMemcpy(raw.data(), logits.data, raw.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<float> out(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) { out[i] = __bfloat162float(raw[i]); }
    return out;
}

std::int32_t argmax(const float* row, std::int32_t n) {
    std::int32_t best = 0;
    for (std::int32_t i = 1; i < n; ++i) {
        if (row[i] > row[best]) { best = i; }
    }
    return best;
}

} // namespace

int main() {
    const char* artifact_path = std::getenv("NINFER_GEMMA4_ARTIFACT");
    const char* tokens_path   = std::getenv("NINFER_GEMMA4_TOKENS");
    if (artifact_path == nullptr || tokens_path == nullptr) {
        std::cout << "SKIP Gemma 4 assistant: NINFER_GEMMA4_ARTIFACT/NINFER_GEMMA4_TOKENS unset\n";
        return 77;
    }
    std::vector<std::int32_t> prompt = read_json(tokens_path).get<std::vector<std::int32_t>>();
    DeviceContext device(0);
    device.bind_to_current_thread();
    const artifact::Reader reader(artifact_path);
    auto model = gemma4::load_model(reader, {.artifact = artifact_path, .assistant = true}, device);
    const std::int32_t domain = std::int32_t(model->resources().public_token_count);
    constexpr std::uint32_t kSteps = 3;
    gemma4::ExecutorOptions options;
    options.max_context   = 4096;
    options.prefill_chunk = 2048;
    options.sequences     = 2;
    options.draft_tokens  = kSteps;
    gemma4::Executor executor(*model, device, options);
    executor.warm_up();

    const auto n = std::uint32_t(prompt.size());
    executor.forward(0, prompt, 1);
    const std::vector<float> first = host_logits(executor, 1);
    const std::int32_t anchor      = argmax(first.data(), domain);
    expect(executor.can_draft(0), "a prefilled sequence can draft");
    std::vector<std::int32_t> drafts(kSteps);
    const std::uint32_t s0 = 0, s1 = 1;
    executor.draft({&s0, 1}, {&anchor, 1}, drafts, kSteps);
    // Twice more: the eager run, the captured graph and its replay give the same chain.
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<std::int32_t> again(kSteps);
        executor.draft({&s0, 1}, {&anchor, 1}, again, kSteps);
        expect(again == drafts, "a replayed draft chain gives the same drafts");
    }
    std::cout << nlohmann::json({{"anchor", anchor}, {"drafts", drafts}}).dump() << '\n';
    if (const char* reference = std::getenv("NINFER_GEMMA4_DRAFTS")) {
        const auto oracle = read_json(reference);
        expect(oracle.at("anchor").get<std::int32_t>() == anchor, "the target's greedy anchor");
        const auto expected = oracle.at("drafts").get<std::vector<std::int32_t>>();
        const auto margins  = oracle.at("margins").get<std::vector<double>>();
        // Each step feeds the last draft: compare until the first draft the oracle's margin leaves
        // to rounding.
        for (std::uint32_t i = 0; i < kSteps && i < expected.size(); ++i) {
            if (margins.at(i) < 0.5) { break; }
            expect(drafts[i] == expected[i],
                   "draft " + std::to_string(i) + ": " + std::to_string(drafts[i]) + " vs oracle " +
                       std::to_string(expected[i]));
        }
    }

    // One verification of the anchor and drafts against the same tokens decoded one by one.
    std::vector<std::int32_t> run{anchor};
    run.insert(run.end(), drafts.begin(), drafts.end());
    executor.verify({&s0, 1}, run);
    const std::vector<float> verified = host_logits(executor, kSteps + 1);
    executor.forward(1, prompt, 1);
    double worst = 0;
    std::size_t equal = 0;
    const std::size_t vocab = verified.size() / (kSteps + 1);
    for (std::uint32_t i = 0; i <= kSteps; ++i) {
        executor.decode({&s1, 1}, {&run[i], 1});
        const std::vector<float> decoded = host_logits(executor, 1);
        const float* v = verified.data() + i * vocab;
        for (std::size_t t = 0; t < std::size_t(domain); ++t) {
            worst = std::max(worst, double(std::fabs(v[t] - decoded[t])));
            equal += v[t] == decoded[t];
        }
        expect(argmax(v, domain) == argmax(decoded.data(), domain),
               "verified and decoded position " + std::to_string(i) + " agree on the greedy token");
    }
    std::cout << "verify vs decode: max |logit difference| " << worst << ", equal "
              << double(equal) / double(std::size_t(domain) * (kSteps + 1)) << '\n';
    expect(worst <= 0.5, "verified logits match decoded ones within BF16 rounding");

    const std::uint32_t kept = 2;
    executor.commit({&s0, 1}, {&kept, 1});
    expect(executor.position(0) == n + kept, "commit keeps the given prefix");
    expect(executor.can_draft(0), "a committed sequence can draft");
    std::vector<std::int32_t> next(kSteps);
    executor.draft({&s0, 1}, {&run[kept], 1}, next, kSteps);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " Gemma 4 assistant\n";
    return failures == 0 ? 0 : 1;
}
