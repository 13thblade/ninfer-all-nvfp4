// Qwen3.8-Flash-Next through the public Engine on the real artifact (milestone M1). Skips (77)
// unless NINFER_TEST_QWEN4EXP_ARTIFACT names a Qwen4Exp artifact (its n-gram volume beside it).
//
// Default mode (the M1 gates):
//   1. Placement invariance end to end: greedy ids of one request are identical with the expert
//      cache cold (first request of a fresh Engine), warm (the same request again), warm-started
//      from the saved state of the first Engine (a second Engine), and small (a 1 GiB cache that
//      must evict every round, in a third Engine whose KV pool starts at 64 tokens and grows from the
//      expert frames, and which decodes without the CUDA graph, so graph replay = eager as well).
//   2. The RAM ledger refuses cleanly: an Engine asked to keep more RAM free than the machine has
//      fails before anything is pinned, with a message that says how much is missing.
//
//   ninfer_qwen4_exp_engine_real_test [TOKENS] [--generate N] [--max-context N]
//
// Bench mode: one Engine, then each job in order (a prompt and greedy output tokens with model
// stops disabled, so every request runs to N). Reports prefill and decode tok/s per job; the
// Program logs each request's expert-cache hit rate.
//
//   ninfer_qwen4_exp_engine_real_test --bench [--max-context N] [--generate N] [--no-graph]
//       [--expert-cache-mib N] [--no-expert-state] TOKENS...
//
// TOKENS is @FILE (ids separated by commas or whitespace) or a comma-separated id list.

#include "ninfer/engine.h"
#include "ninfer/types.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif
#include <thread>

#ifndef NINFER_SOURCE_DIR
#    define NINFER_SOURCE_DIR "."
#endif

using namespace ninfer;

namespace {

std::vector<TokenId> parse_tokens(const std::string& argument) {
    std::string text = argument;
    if (!text.empty() && text.front() == '@') {
        std::ifstream file(text.substr(1));
        if (!file) { throw std::invalid_argument("cannot read " + text.substr(1)); }
        text.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }
    for (char& ch : text) {
        if (ch == ',' || ch == '\n' || ch == '\r' || ch == '\t') { ch = ' '; }
    }
    std::vector<TokenId> out;
    std::stringstream stream(text);
    TokenId id = 0;
    while (stream >> id) { out.push_back(id); }
    return out;
}

double available_ram_gib() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) { return static_cast<double>(status.ullAvailPhys) / (1ULL << 30); }
#endif
    return 0.0;
}

// Between Engines: says how much host RAM came back after one was destroyed, waiting up to a minute
// for it to return near `before`.
void wait_for_ram(double before) {
    const auto start = std::chrono::steady_clock::now();
    double now       = available_ram_gib();
    while (now < before - 1.0 && std::chrono::steady_clock::now() - start < std::chrono::seconds(60)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        now = available_ram_gib();
    }
    std::printf("available RAM %.1f GiB (%.1f before the Engine) after %.1f s\n", now, before,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
}

EngineOptions base_options(const std::string& artifact, std::uint32_t max_context) {
    EngineOptions options;
    options.artifact_path         = artifact;
    options.max_context           = max_context;
    options.kv_capacity           = KvCapacityPolicy::explicit_capacity(max_context);
    options.max_concurrency       = 1;
    options.prefill_chunk         = 1024;
    options.kv_cache              = KvCacheStorage::Int8Group64;
    options.context_cache.enabled = false;
    options.pending_timeout_ms    = 3'600'000;
    return options;
}

struct Run {
    std::vector<TokenId> ids;
    GenerationTimings timings;
    double wall = 0;
};

Run generate(Engine& engine, const std::vector<TokenId>& prompt, std::uint32_t tokens) {
    RequestOptions request;
    request.execution.requested_output_tokens = tokens;
    request.execution.allow_prefix_reuse      = false;
    request.execution.sampling.temperature    = 0.0F;
    request.stop.include_model_defaults       = false;
    request.output.raw                        = true;
    const auto start = std::chrono::steady_clock::now();
    auto result      = engine.generate(engine.prepare_tokens(prompt, false), request);
    Run out;
    out.ids     = std::move(result.generated_token_ids);
    out.timings = result.timings;
    out.wall    = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return out;
}

void print_run(const char* label, const Run& run, std::size_t prompt) {
    const double decode = run.ids.size() > 1 && run.timings.decode_seconds > 0
                              ? static_cast<double>(run.ids.size() - 1) / run.timings.decode_seconds
                              : 0.0;
    // Wall-clock decode: N - 1 tokens over the request's wall time after its prefill, which counts
    // every host step between rounds (commit, the expert cache's update, n-gram reads).
    const double after_prefill = run.wall - run.timings.prefill_seconds;
    const double wall_decode   = run.ids.size() > 1 && after_prefill > 0
                                     ? static_cast<double>(run.ids.size() - 1) / after_prefill
                                     : 0.0;
    // FNV-1a over the generated ids: equal hashes across runs of one prompt mean equal outputs.
    std::uint64_t hash = 1469598103934665603ULL;
    for (const TokenId id : run.ids) {
        hash ^= static_cast<std::uint32_t>(id);
        hash *= 1099511628211ULL;
    }
    std::printf("%s: %zu prompt tokens, %zu generated; prefill %.2f s (%.1f tok/s), decode %.2f s (%.2f tok/s), "
                "after prefill %.2f s (%.2f tok/s), wall %.2f s, ids %016llx\n",
                label, prompt, run.ids.size(), run.timings.prefill_seconds,
                run.timings.prefill_seconds > 0 ? static_cast<double>(prompt) / run.timings.prefill_seconds : 0.0,
                run.timings.decode_seconds, decode, after_prefill, wall_decode, run.wall,
                static_cast<unsigned long long>(hash));
    std::fflush(stdout);
}

bool same(const char* label, const std::vector<TokenId>& reference, const std::vector<TokenId>& ids) {
    if (ids == reference) {
        std::printf("%s: %zu ids identical\n", label, ids.size());
        return true;
    }
    std::size_t first = 0;
    while (first < ids.size() && first < reference.size() && ids[first] == reference[first]) { ++first; }
    std::printf("FAIL: %s differs from the cold run at generated token %zu\n", label, first);
    return false;
}

int bench(const std::string& artifact, const std::vector<std::string>& args) {
    std::uint32_t max_context = 33'792, tokens = 512;
    bool graph = true, state = true;
    std::optional<std::uint64_t> cache;
    std::vector<std::string> jobs;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        const auto value     = [&] {
            if (i + 1 >= args.size()) { throw std::invalid_argument(a + " needs a value"); }
            return std::stoull(args[++i]);
        };
        if (a == "--max-context") {
            max_context = static_cast<std::uint32_t>(value());
        } else if (a == "--generate") {
            tokens = static_cast<std::uint32_t>(value());
        } else if (a == "--expert-cache-mib") {
            cache = value() << 20U;
        } else if (a == "--no-graph") {
            graph = false;
        } else if (a == "--no-expert-state") {
            state = false;
        } else {
            jobs.push_back(a);
        }
    }
    EngineOptions options                  = base_options(artifact, max_context);
    options.use_cuda_graph                 = graph;
    options.flash_next.expert_cache_bytes  = cache;
    options.flash_next.save_expert_state   = state;
    const auto load = std::chrono::steady_clock::now();
    Engine engine(options);
    std::printf("engine loaded in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - load).count());
    for (const std::string& job : jobs) {
        const auto prompt = parse_tokens(job);
        const Run run     = generate(engine, prompt, tokens);
        print_run(job.c_str(), run, prompt.size());
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact_env = std::getenv("NINFER_TEST_QWEN4EXP_ARTIFACT");
    if (artifact_env == nullptr) {
        std::printf("SKIP: set NINFER_TEST_QWEN4EXP_ARTIFACT to a Qwen3.8-Flash-Next artifact\n");
        return 77;
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        const std::string artifact = artifact_env;
        std::vector<std::string> args(argv + 1, argv + argc);
        if (!args.empty() && args.front() == "--bench") {
            return bench(artifact, std::vector<std::string>(args.begin() + 1, args.end()));
        }
        std::string tokens_arg = "@" + std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/qwen4_exp/ppl_tokens-chat.ids";
        std::uint32_t generate_tokens = 48, max_context = 8192, prompt_tokens = 300;
        for (std::size_t i = 0; i < args.size(); ++i) {
            if (args[i] == "--generate" && i + 1 < args.size()) {
                generate_tokens = static_cast<std::uint32_t>(std::stoul(args[++i]));
            } else if (args[i] == "--max-context" && i + 1 < args.size()) {
                max_context = static_cast<std::uint32_t>(std::stoul(args[++i]));
            } else if (args[i] == "--prompt" && i + 1 < args.size()) {
                prompt_tokens = static_cast<std::uint32_t>(std::stoul(args[++i]));
            } else {
                tokens_arg = args[i];
            }
        }
        auto prompt = parse_tokens(tokens_arg);
        if (prompt.size() > prompt_tokens) { prompt.resize(prompt_tokens); }
        if (prompt.empty()) { throw std::invalid_argument("empty prompt"); }
        // A long prompt (every fixture text, ~5,100 tokens): past QSA's 2,048-token selection
        // budget, so attention drops blocks, and past one KV chunk per plane, so the small-cache
        // Engine's 64-token KV base must grow from the expert frames.
        std::vector<TokenId> long_prompt;
        for (const char* name : {"ppl_tokens-code", "ppl_tokens-doc", "ppl_tokens-chat", "infernix_texts-code",
                                 "infernix_texts-doc", "infernix_texts-chat"}) {
            const auto part = parse_tokens("@" + std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/qwen4_exp/" + name + ".ids");
            long_prompt.insert(long_prompt.end(), part.begin(), part.end());
        }
        const std::uint32_t long_tokens = 24;
        std::vector<TokenId> long_cold;
        const auto state_path = std::filesystem::temp_directory_path() / "ninfer-qwen4exp-test.expert-state";
        std::filesystem::remove(state_path);
        bool ok = true;

        // 2 first: the refusal costs no load. More reserve than the machine has must refuse before
        // anything is pinned, naming the shortfall.
        {
            EngineOptions options                 = base_options(artifact, max_context);
            options.flash_next.ram_headroom_bytes = 1ULL << 40; // 1 TiB
            try {
                Engine refused(options);
                std::printf("FAIL: an Engine keeping 1 TiB of RAM free started\n");
                ok = false;
            } catch (const std::exception& error) {
                const std::string what = error.what();
                const bool clean = what.find("cannot start") != std::string::npos && what.find("short") != std::string::npos;
                std::printf("%s: RAM ledger refusal: %s\n", clean ? "ok" : "FAIL", what.c_str());
                ok = ok && clean;
            }
        }

        const double ram_before = available_ram_gib();
        std::vector<TokenId> cold;
        {
            EngineOptions options                = base_options(artifact, max_context);
            options.flash_next.expert_state_path = state_path;
            Engine engine(options);
            const Run first = generate(engine, prompt, generate_tokens);
            print_run("cold cache", first, prompt.size());
            cold = first.ids;
            ok   = ok && cold.size() == generate_tokens;
            const Run again = generate(engine, prompt, generate_tokens);
            print_run("warm cache (same Engine)", again, prompt.size());
            ok = same("warm cache (same Engine)", cold, again.ids) && ok;
            const Run long_run = generate(engine, long_prompt, long_tokens);
            print_run("long prompt", long_run, long_prompt.size());
            long_cold = long_run.ids;
            ok        = ok && long_cold.size() == long_tokens;
        } // the Engine saves the expert state at shutdown
        wait_for_ram(ram_before);
        if (!std::filesystem::exists(state_path)) {
            std::printf("FAIL: the expert state was not saved at shutdown\n");
            ok = false;
        }
        {
            EngineOptions options                = base_options(artifact, max_context);
            options.flash_next.expert_state_path = state_path;
            Engine engine(options);
            const Run run = generate(engine, prompt, generate_tokens);
            print_run("warm start (saved state)", run, prompt.size());
            ok = same("warm start (saved state)", cold, run.ids) && ok;
        }
        wait_for_ram(ram_before);
        {
            EngineOptions options                 = base_options(artifact, max_context);
            options.flash_next.save_expert_state  = false;
            options.flash_next.expert_cache_bytes = 1ULL << 30;
            options.flash_next.kv_base_tokens     = 64;
            options.use_cuda_graph                = false;
            Engine engine(options);
            const Run run = generate(engine, prompt, generate_tokens);
            print_run("small cache (1 GiB, evicting), elastic KV, no CUDA graph", run, prompt.size());
            ok = same("small cache (1 GiB, evicting), elastic KV, no CUDA graph", cold, run.ids) && ok;
            const Run graphless = generate(engine, prompt, generate_tokens);
            ok = same("small cache, second request", cold, graphless.ids) && ok;
            const Run long_run = generate(engine, long_prompt, long_tokens);
            print_run("long prompt, small cache, KV grown", long_run, long_prompt.size());
            ok = same("long prompt, small cache, KV grown", long_cold, long_run.ids) && ok;
        }
        std::filesystem::remove(state_path);
        std::printf(ok ? "qwen4_exp engine checks passed\n" : "FAIL: qwen4_exp engine checks\n");
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
