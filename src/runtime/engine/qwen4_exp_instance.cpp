// Adapted from Infernix a3edb450 src/runtime/engine/qwen4_exp_instance.cpp (Apache-2.0).
// Modified for NInfer-3090: full RAM mode only (no SSD tier: a ledger that cannot hold every expert
// refuses); no prefix cache, MTP, n-gram drafts or vision; the options NInfer's serving can set
// that this model does not support are refused with the flag to change; NInfer's startup phases.

#include "runtime/engine/qwen4_exp_instance.h"

#include "artifact/reader.h"
#include "core/device.h"
#include "core/host_memory.h"
#include "core/startup.h"
#include "core/vram_budget.h"
#include "models/qwen4_exp/memory_plan.h"
#include "models/qwen4_exp/program/ngram_volume.h"
#include "models/registry.h"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::runtime {
namespace {

// Planning allowance for the Program's pinned buffers (io staging, sampling words).
constexpr std::uint64_t kProgramPinnedAllowance = 256ULL << 20;
// The materializer's upload slots and direct-read bounce buffer during the load.
constexpr std::uint64_t kLoadStagingBytes = 4ULL * (64ULL << 20) + (8ULL << 20);
// Below this many expert frames most routed experts miss the VRAM cache and decode slows down
// markedly (Infernix design section 19.2).
constexpr std::uint32_t kFewFrames = 2048;

unsigned long long mib(std::uint64_t bytes) { return static_cast<unsigned long long>(bytes >> 20); }

void report(const std::string& text) {
    std::fprintf(stderr, "[qwen3.8-flash-next] %s\n", text.c_str());
    std::fflush(stderr);
}

// What this release of the model serves: one text request at a time, plain decode, INT8 KV, no
// prefix reuse. Every option a user can set otherwise is refused with what to change.
void validate(const EngineOptions& options) {
    const auto refuse = [](const std::string& why) {
        throw std::invalid_argument("Qwen3.8-Flash-Next: " + why);
    };
    if (options.purpose != EnginePurpose::Generation) {
        refuse("serves generation only (perplexity scoring is not available for this model yet)");
    }
    if (options.max_concurrency != 1) { refuse("serves one request at a time in this release: use --max-concurrency 1"); }
    if (options.kv_cache != KvCacheStorage::Int8Group64) { refuse("its attention cache is INT8: use --kv-dtype int8"); }
    if (options.speculative.backend != SpeculativeBackend::None || options.speculative.ngram_draft_tokens != 0) {
        refuse("has no speculative decoding in this release: remove --spec and --ngram-draft-tokens");
    }
    if (options.kv_capacity.mode == KvCapacityMode::Automatic) {
        refuse("sizes its KV pool itself from free VRAM (it grows and shrinks per request): remove "
               "--kv-capacity auto");
    }
    if (options.enable_vision) { refuse("serves text only in this release: remove --vision"); }
    if (!options.grafts.empty()) { refuse("has no prompt grafts: remove --graft"); }
    if (options.devices.size() > 1) { refuse("runs on one GPU: remove --devices"); }
    if (options.context_store.enabled()) { refuse("has no context store (prefix reuse is off): remove --context-store"); }
    if (options.lm_head_q4 || options.lm_head_q6 || options.embedding_q4 || options.embedding_q6 ||
        options.mtp_experts_q4 || options.mlp_a8_decode || options.prefill_cublas || options.gdn_state_fp16 || !options.prefill_a8 ||
        !options.prefill_cublas_projections) {
        refuse("its weights are the GGUF's, stored exactly: remove the load-time weight flags (--lm-head-q4/q6, "
               "--embedding-q4/q6, --mtp-experts-q4, --mlp-a8-decode, --prefill-cublas, --gdn-state-fp16, "
               "--no-prefill-a8, --no-prefill-cublas-projections)");
    }
}

} // namespace

bool artifact_is_qwen4_exp(const std::filesystem::path& artifact) noexcept {
    try {
        const artifact::Reader reader(artifact);
        return models::qwen4_exp::is_qwen4_exp(reader);
    } catch (...) {
        return false;
    }
}

Qwen4ExpInstance::Qwen4ExpInstance(std::unique_ptr<models::qwen4_exp::Model> source, const EngineOptions& options)
    : model(std::move(source)), parameters(*model),
      frontend(models::qwen3_5::make_frontend(model->resources(),
                                              {.chat_template_path  = options.chat_template_path,
                                               .grammar_cache_bytes = options.grammar_cache_bytes,
                                               .architecture        = models::Architecture::Qwen4Exp,
                                               .vision_enabled      = false,
                                               .max_context         = options.max_context})),
      capacity(options.max_context) {}

Qwen4ExpInstance::~Qwen4ExpInstance() = default;

ConstructedQwen4Exp construct_qwen4_exp(const EngineOptions& options, DeviceContext& device) {
    validate(options);
    const auto start = std::chrono::steady_clock::now();
    const auto& next = options.flash_next;
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    const artifact::Reader reader(options.artifact_path);
    if (!models::qwen4_exp::is_qwen4_exp(reader)) { throw std::invalid_argument("artifact is not Qwen4ExpForCausalLM"); }
    inspect.complete();
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    auto plan = models::qwen4_exp::plan_load(reader);
    binding.complete();
    const auto& config = plan.config();

    // The RAM ledger: every expert is pinned, beside a reserve for Windows and every other planned
    // allocation, before anything is pinned or read.
    models::qwen4_exp::HostMemoryDemand demand;
    demand.reserve      = next.ram_headroom_bytes;
    demand.expert_banks = plan.pinned_bytes();
    demand.later_pinned = kProgramPinnedAllowance;
    demand.pageable     = models::qwen4_exp::NgramVolume::cache_bytes(config.ple.table);
    demand.load_staging = kLoadStagingBytes;
    const auto ledger   = models::qwen4_exp::plan_host_memory(host_memory_snapshot(), demand);
    report(ledger.summary());
    report(ledger.describe());
    if (!ledger.fits) { throw std::runtime_error("Qwen3.8-Flash-Next cannot start: " + ledger.summary()); }
    // A small reserve trims other programs' working sets; keep NInfer's own pageable memory resident
    // so its heap and driver pages never hard-fault from the page file.
    if (const auto refused = reserve_process_working_set(demand.process_growth); !refused.empty()) {
        report("Windows would not guarantee " + std::to_string(mib(demand.process_growth)) +
               " MiB of resident working memory (" + refused + "); if RAM runs short, responses may slow down");
    }

    models::qwen4_exp::ProgramOptions program_options;
    program_options.device          = device.device;
    program_options.max_context     = options.max_context;
    program_options.max_concurrency = options.max_concurrency;
    program_options.prefill_chunk   = options.prefill_chunk;
    program_options.use_cuda_graph  = options.use_cuda_graph;
    program_options.vram_headroom   = next.vram_headroom_bytes;
    program_options.expert_cache_bytes = next.expert_cache_bytes;
    program_options.kv_base_tokens  = next.kv_base_tokens;
    program_options.ngram_volume    = next.ngram_volume_path.empty()
                                          ? models::qwen4_exp::default_ngram_volume(options.artifact_path)
                                          : next.ngram_volume_path;
    if (next.save_expert_state) {
        std::filesystem::path state = next.expert_state_path;
        if (state.empty()) {
            state = options.artifact_path;
            state += ".expert-state";
        }
        // The state belongs to the artifact: its experts and their records.
        std::error_code error;
        const auto size = std::filesystem::file_size(options.artifact_path, error);
        const auto time = std::filesystem::last_write_time(options.artifact_path, error);
        program_options.expert_state          = state;
        program_options.expert_state_identity = "artifact=" + std::filesystem::absolute(options.artifact_path).string() +
                                                ";size=" + std::to_string(error ? 0U : size) +
                                                ";mtime=" + std::to_string(error ? 0 : time.time_since_epoch().count());
    }

    // The VRAM check, before the weights are read: the dense weights, the Program's fixed
    // allocations, the reserve for graph executables and the display headroom must leave room for
    // the expert cache.
    const auto device_plan = models::qwen4_exp::Program::plan_device(program_options, config,
                                                                     plan.resources().public_token_count,
                                                                     plan.expert_record_strides());
    const std::uint64_t dense = plan.device_bytes();
    auto vram                 = open_vram_budget_source(device.device);
    const VramSnapshot before = vram->query();
    {
        models::qwen4_exp::VramDemand check_demand;
        check_demand.headroom    = next.vram_headroom_bytes;
        check_demand.graphs      = device_plan.graph_bound;
        check_demand.fixed       = dense + device_plan.fixed_bytes();
        check_demand.frame_bytes = device_plan.expert_record_stride;
        check_demand.max_frames  = device_plan.max_frames;
        const auto check         = models::qwen4_exp::size_expert_frames(before, check_demand);
        if (check.frame_bytes < 0) {
            throw std::runtime_error(
                "Qwen3.8-Flash-Next does not fit in device memory: " + std::to_string(mib(before.device_free)) +
                " MiB free; dense weights " + std::to_string(mib(dense)) + " MiB, KV " + std::to_string(mib(device_plan.kv)) +
                ", workspace " + std::to_string(mib(device_plan.workspace)) + ", expert staging " +
                std::to_string(mib(device_plan.staging)) + " (" + check.describe() +
                "). Lower --max-context or --prefill-chunk, or free VRAM");
        }
        if (check.frames < kFewFrames) {
            report("decode will be slow: only " + std::to_string(check.frames) + " expert frames fit in VRAM (" +
                   std::to_string(kFewFrames) + " or more recommended); lower --max-context, or free VRAM");
        }
    }

    // The dense arena must land in device memory (WDDM can place it in system memory silently).
    // Device memory is allocated before the expert banks are pinned.
    SpillGuard guard(*vram);
    guard.begin();
    auto model = models::qwen4_exp::materialize_model(std::move(plan), device, &options.startup_observer);
    device.synchronize();
    if (const std::uint64_t spilled = guard.end(dense); spilled != 0) {
        throw std::runtime_error(std::to_string(mib(spilled)) +
                                 " MiB of the dense weights were placed in shared system memory by the driver's "
                                 "sysmem fallback: free VRAM and start again");
    }
    const std::uint64_t after_weights = vram->query().device_free;

    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<Qwen4ExpInstance>(std::move(model), options);
    frontend.complete();

    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    instance->program =
        std::make_unique<models::qwen4_exp::Program>(instance->parameters, device, std::move(program_options));
    program.complete();
    const auto& sizing                 = instance->program->vram_sizing();
    const std::uint64_t frames_bytes   = static_cast<std::uint64_t>(sizing.frames) * device_plan.expert_record_stride;
    const std::uint64_t after_startup  = vram->query().device_free;
    {
        const double gib = static_cast<double>(1ULL << 30);
        char line[400];
        std::snprintf(line, sizeof(line),
                      "VRAM: %llu MiB in use before loading; model %.2f GiB, KV %.2f GiB (up to %.2f), workspace "
                      "%.2f GiB, expert cache %u experts (%.1f GiB); %llu MiB free after startup",
                      mib(before.device_total - before.device_free), static_cast<double>(dense) / gib,
                      static_cast<double>(device_plan.kv) / gib, static_cast<double>(device_plan.kv_max) / gib,
                      static_cast<double>(device_plan.workspace) / gib, sizing.frames,
                      static_cast<double>(frames_bytes) / gib, mib(after_startup));
        report(line);
    }

    const MemorySummary memory               = instance->program->memory_summary();
    auto& resolution                         = instance->kv_capacity_resolution;
    resolution.mode                          = KvCapacityMode::Explicit;
    resolution.main_page_groups              = memory.kv_capacity_page_groups;
    resolution.maximum_main_page_groups      = memory.kv_capacity_max_page_groups;
    resolution.resolved_tokens               = memory.kv_capacity;
    resolution.runtime_reservation_bytes     = device_plan.fixed_bytes() + frames_bytes;
    resolution.available_after_weights_bytes = after_weights;
    resolution.available_after_startup_bytes = after_startup;
    resolution.automatic_headroom_bytes      = sizing.headroom;
    resolution.planned_slack_bytes = after_weights > resolution.runtime_reservation_bytes
                                         ? after_weights - resolution.runtime_reservation_bytes
                                         : 0;

    const auto& stats = instance->model->storage_stats();
    LoadSummary summary;
    summary.architecture         = std::string(models::architecture_name(models::Architecture::Qwen4Exp));
    summary.model_name           = instance->model->info().name;
    summary.load_seconds         = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.pinned_weight_bytes  = stats.pinned_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    summary.weight_formats       = {"ggml_blocks_v1", "ggml_expert_record_v1", "bf16", "fp32"};
    return {std::move(instance), std::move(summary)};
}

} // namespace ninfer::runtime
