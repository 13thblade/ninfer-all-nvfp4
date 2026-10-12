// Adapted from Infernix a3edb450 src/models/qwen4_exp/program/program.cpp and program_impl.h
// (Apache-2.0).
// Modified for NInfer-3090: the execution contract is NInfer's (see program.h); one lane over the
// single-sequence forward of PR 8; prefix reuse, MTP, n-gram drafts, vision, the CPU miss service,
// the SSD tier, landing frames, prefill streaming, the layer walk and /v1/decide are not ported.
// Kept from Infernix: the device plan and its spill checks, the elastic VMM KV pool that borrows
// from the expert frames, the expert cache (LFRU, decode promotion budget and cadence, warm-start
// state), the VRAM control law at round boundaries, the link probe, the power-throttling opt-out
// and the decode CUDA graph.

#include "models/qwen4_exp/program/program.h"

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/link_probe.h"
#include "core/nvtx.h"
#include "core/paged_kv_cache.h"
#include "core/pinned_upload.h"
#include "core/power_throttling.h"
#include "core/tensor.h"
#include "core/vmm_arena.h"
#include "core/vram_budget.h"
#include "models/qwen4_exp/execution/forward.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/frontend/ngram_hash.h"
#include "models/qwen4_exp/model.h"
#include "models/qwen4_exp/program/expert_cache/expert_state.h"
#include "models/qwen4_exp/program/expert_residency.h"
#include "models/qwen4_exp/program/ngram_volume.h"
#include "models/qwen4_exp/program/vram_monitor.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen4_exp {
namespace detail {

using Clock = std::chrono::steady_clock;

struct BasePlanImpl {
    runtime::RequestPlanSummary summary;
    qwen3_5::PreparedContextCache context_cache;
    ops::SamplingConfig sampling;
    std::shared_ptr<const std::vector<TokenId>> prompt;
};

struct ResumeStateImpl {
    std::uint32_t frontier = 0;
};

struct ContractAccess {
    static SequenceHandle make_sequence(const void* owner, std::uint32_t lane, std::uint64_t epoch) noexcept {
        SequenceHandle out;
        out.owner_ = owner;
        out.lane_  = runtime::LaneId{lane};
        out.epoch_ = epoch;
        return out;
    }
    static const void* owner(const SequenceHandle& h) noexcept { return h.owner_; }
    static std::uint32_t lane(const SequenceHandle& h) noexcept { return h.lane_.value; }
    static std::uint64_t epoch(const SequenceHandle& h) noexcept { return h.epoch_; }

    static PendingBatch make_pending(const void* owner, std::uint64_t transaction, SequenceHandle row,
                                     std::span<const TokenId> tokens, runtime::ExecutionTiming timing,
                                     bool constraint_failed) {
        PendingBatch out;
        out.owner_                = owner;
        out.transaction_          = transaction;
        out.row_count_            = 1;
        out.rows_[0]              = row;
        out.tokens_               = tokens;
        out.timing_               = timing;
        out.constraint_failed_[0] = constraint_failed;
        return out;
    }
    static const void* owner(const PendingBatch& p) noexcept { return p.owner_; }
    static std::uint64_t transaction(const PendingBatch& p) noexcept { return p.transaction_; }
    static std::span<const SequenceHandle> rows(const PendingBatch& p) noexcept { return {p.rows_.data(), p.row_count_}; }
    static void consume(PendingBatch& p) noexcept {
        p.owner_       = nullptr;
        p.transaction_ = 0;
        p.row_count_   = 0;
        p.tokens_      = {};
    }
};

namespace {

inline std::uint64_t elapsed_ns(Clock::time_point start) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

inline std::int32_t dim(std::uint64_t v) { return static_cast<std::int32_t>(v); }
inline std::size_t align(std::size_t v, std::size_t a = 256) { return (v + a - 1) / a * a; }

double seconds(std::uint64_t ns) { return static_cast<double>(ns) * 1e-9; }

// The KV pool's chunk: the VMM allocation granularity of the RTX 3090 (2 MiB).
constexpr std::size_t kKvChunkBytes = 2ULL << 20;
// Infernix's elastic pool backs this many tokens at startup and grows above it.
constexpr std::uint32_t kKvBaseTokens = 32768;
// Growth unit (4,096 tokens).
constexpr std::uint32_t kKvGrowPages = 64;
// Misses staged per pass of a layer's experts.
constexpr std::int32_t kStagingSlots = 64;
// Promotions per layer call of a decode round (Infernix design section 19.2: one measured fastest).
constexpr std::size_t kDecodePromotionsPerLayer = 1;
// Once the frames are full, decode promotes only every interval-th round on the reference link.
constexpr std::uint64_t kDecodePromotionInterval = 4;
constexpr double kReferenceLinkBytesPerSecond    = 27.5e9;
// Calls this narrow promote at the decode rate; wider prefill calls promote nothing.
constexpr std::int32_t kServedCallColumns = 8;
// A seeded count is capped, so an expert used heavily in an old session yields to new uses.
constexpr std::uint32_t kSeedCountCap = 16;
// The share of the frames a warm start fills.
constexpr double kSeedShare = 0.5;
constexpr std::chrono::minutes kExpertStateInterval{10};

ops::SamplingConfig translate(const ResolvedSamplingParameters& s) {
    if (!std::isfinite(s.temperature) || !std::isfinite(s.top_p) || !std::isfinite(s.min_p) ||
        !std::isfinite(s.presence_penalty) || !std::isfinite(s.frequency_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    if (s.top_p < 0.0F || s.top_p > 1.0F) { throw std::invalid_argument("top_p must be in [0,1]"); }
    if (s.min_p < 0.0F || s.min_p > 1.0F) { throw std::invalid_argument("min_p must be in [0,1]"); }
    ops::SamplingConfig out;
    out.temperature       = s.temperature;
    out.top_k             = s.top_k;
    out.top_p             = s.top_p;
    out.min_p             = s.min_p;
    out.presence_penalty  = s.presence_penalty;
    out.frequency_penalty = s.frequency_penalty;
    out.seed              = s.seed;
    return out;
}

// The paged INT8-G64 KV of every attention layer and the block table, each plane page-major so
// pages [0, n) are a prefix of every plane: K and V codes, their FP16 group scales and the pooled
// index keys. Planes start on chunk boundaries, so the elastic pool maps each plane's prefix.
struct KvLayout {
    struct Plane {
        std::size_t offset     = 0;
        std::size_t page_bytes = 0;
    };
    std::uint32_t pages = 0;
    std::vector<std::array<Plane, 5>> layers; // k, v, k scales, v scales, pooled keys
    std::size_t table  = 0;
    std::size_t bytes  = 0;

    static KvLayout plan(const TextConfig& c, std::uint32_t pages) {
        KvLayout out;
        out.pages = pages;
        const std::size_t D = c.attention.head_dim, Hkv = c.attention.kv_heads;
        const std::size_t pooled = c.qsa.index_head_dim / c.qsa.compress_ratio;
        const std::size_t codes  = D * kPagedKVPageSize * Hkv;
        const std::size_t scales = (D / 64) * kPagedKVPageSize * Hkv * 2;
        const std::size_t keys   = pooled * kPagedKVPageSize * 2;
        std::size_t at           = align(static_cast<std::size_t>(pages) * 4, kKvChunkBytes);
        out.table                = 0;
        for (std::uint32_t l = 0; l < c.attention_layers; ++l) {
            std::array<Plane, 5> planes{};
            const std::size_t per_page[5] = {codes, codes, scales, scales, keys};
            for (int p = 0; p < 5; ++p) {
                planes[p] = {at, per_page[p]};
                at        = align(at + per_page[p] * pages, kKvChunkBytes);
            }
            out.layers.push_back(planes);
        }
        out.bytes = at;
        return out;
    }

    // The chunks pages [0, pages) of every plane and the block table need.
    [[nodiscard]] std::vector<std::uint8_t> chunks(std::uint32_t backed) const {
        std::vector<std::uint8_t> need(bytes / kKvChunkBytes, 0);
        const auto mark = [&](std::size_t offset, std::size_t n) {
            if (n == 0) { return; }
            for (std::size_t c = offset / kKvChunkBytes; c <= (offset + n - 1) / kKvChunkBytes; ++c) { need[c] = 1; }
        };
        mark(table, static_cast<std::size_t>(pages) * 4);
        for (const auto& planes : layers) {
            for (const auto& plane : planes) { mark(plane.offset, plane.page_bytes * backed); }
        }
        return need;
    }

    [[nodiscard]] std::size_t mapped_bytes(std::uint32_t backed) const {
        const auto need = chunks(backed);
        return static_cast<std::size_t>(std::count(need.begin(), need.end(), std::uint8_t{1})) * kKvChunkBytes;
    }
};

// Per-call inputs: one pinned host block uploaded by the SMs (upload_pinned) before each call.
struct IoLayout {
    std::size_t ids = 0, positions = 0, rope = 0, block_rope = 0, logit_columns = 0, ngram = 0, bytes = 0;

    static IoLayout plan(std::size_t columns, std::size_t row_bytes) {
        IoLayout out;
        out.positions     = align(4 * columns);
        out.rope          = align(out.positions + 4 * columns);
        out.block_rope    = align(out.rope + 12 * columns);
        out.logit_columns = align(out.block_rope + 12);
        out.ngram         = align(out.logit_columns + 4);
        out.bytes         = align(out.ngram + row_bytes * columns);
        return out;
    }
};

struct Layout {
    std::int32_t chunk = 0, token_domain = 0;
    std::uint32_t pages = 0, base_pages = 0;
    KvLayout kv;
    IoLayout io;
    std::size_t state = 0, gdn_conv = 0, gdn_recurrent = 0, ple = 0, tail = 0;
    std::size_t work = 0, staging = 0, mask_words = 0;
    std::uint64_t frame_stride = 0;
    bool vmm                   = false;
    ProgramDevicePlan bytes;
};

Layout plan_layout(const ProgramOptions& o, const TextConfig& c, std::uint32_t public_tokens,
                   std::span<const std::uint64_t> strides) {
    if (o.max_concurrency != 1) {
        throw std::invalid_argument(
            "Qwen3.8-Flash-Next serves one request at a time in this release: use --max-concurrency 1");
    }
    if (o.max_context == 0 || o.prefill_chunk == 0) {
        throw std::invalid_argument("Qwen3.8-Flash-Next: max_context and prefill_chunk must be nonzero");
    }
    if (o.max_context > c.max_position_embeddings) {
        throw std::invalid_argument("Qwen3.8-Flash-Next: max_context " + std::to_string(o.max_context) +
                                    " exceeds the model's " + std::to_string(c.max_position_embeddings) +
                                    " positions");
    }
    if (public_tokens == 0 || public_tokens > c.vocab_size) {
        throw std::invalid_argument("Qwen3.8-Flash-Next: the tokenizer's domain exceeds the vocabulary");
    }
    Layout d;
    d.chunk        = static_cast<std::int32_t>(std::min(o.prefill_chunk, o.max_context));
    d.token_domain = static_cast<std::int32_t>(public_tokens);
    d.pages        = (o.max_context + kPagedKVPageSize - 1) / kPagedKVPageSize;
    const std::uint32_t base_tokens = o.kv_base_tokens != 0 ? o.kv_base_tokens : kKvBaseTokens;
    d.base_pages = std::min(d.pages, std::max(1U, base_tokens / kPagedKVPageSize));
    d.kv         = KvLayout::plan(c, d.pages);
    const NgramHash hash(c.ple.ngram, 0);
    d.io = IoLayout::plan(static_cast<std::size_t>(d.chunk), static_cast<std::size_t>(c.ple.table.row_bytes) * hash.heads());
    d.gdn_conv      = std::size_t(c.gdn.conv_channels()) * (c.gdn.conv_kernel - 1) * 2;
    d.gdn_recurrent = std::size_t(c.gdn.key_head_dim) * c.gdn.value_head_dim * c.gdn.value_heads * 4;
    d.ple           = std::size_t(c.residual_width()) * c.ple.conv_span() * 2;
    d.tail          = std::size_t(c.qsa.index_head_dim) * (c.qsa.compress_ratio - 1) * 2;
    d.state = align(c.gdn_layers * (align(d.gdn_conv) + align(d.gdn_recurrent)) + align(d.ple) +
                    c.attention_layers * align(d.tail));
    for (const std::uint64_t stride : strides) { d.frame_stride = std::max(d.frame_stride, stride); }
    if (d.frame_stride == 0) { throw std::invalid_argument("Qwen3.8-Flash-Next: no expert banks"); }
    d.staging    = static_cast<std::size_t>(kStagingSlots) * d.frame_stride;
    d.mask_words = (static_cast<std::size_t>(public_tokens) + 31) / 32;
    // The forward at the widest call, sampling and the logits cast; the static arena covers prefill
    // chunks too (no wide arena lent from the frames).
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, o.device) != cudaSuccess || sms <= 0) {
        throw std::runtime_error("Qwen3.8-Flash-Next: cannot read the device's multiprocessor count");
    }
    d.work = execution::Forward::workspace_bytes(c, d.chunk, d.pages * kPagedKVPageSize,
                                                 DeviceExecutionView{nullptr, sms}) +
             ops::sampling_workspace_capacity_bytes(d.token_domain, 1, 1);
    d.vmm = VmmArena::supported(o.device);

    auto& b     = d.bytes;
    b.kv_pages  = d.pages;
    b.kv_max    = d.kv.bytes;
    b.kv        = d.vmm ? d.kv.mapped_bytes(d.base_pages) : d.kv.bytes;
    b.workspace = d.work;
    b.staging   = d.staging;
    b.state     = d.state;
    b.io        = d.io.bytes + sizeof(float) * c.vocab_size + 2ULL * c.vocab_size + 256 * 4 +
                  sizeof(ops::SamplingConfig) + 4ULL * public_tokens + 4 * d.mask_words + 4096;
    b.residency            = ExpertResidency::table_bytes(c, d.chunk);
    b.expert_record_stride = d.frame_stride;
    b.max_frames           = ExpertResidency::max_frames(c);
    b.graph_bound          = graph_bound(1, 1, 0);
    return d;
}

} // namespace

class ProgramImpl {
public:
    enum class Phase : std::uint8_t { Free, Binding, Prefill, Decode, Pending, Finishable };

    struct Lane {
        Phase phase         = Phase::Free;
        std::uint64_t epoch = 0;
        std::shared_ptr<const BasePlanImpl> base;
        runtime::BeginSummary begin;
        std::vector<TokenId> history; // prompt, then committed output
        std::uint32_t prompt_tokens = 0;
        std::uint32_t state_tokens  = 0; // positions already in the model state
        std::uint32_t extent        = 0; // positions the request may ever hold
        ops::SamplingConfig sampling;
        bool permit = false;
        std::uint64_t prefill_ns = 0, decode_ns = 0;
        Clock::time_point admitted;
        double first_token_seconds = 0;
        ExpertResidency::Stats cache_at_admission;
        ExpertResidency::Stats cache_at_decode; // at the first token's commit: the decode phase's base
        NgramVolume::Counters ngram_at_decode;
        NgramVolume::Counters ngram_at_admission;
    };

    ProgramImpl(const execution::Parameters& parameters, DeviceContext& device, ProgramOptions options)
        : parameters_(parameters), device_(device), c_(parameters.model.config()), options_(std::move(options)),
          volume_(options_.ngram_volume, c_.ple.table), hash_(c_.ple.ngram, 0),
          plan_(plan_layout(options_, c_, parameters.model.resources().public_token_count, strides(parameters))) {
        exempt_process_from_power_throttling();
        // A read after 100 ms idle within a minute of a request keeps the volume's drive awake.
        volume_.set_keepalive(std::chrono::milliseconds(100), std::chrono::milliseconds(60000));
        vocab_ = dim(c_.vocab_size);

        // Every fixed allocation is checked against the plan and against device free memory (an
        // allocation the driver placed in system memory fails startup).
        vram_ = open_vram_budget_source(device_.device);
        SpillGuard guard(*vram_);
        guard.begin();
        std::uint64_t allocated = 0;

        state_backing_ = DeviceBuffer(plan_.state);
        allocated += plan_.state;
        if (plan_.vmm) {
            kv_vmm_ = std::make_unique<VmmRange>(device_.device, plan_.kv.bytes, kKvChunkBytes);
            kv_base_ = static_cast<std::byte*>(kv_vmm_->base());
            if (!map_kv(plan_.base_pages)) { throw std::runtime_error("Qwen3.8-Flash-Next: the KV pool's base does not fit"); }
            allocated += kv_vmm_->mapped_bytes();
        } else {
            kv_backing_ = DeviceBuffer(plan_.kv.bytes);
            kv_backing_.fill(0);
            kv_base_ = static_cast<std::byte*>(kv_backing_.p);
            allocated += plan_.kv.bytes;
        }
        backed_pages_ = plan_.vmm ? plan_.base_pages : plan_.pages;

        io_device_ = DeviceBuffer(plan_.bytes.io);
        allocated += plan_.bytes.io;
        io_host_ = PinnedHostBuffer(plan_.io.bytes);
        host_token_ = PinnedHostBuffer(64);
        mask_host_  = PinnedHostBuffer(4 * plan_.mask_words + 64);
        config_host_ = PinnedHostBuffer(sizeof(ops::SamplingConfig) + 64);
        auto* io = static_cast<std::byte*>(io_device_.p);
        std::size_t at = align(plan_.io.bytes);
        logits32_ = Tensor(io + at, DType::FP32, {vocab_, 1});
        at = align(at + sizeof(float) * c_.vocab_size);
        logits16_ = Tensor(io + at, DType::BF16, {vocab_, 1});
        at = align(at + 2ULL * c_.vocab_size);
        sampled_ = Tensor(io + at, DType::I32, {1});
        at = align(at + 4);
        sample_position_ = Tensor(io + at, DType::I32, {1});
        at = align(at + 4);
        configs_ = reinterpret_cast<ops::SamplingConfig*>(io + at);
        at = align(at + sizeof(ops::SamplingConfig));
        token_counts_ = Tensor(io + at, DType::I32, {plan_.token_domain});
        at = align(at + 4ULL * plan_.token_domain);
        masks_ = reinterpret_cast<std::uint32_t*>(io + at);
        at = align(at + 4 * plan_.mask_words);
        if (at > io_device_.bytes) { throw std::logic_error("Qwen3.8-Flash-Next: io layout exceeds its plan"); }

        work_ = std::make_unique<WorkspaceArena>(plan_.work);
        allocated += plan_.work;
        staging_ = DeviceBuffer(plan_.staging);
        allocated += plan_.staging;

        std::vector<const std::uint8_t*> banks;
        std::vector<std::uint64_t> bank_strides;
        for (const auto& bank : parameters_.model.expert_banks()) {
            banks.push_back(bank.host_records);
            bank_strides.push_back(bank.record_stride);
        }
        residency_ = std::make_unique<ExpertResidency>(c_, std::move(banks), std::move(bank_strides), plan_.chunk,
                                                       device_.device);
        allocated += plan_.bytes.residency;
        if (allocated != plan_.bytes.fixed_bytes()) {
            throw std::logic_error("Qwen3.8-Flash-Next: the Program's device allocations differ from its plan");
        }
        if (const std::uint64_t spilled = guard.end(allocated); spilled != 0) {
            throw std::runtime_error(std::to_string(spilled >> 20) +
                                     " MiB of NInfer's VRAM allocations were placed in shared system memory by the "
                                     "driver's sysmem fallback: free VRAM, or lower --max-context or --prefill-chunk");
        }
        measure_link();

        // The VRAM expert cache takes what the fixed allocations, the reserve for graph executables
        // and the display headroom leave.
        if (options_.expert_cache) {
            VramDemand demand;
            demand.headroom    = options_.vram_headroom;
            demand.elastic     = residency_->elastic() && options_.vram_monitor;
            demand.graphs      = plan_.bytes.graph_bound;
            demand.frame_bytes = plan_.frame_stride;
            demand.max_frames  = residency_->frame_limit();
            if (options_.expert_cache_bytes) {
                // Bounded before the narrowing so a huge byte budget cannot truncate to a small count.
                demand.max_frames = static_cast<std::uint32_t>(
                    std::min<std::uint64_t>(*options_.expert_cache_bytes / plan_.frame_stride, demand.max_frames));
            }
            // The KV above the base comes out of the frames: keep room for the request's whole extent.
            const std::uint64_t kv_growth = plan_.vmm ? plan_.kv.mapped_bytes(plan_.pages) - kv_vmm_->mapped_bytes() : 0;
            sizing_ = size_expert_frames(vram_->query(), demand);
            const auto grown = residency_->resize(sizing_.frames, device_.stream, *vram_);
            if (grown.spilled != 0 || grown.refused) {
                char reading[200] = "";
                if (grown.spilled != 0) {
                    std::snprintf(reading, sizeof(reading),
                                  " (%llu MiB of a chunk missing from free memory; then free %llu MiB, OS budget %llu, "
                                  "used %llu)",
                                  static_cast<unsigned long long>(grown.spilled >> 20),
                                  static_cast<unsigned long long>(grown.at_spill.device_free >> 20),
                                  static_cast<unsigned long long>(grown.at_spill.local_budget >> 20),
                                  static_cast<unsigned long long>(grown.at_spill.local_usage >> 20));
                }
                diagnostic(std::string("the GPU ") + (grown.spilled ? "placed part of the expert cache in system memory"
                                                                   : "refused expert-cache memory") +
                           reading + "; it starts with " + std::to_string(grown.frames) + " of " +
                           std::to_string(sizing_.frames) + " frames");
            }
            sizing_.frames = grown.frames;
            if (kv_growth > static_cast<std::uint64_t>(sizing_.frames) * plan_.frame_stride) {
                throw std::runtime_error("Qwen3.8-Flash-Next: the KV of --max-context " + std::to_string(options_.max_context) +
                                         " does not fit beside the expert cache: lower --max-context or free VRAM");
            }
            frames_now_ = residency_->frames();
            pool_now_   = residency_->pool_bytes();
            startup_frames_ = frames_now_;
            if (demand.elastic && !options_.expert_cache_bytes) {
                control_.emplace(demand, ExpertResidency::kChunkBytes, vram_->query().local_usage, pool_now_,
                                 options_.vram_grow_delay_seconds);
                monitor_ = std::make_unique<VramMonitor>(device_.device, *vram_, [](const VramSnapshot&) {});
                vram_epoch_ = Clock::now();
            }
        }

        // The forward over the lane's state, KV and the expert sources.
        execution::ForwardState state;
        auto* p = static_cast<std::byte*>(state_backing_.p);
        for (std::uint32_t l = 0; l < c_.gdn_layers; ++l) {
            state.gdn_conv.emplace_back(p, DType::BF16, std::initializer_list<std::int32_t>{
                                                            dim(c_.gdn.conv_channels()), dim(c_.gdn.conv_kernel - 1)});
            p += align(plan_.gdn_conv);
            state.gdn_recurrent.emplace_back(p, DType::FP32, std::initializer_list<std::int32_t>{
                                                                 dim(c_.gdn.key_head_dim), dim(c_.gdn.value_head_dim),
                                                                 dim(c_.gdn.value_heads)});
            p += align(plan_.gdn_recurrent);
        }
        state.ple_conv = Tensor(p, DType::BF16, {dim(c_.residual_width()), dim(c_.ple.conv_span()), 1});
        p += align(plan_.ple);
        const std::int32_t di = dim(c_.qsa.index_head_dim), r = dim(c_.qsa.compress_ratio);
        for (std::uint32_t l = 0; l < c_.attention_layers; ++l) {
            state.qsa_tails.emplace_back(p, DType::BF16, std::initializer_list<std::int32_t>{di, r - 1});
            p += align(plan_.tail);
        }

        const std::int32_t P = dim(plan_.pages), D = dim(c_.attention.head_dim), Hkv = dim(c_.attention.kv_heads);
        execution::ForwardKV kv;
        kv.block_table      = Tensor(kv_base_ + plan_.kv.table, DType::I32, {P});
        kv.max_visible_keys = plan_.pages * kPagedKVPageSize;
        {
            // One lane: the identity table over the pool.
            std::vector<std::int32_t> table(static_cast<std::size_t>(P));
            for (std::int32_t i = 0; i < P; ++i) { table[static_cast<std::size_t>(i)] = i; }
            CUDA_CHECK(cudaMemcpy(kv.block_table.data, table.data(), table.size() * 4, cudaMemcpyHostToDevice));
        }
        for (const auto& planes : plan_.kv.layers) {
            PagedKVLayerView view;
            view.k_pages       = Tensor(kv_base_ + planes[0].offset, DType::I8, {D, kPagedKVPageSize, Hkv, P});
            view.v_pages       = Tensor(kv_base_ + planes[1].offset, DType::I8, {D, kPagedKVPageSize, Hkv, P});
            view.k_scale_pages = Tensor(kv_base_ + planes[2].offset, DType::FP16, {D / 64, kPagedKVPageSize, Hkv, P});
            view.v_scale_pages = Tensor(kv_base_ + planes[3].offset, DType::FP16, {D / 64, kPagedKVPageSize, Hkv, P});
            view.block_table   = kv.block_table;
            view.head_dim      = D;
            view.num_kv_heads  = Hkv;
            view.storage       = KvCacheStorage::Int8Group64;
            kv.layers.push_back(view);
            kv.pooled.emplace_back(kv_base_ + planes[4].offset, DType::BF16,
                                   std::initializer_list<std::int32_t>{di / r, kPagedKVPageSize, 1, P});
        }
        execution::ForwardExperts experts;
        experts.frame_base    = residency_->frame_base();
        experts.frame_stride  = residency_->frame_stride();
        experts.route_log     = residency_->route_log();
        experts.route_stride  = residency_->route_stride();
        experts.staging_base  = static_cast<std::uint8_t*>(staging_.p);
        experts.staging_slots = kStagingSlots;
        for (std::uint32_t l = 0; l < c_.num_hidden_layers; ++l) { experts.frames.push_back(residency_->table(l)); }
        forward_ = std::make_unique<execution::Forward>(parameters_, device_, *work_, std::move(state), std::move(kv),
                                                        std::move(experts));
        warm_start_experts();
        device_.synchronize();
        report_startup();
    }

    ~ProgramImpl() noexcept {
        try {
            device_.synchronize();
        } catch (...) {}
        save_expert_state();
        monitor_.reset();
    }

    ProgramImpl(const ProgramImpl&)            = delete;
    ProgramImpl& operator=(const ProgramImpl&) = delete;

    const ProgramDevicePlan& device_plan() const noexcept { return plan_.bytes; }
    const VramSizing& vram_sizing() const noexcept { return sizing_; }

    Program::ExpertCacheCounters cache_counters() const noexcept {
        const auto& s = residency_->stats();
        return {.routed = s.routed, .hits = s.hits, .promotions = s.promotions, .seeded = s.seeded,
                .frames = residency_->frames()};
    }

    static ProgramDevicePlan plan_device(const ProgramOptions& options, const TextConfig& config,
                                         std::uint32_t public_tokens, std::span<const std::uint64_t> strides) {
        return plan_layout(options, config, public_tokens, strides).bytes;
    }

    // ---------------------------------------------------------------- admission
    RequestBasePlan plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options) {
        auto data = qwen3_5::PreparedPromptAccess::take(std::move(prompt));
        if (data.token_ids.empty()) { throw std::invalid_argument("prompt must contain tokens"); }
        if (data.token_ids.size() > options_.max_context) {
            throw std::invalid_argument("prompt exceeds configured context capacity");
        }
        if (data.has_media()) {
            throw std::invalid_argument("Qwen3.8-Flash-Next serves text only in this release (no images or video)");
        }
        if (data.external_prefix_tokens != 0) {
            throw std::invalid_argument("Qwen3.8-Flash-Next has no prompt grafts (they need the context cache)");
        }
        for (const TokenId id : data.token_ids) {
            if (id < 0 || id >= plan_.token_domain) {
                throw std::invalid_argument("prompt contains token outside the public token domain");
            }
        }
        auto base                            = std::make_shared<BasePlanImpl>();
        base->summary.prompt_tokens          = static_cast<std::uint32_t>(data.token_ids.size());
        base->summary.requested_output_tokens = options.requested_output_tokens;
        const std::uint32_t capacity_output  = options_.max_context - base->summary.prompt_tokens + 1U;
        base->summary.effective_output_tokens = std::min(options.requested_output_tokens, capacity_output);
        base->summary.effective_limit_reason  = options.requested_output_tokens <= capacity_output
                                                    ? FinishReason::OutputLimit
                                                    : FinishReason::ContextCapacity;
        base->summary.publish_continuation = false; // prefix reuse is off for this model
        base->sampling                     = translate(options.sampling);
        base->context_cache                = data.context_cache;
        base->prompt = std::make_shared<const std::vector<TokenId>>(std::move(data.token_ids));
        return RequestBasePlan(std::move(base));
    }

    static const BasePlanImpl& base_of(const RequestBasePlan& plan) {
        if (!plan.impl_) { throw std::logic_error("Qwen3.8-Flash-Next: empty request plan"); }
        return *plan.impl_;
    }

    std::optional<SourceCandidate> inspect_source(const RequestBasePlan& base, std::optional<CheckpointHandle> checkpoint) const {
        if (checkpoint) { return std::nullopt; }
        SourceCandidate out;
        out.reused_tokens  = 0;
        out.remaining_work = runtime::make_prefill_work(0, base_of(base).summary.prompt_tokens, 0, 0,
                                                        static_cast<std::uint32_t>(plan_.chunk));
        return out;
    }

    BindingReservation start_binding(const RequestBasePlan& plan, runtime::LaneId lane_id, const SourceCandidate& source,
                                     ResumeState* resume) {
        BindingReservation out;
        if (resume != nullptr || source.checkpoint || source.reused_tokens != 0) {
            out.source_valid = false;
            return out;
        }
        if (binding_ || lane_id.value != 0 || lane_.phase != Phase::Free || pending_ != 0) {
            throw std::logic_error("Qwen3.8-Flash-Next: binding on a busy lane");
        }
        const auto& base = base_of(plan);
        const std::uint32_t extent =
            std::min(options_.max_context, base.summary.prompt_tokens + base.summary.effective_output_tokens);
        // The request's whole extent is backed now, so its units never meet a shortage.
        ensure_kv((extent + kPagedKVPageSize - 1) / kPagedKVPageSize);
        lane_            = Lane{};
        lane_.phase      = Phase::Binding;
        lane_.epoch      = ++epoch_;
        lane_.base       = plan.impl_;
        lane_.extent     = extent;
        lane_.admitted   = Clock::now();
        binding_         = true;
        return {.reserved = true};
    }

    ContextProgress poll_context(runtime::CancellationFlagView cancellation) {
        ContextProgress out;
        out.kind = ContextOperationKind::Bind;
        if (!binding_) { return out; }
        binding_     = false;
        out.advanced = true;
        out.complete = true;
        if (cancellation.requested()) {
            lane_ = Lane{.epoch = lane_.epoch};
            settle_idle(); // the same idle reconciliation a finished request gets
            return out; // not published: the Engine completes the request as cancelled
        }
        // A fresh sequence: zero recurrent state (the KV is written before it is read).
        CUDA_CHECK(cudaMemsetAsync(state_backing_.p, 0, state_backing_.bytes, device_.stream));
        const auto& base   = *lane_.base;
        lane_.history      = *base.prompt;
        lane_.prompt_tokens = base.summary.prompt_tokens;
        lane_.begin        = {.prompt_tokens = lane_.prompt_tokens, .reused_prompt_tokens = 0,
                              .prefix_reuse_path = PrefixReusePath::Root};
        install_sampling(base.sampling);
        lane_.cache_at_admission = residency_->stats();
        lane_.ngram_at_admission = volume_.counters();
        lane_.phase              = Phase::Prefill;
        out.published            = true;
        out.sequence             = handle();
        return out;
    }

    runtime::ResourceReservation reserve_units(std::span<const ExecutionUnit> units) {
        for (const auto& unit : units) {
            require_sequence(unit.sequence);
            const std::uint32_t frontier =
                unit.kind == ExecutionUnitKind::Control ? lane_.state_tokens + 1 + unit.tokens : lane_.state_tokens + 1;
            if (unit.kind == ExecutionUnitKind::Replay) { throw std::logic_error("Qwen3.8-Flash-Next: no replay"); }
            if (frontier > options_.max_context + 1) { throw std::logic_error("Qwen3.8-Flash-Next: unit exceeds the context"); }
            lane_.permit = true;
        }
        return {.reserved = true};
    }

    void release_units(std::span<const SequenceHandle> sequences) noexcept {
        for (const auto& s : sequences) {
            if (valid(s)) { lane_.permit = false; }
        }
    }

    // ---------------------------------------------------------------- execution
    PrefillProgress advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming* failed, runtime::TokenMaskProvider* masks) {
        const auto start = Clock::now();
        require_sequence(sequence);
        if (lane_.phase != Phase::Prefill || pending_ != 0) {
            throw std::logic_error("Qwen3.8-Flash-Next: prefill on a lane that is not prefilling");
        }
        const nvtx::ScopedRange range(nvtx::Name::Prefill, nvtx::Category::Prefill);
        const std::int32_t begin = static_cast<std::int32_t>(lane_.state_tokens);
        const std::int32_t width = std::min(plan_.chunk, static_cast<std::int32_t>(lane_.prompt_tokens) - begin);
        const bool last          = begin + width == static_cast<std::int32_t>(lane_.prompt_tokens);
        PrefillProgress out;
        out.summary                 = lane_.begin;
        out.processed_prompt_tokens = static_cast<std::uint32_t>(width);
        out.complete                = last;
        try {
            run_call(begin, width);
            lane_.state_tokens += static_cast<std::uint32_t>(width);
            const std::size_t promotions = width <= kServedCallColumns ? decode_budget(static_cast<std::uint32_t>(width)) : 0;
            if (last) {
                const bool failed_constraint = sample(begin + width - 1, masks);
                residency_->enqueue_route_download(device_.stream, width);
                const auto wait = Clock::now();
                device_.synchronize();
                out.timing.device_wait_ns = elapsed_ns(wait);
                const TokenId token = sampled_token();
                residency_->after_round(device_.stream, width, promotions);
                apply_vram_target(false);
                pending_token_ = token;
                pending_       = ++transaction_;
                lane_.phase    = Phase::Pending;
                out.timing.submit_host_ns = elapsed_ns(start) - out.timing.device_wait_ns;
                out.pending.emplace(ContractAccess::make_pending(this, pending_, handle(), {&pending_token_, 1}, out.timing,
                                                                 failed_constraint));
            } else {
                residency_->enqueue_route_download(device_.stream, width);
                const auto wait = Clock::now();
                device_.synchronize();
                out.timing.device_wait_ns = elapsed_ns(wait);
                residency_->after_round(device_.stream, width, promotions);
                apply_vram_target(false);
                out.timing.submit_host_ns = elapsed_ns(start) - out.timing.device_wait_ns;
            }
            lane_.prefill_ns += elapsed_ns(start);
            lane_.permit = false;
            return out;
        } catch (...) {
            if (failed != nullptr) { failed->submit_host_ns += elapsed_ns(start); }
            release_lane();
            throw;
        }
    }

    PendingBatch decode(std::span<const SequenceHandle> sequences, std::span<const runtime::RoundBudget> budgets,
                        runtime::ExecutionTiming* failed, runtime::TokenMaskProvider* masks) {
        const auto start = Clock::now();
        if (sequences.size() != 1 || budgets.size() != 1) {
            throw std::invalid_argument("Qwen3.8-Flash-Next decodes one row");
        }
        require_sequence(sequences[0]);
        if (lane_.phase != Phase::Decode || pending_ != 0 || lane_.history.size() != lane_.state_tokens + 1ULL) {
            throw std::logic_error("Qwen3.8-Flash-Next: decode row is not at a committed frontier");
        }
        const nvtx::ScopedRange range(nvtx::Name::DecodeOrdinaryRound, nvtx::Category::Decode);
        try {
            const std::int32_t position = static_cast<std::int32_t>(lane_.state_tokens);
            stage(position, 1);
            residency_->before_round(device_.stream);
            run_decode();
            lane_.state_tokens += 1;
            const bool failed_constraint = sample(position, masks);
            residency_->enqueue_route_download(device_.stream, 1);
            const auto wait = Clock::now();
            device_.synchronize();
            runtime::ExecutionTiming timing;
            timing.device_wait_ns = elapsed_ns(wait);
            pending_token_        = sampled_token();
            deferred_             = true; // the round's cache update runs at commit
            pending_              = ++transaction_;
            lane_.phase           = Phase::Pending;
            timing.submit_host_ns = elapsed_ns(start) - timing.device_wait_ns;
            lane_.decode_ns += elapsed_ns(start);
            ++decode_rounds_;
            return ContractAccess::make_pending(this, pending_, sequences[0], {&pending_token_, 1}, timing,
                                                failed_constraint);
        } catch (...) {
            if (failed != nullptr) { failed->submit_host_ns += elapsed_ns(start); }
            release_lane();
            throw;
        }
    }

    runtime::ExecutionTiming append_forced(std::span<const SequenceHandle> sequences, std::span<const TokenId> tokens,
                                           std::uint32_t stride, runtime::ExecutionTiming* failed) {
        const auto start = Clock::now();
        if (sequences.size() != 1 || stride == 0 || tokens.size() != stride) {
            throw std::invalid_argument("Qwen3.8-Flash-Next: forced tokens do not match their row");
        }
        require_sequence(sequences[0]);
        if (lane_.phase != Phase::Decode || pending_ != 0 || lane_.history.size() != lane_.state_tokens + 1ULL ||
            lane_.state_tokens + stride > options_.max_context) {
            throw std::logic_error("Qwen3.8-Flash-Next: forced tokens on a lane that is not at a committed frontier");
        }
        for (const TokenId token : tokens) {
            if (token < 0 || token >= plan_.token_domain) { throw std::invalid_argument("forced token outside the domain"); }
        }
        try {
            // The model reads the pending input token and every forced token but the last, which
            // becomes the next decode input.
            lane_.history.insert(lane_.history.end(), tokens.begin(), tokens.end());
            if (lane_.sampling.token_counts != nullptr) {
                upload_pinned_words(tokens);
                Tensor ids(io_device_.p, DType::I32, {static_cast<std::int32_t>(stride)});
                ops::increment_token_counts(ids, token_counts_, device_.stream);
                device_.synchronize(); // the staging below rewrites the pinned block the upload reads
            }
            std::int32_t begin = static_cast<std::int32_t>(lane_.state_tokens);
            const std::int32_t end = begin + static_cast<std::int32_t>(stride);
            while (begin < end) {
                const std::int32_t width = std::min(plan_.chunk, end - begin);
                run_call(begin, width);
                residency_->enqueue_route_download(device_.stream, width);
                device_.synchronize();
                residency_->after_round(device_.stream, width, kDecodePromotionsPerLayer);
                lane_.state_tokens += static_cast<std::uint32_t>(width);
                begin += width;
            }
            apply_vram_target(false);
            lane_.permit = false;
            runtime::ExecutionTiming timing;
            timing.submit_host_ns = elapsed_ns(start);
            lane_.decode_ns += timing.submit_host_ns;
            return timing;
        } catch (...) {
            if (failed != nullptr) { failed->submit_host_ns += elapsed_ns(start); }
            release_lane();
            throw;
        }
    }

    CommitResult commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions) {
        const auto start = Clock::now();
        if (ContractAccess::owner(pending) != this || ContractAccess::transaction(pending) != pending_ || pending_ == 0) {
            throw std::logic_error("Qwen3.8-Flash-Next: commit of a foreign or stale pending batch");
        }
        const auto rows = ContractAccess::rows(pending);
        if (rows.size() != 1 || decisions.size() != 1 || !valid(rows[0])) {
            throw std::logic_error("Qwen3.8-Flash-Next: commit decisions are not row aligned");
        }
        const auto& d = decisions[0];
        const bool constraint_failed = pending.constraint_failed(0);
        ContractAccess::consume(pending);
        pending_ = 0;
        CommitResult out;
        out.row_count = 1;
        if (d.cancelled || d.failed) {
            if ((d.cancelled && d.failed) || (!d.cancelled && d.failed != constraint_failed) || d.accepted_tokens != 0 ||
                !d.terminal) {
                release_lane();
                throw std::logic_error("Qwen3.8-Flash-Next: pending decision is invalid");
            }
            out.rows[0].timings     = timings();
            out.rows[0].disposition = d.failed ? runtime::CommitDisposition::FailedReleased
                                               : runtime::CommitDisposition::CancelledReleased;
            settle_round();
            release_lane();
            settle_idle(); // a cancelled or failed request gets the idle reconciliation a finished one does
        } else {
            if (d.accepted_tokens != 1) {
                release_lane();
                throw std::logic_error("Qwen3.8-Flash-Next: a row commits exactly one token");
            }
            if (lane_.history.size() == lane_.prompt_tokens) {
                lane_.first_token_seconds = std::chrono::duration<double>(Clock::now() - lane_.admitted).count();
                lane_.cache_at_decode     = residency_->stats();
                lane_.ngram_at_decode     = volume_.counters();
            }
            lane_.history.push_back(pending_token_);
            lane_.phase = d.terminal ? Phase::Finishable : Phase::Decode;
            out.rows[0].disposition = d.terminal ? runtime::CommitDisposition::Finishable : runtime::CommitDisposition::Active;
            out.rows[0].timings     = timings();
            settle_round();
        }
        lane_.permit           = false;
        out.timing.post_host_ns = elapsed_ns(start);
        return out;
    }

    DiscardResult discard(PendingBatch&& pending) noexcept {
        DiscardResult out;
        if (ContractAccess::owner(pending) != this || ContractAccess::transaction(pending) != pending_) { return out; }
        out.row_count = ContractAccess::rows(pending).size();
        ContractAccess::consume(pending);
        pending_  = 0;
        // The round ran on the device: settle its cache update before the lane goes, as a commit would.
        try { settle_round(); } catch (...) { deferred_ = false; }
        release_lane();
        settle_idle();
        out.status = runtime::ConsumeStatus::Consumed;
        return out;
    }

    FinishResult finish(SequenceHandle sequence) noexcept {
        FinishResult out;
        try {
            if (!valid(sequence) || lane_.phase != Phase::Finishable || pending_ != 0) { return out; }
            out.timings = timings();
            report_request();
            release_lane();
            apply_vram_target(true);
            maintain_expert_state();
            out.status = runtime::ConsumeStatus::Consumed;
        } catch (...) {}
        return out;
    }

    AbortResult abort(SequenceHandle sequence) noexcept {
        AbortResult out;
        try {
            if (!valid(sequence) || pending_ != 0) { return out; }
            out.timings = timings();
            device_.synchronize();
            release_lane();
            apply_vram_target(true);
            maintain_expert_state();
            out.status = runtime::ConsumeStatus::Consumed;
        } catch (...) {}
        return out;
    }

    // Between requests, after the lane is gone: the idle VRAM target and the periodic expert state.
    void settle_idle() noexcept {
        try {
            apply_vram_target(true);
            maintain_expert_state();
        } catch (...) {}
    }

    void fail_all_cleanup() noexcept {
        (void)device_.synchronize_status();
        binding_  = false;
        pending_  = 0;
        deferred_ = false;
        release_lane();
    }

    bool has_context_transaction() const noexcept { return binding_; }

    PhysicalUsageSnapshot usage() const noexcept {
        PhysicalUsageSnapshot out;
        out.capacity.state_slots   = 1;
        out.occupied.state_slots   = lane_.phase != Phase::Free ? 1U : 0U;
        out.capacity.main_kv_pages = plan_.pages;
        out.occupied.main_kv_pages =
            lane_.phase != Phase::Free ? (lane_.extent + kPagedKVPageSize - 1) / kPagedKVPageSize : 0U;
        return out;
    }

    MemorySummary memory() const noexcept {
        MemorySummary out;
        out.device                       = device_.device;
        out.max_context                  = options_.max_context;
        out.kv_capacity                  = plan_.pages * kPagedKVPageSize;
        out.kv_capacity_page_groups      = plan_.pages;
        out.kv_capacity_max_page_groups  = plan_.pages;
        out.kv_cache                     = KvCacheStorage::Int8Group64;
        out.kv_payload_bytes             = kv_vmm_ ? kv_vmm_->mapped_bytes() : kv_backing_.bytes;
        out.workspace_logical_peak_bytes = plan_.work;
        out.cuda_graph_allowance_bytes   = kGraphExecutableBytes * plan_.bytes.graph_bound;
        return out;
    }

private:
    static std::vector<std::uint64_t> strides(const execution::Parameters& parameters) {
        std::vector<std::uint64_t> out;
        for (const auto& bank : parameters.model.expert_banks()) { out.push_back(bank.record_stride); }
        return out;
    }

    SequenceHandle handle() const noexcept { return ContractAccess::make_sequence(this, 0, lane_.epoch); }

    bool valid(const SequenceHandle& s) const noexcept {
        return ContractAccess::owner(s) == this && ContractAccess::lane(s) == 0 && ContractAccess::epoch(s) == lane_.epoch &&
               lane_.phase != Phase::Free && lane_.phase != Phase::Binding;
    }

    void require_sequence(const SequenceHandle& s) const {
        if (!valid(s)) { throw std::logic_error("Qwen3.8-Flash-Next: stale or foreign sequence"); }
    }

    void release_lane() noexcept {
        const std::uint64_t epoch = lane_.epoch;
        lane_       = Lane{};
        lane_.epoch = epoch;
        binding_    = false;
    }

    GenerationTimings timings() const {
        GenerationTimings out;
        out.prefill_seconds     = seconds(lane_.prefill_ns);
        out.decode_seconds      = seconds(lane_.decode_ns);
        out.first_token_seconds = lane_.first_token_seconds;
        return out;
    }

    void install_sampling(const ops::SamplingConfig& config) {
        lane_.sampling = config;
        const bool penalties = config.presence_penalty != 0.0F || config.frequency_penalty != 0.0F;
        if (penalties) {
            CUDA_CHECK(cudaMemsetAsync(token_counts_.data, 0, token_counts_.bytes(), device_.stream));
            lane_.sampling.token_counts = static_cast<std::int32_t*>(token_counts_.data);
        } else {
            lane_.sampling.token_counts = nullptr;
        }
    }

    // Copies `tokens` (I32) to the head of the io block through pinned memory.
    void upload_pinned_words(std::span<const TokenId> tokens) {
        if (tokens.size() * 4 > plan_.io.positions) { throw std::logic_error("Qwen3.8-Flash-Next: too many forced tokens"); }
        std::memcpy(io_host_.data(), tokens.data(), tokens.size() * 4);
        upload_pinned(io_device_.p, io_host_.data(), tokens.size() * 4, device_.stream);
    }

    // Stages the inputs of positions [begin, begin + width) of the lane's history: ids, positions,
    // RoPE positions (text: every axis equals the position), the pooled-block start and the
    // n-gram rows each column reads, then uploads them.
    void stage(std::int32_t begin, std::int32_t width) {
        const auto& io = plan_.io;
        auto* host     = static_cast<std::byte*>(io_host_.data());
        auto* ids      = reinterpret_cast<std::int32_t*>(host + io.ids);
        auto* pos      = reinterpret_cast<std::int32_t*>(host + io.positions);
        auto* rope     = reinterpret_cast<std::int32_t*>(host + io.rope);
        auto* block    = reinterpret_cast<std::int32_t*>(host + io.block_rope);
        auto* columns  = reinterpret_cast<std::int32_t*>(host + io.logit_columns);
        for (std::int32_t t = 0; t < width; ++t) {
            ids[t] = lane_.history[static_cast<std::size_t>(begin + t)];
            pos[t] = begin + t;
            for (int a = 0; a < 3; ++a) { rope[a * width + t] = begin + t; }
        }
        const std::int32_t r = dim(c_.qsa.compress_ratio);
        for (int a = 0; a < 3; ++a) { block[a] = begin / r * r; }
        columns[0] = width - 1;
        // n-gram rows: the window of ngram_size - 1 earlier tokens, EOS-padded at the start.
        const std::int32_t context = dim(c_.ple.ngram.ngram_size) - 1;
        ngram_window_.assign(static_cast<std::size_t>(std::max(0, context - begin)), c_.eos_token_id);
        const std::int32_t from = std::max(0, begin - context);
        ngram_window_.insert(ngram_window_.end(), lane_.history.begin() + from, lane_.history.begin() + begin + width);
        const std::size_t heads = hash_.heads();
        row_ids_.resize(static_cast<std::size_t>(width) * heads);
        hash_.row_ids(ngram_window_, static_cast<std::size_t>(width), row_ids_.data());
        const std::size_t row_bytes = c_.ple.table.row_bytes;
        volume_.read_rows(row_ids_, std::span<std::byte>(host + io.ngram, row_ids_.size() * row_bytes));
        upload_pinned(io_device_.p, io_host_.data(), io.ngram + row_ids_.size() * row_bytes, device_.stream);
    }

    execution::ForwardBatch batch(std::int32_t width) const {
        const auto& io = plan_.io;
        auto* base     = static_cast<std::byte*>(io_device_.p);
        execution::ForwardBatch b;
        b.ids              = Tensor(base + io.ids, DType::I32, {width});
        b.positions        = Tensor(base + io.positions, DType::I32, {width});
        b.rope_positions   = Tensor(base + io.rope, DType::I32, {width, 3});
        b.block_start_rope = Tensor(base + io.block_rope, DType::I32, {3});
        b.logit_columns    = Tensor(base + io.logit_columns, DType::I32, {1});
        b.ngram_rows       = Tensor(base + io.ngram, DType::U8,
                                    {dim(c_.ple.table.row_bytes), dim(hash_.heads()), width});
        return b;
    }

    // One eager call over positions [begin, begin + width): stage, table, forward. Not synchronized.
    void run_call(std::int32_t begin, std::int32_t width) {
        stage(begin, width);
        residency_->before_round(device_.stream);
        forward_->run(batch(width), logits32_);
        ops::cast_fp32_to_bf16(logits32_, logits16_, device_.stream);
    }

    // The decode forward: eager on the first round (the kernels' one-time setup runs outside a
    // capture), then captured once and replayed. Inputs were staged at fixed addresses.
    void run_decode() {
        const auto body = [this] {
            forward_->run(batch(1), logits32_);
            ops::cast_fp32_to_bf16(logits32_, logits16_, device_.stream);
        };
        if (!options_.use_cuda_graph || decode_rounds_ == 0) {
            body();
            return;
        }
        if (!graph_exec_.ready()) {
            graph_.capture(device_.stream, body);
            graph_exec_.instantiate(graph_);
            graph_exec_.upload(device_.stream);
        }
        graph_exec_.launch(device_.stream);
    }

    // Samples the next token from the last call's logits at `position` into sampled_, with the
    // row's grammar mask when constrained. Returns whether the grammar has a dead end there.
    bool sample(std::int32_t position, runtime::TokenMaskProvider* masks) {
        ops::SamplingConfig config = lane_.sampling;
        std::uint32_t dead         = 0;
        if (masks != nullptr && masks->constrained(0)) {
            auto* words = static_cast<std::uint32_t*>(mask_host_.data());
            dead        = masks->fill(0, {}, std::span<std::uint32_t>(words, plan_.mask_words));
            upload_pinned(masks_, words, 4 * plan_.mask_words, device_.stream);
            masks->uploaded(0, 4 * plan_.mask_words);
            config.mask = {masks_, static_cast<std::int32_t>(plan_.mask_words)};
        }
        auto* host_config = static_cast<ops::SamplingConfig*>(config_host_.data());
        *host_config      = config;
        upload_pinned(configs_, host_config, sizeof(ops::SamplingConfig), device_.stream);
        upload_pinned(sample_position_.data, &position_word(position), 4, device_.stream);
        // The forward's workspace is free once its kernels precede these in stream order.
        work_->reset();
        ops::sample(logits16_, sampled_, plan_.token_domain, configs_, sample_position_, ops::kSamplePurposeDecode, *work_,
                    device_.stream);
        CUDA_CHECK(cudaMemcpyAsync(host_token_.data(), sampled_.data, 4, cudaMemcpyDeviceToHost, device_.stream));
        return (dead & 1U) != 0;
    }

    std::int32_t& position_word(std::int32_t position) {
        auto* word = reinterpret_cast<std::int32_t*>(static_cast<std::byte*>(host_token_.data()) + 32);
        *word      = position;
        return *word;
    }

    TokenId sampled_token() const {
        const TokenId token = *static_cast<const std::int32_t*>(host_token_.data());
        if (token < 0 || token >= plan_.token_domain) {
            throw std::runtime_error("Qwen3.8-Flash-Next: the forward produced non-finite logits (sampled " +
                                     std::to_string(token) + ")");
        }
        return token;
    }

    // ---------------------------------------------------------------- expert cache
    // Promotions follow the tokens a round advances. The fill phase (fewer promotions than frames
    // so far) promotes every round; afterwards every promotion_interval-th.
    std::size_t decode_budget(std::uint32_t tokens) {
        const ExpertResidency::Stats& stats = residency_->stats();
        if (stats.promotions < residency_->frames()) { return kDecodePromotionsPerLayer * tokens; }
        budget_tokens_ += tokens;
        const std::uint64_t interval = promotion_interval();
        const std::uint64_t due      = budget_tokens_ / interval;
        budget_tokens_ %= interval;
        return static_cast<std::size_t>(due) * kDecodePromotionsPerLayer;
    }

    std::uint64_t promotion_interval() const noexcept {
        const double scaled = static_cast<double>(kDecodePromotionInterval) * kReferenceLinkBytesPerSecond / link_;
        return static_cast<std::uint64_t>(std::clamp<long>(std::lround(scaled), 1L, 8L));
    }

    void settle_round() {
        if (!deferred_) { return; }
        deferred_ = false;
        residency_->after_round(device_.stream, 1, decode_budget(1));
        apply_vram_target(false);
    }

    void measure_link() {
        double rate = options_.link_bytes_per_second;
        const auto banks = parameters_.model.expert_banks();
        if (rate <= 0.0 && !banks.empty() && staging_.p != nullptr) {
            // An idle GPU parks its PCIe link at a low rate: enough repetitions (~0.1-0.3 s) wake it
            // before the best one is taken.
            rate = measure_h2d_bytes_per_second(banks.front().host_records, staging_.p, plan_.staging, 32);
        }
        if (rate > 0.0) { link_ = rate; }
        char text[160];
        std::snprintf(text, sizeof(text), "PCIe link to the GPU: %.1f GB/s%s; decode promotes every %llu tokens once full",
                      link_ / 1e9, options_.link_bytes_per_second > 0.0 ? " (fixed)" : " measured",
                      static_cast<unsigned long long>(promotion_interval()));
        diagnostic(text);
    }

    // At a boundary (no round in flight; `idle`: no request active): resizes the expert cache to
    // the VRAM control law's decision.
    void apply_vram_target(bool idle) {
        if (!control_ || !monitor_) { return; }
        const double now = std::chrono::duration<double>(Clock::now() - vram_epoch_).count();
        const auto d     = control_->decide(monitor_->latest(), pool_now_, frames_now_, now, idle);
        if (d.frames == frames_now_) { return; }
        const std::uint32_t before = frames_now_;
        const auto resized         = residency_->resize(d.frames, device_.stream, *vram_);
        frames_now_                = resized.frames;
        pool_now_                  = residency_->pool_bytes();
        const VramSnapshot after   = monitor_->refresh();
        diagnostic(std::string(d.shrink ? "free VRAM fell to " : "free VRAM rose to ") +
                   std::to_string(after.device_free >> 20) + " MiB: expert cache " + std::to_string(before) + " -> " +
                   std::to_string(frames_now_) + " frames");
        if (d.shrink && frames_now_ * 4 < startup_frames_) {
            diagnostic("another program holds VRAM: the expert cache is down to " + std::to_string(frames_now_) + " of " +
                       std::to_string(startup_frames_) + " frames, so decode is slower until that memory is released");
        }
    }

    void warm_start_experts() {
        if (options_.expert_state.empty() || !options_.expert_cache || residency_->frames() == 0) { return; }
        const auto keys = c_.num_hidden_layers * c_.moe.experts;
        const auto load = expert_cache::load_expert_state(options_.expert_state, options_.expert_state_identity, keys);
        expert_state_saved_ = Clock::now();
        if (!load.state) {
            diagnostic("expert cache: " + load.message + "; it starts empty and fills as requests arrive");
            return;
        }
        const auto start  = Clock::now();
        const auto max_keys = static_cast<std::uint32_t>(kSeedShare * residency_->frames());
        const std::uint32_t loaded = residency_->warm_start(*load.state, kSeedCountCap, max_keys, device_.stream);
        device_.synchronize();
        char line[200];
        std::snprintf(line, sizeof(line), "expert cache: restored the %u most-used experts from the last session (%.1f GiB in %.2f s)",
                      loaded, static_cast<double>(loaded) * static_cast<double>(residency_->frame_stride()) / (1ULL << 30),
                      std::chrono::duration<double>(Clock::now() - start).count());
        diagnostic(line);
        expert_state_routed_ = residency_->stats().routed;
    }

    void save_expert_state() noexcept {
        if (options_.expert_state.empty() || !residency_ || residency_->frames() == 0) { return; }
        try {
            const std::string error = expert_cache::save_expert_state(options_.expert_state, options_.expert_state_identity,
                                                                      residency_->saved_state());
            // The interval restarts either way so a failing save is retried at most every
            // kExpertStateInterval; the routed count only advances when the state reached disk.
            expert_state_saved_ = Clock::now();
            if (!error.empty()) {
                diagnostic("expert state not saved: " + error);
                return;
            }
            expert_state_routed_ = residency_->stats().routed;
        } catch (...) {}
    }

    // Between requests: saves when rounds ran since the last save and 10 minutes have passed.
    void maintain_expert_state() noexcept {
        if (options_.expert_state.empty() || residency_->stats().routed == expert_state_routed_) { return; }
        if (Clock::now() - expert_state_saved_ < kExpertStateInterval) { return; }
        save_expert_state();
    }

    // ---------------------------------------------------------------- elastic KV
    // Maps (and zeroes) the chunks pages [0, pages) need; false when the device has no memory.
    bool map_kv(std::uint32_t pages) {
        const auto need = plan_.kv.chunks(pages);
        for (std::size_t c = 0; c < need.size(); ++c) {
            if (!need[c] || kv_vmm_->mapped(c)) { continue; }
            if (!kv_vmm_->map(c)) { return false; }
            CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(kv_vmm_->base()) + c * kKvChunkBytes, 0, kKvChunkBytes,
                                       device_.stream));
        }
        return true;
    }

    // Backs at least `need` pages, taking the memory from the expert frames; above that lowers the
    // pool to what `need` and the base require, returning memory to the frames. Between requests.
    void ensure_kv(std::uint32_t need) {
        if (!kv_vmm_) { return; }
        need = std::min(need, plan_.pages);
        std::uint32_t target = std::max(plan_.base_pages, (need + kKvGrowPages - 1) / kKvGrowPages * kKvGrowPages);
        target               = std::min(target, plan_.pages);
        if (target == backed_pages_) { return; }
        device_.synchronize();
        residency_->drain(device_.stream);
        const std::uint64_t stride = residency_->frame_stride();
        if (target > backed_pages_) {
            const std::size_t added = plan_.kv.mapped_bytes(target) - kv_vmm_->mapped_bytes();
            const std::uint32_t frames = residency_->frames();
            const auto give = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(frames, (added + ExpertResidency::kChunkBytes + stride - 1) / stride));
            // Growth is all or nothing: a failure unmaps what this attempt mapped and takes back the
            // frames it gave away (net memory use is then what it was, so the fixed-use ledger needs no
            // entry), and refreshes the frame accounting the resize changed.
            try {
                if (!map_kv(target)) {
                    (void)residency_->resize(frames - give, device_.stream, *vram_);
                    if (!map_kv(target)) {
                        throw std::runtime_error("Qwen3.8-Flash-Next: no VRAM left for this request's KV");
                    }
                }
            } catch (...) {
                (void)device_.synchronize_status();
                const auto keep = plan_.kv.chunks(backed_pages_);
                for (std::size_t c = 0; c < keep.size(); ++c) {
                    if (!keep[c] && kv_vmm_->mapped(c)) { kv_vmm_->unmap(c); }
                }
                // The frames given away for this growth go back, best effort: the chunks above are
                // free again, and a refusal here only leaves the cache smaller for the controller to regrow.
                if (residency_->frames() < frames) {
                    try {
                        (void)residency_->resize(frames, device_.stream, *vram_);
                        (void)device_.synchronize_status();
                    } catch (...) {}
                }
                frames_now_ = residency_->frames();
                pool_now_   = residency_->pool_bytes();
                throw;
            }
            if (control_) { control_->account_fixed(static_cast<std::int64_t>(added)); }
            diagnostic("KV pool grew to " + std::to_string(target) + " pages (" +
                       std::to_string(kv_vmm_->mapped_bytes() >> 20) + " MiB); expert cache " + std::to_string(frames) +
                       " -> " + std::to_string(residency_->frames()) + " frames");
        } else {
            const std::size_t before = kv_vmm_->mapped_bytes();
            const auto need_chunks   = plan_.kv.chunks(target);
            for (std::size_t c = 0; c < need_chunks.size(); ++c) {
                if (!need_chunks[c] && kv_vmm_->mapped(c)) { kv_vmm_->unmap(c); }
            }
            const std::size_t freed    = before - kv_vmm_->mapped_bytes();
            const std::uint32_t frames = residency_->frames();
            if (!options_.expert_cache_bytes) {
                (void)residency_->resize(frames + static_cast<std::uint32_t>(freed / stride), device_.stream, *vram_);
            }
            if (control_) { control_->account_fixed(-static_cast<std::int64_t>(freed)); }
            diagnostic("KV pool shrank to " + std::to_string(target) + " pages; expert cache " + std::to_string(frames) +
                       " -> " + std::to_string(residency_->frames()) + " frames");
        }
        backed_pages_ = target;
        frames_now_   = residency_->frames();
        pool_now_     = residency_->pool_bytes();
    }

    // ---------------------------------------------------------------- reporting
    void diagnostic(const std::string& text) const noexcept {
        try {
            std::fprintf(stderr, "[qwen3.8-flash-next] %s\n", text.c_str());
            std::fflush(stderr);
        } catch (...) {}
    }

    void report_startup() {
        const double gib = 1ULL << 30;
        char line[400];
        std::snprintf(line, sizeof(line),
                      "expert cache: %u frames (%.2f GiB, %.1f%% of the %u routed experts); KV %u tokens INT8, %.0f MiB "
                      "backed (grows to %.0f MiB from the frames)",
                      residency_->frames(), static_cast<double>(residency_->frames()) * plan_.frame_stride / gib,
                      100.0 * residency_->frames() / (c_.num_hidden_layers * c_.moe.experts),
                      c_.num_hidden_layers * c_.moe.experts, plan_.pages * kPagedKVPageSize,
                      static_cast<double>(kv_vmm_ ? kv_vmm_->mapped_bytes() : kv_backing_.bytes) / (1 << 20),
                      static_cast<double>(plan_.kv.mapped_bytes(plan_.pages)) / (1 << 20));
        diagnostic(line);
        diagnostic(sizing_.describe());
    }

    // The expert cache and n-gram traffic of the finished request: one line per request.
    void report_request() const noexcept {
        try {
            const auto& s = residency_->stats();
            const auto& a = lane_.cache_at_admission;
            const auto n  = volume_.counters();
            const auto& m = lane_.ngram_at_admission;
            const std::uint64_t routed = s.routed - a.routed, hits = s.hits - a.hits;
            const auto& d = lane_.cache_at_decode;
            const std::uint64_t decode_routed = s.routed - d.routed, decode_hits = s.hits - d.hits;
            const std::uint32_t generated =
                lane_.history.size() > lane_.prompt_tokens ? static_cast<std::uint32_t>(lane_.history.size() - lane_.prompt_tokens) : 0;
            std::size_t free_vram = 0, total_vram = 0;
            if (cudaMemGetInfo(&free_vram, &total_vram) != cudaSuccess) { free_vram = 0; }
            char text[640];
            std::snprintf(text, sizeof(text),
                          "request: %u prompt + %u generated tokens; prefill %.2f s, decode %.2f s (%.2f tok/s); expert "
                          "cache %u frames, %.1f%% of %llu routed experts hit, %llu promotions; decode %.1f%% of %llu hit, "
                          "%llu promotions; since start %.1f%%; n-gram rows %llu (%llu cached, %llu block reads, %.2f s; "
                          "decode %llu block reads, %.2f s); VRAM free %zu MiB",
                          lane_.prompt_tokens, generated, seconds(lane_.prefill_ns), seconds(lane_.decode_ns),
                          lane_.decode_ns ? (generated > 1 ? (generated - 1) / seconds(lane_.decode_ns) : 0.0) : 0.0,
                          residency_->frames(), routed ? 100.0 * static_cast<double>(hits) / static_cast<double>(routed) : 0.0,
                          static_cast<unsigned long long>(routed),
                          static_cast<unsigned long long>(s.promotions - a.promotions),
                          decode_routed ? 100.0 * static_cast<double>(decode_hits) / static_cast<double>(decode_routed) : 0.0,
                          static_cast<unsigned long long>(decode_routed),
                          static_cast<unsigned long long>(s.promotions - d.promotions),
                          s.routed ? 100.0 * static_cast<double>(s.hits) / static_cast<double>(s.routed) : 0.0,
                          static_cast<unsigned long long>(n.rows - m.rows), static_cast<unsigned long long>(n.hits - m.hits),
                          static_cast<unsigned long long>(n.reads - m.reads), seconds(n.read_ns - m.read_ns),
                          static_cast<unsigned long long>(n.reads - lane_.ngram_at_decode.reads),
                          seconds(n.read_ns - lane_.ngram_at_decode.read_ns), free_vram >> 20);
            diagnostic(text);
        } catch (...) {}
    }

    const execution::Parameters& parameters_;
    DeviceContext& device_;
    const TextConfig& c_;
    ProgramOptions options_;
    NgramVolume volume_;
    NgramHash hash_;
    Layout plan_;
    std::int32_t vocab_ = 0;

    std::unique_ptr<VramBudgetSource> vram_;
    VramSizing sizing_;
    std::optional<VramControl> control_;
    std::unique_ptr<VramMonitor> monitor_;
    Clock::time_point vram_epoch_{};
    std::uint32_t frames_now_ = 0, startup_frames_ = 0;
    std::uint64_t pool_now_ = 0;
    double link_            = kReferenceLinkBytesPerSecond;

    DeviceBuffer state_backing_;
    std::unique_ptr<VmmRange> kv_vmm_;
    DeviceBuffer kv_backing_;
    std::byte* kv_base_         = nullptr;
    std::uint32_t backed_pages_ = 0;
    DeviceBuffer io_device_;
    PinnedHostBuffer io_host_{1};
    PinnedHostBuffer host_token_{1};
    PinnedHostBuffer mask_host_{1};
    PinnedHostBuffer config_host_{1};
    Tensor logits32_, logits16_, sampled_, sample_position_, token_counts_;
    ops::SamplingConfig* configs_ = nullptr;
    std::uint32_t* masks_         = nullptr;
    std::unique_ptr<WorkspaceArena> work_;
    DeviceBuffer staging_;
    std::unique_ptr<ExpertResidency> residency_;
    std::unique_ptr<execution::Forward> forward_;
    DecodeGraphDefinition graph_;
    DecodeGraphExecutable graph_exec_;

    Lane lane_;
    std::uint64_t epoch_       = 0;
    bool binding_              = false;
    std::uint64_t pending_     = 0; // the open pending transaction, 0 when none
    std::uint64_t transaction_ = 0;
    TokenId pending_token_     = 0;
    bool deferred_             = false; // a decode round's cache update awaits its commit
    std::uint64_t decode_rounds_ = 0;
    std::uint64_t budget_tokens_ = 0;
    std::vector<std::int32_t> ngram_window_;
    std::vector<std::uint32_t> row_ids_;
    Clock::time_point expert_state_saved_{};
    std::uint64_t expert_state_routed_ = 0;

    friend class qwen4_exp::Program;
};

} // namespace detail

// ---------------------------------------------------------------- contract wrappers

RequestBasePlan::RequestBasePlan(std::shared_ptr<detail::BasePlanImpl> impl) noexcept : impl_(std::move(impl)) {}
RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept            = default;
RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;
RequestBasePlan::~RequestBasePlan()                                     = default;

const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ ? impl_->summary : empty;
}

const qwen3_5::PreparedContextCache& RequestBasePlan::context_cache() const noexcept {
    static const qwen3_5::PreparedContextCache empty;
    return impl_ ? impl_->context_cache : empty;
}

ResumeState::ResumeState(std::unique_ptr<detail::ResumeStateImpl> impl) noexcept : impl_(std::move(impl)) {}
ResumeState::ResumeState(ResumeState&&) noexcept            = default;
ResumeState& ResumeState::operator=(ResumeState&&) noexcept = default;
ResumeState::~ResumeState()                                 = default;
std::uint32_t ResumeState::frontier() const noexcept { return impl_ ? impl_->frontier : 0; }

Program::Program(const execution::Parameters& parameters, DeviceContext& device, ProgramOptions options)
    : impl_(std::make_unique<detail::ProgramImpl>(parameters, device, std::move(options))) {}

Program::~Program() noexcept = default;

ProgramDevicePlan Program::plan_device(const ProgramOptions& options, const TextConfig& config,
                                       std::uint32_t public_tokens, std::span<const std::uint64_t> record_strides) {
    return detail::ProgramImpl::plan_device(options, config, public_tokens, record_strides);
}

const ProgramDevicePlan& Program::device_plan() const noexcept { return impl_->device_plan(); }
const VramSizing& Program::vram_sizing() const noexcept { return impl_->vram_sizing(); }
Program::ExpertCacheCounters Program::expert_cache_counters() const noexcept { return impl_->cache_counters(); }

RequestBasePlan Program::plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options) {
    return impl_->plan_request(std::move(prompt), options);
}

CheckpointHandle Program::install_external_checkpoint(const qwen3_5::PromptGraft&) {
    throw std::invalid_argument("Qwen3.8-Flash-Next has no prompt grafts (they need the context cache)");
}

std::optional<SourceCandidate> Program::inspect_source(const RequestBasePlan& base,
                                                       std::optional<CheckpointHandle> checkpoint, bool,
                                                       std::span<const CheckpointHandle>,
                                                       std::span<const CheckpointHandle>) const {
    return impl_->inspect_source(base, checkpoint);
}

namespace {
[[noreturn]] void no_checkpoint() {
    throw std::logic_error("Qwen3.8-Flash-Next makes no checkpoints (prefix reuse is off)");
}
} // namespace

qwen3_5::PrefixShortlistKey Program::checkpoint_key(CheckpointHandle, std::uint32_t) const { no_checkpoint(); }
runtime::ContextResourceUsage Program::checkpoint_footprint(std::span<const CheckpointHandle> handles) const {
    if (!handles.empty()) { no_checkpoint(); }
    return {};
}
CheckpointSummary Program::checkpoint_summary(CheckpointHandle) const { no_checkpoint(); }
CheckpointMetadata Program::checkpoint_metadata(CheckpointHandle) const { no_checkpoint(); }
bool Program::checkpoint_matches(CheckpointHandle, const RequestBasePlan&) const { return false; }
std::uint32_t Program::checkpoint_recovery_frontier(CheckpointHandle, const RequestBasePlan&, std::uint32_t) const {
    no_checkpoint();
}
std::uint64_t Program::checkpoint_recovery_loss(std::span<const CheckpointHandle>,
                                                std::span<const CheckpointHandle>) const {
    return 0;
}

runtime::ResourceReservation Program::reserve_units(std::span<const ExecutionUnit> units) {
    return impl_->reserve_units(units);
}

void Program::release_units(std::span<const SequenceHandle> sequences) noexcept { impl_->release_units(sequences); }

BindingReservation Program::start_binding(const RequestBasePlan& base, runtime::LaneId lane, const SourceCandidate& source,
                                          ResumeState* resume, ExecutionUnitKind, std::uint32_t) {
    return impl_->start_binding(base, lane, source, resume);
}

ContextProgress Program::poll_context(runtime::CancellationFlagView cancellation) {
    return impl_->poll_context(cancellation);
}

bool Program::has_context_transaction() const noexcept { return impl_->has_context_transaction(); }

PrefillProgress Program::advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming* failed_timing,
                                         runtime::TokenMaskProvider* masks) {
    return impl_->advance_prefill(sequence, failed_timing, masks);
}

ReplayProgress Program::advance_replay(SequenceHandle, runtime::ExecutionTiming*) {
    throw std::logic_error("Qwen3.8-Flash-Next does not pause, so it never replays");
}

PendingBatch Program::decode(std::span<const SequenceHandle> sequences, std::span<const runtime::RoundBudget> budgets,
                             runtime::ExecutionTiming* failed_timing, runtime::TokenMaskProvider* masks) {
    return impl_->decode(sequences, budgets, failed_timing, masks);
}

runtime::ExecutionTiming Program::append_forced_tokens(std::span<const SequenceHandle> sequences,
                                                       std::span<const TokenId> row_major_tokens,
                                                       std::uint32_t row_stride,
                                                       std::span<const std::optional<std::uint32_t>>,
                                                       runtime::ExecutionTiming* failed_timing) {
    return impl_->append_forced(sequences, row_major_tokens, row_stride, failed_timing);
}

CommitResult Program::commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
                             runtime::CommitObservation, runtime::ExecutionTiming*) {
    return impl_->commit(std::move(pending), decisions);
}

DiscardResult Program::abort_pending(PendingBatch&& pending) noexcept { return impl_->discard(std::move(pending)); }
FinishResult Program::finish(SequenceHandle sequence) noexcept { return impl_->finish(sequence); }
AbortResult Program::abort(SequenceHandle sequence) noexcept { return impl_->abort(sequence); }
void Program::fail_all_cleanup() noexcept { impl_->fail_all_cleanup(); }
PhysicalUsageSnapshot Program::physical_usage() const noexcept { return impl_->usage(); }
MemorySummary Program::memory_summary() const noexcept { return impl_->memory(); }

CheckpointImage Program::export_checkpoints(std::span<const CheckpointHandle>, const std::optional<PreparedSessionKey>&,
                                            std::string_view) const {
    no_checkpoint();
}

std::size_t Program::checkpoint_image_host_bytes(std::span<const std::uint8_t>, std::string_view) const {
    throw std::invalid_argument("Qwen3.8-Flash-Next has no context store");
}

std::optional<ImportedCheckpoints> Program::import_checkpoints(std::span<const std::uint8_t>, std::string_view) {
    throw std::invalid_argument("Qwen3.8-Flash-Next has no context store");
}

} // namespace ninfer::models::qwen4_exp
