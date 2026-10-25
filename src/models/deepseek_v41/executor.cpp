#include "models/deepseek_v41/executor.h"

#include "core/decode_graph.h"
#include "core/growable_region.h"
#include "core/vmm.h"
#include "core/weight_view.h"
#include "models/deepseek_v41/kernels.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/weight_input.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::deepseek_v41 {
namespace {

namespace gguf = ops::gguf;

constexpr std::size_t kAlign = 256;
constexpr int kLatent        = 512; // the attention latent: key and value of every head
constexpr int kIndexDim      = 128; // the indexer's head width
constexpr int kTailSlots     = 16;  // a ratio-2 compressor's kept rows per sequence
constexpr int kRing          = 256; // a sequence's window ring: the window and a round's drafts
constexpr int kCandidateBits = 32;
// Index scores of a prompt chunk are computed for as many rows at once as this many bytes hold.
constexpr std::size_t kScoreBudget = std::size_t(64) << 20;

std::size_t round_up(std::size_t value, std::size_t to) { return (value + to - 1) / to * to; }

const float* fp32(const Model& model, WeightId id, std::int64_t count) {
    const Tensor tensor = weight_tensor(model.weight(id).view, {std::int32_t(count)});
    if (tensor.dtype != DType::FP32) {
        throw std::invalid_argument(model.weight(id).name + " must be stored as fp32");
    }
    return static_cast<const float*>(tensor.data);
}

Weight matrix(const Model& model, WeightId id) {
    const Weight w = ops::prepare_linear_weight(model.input(id)).weight;
    if (!is_gguf(w.qtype)) {
        throw std::invalid_argument(model.weight(id).name + ": DeepSeek-V4.1 runs GGUF block matrices");
    }
    return w;
}

// The matrices of `ids`: one when they are rows of one stored parent, else each on its own.
std::vector<Weight> matrices(const Model& model, std::initializer_list<WeightId> ids) {
    std::vector<ops::WeightInput> inputs;
    for (const WeightId id : ids) { inputs.push_back(model.input(id)); }
    std::vector<Weight> out;
    if (inputs.size() > 1 && ops::joins(inputs)) {
        out.push_back(ops::prepare_linear_weight(std::span<const ops::WeightInput>(inputs)).weight);
    } else {
        for (const auto& input : inputs) { out.push_back(ops::prepare_linear_weight(input).weight); }
    }
    for (const Weight& w : out) {
        if (!is_gguf(w.qtype)) { throw std::invalid_argument("DeepSeek-V4.1 runs GGUF block matrices"); }
    }
    return out;
}

std::int64_t row_bytes(const Weight& w) {
    const auto block = gguf_block_shape(w.qtype);
    return std::int64_t(w.k / block.elements) * block.bytes;
}

// The rotation frequencies of the reference (precompute_freqs_cis) in FP32: base^(-2i/dim), and
// with YaRN the long wavelengths divided by `factor`, a linear ramp between the dimensions whose
// wavelengths complete beta_fast and beta_slow turns over the original context.
std::vector<float> rope_freq(int dim, float base, std::uint32_t original, float factor,
                             float beta_fast, float beta_slow) {
    std::vector<float> f(std::size_t(dim) / 2);
    for (std::size_t i = 0; i < f.size(); ++i) {
        f[i] = 1.0f / std::pow(base, float(2 * i) / float(dim));
    }
    if (original == 0) { return f; }
    const auto corrected = [&](double rotations) {
        return dim * std::log(double(original) / (rotations * 2 * M_PI)) / (2 * std::log(double(base)));
    };
    const double low  = std::max(std::floor(corrected(beta_fast)), 0.0);
    const double high = std::min(std::ceil(corrected(beta_slow)), double(dim - 1));
    const float span  = float(std::max(high - low, 1e-3));
    for (std::size_t i = 0; i < f.size(); ++i) {
        const float ramp   = std::clamp((float(i) - float(low)) / span, 0.0f, 1.0f);
        const float smooth = 1.0f - ramp;
        f[i]               = f[i] / factor * (1.0f - smooth) + f[i] * smooth;
    }
    return f;
}

// A context-sized cache plane: a row of `row_bytes` per sequence, `position_bytes` a token
// position. Owned whole, or, beside an elastic expert cache, a growable region whose rows are
// backed only as far as their sequence has reached (Impl::back).
struct KvArea {
    DeviceBuffer owned;
    GrowableRegion region;
    void* p = nullptr;
    std::size_t bytes = 0, row_bytes = 0, position_bytes = 0;
    std::vector<std::uint32_t> backed; // positions backed, by sequence

    KvArea() = default;
    KvArea(std::size_t owners, std::size_t row, std::size_t position, bool growing, int device)
        : bytes(owners * row), row_bytes(row), position_bytes(position), backed(owners, 0) {
        if (growing) {
            region = GrowableRegion(bytes, {.device = device});
            p      = region.data();
        } else {
            owned = DeviceBuffer(bytes);
            owned.fill(0);
            p = owned.p;
        }
    }
};

// One hyper-connection site's weights.
struct Site {
    const float *fn = nullptr, *base = nullptr, *scale = nullptr;
};

struct LayerPlan {
    int ratio = 0;
    bool kv_source = false, index_source = false, candidate_source = false, uses_candidates = false;
    int kv_from = -1; // the kv source whose caches the block reads
    Site hc_attention, hc_ffn;
    std::vector<Weight> query_kv; // [query_a; kv] as one parent, or the two
    Weight query_b, output_a, output_b, shared_down;
    std::vector<Weight> shared_gate_up;
    const float *attention_norm = nullptr, *ffn_norm = nullptr, *query_norm = nullptr,
                *kv_norm = nullptr, *sinks = nullptr, *freq = nullptr;
    // kv source: compressor and indexer key; index source: indexer query and head weights.
    const float *compressor_kv = nullptr, *compressor_gate = nullptr, *compressor_norm = nullptr,
                *index_key = nullptr, *index_key_norm = nullptr, *index_query = nullptr,
                *index_weights = nullptr;
    const float *router = nullptr, *router_bias = nullptr;
    gguf::GgmlType gate_up_type{}, down_type{};
    std::int64_t gate_up_row_bytes = 0, down_row_bytes = 0;
    DeviceBuffer tables; // device experts: [E] gate, [E] up, [E] gate_up, [E] down
    int delivery_layer = -1;
    int engram         = -1; // index among the Engram blocks
    Weight engram_projection;
    const float* engram_gate = nullptr;
    // Every sequence's window ring, [sequences][kRing][512] BF16.
    DeviceBuffer ring;
    // kv source: the compressed latents [sequences][rows][512] and index keys [sequences][rows][128]
    // (BF16), rows = max_context / ratio; ratio 2: the tails [sequences][kTailSlots][512] FP32 of kv
    // and score.
    KvArea latents, keys;
    int rows = 0;
    DeviceBuffer tail_kv, tail_score;
    // The checkpoint of every sequence's ring and tails (snapshot()).
    DeviceBuffer checkpoint_ring, checkpoint_kv, checkpoint_score;
};

struct GraphEntry {
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    bool warmed = false; // ran once eagerly, so every kernel is loaded before the capture
};

// The rows of a pass: a prompt chunk of one sequence, or decode rows of one sequence each.
struct Pass {
    int t     = 0;
    bool chunk = false;
    std::uint32_t sequence = 0, first = 0;
    // The rows whose logits the pass keeps (the last ones), and the logit column of the first.
    int logit_rows = 0, logit_column = 0;
};

} // namespace

struct Executor::Impl {
    const Model& model;
    DeviceContext& device;
    ExecutorOptions options;
    const TextConfig& config;
    cudaStream_t stream = nullptr;
    int hidden = 0, streams = 0, heads = 0, vocab = 0, window = 0, query_rank = 0, groups = 0,
        output_rank = 0, index_heads = 0, index_topk = 0, experts = 0, top_k = 0, width = 0;
    float eps = 0;
    // A prompt chunk attends over a staging ring of the window before it and the chunk itself,
    // which every block and sequence shares in turn.
    int prefill_ring = 0;
    DeviceBuffer staging;
    // Beside an elastic expert cache, the compressed caches grow as far as the sequences reach.
    bool growing = false;
    std::vector<KvArea*> sequence_areas;
    std::atomic<std::uint64_t> kv_backed{0};
    std::uint64_t kv_fixed = 0, kv_bytes = 0;
    std::vector<LayerPlan> layers;
    std::unique_ptr<ExpertDelivery> delivery;
    mutable std::mutex expert_stats_mutex;
    ExpertDeliveryStats expert_startup_stats;
    Weight embedding, head;
    const float* final_norm = nullptr;
    DeviceBuffer freq_window, freq_compressed;
    std::vector<std::uint32_t> positions;
    DeviceBuffer d_tokens, d_positions, d_sequences;
    std::unique_ptr<PinnedHostBuffer> h_staging;
    std::unique_ptr<CudaCompletionEvent> staged;
    DeviceBuffer logits_bf16;
    std::unique_ptr<WorkspaceArena> workspace;
    std::size_t workspace_bytes = 0;
    std::map<std::uint32_t, GraphEntry> graphs;
    std::uint32_t verify_width = 0;
    int capacity_rows = 0;   // the largest compressed cache, a decode pass's score width
    int candidate_words = 0; // words of a row's candidate mask
    int candidate_keep = 0, candidate_block = 0;
    // The Engram: the hash, each sequence's history, and the rows of a pass (pinned ids and rows,
    // then on the device: [Engram block][rows of the pass][columns] rows of 144 bytes).
    EngramHasher hasher;
    std::unique_ptr<qwen4_exp::NgramTableReader> engram;
    std::vector<EngramHasher::History> histories;
    std::vector<std::vector<std::int32_t>> verified; // a verification's tokens until commit
    std::unique_ptr<PinnedHostBuffer> h_engram_ids, h_engram_rows;
    std::unique_ptr<CudaCompletionEvent> engram_copied;
    DeviceBuffer d_engram;
    std::vector<std::uint64_t> engram_scratch;
    // One sequence's rings and tails, the state a branch pass returns to.
    DeviceBuffer branch_state;
    std::vector<int> decode_splits; // by decode rows

    Impl(const Model& m, DeviceContext& d, ExecutorOptions o)
        : model(m), device(d), options(std::move(o)), config(m.config()),
          hasher(m.config().engram, m.engram_token_map()) {
        if (options.sequences == 0 || options.prefill_chunk == 0 || options.max_context == 0) {
            throw std::invalid_argument("deepseek_v41 executor: invalid options");
        }
        if (device.size() != 1) { throw std::invalid_argument("DeepSeek-V4.1 runs on one device"); }
        if (options.draft_tokens > std::uint32_t(gguf::kMaxVectorColumns) - 1) {
            throw std::invalid_argument("deepseek_v41 executor: at most 7 drafts a round");
        }
        stream              = device.stream;
        const int device_id = device.rank(0).device;
        growing = !model.expert_banks().empty() && !options.expert_cache_bytes &&
                  vmm::granularity(device_id) != 0;
        hidden      = int(config.hidden_size);
        streams     = int(config.hc_mult);
        heads       = int(config.num_attention_heads);
        vocab       = int(config.vocab_size);
        window      = int(config.sliding_window);
        query_rank  = int(config.q_lora_rank);
        groups      = int(config.o_groups);
        output_rank = int(config.o_lora_rank);
        index_heads = int(config.index_heads);
        index_topk  = int(config.index_topk);
        experts     = int(config.num_experts);
        top_k       = int(config.top_k_experts);
        width       = int(config.moe_intermediate_size);
        eps         = config.rms_norm_eps;
        options.prefill_chunk = std::min<std::uint32_t>(options.prefill_chunk, 4096);
        prefill_ring          = int(round_up(std::size_t(window) + options.prefill_chunk, 256));
        positions.assign(options.sequences, 0);
        histories.assign(options.sequences, {});
        verified.assign(options.sequences, {});
        const auto& w = model.weights();
        embedding     = native_weight(model.weight(w.token_embedding).view);
        head          = matrix(model, w.output_head);
        final_norm    = fp32(model, w.final_norm, hidden);
        if (!is_gguf(embedding.qtype)) { throw std::invalid_argument("DeepSeek-V4.1: GGUF token table"); }
        {
            const auto window_freq = rope_freq(int(config.rope_dim), config.rope_theta, 0, 1, 0, 0);
            const auto yarn_freq   = rope_freq(int(config.rope_dim), config.compress_rope_theta,
                                               config.yarn_original, config.yarn_factor,
                                               config.yarn_beta_fast, config.yarn_beta_slow);
            freq_window     = DeviceBuffer(window_freq.size() * 4);
            freq_compressed = DeviceBuffer(yarn_freq.size() * 4);
            freq_window.copy_from_host(window_freq.data(), window_freq.size() * 4);
            freq_compressed.copy_from_host(yarn_freq.data(), yarn_freq.size() * 4);
        }
        const int sequences = int(options.sequences);
        {
            // The caches are sized up front for every sequence's whole context: refuse with the
            // numbers rather than fail inside an allocation.
            std::size_t needed = 0;
            for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
                needed += (options.checkpoints ? 2 : 1) * std::size_t(sequences) * kRing * kLatent * 2;
                if (config.kv_source[i]) {
                    const std::size_t rows = options.max_context / config.compress_ratios[i];
                    needed += std::size_t(sequences) * rows * (kLatent + kIndexDim) * 2;
                }
            }
            needed += std::size_t(prefill_ring) * kLatent * 2;
            std::size_t free_bytes = 0, total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            const std::size_t reserve = start_reserve();
            if (needed + reserve > free_bytes) {
                throw InsufficientMemory(
                    "DeepSeek-V4.1: the caches and checkpoints of " + std::to_string(sequences) +
                    " sequence(s) of " + std::to_string(options.max_context) + " tokens need " +
                    std::to_string(needed >> 20) + " MiB beside " + std::to_string(reserve >> 20) +
                    " MiB for the workspace and CUDA graphs, and " +
                    std::to_string(free_bytes >> 20) +
                    " MiB are free after the weights; lower --max-concurrency or --max-context");
            }
        }
        const std::size_t ring_bytes = std::size_t(kRing) * kLatent * 2;
        const std::size_t tail_bytes = std::size_t(kTailSlots) * kLatent * 4;
        std::size_t branch_bytes     = 0;
        for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
            const LayerWeights& lw = w.layers[i];
            LayerPlan p;
            p.ratio            = int(config.compress_ratios[i]);
            p.kv_source        = config.kv_source[i];
            p.index_source     = config.index_source[i];
            p.candidate_source = config.candidate_source == int(i);
            p.uses_candidates  = p.index_source && config.candidate_source >= 0 &&
                                config.candidate_source < int(i);
            p.kv_from = p.ratio > 0 ? config.kv_source_of(i) : -1;
            if (p.ratio > 0 && (p.kv_from < 0 || config.index_source_of(i) < 0)) {
                throw std::invalid_argument("DeepSeek-V4.1: a compressed block before its sources");
            }
            const std::int64_t mixes = config.hc_mixes(), flat = std::int64_t(streams) * hidden;
            p.hc_attention = {fp32(model, lw.hc_attention.fn, mixes * flat),
                              fp32(model, lw.hc_attention.base, mixes),
                              fp32(model, lw.hc_attention.scale, 3)};
            p.hc_ffn       = {fp32(model, lw.hc_ffn.fn, mixes * flat), fp32(model, lw.hc_ffn.base, mixes),
                              fp32(model, lw.hc_ffn.scale, 3)};
            const auto& at   = lw.attention;
            p.query_kv       = matrices(model, {at.query_a, at.kv});
            p.query_b        = matrix(model, at.query_b);
            p.output_a       = matrix(model, at.output_a);
            p.output_b       = matrix(model, at.output_b);
            p.attention_norm = fp32(model, lw.attention_norm, hidden);
            p.ffn_norm       = fp32(model, lw.ffn_norm, hidden);
            p.query_norm     = fp32(model, at.query_a_norm, query_rank);
            p.kv_norm        = fp32(model, at.kv_norm, kLatent);
            p.sinks          = fp32(model, at.sinks, heads);
            p.freq = static_cast<const float*>(p.ratio > 0 ? freq_compressed.p : freq_window.p);
            if (p.kv_source) {
                p.compressor_kv = fp32(model, *at.compressor_kv, std::int64_t(kLatent) * hidden);
                if (p.ratio > 1) {
                    p.compressor_gate = fp32(model, *at.compressor_gate, std::int64_t(kLatent) * hidden);
                }
                p.compressor_norm = fp32(model, *at.compressor_norm, kLatent);
                p.index_key       = fp32(model, *at.index_key, std::int64_t(kIndexDim) * kLatent);
                p.index_key_norm  = fp32(model, *at.index_key_norm, kIndexDim);
            }
            if (p.index_source) {
                p.index_query   = fp32(model, *at.index_query,
                                       std::int64_t(index_heads) * kIndexDim * query_rank);
                p.index_weights = fp32(model, *at.index_weights, std::int64_t(index_heads) * hidden);
            }
            const MoeWeights& mw = lw.moe;
            p.router           = fp32(model, mw.router, std::int64_t(experts) * hidden);
            p.router_bias      = fp32(model, mw.router_bias, experts);
            p.shared_gate_up   = matrices(model, {mw.shared_gate, mw.shared_up});
            p.shared_down      = matrix(model, mw.shared_down);
            if (model.expert_residency() == ExpertResidency::Device) {
                const Weight gate_up = matrix(model, *mw.gate_up);
                const Weight down    = matrix(model, *mw.down);
                p.gate_up_type       = ops::detail::gguf_type(gate_up.qtype);
                p.down_type          = ops::detail::gguf_type(down.qtype);
                p.gate_up_row_bytes  = row_bytes(gate_up);
                p.down_row_bytes     = row_bytes(down);
                std::vector<const void*> table(4 * std::size_t(experts));
                for (int x = 0; x < experts; ++x) {
                    const auto* gu = static_cast<const std::byte*>(gate_up.qdata) +
                                     std::int64_t(x) * 2 * width * p.gate_up_row_bytes;
                    table[x]               = gu;
                    table[experts + x]     = gu + std::int64_t(width) * p.gate_up_row_bytes;
                    table[2 * experts + x] = gu;
                    table[3 * experts + x] = static_cast<const std::byte*>(down.qdata) +
                                             std::int64_t(x) * hidden * p.down_row_bytes;
                }
                p.tables = DeviceBuffer(table.size() * sizeof(void*));
                p.tables.copy_from_host(table.data(), table.size() * sizeof(void*));
            } else {
                p.delivery_layer    = int(i);
                const auto& bank    = model.expert_banks()[i];
                p.gate_up_type      = ops::detail::gguf_type(bank.gate_up_type);
                p.down_type         = ops::detail::gguf_type(bank.down_type);
                p.gate_up_row_bytes = std::int64_t(bank.gate_up_row_bytes);
                p.down_row_bytes    = std::int64_t(bank.down_row_bytes);
            }
            if (lw.engram) {
                p.engram            = config.engram.index(i);
                p.engram_projection = matrix(model, lw.engram->projection);
                p.engram_gate       = fp32(model, lw.engram->gate, std::int64_t(streams) * hidden);
            }
            p.ring = DeviceBuffer(ring_bytes * sequences);
            p.ring.fill(0);
            kv_bytes += p.ring.bytes;
            branch_bytes += ring_bytes;
            if (options.checkpoints) {
                p.checkpoint_ring = DeviceBuffer(ring_bytes * sequences);
                kv_bytes += p.checkpoint_ring.bytes;
            }
            if (p.kv_source) {
                p.rows = int(options.max_context / std::uint32_t(p.ratio));
                p.latents = KvArea(std::size_t(sequences), std::size_t(p.rows) * kLatent * 2,
                                   std::size_t(kLatent) * 2 / p.ratio, growing, device_id);
                p.keys    = KvArea(std::size_t(sequences), std::size_t(p.rows) * kIndexDim * 2,
                                   std::size_t(kIndexDim) * 2 / p.ratio, growing, device_id);
                kv_bytes += p.latents.bytes + p.keys.bytes;
                capacity_rows = std::max(capacity_rows, p.rows);
                if (p.ratio > 1) {
                    p.tail_kv    = DeviceBuffer(tail_bytes * sequences);
                    p.tail_score = DeviceBuffer(tail_bytes * sequences);
                    p.tail_kv.fill(0);
                    p.tail_score.fill(0);
                    kv_bytes += 2 * p.tail_kv.bytes;
                    branch_bytes += 2 * tail_bytes;
                    if (options.checkpoints) {
                        p.checkpoint_kv    = DeviceBuffer(tail_bytes * sequences);
                        p.checkpoint_score = DeviceBuffer(tail_bytes * sequences);
                        kv_bytes += 2 * p.checkpoint_kv.bytes;
                    }
                }
            }
            if (p.candidate_source) {
                candidate_block = int(config.candidate_block_size);
                candidate_keep  = int(config.candidate_blocks);
                const int blocks = (p.rows + candidate_block - 1) / candidate_block;
                candidate_words  = (blocks + kCandidateBits - 1) / kCandidateBits;
            }
            layers.push_back(std::move(p));
        }
        staging = DeviceBuffer(std::size_t(prefill_ring) * kLatent * 2);
        staging.fill(0);
        kv_bytes += staging.bytes;
        branch_state = DeviceBuffer(branch_bytes);
        kv_bytes += branch_state.bytes;
        for (LayerPlan& p : layers) {
            if (!p.kv_source) { continue; }
            sequence_areas.push_back(&p.latents);
            sequence_areas.push_back(&p.keys);
        }
        kv_fixed = kv_bytes;
        if (growing) {
            for (KvArea* area : sequence_areas) { kv_fixed -= area->bytes; }
        }
        kv_backed = kv_fixed;
        const int sms = device.multiprocessor_count();
        decode_splits.assign(gguf::kMaxVectorColumns + 1, 1);
        for (int b = 1; b <= gguf::kMaxVectorColumns; ++b) {
            decode_splits[b] = std::clamp(2 * sms / (heads / 16 * b), 1, 40);
        }
        const std::size_t rows = std::max<std::size_t>(options.prefill_chunk, sequences);
        d_tokens    = DeviceBuffer(rows * 4);
        d_positions = DeviceBuffer(rows * 4);
        d_sequences = DeviceBuffer(rows * 4);
        h_staging   = std::make_unique<PinnedHostBuffer>(3 * rows * 4);
        staged      = std::make_unique<CudaCompletionEvent>(device);
        logits_bf16 = DeviceBuffer(std::size_t(vocab) * kMaxLogitRows * 2);
        if (options.engram_table) {
            const std::size_t engram_rows = rows * hasher.rows_per_token();
            engram = std::make_unique<qwen4_exp::NgramTableReader>(*options.engram_table,
                                                                   options.engram_read);
            h_engram_ids  = std::make_unique<PinnedHostBuffer>(engram_rows * 8);
            h_engram_rows = std::make_unique<PinnedHostBuffer>(engram_rows * kEngramRowBytes);
            d_engram      = DeviceBuffer(engram_rows * kEngramRowBytes);
            engram_copied = std::make_unique<CudaCompletionEvent>(device);
            engram_scratch.resize(engram_rows);
        }
        workspace_bytes = plan_workspace();
        {
            std::size_t free_bytes = 0, total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            if (workspace_bytes + kGraphReserve + unbacked() > free_bytes) {
                throw InsufficientMemory(
                    "DeepSeek-V4.1: the workspace of " + std::to_string(workspace_bytes >> 20) +
                    " MiB and " + std::to_string(kGraphReserve >> 20) +
                    " MiB for the CUDA graphs need more than the " + std::to_string(free_bytes >> 20) +
                    " MiB free after the caches; lower --prefill-chunk, --max-concurrency or "
                    "--max-context");
            }
        }
        workspace = std::make_unique<WorkspaceArena>(workspace_bytes);
        if (!model.expert_banks().empty()) { plan_delivery(); }
    }

    void plan_delivery() {
        std::vector<ExpertDeliveryLayer> sources;
        std::vector<std::uint64_t> sizes;
        for (const auto& bank : model.expert_banks()) {
            sources.push_back(bank.delivery);
            for (const auto& expert : bank.delivery.experts) {
                sizes.resize(std::max(sizes.size(), expert.size()), 0);
                for (std::size_t b = 0; b < expert.size(); ++b) {
                    sizes[b] = std::max(sizes[b], std::uint64_t(round_up(
                        expert[b].bytes + expert[b].zero_tail, kAlign)));
                }
            }
        }
        // One whole-layer bank at least (the transport adds a second where memory allows).
        std::uint64_t slot = 0;
        for (auto bytes : sizes) { slot += bytes; }
        const auto bank     = slot * config.num_experts;
        const auto metadata = std::uint64_t(options.prefill_chunk) * config.top_k_experts * 8 +
            std::uint64_t(config.num_hidden_layers) * config.num_experts * 4 * sizeof(void*) +
            config.num_experts * 4;
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        const auto cache = options.expert_cache_bytes.value_or(0);
        if (bank + metadata + cache + unbacked() + kGraphReserve > free) {
            throw InsufficientMemory("DeepSeek-V4.1: expert delivery needs " +
                std::to_string((bank + metadata + cache) >> 20) + " MiB beside " +
                std::to_string(unbacked() >> 20) +
                " MiB for the growing caches and the CUDA Graph reserve after workspace allocation");
        }
        ExpertDeliveryOptions transport;
        transport.pair_capacity   = options.prefill_chunk * config.top_k_experts;
        transport.cache_bytes     = options.expert_cache_bytes;
        transport.keep_free_bytes = kGraphReserve + (growing ? kKvHeadroom : 0);
        transport.elastic         = growing;
        delivery = std::make_unique<ExpertDelivery>(device.rank(0), model.expert_files(),
                                                    std::move(sources), transport);
    }

    static constexpr std::size_t kGraphReserve = std::size_t(320) << 20;
    static constexpr std::size_t kKvHeadroom   = std::size_t(256) << 20;

    std::size_t start_reserve() const {
        return options.prefill_chunk >= 2048 ? std::size_t(2048) << 20 : std::size_t(768) << 20;
    }

    // ---- Growing caches -------------------------------------------------------------------------

    [[nodiscard]] std::size_t unbacked() const {
        std::size_t bytes = 0;
        for (const KvArea* area : sequence_areas) {
            if (area->region) { bytes += area->region.bytes() - area->region.mapped_bytes(); }
        }
        return bytes;
    }

    void count_backed() {
        std::uint64_t bytes = kv_fixed;
        for (const KvArea* area : sequence_areas) {
            if (area->region) { bytes += area->region.mapped_bytes(); }
        }
        kv_backed.store(bytes, std::memory_order_relaxed);
    }

    [[nodiscard]] std::size_t missing(const KvArea& a, std::size_t owner, std::uint32_t positions) const {
        if (!a.region || positions <= a.backed[owner]) { return 0; }
        return a.region.unmapped_bytes(owner * a.row_bytes,
                                       std::min(a.row_bytes, std::size_t(positions) * a.position_bytes));
    }
    [[nodiscard]] bool extend(KvArea& a, std::size_t owner, std::uint32_t positions) {
        if (!a.region || positions <= a.backed[owner]) { return true; }
        if (!a.region.map(owner * a.row_bytes,
                          std::min(a.row_bytes, std::size_t(positions) * a.position_bytes), stream)) {
            return false;
        }
        a.backed[owner] = positions;
        return true;
    }

    // Unbacks `owner`'s row past `positions`, except granules another row still reaches into.
    void trim(KvArea& a, std::size_t owner, std::uint32_t positions) {
        if (!a.region || a.backed[owner] <= positions) { return; }
        const std::size_t g     = a.region.granule();
        const std::uint32_t was = a.backed[owner];
        a.backed[owner]         = positions;
        const auto reached = [&](std::size_t lo, std::size_t hi) {
            for (std::size_t o = 0; o < a.backed.size(); ++o) {
                const std::size_t row = o * a.row_bytes;
                const std::size_t end = row + std::min(a.row_bytes, std::size_t(a.backed[o]) * a.position_bytes);
                if (row < hi && lo < end) { return true; }
            }
            return false;
        };
        const std::size_t row = owner * a.row_bytes;
        const std::size_t hi  = row + std::min(a.row_bytes, std::size_t(was) * a.position_bytes);
        for (std::size_t at = (row + std::size_t(positions) * a.position_bytes) / g * g; at < hi; at += g) {
            if (!reached(at, at + g)) { a.region.unmap(at, g); }
        }
    }

    // Before a call that writes or reads `sequence`'s positions below `end`: its compressed rows
    // are backed up to the 256-position tile that holds `end`, the expert cache giving memory back
    // when the device keeps less than the CUDA Graph reserve beside them. Outside capture.
    void back(std::uint32_t sequence, std::uint32_t end) {
        if (!growing) { return; }
        const auto upto = std::uint32_t(std::min<std::size_t>(round_up(end, 256), options.max_context));
        std::size_t need = 0;
        for (const KvArea* area : sequence_areas) { need += missing(*area, sequence, upto); }
        const auto apply = [&] {
            bool ok = true;
            for (KvArea* area : sequence_areas) { ok = ok && extend(*area, sequence, upto); }
            return ok;
        };
        if (need == 0) {
            (void)apply();
            return;
        }
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        if (free < need + kGraphReserve) { delivery->shrink_cache(need + kGraphReserve - free); }
        while (!apply()) {
            if (delivery->shrink_cache(std::max(need, std::size_t(64) << 20)) == 0) {
                count_backed();
                throw InsufficientMemory("DeepSeek-V4.1: no device memory left to grow the "
                                         "compressed caches to " + std::to_string(upto) + " positions");
            }
        }
        count_backed();
    }

    void release(std::uint32_t sequence) {
        if (!growing || sequence_areas.empty() || sequence_areas.front()->backed[sequence] == 0) { return; }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (KvArea* area : sequence_areas) { trim(*area, sequence, 0); }
        count_backed();
        delivery->grow_cache();
    }

    ~Impl() {
        if (delivery || growing) { cudaStreamSynchronize(stream); }
    }
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;

    // ---- Workspace -----------------------------------------------------------------------------

    static constexpr int kHeadRows = 64; // LM-head columns per FP32 plane pass

    [[nodiscard]] int decode_rows() const {
        return std::min<int>(gguf::kMaxVectorColumns,
                             int(options.sequences) * (int(options.draft_tokens) + 1));
    }

    std::size_t plan_workspace() const {
        const std::size_t t = options.prefill_chunk, h = hidden, s = streams;
        const std::size_t latent_rows = std::size_t(heads) * kLatent;
        const std::size_t pairs = t * top_k, w = width;
        std::size_t gguf_scratch = 0;
        const auto product = [&](const Weight& m) {
            const ops::detail::GgufShape shape{m.qtype, m.n, m.k};
            gguf_scratch = std::max(gguf_scratch,
                                    ops::detail::gguf_project_workspace_bytes({&shape, 1}, 1, int(t)));
        };
        for (const LayerPlan& p : layers) {
            for (const Weight& m : p.query_kv) { product(m); }
            for (const Weight& m : p.shared_gate_up) { product(m); }
            Weight group = p.output_a;
            group.n      = output_rank;
            for (const Weight* m : {&p.query_b, &p.output_b, &p.shared_down, &std::as_const(group)}) {
                product(*m);
            }
            if (p.engram >= 0) { product(p.engram_projection); }
        }
        {
            Weight rows = head;
            rows.n      = std::min<int>(head.n, vocab);
            const ops::detail::GgufShape hs{rows.qtype, rows.n, rows.k};
            gguf_scratch = std::max(gguf_scratch,
                                    ops::detail::gguf_project_workspace_bytes({&hs, 1}, 1, kHeadRows));
        }
        std::size_t allocations = 0;
        const auto sum = [&](std::initializer_list<std::size_t> sizes) {
            std::size_t bytes = 0;
            for (const std::size_t n : sizes) {
                bytes += round_up(n, kAlign);
                ++allocations;
            }
            return bytes;
        };
        const std::size_t mixes = config.hc_mixes();
        const std::size_t pass = sum({t * s * h * 4, 2 * t * (2 * s + s * s) * 4, t * mixes * 4,
                                      t * std::size_t(index_topk) * 4, t * std::size_t(candidate_words) * 4});
        const std::size_t columns = std::size_t(config.engram.columns()) * config.engram.head_dim;
        const std::size_t engram_bytes = sum({t * columns * 2, t * 4, t * (s + 1) * h * 4});
        const std::size_t decode_width = std::size_t(decode_rows()) * std::size_t(capacity_rows) * 4;
        const std::size_t scores = std::max(kScoreBudget, decode_width);
        std::size_t partial = 0;
        for (int b = 1; b <= decode_rows(); ++b) {
            partial = std::max(partial, kernels::sparse_attention_workspace_floats(
                                            b, heads, decode_splits[std::size_t(b)]) * 4);
        }
        const std::size_t attention = sum({
            t * h * 2, t * h * 4, t * 1792 * 4, t * std::size_t(query_rank) * 2,
            t * std::size_t(query_rank) * 4, t * 4, t * 4, 8 * t * 4,
            t * latent_rows * 2, t * latent_rows * 2,
            t * std::size_t(groups) * output_rank * 4, t * std::size_t(groups) * output_rank * 2,
            t * h * 4,
            // compressor
            t * kLatent * 4, t * kLatent * 4, t * kLatent * 4, t * kIndexDim * 4,
            // indexer
            t * std::size_t(index_heads) * kIndexDim * 4, t * std::size_t(index_heads) * kIndexDim * 2,
            t * std::size_t(index_heads) * 4, scores, scores / 8 + 4096,
            partial});
        const std::size_t ffn = sum({
            t * h * 2, t * h * 4, t * 4, t * h * 4, t * std::size_t(experts) * 4, pairs * 4,
            pairs * 4, gguf::moe_routing_bytes(experts, int(pairs)), pairs * 2 * w * 4,
            pairs * w * 2, std::max(gguf::moe_matrix_activation_bytes(int(h), int(pairs)),
                                    gguf::vector_activation_bytes(int(h), int(t))),
            std::max(gguf::moe_matrix_activation_bytes(int(w), int(pairs)),
                     gguf::vector_activation_bytes(int(w), int(pairs))),
            pairs * h * 4, t * h * 8, t * 2 * w * 4, t * w * 2});
        const std::size_t head_bytes = sum({t * h * 2, std::size_t(kHeadRows) * vocab * 4, kHeadRows * 4});
        return pass + std::max({engram_bytes, attention, ffn, head_bytes}) + gguf_scratch +
               allocations * kAlign + (32u << 20);
    }

    // ---- Shared pieces ---------------------------------------------------------------------------

    template <typename T>
    T* alloc(std::size_t count) {
        return static_cast<T*>(workspace->alloc_bytes(count * sizeof(T)).data);
    }
    Tensor alloc_bf16(int rows, int t) { return workspace->alloc(DType::BF16, {rows, t}); }

    [[nodiscard]] static bool f16_sums(const Weight& w, int t) {
        const auto type = ops::detail::gguf_type(w.qtype);
        return t <= gguf::kMaxVectorColumns ? gguf::vector_activation_f16_sums(type)
                                            : gguf::matrix_activation_f16_sums(type);
    }

    // The products of `parts` (rows one after another) of the BF16 x [K, T] into FP32 [T][rows]
    // (column stride `stride`), x guarded for F16 sums where a part's activation keeps them
    // (kernels::guard_columns), its product's columns unscaled after. x may be scaled in place.
    void project(Tensor& x, std::span<const Weight> parts, float* out, int stride, int row = 0) {
        const int t = x.ne[1];
        bool guard  = false;
        for (const Weight& w : parts) { guard = guard || f16_sums(w, t); }
        float* scale = nullptr;
        if (guard) {
            scale = alloc<float>(t);
            kernels::guard_columns(static_cast<__nv_bfloat16*>(x.data), x.ne[0], t, scale, stream);
        }
        std::vector<ops::detail::GgufProduct> products(parts.size());
        int rows = 0;
        for (std::size_t i = 0; i < parts.size(); ++i) {
            products[i].weight   = &parts[i];
            products[i].f32      = out;
            products[i].f32_rows = stride;
            products[i].row      = row + rows;
            rows += parts[i].n;
        }
        ops::detail::gguf_project(x, products, *workspace, stream);
        if (guard) { kernels::unscale_columns(out + row, stride, rows, t, scale, stream); }
    }
    void project(Tensor& x, const Weight& w, float* out, int stride, int row = 0) {
        project(x, std::span<const Weight>(&w, 1), out, stride, row);
    }
    // The same into BF16 rows [T][w.n].
    void project_bf16(Tensor& x, const Weight& w, Tensor& out) {
        const int t      = x.ne[1];
        const bool guard = f16_sums(w, t);
        float* scale     = nullptr;
        if (guard) {
            scale = alloc<float>(t);
            kernels::guard_columns(static_cast<__nv_bfloat16*>(x.data), x.ne[0], t, scale, stream);
        }
        ops::detail::GgufProduct product;
        product.weight = &w;
        product.out    = &out;
        ops::detail::gguf_project(x, {&product, 1}, *workspace, stream);
        if (guard) {
            kernels::unscale_columns(static_cast<__nv_bfloat16*>(out.data), w.n, w.n, t, scale, stream);
        }
    }

    // Tokens, positions, and when given sequences, to the device.
    void stage(std::span<const std::int32_t> a, std::span<const std::int32_t> b,
               std::span<const std::int32_t> c) {
        staged->synchronize();
        auto* host = static_cast<std::int32_t*>(h_staging->data());
        const std::size_t rows = std::max<std::size_t>(options.prefill_chunk, options.sequences);
        if (a.size() > rows || b.size() > rows || c.size() > rows) {
            throw std::invalid_argument("deepseek_v41 executor: staging overflow");
        }
        std::copy(a.begin(), a.end(), host);
        std::copy(b.begin(), b.end(), host + rows);
        std::copy(c.begin(), c.end(), host + 2 * rows);
        CUDA_CHECK(cudaMemcpyAsync(d_tokens.p, host, a.size() * 4, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_positions.p, host + rows, b.size() * 4,
                                   cudaMemcpyHostToDevice, stream));
        if (!c.empty()) {
            CUDA_CHECK(cudaMemcpyAsync(d_sequences.p, host + 2 * rows, c.size() * 4,
                                       cudaMemcpyHostToDevice, stream));
        }
        staged->record(stream);
    }

    // ---- Engram rows -----------------------------------------------------------------------------

    // The table rows of a pass's rows to the device, block-major: rows[r] is (sequence, token)
    // after the sequence's `history` (advanced when `advance`).
    void read_engram(std::span<const std::int32_t> tokens, std::span<const std::uint32_t> owners,
                     std::span<EngramHasher::History> history_of, bool advance) {
        if (!engram) { return; }
        const std::size_t t = tokens.size();
        const std::uint32_t per = hasher.rows_per_token(), columns = hasher.columns();
        const std::size_t blocks = per / columns;
        for (std::size_t r = 0; r < t;) {
            // Consecutive rows of one sequence hash together.
            std::size_t end = r + 1;
            while (end < t && owners[end] == owners[r]) { ++end; }
            hasher.rows(history_of[r], tokens.subspan(r, end - r), engram_scratch.data() + r * per,
                        advance);
            r = end;
        }
        engram_copied->synchronize();
        auto* ids = static_cast<std::uint64_t*>(h_engram_ids->data());
        for (std::size_t b = 0; b < blocks; ++b) {
            for (std::size_t r = 0; r < t; ++r) {
                std::memcpy(ids + (b * t + r) * columns, engram_scratch.data() + r * per + b * columns,
                            columns * 8);
            }
        }
        const std::size_t n = t * per;
        engram->read_rows({ids, n}, {static_cast<std::uint8_t*>(h_engram_rows->data()), n * kEngramRowBytes});
        CUDA_CHECK(cudaMemcpyAsync(d_engram.p, h_engram_rows->data(), n * kEngramRowBytes,
                                   cudaMemcpyHostToDevice, stream));
        engram_copied->record(stream);
    }

    void advance_history(std::uint32_t sequence, std::span<const std::int32_t> tokens) {
        for (const std::int32_t token : tokens) {
            histories[sequence].push(hasher.compressed(token), hasher.hash().ngram_size - 1);
        }
    }

    // ---- Blocks ----------------------------------------------------------------------------------

    void embed(int t, float* h) {
        gguf::dequantize_rows(ops::detail::gguf_type(embedding.qtype), embedding.qdata,
                              row_bytes(embedding), hidden,
                              static_cast<const std::int32_t*>(d_tokens.p), t, h,
                              std::int64_t(streams) * hidden, stream);
        kernels::expand_streams(h, t, streams, hidden, stream);
    }

    struct Coefficients {
        float *pre = nullptr, *post = nullptr, *comb = nullptr;
    };

    void mixes(const Site& site, const float* h, int t, float* scratch, const Coefficients& out) {
        const int m = int(config.hc_mixes());
        kernels::gemm_f32(m, t, streams * hidden, site.fn, h, scratch, stream);
        kernels::HcSplitArgs a;
        a.h          = h;
        a.mixes      = scratch;
        a.tokens     = t;
        a.streams    = streams;
        a.hidden     = hidden;
        a.scale      = site.scale;
        a.base       = site.base;
        a.norm_eps   = eps;
        a.hc_eps     = config.hc_eps;
        a.iterations = int(config.hc_sinkhorn_iters);
        a.pre        = out.pre;
        a.post       = out.post;
        a.comb       = out.comb;
        kernels::hc_split(a, stream);
    }

    void engram_block(const LayerPlan& p, float* h, int t) {
        if (!engram) { return; }
        auto scope        = workspace->scope();
        const int columns = int(hasher.columns());
        const int rows    = t * columns;
        Tensor e          = alloc_bf16(columns * int(config.engram.head_dim), t);
        kernels::engram_rows(static_cast<const std::uint8_t*>(d_engram.p) +
                                 std::size_t(p.engram) * rows * kEngramRowBytes,
                             rows, static_cast<__nv_bfloat16*>(e.data), stream);
        const int kv_rows = (streams + 1) * hidden;
        auto* kv          = alloc<float>(std::size_t(t) * kv_rows);
        project(e, p.engram_projection, kv, kv_rows);
        kernels::engram_gate(h, kv, p.engram_gate, t, streams, hidden, eps, stream);
    }

    // The block's attention: its output y added to the streams through `post` and `comb`.
    void attention(const LayerPlan& p, float* h, const float* pre, int t, const Pass& s,
                   std::int32_t* topk, std::uint32_t* candidates, const Coefficients& c) {
        auto scope      = workspace->scope();
        const auto* pos = static_cast<const std::int32_t*>(d_positions.p);
        const auto* seq = s.chunk ? nullptr : static_cast<const std::int32_t*>(d_sequences.p);
        Tensor x        = alloc_bf16(hidden, t);
        auto* x32       = alloc<float>(std::size_t(t) * hidden);
        kernels::hc_pre_norm(h, pre, p.attention_norm, t, streams, hidden, eps,
                             static_cast<__nv_bfloat16*>(x.data), x32, stream);
        const int qkv_rows = query_rank + kLatent;
        auto* qkv          = alloc<float>(std::size_t(t) * qkv_rows);
        project(x, p.query_kv, qkv, qkv_rows);
        Tensor qr  = alloc_bf16(query_rank, t);
        auto* qr32 = alloc<float>(std::size_t(t) * query_rank);
        kernels::rms_rows(qkv, qkv_rows, query_rank, t, p.query_norm, eps,
                          static_cast<__nv_bfloat16*>(qr.data), qr32, stream);
        // The window latents: a chunk's into the staging ring after the window before it, decode
        // rows' into their sequences' rings.
        const std::int64_t ring_stride = std::int64_t(kRing) * kLatent;
        auto* ring = static_cast<__nv_bfloat16*>(p.ring.p);
        auto* own  = ring + std::int64_t(s.sequence) * ring_stride;
        auto* stage_ring = static_cast<__nv_bfloat16*>(staging.p);
        if (s.chunk) {
            kernels::ring_copy(own, kRing, stage_ring, prefill_ring,
                               std::max(0, int(s.first) - window + 1), int(s.first), stream);
        }
        kernels::KvPostArgs kv;
        kv.kv        = qkv + query_rank;
        kv.kv_stride = qkv_rows;
        kv.tokens    = t;
        kv.norm      = p.kv_norm;
        kv.eps       = eps;
        kv.positions = pos;
        kv.freq      = p.freq;
        kv.cache     = s.chunk ? stage_ring : ring;
        kv.seq_stride = s.chunk ? 0 : ring_stride;
        kv.sequences = seq;
        kv.ring      = s.chunk ? prefill_ring : kRing;
        kernels::kv_post(kv, stream);
        Tensor q = alloc_bf16(heads * kLatent, t);
        project_bf16(qr, p.query_b, q);
        kernels::rotate_heads(static_cast<const __nv_bfloat16*>(q.data), t, heads, kLatent, pos,
                              p.freq, false, static_cast<__nv_bfloat16*>(q.data), stream);
        kernels::AttentionArgs a;
        if (p.ratio > 0) {
            if (p.kv_source) { compress_block(p, x32, t, s); }
            const LayerPlan& source = layers[std::size_t(p.kv_from)];
            if (p.index_source) { index_block(p, source, qr32, x32, t, s, topk, candidates); }
            a.compressed            = static_cast<const __nv_bfloat16*>(source.latents.p);
            a.compressed_seq_stride = std::int64_t(source.rows) * kLatent;
            a.topk                  = topk;
            a.k                     = index_topk;
        }
        Tensor o            = alloc_bf16(heads * kLatent, t);
        a.q                 = static_cast<const __nv_bfloat16*>(q.data);
        a.tokens            = t;
        a.heads             = heads;
        a.positions         = pos;
        a.sequences         = seq;
        a.sequence          = int(s.sequence);
        a.window            = s.chunk ? stage_ring : ring;
        a.window_seq_stride = s.chunk ? 0 : ring_stride;
        a.ring              = s.chunk ? prefill_ring : kRing;
        a.window_size       = window;
        a.sinks             = p.sinks;
        a.scale             = 1.0f / std::sqrt(float(kLatent));
        a.splits            = s.chunk ? 1 : decode_splits[std::size_t(t)];
        if (a.splits > 1) {
            a.partial = alloc<float>(kernels::sparse_attention_workspace_floats(t, heads, a.splits));
        }
        a.out = static_cast<__nv_bfloat16*>(o.data);
        kernels::sparse_attention(a, stream);
        if (s.chunk) {
            const int end = int(s.first) + t;
            kernels::ring_copy(stage_ring, prefill_ring, own, kRing, std::max(0, end - window), end,
                               stream);
        }
        // The grouped output projection: group g's heads (rounded as the reference rounds them)
        // through rows [g * o_rank, (g + 1) * o_rank) of output_a, into the latent's rows there.
        auto* grouped = static_cast<__nv_bfloat16*>(q.data); // q is spent
        kernels::attention_output(static_cast<const __nv_bfloat16*>(o.data), t, heads, groups, pos,
                                  p.freq, grouped, stream);
        const int latent_rows = groups * output_rank;
        const int group_k     = heads / groups * kLatent;
        auto* latent          = alloc<float>(std::size_t(t) * latent_rows);
        for (int g = 0; g < groups; ++g) {
            Weight part = p.output_a;
            part.qdata  = static_cast<const std::byte*>(p.output_a.qdata) +
                         std::int64_t(g) * output_rank * row_bytes(p.output_a);
            part.n      = output_rank;
            Tensor xg(grouped + std::int64_t(g) * t * group_k, DType::BF16, {group_k, t});
            project(xg, part, latent, latent_rows, g * output_rank);
        }
        Tensor latent16 = alloc_bf16(latent_rows, t);
        kernels::f32_to_bf16(latent, static_cast<__nv_bfloat16*>(latent16.data),
                             std::int64_t(t) * latent_rows, stream);
        auto* y = alloc<float>(std::size_t(t) * hidden);
        project(latent16, p.output_b, y, hidden);
        kernels::hc_post(h, y, c.post, c.comb, t, streams, hidden, stream);
    }

    // A kv source's compressor: the compressed latents of the groups its rows complete, and their
    // indexer keys, into its caches.
    void compress_block(const LayerPlan& p, const float* x32, int t, const Pass& s) {
        auto scope      = workspace->scope();
        const auto* pos = static_cast<const std::int32_t*>(d_positions.p);
        const auto* seq = s.chunk ? nullptr : static_cast<const std::int32_t*>(d_sequences.p);
        auto* kv        = alloc<float>(std::size_t(t) * kLatent);
        kernels::gemm_f32(kLatent, t, hidden, p.compressor_kv, x32, kv, stream);
        float* score = nullptr;
        if (p.ratio > 1) {
            score = alloc<float>(std::size_t(t) * kLatent);
            kernels::gemm_f32(kLatent, t, hidden, p.compressor_gate, x32, score, stream);
        }
        auto* latent = alloc<float>(std::size_t(t) * kLatent);
        kernels::CompressArgs c;
        c.kv              = kv;
        c.score           = score;
        c.tokens          = t;
        c.ratio           = p.ratio;
        c.positions       = pos;
        c.sequences       = seq;
        c.sequence        = int(s.sequence);
        c.tail_kv         = static_cast<float*>(p.tail_kv.p);
        c.tail_score      = static_cast<float*>(p.tail_score.p);
        c.tail_seq_stride = std::int64_t(kTailSlots) * kLatent;
        c.slots           = kTailSlots;
        c.tail_from       = s.chunk ? std::max(0, t - kTailSlots) : 0;
        c.norm            = p.compressor_norm;
        c.eps             = eps;
        c.latent          = latent;
        kernels::compress(c, stream);
        kernels::compress_tail(c, stream);
        auto* key = alloc<float>(std::size_t(t) * kIndexDim);
        kernels::gemm_f32(kIndexDim, t, kLatent, p.index_key, latent, key, stream);
        kernels::LatentArgs l;
        l.latent           = latent;
        l.key              = key;
        l.tokens           = t;
        l.ratio            = p.ratio;
        l.positions        = pos;
        l.sequences        = seq;
        l.sequence         = int(s.sequence);
        l.key_norm         = p.index_key_norm;
        l.eps              = eps;
        l.freq             = p.freq;
        l.index_cache      = static_cast<__nv_bfloat16*>(p.keys.p);
        l.index_seq_stride = std::int64_t(p.rows) * kIndexDim;
        l.kv_cache         = static_cast<__nv_bfloat16*>(p.latents.p);
        l.kv_seq_stride    = std::int64_t(p.rows) * kLatent;
        kernels::latent_post(l, stream);
    }

    // An index source: each row's index_topk compressed positions (topk), and at the candidate
    // source the candidate blocks the later index sources keep to.
    void index_block(const LayerPlan& p, const LayerPlan& source, const float* qr32,
                     const float* x32, int t, const Pass& s, std::int32_t* topk,
                     std::uint32_t* candidates) {
        auto scope      = workspace->scope();
        const auto* pos = static_cast<const std::int32_t*>(d_positions.p);
        const auto* seq = s.chunk ? nullptr : static_cast<const std::int32_t*>(d_sequences.p);
        const int q_rows = index_heads * kIndexDim;
        auto* q32        = alloc<float>(std::size_t(t) * q_rows);
        kernels::gemm_f32(q_rows, t, query_rank, p.index_query, qr32, q32, stream);
        auto* q = alloc<__nv_bfloat16>(std::size_t(t) * q_rows);
        kernels::rotate_heads(q32, t, index_heads, kIndexDim, pos, p.freq, true, q, stream);
        auto* weights = alloc<float>(std::size_t(t) * index_heads);
        kernels::gemm_f32(index_heads, t, hidden, p.index_weights, x32, weights, stream);
        const int width = s.chunk ? std::max(1, int(s.first + std::uint32_t(t)) / p.ratio)
                                  : source.rows;
        int sub = t;
        if (s.chunk) {
            sub = int(std::clamp<std::size_t>(kScoreBudget / (std::size_t(width) * 4), 4, std::size_t(t)));
            sub = std::max(4, sub / 4 * 4);
        }
        auto* scores = alloc<float>(std::size_t(std::min(sub, t)) * width);
        float* maxima = p.candidate_source
                            ? alloc<float>(std::size_t(std::min(sub, t)) * candidate_words * kCandidateBits)
                            : nullptr;
        for (int r0 = 0; r0 < t; r0 += sub) {
            const int n = std::min(sub, t - r0);
            kernels::ScoreArgs a;
            a.q            = q + std::int64_t(r0) * q_rows;
            a.weights      = weights + std::int64_t(r0) * index_heads;
            a.weight_scale = 1.0f / std::sqrt(float(kIndexDim)) / std::sqrt(float(index_heads));
            a.heads        = index_heads;
            a.tokens       = n;
            a.ratio        = p.ratio;
            a.width        = width;
            a.group        = s.chunk ? 4 : 1;
            a.positions    = pos + r0;
            a.sequences    = seq != nullptr ? seq + r0 : nullptr;
            a.sequence     = int(s.sequence);
            a.keys         = static_cast<const __nv_bfloat16*>(source.keys.p);
            a.key_seq_stride = std::int64_t(source.rows) * kIndexDim;
            if (p.uses_candidates) {
                a.candidates      = candidates + std::int64_t(r0) * candidate_words;
                a.candidate_words = candidate_words;
                a.block           = candidate_block;
            }
            a.out = scores;
            kernels::index_scores(a, stream);
            if (p.candidate_source) {
                kernels::select_candidates(scores, width, n, pos + r0, p.ratio, candidate_block,
                                           candidate_keep, candidates + std::int64_t(r0) * candidate_words,
                                           candidate_words, maxima, stream);
            }
            kernels::select_topk(scores, width, n, pos + r0, p.ratio, index_topk,
                                 topk + std::int64_t(r0) * index_topk, stream);
        }
    }

    // The block's MoE: routed experts and the shared expert into y, added to the streams.
    void ffn(const LayerPlan& p, float* h, const float* pre, int t, const Coefficients& c) {
        auto scope = workspace->scope();
        Tensor x   = alloc_bf16(hidden, t);
        auto* x32  = alloc<float>(std::size_t(t) * hidden);
        kernels::hc_pre_norm(h, pre, p.ffn_norm, t, streams, hidden, eps,
                             static_cast<__nv_bfloat16*>(x.data), x32, stream);
        // Every product of x (the routed and the shared experts' gate and up rows) sees it guarded
        // for F16 sums; the clamped SwiGLU takes the factors back before its nonlinearity.
        auto* scale = alloc<float>(t);
        kernels::guard_columns(static_cast<__nv_bfloat16*>(x.data), hidden, t, scale, stream);
        auto* y = alloc<float>(std::size_t(t) * hidden);
        const auto shared = [&] {
            auto shared_scope = workspace->scope();
            auto* gu          = alloc<float>(std::size_t(t) * 2 * width);
            ops::detail::GgufProduct parts[2];
            int rows = 0;
            for (std::size_t i = 0; i < p.shared_gate_up.size(); ++i) {
                parts[i].weight   = &p.shared_gate_up[i];
                parts[i].f32      = gu;
                parts[i].f32_rows = 2 * width;
                parts[i].row      = rows;
                rows += p.shared_gate_up[i].n;
            }
            ops::detail::gguf_project(x, {parts, p.shared_gate_up.size()}, *workspace, stream);
            Tensor middle = alloc_bf16(width, t);
            kernels::clamped_swiglu(gu, 2 * width, t, width, config.swiglu_limit, scale, 1,
                                    static_cast<__nv_bfloat16*>(middle.data), width, stream);
            project(middle, p.shared_down, y, hidden);
        };
        routed(p, x, scale, x32, t, y, shared);
        kernels::hc_post(h, y, c.post, c.comb, t, streams, hidden, stream);
    }

    // The routed experts of the block (their output added to y after `shared` stores the shared
    // expert's there), run while missing experts are copied in.
    template <typename Shared>
    void routed(const LayerPlan& p, const Tensor& x, const float* scale, const float* x32, int t,
                float* y, Shared&& shared) {
        const int e = experts, k = top_k, w = width, pairs = t * k;
        auto* logits = alloc<float>(std::size_t(t) * e);
        kernels::gemm_f32(e, t, hidden, p.router, x32, logits, stream);
        auto* ids     = alloc<std::int32_t>(pairs);
        auto* weights = alloc<float>(pairs);
        kernels::route(logits, p.router_bias, t, e, k, config.routed_scaling_factor, ids, weights,
                       stream);
        void* routing_space = workspace->alloc_bytes(gguf::moe_routing_bytes(e, pairs)).data;
        const gguf::MoeRouting routing = gguf::moe_sort_routes(ids, pairs, e, routing_space, stream);
        std::optional<ExpertDelivery::Call> call;
        if (delivery) {
            call = delivery->begin(std::size_t(p.delivery_layer), ids, std::uint32_t(pairs), stream);
        }
        const auto* tables = delivery ? delivery->tables(std::size_t(p.delivery_layer))
                                      : static_cast<const void* const*>(p.tables.p);
        const gguf::MoeTable gate_up{tables + 2 * e, p.gate_up_row_bytes, 2 * w, hidden};
        const gguf::MoeTable down{tables + 3 * e, p.down_row_bytes, hidden, w};
        const int max_active = std::min(pairs, e);
        const auto* xb       = static_cast<const __nv_bfloat16*>(x.data);
        auto* gu             = alloc<float>(std::size_t(pairs) * 2 * w);
        auto* middle         = alloc<__nv_bfloat16>(std::size_t(pairs) * w);
        kernels::MoeCombineArgs combine;
        combine.y      = y;
        combine.tokens = t;
        combine.hidden = hidden;
        if (t <= gguf::kMaxVectorColumns) {
            const int chunk = t <= 1 ? 1 : t <= 2 ? 2 : t <= 4 ? 4 : 8;
            void* act  = workspace->alloc_bytes(gguf::vector_activation_bytes(hidden, t)).data;
            void* act2 = workspace->alloc_bytes(gguf::vector_activation_bytes(w, pairs)).data;
            auto* fixed = alloc<unsigned long long>(std::size_t(t) * hidden);
            gguf::quantize_vector_activation(xb, hidden, t, nullptr, act, stream);
            CUDA_CHECK(cudaMemsetAsync(fixed, 0, std::size_t(t) * hidden * 8, stream));
            CUDA_CHECK(cudaMemsetAsync(gu, 0, std::size_t(pairs) * 2 * w * 4, stream));
            gguf::MoeOutput rows;
            rows.f32 = gu;
            gguf::MoeOutput out;
            out.weighted = fixed;
            out.weights  = weights;
            const auto compute = [&](const gguf::MoeRouting& part) {
                gguf::moe_vector_product(p.gate_up_type, gate_up, part, max_active, k, k, act, t,
                                         rows, chunk, stream);
                kernels::clamped_swiglu(gu, 2 * w, pairs, w, config.swiglu_limit, scale, k, middle,
                                        w, stream);
                gguf::quantize_vector_activation(middle, w, pairs, nullptr, act2, stream);
                gguf::moe_vector_product(p.down_type, down, part, max_active, 1, k, act2, pairs,
                                         out, chunk, stream);
            };
            if (call) {
                // Both stages add to the same exact fixed-point sum, each over its experts of the
                // call's sorted routes; the shared expert runs between them, while the missing
                // experts arrive. The SwiGLU and its quantization read every pair: the cached stage
                // finds the other pairs' rows zero, the missing stage the cached pairs' rows.
                const auto stage = [&](const std::int32_t* set) {
                    gguf::MoeRouting part = routing;
                    part.active_count     = set;
                    part.active           = set + 1;
                    return part;
                };
                compute(stage(call->cached_experts));
                shared();
                delivery->await(stream);
                compute(stage(call->missing_experts));
            } else {
                compute(routing);
                shared();
            }
            combine.fixed = fixed;
            kernels::moe_combine(combine, stream);
            return;
        }
        shared();
        if (delivery) { delivery->await(stream); }
        if (gguf::moe_matrix_supported(p.gate_up_type, 2 * w, hidden, true)) {
            void* act = workspace->alloc_bytes(gguf::moe_matrix_activation_bytes(hidden, pairs)).data;
            gguf::quantize_moe_matrix_activation(p.gate_up_type, xb, hidden, routing, pairs, k, act,
                                                 stream);
            gguf::moe_matrix_product(p.gate_up_type, gate_up, routing, max_active, pairs, true, act,
                                     gu, stream);
        } else {
            void* act = workspace->alloc_bytes(gguf::vector_activation_bytes(hidden, t)).data;
            gguf::quantize_vector_activation(xb, hidden, t, nullptr, act, stream);
            gguf::MoeOutput out;
            out.f32 = gu;
            gguf::moe_vector_product(p.gate_up_type, gate_up, routing, max_active, k, k, act, t, out,
                                     8, stream);
        }
        kernels::clamped_swiglu(gu, 2 * w, pairs, w, config.swiglu_limit, scale, k, middle, w,
                                stream);
        auto* rows = alloc<float>(std::size_t(pairs) * hidden);
        if (gguf::moe_matrix_supported(p.down_type, hidden, w, true)) {
            void* act = workspace->alloc_bytes(gguf::moe_matrix_activation_bytes(w, pairs)).data;
            gguf::quantize_moe_matrix_activation(p.down_type, middle, w, routing, pairs, 1, act,
                                                 stream);
            gguf::moe_matrix_product(p.down_type, down, routing, max_active, pairs, true, act, rows,
                                     stream);
        } else {
            void* act = workspace->alloc_bytes(gguf::vector_activation_bytes(w, pairs)).data;
            gguf::quantize_vector_activation(middle, w, pairs, nullptr, act, stream);
            gguf::MoeOutput out;
            out.f32 = rows;
            gguf::moe_vector_product(p.down_type, down, routing, max_active, 1, k, act, pairs, out, 8,
                                     stream);
        }
        combine.rows    = rows;
        combine.weights = weights;
        combine.top_k   = k;
        kernels::moe_combine(combine, stream);
    }

    // The head of `rows` final-normed rows into logits_bf16 columns [first, first + rows).
    void head_logits(const __nv_bfloat16* rows_x, int rows, int first) {
        for (int at = 0; at < rows; at += kHeadRows) {
            const int n = std::min(kHeadRows, rows - at);
            auto scope  = workspace->scope();
            Tensor x    = alloc_bf16(hidden, n);
            CUDA_CHECK(cudaMemcpyAsync(x.data, rows_x + std::int64_t(at) * hidden,
                                       std::size_t(n) * hidden * 2, cudaMemcpyDeviceToDevice, stream));
            auto* plane = alloc<float>(std::size_t(n) * vocab);
            project(x, head, plane, vocab);
            kernels::f32_to_bf16(plane,
                                 static_cast<__nv_bfloat16*>(logits_bf16.p) + std::int64_t(first + at) * vocab,
                                 std::int64_t(n) * vocab, stream);
        }
    }

    // A pass over staged rows (tokens, positions and, for decode rows, sequences on the device; the
    // Engram rows copied).
    void run(const Pass& s) {
        const int t = s.t;
        workspace->reset();
        auto scope = workspace->scope();
        auto* h    = alloc<float>(std::size_t(t) * streams * hidden);
        embed(t, h);
        const int site = 2 * streams + streams * streams;
        Coefficients attention_site, ffn_site;
        for (Coefficients* c : {&attention_site, &ffn_site}) {
            auto* base = alloc<float>(std::size_t(t) * site);
            c->pre     = base;
            c->post    = base + std::size_t(t) * streams;
            c->comb    = base + std::size_t(t) * 2 * streams;
        }
        auto* mix_scratch = alloc<float>(std::size_t(t) * config.hc_mixes());
        auto* topk        = alloc<std::int32_t>(std::size_t(t) * index_topk);
        auto* candidates  = candidate_words > 0 ? alloc<std::uint32_t>(std::size_t(t) * candidate_words)
                                                : nullptr;
        const float* pre = nullptr; // the first block reads stream 0
        for (const LayerPlan& p : layers) {
            if (p.engram >= 0) { engram_block(p, h, t); }
            mixes(p.hc_attention, h, t, mix_scratch, attention_site);
            attention(p, h, pre, t, s, topk, candidates, attention_site);
            mixes(p.hc_ffn, h, t, mix_scratch, ffn_site);
            ffn(p, h, attention_site.pre, t, ffn_site);
            pre = ffn_site.pre;
        }
        if (s.logit_rows == 0) { return; }
        const int first = t - s.logit_rows;
        Tensor x        = alloc_bf16(hidden, s.logit_rows);
        kernels::hc_pre_norm(h + std::int64_t(first) * streams * hidden, pre + std::int64_t(first) * streams,
                             final_norm, s.logit_rows, streams, hidden, eps,
                             static_cast<__nv_bfloat16*>(x.data), nullptr, stream);
        head_logits(static_cast<const __nv_bfloat16*>(x.data), s.logit_rows, s.logit_column);
    }

    // ---- Calls -----------------------------------------------------------------------------------

    void forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                 std::uint32_t logit_rows, std::uint32_t logit_column = 0) {
        const int t = int(tokens.size());
        if (sequence >= options.sequences || t <= 0 || std::uint32_t(t) > options.prefill_chunk ||
            logit_rows > std::min<std::uint32_t>(t, kMaxLogitRows) ||
            logit_column + logit_rows > kMaxLogitRows) {
            throw std::invalid_argument("deepseek_v41 forward: invalid call");
        }
        const std::uint32_t first = positions[sequence];
        if (first + std::uint32_t(t) > options.max_context) {
            throw std::invalid_argument("deepseek_v41 forward: context capacity exceeded");
        }
        back(sequence, first + std::uint32_t(t));
        std::vector<std::int32_t> pos(t);
        for (int i = 0; i < t; ++i) { pos[i] = std::int32_t(first) + i; }
        stage(tokens, pos, {});
        read_engram_rows(tokens, std::vector<std::uint32_t>(std::size_t(t), sequence), true);
        Pass s;
        s.t            = t;
        s.chunk        = true;
        s.sequence     = sequence;
        s.first        = first;
        s.logit_rows   = int(logit_rows);
        s.logit_column = int(logit_column);
        run(s);
        positions[sequence] = first + std::uint32_t(t);
    }

    // The Engram rows of a pass whose row r belongs to owners[r], from that sequence's history.
    void read_engram_rows(std::span<const std::int32_t> tokens, std::span<const std::uint32_t> owners,
                          bool advance) {
        if (!engram) { return; }
        std::vector<EngramHasher::History> view(tokens.size());
        // Each run of rows of one sequence continues that sequence's history; a verification's
        // rows hash from a copy (advance false) and commit() advances the history.
        std::size_t r = 0;
        while (r < tokens.size()) {
            std::size_t end = r + 1;
            while (end < tokens.size() && owners[end] == owners[r]) { ++end; }
            view[r] = histories[owners[r]];
            r       = end;
        }
        read_engram(tokens, owners, view, false);
        if (advance) {
            for (std::size_t i = 0; i < tokens.size(); ++i) { advance_history(owners[i], tokens.subspan(i, 1)); }
        }
    }

    void decode(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens) {
        const int b = int(sequences.size());
        if (b <= 0 || std::uint32_t(b) > options.sequences || tokens.size() != sequences.size() ||
            b > gguf::kMaxVectorColumns) {
            throw std::invalid_argument("deepseek_v41 decode: invalid batch");
        }
        std::vector<std::int32_t> pos(b), seq(b);
        for (int i = 0; i < b; ++i) {
            if (sequences[i] >= options.sequences ||
                std::count(sequences.begin(), sequences.begin() + i, sequences[i]) != 0) {
                throw std::invalid_argument("deepseek_v41 decode: invalid sequence");
            }
            if (positions[sequences[i]] + 1 > options.max_context) {
                throw std::invalid_argument("deepseek_v41 decode: context capacity exceeded");
            }
            pos[i] = std::int32_t(positions[sequences[i]]);
            seq[i] = std::int32_t(sequences[i]);
        }
        for (int i = 0; i < b; ++i) { back(sequences[i], positions[sequences[i]] + 1); }
        stage(tokens, pos, seq);
        read_engram_rows(tokens, sequences, true);
        Pass s;
        s.t          = b;
        s.logit_rows = b;
        run_graph(std::uint32_t(b), [&] { run(s); });
        for (int i = 0; i < b; ++i) { positions[sequences[i]] += 1; }
    }

    template <typename Body>
    void run_graph(std::uint32_t key, Body&& body) {
        if (!options.cuda_graphs) {
            body();
            return;
        }
        GraphEntry& entry = graphs[key];
        if (!entry.warmed) {
            body();
            entry.warmed = true;
            return;
        }
        if (!entry.executable.ready()) {
            entry.definition.capture(stream, [&] { body(); });
            entry.executable.instantiate(entry.definition);
        }
        entry.executable.launch(stream);
    }

    void verify(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens) {
        const int n = int(sequences.size());
        if (options.draft_tokens == 0 || n <= 0 || tokens.size() % sequences.size() != 0) {
            throw std::invalid_argument("deepseek_v41 verify: invalid call");
        }
        const int w = int(tokens.size()) / n, rows = n * w;
        if (w < 1 || std::uint32_t(w) > options.draft_tokens + 1 || rows > gguf::kMaxVectorColumns) {
            throw std::invalid_argument("deepseek_v41 verify: invalid width");
        }
        std::vector<std::int32_t> pos(rows), seq(rows);
        std::vector<std::uint32_t> owners(rows);
        for (int j = 0; j < n; ++j) {
            const std::uint32_t s = sequences[j];
            if (s >= options.sequences || std::count(sequences.begin(), sequences.begin() + j, s) != 0) {
                throw std::invalid_argument("deepseek_v41 verify: invalid sequence");
            }
            if (positions[s] + std::uint32_t(w) > options.max_context) {
                throw std::invalid_argument("deepseek_v41 verify: context capacity exceeded");
            }
            for (int i = 0; i < w; ++i) {
                pos[j * w + i]    = std::int32_t(positions[s]) + i;
                seq[j * w + i]    = std::int32_t(s);
                owners[j * w + i] = s;
            }
            verified[s].assign(tokens.begin() + j * w, tokens.begin() + (j + 1) * w);
        }
        for (const std::uint32_t s : sequences) { back(s, positions[s] + std::uint32_t(w)); }
        stage(tokens, pos, seq);
        read_engram_rows(tokens, owners, false);
        Pass s;
        s.t          = rows;
        s.logit_rows = rows;
        run_graph(16 + std::uint32_t(rows), [&] { run(s); });
        verify_width = std::uint32_t(w);
    }

    void commit(std::span<const std::uint32_t> sequences, std::span<const std::uint32_t> kept) {
        if (kept.size() != sequences.size() || verify_width == 0) {
            throw std::invalid_argument("deepseek_v41 commit: invalid call");
        }
        for (std::size_t j = 0; j < sequences.size(); ++j) {
            const std::uint32_t s = sequences[j];
            if (s >= options.sequences || kept[j] < 1 || kept[j] > verify_width ||
                verified[s].size() != verify_width) {
                throw std::invalid_argument("deepseek_v41 commit: invalid row");
            }
            advance_history(s, std::span<const std::int32_t>(verified[s]).first(kept[j]));
            verified[s].clear();
            positions[s] += kept[j];
        }
        verify_width = 0;
    }

    // ---- Branches -------------------------------------------------------------------------------

    // One sequence's rings and tails, (device, bytes) in order: the live state or its checkpoint.
    std::vector<std::pair<std::uint8_t*, std::size_t>> state(std::uint32_t sequence, bool checkpoint) const {
        std::vector<std::pair<std::uint8_t*, std::size_t>> out;
        const std::size_t ring_bytes = std::size_t(kRing) * kLatent * 2;
        const std::size_t tail_bytes = std::size_t(kTailSlots) * kLatent * 4;
        for (const LayerPlan& p : layers) {
            const auto& ring = checkpoint ? p.checkpoint_ring : p.ring;
            out.emplace_back(static_cast<std::uint8_t*>(ring.p) + sequence * ring_bytes, ring_bytes);
            if (p.kv_source && p.ratio > 1) {
                const auto& kv    = checkpoint ? p.checkpoint_kv : p.tail_kv;
                const auto& score = checkpoint ? p.checkpoint_score : p.tail_score;
                out.emplace_back(static_cast<std::uint8_t*>(kv.p) + sequence * tail_bytes, tail_bytes);
                out.emplace_back(static_cast<std::uint8_t*>(score.p) + sequence * tail_bytes, tail_bytes);
            }
        }
        return out;
    }

    void copy_state(std::uint32_t sequence, bool to_checkpoint, bool from_checkpoint) {
        const auto from = state(sequence, from_checkpoint);
        const auto to   = state(sequence, to_checkpoint);
        for (std::size_t i = 0; i < from.size(); ++i) {
            CUDA_CHECK(cudaMemcpyAsync(to[i].first, from[i].first, from[i].second,
                                       cudaMemcpyDeviceToDevice, stream));
        }
    }

    void copy_branch_state(std::uint32_t sequence, bool save) {
        auto* kept = static_cast<std::uint8_t*>(branch_state.p);
        for (const auto& [device_state, bytes] : state(sequence, false)) {
            CUDA_CHECK(cudaMemcpyAsync(save ? kept : device_state, save ? device_state : kept, bytes,
                                       cudaMemcpyDeviceToDevice, stream));
            kept += bytes;
        }
    }

    void branches(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                  std::span<const std::uint32_t> lengths) {
        if (sequence >= options.sequences || lengths.empty() || lengths.size() > kMaxLogitRows ||
            std::find(lengths.begin(), lengths.end(), 0U) != lengths.end() ||
            std::accumulate(lengths.begin(), lengths.end(), std::uint64_t{0}) != tokens.size()) {
            throw std::invalid_argument("deepseek_v41 forward_branches: invalid branches");
        }
        for (const std::uint32_t length : lengths) {
            if (length > options.prefill_chunk) {
                throw std::invalid_argument("deepseek_v41 forward_branches: a branch longer than a chunk");
            }
        }
        const std::uint32_t position         = positions[sequence];
        const EngramHasher::History history  = histories[sequence];
        copy_branch_state(sequence, true);
        std::size_t at = 0;
        for (std::size_t b = 0; b < lengths.size(); ++b) {
            forward(sequence, tokens.subspan(at, lengths[b]), 1, std::uint32_t(b));
            at += lengths[b];
            copy_branch_state(sequence, false);
            positions[sequence] = position;
            histories[sequence] = history;
        }
    }

    void warm_up() {
        // The prompt route once (kernels load at first launch), then every decode batch and every
        // verification's row count eagerly, then their graphs. Under lazy module loading a kernel
        // loaded while a graph waits on the device for the experts' host service deadlocks the
        // two, so every route has loaded its kernels before the first graph runs, and each graph
        // has finished before the next call.
        const std::vector<std::int32_t> prompt(std::min<std::uint32_t>(options.prefill_chunk, 16), 2);
        forward(0, prompt, 1);
        reset_state(0);
        const std::uint32_t batches = std::min<std::uint32_t>(options.sequences, 8);
        // A verification's graph is keyed by its rows: one (sequences, width) shape for each.
        std::vector<std::pair<std::uint32_t, std::uint32_t>> rounds;
        for (std::uint32_t rows = 1; options.draft_tokens > 0 && rows <= 8; ++rows) {
            for (std::uint32_t w = std::min(options.draft_tokens + 1, rows); w >= 1; --w) {
                if (rows % w == 0 && rows / w <= batches) {
                    rounds.emplace_back(rows / w, w);
                    break;
                }
            }
        }
        for (int pass = 0; pass < 2; ++pass) { // eagerly, then captured
            std::vector<std::uint32_t> sequences;
            std::vector<std::int32_t> tokens;
            for (std::uint32_t b = 1; b <= batches; ++b) {
                sequences.push_back(b - 1);
                tokens.push_back(2);
                decode(sequences, tokens);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                for (const std::uint32_t s : sequences) { reset_state(s); }
            }
            for (const auto& [n, w] : rounds) {
                const std::vector<std::uint32_t> owners(sequences.begin(), sequences.begin() + n);
                verify(owners, std::vector<std::int32_t>(std::size_t(n) * w, 2));
                commit(owners, std::vector<std::uint32_t>(n, 1));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                for (const std::uint32_t s : owners) { reset_state(s); }
            }
        }
        if (delivery) {
            delivery->drain();
            delivery->prime(options.expert_profile ? &*options.expert_profile : nullptr);
            std::lock_guard lock(expert_stats_mutex);
            expert_startup_stats = delivery->stats();
        }
    }

    void reset_state(std::uint32_t sequence) {
        positions.at(sequence) = 0;
        histories.at(sequence) = {};
        verified.at(sequence).clear();
    }

    // ---- Snapshots and images ----------------------------------------------------------------------

    [[nodiscard]] std::uint64_t image_bytes(std::uint32_t position) const {
        std::uint64_t bytes = 0;
        for (const auto& [unused, n] : state(0, false)) { bytes += n; }
        for (const LayerPlan& p : layers) {
            if (!p.kv_source) { continue; }
            bytes += std::uint64_t(position / std::uint32_t(p.ratio)) * (kLatent + kIndexDim) * 2;
        }
        return bytes;
    }

    // Copies a sequence's image between the device and the host pages: its rings and tails (from
    // or to its checkpoint with `checkpoint`), then the compressed rows below `position`.
    void copy_image(std::uint32_t sequence, std::uint32_t position, std::span<std::byte* const> pages,
                    std::size_t page_bytes, bool save, bool checkpoint) {
        if (page_bytes == 0 || pages.size() < (image_bytes(position) + page_bytes - 1) / page_bytes) {
            throw std::invalid_argument("deepseek_v41 image: pages do not hold the image");
        }
        std::size_t at = 0;
        const auto copy = [&](std::uint8_t* device_bytes, std::size_t bytes) {
            while (bytes > 0) {
                const std::size_t in_page = std::min(bytes, page_bytes - at % page_bytes);
                std::byte* host           = pages[at / page_bytes] + at % page_bytes;
                if (save) {
                    CUDA_CHECK(cudaMemcpyAsync(host, device_bytes, in_page, cudaMemcpyDeviceToHost, stream));
                } else {
                    CUDA_CHECK(cudaMemcpyAsync(device_bytes, host, in_page, cudaMemcpyHostToDevice, stream));
                }
                device_bytes += in_page;
                bytes -= in_page;
                at += in_page;
            }
        };
        for (const auto& [device_state, bytes] : state(sequence, checkpoint)) { copy(device_state, bytes); }
        for (const LayerPlan& p : layers) {
            if (!p.kv_source) { continue; }
            const std::size_t rows = position / std::uint32_t(p.ratio);
            copy(static_cast<std::uint8_t*>(p.latents.p) + std::size_t(sequence) * p.latents.row_bytes,
                 rows * kLatent * 2);
            copy(static_cast<std::uint8_t*>(p.keys.p) + std::size_t(sequence) * p.keys.row_bytes,
                 rows * kIndexDim * 2);
        }
    }

    SequenceImage save_image(std::uint32_t sequence, const SequenceSnapshot* at,
                             std::span<std::byte* const> pages, std::size_t page_bytes) {
        const std::uint32_t position = at != nullptr ? at->position : positions.at(sequence);
        copy_image(sequence, position, pages, page_bytes, true, at != nullptr);
        return {position, at != nullptr ? at->history : histories.at(sequence)};
    }

    void load_image(std::uint32_t sequence, const SequenceImage& image,
                    std::span<std::byte* const> pages, std::size_t page_bytes) {
        if (image.position > options.max_context) {
            throw std::invalid_argument("deepseek_v41 load_image: invalid image");
        }
        back(sequence, image.position);
        copy_image(sequence, image.position, pages, page_bytes, false, false);
        positions.at(sequence) = image.position;
        histories.at(sequence) = image.history;
        verified.at(sequence).clear();
    }

    void snapshot(std::uint32_t sequence, SequenceSnapshot& out) {
        if (!options.checkpoints) { throw std::logic_error("deepseek_v41: an executor without checkpoints"); }
        copy_state(sequence, true, false);
        out.position = positions[sequence];
        out.history  = histories[sequence];
    }

    void restore(std::uint32_t sequence, const SequenceSnapshot& from) {
        if (!options.checkpoints) { throw std::logic_error("deepseek_v41: an executor without checkpoints"); }
        copy_state(sequence, false, true);
        positions[sequence] = from.position;
        histories[sequence] = from.history;
        verified[sequence].clear();
    }
};

Executor::Executor(const Model& model, DeviceContext& device, ExecutorOptions options)
    : impl_(std::make_unique<Impl>(model, device, std::move(options))) {}

Executor::~Executor() = default;

const ExecutorOptions& Executor::options() const noexcept { return impl_->options; }

ExecutorMemory Executor::memory() const noexcept {
    return ExecutorMemory{.kv_bytes        = impl_->kv_bytes,
                          .kv_backed_bytes = impl_->kv_backed.load(std::memory_order_relaxed),
                          .kv_grows        = impl_->growing,
                          .workspace_bytes = impl_->workspace_bytes,
                          .engram_resident_bytes = impl_->engram ? impl_->engram->resident_bytes() : 0,
                          .experts = impl_->delivery ? impl_->delivery->memory() : ExpertDeliveryMemory{}};
}

std::uint64_t Executor::image_bytes(std::uint32_t position) const { return impl_->image_bytes(position); }

SequenceImage Executor::save_image(std::uint32_t sequence, const SequenceSnapshot* at,
                                   std::span<std::byte* const> pages, std::size_t page_bytes) {
    return impl_->save_image(sequence, at, pages, page_bytes);
}

void Executor::load_image(std::uint32_t sequence, const SequenceImage& image,
                          std::span<std::byte* const> pages, std::size_t page_bytes) {
    impl_->load_image(sequence, image, pages, page_bytes);
}

void Executor::reset(std::uint32_t sequence) {
    impl_->reset_state(sequence);
    impl_->release(sequence);
}

std::uint32_t Executor::position(std::uint32_t sequence) const { return impl_->positions.at(sequence); }

void Executor::warm_up() {
    impl_->warm_up();
    check_experts();
}

void Executor::check_experts() const {
    if (impl_->delivery) { impl_->delivery->check(); }
}
ExpertDeliveryStats Executor::expert_stats() const {
    if (!impl_->delivery) { return {}; }
    std::lock_guard lock(impl_->expert_stats_mutex);
    return impl_->delivery->stats() - impl_->expert_startup_stats;
}
ExpertRouteCounts Executor::expert_totals() const {
    return impl_->delivery ? impl_->delivery->totals() : ExpertRouteCounts{};
}

std::optional<NgramTableStats> Executor::engram_stats() const {
    if (!impl_->engram) { return std::nullopt; }
    return impl_->engram->counters();
}

std::string Executor::engram_description() const {
    if (!impl_->engram) { return {}; }
    return impl_->engram->io_backend();
}

void Executor::forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                       std::uint32_t logit_rows) {
    if (logit_rows == 0) { throw std::invalid_argument("deepseek_v41 forward: no logit rows"); }
    impl_->forward(sequence, tokens, logit_rows);
}

void Executor::forward_branches(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                                std::span<const std::uint32_t> lengths) {
    impl_->branches(sequence, tokens, lengths);
}

void Executor::forward_prompt_branches(std::uint32_t sequence, std::span<const std::int32_t> prompt,
                                       std::span<const std::int32_t> tokens,
                                       std::span<const std::uint32_t> lengths) {
    if (!prompt.empty()) { impl_->forward(sequence, prompt, 0); }
    impl_->branches(sequence, tokens, lengths);
}

void Executor::decode(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens) {
    impl_->decode(sequences, tokens);
}

Tensor Executor::logits(std::uint32_t rows) const {
    if (rows == 0 || rows > kMaxLogitRows) {
        throw std::invalid_argument("deepseek_v41 logits: invalid row count");
    }
    return Tensor(impl_->logits_bf16.p, DType::BF16, {impl_->vocab, static_cast<std::int32_t>(rows)});
}

cudaStream_t Executor::head_stream() const noexcept { return impl_->stream; }

std::uint32_t Executor::draft_tokens() const noexcept { return impl_->options.draft_tokens; }

void Executor::verify(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens) {
    impl_->verify(sequences, tokens);
}

void Executor::commit(std::span<const std::uint32_t> sequences, std::span<const std::uint32_t> kept) {
    impl_->commit(sequences, kept);
}

void Executor::snapshot(std::uint32_t sequence, SequenceSnapshot& out) { impl_->snapshot(sequence, out); }

void Executor::restore(std::uint32_t sequence, const SequenceSnapshot& from) {
    impl_->restore(sequence, from);
}

} // namespace ninfer::models::deepseek_v41
