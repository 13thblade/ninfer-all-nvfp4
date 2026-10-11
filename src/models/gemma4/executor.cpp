#include "models/gemma4/executor.h"

#include "core/decode_graph.h"
#include "core/weight_view.h"
#include "models/gemma4/kernels.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/weight_input.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/ggml_fattn.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::gemma4 {
namespace {

namespace gguf = ops::gguf;

constexpr std::size_t kAlign = 256;

std::size_t round_up(std::size_t value, std::size_t to) { return (value + to - 1) / to * to; }

const float* fp32(const Model& model, WeightId id, std::int32_t count) {
    const Tensor tensor = weight_tensor(model.weight(id).view, {count});
    if (tensor.dtype != DType::FP32) {
        throw std::invalid_argument(model.weight(id).name + " must be stored as fp32");
    }
    return static_cast<const float*>(tensor.data);
}

Weight matrix(const Model& model, std::initializer_list<WeightId> ids) {
    std::vector<ops::WeightInput> inputs;
    for (const WeightId id : ids) { inputs.push_back(model.input(id)); }
    if (inputs.size() == 1) { return ops::prepare_linear_weight(inputs.front()).weight; }
    if (!ops::joins(inputs)) {
        throw std::invalid_argument("gemma4: projection parts are not one stored parent");
    }
    return ops::prepare_linear_weight(std::span<const ops::WeightInput>(inputs)).weight;
}

std::int64_t row_bytes(const Weight& w) {
    const auto block = gguf_block_shape(w.qtype);
    return std::int64_t(w.k / block.elements) * block.bytes;
}

struct LayerPlan {
    bool full    = false;
    int head_dim = 0, kv_heads = 0, qkv_rows = 0;
    Weight qkv, output, gate_up, down;
    const float *input_norm = nullptr, *post_attention_norm = nullptr, *ffn_norm = nullptr,
                *post_ffn_norm = nullptr, *layer_scalar = nullptr, *q_norm = nullptr,
                *k_norm = nullptr;
    // MoE.
    bool moe = false;
    const float *router = nullptr, *router_scale = nullptr, *expert_scale = nullptr,
                *pre_norm = nullptr, *post_norm = nullptr, *mlp_post_norm = nullptr;
    gguf::GgmlType expert_type{};
    std::int64_t gate_up_row_bytes = 0, down_row_bytes = 0;
    DeviceBuffer tables; // pointers: [E] gate, [E] up, [E] gate_up, [E] down
    // The F16 K/V cache: [sequences][kv heads][slots][D].
    DeviceBuffer k, v;
    int slots = 0, ring = 0;
    std::int64_t seq_stride = 0, head_stride = 0;
};

// Keys of the decode graphs: the batch size.
struct GraphEntry {
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    bool warmed = false; // ran once eagerly, so every kernel is loaded before the capture
};

} // namespace

struct Executor::Impl {
    const Model& model;
    DeviceContext& device;
    ExecutorOptions options;
    const TextConfig& config;
    cudaStream_t stream = nullptr;
    int hidden = 0, query_heads = 0, vocab = 0, ring = 0, global_slots = 0;
    // A prompt chunk's sliding-window attention needs the window before it and the chunk itself:
    // a ring of prefill_ring slots that every sequence's chunks share, one block at a time
    // (staging_k/v), while each sequence keeps only its last `ring` (the window) positions.
    int prefill_ring = 0;
    DeviceBuffer staging_k, staging_v;
    float eps = 1e-6f;
    std::vector<LayerPlan> layers;
    Weight head; // tied token table
    const float* final_norm = nullptr;
    DeviceBuffer inv_freq_sliding, inv_freq_full;
    std::vector<std::uint32_t> positions;
    // Device staging: tokens, positions, sequences for up to max(prefill_chunk, sequences) rows.
    DeviceBuffer d_tokens, d_positions, d_sequences;
    std::unique_ptr<PinnedHostBuffer> h_staging;
    // Recorded after the staging copies: the next stage() rewrites the pinned buffer only once
    // the previous copies have read it (a prompt chunk that keeps no logits is not synchronized).
    std::unique_ptr<CudaCompletionEvent> staged;
    DeviceBuffer logits_bf16;
    std::unique_ptr<WorkspaceArena> workspace;
    std::size_t workspace_bytes = 0;
    std::uint64_t kv_bytes      = 0;
    int decode_splits_full = 1, decode_splits_sliding = 1;
    std::map<std::uint32_t, GraphEntry> graphs;

    Impl(const Model& m, DeviceContext& d, ExecutorOptions o)
        : model(m), device(d), options(o), config(m.config()) {
        if (options.sequences == 0 || options.prefill_chunk == 0 || options.max_context == 0) {
            throw std::invalid_argument("gemma4 executor: invalid options");
        }
        if (device.size() != 1) {
            throw std::invalid_argument("Gemma 4 runs on one device");
        }
        stream      = device.stream;
        hidden      = int(config.hidden_size);
        query_heads = int(config.num_attention_heads);
        vocab       = int(config.vocab_size);
        eps         = config.rms_norm_eps;
        options.prefill_chunk = std::min<std::uint32_t>(options.prefill_chunk, 4096);
        ring         = int(round_up(config.sliding_window, 256));
        prefill_ring = int(round_up(config.sliding_window + options.prefill_chunk, 256));
        global_slots = int(round_up(options.max_context + 1, 256));
        positions.assign(options.sequences, 0);
        const auto& w = model.weights();
        head          = matrix(model, {w.output_head});
        final_norm    = fp32(model, w.final_norm, hidden);
        // RoPE frequencies, as the reference computes them in FP32.
        {
            std::vector<float> sliding(config.head_dim / 2), full(config.global_head_dim / 2, 0.0f);
            for (std::size_t i = 0; i < sliding.size(); ++i) {
                sliding[i] = float(1.0 / std::pow(double(config.rope_theta_sliding),
                                                  double(2 * i) / double(config.head_dim)));
            }
            for (std::uint32_t i = 0; i < config.rotary_pairs_full; ++i) {
                full[i] = float(1.0 / std::pow(double(config.rope_theta_full),
                                               double(2 * i) / double(config.global_head_dim)));
            }
            inv_freq_sliding = DeviceBuffer(sliding.size() * 4);
            inv_freq_sliding.copy_from_host(sliding.data(), sliding.size() * 4);
            inv_freq_full = DeviceBuffer(full.size() * 4);
            inv_freq_full.copy_from_host(full.data(), full.size() * 4);
        }
        const int sequences = int(options.sequences);
        {
            // The caches are sized up front for every sequence's whole context: refuse with the
            // numbers rather than fail inside an allocation.
            // Each sequence may also hold one checkpoint of its sliding rings (snapshot()).
            std::size_t needed = 0, staging = 0;
            for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
                const bool full         = config.full_attention[i];
                const std::size_t slots = full ? global_slots : ring;
                const std::size_t row   = 2 * std::size_t(config.layer_kv_heads(i)) *
                                        config.layer_head_dim(i) * sizeof(__half);
                needed += slots * row * std::size_t(sequences) * (full ? 1 : 2);
                if (!full) { staging = std::max(staging, std::size_t(prefill_ring) * row); }
            }
            needed += staging;
            std::size_t free_bytes = 0, total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            constexpr std::size_t kReserve = std::size_t(1536) << 20; // workspace and graphs
            if (needed + kReserve > free_bytes) {
                throw std::runtime_error(
                    "Gemma 4: the KV cache and checkpoints of " + std::to_string(sequences) +
                    " sequence(s) of " +
                    std::to_string(options.max_context) + " tokens needs " +
                    std::to_string(needed >> 20) + " MiB beside ~1.5 GiB of workspace, and " +
                    std::to_string(free_bytes >> 20) +
                    " MiB are free after the weights; lower --max-concurrency or --max-context");
            }
        }
        for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
            const LayerWeights& lw = w.layers[i];
            LayerPlan p;
            p.full     = config.full_attention[i];
            p.head_dim = int(config.layer_head_dim(i));
            p.kv_heads = int(config.layer_kv_heads(i));
            p.qkv      = lw.attention.value
                             ? matrix(model, {lw.attention.query, lw.attention.key, *lw.attention.value})
                             : matrix(model, {lw.attention.query, lw.attention.key});
            p.qkv_rows = p.qkv.n;
            p.output   = matrix(model, {lw.attention.output});
            p.gate_up  = matrix(model, {lw.mlp_gate, lw.mlp_up});
            p.down     = matrix(model, {lw.mlp_down});
            p.input_norm          = fp32(model, lw.input_norm, hidden);
            p.post_attention_norm = fp32(model, lw.post_attention_norm, hidden);
            p.ffn_norm            = fp32(model, lw.ffn_norm, hidden);
            p.post_ffn_norm       = fp32(model, lw.post_ffn_norm, hidden);
            p.layer_scalar        = fp32(model, lw.layer_scalar, 1);
            p.q_norm              = fp32(model, lw.attention.query_norm, p.head_dim);
            p.k_norm              = fp32(model, lw.attention.key_norm, p.head_dim);
            for (const Weight* m : {&p.qkv, &p.output, &p.gate_up, &p.down}) {
                if (!is_gguf(m->qtype)) {
                    throw std::invalid_argument("Gemma 4 runs GGUF block matrices");
                }
            }
            if (lw.moe) {
                const MoeWeights& mw = *lw.moe;
                const int e = int(config.num_experts), width = int(config.moe_intermediate_size);
                p.moe           = true;
                p.router        = fp32(model, mw.router, e * hidden);
                p.router_scale  = fp32(model, mw.router_scale, hidden);
                p.expert_scale  = fp32(model, mw.expert_scale, e);
                p.pre_norm      = fp32(model, mw.pre_norm, hidden);
                p.post_norm     = fp32(model, mw.post_norm, hidden);
                p.mlp_post_norm = fp32(model, mw.mlp_post_norm, hidden);
                const Weight gate_up = matrix(model, {mw.gate_up});
                const Weight down    = matrix(model, {mw.down});
                if (gate_up.qtype != down.qtype || !is_gguf(gate_up.qtype)) {
                    throw std::invalid_argument("Gemma 4 experts must share one GGUF block type");
                }
                p.expert_type       = ops::detail::gguf_type(gate_up.qtype);
                p.gate_up_row_bytes = row_bytes(gate_up);
                p.down_row_bytes    = row_bytes(down);
                std::vector<const void*> table(4 * std::size_t(e));
                for (int x = 0; x < e; ++x) {
                    const auto* gu = static_cast<const std::byte*>(gate_up.qdata) +
                                     std::int64_t(x) * 2 * width * p.gate_up_row_bytes;
                    table[x]         = gu;
                    table[e + x]     = gu + std::int64_t(width) * p.gate_up_row_bytes;
                    table[2 * e + x] = gu;
                    table[3 * e + x] = static_cast<const std::byte*>(down.qdata) +
                                       std::int64_t(x) * hidden * p.down_row_bytes;
                }
                p.tables = DeviceBuffer(table.size() * sizeof(void*));
                p.tables.copy_from_host(table.data(), table.size() * sizeof(void*));
            }
            p.slots       = p.full ? global_slots : ring;
            p.ring        = p.full ? 0 : ring;
            p.head_stride = std::int64_t(p.slots) * p.head_dim;
            p.seq_stride  = p.head_stride * p.kv_heads;
            const std::size_t bytes = std::size_t(p.seq_stride) * sequences * sizeof(__half);
            p.k = DeviceBuffer(bytes);
            p.v = DeviceBuffer(bytes);
            p.k.fill(0);
            p.v.fill(0);
            kv_bytes += 2 * bytes;
            layers.push_back(std::move(p));
        }
        {
            std::size_t staging = 0;
            for (const LayerPlan& p : layers) {
                if (!p.full) {
                    staging = std::max(staging, std::size_t(prefill_ring) * p.kv_heads *
                                                    p.head_dim * sizeof(__half));
                }
            }
            if (staging != 0) {
                staging_k = DeviceBuffer(staging);
                staging_v = DeviceBuffer(staging);
                staging_k.fill(0);
                staging_v.fill(0);
                kv_bytes += 2 * staging;
            }
        }
        const int sms = device.multiprocessor_count();
        // Upper bounds: a step uses as many as its keys fill (a tile each), so the sliding
        // window's 1024 keys take 16 and a long full-attention context spreads over ~3 blocks per SM.
        decode_splits_full    = std::clamp(3 * sms / std::max(1, int(config.num_global_key_value_heads)), 1, 128);
        decode_splits_sliding = 16;
        const std::size_t rows = std::max<std::size_t>(options.prefill_chunk, sequences);
        d_tokens    = DeviceBuffer(rows * 4);
        d_positions = DeviceBuffer(rows * 4);
        d_sequences = DeviceBuffer(rows * 4);
        h_staging   = std::make_unique<PinnedHostBuffer>(3 * rows * 4);
        staged      = std::make_unique<CudaCompletionEvent>(device);
        logits_bf16 = DeviceBuffer(std::size_t(vocab) * kMaxLogitRows * 2);
        workspace_bytes = plan_workspace();
        workspace       = std::make_unique<WorkspaceArena>(workspace_bytes);
    }

    // ---- Workspace -----------------------------------------------------------------------------

    std::size_t plan_workspace() const {
        const std::size_t t     = options.prefill_chunk;
        const std::size_t h     = hidden;
        std::size_t qkv_rows    = 0, q_floats = 0;
        std::size_t gguf_scratch = 0;
        for (const LayerPlan& p : layers) {
            qkv_rows = std::max<std::size_t>(qkv_rows, p.qkv_rows);
            q_floats = std::max<std::size_t>(q_floats, std::size_t(query_heads) * p.head_dim);
            for (const Weight* m : {&p.qkv, &p.output, &p.down}) {
                const ops::detail::GgufShape shape{m->qtype, m->n, m->k};
                gguf_scratch = std::max(gguf_scratch, ops::detail::gguf_project_workspace_bytes(
                                                          {&shape, 1}, 1, int(t)));
            }
            const ops::detail::GgufShape gu{p.gate_up.qtype, p.gate_up.n, p.gate_up.k};
            gguf_scratch = std::max(gguf_scratch,
                                    ops::detail::gguf_swiglu_workspace_bytes(gu, nullptr, 1, 8));
            gguf_scratch = std::max(gguf_scratch, ops::detail::gguf_project_workspace_bytes(
                                                      {&gu, 1}, 1, int(t)));
        }
        const ops::detail::GgufShape hs{head.qtype, std::min<int>(head.n, vocab), head.k};
        gguf_scratch = std::max(gguf_scratch,
                                ops::detail::gguf_project_workspace_bytes({&hs, 1}, 1, kHeadRows));
        std::size_t bytes = 0;
        const auto add    = [&](std::size_t n) { bytes += round_up(n, kAlign); };
        add(t * h * 4);              // residual
        add(t * h * 2);              // normalized input
        add(t * qkv_rows * 2);       // qkv
        add(t * q_floats * 4);       // q
        add(t * q_floats * 4);       // attention out FP32
        add(t * q_floats * 2);       // attention out BF16
        add(t * h * 2);              // o
        add(t * h * 2 * 2);          // mlp / moe inputs
        add(t * config.intermediate_size * 2);
        add(t * config.intermediate_size * 2 * 4); // [gate; up] FP32 plane
        add(t * h * 2);              // dense out
        add(t * std::size_t(global_slots) * 2); // full-attention mask
        add(t * std::size_t(prefill_ring) * 2); // sliding-window mask
        add(gguf::flash_attention_workspace_bytes(512, query_heads, 1, int(t)));
        add(gguf::flash_attention_workspace_bytes(256, query_heads, 1, int(t)));
        add(std::size_t(kHeadRows) * vocab * 4); // head FP32 plane
        add(kHeadRows * h * 2);
        std::size_t moe = 0;
        if (config.moe) {
            const std::size_t pairs = t * config.top_k_experts;
            const std::size_t w     = config.moe_intermediate_size;
            moe += round_up(t * config.top_k_experts * 8, kAlign) * 2; // ids, weights
            moe += round_up(t * (h + config.num_experts) * 4, kAlign);   // router input, logits
            moe += round_up(gguf::moe_routing_bytes(int(config.num_experts), int(pairs)), kAlign);
            moe += round_up(gguf::moe_matrix_activation_bytes(int(h), int(pairs)), kAlign);
            moe += round_up(pairs * 2 * w * 4, kAlign);
            moe += round_up(pairs * w * 2, kAlign);
            moe += round_up(gguf::moe_matrix_activation_bytes(int(w), int(pairs)), kAlign);
            moe += round_up(pairs * h * 4, kAlign);
            moe += round_up(t * h * 8, kAlign);
        }
        const std::size_t decode =
            kernels::decode_attention_workspace_floats(int(options.sequences), query_heads, 512,
                                                       decode_splits_full) * 4 +
            kernels::decode_attention_workspace_floats(int(options.sequences), query_heads, 256,
                                                       decode_splits_sliding) * 4;
        return bytes + moe + round_up(decode, kAlign) + gguf_scratch + (16u << 20);
    }

    static constexpr int kHeadRows = 64; // LM-head columns per FP32 plane pass

    // ---- Shared pieces ---------------------------------------------------------------------------

    void project(const Tensor& x, const Weight& w, Tensor& out) {
        ops::linear(x, w, out, ops::LinearPolicy::AllowA8, *workspace, stream);
    }

    // Up to kMaxVectorColumns rows of width k in the vector kernels' q8_1 form, written by the
    // kernel that produces them instead of a BF16 row and a separate quantization.
    struct Q8Act {
        void* base = nullptr;
        kernels::Q8Rows rows;
    };
    Q8Act alloc_q8(int k, int t) {
        Q8Act q;
        q.base    = workspace->alloc_bytes(gguf::vector_activation_bytes(k, t)).data;
        q.rows.qs = static_cast<std::int8_t*>(q.base);
        q.rows.ds = reinterpret_cast<half2*>(static_cast<std::uint8_t*>(q.base) +
                                             gguf::vector_activation_scales_offset(k, t));
        return q;
    }
    void project_q8(const Q8Act& x, const Weight& w, int t, Tensor& out) {
        gguf::VectorOutput o;
        o.bf16          = static_cast<__nv_bfloat16*>(out.data);
        o.column_stride = w.n;
        gguf::vector_product(ops::detail::gguf_type(w.qtype), w.qdata, row_bytes(w), w.n, w.k,
                             x.base, t, o, stream);
    }

    void stage(std::span<const std::int32_t> a, std::span<const std::int32_t> b,
               std::span<const std::int32_t> c) {
        staged->synchronize();
        auto* host = static_cast<std::int32_t*>(h_staging->data());
        const std::size_t rows = std::max<std::size_t>(options.prefill_chunk, options.sequences);
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

    // The token rows' embeddings, scaled, into the FP32 residual h [T][H].
    void embed(int tokens, float* h) {
        gguf::dequantize_rows(ops::detail::gguf_type(head.qtype), head.qdata, row_bytes(head),
                              hidden, static_cast<const std::int32_t*>(d_tokens.p), tokens, h,
                              hidden, stream);
        kernels::scale(h, std::int64_t(tokens) * hidden, std::sqrt(float(hidden)), stream);
    }

    // The rest of block `p` after its attention output o (BF16 [T][H]): the attention residual,
    // the dense MLP and the routed experts into the residual h, and the normalized input of what
    // follows (`next_norm`: the next block's input norm, or the final norm) into `next`.
    void block_tail(const LayerPlan& p, float* h, int t, const Tensor& o, const float* next_norm,
                    __nv_bfloat16* next, const Q8Act* next_q) {
        auto scope    = workspace->scope();
        const bool q8 = t <= gguf::kMaxVectorColumns;
        Tensor mlp_in = workspace->alloc(DType::BF16, {hidden, t});
        Q8Act mlp_q, moe_q;
        kernels::AttnResidualArgs residual;
        residual.h        = h;
        residual.o        = static_cast<const __nv_bfloat16*>(o.data);
        residual.w_post   = p.post_attention_norm;
        residual.hidden   = hidden;
        residual.tokens   = t;
        residual.eps      = eps;
        residual.ffn_norm = p.ffn_norm;
        if (q8) {
            mlp_q          = alloc_q8(hidden, t);
            residual.mlp_q = mlp_q.rows;
        } else {
            residual.mlp_in = static_cast<__nv_bfloat16*>(mlp_in.data);
        }
        if (p.moe) {
            residual.pre_norm = p.pre_norm;
            if (q8) {
                moe_q          = alloc_q8(hidden, t);
                residual.moe_q = moe_q.rows;
            } else {
                residual.moe_in = static_cast<__nv_bfloat16*>(
                    workspace->alloc_bytes(std::size_t(t) * hidden * 2).data);
            }
            residual.router_scale = p.router_scale;
            residual.router_x     = static_cast<float*>(
                workspace->alloc_bytes(std::size_t(t) * hidden * 4).data);
        }
        kernels::attn_residual(residual, stream);
        const int width = int(config.intermediate_size);
        Tensor middle   = workspace->alloc(DType::BF16, {width, t});
        if (q8) {
            gguf::vector_swiglu(ops::detail::gguf_type(p.gate_up.qtype), p.gate_up.qdata,
                                row_bytes(p.gate_up), width, hidden, mlp_q.base, t,
                                static_cast<__nv_bfloat16*>(middle.data), width, stream,
                                gguf::GateActivation::GeluTanh);
        } else {
            // The integer matrix kernel takes whole 128-row tiles: the [gate; up] parent at once
            // (2 x 2112 rows), then GeGLU over its FP32 rows.
            auto* plane = static_cast<float*>(
                workspace->alloc_bytes(std::size_t(t) * 2 * width * 4).data);
            ops::detail::GgufProduct product;
            product.weight   = &p.gate_up;
            product.f32      = plane;
            product.f32_rows = 2 * width;
            ops::detail::gguf_project(mlp_in, {&product, 1}, *workspace, stream);
            kernels::geglu_rows(plane, t, width, static_cast<__nv_bfloat16*>(middle.data), stream);
        }
        Tensor dense = workspace->alloc(DType::BF16, {hidden, t});
        project(middle, p.down, dense);
        kernels::FfnCombineArgs combine;
        combine.h         = h;
        combine.hidden    = hidden;
        combine.tokens    = t;
        combine.eps       = eps;
        combine.dense     = static_cast<const __nv_bfloat16*>(dense.data);
        combine.w_post    = p.post_ffn_norm;
        combine.scalar    = p.layer_scalar;
        combine.next_norm = next_norm;
        combine.next_out  = next;
        if (next_q != nullptr) { combine.next_q = next_q->rows; }
        if (p.moe) {
            combine.w_dense = p.mlp_post_norm;
            combine.w_moe   = p.post_norm;
            experts(p, residual.moe_in, q8 ? &moe_q : nullptr, residual.router_x, t, combine);
        }
        kernels::ffn_combine(combine, stream);
    }

    // The routed experts of block `p`; their output lands where `combine` reads it.
    void experts(const LayerPlan& p, const __nv_bfloat16* x, const Q8Act* xq, const float* router_x,
                 int t, kernels::FfnCombineArgs& combine) {
        const int e = int(config.num_experts), k = int(config.top_k_experts);
        const int width = int(config.moe_intermediate_size);
        const int pairs = t * k;
        auto* ids      = static_cast<std::int32_t*>(workspace->alloc_bytes(std::size_t(pairs) * 4).data);
        auto* weights  = static_cast<float*>(workspace->alloc_bytes(std::size_t(pairs) * 4).data);
        auto* logits   = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * e * 4).data);
        void* routing_space = workspace->alloc_bytes(gguf::moe_routing_bytes(e, pairs)).data;
        gguf::MoeRouting routing;
        if (t <= gguf::kMaxVectorColumns) {
            auto* base = static_cast<std::int32_t*>(routing_space);
            routing.bounds       = base;
            routing.sorted       = base + e + 1;
            routing.active       = base + e + 1 + pairs;
            routing.active_count = base + e + 1 + pairs + e;
            kernels::route_select(router_x, hidden, t, p.router, p.expert_scale, e, k, logits, ids,
                                  weights, base, base + e + 1, base + e + 1 + pairs,
                                  base + e + 1 + pairs + e, stream);
        } else {
            kernels::route(router_x, hidden, t, p.router, p.expert_scale, e, k, logits, ids,
                           weights, stream);
            routing = gguf::moe_sort_routes(ids, pairs, e, routing_space, stream);
        }
        const auto* tables = static_cast<const void* const*>(p.tables.p);
        const gguf::MoeTable gate{tables, p.gate_up_row_bytes, width, hidden};
        const gguf::MoeTable up{tables + e, p.gate_up_row_bytes, width, hidden};
        const gguf::MoeTable gate_up{tables + 2 * e, p.gate_up_row_bytes, 2 * width, hidden};
        const gguf::MoeTable down{tables + 3 * e, p.down_row_bytes, hidden, width};
        const int max_active = std::min(pairs, e);
        auto* middle = static_cast<__nv_bfloat16*>(
            workspace->alloc_bytes(std::size_t(pairs) * width * 2).data);
        if (t <= gguf::kMaxVectorColumns) {
            const int chunk = t <= 1 ? 1 : t <= 2 ? 2 : t <= 4 ? 4 : 8;
            void* act = xq != nullptr ? xq->base : nullptr;
            if (act == nullptr) {
                act = workspace->alloc_bytes(gguf::vector_activation_bytes(hidden, t)).data;
                gguf::quantize_vector_activation(x, hidden, t, nullptr, act, stream);
            }
            gguf::moe_vector_swiglu(p.expert_type, gate, up, routing, max_active, k, act, t,
                                    middle, chunk, stream, gguf::GateActivation::GeluTanh);
            void* act2 = workspace->alloc_bytes(gguf::vector_activation_bytes(width, pairs)).data;
            gguf::quantize_vector_activation(middle, width, pairs, nullptr, act2, stream);
            auto* fixed = static_cast<unsigned long long*>(
                workspace->alloc_bytes(std::size_t(t) * hidden * 8).data);
            CUDA_CHECK(cudaMemsetAsync(fixed, 0, std::size_t(t) * hidden * 8, stream));
            gguf::MoeOutput out;
            out.weighted = fixed;
            out.weights  = weights;
            gguf::moe_vector_product(p.expert_type, down, routing, max_active, 1, k, act2, pairs,
                                     out, chunk, stream);
            combine.moe_fixed = fixed;
            return;
        }
        void* act = workspace->alloc_bytes(gguf::moe_matrix_activation_bytes(hidden, pairs)).data;
        gguf::quantize_moe_matrix_activation(p.expert_type, x, hidden, routing, pairs, k, act,
                                             stream);
        auto* gu = static_cast<float*>(
            workspace->alloc_bytes(std::size_t(pairs) * 2 * width * 4).data);
        gguf::moe_matrix_product(p.expert_type, gate_up, routing, max_active, pairs, t, true, act,
                                 gu, stream);
        kernels::geglu_rows(gu, pairs, width, middle, stream);
        void* act2 = workspace->alloc_bytes(gguf::moe_matrix_activation_bytes(width, pairs)).data;
        gguf::quantize_moe_matrix_activation(p.expert_type, middle, width, routing, pairs, 1, act2,
                                             stream);
        auto* rows = static_cast<float*>(
            workspace->alloc_bytes(std::size_t(pairs) * hidden * 4).data);
        gguf::moe_matrix_product(p.expert_type, down, routing, max_active, pairs, t, true, act2,
                                 rows, stream);
        combine.moe_rows    = rows;
        combine.moe_weights = weights;
        combine.top_k       = k;
    }

    // The head of `rows` final-normed rows x into logits_bf16 columns [first, first + rows).
    void head_logits(const __nv_bfloat16* rows_x, int rows, int first) {
        for (int at = 0; at < rows; at += kHeadRows) {
            const int n  = std::min(kHeadRows, rows - at);
            auto scope   = workspace->scope();
            Tensor x(const_cast<__nv_bfloat16*>(rows_x) + std::int64_t(at) * hidden, DType::BF16,
                     {hidden, n});
            auto* plane = static_cast<float*>(
                workspace->alloc_bytes(std::size_t(n) * vocab * 4).data);
            ops::detail::GgufProduct product;
            product.weight   = &head;
            product.f32      = plane;
            product.f32_rows = vocab;
            ops::detail::gguf_project(x, {&product, 1}, *workspace, stream);
            kernels::softcap(plane, std::int64_t(n) * vocab, config.final_logit_softcapping,
                             static_cast<__nv_bfloat16*>(logits_bf16.p) +
                                 std::int64_t(first + at) * vocab,
                             stream);
        }
    }

    // The head of the last `rows` of t <= kMaxVectorColumns final-normed q8_1 rows into
    // logits_bf16 columns [0, rows).
    void head_logits_q8(const Q8Act& x, int t, int rows) {
        auto scope  = workspace->scope();
        auto* plane = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * vocab * 4).data);
        gguf::VectorOutput o;
        o.f32           = plane;
        o.column_stride = vocab;
        gguf::vector_product(ops::detail::gguf_type(head.qtype), head.qdata, row_bytes(head), vocab,
                             head.k, x.base, t, o, stream);
        kernels::softcap(plane + std::int64_t(t - rows) * vocab, std::int64_t(rows) * vocab,
                         config.final_logit_softcapping,
                         static_cast<__nv_bfloat16*>(logits_bf16.p), stream);
    }

    // ---- Prompt chunks -----------------------------------------------------------------------------

    void forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                 std::uint32_t logit_rows) {
        const int t = int(tokens.size());
        if (sequence >= options.sequences || t <= 0 || std::uint32_t(t) > options.prefill_chunk ||
            logit_rows == 0 || logit_rows > std::min<std::uint32_t>(t, kMaxLogitRows)) {
            throw std::invalid_argument("gemma4 forward: invalid call");
        }
        const std::uint32_t first = positions[sequence];
        if (first + std::uint32_t(t) > options.max_context) {
            throw std::invalid_argument("gemma4 forward: context capacity exceeded");
        }
        std::vector<std::int32_t> pos(t);
        for (int i = 0; i < t; ++i) { pos[i] = std::int32_t(first) + i; }
        stage(tokens, pos, {});
        workspace->reset();
        auto scope = workspace->scope();
        auto* h    = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * hidden * 4).data);
        embed(t, h);
        // Masks: the sliding ring and the full-attention prefix.
        const int full_slots = int(round_up(first + t, 256));
        auto* mask_ring = static_cast<__half*>(
            workspace->alloc_bytes(std::size_t(t) * prefill_ring * 2).data);
        auto* mask_full =
            static_cast<__half*>(workspace->alloc_bytes(std::size_t(t) * full_slots * 2).data);
        kernels::build_mask(mask_ring, t, prefill_ring, int(first), prefill_ring,
                            int(config.sliding_window), stream);
        const int window = int(config.sliding_window);
        kernels::build_mask(mask_full, t, full_slots, int(first), 0, 0, stream);
        // Each block's normalized input rows; after the last block, the final-normed rows (q8_1
        // for a vector-width chunk).
        const bool q8 = t <= gguf::kMaxVectorColumns;
        Tensor x      = workspace->alloc(DType::BF16, {hidden, t});
        Q8Act xq;
        if (q8) { xq = alloc_q8(hidden, t); }
        kernels::rms_norm(h, layers.front().input_norm, hidden, t, eps,
                          q8 ? nullptr : static_cast<__nv_bfloat16*>(x.data), stream, xq.rows);
        for (std::size_t li = 0; li < layers.size(); ++li) {
            const LayerPlan& p = layers[li];
            auto layer_scope   = workspace->scope();
            Tensor qkv = workspace->alloc(DType::BF16, {p.qkv_rows, t});
            if (q8) {
                project_q8(xq, p.qkv, t, qkv);
            } else {
                project(x, p.qkv, qkv);
            }
            const int q_floats = query_heads * p.head_dim;
            auto* q = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * q_floats * 4).data);
            kernels::QkvArgs qa;
            qa.qkv         = static_cast<const __nv_bfloat16*>(qkv.data);
            qa.tokens      = t;
            qa.query_heads = query_heads;
            qa.kv_heads    = p.kv_heads;
            qa.head_dim    = p.head_dim;
            qa.has_v       = !p.full;
            qa.q_norm      = p.q_norm;
            qa.k_norm      = p.k_norm;
            qa.eps         = eps;
            qa.positions   = static_cast<const std::int32_t*>(d_positions.p);
            qa.inv_freq    = static_cast<const float*>(p.full ? inv_freq_full.p : inv_freq_sliding.p);
            qa.q_out       = q;
            // A sliding block attends over the staging ring: the window before the chunk first
            // comes over from the sequence's ring, the chunk's keys land there, and the window that
            // ends the chunk goes back.
            const std::int64_t seq_offset   = std::int64_t(sequence) * p.seq_stride;
            const std::int64_t stage_stride = std::int64_t(prefill_ring) * p.head_dim;
            kernels::RingCopyArgs keep;
            keep.kv_heads = p.kv_heads;
            keep.head_dim = p.head_dim;
            if (!p.full) {
                keep.src_k           = static_cast<const __half*>(p.k.p) + seq_offset;
                keep.src_v           = static_cast<const __half*>(p.v.p) + seq_offset;
                keep.dst_k           = static_cast<__half*>(staging_k.p);
                keep.dst_v           = static_cast<__half*>(staging_v.p);
                keep.src_head_stride = p.head_stride;
                keep.dst_head_stride = stage_stride;
                keep.src_ring        = p.ring;
                keep.dst_ring        = prefill_ring;
                keep.begin           = std::max(0, int(first) - window);
                keep.end             = int(first);
                kernels::ring_copy(keep, stream);
            }
            qa.k_cache     = static_cast<__half*>(p.full ? p.k.p : staging_k.p);
            qa.v_cache     = static_cast<__half*>(p.full ? p.v.p : staging_v.p);
            qa.seq_stride  = p.full ? p.seq_stride : 0;
            qa.head_stride = p.full ? p.head_stride : stage_stride;
            qa.sequence    = p.full ? int(sequence) : 0;
            qa.ring        = p.full ? 0 : prefill_ring;
            kernels::qkv_post(qa, stream);
            auto* attn = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * q_floats * 4).data);
            const std::size_t fa_bytes =
                gguf::flash_attention_workspace_bytes(p.head_dim, query_heads, p.kv_heads, t);
            void* fa_space = workspace->alloc_bytes(fa_bytes).data;
            gguf::FlashAttentionArgs fa;
            fa.head_dim        = p.head_dim;
            fa.query_heads     = query_heads;
            fa.kv_heads        = p.kv_heads;
            fa.tokens          = t;
            fa.q               = q;
            fa.q_token_stride  = q_floats;
            fa.k = p.full ? static_cast<const __half*>(p.k.p) + seq_offset
                          : static_cast<const __half*>(staging_k.p);
            fa.v = p.full ? static_cast<const __half*>(p.v.p) + seq_offset
                          : static_cast<const __half*>(staging_v.p);
            fa.row_bytes       = std::int64_t(p.head_dim) * 2;
            fa.head_bytes      = (p.full ? p.head_stride : stage_stride) * 2;
            fa.slots           = p.full ? full_slots : prefill_ring;
            fa.mask            = p.full ? mask_full : mask_ring;
            fa.mask_row_halves = fa.slots;
            fa.out             = attn;
            fa.scale           = 1.0f;
            gguf::flash_attention(fa, fa_space, fa_bytes, stream);
            if (!p.full) {
                kernels::RingCopyArgs back = keep;
                std::swap(back.src_head_stride, back.dst_head_stride);
                std::swap(back.src_ring, back.dst_ring);
                back.src_k = static_cast<const __half*>(staging_k.p);
                back.src_v = static_cast<const __half*>(staging_v.p);
                back.dst_k = static_cast<__half*>(p.k.p) + seq_offset;
                back.dst_v = static_cast<__half*>(p.v.p) + seq_offset;
                back.begin = std::max(0, int(first) + t - window);
                back.end   = int(first) + t;
                kernels::ring_copy(back, stream);
            }
            Tensor attn_bf16 = workspace->alloc(DType::BF16, {q_floats, t});
            kernels::f32_to_bf16(attn, static_cast<__nv_bfloat16*>(attn_bf16.data),
                                 std::int64_t(t) * q_floats, stream);
            Tensor o = workspace->alloc(DType::BF16, {hidden, t});
            project(attn_bf16, p.output, o);
            block_tail(p, h, t, o, li + 1 < layers.size() ? layers[li + 1].input_norm : final_norm,
                       q8 ? nullptr : static_cast<__nv_bfloat16*>(x.data), q8 ? &xq : nullptr);
        }
        if (q8) {
            head_logits_q8(xq, t, int(logit_rows));
        } else {
            head_logits(static_cast<const __nv_bfloat16*>(x.data) +
                            std::int64_t(t - int(logit_rows)) * hidden,
                        int(logit_rows), 0);
        }
        positions[sequence] = first + std::uint32_t(t);
    }

    // ---- Decode ---------------------------------------------------------------------------------------

    // One token per row; tokens, positions and sequences already staged on the device.
    void decode_body(int b) {
        workspace->reset();
        auto scope = workspace->scope();
        auto* h    = static_cast<float*>(workspace->alloc_bytes(std::size_t(b) * hidden * 4).data);
        embed(b, h);
        // Each block's normalized input rows in q8_1 form; after the last block, the final-normed
        // rows.
        const Q8Act xq = alloc_q8(hidden, b);
        kernels::rms_norm(h, layers.front().input_norm, hidden, b, eps, nullptr, stream, xq.rows);
        for (std::size_t li = 0; li < layers.size(); ++li) {
            const LayerPlan& p = layers[li];
            auto layer_scope   = workspace->scope();
            Tensor qkv = workspace->alloc(DType::BF16, {p.qkv_rows, b});
            project_q8(xq, p.qkv, b, qkv);
            const int q_floats = query_heads * p.head_dim;
            auto* q = static_cast<float*>(workspace->alloc_bytes(std::size_t(b) * q_floats * 4).data);
            kernels::QkvArgs qa;
            qa.qkv         = static_cast<const __nv_bfloat16*>(qkv.data);
            qa.tokens      = b;
            qa.query_heads = query_heads;
            qa.kv_heads    = p.kv_heads;
            qa.head_dim    = p.head_dim;
            qa.has_v       = !p.full;
            qa.q_norm      = p.q_norm;
            qa.k_norm      = p.k_norm;
            qa.eps         = eps;
            qa.positions   = static_cast<const std::int32_t*>(d_positions.p);
            qa.inv_freq    = static_cast<const float*>(p.full ? inv_freq_full.p : inv_freq_sliding.p);
            qa.q_out       = q;
            qa.k_cache     = static_cast<__half*>(p.k.p);
            qa.v_cache     = static_cast<__half*>(p.v.p);
            qa.seq_stride  = p.seq_stride;
            qa.head_stride = p.head_stride;
            qa.sequences   = static_cast<const std::int32_t*>(d_sequences.p);
            qa.ring        = p.ring;
            kernels::qkv_post(qa, stream);
            const int splits = p.full ? decode_splits_full : decode_splits_sliding;
            auto* partial    = static_cast<float*>(
                workspace->alloc_bytes(kernels::decode_attention_workspace_floats(
                                           b, query_heads, p.head_dim, splits) * 4).data);
            const Q8Act attn = alloc_q8(q_floats, b);
            kernels::DecodeAttentionArgs da;
            da.q           = q;
            da.k_cache     = static_cast<const __half*>(p.k.p);
            da.v_cache     = static_cast<const __half*>(p.v.p);
            da.seq_stride  = p.seq_stride;
            da.head_stride = p.head_stride;
            da.sequences   = static_cast<const std::int32_t*>(d_sequences.p);
            da.positions   = static_cast<const std::int32_t*>(d_positions.p);
            da.batch       = b;
            da.query_heads = query_heads;
            da.kv_heads    = p.kv_heads;
            da.head_dim    = p.head_dim;
            da.ring        = p.ring;
            da.window      = int(config.sliding_window);
            da.splits      = splits;
            da.scale       = 1.0f;
            da.partial     = partial;
            da.out_q       = attn.rows;
            kernels::decode_attention(da, stream);
            Tensor o = workspace->alloc(DType::BF16, {hidden, b});
            project_q8(attn, p.output, b, o);
            block_tail(p, h, b, o, li + 1 < layers.size() ? layers[li + 1].input_norm : final_norm,
                       nullptr, &xq);
        }
        head_logits_q8(xq, b, b);
    }

    void decode(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens) {
        const int b = int(sequences.size());
        if (b <= 0 || std::uint32_t(b) > options.sequences || tokens.size() != sequences.size() ||
            b > gguf::kMaxVectorColumns) {
            throw std::invalid_argument("gemma4 decode: invalid batch");
        }
        std::vector<std::int32_t> pos(b), seq(b);
        for (int i = 0; i < b; ++i) {
            if (sequences[i] >= options.sequences) {
                throw std::invalid_argument("gemma4 decode: invalid sequence");
            }
            if (positions[sequences[i]] + 1 > options.max_context) {
                throw std::invalid_argument("gemma4 decode: context capacity exceeded");
            }
            pos[i] = std::int32_t(positions[sequences[i]]);
            seq[i] = std::int32_t(sequences[i]);
        }
        stage(tokens, pos, seq);
        if (!options.cuda_graphs) {
            decode_body(b);
        } else {
            GraphEntry& entry = graphs[std::uint32_t(b)];
            if (!entry.warmed) {
                decode_body(b);
                entry.warmed = true;
            } else {
                if (!entry.executable.ready()) {
                    entry.definition.capture(stream, [&] { decode_body(b); });
                    entry.executable.instantiate(entry.definition);
                }
                entry.executable.launch(stream);
            }
        }
        for (int i = 0; i < b; ++i) { positions[sequences[i]] += 1; }
    }

    void warm_up() {
        // The prompt route once (kernels load at first launch), then each decode batch's graph.
        const std::vector<std::int32_t> prompt(std::min<std::uint32_t>(options.prefill_chunk, 16), 2);
        forward(0, prompt, 1);
        positions[0] = 0;
        std::vector<std::uint32_t> sequences;
        std::vector<std::int32_t> tokens;
        for (std::uint32_t b = 1; b <= std::min<std::uint32_t>(options.sequences, 8); ++b) {
            sequences.push_back(b - 1);
            tokens.push_back(2);
            for (int pass = 0; pass < 2; ++pass) { // eagerly, then captured
                decode(sequences, tokens);
                for (const std::uint32_t s : sequences) { positions[s] = 0; }
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void snapshot(std::uint32_t sequence, SequenceSnapshot& out) {
        std::size_t index = 0;
        for (const LayerPlan& p : layers) {
            if (p.full) { continue; }
            const std::size_t bytes = std::size_t(p.seq_stride) * sizeof(__half);
            if (out.rings.size() < index + 2) {
                out.rings.emplace_back(bytes);
                out.rings.emplace_back(bytes);
            }
            CUDA_CHECK(cudaMemcpyAsync(out.rings[index].p,
                                       static_cast<const __half*>(p.k.p) + sequence * p.seq_stride,
                                       bytes, cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(out.rings[index + 1].p,
                                       static_cast<const __half*>(p.v.p) + sequence * p.seq_stride,
                                       bytes, cudaMemcpyDeviceToDevice, stream));
            index += 2;
        }
        out.position = positions[sequence];
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void restore(std::uint32_t sequence, const SequenceSnapshot& from) {
        std::size_t index = 0;
        for (const LayerPlan& p : layers) {
            if (p.full) { continue; }
            const std::size_t bytes = std::size_t(p.seq_stride) * sizeof(__half);
            CUDA_CHECK(cudaMemcpyAsync(static_cast<__half*>(p.k.p) + sequence * p.seq_stride,
                                       from.rings.at(index).p, bytes, cudaMemcpyDeviceToDevice,
                                       stream));
            CUDA_CHECK(cudaMemcpyAsync(static_cast<__half*>(p.v.p) + sequence * p.seq_stride,
                                       from.rings.at(index + 1).p, bytes,
                                       cudaMemcpyDeviceToDevice, stream));
            index += 2;
        }
        positions[sequence] = from.position;
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
};

Executor::Executor(const Model& model, DeviceContext& device, ExecutorOptions options)
    : impl_(std::make_unique<Impl>(model, device, options)) {}

Executor::~Executor() = default;

const ExecutorOptions& Executor::options() const noexcept { return impl_->options; }

ExecutorMemory Executor::memory() const noexcept {
    return ExecutorMemory{.kv_bytes = impl_->kv_bytes, .workspace_bytes = impl_->workspace_bytes};
}

void Executor::reset(std::uint32_t sequence) { impl_->positions.at(sequence) = 0; }

std::uint32_t Executor::position(std::uint32_t sequence) const {
    return impl_->positions.at(sequence);
}

void Executor::warm_up() { impl_->warm_up(); }

void Executor::forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                       std::uint32_t logit_rows) {
    impl_->forward(sequence, tokens, logit_rows);
}

void Executor::decode(std::span<const std::uint32_t> sequences,
                      std::span<const std::int32_t> tokens) {
    impl_->decode(sequences, tokens);
}

Tensor Executor::logits(std::uint32_t rows) const {
    if (rows == 0 || rows > kMaxLogitRows) {
        throw std::invalid_argument("gemma4 logits: invalid row count");
    }
    return Tensor(impl_->logits_bf16.p, DType::BF16,
                  {impl_->vocab, static_cast<std::int32_t>(rows)});
}

cudaStream_t Executor::head_stream() const noexcept { return impl_->stream; }

void Executor::snapshot(std::uint32_t sequence, SequenceSnapshot& out) {
    impl_->snapshot(sequence, out);
}

void Executor::restore(std::uint32_t sequence, const SequenceSnapshot& from) {
    impl_->restore(sequence, from);
}

} // namespace ninfer::models::gemma4
