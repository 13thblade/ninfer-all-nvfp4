#include "models/deepseek_v41/model.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen3_5/load/bindings.h"

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::deepseek_v41 {
namespace {

using qwen3_5::loading::Bindings;

// Zero bytes after a matrix: the integer matrix kernel reads whole 256-value steps of a row
// (gguf_project, moe_matrix_product), so a stored row whose k is not a whole number of them reads
// past its end.
constexpr std::uint64_t kRowTail = 512;

FrontendResources bind_resources(artifact::Binder& binder, const TextConfig& config,
                                 std::vector<std::int32_t>& token_map) {
    const auto resource = [&](std::string_view role) {
        const auto bytes = binder.host_object(binder.resource("text", role));
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    FrontendResources out;
    out.tokenizer_json         = resource("tokenizer.json");
    out.tokenizer_config_json  = resource("tokenizer_config.json");
    out.chat_template_jinja    = resource("chat_template.jinja");
    out.generation_config_json = resource("generation_config.json");
    out.tokenizer =
        std::make_shared<const qwen3_5::frontend::Tokenizer>(qwen3_5::frontend::TokenizerResources{
            out.tokenizer_json, out.tokenizer_config_json, out.generation_config_json});
    const auto count = out.tokenizer->vocab_size();
    if (!count || count > config.vocab_size || !out.tokenizer->has_exact_token_domain(count)) {
        throw artifact::ArtifactError(
            "tokenizer must expose a contiguous public domain within embedding rows");
    }
    out.public_token_count = static_cast<std::uint32_t>(count);
    const auto map = resource("engram_token_map");
    if (map.size() != std::size_t(config.vocab_size) * 4) {
        throw artifact::ArtifactError("the Engram token map must hold a compressed id per token");
    }
    token_map.resize(config.vocab_size);
    std::memcpy(token_map.data(), map.data(), map.size());
    for (const auto id : token_map) {
        if (id < 0) { throw artifact::ArtifactError("the Engram token map holds a negative id"); }
    }
    return out;
}

void tail(Bindings& b, WeightId id) {
    for (const auto& part : b.at(id).reference.binding.parts) {
        if (is_gguf(b.binder.reader().geometry(part.object).format)) {
            b.binder.device_tail(part.object, kRowTail);
        }
    }
}

// Resolve stored GGUF rows directly. Disk mode validates without materializing the bank.
// Host mode borrows the materializer's pinned parent. Neither path repacks model weights.
std::vector<ExpertBlob> expert_blobs(const artifact::Reader& reader,
                                    const artifact::ParameterReference& reference,
                                    const artifact::MaterializedArtifact& backing,
                                    std::uint64_t experts, std::uint64_t rows, std::uint64_t k,
                                    QType& format, std::uint64_t& row_bytes) {
    std::vector<ExpertBlob> result;
    const auto& parts = reference.binding.parts;
    if (parts.size() != 1) {
        throw artifact::ArtifactError(reference.name + ": expert bank must be one stored region");
    }
    const auto& part = parts.front();
    const auto& geometry = reader.geometry(part.object);
    if (!is_gguf(geometry.format) || geometry.shape.size() != 2 || geometry.shape[1] != k ||
        part.begin % k || part.end - part.begin != experts * rows * k) {
        throw artifact::ArtifactError(reference.name + ": experts must be stored GGUF block rows");
    }
    format = geometry.format;
    const auto block = gguf_block_shape(format);
    if (k % block.elements) {
        throw artifact::ArtifactError(reference.name + ": partial stored quantization block");
    }
    row_bytes = k / block.elements * block.bytes;
    reader.validate_object(part.object);
    const auto start = part.begin / k * row_bytes;
    for (std::uint64_t e = 0; e < experts; ++e) {
        ExpertBlob blob;
        blob.bytes = rows * row_bytes;
        blob.zero_tail = kRowTail;
        const auto offset = start + e * blob.bytes;
        if (reference.residency == artifact::Residency::Pinned) {
            const auto* parent = backing.pinned_parent(part.object).data;
            blob.pinned = parent + offset;
            blob.pinned_region = reinterpret_cast<std::uintptr_t>(parent);
        } else {
            const auto logical = reader.directory().tensor(part.object).offset + offset;
            for (const auto& segment : reader.segments(logical, blob.bytes)) {
                blob.extents.push_back({segment.file_index, segment.file_offset, segment.bytes});
            }
        }
        result.push_back(std::move(blob));
    }
    return result;
}

} // namespace

Model::Model(TextConfig config, TextWeights weights, std::vector<BoundWeight> bound,
             FrontendResources resources, InstanceInfo info, artifact::MaterializedArtifact backing)
    : backing_(std::move(backing)), config_(std::move(config)), weights_(std::move(weights)),
      bound_(std::move(bound)), resources_(std::move(resources)), info_(std::move(info)) {}

Model::~Model() = default;

ops::WeightInput Model::input(WeightId id) const {
    const auto& parameter = weight(id);
    if (parameter.uses.size() != 1) {
        throw std::invalid_argument(parameter.name + ": a weight input needs exactly one use");
    }
    const auto& use = parameter.uses.front();
    return ops::WeightInput{parameter.view, use.policy, use.activation_input_divisor};
}

bool is_deepseek_v41(const artifact::Reader& reader) {
    const auto& components = reader.directory().components;
    const auto text        = components.find("text");
    if (text == components.end()) { return false; }
    const auto& config = text->second.config;
    return config.is_object() && config.contains("architectures") &&
           config.at("architectures").is_array() && config.at("architectures").size() == 1 &&
           config.at("architectures")[0] == "DeepseekV41ForCausalLM";
}

std::unique_ptr<Model> load_model(const artifact::Reader& reader, const LoadOptions& options,
                                  DeviceContext& device, const StartupObserver* observer) {
    if (!is_deepseek_v41(reader)) {
        throw artifact::ArtifactError("the artifact's text component is not DeepseekV41ForCausalLM");
    }
    TextConfig config = parse_text_config(reader.directory().component("text").config);
    artifact::Binder binder(reader);
    std::vector<std::int32_t> token_map;
    FrontendResources resources = bind_resources(binder, config, token_map);
    Bindings b(binder);
    const bool offload = options.experts != ExpertResidency::Device;
    std::vector<std::array<artifact::ParameterReference, 2>> expert_references;
    const std::uint64_t h  = config.hidden_size;
    const std::uint64_t d  = config.head_dim;
    const std::uint64_t hc = config.hc_mult;
    TextWeights weights;
    weights.token_embedding = b.parameter("text/token_embedding", {config.vocab_size, h});
    weights.output_head     = b.parameter("text/output_head", {config.vocab_size, h},
                                          {"text/final_hidden"});
    weights.final_norm      = b.direct("text/final_norm", {h}, QType::FP32);
    tail(b, weights.output_head);
    const std::uint64_t q_rank = config.q_lora_rank;
    const std::uint64_t heads  = config.num_attention_heads;
    const std::uint64_t groups = config.o_groups, o_rank = config.o_lora_rank;
    const std::uint64_t ih = config.index_heads, idim = config.index_head_dim;
    const std::uint64_t e = config.num_experts, w = config.moe_intermediate_size;
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        const std::string p = "text/layers/" + std::to_string(i) + "/";
        const std::string a = p + "attention/";
        LayerWeights layer;
        for (auto [site, prefix] : {std::pair{&layer.hc_attention, p + "hc_attention/"},
                                    std::pair{&layer.hc_ffn, p + "hc_ffn/"}}) {
            site->fn    = b.direct(prefix + "fn", {config.hc_mixes(), hc * h}, QType::FP32);
            site->base  = b.direct(prefix + "base", {config.hc_mixes()}, QType::FP32);
            site->scale = b.direct(prefix + "scale", {3}, QType::FP32);
        }
        layer.attention_norm = b.direct(p + "attention_norm", {h}, QType::FP32);
        layer.ffn_norm       = b.direct(p + "ffn_norm", {h}, QType::FP32);
        auto& at             = layer.attention;
        at.query_a      = b.parameter(a + "query_a", {q_rank, h}, {p + "attention_input"});
        at.kv           = b.parameter(a + "kv", {d, h}, {p + "attention_input"});
        at.query_a_norm = b.direct(a + "query_a_norm", {q_rank}, QType::FP32);
        at.kv_norm      = b.direct(a + "kv_norm", {d}, QType::FP32);
        at.query_b      = b.parameter(a + "query_b", {heads * d, q_rank}, {a + "query_latent"});
        at.output_a     = b.parameter(a + "output_a", {groups * o_rank, heads * d / groups},
                                      {a + "grouped_output"});
        at.output_b     = b.parameter(a + "output_b", {h, groups * o_rank}, {a + "output_latent"});
        at.sinks        = b.direct(a + "sinks", {heads}, QType::FP32);
        for (const WeightId id : {at.query_a, at.kv, at.query_b, at.output_a, at.output_b}) {
            tail(b, id);
        }
        if (config.kv_source[i]) {
            at.compressor_kv = b.direct(a + "compressor/kv", {d, h}, QType::FP32);
            if (config.compress_ratios[i] > 1) {
                at.compressor_gate = b.direct(a + "compressor/gate", {d, h}, QType::FP32);
            }
            at.compressor_norm = b.direct(a + "compressor/norm", {d}, QType::FP32);
            at.index_key       = b.direct(a + "indexer/key", {idim, d}, QType::FP32);
            at.index_key_norm  = b.direct(a + "indexer/key_norm", {idim}, QType::FP32);
        }
        if (config.index_source[i]) {
            at.index_query   = b.direct(a + "indexer/query", {ih * idim, q_rank}, QType::FP32);
            at.index_weights = b.direct(a + "indexer/weights", {ih, h}, QType::FP32);
        }
        const std::string m = p + "moe/";
        MoeWeights& moe     = layer.moe;
        moe.router          = b.direct(m + "router", {e, h}, QType::FP32);
        moe.router_bias     = b.direct(m + "router_bias", {e}, QType::FP32);
        moe.shared_gate     = b.parameter(m + "shared_gate", {w, h}, {m + "input"});
        moe.shared_up       = b.parameter(m + "shared_up", {w, h}, {m + "input"});
        moe.shared_down     = b.parameter(m + "shared_down", {h, w}, {m + "shared_product"});
        tail(b, moe.shared_down);
        tail(b, moe.shared_gate);
        if (offload) {
            const auto residency = options.experts == ExpertResidency::Host
                                       ? artifact::Residency::Pinned : artifact::Residency::Values;
            expert_references.push_back({
                binder.parameter(m + "gate_up", {e * 2 * w, h}, residency),
                binder.parameter(m + "down", {e * h, w}, residency)});
            // Uses still belong to the family contract, even without a resident operand.
            for (const auto& use : {std::pair{m + "gate_up", m + "input"},
                                    std::pair{m + "down", m + "product"}}) {
                const auto& contract = binder.use(use.first, use.second);
                if (!contract.activation_policy || !contract.auxiliaries.empty()) {
                    throw artifact::ArtifactError(use.first +
                        ": streamed GGUF experts require an activation policy and no auxiliaries");
                }
            }
        } else {
            moe.gate_up = b.parameter(m + "gate_up", {e * 2 * w, h}, {m + "input"});
            moe.down    = b.parameter(m + "down", {e * h, w}, {m + "product"});
            tail(b, *moe.down);
            tail(b, *moe.gate_up);
        }
        if (config.engram.index(i) >= 0) {
            const std::string n = p + "engram/";
            const std::uint64_t width = std::uint64_t(config.engram.columns()) *
                                        config.engram.head_dim;
            EngramWeights engram;
            engram.projection = b.parameter(n + "projection", {(hc + 1) * h, width}, {n + "rows"});
            engram.gate       = b.direct(n + "gate", {hc, h}, QType::FP32);
            tail(b, engram.projection);
            layer.engram = engram;
        }
        weights.layers.push_back(std::move(layer));
    }
    InstanceInfo info;
    info.name            = reader.directory().metadata.value("name", std::string("DeepSeek-V4.1-Flash"));
    info.metadata_json   = reader.directory().metadata.dump();
    info.provenance_json = reader.directory().provenance.dump();
    info.artifact_id     = reader.artifact_id();
    auto pending         = std::move(b.weights);
    auto materialization = std::move(binder).finish();
    auto backing = artifact::materialize(reader, std::move(materialization), device, observer);
    auto bound   = qwen3_5::loading::resolve_weights(std::move(pending), backing);
    std::vector<ExpertBanks> expert_banks;
    for (const auto& reference : expert_references) {
        ExpertBanks bank;
        auto gu = expert_blobs(reader, reference[0], backing, e, 2 * w, h, bank.gate_up_type,
                               bank.gate_up_row_bytes);
        auto down = expert_blobs(reader, reference[1], backing, e, h, w, bank.down_type,
                                 bank.down_row_bytes);
        for (std::size_t x = 0; x < gu.size(); ++x) {
            bank.delivery.experts.push_back({std::move(gu[x]), std::move(down[x])});
        }
        // Gate rows, up rows, the [gate; up] parent, down rows: the executor's expert tables.
        bank.delivery.aliases = {{0, 0}, {0, w * bank.gate_up_row_bytes}, {0, 0}, {1, 0}};
        expert_banks.push_back(std::move(bank));
    }
    auto model = std::unique_ptr<Model>(new Model(std::move(config), std::move(weights),
                                                  std::move(bound), std::move(resources),
                                                  std::move(info), std::move(backing)));
    model->engram_token_map_ = std::move(token_map);
    model->expert_residency_ = options.experts;
    model->expert_banks_     = std::move(expert_banks);
    if (options.experts == ExpertResidency::Disk) {
        for (const auto& file : reader.directory().files) {
            model->expert_files_.push_back(file.path ? options.artifact.parent_path() / *file.path
                                                     : options.artifact);
        }
    }
    return model;
}

} // namespace ninfer::models::deepseek_v41
