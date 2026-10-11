#include "models/gemma4/model.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen3_5/load/bindings.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::gemma4 {
namespace {

using qwen3_5::loading::Bindings;

// Zero bytes after a matrix whose rows are not whole 256-value steps: the integer matrix kernel
// reads past the last row's end (gguf_project, moe_matrix_product).
constexpr std::uint64_t kRowTail = 512;

FrontendResources bind_resources(artifact::Binder& binder, const TextConfig& config,
                                 bool vision) {
    const auto resource = [&](std::string_view component, std::string_view role) {
        const auto bytes = binder.host_object(binder.resource(component, role));
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    FrontendResources out;
    out.tokenizer_json         = resource("text", "tokenizer.json");
    out.tokenizer_config_json  = resource("text", "tokenizer_config.json");
    out.chat_template_jinja    = resource("text", "chat_template.jinja");
    out.generation_config_json = resource("text", "generation_config.json");
    if (vision) { out.processor_config_json = resource("vision", "processor_config.json"); }
    out.tokenizer =
        std::make_shared<const qwen3_5::frontend::Tokenizer>(qwen3_5::frontend::TokenizerResources{
            out.tokenizer_json, out.tokenizer_config_json, out.generation_config_json});
    const auto count = out.tokenizer->vocab_size();
    if (!count || count > config.vocab_size || !out.tokenizer->has_exact_token_domain(count)) {
        throw artifact::ArtifactError(
            "tokenizer must expose a contiguous public domain within embedding rows");
    }
    out.public_token_count = static_cast<std::uint32_t>(count);
    return out;
}

void tail(Bindings& b, WeightId id) {
    for (const auto& part : b.at(id).reference.binding.parts) {
        if (is_gguf(b.binder.reader().geometry(part.object).format)) {
            b.binder.device_tail(part.object, kRowTail);
        }
    }
}

} // namespace

Model::Model(TextConfig config, TextWeights weights, std::optional<AssistantConfig> assistant_config,
             std::optional<AssistantWeights> assistant, VisionParts vision,
             std::vector<BoundWeight> bound, FrontendResources resources, InstanceInfo info,
             artifact::MaterializedArtifact backing)
    : backing_(std::move(backing)), config_(std::move(config)), weights_(std::move(weights)),
      assistant_config_(std::move(assistant_config)), assistant_(std::move(assistant)),
      vision_config_(std::move(vision.config)), vision_(std::move(vision.weights)),
      vision_placement_(vision.placement), bound_(std::move(bound)),
      resources_(std::move(resources)), info_(std::move(info)) {}

Model::~Model() = default;

ops::WeightInput Model::input(WeightId id) const {
    const auto& parameter = weight(id);
    if (parameter.uses.size() != 1) {
        throw std::invalid_argument(parameter.name + ": a weight input needs exactly one use");
    }
    const auto& use = parameter.uses.front();
    return ops::WeightInput{parameter.view, use.policy, use.activation_input_divisor};
}

bool is_gemma4(const artifact::Reader& reader) {
    const auto& components = reader.directory().components;
    const auto text        = components.find("text");
    if (text == components.end()) { return false; }
    const auto& config = text->second.config;
    return config.is_object() && config.contains("architectures") &&
           config.at("architectures").is_array() && config.at("architectures").size() == 1 &&
           config.at("architectures")[0] == "Gemma4ForCausalLM";
}

bool has_assistant(const artifact::Reader& reader) {
    return is_gemma4(reader) && reader.directory().components.contains("mtp");
}

bool has_vision(const artifact::Reader& reader) {
    return is_gemma4(reader) && reader.directory().components.contains("vision");
}

std::unique_ptr<Model> load_model(const artifact::Reader& reader, const LoadOptions& options,
                                  DeviceContext& device, const StartupObserver* observer) {
    if (!is_gemma4(reader)) {
        throw artifact::ArtifactError("the artifact's text component is not Gemma4ForCausalLM");
    }
    TextConfig config = parse_text_config(reader.directory().component("text").config);
    const bool vision = options.vision != VisionPlacement::None;
    if (vision && !has_vision(reader)) {
        throw artifact::ArtifactError(
            "the artifact holds no Gemma 4 vision tower (convert with --components text,vision)");
    }
    artifact::Binder binder(reader);
    FrontendResources resources = bind_resources(binder, config, vision);
    Bindings b(binder);
    const std::uint64_t h = config.hidden_size;
    TextWeights weights;
    weights.output_head = b.parameter("text/output_head", {config.vocab_size, h},
                                      {"text/final_hidden"});
    weights.final_norm  = b.direct("text/final_norm", {h}, QType::FP32);
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        const std::string p = "text/layers/" + std::to_string(i) + "/";
        const std::string a = p + "attention/";
        const bool full     = config.full_attention[i];
        const std::uint64_t d  = config.layer_head_dim(i);
        const std::uint64_t kv = config.layer_kv_heads(i);
        const std::uint64_t q  = std::uint64_t(config.num_attention_heads) * d;
        LayerWeights layer;
        layer.input_norm          = b.direct(p + "input_norm", {h}, QType::FP32);
        layer.post_attention_norm = b.direct(p + "post_attention_norm", {h}, QType::FP32);
        layer.ffn_norm            = b.direct(p + "ffn_norm", {h}, QType::FP32);
        layer.post_ffn_norm       = b.direct(p + "post_ffn_norm", {h}, QType::FP32);
        layer.layer_scalar        = b.direct(p + "layer_scalar", {1}, QType::FP32);
        layer.attention.query     = b.parameter(a + "query", {q, h}, {p + "mixer_input"});
        layer.attention.key       = b.parameter(a + "key", {kv * d, h}, {p + "mixer_input"});
        if (!full) {
            layer.attention.value = b.parameter(a + "value", {kv * d, h}, {p + "mixer_input"});
        }
        layer.attention.output = b.parameter(a + "output", {h, q}, {p + "attention_output"});
        layer.attention.query_norm = b.direct(a + "query_norm", {d}, QType::FP32);
        layer.attention.key_norm   = b.direct(a + "key_norm", {d}, QType::FP32);
        const std::uint64_t width  = config.intermediate_size;
        layer.mlp_gate = b.parameter(p + "mlp/gate", {width, h}, {p + "mlp_input"});
        layer.mlp_up   = b.parameter(p + "mlp/up", {width, h}, {p + "mlp_input"});
        layer.mlp_down = b.parameter(p + "mlp/down", {h, width}, {p + "mlp_product"});
        tail(b, layer.mlp_down);
        if (config.moe) {
            const std::uint64_t e = config.num_experts, w = config.moe_intermediate_size;
            const std::string m   = p + "moe/";
            MoeWeights moe;
            moe.mlp_post_norm = b.direct(p + "mlp/post_norm", {h}, QType::FP32);
            moe.pre_norm      = b.direct(m + "pre_norm", {h}, QType::FP32);
            moe.post_norm     = b.direct(m + "post_norm", {h}, QType::FP32);
            moe.router        = b.direct(m + "router", {e, h}, QType::FP32);
            moe.router_scale  = b.direct(m + "router_scale", {h}, QType::FP32);
            moe.expert_scale  = b.direct(m + "expert_scale", {e}, QType::FP32);
            moe.gate_up       = b.parameter(m + "gate_up", {e * 2 * w, h}, {m + "input"});
            moe.down          = b.parameter(m + "down", {e * h, w}, {m + "product"});
            tail(b, moe.down);
            tail(b, moe.gate_up);
            layer.moe = moe;
        }
        weights.layers.push_back(std::move(layer));
    }
    std::optional<AssistantConfig> assistant_config;
    std::optional<AssistantWeights> assistant;
    if (options.assistant) {
        if (!has_assistant(reader)) {
            throw artifact::ArtifactError(
                "the artifact holds no Gemma 4 assistant (convert with --components text,mtp)");
        }
        const AssistantConfig& ac = assistant_config.emplace(
            parse_assistant_config(reader.directory().component("mtp").config, config));
        const std::uint64_t ah = ac.hidden_size;
        AssistantWeights aw;
        aw.output_head     = b.parameter("mtp/output_head", {ac.vocab_size, ah},
                                         {"mtp/final_hidden"});
        aw.final_norm      = b.direct("mtp/final_norm", {ah}, QType::FP32);
        aw.pre_projection  = b.parameter("mtp/pre_projection", {ah, 2 * h}, {"mtp/stem_input"});
        aw.post_projection = b.parameter("mtp/post_projection", {h, ah}, {"mtp/final_hidden"});
        for (std::uint32_t i = 0; i < ac.full_attention.size(); ++i) {
            const std::string p = "mtp/layers/" + std::to_string(i) + "/";
            const std::string a = p + "attention/";
            const std::uint64_t d =
                ac.full_attention[i] ? config.global_head_dim : config.head_dim;
            const std::uint64_t q = std::uint64_t(ac.num_attention_heads) * d;
            AssistantLayerWeights layer;
            layer.input_norm          = b.direct(p + "input_norm", {ah}, QType::FP32);
            layer.post_attention_norm = b.direct(p + "post_attention_norm", {ah}, QType::FP32);
            layer.ffn_norm            = b.direct(p + "ffn_norm", {ah}, QType::FP32);
            layer.post_ffn_norm       = b.direct(p + "post_ffn_norm", {ah}, QType::FP32);
            layer.layer_scalar        = b.direct(p + "layer_scalar", {1}, QType::FP32);
            layer.query      = b.parameter(a + "query", {q, ah}, {p + "mixer_input"});
            layer.query_norm = b.direct(a + "query_norm", {d}, QType::FP32);
            layer.output     = b.parameter(a + "output", {ah, q}, {p + "attention_output"});
            const std::uint64_t width = ac.intermediate_size;
            layer.mlp_gate = b.parameter(p + "mlp/gate", {width, ah}, {p + "mlp_input"});
            layer.mlp_up   = b.parameter(p + "mlp/up", {width, ah}, {p + "mlp_input"});
            layer.mlp_down = b.parameter(p + "mlp/down", {ah, width}, {p + "mlp_product"});
            tail(b, layer.mlp_down);
            aw.layers.push_back(std::move(layer));
        }
        assistant = std::move(aw);
    }
    Model::VisionParts vision_parts;
    if (vision) {
        const VisionConfig& vc = vision_parts.config.emplace(
            parse_vision_config(reader.directory().component("vision").config, config));
        const auto residency = options.vision == VisionPlacement::Host     ? artifact::Residency::Host
                               : options.vision == VisionPlacement::Pinned ? artifact::Residency::Pinned
                                                                           : artifact::Residency::Device;
        const std::uint64_t vh = vc.hidden_size, vw = vc.intermediate_size;
        const std::uint64_t d  = vc.head_dim;
        const auto matrix = [&](const std::string& name, artifact::Shape shape,
                                const std::string& input) {
            return b.parameter(name, std::move(shape), {input}, QType::BF16, residency);
        };
        const auto vector = [&](const std::string& name, std::uint64_t n) {
            return b.direct(name, {n}, QType::FP32, residency);
        };
        VisionWeights vw_;
        vw_.patch_embedding = matrix("vision/patch_embedding", {vh, vc.patch_features()},
                                     "vision/patches");
        vw_.position_table  = b.direct("vision/position_table",
                                       {2 * std::uint64_t(vc.position_embedding_size), vh},
                                       QType::BF16, residency);
        vw_.projection      = matrix("vision/projection", {vc.projection_dim, vh},
                                     "vision/soft_tokens");
        if (vc.standardize) {
            vw_.std_bias  = vector("vision/std_bias", vh);
            vw_.std_scale = vector("vision/std_scale", vh);
        }
        for (std::uint32_t i = 0; i < vc.num_hidden_layers; ++i) {
            const std::string p = "vision/layers/" + std::to_string(i) + "/";
            const std::string a = p + "attention/";
            VisionLayerWeights layer;
            layer.input_norm          = vector(p + "input_norm", vh);
            layer.post_attention_norm = vector(p + "post_attention_norm", vh);
            layer.ffn_norm            = vector(p + "ffn_norm", vh);
            layer.post_ffn_norm       = vector(p + "post_ffn_norm", vh);
            layer.query               = matrix(a + "query", {vh, vh}, p + "mixer_input");
            layer.key                 = matrix(a + "key", {vh, vh}, p + "mixer_input");
            layer.value               = matrix(a + "value", {vh, vh}, p + "mixer_input");
            layer.output              = matrix(a + "output", {vh, vh}, p + "attention_output");
            layer.query_norm          = vector(a + "query_norm", d);
            layer.key_norm            = vector(a + "key_norm", d);
            layer.mlp_gate            = matrix(p + "mlp/gate", {vw, vh}, p + "mlp_input");
            layer.mlp_up              = matrix(p + "mlp/up", {vw, vh}, p + "mlp_input");
            layer.mlp_down            = matrix(p + "mlp/down", {vh, vw}, p + "mlp_product");
            vw_.layers.push_back(layer);
        }
        vision_parts.weights   = std::move(vw_);
        vision_parts.placement = options.vision;
    }
    InstanceInfo info;
    info.name            = reader.directory().metadata.value("name", std::string("Gemma 4"));
    info.metadata_json   = reader.directory().metadata.dump();
    info.provenance_json = reader.directory().provenance.dump();
    info.artifact_id     = reader.artifact_id();
    auto pending         = std::move(b.weights);
    auto materialization = std::move(binder).finish();
    auto backing = artifact::materialize(reader, std::move(materialization), device, observer);
    auto bound   = qwen3_5::loading::resolve_weights(std::move(pending), backing);
    return std::unique_ptr<Model>(new Model(std::move(config), std::move(weights),
                                            std::move(assistant_config), std::move(assistant),
                                            std::move(vision_parts), std::move(bound),
                                            std::move(resources), std::move(info),
                                            std::move(backing)));
}

} // namespace ninfer::models::gemma4
