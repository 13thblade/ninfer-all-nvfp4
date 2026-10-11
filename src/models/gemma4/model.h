#pragma once

// Gemma 4 (Gemma4ForCausalLM): the immutable model a v3 artifact describes. Matrices stay in the
// GGUF block formats the converter imported; vectors are FP32. The frontend resources are Gemma 4's
// tokenizer and chat template, which the shared frontend serves in its Gemma dialect.

#include "artifact/materializer.h"
#include "core/device.h"
#include "core/startup.h"
#include "models/gemma4/config.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/model.h"
#include "models/qwen3_5/weights.h"
#include "ninfer/ops/weight_input.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::artifact {
class Reader;
}

namespace ninfer::models::gemma4 {

using qwen3_5::BoundWeight;
using qwen3_5::FrontendResources;
using qwen3_5::InstanceInfo;
using qwen3_5::WeightId;

struct AttentionWeights {
    WeightId query, key, output, query_norm, key_norm;
    std::optional<WeightId> value; // sliding-window blocks; full attention reads K as V
};

struct MoeWeights {
    WeightId router, router_scale, expert_scale, gate_up, down, pre_norm, post_norm;
    WeightId mlp_post_norm; // post_feedforward_layernorm_1 of the dense branch
};

struct LayerWeights {
    WeightId input_norm, post_attention_norm, ffn_norm, post_ffn_norm, layer_scalar;
    AttentionWeights attention;
    WeightId mlp_gate, mlp_up, mlp_down;
    std::optional<MoeWeights> moe;
};

struct TextWeights {
    WeightId output_head, final_norm; // the head is also the token table
    std::vector<LayerWeights> layers;
};

struct LoadOptions {
    std::filesystem::path artifact;
};

class Model {
public:
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;

    [[nodiscard]] const TextConfig& config() const noexcept { return config_; }
    [[nodiscard]] const TextWeights& weights() const noexcept { return weights_; }
    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }
    [[nodiscard]] std::span<const BoundWeight> weight_data() const noexcept { return bound_; }
    [[nodiscard]] ops::WeightInput input(WeightId id) const;
    [[nodiscard]] const FrontendResources& resources() const noexcept { return resources_; }
    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }
    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }

private:
    friend std::unique_ptr<Model> load_model(const artifact::Reader&, const LoadOptions&,
                                             DeviceContext&, const StartupObserver*);
    Model(TextConfig config, TextWeights weights, std::vector<BoundWeight> bound,
          FrontendResources resources, InstanceInfo info, artifact::MaterializedArtifact backing);

    artifact::MaterializedArtifact backing_;
    TextConfig config_;
    TextWeights weights_;
    std::vector<BoundWeight> bound_;
    FrontendResources resources_;
    InstanceInfo info_;
};

// Whether the artifact's text component is Gemma 4.
[[nodiscard]] bool is_gemma4(const artifact::Reader& reader);

[[nodiscard]] std::unique_ptr<Model> load_model(const artifact::Reader& reader,
                                                const LoadOptions& options, DeviceContext& device,
                                                const StartupObserver* observer = nullptr);

} // namespace ninfer::models::gemma4
