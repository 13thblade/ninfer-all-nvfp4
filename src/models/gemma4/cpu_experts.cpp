#include "models/gemma4/cpu_experts.h"

#include "core/cpu_spin_pool.h"
#include "ops/linear/gguf/ggml_bridge.h"
// After the bridge, which brings the CUDA host definitions fixed_point.cuh's qualifiers need.
#include "ops/common/fixed_point.cuh"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ninfer::models::gemma4 {
namespace {

namespace gguf = ops::gguf;

// Rows of one task: a [gate; up] task takes 64 of an expert's gate or up rows, a down task 64
// rows of every CPU expert, so that no two tasks add to the same sums.
constexpr int kRows = 64;

template <class Job>
void run(CpuSpinPool* pool, std::size_t count, Job&& job) {
    if (pool) { return pool->run(count, job); }
    for (std::size_t i = 0; i < count; ++i) { job(i); }
}

} // namespace

CpuExperts::CpuExperts(std::vector<CpuExpertLayer> layers, int hidden, int width, int top_k)
    : layers_(std::move(layers)), hidden_(hidden), width_(width), top_k_(top_k),
      backend_(ops::gguf_cpu_backend_selected(ops::GgufCpuBackend::Automatic)) {
    if (hidden <= 0 || width <= 0 || hidden % 32 != 0 || width % 32 != 0 || top_k <= 0) {
        throw std::invalid_argument("Gemma 4 CPU experts: invalid expert geometry");
    }
}

bool CpuExperts::computes(const CpuExpertLayer& layer, int hidden, int width) noexcept {
    const auto fits = [](QType format, int k) {
        return ops::gguf_cpu_supported(format) && k % gguf_block_shape(format).elements == 0;
    };
    return fits(layer.gate_up, hidden) && fits(layer.down, width);
}

std::size_t CpuExperts::activation_bytes(int tokens) const {
    return gguf::vector_activation_bytes(hidden_, tokens);
}
std::size_t CpuExperts::weight_bytes(int tokens) const { return std::size_t(tokens) * top_k_ * 4; }
std::size_t CpuExperts::input_bytes() const {
    return activation_bytes(kMaxTokens) + weight_bytes(kMaxTokens);
}
std::size_t CpuExperts::product_bytes() const { return std::size_t(kMaxTokens) * hidden_ * 8; }

bool CpuExperts::supports(std::size_t layer) const {
    return layer < layers_.size() && computes(layers_[layer], hidden_, width_);
}

std::size_t CpuExperts::compute(const Work& work) {
    const CpuExpertLayer& layer = layers_.at(work.layer);
    const int pairs  = int(work.ids.size());
    const int tokens = pairs / top_k_;
    if (pairs % top_k_ != 0 || tokens < 1 || tokens > kMaxTokens) {
        throw std::invalid_argument("Gemma 4 CPU experts: a call of 1 to 8 tokens");
    }
    const auto* base    = reinterpret_cast<const std::uint8_t*>(work.inputs);
    const auto* codes   = reinterpret_cast<const std::int8_t*>(base);
    const auto* ds      = reinterpret_cast<const std::uint16_t*>(
        base + gguf::vector_activation_scales_offset(hidden_, tokens));
    const auto* weights = reinterpret_cast<const float*>(base + activation_bytes(tokens));
    auto* sums          = reinterpret_cast<std::uint64_t*>(work.products);
    std::memset(sums, 0, std::size_t(tokens) * hidden_ * 8);

    const std::size_t count = work.experts.size();
    if (jobs_.size() < count) { jobs_.resize(count); }
    for (std::size_t i = 0; i < count; ++i) {
        Job& job   = jobs_[i];
        job.expert = work.experts[i];
        job.pairs.clear();
        job.tokens.clear();
        for (int p = 0; p < pairs; ++p) {
            if (work.ids[std::size_t(p)] != job.expert) { continue; }
            job.pairs.push_back(p);
            job.tokens.push_back(p / top_k_);
        }
        if (job.pairs.empty() || work.blobs[i].size() < 2) {
            throw std::invalid_argument("Gemma 4 CPU experts: an expert without pairs or blobs");
        }
        job.gate_up = reinterpret_cast<const std::uint8_t*>(work.blobs[i][0]);
        job.down    = reinterpret_cast<const std::uint8_t*>(work.blobs[i][1]);
        if (job.input.k() != hidden_) { job.input = ops::GgufCpuActivation(kMaxTokens, hidden_); }
        if (job.middle.k() != width_) { job.middle = ops::GgufCpuActivation(kMaxTokens, width_); }
        job.input.load_q8_1(layer.gate_up, codes, ds, job.tokens);
        const std::size_t n = job.pairs.size();
        job.gate.resize(n * width_);
        job.up.resize(n * width_);
        job.middle_bf16.resize(n * width_);
        job.rows.resize(n * hidden_);
    }

    // Gate and up rows: the stored [gate; up] matrix's first and second `width` rows.
    const int up_tasks = (width_ + kRows - 1) / kRows;
    run(work.pool, count * std::size_t(2 * up_tasks), [&](std::size_t i) {
        Job& job         = jobs_[i / (2 * up_tasks)];
        const int part   = int(i % (2 * up_tasks));
        const bool is_up = part >= up_tasks;
        const int begin  = (part % up_tasks) * kRows;
        const int end    = std::min(begin + kRows, width_);
        const ops::GgufCpuMatrix w{layer.gate_up,
                                   job.gate_up + (is_up ? std::size_t(width_) * layer.gate_up_row_bytes : 0),
                                   layer.gate_up_row_bytes, width_, hidden_};
        ops::gguf_cpu_rows(w, job.input, begin, end, (is_up ? job.up : job.gate).data(), width_,
                           backend_);
    });
    // The BF16 middle, as the GPU stores it before quantizing it for the down rows.
    run(work.pool, count, [&](std::size_t i) {
        Job& job = jobs_[i];
        ops::gguf_cpu_geglu_bf16(job.gate.data(), job.up.data(), job.gate.size(), job.middle_bf16.data(),
                                 backend_);
        job.middle.prepare_bf16(layer.down, job.middle_bf16.data(), std::size_t(width_),
                                int(job.pairs.size()));
    });
    const int down_tasks = (hidden_ + kRows - 1) / kRows;
    run(work.pool, std::size_t(down_tasks), [&](std::size_t task) {
        const int begin = int(task) * kRows;
        const int end   = std::min(begin + kRows, hidden_);
        for (std::size_t i = 0; i < count; ++i) {
            Job& job = jobs_[i];
            const ops::GgufCpuMatrix w{layer.down, job.down, layer.down_row_bytes, hidden_, width_};
            ops::gguf_cpu_rows(w, job.middle, begin, end, job.rows.data(), std::size_t(hidden_), backend_);
            for (std::size_t j = 0; j < job.pairs.size(); ++j) {
                const float weight = weights[job.pairs[j]];
                const float* row   = job.rows.data() + j * hidden_;
                std::uint64_t* sum = sums + std::size_t(job.tokens[j]) * hidden_;
                // Two's complement: the GPU's unsigned atomic additions, modulo 2^64.
                for (int r = begin; r < end; ++r) {
                    sum[r] += static_cast<std::uint64_t>(ops::fixed_product(row[r], weight));
                }
            }
        }
    });
    return std::size_t(tokens) * hidden_ * 8;
}

} // namespace ninfer::models::gemma4
