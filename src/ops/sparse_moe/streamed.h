#pragma once

#include "ninfer/ops/sparse_moe.h"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"
#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"
#include "core/expert_delivery.h"

namespace ninfer::ops::detail {

// Narrow calls overlap cached arithmetic with delivery in eight-row waves. Grouped prefill
// resolves at most all 256 experts once per bounded native slice, through the same raw tables.
// Tables hold [gate/up code, high, scale, divisor, down ...][256].
inline constexpr std::int32_t kSparseMoeStreamRows = 8;
inline constexpr std::int32_t kSparseMoeStreamPairs = 8 * kSparseMoePrefillSliceMax;
void sparse_moe_stream_prefill(const Tensor& x, const SparseMoeWeights& weights,
                               Tensor& destination, const SparseMoePrefillPlan& plan,
                               const SparseMoePrefillWorkspace& workspace, ExpertDelivery& delivery,
                               std::size_t layer, cudaStream_t stream);

void sparse_moe_stream_route(const Tensor& x, const Weight& router,
                              const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream);
void sparse_moe_stream_decode_route(const Tensor& x, const Weight& router,
                                     const SparseMoeSmallTWorkspace& workspace,
                                     cudaStream_t stream);
void sparse_moe_stream_experts(const Tensor& x, const SparseMoeWeights& weights,
                                const void* const* tables, const std::int32_t* ids,
                                const float* alpha, const float* shared_scale,
                                float* activations, float* products, bool shared,
                                cudaStream_t stream);
void sparse_moe_stream_merge(const float* products, Tensor& destination, cudaStream_t stream);

} // namespace ninfer::ops::detail
