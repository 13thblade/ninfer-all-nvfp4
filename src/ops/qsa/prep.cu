// QSA index-query preparation and pooled index keys (include/ninfer/ops/qsa.h).
// Adapted from Infernix fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de src/ops/qsa/qsa.cu (Apache-2.0):
// index_query_kernel, pool_kernel and tail_kernel. Modified for NInfer: one warp per vector, the
// norm output is not rounded before the rotation, the gain may be plain or unit-offset, and the
// rotation angle is formed and range-reduced in FP64.
// commit_tail_kernel: adapted from Infernix a3edb450 src/ops/qsa/qsa.cu (Apache-2.0); modified for
// NInfer-3090: one sequence, and it writes every committed slot in order (the tail of n one-column
// calls, bit for bit) instead of only the last block's.

#include "ops/qsa/launch.h"

#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kDi       = kQsaIndexDim;
constexpr int kR        = kQsaBlockTokens;
constexpr int kHalfRot  = kQsaRotaryDim / 2;
constexpr int kSlot     = kDi / kR; // pooled-plane lanes per token slot
constexpr unsigned kAll = 0xffffffffU;

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("qsa ") + what + ": " + cudaGetErrorString(error));
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(kAll, v, o); }
    return v;
}

// cos and sin of pair i at RoPE axis value `position`: the angle in FP64, reduced to one turn.
__device__ __forceinline__ void rope_sincos(int position, double inverse_frequency, float& c,
                                            float& s) {
    constexpr double kInvTwoPi = 1.59154943091895336e-01;
    constexpr double kTwoPi    = 6.28318530717958648e+00;
    const double angle         = static_cast<double>(position) * inverse_frequency;
    const float reduced = static_cast<float>(angle - nearbyint(angle * kInvTwoPi) * kTwoPi);
    sincosf(reduced, &s, &c);
}

// A warp normalizes and rotates one 128-vector held as x[j] = element lane + 32 j, rotating pair
// (lane, lane + 32) by axis lane % 3 of `rope` (axis a at rope[a * rope_stride]).
__device__ __forceinline__ void norm_rotate(float (&x)[4], const bf16* __restrict__ weight,
                                            bool unit_offset, float eps,
                                            const std::int32_t* __restrict__ rope,
                                            std::int64_t rope_stride, const QsaRopeTable& table,
                                            int lane) {
    const float squares = warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
    const float inv     = rsqrtf(squares / static_cast<float>(kDi) + eps);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float w = __bfloat162float(weight[lane + 32 * j]);
        x[j]          = x[j] * inv * (unit_offset ? 1.0F + w : w);
    }
    float c, s;
    rope_sincos(rope[(lane % 3) * rope_stride], table.inverse_frequency[lane], c, s);
    const float a = x[0], b = x[1];
    x[0]          = a * c - b * s;
    x[1]          = b * c + a * s;
}

// One warp per (index head, column).
__global__ void __launch_bounds__(128) index_query_kernel(bf16* __restrict__ q,
                                                          const bf16* __restrict__ weight,
                                                          const std::int32_t* __restrict__ rope,
                                                          int columns, bool unit_offset, float eps,
                                                          QsaRopeTable table) {
    const int lane = threadIdx.x & 31, head = threadIdx.x >> 5, t = blockIdx.x;
    bf16* row      = q + (static_cast<std::int64_t>(t) * kQsaIndexHeads + head) * kDi;
    float x[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) { x[j] = __bfloat162float(row[lane + 32 * j]); }
    norm_rotate(x, weight, unit_offset, eps, rope + t, columns, table, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) { row[lane + 32 * j] = __float2bfloat16_rn(x[j]); }
}

// One warp per column; a column that completes a block pools it.
__global__ void __launch_bounds__(128)
    pool_kernel(const bf16* __restrict__ raw, const std::int32_t* __restrict__ positions,
                const std::int32_t* __restrict__ rope, const std::int32_t* __restrict__ block_rope,
                const bf16* __restrict__ weight, const std::int32_t* __restrict__ table,
                const bf16* __restrict__ tail, bf16* __restrict__ pooled, int columns,
                bool unit_offset, float eps, QsaRopeTable table_rope) {
    const int lane = threadIdx.x & 31;
    const int t    = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (t >= columns) { return; }
    const int start = positions[0];
    const int p     = start + t;
    if ((p + 1) % kR != 0) { return; }
    const int first = p - kR + 1;
    float x[4]      = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
    for (int j = 0; j < kR; ++j) {
        const int q        = first + j;
        const bf16* source = q >= start ? raw + static_cast<std::int64_t>(t - (p - q)) * kDi
                                        : tail + static_cast<std::int64_t>(q % kR) * kDi;
#pragma unroll
        for (int e = 0; e < 4; ++e) { x[e] += __bfloat162float(source[lane + 32 * e]); }
    }
    // The BF16 mean is the contract's boundary (scaling by 1/4 is exact).
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        x[e] = __bfloat162float(__float2bfloat16_rn(x[e] * (1.0F / static_cast<float>(kR))));
    }
    const std::int32_t* rope_at = first >= start ? rope + (t - (p - first)) : block_rope;
    const std::int64_t stride   = first >= start ? columns : 1;
    norm_rotate(x, weight, unit_offset, eps, rope_at, stride, table_rope, lane);
    const int page   = paged_kv_physical_page(table, first);
    bf16* block_base = pooled + static_cast<std::int64_t>(kSlot) * kPagedKVPageSize * page +
                       static_cast<std::int64_t>(kSlot) * (first & kPagedKVPageMask);
#pragma unroll
    for (int e = 0; e < 4; ++e) { block_base[lane + 32 * e] = __float2bfloat16_rn(x[e]); }
}

// One warp: the latest block's positions q % R < R - 1 up to the call's last position.
__global__ void __launch_bounds__(32) tail_kernel(const bf16* __restrict__ raw,
                                                  const std::int32_t* __restrict__ positions,
                                                  int columns, bf16* __restrict__ tail) {
    const int lane  = threadIdx.x;
    const int start = positions[0];
    const int last  = start + columns - 1;
    const int base  = last - last % kR;
    const int end   = min(last, base + kR - 2);
    for (int q = max(base, start); q <= end; ++q) {
        const bf16* source = raw + static_cast<std::int64_t>(q - start) * kDi;
        bf16* target       = tail + static_cast<std::int64_t>(q % kR) * kDi;
#pragma unroll
        for (int e = 0; e < 4; ++e) { target[lane + 32 * e] = source[lane + 32 * e]; }
    }
}

// One warp per layer: the tail after the first commit[0] of the recorded columns (Infernix
// a3edb450 src/ops/qsa/qsa.cu commit_tail_kernel, one sequence).
__global__ void __launch_bounds__(32) commit_tail_kernel(const bf16* __restrict__ raw,
                                                         const std::int32_t* __restrict__ positions,
                                                         const std::int32_t* __restrict__ commit,
                                                         int width, bf16* __restrict__ tails,
                                                         std::int64_t layer_stride) {
    const int lane = threadIdx.x, layer = blockIdx.x;
    const int n    = commit[0];
    if (n <= 0) { return; }
    const int start = positions[0];
    const int last  = start + n - 1;
    const bf16* layer_raw = raw + static_cast<std::int64_t>(layer) * width * kDi;
    bf16* tail            = tails + static_cast<std::int64_t>(layer) * layer_stride;
    // Every committed position that does not complete a block, in order: the tail n one-column
    // calls leave, including the slots a commit spanning two blocks fills from the earlier block
    // (pooling never reads them, but the committed state equals the decoded state bit for bit).
    for (int q = start; q <= last; ++q) {
        if (q % kR == kR - 1) { continue; }
        const bf16* source = layer_raw + static_cast<std::int64_t>(q - start) * kDi;
        bf16* target       = tail + static_cast<std::int64_t>(q % kR) * kDi;
#pragma unroll
        for (int e = 0; e < 4; ++e) { target[lane + 32 * e] = source[lane + 32 * e]; }
    }
}

} // namespace

void qsa_commit_tails_launch(const Tensor& raw_keys, const Tensor& positions,
                             const Tensor& commit_columns, Tensor& tail0,
                             std::int64_t tail_stride_elements, cudaStream_t stream) {
    commit_tail_kernel<<<raw_keys.ne[2], 32, 0, stream>>>(
        static_cast<const bf16*>(raw_keys.data), static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(commit_columns.data), raw_keys.ne[1],
        static_cast<bf16*>(tail0.data), tail_stride_elements);
    check_launch("commit tails");
}

QsaRopeTable qsa_rope_table(const QsaIndexerGeometry& geometry) {
    QsaRopeTable table{};
    for (int i = 0; i < kHalfRot; ++i) {
        table.inverse_frequency[i] =
            std::pow(static_cast<double>(geometry.theta),
                     -2.0 * static_cast<double>(i) / static_cast<double>(geometry.rotary_dim));
    }
    return table;
}

void qsa_index_query_launch(const Tensor& rope_positions, const Tensor& norm_weight,
                            const QsaIndexerGeometry& geometry, Tensor& q, cudaStream_t stream) {
    const int columns = q.ne[2];
    index_query_kernel<<<columns, 32 * kQsaIndexHeads, 0, stream>>>(
        static_cast<bf16*>(q.data), static_cast<const bf16*>(norm_weight.data),
        static_cast<const std::int32_t*>(rope_positions.data), columns, geometry.unit_offset_norm,
        geometry.eps, qsa_rope_table(geometry));
    check_launch("index query");
}

void qsa_pool_keys_launch(const Tensor& raw_keys, const Tensor& positions,
                          const Tensor& rope_positions, const Tensor& block_start_rope,
                          const Tensor& norm_weight, const QsaIndexerGeometry& geometry,
                          const Tensor& block_table, Tensor& tail, Tensor& pooled_pages,
                          cudaStream_t stream) {
    const int columns = raw_keys.ne[1];
    // Pooling reads the old tail; the tail update runs after it.
    pool_kernel<<<(columns + 3) / 4, 128, 0, stream>>>(
        static_cast<const bf16*>(raw_keys.data), static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(rope_positions.data),
        static_cast<const std::int32_t*>(block_start_rope.data),
        static_cast<const bf16*>(norm_weight.data),
        static_cast<const std::int32_t*>(block_table.data), static_cast<const bf16*>(tail.data),
        static_cast<bf16*>(pooled_pages.data), columns, geometry.unit_offset_norm, geometry.eps,
        qsa_rope_table(geometry));
    check_launch("pool keys");
    tail_kernel<<<1, 32, 0, stream>>>(static_cast<const bf16*>(raw_keys.data),
                                      static_cast<const std::int32_t*>(positions.data), columns,
                                      static_cast<bf16*>(tail.data));
    check_launch("pool tail");
}

} // namespace ninfer::ops::detail
