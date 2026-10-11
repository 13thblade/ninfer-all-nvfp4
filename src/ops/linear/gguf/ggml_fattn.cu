// llama.cpp's flash_attn_ext_f16 (fattn-mma-f16.cuh) launched without ggml's backend context: the
// host side of launch_fattn and ggml_cuda_flash_attn_ext_mma_f16_case, for the two Gemma 4 shapes.

#include "ggml_fattn.h"
#include "ggml_bridge_internal.cuh"

#include "ggml-cuda/fattn-mma-f16.cuh"

#include <algorithm>
#include <mutex>
#include <stdexcept>

namespace ninfer::ops::gguf {
namespace {

using detail::check;
using detail::device_facts;

// The configurations instantiated: width 256 with two query heads per key/value head, 32 query rows
// per tile; width 512 with eight, 8 rows per tile (64 columns each, ggml's prompt choice).
template <int D> struct Shape;
template <> struct Shape<256> {
    static constexpr int ncols1 = 32;
    static constexpr int ncols2 = 2;
};
template <> struct Shape<512> {
    static constexpr int ncols1 = 8;
    static constexpr int ncols2 = 8;
};

struct Launch {
    fattn_kernel_t kernel = nullptr;
    int threads           = 0;
    std::size_t shared    = 0;
    int nbatch_fa         = 0;
    int blocks_per_sm     = 0;
};

template <int D>
const Launch& launch_for() {
    constexpr int ncols1 = Shape<D>::ncols1;
    constexpr int ncols2 = Shape<D>::ncols2;
    constexpr int ncols  = ncols1 * ncols2;
    static Launch launches[16];
    static std::once_flag once[16];
    int device = 0;
    check(cudaGetDevice(&device), "device");
    std::call_once(once[device & 15], [&] {
        const int cc               = device_facts().cc;
        const int nthreads         = ggml_cuda_fattn_mma_get_nthreads(D, D, ncols, cc);
        const int nbatch_fa        = ggml_cuda_fattn_mma_get_nbatch_fa(D, D, ncols, cc);
        const int nbatch_K2        = ggml_cuda_fattn_mma_get_nbatch_K2(D, D, ncols, cc);
        const int nbatch_V2        = ggml_cuda_fattn_mma_get_nbatch_V2(D, D, ncols, cc);
        const int nbatch_combine   = ggml_cuda_fattn_mma_get_nbatch_combine(D, D, ncols, cc);
        const bool Q_in_reg        = ggml_cuda_fattn_mma_get_Q_in_reg(D, D, ncols, cc);
        const int nstages          = ggml_cuda_fattn_mma_get_nstages(D, D, ncols1, ncols2, cc);
        const int cols_per_warp    = std::min(ncols, get_cols_per_warp(cc));
        const int nwarps           = nthreads / WARP_SIZE;
        const bool swizzled        = ggml_cuda_fattn_mma_get_swizzled(D, D, ncols1, ncols2, cc);
        const int stride_tile_K    = ggml_cuda_fattn_mma_get_stride_tile(nbatch_K2, swizzled);
        const int stride_tile_V    = ggml_cuda_fattn_mma_get_stride_tile(nbatch_V2, swizzled);
        const std::size_t kv_1     = std::size_t(nbatch_fa) * std::max(stride_tile_K, stride_tile_V) * sizeof(half2);
        const std::size_t kv_2     = std::size_t(nbatch_fa) * (stride_tile_K + stride_tile_V) * sizeof(half2);
        const std::size_t q_bytes  = std::size_t(ncols) * (D / 2 + 4) * sizeof(half2);
        const std::size_t mask     = std::size_t(ncols1) * (nbatch_fa / 2 + 4) * sizeof(half2);
        const std::size_t combine  = std::size_t(nwarps) * cols_per_warp * (nbatch_combine + 4) * sizeof(half2);
        const std::size_t kv       = nstages <= 1 ? kv_1 : kv_2;
        const std::size_t shared   = std::max(combine, Q_in_reg ? std::max(q_bytes, kv + mask)
                                                                : q_bytes + kv + mask);
        Launch& out = launches[device & 15];
        out.kernel  = flash_attn_ext_f16<D, D, ncols1, ncols2, false, false, false>;
        out.threads = nthreads;
        out.shared  = shared;
        out.nbatch_fa = nbatch_fa;
        check(cudaFuncSetAttribute(reinterpret_cast<const void*>(out.kernel),
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(shared)),
              "flash attention shared memory");
        int blocks = 0;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, out.kernel, nthreads, shared),
              "flash attention occupancy");
        out.blocks_per_sm = std::max(blocks, 1);
    });
    return launches[device & 15];
}

struct Grid {
    int blocks     = 0;
    int ntiles_dst = 0;
    int ntiles_x   = 0;
    int ntiles_kv  = 0;
    bool fixup     = false;
};

template <int D>
Grid grid_for(const Launch& launch, int tokens, int kv_heads, int slots) {
    constexpr int ncols1 = Shape<D>::ncols1;
    const auto& facts = device_facts();
    Grid g;
    g.ntiles_x   = (tokens + ncols1 - 1) / ncols1;
    g.ntiles_dst = g.ntiles_x * kv_heads;
    g.ntiles_kv  = (slots + launch.nbatch_fa - 1) / launch.nbatch_fa;
    const int max_blocks = launch.blocks_per_sm * facts.sm_count;
    const int waves      = (g.ntiles_dst + max_blocks - 1) / max_blocks;
    const int efficiency = 100 * g.ntiles_dst / (max_blocks * waves);
    const bool stream_k  = facts.cc >= GGML_CUDA_CC_ADA_LOVELACE || efficiency < 75;
    g.blocks = g.ntiles_dst;
    if (stream_k) {
        const int raw     = std::min(max_blocks, g.ntiles_kv * g.ntiles_dst);
        const int rounded = (raw / g.ntiles_dst) * g.ntiles_dst;
        const int loss    = rounded > 0 ? 100 * (raw - rounded) / raw : 100;
        g.blocks          = loss <= 5 ? rounded : raw;
    }
    g.fixup = g.ntiles_dst % g.blocks != 0;
    return g;
}

template <int D>
std::size_t workspace_for(int query_heads, int kv_heads, int max_tokens) {
    constexpr int ncols = Shape<D>::ncols1 * Shape<D>::ncols2;
    (void)query_heads;
    (void)kv_heads;
    (void)max_tokens;
    // Stream-k never launches more than a wave of blocks; each keeps its partial rows and their
    // (max, sum) pairs.
    const Launch& launch = launch_for<D>();
    const std::size_t blocks = std::size_t(launch.blocks_per_sm) * device_facts().sm_count;
    return blocks * ncols * (2 + D / 2) * sizeof(float2) + 256;
}

template <int D>
void run(const FlashAttentionArgs& a, void* workspace, std::size_t workspace_bytes,
         cudaStream_t stream) {
    constexpr int ncols1 = Shape<D>::ncols1;
    constexpr int ncols2 = Shape<D>::ncols2;
    constexpr int ncols  = ncols1 * ncols2;
    if (a.query_heads != a.kv_heads * ncols2) {
        throw std::invalid_argument("gguf flash attention: unsupported grouping for this width");
    }
    const Launch& launch = launch_for<D>();
    const Grid g         = grid_for<D>(launch, a.tokens, a.kv_heads, a.slots);
    float2* meta         = nullptr;
    if (g.fixup) {
        const std::size_t need = std::size_t(g.blocks) * ncols * (2 + D / 2) * sizeof(float2);
        if (workspace == nullptr || workspace_bytes < need) {
            throw std::invalid_argument("gguf flash attention: workspace too small");
        }
        meta = static_cast<float2*>(workspace);
    }
    const uint3 ne01 = init_fastdiv_values(a.tokens);
    const std::int32_t nb01 = static_cast<std::int32_t>(a.q_token_stride * sizeof(float));
    const std::int32_t nb02 = static_cast<std::int32_t>(D * sizeof(float));
    const std::int32_t nb03 = static_cast<std::int32_t>(nb01) * a.tokens;
    const std::int32_t nb31 = static_cast<std::int32_t>(a.mask_row_halves * sizeof(half));
    const dim3 block(WARP_SIZE, launch.threads / WARP_SIZE, 1);
    launch.kernel<<<dim3(g.blocks, 1, 1), block, launch.shared, stream>>>(
        reinterpret_cast<const char*>(a.q), static_cast<const char*>(a.k),
        static_cast<const char*>(a.v), static_cast<const char*>(a.mask), nullptr, nullptr, a.out,
        meta, a.scale, 0.0f, 1.0f, 1.0f, 1u, 0.0f, D, ne01, a.query_heads, 1, nb01, nb02, nb03, D,
        a.slots, a.kv_heads, 1, static_cast<std::int32_t>(a.row_bytes),
        static_cast<std::int32_t>(a.head_bytes), a.head_bytes * a.kv_heads,
        static_cast<std::int32_t>(a.row_bytes), static_cast<std::int32_t>(a.head_bytes),
        a.head_bytes * a.kv_heads, a.tokens, 1, 1, nb31, nb31 * a.tokens,
        std::int64_t(nb31) * a.tokens);
    check(cudaGetLastError(), "flash attention launch");
    if (!g.fixup) { return; }
    const int gqa = a.query_heads / a.kv_heads;
    if (g.blocks % g.ntiles_dst == 0 && g.blocks > g.ntiles_dst) {
        const int per_tile = g.blocks / g.ntiles_dst;
        const uint3 fd0    = init_fastdiv_values(g.ntiles_x * a.kv_heads);
        const uint3 fd1    = init_fastdiv_values(g.ntiles_x);
        const uint3 fd2    = init_fastdiv_values(g.ntiles_x);
        flash_attn_stream_k_fixup_uniform<D, ncols1, ncols2>
            <<<dim3(g.ntiles_dst, ncols1, ncols2), dim3(D, 1, 1), 0, stream>>>(
                a.out, meta, a.tokens, a.query_heads, a.kv_heads, g.blocks, gqa, per_tile, fd0, fd1,
                fd2);
    } else {
        const int total = g.ntiles_kv * g.ntiles_dst;
        const uint3 fd0 = init_fastdiv_values(g.ntiles_kv * g.ntiles_x * a.kv_heads);
        const uint3 fd1 = init_fastdiv_values(g.ntiles_kv * g.ntiles_x);
        const uint3 fd2 = init_fastdiv_values(g.ntiles_kv * g.ntiles_x);
        const uint3 fd3 = init_fastdiv_values(g.ntiles_kv);
        flash_attn_stream_k_fixup_general<D, ncols1, ncols2>
            <<<dim3(g.blocks, ncols1, ncols2), dim3(D, 1, 1), 0, stream>>>(
                a.out, meta, a.tokens, a.query_heads, gqa, total, fd0, fd1, fd2, fd3);
    }
    check(cudaGetLastError(), "flash attention fixup launch");
}

} // namespace

std::size_t flash_attention_workspace_bytes(int head_dim, int query_heads, int kv_heads,
                                            int max_tokens) {
    switch (head_dim) {
    case 256: return workspace_for<256>(query_heads, kv_heads, max_tokens);
    case 512: return workspace_for<512>(query_heads, kv_heads, max_tokens);
    default: break;
    }
    throw std::invalid_argument("gguf flash attention: head width must be 256 or 512");
}

void flash_attention(const FlashAttentionArgs& args, void* workspace, std::size_t workspace_bytes,
                     cudaStream_t stream) {
    if (args.q == nullptr || args.k == nullptr || args.v == nullptr || args.mask == nullptr ||
        args.out == nullptr || args.tokens <= 0 || args.kv_heads <= 0 || args.slots <= 0 ||
        args.slots % FATTN_KQ_STRIDE != 0 || args.mask_row_halves < args.slots ||
        args.row_bytes % 16 != 0 || args.head_bytes % 16 != 0 ||
        (args.q_token_stride * sizeof(float)) % 16 != 0 || args.mask_row_halves % 8 != 0) {
        throw std::invalid_argument("gguf flash attention: invalid arguments");
    }
    switch (args.head_dim) {
    case 256: return run<256>(args, workspace, workspace_bytes, stream);
    case 512: return run<512>(args, workspace, workspace_bytes, stream);
    default: break;
    }
    throw std::invalid_argument("gguf flash attention: head width must be 256 or 512");
}

} // namespace ninfer::ops::gguf
