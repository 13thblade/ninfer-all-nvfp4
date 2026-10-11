#include "models/gemma4/kernels.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::models::gemma4::kernels {
namespace {

void check_launch(const char* what) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("gemma4 ") + what + ": " + cudaGetErrorString(status));
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFu, v, o); }
    return v;
}

// Sum over the block (blockDim.x a multiple of 32, at most 1024); every thread gets the total.
__device__ float block_sum(float v) {
    __shared__ float partial[32];
    __shared__ float total;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) { partial[warp] = v; }
    __syncthreads();
    if (warp == 0) {
        float s = lane < int(blockDim.x >> 5) ? partial[lane] : 0.0f;
        s = warp_sum(s);
        if (lane == 0) { total = s; }
    }
    __syncthreads();
    return total;
}

// Two sums over the block at once (blockDim.x a multiple of 32, at most 1024).
__device__ float2 block_sum2(float a, float b) {
    __shared__ float2 partial2[32];
    __shared__ float2 total2;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    a = warp_sum(a);
    b = warp_sum(b);
    __syncthreads();
    if (lane == 0) { partial2[warp] = make_float2(a, b); }
    __syncthreads();
    if (warp == 0) {
        float2 v = lane < int(blockDim.x >> 5) ? partial2[lane] : make_float2(0.0f, 0.0f);
        v.x      = warp_sum(v.x);
        v.y      = warp_sum(v.y);
        if (lane == 0) { total2 = v; }
    }
    __syncthreads();
    return total2;
}

__device__ __forceinline__ float gelu_tanh(float x) {
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Row kernels hold one float4 of a [hidden] row per thread (hidden % 4 == 0, hidden / 4 <= 1024).
__device__ __forceinline__ float4 load_bf16x4(const __nv_bfloat16* p) {
    const uint2 w   = *reinterpret_cast<const uint2*>(p);
    const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w.x));
    const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w.y));
    return make_float4(lo.x, lo.y, hi.x, hi.y);
}
__device__ __forceinline__ void store_bf16x4(__nv_bfloat16* p, float4 v) {
    const __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y);
    const __nv_bfloat162 hi = __floats2bfloat162_rn(v.z, v.w);
    uint2 w;
    w.x = *reinterpret_cast<const unsigned*>(&lo);
    w.y = *reinterpret_cast<const unsigned*>(&hi);
    *reinterpret_cast<uint2*>(p) = w;
}
__device__ __forceinline__ float dot4(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
__device__ __forceinline__ float4 mul4(float4 a, float4 b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
__device__ __forceinline__ float4 scale4(float4 a, float s) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
__device__ __forceinline__ float4 add4(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__device__ __forceinline__ float4 ld4(const float* p, int i) {
    return reinterpret_cast<const float4*>(p)[i];
}
__device__ __forceinline__ float4 fixed4(const unsigned long long* p, int i) {
    const auto* v  = p + 4 * std::int64_t(i);
    const double s = 1.0 / 4294967296.0;
    return make_float4(float(double(static_cast<long long>(v[0])) * s),
                       float(double(static_cast<long long>(v[1])) * s),
                       float(double(static_cast<long long>(v[2])) * s),
                       float(double(static_cast<long long>(v[3])) * s));
}

// The q8_1 form of this thread's four values, elements 4i..4i+3 of row `row` of width k: eight
// consecutive threads hold one 32-value block (hidden % 32 == 0 keeps every group of eight either
// in or out of the row). Every lane of the warp calls it; `valid` lanes store.
__device__ __forceinline__ void store_q8x4(const Q8Rows& q, int k, std::int64_t row, int i, float4 v,
                                           bool valid) {
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    float sum  = v.x + v.y + v.z + v.w;
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, o);
    }
    if (!valid) { return; }
    const float d = amax / 127.0f;
    char4 c       = make_char4(0, 0, 0, 0);
    if (amax != 0.0f) {
        c = make_char4(static_cast<signed char>(roundf(v.x / d)), static_cast<signed char>(roundf(v.y / d)),
                       static_cast<signed char>(roundf(v.z / d)), static_cast<signed char>(roundf(v.w / d)));
    }
    *reinterpret_cast<char4*>(q.qs + row * k + 4 * i) = c;
    if ((i & 7) == 0) { q.ds[row * (k / 32) + i / 8] = make_half2(__float2half(d), __float2half(sum)); }
}

__global__ void scale_kernel(float* x, std::int64_t n, float s) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) { x[i] *= s; }
}

__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                int hidden, float eps, __nv_bfloat16* __restrict__ out, Q8Rows q8) {
    const int q      = hidden >> 2, i = threadIdx.x;
    const float* row = x + std::int64_t(blockIdx.x) * hidden;
    const float4 v   = i < q ? ld4(row, i) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float inv  = rsqrtf(block_sum(dot4(v, v)) / float(hidden) + eps);
    float4 y         = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (i < q) {
        const float4 wv = w != nullptr ? ld4(w, i) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        y               = mul4(scale4(v, inv), wv);
        if (out != nullptr) { store_bf16x4(out + std::int64_t(blockIdx.x) * hidden + 4 * i, y); }
    }
    if (q8.qs != nullptr) { store_q8x4(q8, hidden, blockIdx.x, i, y, i < q); }
}

// Block (t, head): Hq query heads then Hkv key/value heads; D/2 threads, thread i the pair
// (i, i + D/2).
__global__ void qkv_post_kernel(QkvArgs a) {
    const int t      = blockIdx.x;
    const int head   = blockIdx.y;
    const int d      = a.head_dim;
    const int half   = d >> 1;
    const int i      = threadIdx.x;
    const int rows   = (a.query_heads + a.kv_heads * (a.has_v ? 2 : 1)) * d;
    const __nv_bfloat16* row = a.qkv + std::int64_t(t) * rows;
    const int position = a.positions[t];
    const float angle  = float(position) * a.inv_freq[i];
    float c = 1.0f, s = 0.0f;
    if (a.inv_freq[i] != 0.0f) { sincosf(angle, &s, &c); }
    if (head < a.query_heads) {
        const __nv_bfloat16* x = row + head * d;
        const float x0 = __bfloat162float(x[i]), x1 = __bfloat162float(x[i + half]);
        const float inv = rsqrtf(block_sum(x0 * x0 + x1 * x1) / float(d) + a.eps);
        const float n0 = x0 * inv * a.q_norm[i], n1 = x1 * inv * a.q_norm[i + half];
        float* q = a.q_out + (std::int64_t(t) * a.query_heads + head) * d;
        q[i]        = n0 * c - n1 * s;
        q[i + half] = n1 * c + n0 * s;
        return;
    }
    const int kh = head - a.query_heads;
    const __nv_bfloat16* k = row + (a.query_heads + kh) * d;
    const __nv_bfloat16* v = a.has_v ? row + (a.query_heads + a.kv_heads + kh) * d : k;
    const float k0 = __bfloat162float(k[i]), k1 = __bfloat162float(k[i + half]);
    const float v0 = __bfloat162float(v[i]), v1 = __bfloat162float(v[i + half]);
    const float kinv = rsqrtf(block_sum(k0 * k0 + k1 * k1) / float(d) + a.eps);
    const float vinv = rsqrtf(block_sum(v0 * v0 + v1 * v1) / float(d) + a.eps);
    const float n0 = k0 * kinv * a.k_norm[i], n1 = k1 * kinv * a.k_norm[i + half];
    const int sequence = a.sequences != nullptr ? a.sequences[t] : a.sequence;
    const int slot     = a.ring != 0 ? position % a.ring : position;
    const std::int64_t at =
        std::int64_t(sequence) * a.seq_stride + std::int64_t(kh) * a.head_stride +
        std::int64_t(slot) * d;
    a.k_cache[at + i]        = __float2half(n0 * c - n1 * s);
    a.k_cache[at + i + half] = __float2half(n1 * c + n0 * s);
    a.v_cache[at + i]        = __float2half(v0 * vinv);
    a.v_cache[at + i + half] = __float2half(v1 * vinv);
}

__global__ void build_mask_kernel(__half* mask, int tokens, int slots, int first, int ring,
                                  int window) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (s >= slots) { return; }
    const int query = first + t;
    bool visible    = false;
    if (ring == 0) {
        visible = s <= query;
    } else {
        const int last = first + tokens - 1;
        if (s <= last) {
            const int p = s + ((last - s) / ring) * ring; // latest position in slot s
            visible     = p <= query && p > query - window;
        }
    }
    mask[std::int64_t(t) * slots + s] = visible ? __float2half(0.0f) : __float2half(-INFINITY);
}

// Block (position, head): the position's K and V rows, 16-byte words.
__global__ void ring_copy_kernel(RingCopyArgs a) {
    const int p = a.begin + blockIdx.x, h = blockIdx.y;
    const std::int64_t src = h * a.src_head_stride + std::int64_t(p % a.src_ring) * a.head_dim;
    const std::int64_t dst = h * a.dst_head_stride + std::int64_t(p % a.dst_ring) * a.head_dim;
    for (int w = threadIdx.x; w < a.head_dim / 8; w += blockDim.x) {
        reinterpret_cast<uint4*>(a.dst_k + dst)[w] = reinterpret_cast<const uint4*>(a.src_k + src)[w];
        reinterpret_cast<uint4*>(a.dst_v + dst)[w] = reinterpret_cast<const uint4*>(a.src_v + src)[w];
    }
}

__global__ void f32_to_bf16_kernel(const float* in, __nv_bfloat16* out, std::int64_t n) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = __float2bfloat16(in[i]); }
}

// ---- Decode attention ---------------------------------------------------------------------------
// Block (kv head, split, row): the G query heads of the KV head over the split's keys, in tiles. Four
// warps; warp w scores heads [w * HPW, (w + 1) * HPW) over keys of its key group (G < 4: the tile's
// keys are split between 4 / G warps per head). Every thread then accumulates D / 128 dimensions of
// every head's output.
constexpr int kDecodeThreads = 128;

// The sums of 16 per-lane values v[k] over the warp, by halving exchanges: afterwards lanes 2j and
// 2j + 1 both hold the total of key (lane & 16 ? 8 : 0) + (lane & 8 ? 4 : 0) + (lane & 4 ? 2 : 0) +
// (lane & 2 ? 1 : 0).
__device__ __forceinline__ float key_reduce16(float (&v)[16], int lane) {
#pragma unroll
    for (int width = 8, offset = 16; width >= 1; width >>= 1, offset >>= 1) {
        const bool upper = (lane & offset) != 0;
#pragma unroll
        for (int i = 0; i < width; ++i) {
            const float send = upper ? v[i] : v[i + width];
            const float keep = upper ? v[i + width] : v[i];
            v[i]             = keep + __shfl_xor_sync(0xFFFFFFFFu, send, offset);
        }
    }
    return v[0] + __shfl_xor_sync(0xFFFFFFFFu, v[0], 1);
}

// The splits a row's keys actually use: at least a tile of keys each, so a short context does not
// pay for empty blocks and their partial rows. The attention and its combine agree on it.
__device__ __forceinline__ int decode_active_splits(int keys, int splits, int tile) {
    return min(splits, max(1, (keys + tile - 1) / tile));
}

template <int D, int G>
struct DecodeShape {
    static constexpr int kTile      = D == 512 ? 16 : 32;
    static constexpr int kHpw       = G >= 4 ? G / 4 : 1;   // heads per warp
    static constexpr int kGroups    = G >= 4 ? 1 : 4 / G;   // key groups per head
    static constexpr int kDpl       = D / 32;               // dims per lane
    static constexpr int kDpt       = D / kDecodeThreads;   // dims per thread (PV)
};

template <int D, int G>
__launch_bounds__(kDecodeThreads) __global__ void decode_attention_kernel(DecodeAttentionArgs a) {
    using S              = DecodeShape<D, G>;
    constexpr int kTile  = S::kTile;
    const int kv_head    = blockIdx.x;
    const int split      = blockIdx.y;
    const int b          = blockIdx.z;
    const int warp       = threadIdx.x >> 5;
    const int lane       = threadIdx.x & 31;
    const int position   = a.positions[b];
    const int sequence   = a.sequences[b];
    const int lo_all     = a.ring != 0 ? max(0, position - a.window + 1) : 0;
    const int n_all      = position + 1 - lo_all;
    const int active     = decode_active_splits(n_all, a.splits, kTile);
    if (split >= active) { return; }
    const int chunk      = ((n_all + active - 1) / active + kTile - 1) / kTile * kTile;
    const int lo         = lo_all + split * chunk;
    const int hi         = min(position + 1, lo + chunk);

    __shared__ __align__(16) __half k_s[kTile][D];
    __shared__ __align__(16) __half v_s[kTile][D];
    __shared__ float p_s[G][kTile];
    __shared__ float m_s[G], l_s[G], scale_s[G];

    const int q_head0 = kv_head * G;
    if (threadIdx.x < G) {
        m_s[threadIdx.x] = -INFINITY;
        l_s[threadIdx.x] = 0.0f;
    }
    float acc[G][S::kDpt];
#pragma unroll
    for (int g = 0; g < G; ++g) {
#pragma unroll
        for (int j = 0; j < S::kDpt; ++j) { acc[g][j] = 0.0f; }
    }
    __syncthreads();

    // This lane's slice of each of its heads' queries.
    const int head_base = (warp / S::kGroups) * S::kHpw;
    const int key_group = warp % S::kGroups;
    float q_r[S::kHpw][S::kDpl];
#pragma unroll
    for (int h = 0; h < S::kHpw; ++h) {
        const float* q = a.q + (std::int64_t(b) * a.query_heads + q_head0 + head_base + h) * D +
                         lane * S::kDpl;
#pragma unroll
        for (int j = 0; j < S::kDpl; ++j) { q_r[h][j] = q[j] * a.scale; }
    }

    const std::int64_t base = std::int64_t(sequence) * a.seq_stride +
                              std::int64_t(kv_head) * a.head_stride;
    // The K and V rows of a tile in 16-byte words, kLoads per thread; the next tile's words are
    // fetched into registers while the current one is scored.
    constexpr int kWords = D / 8;
    constexpr int kLoads = kTile * kWords / kDecodeThreads;
    static_assert(kTile * kWords % kDecodeThreads == 0);
    uint4 k_next[kLoads], v_next[kLoads];
    const auto fetch = [&](int tile, int count) {
#pragma unroll
        for (int u = 0; u < kLoads; ++u) {
            const int w = threadIdx.x + u * kDecodeThreads;
            const int r = w / kWords, c = w % kWords;
            k_next[u] = make_uint4(0, 0, 0, 0);
            v_next[u] = make_uint4(0, 0, 0, 0);
            if (r < count) {
                const int p    = tile + r;
                const int slot = a.ring != 0 ? p % a.ring : p;
                const std::int64_t at = base + std::int64_t(slot) * D + c * 8;
                k_next[u] = *reinterpret_cast<const uint4*>(a.k_cache + at);
                v_next[u] = *reinterpret_cast<const uint4*>(a.v_cache + at);
            }
        }
    };
    if (lo < hi) { fetch(lo, min(kTile, hi - lo)); }
    for (int tile = lo; tile < hi; tile += kTile) {
        const int count = min(kTile, hi - tile);
#pragma unroll
        for (int u = 0; u < kLoads; ++u) {
            const int w = threadIdx.x + u * kDecodeThreads;
            const int r = w / kWords, c = w % kWords;
            *reinterpret_cast<uint4*>(&k_s[r][c * 8]) = k_next[u];
            *reinterpret_cast<uint4*>(&v_s[r][c * 8]) = v_next[u];
        }
        __syncthreads();
        if (tile + kTile < hi) { fetch(tile + kTile, min(kTile, hi - tile - kTile)); }
        // Scores: each lane's partial dot products over its kDpl dimensions for the warp's 16 keys,
        // then one butterfly that leaves every key's sum on a pair of lanes (16 shuffles per head
        // instead of 16 full reductions).
        constexpr int kKeysPerGroup = kTile / S::kGroups;
        static_assert(kKeysPerGroup == 16, "the score butterfly reduces 16 keys");
        float part[S::kHpw][kKeysPerGroup];
#pragma unroll
        for (int kk = 0; kk < kKeysPerGroup; ++kk) {
            const int key = key_group * kKeysPerGroup + kk;
            float kf[S::kDpl];
#pragma unroll
            for (int j = 0; j < S::kDpl; j += 2) {
                const float2 f = __half22float2(
                    *reinterpret_cast<const __half2*>(&k_s[key][lane * S::kDpl + j]));
                kf[j]     = f.x;
                kf[j + 1] = f.y;
            }
#pragma unroll
            for (int h = 0; h < S::kHpw; ++h) {
                float dot = 0.0f;
#pragma unroll
                for (int j = 0; j < S::kDpl; ++j) { dot += q_r[h][j] * kf[j]; }
                part[h][kk] = dot;
            }
        }
        const int lane_key = key_group * kKeysPerGroup + ((lane & 16) ? 8 : 0) + ((lane & 8) ? 4 : 0) +
                             ((lane & 4) ? 2 : 0) + ((lane & 2) ? 1 : 0);
#pragma unroll
        for (int h = 0; h < S::kHpw; ++h) {
            const float dot = key_reduce16(part[h], lane);
            if ((lane & 1) == 0) {
                p_s[head_base + h][lane_key] = lane_key < count ? dot : -INFINITY;
            }
        }
        __syncthreads();
        // Online softmax: warp w takes heads w, w + 4, ...; lane k the tile's key k.
        static_assert(kTile <= 32, "one lane per key of a tile");
        for (int g = warp; g < G; g += kDecodeThreads / 32) {
            const float p     = lane < count ? p_s[g][lane] : -INFINITY;
            float m_tile      = p;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) { m_tile = fmaxf(m_tile, __shfl_xor_sync(0xFFFFFFFFu, m_tile, o)); }
            const float m_old = m_s[g];
            const float m_new = fmaxf(m_old, m_tile);
            const float scale = expf(m_old - m_new);
            const float e     = lane < count ? expf(p - m_new) : 0.0f;
            if (lane < kTile) { p_s[g][lane] = e; }
            const float sum = warp_sum(e);
            if (lane == 0) {
                l_s[g]     = l_s[g] * scale + sum;
                m_s[g]     = m_new;
                scale_s[g] = scale;
            }
        }
        __syncthreads();
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float scale = scale_s[g];
#pragma unroll
            for (int j = 0; j < S::kDpt; ++j) { acc[g][j] *= scale; }
        }
        for (int k = 0; k < count; ++k) {
#pragma unroll
            for (int j = 0; j < S::kDpt; ++j) {
                const float v = __half2float(v_s[k][threadIdx.x + j * kDecodeThreads]);
#pragma unroll
                for (int g = 0; g < G; ++g) { acc[g][j] += p_s[g][k] * v; }
            }
        }
        __syncthreads();
    }
    // Partials: [B][Hq][splits] rows of D values, then [B][Hq][splits] (m, l) pairs.
    const std::int64_t rows = std::int64_t(a.batch) * a.query_heads * a.splits;
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const std::int64_t row =
            (std::int64_t(b) * a.query_heads + q_head0 + g) * a.splits + split;
#pragma unroll
        for (int j = 0; j < S::kDpt; ++j) {
            a.partial[row * D + threadIdx.x + j * kDecodeThreads] = acc[g][j];
        }
        if (threadIdx.x == 0) {
            a.partial[rows * D + 2 * row]     = m_s[g];
            a.partial[rows * D + 2 * row + 1] = l_s[g];
        }
    }
}

// Block (head, row): merges the splits of one query head.
constexpr int kDecodeCombineThreads = 256; // at least the largest split count

template <int D>
__global__ void decode_combine_kernel(DecodeAttentionArgs a) {
    const int head = blockIdx.x, b = blockIdx.y;
    const std::int64_t rows  = std::int64_t(a.batch) * a.query_heads * a.splits;
    const std::int64_t first = (std::int64_t(b) * a.query_heads + head) * a.splits;
    const int position = a.positions[b];
    const int lo_all   = a.ring != 0 ? max(0, position - a.window + 1) : 0;
    const int active   = decode_active_splits(position + 1 - lo_all, a.splits, D == 512 ? 16 : 32);
    // Each split's weight exp(m_s - m) once, in shared memory (blockDim >= splits).
    __shared__ float weight[kDecodeCombineThreads];
    __shared__ float max_s, inv_l;
    const int s0   = threadIdx.x;
    const float ms = s0 < active ? a.partial[rows * D + 2 * (first + s0)] : -INFINITY;
    float m        = ms;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o)); }
    if ((threadIdx.x & 31) == 0) { weight[threadIdx.x >> 5] = m; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float all = -INFINITY;
        for (int w = 0; w < int(blockDim.x >> 5); ++w) { all = fmaxf(all, weight[w]); }
        max_s = all;
    }
    __syncthreads();
    const float w_s = ms != -INFINITY ? expf(ms - max_s) : 0.0f;
    const float l_s = w_s != 0.0f ? a.partial[rows * D + 2 * (first + s0) + 1] * w_s : 0.0f;
    const float l   = block_sum(l_s);
    weight[threadIdx.x] = w_s;
    if (threadIdx.x == 0) { inv_l = 1.0f / l; }
    __syncthreads();
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float o = 0.0f;
        for (int s = 0; s < active; ++s) {
            const float w = weight[s];
            if (w != 0.0f) { o += a.partial[(first + s) * D + d] * w; }
        }
        const float y = o * inv_l;
        if (a.out_q.qs == nullptr) {
            a.out[(std::int64_t(b) * a.query_heads + head) * D + d] = __float2bfloat16(y);
            continue;
        }
        // One warp holds 32 consecutive values: one q8_1 block of the o-projection input.
        const int width = a.query_heads * D, at = head * D + d;
        float amax = fabsf(y), sum = y;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
            sum += __shfl_xor_sync(0xFFFFFFFFu, sum, off);
        }
        const float dq = amax / 127.0f;
        a.out_q.qs[std::int64_t(b) * width + at] =
            amax == 0.0f ? 0 : static_cast<std::int8_t>(roundf(y / dq));
        if ((threadIdx.x & 31) == 0) {
            a.out_q.ds[std::int64_t(b) * (width / 32) + at / 32] =
                make_half2(__float2half(dq), __float2half(sum));
        }
    }
}

// h += rms_norm(o) * w_post; then the FFN inputs of the new h (see AttnResidualArgs).
__global__ void attn_residual_kernel(AttnResidualArgs a) {
    const int q = a.hidden >> 2, i = threadIdx.x;
    const std::int64_t row = std::int64_t(blockIdx.x) * a.hidden;
    const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float4 o    = i < q ? load_bf16x4(a.o + row + 4 * i) : zero;
    const float inv_o = rsqrtf(block_sum(dot4(o, o)) / float(a.hidden) + a.eps);
    float4 h = zero;
    if (i < q) {
        h = add4(ld4(a.h + row, i), mul4(scale4(o, inv_o), ld4(a.w_post, i)));
        reinterpret_cast<float4*>(a.h + row)[i] = h;
    }
    const float inv = rsqrtf(block_sum(dot4(h, h)) / float(a.hidden) + a.eps);
    const bool valid = i < q;
    const float4 n   = scale4(h, inv);
    const float4 mlp = valid ? mul4(n, ld4(a.ffn_norm, i)) : zero;
    if (valid && a.mlp_in != nullptr) { store_bf16x4(a.mlp_in + row + 4 * i, mlp); }
    if (a.mlp_q.qs != nullptr) { store_q8x4(a.mlp_q, a.hidden, blockIdx.x, i, mlp, valid); }
    if (a.pre_norm != nullptr) {
        const float4 moe = valid ? mul4(n, ld4(a.pre_norm, i)) : zero;
        if (valid && a.moe_in != nullptr) { store_bf16x4(a.moe_in + row + 4 * i, moe); }
        if (a.moe_q.qs != nullptr) { store_q8x4(a.moe_q, a.hidden, blockIdx.x, i, moe, valid); }
    }
    if (valid && a.router_x != nullptr) {
        reinterpret_cast<float4*>(a.router_x + row)[i] =
            scale4(mul4(n, ld4(a.router_scale, i)), rsqrtf(float(a.hidden)));
    }
}

// Router logits of a prompt chunk, a tiled FP32 product: block (expert tile, token tile) computes
// kRouteTileT tokens x kRouteTileE experts over K steps of kRouteTileK staged in shared memory;
// thread (row group ty, expert tx) keeps kRouteTileT / 8 token sums of one expert.
constexpr int kRouteTileT = 64, kRouteTileE = 32, kRouteTileK = 32;
constexpr int kRouteTokens = 8; // decode: the small kernel below covers up to this many tokens

__global__ void route_logits_kernel(const float* __restrict__ x, int hidden, int tokens,
                                    const float* __restrict__ router, int experts,
                                    float* __restrict__ logits) {
    __shared__ float xs[kRouteTileT][kRouteTileK + 1];
    __shared__ float rs[kRouteTileE][kRouteTileK + 1];
    const int tx = threadIdx.x & 31, ty = threadIdx.x >> 5; // 8 warps
    const int e0 = blockIdx.x * kRouteTileE, t0 = blockIdx.y * kRouteTileT;
    constexpr int kPer = kRouteTileT / 8;
    // Routing turns on near ties, so each K tile's partial sum enters a compensated total: the
    // logits keep the accuracy of the decode kernel's lane-split sums.
    float acc[kPer] = {}, err[kPer] = {};
    for (int k0 = 0; k0 < hidden; k0 += kRouteTileK) {
        for (int i = threadIdx.x; i < kRouteTileT * kRouteTileK; i += 256) {
            const int r = i / kRouteTileK, c = i % kRouteTileK;
            const int t = t0 + r, kk = k0 + c;
            xs[r][c] = t < tokens && kk < hidden ? x[std::int64_t(t) * hidden + kk] : 0.0f;
        }
        for (int i = threadIdx.x; i < kRouteTileE * kRouteTileK; i += 256) {
            const int r = i / kRouteTileK, c = i % kRouteTileK;
            const int e = e0 + r, kk = k0 + c;
            rs[r][c] = e < experts && kk < hidden ? router[std::int64_t(e) * hidden + kk] : 0.0f;
        }
        __syncthreads();
        float part[kPer] = {};
#pragma unroll 8
        for (int c = 0; c < kRouteTileK; ++c) {
            const float w = rs[tx][c];
#pragma unroll
            for (int i = 0; i < kPer; ++i) { part[i] = fmaf(xs[ty * kPer + i][c], w, part[i]); }
        }
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const float y = part[i] - err[i];
            const float t = acc[i] + y;
            err[i]        = (t - acc[i]) - y;
            acc[i]        = t;
        }
        __syncthreads();
    }
    const int e = e0 + tx;
    if (e >= experts) { return; }
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int t = t0 + ty * kPer + i;
        if (t < tokens) { logits[std::int64_t(t) * experts + e] = acc[i]; }
    }
}

// Router logits for up to kRouteTokens tokens (decode): one block per expert, its row split over
// four warps, so every SM streams part of the router.
constexpr int kRouteSmallThreads = 256;

__global__ void route_logits_small_kernel(const float* __restrict__ x, int hidden, int tokens,
                                          const float* __restrict__ router, int experts,
                                          float* __restrict__ logits) {
    __shared__ float part[kRouteSmallThreads / 32][kRouteTokens];
    const int e = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float4* w = reinterpret_cast<const float4*>(router + std::int64_t(e) * hidden);
    const int quads = hidden / 4;
    float dot[kRouteTokens] = {};
    for (int i = threadIdx.x; i < quads; i += kRouteSmallThreads) {
        const float4 wv = w[i];
#pragma unroll
        for (int k = 0; k < kRouteTokens; ++k) {
            if (k < tokens) {
                dot[k] += dot4(wv, reinterpret_cast<const float4*>(x)[std::int64_t(k) * quads + i]);
            }
        }
    }
#pragma unroll
    for (int k = 0; k < kRouteTokens; ++k) {
        const float v = warp_sum(dot[k]);
        if (lane == 0) { part[warp][k] = v; }
    }
    __syncthreads();
    if (threadIdx.x < tokens) {
        float v = 0.0f;
#pragma unroll
        for (int w4 = 0; w4 < kRouteSmallThreads / 32; ++w4) { v += part[w4][threadIdx.x]; }
        logits[std::int64_t(threadIdx.x) * experts + e] = v;
    }
}

// One block per token, one thread per expert: each expert's rank among the logits (larger first,
// lower id on ties) selects the top_k; their softmax times expert_scale.
constexpr int kTopkMaxExperts = 256;

__global__ void route_topk_kernel(const float* __restrict__ logits, int experts, int top_k,
                                  const float* __restrict__ expert_scale,
                                  std::int32_t* __restrict__ ids, float* __restrict__ weights) {
    __shared__ float v[kTopkMaxExperts];
    __shared__ float chosen[8];
    __shared__ int chosen_id[8];
    const int t = blockIdx.x, e = threadIdx.x;
    if (e < experts) { v[e] = logits[std::int64_t(t) * experts + e]; }
    __syncthreads();
    if (e < experts) {
        const float mine = v[e];
        int rank         = 0;
        for (int j = 0; j < experts; ++j) {
            const float other = v[j];
            rank += (other > mine || (other == mine && j < e)) ? 1 : 0;
        }
        if (rank < top_k) {
            chosen[rank]    = mine;
            chosen_id[rank] = e;
        }
    }
    __syncthreads();
    if (e < top_k) {
        float sum = 0.0f;
        for (int k = 0; k < top_k; ++k) { sum += expf(chosen[k] - chosen[0]); }
        ids[std::int64_t(t) * top_k + e]     = chosen_id[e];
        weights[std::int64_t(t) * top_k + e] =
            expf(chosen[e] - chosen[0]) / sum * expert_scale[chosen_id[e]];
    }
}

// Decode routing after the logits, in one block of at least max(experts, tokens * top_k)
// threads: each token's top_k as route_topk_kernel picks them, then the pairs grouped by expert as
// moe_sort_routes lists them (bounds, pairs per expert, active experts ascending).
__global__ void route_select_kernel(const float* __restrict__ logits, int tokens, int experts,
                                    int top_k, const float* __restrict__ expert_scale,
                                    std::int32_t* __restrict__ ids, float* __restrict__ weights,
                                    std::int32_t* __restrict__ bounds,
                                    std::int32_t* __restrict__ sorted,
                                    std::int32_t* __restrict__ active,
                                    std::int32_t* __restrict__ active_count) {
    __shared__ float v[kTopkMaxExperts];
    __shared__ float chosen[kRouteTokens * 8];
    __shared__ int chosen_id[kRouteTokens * 8];
    __shared__ int cursor[kTopkMaxExperts];
    __shared__ int warp_total[kTopkMaxExperts / 32][2];
    const int e = threadIdx.x, lane = e & 31, warp = e >> 5;
    for (int t = 0; t < tokens; ++t) {
        if (e < experts) { v[e] = logits[std::int64_t(t) * experts + e]; }
        __syncthreads();
        if (e < experts) {
            const float mine = v[e];
            int rank         = 0;
            for (int j = 0; j < experts; ++j) {
                const float other = v[j];
                rank += (other > mine || (other == mine && j < e)) ? 1 : 0;
            }
            if (rank < top_k) {
                chosen[t * top_k + rank]    = mine;
                chosen_id[t * top_k + rank] = e;
            }
        }
        __syncthreads();
    }
    const int pairs = tokens * top_k;
    if (e < pairs) {
        const int base = (e / top_k) * top_k;
        float sum      = 0.0f;
        for (int k = 0; k < top_k; ++k) { sum += expf(chosen[base + k] - chosen[base]); }
        ids[e]     = chosen_id[e];
        weights[e] = expf(chosen[e] - chosen[base]) / sum * expert_scale[chosen_id[e]];
    }
    // Exclusive scans over the experts of their pair counts and of whether they have any.
    int count = 0;
    if (e < experts) {
        for (int p = 0; p < pairs; ++p) { count += chosen_id[p] == e ? 1 : 0; }
    }
    int c = count, f = count != 0 ? 1 : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int uc = __shfl_up_sync(0xFFFFFFFFu, c, o), uf = __shfl_up_sync(0xFFFFFFFFu, f, o);
        if (lane >= o) {
            c += uc;
            f += uf;
        }
    }
    if (lane == 31) {
        warp_total[warp][0] = c;
        warp_total[warp][1] = f;
    }
    __syncthreads();
    int before_c = 0, before_f = 0;
    for (int w = 0; w < warp; ++w) {
        before_c += warp_total[w][0];
        before_f += warp_total[w][1];
    }
    const int start = before_c + c - count, slot = before_f + f - (count != 0 ? 1 : 0);
    if (e < experts) {
        bounds[e] = start;
        cursor[e] = start;
        if (count != 0) { active[slot] = e; }
        if (e == experts - 1) {
            bounds[experts] = start + count;
            *active_count   = slot + (count != 0 ? 1 : 0);
        }
    }
    __syncthreads();
    if (e < pairs) { sorted[atomicAdd(cursor + chosen_id[e], 1)] = e; }
}

__global__ void geglu_rows_kernel(const float* __restrict__ x, int pairs, int width,
                                  __nv_bfloat16* __restrict__ out) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::int64_t(pairs) * width) { return; }
    const std::int64_t p = i / width, r = i % width;
    const float* row     = x + p * 2 * width;
    out[i]               = __float2bfloat16(gelu_tanh(row[r]) * row[width + r]);
}

__global__ void ffn_combine_kernel(FfnCombineArgs a) {
    const int q = a.hidden >> 2, i = threadIdx.x;
    const std::int64_t row = std::int64_t(blockIdx.x) * a.hidden;
    const float hidden = float(a.hidden);
    const float4 zero  = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 y = i < q ? load_bf16x4(a.dense + row + 4 * i) : zero;
    const bool moe = a.moe_rows != nullptr || a.moe_fixed != nullptr;
    float4 m       = zero;
    if (moe && i < q) {
        if (a.moe_rows != nullptr) {
            for (int k = 0; k < a.top_k; ++k) {
                const std::int64_t pair = std::int64_t(blockIdx.x) * a.top_k + k;
                m = add4(m, scale4(ld4(a.moe_rows + pair * a.hidden, i), a.moe_weights[pair]));
            }
        } else {
            m = fixed4(a.moe_fixed + row, i);
        }
    }
    if (a.w_dense != nullptr || moe) {
        // The dense and expert norms in one reduction.
        const float2 ss = block_sum2(dot4(y, y), dot4(m, m));
        if (i < q) {
            if (a.w_dense != nullptr) { y = mul4(scale4(y, rsqrtf(ss.x / hidden + a.eps)), ld4(a.w_dense, i)); }
            if (moe) { y = add4(y, mul4(scale4(m, rsqrtf(ss.y / hidden + a.eps)), ld4(a.w_moe, i))); }
        }
    }
    const float inv_y = rsqrtf(block_sum(dot4(y, y)) / hidden + a.eps);
    float4 h = zero;
    if (i < q) {
        h = scale4(add4(ld4(a.h + row, i), mul4(scale4(y, inv_y), ld4(a.w_post, i))), a.scalar[0]);
        reinterpret_cast<float4*>(a.h + row)[i] = h;
    }
    if (a.next_norm == nullptr) { return; }
    const float inv  = rsqrtf(block_sum(dot4(h, h)) / hidden + a.eps);
    const float4 nxt = i < q ? mul4(scale4(h, inv), ld4(a.next_norm, i)) : zero;
    if (i < q && a.next_out != nullptr) { store_bf16x4(a.next_out + row + 4 * i, nxt); }
    if (a.next_q.qs != nullptr) { store_q8x4(a.next_q, a.hidden, blockIdx.x, i, nxt, i < q); }
}

__global__ void softcap_kernel(const float* x, std::int64_t n, float cap, __nv_bfloat16* out) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    const float v = cap > 0.0f ? cap * tanhf(x[i] / cap) : x[i];
    out[i]        = __float2bfloat16(v);
}

unsigned blocks_for(std::int64_t n, int threads) {
    return static_cast<unsigned>((n + threads - 1) / threads);
}

} // namespace

void scale(float* x, std::int64_t n, float s, cudaStream_t stream) {
    if (n <= 0) { return; }
    scale_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(x, n, s);
    check_launch("scale");
}

int row_threads(int hidden) {
    if (hidden % 4 != 0 || hidden / 4 > 1024) {
        throw std::invalid_argument("gemma4 row kernels: hidden must be whole float4s, at most 4096");
    }
    return (hidden / 4 + 31) / 32 * 32;
}

void rms_norm(const float* x, const float* w, int hidden, int tokens, float eps,
              __nv_bfloat16* out, cudaStream_t stream, Q8Rows q) {
    rms_norm_kernel<<<tokens, row_threads(hidden), 0, stream>>>(x, w, hidden, eps, out, q);
    check_launch("rms_norm");
}

void qkv_post(const QkvArgs& args, cudaStream_t stream) {
    if (args.head_dim != 256 && args.head_dim != 512) {
        throw std::invalid_argument("gemma4 qkv_post: head width must be 256 or 512");
    }
    const dim3 grid(args.tokens, args.query_heads + args.kv_heads, 1);
    qkv_post_kernel<<<grid, args.head_dim / 2, 0, stream>>>(args);
    check_launch("qkv_post");
}

void build_mask(__half* mask, int tokens, int slots, int first, int ring, int window,
                cudaStream_t stream) {
    const dim3 grid(blocks_for(slots, 256), tokens, 1);
    build_mask_kernel<<<grid, 256, 0, stream>>>(mask, tokens, slots, first, ring, window);
    check_launch("build_mask");
}

void ring_copy(const RingCopyArgs& a, cudaStream_t stream) {
    if (a.end <= a.begin) { return; }
    if (a.head_dim % 8 != 0) { throw std::invalid_argument("gemma4 ring_copy: head width"); }
    ring_copy_kernel<<<dim3(a.end - a.begin, a.kv_heads, 1), 64, 0, stream>>>(a);
    check_launch("ring_copy");
}

void f32_to_bf16(const float* in, __nv_bfloat16* out, std::int64_t n, cudaStream_t stream) {
    if (n <= 0) { return; }
    f32_to_bf16_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(in, out, n);
    check_launch("f32_to_bf16");
}

std::size_t decode_attention_workspace_floats(int batch, int query_heads, int head_dim,
                                              int splits) {
    const std::size_t rows = std::size_t(batch) * query_heads * splits;
    return rows * head_dim + 2 * rows;
}

void decode_attention(const DecodeAttentionArgs& args, cudaStream_t stream) {
    const int group = args.query_heads / args.kv_heads;
    if (args.splits > kDecodeCombineThreads) {
        throw std::invalid_argument("gemma4 decode attention: too many splits");
    }
    const dim3 grid(args.kv_heads, args.splits, args.batch);
    if (args.head_dim == 256 && group == 2) {
        decode_attention_kernel<256, 2><<<grid, kDecodeThreads, 0, stream>>>(args);
        check_launch("decode_attention");
        decode_combine_kernel<256><<<dim3(args.query_heads, args.batch), kDecodeCombineThreads, 0, stream>>>(args);
    } else if (args.head_dim == 512 && group == 8) {
        decode_attention_kernel<512, 8><<<grid, kDecodeThreads, 0, stream>>>(args);
        check_launch("decode_attention");
        decode_combine_kernel<512><<<dim3(args.query_heads, args.batch), kDecodeCombineThreads, 0, stream>>>(args);
    } else {
        throw std::invalid_argument("gemma4 decode attention: unsupported head geometry");
    }
    check_launch("decode_combine");
}

void attn_residual(const AttnResidualArgs& args, cudaStream_t stream) {
    attn_residual_kernel<<<args.tokens, row_threads(args.hidden), 0, stream>>>(args);
    check_launch("attn_residual");
}

void route(const float* x, int hidden, int tokens, const float* router, const float* expert_scale,
           int experts, int top_k, float* logits, std::int32_t* ids, float* weights,
           cudaStream_t stream) {
    if (top_k > 8 || top_k > experts || experts > kTopkMaxExperts || hidden % 4 != 0) {
        throw std::invalid_argument("gemma4 route: at most 8 of 256 experts over whole float4s");
    }
    if (tokens <= kRouteTokens) {
        route_logits_small_kernel<<<experts, kRouteSmallThreads, 0, stream>>>(x, hidden, tokens,
                                                                             router, experts, logits);
    } else {
        const dim3 grid(blocks_for(experts, kRouteTileE), blocks_for(tokens, kRouteTileT), 1);
        route_logits_kernel<<<grid, 256, 0, stream>>>(x, hidden, tokens, router, experts, logits);
    }
    check_launch("route logits");
    route_topk_kernel<<<tokens, (experts + 31) / 32 * 32, 0, stream>>>(logits, experts, top_k,
                                                                       expert_scale, ids, weights);
    check_launch("route top-k");
}

void route_select(const float* x, int hidden, int tokens, const float* router,
                  const float* expert_scale, int experts, int top_k, float* logits,
                  std::int32_t* ids, float* weights, std::int32_t* bounds, std::int32_t* sorted,
                  std::int32_t* active, std::int32_t* active_count, cudaStream_t stream) {
    if (tokens > kRouteTokens || top_k > 8 || top_k > experts || experts > kTopkMaxExperts ||
        hidden % 4 != 0) {
        throw std::invalid_argument("gemma4 route_select: at most 8 tokens, 8 of 256 experts");
    }
    route_logits_small_kernel<<<experts, kRouteSmallThreads, 0, stream>>>(x, hidden, tokens, router,
                                                                         experts, logits);
    check_launch("route logits");
    const int threads = (std::max(experts, tokens * top_k) + 31) / 32 * 32;
    route_select_kernel<<<1, threads, 0, stream>>>(logits, tokens, experts, top_k, expert_scale, ids,
                                                   weights, bounds, sorted, active, active_count);
    check_launch("route select");
}

void geglu_rows(const float* x, int pairs, int width, __nv_bfloat16* out, cudaStream_t stream) {
    const std::int64_t n = std::int64_t(pairs) * width;
    if (n <= 0) { return; }
    geglu_rows_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(x, pairs, width, out);
    check_launch("geglu_rows");
}

void ffn_combine(const FfnCombineArgs& args, cudaStream_t stream) {
    ffn_combine_kernel<<<args.tokens, row_threads(args.hidden), 0, stream>>>(args);
    check_launch("ffn_combine");
}

void softcap(const float* x, std::int64_t n, float cap, __nv_bfloat16* out, cudaStream_t stream) {
    if (n <= 0) { return; }
    softcap_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(x, n, cap, out);
    check_launch("softcap");
}

} // namespace ninfer::models::gemma4::kernels
