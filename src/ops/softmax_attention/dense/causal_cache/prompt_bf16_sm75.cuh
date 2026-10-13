#pragma once

// BF16-cache causal prompt kernel of a Turing (sm_75) build, included by prompt_bf16.cuh after its
// staging helper; the sm_80+ kernel there keeps this one's contract, tiles and grid. Turing has 64
// KiB of shared memory per block and no cp.async: one buffer holds K(kb), then V(kb), beside Q, and
// the kernel takes (Br + Bc) * D * 2 bytes, exactly 64 KiB. It has no BF16 MMA either, so a CTA
// sweeps its rows on FP16 MMA with Q, K and V as their FP16 roundings
// (common/bf16_attention.cuh) and sweeps them again on the BF16 emulation, an out-of-line function,
// only when a tile holds a value FP16 cannot hold: in one loop body with the fast path, the
// emulation's code cost the fast path most of its speed. Nothing is written before a sweep ends.

#include <math_constants.h>

#include "ops/common/device_limits.h"
#include "ops/softmax_attention/common/bf16_attention.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"

namespace ninfer::ops {

inline constexpr int kCausalPromptBf16SmemBytes =
    (kCausalPromptBr + kCausalPromptBc) * kCausalPromptHeadDim *
    static_cast<int>(sizeof(__nv_bfloat16));
static_assert(kCausalPromptBf16SmemBytes <= kMaxSharedBytesPerBlock);

namespace sm75_prompt_bf16 {

// causal_prompt_stage_kv as FP16 roundings; false when one of this thread's values is beyond FP16's
// range.
template <typename Geometry>
__device__ __forceinline__ bool stage_kv_half(__nv_bfloat16* dst, const __nv_bfloat16* cache,
                                              int kv_head, int k0, int max_query_abs,
                                              int physical_page, int tid) {
    constexpr int D         = kCausalPromptHeadDim;
    constexpr int Bc        = kCausalPromptBc;
    constexpr int Threads   = kCausalPromptThreads;
    constexpr int VecPerRow = D / 8;
    const __nv_bfloat16* cache_block =
        cache + paged_kv_element_offset<kCausalPromptHeadDim, Geometry::KVHeads>(
                    physical_page, kv_head, k0 & kPagedKVPageMask, 0);
    bool fits = true;
#pragma unroll
    for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
        const int key_l  = chunk >> 5;
        const int d      = (chunk & 31) << 3;
        __nv_bfloat16* p = &dst[key_l * D + causal_prompt_swz(key_l, d)];
        if ((k0 + key_l) <= max_query_abs) {
            fits = copy_bf16x8_as_half(p, &cache_block[key_l * D + d]) && fits;
        } else {
            store_vec(p, make_int4(0, 0, 0, 0));
        }
    }
    return fits;
}

// The CTA's Q rows into q_s, as FP16 roundings (Half; the return says whether every value the
// thread copied fits FP16) or as BF16.
template <typename Geometry, bool Half>
__device__ __forceinline__ bool stage_q(__nv_bfloat16* q_s, const __nv_bfloat16* q, int q_head,
                                        int q0, int tokens, int tid) {
    constexpr int D          = kCausalPromptHeadDim;
    constexpr int Br         = kCausalPromptBr;
    constexpr int Threads    = kCausalPromptThreads;
    constexpr int VecPerRow  = D / 8;
    constexpr int QRowStride = D * Geometry::QHeads;
    const __nv_bfloat16* q_block = q + causal_prompt_q_index<Geometry>(q_head, 0, q0);
    bool fits                    = true;
#pragma unroll
    for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
        const int row    = chunk >> 5;
        const int d      = (chunk & 31) << 3;
        __nv_bfloat16* p = &q_s[row * D + causal_prompt_swz(row, d)];
        if (q0 + row < tokens) {
            if constexpr (Half) {
                fits = copy_bf16x8_as_half(p, &q_block[row * QRowStride + d]) && fits;
            } else {
                cp_async<16, Cache::cg>(p, &q_block[row * QRowStride + d]);
            }
        } else {
            store_vec(p, make_int4(0, 0, 0, 0));
        }
    }
    return fits;
}

// One sweep of the CTA's rows over its keys, ending with their outputs, with Q already in q_s.
// Fast: K and V tiles as FP16 roundings on FP16 MMA; false, with nothing written, once a tile holds
// a value FP16 cannot hold. Exact: tiles as BF16 on mma_bf16_rounded and mma_f16_bf16_rounded,
// whose products equal the fast sweep's for every operand FP16 holds.
template <typename Geometry, bool Exact>
__device__ __forceinline__ bool sweep(__nv_bfloat16* q_s, const __nv_bfloat16* __restrict__ cache_k,
                                      const __nv_bfloat16* __restrict__ cache_v,
                                      const std::int32_t* block_table, int base_pos, float scale,
                                      __nv_bfloat16* __restrict__ out, int q_head, int q0,
                                      int tokens, int width) {
    constexpr int D             = kCausalPromptHeadDim;
    constexpr int Br            = kCausalPromptBr;
    constexpr int Bc            = kCausalPromptBc;
    constexpr int Threads       = kCausalPromptThreads;
    constexpr int QKNt          = Bc / 8;
    constexpr int QKKs          = D / 16;
    constexpr int PVNt          = D / 8;
    constexpr int PVKs          = Bc / 16;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    __nv_bfloat16* kv_s = q_s + Br * D; // K(kb), then V(kb)
    const int tid       = static_cast<int>(threadIdx.x);
    const int warp      = tid >> 5;
    const int lane      = tid & 31;
    const int kv_head   = q_head / Geometry::GroupSize;
    const int gid       = lane >> 2;
    const int lid       = lane & 3;
    const int a_mat     = lane >> 3;
    const int a_rin     = lane & 7;
    const int a_rowoff  = a_rin + ((a_mat & 1) << 3);
    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const int warp_row0 = warp * 16;

    // Per-lane swizzled ldmatrix bases (causal_prompt_swz_addr), as the sm_80+ kernel's.
    const unsigned q_lane_base =
        smem_addr(q_s) + static_cast<unsigned>((warp_row0 + a_rowoff) * 512);
    const unsigned q_as = static_cast<unsigned>((a_mat >> 1) << 4);
    const unsigned q_r  = static_cast<unsigned>(a_rin << 4);
    const unsigned k_lane_base = smem_addr(kv_s) + static_cast<unsigned>(b_rin * 512) +
                                 (static_cast<unsigned>(lane >> 4) << 12);
    const unsigned k_as = static_cast<unsigned>((b_koff >> 3) << 4);
    const unsigned k_r  = static_cast<unsigned>(b_rin << 4);
    const unsigned v_lane_base = smem_addr(kv_s) + static_cast<unsigned>(((lane >> 3) & 1) * 4096) +
                                 static_cast<unsigned>(b_rin * 512);
    const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r  = static_cast<unsigned>(b_rin << 4);

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int n_block_max   = (max_query_abs / Bc) + 1;
    const float scale_l2    = scale * Log2E;
    int physical_page       = block_table[0];

    // Stages one K or V tile into kv_s and publishes it; false (fast sweep) when it does not fit.
    const auto stage = [&](const __nv_bfloat16* cache, int k0) -> bool {
        if constexpr (Exact) {
            causal_prompt_stage_kv<Geometry>(kv_s, cache, kv_head, k0, max_query_abs,
                                             physical_page, tid);
            cp_wait<0>();
            __syncthreads();
            return true;
        } else {
            const bool fits = stage_kv_half<Geometry>(kv_s, cache, kv_head, k0, max_query_abs,
                                                      physical_page, tid);
            return __syncthreads_and(fits ? 1 : 0) != 0;
        }
    };

    for (int kb = 0; kb < n_block_max; ++kb) {
        const int k0 = kb * Bc;
        if (kb != 0) {
            physical_page = block_table[kb];
            __syncthreads(); // every warp's PV is done with V(kb - 1)
        }
        if (!stage(cache_k, k0)) { return false; }

        // S = Q Kᵀ for this warp's 16 rows over all Bc keys, ldmatrix of step k + 1 issued while
        // the MMAs of step k run.
        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
        unsigned af[2][4];
        unsigned bf[2][QKNt][2];
        ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                    causal_prompt_swz_addr(q_lane_base, 0u, q_as, q_r));
#pragma unroll
        for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
            ldmatrix_x4(bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                        causal_prompt_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 4096), 0u,
                                               k_as, k_r));
        }
#pragma unroll
        for (int k = 0; k < QKKs; ++k) {
            const int cur = k & 1;
            const int nxt = cur ^ 1;
            if (k + 1 < QKKs) {
                const unsigned ck = static_cast<unsigned>((k + 1) << 5);
                ldmatrix_x4(af[nxt][0], af[nxt][1], af[nxt][2], af[nxt][3],
                            causal_prompt_swz_addr(q_lane_base, ck, q_as, q_r));
#pragma unroll
                for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                    ldmatrix_x4(
                        bf[nxt][nt2][0], bf[nxt][nt2][1], bf[nxt][nt2 + 1][0], bf[nxt][nt2 + 1][1],
                        causal_prompt_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 4096), ck,
                                               k_as, k_r));
                }
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                if constexpr (Exact) {
                    mma_bf16_rounded(score[nt][0], score[nt][1], score[nt][2], score[nt][3],
                                     af[cur][0], af[cur][1], af[cur][2], af[cur][3],
                                     bf[cur][nt][0], bf[cur][nt][1]);
                } else {
                    mma_f16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                            af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
                }
            }
        }

        const int row0             = warp_row0 + gid;
        const int row1             = warp_row0 + gid + 8;
        const int qrow0            = q0 + row0;
        const int qrow1            = q0 + row1;
        const int qabs0            = (qrow0 < tokens) ? base_pos + qrow0 : -1;
        const int qabs1            = (qrow1 < tokens) ? base_pos + qrow1 : -1;
        const bool full_score_tile = (q0 + Br <= tokens) && ((k0 + Bc - 1) <= (base_pos + q0));

        // Block row-max on raw scores; the scale folds into exp2 below.
        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
        if (full_score_tile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                score[nt][0]   = (qrow0 < tokens && key0 <= qabs0) ? score[nt][0] : -CUDART_INF_F;
                score[nt][1]   = (qrow0 < tokens && key1 <= qabs0) ? score[nt][1] : -CUDART_INF_F;
                score[nt][2]   = (qrow1 < tokens && key0 <= qabs1) ? score[nt][2] : -CUDART_INF_F;
                score[nt][3]   = (qrow1 < tokens && key1 <= qabs1) ? score[nt][3] : -CUDART_INF_F;
                bm0            = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1            = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0        = fmaxf(m0, bm0);
        const float nm1        = fmaxf(m1, bm1);
        const float nm0_scaled = nm0 * scale_l2;
        const float nm1_scaled = nm1 * scale_l2;
        const float alpha0     = exp2_approx(__fmaf_rn(m0, scale_l2, -nm0_scaled));
        const float alpha1     = exp2_approx(__fmaf_rn(m1, scale_l2, -nm1_scaled));

        // P = exp2(S - m) as the PV A fragment (FP16 on Turing), plus this lane's row sums.
        float bl0 = 0.0f, bl1 = 0.0f;
        unsigned p_frag[PVKs][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            float p00, p01, p10, p11;
            if (full_score_tile) {
                p00 = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled));
                p01 = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled));
                p10 = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled));
                p11 = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled));
            } else {
                p00 = (score[nt][0] > -CUDART_INF_F)
                          ? exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled))
                          : 0.0f;
                p01 = (score[nt][1] > -CUDART_INF_F)
                          ? exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled))
                          : 0.0f;
                p10 = (score[nt][2] > -CUDART_INF_F)
                          ? exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled))
                          : 0.0f;
                p11 = (score[nt][3] > -CUDART_INF_F)
                          ? exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled))
                          : 0.0f;
            }
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            const int slot            = (nt & 1) * 2;
            p_frag[nt >> 1][slot]     = pack_attention_p(p00, p01);
            p_frag[nt >> 1][slot + 1] = pack_attention_p(p10, p11);
        }

        l0 = __fmaf_rn(l0, alpha0, bl0);
        l1 = __fmaf_rn(l1, alpha1, bl1);
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        // V(kb) replaces K(kb) once every warp's QK is done with it.
        __syncthreads();
        if (!stage(cache_v, k0)) { return false; }

        // O += P V over the Bc keys, the transposed ldmatrix of the next V fragment pair issued
        // while the current MMAs run.
        constexpr int PVHalf  = PVNt / 2;
        constexpr int PVLoads = PVKs * PVHalf;
        unsigned vf[2][4];
        ldmatrix_x4_t(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                      causal_prompt_swz_addr(v_lane_base, 0u, v_as, v_r));
#pragma unroll
        for (int li = 0; li < PVLoads; ++li) {
            const int k   = li / PVHalf;
            const int n2  = (li % PVHalf) * 2;
            const int cur = li & 1;
            const int nxt = cur ^ 1;
            if (li + 1 < PVLoads) {
                const int k2       = (li + 1) / PVHalf;
                const int n2b      = ((li + 1) % PVHalf) * 2;
                const unsigned ckv = static_cast<unsigned>(n2b << 4);
                ldmatrix_x4_t(vf[nxt][0], vf[nxt][1], vf[nxt][2], vf[nxt][3],
                              causal_prompt_swz_addr(v_lane_base + static_cast<unsigned>(k2 * 8192),
                                                     ckv, v_as, v_r));
            }
            if constexpr (Exact) {
                mma_f16_bf16_rounded(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3],
                                     p_frag[k][0], p_frag[k][1], p_frag[k][2], p_frag[k][3],
                                     vf[cur][0], vf[cur][1]);
                mma_f16_bf16_rounded(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2],
                                     acc[n2 + 1][3], p_frag[k][0], p_frag[k][1], p_frag[k][2],
                                     p_frag[k][3], vf[cur][2], vf[cur][3]);
            } else {
                mma_f16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_frag[k][0], p_frag[k][1],
                        p_frag[k][2], p_frag[k][3], vf[cur][0], vf[cur][1]);
                mma_f16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3],
                        p_frag[k][0], p_frag[k][1], p_frag[k][2], p_frag[k][3], vf[cur][2],
                        vf[cur][3]);
            }
        }
    }

    l0 = warp_sum<4>(l0, FullMask);
    l1 = warp_sum<4>(l1, FullMask);

    // Normalize once per row via reciprocal-multiply instead of 128 IEEE divides.
    const float inv_l0 = (l0 > 0.0f) ? __frcp_rn(l0) : 0.0f;
    const float inv_l1 = (l1 > 0.0f) ? __frcp_rn(l1) : 0.0f;
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0    = n * 8 + 2 * lid;
        const int qrow0 = q0 + warp_row0 + gid;
        const int qrow1 = q0 + warp_row0 + gid + 8;
        if (qrow0 < tokens) {
            *reinterpret_cast<unsigned*>(&out[causal_prompt_q_index<Geometry>(q_head, d0, qrow0)]) =
                pack_bf16x2(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
        }
        if (qrow1 < tokens) {
            *reinterpret_cast<unsigned*>(&out[causal_prompt_q_index<Geometry>(q_head, d0, qrow1)]) =
                pack_bf16x2(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
        }
    }
    causal_prompt_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid,
                                             Threads);
    return true;
}

// The BF16 emulation's sweep, out of line: stages Q as BF16 over q_s and sweeps every key block
// again, over shared memory nothing else still reads.
template <typename Geometry>
__device__ __noinline__ void sweep_exact(__nv_bfloat16* q_s, const __nv_bfloat16* q,
                                         const __nv_bfloat16* cache_k,
                                         const __nv_bfloat16* cache_v,
                                         const std::int32_t* block_table, int base_pos,
                                         float scale, __nv_bfloat16* out, int q_head, int q0,
                                         int tokens, int width) {
    const int tid = static_cast<int>(threadIdx.x);
    __syncthreads(); // every warp is done with the fast sweep's q_s and K/V buffer
    (void)stage_q<Geometry, false>(q_s, q, q_head, q0, tokens, tid);
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    (void)sweep<Geometry, true>(q_s, cache_k, cache_v, block_table, base_pos, scale, out, q_head, q0,
                                tokens, width);
}

} // namespace sm75_prompt_bf16

// FlashAttention-2 forward, one CTA per (query 64-row block, query head). Grid is
// (ceil(tokens/64), q_heads). seqlen_q = tokens, seqlen_k = base_pos + tokens, with
// bottom-right causal alignment (query row i sees keys [0, base_pos + i]).
template <typename Geometry, typename Metadata>
__launch_bounds__(kCausalPromptThreads, 1) __global__
    void causal_attention_prompt_bf16_kernel(const __nv_bfloat16* __restrict__ q,
                                             const __nv_bfloat16* __restrict__ cache_k,
                                             const __nv_bfloat16* __restrict__ cache_v,
                                             Metadata metadata,
                                             const std::int32_t* __restrict__ positions,
                                             float scale, __nv_bfloat16* __restrict__ out,
                                             std::int32_t width) {
    constexpr int Br      = kCausalPromptBr;
    constexpr int Threads = kCausalPromptThreads;
    static_assert(Threads == 128);

    extern __shared__ __align__(16) __nv_bfloat16 causal_smem[];
    __nv_bfloat16* q_s = causal_smem; // [Br, D] swizzled, then the K/V buffer [Bc, D]

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int q0      = q_block * Br;
    const int tokens  = metadata.valid_tokens(width);

    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        causal_prompt_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid,
                                                 Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    const bool q_fits =
        sm75_prompt_bf16::stage_q<Geometry, true>(q_s, q, q_head, q0, tokens, tid);
    if (__syncthreads_and(q_fits ? 1 : 0) != 0 &&
        sm75_prompt_bf16::sweep<Geometry, false>(q_s, cache_k, cache_v, block_table, base_pos,
                                                 scale, out, q_head, q0, tokens, width)) {
        return;
    }
    sm75_prompt_bf16::sweep_exact<Geometry>(q_s, q, cache_k, cache_v, block_table, base_pos, scale,
                                            out, q_head, q0, tokens, width);
}

} // namespace ninfer::ops
