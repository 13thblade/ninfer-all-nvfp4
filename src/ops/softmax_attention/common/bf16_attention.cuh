#pragma once

// MMA operands of the BF16 attention kernels on Turing. sm_80 and newer multiply BF16 Q, K and V
// and BF16-rounded probabilities on BF16 m16n8k16. Turing has no BF16 MMA: mma_bf16's emulation
// converts every fragment and votes on its exactness at every call, and runs a product on FP16
// only where every operand is an FP16 value. Two things make that slow in attention:
//
// - Softmax probabilities span far more binades than FP16's, so as BF16 operands nearly every PV
//   tile of a long context fell back to FP32. A Turing build rounds the probabilities to FP16
//   (three more significant bits from 2^-14 to one, FP16 subnormals below, zero under 2^-24, as the
//   INT8-family kernels round them on every architecture): pack_attention_p.
// - The per-call conversions and votes cost more than the MMAs, and an exactness vote fails for a
//   whole tile over one value under 2^-17. The causal-cache kernels multiply Q, K and V as their
//   FP16 roundings instead: the BF16 value itself from 2^-17 up, an FP16 subnormal or zero below,
//   a change of at most 2^-25, so a product moves by at most 2^-25 times its other operand (the
//   INT8-family kernels hold V as FP16 too). They stage a tile as FP16 when every value is below
//   2^16 (store_bf16x8_as_half) and run it on native FP16 MMA. A tile with a larger value, which
//   FP16 cannot hold, stages as BF16 and multiplies on mma_bf16_rounded and mma_f16_bf16_rounded,
//   which take the same rounding of every value below 2^16 and keep the larger ones: the outputs
//   whose row and column fit run on the same FP16 MMA, the others in FP32. An output's bits
//   therefore depend on its own row and column alone, whichever way its tile was staged.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

#if defined(NINFER_SM75)
inline constexpr bool kAttentionPvHalfProbabilities = true;
#else
inline constexpr bool kAttentionPvHalfProbabilities = false;
#endif

// Two probabilities as the A operand of attention_pv_mma.
__device__ __forceinline__ std::uint32_t pack_attention_p(float lo, float hi) {
    if constexpr (kAttentionPvHalfProbabilities) {
        return pack_f16x2(lo, hi);
    } else {
        return pack_bf16x2(lo, hi);
    }
}

#if defined(NINFER_SM75)
namespace sm75 {

// FP32 tile product of m16n8k16 fragments with FP16 A and BF16 B, laid out as mma_bf16_fp32's. Out
// of line for the same reason.
static __device__ __noinline__ float4 mma_f16_bf16_fp32(float4 acc, unsigned a0, unsigned a1,
                                                        unsigned a2, unsigned a3, unsigned b0,
                                                        unsigned b1) {
    float c0 = acc.x, c1 = acc.y, c2 = acc.z, c3 = acc.w;
    constexpr unsigned kAll = 0xffffffffu;
    const unsigned lane     = lane_id();
    const unsigned group    = lane >> 2;
    const unsigned col_lane = (lane & 3u) * 8u;
    const auto lo = [](unsigned x) { return __low2float(load_vec<__half2>(&x)); };
    const auto hi = [](unsigned x) { return __high2float(load_vec<__half2>(&x)); };
#pragma unroll
    for (unsigned t = 0; t < 4; ++t) {
        const unsigned ra0 = __shfl_sync(kAll, a0, group * 4u + t);
        const unsigned ra1 = __shfl_sync(kAll, a1, group * 4u + t);
        const unsigned ra2 = __shfl_sync(kAll, a2, group * 4u + t);
        const unsigned ra3 = __shfl_sync(kAll, a3, group * 4u + t);
        const unsigned b00 = __shfl_sync(kAll, b0, col_lane + t);
        const unsigned b01 = __shfl_sync(kAll, b1, col_lane + t);
        const unsigned b10 = __shfl_sync(kAll, b0, col_lane + 4u + t);
        const unsigned b11 = __shfl_sync(kAll, b1, col_lane + 4u + t);
        c0 = fmaf(lo(ra0), bf16_lo(b00), c0);
        c0 = fmaf(hi(ra0), bf16_hi(b00), c0);
        c0 = fmaf(lo(ra2), bf16_lo(b01), c0);
        c0 = fmaf(hi(ra2), bf16_hi(b01), c0);
        c1 = fmaf(lo(ra0), bf16_lo(b10), c1);
        c1 = fmaf(hi(ra0), bf16_hi(b10), c1);
        c1 = fmaf(lo(ra2), bf16_lo(b11), c1);
        c1 = fmaf(hi(ra2), bf16_hi(b11), c1);
        c2 = fmaf(lo(ra1), bf16_lo(b00), c2);
        c2 = fmaf(hi(ra1), bf16_hi(b00), c2);
        c2 = fmaf(lo(ra3), bf16_lo(b01), c2);
        c2 = fmaf(hi(ra3), bf16_hi(b01), c2);
        c3 = fmaf(lo(ra1), bf16_lo(b10), c3);
        c3 = fmaf(hi(ra1), bf16_hi(b10), c3);
        c3 = fmaf(lo(ra3), bf16_lo(b11), c3);
        c3 = fmaf(hi(ra3), bf16_hi(b11), c3);
    }
    return make_float4(c0, c1, c2, c3);
}

// An FP16 fragment register staged from BF16 values, as BF16 values (exact: an FP16 rounding of a
// BF16 value has at most its eight significant bits).
__device__ __forceinline__ unsigned half2_to_bf16x2(unsigned x) {
    const __half2 value = load_vec<__half2>(&x);
    return pack_bf16x2(__low2float(value), __high2float(value));
}

// Both BF16 halves are below 2^16 in magnitude (a biased exponent of at most 142; zero and BF16
// subnormals included), so their FP16 roundings are finite.
__device__ __forceinline__ bool bf16x2_fits_fp16(unsigned x) {
    return ((x >> 7) & 0xffu) <= 142u && ((x >> 23) & 0xffu) <= 142u;
}

// Each BF16 half below 2^16 as its FP16 rounding (still a BF16 value: at most eight significant
// bits), a larger one unchanged.
__device__ __forceinline__ unsigned bf16x2_fp16_rounded(unsigned x) {
    const auto rounded = [](unsigned bits) {
        if (((bits >> 7) & 0xffu) > 142u) { return bits; }
        return __float_as_uint(__half2float(__float2half_rn(__uint_as_float(bits << 16)))) >> 16;
    };
    return rounded(x & 0xffffu) | (rounded(x >> 16) << 16);
}

} // namespace sm75

// m16n8k16 with FP16 A and BF16 B. B runs on FP16 m16n8k8 where it is exactly an FP16 value; the
// outputs of a column that holds another value take the FP32 product of the same operands, so an
// output's bits depend on its own column alone (sm75::take_inexact_outputs).
__device__ __forceinline__ void mma_f16_bf16(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    constexpr unsigned kAll = 0xffffffffu;
    const unsigned cols     = __ballot_sync(
        kAll, sm75::bf16x2_exact_in_fp16(b0) && sm75::bf16x2_exact_in_fp16(b1));
    const float4 acc = make_float4(c0, c1, c2, c3);
    sm75::mma_f16_k8(c0, c1, c2, c3, a0, a1, sm75::bf16x2_to_half2(b0));
    sm75::mma_f16_k8(c0, c1, c2, c3, a2, a3, sm75::bf16x2_to_half2(b1));
    if (cols != kAll) {
        sm75::take_inexact_outputs(c0, c1, c2, c3,
                                   sm75::mma_f16_bf16_fp32(acc, a0, a1, a2, a3, b0, b1), kAll,
                                   kAll, cols);
    }
}

// m16n8k16 of BF16 fragments as the causal-cache kernels multiply Q and K: every value below 2^16
// as its FP16 rounding, on FP16 m16n8k8 for the outputs whose row of A and column of B fit FP16
// and in FP32 for the others (mma_bf16's per-output routing).
__device__ __forceinline__ void mma_bf16_rounded(float& c0, float& c1, float& c2, float& c3,
                                                 unsigned a0, unsigned a1, unsigned a2,
                                                 unsigned a3, unsigned b0, unsigned b1) {
    constexpr unsigned kAll = 0xffffffffu;
    const unsigned rows_lo =
        __ballot_sync(kAll, sm75::bf16x2_fits_fp16(a0) && sm75::bf16x2_fits_fp16(a2));
    const unsigned rows_hi =
        __ballot_sync(kAll, sm75::bf16x2_fits_fp16(a1) && sm75::bf16x2_fits_fp16(a3));
    const unsigned cols =
        __ballot_sync(kAll, sm75::bf16x2_fits_fp16(b0) && sm75::bf16x2_fits_fp16(b1));
    const float4 acc = make_float4(c0, c1, c2, c3);
    sm75::mma_f16_k8(c0, c1, c2, c3, sm75::bf16x2_to_half2(a0), sm75::bf16x2_to_half2(a1),
                     sm75::bf16x2_to_half2(b0));
    sm75::mma_f16_k8(c0, c1, c2, c3, sm75::bf16x2_to_half2(a2), sm75::bf16x2_to_half2(a3),
                     sm75::bf16x2_to_half2(b1));
    if ((rows_lo & rows_hi & cols) != kAll) {
        const float4 exact = sm75::mma_bf16_fp32(
            acc, sm75::bf16x2_fp16_rounded(a0), sm75::bf16x2_fp16_rounded(a1),
            sm75::bf16x2_fp16_rounded(a2), sm75::bf16x2_fp16_rounded(a3),
            sm75::bf16x2_fp16_rounded(b0), sm75::bf16x2_fp16_rounded(b1));
        sm75::take_inexact_outputs(c0, c1, c2, c3, exact, rows_lo, rows_hi, cols);
    }
}

// mma_f16_bf16 as the causal-cache kernels multiply P and V: every value of B below 2^16 as its
// FP16 rounding, on FP16 m16n8k8 for the columns that fit FP16 and in FP32 for the others.
__device__ __forceinline__ void mma_f16_bf16_rounded(float& c0, float& c1, float& c2, float& c3,
                                                     unsigned a0, unsigned a1, unsigned a2,
                                                     unsigned a3, unsigned b0, unsigned b1) {
    constexpr unsigned kAll = 0xffffffffu;
    const unsigned cols =
        __ballot_sync(kAll, sm75::bf16x2_fits_fp16(b0) && sm75::bf16x2_fits_fp16(b1));
    const float4 acc = make_float4(c0, c1, c2, c3);
    sm75::mma_f16_k8(c0, c1, c2, c3, a0, a1, sm75::bf16x2_to_half2(b0));
    sm75::mma_f16_k8(c0, c1, c2, c3, a2, a3, sm75::bf16x2_to_half2(b1));
    if (cols != kAll) {
        const float4 exact =
            sm75::mma_f16_bf16_fp32(acc, a0, a1, a2, a3, sm75::bf16x2_fp16_rounded(b0),
                                    sm75::bf16x2_fp16_rounded(b1));
        sm75::take_inexact_outputs(c0, c1, c2, c3, exact, kAll, kAll, cols);
    }
}

// Stores eight BF16 values (16 bytes) as their FP16 roundings and says whether every one was
// within FP16's range; when one was not, the destination holds values the caller must replace.
__device__ __forceinline__ bool store_bf16x8_as_half(void* dst, uint4 raw) {
    const bool fits = sm75::bf16x2_fits_fp16(raw.x) && sm75::bf16x2_fits_fp16(raw.y) &&
                      sm75::bf16x2_fits_fp16(raw.z) && sm75::bf16x2_fits_fp16(raw.w);
    store_vec(dst, make_uint4(sm75::bf16x2_to_half2(raw.x), sm75::bf16x2_to_half2(raw.y),
                              sm75::bf16x2_to_half2(raw.z), sm75::bf16x2_to_half2(raw.w)));
    return fits;
}

__device__ __forceinline__ bool copy_bf16x8_as_half(void* dst, const void* src) {
    return store_bf16x8_as_half(dst, load_vec<uint4>(src));
}
#endif

// O += P V for one m16n8k16 fragment: probabilities from pack_attention_p, BF16 values.
__device__ __forceinline__ void attention_pv_mma(float& c0, float& c1, float& c2, float& c3,
                                                 unsigned p0, unsigned p1, unsigned p2,
                                                 unsigned p3, unsigned v0, unsigned v1) {
#if defined(NINFER_SM75)
    mma_f16_bf16(c0, c1, c2, c3, p0, p1, p2, p3, v0, v1);
#else
    mma_bf16(c0, c1, c2, c3, p0, p1, p2, p3, v0, v1);
#endif
}

} // namespace ninfer::ops
