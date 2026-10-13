#pragma once

#include "ops/common/memory.cuh"

#include <cuda_fp16.h>

namespace ninfer::ops {

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
#if defined(NINFER_SM70_DEVICE)
    unsigned r[2];
    sm70::ldmatrix<2>(r, addr);
    r0 = r[0];
    r1 = r[1];
#else
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
#endif
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
#if defined(NINFER_SM70_DEVICE)
    unsigned r[4];
    sm70::ldmatrix<4>(r, addr);
    r0 = r[0];
    r1 = r[1];
    r2 = r[2];
    r3 = r[3];
#else
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
#endif
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
#if defined(NINFER_SM70_DEVICE)
    unsigned r[2];
    sm70::ldmatrix_trans<2>(r, addr);
    r0 = r[0];
    r1 = r[1];
#else
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
#endif
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                              unsigned& r3, unsigned addr) {
#if defined(NINFER_SM70_DEVICE)
    unsigned r[4];
    sm70::ldmatrix_trans<4>(r, addr);
    r0 = r[0];
    r1 = r[1];
    r2 = r[2];
    r3 = r[3];
#else
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
#endif
}

#if defined(NINFER_SM75)
// Turing (sm_75) tensor cores have the FP16 m16n8k8 and INT8 m8n8k16 shapes only: no BF16 or TF32
// operands and no m16n8k16/m16n8k32. The helpers below keep every caller's fragment contract.
//
// - FP16 m16n8k16 is two m16n8k8 over k 0..7 ({a0,a1}, b0) and k 8..15 ({a2,a3}, b1): the k8
//   fragments are exactly the halves of the k16 ones.
// - INT8 m16n8k32 is four m8n8k16: m8n8k16 holds one row g with k = 4*tig..4*tig+3 for A and the
//   same k of column g for B, which is a0 (row g), a1 (row g+8), a2/a3 (k + 16) and b0/b1 (k + 16)
//   of m16n8k32. Integer accumulation is exact, so the result is identical.
// - BF16 and TF32 are not narrowed. A BF16 value (8 significant bits) is exactly an FP16 value when
//   it is zero or its exponent is in [-17, 15]; a TF32 value (11 significant bits) when it is zero
//   or its exponent is in [-14, 15]. The tile runs on FP16 m16n8k8 with FP32 accumulation, which
//   forms the same exact products the native instruction does for every output whose row of A and
//   column of B qualify. When some do not, the warp also computes the tile in FP32 from the same
//   operands (exact products, FP32 sums, rows and columns gathered with shuffles), and exactly the
//   outputs with a non-qualifying row or column take that result. Correctness therefore never
//   depends on the range of the data, only the speed does, and an output's bits depend on its own
//   operands alone, whatever else shares its tile.
namespace sm75 {

__device__ __forceinline__ unsigned lane_id() {
    unsigned lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    return lane;
}

__device__ __forceinline__ void mma_f16_k8(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                           unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ void mma_f16_k8_f16acc(unsigned& d0, unsigned& d1, unsigned a0,
                                                  unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
                 : "+r"(d0), "+r"(d1)
                 : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ void mma_s8_k16(int& c0, int& c1, unsigned a, unsigned b) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a), "r"(b));
}

// Both BF16 halves are zero or have a biased exponent in [110, 142].
__device__ __forceinline__ bool bf16x2_exact_in_fp16(unsigned x) {
    const unsigned lo = x & 0xffffu, hi = x >> 16;
    const bool lo_ok  = (lo & 0x7fffu) == 0u || ((lo >> 7) & 0xffu) - 110u <= 32u;
    const bool hi_ok  = (hi & 0x7fffu) == 0u || ((hi >> 7) & 0xffu) - 110u <= 32u;
    return lo_ok && hi_ok;
}

__device__ __forceinline__ float bf16_lo(unsigned x) { return __uint_as_float(x << 16); }
__device__ __forceinline__ float bf16_hi(unsigned x) { return __uint_as_float(x & 0xffff0000u); }

__device__ __forceinline__ unsigned half2_bits(float lo, float hi) {
    const __half2 h = __halves2half2(__float2half_rn(lo), __float2half_rn(hi));
    return *reinterpret_cast<const unsigned*>(&h);
}

// Exact for operands that passed bf16x2_exact_in_fp16.
__device__ __forceinline__ unsigned bf16x2_to_half2(unsigned x) {
    return half2_bits(bf16_lo(x), bf16_hi(x));
}

// The operand bits a TF32 MMA reads: the low 13 mantissa bits are not part of the value.
__device__ __forceinline__ float tf32_value(unsigned x) { return __uint_as_float(x & 0xffffe000u); }

__device__ __forceinline__ bool tf32_exact_in_fp16(unsigned x) {
    const unsigned v = x & 0x7fffe000u;
    return v == 0u || ((v >> 23) & 0xffu) - 113u <= 29u;
}

// FP32 tile product of m16n8k16 BF16 fragments; every lane's quad holds its rows, and the quad of
// group n holds column n of B. Out of line: it is the rare route, and inlining it at every MMA site
// multiplies the size of the kernels (and their ptxas time) several times over.
static __device__ __noinline__ float4 mma_bf16_fp32(float4 acc, unsigned a0, unsigned a1,
                                                    unsigned a2, unsigned a3, unsigned b0,
                                                    unsigned b1) {
    float c0 = acc.x, c1 = acc.y, c2 = acc.z, c3 = acc.w;
    constexpr unsigned kAll = 0xffffffffu;
    const unsigned lane     = lane_id();
    const unsigned group    = lane >> 2;
    const unsigned col_lane = (lane & 3u) * 8u; // quad of column 2*tig
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
        c0 = fmaf(bf16_lo(ra0), bf16_lo(b00), c0);
        c0 = fmaf(bf16_hi(ra0), bf16_hi(b00), c0);
        c0 = fmaf(bf16_lo(ra2), bf16_lo(b01), c0);
        c0 = fmaf(bf16_hi(ra2), bf16_hi(b01), c0);
        c1 = fmaf(bf16_lo(ra0), bf16_lo(b10), c1);
        c1 = fmaf(bf16_hi(ra0), bf16_hi(b10), c1);
        c1 = fmaf(bf16_lo(ra2), bf16_lo(b11), c1);
        c1 = fmaf(bf16_hi(ra2), bf16_hi(b11), c1);
        c2 = fmaf(bf16_lo(ra1), bf16_lo(b00), c2);
        c2 = fmaf(bf16_hi(ra1), bf16_hi(b00), c2);
        c2 = fmaf(bf16_lo(ra3), bf16_lo(b01), c2);
        c2 = fmaf(bf16_hi(ra3), bf16_hi(b01), c2);
        c3 = fmaf(bf16_lo(ra1), bf16_lo(b10), c3);
        c3 = fmaf(bf16_hi(ra1), bf16_hi(b10), c3);
        c3 = fmaf(bf16_lo(ra3), bf16_lo(b11), c3);
        c3 = fmaf(bf16_hi(ra3), bf16_hi(b11), c3);
    }
    return make_float4(c0, c1, c2, c3);
}

// FP32 tile product of m16n8k8 TF32 fragments (a0/a1: k = tig, a2/a3: k = tig + 4; b0/b1 the same
// k). Out of line for the same reason as mma_bf16_fp32.
static __device__ __noinline__ float4 mma_tf32_fp32(float4 acc, unsigned a0, unsigned a1,
                                                    unsigned a2, unsigned a3, unsigned b0,
                                                    unsigned b1) {
    float c0 = acc.x, c1 = acc.y, c2 = acc.z, c3 = acc.w;
    constexpr unsigned kAll = 0xffffffffu;
    const unsigned lane     = lane_id();
    const unsigned group    = lane >> 2;
    const unsigned col_lane = (lane & 3u) * 8u;
#pragma unroll
    for (unsigned t = 0; t < 4; ++t) {
        const float ra0 = tf32_value(__shfl_sync(kAll, a0, group * 4u + t));
        const float ra1 = tf32_value(__shfl_sync(kAll, a1, group * 4u + t));
        const float ra2 = tf32_value(__shfl_sync(kAll, a2, group * 4u + t));
        const float ra3 = tf32_value(__shfl_sync(kAll, a3, group * 4u + t));
        const float b00 = tf32_value(__shfl_sync(kAll, b0, col_lane + t));
        const float b01 = tf32_value(__shfl_sync(kAll, b1, col_lane + t));
        const float b10 = tf32_value(__shfl_sync(kAll, b0, col_lane + 4u + t));
        const float b11 = tf32_value(__shfl_sync(kAll, b1, col_lane + 4u + t));
        c0 = fmaf(ra0, b00, c0);
        c0 = fmaf(ra2, b01, c0);
        c1 = fmaf(ra0, b10, c1);
        c1 = fmaf(ra2, b11, c1);
        c2 = fmaf(ra1, b00, c2);
        c2 = fmaf(ra3, b01, c2);
        c3 = fmaf(ra1, b10, c3);
        c3 = fmaf(ra3, b11, c3);
    }
    return make_float4(c0, c1, c2, c3);
}

// The lane's outputs (rows g and g + 8, columns 2t and 2t + 1) whose own row of A or column of B
// holds an operand FP16 cannot represent take the FP32 tile product. The ballots say which pieces
// qualify: row g in quad g of `rows_lo`, row g + 8 in quad g of `rows_hi`, column n in quad n of
// `cols`. An output's route therefore depends on its own row and column alone, never on the other
// rows and columns that share its tile, so every tiling of a product gives the same bits.
__device__ __forceinline__ void take_inexact_outputs(float& c0, float& c1, float& c2, float& c3,
                                                     float4 exact, unsigned rows_lo,
                                                     unsigned rows_hi, unsigned cols) {
    const unsigned lane = lane_id();
    const unsigned g = lane >> 2, t = lane & 3u;
    const auto quad = [](unsigned mask, unsigned q) { return ((mask >> (4u * q)) & 0xfu) == 0xfu; };
    const bool row_lo = quad(rows_lo, g), row_hi = quad(rows_hi, g);
    const bool col_0 = quad(cols, 2u * t), col_1 = quad(cols, 2u * t + 1u);
    if (!(row_lo && col_0)) { c0 = exact.x; }
    if (!(row_lo && col_1)) { c1 = exact.y; }
    if (!(row_hi && col_0)) { c2 = exact.z; }
    if (!(row_hi && col_1)) { c3 = exact.w; }
}

} // namespace sm75

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    constexpr unsigned kAll = 0xffffffffu;
    const unsigned rows_lo  = __ballot_sync(
        kAll, sm75::bf16x2_exact_in_fp16(a0) && sm75::bf16x2_exact_in_fp16(a2));
    const unsigned rows_hi = __ballot_sync(
        kAll, sm75::bf16x2_exact_in_fp16(a1) && sm75::bf16x2_exact_in_fp16(a3));
    const unsigned cols = __ballot_sync(
        kAll, sm75::bf16x2_exact_in_fp16(b0) && sm75::bf16x2_exact_in_fp16(b1));
    const float4 acc = make_float4(c0, c1, c2, c3);
    sm75::mma_f16_k8(c0, c1, c2, c3, sm75::bf16x2_to_half2(a0), sm75::bf16x2_to_half2(a1),
                     sm75::bf16x2_to_half2(b0));
    sm75::mma_f16_k8(c0, c1, c2, c3, sm75::bf16x2_to_half2(a2), sm75::bf16x2_to_half2(a3),
                     sm75::bf16x2_to_half2(b1));
    if ((rows_lo & rows_hi & cols) != kAll) {
        sm75::take_inexact_outputs(c0, c1, c2, c3,
                                   sm75::mma_bf16_fp32(acc, a0, a1, a2, a3, b0, b1), rows_lo,
                                   rows_hi, cols);
    }
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
    sm75::mma_f16_k8(c0, c1, c2, c3, a0, a1, b0);
    sm75::mma_f16_k8(c0, c1, c2, c3, a2, a3, b1);
}

// FP16 accumulation: d0 holds the row-gid pair and d1 the row-gid+8 pair of the fragment, in the
// same order as mma_f16's c0..c3. Two k8 steps round the partial sum to FP16 once more than k16.
__device__ __forceinline__ void mma_f16_f16acc(unsigned& d0, unsigned& d1, unsigned a0,
                                               unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                               unsigned b1) {
    sm75::mma_f16_k8_f16acc(d0, d1, a0, a1, b0);
    sm75::mma_f16_k8_f16acc(d0, d1, a2, a3, b1);
}

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    sm75::mma_s8_k16(c0, c1, a0, b0);
    sm75::mma_s8_k16(c0, c1, a2, b1);
    sm75::mma_s8_k16(c2, c3, a1, b0);
    sm75::mma_s8_k16(c2, c3, a3, b1);
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
    constexpr unsigned kAll = 0xffffffffu;
    const unsigned rows_lo  = __ballot_sync(
        kAll, sm75::tf32_exact_in_fp16(a0) && sm75::tf32_exact_in_fp16(a2));
    const unsigned rows_hi = __ballot_sync(
        kAll, sm75::tf32_exact_in_fp16(a1) && sm75::tf32_exact_in_fp16(a3));
    const unsigned cols = __ballot_sync(
        kAll, sm75::tf32_exact_in_fp16(b0) && sm75::tf32_exact_in_fp16(b1));
    const float4 acc = make_float4(c0, c1, c2, c3);
    // FP16 m16n8k8 holds k = 2*tig, 2*tig+1 where TF32 holds k = tig, tig+4. Pairing them the same
    // way in A and B permutes k alike on both sides, which leaves every dot product unchanged.
    sm75::mma_f16_k8(c0, c1, c2, c3, sm75::half2_bits(sm75::tf32_value(a0), sm75::tf32_value(a2)),
                     sm75::half2_bits(sm75::tf32_value(a1), sm75::tf32_value(a3)),
                     sm75::half2_bits(sm75::tf32_value(b0), sm75::tf32_value(b1)));
    if ((rows_lo & rows_hi & cols) != kAll) {
        sm75::take_inexact_outputs(c0, c1, c2, c3,
                                   sm75::mma_tf32_fp32(acc, a0, a1, a2, a3, b0, b1), rows_lo,
                                   rows_hi, cols);
    }
}

__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                         float a1, float a2, float a3, float b0, float b1) {
    mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                  __float_as_uint(a3), __float_as_uint(b0), __float_as_uint(b1));
}
#else
__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
#if defined(NINFER_SM70_DEVICE)
    sm70::mma_m16n8k16<sm70::Bf16Operands>(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
#if defined(NINFER_SM70_DEVICE)
    sm70::mma_m16n8k16<sm70::F16Operands>(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// FP16 accumulation: twice the FP32-accumulator rate on GeForce parts. d0 holds the row-gid pair
// and d1 the row-gid+8 pair of the fragment, in the same order as mma_f16's c0..c3.
__device__ __forceinline__ void mma_f16_f16acc(unsigned& d0, unsigned& d1, unsigned a0,
                                               unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                               unsigned b1) {
#if defined(NINFER_SM70_DEVICE)
    sm70::mma_m16n8k16_f16acc(d0, d1, a0, a1, a2, a3, b0, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(d0), "+r"(d1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if defined(NINFER_SM70_DEVICE)
    sm70::mma_m16n8k32_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
#if defined(NINFER_SM70_DEVICE)
    sm70::mma_m16n8k8_tf32(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                         float a1, float a2, float a3, float b0, float b1) {
    mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                  __float_as_uint(a3), __float_as_uint(b0), __float_as_uint(b1));
}
#endif

// On RTX 5090, block-scaled FP8 avoids the plain form's half-rate FP32 accumulation.
// UE8M0 0x7f is exactly 1: preserve the raw E4M3 dot product and keep the callers'
// existing row scales in FP32 after the MMA. No operand repacking is needed.
__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    constexpr unsigned kUnitScale8          = 0x7Fu;
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
                 "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                 "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(kUnitScale8),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(kUnitScale8), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}

__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, "
                 "{%8,%9}, "
                 "{%0,%1,%2,%3}, "
                 "{%10}, "
                 "{%11,%12}, "
                 "{%13}, "
                 "{%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}

} // namespace ninfer::ops
