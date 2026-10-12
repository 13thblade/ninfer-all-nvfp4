#pragma once

// Volta (sm_70) has none of the Ampere warp primitives the kernels are written against: no
// cp.async, no ldmatrix, no m16n8 mma.sync shapes and no integer tensor cores. A 70 build keeps
// every kernel and replaces those primitives with the exact equivalents below, built from warp
// shuffles, shared-memory loads, FP32 FMA and dp4a. Each one reproduces the PTX fragment layout of
// the instruction it stands for, so the callers' register bookkeeping is unchanged. The hot prefill
// and attention paths of a 70 build take their own routes; these keep every other kernel correct.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#    define NINFER_SM70_DEVICE 1
#endif

namespace ninfer::ops::sm70 {

__device__ __forceinline__ unsigned lane_id() {
    unsigned lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    return lane;
}

__device__ __forceinline__ unsigned lds32(unsigned addr) {
    unsigned value;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(value) : "r"(addr));
    return value;
}

__device__ __forceinline__ unsigned lds16(unsigned addr) {
    unsigned short value;
    asm volatile("ld.shared.b16 %0, [%1];" : "=h"(value) : "r"(addr));
    return value;
}

// ldmatrix.m8n8.b16: thread 8i+r names row r of matrix i; every thread receives, from matrix i,
// the 32-bit word at row lane/4, columns 2(lane%4) and 2(lane%4)+1.
template <int Matrices>
__device__ __forceinline__ void ldmatrix(unsigned (&r)[Matrices], unsigned addr) {
    const unsigned lane = lane_id();
#pragma unroll
    for (int i = 0; i < Matrices; ++i) {
        const unsigned row = __shfl_sync(0xffffffffu, addr, (i << 3) | (lane >> 2));
        r[i]               = lds32(row + ((lane & 3u) << 2));
    }
}

// ldmatrix.m8n8.trans.b16: every thread receives, from matrix i, the elements at rows 2(lane%4)
// and 2(lane%4)+1 of column lane/4, low half first.
template <int Matrices>
__device__ __forceinline__ void ldmatrix_trans(unsigned (&r)[Matrices], unsigned addr) {
    const unsigned lane   = lane_id();
    const unsigned column = (lane >> 2) << 1;
#pragma unroll
    for (int i = 0; i < Matrices; ++i) {
        const unsigned row0 = __shfl_sync(0xffffffffu, addr, (i << 3) | ((lane & 3u) << 1));
        const unsigned row1 = __shfl_sync(0xffffffffu, addr, (i << 3) | ((lane & 3u) << 1) | 1u);
        r[i]                = lds16(row0 + column) | (lds16(row1 + column) << 16);
    }
}

__device__ __forceinline__ float2 bf16x2_bits_to_float2(unsigned bits) {
    return make_float2(__uint_as_float(bits << 16), __uint_as_float(bits & 0xffff0000u));
}

__device__ __forceinline__ float2 f16x2_bits_to_float2(unsigned bits) {
    __half2 value;
    *reinterpret_cast<unsigned*>(&value) = bits;
    return __half22float2(value);
}

struct Bf16Operands {
    __device__ __forceinline__ static float2 widen(unsigned bits) {
        return bf16x2_bits_to_float2(bits);
    }
};

struct F16Operands {
    __device__ __forceinline__ static float2 widen(unsigned bits) {
        return f16x2_bits_to_float2(bits);
    }
};

// mma.m16n8k16.row.col with FP32 accumulators. Thread (g = lane/4, t = lane%4) holds
// A[g][2t..], A[g+8][2t..], A[g][2t+8..], A[g+8][2t+8..], B[2t..][g], B[2t+8..][g] and the
// accumulators C[g][2t..2t+1], C[g+8][2t..2t+1]. Row g of A lives in quad g; column n of B in
// quad n. Each pass j gathers lane j of the thread's own quad (its two A rows) and lane j of
// quads 2t and 2t+1 (its two B columns), which together cover k = 2j, 2j+1, 2j+8 and 2j+9.
template <class Operands>
__device__ __forceinline__ void mma_m16n8k16(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    const unsigned lane = lane_id();
    const unsigned quad = lane & ~3u;
    const unsigned col0 = (lane & 3u) << 3;
#pragma unroll 1
    for (unsigned j = 0; j < 4; ++j) {
        const float2 a_g_lo  = Operands::widen(__shfl_sync(0xffffffffu, a0, quad | j));
        const float2 a_g8_lo = Operands::widen(__shfl_sync(0xffffffffu, a1, quad | j));
        const float2 a_g_hi  = Operands::widen(__shfl_sync(0xffffffffu, a2, quad | j));
        const float2 a_g8_hi = Operands::widen(__shfl_sync(0xffffffffu, a3, quad | j));
        const float2 b_n0_lo = Operands::widen(__shfl_sync(0xffffffffu, b0, col0 | j));
        const float2 b_n0_hi = Operands::widen(__shfl_sync(0xffffffffu, b1, col0 | j));
        const float2 b_n1_lo = Operands::widen(__shfl_sync(0xffffffffu, b0, (col0 + 4u) | j));
        const float2 b_n1_hi = Operands::widen(__shfl_sync(0xffffffffu, b1, (col0 + 4u) | j));
        c0                   = fmaf(a_g_lo.x, b_n0_lo.x, c0);
        c0                   = fmaf(a_g_lo.y, b_n0_lo.y, c0);
        c0                   = fmaf(a_g_hi.x, b_n0_hi.x, c0);
        c0                   = fmaf(a_g_hi.y, b_n0_hi.y, c0);
        c1                   = fmaf(a_g_lo.x, b_n1_lo.x, c1);
        c1                   = fmaf(a_g_lo.y, b_n1_lo.y, c1);
        c1                   = fmaf(a_g_hi.x, b_n1_hi.x, c1);
        c1                   = fmaf(a_g_hi.y, b_n1_hi.y, c1);
        c2                   = fmaf(a_g8_lo.x, b_n0_lo.x, c2);
        c2                   = fmaf(a_g8_lo.y, b_n0_lo.y, c2);
        c2                   = fmaf(a_g8_hi.x, b_n0_hi.x, c2);
        c2                   = fmaf(a_g8_hi.y, b_n0_hi.y, c2);
        c3                   = fmaf(a_g8_lo.x, b_n1_lo.x, c3);
        c3                   = fmaf(a_g8_lo.y, b_n1_lo.y, c3);
        c3                   = fmaf(a_g8_hi.x, b_n1_hi.x, c3);
        c3                   = fmaf(a_g8_hi.y, b_n1_hi.y, c3);
    }
}

// mma.m16n8k16.row.col.f16.f16.f16.f16: the same product with FP16 accumulators, d0 holding the
// row-g pair and d1 the row-(g+8) pair. The emulation accumulates in FP32 and rounds once.
__device__ __forceinline__ void mma_m16n8k16_f16acc(unsigned& d0, unsigned& d1, unsigned a0,
                                                    unsigned a1, unsigned a2, unsigned a3,
                                                    unsigned b0, unsigned b1) {
    const float2 lo = f16x2_bits_to_float2(d0);
    const float2 hi = f16x2_bits_to_float2(d1);
    float c0 = lo.x, c1 = lo.y, c2 = hi.x, c3 = hi.y;
    mma_m16n8k16<F16Operands>(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    const __half2 out_lo = __floats2half2_rn(c0, c1);
    const __half2 out_hi = __floats2half2_rn(c2, c3);
    d0                   = *reinterpret_cast<const unsigned*>(&out_lo);
    d1                   = *reinterpret_cast<const unsigned*>(&out_hi);
}

// mma.m16n8k8.row.col.f32.f16.f16.f32: thread (g, t) holds A[g][2t..], A[g+8][2t..] and
// B[2t..][g]; the accumulators match the k16 shape.
__device__ __forceinline__ void mma_m16n8k8_f16(float& c0, float& c1, float& c2, float& c3,
                                                unsigned a0, unsigned a1, unsigned b0) {
    const unsigned lane = lane_id();
    const unsigned quad = lane & ~3u;
    const unsigned col0 = (lane & 3u) << 3;
#pragma unroll 1
    for (unsigned j = 0; j < 4; ++j) {
        const float2 a_g  = f16x2_bits_to_float2(__shfl_sync(0xffffffffu, a0, quad | j));
        const float2 a_g8 = f16x2_bits_to_float2(__shfl_sync(0xffffffffu, a1, quad | j));
        const float2 b_n0 = f16x2_bits_to_float2(__shfl_sync(0xffffffffu, b0, col0 | j));
        const float2 b_n1 = f16x2_bits_to_float2(__shfl_sync(0xffffffffu, b0, (col0 + 4u) | j));
        c0                = fmaf(a_g.y, b_n0.y, fmaf(a_g.x, b_n0.x, c0));
        c1                = fmaf(a_g.y, b_n1.y, fmaf(a_g.x, b_n1.x, c1));
        c2                = fmaf(a_g8.y, b_n0.y, fmaf(a_g8.x, b_n0.x, c2));
        c3                = fmaf(a_g8.y, b_n1.y, fmaf(a_g8.x, b_n1.x, c3));
    }
}

// mma.m16n8k32.row.col.s32.s8.s8.s32: the k16 layout with four int8 codes per register, so every
// pass covers k = 4j..4j+3 and 4j+16..4j+19 through dp4a. Integer accumulation is exact.
__device__ __forceinline__ void mma_m16n8k32_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0,
                                                unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                                unsigned b1) {
    const unsigned lane = lane_id();
    const unsigned quad = lane & ~3u;
    const unsigned col0 = (lane & 3u) << 3;
#pragma unroll 1
    for (unsigned j = 0; j < 4; ++j) {
        const int a_g_lo  = static_cast<int>(__shfl_sync(0xffffffffu, a0, quad | j));
        const int a_g8_lo = static_cast<int>(__shfl_sync(0xffffffffu, a1, quad | j));
        const int a_g_hi  = static_cast<int>(__shfl_sync(0xffffffffu, a2, quad | j));
        const int a_g8_hi = static_cast<int>(__shfl_sync(0xffffffffu, a3, quad | j));
        const int b_n0_lo = static_cast<int>(__shfl_sync(0xffffffffu, b0, col0 | j));
        const int b_n0_hi = static_cast<int>(__shfl_sync(0xffffffffu, b1, col0 | j));
        const int b_n1_lo = static_cast<int>(__shfl_sync(0xffffffffu, b0, (col0 + 4u) | j));
        const int b_n1_hi = static_cast<int>(__shfl_sync(0xffffffffu, b1, (col0 + 4u) | j));
        c0                = __dp4a(a_g_hi, b_n0_hi, __dp4a(a_g_lo, b_n0_lo, c0));
        c1                = __dp4a(a_g_hi, b_n1_hi, __dp4a(a_g_lo, b_n1_lo, c1));
        c2                = __dp4a(a_g8_hi, b_n0_hi, __dp4a(a_g8_lo, b_n0_lo, c2));
        c3                = __dp4a(a_g8_hi, b_n1_hi, __dp4a(a_g8_lo, b_n1_lo, c3));
    }
}

// mma.m16n8k16.row.col.s32.s8.s8.s32: thread (g, t) holds A[g][4t..4t+3], A[g+8][4t..] and
// B[4t..4t+3][g].
__device__ __forceinline__ void mma_m16n8k16_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0,
                                                unsigned a1, unsigned b0) {
    const unsigned lane = lane_id();
    const unsigned quad = lane & ~3u;
    const unsigned col0 = (lane & 3u) << 3;
#pragma unroll 1
    for (unsigned j = 0; j < 4; ++j) {
        const int a_g  = static_cast<int>(__shfl_sync(0xffffffffu, a0, quad | j));
        const int a_g8 = static_cast<int>(__shfl_sync(0xffffffffu, a1, quad | j));
        const int b_n0 = static_cast<int>(__shfl_sync(0xffffffffu, b0, col0 | j));
        const int b_n1 = static_cast<int>(__shfl_sync(0xffffffffu, b0, (col0 + 4u) | j));
        c0             = __dp4a(a_g, b_n0, c0);
        c1             = __dp4a(a_g, b_n1, c1);
        c2             = __dp4a(a_g8, b_n0, c2);
        c3             = __dp4a(a_g8, b_n1, c3);
    }
}

// mma.m16n8k8.row.col.f32.tf32.tf32.f32: thread (g, t) holds A[g][t], A[g+8][t], A[g][t+4],
// A[g+8][t+4], B[t][g] and B[t+4][g]. The tensor core ignores the low 13 mantissa bits of each
// operand, which the emulation reproduces by masking them.
__device__ __forceinline__ float tf32_operand(unsigned bits) {
    return __uint_as_float(bits & 0xffffe000u);
}

__device__ __forceinline__ void mma_m16n8k8_tf32(float& c0, float& c1, float& c2, float& c3,
                                                 unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                                 unsigned b0, unsigned b1) {
    const unsigned lane = lane_id();
    const unsigned quad = lane & ~3u;
    const unsigned col0 = (lane & 3u) << 3;
#pragma unroll 1
    for (unsigned j = 0; j < 4; ++j) {
        const float a_g_lo  = tf32_operand(__shfl_sync(0xffffffffu, a0, quad | j));
        const float a_g8_lo = tf32_operand(__shfl_sync(0xffffffffu, a1, quad | j));
        const float a_g_hi  = tf32_operand(__shfl_sync(0xffffffffu, a2, quad | j));
        const float a_g8_hi = tf32_operand(__shfl_sync(0xffffffffu, a3, quad | j));
        const float b_n0_lo = tf32_operand(__shfl_sync(0xffffffffu, b0, col0 | j));
        const float b_n0_hi = tf32_operand(__shfl_sync(0xffffffffu, b1, col0 | j));
        const float b_n1_lo = tf32_operand(__shfl_sync(0xffffffffu, b0, (col0 + 4u) | j));
        const float b_n1_hi = tf32_operand(__shfl_sync(0xffffffffu, b1, (col0 + 4u) | j));
        c0                  = fmaf(a_g_hi, b_n0_hi, fmaf(a_g_lo, b_n0_lo, c0));
        c1                  = fmaf(a_g_hi, b_n1_hi, fmaf(a_g_lo, b_n1_lo, c1));
        c2                  = fmaf(a_g8_hi, b_n0_hi, fmaf(a_g8_lo, b_n0_lo, c2));
        c3                  = fmaf(a_g8_hi, b_n1_hi, fmaf(a_g8_lo, b_n1_lo, c3));
    }
}

// __reduce_max_sync (redux.sync, sm_80) over the lanes 0..Width-1 named by mask, as a butterfly.
template <int Width>
__device__ __forceinline__ unsigned reduce_max(unsigned mask, unsigned value) {
    static_assert(Width > 0 && Width <= 32 && (Width & (Width - 1)) == 0);
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        const unsigned other = __shfl_xor_sync(mask, value, offset, Width);
        value                = other > value ? other : value;
    }
    return value;
}

// cp.async with its zero fill, as a plain copy: the data is in shared memory when the call
// returns, which satisfies every later wait_group.
template <int Bytes, bool BypassL1>
__device__ __forceinline__ void copy_to_shared(void* smem_dst, const void* gmem_src) {
    if constexpr (Bytes == 16) {
        *static_cast<int4*>(smem_dst) = BypassL1 ? __ldcg(static_cast<const int4*>(gmem_src))
                                                 : *static_cast<const int4*>(gmem_src);
    } else if constexpr (Bytes == 8) {
        *static_cast<int2*>(smem_dst) = *static_cast<const int2*>(gmem_src);
    } else {
        static_assert(Bytes == 4);
        *static_cast<int*>(smem_dst) = *static_cast<const int*>(gmem_src);
    }
}

template <int Bytes, bool BypassL1>
__device__ __forceinline__ void copy_to_shared_zfill(void* smem_dst, const void* gmem_src,
                                                     int src_bytes) {
    if (src_bytes >= Bytes) {
        copy_to_shared<Bytes, BypassL1>(smem_dst, gmem_src);
        return;
    }
    unsigned char* dst       = static_cast<unsigned char*>(smem_dst);
    const unsigned char* src = static_cast<const unsigned char*>(gmem_src);
#pragma unroll
    for (int i = 0; i < Bytes; ++i) { dst[i] = i < src_bytes ? src[i] : 0; }
}

} // namespace ninfer::ops::sm70
