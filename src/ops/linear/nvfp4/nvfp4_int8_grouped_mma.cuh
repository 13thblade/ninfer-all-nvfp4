#pragma once

// NVFP4 weights on the integer tensor cores (Tier-2, sm_86): the stored E2M1 codes are widened to
// s8 (v = 2*e2m1, an exact int8 grid) and fed to mma.m16n8k16.s8 -- one NVFP4 G16 group per MMA --
// with the per-16 E4M3 group scales rescaling the int32 partials in FP32. Activations are
// quantised to s8 with one FP32 scale per (token, 32 columns): a per-token scalar lets one
// outlier channel starve the whole token, and the 32-column group confines a channel outlier to
// its own half of a BK=64 tile -- two rescale constants per k-tile, one per k-step pair, so the
// rescale still folds next to the weight scale at no extra bookkeeping.
//
// The E2M1 1/2 factor lives in the E4M3 scale decode (nvfp4_e4m3_scale.h) -- the codec widens E2M1
// to 2*e2m1, and llama.cpp folds the same 1/2 into its ue4m3 decode -- so the global weight divisor
// reaches the mainloop as alpha = 1/divisor and is applied once to the FP32 accumulators after the
// K loop, matching the A16 route's contract. Two half factors would halve the result silently, one
// in each place is exactly right; the enumeration test pins the decode and PPL pins the product.
//
// The mainloop stages 128 output rows and serves every projection route on this representation:
// the plain Linear / LinearAdd routes and both halves of gate_up (the gate projection to an FP32
// plane, the up projection with the GateProduct store that reads it back) differ only in their
// output adapter. Performance model and measured instruction budget: the issue-port bound is
// ~45% MFU (measured), ~50-55% after the staging optimisations; the real-world expectation is
// 30-45% MFU, the same staging-bound conclusion the A8 integer routes already live with.

#include "ops/common/mma.cuh"  // sm70:: and sm75:: mma emulations, and the NINFER_SM70_DEVICE guard
#include "ops/linear/common/output.cuh"
#include "ops/linear/nvfp4/nvfp4_codec_unified.cuh"
#include "ops/linear/nvfp4/nvfp4_e4m3_scale.h"
#include "ops/linear/nvfp4/nvfp4_int8_grouped_host.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops::detail::unified {

// ---------------------------------------------------------------------------
// E2M1 code -> s8, four codes per word (PRMT/LOP3 form).
//
// The nibble v = 2*e2m1 takes the values {0,1,2,3,4,6,8,12} over u = n&7 and negates with bit 3,
// so the sign select is a byte-wise mux between two PRMT tables and nvcc folds it into one LOP3
// (immLut 0xD8). Consuming the raw uint32 words directly (packed nibble selectors, low half then
// high half) makes the split/interleave round-trip of a byte-shaped selector unnecessary: the
// words leave the codec already in k order.
//
// The sign masks are extracted with inline prmt.b32, NOT the __byte_perm intrinsic: the intrinsic
// lowers to PRMT with the selector masked by 0x7777 (the sign-replicate mode bit is stripped --
// measured on CUDA 13.4 / sm_86, the intrinsic performs a plain nibble&7 byte select), while
// inline asm reaches the documented PTX sign-replicate mode. The table lookups below use
// 0x7777-masked selectors, so the intrinsic is safe for them.
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned prmt_sign_replicate(unsigned a, unsigned b, unsigned s) {
    unsigned result;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(s));
    return result;
}

__device__ __forceinline__ void e2m1_word(unsigned x, unsigned& q0, unsigned& q1) {
    const unsigned sel0 = x & 0x7777u;                               // low 16 bits: 4 nibbles
    const unsigned sel1 = (x >> 16) & 0x7777u;
    const unsigned p0 = __byte_perm(0x03020100u, 0x0C080604u, sel0); // +V(u) = {0,1,2,3,4,6,8,12}
    const unsigned p1 = __byte_perm(0x03020100u, 0x0C080604u, sel1);
    const unsigned n0 = __byte_perm(0xFDFEFF00u, 0xF4F8FAFCu, sel0); // -V(u) = {0,-1,-2,-3,-4,-6,-8,-12}
    const unsigned n1 = __byte_perm(0xFDFEFF00u, 0xF4F8FAFCu, sel1);
    const unsigned me = prmt_sign_replicate(x << 4, 0u, 0xBA98u);    // even-code sign masks
    const unsigned mo = prmt_sign_replicate(x, 0u, 0xBA98u);         // odd-code sign masks
    const unsigned m0 = __byte_perm(me, mo, 0x5140u);                // codes 0..3 signs
    const unsigned m1 = __byte_perm(me, mo, 0x7362u);                // codes 4..7 signs
    q0 = (p0 & ~m0) | (n0 & m0);   // nvcc synthesises one LOP3 (immLut 0xD8) per mux
    q1 = (p1 & ~m1) | (n1 & m1);
}

// ---------------------------------------------------------------------------
// E4M3FN scale byte -> FP32, llama.cpp's ue4m3 semantics (see nvfp4_e4m3_scale.h for the whole
// argument: the trailing /2, NaN -> 0, and the unreachable >= 0x80 domain). The full 256-entry
// table costs 1KB of constant-storage rom and one cached load per scale byte, which beats both the
// CUDA fp8 header conversion (compiled to an outlined call with BSSY/BSYNC bookkeeping on sm_86,
// ~53 instructions per k-tile per thread) and an inline bit assembly of similar size. The host and
// the device table expand one list, so the enumeration test on the host covers what this reads.
// ---------------------------------------------------------------------------
__device__ constexpr float kNvfp4E4M3ScaleLut[256] = {
    NINFER_E4M3_SCALE_LIST(NINFER_E4M3_SCALE_VALUE)};

// ---------------------------------------------------------------------------
// Activation quantiser: one FP32 scale per (token, 32 columns). A 32-column group confines an
// outlier channel to its own half of a k-tile (a per-token absmax starves every other channel of
// the token); with BK=64 the two halves of a tile are the two 32-blocks, so the kernel reads two
// scales per token per tile -- one per k-step pair.
//
// One thread owns one group. The 32 BF16 values arrive as four uint4 loads, and the abs-max runs
// on the raw bit patterns: masking the sign bit orders non-negative BF16 exactly as the floats
// order, so the integer max is the float max without a single conversion. The quantise path keeps
// the scalar float arithmetic of the first version untouched (convert, multiply by 127/amax,
// round-to-nearest, clamp), so codes and scales are bit-identical to it; only the loads widened
// and the max moved to integers. The first version's per-element scalar loads ran this pass at
// ~160 GB/s -- measured on the RTX 3090 Ti at the production planes ([17408,2048]: 695 us,
// [5120,2048]: 208 us) -- which made the four per-projection quantise passes ~4.6% of a prefill
// chunk; the vector form moves the same planes near streaming rate (143 / 43 us).
// ---------------------------------------------------------------------------
constexpr int kNvfp4Int8QuantBlock = 32;

__device__ __forceinline__ unsigned nvfp4_int8_bf16_absmax_word(unsigned word, unsigned m) {
    const unsigned both = word & 0x7FFF7FFFu; // abs of the two packed BF16 lanes
    m = max(m, both & 0xFFFFu);
    return max(m, both >> 16);
}

// One 32-column group: the integer abs-max of its four uint4 loads and the two uint4 of s8 codes
// they quantise to. Both activation layouts call this and only differ in where the results land,
// so the codes and the scale of a group cannot drift between the two-plane and the MMQ-row form.
// The values are re-read through L1 rather than aliased from the staged registers: taking the
// address of a local uint4 would push it to local memory.
__device__ __forceinline__ void nvfp4_int8_quantize_group(const uint4* source, float& scale,
                                                          uint4 (&packed)[2]) {
    const uint4 w0 = source[0];
    const uint4 w1 = source[1];
    const uint4 w2 = source[2];
    const uint4 w3 = source[3];
    unsigned m = 0;
    m = nvfp4_int8_bf16_absmax_word(w0.x, m);
    m = nvfp4_int8_bf16_absmax_word(w0.y, m);
    m = nvfp4_int8_bf16_absmax_word(w0.z, m);
    m = nvfp4_int8_bf16_absmax_word(w0.w, m);
    m = nvfp4_int8_bf16_absmax_word(w1.x, m);
    m = nvfp4_int8_bf16_absmax_word(w1.y, m);
    m = nvfp4_int8_bf16_absmax_word(w1.z, m);
    m = nvfp4_int8_bf16_absmax_word(w1.w, m);
    m = nvfp4_int8_bf16_absmax_word(w2.x, m);
    m = nvfp4_int8_bf16_absmax_word(w2.y, m);
    m = nvfp4_int8_bf16_absmax_word(w2.z, m);
    m = nvfp4_int8_bf16_absmax_word(w2.w, m);
    m = nvfp4_int8_bf16_absmax_word(w3.x, m);
    m = nvfp4_int8_bf16_absmax_word(w3.y, m);
    m = nvfp4_int8_bf16_absmax_word(w3.z, m);
    m = nvfp4_int8_bf16_absmax_word(w3.w, m);
    const float amax = fmaxf(__uint_as_float(m << 16), 1.0e-20F);
    scale            = amax / 127.0F;
    const float inv  = 127.0F / amax;
    const __nv_bfloat16* values = reinterpret_cast<const __nv_bfloat16*>(source);
    auto* bytes                 = reinterpret_cast<std::int8_t*>(packed);
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        const float value = __bfloat162float(values[j]) * inv;
        bytes[j] = static_cast<std::int8_t>(max(-127, min(127, __float2int_rn(value))));
    }
}

// A padding row of a ragged token tile: zero the activation planes for one token and do not touch
// x, whose rows stop at the caller's real token count. Zero is the only safe filler -- e2m1 code 0
// is the value 0, so a padded row contributes nothing to any accumulator, and the row is never
// stored, so it merely has to be defined. The vector path needs 16-byte alignment, which the
// route's own k%64==0 gate and the arena's 256-byte alignment already give; the scalar loop is the
// same defensive fallback the quantiser below uses for exotic callers.
static __device__ void zero_nvfp4_int8_activation_row(std::int8_t* __restrict__ codes,
                                                     float* __restrict__ scales,
                                                     std::int32_t token, std::int32_t k,
                                                     std::int32_t groups) {
    std::int8_t* const row = codes + static_cast<std::size_t>(token) * k;
    if ((k & 15) == 0 && (reinterpret_cast<std::uintptr_t>(row) & 15) == 0) {
        auto* words = reinterpret_cast<uint4*>(row);
        for (std::int32_t v = static_cast<std::int32_t>(threadIdx.x); v < (k >> 4);
             v += static_cast<std::int32_t>(blockDim.x)) {
            words[v] = make_uint4(0u, 0u, 0u, 0u);
        }
    } else {
        for (std::int32_t b = static_cast<std::int32_t>(threadIdx.x); b < k;
             b += static_cast<std::int32_t>(blockDim.x)) {
            row[b] = 0;
        }
    }
    for (std::int32_t g = static_cast<std::int32_t>(threadIdx.x); g < groups;
         g += static_cast<std::int32_t>(blockDim.x)) {
        scales[static_cast<std::size_t>(token) * groups + g] = 0.0F;
    }
}

// static: the header is included by several .cu files (linear, linear_swiglu, linear_add,
// attn/gdn input projections); without internal linkage each TU emits its own host stub and
// kernel symbol and the bundle exe link fails with LNK2005. Repo precedent: rope.cuh:230.
static __global__ void quantize_nvfp4_int8_grouped(const __nv_bfloat16* __restrict__ x,
                                            std::int32_t tokens, std::int32_t k,
                                            std::int8_t* __restrict__ codes,
                                            float* __restrict__ scales) {
    const std::int32_t token = blockIdx.x;
    const std::int32_t groups = k >> 5;
    // Launched over the padded width, so every row the mainloop will read is written: real tokens
    // quantise, the surplus rows zero.
    if (token >= tokens) {
        zero_nvfp4_int8_activation_row(codes, scales, token, k, groups);
        return;
    }
    // The vector path needs 32-byte alignment on both planes; k%32==0 keeps every group's
    // load and store 32-byte aligned for any token. The layer buffers are arena-aligned, so
    // the guard is uniform per launch and the scalar fallback only covers exotic callers.
    if ((k & 31) == 0 && (reinterpret_cast<std::uintptr_t>(x) & 31) == 0 &&
        (reinterpret_cast<std::uintptr_t>(codes) & 31) == 0) {
        for (std::int32_t g = threadIdx.x; g < groups; g += blockDim.x) {
            const uint4* source = reinterpret_cast<const uint4*>(
                x + static_cast<std::size_t>(token) * k + static_cast<std::size_t>(g) * 32);
            float scale  = 0.0F;
            uint4 packed[2];
            nvfp4_int8_quantize_group(source, scale, packed);
            scales[static_cast<std::size_t>(token) * groups + g] = scale;
            std::int8_t* dst = codes + static_cast<std::size_t>(token) * k +
                               static_cast<std::size_t>(g) * 32;
            reinterpret_cast<uint4*>(dst)[0] = packed[0];
            reinterpret_cast<uint4*>(dst)[1] = packed[1];
        }
        return;
    }
    for (std::int32_t g = threadIdx.x; g < groups; g += blockDim.x) {
        const __nv_bfloat16* source = x + static_cast<std::size_t>(token) * k + g * 32;
        float amax                  = 0.0F;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            amax = fmaxf(amax, fabsf(__bfloat162float(source[j])));
        }
        amax = fmaxf(amax, 1.0e-20F);
        scales[static_cast<std::size_t>(token) * groups + g] = amax / 127.0F;
        const float inv  = 127.0F / amax;
        std::int8_t* dst = codes + static_cast<std::size_t>(token) * k +
                           static_cast<std::size_t>(g) * 32;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const float value = __bfloat162float(source[j]) * inv;
            dst[j] = static_cast<std::int8_t>(max(-127, min(127, __float2int_rn(value))));
        }
    }
}

// ---------------------------------------------------------------------------
// Row layout. A block stages 128 consecutive output rows and every thread's fragment tiles land
// on rows the warp owns. The block's J tokens are shared by NT token warps, each taking the NIT
// eight-token tile_C columns its participation buys; the remaining 8/NT warps take consecutive
// output-row bands of kWarpRows rows, held as MT fragment tiles 16 rows apart.
//
// J = 128 reproduces the layout this kernel shipped with: NT = 2 token warps of 64 tokens against
// four row bands of 32 rows. Narrower tiles coalesce the token axis into one warp (NT = 1) and spend
// the seven warps that frees on eight row bands of 16 rows -- same 128 output rows, same eight
// warps, one MT fragment per warp instead of two. That is the whole point of the ladder: the row
// tile, the staging and the weight traffic do not move, and the tensor-core work drops in
// proportion to J because the activation side of every MMA is J tokens wide instead of 128.
// ---------------------------------------------------------------------------
template <int J>
struct Nvfp4Int8RowPolicy {
    static_assert(nvfp4_int8_j_admitted(J), "J must be a stage of the token-tile ladder");

    static constexpr int kNTiles   = J / 8;             // eight-token MMA columns in the tile
    static constexpr int NT        = (kNTiles + 7) / 8; // token warps the tile keeps busy
    static constexpr int NIT       = kNTiles / NT;      // tile_C columns per token warp
    static constexpr int kRowWarps = detail::kInt8Threads / 32 / NT;
    static constexpr int kWarpRows = detail::kInt8TileRows / kRowWarps;
    static constexpr int MT        = kWarpRows / 16;

    static_assert(NT > 0 && NT <= detail::kInt8Threads / 32, "warp budget must cover the warps");
    static_assert(MT * 16 * kRowWarps == detail::kInt8TileRows, "row bands must tile the tile");
    static_assert(NIT * NT * 8 == J, "token warps must tile the token tile");

    static constexpr int kStagedRows    = detail::kInt8TileRows;
    static constexpr int kOutputRows    = detail::kInt8TileRows;
    static constexpr int kFragmentStep  = 16;
    static constexpr int kWarpRowStride = kWarpRows;

    static __device__ int weight_row(int begin, int local) { return begin + local; }

    // Staged row of fragment tile (wm, m) for lane row `gid`.
    static __device__ int fragment_row(int wm, int m, int gid) {
        return wm * kWarpRowStride + gid + m * kFragmentStep;
    }

    // First token column of tile_C column index `n` for token warp `wt`.
    static __device__ int token_column(int wt, int n, int tig) {
        return wt * (NIT * 8) + n * 8 + tig * 2;
    }
};

namespace detail {

// Shared launch geometry lives in nvfp4_int8_grouped_host.h so host-built plan/dispatch
// translation units can query admission and workspace sizes.

// A uint4 read through a char* is emitted as four 32-bit loads unless the compiler can prove the
// alignment, which it cannot through this pointer arithmetic.
__device__ __forceinline__ uint4 int8_lds128(const void* pointer) {
    uint4 value;
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(pointer));
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "r"(address));
    return value;
}

__device__ __forceinline__ uint2 int8_lds64(const void* pointer) {
    uint2 value;
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(pointer));
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];"
                 : "=r"(value.x), "=r"(value.y)
                 : "r"(address));
    return value;
}

__device__ __forceinline__ void mma_s8_k16(int& c0, int& c1, int& c2, int& c3, unsigned a0,
                                           unsigned a1, unsigned b0) {
#if defined(NINFER_SM70_DEVICE)
    sm70::mma_m16n8k16_s8(c0, c1, c2, c3, a0, a1, b0);
#elif defined(NINFER_SM75)
    // m16n8k16 splits into two m8n8k16 over the row halves: a0 is row g, a1 is row g+8 (the same
    // contract mma_s8 uses for its k32 form), so the four accumulators pair up without reordering.
    sm75::mma_s8_k16(c0, c1, a0, b0);
    sm75::mma_s8_k16(c2, c3, a1, b0);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
#endif
}

} // namespace detail

// ---------------------------------------------------------------------------
// Main kernel. One block covers kInt8TileRows output rows x J tokens x one k-tile
// of kInt8TileK columns per iteration, where J is the stage of the token-tile ladder the caller's
// width selects (nvfp4_int8_select_j) -- see Nvfp4Int8RowPolicy for how the block's eight warps are
// spread over those two axes. There is no predicated tail inside the mainloop, so the surplus
// columns of a padded tile are computed and dropped by the epilogue guard.
// The output adapter owns the projection's epilogue -- a plain BF16 row store for the Linear and
// LinearAdd routes, an FP32 plane for the gate half of gate_up, and silu(gate)*up read back from
// that plane for its up half.
//
// The CTA is ggml-MMQ shaped: 256 threads / 8 warps, each warp owning kWarpRows x (J/NT) tokens
// when J = 128 that is the 4x2 grid it has always used -- 32 rows x 64 tokens, MT=2, NT=2, NIT=8,
// acc[2][8][4] = 64 floats. The 128-register ceiling of the old 512-thread CTA (512 threads * 128 =
// the SM's whole register file) is gone, so the kernel hoists the A operands (weights) for the whole
// BK=64 tile with lds128 and streams the activations one n-tile at a time. The measured shared-read
// census per warp and tile falls from ~40 LDS on the 512-thread CTA (A 8 + B 8 + ws 8 + xs 16) to 24
// lds128 (A 4 + ws 4 + xs 8 + B 8). The staging pipeline runs two tiles deep, so a tile's LDG lands
// a whole extra tile of compute early.
// ---------------------------------------------------------------------------
template <int J, class Output>
__global__ __launch_bounds__(detail::kInt8Threads) void nvfp4_int8_grouped_kernel(
    const std::uint8_t* __restrict__ weight_codes, const std::uint8_t* __restrict__ weight_scales,
    const std::int8_t* __restrict__ x_codes, const float* __restrict__ x_scales, float alpha_half,
    Output output, std::int32_t k, std::int32_t tokens) {
    using Policy = Nvfp4Int8RowPolicy<J>;
    namespace d  = detail;
    constexpr int BM      = d::kInt8TileRows;
    constexpr int BN      = J;
    constexpr int MT      = Policy::MT;
    constexpr int N_IT    = Policy::NIT;
    constexpr int NT      = Policy::NT;
    constexpr int kSRow   = d::kInt8SRow;
    constexpr int kThreads = d::kInt8Threads;
    // A 128x64 tile is staged as BM x 4 k-quarters 16-byte slots; 256 threads sweep them in two
    // passes of the same (row, quarter) mapping the 512-thread CTA used, so the smem layout and
    // its bank behaviour are untouched.
    constexpr int kStagingPasses = (BM * 4) / kThreads;
    static_assert((BM * 4) % kThreads == 0, "staging sweep must cover the whole tile");

    extern __shared__ __align__(16) unsigned char int8_smem[];
    auto* sa = reinterpret_cast<std::int8_t*>(int8_smem);      // [BM][kSRow] weight codes
    auto* sb = sa + BM * kSRow;                                // [BM][kSRow] activation codes
    auto* sws = reinterpret_cast<float*>(sb + BM * kSRow);     // [BM][4] weight group scales
    auto* sxs = sws + BM * d::kInt8ScaleGroups;                // [BM][2] per-32 activation scales

    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm   = warp / NT; // row band the warp owns
    const int wt   = warp % NT; // token slice of the band the warp owns
    const int gid  = lane >> 2; // 0..7, row inside a 16-row tile
    const int tig  = lane & 3;  // 0..3, k quarter

    const int row_block = static_cast<int>(blockIdx.x) * Policy::kOutputRows;
    const int col_block = static_cast<int>(blockIdx.y) * BN;
    const int groups    = k >> 6; // 64-column k-tiles == weight scale tiles

    // The block is a whole 128-row output tile, so a segmented adapter binds its segment once,
    // up front: resolving it here (not in the epilogue) lets the unused segment pointers die
    // before the mainloop and keeps the adapter off the register critical path.
    const auto epilogue = linear_output_tile<Policy::kOutputRows>(output, row_block);

    // Register-staged prefetch: the codes/scales for tile g+1 issue before the MMAs for tile g,
    // so their latency hides under compute instead of stalling on the barrier. The activation
    // scales deliberately stay OUT of the Stage and are read straight from global in
    // store_stage: staging them here would lengthen the already-covered staging dependency
    // without shortening the mainloop's critical path.
    struct Stage {
        uint2 w;
        uint4 x;
        unsigned ws;
    };
    const auto load_stage = [&](int tidw, int tile, Stage& stage) {
        const int ld_row = tidw >> 2; // 0..127
        const int ld_q   = tidw & 3;  // 0..3, sixteen codes each
        stage.w          = *reinterpret_cast<const uint2*>(
            weight_codes + static_cast<std::size_t>(Policy::weight_row(row_block, ld_row)) * (k / 2) +
            static_cast<std::size_t>(tile) * (d::kInt8TileK / 2) + ld_q * 8);
        // Only the first J rows of the activation tile exist -- the plane was allocated for the
        // padded width, so a row past J would be a read past it. The weight side keeps all BM rows
        // regardless: that is what the title of the loop is, and it is why the same (row, quarter)
        // sweep stays unconditional while this half of it does not.
        stage.x = make_uint4(0u, 0u, 0u, 0u);
        if (ld_row < BN) {
            stage.x = *reinterpret_cast<const uint4*>(
                x_codes + static_cast<std::size_t>(col_block + ld_row) * k +
                static_cast<std::size_t>(tile) * d::kInt8TileK + ld_q * 16);
        }
        stage.ws = 0u;
        if (tidw < BM) {
            // group%4 == 0: one 4-byte load carries the k-tile's four group scales.
            stage.ws = *reinterpret_cast<const unsigned*>(
                weight_scales + nvfp4_scale_byte_offset(Policy::weight_row(row_block, tidw), tile * 4, k));
        }
    };
    // Shared rows hold k permuted so a thread's four MMA operand words for one row are adjacent:
    // k = ld_q*16 + i*4 + j lands at byte ld_q*4 + i*16 + j, and lds128 at tig*16 returns the four
    // k-quarters in order.
    const auto store_stage = [&](int tidw, const Stage& stage, int tile) {
        const int ld_row = tidw >> 2;
        const int ld_q   = tidw & 3;
        unsigned q[4];
        e2m1_word(stage.w.x, q[0], q[1]);
        e2m1_word(stage.w.y, q[2], q[3]);
        auto* weight_row_smem = sa + ld_row * kSRow + ld_q * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            *reinterpret_cast<unsigned*>(weight_row_smem + i * 16) = q[i];
        }
        const unsigned x_words[4] = {stage.x.x, stage.x.y, stage.x.z, stage.x.w};
        if (ld_row < BN) {
            auto* x_row_smem = sb + ld_row * kSRow + ld_q * 4;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                *reinterpret_cast<unsigned*>(x_row_smem + i * 16) = x_words[i];
            }
        }
        if (tidw < BM) {
            float4 decoded;
            decoded.x = kNvfp4E4M3ScaleLut[stage.ws & 0xFFu];
            decoded.y = kNvfp4E4M3ScaleLut[(stage.ws >> 8) & 0xFFu];
            decoded.z = kNvfp4E4M3ScaleLut[(stage.ws >> 16) & 0xFFu];
            decoded.w = kNvfp4E4M3ScaleLut[stage.ws >> 24];
            *reinterpret_cast<float4*>(sws + tidw * 4) = decoded;
            if (tidw < BN) {
                // One 8-byte load carries both 32-block scales of the tile (k%64==0 makes
                // token*k/32 even, so the pair is 8-byte aligned).
                const uint2 xs_pair_raw = *reinterpret_cast<const uint2*>(
                    x_scales + (col_block + tidw) * (k >> 5) + tile * 2);
                *reinterpret_cast<uint2*>(sxs + tidw * 2) = xs_pair_raw;
            }
        }
    };

    float acc[MT][N_IT][4]; // m: rows of the warp band, n: eight-token columns of the band
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int n = 0; n < N_IT; ++n)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[m][n][j] = 0.0F;

    Stage staging;
#pragma unroll
    for (int t2 = 0; t2 < kStagingPasses; ++t2) {
        load_stage(tid + t2 * kThreads, 0, staging);
        store_stage(tid + t2 * kThreads, staging, 0);
    }
    // Two-deep staging pipeline. A depth-1 prefetch hides one tile's worth of LDG latency, but
    // the staging decode + STS still sits between two barriers; issuing the tile-after-next here
    // gives it a whole extra tile of compute to land under. The loop is written as explicit
    // low/high slots (not an indexed array) so the Stage sets stay in registers instead of
    // spilling to local memory.
    Stage pend_lo[kStagingPasses];
    Stage pend_hi[kStagingPasses];
    if (groups > 1) {
#pragma unroll
        for (int t2 = 0; t2 < kStagingPasses; ++t2) {
            load_stage(tid + t2 * kThreads, 1, pend_lo[t2]);
        }
    }
    __syncthreads();

    const auto compute_tile = [&]() {
        // The A operands and weight scales are hoisted for the whole BK=64 tile (four k-steps):
        // two lds128 per m row-group instead of an 8-byte pair per half-pass.
        unsigned af[MT][4][2];      // [m][k-step 0..3][lo/hi row]
        float    ws_pair[MT][2][4]; // [m][row 0/1][k-step 0..3]
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int r0   = Policy::fragment_row(wm, m, gid);
            const uint4 lo = d::int8_lds128(sa + r0 * kSRow + tig * 16);
            const uint4 hi = d::int8_lds128(sa + (r0 + 8) * kSRow + tig * 16);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                af[m][ks][0] = (&lo.x)[ks];
                af[m][ks][1] = (&hi.x)[ks];
            }
            // The staged row's four FP32 group scales; read as raw words, not numerically
            // converted (a float conversion of the raw bit pattern of 3.5 yields ~1.08e9).
            const uint4 w0 = d::int8_lds128(sws + r0 * 4);
            const uint4 w1 = d::int8_lds128(sws + (r0 + 8) * 4);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                ws_pair[m][0][ks] = __uint_as_float((&w0.x)[ks]);
                ws_pair[m][1][ks] = __uint_as_float((&w1.x)[ks]);
            }
        }

        // One n at a time: an lds128 of the activation codes carries all four k-steps of the
        // 64-column tile, and an lds128 of the two token rows' per-32 scales carries both
        // halves. Keeping B and xs transient per n -- instead of staging a whole half-pass of
        // eight n-tiles -- drops the activation shared reads from 16 LDS.64 to 8 LDS.128 per
        // warp/tile and frees the registers the old per-half bf[8][2] + xs_pair[8][2] held. The
        // A operands stay hoisted for the whole tile, so the weights are still read once per
        // tile and reused across all 64 tokens the warp owns.
#pragma unroll
        for (int n = 0; n < N_IT; ++n) {
            const uint4 b4 = d::int8_lds128(sb + ((wt * N_IT + n) * 8 + gid) * kSRow + tig * 16);
            const int c0   = (wt * N_IT + n) * 8 + tig * 2;
            const uint4 x4 = d::int8_lds128(sxs + c0 * 2);
            // sxs is laid out [token][half]: token c0 half 0/1 then token c0+1 half 0/1.
            const float xa[2][2] = {{__uint_as_float(x4.x), __uint_as_float(x4.y)},
                                   {__uint_as_float(x4.z), __uint_as_float(x4.w)}};
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const float xa0 = xa[0][half];
                const float xa1 = xa[1][half];
#pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    const int kk      = half * 2 + ks;
                    const unsigned bw = (&b4.x)[kk];
#pragma unroll
                    for (int m = 0; m < MT; ++m) {
                        const float w0 = ws_pair[m][0][kk];
                        const float w1 = ws_pair[m][1][kk];
                        int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                        d::mma_s8_k16(s0, s1, s2, s3, af[m][kk][0], af[m][kk][1], bw);
                        acc[m][n][0] = fmaf(static_cast<float>(s0), w0 * xa0, acc[m][n][0]);
                        acc[m][n][1] = fmaf(static_cast<float>(s1), w0 * xa1, acc[m][n][1]);
                        acc[m][n][2] = fmaf(static_cast<float>(s2), w1 * xa0, acc[m][n][2]);
                        acc[m][n][3] = fmaf(static_cast<float>(s3), w1 * xa1, acc[m][n][3]);
                    }
                }
            }
        }
    };

    // Pair-unrolled mainloop: pend_lo holds the tile the even iteration will store, pend_hi the
    // odd iteration's. Each iteration issues the tile two ahead into the other slot, then
    // computes, then stores the ready slot back to smem behind a barrier.
    int tile = 0;
    for (; tile + 1 < groups; tile += 2) {
        if (tile + 2 < groups) {
#pragma unroll
            for (int t2 = 0; t2 < kStagingPasses; ++t2) {
                load_stage(tid + t2 * kThreads, tile + 2, pend_hi[t2]);
            }
        }
        compute_tile();
        __syncthreads();
#pragma unroll
        for (int t2 = 0; t2 < kStagingPasses; ++t2) {
            store_stage(tid + t2 * kThreads, pend_lo[t2], tile + 1);
        }
        __syncthreads();

        if (tile + 3 < groups) {
#pragma unroll
            for (int t2 = 0; t2 < kStagingPasses; ++t2) {
                load_stage(tid + t2 * kThreads, tile + 3, pend_lo[t2]);
            }
        }
        compute_tile();
        __syncthreads();
        if (tile + 2 < groups) {
#pragma unroll
            for (int t2 = 0; t2 < kStagingPasses; ++t2) {
                store_stage(tid + t2 * kThreads, pend_hi[t2], tile + 2);
            }
            __syncthreads();
        }
    }
    if (tile < groups) {
        compute_tile();
        __syncthreads();
    }

    // alpha carries 0.5/weight_scale_divisor; the activation scales already left through the
    // per-group rescale. The adapter owns everything the projection adds on top.
    //
    // `tokens` is the caller's real width, not the padded width the grid was built from: a forced
    // route runs whole 128-token tiles at any width, so the surplus columns are computed and
    // discarded. They cannot simply be stored -- linear_add's store reads back the residual it is
    // about to update, so an unguarded padding column is an out-of-bounds read as well as a write,
    // and the swiglu gate plane would leak uninitialised columns into the up projection's silu.
    // c0 rises with n for a fixed thread, so one break retires the rest of the band; for a width
    // that already lands on a whole tile every guard passes and the stores are exactly what they
    // were before padding existed.
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int row0 = row_block + Policy::fragment_row(wm, m, gid);
        const int row1 = row0 + 8;
#pragma unroll
        for (int n = 0; n < N_IT; ++n) {
            const int c0 = col_block + Policy::token_column(wt, n, tig);
            if (c0 >= tokens) { break; }
            epilogue.store(row0, c0, acc[m][n][0] * alpha_half);
            epilogue.store(row1, c0, acc[m][n][2] * alpha_half);
            if (c0 + 1 < tokens) {
                epilogue.store(row0, c0 + 1, acc[m][n][1] * alpha_half);
                epilogue.store(row1, c0 + 1, acc[m][n][3] * alpha_half);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Launch front end for the token-tile ladder. It exists so no route hard-codes a tile stage the way
// every one of them used to hard-code 128: the width picks the stage (nvfp4_int8_select_j), the grid
// and the padding follow it, and adding a stage is one line here rather than a sweep of six routes.
// A J that no width can select would be a compile-time error in every instantiation instead of a
// runtime surprise, so the switch below is written over exactly the admitted stages.
// ---------------------------------------------------------------------------
template <int J, class Output>
inline void nvfp4_int8_grouped_launch_j(const std::uint8_t* weight_codes,
                                        const std::uint8_t* weight_scales,
                                        const std::int8_t* x_codes, const float* x_scales,
                                        float alpha_half, Output output, std::int32_t k,
                                        std::int32_t tokens, std::int32_t rows,
                                        cudaStream_t stream) {
    static_assert(nvfp4_int8_j_admitted(J), "J must be a stage of the token-tile ladder");
    const std::int32_t padded = nvfp4_int8_padded_tokens(tokens);
    const dim3 grid(rows / detail::kInt8TileRows, padded / J);
    nvfp4_int8_grouped_kernel<J, Output>
        <<<grid, detail::kInt8Threads, nvfp4_int8_grouped_smem_bytes(), stream>>>(
            weight_codes, weight_scales, x_codes, x_scales, alpha_half, output, k, tokens);
}

template <class Output>
inline void nvfp4_int8_grouped_launch(const std::uint8_t* weight_codes,
                                      const std::uint8_t* weight_scales,
                                      const std::int8_t* x_codes, const float* x_scales,
                                      float alpha_half, Output output, std::int32_t k,
                                      std::int32_t tokens, std::int32_t rows, cudaStream_t stream) {
    switch (nvfp4_int8_select_j(tokens)) {
        case 8:
            nvfp4_int8_grouped_launch_j<8, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                   alpha_half, output, k, tokens, rows, stream);
            return;
        case 16:
            nvfp4_int8_grouped_launch_j<16, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                    alpha_half, output, k, tokens, rows, stream);
            return;
        case 24:
            nvfp4_int8_grouped_launch_j<24, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                    alpha_half, output, k, tokens, rows, stream);
            return;
        case 32:
            nvfp4_int8_grouped_launch_j<32, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                    alpha_half, output, k, tokens, rows, stream);
            return;
        case 40:
            nvfp4_int8_grouped_launch_j<40, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                    alpha_half, output, k, tokens, rows, stream);
            return;
        case 48:
            nvfp4_int8_grouped_launch_j<48, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                    alpha_half, output, k, tokens, rows, stream);
            return;
        case 64:
            nvfp4_int8_grouped_launch_j<64, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                    alpha_half, output, k, tokens, rows, stream);
            return;
        case 80:
            nvfp4_int8_grouped_launch_j<80, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                    alpha_half, output, k, tokens, rows, stream);
            return;
        case 96:
            nvfp4_int8_grouped_launch_j<96, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                    alpha_half, output, k, tokens, rows, stream);
            return;
        case 112:
            nvfp4_int8_grouped_launch_j<112, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                     alpha_half, output, k, tokens, rows, stream);
            return;
        case 128:
            nvfp4_int8_grouped_launch_j<128, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                     alpha_half, output, k, tokens, rows, stream);
            return;
        default:
            break;
    }
    throw std::invalid_argument("nvfp4 int8 grouped: no admissible token tile for T=" +
                                std::to_string(tokens));
}

} // namespace ninfer::ops::detail::unified
