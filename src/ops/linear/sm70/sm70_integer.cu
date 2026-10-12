#include "ops/linear/sm70/sm70_integer.h"

#include "core/device.h"
#include "core/layout.h"
#include "core/weight_view.h"
#include "ops/common/math.cuh"
#include "ops/linear/sm70/sm70_dequantize.h"
#include "ops/linear/sm70/sm70_epilogue.cuh"
#include "ops/linear/sm70/sm70_gemm.h"
#include "ops/linear/sm70/sm70_small.cuh"
#include "ops/linear/t2/t2_a8.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace ninfer::ops::detail::sm70 {
namespace {

constexpr int kGroup   = 64;  // values per activation scale
constexpr int kWarps   = 8;   // a warp per four-row panel, so a CTA owns 32 rows
constexpr int kT2Group = 128; // values per T2 weight scale (a 32-byte record)

// Element e of a 64-value group in the stored order: within each sixteen-value block, word s holds
// elements 4i + s (i = 0..3) in byte i, the order in which a T2 code word's fields come out of
// (word >> 2s) & 0x03030303.
__device__ __forceinline__ int place(int e) { return (e & ~15) | ((e & 3) << 2) | ((e & 15) >> 2); }

// One warp per (column, 64-value group): codes at [T][K] in the stored order, scale FP16(absmax /
// 127) as a float and twice the codes' sum (the offset term of integer_dot) at [T][K / 64].
__global__ void integer_quantize_kernel(const __nv_bfloat16* __restrict__ x, std::int32_t k,
                                        std::int8_t* __restrict__ codes, float* __restrict__ scales,
                                        std::int32_t* __restrict__ sums) {
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int group  = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) >> 5) +
                       (static_cast<int>(threadIdx.x) >> 5);
    const int token  = static_cast<int>(blockIdx.y);
    const int groups = k / kGroup;
    if (group >= groups) { return; }
    const std::int64_t base = static_cast<std::int64_t>(token) * k + group * kGroup;
    const float2 value =
        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + base + 2 * lane));
    float amax = fmaxf(fabsf(value.x), fabsf(value.y));
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
    }
    amax                              = fmaxf(amax, 1.0e-20F);
    const float inv                   = 127.0F / amax;
    const int low                     = max(-127, min(127, __float2int_rn(value.x * inv)));
    const int high                    = max(-127, min(127, __float2int_rn(value.y * inv)));
    codes[base + place(2 * lane)]     = static_cast<std::int8_t>(low);
    codes[base + place(2 * lane + 1)] = static_cast<std::int8_t>(high);
    int sum                           = low + high;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, mask); }
    if (lane == 0) {
        const std::int64_t at = static_cast<std::int64_t>(token) * groups + group;
        scales[at]            = __half2float(__float2half(amax / 127.0F));
        sums[at]              = 2 * sum;
    }
}

// The CTA's copy of the activations: codes [Cols][K], scales and offset terms [Cols][K / 64].
template <int Cols>
struct StagedActivations {
    const std::int8_t* codes;
    const float* scales;
    const std::int32_t* sums;
    std::int32_t groups;
};

template <int Cols>
__device__ __forceinline__ StagedActivations<Cols>
stage_activations(const std::int8_t* __restrict__ codes, const float* __restrict__ scales,
                  const std::int32_t* __restrict__ sums, std::int32_t k, std::int32_t cols) {
    extern __shared__ int4 shared_words[];
    auto* code_s      = reinterpret_cast<std::int8_t*>(shared_words);
    const int groups  = k / kGroup;
    auto* scale_s     = reinterpret_cast<float*>(code_s + Cols * k);
    auto* sum_s       = reinterpret_cast<std::int32_t*>(scale_s + Cols * groups);
    const int threads = static_cast<int>(blockDim.x);
    for (int i = static_cast<int>(threadIdx.x); i < Cols * k / 16; i += threads) {
        const int c = i / (k / 16);
        shared_words[i] =
            c < cols ? __ldg(reinterpret_cast<const int4*>(codes) + i) : make_int4(0, 0, 0, 0);
    }
    for (int i = static_cast<int>(threadIdx.x); i < Cols * groups; i += threads) {
        const bool live = i / groups < cols;
        scale_s[i]      = live ? __ldg(scales + i) : 0.0F;
        sum_s[i]        = live ? __ldg(sums + i) : 0;
    }
    __syncthreads();
    return {code_s, scale_s, sum_s, groups};
}

// The 64 codes of a T2 record half as sixteen offset words: byte i of word 4j + s is code 16j +
// 4i + s plus 2, a value in {1, 2, 3}.
__device__ __forceinline__ void offset_codes(uint4 record, unsigned (&b)[16]) {
    const unsigned words[4] = {record.x, record.y, record.z, record.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            b[4 * j + s] = ((words[j] >> (2 * s)) & 0x03030303u) ^ 0x02020202u;
        }
    }
}

// sum (code + 2) * x over a 64-value group, less 2 sum x: the exact integer dot product.
template <int Cols>
__device__ __forceinline__ int integer_dot(const unsigned (&b)[16],
                                           const StagedActivations<Cols>& x, int c, int group) {
    const int4* words = reinterpret_cast<const int4*>(
        x.codes + static_cast<std::int64_t>(c) * x.groups * kGroup + group * kGroup);
    int dot = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int4 v = words[j];
        dot          = __dp4a(static_cast<int>(b[4 * j + 0]), v.x, dot);
        dot          = __dp4a(static_cast<int>(b[4 * j + 1]), v.y, dot);
        dot          = __dp4a(static_cast<int>(b[4 * j + 2]), v.z, dot);
        dot          = __dp4a(static_cast<int>(b[4 * j + 3]), v.w, dot);
    }
    return dot - x.sums[c * x.groups + group];
}

// The lanes of a warp on the four-row panel at `row0`: lane l reads, at step s, half (l % 2) of
// row row0 + (l % 8) / 2's record for group 4s + l / 8. With the panel layout the warp's 32 reads
// are four whole 128-byte lines.
struct PanelLane {
    int row_in_panel;
    int half;
    int group_offset;
};

__device__ __forceinline__ PanelLane panel_lane(int lane) {
    return {(lane & 7) >> 1, lane & 1, lane >> 3};
}

// Adds row partials over the lanes that share a row (halves and group offsets); every lane ends
// with its row's sum.
__device__ __forceinline__ float row_sum(float value) {
    value += __shfl_xor_sync(0xffffffffu, value, 1);
    value += __shfl_xor_sync(0xffffffffu, value, 8);
    value += __shfl_xor_sync(0xffffffffu, value, 16);
    return value;
}

template <int Cols>
__device__ __forceinline__ void
accumulate_row(const RowSplitView& view, std::int32_t row, const PanelLane& p, int step,
               int t2_groups, const StagedActivations<Cols>& x, int cols, float (&acc)[Cols]) {
    const int group = 4 * step + p.group_offset;
    if (group >= t2_groups) { return; }
    const uint4 record = __ldg(
        reinterpret_cast<const uint4*>(view.record(view.base, kT2Group / 4, row, group)) + p.half);
    const float weight_scale = view.scale(row, group);
    unsigned b[16];
    offset_codes(record, b);
    const int x_group = 2 * group + p.half;
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
        if (c >= cols) { break; }
        acc[c] = fmaf(static_cast<float>(integer_dot(b, x, c, x_group)),
                      weight_scale * x.scales[c * x.groups + x_group], acc[c]);
    }
}

template <int Cols>
__launch_bounds__(kWarps * 32) __global__
    void integer_project_kernel(RowSplitView view, std::int32_t rows, std::int32_t k,
                                const std::int8_t* __restrict__ codes,
                                const float* __restrict__ scales,
                                const std::int32_t* __restrict__ sums, std::int32_t cols,
                                DestinationEpilogue epilogue) {
    const StagedActivations<Cols> x = stage_activations<Cols>(codes, scales, sums, k, cols);
    const int lane                  = static_cast<int>(threadIdx.x) & 31;
    const std::int32_t row0 =
        (static_cast<std::int32_t>(blockIdx.x) * kWarps + (static_cast<int>(threadIdx.x) >> 5)) * 4;
    if (row0 >= rows) { return; }
    const PanelLane p      = panel_lane(lane);
    const std::int32_t row = row0 + p.row_in_panel;
    const int t2_groups    = k / kT2Group;
    float acc[Cols];
#pragma unroll
    for (int c = 0; c < Cols; ++c) { acc[c] = 0.0F; }
    for (int step = 0; 4 * step < t2_groups; ++step) {
        accumulate_row<Cols>(view, row, p, step, t2_groups, x, cols, acc);
    }
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
        const float value = row_sum(acc[c]);
        if ((lane & 0x19) == 0 && c < cols) { epilogue(row, c, value); }
    }
}

// Panel rows [i0, i0 + 4) of the gate and the matching up rows.
template <int Cols>
__launch_bounds__(kWarps * 32) __global__
    void integer_swiglu_kernel(RowSplitView gate, RowSplitView up, std::int32_t rows,
                               std::int32_t up_first, std::int32_t k,
                               const std::int8_t* __restrict__ codes,
                               const float* __restrict__ scales,
                               const std::int32_t* __restrict__ sums, std::int32_t cols,
                               RowFactor gate_factor, RowFactor up_factor,
                               __nv_bfloat16* __restrict__ out) {
    const StagedActivations<Cols> x = stage_activations<Cols>(codes, scales, sums, k, cols);
    const int lane                  = static_cast<int>(threadIdx.x) & 31;
    const std::int32_t i0 =
        (static_cast<std::int32_t>(blockIdx.x) * kWarps + (static_cast<int>(threadIdx.x) >> 5)) * 4;
    if (i0 >= rows) { return; }
    const PanelLane p    = panel_lane(lane);
    const std::int32_t i = i0 + p.row_in_panel;
    const int t2_groups  = k / kT2Group;
    float g[Cols];
    float u[Cols];
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
        g[c] = 0.0F;
        u[c] = 0.0F;
    }
    for (int step = 0; 4 * step < t2_groups; ++step) {
        accumulate_row<Cols>(gate, i, p, step, t2_groups, x, cols, g);
        accumulate_row<Cols>(up, up_first + i, p, step, t2_groups, x, cols, u);
    }
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
        const float gate_value = row_sum(g[c]) * gate_factor(i);
        const float up_value   = row_sum(u[c]) * up_factor(up_first + i);
        if ((lane & 0x19) == 0 && c < cols) {
            out[static_cast<std::size_t>(c) * rows + i] =
                __float2bfloat16_rn(silu(gate_value) * up_value);
        }
    }
}

RowSplitView t2_view(const Weight& w) {
    RowSplitView view;
    view.base           = static_cast<const std::uint8_t*>(w.qdata);
    view.scales         = static_cast<const __half*>(w.scales);
    view.groups_per_row = w.padded_shape[1] / kT2Group;
    view.panel_shift    = row_split_panel_shift(w.layout);
    return view;
}

std::size_t shared_bytes(std::int32_t k, int cols) {
    return static_cast<std::size_t>(cols) * k +
           static_cast<std::size_t>(cols) * (k / kGroup) * (sizeof(float) + sizeof(std::int32_t));
}

// f.template operator()<Cols>() for the call's width.
template <class F>
void with_cols(std::int32_t tokens, F&& f) {
    if (tokens == 1) {
        f.template operator()<1>();
    } else {
        f.template operator()<2>();
    }
}

// The kernels' shapes: whole T2 groups, and staged activations within the static 48 KiB a CTA
// takes without opting in (two columns of the widest registered T2 input, 17,408, need 38 KiB).
bool integer_shape(QType format, std::int32_t k, std::int32_t tokens) {
    return enabled() && format == QType::T2_G128_FP16 && tokens >= 1 &&
           tokens <= kIntegerMaxTokens && k % kT2Group == 0 &&
           shared_bytes(k, tokens) <= 48u * 1024u;
}

void require(const Activation& x, const Weight& w) {
    if (x.codes == nullptr || w.k != x.k || !integer_shape(w.qtype, w.k, x.tokens) ||
        !dequantizable(w) || w.n % 4 != 0) {
        throw std::invalid_argument("sm70 integer route: unsupported weight or activations");
    }
}

} // namespace

bool integer_format(QType format, std::int32_t k, std::int32_t tokens, LinearPolicy policy) {
    return t2_a8_admits(policy) && integer_shape(format, k, tokens);
}

bool integer_takes(const Weight& w, std::int32_t tokens, LinearPolicy policy) {
    return integer_format(w.qtype, w.k, tokens, policy) && dequantizable(w) && w.n % 4 == 0;
}

std::size_t integer_bytes(std::int32_t k, std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    const auto groups = static_cast<std::size_t>(k / kGroup) * tokens;
    (void)layout.alloc_bytes(static_cast<std::size_t>(k) * tokens);
    (void)layout.alloc_bytes(groups * sizeof(float));
    (void)layout.alloc_bytes(groups * sizeof(std::int32_t));
    return layout.peak_bytes(1);
}

void integer_prepare(const Tensor& x, WorkspaceArena& workspace, cudaStream_t stream,
                     Activation& x_out) {
    const std::int32_t k      = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    if (x.dtype != DType::BF16 || x.data == nullptr || !x.is_contiguous() || x.ne[2] != 1 ||
        x.ne[3] != 1 || tokens < 1 || tokens > kIntegerMaxTokens || k % kT2Group != 0) {
        throw std::invalid_argument("sm70 integer route: x must be a contiguous BF16 [K, T<=2]");
    }
#if defined(NINFER_SM70)
    const auto groups = static_cast<std::size_t>(k / kGroup) * tokens;
    auto* codes =
        static_cast<std::int8_t*>(workspace.alloc_bytes(static_cast<std::size_t>(k) * tokens).data);
    auto* scales = static_cast<float*>(workspace.alloc_bytes(groups * sizeof(float)).data);
    auto* sums =
        static_cast<std::int32_t*>(workspace.alloc_bytes(groups * sizeof(std::int32_t)).data);
    constexpr int kQuantizeWarps = 8;
    const dim3 grid(static_cast<unsigned>((k / kGroup + kQuantizeWarps - 1) / kQuantizeWarps),
                    static_cast<unsigned>(tokens));
    integer_quantize_kernel<<<grid, kQuantizeWarps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), k, codes, scales, sums);
    CUDA_CHECK(cudaGetLastError());
    x_out.k           = k;
    x_out.tokens      = tokens;
    x_out.codes       = codes;
    x_out.code_scales = scales;
    x_out.code_sums   = sums;
#else
    (void)workspace;
    (void)stream;
    (void)x_out;
    throw std::logic_error("sm70 integer route: this build does not target sm_70");
#endif
}

void integer_project(const Activation& x, const Weight& w,
                     std::span<const Destination> destinations, cudaStream_t stream) {
    require(x, w);
    if (destinations.empty() || destinations.size() > kMaxDestinations) {
        throw std::invalid_argument("sm70 integer route: unsupported destinations");
    }
#if defined(NINFER_SM70)
    DestinationEpilogue epilogue;
    for (const Destination& d : destinations) { epilogue.set[epilogue.count++] = d; }
    epilogue.factor = row_factor(w);
    const dim3 grid(static_cast<unsigned>((w.n / 4 + kWarps - 1) / kWarps));
    with_cols(x.tokens, [&]<int Cols>() {
        const std::size_t shared = shared_bytes(x.k, Cols);
        integer_project_kernel<Cols><<<grid, kWarps * 32, shared, stream>>>(
            t2_view(w), w.n, x.k, x.codes, x.code_scales, x.code_sums, x.tokens, epilogue);
        CUDA_CHECK(cudaGetLastError());
    });
#else
    (void)stream;
    throw std::logic_error("sm70 integer route: this build does not target sm_70");
#endif
}

void integer_swiglu(const Activation& x, const Weight& gate, const Weight* up, Tensor& out,
                    cudaStream_t stream) {
    const std::int32_t rows = out.ne[0];
    const bool joined       = up == nullptr;
    const Weight& up_weight = joined ? gate : *up;
    require(x, gate);
    require(x, up_weight);
    if (out.dtype != DType::BF16 || out.ne[1] != x.tokens || rows % 4 != 0 ||
        (joined ? gate.n != 2 * rows : (gate.n != rows || up->n != rows))) {
        throw std::invalid_argument("sm70 integer swiglu: unsupported weights or output");
    }
#if defined(NINFER_SM70)
    const dim3 grid(static_cast<unsigned>((rows / 4 + kWarps - 1) / kWarps));
    with_cols(x.tokens, [&]<int Cols>() {
        const std::size_t shared = shared_bytes(x.k, Cols);
        integer_swiglu_kernel<Cols><<<grid, kWarps * 32, shared, stream>>>(
            t2_view(gate), t2_view(up_weight), rows, joined ? rows : 0, x.k, x.codes, x.code_scales,
            x.code_sums, x.tokens, row_factor(gate), row_factor(up_weight),
            static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());
    });
#else
    (void)stream;
    throw std::logic_error("sm70 integer swiglu: this build does not target sm_70");
#endif
}

} // namespace ninfer::ops::detail::sm70
