#include "ops/linear/q6/q6_fp32_launch.cuh"
#include "ops/linear/q6/q6_instances.cuh"
#include "ops/linear/q6/q6_shapes.h"

namespace ninfer::ops::detail {
namespace {

void launch_mma_r64_t112(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    // The full tile and the capacity tail have different measured warp mappings.
    if (x.ne[1] % 112 == 0) {
        launch_q6_fp32_mma<q6_instances::MmaR64T112>(x, weight, out, stream);
    } else {
        launch_q6_fp32_mma<q6_instances::MmaR64T112Partial>(x, weight, out, stream);
    }
}

} // namespace

// The vocabulary head's FP32 logits on the schedules its BF16 tables take at each width
// (n248320_k5120.cpp and unified.cpp).

Q6Launch select_q6_n248320_k5120_fp32(std::int32_t tokens) {
    using namespace q6_instances;
    if (tokens <= 4) return launch_q6_fp32_simt<SimtR8T4Cg>;
    if (tokens <= 5) return launch_q6_fp32_simt<SimtR8T5Cg>;
    if (tokens <= 6) return launch_q6_fp32_simt<SimtR8T6Cg>;
    if (tokens <= 7) return launch_q6_fp32_simt<SimtR8T7Cg>;
    if (tokens <= 16) return launch_q6_fp32_mma<MmaR64T16K128>;
    if (tokens <= 24) return launch_q6_fp32_mma<MmaR64T24K128>;
    if (tokens <= 32) return launch_q6_fp32_mma<MmaR64T32K128>;
    if (tokens <= 48) return launch_q6_fp32_mma<MmaR64T48K128>;
    return launch_q6_fp32_mma<MmaR64T128>;
}

Q6Launch select_q6_n248320_k5120_unified_fp32(std::int32_t tokens) {
    using namespace q6_instances;
    if (tokens <= 2) return launch_q6_fp32_simt<SimtR8T4>;
    if (tokens <= 8) return launch_q6_fp32_sliced_k_mma<SlicedR16T8W4S2>;
    if (tokens <= 16) return launch_q6_fp32_sliced_k_mma<SlicedR32T16W4S2>;
    if (tokens <= 32) return launch_q6_fp32_sliced_k_mma<SlicedR32T32W4S1>;
    if (tokens <= 48) return launch_q6_fp32_mma<MmaR64T48K128>;
    if (tokens <= 56) return launch_q6_fp32_mma<MmaR64T56K128>;
    if (tokens <= 64) return launch_q6_fp32_mma<MmaR64T64K128>;
    if (tokens <= 80) return launch_q6_fp32_mma<MmaR64T80>;
    if (tokens <= 96) return launch_q6_fp32_mma<MmaR64T96>;
    if (tokens <= 112) return launch_mma_r64_t112;
    return launch_q6_fp32_mma<MmaR64T128>;
}

} // namespace ninfer::ops::detail
