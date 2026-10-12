// Decode-route instances of linear_pair over GGML weights (ggml_shapes.h kGgmlPairShapes): every
// ordered pair of the pair formats at K = 2560 and T = 1..8, in one launch per call.

#include "ops/linear/ggml/ggml_launch.h"
#include "ops/linear/ggml/ggml_mmvq.cuh"
#include "ops/linear/ggml/ggml_shapes.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kPairK = 2560;
static_assert(kGgmlPairShapes[0].k == kPairK && kGgmlPairShapes[1].k == kPairK,
              "the pair instances are compiled for K = 2560");

template <QType F1, QType F2>
void launch_t(const std::uint8_t* w1, const std::uint8_t* w2, const __nv_bfloat16* x, __nv_bfloat16* out1,
              __nv_bfloat16* out2, std::int32_t n, std::int32_t t, cudaStream_t stream) {
    switch (t) {
    case 1: return ggml::launch_mmvq_pair_instance<F1, F2, kPairK, 1>(w1, w2, x, out1, out2, n, stream);
    case 2: return ggml::launch_mmvq_pair_instance<F1, F2, kPairK, 2>(w1, w2, x, out1, out2, n, stream);
    case 3: return ggml::launch_mmvq_pair_instance<F1, F2, kPairK, 3>(w1, w2, x, out1, out2, n, stream);
    case 4: return ggml::launch_mmvq_pair_instance<F1, F2, kPairK, 4>(w1, w2, x, out1, out2, n, stream);
    case 5: return ggml::launch_mmvq_pair_instance<F1, F2, kPairK, 5>(w1, w2, x, out1, out2, n, stream);
    case 6: return ggml::launch_mmvq_pair_instance<F1, F2, kPairK, 6>(w1, w2, x, out1, out2, n, stream);
    case 7: return ggml::launch_mmvq_pair_instance<F1, F2, kPairK, 7>(w1, w2, x, out1, out2, n, stream);
    case 8: return ggml::launch_mmvq_pair_instance<F1, F2, kPairK, 8>(w1, w2, x, out1, out2, n, stream);
    default: break;
    }
    throw std::invalid_argument("ggml linear_pair: T must be in [1, 8]");
}

template <QType F1>
void launch_second(QType second, const std::uint8_t* w1, const std::uint8_t* w2, const __nv_bfloat16* x,
                   __nv_bfloat16* out1, __nv_bfloat16* out2, std::int32_t n, std::int32_t t, cudaStream_t stream) {
    switch (second) {
    case QType::GGML_IQ4_XS: return launch_t<F1, QType::GGML_IQ4_XS>(w1, w2, x, out1, out2, n, t, stream);
    case QType::GGML_IQ3_S: return launch_t<F1, QType::GGML_IQ3_S>(w1, w2, x, out1, out2, n, t, stream);
    case QType::GGML_Q6_K: return launch_t<F1, QType::GGML_Q6_K>(w1, w2, x, out1, out2, n, t, stream);
    default: break;
    }
    throw std::invalid_argument("ggml linear_pair: unsupported second format");
}

} // namespace

void ggml_mmvq_pair_launch(QType first, QType second, const std::uint8_t* w1, const std::uint8_t* w2,
                           const __nv_bfloat16* x, __nv_bfloat16* out1, __nv_bfloat16* out2, std::int32_t n,
                           std::int32_t k, std::int32_t t, cudaStream_t stream) {
    if (k != kPairK) { throw std::invalid_argument("ggml linear_pair: K has no compiled instance"); }
    switch (first) {
    case QType::GGML_IQ4_XS: return launch_second<QType::GGML_IQ4_XS>(second, w1, w2, x, out1, out2, n, t, stream);
    case QType::GGML_IQ3_S: return launch_second<QType::GGML_IQ3_S>(second, w1, w2, x, out1, out2, n, t, stream);
    case QType::GGML_Q6_K: return launch_second<QType::GGML_Q6_K>(second, w1, w2, x, out1, out2, n, t, stream);
    default: break;
    }
    throw std::invalid_argument("ggml linear_pair: unsupported first format");
}

} // namespace ninfer::ops::detail
