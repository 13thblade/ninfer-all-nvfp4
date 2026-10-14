#include "ops/linear/linear_test_common.h"

#include "ops/linear/common/route_table.h"
#include "ops/linear/nvfp4/nvfp4_e4m3_scale.h"
#include "ops/linear/nvfp4/nvfp4_int8_grouped_host.h"
#include "ops/linear/nvfp4/nvfp4_mmq_host.h"
#include "ops/op_tester.h" // test::ScopedNvfp4GemmRoute

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

int run_generic_geometries();
int run_forced_padded_widths();
int verify_geometry_diagnostics();

// Two things have to be said for an int8 measurement to be real, and they are the same shape of
// hazard. (1) The int8 route lives on the unified table only: its admission, its shapes registry and
// its kernel instances are all `ops::linear::nvfp4::unified`. This binary installs no device profile,
// so without a table pin the schedule would be Legacy for most widths and every int8 invocation would
// silently run the Legacy A16 composition instead -- green, but not the route under test. (2) Which
// mainloop answers is named in the environment and an unset environment names A16, so without a route
// pin the admission tests below would be measuring a route nobody selected -- or throwing "no A16
// route to fall back to" from a capacity query, which is a configuration failure masquerading as a
// test result. Pin both for the duration; a case that wants another route says so locally and nests.
struct UnifiedInt8Route {
    UnifiedInt8Route()
        : previous((ops::detail::validate_nvfp4_gemm_env(),
                   ops::detail::force_nvfp4_gemm_route(ops::detail::Nvfp4GemmRoute::Grouped))) {
        ops::detail::force_linear_route_table(ops::detail::LinearRouteTable::Unified);
    }
    ~UnifiedInt8Route() {
        ops::detail::force_linear_route_table(std::nullopt);
        ops::detail::force_nvfp4_gemm_route(previous);
    }
    UnifiedInt8Route(const UnifiedInt8Route&)            = delete;
    UnifiedInt8Route& operator=(const UnifiedInt8Route&) = delete;

    std::optional<ops::detail::Nvfp4GemmRoute> previous;
};

// Pin a route for the duration of one block. Shared with the linear harnesses, which pin Grouped for
// an Int8Grouped shape: save/restore rather than clear means the two nest instead of fighting.
using ForcedRoute = test::ScopedNvfp4GemmRoute;

// Intervals whose widest point has a whole 128-token tile inside them. A registered geometry also
// admits the empty intervals below (it has an A16 composition to fall back to); the alignment-only
// geometry has no fallback, so only the whole-tile points inside an interval are reachable --
// verify_aligned_only_envelopes states its envelope over exactly those.
constexpr std::pair<std::int32_t, std::int32_t> kRegisteredIntervals[] = {
    {1, 4}, {2, 4}, {1, 128}, {9, 25}, {24, 129}, {128, 512}};
constexpr std::pair<std::int32_t, std::int32_t> kAlignedOnlyIntervals[] = {
    {1, 128}, {1, 129}, {9, 129}, {129, 256}, {128, 512}};

// The shared verify_workspace_envelopes covers the A16/A8/A4 policies; the int8 route's
// admission gate makes its envelope worth checking under its own policy, where ragged points
// reserve nothing and tile-aligned points reserve their exact planes.
int verify_int8_workspace_envelopes(
    std::int32_t n, std::int32_t k,
    std::span<const std::pair<std::int32_t, std::int32_t>> intervals = kRegisteredIntervals) {
    int failures = 0;
    const auto policy = ops::LinearPolicy::AllowNvfp4Int8;
    for (auto [first, last] : intervals) {
        const auto capacity =
            ops::linear_workspace_capacity_bytes(QType::NVFP4, n, k, policy, first, last);
        for (int t = first; t <= last; ++t) {
            const auto point =
                ops::linear_workspace_capacity_bytes(QType::NVFP4, n, k, policy, t, t);
            if (point > capacity) {
                std::cerr << "Linear workspace interval [" << first << ',' << last
                          << "] cannot cover T=" << t << " for [" << n << ',' << k << "]\n";
                ++failures;
                break;
            }
        }
    }
    // The MMQ mainloop quantises the same activations into one interleaved plane instead of two.
    // It costs the same K + K/8 bytes per token, so every token count it can be called on has to fit
    // inside the envelope declared above -- a silent overrun would only show up as a corrupted
    // activation plane on the shapes that actually reach the MMQ route. The width filter is now just
    // "can a route get here": MMQ pads like the grouped kernel, so the ragged widths are exactly the
    // interesting ones, and the only geometry condition left is the weight-side K % ITER_K == 0.
    for (auto [first, last] : intervals) {
        const auto capacity =
            ops::linear_workspace_capacity_bytes(QType::NVFP4, n, k, policy, first, last);
        for (int t = first; t <= last; ++t) {
            if (!(ops::detail::mmq::k_admitted(k) &&
                  ops::detail::unified::nvfp4_int8_route_admits(t))) {
                continue;
            }
            const auto tile = ops::detail::mmq::activation_bytes(t, k);
            if (tile > capacity) {
                std::cerr << "MMQ activation plane T=" << t << " needs " << tile
                          << " B, interval [" << first << ',' << last << "] declares " << capacity
                          << " for [" << n << ',' << k << "]\n";
                ++failures;
                break;
            }
        }
    }
    return failures;
}

// An unregistered alignment-only geometry has no A16 fallback, so a width its route does not admit
// has no route at all -- it throws, which is a different statement from "reserves nothing". Its
// envelope is therefore stated over the widths its route can reach: a declared interval still has to
// cover each of those, the property the A16 fallback would otherwise hide.
int verify_aligned_only_envelopes(
    std::int32_t n, std::int32_t k,
    std::span<const std::pair<std::int32_t, std::int32_t>> intervals) {
    int failures = 0;
    const auto policy = ops::LinearPolicy::AllowNvfp4Int8;
    for (auto [first, last] : intervals) {
        const auto capacity =
            ops::linear_workspace_capacity_bytes(QType::NVFP4, n, k, policy, first, last);
        for (int t = first; t <= last; ++t) {
            if (!ops::detail::unified::nvfp4_int8_route_admits(t)) { continue; }
            const auto point =
                ops::linear_workspace_capacity_bytes(QType::NVFP4, n, k, policy, t, t);
            if (point > capacity) {
                std::cerr << "Linear workspace interval [" << first << ',' << last
                          << "] cannot cover T=" << t << " for [" << n << ',' << k << "]\n";
                ++failures;
                break;
            }
        }
    }
    return failures;
}

// ---------------------------------------------------------------------------
// The per-16 E4M3 scale table, enumerated against llama.cpp's ue4m3 decode.
//
// The reference below is an independent restatement of ggml_cuda_ue4m3_to_fp32
// (ggml/src/ggml-cuda/common.cuh:847, software fallback): ldexp-based, in double, bit-compared --
// not a copy of the production expression, which is written with power-of-two factors. Every one
// of the 256 bytes is checked, including the corners llama's two paths disagree on.
// ---------------------------------------------------------------------------
double llama_ue4m3_software(std::uint8_t byte) {
    if (byte == 0u || byte == 0x7Fu) { return 0.0; }
    const int exponent = (byte >> 3) & 0x0F;
    const int mantissa = byte & 0x07;
    const double raw   = exponent == 0 ? std::ldexp(static_cast<double>(mantissa), -9)
                                       : std::ldexp(1.0 + static_cast<double>(mantissa) / 8.0,
                                                    exponent - 7);
    return raw / 2.0;
}

// The signed E4M3FN reading this table used before it took llama's semantics -- the one whose
// values the old alpha = 0.5/divisor halved. Kept here as the other side of the half-factor check.
double signed_e4m3fn(std::uint8_t byte) {
    const bool negative = (byte & 0x80u) != 0;
    const int  exponent = (byte >> 3) & 0x0F;
    const int  mantissa = byte & 0x07;
    if (exponent == 0x0F && mantissa == 0x07) { return std::nan(""); }
    const double magnitude =
        exponent == 0 ? std::ldexp(static_cast<double>(mantissa), -9)
                      : std::ldexp(1.0 + static_cast<double>(mantissa) / 8.0, exponent - 7);
    return negative ? -magnitude : magnitude;
}

std::uint32_t float_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

int verify_ue4m3_scale_table() {
    int failures = 0;
    for (int byte = 0; byte < 256; ++byte) {
        const float produced = ops::detail::nvfp4::kNvfp4E4M3Scale[byte];
        const float expected =
            static_cast<float>(llama_ue4m3_software(static_cast<std::uint8_t>(byte)));
        if (float_bits(produced) != float_bits(expected)) {
            std::cerr << "NVFP4 E4M3 scale 0x" << std::hex << byte << std::dec << ": got "
                      << produced << ", llama.cpp ue4m3 gives " << expected << '\n';
            ++failures;
        }
    }
    // The half factor moved out of alpha and into the decode: over the reachable domain -- a block
    // scale is a magnitude, so bytes above 0x7F never occur -- the new entry is exactly half the
    // old one. 0x7F is the one reachable byte that changes for another reason: NaN -> 0.0f.
    int halved = 0;
    for (int byte = 0; byte <= 0x7E; ++byte) {
        const float produced = ops::detail::nvfp4::kNvfp4E4M3Scale[byte];
        const float old_half =
            static_cast<float>(signed_e4m3fn(static_cast<std::uint8_t>(byte)) / 2.0);
        if (float_bits(produced) != float_bits(old_half)) {
            std::cerr << "NVFP4 E4M3 scale 0x" << std::hex << byte << std::dec << ": got "
                      << produced << ", half the previous decode is " << old_half << '\n';
            ++failures;
        } else {
            ++halved;
        }
    }
    if (halved != 127) {
        std::cerr << "NVFP4 E4M3 scale: " << halved << "/127 reachable bytes are exactly halved\n";
        ++failures;
    }
    if (ops::detail::nvfp4::kNvfp4E4M3Scale[0x7F] != 0.0F) {
        std::cerr << "NVFP4 E4M3 scale 0x7F (NaN) must decode to 0.0f\n";
        ++failures;
    }
    std::cout << "NVFP4 E4M3 scale table: 256/256 bytes match llama.cpp ue4m3, " << halved
              << "/127 reachable bytes are half the previous decode\n";
    return failures;
}

int run_nvfp4_int8() {
    const UnifiedInt8Route unified;
    std::vector<Invocation> invocations;
    // Ragged widths fall back to the A16 composition under the same policy: 1/4/16/64 sit under
    // the 128-token tile floor and 129/257 just past tile boundaries.
    for (int t : {1, 4, 16, 64, 129, 257})
        invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    // Whole 128-token tiles take the grouped-integer route.
    for (int t : {128, 256, 384, 512})
        invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    // Graph replay on both sides of the admission gate, plus one tile-aligned replay.
    for (int t : {128, 512})
        invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8, true});
    invocations.push_back({129, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8, true});

    int failures = 0;
    failures += run_shape("NVFP4_Int8Grouped", ActivationCompute::Int8Grouped, make_nvfp4_weight,
                          {14336, 5120, 801U, Comparison::SampledRows, true, invocations});
    failures += run_shape("NVFP4_Int8Grouped", ActivationCompute::Int8Grouped, make_nvfp4_weight,
                          {16384, 5120, 803U, Comparison::SampledRows, true, invocations});
    failures += run_shape("NVFP4_Int8Grouped", ActivationCompute::Int8Grouped, make_nvfp4_weight,
                          {34816, 5120, 804U, Comparison::SampledRows, true, invocations});
    failures += run_shape("NVFP4_Int8Grouped", ActivationCompute::Int8Grouped, make_nvfp4_weight,
                          {5120, 6144, 805U, Comparison::SampledRows, true, invocations});
    failures += run_shape("NVFP4_Int8Grouped", ActivationCompute::Int8Grouped, make_nvfp4_weight,
                          {5120, 17408, 807U, Comparison::SampledRows, true, invocations});
    for (auto [n, k] : {std::pair{14336, 5120}, std::pair{16384, 5120}, std::pair{34816, 5120},
                        std::pair{5120, 6144}, std::pair{5120, 17408}}) {
        failures += verify_int8_workspace_envelopes(n, k);
    }
    failures += run_generic_geometries();
    failures += run_forced_padded_widths();
    failures += verify_ue4m3_scale_table();
    return failures;
}

// The registration contract: a geometry outside the 27B artifact's set reaches the same
// (kernels-free) int8 mainloop through one table row, and an aligned geometry with no row at all
// reaches it through the alignment admission. Neither has an A16 composition to hide behind, which
// is what keeps the oracle comparisons above from being vacuously green.
int run_generic_geometries() {
    int failures = 0;
    // [8192, 4096] has a row (shapes/n8192_k4096_unified.cu): ragged widths take its A16 rows, the
    // whole tiles take the shared int8 mainloop.
    std::vector<Invocation> registered;
    for (int t : {1, 64, 129}) {
        registered.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    }
    for (int t : {128, 256, 512}) {
        registered.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    }
    failures += run_shape("NVFP4_Int8Grouped", ActivationCompute::Int8Grouped, make_nvfp4_weight,
                          {8192, 4096, 901U, Comparison::SampledRows, true, registered});
    failures += verify_int8_workspace_envelopes(8192, 4096);

    // [4096, 2048] has no row: the alignment admission is the only way in, and the int8 mainloop is
    // the only route behind it. An unregistered geometry therefore cannot pass by silently falling
    // back to A16 -- the policy overload refuses it.
    std::vector<Invocation> unregistered;
    for (int t : {128, 256, 384}) {
        unregistered.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    }
    failures += run_shape("NVFP4_Int8Grouped", ActivationCompute::Int8Grouped, make_nvfp4_weight,
                          {4096, 2048, 903U, Comparison::SampledRows, true, unregistered});
    failures += verify_aligned_only_envelopes(4096, 2048, kAlignedOnlyIntervals);

    failures += verify_geometry_diagnostics();
    return failures;
}

// The admission layer's diagnostics: a table miss under a policy that has no route for the
// geometry, a ragged width on the alignment-only geometry, and a geometry that misses either
// alignment axis, each named rather than reported as an unqualified "unsupported shape".
int verify_geometry_diagnostics() {
    // These are the diagnostics of a refusal: an alignment-only geometry has no table row and so no
    // A16 route to fall back to, which only bites when the named route does not admit the width.
    // A16 is that route -- it admits none -- so pin it; under grouped or mmq the same call is legal
    // and run_forced_padded_widths is what proves it.
    const ForcedRoute refusing(ops::detail::Nvfp4GemmRoute::A16);
    struct Case {
        std::int32_t n, k;
        ops::LinearPolicy policy;
        std::int32_t tokens;
        const char* expected;
    };
    const Case cases[] = {
        // No row and no A16 route under this policy.
        {4096, 2048, ops::LinearPolicy::A16Only, 128, "no registered shape"},
        // A row would bring an A16 fallback; the alignment admission does not, and under the
        // default route a width that is not a whole 128-token tile is not admitted. A forced
        // grouped route does admit it -- the tile is padded, not refused -- which
        // run_forced_padded_widths below is what actually exercises.
        {4096, 2048, ops::LinearPolicy::AllowNvfp4Int8, 129, "does not admit"},
        // N is not a whole number of 128-row tiles.
        {4000, 2048, ops::LinearPolicy::AllowNvfp4Int8, 128, "N % 128 == 0"},
        // K is not a whole number of 64-column k-tiles.
        {4096, 2080, ops::LinearPolicy::AllowNvfp4Int8, 128, "K % 64 == 0"},
    };
    int failures = 0;
    for (const Case& probe : cases) {
        bool threw = false;
        std::string message;
        try {
            (void)ops::linear_workspace_capacity_bytes(QType::NVFP4, probe.n, probe.k, probe.policy,
                                                       probe.tokens, probe.tokens);
        } catch (const std::invalid_argument& error) {
            threw = true;
            message = error.what();
        }
        if (!threw || message.find(probe.expected) == std::string::npos) {
            std::cerr << "NVFP4 int8 [" << probe.n << ',' << probe.k << "] T=" << probe.tokens
                      << " (" << static_cast<int>(probe.policy) << "): expected \"" << probe.expected
                      << "\", got " << (threw ? message : std::string("no exception")) << '\n';
            ++failures;
        }
    }
    return failures;
}

// The grouped route answers at every width, so what has to be right here is the padded tile: the
// widths the old cascade never sent to the int8 kernels -- one token, a handful, a width landing in
// the middle of a tile -- each run with the last tile full and its surplus columns discarded. Every
// column is compared against the FP64 oracle, and the input-preservation check catches the failure
// padding actually creates: a padding row reading past the caller's activation. The grouped cases run
// on the route run_nvfp4_int8 already pinned; only MMQ needs to say so itself.
int run_forced_padded_widths() {
    int failures = 0;

    // Every residue class under a tile, both sides of the 128 and 256 tile boundaries, and one
    // width past each -- a grid that floored instead of ceiling the token count would show up here
    // as missing columns, and one that ceiled the stores as corrupt ones.
    std::vector<Invocation> narrow;
    for (int t : {1, 2, 3, 31, 32, 33, 64, 65, 100, 127}) {
        narrow.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    }
    failures += run_shape("NVFP4_Int8Grouped_PaddedNarrow", ActivationCompute::Int8Grouped,
                          make_nvfp4_weight,
                          {8192, 4096, 911U, Comparison::SampledRows, true, narrow});

    std::vector<Invocation> ragged;
    for (int t : {128, 129, 255, 256, 257, 300, 383}) {
        ragged.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    }
    failures += run_shape("NVFP4_Int8Grouped_PaddedRagged", ActivationCompute::Int8Grouped,
                          make_nvfp4_weight,
                          {8192, 4096, 913U, Comparison::SampledRows, true, ragged});

    // [4096, 2048] has no table row and no A16 composition, so a green run here can only mean the
    // padded int8 kernel answered these widths -- the positive-execution assertion the registered
    // geometry cannot offer on its own, where a refused width would quietly pass on A16 instead.
    std::vector<Invocation> alignment_only;
    for (int t : {1, 37, 129, 915}) {
        alignment_only.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    }
    failures += run_shape("NVFP4_Int8Grouped_PaddedAlignedOnly", ActivationCompute::Int8Grouped,
                          make_nvfp4_weight,
                          {4096, 2048, 917U, Comparison::SampledRows, true, alignment_only});

    // The MMQ mainloop pads against a different shape of problem: its plane is window-major,
    // [window][token][row], so the STRIDE is a function of the token count and the writer and the
    // reader must both use the padded number. Same widths, same oracle, same guarded outputs.
    const ForcedRoute mmq_route(ops::detail::Nvfp4GemmRoute::Mmq);
    std::vector<Invocation> mmq_narrow;
    for (int t : {1, 2, 3, 31, 32, 33, 64, 65, 100, 127}) {
        mmq_narrow.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    }
    failures += run_shape("NVFP4_Int8Mmq_PaddedNarrow", ActivationCompute::Int8Grouped,
                          make_nvfp4_weight,
                          {8192, 4096, 921U, Comparison::SampledRows, true, mmq_narrow});

    std::vector<Invocation> mmq_ragged;
    for (int t : {128, 129, 255, 256, 257, 300, 383}) {
        mmq_ragged.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    }
    failures += run_shape("NVFP4_Int8Mmq_PaddedRagged", ActivationCompute::Int8Grouped,
                          make_nvfp4_weight,
                          {8192, 4096, 923U, Comparison::SampledRows, true, mmq_ragged});

    std::vector<Invocation> mmq_alignment_only;
    for (int t : {1, 37, 129, 915}) {
        mmq_alignment_only.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    }
    failures += run_shape("NVFP4_Int8Mmq_PaddedAlignedOnly", ActivationCompute::Int8Grouped,
                          make_nvfp4_weight,
                          {4096, 2048, 927U, Comparison::SampledRows, true, mmq_alignment_only});
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_nvfp4_int8();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4_Int8Grouped Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4_Int8Grouped Linear: " << error.what() << '\n';
        return 1;
    }
}
