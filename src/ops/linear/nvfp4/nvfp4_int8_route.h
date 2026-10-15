#pragma once

// The NVFP4 GEMM paths on hardware without native FP4 tensor cores (sm_86 / sm_89): named one at a
// time. `mmvq` is one route over two widths rather than one kernel: below llama.cpp's tensor-core
// crossover it runs the MMVQ-shaped mainloop, above it the MMQ mainloop, exactly the division of
// labour `ggml_cuda_should_use_mmvq` makes. It exists so the narrow half of the width ladder is
// measurable as its own thing instead of being hidden inside whichever wide-kernel answers.
//
// There used to be a fourth answer, and it was not a route: the dispatcher compared the call's
// token count against the int8 mainloop's whole-128-token-tile admission and fell through to the
// A16 (BF16 MMA / SIMT) family for every other width, while the MMQ mainloop was a nested switch
// *inside* that int8 branch. That hybrid cascade is gone. Each of the three paths is selected
// outright and served above the crossover, because a same-geometry A/B is impossible while one of
// them is a function of the other's refusal -- and, in the other direction, a route that never got
// taken reads as a route that made no difference.
//
// The crossover itself is now a property of the int8 family rather than of the name `mmvq`:
// `mmvq::applies` claims T <= 8 whichever route is named, so `grouped` and `mmq` differ only from
// nine columns up. This is not the deleted cascade coming back -- nothing falls through to A16 on a
// width, and every int8 route still answers every width. What it does cost is the ability to A/B the
// two MMA mainloops against each other *at decode widths*: the §15.5 decode column (grouped 18.7 /
// mmq 26.5 / mmvq 36.8 tok/s) is history, not a reproducible pair of arms. What it buys is that
// decode has one implementation, and that the kernel reading each weight byte exactly once -- rather
// than staging a J-token tile for a single column -- is what answers there, as it does upstream.
//
// Unset resolves to A16, the upstream implementation, which is the shipping default: the int8
// mainloops pad their token tile up to 128 (nvfp4_int8_padded_tokens), so a forced int8 route at
// decode multiplies the MMA work ~128x for the same weight traffic. That is a measurement choice,
// not a good default, and A16 also remains the path upstream is still improving.

#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

enum class Nvfp4GemmRoute : std::int8_t {
    A16,     ///< The BF16 MMA / SIMT family. Resolved when the environment names nothing.
    Grouped, ///< The int8-grouped mainloop above the crossover; T <= 8 goes to the row-form kernel.
    Mmq,     ///< The ggml-shaped MMQ mainloop above the crossover; same narrow-band exception.
    Mmvq,    ///< ggml's width ladder: MMVQ up to 8 columns, MMQ above it.
};

inline const char* to_string(Nvfp4GemmRoute route) {
    switch (route) {
        case Nvfp4GemmRoute::A16: return "a16";
        case Nvfp4GemmRoute::Grouped: return "grouped";
        case Nvfp4GemmRoute::Mmq: return "mmq";
        case Nvfp4GemmRoute::Mmvq: return "mmvq";
    }
    return "?";
}

namespace route_detail {

inline bool iequals(const char* lhs, const std::string& rhs) {
    const std::size_t n = std::strlen(lhs);
    if (n != rhs.size()) { return false; }
    for (std::size_t i = 0; i < n; ++i) {
        if (std::tolower(static_cast<unsigned char>(lhs[i])) !=
            std::tolower(static_cast<unsigned char>(rhs[i]))) {
            return false;
        }
    }
    return true;
}

// A value that is not a route name is an error, not a fallback. `set NINFER_NVFP4_GEMM=mmq ` in
// cmd.exe folds the trailing space into the value; under the old boolean env check that silently
// read as "route off", which is how a real difference between two mainloops once got reported as
// no difference at all.
[[noreturn]] inline void reject(const std::string& value) {
    throw std::invalid_argument(
        "NINFER_NVFP4_GEMM: unrecognised route '" + value +
        "'; expected one of a16|grouped|mmq|mmvq. Note that in cmd.exe a trailing space is part of "
        "the value -- write set \"NINFER_NVFP4_GEMM=grouped\" or use PowerShell's $env:.");
}

// -1: nothing forced; otherwise the forced route, plus one. Same shape as the route table's
// forced_for_tests() slot, which is why this seam exists at all: the environment is read into a
// function-local static, so a single test binary cannot see both routes without it.
inline std::atomic<int>& forced() {
    static std::atomic<int> slot{-1};
    return slot;
}

} // namespace route_detail

/// Test seam: override the named route for the whole process, returning whatever was in force so a
/// nested scope can put it back. Production code reads the environment and never calls this.
///
/// The return value is load-bearing, not convenience: a test harness pins the route a profile claims
/// while the case that owns it may pin one of its own, and with an absolute setter the inner scope
/// would clear the outer one on exit and leave the rest of the block on the shipping default -- A16,
/// answering where the kernel under test should have.
inline std::optional<Nvfp4GemmRoute> force_nvfp4_gemm_route(std::optional<Nvfp4GemmRoute> route) {
    const int previous = route_detail::forced().exchange(
        route ? static_cast<int>(*route) + 1 : -1, std::memory_order_relaxed);
    if (previous < 0) { return std::nullopt; }
    return static_cast<Nvfp4GemmRoute>(previous - 1);
}

/// The route named by NINFER_NVFP4_GEMM, resolved once per process; unset is A16.
///
/// One gate stays outside this header, and it is configuration rather than routing: which SHAPE
/// TABLE a width resolves to is the device profile's business (its nvfp4 width bands), shared with
/// every other quant format, and the own (non-unified) table registers no int8 route at all. The
/// shipping profile bands narrow widths away from the unified table, so measuring an int8 route at
/// decode takes both switches --
///
///   NINFER_NVFP4_GEMM=grouped  NINFER_LINEAR_ROUTES=unified
///
/// The second is the repo's existing whole-table override. Neither implies the other, and naming the
/// first without the second is a loud error rather than a quiet run of the A16 family.
inline Nvfp4GemmRoute nvfp4_gemm_route() {
    if (const int value = route_detail::forced().load(std::memory_order_relaxed); value >= 0) {
        return static_cast<Nvfp4GemmRoute>(value - 1);
    }
    static const Nvfp4GemmRoute route = [] {
        const char* raw = std::getenv("NINFER_NVFP4_GEMM");
        if (raw == nullptr || raw[0] == '\0') { return Nvfp4GemmRoute::A16; }
        const std::string value(raw);
        if (route_detail::iequals("default", value) || route_detail::iequals("cascade", value)) {
            // Named deliberately, because a script that still passes it would otherwise resolve to
            // A16 and quietly stop measuring the int8 kernels it thinks it is measuring.
            throw std::invalid_argument(
                "NINFER_NVFP4_GEMM: '" + value +
                "' named the old hybrid width cascade, which no longer exists; name the route "
                "outright -- a16 (also what an unset environment selects), grouped, or mmq");
        }
        if (route_detail::iequals("a16", value)) { return Nvfp4GemmRoute::A16; }
        if (route_detail::iequals("grouped", value)) { return Nvfp4GemmRoute::Grouped; }
        if (route_detail::iequals("mmvq", value)) { return Nvfp4GemmRoute::Mmvq; }
        if (route_detail::iequals("mmq", value) || value == "1") {
            // "1" is the legacy spelling of the MMQ switch, kept so existing A/B scripts still work.
            return Nvfp4GemmRoute::Mmq;
        }
        route_detail::reject(value);
    }();
    return route;
}

/// Validate an ambient NINFER_NVFP4_GEMM without consulting the test seam.
///
/// A binary that pins its own route never resolves the environment, so a mistyped value would pass
/// unnoticed there -- and the whole reason an unrecognised value is an error rather than a fallback is
/// that a silent one has been mistaken for a measurement before. Anything that pins a route over the
/// top of an ambient setting should call this first.
inline void validate_nvfp4_gemm_env() {
    const char* raw = std::getenv("NINFER_NVFP4_GEMM");
    if (raw == nullptr || raw[0] == '\0') { return; }
    const std::string value(raw);
    if (route_detail::iequals("a16", value) || route_detail::iequals("grouped", value) ||
        route_detail::iequals("mmq", value) || route_detail::iequals("mmvq", value) ||
        value == "1") {
        return;
    }
    if (route_detail::iequals("default", value) || route_detail::iequals("cascade", value)) {
        throw std::invalid_argument(
            "NINFER_NVFP4_GEMM: '" + value +
            "' named the old hybrid width cascade, which no longer exists; name the route outright "
            "-- a16 (also what an unset environment selects), grouped, or mmq");
    }
    route_detail::reject(value);
}

/// Whether the named route reaches either int8 mainloop. The own (non-unified) shape table registers
/// no int8 route at all, so a caller that forces one and lands there must say so rather than quietly
/// running the A16 family.
inline bool nvfp4_gemm_route_forces_int8() {
    const Nvfp4GemmRoute route = nvfp4_gemm_route();
    return route == Nvfp4GemmRoute::Grouped || route == Nvfp4GemmRoute::Mmq ||
           route == Nvfp4GemmRoute::Mmvq;
}

} // namespace ninfer::ops::detail
