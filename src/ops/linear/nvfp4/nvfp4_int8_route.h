#pragma once

// Which wide mainloop an NVFP4 int8 call runs, and where that answer comes from.
//
// The NVFP4 W4A8 family has three mainloops: the grouped s8-tensor-core ladder
// (nvfp4_int8_grouped_mma.cuh), the ggml-shaped MMQ mainloop (nvfp4_mmq_mma.cuh), and a row-form
// kernel that answers T <= 8 whichever wide mainloop the call would otherwise take -- llama.cpp's own
// crossover, so decode and verify widths have one implementation rather than three. The row form is
// on the table too, not on a name of its own: the legacy chain runs the ported MMVQ kernel
// (nvfp4_mmvq_mma.cuh) and the unified chain runs its own narrow kernel
// (nvfp4_int8_narrow_mma.cuh) at the call's exact width.
//
// The wide mainloop is a property of the ROUTE TABLE, not of a second switch: a call the own
// (legacy) table serves runs MMQ, a call the unified table serves runs the grouped mainloop. That
// is the same axis the device profile's `nvfp4_int8/*` bands calibrate, so a calibrated band chooses
// a mainloop and there is no separate name to keep in sync with it.
//
// The family itself has exactly one switch: --nvfp4-w4a8-int (LoadOptions::nvfp4_w4a8_int), which
// registers the per-shape AllowNvfp4Int8 policy at the model layer
// (models/qwen3_5/execution/parameters.cpp). While the flag is on no width falls back to A16 --
// prefill and decode are the same decision on purpose, because a per-phase switch would make a
// calibrated band mean different kernels at different phases.

#include <cstdint>

namespace ninfer::ops::detail {

enum class Nvfp4Int8Mainloop : std::int8_t {
    Grouped, ///< The int8-grouped ladder above the crossover; T <= 8 goes to the row-form kernel.
    Mmq,     ///< The ggml-shaped MMQ mainloop above the crossover; same narrow-band exception.
};

inline const char* to_string(Nvfp4Int8Mainloop mainloop) {
    switch (mainloop) {
        case Nvfp4Int8Mainloop::Grouped: return "grouped";
        case Nvfp4Int8Mainloop::Mmq: return "mmq";
    }
    return "?";
}

} // namespace ninfer::ops::detail
