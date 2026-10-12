#pragma once

#include "ninfer/types.h"

namespace ninfer::runtime {

// The checks every model's route shares: a .ninfer artifact path, nonzero context, a coherent
// KV-capacity request and context fit, and concurrency, pending-request and media bounds. Throws
// std::invalid_argument.
void validate_engine_options(const EngineOptions& options);

} // namespace ninfer::runtime
