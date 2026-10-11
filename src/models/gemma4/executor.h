#pragma once

// The forward pass of Gemma 4 over a loaded Model. Each sequence owns an F16 K/V cache per block:
// a ring of the last window plus a prompt chunk for the sliding-window blocks, every position for
// the full-attention blocks. Prompt chunks run llama.cpp's tensor-core flash attention; decode steps
// run one token of each of several sequences at once, replayed from a CUDA graph per batch size.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/gemma4/model.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::models::gemma4 {

struct ExecutorOptions {
    std::uint32_t max_context   = 32768;
    std::uint32_t prefill_chunk = 1024; // prompt tokens per forward()
    std::uint32_t sequences     = 1;
    bool cuda_graphs            = true;
};

struct ExecutorMemory {
    std::uint64_t kv_bytes        = 0;
    std::uint64_t workspace_bytes = 0;
};

// A sequence's sliding-window rings and position, which Executor::restore resumes from; the
// full-attention caches of earlier positions are not copied, since a sequence only writes
// positions at or past its own.
struct SequenceSnapshot {
    std::vector<DeviceBuffer> rings;
    std::uint32_t position = 0;
};

class Executor {
public:
    // Logit rows forward() may leave at once.
    static constexpr std::uint32_t kMaxLogitRows = 256;

    Executor(const Model& model, DeviceContext& device, ExecutorOptions options);
    ~Executor();
    Executor(const Executor&)            = delete;
    Executor& operator=(const Executor&) = delete;

    [[nodiscard]] const ExecutorOptions& options() const noexcept;
    [[nodiscard]] ExecutorMemory memory() const noexcept;
    void reset(std::uint32_t sequence);
    [[nodiscard]] std::uint32_t position(std::uint32_t sequence) const;
    // Loads the kernels every route reaches, and captures the decode graphs.
    void warm_up();
    // Runs up to prefill_chunk tokens at the sequence's next positions and leaves the logits of the
    // last `logit_rows` (at most kMaxLogitRows) in logits().
    void forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                 std::uint32_t logit_rows);
    // One token of each of `sequences` (distinct) at its next position; logits one column each.
    void decode(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens);
    // BF16 [vocab, rows], ready on head_stream().
    [[nodiscard]] Tensor logits(std::uint32_t rows) const;
    [[nodiscard]] cudaStream_t head_stream() const noexcept;
    void snapshot(std::uint32_t sequence, SequenceSnapshot& out);
    void restore(std::uint32_t sequence, const SequenceSnapshot& from);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::gemma4
