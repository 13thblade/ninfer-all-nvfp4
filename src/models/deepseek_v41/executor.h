#pragma once

// The forward pass of DeepSeek-V4.1-Flash over a loaded Model. Each sequence owns, per block, a
// ring of its last window of attention latents (and a round's drafts), and per kv source block
// the compressed latents and the indexer's keys of every compressed position, with the ratio-2
// compressor's incomplete group in a small tail. Prompt chunks attend through a staging ring that
// every sequence's chunks share, a span of several chunks block by block when experts stay off
// the device; decode steps run one token of each of several sequences at once,
// replayed from a CUDA graph per batch size. The Engram rows a pass addresses are read before it
// from the table's files (or RAM) and copied to the device.
//
// With draft_tokens, verify() runs a sequence's next token (its anchor) and up to draft_tokens
// proposed tokens after it (a context lookup's) as decode rows, and commit() keeps the first ones.
// Branches (forward_branches) run one after another from a copy of the sequence's state.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/deepseek_v41/engram.h"
#include "models/deepseek_v41/model.h"
#include "models/expert_profile.h"
#include "models/qwen4_exp/ngram_table.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::deepseek_v41 {

// The device lacks the memory the options need (the caches, or the workspace and CUDA graphs
// beside them); the message names the amounts.
struct InsufficientMemory : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct ExecutorOptions {
    std::uint32_t max_context   = 32768;
    std::uint32_t prefill_chunk = 2048; // prompt tokens per forward()
    std::uint32_t sequences     = 1;
    bool cuda_graphs            = true;
    // Proposed tokens a verification may carry per sequence after its anchor (0: no
    // verification); a verification's rows, the anchors and drafts of all its sequences, are at
    // most 8.
    std::uint32_t draft_tokens = 0;
    // One checkpoint of every sequence's rings and tails (snapshot/restore), allocated with the
    // caches.
    bool checkpoints = true;
    std::optional<std::uint64_t> expert_cache_bytes;
    // Host or disk experts: the cache's first contents, the most routed of these counts
    // ([expert layer][expert]) once warm-up is done; none leaves the cache empty then.
    std::optional<ExpertRouteCounts> expert_profile;
    // The Engram table's rows and how they are read; none runs the model without its Engram (a
    // non-standard mode that degrades the output).
    std::optional<qwen4_exp::NgramTableLayout> engram_table;
    qwen4_exp::NgramReadOptions engram_read;
};

struct ExecutorMemory {
    std::uint64_t kv_bytes        = 0;
    // Beside an elastic expert cache, the compressed caches grow with their sequences: the bytes
    // backed now, of kv_bytes at most.
    std::uint64_t kv_backed_bytes = 0;
    bool kv_grows                 = false;
    std::uint64_t workspace_bytes = 0;
    // The most a prompt span borrows from an elastic expert cache while it runs (0: none).
    std::uint64_t span_borrow_bytes = 0;
    // Engram rows held in RAM (a table read whole, or the reader's row cache).
    std::uint64_t engram_resident_bytes = 0;
    ExpertDeliveryMemory experts;
};

// A sequence's checkpoint: its window rings and compressor tails, kept in the executor's
// checkpoint storage of that sequence (one per sequence), and its position and Engram history,
// which Executor::restore resumes from. The compressed caches of earlier positions are not
// copied, since a sequence only rewrites the compressed rows of positions at or past its own.
struct SequenceSnapshot {
    std::uint32_t position = 0;
    EngramHasher::History history;
};

// A sequence's state in host memory, laid out as image_bytes() sizes it: its rings, tails and
// compressed rows below `position`. The context cache keeps prefixes this way beyond the device's
// sequences; the header (position and Engram history) travels beside the bytes.
struct SequenceImage {
    std::uint32_t position = 0;
    EngramHasher::History history;
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
    // Position 0; growing caches give the sequence's rows back (ExecutorMemory::kv_grows).
    void reset(std::uint32_t sequence);
    [[nodiscard]] std::uint32_t position(std::uint32_t sequence) const;
    // Loads the kernels every route reaches, and captures the decode graphs.
    void warm_up();
    // Check after a stream completion and before output publication.
    void check_experts() const;
    [[nodiscard]] ExpertDeliveryStats expert_stats() const;
    [[nodiscard]] ExpertRouteCounts expert_totals() const;
    // The Engram reader's counters, or empty without the Engram.
    [[nodiscard]] std::optional<NgramTableStats> engram_stats() const;
    [[nodiscard]] std::string engram_description() const;
    // Prompt tokens one forward() may take: prefill_chunk, or with host or disk experts a span of
    // several chunks (as many as memory allows beside the expert cache), which forward() runs
    // layer by layer so that each block's missing experts cross the bus once for all of them.
    [[nodiscard]] std::uint32_t prompt_step() const noexcept;
    // Runs up to prompt_step() tokens at the sequence's next positions and leaves the logits of
    // the last `logit_rows` (at most kMaxLogitRows) in logits().
    void forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                 std::uint32_t logit_rows);
    // Branches of a prompt, each continuing the sequence from its position: branch b runs
    // `lengths[b]` of `tokens` (in branch order, each at most prefill_chunk, at most kMaxLogitRows
    // branches). Leaves the logits of each branch's last token in logits(), one column per branch
    // in order. The sequence keeps its position and state.
    void forward_branches(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                          std::span<const std::uint32_t> lengths);
    // A prompt's last chunk, then branches continuing it from there as forward_branches has
    // them. The sequence ends at the prompt's end.
    void forward_prompt_branches(std::uint32_t sequence, std::span<const std::int32_t> prompt,
                                 std::span<const std::int32_t> tokens,
                                 std::span<const std::uint32_t> lengths);
    // One token of each of `sequences` (distinct) at its next position; logits one column each.
    void decode(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens);
    // BF16 [vocab, rows], ready on head_stream().
    [[nodiscard]] Tensor logits(std::uint32_t rows) const;

    [[nodiscard]] std::uint32_t draft_tokens() const noexcept;
    // tokens.size() / sequences.size() tokens of each sequence (its anchor, then its drafts) at its
    // next positions, as decode rows: logits one column each, sequence by sequence. The positions
    // advance only with commit().
    void verify(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens);
    // Keeps the first kept[j] (1..width) of sequence j's verified tokens.
    void commit(std::span<const std::uint32_t> sequences, std::span<const std::uint32_t> kept);
    [[nodiscard]] cudaStream_t head_stream() const noexcept;
    void snapshot(std::uint32_t sequence, SequenceSnapshot& out);
    void restore(std::uint32_t sequence, const SequenceSnapshot& from);
    // Bytes of an image at `position`.
    [[nodiscard]] std::uint64_t image_bytes(std::uint32_t position) const;
    // Copies the sequence's state into `pages`, pinned host memory that holds image_bytes() of its
    // position in consecutive slices of `page_bytes` (the last one may be partly used): its live
    // state, or the state at `at`, its checkpoint, whose positions it has not written since.
    // Snapshots, restores and image copies are enqueued on the executor's stream, ordered with
    // the work around them, and the host does not wait for them: device work enqueued later sees
    // them done, and pages are only to be rewritten by later image copies.
    SequenceImage save_image(std::uint32_t sequence, const SequenceSnapshot* at,
                             std::span<std::byte* const> pages, std::size_t page_bytes);
    // Makes the sequence hold the image's state.
    void load_image(std::uint32_t sequence, const SequenceImage& image,
                    std::span<std::byte* const> pages, std::size_t page_bytes);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::deepseek_v41
