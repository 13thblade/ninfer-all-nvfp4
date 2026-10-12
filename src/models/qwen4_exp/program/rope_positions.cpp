// Adapted from Infernix a3edb450 src/models/qwen4_exp/program/rope_positions.cpp and
// vision_window.cpp (visual_slice) (Apache-2.0).
// Modified for NInfer-3090: NInfer's namespace; see rope_positions.h.

#include "models/qwen4_exp/program/rope_positions.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace ninfer::models::qwen4_exp {

RopePosition rope_of(const LaneRope& rope, std::uint32_t index) noexcept {
    if (index < rope.prompt_tokens && !rope.prompt.empty()) {
        const std::size_t n = rope.prompt_tokens;
        return {rope.prompt[index], rope.prompt[n + index], rope.prompt[2 * n + index]};
    }
    const auto v = static_cast<std::int32_t>(static_cast<std::int64_t>(index) + rope.delta);
    return {v, v, v};
}

RopePosition block_start_rope(const LaneRope& rope, std::uint32_t start, std::uint32_t ratio) noexcept {
    return rope_of(rope, start / ratio * ratio);
}

void stage_rope_value(const RopePosition& value, std::span<std::int32_t> words, std::int32_t columns,
                      std::int32_t column) {
    for (std::size_t a = 0; a < 3; ++a) {
        words[a * static_cast<std::size_t>(columns) + static_cast<std::size_t>(column)] = value[a];
    }
}

void stage_rope(const LaneRope& rope, std::uint32_t first, std::int32_t count, std::span<std::int32_t> words,
                std::int32_t columns, std::int32_t column) {
    if (count < 0 || column < 0 || column + count > columns || words.size() < 3ULL * static_cast<std::size_t>(columns)) {
        throw std::invalid_argument("Qwen4Exp RoPE staging: columns out of range");
    }
    for (std::int32_t i = 0; i < count; ++i) {
        stage_rope_value(rope_of(rope, first + static_cast<std::uint32_t>(i)), words, columns, column + i);
    }
}

void stage_mtp_chunk_rope(const LaneRope& rope, std::uint32_t first, std::int32_t cells, std::int32_t sub,
                          std::uint32_t ratio, std::span<std::int32_t> cells_words,
                          std::span<std::int32_t> block_words) {
    if (cells < 0 || sub <= 0 || ratio == 0) { throw std::invalid_argument("Qwen4Exp MTP RoPE staging: bad geometry"); }
    const std::int32_t chunks = (cells + sub - 1) / sub;
    if (cells_words.size() < 3ULL * static_cast<std::size_t>(cells) ||
        block_words.size() < 3ULL * static_cast<std::size_t>(chunks)) {
        throw std::invalid_argument("Qwen4Exp MTP RoPE staging: io is too small");
    }
    for (std::int32_t k = 0; k < chunks; ++k) {
        const std::int32_t n     = std::min(sub, cells - k * sub);
        const std::uint32_t base = first + static_cast<std::uint32_t>(k * sub);
        stage_rope(rope, base, n, cells_words.subspan(3ULL * static_cast<std::size_t>(sub) * k), n, 0);
        stage_rope_value(block_start_rope(rope, base, ratio), block_words.subspan(3ULL * static_cast<std::size_t>(k)), 1, 0);
    }
}

VisualSlice visual_slice(std::span<const std::uint32_t> visual, std::uint32_t a, std::uint32_t b, std::uint32_t base,
                         std::span<std::int32_t> columns) {
    const auto lo = std::lower_bound(visual.begin(), visual.end(), a);
    const auto hi = std::lower_bound(lo, visual.end(), b);
    VisualSlice out{static_cast<std::uint32_t>(lo - visual.begin()), static_cast<std::uint32_t>(hi - lo)};
    if (columns.size() < out.count) { throw std::invalid_argument("Vision: too few column words"); }
    for (std::uint32_t i = 0; i < out.count; ++i) {
        columns[i] = static_cast<std::int32_t>(visual[out.first + i] - base);
    }
    return out;
}

} // namespace ninfer::models::qwen4_exp
