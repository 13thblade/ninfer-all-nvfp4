#pragma once

// Gemma 4's output markup in the shared frontend: the reasoning channel `<|channel>thought\n ...
// <channel|>` and tool calls `<|tool_call>call:NAME{key:value,...}<tool_call|>`, whose strings are
// delimited by `<|"|>`.

#include "ninfer/types.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

class Tokenizer;

// The special tokens of the dialect; -1 where the vocabulary lacks one.
struct GemmaMarkers {
    std::int32_t channel_open  = -1; // <|channel>
    std::int32_t channel_close = -1; // <channel|>
    std::int32_t call_open     = -1; // <|tool_call>
    std::int32_t call_close    = -1; // <tool_call|>
    std::int32_t quote         = -1; // <|"|>

    [[nodiscard]] static GemmaMarkers from(const Tokenizer& tokenizer);
    // Whether the token's text is kept in the content stream (tool-call markup).
    [[nodiscard]] bool markup(std::int32_t token) const noexcept {
        return token == call_open || token == call_close || token == quote;
    }
};

inline constexpr std::string_view kGemmaCallOpen  = "<|tool_call>";
inline constexpr std::string_view kGemmaCallClose = "<tool_call|>";
inline constexpr std::string_view kGemmaQuote     = "<|\"|>";

// Pulls Gemma tool calls out of the content stream: text outside calls is published as it
// arrives (a possible opener's prefix is held back), each complete call becomes a structured
// call with JSON arguments. An unterminated or unreadable call returns to content at the end.
class GemmaToolCallDecoder {
public:
    struct Terminal {
        std::string content;
        std::vector<GeneratedToolCall> tool_calls;
        ToolCallParseDiagnostics diagnostics;
    };

    [[nodiscard]] std::string feed(std::string_view text);
    [[nodiscard]] Terminal finish();

private:
    std::string pending_;
    std::string region_;
    bool in_call_ = false;
    std::vector<GeneratedToolCall> calls_;
    ToolCallParseDiagnostics diagnostics_;
};

// `call:NAME{...}` (the text between the call markers) as a structured call; false when unreadable.
[[nodiscard]] bool parse_gemma_call(std::string_view region, GeneratedToolCall& out);

} // namespace ninfer::models::qwen3_5::frontend
