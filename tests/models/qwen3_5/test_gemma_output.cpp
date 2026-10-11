// Gemma 4's tool-call markup in the shared frontend: GemmaToolCallDecoder pulls
// `<|tool_call>call:NAME{key:value,...}<tool_call|>` regions out of streamed content (markers split
// across feeds included) and returns them as structured calls with JSON arguments; text around
// them stays content, and an unreadable or unterminated call returns to content.

#include "models/qwen3_5/frontend/gemma_output.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::models::qwen3_5::frontend::GemmaToolCallDecoder;
using ninfer::models::qwen3_5::frontend::parse_gemma_call;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

GemmaToolCallDecoder::Terminal run(const std::vector<std::string>& pieces, std::string& streamed) {
    GemmaToolCallDecoder decoder;
    for (const auto& piece : pieces) { streamed += decoder.feed(piece); }
    auto terminal = decoder.finish();
    streamed += terminal.content;
    return terminal;
}

} // namespace

int main() {
    const std::string q = "<|\"|>";
    {
        ninfer::GeneratedToolCall call;
        expect(parse_gemma_call("call:get_weather{city:" + q + "Paris" + q + ",days:3}", call),
               "parse a call");
        expect(call.name == "get_weather", "call name");
        const auto args = nlohmann::json::parse(call.arguments_json);
        expect(args == nlohmann::json({{"city", "Paris"}, {"days", 3}}), "call arguments");
    }
    {
        // Nested object and array values, a boolean, a string holding braces and commas.
        ninfer::GeneratedToolCall call;
        expect(parse_gemma_call("call:f{a:{b:[1,2," + q + "x,}" + q + "]},flag:true}", call),
               "parse nested values");
        const auto args = nlohmann::json::parse(call.arguments_json);
        expect(args == nlohmann::json({{"a", {{"b", {1, 2, "x,}"}}}}, {"flag", true}}),
               "nested arguments");
    }
    {
        ninfer::GeneratedToolCall call;
        expect(!parse_gemma_call("get_weather{city:Paris}", call), "reject a region without call:");
    }
    {
        // Streamed with the markers split across feeds, text before the call kept as content.
        std::string streamed;
        const auto terminal = run({"Checking. <|tool", "_call>call:get_weather{city:" + q, "Paris",
                                   q + ",days:3}<tool_", "call|>"},
                                  streamed);
        expect(terminal.tool_calls.size() == 1, "one streamed call");
        expect(streamed == "Checking. ", "content around the call: '" + streamed + "'");
        if (!terminal.tool_calls.empty()) {
            expect(terminal.tool_calls[0].name == "get_weather", "streamed call name");
        }
    }
    {
        // Two calls in one turn.
        std::string streamed;
        const auto terminal =
            run({"<|tool_call>call:a{x:1}<tool_call|><|tool_call>call:b{y:" + q + "z" + q +
                 "}<tool_call|>"},
                streamed);
        expect(terminal.tool_calls.size() == 2, "two calls");
        expect(streamed.empty(), "no content beside two calls");
    }
    {
        // An unterminated call returns to content.
        std::string streamed;
        const auto terminal = run({"<|tool_call>call:a{x:1"}, streamed);
        expect(terminal.tool_calls.empty(), "unterminated call is not a call");
        expect(streamed.find("call:a{x:1") != std::string::npos, "unterminated call stays content");
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " Gemma output\n";
    return failures == 0 ? 0 : 1;
}
