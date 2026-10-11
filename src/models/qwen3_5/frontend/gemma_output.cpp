#include "models/qwen3_5/frontend/gemma_output.h"

#include "models/qwen3_5/frontend/tokenizer.h"

#include <nlohmann/json.hpp>

#include <cctype>
#include <charconv>
#include <optional>

namespace ninfer::models::qwen3_5::frontend {
namespace {

using Json = nlohmann::ordered_json;

std::int32_t single_token(const Tokenizer& tokenizer, std::string_view text) {
    const std::vector<int> ids = tokenizer.encode(text);
    return ids.size() == 1 ? ids.front() : -1;
}

struct Reader {
    std::string_view text;
    std::size_t at = 0;

    void skip() {
        while (at < text.size() && std::isspace(static_cast<unsigned char>(text[at])) != 0) { ++at; }
    }
    bool take(std::string_view token) {
        skip();
        if (text.substr(at).starts_with(token)) {
            at += token.size();
            return true;
        }
        return false;
    }
    [[nodiscard]] bool peek(std::string_view token) {
        skip();
        return text.substr(at).starts_with(token);
    }

    std::optional<std::string> quoted() {
        if (!take(kGemmaQuote)) { return std::nullopt; }
        const std::size_t end = text.find(kGemmaQuote, at);
        if (end == std::string_view::npos) { return std::nullopt; }
        std::string out(text.substr(at, end - at));
        at = end + kGemmaQuote.size();
        return out;
    }

    std::optional<std::string> key() {
        if (peek(kGemmaQuote)) { return quoted(); }
        skip();
        const std::size_t end = text.find(':', at);
        if (end == std::string_view::npos) { return std::nullopt; }
        std::string_view raw = text.substr(at, end - at);
        while (!raw.empty() && std::isspace(static_cast<unsigned char>(raw.back())) != 0) {
            raw.remove_suffix(1);
        }
        if (raw.empty()) { return std::nullopt; }
        at = end;
        return std::string(raw);
    }

    std::optional<Json> value() {
        skip();
        if (peek(kGemmaQuote)) {
            auto s = quoted();
            if (!s) { return std::nullopt; }
            return Json(std::move(*s));
        }
        if (at >= text.size()) { return std::nullopt; }
        if (text[at] == '{') { return object(); }
        if (text[at] == '[') {
            ++at;
            Json array = Json::array();
            if (take("]")) { return array; }
            for (;;) {
                auto item = value();
                if (!item) { return std::nullopt; }
                array.push_back(std::move(*item));
                if (take(",")) { continue; }
                if (take("]")) { return array; }
                return std::nullopt;
            }
        }
        // A bare scalar up to the next separator.
        std::size_t end = at;
        while (end < text.size() && text[end] != ',' && text[end] != '}' && text[end] != ']') {
            ++end;
        }
        std::string_view raw = text.substr(at, end - at);
        while (!raw.empty() && std::isspace(static_cast<unsigned char>(raw.back())) != 0) {
            raw.remove_suffix(1);
        }
        at = end;
        if (raw == "true") { return Json(true); }
        if (raw == "false") { return Json(false); }
        if (raw == "null") { return Json(nullptr); }
        std::int64_t integer = 0;
        auto [iend, ierr]    = std::from_chars(raw.data(), raw.data() + raw.size(), integer);
        if (ierr == std::errc{} && iend == raw.data() + raw.size() && !raw.empty()) {
            return Json(integer);
        }
        try {
            std::size_t used = 0;
            const double number = std::stod(std::string(raw), &used);
            if (used == raw.size() && !raw.empty()) { return Json(number); }
        } catch (const std::exception&) {}
        return Json(std::string(raw));
    }

    std::optional<Json> object() {
        if (!take("{")) { return std::nullopt; }
        Json out = Json::object();
        if (take("}")) { return out; }
        for (;;) {
            auto k = key();
            if (!k || !take(":")) { return std::nullopt; }
            auto v = value();
            if (!v) { return std::nullopt; }
            out[*k] = std::move(*v);
            if (take(",")) { continue; }
            if (take("}")) { return out; }
            return std::nullopt;
        }
    }
};

// The length of the longest suffix of `text` that begins `marker` (shorter than the marker).
std::size_t partial_suffix(std::string_view text, std::string_view marker) {
    const std::size_t maximum = std::min(text.size(), marker.size() - 1);
    for (std::size_t size = maximum; size != 0; --size) {
        if (text.substr(text.size() - size) == marker.substr(0, size)) { return size; }
    }
    return 0;
}

} // namespace

GemmaMarkers GemmaMarkers::from(const Tokenizer& tokenizer) {
    GemmaMarkers out;
    out.channel_open  = single_token(tokenizer, "<|channel>");
    out.channel_close = single_token(tokenizer, "<channel|>");
    out.call_open     = single_token(tokenizer, kGemmaCallOpen);
    out.call_close    = single_token(tokenizer, kGemmaCallClose);
    out.quote         = single_token(tokenizer, kGemmaQuote);
    return out;
}

bool parse_gemma_call(std::string_view region, GeneratedToolCall& out) {
    Reader reader{region};
    if (!reader.take("call:")) { return false; }
    const std::size_t brace = region.find('{', reader.at);
    if (brace == std::string_view::npos) { return false; }
    std::string_view name = region.substr(reader.at, brace - reader.at);
    while (!name.empty() && std::isspace(static_cast<unsigned char>(name.back())) != 0) {
        name.remove_suffix(1);
    }
    if (name.empty()) { return false; }
    reader.at = brace;
    auto arguments = reader.object();
    if (!arguments) { return false; }
    reader.skip();
    if (reader.at != region.size()) { return false; }
    out.name           = std::string(name);
    out.arguments_json = arguments->dump();
    return true;
}

std::string GemmaToolCallDecoder::feed(std::string_view text) {
    std::string published;
    pending_.append(text);
    for (;;) {
        if (!in_call_) {
            const std::size_t open = pending_.find(kGemmaCallOpen);
            if (open == std::string::npos) {
                const std::size_t hold = partial_suffix(pending_, kGemmaCallOpen);
                published.append(pending_, 0, pending_.size() - hold);
                pending_.erase(0, pending_.size() - hold);
                return published;
            }
            published.append(pending_, 0, open);
            pending_.erase(0, open + kGemmaCallOpen.size());
            in_call_                 = true;
            diagnostics_.marker_seen = true;
            region_.clear();
        }
        const std::size_t close = pending_.find(kGemmaCallClose);
        if (close == std::string::npos) {
            const std::size_t hold = partial_suffix(pending_, kGemmaCallClose);
            region_.append(pending_, 0, pending_.size() - hold);
            pending_.erase(0, pending_.size() - hold);
            return published;
        }
        region_.append(pending_, 0, close);
        pending_.erase(0, close + kGemmaCallClose.size());
        in_call_ = false;
        GeneratedToolCall call;
        if (parse_gemma_call(region_, call)) {
            calls_.push_back(std::move(call));
            ++diagnostics_.structured_call_count;
        } else {
            diagnostics_.fallback_reason = ToolCallParseFallbackReason::MalformedStructure;
            published.append(kGemmaCallOpen);
            published.append(region_);
            published.append(kGemmaCallClose);
        }
        region_.clear();
    }
}

GemmaToolCallDecoder::Terminal GemmaToolCallDecoder::finish() {
    Terminal out;
    if (in_call_) {
        region_.append(pending_);
        GeneratedToolCall call;
        if (parse_gemma_call(region_, call)) {
            calls_.push_back(std::move(call));
            ++diagnostics_.structured_call_count;
        } else {
            out.content.append(kGemmaCallOpen);
            out.content.append(region_);
        }
    } else {
        out.content = pending_;
    }
    pending_.clear();
    region_.clear();
    in_call_        = false;
    out.tool_calls  = std::move(calls_);
    out.diagnostics = diagnostics_;
    return out;
}

} // namespace ninfer::models::qwen3_5::frontend
