#include "text/unicode.h"

#include <utf8proc/utf8proc.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>

namespace {

// Exact oracle: one utf8proc NFC pass over the whole text. normalize_nfc hands utf8proc only the
// stretches between NFC boundaries, so its splitting must never change the result.
std::string one_pass_nfc(const std::string& text) {
    utf8proc_uint8_t* mapped = nullptr;
    const utf8proc_ssize_t length =
        utf8proc_map(reinterpret_cast<const utf8proc_uint8_t*>(text.data()),
                     static_cast<utf8proc_ssize_t>(text.size()), &mapped,
                     static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
    if (length < 0) { throw std::runtime_error("oracle rejected the input"); }
    std::string normalized(reinterpret_cast<const char*>(mapped), static_cast<std::size_t>(length));
    std::free(mapped);
    return normalized;
}

void append(std::string& text, std::int32_t codepoint) {
    utf8proc_uint8_t bytes[4];
    const utf8proc_ssize_t length = utf8proc_encode_char(codepoint, bytes);
    text.append(reinterpret_cast<const char*>(bytes), static_cast<std::size_t>(length));
}

} // namespace

int main() {
    namespace unicode = ninfer::text::unicode_internal;
    // Combining marks of several classes (including ones that compose with ASCII bases, like the
    // long solidus overlay with '<' '=' '>'), precomposed and singleton characters, a starter that
    // decomposes into marks (U+0F73), Hangul jamo and syllables, compatibility-only characters and
    // plain symbols, so stretches start, merge and end at every kind of neighbour.
    const std::int32_t pool[] = {
        0x0300, 0x0301, 0x0308, 0x0323, 0x0327, 0x0338, 0x0344, 0x0345, 0x05b0, 0x093c,
        0x0958, 0x0f71, 0x0f72, 0x0f73, 0x0f75, 0x0f81, 0x09be, 0x09c7, 0x0b47, 0x0b56,
        0x0cc2, 0x0cc6, 0x0cd5, 0x1100, 0x1112, 0x1161, 0x1175, 0x11a8, 0x11c2, 0xac00,
        0xac01, 0xd7a3, 0x00e9, 0x00a8, 0x0385, 0x0387, 0x037e, 0x1fee, 0x212b, 0x2126,
        0x1e0a, 0x2adc, 0x1d15e, 0x3099, 0x304b, 0x302a, 0xfb01, 0x2000, 0x03b1, 0x2211,
        0x4e2d, 0x1f600};
    constexpr char ascii[] = "eAo<=>nc z\n\t.KR";
    std::mt19937 random(20261028);
    for (int iteration = 0; iteration < 100000; ++iteration) {
        std::string text;
        const int length = 1 + static_cast<int>(random() % 100);
        for (int i = 0; i < length; ++i) {
            const unsigned choice = random() % 100;
            if (choice < 55) {
                text.push_back(ascii[random() % (sizeof(ascii) - 1)]);
            } else if (choice < 62) {
                text.append(random() % 80, 'x');
            } else {
                append(text, pool[random() % (sizeof(pool) / sizeof(pool[0]))]);
            }
        }
        if (unicode::normalize_nfc(text) != one_pass_nfc(text)) {
            std::fprintf(stderr, "segmented NFC differs from one utf8proc pass (case %d)\n",
                         iteration);
            return 1;
        }
    }
    // Fixed cases with their NFC spelled out: the empty text, an embedded NUL, a leading mark, an
    // ASCII base composing with the mark after it, reordered marks, singletons, and Hangul L+V+T
    // and LV+T composing into one syllable after a run of other text.
    const struct {
        std::string input;
        std::string expected;
    } fixed[] = {
        {"", ""},
        {std::string("x\0a\xcc\x81", 5), std::string("x\0\xc3\xa1", 4)},
        {"\xcc\x81" "e", "\xcc\x81" "e"},
        {"xe\xcc\x81", "x\xc3\xa9"},
        {"a\xcc\xa3\xcc\x82", "\xe1\xba\xad"},
        {"a\xcc\x82\xcc\xa3", "\xe1\xba\xad"},
        {"\xe2\x84\xab\xe2\x84\xa6", "\xc3\x85\xce\xa9"},
        {"ab \xe1\x84\x80\xe1\x85\xa1\xe1\x86\xa8", "ab \xea\xb0\x81"},
        {"\xe4\xb8\xad\xea\xb0\x80\xe1\x86\xa8", "\xe4\xb8\xad\xea\xb0\x81"},
    };
    for (const auto& c : fixed) {
        if (unicode::normalize_nfc(c.input) != c.expected || one_pass_nfc(c.input) != c.expected) {
            std::fprintf(stderr, "NFC of a fixed case differs from its expected form\n");
            return 1;
        }
    }
    // Invalid UTF-8 is rejected wherever it sits: truncated, a stray continuation byte, an
    // overlong encoding, an encoded surrogate, and a bad byte after a long ASCII run.
    const std::string invalid[] = {"ok \xe2\x82 bad", "\x80", "a\xc0\xaf", "\xed\xa0\x80",
                                   std::string(4096, 'x') + "\xff"};
    for (const std::string& text : invalid) {
        bool rejected = false;
        try {
            (void)unicode::normalize_nfc(text);
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) {
            std::fprintf(stderr, "segmented NFC accepted invalid UTF-8\n");
            return 1;
        }
    }
    std::printf("ok (segmented NFC)\n");
    return 0;
}
