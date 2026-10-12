// ReadOnlyFile's range contract: a direct read whose [offset, offset + size) does not fit the
// platform's signed file offset is refused before any I/O, on every platform; an ordinary aligned
// read still returns the file's bytes.

#include "core/read_only_file.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t kBlock = 4096; // the alignment unbuffered I/O needs

struct alignas(kBlock) Block {
    std::byte bytes[kBlock];
};
Block g_block;

int expect_overflow(const ninfer::ReadOnlyFile& file, std::uint64_t offset, std::span<std::byte> buffer,
                    const char* label) {
    try {
        (void)file.read_direct(offset, buffer);
    } catch (const std::overflow_error&) { return 0; }
    std::cerr << label << " did not throw std::overflow_error\n";
    return 1;
}

} // namespace

int main() {
    // In the working directory (the build tree), not a tmpfs that may refuse unbuffered opens.
    const auto path = std::filesystem::current_path() / "ninfer_read_only_file_test.bin";
    {
        std::vector<char> bytes(2 * kBlock);
        for (std::size_t i = 0; i < bytes.size(); ++i) { bytes[i] = static_cast<char>(i * 7 + 1); }
        std::ofstream(path, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    int failures = 0;
    {
        const ninfer::ReadOnlyFile file(path);
        const std::span<std::byte> buffer(g_block.bytes);
        if (file.read_direct(kBlock, buffer) != kBlock || buffer[0] != static_cast<std::byte>(kBlock * 7 + 1)) {
            std::cerr << "an aligned read did not return the file's bytes\n";
            ++failures;
        }
        constexpr auto max = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
        // The aligned offset 2^63 - 4096 fits on its own; with 4096 bytes the range ends at 2^63.
        failures += expect_overflow(file, max - kBlock + 1, buffer, "a read ending past the largest offset");
        failures += expect_overflow(file, max, buffer, "a read at the largest offset");
        failures += expect_overflow(file, max + 1, buffer, "a read past the largest offset");
    }
    std::filesystem::remove(path);
    if (failures == 0) { std::cout << "read_only_file: passed\n"; }
    return failures == 0 ? 0 : 1;
}
