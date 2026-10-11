#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

namespace ninfer::text {
// Request-owned incremental string constraints the grammar does not carry: uniqueness for JSON
// string arrays, and the minLength/maxLength of plain string values in a schema without
// alternatives (a counted string in the grammar has no cacheable token masks, so those bounds are
// taken out of the schema the grammar compiles and enforced here). The immutable schema plan is
// shared; a copy owns independent parser, branch, seen-value and length state.
class StringConstraintState {
public:
    static void AnalyzeSchema(const std::string& normalized_schema);
    explicit StringConstraintState(const std::string& normalized_schema,
                                   const std::string& reasoning_close = {});
    ~StringConstraintState();
    StringConstraintState(const StringConstraintState&);
    StringConstraintState& operator=(const StringConstraintState&);
    StringConstraintState(StringConstraintState&&) noexcept;
    StringConstraintState& operator=(StringConstraintState&&) noexcept;
    // The schema for the grammar: the normalized schema less the bounds enforced here.
    [[nodiscard]] const std::string& grammar_schema() const noexcept;
    [[nodiscard]] bool accepts(std::string_view decoded_token) const;
    // False never changes the committed state.
    bool accept(std::string_view decoded_token);
    [[nodiscard]] bool has_constraints() const noexcept;
    // Whether the next token's mask needs a per-token check at all; `longest_token` is the
    // vocabulary's longest token in bytes.
    [[nodiscard]] bool needs_scan(std::size_t longest_token) const noexcept;
    [[nodiscard]] bool needs_check(std::string_view decoded_token) const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ninfer::text
