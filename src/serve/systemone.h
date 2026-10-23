#pragma once

// Typed decisions: POST /v1/systemone (TypeSafe's System One API, as llama.cpp, vLLM and Ollama
// serve it) and POST /v1/decisions (OpenAI's Decisions API), answered by the served chat model
// through a label readout. Each question is one prompt -- a fixed system turn asking for one
// label, then the state, the question and its options under one-token labels (A.. for a choice,
// yes/no for a noul, 0.. for a score), thinking off -- and the answer is read from the first reply
// token's exact log probabilities of those labels (a noul's yes and no in both cases): softmax
// over the options gives the probabilities, their total over the vocabulary is the label mass,
// and the winner's probability over the whole vocabulary is the confidence.

#include "serve/request.h"
#include "serve/request_json.h"

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::serve {

struct GenerationOutcome;

// vLLM's bounds, within the readout's kMaximumReadoutTokens and the 26 letters.
inline constexpr std::size_t kMaximumDecisionQuestions = 64;
inline constexpr std::size_t kMaximumDecisionOptions   = 26;
inline constexpr std::size_t kMaximumDecisionLevels    = 10;
inline constexpr std::size_t kMaximumDecisionIdLength  = 100;

enum class DecisionDialect { SystemOne, OpenAI };

// System One's choice / noul / score; OpenAI's predicate is a noul.
enum class DecisionType { Choice, Noul, Score };

struct DecisionOption {
    // The option's wire value: a System One option name or score level, or an OpenAI choice value
    // (a string or a boolean, echoed with its type) or level label.
    RequestJson value;
    // The value as the prompt shows it.
    std::string text;
    std::optional<std::string> description;
};

struct DecisionQuestion {
    // System One's question id, or OpenAI's optional name (`named` false echoes null).
    std::string id;
    bool named        = true;
    DecisionType type = DecisionType::Choice;
    std::string instructions;
    // A choice's options or a score's levels, lowest first; a noul's optional yes and no
    // descriptions as two entries (text "yes" and "no").
    std::vector<DecisionOption> options;
};

struct DecisionRequest {
    DecisionDialect dialect = DecisionDialect::SystemOne;
    std::string model;
    // The state as the prompt shows it: a string as given, other JSON as its compact text.
    std::string state;
    // Optional context placed ahead of the state (System One's `instructions`).
    std::string instructions;
    std::vector<DecisionQuestion> questions;
};

// Strict: unknown fields are refused, as vLLM refuses them; media fields are refused as not
// implemented.
[[nodiscard]] DecisionRequest parse_systemone_request(const RequestJson& body);
[[nodiscard]] DecisionRequest parse_decisions_request(const RequestJson& body);

struct DecisionLabels {
    // The label texts the read looks up, each one token: A..Z for a choice, yes, Yes, no and No
    // for a noul, 0..9 for a score.
    std::vector<std::string> texts;
    // The option each label answers for, aligned with texts.
    std::vector<std::size_t> option;
};

[[nodiscard]] DecisionLabels decision_labels(const DecisionQuestion& question);

// The request that reads `question`: the system turn, then one user turn with the state, the
// question and its options under the labels, thinking off, one output token, with the labels'
// token ids (aligned with decision_labels(question).texts) read out exactly. The user turn's first
// part holds the context and the state, with an explicit cache boundary after it; the second holds
// the question.
[[nodiscard]] GenerationRequest make_decision_read(const DecisionRequest& request,
                                                   const DecisionQuestion& question,
                                                   std::vector<ninfer::TokenId> label_tokens);

struct DecisionAnswer {
    // Over the options, in order; sums to 1.
    std::vector<double> probabilities;
    // The labels' total probability over the whole vocabulary.
    double label_mass = 0.0;
    // The model's most likely token is one of the labels.
    bool argmax_is_label = false;
    // Read from the exact readout; false when the core has none and the labels were read from the
    // top alternatives (a label outside them counts as likely as the least likely alternative).
    bool exact = true;
};

// The answer of one read, `label_tokens` aligned with `labels.texts`, or std::runtime_error when
// the read produced no answer token.
[[nodiscard]] DecisionAnswer read_decision(const GenerationOutcome& outcome,
                                           const DecisionLabels& labels,
                                           std::span<const ninfer::TokenId> label_tokens);

// One response, the answers aligned with the request's questions; `input_tokens` and
// `output_tokens` summed over the reads.
[[nodiscard]] std::string make_decision_response(const DecisionRequest& request,
                                                 std::span<const DecisionAnswer> answers,
                                                 int input_tokens, int output_tokens);

} // namespace ninfer::serve
