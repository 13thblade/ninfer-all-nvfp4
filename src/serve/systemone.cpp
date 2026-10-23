#include "serve/systemone.h"

#include "serve/generation_service.h"
#include "serve/openai_common.h"
#include "serve/request_validation.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::serve {
namespace {

using Json    = RequestJson;
using OutJson = nlohmann::ordered_json;

// The system turn of every read. It keeps the reply's first token on the labels: on a 27B chat
// model the labels' probability mass rose from 0.6-0.9 to 0.98-0.99 with it.
constexpr const char* kSystem =
    "You answer typed questions about a state with one label only: the letter of the best "
    "option, yes or no, or the number of the level. No words before or after the label.";

bool present(const Json& body, const char* key) {
    return body.contains(key) && !body.at(key).is_null();
}

[[noreturn]] void not_implemented(std::string message, std::string param) {
    ApiError error;
    error.status  = 501;
    error.type    = "not_implemented_error";
    error.message = std::move(message);
    error.param   = std::move(param);
    throw ApiException(std::move(error));
}

// A string as given, other JSON as its compact text.
std::string json_text(const Json& value) {
    if (value.is_string()) { return value.get<std::string>(); }
    return value.dump(-1, ' ', false, Json::error_handler_t::replace);
}

std::string required_text(const Json& object, const char* key, const std::string& param) {
    if (!present(object, key)) { bad_request(param + " is required", param); }
    const Json& value = object.at(key);
    if (!value.is_string() && !value.is_object() && !value.is_array()) {
        bad_request(param + " must be a string, an object or an array", param);
    }
    return json_text(value);
}

std::optional<std::string> optional_description(const Json& value, const std::string& param) {
    if (value.is_null()) { return std::nullopt; }
    if (!value.is_string()) { bad_request(param + " must be a string or null", param); }
    return value.get<std::string>();
}

void refuse_unknown_fields(const Json& object, std::initializer_list<const char*> known,
                           const std::string& scope) {
    for (const auto& [key, value] : object.items()) {
        if (std::find_if(known.begin(), known.end(),
                         [&key](const char* name) { return key == name; }) == known.end()) {
            bad_request("unknown field " + scope + key, scope + key);
        }
    }
}

void refuse_media(const Json& body) {
    for (const char* key : {"images", "files", "audio", "videos"}) {
        if (present(body, key)) {
            not_implemented(std::string(key) + " are not supported: decisions read text", key);
        }
    }
}

DecisionType parse_type(const Json& question, const std::string& param, DecisionDialect dialect) {
    const std::string field = param + ".type";
    if (!present(question, "type") || !question.at("type").is_string()) {
        bad_request(field + " must be a string", field);
    }
    const std::string type = question.at("type").get<std::string>();
    if (type == "choice") { return DecisionType::Choice; }
    if (type == "score") { return DecisionType::Score; }
    if (dialect == DecisionDialect::SystemOne && type == "noul") { return DecisionType::Noul; }
    if (dialect == DecisionDialect::OpenAI && type == "predicate") { return DecisionType::Noul; }
    bad_request(field + " must be " +
                    (dialect == DecisionDialect::SystemOne ? "choice, noul or score"
                                                           : "predicate, choice or score"),
                field);
}

void require_option_count(std::size_t count, DecisionType type, const std::string& field) {
    if (type == DecisionType::Choice &&
        (count < 2 || count > kMaximumDecisionOptions)) {
        bad_request(field + " must list 2 to " + std::to_string(kMaximumDecisionOptions) +
                        " options",
                    field);
    }
    if (type == DecisionType::Score && (count < 2 || count > kMaximumDecisionLevels)) {
        bad_request(field + " must list 2 to " + std::to_string(kMaximumDecisionLevels) +
                        " levels",
                    field);
    }
}

// System One's criteria: a choice's {option: description|null}, a score's [level, ...], a noul's
// optional {true: ..., false: ...}.
std::vector<DecisionOption> parse_criteria(const Json& question, DecisionType type,
                                           const std::string& param) {
    const std::string field = param + ".criteria";
    std::vector<DecisionOption> options;
    switch (type) {
    case DecisionType::Choice: {
        if (!present(question, "criteria") || !question.at("criteria").is_object()) {
            bad_request(field + " must map each option to its description or null", field);
        }
        const Json& criteria = question.at("criteria");
        require_option_count(criteria.size(), type, field);
        for (const auto& [name, description] : criteria.items()) {
            if (name.empty()) { bad_request(field + " has an empty option name", field); }
            options.push_back({.value       = Json(name),
                               .text        = name,
                               .description = optional_description(description, field + "." + name)});
        }
        break;
    }
    case DecisionType::Score: {
        if (!present(question, "criteria") || !question.at("criteria").is_array()) {
            bad_request(field + " must be an array of level descriptions, lowest first", field);
        }
        const Json& criteria = question.at("criteria");
        require_option_count(criteria.size(), type, field);
        for (const Json& level : criteria) {
            if (!level.is_string() || level.get_ref<const std::string&>().empty()) {
                bad_request(field + " levels must be non-empty strings", field);
            }
            options.push_back({.value       = Json(options.size()),
                               .text        = level.get<std::string>(),
                               .description = std::nullopt});
        }
        break;
    }
    case DecisionType::Noul: {
        DecisionOption yes{.value = Json("yes"), .text = "yes", .description = std::nullopt};
        DecisionOption no{.value = Json("no"), .text = "no", .description = std::nullopt};
        if (present(question, "criteria")) {
            const Json& criteria = question.at("criteria");
            if (!criteria.is_object()) {
                bad_request(field + " must be null or an object with true and false descriptions",
                            field);
            }
            refuse_unknown_fields(criteria, {"true", "false"}, field + ".");
            if (criteria.contains("true")) {
                yes.description = optional_description(criteria.at("true"), field + ".true");
            }
            if (criteria.contains("false")) {
                no.description = optional_description(criteria.at("false"), field + ".false");
            }
        }
        options.push_back(std::move(yes));
        options.push_back(std::move(no));
        break;
    }
    }
    return options;
}

// OpenAI's choices [{value, description}] and levels [{label, description}].
std::vector<DecisionOption> parse_openai_options(const Json& question, DecisionType type,
                                                 const std::string& param) {
    std::vector<DecisionOption> options;
    const bool choice       = type == DecisionType::Choice;
    const std::string field = param + (choice ? ".choices" : ".levels");
    if (type == DecisionType::Noul) {
        if (present(question, "choices") || present(question, "levels")) {
            bad_request(param + " is a predicate and takes neither choices nor levels", param);
        }
        options.push_back({.value = Json("yes"), .text = "yes", .description = std::nullopt});
        options.push_back({.value = Json("no"), .text = "no", .description = std::nullopt});
        return options;
    }
    if (present(question, choice ? "levels" : "choices")) {
        bad_request(param + " takes " + (choice ? "choices, not levels" : "levels, not choices"),
                    param);
    }
    if (!present(question, choice ? "choices" : "levels") ||
        !question.at(choice ? "choices" : "levels").is_array()) {
        bad_request(field + " must be an array", field);
    }
    const Json& entries = question.at(choice ? "choices" : "levels");
    require_option_count(entries.size(), type, field);
    for (const Json& entry : entries) {
        const std::string scope = field + "[" + std::to_string(options.size()) + "]";
        if (!entry.is_object()) { bad_request(scope + " must be an object", scope); }
        refuse_unknown_fields(entry, {choice ? "value" : "label", "description"}, scope + ".");
        DecisionOption option;
        if (choice) {
            if (!present(entry, "value")) { bad_request(scope + ".value is required", scope); }
            const Json& value = entry.at("value");
            if (value.is_string()) {
                option.text = value.get<std::string>();
            } else if (value.is_boolean()) {
                option.text = value.get<bool>() ? "true" : "false";
            } else {
                bad_request(scope + ".value must be a string or a boolean", scope + ".value");
            }
            option.value = value;
        } else {
            if (!present(entry, "label") || !entry.at("label").is_string()) {
                bad_request(scope + ".label must be a string", scope + ".label");
            }
            option.text  = entry.at("label").get<std::string>();
            option.value = Json(options.size());
        }
        if (option.text.empty()) { bad_request(scope + " has an empty value", scope); }
        if (entry.contains("description")) {
            option.description =
                optional_description(entry.at("description"), scope + ".description");
        }
        for (const DecisionOption& earlier : options) {
            if (earlier.text == option.text) {
                bad_request(scope + " repeats " + option.text, scope);
            }
        }
        options.push_back(std::move(option));
    }
    return options;
}

void require_question_count(std::size_t count, const char* param) {
    if (count == 0) { bad_request(std::string(param) + " must hold at least one question", param); }
    if (count > kMaximumDecisionQuestions) {
        bad_request(std::string(param) + " holds more than " +
                        std::to_string(kMaximumDecisionQuestions) + " questions",
                    param);
    }
}

std::string openai_input_text(const Json& input) {
    if (input.is_string()) { return input.get<std::string>(); }
    if (!input.is_array() || input.empty()) {
        bad_request("input must be a string or a non-empty array of user messages", "input");
    }
    std::string text;
    for (std::size_t index = 0; index < input.size(); ++index) {
        const std::string scope = "input[" + std::to_string(index) + "]";
        const Json& message     = input[index];
        if (!message.is_object()) { bad_request(scope + " must be a message object", scope); }
        refuse_unknown_fields(message, {"role", "content", "type"}, scope + ".");
        if (present(message, "type") && message.at("type") != "message") {
            bad_request(scope + ".type must be message", scope + ".type");
        }
        if (!present(message, "role") || message.at("role") != "user") {
            bad_request(scope + ".role must be user", scope + ".role");
        }
        if (!present(message, "content")) {
            bad_request(scope + ".content is required", scope + ".content");
        }
        const Json& content = message.at("content");
        std::string piece;
        if (content.is_string()) {
            piece = content.get<std::string>();
        } else if (content.is_array()) {
            for (std::size_t part = 0; part < content.size(); ++part) {
                const std::string part_scope = scope + ".content[" + std::to_string(part) + "]";
                const Json& entry            = content[part];
                if (!entry.is_object() || !present(entry, "type") ||
                    entry.at("type") != "input_text") {
                    bad_request(part_scope + " must be an input_text part", part_scope);
                }
                refuse_unknown_fields(entry, {"type", "text"}, part_scope + ".");
                if (!present(entry, "text") || !entry.at("text").is_string()) {
                    bad_request(part_scope + ".text must be a string", part_scope + ".text");
                }
                if (!piece.empty()) { piece += '\n'; }
                piece += entry.at("text").get<std::string>();
            }
        } else {
            bad_request(scope + ".content must be a string or an array of input_text parts",
                        scope + ".content");
        }
        if (!text.empty()) { text += "\n\n"; }
        text += piece;
    }
    return text;
}

ChatTurn text_turn(ChatRole role, std::string text) {
    ChatTurn turn;
    turn.role = role;
    turn.content.push_back(ContentPart{.kind = ContentKind::Text, .text = std::move(text)});
    return turn;
}

std::string label_line(const std::string& label, const DecisionOption& option) {
    std::string line = label + ". " + option.text;
    if (option.description && !option.description->empty()) {
        line += " - " + *option.description;
    }
    return line + "\n";
}

// "A, B or C" for a few labels, "A to Z" for more.
std::string label_span(const std::vector<std::string>& labels) {
    if (labels.size() > 4) { return labels.front() + " to " + labels.back(); }
    std::string span;
    for (std::size_t k = 0; k < labels.size(); ++k) {
        if (k > 0) { span += k + 1 == labels.size() ? " or " : ", "; }
        span += labels[k];
    }
    return span;
}

// The label the prompt shows for each option: the first of its labels.
std::vector<std::string> shown_labels(const DecisionQuestion& question) {
    const DecisionLabels labels = decision_labels(question);
    std::vector<std::string> shown(question.options.size());
    for (std::size_t k = labels.texts.size(); k-- > 0;) {
        shown[labels.option[k]] = labels.texts[k];
    }
    return shown;
}

double finite_probability(double value) {
    return std::isfinite(value) && value > 0.0 ? value : 0.0;
}

} // namespace

DecisionRequest parse_systemone_request(const Json& body) {
    if (!body.is_object()) { bad_request("request body must be a JSON object"); }
    refuse_media(body);
    refuse_unknown_fields(
        body, {"model", "state", "questions", "instructions", "chat_template_kwargs", "joint"},
        "");
    if (present(body, "joint")) {
        if (!body.at("joint").is_boolean()) {
            bad_request("joint must be a boolean", "joint");
        }
        if (body.at("joint").get<bool>()) {
            bad_request("joint reads are not supported: questions are read one at a time",
                        "joint");
        }
    }
    DecisionRequest out;
    out.dialect = DecisionDialect::SystemOne;
    if (present(body, "model")) {
        if (!body.at("model").is_string()) { bad_request("model must be a string", "model"); }
        out.model = body.at("model").get<std::string>();
    }
    out.state = required_text(body, "state", "state");
    if (present(body, "instructions")) { out.instructions = json_text(body.at("instructions")); }
    if (present(body, "chat_template_kwargs")) {
        const Json& kwargs = body.at("chat_template_kwargs");
        if (!kwargs.is_object()) {
            bad_request("chat_template_kwargs must be an object", "chat_template_kwargs");
        }
        if (kwargs.contains("enable_thinking") && kwargs.at("enable_thinking") == true) {
            bad_request("decisions are read with thinking off",
                        "chat_template_kwargs.enable_thinking");
        }
    }
    if (!present(body, "questions") || !body.at("questions").is_object()) {
        bad_request("questions must be an object of question ids", "questions");
    }
    const Json& questions = body.at("questions");
    require_question_count(questions.size(), "questions");
    for (const auto& [id, question] : questions.items()) {
        const std::string param = "questions." + id;
        if (id.empty() || id.size() > kMaximumDecisionIdLength) {
            bad_request("question ids are 1 to " + std::to_string(kMaximumDecisionIdLength) +
                            " characters",
                        "questions");
        }
        if (!question.is_object()) { bad_request(param + " must be an object", param); }
        refuse_unknown_fields(question, {"type", "instructions", "criteria"}, param + ".");
        DecisionQuestion parsed;
        parsed.id           = id;
        parsed.type         = parse_type(question, param, DecisionDialect::SystemOne);
        parsed.instructions = required_text(question, "instructions", param + ".instructions");
        parsed.options      = parse_criteria(question, parsed.type, param);
        out.questions.push_back(std::move(parsed));
    }
    return out;
}

DecisionRequest parse_decisions_request(const Json& body) {
    if (!body.is_object()) { bad_request("request body must be a JSON object"); }
    refuse_media(body);
    refuse_unknown_fields(body, {"model", "input", "questions", "safety_identifier"}, "");
    DecisionRequest out;
    out.dialect = DecisionDialect::OpenAI;
    if (present(body, "model")) {
        if (!body.at("model").is_string()) { bad_request("model must be a string", "model"); }
        out.model = body.at("model").get<std::string>();
    }
    if (!present(body, "input")) { bad_request("input is required", "input"); }
    out.state = openai_input_text(body.at("input"));
    if (!present(body, "questions") || !body.at("questions").is_array()) {
        bad_request("questions must be an array", "questions");
    }
    const Json& questions = body.at("questions");
    require_question_count(questions.size(), "questions");
    for (std::size_t index = 0; index < questions.size(); ++index) {
        const std::string param = "questions[" + std::to_string(index) + "]";
        const Json& question    = questions[index];
        if (!question.is_object()) { bad_request(param + " must be an object", param); }
        refuse_unknown_fields(question, {"type", "name", "instructions", "choices", "levels"},
                              param + ".");
        DecisionQuestion parsed;
        parsed.named = present(question, "name");
        if (parsed.named) {
            if (!question.at("name").is_string()) {
                bad_request(param + ".name must be a string or null", param + ".name");
            }
            parsed.id = question.at("name").get<std::string>();
        }
        parsed.type = parse_type(question, param, DecisionDialect::OpenAI);
        if (!present(question, "instructions") || !question.at("instructions").is_string()) {
            bad_request(param + ".instructions must be a string", param + ".instructions");
        }
        parsed.instructions = question.at("instructions").get<std::string>();
        parsed.options      = parse_openai_options(question, parsed.type, param);
        out.questions.push_back(std::move(parsed));
    }
    return out;
}

DecisionLabels decision_labels(const DecisionQuestion& question) {
    DecisionLabels labels;
    const auto add = [&labels](std::string text, std::size_t option) {
        labels.texts.push_back(std::move(text));
        labels.option.push_back(option);
    };
    switch (question.type) {
    case DecisionType::Choice:
        for (std::size_t k = 0; k < question.options.size(); ++k) {
            add(std::string(1, static_cast<char>('A' + k)), k);
        }
        break;
    case DecisionType::Score:
        for (std::size_t k = 0; k < question.options.size(); ++k) { add(std::to_string(k), k); }
        break;
    case DecisionType::Noul:
        // Both cases: a model that answers "Yes" is answering yes.
        add("yes", 0);
        add("Yes", 0);
        add("no", 1);
        add("No", 1);
        break;
    }
    return labels;
}

GenerationRequest make_decision_read(const DecisionRequest& request,
                                     const DecisionQuestion& question,
                                     std::vector<ninfer::TokenId> label_tokens) {
    const std::vector<std::string> labels = shown_labels(question);
    // The context and the state are the same for every question of the request: a cache boundary
    // after them lets the later reads resume from the first one's state instead of prefilling it
    // again. The two parts render as one user turn, byte for byte.
    std::string shared;
    if (!request.instructions.empty()) { shared += request.instructions + "\n\n"; }
    shared += "State:\n" + request.state + "\n\n";
    std::string text = "Question: " + question.instructions + "\n";
    switch (question.type) {
    case DecisionType::Choice:
        text += "Options:\n";
        for (std::size_t k = 0; k < question.options.size(); ++k) {
            text += label_line(labels[k], question.options[k]);
        }
        text += "Reply with a single letter (" + label_span(labels) + ") and nothing else.";
        break;
    case DecisionType::Noul:
        for (std::size_t k = 0; k < question.options.size(); ++k) {
            const DecisionOption& option = question.options[k];
            if (option.description && !option.description->empty()) {
                text += labels[k] + ": " + *option.description + "\n";
            }
        }
        text += "Answer with one word, yes or no, in lower case.";
        break;
    case DecisionType::Score:
        text += "Levels, lowest first:\n";
        for (std::size_t k = 0; k < question.options.size(); ++k) {
            text += label_line(labels[k], question.options[k]);
        }
        text += "Reply with a single number (" + label_span(labels) + ") and nothing else.";
        break;
    }
    GenerationRequest out;
    // Every read of every request starts with this turn; a boundary after it lets a request's
    // first read resume there. The planner drops a structural boundary this close to the prompt
    // start, an explicit one it keeps.
    ChatTurn system            = text_turn(ChatRole::System, kSystem);
    system.cache_boundary_after = CacheBoundary{};
    out.messages.push_back(std::move(system));
    ChatTurn user = text_turn(ChatRole::User, std::move(shared));
    user.content.front().cache_boundary_after = CacheBoundary{};
    user.content.push_back(ContentPart{.kind = ContentKind::Text, .text = std::move(text)});
    out.messages.push_back(std::move(user));
    out.enable_thinking            = false;
    out.max_tokens                 = 1;
    out.sampling.temperature       = 0.0;
    out.sampling.presence_penalty  = 0.0;
    out.sampling.frequency_penalty = 0.0;
    out.logprobs                   = true;
    out.readout_tokens             = std::move(label_tokens);
    // One token drafts nothing; --spec's drafter would only run over the prompt.
    out.speculative_drafting = false;
    // The server's default graft would be a hidden prefix the question does not expect.
    out.graft = std::string();
    return out;
}

DecisionAnswer read_decision(const GenerationOutcome& outcome, const DecisionLabels& labels,
                             std::span<const ninfer::TokenId> label_tokens) {
    if (outcome.content_logprobs.empty() || outcome.content_logprobs.front().top_ids[0] < 0) {
        throw std::runtime_error("the decision read produced no answer token");
    }
    if (label_tokens.size() != labels.option.size() || labels.option.empty()) {
        throw std::logic_error("decision label tokens do not match the labels");
    }
    const ninfer::TokenLogprob& first = outcome.content_logprobs.front();
    const std::size_t count           = label_tokens.size();
    const std::size_t options = 1 + *std::max_element(labels.option.begin(), labels.option.end());
    DecisionAnswer answer;
    answer.exact = first.readout.size() == count;
    std::vector<double> logprobs(count, -std::numeric_limits<double>::infinity());
    if (answer.exact) {
        for (std::size_t k = 0; k < count; ++k) { logprobs[k] = first.readout[k]; }
    } else {
        // The top alternatives only: a label outside them counts as likely as the least likely of
        // them, its upper bound.
        double least = -std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < ninfer::kMaximumTokenLogprobs && first.top_ids[j] >= 0; ++j) {
            least = first.top_values[j];
            for (std::size_t k = 0; k < count; ++k) {
                if (first.top_ids[j] == label_tokens[k]) { logprobs[k] = least; }
            }
        }
        for (double& value : logprobs) {
            if (!std::isfinite(value)) { value = least; }
        }
    }
    answer.probabilities.assign(options, 0.0);
    for (std::size_t k = 0; k < count; ++k) {
        const double probability = finite_probability(std::exp(logprobs[k]));
        answer.probabilities[labels.option[k]] += probability;
        answer.label_mass += probability;
    }
    if (answer.label_mass > 0.0) {
        for (double& probability : answer.probabilities) { probability /= answer.label_mass; }
    } else {
        for (double& probability : answer.probabilities) {
            probability = 1.0 / static_cast<double>(options);
        }
    }
    answer.argmax_is_label =
        std::find(label_tokens.begin(), label_tokens.end(), first.top_ids[0]) != label_tokens.end();
    return answer;
}

std::string make_decision_response(const DecisionRequest& request,
                                   std::span<const DecisionAnswer> answers, int input_tokens,
                                   int output_tokens) {
    if (answers.size() != request.questions.size()) {
        throw std::logic_error("decision answers do not match the questions");
    }
    const bool system_one = request.dialect == DecisionDialect::SystemOne;
    OutJson body;
    body["id"]      = new_openai_decision_id();
    body["object"]  = system_one ? "structured_decision" : "decision";
    body["created"] = unix_time_now();
    body["model"]   = request.model;
    OutJson rendered    = system_one ? OutJson::object() : OutJson::array();
    OutJson diagnostics = OutJson::object();
    for (std::size_t q = 0; q < answers.size(); ++q) {
        const DecisionQuestion& question = request.questions[q];
        const DecisionAnswer& answer     = answers[q];
        const auto best                  = static_cast<std::size_t>(
            std::max_element(answer.probabilities.begin(), answer.probabilities.end()) -
            answer.probabilities.begin());
        const double confidence = answer.probabilities[best] * answer.label_mass;
        OutJson entry;
        if (system_one) {
            switch (question.type) {
            case DecisionType::Choice: {
                OutJson probabilities = OutJson::object();
                for (std::size_t k = 0; k < question.options.size(); ++k) {
                    probabilities[question.options[k].text] = answer.probabilities[k];
                }
                entry = {{"type", "choice"},
                         {"choice", question.options[best].text},
                         {"probabilities", std::move(probabilities)},
                         {"confidence", confidence}};
                break;
            }
            case DecisionType::Noul:
                entry = {{"type", "noul"},
                         {"noul", answer.probabilities[0]},
                         {"probabilities",
                          {{"yes", answer.probabilities[0]}, {"no", answer.probabilities[1]}}},
                         {"confidence", confidence}};
                break;
            case DecisionType::Score: {
                OutJson legend        = OutJson::object();
                OutJson probabilities = OutJson::object();
                double score          = 0.0;
                for (std::size_t k = 0; k < question.options.size(); ++k) {
                    const std::string index = std::to_string(k);
                    legend[index]           = question.options[k].text;
                    probabilities[index]    = answer.probabilities[k];
                    score += answer.probabilities[k] * static_cast<double>(k);
                }
                entry = {{"type", "score"},
                         {"score", score},
                         {"legend", std::move(legend)},
                         {"probabilities", std::move(probabilities)},
                         {"confidence", confidence}};
                break;
            }
            }
            rendered[question.id] = std::move(entry);
            diagnostics[question.id] = {{"label_mass", answer.label_mass},
                                        {"argmax_is_label", answer.argmax_is_label},
                                        {"readout", answer.exact ? "exact" : "top_k"}};
            continue;
        }
        const OutJson name = question.named ? OutJson(question.id) : OutJson(nullptr);
        switch (question.type) {
        case DecisionType::Noul:
            entry = {{"type", "predicate"}, {"name", name}, {"probability", answer.probabilities[0]}};
            break;
        case DecisionType::Choice: {
            OutJson probabilities = OutJson::array();
            for (std::size_t k = 0; k < question.options.size(); ++k) {
                probabilities.push_back({{"value", question.options[k].value},
                                         {"probability", answer.probabilities[k]}});
            }
            entry = {{"type", "choice"},
                     {"name", name},
                     {"choice", question.options[best].value},
                     {"probabilities", std::move(probabilities)},
                     {"confidence", confidence}};
            break;
        }
        case DecisionType::Score: {
            OutJson probabilities = OutJson::array();
            double score          = 0.0;
            for (std::size_t k = 0; k < question.options.size(); ++k) {
                probabilities.push_back({{"value", k},
                                         {"label", question.options[k].text},
                                         {"probability", answer.probabilities[k]}});
                score += answer.probabilities[k] * static_cast<double>(k);
            }
            entry = {{"type", "score"},
                     {"name", name},
                     {"score", score},
                     {"probabilities", std::move(probabilities)},
                     {"confidence", confidence}};
            break;
        }
        }
        rendered.push_back(std::move(entry));
    }
    body["answers"] = std::move(rendered);
    if (system_one) {
        body["usage"] = {{"input_tokens", input_tokens}, {"output_tokens", output_tokens}};
        body["diagnostics"] = std::move(diagnostics);
    } else {
        body["usage"] = {{"input_tokens", input_tokens},
                         {"output_tokens", output_tokens},
                         {"total_tokens", input_tokens + output_tokens}};
    }
    return body.dump(-1, ' ', false, OutJson::error_handler_t::replace);
}

} // namespace ninfer::serve
