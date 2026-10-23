// HTTP handling of the typed decision endpoints: POST /v1/systemone and POST /v1/decisions, one
// read per question on the served chat model.
#include "ninfer/request.h"
#include "serve/http_server.h"
#include "serve/http_transport.h"
#include "serve/request_events.h"
#include "serve/request_validation.h"
#include "serve/systemone.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::serve {

void HttpServer::handle_decisions(const httplib::Request& req, httplib::Response& res,
                                  DecisionDialect dialect) {
    const bool system_one = dialect == DecisionDialect::SystemOne;
    const char* protocol  = system_one ? "systemone" : "openai_decisions";
    DecisionRequest request;
    ModelRegistry::Lease lease;
    // The questions' labels as the model's tokens: each must be one token at the reply's start.
    std::vector<DecisionLabels> labels;
    std::vector<std::vector<ninfer::TokenId>> label_tokens;
    try {
        const RequestJson body = parse_json_body(req);
        request       = system_one ? parse_systemone_request(body) : parse_decisions_request(body);
        lease         = lease_model(request.model, req);
        request.model = lease.model();
        std::map<std::string, ninfer::TokenId> resolved;
        for (const DecisionQuestion& question : request.questions) {
            labels.push_back(decision_labels(question));
            std::vector<ninfer::TokenId> tokens;
            for (const std::string& label : labels.back().texts) {
                auto found = resolved.find(label);
                if (found == resolved.end()) {
                    const std::vector<ninfer::TokenId> ids = lease.service().tokenize(label, false);
                    if (ids.size() != 1) {
                        bad_request("label " + label + " is not one token of the model's tokenizer",
                                    "questions");
                    }
                    found = resolved.emplace(label, ids.front()).first;
                }
                tokens.push_back(found->second);
            }
            label_tokens.push_back(std::move(tokens));
        }
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }
    GenerationService& service = lease.service();
    const auto cancelled       = [&req] { return client_disconnected(req); };
    // One read per question. The first runs alone and leaves the state at its cache boundary; the
    // rest resume from there, a lane's worth at a time so the ingress queue keeps room for other
    // clients' requests. A window's reads are submitted as one group: they enter the Engine's queue
    // together, so their questions share one prefill pass. A read is never continued, so its own
    // continuation is the first the cache evicts, and it starts no session lineage.
    const std::size_t lanes =
        std::max<std::size_t>(1, static_cast<std::size_t>(service.options().max_concurrency));
    const ninfer::ContextCacheHints read_cache{
        .retention            = ninfer::CacheRetentionHint::Disposable,
        .update_session_index = false,
    };
    std::vector<DecisionAnswer> answers;
    answers.reserve(request.questions.size());
    int input_tokens  = 0;
    int output_tokens = 0;
    try {
        for (std::size_t begin = 0; begin < request.questions.size();) {
            const std::size_t end =
                std::min(begin == 0 ? std::size_t{1} : begin + lanes, request.questions.size());
            std::vector<GenerationRequest> reads;
            std::vector<StagedRequest> staged;
            for (std::size_t index = begin; index < end; ++index) {
                reads.push_back(make_decision_read(request, request.questions[index],
                                                   label_tokens[index]));
                staged.push_back(service.stage(reads.back(), GenerationConsumerMode::Aggregate,
                                               {.phase_timings = true}, cancelled, read_cache));
            }
            std::vector<PreparedRequest> batch = service.submit_group(std::move(staged));
            std::vector<std::shared_ptr<RequestLifecycle>> lifecycles;
            for (std::size_t offset = 0; offset < batch.size(); ++offset) {
                const RequestLogMetadata metadata{.model = request.model};
                lifecycles.push_back(begin_request(make_request_log_context(
                    ++request_seq_, protocol, reads[offset], metadata, batch[offset])));
            }
            for (std::size_t offset = 0; offset < batch.size(); ++offset) {
                GenerationOutcome outcome;
                try {
                    outcome = service.run(batch[offset], nullptr, cancelled);
                } catch (const ApiException& exception) {
                    lifecycles[offset]->failure(make_generation_request_failure(exception.error()));
                    throw;
                }
                lifecycles[offset]->done(outcome);
                input_tokens += outcome.prompt_tokens;
                output_tokens += outcome.completion_tokens;
                answers.push_back(
                    read_decision(outcome, labels[begin + offset], label_tokens[begin + offset]));
            }
            begin = end;
        }
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        write_openai_error(res, error);
        return;
    }
    res.set_content(make_decision_response(request, answers, input_tokens, output_tokens),
                    "application/json");
}

} // namespace ninfer::serve
