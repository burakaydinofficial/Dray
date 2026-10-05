#include "server/chat_response.h"

namespace dray::server {

using nlohmann::json;

const char* finish_reason(const engine::GenResult& r, bool tool_calls) {
    if (r.cancelled) return "cancelled";
    if (!r.truncated_by_eog) return "length";
    return tool_calls ? "tool_calls" : "stop";
}

json error_body(const ApiError& e) {
    json err = { { "message", e.message } };
    if (!e.code.empty()) err["code"] = e.code;
    return json{ { "error", err } };
}

bool untrustworthy(const engine::GenResult& r, bool tainted) {
    return r.aborted || r.cancelled || tainted;
}

json untrustworthy_body(const engine::GenResult& r, bool tainted) {
    const char* why =
        r.aborted   ? "generation aborted: a weight failed to materialise, so the output was "
                      "computed against poison and is not trustworthy"
      : r.cancelled ? "generation cancelled before completion"
                    : "engine is tainted; earlier failures make this output untrustworthy";
    return json{
        { "error", { { "message", why }, { "type", "dray_untrustworthy_output" } } },
        { "dray", { { "aborted", r.aborted },
                       { "cancelled", r.cancelled },
                       { "tainted", tainted },
                       { "partial_tokens", r.tokens_out } } },
    };
}

json completion_body(const ResponseId& id, const engine::GenResult& r, const ChatOutput& out,
                     bool tainted) {
    return json{
        { "id", id.id },
        { "object", "chat.completion" },
        { "created", id.created },                                   // T9
        { "model", id.model },
        { "choices", { {
            { "index", 0 },
            // The model's own format parsed the text: content,
            // reasoning_content and tool_calls, as OpenAI clients expect.
            { "message", json::parse(out.message_json()) },
            { "logprobs", nullptr },                                  // T9
            { "finish_reason", finish_reason(r, out.has_tool_calls()) },
        } } },
        // T9: prompt_tokens and total_tokens are non-optional in every real
        // client's schema; the partial object crashed openai-python.
        { "usage", { { "prompt_tokens", r.tokens_in },
                     { "completion_tokens", r.tokens_out },
                     { "total_tokens", r.tokens_in + r.tokens_out } } },
        { "dray", { { "chat_template", "model (jinja)" },
                       { "aborted", r.aborted },
                       { "tainted", tainted } } },
    };
}

json chunk_body(const ResponseId& id, const json& delta, const char* finish) {
    return json{
        { "id", id.id },
        { "object", "chat.completion.chunk" },
        { "created", id.created },
        { "model", id.model },
        { "choices", { { { "index", 0 },
                         { "delta", delta },
                         { "finish_reason", finish ? json(finish) : json(nullptr) } } } },
    };
}

json error_chunk_body(const ResponseId& id, const std::string& message) {
    json j = chunk_body(id, json::object(), "error");
    j["error"] = { { "message", message } };
    return j;
}

}  // namespace dray::server
