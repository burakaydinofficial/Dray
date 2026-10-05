// The OpenAI wire format of /v1/chat/completions: the completion object, the
// stream chunks, the finish reason and the error objects. Pure functions of a
// result -- no transport, no engine -- so each rule lives in exactly one place.
#pragma once

#include <cstdint>
#include <string>

#include "json.hpp"

#include "engine/engine_types.h"
#include "server/chat_format.h"
#include "server/chat_request.h"

namespace dray::server {

// Identifies one response on the wire.
struct ResponseId {
    std::string id;
    std::string model;
    int64_t     created = 0;   // F14: one timestamp per response
};

// Why the generation ended. A limit that cut the output off wins: a truncated
// tool call is "length", never a call for the client to execute.
const char* finish_reason(const engine::GenResult& r, bool tool_calls);

// {"error": {message, code?}}
nlohmann::json error_body(const ApiError& e);

// A run that computed against poison, was cancelled, or ran on a tainted
// engine is NOT a completion (2026-08-24 audit: the one shape where a
// materialise failure reached a user was a clean-looking 200). Null = fine.
bool untrustworthy(const engine::GenResult& r, bool tainted);
nlohmann::json untrustworthy_body(const engine::GenResult& r, bool tainted);

// The non-streaming response.
nlohmann::json completion_body(const ResponseId& id, const engine::GenResult& r,
                               const ChatOutput& out, bool tainted);

// One streaming chunk; `finish` null while the response is still going.
nlohmann::json chunk_body(const ResponseId& id, const nlohmann::json& delta, const char* finish);
// The terminal chunk of a failed stream: the error reaches the client in-band
// (Swarm S1), with finish_reason "error".
nlohmann::json error_chunk_body(const ResponseId& id, const std::string& message);

}  // namespace dray::server
