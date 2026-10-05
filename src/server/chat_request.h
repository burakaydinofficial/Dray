// An OpenAI chat request, read: the body's fields become generation parameters
// and the model's chat template renders the prompt, or the request is refused
// with the reason. No transport, no scheduling -- pure, and tested as such.
#pragma once

#include <memory>
#include <string>

#include "engine/engine_types.h"
#include "server/chat_format.h"

namespace dray::server {

// A refusal the client gets as an OpenAI error object.
struct ApiError {
    int         status = 400;
    std::string message;
    std::string code;   // e.g. "unsupported_parameter"; empty = none
};

struct ChatRequest {
    engine::GenParams params;   // prompt, max_tokens, stops, temperature, seed
    std::shared_ptr<const ChatFormat::Rendered> rendered;   // how to parse the output
    bool stream = false;
    bool background = false;
};

// False: *error says why (a 400).
bool parse_chat_request(const std::string& body, const ChatFormat& format,
                        ChatRequest* out, ApiError* error);

}  // namespace dray::server
