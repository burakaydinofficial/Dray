#include "server/chat_request.h"

#include <cstdint>
#include <exception>

#include "json.hpp"

namespace dray::server {

using nlohmann::json;

bool parse_chat_request(const std::string& text, const ChatFormat& format,
                        ChatRequest* out, ApiError* error) {
    auto refuse = [error](const std::string& message, const std::string& code = "") {
        error->status = 400;
        error->message = message;
        error->code = code;
        return false;
    };
    json body;
    try {
        body = json::parse(text);
    } catch (const std::exception& ex) {
        return refuse(ex.what());
    }
    if (!body.is_object() || !body.contains("messages") || !body["messages"].is_array()) {
        return refuse("messages[] required");
    }
    // T10: refuse what would be silently discarded -- an ignored parameter is
    // indistinguishable from a honoured one, which Invariant 6 forbids.
    for (const char* unsup : { "response_format", "logit_bias", "logprobs" }) {
        auto it = body.find(unsup);
        if (it != body.end() && !it->is_null()) {
            return refuse(std::string(unsup) + " is not supported by this server", "unsupported_parameter");
        }
    }
    {
        auto it = body.find("n");
        if (it != body.end() && it->is_number() && it->get<double>() > 1) {
            return refuse("n>1 is not supported", "unsupported_parameter");
        }
    }

    engine::GenParams& gp = out->params;
    // Swarm S8: a present null ("max_tokens": null is an ordinary client
    // payload) or a wrong type is a 400 with a message, never an opaque 500.
    try {
        std::string rerr;
        out->rendered = format.render(text, &rerr);
        if (!out->rendered) return refuse(rerr);
        gp.prompt = ChatFormat::prompt(*out->rendered);

        auto num = [&body](const char* k, double def) {
            auto it = body.find(k);
            return (it == body.end() || it->is_null()) ? def : it->get<double>();
        };
        // Clamp BEFORE casting: converting an out-of-range double to an integer
        // is undefined behaviour, and {"max_tokens": 1e20} is well-formed JSON
        // that once reached static_cast<int32_t> raw -- INT32_MIN and a 200 with
        // empty content instead of a 400 (2026-08-24 audit). NaN -> the default.
        auto num_i32 = [&num](const char* k, int32_t def, int32_t lo, int32_t hi) {
            const double d = num(k, static_cast<double>(def));
            if (d != d) return def;
            if (d < static_cast<double>(lo)) return lo;
            if (d > static_cast<double>(hi)) return hi;
            return static_cast<int32_t>(d);
        };
        auto num_u32 = [&num](const char* k, uint32_t def) {
            const double d = num(k, static_cast<double>(def));
            if (d != d || d < 0.0) return def;
            if (d > 4294967295.0) return static_cast<uint32_t>(4294967295u);
            return static_cast<uint32_t>(d);
        };
        auto flag = [&body](const char* k) {
            auto it = body.find(k);
            return it != body.end() && it->is_boolean() && it->get<bool>();
        };
        // T10: max_completion_tokens is what current SDKs send; it wins over the
        // deprecated max_tokens. Neither: until the context ends, as the OpenAI
        // API does -- agents rarely send one, and a fixed default cut their edits.
        gp.max_tokens = num_i32("max_completion_tokens",
                                num_i32("max_tokens", 1 << 24, 1, 1 << 24), 1, 1 << 24);
        // T10: stop may be a string or an array of strings.
        auto sit = body.find("stop");
        if (sit != body.end() && !sit->is_null()) {
            if (sit->is_string()) {
                gp.stop.push_back(sit->get<std::string>());
            } else if (sit->is_array()) {
                for (const auto& s : *sit) {
                    if (s.is_string()) gp.stop.push_back(s.get<std::string>());
                }
            }
        }
        // End-of-turn markers the model's template needs as stops.
        for (const std::string& s : ChatFormat::stops(*out->rendered)) gp.stop.push_back(s);
        gp.temperature = static_cast<float>(num("temperature", 0.8));
        gp.seed = num_u32("seed", 0);
        out->stream = flag("stream");
        out->background = flag("background");
    } catch (const std::exception& ex) {
        return refuse(std::string("invalid request field: ") + ex.what());
    }
    return true;
}

}  // namespace dray::server
