// chat_request / chat_response: the OpenAI /v1/chat/completions contract --
// what a request body becomes, what is refused and why, and the exact shape of
// every object sent back. Pure functions; a real template, no model.

#include "harness.h"

#include <fstream>
#include <iterator>
#include <string>

#include "json.hpp"

#include "server/chat_format.h"
#include "server/chat_request.h"
#include "server/chat_response.h"

using dray::engine::GenResult;
using dray::server::ApiError;
using dray::server::ChatFormat;
using dray::server::ChatOutput;
using dray::server::ChatRequest;
using dray::server::ResponseId;
using nlohmann::json;
namespace srv = dray::server;

namespace {

const ChatFormat& qwen() {
    static const ChatFormat f([] {
        std::ifstream in(std::string(DRAY_LLAMA_TEMPLATES) + "/Qwen-Qwen3-0.6B.jinja", std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }());
    return f;
}

const std::string kMsgs = R"("messages":[{"role":"user","content":"hi"}])";

bool parse(const std::string& extra, ChatRequest* rq, ApiError* err) {
    return srv::parse_chat_request("{" + kMsgs + (extra.empty() ? "" : "," + extra) + "}", qwen(), rq, err);
}

}  // namespace

LZ_TEST(a_plain_request_becomes_generation_parameters) {
    ChatRequest rq;
    ApiError err;
    LZ_REQUIRE(parse(R"("temperature":0,"seed":7,"stream":true,"stop":"END")", &rq, &err));
    LZ_CHECK(rq.params.prompt.find("hi") != std::string::npos);
    LZ_CHECK(rq.params.temperature == 0.0f);
    LZ_CHECK_EQ(rq.params.seed, 7u);
    LZ_CHECK(rq.stream);
    LZ_CHECK(!rq.background);
    LZ_CHECK(!rq.params.stop.empty());
    LZ_CHECK_EQ(rq.params.stop.front(), std::string("END"));
    LZ_CHECK(rq.rendered != nullptr);
}

LZ_TEST(max_tokens_defaults_to_the_context_and_is_clamped) {
    ChatRequest a, b, c, d, e;
    ApiError err;
    LZ_REQUIRE(parse("", &a, &err));
    LZ_CHECK_EQ(a.params.max_tokens, 1 << 24);                 // until the context ends
    LZ_REQUIRE(parse(R"("max_tokens":null)", &b, &err));
    LZ_CHECK_EQ(b.params.max_tokens, 1 << 24);
    LZ_REQUIRE(parse(R"("max_tokens":50,"max_completion_tokens":20)", &c, &err));
    LZ_CHECK_EQ(c.params.max_tokens, 20);                      // the current field wins
    LZ_REQUIRE(parse(R"("max_tokens":1e20)", &d, &err));
    LZ_CHECK_EQ(d.params.max_tokens, 1 << 24);                 // clamped, not UB
    LZ_REQUIRE(parse(R"("max_tokens":-5)", &e, &err));
    LZ_CHECK_EQ(e.params.max_tokens, 1);
}

LZ_TEST(what_cannot_be_honoured_is_refused_not_ignored) {
    ChatRequest rq;
    ApiError err;
    LZ_CHECK(!parse(R"("response_format":{"type":"json_object"})", &rq, &err));
    LZ_CHECK_EQ(err.status, 400);
    LZ_CHECK_EQ(err.code, std::string("unsupported_parameter"));
    LZ_CHECK(!parse(R"("n":2)", &rq, &err));
    LZ_CHECK(!srv::parse_chat_request("{}", qwen(), &rq, &err));
    LZ_CHECK_EQ(err.message, std::string("messages[] required"));
    LZ_CHECK(!srv::parse_chat_request("{", qwen(), &rq, &err));
    LZ_CHECK(!parse(R"("temperature":"hot")", &rq, &err));   // wrong type: a 400, not a 500
    LZ_CHECK(err.message.find("invalid request field") != std::string::npos);
}

LZ_TEST(tools_are_accepted_now) {
    ChatRequest rq;
    ApiError err;
    LZ_CHECK(parse(R"("tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{}}}}])",
                   &rq, &err));
    LZ_CHECK(ChatFormat::has_tools(*rq.rendered));
}

LZ_TEST(the_finish_reason_rule) {
    GenResult r;
    r.truncated_by_eog = true;
    LZ_CHECK_EQ(std::string(srv::finish_reason(r, false)), std::string("stop"));
    LZ_CHECK_EQ(std::string(srv::finish_reason(r, true)), std::string("tool_calls"));
    r.truncated_by_eog = false;   // a limit ended it
    LZ_CHECK_EQ(std::string(srv::finish_reason(r, true)), std::string("length"));   // never a cut-off call
    r.cancelled = true;
    LZ_CHECK_EQ(std::string(srv::finish_reason(r, false)), std::string("cancelled"));
}

LZ_TEST(an_untrustworthy_run_is_never_a_completion) {
    GenResult ok, aborted, cancelled;
    aborted.aborted = true;
    cancelled.cancelled = true;
    LZ_CHECK(!srv::untrustworthy(ok, false));
    LZ_CHECK(srv::untrustworthy(ok, true));          // tainted engine
    LZ_CHECK(srv::untrustworthy(aborted, false));
    LZ_CHECK(srv::untrustworthy(cancelled, false));
    const json j = srv::untrustworthy_body(aborted, false);
    LZ_CHECK_EQ(j["error"]["type"].get<std::string>(), std::string("dray_untrustworthy_output"));
    LZ_CHECK(j["dray"]["aborted"].get<bool>());
}

LZ_TEST(the_completion_and_chunk_shapes) {
    ChatRequest rq;
    ApiError err;
    LZ_REQUIRE(parse("", &rq, &err));
    GenResult r;
    r.truncated_by_eog = true;
    r.tokens_in = 12;
    r.tokens_out = 3;
    ChatOutput out(rq.rendered);
    out.feed("Hello!");
    out.finish();
    const ResponseId id{ "ldg-9", "m.gguf", 1234 };
    const json c = srv::completion_body(id, r, out, false);
    LZ_CHECK_EQ(c["object"].get<std::string>(), std::string("chat.completion"));
    LZ_CHECK_EQ(c["choices"][0]["message"]["content"].get<std::string>(), std::string("Hello!"));
    LZ_CHECK_EQ(c["choices"][0]["finish_reason"].get<std::string>(), std::string("stop"));
    LZ_CHECK_EQ(c["usage"]["total_tokens"].get<int>(), 15);
    LZ_CHECK(c["choices"][0]["logprobs"].is_null());

    const json k = srv::chunk_body(id, json{ { "content", "x" } }, nullptr);
    LZ_CHECK_EQ(k["object"].get<std::string>(), std::string("chat.completion.chunk"));
    LZ_CHECK(k["choices"][0]["finish_reason"].is_null());
    LZ_CHECK_EQ(k["created"].get<int64_t>(), 1234);
    const json e = srv::error_chunk_body(id, "boom");
    LZ_CHECK_EQ(e["choices"][0]["finish_reason"].get<std::string>(), std::string("error"));
    LZ_CHECK_EQ(e["error"]["message"].get<std::string>(), std::string("boom"));
}
