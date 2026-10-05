// ChatFormat / ChatOutput: OpenAI requests rendered through a model's own Jinja
// template (tools included), and generated text parsed back into content,
// reasoning_content and tool_calls -- whole and incrementally. Real templates
// from llama.cpp's repo; no model needed.

#include "harness.h"

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "json.hpp"

#include "server/chat_format.h"

using dray::server::ChatFormat;
using dray::server::ChatOutput;
using nlohmann::json;

namespace {

std::string read_template(const char* name) {
    std::ifstream f(std::string(DRAY_LLAMA_TEMPLATES) + "/" + name, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

const char* kTools = R"([{"type":"function","function":{"name":"read_file",
  "description":"Read a file","parameters":{"type":"object",
  "properties":{"path":{"type":"string"}},"required":["path"]}}}])";

std::string request(const std::string& messages, bool tools) {
    return std::string("{\"messages\":") + messages + (tools ? std::string(",\"tools\":") + kTools : "") + "}";
}

// Everything the deltas carried, reassembled as a client would.
struct Reassembled {
    std::string content, reasoning, name, arguments, id;
};
Reassembled reassemble(const std::vector<std::string>& deltas) {
    Reassembled r;
    for (const std::string& d : deltas) {
        const json j = json::parse(d);
        if (j.contains("content")) r.content += j["content"].get<std::string>();
        if (j.contains("reasoning_content")) r.reasoning += j["reasoning_content"].get<std::string>();
        if (j.contains("tool_calls")) {
            const json& t = j["tool_calls"][0];
            if (t.contains("id")) r.id = t["id"].get<std::string>();
            if (t.contains("function")) {
                if (t["function"].contains("name")) r.name += t["function"]["name"].get<std::string>();
                if (t["function"].contains("arguments")) r.arguments += t["function"]["arguments"].get<std::string>();
            }
        }
    }
    return r;
}

// Feeds `text` in pieces of `step` bytes, then finishes.
std::vector<std::string> stream(ChatOutput& out, const std::string& text, size_t step) {
    std::vector<std::string> all;
    for (size_t i = 0; i < text.size(); i += step) {
        for (auto& d : out.feed(text.substr(i, step))) all.push_back(d);
    }
    for (auto& d : out.finish()) all.push_back(d);
    return all;
}

const std::string kUser = R"([{"role":"user","content":"What is in a.txt?"}])";

}  // namespace

LZ_TEST(tools_are_rendered_into_the_prompt_by_the_models_template) {
    ChatFormat f(read_template("Qwen-Qwen3-0.6B.jinja"));
    std::string err;
    auto r = f.render(request(kUser, true), &err);
    LZ_REQUIRE(r != nullptr);
    const std::string& p = ChatFormat::prompt(*r);
    LZ_CHECK(p.find("read_file") != std::string::npos);        // the tool definition
    LZ_CHECK(p.find("What is in a.txt?") != std::string::npos);
    LZ_CHECK(p.find("<|im_start|>assistant") != std::string::npos);
    LZ_CHECK(ChatFormat::has_tools(*r));
    auto plain = f.render(request(kUser, false), &err);
    LZ_REQUIRE(plain != nullptr);
    LZ_CHECK(ChatFormat::prompt(*plain).find("read_file") == std::string::npos);
    LZ_CHECK(!ChatFormat::has_tools(*plain));
}

LZ_TEST(thinking_goes_to_reasoning_content_whole_and_streamed) {
    ChatFormat f(read_template("Qwen-Qwen3-0.6B.jinja"));
    std::string err;
    auto r = f.render(request(kUser, false), &err);
    LZ_REQUIRE(r != nullptr);
    const std::string text = "<think>\nThe user wants a file.\n</think>\n\nIt says hello.";
    for (size_t step : {size_t(1), size_t(3), text.size()}) {
        ChatOutput out(r);
        const Reassembled got = reassemble(stream(out, text, step));
        LZ_CHECK_EQ(got.content, std::string("It says hello."));
        LZ_CHECK(got.reasoning.find("The user wants a file.") != std::string::npos);
        const json m = json::parse(out.message_json());
        LZ_CHECK_EQ(m["content"].get<std::string>(), std::string("It says hello."));
        LZ_CHECK(!out.has_tool_calls());
    }
}

LZ_TEST(a_tool_call_is_parsed_into_tool_calls_whole_and_streamed) {
    ChatFormat f(read_template("Qwen-Qwen3-0.6B.jinja"));
    std::string err;
    auto r = f.render(request(kUser, true), &err);
    LZ_REQUIRE(r != nullptr);
    const std::string text =
        "<think>\nI should read it.\n</think>\n\n"
        "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.txt\"}}\n</tool_call>";
    for (size_t step : {size_t(1), size_t(7), text.size()}) {
        ChatOutput out(r);
        const Reassembled got = reassemble(stream(out, text, step));
        LZ_CHECK_EQ(got.name, std::string("read_file"));
        LZ_CHECK(!got.id.empty());
        LZ_CHECK_EQ(json::parse(got.arguments)["path"].get<std::string>(), std::string("a.txt"));
        LZ_CHECK(got.content.find("tool_call") == std::string::npos);   // not leaked as text
        LZ_CHECK(out.has_tool_calls());
        const json m = json::parse(out.message_json());
        LZ_CHECK_EQ(m["tool_calls"][0]["function"]["name"].get<std::string>(), std::string("read_file"));
        LZ_CHECK_EQ(m["tool_calls"][0]["type"].get<std::string>(), std::string("function"));
    }
}

LZ_TEST(a_conversation_with_tool_results_renders) {
    ChatFormat f(read_template("Qwen-Qwen3-0.6B.jinja"));
    const std::string msgs = R"([
      {"role":"user","content":"What is in a.txt?"},
      {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function",
        "function":{"name":"read_file","arguments":"{\"path\":\"a.txt\"}"}}]},
      {"role":"tool","tool_call_id":"call_1","content":"hello world"}])";
    std::string err;
    auto r = f.render(request(msgs, true), &err);
    LZ_REQUIRE(r != nullptr);
    const std::string& p = ChatFormat::prompt(*r);
    LZ_CHECK(p.find("hello world") != std::string::npos);   // the tool result is in the prompt
    LZ_CHECK(p.find("a.txt") != std::string::npos);
}

LZ_TEST(an_xml_tool_call_format_is_parsed_too) {
    // Qwen3-Coder writes tool calls as XML, not JSON: the model's own format decides.
    ChatFormat f(read_template("Qwen3-Coder.jinja"));
    std::string err;
    auto r = f.render(request(kUser, true), &err);
    LZ_REQUIRE(r != nullptr);
    const std::string text =
        "<tool_call>\n<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>";
    ChatOutput out(r);
    const Reassembled got = reassemble(stream(out, text, 5));
    LZ_CHECK_EQ(got.name, std::string("read_file"));
    LZ_CHECK_EQ(json::parse(got.arguments)["path"].get<std::string>(), std::string("a.txt"));
}

LZ_TEST(bad_requests_are_refused_with_a_reason) {
    ChatFormat f(read_template("Qwen-Qwen3-0.6B.jinja"));
    std::string err;
    LZ_CHECK(f.render("{\"nomessages\":[]}", &err) == nullptr);
    LZ_CHECK(!err.empty());
    err.clear();
    LZ_CHECK(f.render("not json", &err) == nullptr);
    LZ_CHECK(!err.empty());
}

LZ_TEST(output_that_never_parses_is_kept_as_content) {
    ChatFormat f(read_template("Qwen-Qwen3-0.6B.jinja"));
    std::string err;
    auto r = f.render(request(kUser, true), &err);
    LZ_REQUIRE(r != nullptr);
    ChatOutput out(r);
    const std::string text = "Plain answer with a stray { brace.";
    const Reassembled got = reassemble(stream(out, text, 4));
    LZ_CHECK_EQ(got.content, text);   // nothing dropped
    LZ_CHECK(!out.has_tool_calls());
}

LZ_TEST(a_tool_call_cut_off_by_the_token_limit_is_not_lost) {
    // max_tokens (or the context) ended the generation inside a tool call: the
    // final parse cannot complete it, and the text must still reach the client.
    ChatFormat f(read_template("Qwen-Qwen3-0.6B.jinja"));
    std::string err;
    auto r = f.render(request(kUser, true), &err);
    LZ_REQUIRE(r != nullptr);
    const std::string text = "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"pa";
    ChatOutput out(r);
    const Reassembled got = reassemble(stream(out, text, 6));
    const json m = json::parse(out.message_json());
    const bool delivered = got.content.find("read_file") != std::string::npos ||
                           got.name == "read_file";
    LZ_CHECK(delivered);
    LZ_CHECK(!m.dump().empty());
}
