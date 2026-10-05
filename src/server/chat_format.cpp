#include "server/chat_format.h"

#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <utility>

#include "chat.h"
#include "llama.h"

// llama.cpp's common library changed its JSON type between the two bases this
// project builds against; the calls used here exist on both.
#if __has_include("json.h")
#include "json.h"
using ChatJson = common_json;
#else
#include <nlohmann/json.hpp>   // llama.cpp's vendored copy; chat.h only forward-declares it
using ChatJson = nlohmann::ordered_json;
#endif

namespace dray::server {

namespace {

// JSON string literal, escaped (the outgoing objects are built by hand so no
// JSON type crosses the library boundary).
std::string quote(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 2);
    o += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            case '\b': o += "\\b"; break;
            case '\f': o += "\\f"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o += static_cast<char>(c);
                }
        }
    }
    o += '"';
    return o;
}

std::string delta_json(const common_chat_msg_diff& d) {
    std::string o = "{";
    bool first = true;
    auto field = [&](const char* k, const std::string& v) {
        if (!first) o += ",";
        first = false;
        o += quote(k) + ":" + v;
    };
    if (!d.reasoning_content_delta.empty()) field("reasoning_content", quote(d.reasoning_content_delta));
    if (!d.content_delta.empty()) field("content", quote(d.content_delta));
    if (d.tool_call_index != std::string::npos) {
        std::string tc = "{\"index\":" + std::to_string(d.tool_call_index);
        if (!d.tool_call_delta.id.empty()) {
            tc += ",\"id\":" + quote(d.tool_call_delta.id) + ",\"type\":\"function\"";
        }
        if (!d.tool_call_delta.name.empty() || !d.tool_call_delta.arguments.empty()) {
            tc += ",\"function\":{";
            bool f1 = true;
            if (!d.tool_call_delta.name.empty()) {
                tc += "\"name\":" + quote(d.tool_call_delta.name);
                f1 = false;
            }
            if (!d.tool_call_delta.arguments.empty()) {
                if (!f1) tc += ",";
                tc += "\"arguments\":" + quote(d.tool_call_delta.arguments);
            }
            tc += "}";
        }
        tc += "}";
        field("tool_calls", "[" + tc + "]");
    }
    o += "}";
    return o;
}

std::string new_call_id() {
    static std::atomic<uint64_t> n{0};
    return "call_" + std::to_string(++n);
}

}  // namespace

struct ChatFormat::Rendered {
    std::string prompt;
    std::vector<std::string> stops;
    bool tools = false;
    common_chat_parser_params parser;
};

struct ChatFormat::Impl {
    common_chat_templates_ptr tmpls;
};

ChatFormat::ChatFormat(const llama_model* model) : impl_(std::make_unique<Impl>()) {
    // The model's own template (GGUF tokenizer.chat_template), no override.
    impl_->tmpls = common_chat_templates_init(model, "");
}

ChatFormat::ChatFormat(const std::string& jinja_template) : impl_(std::make_unique<Impl>()) {
    impl_->tmpls = common_chat_templates_init(nullptr, jinja_template);
}

ChatFormat::~ChatFormat() = default;

const std::string& ChatFormat::prompt(const Rendered& r) { return r.prompt; }
const std::vector<std::string>& ChatFormat::stops(const Rendered& r) { return r.stops; }
bool ChatFormat::has_tools(const Rendered& r) { return r.tools; }

std::shared_ptr<const ChatFormat::Rendered> ChatFormat::render(const std::string& body,
                                                               std::string* err) const {
    try {
        const ChatJson j = ChatJson::parse(body);
        common_chat_templates_inputs in;
        in.messages = common_chat_msgs_parse_oaicompat(j.at("messages"));
        if (j.contains("tools") && !j.at("tools").is_null()) {
            in.tools = common_chat_tools_parse_oaicompat(j.at("tools"));
        }
        if (j.contains("tool_choice") && j.at("tool_choice").is_string()) {
            in.tool_choice = common_chat_tool_choice_parse_oaicompat(j.at("tool_choice").get<std::string>());
        }
        if (j.contains("parallel_tool_calls") && j.at("parallel_tool_calls").is_boolean()) {
            in.parallel_tool_calls = j.at("parallel_tool_calls").get<bool>();
        }
        in.use_jinja = true;
        in.add_generation_prompt = true;
        // Reasoning goes to message.reasoning_content, in streaming deltas too.
        in.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        in.enable_thinking = true;
        if (j.contains("chat_template_kwargs") && j.at("chat_template_kwargs").is_object() &&
            j.at("chat_template_kwargs").contains("enable_thinking") &&
            j.at("chat_template_kwargs").at("enable_thinking").is_boolean()) {
            in.enable_thinking = j.at("chat_template_kwargs").at("enable_thinking").get<bool>();
        }
        if (j.contains("reasoning_effort") && j.at("reasoning_effort").is_string() &&
            j.at("reasoning_effort").get<std::string>() == "none") {
            in.enable_thinking = false;
        }

        const common_chat_params p = common_chat_templates_apply(impl_->tmpls.get(), in);

        auto r = std::make_shared<Rendered>();
        r->prompt = p.prompt;
        r->stops = p.additional_stops;
        r->tools = !in.tools.empty() && in.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
        r->parser = common_chat_parser_params(p);
        if (!p.parser.empty()) r->parser.parser.load(p.parser);
        return r;
    } catch (const std::exception& e) {
        *err = std::string("cannot render this request with the model's chat template: ") + e.what();
        return nullptr;
    }
}

struct ChatOutput::Impl {
    std::shared_ptr<const ChatFormat::Rendered> r;
    std::string text;
    common_chat_msg msg;
    std::vector<std::string> ids;
    bool parse_failed = false;

    std::vector<std::string> update(bool partial) {
        std::vector<std::string> out;
        if (parse_failed) return out;
        common_chat_msg next;
        try {
            next = common_chat_parse(text, partial, r->parser);
        } catch (const std::exception&) {
            if (partial) return out;   // an unfinished construct: wait for more text
            // A guard: llama.cpp's parser has not been seen to throw on finished
            // text (a truncated tool call comes back as a partial call). If it
            // ever does, the text reaches the client verbatim as content.
            parse_failed = true;
            next = common_chat_msg{};
            next.role = "assistant";
            next.content = text;
        }
        if (next.empty()) return out;
        next.set_tool_call_ids(ids, new_call_id);
        for (const common_chat_msg_diff& d : common_chat_msg_diff::compute_diffs(msg, next)) {
            out.push_back(delta_json(d));
        }
        msg = std::move(next);
        return out;
    }
};

ChatOutput::ChatOutput(std::shared_ptr<const ChatFormat::Rendered> rendered)
    : impl_(std::make_unique<Impl>()) {
    impl_->r = std::move(rendered);
    impl_->msg.role = "assistant";
}

ChatOutput::~ChatOutput() = default;

std::vector<std::string> ChatOutput::feed(const std::string& piece) {
    impl_->text += piece;
    return impl_->update(true);
}

std::vector<std::string> ChatOutput::finish() { return impl_->update(false); }

bool ChatOutput::has_tool_calls() const { return !impl_->msg.tool_calls.empty(); }

std::string ChatOutput::message_json() const {
    const common_chat_msg& m = impl_->msg;
    std::string o = "{\"role\":\"assistant\",\"content\":";
    // OpenAI: content is null when the message is only tool calls.
    o += (m.content.empty() && !m.tool_calls.empty()) ? std::string("null") : quote(m.content);
    if (!m.reasoning_content.empty()) o += ",\"reasoning_content\":" + quote(m.reasoning_content);
    if (!m.tool_calls.empty()) {
        o += ",\"tool_calls\":[";
        for (size_t i = 0; i < m.tool_calls.size(); ++i) {
            const common_chat_tool_call& t = m.tool_calls[i];
            if (i) o += ",";
            o += "{\"id\":" + quote(t.id) + ",\"type\":\"function\",\"function\":{\"name\":" +
                 quote(t.name) + ",\"arguments\":" + quote(t.arguments) + "}}";
        }
        o += "]";
    }
    o += "}";
    return o;
}

}  // namespace dray::server
