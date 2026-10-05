// OpenAI chat requests <-> the model's own chat format, through llama.cpp's chat
// library (the one llama-server uses): the model's Jinja template renders the
// messages AND the tool definitions, and the generated text is parsed back into
// content, reasoning_content and tool_calls, incrementally for streaming.
//
// Only strings cross this header: llama.cpp's common library carries its own
// JSON type (and the two bases disagree on which), so it stays inside
// chat_format.cpp.
#pragma once

#include <memory>
#include <string>
#include <vector>

struct llama_model;

namespace dray::server {

class ChatFormat {
public:
    explicit ChatFormat(const llama_model* model);
    // An explicit Jinja template instead of the model's own (tests; overrides).
    explicit ChatFormat(const std::string& jinja_template);
    ~ChatFormat();
    ChatFormat(const ChatFormat&) = delete;
    ChatFormat& operator=(const ChatFormat&) = delete;

    struct Rendered;   // the prompt and how to parse what comes back (opaque)
    // Renders a request body (the raw JSON text). nullptr and *err set: the
    // request cannot be rendered (a 400 for the client).
    std::shared_ptr<const Rendered> render(const std::string& body, std::string* err) const;

    static const std::string& prompt(const Rendered& r);
    // Stop strings the template adds (end-of-turn markers some formats need).
    static const std::vector<std::string>& stops(const Rendered& r);
    static bool has_tools(const Rendered& r);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// One response's output, parsed as it grows.
class ChatOutput {
public:
    explicit ChatOutput(std::shared_ptr<const ChatFormat::Rendered> rendered);
    ~ChatOutput();
    ChatOutput(const ChatOutput&) = delete;
    ChatOutput& operator=(const ChatOutput&) = delete;

    // Appends generated text; returns the OpenAI `delta` objects (JSON text)
    // it completes -- possibly none while a tool call is still being written.
    std::vector<std::string> feed(const std::string& piece);
    // The generation ended: the last deltas.
    std::vector<std::string> finish();
    // The whole assistant message as an OpenAI `message` object (JSON text).
    std::string message_json() const;
    bool has_tool_calls() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace dray::server
