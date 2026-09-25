#include "server/chat_prompt.h"

#include <vector>

#include "llama.h"

namespace dray::server {

using nlohmann::json;

std::string build_prompt(const llama_model* model, const json& messages,
                         bool* used_template) {
    *used_template = false;
    std::vector<llama_chat_message> msgs;
    std::vector<std::string> keep;  // llama_chat_message holds char*: keep alive
    keep.reserve(messages.size() * 2);
    for (const auto& m : messages) {
        keep.push_back(m.is_object() ? m.value("role", "user") : "user");
        // Swarm S8: content may be a string, null, or the spec's content-part
        // array; value() throws on a present null and on arrays.
        std::string content;
        const auto ci = m.is_object() ? m.find("content") : m.end();
        if (ci != m.end()) {
            if (ci->is_string()) content = ci->get<std::string>();
            else if (ci->is_array()) {
                for (const auto& part : *ci) {
                    if (part.is_object() && part.value("type", "") == "text")
                        content += part.value("text", "");
                }
            }
        }
        keep.push_back(content);
        msgs.push_back({ keep[keep.size() - 2].c_str(), keep.back().c_str() });
    }

    const char* tmpl = llama_model_chat_template(model, nullptr);
    if (tmpl) {
        std::vector<char> buf(1 << 16);
        int32_t r = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(),
                                              true, buf.data(),
                                              static_cast<int32_t>(buf.size()));
        if (r > static_cast<int32_t>(buf.size())) {
            buf.resize(static_cast<size_t>(r));
            r = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), true,
                                          buf.data(), static_cast<int32_t>(buf.size()));
        }
        if (r > 0) {
            *used_template = true;
            return std::string(buf.data(), static_cast<size_t>(r));
        }
    }
    // Fallback template, built from the already-sanitized pairs above -- the
    // raw json re-read this replaced was the same S8 throw in other clothes.
    std::string p;
    for (size_t i = 0; i + 1 < keep.size(); i += 2) {
        p += keep[i];
        p += ": ";
        p += keep[i + 1];
        p += "\n";
    }
    p += "assistant:";
    return p;
}

}  // namespace dray::server
