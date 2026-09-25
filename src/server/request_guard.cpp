#include "server/request_guard.h"

#include <cctype>

namespace dray::server {

namespace {

httplib::Server::HandlerResponse refuse(httplib::Response& res, int status, const char* body) {
    res.status = status;
    res.set_content(body, "application/json");
    return httplib::Server::HandlerResponse::Handled;
}

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// Constant-time comparison: operator== short-circuits at the first differing
// byte, which leaks match length over loopback timing. The length check leaks
// only the length, which the "Bearer " prefix already makes guessable.
bool ct_equal(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    unsigned char acc = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        acc = static_cast<unsigned char>(
            acc | (static_cast<unsigned char>(a[i]) ^ static_cast<unsigned char>(b[i])));
    }
    return acc == 0;
}

}  // namespace

RequestGuard::RequestGuard(const std::string& api_key)
    : want_(api_key.empty() ? std::string() : "Bearer " + api_key) {}

httplib::Server::HandlerResponse RequestGuard::check(const httplib::Request& req,
                                                     httplib::Response& res) const {
    if (req.path == "/health") return httplib::Server::HandlerResponse::Unhandled;

    if (!req.get_header_value("Origin").empty()) {
        return refuse(res, 403, "{\"error\":{\"message\":\"browser-origin requests are not accepted\"}}");
    }
    std::string host = lower(req.get_header_value("Host"));   // H17
    {
        const size_t rb = host.rfind(']');   // [::1]:port keeps its brackets
        const size_t cp = host.rfind(':');
        if (cp != std::string::npos && (rb == std::string::npos || cp > rb)) host.erase(cp);
    }
    if (host != "127.0.0.1" && host != "localhost" && host != "::1" && host != "[::1]") {
        return refuse(res, 403, "{\"error\":{\"message\":\"host not allowed\"}}");
    }
    // Pass-4b: httplib decides chunked-ness case-insensitively, so this must too.
    if (lower(req.get_header_value("Transfer-Encoding")).find("chunked") != std::string::npos) {
        return refuse(res, 411, "{\"error\":{\"message\":\"chunked transfer not supported; send Content-Length\"}}");
    }
    // Pass-4c: set_payload_max_length is consulted only on the Content-Length
    // branch. Only methods that carry a body: GET/DELETE legitimately have none.
    if ((req.method == "POST" || req.method == "PUT" || req.method == "PATCH") &&
        !req.has_header("Content-Length")) {
        return refuse(res, 411, "{\"error\":{\"message\":\"Content-Length required\"}}");
    }
    if (want_.empty() || ct_equal(req.get_header_value("Authorization"), want_)) {
        return httplib::Server::HandlerResponse::Unhandled;
    }
    res.set_header("WWW-Authenticate", "Bearer");
    return refuse(res, 401, "{\"error\":{\"message\":\"invalid or missing API key\"}}");
}

}  // namespace dray::server
