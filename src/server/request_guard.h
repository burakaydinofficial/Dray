// Request admission policy, applied before routing to every path but /health.
//
// The server binds loopback only; these gates keep it that way against the
// ways a browser or a sloppy client could still reach it:
//
//   * Origin present            -> 403 (no CLI or SDK sends one; every browser does)
//   * Host not a loopback name  -> 403 (DNS rebinding; EQUALITY after stripping
//                                  an optional :port -- a prefix test once let
//                                  localhost.attacker.com through)
//   * chunked transfer          -> 411 (httplib's payload cap does not bound it)
//   * body method without
//     Content-Length            -> 411 (the unbounded read-until-close path)
//   * API key set and missing
//     or wrong                  -> 401 (constant-time comparison)
//
// /health is the ONLY exempt path (monitors). /admin/* passes the gates too:
// the old /v1-prefix test left /admin/shutdown as the one unauthenticated state
// change after the operator paid for auth (F11).
#pragma once

#include <string>

#include "httplib.h"

namespace dray::server {

class RequestGuard {
public:
    // Empty api_key = open on localhost.
    explicit RequestGuard(const std::string& api_key);

    // Handled = a refusal was written to res; Unhandled = route normally.
    httplib::Server::HandlerResponse check(const httplib::Request& req,
                                           httplib::Response& res) const;

private:
    std::string want_;   // "Bearer <key>", or empty
};

}  // namespace dray::server
