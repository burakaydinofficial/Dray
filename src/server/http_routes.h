// The OpenAI-compatible surface, installed onto an httplib server. No
// interface, no dashboard, ever (owner, 2026-08-14):
//
//   POST   /v1/chat/completions      stream:true -> SSE; background:true -> job
//   GET    /v1/models
//   GET    /v1/responses/{id}        poll a background job
//   POST   /v1/responses/{id}/cancel
//   DELETE /v1/responses/{id}
//   POST   /admin/shutdown           graceful stop (Content-Type gated)
//   GET    /health                   includes the streamer's honest report line
#pragma once

#include "httplib.h"

#include "server/server_context.h"

namespace dray::server {

void install_chat_completions(httplib::Server& srv, const ServerContext& ctx);
void install_status_routes(httplib::Server& srv, const ServerContext& ctx);   // /health, /v1/models
void install_job_routes(httplib::Server& srv, const ServerContext& ctx);      // /v1/responses/*
void install_admin_routes(httplib::Server& srv);                              // /admin/shutdown

}  // namespace dray::server
