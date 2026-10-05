// The subcommands. Each returns the process exit code: 0 success, 1 failure,
// 2 an admission refusal (the cap cannot serve what was asked, and the message
// says what would).
#pragma once

#include "cli/args.h"

namespace dray::cli {

int cmd_plan(const Args& a);           // residency plan + RAM curve, no load
int cmd_config(const Args& a);        // effective settings and their sources
int cmd_calibrate(const Args& a);      // drive bandwidth per traffic class
int cmd_verify(const Args& a);         // uncached reads vs an independent reference
int cmd_run(const Args& a);            // one generation with the live readout
int cmd_run_reference(const Args& a);  // --no-stream: llama.cpp loads normally (difftest reference)
int cmd_batch(const Args& a);          // lockstep batched decode, optional cohort rotation
int cmd_snaptest(const Args& a);       // snapshot save/restore round trip on a live context
int cmd_serve(const Args& a);          // the OpenAI-compatible server
int cmd_repack(const Args& a);         // install-time expert-major companion

}  // namespace dray::cli
