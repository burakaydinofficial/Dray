#include "cli/commands.h"
#include "cli/engine_config.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "calib/calibrate.h"
#include "config/env_report.h"
#include "engine/engine.h"
#include "mem/accountant.h"
#include "report/readout.h"

namespace dray::cli {


// THE STREAMING RUN, consolidated onto the server engine: one setup sequence,
// one generate loop, two consumers (this and serve). The readout drives through
// the observer hooks; its output format is byte-compatible with the pre-
// consolidation path. (T28: difftest reads only the generated text after the
// last readout line; no fuller parser exists in scripts/ today.)
// One display-only difference, deliberate: the Decode phase mark shifts by one
// token's sampling (~ms), because the engine has no hook between the prefill
// decode and the first sample.
int cmd_run(const Args& a) {
    if (a.no_stream) return cmd_run_reference(a);

    dray::engine::EngineConfig ec = engine_config_from(a);
    ec.plan_to_stdout = true;   // the plan is FOR the user here, not the operator

    std::string err;
    std::unique_ptr<dray::engine::Engine> eng = dray::engine::Engine::open(ec, &err);
    if (!eng) {
        if (err.rfind("REFUSED", 0) == 0) { std::printf("\n%s\n", err.c_str()); return 2; }
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }

    namespace rep = dray::report;
    rep::Readout::Options ropt;
    ropt.model = a.model;
    ropt.jsonl_path = a.status_path;
    rep::Readout ro(ropt);
    ro.set_context("llama.cpp direct-io load",
                   eng->plan_report() + "\n" +
                       dray::config::env_report());

    // T5: the README's limiter promise, finally wired. A persisted calibration
    // sets the per-class ceilings, quoted WITH its age on every readout line;
    // no calibration means the verdict stays honestly unknown, and says how to
    // get one. Never a default, never fresh-truth theatre.
    {
        dray::calib::Calibration cal;
        std::string when, lerr;
        if (dray::calib::load_calibration(&cal, &when, a.model + ".lzcal.json", &lerr)) {
            rep::Ceilings ceil;
            ceil.set(rep::ByteClass::Stream, cal.stream_bw);
            ceil.set(rep::ByteClass::Gather, cal.gather_bw);
            ceil.provenance = "calibrated " + when;
            ro.set_ceilings(ceil);
            std::fprintf(stderr,
                         "ceilings: %s; thermal state may differ, re-run calibrate to refresh\n",
                         ceil.provenance.c_str());
        } else {
            // A calibration measures the DRIVE, not the model, but it is stored
            // beside the model file -- so sweeping one model taught the engine
            // nothing about the next one on the same disk, and most runs went
            // out with "ceiling unknown". That is how a 2.75 GB/s realised rate
            // sat unremarked next to a 6.8 GB/s drive for days (2026-08-24).
            //
            // So fall back to any calibration on the SAME VOLUME, and name the
            // file it came from. Borrowed, disclosed, never silent.
            std::string borrowed, bwhen;
            try {
                const std::filesystem::path mp(a.model);
                const auto root = mp.root_name();
                // Walk up, and at each level look one directory deeper too:
                // models are laid out as <root>/<repo>/<quant>/file.gguf, so a
                // calibration for a different model is a SIBLING subtree away,
                // never in a shared parent's own file list.
                auto scan = [&](const std::filesystem::path& dir) {
                    std::error_code ec;
                    for (const auto& e : std::filesystem::directory_iterator(
                             dir, std::filesystem::directory_options::skip_permission_denied, ec)) {
                        if (ec) return;
                        if (!e.is_regular_file(ec)) continue;
                        const std::string p = e.path().string();
                        if (p.size() > 11 && p.compare(p.size() - 11, 11, ".lzcal.json") == 0) {
                            dray::calib::Calibration c2;
                            std::string e2;
                            if (dray::calib::load_calibration(&c2, &bwhen, p, &e2)) {
                                cal = c2;
                                borrowed = e.path().filename().string();
                                return;
                            }
                        }
                    }
                };
                for (std::filesystem::path dir = mp.parent_path();
                     !dir.empty() && dir.root_name() == root; dir = dir.parent_path()) {
                    scan(dir);
                    if (!borrowed.empty()) break;
                    std::error_code ec;
                    for (const auto& e : std::filesystem::directory_iterator(
                             dir, std::filesystem::directory_options::skip_permission_denied, ec)) {
                        if (ec) break;
                        if (e.is_directory(ec)) { scan(e.path()); if (!borrowed.empty()) break; }
                    }
                    if (!borrowed.empty()) break;
                    if (dir == dir.parent_path()) break;
                }
            } catch (const std::exception&) { /* no calibration, say so below */ }

            if (!borrowed.empty()) {
                rep::Ceilings ceil;
                ceil.set(rep::ByteClass::Stream, cal.stream_bw);
                ceil.set(rep::ByteClass::Gather, cal.gather_bw);
                ceil.provenance = "calibrated " + bwhen + " for a DIFFERENT file on this volume (" +
                                  borrowed + ")";
                ro.set_ceilings(ceil);
                std::fprintf(stderr,
                             "ceilings: %s. The drive is the same; the model is not. Re-run "
                             "calibrate on THIS model to remove the borrow.\n",
                             ceil.provenance.c_str());
            } else {
                std::fprintf(stderr,
                             "no drive calibration; limiter verdicts unknown. run: dray calibrate -m <model>\n");
            }
        }
    }

    rep::Counters c;
    c.tokens_requested = a.n_predict;
    c.resident_cap = static_cast<int64_t>(a.cap);
    c.context_cap = static_cast<int64_t>(a.n_ctx ? a.n_ctx : 32768);

    int32_t n_prompt = 0;
    dray::engine::GenParams gp;
    gp.prompt = a.prompt;
    gp.max_tokens = a.n_predict;
    // --stop was parsed, plumbed into batch and rotate, and dropped HERE -- the
    // one path a user trying the flag reaches first. A silently ignored flag is
    // indistinguishable from a honoured one (2026-08-24 audit).
    gp.stop = a.stop_seqs;
    gp.temperature = a.greedy ? 0.0f : 0.8f;
    gp.seed = a.seed;
    gp.on_prefill = [&](int32_t n) {
        n_prompt = n;
        std::fprintf(stderr, "prefill: %d tokens\n", n);
        c.phase = rep::Phase::Prefill;
        ro.update(c, rep::monotonic_ns());
    };

    // Swarm R6/R7/S13/S37: the streaming path now feeds the readout the same
    // truths the reference path always had -- byte classes from the streamer's
    // own counters, ledger residency, and the cap-honesty verdict, per token.
    uint64_t prev_stream = 0, prev_gather = 0;
    bool over_cap_seen = false;
    dray::engine::GenResult r = eng->generate(
        gp, [&](const std::string&) {
            if (c.phase != rep::Phase::Decode) {
                c.phase = rep::Phase::Decode;
            }
            ++c.tokens_done;
            c.context_len = static_cast<int64_t>(n_prompt) + c.tokens_done;
            const size_t rss = dray::mem::Accountant::process_rss();
            if (rss) c.rss_bytes = static_cast<int64_t>(rss);
            const auto ec2 = eng->counters();
            const uint64_t g = ec2.bytes_gather;
            const uint64_t s = ec2.bytes_streamed - ec2.bytes_gather;
            c.add_bytes(rep::ByteClass::Gather, static_cast<int64_t>(g - prev_gather));
            c.add_bytes(rep::ByteClass::Stream, static_cast<int64_t>(s - prev_stream));
            prev_gather = g;
            prev_stream = s;
            c.resident_bytes = static_cast<int64_t>(ec2.resident_bytes);
            // R7 residual: the two hit rates, from the streamer's own ledgers.
            c.routed_bytes_demanded = static_cast<int64_t>(ec2.routed_needed);
            c.routed_bytes_hit = static_cast<int64_t>(
                ec2.routed_needed - std::min(ec2.bytes_gather, ec2.routed_needed));
            const uint64_t active_needed = ec2.routed_needed + ec2.uncond_needed;
            const uint64_t active_read = ec2.bytes_gather + ec2.uncond_read;
            c.active_bytes_demanded = static_cast<int64_t>(active_needed);
            c.active_bytes_hit = static_cast<int64_t>(
                active_needed - std::min(active_read, active_needed));
            if (ec2.over_cap && !over_cap_seen) {
                over_cap_seen = true;
                std::fprintf(stderr, "\nCAP BREACH: resident bytes exceed the configured cap; see final accountant report\n");
            }
            ro.update(c, rep::monotonic_ns());
            ro.render_terminal(std::cerr);
            ro.write_jsonl();
        });
    ro.end_terminal(std::cerr);

    if (!r.error.empty()) {
        std::fprintf(stderr, "%s\n", r.error.c_str());
    }
    std::printf("\n%s\n", r.text.c_str());
    std::fprintf(stderr, "\n%s\n", ro.final_summary().c_str());
    std::fprintf(stderr, "%s\n", eng->streamer_report().c_str());
    // R6: the per-category ledger, previously printed only by the reference
    // path, closes every streaming run too.
    std::fprintf(stderr, "%s\n", eng->accountant_report().c_str());
    if (over_cap_seen) {
        std::fprintf(stderr, "CAP BREACH occurred during this run; figures above are the ledger's.\n");
    }
    const uint64_t failures = eng->failures();
    if (failures > 0) {
        std::fprintf(stderr,
                     "\nOUTPUT NOT TRUSTWORTHY: %llu weights failed to materialise.\n",
                     static_cast<unsigned long long>(failures));
    }
    eng.reset();
    // T19: an untrustworthy run must not score as clean on the primary gate.
    return (r.error.empty() && failures == 0 && !over_cap_seen && !r.aborted) ? 0 : 1;
}


}  // namespace dray::cli
