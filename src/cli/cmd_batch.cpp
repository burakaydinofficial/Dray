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

#include "engine/batch_generator.h"
#include "engine/cohort_rotator.h"
#include "engine/engine.h"
#include "engine/llama_stepper.h"
#include "engine/scheduler.h"

namespace dray::cli {

// Batched offline decode: N prompts, lockstep steps, one shared stream -- the
// throughput operating point the projection table models, and the run that
// grades that table. The server gets the same economics from engine::Scheduler
// (serve --parallel); DRAY_BATCH_SCHEDULER=1 runs this command through it.
int cmd_batch(const Args& a) {
    namespace srv = dray::engine;
    if (a.n_batch_seq < 1) { std::fprintf(stderr, "batch: --batch must be >= 1\n"); return 1; }

    std::vector<std::string> prompts;
    if (!a.prompts_file.empty()) {
        std::ifstream pf(a.prompts_file);
        if (!pf.good()) { std::fprintf(stderr, "batch: cannot read %s\n", a.prompts_file.c_str()); return 1; }
        std::string line;
        while (std::getline(pf, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) prompts.push_back(line);
        }
        if (prompts.empty()) { std::fprintf(stderr, "batch: %s has no prompts\n", a.prompts_file.c_str()); return 1; }
        if (prompts.size() > a.n_batch_seq && a.rotate_span <= 0) {
            std::fprintf(stderr, "batch: %zu prompts but --batch %u; refusing to truncate"
                                 " (or add --rotate SPAN --state-dir DIR to serve them in cohorts)\n",
                         prompts.size(), a.n_batch_seq);
            return 1;
        }
    } else {
        prompts.assign(a.n_batch_seq, a.prompt);
    }
    if (a.rotate_span > 0 && a.state_dir.empty()) {
        std::fprintf(stderr, "batch: --rotate writes parked state; --state-dir is a required,"
                             " explicit choice (a different device than the model is wise)\n");
        return 1;
    }

    srv::EngineConfig ec = engine_config_from(a);
    ec.plan_to_stdout = true;
    // Rotation: the context funds the COHORT width (--batch); prompts beyond
    // it rotate through. Unrotated: every prompt is resident, as before.
    ec.n_seq = a.rotate_span > 0 ? a.n_batch_seq
                                 : static_cast<uint32_t>(prompts.size());
    std::string err;
    std::unique_ptr<srv::Engine> eng = srv::Engine::open(ec, &err);
    if (!eng) { std::fprintf(stderr, "batch: %s\n", err.c_str()); return 1; }

    if (a.rotate_span > 0) {
        // Cohort rotation: N prompts through a --batch-wide window, state
        // parked to --state-dir between residencies. Writes are reported
        // with the same rigor as reads -- the honesty contract covers them.
        srv::RotateParams rp;
        rp.prompts = prompts;
        rp.max_tokens = a.n_predict;
        rp.temperature = a.greedy ? 0.0f : 0.8f;
        rp.seed = a.seed;
        rp.stop = a.stop_seqs;
        rp.span = a.rotate_span;
        rp.state_dir = a.state_dir;
        rp.on_round = [](int32_t round, int32_t cohort, int32_t done) {
            std::fprintf(stderr, "rotate: round %d, cohort %d, %d sequences done\n",
                         round, cohort, done);
        };
        const auto ec_open = eng->counters();
        const auto t0 = std::chrono::steady_clock::now();
        srv::RotateResult rres = srv::CohortRotator(*eng).run(rp);
        const auto ec1 = eng->counters();
        const double secs =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (!rres.error.empty()) {
            std::fprintf(stderr, "rotate: %s\n", rres.error.c_str());
            eng.reset();
            return 1;
        }
        bool any_bad = false;
        uint64_t total_out = 0;
        for (size_t i = 0; i < rres.seqs.size(); ++i) {
            const auto& r = rres.seqs[i];
            std::printf("--- seq %zu (%d tokens%s%s%s) ---\n%s\n", i, r.tokens_out,
                        r.truncated_by_eog ? ", eog/stop" : "",
                        r.ctx_wall ? ", ctx-wall" : "",
                        r.cancelled ? ", cancelled" : "",
                        r.text.c_str());
            if (!r.error.empty()) std::printf("    ERROR: %s\n", r.error.c_str());
            total_out += static_cast<uint64_t>(r.tokens_out);
            any_bad = any_bad || r.aborted || !r.error.empty();
        }
        const uint64_t bytes_r = ec1.bytes_streamed - ec_open.bytes_streamed;
        const uint64_t fails = ec1.failures - ec_open.failures;
        std::printf("\nrotate summary: %zu seqs in cohorts of %u, %llu residencies, %llu tokens\n",
                    rres.seqs.size(), a.n_batch_seq,
                    static_cast<unsigned long long>(rres.rounds),
                    static_cast<unsigned long long>(total_out));
        std::printf("  model bytes read    %.3f GB\n", bytes_r / 1e9);
        std::printf("  state bytes WRITTEN %.3f GB, read back %.3f GB (%.1f MB/token amortized)\n",
                    rres.state_bytes_written / 1e9, rres.state_bytes_read / 1e9,
                    total_out ? rres.state_bytes_written / 1e6 / static_cast<double>(total_out) : 0.0);
        std::printf("  wall                %.1f s  (%.2f tok/s aggregate; wall is noisy, bytes decide)\n",
                    secs, secs > 0 ? total_out / secs : 0.0);
        if (fails) std::printf("  *** %llu materialise FAILURES -- OUTPUT NOT TRUSTWORTHY ***\n",
                               static_cast<unsigned long long>(fails));
        const bool over = eng->live_over_cap();
        if (over) std::printf("  *** CAP BREACH occurred during this run ***\n");
        std::fprintf(stderr, "%s\n", eng->streamer_report().c_str());
        std::printf("\n%s\n", eng->accountant_report().c_str());
        eng.reset();
        return (!any_bad && fails == 0 && !over) ? 0 : 1;
    }

    srv::BatchParams bp;
    bp.prompts = prompts;
    bp.max_tokens = a.n_predict;
    bp.temperature = a.greedy ? 0.0f : 0.8f;
    bp.seed = a.seed;
    bp.stop = a.stop_seqs;   // battery-caught: the engine implemented stop, the CLI never plumbed it

    const auto ec_open = eng->counters();
    const auto t0 = std::chrono::steady_clock::now();
    bp.on_step = [&](int32_t step, int32_t live) {
        if (step % 8 == 0) {
            const auto ecs = eng->counters();
            std::fprintf(stderr, "batch: step %d, %d live, %.2f GB read\n", step, live,
                         (ecs.bytes_streamed - ec_open.bytes_streamed) / 1e9);
        }
    };
    // DRAY_BATCH_SCHEDULER=1: the same prompts through the continuous-
    // batching scheduler (all submitted at once, driven synchronously). With
    // prompts that fit one step it builds the same steps as BatchGenerator,
    // so the output must be identical -- that is the check this lever exists for.
    const bool via_scheduler = [] {
        const char* v = std::getenv("DRAY_BATCH_SCHEDULER");
        return v && v[0] == '1';
    }();
    srv::BatchResult br;
    if (via_scheduler) {
        srv::LlamaStepper stepper(*eng);
        srv::Scheduler::Config scfg;
        scfg.prefill_tokens_per_step = eng->prefill_chunk();
        srv::Scheduler sched(stepper, scfg);
        br.seqs.resize(bp.prompts.size());
        for (size_t s = 0; s < bp.prompts.size(); ++s) {
            srv::Request q;
            q.params.prompt = bp.prompts[s];
            q.params.max_tokens = bp.max_tokens;
            q.params.temperature = bp.temperature;
            q.params.seed = (bp.seed ? bp.seed : 1234u) + static_cast<uint32_t>(s);   // as BatchGenerator
            q.params.stop = bp.stop;
            q.done = [&br, s](srv::GenResult r) { br.seqs[s] = std::move(r); };
            sched.submit(std::move(q));
        }
        br.prefill_end_bytes = ec_open.bytes_streamed;
        bool prefill_recorded = false;
        while (sched.step()) {
            const auto st = sched.stats();
            if (!prefill_recorded && st.prefilling == 0 && st.waiting == 0) {
                br.prefill_end_bytes = eng->counters().bytes_streamed;
                prefill_recorded = true;
            }
            if (bp.on_step) bp.on_step(static_cast<int32_t>(st.steps), st.active);
        }
        br.steps = static_cast<uint64_t>(sched.stats().steps);
    } else {
        br = srv::BatchGenerator(*eng).run(bp);
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const auto ec1 = eng->counters();

    if (!br.error.empty()) {
        std::fprintf(stderr, "batch: %s\n", br.error.c_str());
        eng.reset();
        return 1;
    }

    uint64_t total_out = 0;
    bool any_bad = false;
    for (size_t s = 0; s < br.seqs.size(); ++s) {
        const auto& r = br.seqs[s];
        std::printf("--- seq %zu (%d tokens%s%s%s) ---\n%s\n", s, r.tokens_out,
                    r.aborted ? ", ABORTED" : "",
                    r.cancelled ? ", cancelled" : "",
                    r.error.empty() ? "" : ", ERROR",
                    r.text.c_str());
        if (!r.error.empty()) std::printf("    error: %s\n", r.error.c_str());
        total_out += static_cast<uint64_t>(r.tokens_out);
        any_bad = any_bad || r.aborted || !r.error.empty();
    }
    // J7: prefill and decode reported SEPARATELY -- the first benchmark
    // divided prefill-inclusive bytes by decode-only steps, inflating every
    // per-step figure by a term that grows with B.
    const uint64_t total_bytes = ec1.bytes_streamed - ec_open.bytes_streamed;
    const uint64_t prefill_bytes = br.prefill_end_bytes > ec_open.bytes_streamed
                                       ? br.prefill_end_bytes - ec_open.bytes_streamed : 0;
    const uint64_t decode_bytes = total_bytes > prefill_bytes ? total_bytes - prefill_bytes : 0;
    const uint64_t fails = ec1.failures - ec_open.failures;
    std::printf("\nbatch summary: %zu seqs, %llu steps, %llu tokens total\n",
                br.seqs.size(), static_cast<unsigned long long>(br.steps),
                static_cast<unsigned long long>(total_out));
    std::printf("  prefill bytes  %.3f GB (all %zu sequences)\n",
                prefill_bytes / 1e9, br.seqs.size());
    std::printf("  decode bytes   %.3f GB  (%.3f GB/step, %.3f GB/token-aggregate)\n",
                decode_bytes / 1e9,
                br.steps ? decode_bytes / 1e9 / static_cast<double>(br.steps) : 0.0,
                total_out ? decode_bytes / 1e9 / static_cast<double>(total_out) : 0.0);
    std::printf("  total bytes    %.3f GB\n", total_bytes / 1e9);
    std::printf("  wall           %.1f s  (%.2f tok/s aggregate; wall is noisy, bytes decide)\n",
                secs, secs > 0 ? total_out / secs : 0.0);
    if (fails) std::printf("  *** %llu materialise FAILURES -- OUTPUT NOT TRUSTWORTHY ***\n",
                           static_cast<unsigned long long>(fails));
    // J4: the LATCHED breach signal, not a point sample of a self-clearing
    // flag -- and the loud line, so text greps see it like cmd_run's.
    const bool over = eng->live_over_cap();
    if (over) std::printf("  *** CAP BREACH occurred during this run ***\n");
    // As run does: the cache, the slot pool, the reads -- on stderr, beside the
    // live progress, so the recorded stdout stays what golden compares.
    std::fprintf(stderr, "%s\n", eng->streamer_report().c_str());
    std::printf("\n%s\n", eng->accountant_report().c_str());
    eng.reset();
    return (!any_bad && fails == 0 && !over) ? 0 : 1;
}


}  // namespace dray::cli
