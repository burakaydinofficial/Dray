#include "cli/commands.h"

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

#include "config/env_report.h"
#include "plan/residency.h"

namespace dray::cli {

int cmd_plan(const Args& a) {
    std::string err;
    uint32_t ctx = a.n_ctx ? a.n_ctx : 32768;
    // plan --batch N: the admission maths at width N, instantly, from the
    // tensor table alone -- the same floor and refusal the engine would
    // compute at load, without touching the drive for the weights.
    const uint32_t plan_seq = a.n_batch_seq > 1 ? a.n_batch_seq : 1;
    // --kv must reach the planner too: the floor it prints is the floor the
    // engine will reserve (this was silently dropped when --kv landed).
    const double plan_kv_el = a.kv == "q4" ? 0.5625 : (a.kv == "q8" ? 1.0625 : 2.0);
    dray::plan::Plan p = dray::plan::build_plan(a.model, a.cap, ctx, &err, plan_seq, plan_kv_el);
    if (!err.empty()) { std::fprintf(stderr, "plan failed: %s\n", err.c_str()); return 1; }
    for (const auto& w : p.warnings) std::fprintf(stderr, "%s\n", w.c_str());
    std::printf("%s\n", p.report().c_str());
    // T28: the README says plan output names the pulled levers; make it true.
    {
        const std::string envs_plan = dray::config::env_report();
        if (!envs_plan.empty()) std::printf("%s\n", envs_plan.c_str());
    }

// Batching PROJECTION (Invariant 8: projections are labeled, never quoted
    // as results; and GRADED -- the K3 measurement confirmed the sharing at
    // B=8 within 20% and FALSIFIED the old feasibility column, which counted
    // only KV state: the per-step expert-UNION working set must also fit the
    // cache or the step thrashes within itself, measured at B=32/28G reading
    // 1.8x its own cold bound). Pure arithmetic from this file's own numbers. The mechanism:
    // the unconditional stream is read ONCE per decode step regardless of how
    // many sequences share it, while routed-expert overlap between sequences
    // is the birthday-ish 1-(1-k/E)^B, near-nil at frontier sparsity. So per-
    // sequence bytes fall as uncond/B + routed*u(B)/B, and at large B every
    // step converges to one full-model scan shared by all sequences: decode
    // becomes prefill. Cold bound (h_routed = 0, uncond fully streaming);
    // caching only improves it. Feasibility: each extra sequence carries its
    // own KV + recurrent state inside the cap.
    if (p.n_moe_layers > 0 && p.n_experts > 0 && p.n_expert_used > 0) {
        const double E = static_cast<double>(p.n_experts);
        const double k = static_cast<double>(p.n_expert_used);
        const double uncond = static_cast<double>(p.unconditional_bytes);
        const double routed = static_cast<double>(p.routed_bytes);
        const uint64_t seq_state = p.floor.kv_cache + p.floor.recurrent_state;
        const uint64_t f_total = p.floor.total();
        const uint64_t spare = a.cap > f_total ? a.cap - f_total : 0;
        const uint64_t b_max = seq_state ? 1 + spare / seq_state : 1;
        const double per1 = uncond + routed * (1.0 - std::pow(1.0 - k / E, 1.0));
        std::printf("\nbatched-decode projection (cold bound; PROJECTION, not a measurement):\n");
        std::printf("  %-6s %14s %14s %10s %s\n",
                    "batch", "bytes/tok/seq", "step bytes", "aggregate", "fits this cap?");
        for (uint64_t B : { 1ull, 2ull, 4ull, 8ull, 16ull, 32ull, 64ull }) {
            const double u = 1.0 - std::pow(1.0 - k / E, static_cast<double>(B));
            const double step = uncond + routed * u;
            const double per_seq = step / static_cast<double>(B);
            // J2: the previous "union working set" verdict compared one layer's
            // union against the WHOLE cache budget (n_layers too permissive)
            // and spent the same spare bytes on state and cache at once -- it
            // could never fire on the measured B=32 thrash it was added for.
            // No validated intra-step thrash model exists yet, so the honest
            // column is DISCLOSURE: the per-B cache remainder (state funded
            // first, per row) and the union/cache pressure ratio, with the
            // measured anchor in the footnote. "measure" is the verdict where
            // pressure is material; fake precision was worse than none.
            const uint64_t extra = (B - 1) * seq_state;
            const uint64_t cache_b = a.cap > f_total + extra ? a.cap - f_total - extra : 0;
            const double union_all = routed * u;
            const bool state_ok = B <= b_max;
            const double pressure = cache_b ? union_all / static_cast<double>(cache_b) : 1e9;
            std::printf("  %-6llu %11.2f GB %11.2f GB %9.2fx %s\n",
                        static_cast<unsigned long long>(B), per_seq / 1e9, step / 1e9,
                        per1 / per_seq,
                        !state_ok ? "no (per-seq state)"
                                  : (pressure > 4.0 ? "measure (union >> cache)" : "yes"));
        }
        std::printf("  per-seq state %.2f GB (KV + recurrent); max batch at this cap ~%llu\n",
                    seq_state / 1e9, static_cast<unsigned long long>(b_max));
        std::printf("  at large batch every step is one full-model scan shared by all\n"
                    "  sequences: decode converges to prefill and the SEQUENTIAL ceiling\n"
                    "  (calibrate's stream class) becomes the relevant one.\n");
        // plan --batch N: preview the engine's own admission verdict at this
        // width -- the measured bounds (below 1x the widest union region is
        // certain death, 2x proven clean, between is the untested band).
        if (plan_seq > 1) {
            const double uw = E > 0.0
                ? E * (1.0 - std::pow(1.0 - k / E, static_cast<double>(plan_seq))) : k;
            uint64_t max_slot = 0;
            for (const auto& sc : p.slot_classes) max_slot = std::max(max_slot, sc.slot_bytes);
            const uint64_t region = static_cast<uint64_t>(std::ceil(uw)) * max_slot;
            const uint64_t remainder = a.cap > f_total ? a.cap - f_total : 0;
            const char* verdict = region > remainder
                ? "REFUSED at load (one union region exceeds the cache remainder)"
                : (region * 2 > remainder
                       ? "admits with WARNING (1x-2x the union region: untested band)"
                       : "admits");
            std::printf("  batch %u admission preview: floor %.2f GB, cache remainder %.2f GB,\n"
                        "  widest union region %.2f GB -> %s\n",
                        plan_seq, f_total / 1e9, remainder / 1e9, region / 1e9, verdict);
        }
    }
    if (!p.feasible) {
        std::printf("\nREFUSED: %s\n", p.refusal.c_str());
        return 2;
    }
    return 0;
}


}  // namespace dray::cli
