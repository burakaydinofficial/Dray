#include "engine/cohort_rotator.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <system_error>
#include <string>
#include <vector>

#include "llama.h"

#include "engine/engine.h"
#include "engine/streamed_text.h"

namespace dray::engine {

RotateResult CohortRotator::run(const RotateParams& p) {
    RotateResult rr;
    const int32_t N = static_cast<int32_t>(p.prompts.size());
    const int32_t W = static_cast<int32_t>(engine_.funded_sequences());
    if (N < 1) { rr.error = "rotate: no prompts"; return rr; }
    if (W < 1) { rr.error = "rotate: plan funded no sequences"; return rr; }
    if (p.span < 1) { rr.error = "rotate: span must be >= 1"; return rr; }
    // MEASURED (2026-08-19, testbed bisection): cohort slot recycling is
    // deterministic; the llama_state_seq park/restore roundtrip is NOT --
    // set_data reports success yet reruns of the identical command diverge,
    // the signature of restored-state bookkeeping desync reading
    // uninitialized cells. Until the fork's multi-stream state serialization
    // is proven, rotation refuses mid-generation parking: each cohort runs
    // to completion, then rotates. That is the throughput use case; spans
    // below max_tokens buy only latency fairness and are not worth wrong
    // tokens.
    const bool rot_unsafe = [] {
        const char* v = std::getenv("DRAY_ROTATE_UNSAFE");
        return v && v[0] == '1';
    }();
    if (p.span < p.max_tokens && !rot_unsafe) {
        rr.error = "rotate: span " + std::to_string(p.span) + " < max_tokens " +
                   std::to_string(p.max_tokens) + " requires mid-generation state parking, which is "
                   "NOT yet proven bit-exact in this build (reruns diverge); use --rotate >= " +
                   std::to_string(p.max_tokens) + " (run-to-completion cohorts)";
        return rr;
    }
    if (p.state_dir.empty()) { rr.error = "rotate: --state-dir is required (rotation writes; where is an explicit choice)"; return rr; }
    // Created here, or refused here with the reason: a missing directory used to
    // surface only as "state park failed" on every sequence, mid-run.
    {
        std::error_code ec;
        std::filesystem::create_directories(p.state_dir, ec);
        if (ec || !std::filesystem::is_directory(p.state_dir, ec)) {
            rr.error = "rotate: cannot create --state-dir " + p.state_dir + ": " + ec.message();
            return rr;
        }
    }
    rr.seqs.resize(static_cast<size_t>(N));

    llama_memory_clear(llama_get_memory(engine_.context()), true);
    const int32_t ctx_seq = static_cast<int32_t>(llama_n_ctx_seq(engine_.context()));

    // Host-persistent per-GLOBAL-sequence machinery. Only KV/recurrent state
    // rotates through llama slots; samplers, pending tokens and text live in
    // host RAM for the whole run, so a residency is pure state scheduling.
    struct RSeq {
        llama_sampler* smpl = nullptr;
        llama_token pending = 0;
        bool live = false;
        bool admitted = false;
        bool prefilled = false;
        int32_t n_prompt = 0;
        int32_t done = 0;               // decode steps completed
        std::vector<llama_token> toks;
    };
    std::vector<RSeq> gs(static_cast<size_t>(N));

    auto feed_g = [&](int32_t g, const char* b, size_t np) {
        GenResult& r = rr.seqs[static_cast<size_t>(g)];
        r.text.append(b, np);
        const StopMatch m = find_stop(r.text, np, p.stop);
        if (m.found()) {
            r.text.resize(m.pos);
            r.stop_hit = true;
        }
    };

    // Tokenize + admit + build samplers up front, exactly the batch path's
    // discipline: refuse per sequence before any drive time is spent.
    for (int32_t g = 0; g < N; ++g) {
        GenResult& r = rr.seqs[static_cast<size_t>(g)];
        RSeq& q = gs[static_cast<size_t>(g)];
        const std::string& pr = p.prompts[static_cast<size_t>(g)];
        std::vector<llama_token> toks(pr.size() + 8);
        int32_t n = llama_tokenize(engine_.vocab(), pr.c_str(), static_cast<int32_t>(pr.size()),
                                   toks.data(), static_cast<int32_t>(toks.size()), true, true);
        if (n < 0) {
            toks.resize(static_cast<size_t>(-n));
            n = llama_tokenize(engine_.vocab(), pr.c_str(), static_cast<int32_t>(pr.size()),
                               toks.data(), static_cast<int32_t>(toks.size()), true, true);
        }
        if (n <= 0) { r.error = "rotate: tokenize failed"; continue; }
        if (n >= ctx_seq) {
            r.bad_request = true;
            r.error = "rotate: prompt of " + std::to_string(n) + " tokens >= per-sequence context " +
                      std::to_string(ctx_seq);
            continue;
        }
        q.n_prompt = n;
        r.tokens_in = n;
        q.toks.assign(toks.begin(), toks.begin() + n);
        llama_sampler_chain_params sp = llama_sampler_chain_default_params();
        q.smpl = llama_sampler_chain_init(sp);
        if (p.temperature > 0.0f) {
            llama_sampler_chain_add(q.smpl, llama_sampler_init_temp(p.temperature));
            // Seed by GLOBAL index so a rotated run and split unrotated runs
            // of the same prompts sample identically -- the oracle depends on it.
            const uint32_t sseed = (p.seed ? p.seed : 1234u) + static_cast<uint32_t>(g);
            llama_sampler_chain_add(q.smpl, llama_sampler_init_dist(sseed));
        } else {
            llama_sampler_chain_add(q.smpl, llama_sampler_init_greedy());
        }
        q.admitted = true;
    }

    auto state_path = [&](int32_t g) {
        return p.state_dir + "/g" + std::to_string(g) + ".state";
    };
    const bool keep_states = rot_unsafe;   // diagnostics: copy each park aside
    int32_t park_no = 0;
    auto park = [&](int32_t g, int32_t slot) -> bool {
        const size_t sz = llama_state_seq_get_size(engine_.context(), slot);
        if (sz == 0) return false;
        std::vector<uint8_t> buf(sz);
        const size_t got = llama_state_seq_get_data(engine_.context(), buf.data(), sz, slot);
        if (got == 0) return false;
        FILE* f = std::fopen(state_path(g).c_str(), "wb");
        if (!f) return false;
        const size_t wr = std::fwrite(buf.data(), 1, got, f);
        std::fclose(f);
        if (wr != got) return false;
        rr.state_bytes_written += wr;
        if (keep_states) {
            FILE* k = std::fopen((state_path(g) + ".park" + std::to_string(park_no++)).c_str(), "wb");
            if (k) { std::fwrite(buf.data(), 1, got, k); std::fclose(k); }
        }
        return true;
    };
    auto unpark = [&](int32_t g, int32_t slot) -> bool {
        FILE* f = std::fopen(state_path(g).c_str(), "rb");
        if (!f) return false;
        std::fseek(f, 0, SEEK_END);
        const long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (sz <= 0) { std::fclose(f); return false; }
        std::vector<uint8_t> buf(static_cast<size_t>(sz));
        const size_t rd = std::fread(buf.data(), 1, buf.size(), f);
        std::fclose(f);
        if (rd != buf.size()) return false;
        if (llama_state_seq_set_data(engine_.context(), buf.data(), buf.size(), slot) == 0) return false;
        rr.state_bytes_read += rd;
        return true;
    };

    const int32_t cohorts = (N + W - 1) / W;
    bool all_done = false;
    bool cancelled = false;
    while (!all_done && !cancelled) {
        all_done = true;
        for (int32_t c = 0; c < cohorts && !cancelled; ++c) {
            const int32_t g0 = c * W;
            const int32_t cw = std::min(W, N - g0);
            // Anything left to do in this cohort this round?
            bool work = false;
            for (int32_t s = 0; s < cw; ++s) {
                const RSeq& q = gs[static_cast<size_t>(g0 + s)];
                if (q.admitted && (!q.prefilled || (q.live && q.done < p.max_tokens))) { work = true; break; }
            }
            if (!work) continue;
            if (p.should_continue && !p.should_continue()) { cancelled = true; break; }

            // RESTORE: previously-parked live sequences re-enter their slots.
            for (int32_t s = 0; s < cw; ++s) {
                const int32_t g = g0 + s;
                RSeq& q = gs[static_cast<size_t>(g)];
                if (!q.admitted || !q.prefilled || !q.live) continue;
                if (!unpark(g, s)) {
                    q.live = false;
                    rr.seqs[static_cast<size_t>(g)].error = "rotate: state restore failed";
                }
            }

            // FIRST RESIDENCY: joint prefill for this cohort's unprefilled
            // sequences -- the batch path's shared-chunk scheme, slots local.
            {
                llama_batch pb = llama_batch_init(engine_.prefill_chunk(), 0, 1);
                std::vector<int32_t> fed(static_cast<size_t>(cw), 0);
                std::vector<int32_t> lseq(static_cast<size_t>(engine_.prefill_chunk()));
                std::vector<int32_t> lrow(static_cast<size_t>(engine_.prefill_chunk()));
                for (;;) {
                    pb.n_tokens = 0;
                    int32_t nl = 0;
                    for (int32_t s = 0; s < cw && pb.n_tokens < engine_.prefill_chunk(); ++s) {
                        RSeq& q = gs[static_cast<size_t>(g0 + s)];
                        if (!q.admitted || q.prefilled) continue;
                        while (fed[static_cast<size_t>(s)] < q.n_prompt && pb.n_tokens < engine_.prefill_chunk()) {
                            const int32_t i0 = fed[static_cast<size_t>(s)]++;
                            const int32_t j = pb.n_tokens++;
                            pb.token[j] = q.toks[static_cast<size_t>(i0)];
                            pb.pos[j] = i0;
                            pb.n_seq_id[j] = 1;
                            pb.seq_id[j][0] = s;
                            const bool last = (i0 == q.n_prompt - 1);
                            pb.logits[j] = static_cast<int8_t>(last);
                            if (last) { lseq[static_cast<size_t>(nl)] = s; lrow[static_cast<size_t>(nl++)] = j; }
                        }
                    }
                    if (pb.n_tokens == 0) break;
                    if (llama_decode(engine_.context(), pb) != 0) {
                        for (int32_t s = 0; s < cw; ++s) {
                            RSeq& q = gs[static_cast<size_t>(g0 + s)];
                            if (q.admitted && !q.prefilled && fed[static_cast<size_t>(s)] > 0) {
                                q.admitted = false;
                                rr.seqs[static_cast<size_t>(g0 + s)].error = "rotate: prefill decode failed";
                            }
                        }
                        break;
                    }
                    for (int32_t li = 0; li < nl; ++li) {
                        const int32_t s = lseq[static_cast<size_t>(li)];
                        const int32_t g = g0 + s;
                        RSeq& q = gs[static_cast<size_t>(g)];
                        q.pending = llama_sampler_sample(q.smpl, engine_.context(), lrow[static_cast<size_t>(li)]);
                        q.prefilled = true;
                        q.live = !llama_vocab_is_eog(engine_.vocab(), q.pending);
                        if (!q.live) rr.seqs[static_cast<size_t>(g)].truncated_by_eog = true;
                    }
                }
                llama_batch_free(pb);
            }

            // SPAN of lockstep decode, the batch loop's exact shape on slots.
            llama_batch db = llama_batch_init(cw, 0, 1);
            std::vector<int32_t> row_of(static_cast<size_t>(cw), -1);
            for (int32_t t = 0; t < p.span; ++t) {
                if (p.should_continue && !p.should_continue()) { cancelled = true; break; }
                int32_t live = 0;
                db.n_tokens = 0;
                for (int32_t s = 0; s < cw; ++s) {
                    const int32_t g = g0 + s;
                    RSeq& q = gs[static_cast<size_t>(g)];
                    row_of[static_cast<size_t>(s)] = -1;
                    if (!q.admitted || !q.live || q.done >= p.max_tokens) continue;
                    if (q.n_prompt + q.done + 1 > ctx_seq) {
                        q.live = false;
                        rr.seqs[static_cast<size_t>(g)].ctx_wall = true;
                        continue;
                    }
                    const int32_t j = db.n_tokens++;
                    db.token[j] = q.pending;
                    db.pos[j] = q.n_prompt + q.done;
                    db.n_seq_id[j] = 1;
                    db.seq_id[j][0] = s;
                    db.logits[j] = 1;
                    row_of[static_cast<size_t>(s)] = j;
                    ++live;
                }
                if (live == 0) break;
                if (llama_decode(engine_.context(), db) != 0) {
                    const bool tainted = engine_.weights_failed();
                    for (int32_t s = 0; s < cw; ++s) {
                        RSeq& q = gs[static_cast<size_t>(g0 + s)];
                        if (!q.live) continue;
                        if (tainted) rr.seqs[static_cast<size_t>(g0 + s)].aborted = true;
                        else rr.seqs[static_cast<size_t>(g0 + s)].error = "rotate: decode failed";
                        q.live = false;
                    }
                    break;
                }
                engine_.latch_over_cap();
                engine_.bind_cap_to_process();
                for (int32_t s = 0; s < cw; ++s) {
                    const int32_t g = g0 + s;
                    RSeq& q = gs[static_cast<size_t>(g)];
                    if (row_of[static_cast<size_t>(s)] < 0) continue;
                    GenResult& r = rr.seqs[static_cast<size_t>(g)];
                    char buf[256];
                    const int32_t np = llama_token_to_piece(engine_.vocab(), q.pending, buf, sizeof(buf), 0, true);
                    if (np > 0) feed_g(g, buf, static_cast<size_t>(np));
                    q.done += 1;
                    r.tokens_out = q.done;
                    if (r.stop_hit) { r.truncated_by_eog = true; q.live = false; continue; }
                    if (q.done >= p.max_tokens) { q.live = false; continue; }
                    const llama_token nxt =
                        llama_sampler_sample(q.smpl, engine_.context(), row_of[static_cast<size_t>(s)]);
                    if (llama_vocab_is_eog(engine_.vocab(), nxt)) {
                        r.truncated_by_eog = true;
                        q.live = false;
                        continue;
                    }
                    q.pending = nxt;
                }
            }
            llama_batch_free(db);
            ++rr.rounds;

            // PARK live sequences; finished ones just vacate. Slots must be
            // empty either way -- the next cohort owns them.
            for (int32_t s = 0; s < cw; ++s) {
                const int32_t g = g0 + s;
                RSeq& q = gs[static_cast<size_t>(g)];
                if (q.admitted && q.prefilled && q.live && q.done < p.max_tokens) {
                    // Diagnostics under DRAY_ROTATE_UNSAFE: keep a copy of
                    // each park for byte-comparison across reruns.
                    if (!park(g, s)) {
                        q.live = false;
                        rr.seqs[static_cast<size_t>(g)].error = "rotate: state park failed";
                    }
                    all_done = false;
                }
                llama_memory_seq_rm(llama_get_memory(engine_.context()), s, -1, -1);
            }
            int32_t done_ct = 0;
            for (int32_t g = 0; g < N; ++g) {
                const RSeq& q = gs[static_cast<size_t>(g)];
                if (!q.admitted || !q.live || q.done >= p.max_tokens) ++done_ct;
            }
            if (p.on_round) p.on_round(static_cast<int32_t>(rr.rounds), c, done_ct);
        }
    }
    if (cancelled) {
        for (int32_t g = 0; g < N; ++g) {
            if (gs[static_cast<size_t>(g)].live) rr.seqs[static_cast<size_t>(g)].cancelled = true;
        }
    }
    for (int32_t g = 0; g < N; ++g) {
        RSeq& q = gs[static_cast<size_t>(g)];
        if (q.smpl) llama_sampler_free(q.smpl);
        std::remove(state_path(g).c_str());
    }
    return rr;
}

}  // namespace dray::engine
