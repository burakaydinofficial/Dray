#include "engine/batch_generator.h"

#include <algorithm>
#include <string>
#include <vector>

#include "llama.h"

#include "engine/engine.h"
#include "engine/streamed_text.h"

namespace dray::engine {

// The ids tensors run [k, B] through the compaction machinery here -- the
// ne[1]>1 paths three audits called untested get their exercise WITH an oracle
// attached.
BatchResult BatchGenerator::run(const BatchParams& p) {
    BatchResult br;
    const int32_t B = static_cast<int32_t>(p.prompts.size());
    if (B < 1) { br.error = "batch: no prompts"; return br; }
    if (static_cast<uint32_t>(B) > engine_.funded_sequences()) {
        // Invariant 1: the admission funded plan.n_seq sequences; more would
        // run KV outside the cap. Refused here, not discovered mid-decode.
        br.error = "batch: " + std::to_string(B) + " sequences but the plan admitted " +
                   std::to_string(engine_.funded_sequences()) + "; re-plan with --batch " +
                   std::to_string(B);
        return br;
    }
    br.seqs.resize(static_cast<size_t>(B));

    llama_memory_clear(llama_get_memory(engine_.context()), true);

    // Per-sequence machinery: sampler (seed+s, disclosed), UTF-8-free feed
    // (offline mode buffers whole texts; no SSE hold-back needed), stop
    // matcher, liveness.
    struct Seq {
        llama_sampler* smpl = nullptr;
        llama_token pending = 0;
        bool live = false;
        bool admitted = false;
        int32_t n_prompt = 0;
        std::vector<llama_token> toks;
    };
    std::vector<Seq> seqs(static_cast<size_t>(B));
    auto feed_seq = [&](int32_t s, const char* b, size_t np) {
        GenResult& r = br.seqs[static_cast<size_t>(s)];
        r.text.append(b, np);
        const StopMatch m = find_stop(r.text, np, p.stop);
        if (m.found()) {
            r.text.erase(m.pos);
            r.stop_hit = true;
        }
    };

    // Prefill each sequence in turn (chunked; activation scratch reused, which
    // is why the planner funds it once).
    for (int32_t s = 0; s < B; ++s) {
        GenResult& r = br.seqs[static_cast<size_t>(s)];
        Seq& q = seqs[static_cast<size_t>(s)];
        const std::string& pr = p.prompts[static_cast<size_t>(s)];
        std::vector<llama_token> toks(pr.size() + 8);
        // J3: per-sequence admission BEFORE the prefill is paid for in drive
        // time -- against the PER-SEQUENCE ring, not the aggregate.
        int32_t n = llama_tokenize(engine_.vocab(), pr.c_str(), static_cast<int32_t>(pr.size()),
                                   toks.data(), static_cast<int32_t>(toks.size()), true, true);
        if (n < 0) {
            toks.resize(static_cast<size_t>(-n));
            n = llama_tokenize(engine_.vocab(), pr.c_str(), static_cast<int32_t>(pr.size()),
                               toks.data(), static_cast<int32_t>(toks.size()), true, true);
        }
        if (n <= 0) { r.error = "batch: tokenize failed"; continue; }
        {
            const int32_t cseq = static_cast<int32_t>(llama_n_ctx_seq(engine_.context()));
            if (n >= cseq) {
                r.bad_request = true;
                r.error = "batch: prompt of " + std::to_string(n) + " tokens >= per-sequence context " +
                          std::to_string(cseq) + "; raise --ctx or shorten the prompt";
                continue;
            }
        }
        q.n_prompt = n;
        r.tokens_in = n;
        q.toks.assign(toks.begin(), toks.begin() + n);
        llama_sampler_chain_params sp = llama_sampler_chain_default_params();
        q.smpl = llama_sampler_chain_init(sp);
        if (p.temperature > 0.0f) {
            llama_sampler_chain_add(q.smpl, llama_sampler_init_temp(p.temperature));
            const uint32_t sseed = (p.seed ? p.seed : 1234u) + static_cast<uint32_t>(s);
            llama_sampler_chain_add(q.smpl, llama_sampler_init_dist(sseed));
        } else {
            llama_sampler_chain_add(q.smpl, llama_sampler_init_greedy());
        }
        q.admitted = true;
    }

    // JOINT prefill: pack ALL sequences' prompt tokens into shared
    // engine_.prefill_chunk() chunks -- ceil(total/512) model sweeps instead of one
    // sweep per sequence (measured: 32 sequential prefills were 72% of the
    // crown run's bytes). Chunks carry ne[1] > n_seq, so the streamer keeps
    // them on the prefill SCAN path (whole tensors through the ring, no
    // union-region materialisation) -- the same path the sequential version
    // ran on, with the same hazards, just fewer sweeps; never more than the
    // sequential count in any prompt/batch shape. Each token carries its own
    // seq_id and position; a sequence's last prompt token requests logits so
    // its first generated token samples from that row.
    {
        const int32_t chunk_cap = engine_.prefill_chunk();
        llama_batch pb = llama_batch_init(chunk_cap, 0, 1);
        std::vector<int32_t> fed(static_cast<size_t>(B), 0);
        std::vector<int32_t> logit_seq(static_cast<size_t>(chunk_cap));
        std::vector<int32_t> logit_row(static_cast<size_t>(chunk_cap));
        bool cancelled = false;
        for (;;) {
            if (p.should_continue && !p.should_continue()) { cancelled = true; break; }
            pb.n_tokens = 0;
            int32_t n_logit = 0;
            for (int32_t s = 0; s < B && pb.n_tokens < chunk_cap; ++s) {
                Seq& q = seqs[static_cast<size_t>(s)];
                if (!q.admitted) continue;
                while (fed[static_cast<size_t>(s)] < q.n_prompt && pb.n_tokens < chunk_cap) {
                    const int32_t i0 = fed[static_cast<size_t>(s)]++;
                    const int32_t j = pb.n_tokens++;
                    pb.token[j] = q.toks[static_cast<size_t>(i0)];
                    pb.pos[j] = i0;
                    pb.n_seq_id[j] = 1;
                    pb.seq_id[j][0] = s;
                    const bool last = (i0 == q.n_prompt - 1);
                    pb.logits[j] = static_cast<int8_t>(last);
                    // llama_get_logits_ith wants the BATCH TOKEN index (it maps
                    // through output_ids itself); a dense logits ordinal aborts.
                    if (last) { logit_seq[static_cast<size_t>(n_logit)] = s;
                                logit_row[static_cast<size_t>(n_logit++)] = j; }
                }
            }
            if (pb.n_tokens == 0) break;
            if (llama_decode(engine_.context(), pb) != 0) {
                for (int32_t s = 0; s < B; ++s) {
                    Seq& q = seqs[static_cast<size_t>(s)];
                    if (q.admitted && fed[static_cast<size_t>(s)] > 0) {
                        q.admitted = false;
                        br.seqs[static_cast<size_t>(s)].error = "batch: prefill decode failed";
                    }
                }
                break;
            }
            // Sample the first token of every sequence whose prompt completed
            // in this chunk; logits rows appear in request order.
            for (int32_t li = 0; li < n_logit; ++li) {
                const int32_t s = logit_seq[static_cast<size_t>(li)];
                Seq& q = seqs[static_cast<size_t>(s)];
                q.pending = llama_sampler_sample(q.smpl, engine_.context(), logit_row[static_cast<size_t>(li)]);
                q.live = !llama_vocab_is_eog(engine_.vocab(), q.pending);
                if (!q.live) br.seqs[static_cast<size_t>(s)].truncated_by_eog = true;
            }
        }
        llama_batch_free(pb);
        if (cancelled) {
            for (int32_t s = 0; s < B; ++s) {
                if (seqs[static_cast<size_t>(s)].admitted)
                    br.seqs[static_cast<size_t>(s)].cancelled = true;
            }
        }
    }

    br.prefill_end_bytes = engine_.counters().bytes_streamed;   // J7 boundary

    // Lockstep decode: one llama_decode per step carrying every live
    // sequence's pending token; the streamer sees ids [k, live] and reads the
    // union once.
    llama_batch db = llama_batch_init(B, 0, 1);
    std::vector<int32_t> row_of(static_cast<size_t>(B), -1);
    // J1: this build does NOT use a unified KV pool -- kv_unified defaults
    // false, so each sequence owns a PRIVATE ring of llama_n_ctx_seq() cells
    // and the binding constraint is PER SEQUENCE. (An earlier total-cells
    // guard here was n_seq times too loose; every proven run was symmetric,
    // which made its algebra accidentally correct.)
    const int32_t ctx_seq = static_cast<int32_t>(llama_n_ctx_seq(engine_.context()));
    for (int32_t step = 0; step < p.max_tokens; ++step) {
        if (p.should_continue && !p.should_continue()) {
            for (int32_t s = 0; s < B; ++s) {
                if (seqs[static_cast<size_t>(s)].live) br.seqs[static_cast<size_t>(s)].cancelled = true;
            }
            break;
        }
        int32_t live = 0;
        db.n_tokens = 0;
        for (int32_t s = 0; s < B; ++s) {
            Seq& q = seqs[static_cast<size_t>(s)];
            row_of[static_cast<size_t>(s)] = -1;
            if (!q.live) continue;
            // J1: the per-sequence wall, checked on the sequence it binds --
            // only THIS row retires, honestly marked, the rest decode on.
            if (q.n_prompt + step + 1 > ctx_seq) {
                q.live = false;
                br.seqs[static_cast<size_t>(s)].ctx_wall = true;
                continue;
            }
            const int32_t j = db.n_tokens++;
            db.token[j] = q.pending;
            db.pos[j] = q.n_prompt + step;
            db.n_seq_id[j] = 1;
            db.seq_id[j][0] = s;
            db.logits[j] = 1;
            row_of[static_cast<size_t>(s)] = j;
            ++live;
        }
        if (live == 0) break;
        if (llama_decode(engine_.context(), db) != 0) {
            const bool tainted = engine_.weights_failed();
            for (int32_t s = 0; s < B; ++s) {
                if (!seqs[static_cast<size_t>(s)].live) continue;
                if (tainted) br.seqs[static_cast<size_t>(s)].aborted = true;
                else br.seqs[static_cast<size_t>(s)].error = "batch: decode failed";
            }
            break;
        }
        ++br.steps;
        engine_.note_progress(static_cast<int32_t>(br.steps));
        engine_.latch_over_cap();
        // J5: the multi-day discipline, per STEP -- bind the cap on the worse
        // of RSS and commit exactly as the single-stream loop does; without
        // this the unreserved charge froze at open and over_cap was blinded.
        engine_.bind_cap_to_process();
        for (int32_t s = 0; s < B; ++s) {
            Seq& q = seqs[static_cast<size_t>(s)];
            if (!q.live) continue;
            GenResult& r = br.seqs[static_cast<size_t>(s)];
            char buf[256];
            const int32_t np = llama_token_to_piece(engine_.vocab(), q.pending, buf, sizeof(buf), 0, true);
            if (np > 0) feed_seq(s, buf, static_cast<size_t>(np));
            r.tokens_out = step + 1;
            if (r.stop_hit) { r.truncated_by_eog = true; q.live = false; continue; }
            const llama_token nxt =
                llama_sampler_sample(q.smpl, engine_.context(), row_of[static_cast<size_t>(s)]);
            if (llama_vocab_is_eog(engine_.vocab(), nxt)) {
                r.truncated_by_eog = true;
                q.live = false;
            } else {
                q.pending = nxt;
            }
        }
        if (p.on_step) p.on_step(step + 1, live);
    }
    llama_batch_free(db);
    for (auto& q : seqs) { if (q.smpl) llama_sampler_free(q.smpl); }
    if (engine_.weights_failed()) {
        for (auto& r : br.seqs) r.aborted = true;
    }
    return br;
}

}  // namespace dray::engine
