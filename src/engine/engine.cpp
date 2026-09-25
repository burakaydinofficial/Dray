// Engine lifetime, readouts, and the single-stream generate loop. Setup lives in
// engine_setup.cpp; batch and rotation in their own generators.

#include "engine/engine.h"

#include <algorithm>

#include "llama.h"

#include "backend/stream_buffer.h"
#include "engine/engine_internal.h"
#include "mem/accountant.h"
#include "plan/residency.h"

namespace dray::engine {

// Release order matters and is the old engine_close order: llama's context and
// model first (they point into streamer memory), then the metadata they were
// built from, then the streamer and the ledger it charges, then the backend.
Engine::~Engine() {
    if (lctx_) llama_free(lctx_);
    if (model_) llama_model_free(model_);
    if (meta_) backend::free_merged_metadata(*meta_);
    streamer_.reset();
    acct_.reset();
    llama_backend_free();
}

EngineCounters Engine::counters() const {
    EngineCounters c;
    c.bytes_streamed = streamer_->bytes_streamed();
    c.bytes_gather   = streamer_->routed_bytes_read();
    c.routed_needed  = streamer_->routed_bytes_needed();
    c.uncond_needed  = streamer_->uncond_bytes_needed();
    c.uncond_read    = streamer_->uncond_bytes_read();
    c.nodes          = streamer_->nodes_materialised();
    c.failures       = streamer_->failures();
    c.resident_bytes = acct_->used();
    c.over_cap       = streamer_->over_cap();
    return c;
}

std::string Engine::plan_report() const       { return plan_->report(); }
std::string Engine::streamer_report() const   { return streamer_->report(); }
std::string Engine::accountant_report() const { return streamer_->accountant_report(); }
uint64_t Engine::failures() const             { return streamer_->failures(); }
uint32_t Engine::funded_sequences() const     { return plan_->n_seq; }
bool Engine::weights_failed() const           { return streamer_->aborted(); }

void Engine::latch_over_cap() {
    if (streamer_->over_cap()) live_over_cap_.store(true, std::memory_order_relaxed);
}

void Engine::bind_cap_to_process() {
    const uint64_t worst = std::max<uint64_t>(
        mem::Accountant::process_rss(),
        mem::Accountant::process_committed());
    if (worst) streamer_->rebudget_against_rss(worst);
}

GenResult Engine::generate(const GenParams& p, const TokenSink& token_cb) {
    GenResult r;
    int32_t max_out = p.max_tokens;
    StreamedText text(&r, p.stop, p.stop_seed, token_cb);

    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sp);
    if (p.temperature <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(p.temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(
            p.seed == 0 ? LLAMA_DEFAULT_SEED : p.seed));
    }

    if (p.resume) {
        // I3: a spent budget must be decided BEFORE the pending decode -- the
        // old order decoded, fed and counted the token, then declared a clean
        // finish one token past the bound the client set.
        if (max_out < 1) {
            llama_sampler_free(smpl);
            r.tokens_out = 0;
            r.truncated_by_eog = false;   // finish_reason: length, honestly
            return r;
        }
        // The caller restored a snapshot taken at a safepoint: state excludes
        // the pending token, whose decode here regenerates the logits. No
        // clear, no prefill -- the restored state IS the prefix.
        llama_token pending = p.resume_pending;
        char buf0[256];
        int32_t np0 = llama_token_to_piece(vocab_, pending, buf0, sizeof(buf0), 0, true);
        llama_batch b0 = llama_batch_get_one(&pending, 1);
        if (llama_decode(lctx_, b0) != 0) {
            llama_sampler_free(smpl);
            r.error = "resume redecode failed";
            return r;
        }
        if (np0 > 0) text.feed(buf0, static_cast<size_t>(np0));
        r.tokens_out = 1;
        // G3 FIRST (H7c): a stop completed by the pending token's own piece
        // completes the job regardless of context arithmetic below.
        if (r.stop_hit) {
            r.truncated_by_eog = true;
            text.flush();
            llama_sampler_free(smpl);
            return r;
        }
        // Pass-4b correction + H7: seq_len is read AFTER the pending decode so
        // it already counts that slot -- the bound for the remaining
        // (max_out - 1) decodes is room + 1. max_out can also arrive at 0
        // legitimately (the budget was spent exactly at the interruption):
        // that is a CLEAN FINISH, not an error -- erroring resurrected the job
        // every restart until quarantine.
        {
            if (max_out < 1) {
                r.truncated_by_eog = false;   // finish_reason: length, honestly
                text.flush();
                llama_sampler_free(smpl);
                return r;
            }
            const int32_t seq_len = static_cast<int32_t>(
                llama_memory_seq_pos_max(llama_get_memory(lctx_), 0)) + 1;
            const int32_t room = static_cast<int32_t>(llama_n_ctx_seq(lctx_)) - seq_len;
            if (max_out > room + 1) max_out = room + 1;
            if (max_out < 1) {
                text.flush();
                llama_sampler_free(smpl);
                r.error = "resume: context is full; nothing can be generated";
                return r;
            }
        }
    } else {
        // Requests are independent: clear KV and recurrent state, then prefill.
        llama_memory_clear(llama_get_memory(lctx_), true);

        std::vector<llama_token> toks(p.prompt.size() + 8);
        int32_t n = llama_tokenize(vocab_, p.prompt.c_str(),
                                   static_cast<int32_t>(p.prompt.size()),
                                   toks.data(), static_cast<int32_t>(toks.size()), true, true);
        if (n < 0) {
            toks.resize(static_cast<size_t>(-n));
            n = llama_tokenize(vocab_, p.prompt.c_str(),
                               static_cast<int32_t>(p.prompt.size()),
                               toks.data(), static_cast<int32_t>(toks.size()), true, true);
        }
        if (n <= 0) { llama_sampler_free(smpl); r.error = "tokenization failed"; return r; }
        toks.resize(static_cast<size_t>(n));

        // Admission against the context (swarm S3): a prompt that cannot fit is
        // an error the client can act on, never an abort hours of work deep.
        // J3: under n_seq>1 llama_n_ctx() is the AGGREGATE; the per-sequence
        // bound is llama_n_ctx_seq (identical when n_seq==1).
        const int32_t nctx = static_cast<int32_t>(llama_n_ctx_seq(lctx_));
        if (n >= nctx) {
            llama_sampler_free(smpl);
            r.error = "prompt is " + std::to_string(n) + " tokens but the context is " +
                      std::to_string(nctx) + "; raise --ctx or shorten the prompt";
            r.bad_request = true;
            return r;
        }

        // Generation cannot outrun the KV either: clamp instead of dying at
        // "decode failed" hours in (the client sees the honest finish_reason).
        if (max_out > nctx - n) max_out = nctx - n;

        r.tokens_in = n;   // T9
        if (p.on_prefill) p.on_prefill(n);
        // Chunked prefill: n_batch is a compute-sizing knob, not a prompt limit.
        for (int32_t i0 = 0; i0 < n; i0 += prefill_batch_) {
            if (p.on_prefill_progress) p.on_prefill_progress(i0);   // F13
            // T3: prefill is minutes on the large models; cancel, disconnect
            // and shutdown must not be ignored for its whole duration.
            if (p.should_continue && !p.should_continue()) {
                llama_sampler_free(smpl);
                r.cancelled = true;
                return r;
            }
            const int32_t take = std::min(prefill_batch_, n - i0);
            llama_batch batch = llama_batch_get_one(toks.data() + i0, take);
            if (llama_decode(lctx_, batch) != 0) {
                llama_sampler_free(smpl);
                // A weight that failed to materialise during the prompt must
                // not come back as a bare "prefill failed".
                if (weights_failed()) {
                    r.aborted = true;
                    r.error = "prefill failed: a weight failed to materialise";
                } else {
                    r.error = "prefill failed";
                }
                return r;
            }
        }
    }

    for (int32_t i = r.tokens_out; i < max_out; ++i) {
        llama_token t = llama_sampler_sample(smpl, lctx_, -1);
        if (llama_vocab_is_eog(vocab_, t)) { r.truncated_by_eog = true; break; }
        if (p.on_safepoint) {
            // T8: flush the held-back UTF-8 tail into the consumer BEFORE the
            // checkpoint reads its text, or resumed text loses those bytes and
            // gains a U+FFFD. Safepoints exist only on the job path, whose
            // token_cb appends to the registry -- no SSE sees a partial glyph.
            text.flush();
            p.on_safepoint(t, p.resume_tokens_done + i);   // F5: absolute
        }

        char buf[256];
        int32_t np = llama_token_to_piece(vocab_, t, buf, sizeof(buf), 0, true);

        llama_batch b1 = llama_batch_get_one(&t, 1);
        if (llama_decode(lctx_, b1) != 0) {
            // ASK WHY FIRST. A materialise failure makes ggml return ABORTED,
            // so llama_decode fails here; callers keying on r.aborted (the
            // streaming error frame, the job status) must be able to tell an
            // untrustworthy run from a broken request (2026-08-24 audit).
            if (weights_failed()) {
                r.aborted = true;
                r.error = "decode failed: a weight failed to materialise";
            } else {
                r.error = "decode failed";
            }
            break;
        }
        note_progress(i + 1);
        latch_over_cap();
        if (weights_failed()) { r.aborted = true; break; }
        if (p.should_continue && !p.should_continue()) {
            r.cancelled = true;
            r.tokens_out = i + 1;
            // T1: a cancellation that can resume needs the pending-token shape
            // (state BEFORE a sampled token's decode). F6/pass-4b: token t was
            // DECODED (it is in the snapshot state) but its piece had not been
            // fed. Feed it for EVERY cancel.
            if (np > 0) text.feed(buf, static_cast<size_t>(np));
            if (r.stop_hit) {
                r.cancelled = false;
                r.truncated_by_eog = true;
                break;
            }
            if (p.on_safepoint) {
                // G4: the feed above is the stop matcher; the stop-hit exit
                // before this block keeps a checkpoint from recording a
                // pending token past the stop.
                text.flush();
                // I3: never checkpoint a pending token PAST the client's budget
                // -- the restarted session would compute max_tokens 0 and the
                // poll would report orig_max + 1.
                if (i + 1 < max_out) {
                    const llama_token pt = llama_sampler_sample(smpl, lctx_, -1);
                    if (!llama_vocab_is_eog(vocab_, pt)) {
                        // F5: absolute count, base + session-local.
                        p.on_safepoint(pt, p.resume_tokens_done + i + 1);
                    }
                }
            }
            break;
        }
        r.tokens_out = i + 1;
        // Post-decode on purpose: a readout ticking here matches the verified
        // measurement path's timing exactly.
        if (np > 0) text.feed(buf, static_cast<size_t>(np));
        if (r.stop_hit) { r.truncated_by_eog = true; break; }   // T10: honest "stop"

        bind_cap_to_process();
    }

    // Flush any held-back tail; if genuinely malformed, the dump-side replace
    // backstop renders it as U+FFFD instead of killing the process.
    text.flush();
    llama_sampler_free(smpl);
    return r;
}

}  // namespace dray::engine
