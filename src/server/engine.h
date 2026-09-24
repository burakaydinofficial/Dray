// The engine behind the server: one loaded model + streaming context, reused
// across requests. This is cmd_run's setup sequence made reusable -- cmd_run
// itself is deliberately untouched (it is the verified measurement path; the
// two will be consolidated in a calm moment, not the night the server lands).
//
// Serialized admission is the caller's job (Invariant 7): generate() assumes one
// generation at a time and the server holds a mutex across each request.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;
struct llama_vocab;
namespace dray::mem { class Accountant; }

namespace dray::server {

struct EngineConfig {
    std::string model_path;
    uint64_t    cap = 0;
    uint32_t    n_ctx = 0;      // 0 = default (32768)
    std::string repack_dir;     // apply a repacked companion when non-empty
    bool        plan_to_stdout = false;  // cmd_run prints the plan for the USER;
                                         // serve logs it to stderr for the operator
    // F4: reserve a standing checkpoint-blob allowance at open, so a cap that
    // cannot hold checkpoints refuses at admission instead of every safepoint
    // silently failing forever once the cache saturates (which is steady state).
    bool        reserve_checkpoint = false;
    // Batch: sequences the admission funds and the context is sized for.
    // 1 = classic single-stream. J1: each sequence gets a PRIVATE ring of
    // n_ctx cells (kv_unified is false); llama receives n_ctx*n_seq total and
    // splits it per stream.
    uint32_t    n_seq = 1;
    // llama --override-kv strings ("key=type:value"); applied at model load,
    // taking precedence over the profile's kv_defaults for the same key.
    std::vector<std::string> kv_overrides;
    // Runtime GPU consent (--gpu). Without it a backend-carrying build still
    // restricts llama to CPU devices -- one binary, opt-in at start.
    bool gpu = false;
    // Prefill chunk tokens (--prefill-chunk, default 512). The GPU compute
    // buffer scales with this; small values let low-VRAM cards prefill.
    int32_t prefill_chunk = 0;
    // Compute threads (--threads). 0 = the measured optimum (4). llama's own
    // default is FOUR regardless of machine size, which silently capped every
    // compute-bound measurement this project made before 2026-08-21.
    int32_t n_threads = 0;
    // --force-stream: keep the streaming buffer even when the whole model
    // would fit the cap. Default behaviour hands allocation to llama in that
    // case (3x faster: its repacked-weight kernels), but MEASUREMENTS and
    // correctness gates must be able to pin the streaming path or they
    // silently stop testing this engine at all.
    bool force_stream = false;
    // --resident: hand allocation to llama when the model fits (EXPERIMENTAL,
    // currently produces wrong output -- see DECISIONS). Intended to become
    // the default once a correctness gate covers it.
    bool resident = false;
    // KV cache quantization (--kv q4|q8|f16, default f16). Quality is
    // model-dependent and never assumed (Invariant 8): the owner measured
    // q4/q4 as near-lossless on Qwen3.8-27B; other models unmeasured.
    std::string kv_quant;
};

struct Engine;  // owns accountant, streamer, plan, metadata, model, context

// Opens the full streaming stack. On failure returns nullptr and sets *err.
// Prints the plan and integrity report to stderr exactly like cmd_run, because
// the operator of a server deserves the same honesty as the operator of a run.
Engine* engine_open(const EngineConfig& cfg, std::string* err);
void    engine_close(Engine* e);

struct GenParams {
    std::string prompt;
    int32_t     max_tokens  = 512;
    float       temperature = 0.8f;   // <= 0 means greedy
    // T10: stop sequences, matched against the decoded text as it grows; a hit
    // trims the text at the match and ends the generation with stop_hit set.
    std::vector<std::string> stop;
    // Resume: the TAIL of the prior sessions' text (longest stop minus one
    // byte suffices). Without it a stop straddling the resume boundary --
    // prefix generated before the crash, suffix after -- was invisible, which
    // falsified the "greedy jobs resume identical to an uninterrupted run"
    // guarantee. A match that begins inside this seed reports the overhang in
    // GenResult::base_trim so the caller can trim its persisted prefix
    // (2026-08-24 audit).
    std::string stop_seed;
    // F13: called at the top of EVERY prefill chunk with cumulative tokens
    // decoded so far -- the wire needs a sign of life during minutes-long
    // prefills or SDK read-timeouts retry whole generations. Optional.
    std::function<void(int32_t)> on_prefill_progress;
    // F5: cumulative tokens decoded by PRIOR sessions of a resumed job (0 for
    // fresh). on_safepoint's second argument is resume_tokens_done + the
    // session-local index, i.e. ALWAYS absolute.
    int32_t     resume_tokens_done = 0;
    uint32_t    seed        = 0;      // 0 maps to LLAMA_DEFAULT_SEED (llama's
                                      // "random"); pass a concrete seed for
                                      // reproducibility (R10: 0 is NOT random
                                      // by itself, it was a shared fixed seed)
    // Observer hooks, all optional. on_prefill fires with the prompt token count
    // before the prefill decode; on_token fires AFTER each token's decode and
    // abort check (post-decode so a readout ticking on it matches the verified
    // measurement path's timing exactly; the one-token latency cost to SSE is
    // the price of one semantics instead of two).
    std::function<void(int32_t)> on_prefill;
    // Fires after sampling a token and BEFORE its decode -- the exact instant a
    // snapshot has a well-defined resume shape (state excludes the token; the
    // token is the pending redecode). The job layer checkpoints here.
    std::function<void(int32_t next_token, int32_t tokens_done)> on_safepoint;
    // Resume mode: skip the clear/tokenize/prefill entirely; the caller has
    // restored a snapshot and supplies the pending token, which is decoded
    // first to regenerate logits. on_prefill does not fire.
    bool        resume = false;
    int32_t     resume_pending = 0;
    // Cooperative stop, checked between decodes (swarm S5): return false to end
    // the generation with r.cancelled set. Foreground streams wire this to the
    // sink's liveness; background jobs to their cancel flag. Null = run to end.
    std::function<bool()> should_continue;
};

struct GenResult {
    std::string text;
    int32_t     tokens_in = 0;    // prompt tokens (0 on resume: unknown here, T9)
    int32_t     tokens_out = 0;
    bool        aborted = false;       // a weight failed to materialise mid-run
    bool        bad_request = false;   // the client's ask was invalid (400, not 500)
    bool        cancelled = false;     // should_continue() said stop; output is a clean partial
    bool        stop_hit = false;      // a stop sequence matched (T10)
    // Bytes to remove from the END of the caller's persisted prefix when a stop
    // began inside stop_seed (resume straddle). 0 otherwise.
    size_t      base_trim = 0;
    bool        ctx_wall = false;      // J1: this sequence hit its private KV ring's end
    bool        truncated_by_eog = false;
    std::string error;                 // non-empty = request failed outright
};

// Runs one generation. token_cb (may be null) receives each detokenized piece as
// it is produced -- the streaming hook. Clears the model memory (KV/recurrent)
// first: requests are independent in v1; prefix reuse is a later feature.
GenResult engine_generate(Engine* e, const GenParams& p,
                          const std::function<void(const std::string&)>& token_cb);

// The raw context and vocab, for the persistence layer and its diagnostics:
// capture/restore work BETWEEN decodes on the live context by design.
struct llama_context* engine_ctx(Engine* e);
const llama_vocab*    engine_vocab(Engine* e);
// The model handle, for chat-template application in the server layer.
const llama_model* engine_model(Engine* e);

// Structured live counters for readouts and /health: byte totals by class,
// node count, failures, ledger-resident bytes, and the cap-honesty verdict.
// Read these only when no generation is running (or from the generating
// thread itself): the underlying counters are plain integers by design.
struct EngineCounters {
    uint64_t bytes_streamed = 0;   // total from disk
    uint64_t bytes_gather = 0;     // routed-expert share of the total
    // R7 residual: hit-rate inputs (h_routed = 1 - routed_read/routed_needed;
    // h_bytes over ALL active bytes).
    uint64_t routed_needed = 0;
    uint64_t uncond_needed = 0;
    uint64_t uncond_read = 0;
    uint64_t nodes = 0;
    uint64_t failures = 0;
    uint64_t resident_bytes = 0;   // accountant ledger, used()
    bool     over_cap = false;
};
EngineCounters engine_counters(Engine* e);
std::string engine_accountant_report(Engine* e);
// T6: for wiring SnapshotCache's ledger; checkpoint blobs must not bypass the cap.
dray::mem::Accountant* engine_accountant(Engine* e);
// G1: bytes of the standing checkpoint allowance (0 = none reserved).
uint64_t engine_checkpoint_allowance(Engine* e);
// T13: lock-free live generation state, safe from any thread at any time.
int32_t engine_live_tokens(Engine* e);
bool engine_live_over_cap(Engine* e);
std::string engine_plan_report(Engine* e);
// The streamer's honest report line, for /health.
std::string engine_report(Engine* e);
// Model identity for /v1/models.
std::string engine_model_id(Engine* e);
// True if any weight ever failed to materialise (output untrustworthy).
bool engine_tainted(Engine* e);
uint64_t engine_failures(Engine* e);

// Blocking server loop. Returns process exit code.
int serve_main(const EngineConfig& cfg, int port, const std::string& jobs_dir,
               const std::string& api_key = std::string());

// ---------------------------------------------------------------------------
// Batched decode (offline; owner-mandated production batching, phase 2).
// N sequences decoded in LOCKSTEP through the same streamer: the unconditional
// stream is read once per step for all of them, routed experts once per step
// for the UNION of selections -- the entire economic case for batching on this
// engine. The server stays serialized (Invariant 7); this API is for the
// `dray batch` subcommand. Sequences finish independently (EOG/stop/budget)
// and the batch shrinks; per-sequence seeds are seed+s, disclosed in results.
struct BatchParams {
    std::vector<std::string> prompts;   // one per sequence; size = batch size
    int32_t     max_tokens  = 512;
    float       temperature = 0.0f;     // <= 0 greedy (recommended for verify)
    uint32_t    seed        = 0;
    std::vector<std::string> stop;
    std::function<bool()> should_continue;
    // Fires after each lockstep decode with (step, live_sequences).
    std::function<void(int32_t, int32_t)> on_step;
};
struct BatchResult {
    std::vector<GenResult> seqs;   // one per input prompt, same order
    uint64_t steps = 0;            // lockstep decodes performed
    // J7: cumulative streamed bytes at the prefill/decode boundary, so the
    // caller can report decode-only figures -- the first benchmark divided
    // prefill-inclusive bytes by decode-only steps, inflating GB/step by a
    // term that grows with B.
    uint64_t prefill_end_bytes = 0;
    std::string error;             // batch-level failure (admission, decode)
};
BatchResult engine_generate_batch(Engine* e, const BatchParams& p);

// Cohort rotation (owner-ratified 2026-08-19): more logical sequences than
// the admission funds, served by rotating cohorts of the funded width. A
// cohort decodes for `span` tokens, parks its per-sequence KV/recurrent
// state to `state_dir`, and the next cohort loads. OPT-IN ONLY -- this is
// the engine's first sustained-write feature, and the state files may (and
// on endurance grounds should) live on a different device than the model.
// Rotation is pure scheduling: outputs must be byte-identical to unrotated
// runs of the same cohorts, and the oracle asserts exactly that. State
// files are working scratch, not checkpoints -- no fsync, no crash
// durability; a lost state means rerunning the batch.
struct RotateParams {
    std::vector<std::string> prompts;   // may exceed the funded width
    int32_t max_tokens = 64;
    float temperature = 0.0f;           // 0 = greedy
    uint32_t seed = 0;                  // per-seq: seed + GLOBAL index
    std::vector<std::string> stop;
    int32_t span = 64;                  // tokens per residency
    std::string state_dir;              // REQUIRED: parked-state directory
    std::function<bool()> should_continue;
    std::function<void(int32_t round, int32_t cohort, int32_t done)> on_round;
};
struct RotateResult {
    std::vector<GenResult> seqs;        // one per prompt, same order
    uint64_t rounds = 0;                // cohort residencies executed
    uint64_t state_bytes_written = 0;   // the honesty contract covers writes
    uint64_t state_bytes_read = 0;
    std::string error;
};
RotateResult engine_generate_rotated(Engine* e, const RotateParams& p);

}  // namespace dray::server
