// Scheduler: continuous batching over a scripted Stepper. What each step
// carries (every decoding slot + a bounded share of prompt tokens), requests
// joining mid-stream without stalling the others, the queue, cancellation,
// context and token limits, stops, failures.

#include "harness.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "engine/ram_kv_store.h"
#include "engine/scheduler.h"
#include "engine/stepper.h"

using dray::engine::GenResult;
using dray::engine::Request;
using dray::engine::Scheduler;
using dray::engine::Stepper;

namespace {

constexpr int32_t kEnd = 0;

// One token per prompt byte. The answer is a function of the slot's KV alone:
// its non-capital tokens upper-cased, then kEnd, continuing after the capitals
// already there (tokens this fake generated earlier). So a request produces a
// known text whatever else runs, and a KV reused from a kept conversation must
// produce exactly what the same prompt produces from scratch.
class FakeStepper final : public Stepper {
public:
    FakeStepper(int32_t slots, int32_t ctx, int32_t step_tokens)
        : slots_(slots), ctx_(ctx), step_(step_tokens) {}

    int32_t slots() const override { return slots_; }
    int32_t context_per_slot() const override { return ctx_; }
    int32_t max_step_tokens() const override { return step_; }

    bool tokenize(const std::string& t, std::vector<int32_t>* out) override {
        out->clear();
        for (unsigned char c : t) out->push_back(c);
        return true;
    }
    std::string piece(int32_t token) override { return std::string(1, static_cast<char>(token)); }
    bool is_end(int32_t token) override { return token == kEnd; }

    void clear_kv(int32_t slot) override { seen[slot].clear(); cleared[slot]++; }
    int32_t kv_tokens(int32_t slot) override { return static_cast<int32_t>(seen[slot].size()); }
    // `recurrent`: like recurrent state, the KV cannot be cut back (only a
    // no-op cut succeeds); checkpoints save and restore it whole.
    bool truncate_kv(int32_t slot, int32_t n) override {
        std::vector<int32_t>& s = seen[slot];
        if (n >= static_cast<int32_t>(s.size())) return true;
        if (recurrent) return false;
        s.resize(static_cast<size_t>(n));
        return true;
    }
    bool needs_checkpoints() const override { return recurrent; }
    bool save_checkpoint(int32_t slot, std::vector<uint8_t>* out) override {
        const std::vector<int32_t>& s = seen[slot];
        out->assign(reinterpret_cast<const uint8_t*>(s.data()),
                    reinterpret_cast<const uint8_t*>(s.data() + s.size()));
        ++saves;
        return true;
    }
    bool load_checkpoint(int32_t slot, const std::vector<uint8_t>& in) override {
        std::vector<int32_t>& s = seen[slot];
        s.resize(in.size() / sizeof(int32_t));
        if (!in.empty()) std::memcpy(s.data(), in.data(), in.size());
        ++loads;
        return true;
    }
    // The whole state of the fake IS its KV, so saving it is a copy of `seen`.
    bool save_state(int32_t slot, std::vector<uint8_t>* out) override {
        const std::vector<int32_t>& s = seen[slot];
        out->assign(reinterpret_cast<const uint8_t*>(s.data()),
                    reinterpret_cast<const uint8_t*>(s.data() + s.size()));
        ++state_saves;
        return true;
    }
    bool load_state(int32_t slot, const std::vector<uint8_t>& in) override {
        std::vector<int32_t>& s = seen[slot];
        s.resize(in.size() / sizeof(int32_t));
        if (!in.empty()) std::memcpy(s.data(), in.data(), in.size());
        ++state_loads;
        return true;
    }
    void open_slot(int32_t slot, float, uint32_t) override {
        opened[slot]++;
        answer[slot].clear();   // the sampler starts afresh; the KV stays
    }
    void close_slot(int32_t slot) override { closed[slot]++; }

    bool decode(const std::vector<Entry>& batch) override {
        if (fail_next) { fail_next = false; return false; }
        steps.push_back(batch);
        for (const Entry& e : batch) {
            // Positions in a slot are consecutive from 0: the KV is private.
            if (e.pos != static_cast<int32_t>(seen[e.slot].size())) position_errors++;
            seen[e.slot].push_back(e.token);
        }
        return true;
    }
    int32_t sample(int32_t slot, int32_t row) override {
        const Entry& e = steps.back()[static_cast<size_t>(row)];
        if (!e.logits) row_errors++;
        if (answer[slot].empty()) {
            int32_t done = 0;
            for (int32_t t : seen[slot]) {
                if (t >= 'A' && t <= 'Z') { ++done; continue; }
                answer[slot].push_back(t >= 'a' && t <= 'z' ? t - 32 : t);
            }
            answer[slot].push_back(kEnd);
            next[slot] = done;
        }
        const size_t i = static_cast<size_t>(std::min<int32_t>(next[slot]++,
                                             static_cast<int32_t>(answer[slot].size()) - 1));
        return answer[slot][i];
    }
    bool tainted() const override { return taint; }

    int32_t slots_, ctx_, step_;
    bool fail_next = false, taint = false, recurrent = false;
    int position_errors = 0, row_errors = 0, saves = 0, loads = 0, state_saves = 0, state_loads = 0;
    std::vector<std::vector<Entry>> steps;
    std::map<int32_t, std::vector<int32_t>> seen, answer;
    std::map<int32_t, int32_t> next, opened, closed, cleared;
};

struct Outcome {
    bool done = false;
    GenResult r;
    std::string streamed;
};

Request make(const std::string& prompt, Outcome* o, int32_t max_tokens = 100) {
    Request q;
    q.params.prompt = prompt;
    q.params.max_tokens = max_tokens;
    q.params.temperature = 0;
    q.sink = [o](const std::string& p) { o->streamed += p; };
    q.done = [o](GenResult r) { o->done = true; o->r = std::move(r); };
    return q;
}

void drain(Scheduler& s, int limit = 1000) {
    for (int i = 0; i < limit && s.step(); ++i) {}
}

}  // namespace

LZ_TEST(every_request_gets_its_own_answer_while_sharing_steps) {
    FakeStepper st(4, 64, 64);
    Scheduler s(st, {16});
    Outcome a, b, c;
    s.submit(make("abc", &a));
    s.submit(make("hello", &b));
    s.submit(make("x", &c));
    drain(s);
    LZ_REQUIRE(a.done && b.done && c.done);
    LZ_CHECK_EQ(a.r.text, std::string("ABC"));
    LZ_CHECK_EQ(b.r.text, std::string("HELLO"));
    LZ_CHECK_EQ(c.r.text, std::string("X"));
    LZ_CHECK_EQ(a.streamed, a.r.text);            // the sink saw exactly the result
    LZ_CHECK(a.r.truncated_by_eog);
    LZ_CHECK_EQ(st.position_errors, 0);
    LZ_CHECK_EQ(st.row_errors, 0);
    // Shared steps: the three decoded together, not one after another.
    size_t max_width = 0;
    for (const auto& step : st.steps) max_width = std::max(max_width, step.size());
    LZ_CHECK_GE(max_width, 3u);
}

LZ_TEST(a_late_request_joins_without_stalling_the_running_ones) {
    FakeStepper st(2, 256, 64);
    Scheduler s(st, {4});                         // small prefill share per step
    Outcome a, b;
    s.submit(make("aaaaaaaaaaaaaaaaaaaa", &a));   // 20-token answer
    for (int i = 0; i < 8; ++i) s.step();         // a is decoding now
    LZ_REQUIRE(!a.done);
    s.submit(make(std::string(40, 'b'), &b));     // a long prompt arrives
    const size_t from = st.steps.size();
    s.step();
    // While b prefills, every step still carries a's decode token, and b's
    // prompt arrives at most 4 tokens per step.
    for (size_t k = from; k < st.steps.size(); ++k) {
        int a_tokens = 0, b_tokens = 0;
        for (const auto& e : st.steps[k]) (e.slot == 0 ? a_tokens : b_tokens)++;
        LZ_CHECK_EQ(a_tokens, 1);
        LZ_CHECK_LE(b_tokens, 4);
    }
    drain(s);
    LZ_CHECK_EQ(a.r.text, std::string(20, 'A'));
    LZ_CHECK_EQ(b.r.text, std::string(40, 'B'));
}

LZ_TEST(requests_beyond_the_slots_wait_and_then_run) {
    FakeStepper st(2, 64, 64);
    Scheduler s(st, {64});
    Outcome o[5];
    for (int i = 0; i < 5; ++i) s.submit(make(std::string(1, static_cast<char>('a' + i)), &o[i]));
    LZ_CHECK_EQ(s.stats().waiting, 5);
    s.step();
    LZ_CHECK_EQ(s.stats().waiting, 3);             // two admitted
    drain(s);
    for (int i = 0; i < 5; ++i) {
        LZ_CHECK(o[i].done);
        LZ_CHECK_EQ(o[i].r.text, std::string(1, static_cast<char>('A' + i)));
    }
    LZ_CHECK_EQ(st.opened[0] + st.opened[1], 5);   // slots reused, each opened afresh
    LZ_CHECK_EQ(st.closed[0] + st.closed[1], 5);   // and closed after each request
}

LZ_TEST(cancel_ends_a_running_request_and_drops_a_queued_one) {
    FakeStepper st(1, 64, 64);
    Scheduler s(st, {64});
    Outcome run, queued;
    const uint64_t a = s.submit(make(std::string(30, 'a'), &run));
    const uint64_t b = s.submit(make("b", &queued));
    for (int i = 0; i < 5; ++i) s.step();
    s.cancel(b);
    s.cancel(a);
    s.step();
    LZ_CHECK(run.done && run.r.cancelled);
    LZ_CHECK(queued.done && queued.r.cancelled);
    LZ_CHECK_EQ(queued.r.tokens_out, 0);           // it never ran
    LZ_CHECK(!s.step());                           // nothing left
}

LZ_TEST(limits_end_requests_cleanly) {
    FakeStepper st(2, 8, 64);
    Scheduler s(st, {64});
    Outcome too_long, capped, walled;
    s.submit(make("0123456789", &too_long));       // 10 tokens >= context 8
    s.submit(make("abc", &capped, 2));             // max_tokens 2
    s.submit(make("abcdef", &walled));             // 6-token prompt: hits the wall at 8
    drain(s);
    LZ_CHECK(too_long.done && too_long.r.bad_request);
    LZ_CHECK_EQ(capped.r.text, std::string("AB"));
    LZ_CHECK_EQ(capped.r.tokens_out, 2);
    LZ_CHECK(walled.r.ctx_wall);
    LZ_CHECK_EQ(walled.r.tokens_out, 2);           // positions 6 and 7, then the wall
}

LZ_TEST(stop_sequences_trim_and_end) {
    FakeStepper st(1, 64, 64);
    Scheduler s(st, {64});
    Outcome o;
    Request q = make("abcdef", &o);
    q.params.stop = {"CD"};
    s.submit(std::move(q));
    drain(s);
    LZ_CHECK(o.r.stop_hit);
    LZ_CHECK_EQ(o.r.text, std::string("AB"));
    // StreamedText's documented limit, the same on every path: a stop whose
    // prefix arrived in an EARLIER piece ("C", then "D") has already streamed
    // that prefix. The result is trimmed; the wire carries the prefix too.
    LZ_CHECK_EQ(o.streamed.substr(0, 2), std::string("AB"));
}

LZ_TEST(a_failed_step_fails_only_the_requests_in_it) {
    FakeStepper st(2, 64, 64);
    Scheduler s(st, {64});
    Outcome a, b;
    s.submit(make("abc", &a));
    s.step();                                       // a prefilled
    st.fail_next = true;
    st.taint = true;
    s.step();                                       // a's decode step fails
    LZ_CHECK(a.done && a.r.aborted);
    s.submit(make("xy", &b));                       // the engine keeps serving
    st.taint = false;
    drain(s);
    LZ_CHECK_EQ(b.r.text, std::string("XY"));
}

LZ_TEST(prefill_is_shared_fairly_among_new_requests_while_another_decodes) {
    FakeStepper st(4, 256, 64);
    Scheduler s(st, {6});
    Outcome d, o[3];
    s.submit(make("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzz", &d));   // decoding while the others prefill
    s.step();                                                // d's prompt, whole: nobody decodes yet
    LZ_REQUIRE(!d.done);
    for (int i = 0; i < 3; ++i) s.submit(make(std::string(12, static_cast<char>('a' + i)), &o[i]));
    const size_t first = st.steps.size();
    s.step();
    s.step();
    // With a client decoding, prefill is capped at 6 tokens a step, round-robin:
    // no new slot got all 12 while another got none.
    std::map<int32_t, int> fed;
    for (size_t k = first; k < first + 2; ++k) {
        for (const auto& e : st.steps[k]) if (e.slot != 0) fed[e.slot]++;
    }
    LZ_CHECK_EQ(fed[1] + fed[2] + fed[3], 12);
    LZ_CHECK(fed[1] < 12 || fed[2] > 0);
    drain(s);
    for (int i = 0; i < 3; ++i) LZ_CHECK_EQ(o[i].r.text, std::string(12, static_cast<char>('A' + i)));
}

LZ_TEST(with_nobody_decoding_a_step_takes_as_much_prompt_as_it_holds) {
    // prefill_tokens_per_step only protects other clients' decoding: alone, a
    // prompt goes in steps of the full step size (fewer weight re-reads).
    FakeStepper st(2, 256, 64);
    Scheduler s(st, {6});
    Outcome o;
    s.submit(make(std::string(40, 'q'), &o));
    s.step();
    LZ_REQUIRE_EQ(st.steps.size(), 1u);
    LZ_CHECK_EQ(st.steps[0].size(), 40u);   // not 6
    drain(s);
    LZ_CHECK_EQ(o.r.text, std::string(40, 'Q'));
}

LZ_TEST(the_loop_thread_serves_concurrent_submitters) {
    FakeStepper st(3, 64, 64);
    Scheduler s(st, {16});
    s.start();
    std::vector<Outcome> o(8);
    std::mutex m;
    std::condition_variable cv;
    int finished = 0;
    for (int i = 0; i < 8; ++i) {
        Request q;
        q.params.prompt = std::string(3 + i, static_cast<char>('a' + i));
        q.params.max_tokens = 100;
        q.params.temperature = 0;
        Outcome* out = &o[static_cast<size_t>(i)];
        q.done = [out, &m, &cv, &finished](GenResult r) {
            std::lock_guard<std::mutex> lk(m);
            out->r = std::move(r);
            out->done = true;
            ++finished;
            cv.notify_all();
        };
        s.submit(std::move(q));
    }
    {
        std::unique_lock<std::mutex> lk(m);
        cv.wait_for(lk, std::chrono::seconds(10), [&] { return finished == 8; });
    }
    s.stop();
    LZ_REQUIRE_EQ(finished, 8);
    for (int i = 0; i < 8; ++i) {
        LZ_CHECK_EQ(o[static_cast<size_t>(i)].r.text, std::string(3 + i, static_cast<char>('A' + i)));
    }
}

// --- the job hooks: safepoints, cancel's final safepoint, restore -----------

namespace {

// What a checkpoint would hold in the fake: the slot's KV and its answer script.
struct Snapshot {
    int32_t next_token = 0;
    int32_t done = 0;
    std::vector<int32_t> seen;   // the fake's KV: all a checkpoint needs
    std::string text;
};

}  // namespace

LZ_TEST(a_cancelled_job_resumes_from_its_last_safepoint_to_the_same_text) {
    FakeStepper st(2, 128, 64);
    Scheduler s(st, {64});
    // Uninterrupted reference.
    Outcome ref;
    s.submit(make("abcdefgh", &ref));
    drain(s);
    LZ_REQUIRE_EQ(ref.r.text, std::string("ABCDEFGH"));

    // Session 1: checkpoint at every safepoint, cancel after a few steps.
    Outcome one;
    Snapshot snap;
    std::vector<int32_t> dones;
    bool draining = false;
    Request q = make("abcdefgh", &one);
    // Like the job layer: a checkpoint every 4 tokens, or whenever draining.
    q.safepoint = [&](int32_t seq, int32_t next_tok, int32_t done) {
        dones.push_back(done);
        if (done % 4 != 0 && !draining) return;
        snap.next_token = next_tok;
        snap.done = done;
        snap.seen = st.seen[seq];
        snap.text = one.streamed;
    };
    const uint64_t id = s.submit(std::move(q));
    for (int i = 0; i < 6; ++i) s.step();
    draining = true;
    s.cancel(id);
    s.step();
    LZ_REQUIRE(one.done && one.r.cancelled);
    LZ_REQUIRE(one.r.tokens_out % 4 != 0);   // so only the final safepoint can be current
    // Safepoints carry the absolute count, from 0 (the first sampled token).
    LZ_CHECK_EQ(dones.front(), 0);
    for (size_t i = 1; i < dones.size(); ++i) LZ_CHECK_GE(dones[i], dones[i - 1]);
    LZ_CHECK_EQ(snap.done, one.r.tokens_out);   // cancel's final safepoint is the latest state

    // Session 2: restore the snapshot into whatever slot it gets, and finish.
    Outcome two;
    Request r = make("abcdefgh", &two, 100);
    r.restore = [&](int32_t seq, dray::engine::GenParams& gp) {
        st.seen[seq] = snap.seen;
        gp.resume_pending = snap.next_token;
        gp.resume_tokens_done = snap.done;
        gp.max_tokens = 100 - snap.done;
        return dray::engine::Restored::Resumed;
    };
    s.submit(std::move(r));
    drain(s);
    LZ_REQUIRE(two.done);
    LZ_CHECK_EQ(snap.text + two.r.text, ref.r.text);   // the same text, uninterrupted
    LZ_CHECK_EQ(two.r.tokens_in, 0);                   // unknown on a resume
    LZ_CHECK_EQ(st.position_errors, 0);                // positions continue from the restore
}

LZ_TEST(no_final_safepoint_past_the_budget) {
    FakeStepper st(1, 128, 64);
    Scheduler s(st, {64});
    Outcome o;
    int after_cancel = 0;
    bool cancelling = false;
    Request q = make("abcdefgh", &o, 3);
    q.safepoint = [&](int32_t, int32_t, int32_t) { if (cancelling) ++after_cancel; };
    const uint64_t id = s.submit(std::move(q));
    // Run until 3 tokens are out (the budget): the request finishes by itself;
    // a cancel arriving afterwards finds nothing to checkpoint.
    drain(s);
    cancelling = true;
    s.cancel(id);
    s.step();
    LZ_CHECK_EQ(o.r.tokens_out, 3);
    LZ_CHECK_EQ(after_cancel, 0);
}

LZ_TEST(restore_outcomes_never_hold_a_slot) {
    FakeStepper st(1, 128, 64);
    Scheduler s(st, {64});
    Outcome abandoned, spent, after;
    Request a = make("abc", &abandoned);
    a.restore = [](int32_t, dray::engine::GenParams&) { return dray::engine::Restored::Abandon; };
    Request b = make("abc", &spent);
    b.restore = [](int32_t, dray::engine::GenParams& gp) {
        gp.max_tokens = 0;   // the budget was spent exactly at the interruption
        return dray::engine::Restored::Resumed;
    };
    s.submit(std::move(a));
    s.submit(std::move(b));
    s.submit(make("xyz", &after));
    drain(s);
    LZ_CHECK(abandoned.done && abandoned.r.cancelled);
    LZ_CHECK(spent.done && spent.r.error.empty() && !spent.r.cancelled);   // a clean finish
    LZ_CHECK_EQ(spent.r.tokens_out, 0);
    LZ_CHECK_EQ(after.r.text, std::string("XYZ"));   // the one slot was free for it
}

LZ_TEST(posted_work_runs_on_the_loop_thread_between_steps_busy_or_idle) {
    FakeStepper st(2, 128, 64);
    Scheduler s(st, {64});
    s.start();
    // Idle: runs promptly.
    std::thread::id where{};
    auto f1 = s.post([&] { where = std::this_thread::get_id(); });
    LZ_REQUIRE(f1.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    LZ_CHECK(where != std::this_thread::get_id());
    // Busy: runs between steps, never inside one (the fake counts decodes).
    Outcome a;
    s.submit(make(std::string(40, 'a'), &a));
    size_t seen_steps = 0;
    auto f2 = s.post([&] { seen_steps = st.steps.size(); });
    LZ_REQUIRE(f2.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    LZ_CHECK(s.wait_idle(std::chrono::seconds(10)));
    s.stop();
    LZ_CHECK(a.done);
    LZ_CHECK_LE(seen_steps, st.steps.size());
}

LZ_TEST(wait_idle_waits_for_queued_and_active_requests) {
    FakeStepper st(1, 128, 64);
    Scheduler s(st, {64});
    Outcome o[3];
    for (int i = 0; i < 3; ++i) s.submit(make("abcdef", &o[i]));
    LZ_CHECK(!s.wait_idle(std::chrono::milliseconds(20)));   // nothing is running them yet
    s.start();
    LZ_CHECK(s.wait_idle(std::chrono::seconds(10)));
    s.stop();
    for (auto& x : o) LZ_CHECK(x.done);
}

// --- conversation reuse (RamKvStore) -------------------------------------------

namespace {

std::string run_alone(const std::string& prompt, bool recurrent) {
    FakeStepper st(1, 256, 64);
    st.recurrent = recurrent;
    Scheduler s(st, {64});
    Outcome o;
    s.submit(make(prompt, &o));
    drain(s);
    return o.r.text;
}

int prompt_tokens_fed(const FakeStepper& st, size_t from_step) {
    int n = 0;
    for (size_t k = from_step; k < st.steps.size(); ++k) {
        for (const auto& e : st.steps[k]) n += e.logits ? 0 : 1;   // prompt tokens, except the last
    }
    return n;
}

}  // namespace

using dray::engine::RamKvStore;

LZ_TEST(a_next_turn_reuses_the_kept_conversation_and_answers_the_same) {
    FakeStepper st(2, 256, 64);
    RamKvStore::Config kc;
    kc.checkpoints_per_slot = 0;              // attention only: truncation is enough
    RamKvStore store(st, kc);
    Scheduler s(st, {64}, &store);
    Outcome one, two;
    s.submit(make("abc", &one));
    drain(s);
    LZ_REQUIRE_EQ(one.r.text, std::string("ABC"));
    // The client sends the whole conversation back, plus a new message.
    const size_t from = st.steps.size();
    s.submit(make("abcABCde", &two));
    drain(s);
    LZ_CHECK_EQ(two.r.text, run_alone("abcABCde", false));   // same as from scratch
    LZ_CHECK_EQ(store.stats().reused_tokens, 6);             // "abcABC" was already there
    LZ_CHECK_EQ(prompt_tokens_fed(st, from), 1);             // only "d" (and "e" with logits)
    LZ_CHECK_EQ(st.position_errors, 0);
}

LZ_TEST(recurrent_state_is_restored_from_a_checkpoint_before_the_divergence) {
    FakeStepper st(1, 256, 64);
    st.recurrent = true;
    RamKvStore::Config kc;
    kc.checkpoints_per_slot = 2;
    kc.checkpoint_offset = 4;
    RamKvStore store(st, kc);
    Scheduler s(st, {64}, &store);
    Outcome one, two;
    s.submit(make("abcdefghij", &one));
    drain(s);
    LZ_REQUIRE_EQ(one.r.text, std::string("ABCDEFGHIJ"));
    LZ_CHECK_EQ(store.stats().checkpoints_taken, 1);   // at 10 - 4 = 6 prompt tokens
    // The client changed the assistant's text: the conversations share 15
    // tokens, but recurrent state cannot be cut back to 15 -- the checkpoint at
    // 6 is the newest usable point.
    s.submit(make("abcdefghijABCDExy", &two));
    drain(s);
    LZ_CHECK_EQ(two.r.text, run_alone("abcdefghijABCDExy", true));
    LZ_CHECK_EQ(store.stats().checkpoints_restored, 1);
    LZ_CHECK_EQ(store.stats().reused_tokens, 6);
    LZ_CHECK_EQ(st.position_errors, 0);
}

LZ_TEST(an_exact_continuation_needs_no_checkpoint_even_when_recurrent) {
    FakeStepper st(1, 256, 64);
    st.recurrent = true;
    RamKvStore store(st, {});
    Scheduler s(st, {64}, &store);
    Outcome one, two;
    s.submit(make("abcdefghij", &one));
    drain(s);
    s.submit(make("abcdefghijABCDEFGHIJxy", &two));   // the kept conversation, extended
    drain(s);
    LZ_CHECK_EQ(two.r.text, run_alone("abcdefghijABCDEFGHIJxy", true));
    LZ_CHECK_EQ(store.stats().reused_tokens, 20);      // all of it: nothing to cut
    LZ_CHECK_EQ(store.stats().checkpoints_restored, 0);
}

LZ_TEST(a_prompt_goes_to_the_slot_holding_its_conversation) {
    FakeStepper st(2, 256, 64);
    RamKvStore::Config kc;
    kc.checkpoints_per_slot = 0;
    RamKvStore store(st, kc);
    Scheduler s(st, {64}, &store);
    Outcome a, b, b2;
    s.submit(make("aaaa", &a));
    s.submit(make("bbbb", &b));
    drain(s);
    s.submit(make("bbbbBBBBcc", &b2));
    drain(s);
    LZ_CHECK_EQ(b2.r.text, run_alone("bbbbBBBBcc", false));
    LZ_CHECK_EQ(store.stats().reused_tokens, 8);
    // And the other conversation is still kept, in the other slot.
    Outcome a2;
    s.submit(make("aaaaAAAAdd", &a2));
    drain(s);
    LZ_CHECK_EQ(store.stats().reused_tokens, 16);
}

LZ_TEST(idle_conversations_expire_and_failed_ones_are_forgotten) {
    FakeStepper st(1, 256, 64);
    int64_t clock = 0;
    RamKvStore::Config kc;
    kc.checkpoints_per_slot = 0;
    kc.idle_timeout = std::chrono::seconds(60);
    kc.clock = [&clock] { return clock; };
    RamKvStore store(st, kc);
    Scheduler s(st, {64}, &store);
    Outcome one, late;
    s.submit(make("abc", &one));
    drain(s);
    clock += 61'000;                                    // past the idle timeout
    s.submit(make("abcABCde", &late));
    drain(s);
    LZ_CHECK_EQ(store.stats().reused_tokens, 0);        // expired: prefilled afresh
    LZ_CHECK_EQ(late.r.text, run_alone("abcABCde", false));

    // A failed step leaves nothing to build on.
    Outcome bad, after;
    s.submit(make("xyz", &bad));
    s.step();
    st.fail_next = true;
    s.step();
    LZ_REQUIRE(bad.done && !bad.r.error.empty());
    s.submit(make("xyzXYZw", &after));
    drain(s);
    LZ_CHECK_EQ(store.stats().reused_tokens, 0);
}

LZ_TEST(prefill_pauses_exactly_at_the_checkpoint_position) {
    FakeStepper st(1, 256, 64);
    st.recurrent = true;
    RamKvStore::Config kc;
    kc.checkpoint_offset = 4;
    RamKvStore store(st, kc);
    Scheduler s(st, {64}, &store);
    Outcome o;
    s.submit(make("abcdefghijkl", &o));   // 12 tokens: checkpoint after 8
    s.step();
    LZ_REQUIRE_EQ(st.steps.size(), 1u);
    LZ_CHECK_EQ(st.steps[0].size(), 8u);   // the first step stops at the checkpoint
    s.step();
    LZ_CHECK_EQ(st.saves, 1);              // taken at the boundary, holding 8 tokens
    drain(s);
    LZ_CHECK_EQ(o.r.text, std::string("ABCDEFGHIJKL"));
}

LZ_TEST(the_same_prompt_again_still_feeds_its_last_token) {
    // The kept conversation covers the whole prompt: one prompt token must
    // still be fed, or there are no logits for the first answer token.
    FakeStepper st(1, 256, 64);
    RamKvStore::Config kc;
    kc.checkpoints_per_slot = 0;
    RamKvStore store(st, kc);
    Scheduler s(st, {64}, &store);
    Outcome one, again;
    s.submit(make("abc", &one));
    drain(s);
    s.submit(make("abc", &again));
    for (int i = 0; i < 50 && !again.done; ++i) s.step();
    LZ_REQUIRE(again.done);
    LZ_CHECK_EQ(again.r.text, std::string("ABC"));
    LZ_CHECK_EQ(store.stats().reused_tokens, 2);
}

LZ_TEST(only_checkpoints_before_the_divergence_and_still_valid_are_restored) {
    FakeStepper st(1, 256, 64);
    st.recurrent = true;
    RamKvStore::Config kc;
    kc.checkpoints_per_slot = 4;
    kc.checkpoint_offset = 4;
    RamKvStore store(st, kc);
    Scheduler s(st, {64}, &store);
    auto turn = [&](const std::string& p) {
        Outcome o;
        s.submit(make(p, &o));
        drain(s);
        LZ_CHECK_EQ(o.r.text, run_alone(p, true));
        return o.r.text;
    };
    turn("abcdefghij");                          // checkpoint at 6
    turn("abcdefghijABCDEFGHIJklmnopqrst");       // exact continuation; checkpoint at 26
    // Diverges at 15: the checkpoint at 26 is past it, the one at 6 is not.
    const std::string r3 = "abcdefghijABCDExyzuvwxyzuvwxyzuv";   // checkpoint at 28
    turn(r3);
    // Diverges at 27 of r3's own history. The checkpoint at 26 belonged to the
    // conversation r3 replaced -- restoring it would be the wrong state. The
    // only valid one at or before 27 is at 6.
    turn(r3.substr(0, 27) + "qq");
    LZ_CHECK_EQ(st.position_errors, 0);
}

// --- the pool of parked conversations -------------------------------------------

namespace {

// A ledger with a limit, to see what the pool charges and that it pays back.
struct FakeMemory final : dray::engine::MemoryCharge {
    uint64_t limit = UINT64_MAX, used = 0;
    bool charge(uint64_t b) override {
        if (used + b > limit) return false;
        used += b;
        return true;
    }
    void refund(uint64_t b) override { used -= b; }
};

std::string turn(Scheduler& s, const std::string& prompt) {
    Outcome o;
    s.submit(make(prompt, &o));
    drain(s);
    return o.r.text;
}

}  // namespace

LZ_TEST(conversations_outlive_their_slot_in_the_pool) {
    // One slot, two alternating conversations: each one's next turn finds its
    // own conversation again, parked while the other one used the slot.
    FakeStepper st(1, 256, 64);
    RamKvStore::Config kc;
    kc.checkpoints_per_slot = 0;
    kc.pool_max_tokens = 1000;
    FakeMemory mem;
    kc.memory = &mem;
    RamKvStore store(st, kc);
    Scheduler s(st, {64}, &store);
    turn(s, "aaaa");
    turn(s, "bbbb");                                     // parks "aaaaAAAA"
    LZ_CHECK_EQ(store.stats().parked, 1);
    const std::string a2 = turn(s, "aaaaAAAAxy");        // restored from the pool
    LZ_CHECK_EQ(a2, run_alone("aaaaAAAAxy", false));
    LZ_CHECK_EQ(store.stats().restored, 1);
    LZ_CHECK_EQ(store.stats().reused_tokens, 8);
    const std::string b2 = turn(s, "bbbbBBBBzz");        // and B from the pool in turn
    LZ_CHECK_EQ(b2, run_alone("bbbbBBBBzz", false));
    LZ_CHECK_EQ(store.stats().restored, 2);
    LZ_CHECK_EQ(st.position_errors, 0);
    // What left the pool was paid back: the ledger holds exactly what is parked.
    LZ_CHECK_EQ(mem.used, store.stats().pool_bytes);
}

LZ_TEST(a_shared_system_prompt_no_longer_destroys_the_other_conversation) {
    FakeStepper st(1, 256, 64);
    RamKvStore::Config kc;
    kc.checkpoints_per_slot = 0;
    kc.pool_max_tokens = 1000;
    RamKvStore store(st, kc);
    Scheduler s(st, {64}, &store);
    turn(s, "sysaaa");
    turn(s, "sysbbb");            // matches "sys" only: a full copy of A is parked first
    const std::string a2 = turn(s, "sysaaaSYSAAAq");
    LZ_CHECK_EQ(a2, run_alone("sysaaaSYSAAAq", false));
    LZ_CHECK_EQ(store.stats().reused_tokens, 3 + 12);   // "sys" for B, all of A back
}

LZ_TEST(the_pool_keeps_to_its_token_budget_and_its_memory) {
    FakeStepper st(1, 256, 64);
    FakeMemory mem;
    RamKvStore::Config kc;
    kc.checkpoints_per_slot = 0;
    kc.pool_max_tokens = 10;      // one 8-token conversation, not two
    kc.memory = &mem;
    {
        RamKvStore store(st, kc);
        Scheduler s(st, {64}, &store);
        turn(s, "aaaa");
        turn(s, "bbbb");          // parks A (8 tokens)
        turn(s, "cccc");          // parks B: A is evicted for the budget
        LZ_CHECK_EQ(store.stats().pool_entries, 1);
        LZ_CHECK_LE(store.stats().pool_tokens, 10);
        LZ_CHECK_GE(store.stats().pool_evicted, 1);
        LZ_CHECK_EQ(mem.used, store.stats().pool_bytes);   // charged exactly what it holds
        // A ledger with no room: nothing is parked, nothing breaks.
        mem.limit = mem.used;
        turn(s, "dddd");          // would park C: refused, the oldest goes instead
        LZ_CHECK_LE(mem.used, mem.limit);
    }
    LZ_CHECK_EQ(mem.used, 0u);    // everything paid back when the store goes
}

LZ_TEST(parked_conversations_expire) {
    FakeStepper st(1, 256, 64);
    int64_t clock = 0;
    RamKvStore::Config kc;
    kc.checkpoints_per_slot = 0;
    kc.pool_max_tokens = 1000;
    kc.idle_timeout = std::chrono::seconds(60);
    kc.clock = [&clock] { return clock; };
    RamKvStore store(st, kc);
    Scheduler s(st, {64}, &store);
    turn(s, "aaaa");
    turn(s, "bbbb");
    LZ_CHECK_EQ(store.stats().pool_entries, 1);
    clock += 61'000;
    turn(s, "aaaaAAAAxy");        // too late: A expired, prefilled from scratch
    LZ_CHECK_EQ(store.stats().restored, 0);
    LZ_CHECK_EQ(store.stats().pool_evicted, 1);
}

LZ_TEST(a_recurrent_conversation_comes_back_with_its_checkpoints) {
    FakeStepper st(1, 256, 64);
    st.recurrent = true;
    RamKvStore::Config kc;
    kc.checkpoints_per_slot = 2;
    kc.checkpoint_offset = 4;
    kc.pool_max_tokens = 1000;
    RamKvStore store(st, kc);
    Scheduler s(st, {64}, &store);
    turn(s, "abcdefghij");        // checkpoint at 6
    turn(s, "zzzzzzzz");          // parks A with its checkpoint
    // A diverges inside its old answer: restored whole, then cut back via the
    // checkpoint that came back with it.
    const std::string a2 = turn(s, "abcdefghijABCDExy");
    LZ_CHECK_EQ(a2, run_alone("abcdefghijABCDExy", true));
    LZ_CHECK_EQ(store.stats().restored, 1);
    LZ_CHECK_GE(store.stats().checkpoints_restored, 1);
    LZ_CHECK_EQ(st.position_errors, 0);
}
