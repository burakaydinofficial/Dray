#include "engine/scheduler.h"

#include <algorithm>
#include <string>
#include <utility>

namespace dray::engine {

Scheduler::Scheduler(Stepper& stepper, Config config, KvStore* store)
    : stepper_(stepper), store_(store), cfg_(config) {
    const int32_t n = std::max(1, stepper_.slots());
    for (int32_t i = 0; i < n; ++i) slots_.push_back(std::make_unique<Slot>(i));
}

Scheduler::~Scheduler() {
    stop();
    // Driven without the thread (tests), or stopped mid-work: nothing may be
    // left without its `done`.
    for (auto& s : slots_) {
        if (s->free()) continue;
        s->result().cancelled = true;
        finish(*s);
    }
    std::deque<Queued> left;
    {
        std::lock_guard<std::mutex> lk(mu_);
        left.swap(queue_);
    }
    for (Queued& q : left) {
        GenResult r;
        r.cancelled = true;
        if (q.request.done) q.request.done(std::move(r));
    }
}

uint64_t Scheduler::submit(Request request) {
    std::lock_guard<std::mutex> lk(mu_);
    const uint64_t id = next_id_++;
    ++in_flight_;
    queue_.push_back(Queued{id, std::move(request)});
    wake_.notify_one();
    return id;
}

void Scheduler::cancel(uint64_t id) {
    std::lock_guard<std::mutex> lk(mu_);
    cancels_.push_back(id);
    wake_.notify_one();
}

std::future<void> Scheduler::post(std::function<void()> fn) {
    std::lock_guard<std::mutex> lk(mu_);
    posted_.emplace_back(std::move(fn));
    std::future<void> f = posted_.back().get_future();
    wake_.notify_one();
    return f;
}

bool Scheduler::wait_idle(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(mu_);
    return idle_.wait_for(lk, timeout, [this] { return in_flight_ == 0; });
}

void Scheduler::start() {
    std::lock_guard<std::mutex> lk(mu_);
    if (running_) return;
    running_ = true;
    loop_ = std::thread([this] {
        for (;;) {
            {
                std::lock_guard<std::mutex> g(mu_);
                if (!running_) return;
            }
            if (step()) continue;
            std::unique_lock<std::mutex> g(mu_);
            wake_.wait(g, [this] {
                return !running_ || !queue_.empty() || !cancels_.empty() || !posted_.empty();
            });
        }
    });
}

void Scheduler::stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!running_) return;
        running_ = false;
        wake_.notify_all();
    }
    if (loop_.joinable()) loop_.join();
}

Scheduler::Stats Scheduler::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    Stats s = stats_;
    s.waiting = static_cast<int32_t>(queue_.size());
    return s;
}

void Scheduler::finish(Slot& slot) {
    if (store_) {
        // A clean or cancelled request leaves a KV worth keeping; a failed one
        // leaves nothing anyone should build on.
        const GenResult& r = slot.result();
        if (slot.known_history() && r.error.empty() && !r.aborted) {
            store_->retain(slot.index(), slot.kv_tokens());
        } else {
            store_->forget(slot.index());
        }
    }
    stepper_.close_slot(slot.index());
    slot.finish();
    retire();
}

void Scheduler::retire() {
    std::lock_guard<std::mutex> lk(mu_);
    if (--in_flight_ == 0) idle_.notify_all();
}

void Scheduler::admit() {
    // Cancellations first: a queued request never runs; an active one is
    // marked and ends at this step's boundary.
    std::vector<uint64_t> cancels;
    std::vector<Queued> dropped;
    {
        std::lock_guard<std::mutex> lk(mu_);
        cancels.swap(cancels_);
        for (uint64_t id : cancels) {
            auto it = std::find_if(queue_.begin(), queue_.end(),
                                   [id](const Queued& q) { return q.id == id; });
            if (it != queue_.end()) {
                dropped.push_back(std::move(*it));
                queue_.erase(it);
            }
        }
    }
    for (Queued& q : dropped) {
        GenResult r;
        r.cancelled = true;
        if (q.request.done) q.request.done(std::move(r));
        retire();
    }
    for (uint64_t id : cancels) {
        for (auto& s : slots_) {
            if (!s->free() && s->request_id() == id) s->request_cancel();
        }
    }

    // A free slot keeps taking requests until one of them runs: a refused
    // request (bad prompt, spent budget, abandoned restore) never holds it.
    for (;;) {
        std::vector<int32_t> free_slots;
        for (auto& s : slots_) {
            if (s->free()) free_slots.push_back(s->index());
        }
        if (free_slots.empty()) return;
        Queued q;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (queue_.empty()) return;
            q = std::move(queue_.front());
            queue_.pop_front();
        }
        GenParams& gp = q.request.params;
        auto refuse = [&](GenResult r) {
            if (q.request.done) q.request.done(std::move(r));
            retire();
        };

        // A job resuming from its checkpoint: any free slot, emptied, then
        // restored. Nothing kept there is reusable afterwards.
        if (q.request.restore) {
            const int32_t idx = free_slots.front();
            Slot& s = *slots_[static_cast<size_t>(idx)];
            if (store_) store_->forget(idx);
            stepper_.clear_kv(idx);
            const Restored how = q.request.restore(idx, gp);
            if (how == Restored::Abandon) {
                stepper_.clear_kv(idx);
                GenResult r;
                r.cancelled = true;
                refuse(std::move(r));
                continue;
            }
            if (how == Restored::Resumed) {
                // I3: a spent budget is decided BEFORE the pending token's decode
                // -- a clean finish ("length"), never one token past the bound.
                if (gp.max_tokens < 1) {
                    stepper_.clear_kv(idx);
                    refuse(GenResult{});
                    continue;
                }
                const int32_t n_past = stepper_.kv_tokens(idx);
                if (n_past + 1 > stepper_.context_per_slot()) {
                    stepper_.clear_kv(idx);
                    GenResult r;
                    r.error = "resume: context is full; nothing can be generated";
                    refuse(std::move(r));
                    continue;
                }
                stepper_.open_slot(idx, gp.temperature, gp.seed);
                s.begin_resumed(q.id, std::move(q.request), n_past);
                continue;
            }
            // Restored::Prefill: nothing to resume; the prompt below, in this slot.
            std::vector<int32_t> prompt;
            if (!stepper_.tokenize(gp.prompt, &prompt) || prompt.empty() ||
                static_cast<int32_t>(prompt.size()) >= stepper_.context_per_slot()) {
                GenResult r;
                r.error = "prompt could not be tokenized or does not fit the slot's context";
                refuse(std::move(r));
                continue;
            }
            const int32_t cp = store_ ? store_->checkpoint_position(static_cast<int32_t>(prompt.size()), 0) : -1;
            stepper_.open_slot(idx, gp.temperature, gp.seed);
            s.begin(q.id, std::move(q.request), std::move(prompt), 0, cp);
            continue;
        }

        std::vector<int32_t> prompt;
        GenResult bad;
        if (!stepper_.tokenize(gp.prompt, &prompt) || prompt.empty()) {
            bad.error = "prompt could not be tokenized";
        } else if (static_cast<int32_t>(prompt.size()) >= stepper_.context_per_slot()) {
            bad.bad_request = true;
            bad.error = "prompt of " + std::to_string(prompt.size()) +
                        " tokens does not fit this slot's context of " +
                        std::to_string(stepper_.context_per_slot()) + " tokens";
        }
        if (!bad.error.empty()) {
            refuse(std::move(bad));
            continue;
        }
        // The store picks the slot (the longest kept prefix of this prompt) and
        // leaves its KV holding exactly the reusable part; without a store, the
        // first free slot, emptied.
        KvStore::Placement pl;
        if (store_) {
            pl = store_->place(prompt, free_slots);
        }
        if (pl.slot < 0) {
            pl.slot = free_slots.front();
            pl.reuse = 0;
            stepper_.clear_kv(pl.slot);
        }
        if (pl.reuse > 0) {
            std::lock_guard<std::mutex> lk(mu_);
            ++stats_.reused_requests;
            stats_.reused_tokens += pl.reuse;
        }
        const int32_t cp = store_ ? store_->checkpoint_position(static_cast<int32_t>(prompt.size()), pl.reuse) : -1;
        stepper_.open_slot(pl.slot, gp.temperature, gp.seed);
        slots_[static_cast<size_t>(pl.slot)]->begin(q.id, std::move(q.request), std::move(prompt),
                                                    pl.reuse, cp);
    }
}

bool Scheduler::step() {
    // Posted work first: between two steps, where engine internals are quiescent.
    std::vector<std::packaged_task<void()>> posted;
    {
        std::lock_guard<std::mutex> lk(mu_);
        posted.swap(posted_);
    }
    for (auto& t : posted) t();

    admit();

    for (auto& s : slots_) {
        if (s->free() || !s->cancel_requested()) continue;
        // T1: a cancellation that can resume needs a checkpoint in the
        // resumable shape -- and at a step boundary a decoding slot IS in it:
        // its pending token is sampled, not yet decoded. Offered once more so
        // a draining shutdown loses no progress; never past the client's budget
        // (I3), where the resumed session would have nothing left to do.
        const Request& rq = s->request();
        if (s->state() == Slot::State::Decoding && rq.safepoint &&
            s->result().tokens_out < rq.params.max_tokens) {
            s->flush_text();
            rq.safepoint(s->index(), s->pending(), s->tokens_done());
        }
        s->result().cancelled = true;
        finish(*s);
    }

    // Checkpoints due: prefill paused at the position the store asked for, and
    // the KV now holds exactly the tokens fed so far.
    if (store_) {
        for (auto& s : slots_) {
            if (!s->checkpoint_due()) continue;
            store_->checkpoint(s->index(), s->kv_tokens());
            s->mark_checkpointed();
        }
    }

    // --- build: every decoding slot's pending token, then prompt tokens.
    enum class Kind : uint8_t { Decode, LastPrompt, Prompt };
    std::vector<Stepper::Entry> batch;
    std::vector<Kind> kind;
    const int32_t ctx = stepper_.context_per_slot();
    for (auto& s : slots_) {
        if (s->state() != Slot::State::Decoding) continue;
        if (s->position() + 1 > ctx) {
            s->result().ctx_wall = true;
            finish(*s);
            continue;
        }
        batch.push_back({s->index(), s->pending(), s->position(), true});
        kind.push_back(Kind::Decode);
    }
    const int32_t n_decode = static_cast<int32_t>(batch.size());
    // prefill_tokens_per_step bounds a step only to keep OTHER clients' decoding
    // responsive. With nobody decoding, a step takes as much prompt as it holds:
    // each step re-reads the experts the prompt routes to (nearly all of them on a
    // 512-expert model), so 17 steps of 256 read ~1 TB for a 4.4k-token prompt
    // where 3 of 2048 read a fifth of that (Flash Next, 2 x 600k server: 922 s).
    const int32_t per_step = n_decode > 0 ? cfg_.prefill_tokens_per_step : stepper_.max_step_tokens();
    int32_t budget = std::min(per_step, stepper_.max_step_tokens() - n_decode);
    const int32_t n_slots = static_cast<int32_t>(slots_.size());
    int32_t last_served = -1;
    for (int32_t k = 0; k < n_slots && budget > 0; ++k) {
        Slot& s = *slots_[static_cast<size_t>((next_prefill_ + k) % n_slots)];
        if (s.state() != Slot::State::Prefilling) continue;
        while (budget > 0 && s.prefill_remaining() > 0 && !s.checkpoint_due()) {
            int32_t tok = 0, pos = 0;
            const bool last = s.next_prompt_token(&tok, &pos);
            batch.push_back({s.index(), tok, pos, last});
            kind.push_back(last ? Kind::LastPrompt : Kind::Prompt);
            --budget;
        }
        last_served = s.index();
    }
    // Fairness: the next step's prefill starts after the slot served last.
    if (last_served >= 0) next_prefill_ = (last_served + 1) % n_slots;

    if (batch.empty()) {
        std::lock_guard<std::mutex> lk(mu_);
        stats_.active = 0;
        stats_.prefilling = 0;
        return !queue_.empty() || !posted_.empty();
    }

    // --- one decode for all of them.
    if (!stepper_.decode(batch)) {
        const bool tainted = stepper_.tainted();
        std::vector<bool> hit(slots_.size(), false);
        for (const auto& e : batch) hit[static_cast<size_t>(e.slot)] = true;
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (!hit[i] || slots_[i]->free()) continue;
            if (tainted) slots_[i]->result().aborted = true;
            else slots_[i]->result().error = "decode failed";
            finish(*slots_[i]);
        }
        return true;
    }

    // --- deliver: text for decoded tokens, then the next token for each.
    for (size_t row = 0; row < batch.size(); ++row) {
        if (kind[row] == Kind::Prompt) continue;
        Slot& s = *slots_[static_cast<size_t>(batch[row].slot)];
        if (kind[row] == Kind::Decode) {
            if (!s.emit(stepper_.piece(s.pending()))) {
                finish(s);
                continue;
            }
        }
        const int32_t next = stepper_.sample(s.index(), static_cast<int32_t>(row));
        if (stepper_.is_end(next)) {
            s.result().truncated_by_eog = true;
            finish(s);
            continue;
        }
        if (kind[row] == Kind::LastPrompt) s.start_decoding(next);
        else s.set_pending(next);
        // The safepoint: sampled, not yet decoded. T8: held-back UTF-8 goes to
        // the consumer first, or a checkpoint's text would lose those bytes.
        if (s.request().safepoint) {
            s.flush_text();
            s.request().safepoint(s.index(), next, s.tokens_done());
        }
    }

    int32_t active = 0, prefilling = 0;
    for (auto& s : slots_) {
        active += s->free() ? 0 : 1;
        prefilling += s->state() == Slot::State::Prefilling ? 1 : 0;
    }
    int64_t steps = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        steps = ++stats_.steps;
        stats_.decode_tokens += n_decode;
        stats_.prefill_tokens += static_cast<int64_t>(batch.size()) - n_decode;
        stats_.active = active;
        stats_.prefilling = prefilling;
    }
    stepper_.after_step(steps);
    return true;
}

}  // namespace dray::engine
