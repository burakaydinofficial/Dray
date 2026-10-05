#include "backend/expert_compactor.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace dray::backend {


namespace {

void log_region_leak() {
    std::fprintf(stderr, "[dray] FATAL: backend dead with region reads outstanding; "
                         "leaking the region (no-cancel contract)\n");
}

}  // namespace

// ---------------------------------------------------------------------------
// row slicing (GET_ROWS)

// ggml_get_rows indexes by the id VALUE and does not range-check against ne[1]
// at compute time, so a compact buffer of k rows addressed by ids in 0..k-1 is
// in bounds by construction.
bool ExpertCompactor::compact_rows(ggml_tensor* node, ggml_tensor* w, ggml_tensor* ids,
                                   uint64_t* streamed) {
    if (!w || !ids || ids->type != GGML_TYPE_I32) return false;

    const Source* src = p_.tensors.source_of(w);
    if (!src) return false;                            // not ours
    if (w->ne[1] <= 1 || w->ne[2] > 1) return false;   // a plain 2-D table only

    const int64_t n_rows  = w->ne[1];
    const uint64_t stride = static_cast<uint64_t>(w->nb[1]);
    if (stride == 0) return false;

    std::vector<int32_t> uniq, remapped;
    if (!derive_ids_map(ids, n_rows, &uniq, &remapped)) return false;

    // Not worth it if the "slice" is most of the table: fall back to the whole
    // tensor, which is cacheable and reusable across tokens.
    if (static_cast<int64_t>(uniq.size()) * 4 >= n_rows) return false;

    if (const Resident* ex = p_.cache.find(w)) {
        if (ex->uniq == uniq) {
            w->data = ex->mem;
            p_.ids.install(node, ids, remapped, 1);
            return true;
        }
        p_.cache.drop(w);
    }

    const uint64_t need = static_cast<uint64_t>(uniq.size()) * stride;
    if (!p_.cache.make_room(need)) return false;

    uint8_t* mem = static_cast<uint8_t*>(
        p_.mem.alloc(mem::Category::ExpertCache, need, kHostAlign));
    if (!mem) return false;

    std::vector<Slice> slices(uniq.size());
    for (size_t k = 0; k < uniq.size(); ++k) {
        slices[k].src = *src;
        slices[k].src.offset += static_cast<uint64_t>(uniq[k]) * stride;
        slices[k].dst = mem + k * stride;
        slices[k].len = stride;
    }
    if (!p_.io.read_batch(slices)) {
        p_.mem.free(mem::Category::ExpertCache, mem, need);
        return false;
    }
    *streamed += need;

    Resident nr; nr.mem = mem; nr.bytes = need; nr.pinned = false;
    nr.prio = Prio::RoutedExpert;   // valid only for this token's ids, like a compact
    nr.uniq = uniq;
    p_.cache.add(w, nr);
    w->data = mem;
    p_.ids.install(node, ids, remapped, 1);
    return true;
}

// ---------------------------------------------------------------------------
// Phase A / Phase C: sibling regions, submitted unwaited

ExpertCompactor::RegionGeom ExpertCompactor::region_geom(const Source& src, uint64_t stride,
                                                         uint64_t need) const {
    const uint32_t a = p_.io.align();
    const uint64_t dstride = src.disk_stride ? src.disk_stride : stride;
    RegionGeom g;
    g.head = (a && dstride % a == 0) ? static_cast<uint32_t>(src.offset % a) : 0;
    g.bytes = need + (g.head ? a : 0);
    g.align = g.head ? a : kHostAlign;
    return g;
}

bool ExpertCompactor::alloc_region(const Source& src, uint64_t stride, uint64_t need,
                                   uint8_t** mem, uint64_t* bytes, uint32_t* head, void* spare) {
    const RegionGeom g = region_geom(src, stride, need);
    *head = g.head;
    *bytes = g.bytes;
    const uint32_t align = g.align;
    // The tensor's own previous region, already the right size: the common
    // decode case (k experts every token), and the one that used to free a
    // region and commit a fresh one per tensor per token -- 16% of Qwen3.8-
    // Flash-Next decode at 36 GiB, almost all of it OS zero-fill on first touch.
    if (spare) {
        *mem = static_cast<uint8_t*>(spare);
        return true;
    }
    // A dead region of the same size first: its pages are committed already, and
    // its ledger charge simply carries over (see ResidencyCache::take_region).
    // ONLY when a fresh allocation would have to evict: then the swap is that
    // eviction with the free and the alloc skipped. Taking a region while there
    // is free room throws away a cached region for nothing (fewer reuse hits; the
    // pressure test caught it as a smaller cache at its rebudget). And not when
    // usage is already over budget (after a rebudget shrink): a same-size swap
    // cannot bring it down, so make_room below must evict first, as before.
    const uint64_t used = p_.mem.cache_used(), budget = p_.mem.cache_budget();
    if (used <= budget && used + *bytes > budget &&
        (*mem = static_cast<uint8_t*>(p_.cache.take_region(*bytes, align))) != nullptr) {
        return true;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const bool room = p_.cache.make_room(*bytes);
    const auto t1 = std::chrono::steady_clock::now();
    ns_room_ += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    if (!room) return false;
    *mem = static_cast<uint8_t*>(
        p_.mem.alloc(mem::Category::ExpertCache, *bytes, align));
    ns_alloc_ += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t1).count());
    return *mem != nullptr;
}

void ExpertCompactor::append_timing(std::ostream& o) const {
    o << ", phase C " << ns_unlock_ / 1000000 << "ms over " << early_unlocks_
      << " unlocks (region room " << ns_room_ / 1000000 << "ms, alloc " << ns_alloc_ / 1000000
      << "ms, both paths)";
}

// Any refusal (no room, queue full, alloc fail) just means the sibling reads
// synchronously when its own node arrives -- never an error, only lost overlap.
void ExpertCompactor::submit_sibling_region(ggml_tensor* w, const std::vector<int32_t>& uniq) {
    if (!w || pending_.count(w)) return;
    const Source* src = p_.tensors.source_of(w);
    if (!src || w->ne[2] <= 1) return;
    const Resident* ex = p_.cache.find(w);
    if (ex && ex->uniq == uniq) return;   // reusable as is

    // Strides DIFFER across gate/up/down -- always this tensor's own nb[2].
    const uint64_t stride = static_cast<uint64_t>(w->nb[2]);
    const uint64_t need   = static_cast<uint64_t>(uniq.size()) * stride;
    // The old region holds the previous token's routing for this tensor, which
    // this routing replaces; nothing computes from it now (the siblings are
    // ahead of their nodes). Reclaim it rather than hold both.
    void* spare = nullptr;
    if (ex) {
        const RegionGeom g = region_geom(*src, stride, need);
        spare = p_.cache.reclaim_own(w, g.bytes, g.align, ResidencyCache::Mismatch::Keep);
    }
    uint8_t* mem = nullptr;
    uint64_t alloc_bytes = 0;
    uint32_t head = 0;
    if (!alloc_region(*src, stride, need, &mem, &alloc_bytes, &head, spare)) return;
    uint8_t* data = mem + head;

    PendingRegion pr;
    pr.mem = mem; pr.bytes = alloc_bytes; pr.head = head; pr.uniq = uniq;
    bool fail = false;
    for (size_t k = 0; k < uniq.size() && !fail; ++k) {
        // Frequency cache first: a hit is a memcpy at RAM speed, no disk request,
        // and no tag to wait on at adoption.
        if (const void* c = p_.slots.hit(w, uniq[k])) {
            std::memcpy(data + k * stride, c, static_cast<size_t>(stride));
            continue;
        }
        pr.miss.push_back(k);
        Source s = *src;
        s.offset += static_cast<uint64_t>(uniq[k]) *
                    (s.disk_stride ? s.disk_stride : stride);
        if (!p_.io.submit_exact(s, data + k * stride, stride, &pr.tags)) { fail = true; break; }
    }
    if (fail) {
        // In-flight DMA writes into `mem`: it MUST NOT be freed until every
        // submitted read has completed, success or not. And "settle returned
        // false" is not the same as "completed": a dead backend leaves the reads
        // OUTSTANDING, so the free below was the F2 free-under-DMA class through
        // a nullptr out-param (2026-08-24 audit). Leak deliberately instead.
        bool dead = false;
        p_.io.settle(pr.tags, &dead);
        if (dead) { log_region_leak(); return; }
        p_.mem.free(mem::Category::ExpertCache, mem, alloc_bytes);
        return;
    }
    pending_[w] = std::move(pr);
}

// From any one of a layer's expert tensors, submit the other two. The suffixes are
// the ggml MoE convention shared by every target model, not an architecture branch.
void ExpertCompactor::submit_layer_siblings(ggml_tensor* w, const std::vector<int32_t>& uniq) {
    static const char* kinds[3] = {"_gate_exps", "_up_exps", "_down_exps"};
    const char* mine = nullptr;
    for (const char* k : kinds) if (std::strstr(w->name, k)) { mine = k; break; }
    if (!mine) return;
    for (const char* k : kinds) {
        if (k == mine) continue;
        std::string n(w->name);
        n.replace(n.find(mine), std::strlen(mine), k);
        if (ggml_tensor* sib = p_.tensors.named(n)) submit_sibling_region(sib, uniq);
    }
}

void ExpertCompactor::unlock_early(ggml_tensor* node) {
    if (flags_.no_early || !node || trio_by_ids_.empty()) return;
    auto it = trio_by_ids_.find(node);
    if (it == trio_by_ids_.end()) return;
    const auto t0 = std::chrono::steady_clock::now();
    struct Timed {
        uint64_t* ns; std::chrono::steady_clock::time_point t0;
        ~Timed() {
            *ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count());
        }
    } timed{&ns_unlock_, t0};
    std::vector<int32_t> uniq, remapped;
    if (!derive_ids_map(node, it->second.n_expert, &uniq, &remapped)) return;
    if (!fast_path(node, uniq.size())) return;
    for (int s = 0; s < 3; ++s) {
        if (it->second.w[s]) submit_sibling_region(it->second.w[s], uniq);
    }
    ++early_unlocks_;
}

// ---------------------------------------------------------------------------
// expert compaction (MUL_MAT_ID)

bool ExpertCompactor::compact_experts(ggml_tensor* node, ggml_tensor* w, ggml_tensor* ids,
                                      uint64_t* streamed) {
    if (!w || !ids || ids->type != GGML_TYPE_I32) return false;

    const Source* src = p_.tensors.source_of(w);
    if (!src) return false;          // weights not ours
    if (w->ne[2] <= 1) return false;

    const int64_t n_expert = w->ne[2];
    const uint64_t stride  = static_cast<uint64_t>(w->nb[2]);

    // Read-only derivation from the ORIGINAL ids. Every node of this layer derives
    // the same ordering because it derives it from the same untouched source, so no
    // generation counter or "already remapped?" test is needed -- the ambiguity
    // those existed to resolve cannot arise.
    std::vector<int32_t> uniq, remapped;
    if (!derive_ids_map(ids, n_expert, &uniq, &remapped)) return false;

    // IDENTITY COMPACTION: same machinery -- compact region, private ids, repointed
    // data -- but every expert included in original order, so the mapping is the
    // identity. Separates "the compaction mechanism is broken" from "selecting a
    // subset is broken", which the trace shows are both individually plausible.
    const bool compact_all = flags_.compact_all;
    if (compact_all) {
        uniq.resize(static_cast<size_t>(n_expert));
        for (int64_t e = 0; e < n_expert; ++e) uniq[static_cast<size_t>(e)] = static_cast<int32_t>(e);
        const int64_t n0 = ids->ne[0], n1 = ids->ne[1];
        remapped.assign(static_cast<size_t>(n0 * n1), 0);
        size_t k = 0;
        for (int64_t i1 = 0; i1 < n1; ++i1)
            for (int64_t i0 = 0; i0 < n0; ++i0, ++k)
                remapped[k] = *ids_at(ids, i0, i1);
    }

    // PHASE C bookkeeping and the skew histogram, decode only. Decode vs prefill:
    // ne[1]==1 was only ever a proxy. A batched decode step carries ne[1]==B<=n_seq;
    // prefill chunks carry hundreds. The bound is the admitted width, and at
    // n_seq==1 it reduces to the old test exactly.
    const bool fast = fast_path(ids, uniq.size());
    if (fast) {
        ExpertTrio& trio = trio_by_ids_[ids];
        trio.n_expert = n_expert;
        bool known = false;
        for (int s = 0; s < 3; ++s) if (trio.w[s] == w) { known = true; break; }
        if (!known) {
            for (int s = 0; s < 3; ++s) if (!trio.w[s]) { trio.w[s] = w; break; }
        }
        p_.skew.record(w, uniq, n_expert);
    }

    // Reuse only if the region holds exactly this ordering.
    if (const Resident* ex = p_.cache.find(w)) {
        if (!flags_.no_reuse && ex->uniq == uniq) {
            w->data = static_cast<uint8_t*>(ex->mem) + ex->head;
            p_.hits.add_routed(static_cast<uint64_t>(uniq.size()) * stride, 0);   // from RAM: read 0
            if (!compact_all) p_.ids.install(node, ids, remapped);
            return true;
        }
    }
    // The previous routing's region is dead from here on: keep its memory for
    // this routing's region if it fits (alloc_region), else it is freed now.
    // Held until then, so it is freed on every path that does not consume it.
    const uint64_t need = static_cast<uint64_t>(uniq.size()) * stride;
    const RegionGeom geom = region_geom(*src, stride, need);
    void* spare = p_.cache.reclaim_own(w, geom.bytes, geom.align, ResidencyCache::Mismatch::Drop);
    auto free_spare = [&] {
        if (spare) p_.mem.free(mem::Category::ExpertCache, spare, geom.bytes);
        spare = nullptr;
    };

    // PHASE A: get the whole layer's reads in flight together. Submit the two
    // sibling regions first (unwaited), then handle this tensor -- whose own reads
    // overlap the siblings', and whose waits route their completions to the
    // registry via pump().
    //
    // DECODE ONLY. During prefill the ids carry every prompt token, so uniq grows
    // toward n_expert and each region toward the whole tensor -- two unwaited,
    // unreclaimable sibling regions then eat exactly the churn slack the current
    // tensor needs (measured: blk.5 gate failed with 107 LRU entries, all pinned,
    // nothing evictable, at a cap that worked unfused). At decode ne[1]<=n_seq
    // caps the trio at ~3 x uniq(n_seq) x stride -- which is precisely what the
    // width-scaled churn reserve funds, so batch fusion is inside the reserve by
    // construction.
    if (!flags_.no_fuse && fast) submit_layer_siblings(w, uniq);

    // Adopt a region a sibling's call already submitted for this routing.
    auto pr = pending_.find(w);
    if (pr != pending_.end()) {
        if (pr->second.uniq == uniq) {
            free_spare();   // the pending region replaces it
            bool dead = false;
            const bool ok = p_.io.settle(pr->second.tags, &dead);
            uint8_t* mem = pr->second.mem;
            const uint64_t need_p = pr->second.bytes;   // the allocation
            const uint32_t head_p = pr->second.head;
            uint8_t* data = mem + head_p;
            const std::vector<size_t> miss = std::move(pr->second.miss);
            pending_.erase(pr);
            if (dead) {
                log_region_leak();
                return false;   // caller falls back; the memory is never reused
            }
            if (!ok) {
                p_.mem.free(mem::Category::ExpertCache, mem, need_p);
                return false;
            }
            // Bytes-read counts DISK bytes only: cache-hit slots were memcpy'd at
            // submission and never touched the drive. h_routed splits the same way.
            const uint64_t miss_bytes = static_cast<uint64_t>(miss.size()) * stride;
            *streamed += miss_bytes;
            p_.hits.add_routed(static_cast<uint64_t>(uniq.size()) * stride, miss_bytes);
            if (admits(ids)) {
                for (size_t k : miss) p_.slots.admit(w, uniq[k], data + k * stride, stride);
            }
            if (!compact_all) p_.ids.install(node, ids, remapped);
            Resident nr; nr.mem = mem; nr.bytes = need_p; nr.head = head_p; nr.pinned = false;
            nr.uniq = uniq; nr.prio = Prio::RoutedExpert;
            p_.cache.add(w, nr);
            w->data = data;
            return true;
        }
        // Routing changed before the region was consumed (abort/edge path): the
        // layout is for the wrong experts. Drain the DMA, free, fall through --
        // unless the backend died, in which case the DMA never drained and the
        // region must be leaked, not freed.
        bool dead = false;
        p_.io.settle(pr->second.tags, &dead);
        if (dead) {
            log_region_leak();
            pending_.erase(pr);
        } else {
            p_.mem.free(mem::Category::ExpertCache, pr->second.mem, pr->second.bytes);
            pending_.erase(pr);
        }
    }

    const bool trace = flags_.trace_compact;
    if (trace && ids->ne[1] == 1 && traced_ < 20) {
        ++traced_;
        std::fprintf(stderr, "[dray] COMPACT %s ids.ne=[%lld,%lld,%lld] nb=[%zu,%zu] uniq=%zu first=%d last=%d\n",
                     w->name, (long long)ids->ne[0], (long long)ids->ne[1],
                     (long long)ids->ne[2], ids->nb[0], ids->nb[1],
                     uniq.size(), uniq.empty() ? -1 : uniq.front(),
                     uniq.empty() ? -1 : uniq.back());
        std::fflush(stderr);
    }

    uint8_t* mem = nullptr;
    uint64_t alloc_bytes = 0;
    uint32_t head = 0;
    const bool got = alloc_region(*src, stride, need, &mem, &alloc_bytes, &head, spare);
    spare = nullptr;   // a non-null spare is always consumed (alloc_region cannot fail with one)
    if (!got) {
        if (failures_ < 8) {
            std::fprintf(stderr,
                "[dray] COMPACT FAIL %s: no room  uniq=%zu need=%.2fGB in_use=%.2fGB budget=%.2fGB lru=%zu\n",
                w->name, uniq.size(), alloc_bytes / 1e9, p_.mem.cache_used() / 1e9,
                p_.mem.cache_budget() / 1e9, p_.cache.lru_size());
            std::fflush(stderr);
        }
        return false;
    }
    uint8_t* data = mem + head;

    // Frequency cache first, then ONE batch for the misses, so the drive sees
    // every remaining request in flight together.
    std::vector<size_t> miss_k;
    miss_k.reserve(uniq.size());
    for (size_t k = 0; k < uniq.size(); ++k) {
        if (const void* c = p_.slots.hit(w, uniq[k])) {
            std::memcpy(data + k * stride, c, static_cast<size_t>(stride));
        } else {
            miss_k.push_back(k);
        }
    }
    const uint64_t dstride = src->disk_stride ? src->disk_stride : stride;
    std::vector<Slice> slices(miss_k.size());
    for (size_t i = 0; i < miss_k.size(); ++i) {
        const size_t k = miss_k[i];
        slices[i].src = *src;
        slices[i].src.offset += static_cast<uint64_t>(uniq[k]) * dstride;
        slices[i].dst = data + k * stride;
        slices[i].len = stride;
    }
    if (!slices.empty() && !p_.io.read_batch(slices)) {
        p_.mem.free(mem::Category::ExpertCache, mem, alloc_bytes);
        return false;
    }
    const uint64_t miss_bytes = static_cast<uint64_t>(miss_k.size()) * stride;
    *streamed += miss_bytes;
    p_.hits.add_routed(static_cast<uint64_t>(uniq.size()) * stride, miss_bytes);
    // Decode-width steps only (admits(): see there). Hits stay enabled at
    // prefill -- they allocate nothing. Batch decode admits too since slots
    // became evictable: the 2026-08-20 M3 28G B=16 failure (need 0.70 GB, lru=421,
    // nothing evictable) was a committed pool make_room could not ask back.
    if (admits(ids)) {
        for (size_t k : miss_k) {
            p_.slots.admit(w, uniq[k], data + k * stride, stride);
        }
    }

    // Does read_batch actually deliver the bytes compact_experts asked for?
    // Checked against an independent direct read of the first two slots.
    if (trace) {
        for (size_t k = 0; k < uniq.size() && k < 2; ++k) {
            Source one = *src;
            one.offset += static_cast<uint64_t>(uniq[k]) * stride;
            uint64_t ab = 0; uint32_t hd = 0;
            void* chk = p_.io.read_whole(mem::Category::Misc, one, stride, &ab, &hd);
            if (chk) {
                const bool same = std::memcmp(static_cast<uint8_t*>(chk) + hd,
                                              data + k * stride, static_cast<size_t>(stride)) == 0;
                std::fprintf(stderr, "[dray] READBATCH slot %zu expert %d: %s\n",
                             k, uniq[k], same ? "MATCHES direct read" : "*** MISMATCH ***");
                std::fflush(stderr);
                p_.mem.free(mem::Category::Misc, chk, ab);
            }
        }
    }

    // Under identity compaction the remapped ids are numerically identical to the
    // originals, so a private tensor is semantically a no-op. Skipping it isolates
    // "the private ids tensor is wrong" from "the compact buffer is wrong".
    if (!compact_all) p_.ids.install(node, ids, remapped);

    // A compacted region is valid for ONE routing decision, so it is the first
    // thing that should go when room is needed.
    Resident nr; nr.mem = mem; nr.bytes = alloc_bytes; nr.head = head; nr.pinned = false; nr.uniq = uniq;
    nr.prio = Prio::RoutedExpert;
    p_.cache.add(w, nr);
    w->data = data;
    return true;
}

// ---------------------------------------------------------------------------

void ExpertCompactor::release_all() {
    for (auto& kv : pending_) {
        // Pass-4b teardown: a FAILED settle means the backend died with reads
        // outstanding into this region -- the F2 rule applies in the dtor too:
        // leak the memory, never free under DMA, even on the way out.
        bool dead = false;
        const bool ok = p_.io.settle(kv.second.tags, &dead);
        // H15: a completed-but-FAILED read is fully landed -- freeing is safe;
        // only a dead backend (reads still in flight) forces the leak.
        if (ok || !dead) {
            if (kv.second.mem) p_.mem.free_uncharged(kv.second.mem, kv.second.bytes);
        } else if (kv.second.mem) {
            std::fprintf(stderr, "[dray] FATAL: teardown leaking a pending region "
                                 "(backend died with reads outstanding)\n");
        }
    }
    pending_.clear();
}

}  // namespace dray::backend

namespace dray::backend {

int ExpertCompactor::copy_kind_of(const char* name) {
    // The ggml MoE naming convention shared by every target model (as in
    // submit_layer_siblings), not an architecture branch.
    if (std::strstr(name, "_up_exps") && !std::strstr(name, "_gate_up_exps")) return 1;
    if (std::strstr(name, "_down_exps")) return 2;
    return 0;
}

int ExpertCompactor::copy_kind(const ggml_tensor* w) { return copy_kind_of(w->name); }

void ExpertCompactor::set_copy_slots(uint8_t* const base[kCopyKinds], const uint64_t bytes[kCopyKinds]) {
    settle_copy_slots();
    for (int k = 0; k < kCopyKinds; ++k) {
        copy_slots_[k] = CopySlot{};
        copy_slots_[k].mem = base[k];
        copy_slots_[k].bytes = bytes[k];
    }
}

bool ExpertCompactor::settle_slot(CopySlot& s) {
    if (!s.pending) return true;
    bool dead = false;
    const bool ok = p_.io.settle(s.tags, &dead);
    s.tags.clear();
    s.pending = false;
    if (dead) {
        log_region_leak();
        s.mem = nullptr;   // still a DMA target: never land in it again
        s.bytes = 0;
    }
    return ok;
}

void ExpertCompactor::settle_copy_slots() {
    for (CopySlot& s : copy_slots_) settle_slot(s);
}

namespace {

// Where expert e of a full-size landing region sits in `slot`, placed by `g`
// (alignment for direct reads), or null if the slot is too small.
uint8_t* landing_in(uint8_t* slot, uint64_t slot_bytes, uint64_t g_bytes, uint32_t g_align,
                    uint32_t g_head) {
    if (!slot || g_bytes + g_align > slot_bytes) return nullptr;
    const uintptr_t a = g_align ? g_align : 1;
    uint8_t* base = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(slot) + a - 1) / a * a);
    return base + g_head;
}

}  // namespace

uint8_t* ExpertCompactor::read_experts_for_copy(ggml_tensor* w, const int32_t* experts, int64_t n,
                                                uint8_t** mem, uint64_t* bytes, uint64_t* streamed) {
    const Source* src = p_.tensors.source_of(w);
    if (!src || n <= 0 || w->ne[2] <= 1) return nullptr;
    for (int64_t i = 0; i < n; ++i) {
        if (experts[i] < 0 || experts[i] >= w->ne[2]) return nullptr;   // never guess
    }
    const uint64_t stride = static_cast<uint64_t>(w->nb[2]);
    const uint64_t need = stride * static_cast<uint64_t>(w->ne[2]);
    const RegionGeom g = region_geom(*src, stride, need);

    uint8_t* data = nullptr;
    CopySlot& slot = copy_slots_[copy_kind(w)];
    // The slot holds this tensor with this routing: read ahead (wait for it) or still
    // there from an earlier copy (its contents are unchanged since).
    const bool same = slot.data && slot.w == w && slot.experts.size() == static_cast<size_t>(n) &&
                      std::equal(slot.experts.begin(), slot.experts.end(), experts);
    const bool was_pending = slot.pending;
    const bool ok = settle_slot(slot);
    if (same && ok && slot.mem) {
        ++copy_prefetch_hits_;
        if (!was_pending) p_.hits.add_routed(static_cast<uint64_t>(n) * stride, 0);   // served from RAM
        *mem = slot.data;
        *bytes = 0;
        return slot.data;
    }
    if (was_pending) ++copy_prefetch_misses_;
    slot.w = nullptr;
    if ((data = landing_in(slot.mem, slot.bytes, g.bytes, g.align, g.head)) != nullptr) {
        *mem = data;
        *bytes = 0;
    } else {
        uint32_t head = 0;
        if (!alloc_region(*src, stride, need, mem, bytes, &head, nullptr)) return nullptr;
        data = *mem + head;
    }
    const uint64_t dstride = src->disk_stride ? src->disk_stride : stride;
    std::vector<Slice> slices;
    slices.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const int32_t e = experts[i];
        uint8_t* dst = data + static_cast<uint64_t>(e) * stride;
        if (const void* c = p_.slots.hit(w, e)) {
            std::memcpy(dst, c, static_cast<size_t>(stride));
            continue;
        }
        Slice s;
        s.src = *src;
        s.src.offset += static_cast<uint64_t>(e) * dstride;
        s.dst = dst;
        s.len = stride;
        slices.push_back(s);
    }
    if (!slices.empty() && !p_.io.read_batch(slices)) {
        if (*bytes) p_.mem.free(mem::Category::ExpertCache, *mem, *bytes);
        return nullptr;
    }
    const uint64_t read = static_cast<uint64_t>(slices.size()) * stride;
    *streamed += read;
    p_.hits.add_routed(static_cast<uint64_t>(n) * stride, read);
    if (*bytes == 0) {   // landed in the slot: remember what it holds
        slot.w = w;
        slot.experts.assign(experts, experts + n);
        slot.data = data;
    }
    return data;
}

void ExpertCompactor::prefetch_copy_siblings(ggml_tensor* w, const int32_t* experts, int64_t n,
                                             uint64_t* streamed) {
    static const char* kinds[kCopyKinds] = {"_gate_exps", "_up_exps", "_down_exps"};
    const int mine = copy_kind(w);
    if (!std::strstr(w->name, kinds[mine])) return;   // fused or unnamed: no siblings known
    for (int k = 0; k < kCopyKinds; ++k) {
        if (k == mine) continue;
        CopySlot& slot = copy_slots_[k];
        if (!slot.mem) continue;
        std::string name(w->name);
        name.replace(name.find(kinds[mine]), std::strlen(kinds[mine]), kinds[k]);
        ggml_tensor* sib = p_.tensors.named(name);
        const Source* src = sib ? p_.tensors.source_of(sib) : nullptr;
        if (!src || sib->ne[2] <= 1) continue;
        if (const Resident* r = p_.cache.find(sib); r && r->mem && r->uniq.empty()) continue;   // whole in RAM
        bool in_range = true;
        for (int64_t i = 0; i < n && in_range; ++i) in_range = experts[i] >= 0 && experts[i] < sib->ne[2];
        if (!in_range) continue;
        // Already there (read ahead, or copied earlier with this routing): nothing to do.
        if (slot.w == sib && slot.experts.size() == static_cast<size_t>(n) &&
            std::equal(slot.experts.begin(), slot.experts.end(), experts)) {
            continue;
        }
        settle_slot(slot);   // an unclaimed read-ahead: its slot is about to be reused
        slot.w = nullptr;
        if (!slot.mem) continue;
        const uint64_t stride = static_cast<uint64_t>(sib->nb[2]);
        const RegionGeom g = region_geom(*src, stride, stride * static_cast<uint64_t>(sib->ne[2]));
        uint8_t* data = landing_in(slot.mem, slot.bytes, g.bytes, g.align, g.head);
        if (!data) continue;
        const uint64_t dstride = src->disk_stride ? src->disk_stride : stride;
        uint64_t misses = 0;
        bool fail = false;
        for (int64_t i = 0; i < n && !fail; ++i) {
            const int32_t e = experts[i];
            uint8_t* dst = data + static_cast<uint64_t>(e) * stride;
            if (const void* c = p_.slots.hit(sib, e)) {
                std::memcpy(dst, c, static_cast<size_t>(stride));
                continue;
            }
            Source s = *src;
            s.offset += static_cast<uint64_t>(e) * dstride;
            fail = !p_.io.submit_exact(s, dst, stride, &slot.tags);
            ++misses;
        }
        slot.pending = true;   // even on a failed submit: what was submitted must settle
        if (fail) {
            settle_slot(slot);
            continue;
        }
        slot.w = sib;
        slot.experts.assign(experts, experts + n);
        slot.data = data;
        *streamed += misses * stride;
        p_.hits.add_routed(static_cast<uint64_t>(n) * stride, misses * stride);
    }
}

}  // namespace dray::backend
