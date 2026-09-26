#include "backend/expert_compactor.h"

#include <cstdio>
#include <cstring>
#include <string>

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
            w->extra = nullptr;   // compacted slab is not the repacked layout
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
    w->extra = nullptr;   // freshly compacted bytes; drop any stale repack traits
    w->data = mem;
    p_.ids.install(node, ids, remapped, 1);
    return true;
}

// ---------------------------------------------------------------------------
// Phase A / Phase C: sibling regions, submitted unwaited

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
    if (!p_.cache.make_room(need)) return;
    uint8_t* mem = static_cast<uint8_t*>(
        p_.mem.alloc(mem::Category::ExpertCache, need, kHostAlign));
    if (!mem) return;

    PendingRegion pr;
    pr.mem = mem; pr.bytes = need; pr.uniq = uniq;
    bool fail = false;
    for (size_t k = 0; k < uniq.size() && !fail; ++k) {
        // Frequency cache first: a hit is a memcpy at RAM speed, no disk request,
        // and no tag to wait on at adoption.
        if (const void* c = p_.slots.hit(w, uniq[k])) {
            std::memcpy(mem + k * stride, c, static_cast<size_t>(stride));
            continue;
        }
        pr.miss.push_back(k);
        Source s = *src;
        s.offset += static_cast<uint64_t>(uniq[k]) *
                    (s.disk_stride ? s.disk_stride : stride);
        const uint64_t tag = p_.io.submit_staged(s, mem + k * stride, stride);
        if (tag == 0) { fail = true; break; }
        pr.tags.push_back(tag);
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
        p_.mem.free(mem::Category::ExpertCache, mem, need);
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
    if (flags_.no_early || !node || trio_by_ids_.empty() || !is_decode(node)) return;
    auto it = trio_by_ids_.find(node);
    if (it == trio_by_ids_.end()) return;
    std::vector<int32_t> uniq, remapped;
    if (!derive_ids_map(node, it->second.n_expert, &uniq, &remapped)) return;
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
    if (is_decode(ids)) {
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
            w->extra = nullptr;   // compacted slab is not the repacked layout
            w->data = ex->mem;
            p_.hits.add_routed(ex->bytes, 0);   // served from RAM: read 0
            if (!compact_all) p_.ids.install(node, ids, remapped);
            return true;
        }
        p_.cache.drop(w);
    }

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
    if (!flags_.no_fuse && is_decode(ids)) submit_layer_siblings(w, uniq);

    // Adopt a region a sibling's call already submitted for this routing.
    auto pr = pending_.find(w);
    if (pr != pending_.end()) {
        if (pr->second.uniq == uniq) {
            bool dead = false;
            const bool ok = p_.io.settle(pr->second.tags, &dead);
            uint8_t* mem = pr->second.mem;
            const uint64_t need_p = pr->second.bytes;
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
            p_.hits.add_routed(need_p, miss_bytes);
            for (size_t k : miss) {
                p_.slots.admit(w, uniq[k], mem + k * stride, stride);
            }
            if (!compact_all) p_.ids.install(node, ids, remapped);
            Resident nr; nr.mem = mem; nr.bytes = need_p; nr.pinned = false;
            nr.uniq = uniq; nr.prio = Prio::RoutedExpert;
            p_.cache.add(w, nr);
            w->data = mem;
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

    const uint64_t need = static_cast<uint64_t>(uniq.size()) * stride;
    if (!p_.cache.make_room(need)) {
        if (failures_ < 8) {
            std::fprintf(stderr,
                "[dray] COMPACT FAIL %s: no room  uniq=%zu need=%.2fGB in_use=%.2fGB budget=%.2fGB lru=%zu\n",
                w->name, uniq.size(), need / 1e9, p_.mem.cache_used() / 1e9,
                p_.mem.cache_budget() / 1e9, p_.cache.lru_size());
            std::fflush(stderr);
        }
        return false;
    }
    uint8_t* mem = static_cast<uint8_t*>(
        p_.mem.alloc(mem::Category::ExpertCache, need, kHostAlign));
    if (!mem) return false;

    // Frequency cache first, then ONE batch for the misses, so the drive sees
    // every remaining request in flight together.
    std::vector<size_t> miss_k;
    miss_k.reserve(uniq.size());
    for (size_t k = 0; k < uniq.size(); ++k) {
        if (const void* c = p_.slots.hit(w, uniq[k])) {
            std::memcpy(mem + k * stride, c, static_cast<size_t>(stride));
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
        slices[i].dst = mem + k * stride;
        slices[i].len = stride;
    }
    if (!slices.empty() && !p_.io.read_batch(slices)) {
        p_.mem.free(mem::Category::ExpertCache, mem, need);
        return false;
    }
    const uint64_t miss_bytes = static_cast<uint64_t>(miss_k.size()) * stride;
    *streamed += miss_bytes;
    p_.hits.add_routed(static_cast<uint64_t>(uniq.size()) * stride, miss_bytes);
    // DECODE ONLY -- the Phase A rule, third appearance: an unreclaimable
    // allocation made during prefill competes with the static pinning claim that
    // has not settled yet. GLM at 16 GiB: slots admitted at prefill packed the
    // budget by layer 52 and the region had no room, nothing evictable. Hits
    // stay enabled at prefill (they allocate nothing).
    // SINGLE-STREAM ONLY (measured 2026-08-20, M3 28G B=16 bisect): under
    // batch the pool pins union experts and, at caps where the unconditional
    // set fully fits, pool+statics strangle the evictable remainder -- one
    // materialise failure at the first wide union (need 0.70GB, lru=421,
    // nothing evictable). NO_ESLOTS cleared it: 2.263 GB/token vs FAIL. The
    // batch eslot case measured: testbed -22% bytes, K3 nil, M3 fatal -- a
    // properly EVICTABLE pool is the filed fix; until then batch runs
    // without eslot admission.
    if (ids->ne[1] == 1) {
        for (size_t k : miss_k) {
            p_.slots.admit(w, uniq[k], mem + k * stride, stride);
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
                                              mem + k * stride, static_cast<size_t>(stride)) == 0;
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
    Resident nr; nr.mem = mem; nr.bytes = need; nr.pinned = false; nr.uniq = uniq;
    nr.prio = Prio::RoutedExpert;
    p_.cache.add(w, nr);
    w->extra = nullptr;   // freshly compacted bytes; drop any stale repack traits
    w->data = mem;
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
