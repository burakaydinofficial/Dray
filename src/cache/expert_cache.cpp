#include "cache/expert_cache.h"

#include <algorithm>
#include <cstring>
#include <sstream>

namespace dray::cache {

namespace {

// A layer's routed experts live in THREE fused GGUF tensors (ffn_gate_exps,
// ffn_up_exps, ffn_down_exps), each shaped [.., .., n_expert]. ggml_mul_mat_id
// indexes dim 2 by a uniform stride, so each of the three needs its OWN
// uniformly-strided arena -- one expert is three non-adjacent slices, not one
// contiguous block. (This is also why a single Qwen3.8 token touches ~2,760
// discontiguous extents rather than 920.)
//
// We therefore lay one layer's slab out as three back-to-back regions:
//     [ gate : n_slots * gate_stride ][ up : ... ][ down : ... ]
// Slot index i means the same expert in all three, so one remap serves all three
// mul_mat_id calls.
struct ExpertsTensor {
    uint64_t offset = 0;   // offset WITHIN its shard
    uint64_t stride = 0;   // bytes per expert = tensor_bytes / n_experts
    int32_t  shard = 0;
    bool     found = false;
};

struct LayerGeometry {
    ExpertsTensor gate, up, down;
    bool complete() const { return gate.found && up.found && down.found; }
    uint64_t expert_bytes() const { return gate.stride + up.stride + down.stride; }
};

bool name_has(const std::string& n, const char* needle) {
    return n.find(needle) != std::string::npos;
}

}  // namespace

// ---------------------------------------------------------------- LayerCache

LayerCache::LayerCache(mem::Accountant& acct, const plan::LayerSlotClass& spec,
                       size_t n_slots, uint32_t align)
    : spec_(spec) {
    if (n_slots == 0 || spec.slot_bytes == 0) return;
    slab_ = std::make_unique<mem::Slab>(acct, mem::Category::ExpertCache,
                                        static_cast<size_t>(spec.slot_bytes),
                                        n_slots, align);
    if (slab_ && slab_->valid()) {
        slots_.resize(n_slots);
    } else {
        slab_.reset();
    }
}

int32_t LayerCache::find(int32_t expert_id) const {
    for (size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].valid && slots_[i].expert_id == expert_id) {
            return static_cast<int32_t>(i);
        }
    }
    return kNoSlot;
}

int32_t LayerCache::acquire(int32_t expert_id, uint64_t clock) {
    // Already resident: a hit.
    int32_t existing = find(expert_id);
    if (existing != kNoSlot) {
        slots_[existing].last_used = clock;
        ++hits_;
        return existing;
    }
    ++misses_;

    // Prefer an empty slot, else the least-recently-used UNPINNED one. A pinned
    // slot has a read in flight; the storage interface has no cancel() because the
    // portable thread-pool floor cannot cancel, so evicting it is not an option.
    int32_t victim = kNoSlot;
    uint64_t oldest = UINT64_MAX;
    for (size_t i = 0; i < slots_.size(); ++i) {
        const SlotState& s = slots_[i];
        if (s.refcount > 0) continue;
        if (!s.valid) { victim = static_cast<int32_t>(i); break; }
        if (s.last_used < oldest) { oldest = s.last_used; victim = static_cast<int32_t>(i); }
    }
    if (victim == kNoSlot) return kNoSlot;  // every slot pinned

    slots_[victim].expert_id = expert_id;
    slots_[victim].valid = false;
    slots_[victim].last_used = clock;
    return victim;
}

void LayerCache::mark_valid(int32_t slot) {
    if (slot >= 0 && static_cast<size_t>(slot) < slots_.size()) slots_[slot].valid = true;
}

void LayerCache::mark_failed(int32_t slot) {
    if (slot >= 0 && static_cast<size_t>(slot) < slots_.size()) {
        slots_[slot].valid = false;
        slots_[slot].expert_id = -1;
    }
}

void LayerCache::pin(int32_t slot) {
    if (slot >= 0 && static_cast<size_t>(slot) < slots_.size()) ++slots_[slot].refcount;
}

void LayerCache::unpin(int32_t slot) {
    if (slot >= 0 && static_cast<size_t>(slot) < slots_.size() && slots_[slot].refcount > 0) {
        --slots_[slot].refcount;
    }
}

void* LayerCache::slot_ptr(int32_t slot) {
    if (!slab_ || slot < 0 || static_cast<size_t>(slot) >= slab_->n_slots()) return nullptr;
    return slab_->slot(static_cast<size_t>(slot));
}

// --------------------------------------------------------------- ExpertCache

ExpertCache::ExpertCache(mem::Accountant& acct, io::Backend& backend,
                         const plan::Plan& p, uint32_t align)
    : acct_(acct), backend_(backend), align_(align) {
    n_experts_ = p.n_experts;
    if (n_experts_ == 0 || p.slot_classes.empty()) {
        error_ = "plan has no MoE layers";
        return;
    }
    if (p.shard_paths.empty()) {
        error_ = "plan carries no shard paths";
        return;
    }

    shard_files_.assign(p.shard_paths.size(), io::kInvalidFile);
    for (size_t i = 0; i < p.shard_paths.size(); ++i) {
        shard_files_[i] = backend_.open(p.shard_paths[i]);
        if (shard_files_[i] == io::kInvalidFile) {
            error_ = "cannot open shard uncached: " + p.shard_paths[i];
            return;
        }
    }

    // Derive per-layer, per-tensor geometry from the plan's tensor table. We do
    // this here rather than widening LayerSlotClass so the planner's contract
    // stays stable.
    int32_t max_layer = -1;
    for (const auto& sc : p.slot_classes) max_layer = std::max(max_layer, sc.layer);
    if (max_layer < 0) return;

    std::vector<LayerGeometry> geo(static_cast<size_t>(max_layer) + 1);
    for (const plan::TensorInfo& t : p.tensors) {
        if (t.cls != plan::TensorClass::RoutedExpert) continue;
        if (t.layer < 0 || t.layer > max_layer) continue;
        LayerGeometry& g = geo[static_cast<size_t>(t.layer)];
        ExpertsTensor* dst = nullptr;
        if (name_has(t.name, "ffn_gate_exps")) dst = &g.gate;
        else if (name_has(t.name, "ffn_up_exps")) dst = &g.up;
        else if (name_has(t.name, "ffn_down_exps")) dst = &g.down;
        if (!dst) continue;
        dst->offset = t.offset;
        dst->stride = t.bytes / n_experts_;
        dst->shard  = t.shard;
        dst->found  = true;
    }

    // Distribute the cache budget across layers. Uniform slot count per layer is
    // the right default: expert access follows deterministic layer order, so
    // every MoE layer is touched exactly once per token and none deserves more
    // slots than another. (Skew-aware distribution is an open question that the
    // first trace is meant to inform -- see the eviction hypothesis in memory.)
    uint64_t bytes_per_slot_all_layers = 0;
    for (const auto& sc : p.slot_classes) bytes_per_slot_all_layers += sc.slot_bytes;
    if (bytes_per_slot_all_layers == 0) return;

    size_t slots_per_layer = static_cast<size_t>(p.cache_budget / bytes_per_slot_all_layers);
    // Never fewer than n_expert_used, or a single token cannot be served at all.
    if (slots_per_layer < p.n_expert_used) slots_per_layer = p.n_expert_used;
    slots_per_layer = std::min<size_t>(slots_per_layer, n_experts_);

    layers_.resize(static_cast<size_t>(max_layer) + 1);
    expert_offsets_.assign(static_cast<size_t>(max_layer + 1) * 3, 0);
    expert_strides_.assign(static_cast<size_t>(max_layer + 1) * 3, 0);
    expert_shards_.assign(static_cast<size_t>(max_layer + 1) * 3, -1);

    for (const plan::LayerSlotClass& sc : p.slot_classes) {
        const LayerGeometry& g = geo[static_cast<size_t>(sc.layer)];
        if (!g.complete()) continue;
        layers_[static_cast<size_t>(sc.layer)] =
            std::make_unique<LayerCache>(acct_, sc, slots_per_layer, align_);
        const size_t b = static_cast<size_t>(sc.layer) * 3;
        expert_offsets_[b + 0] = g.gate.offset;
        expert_offsets_[b + 1] = g.up.offset;
        expert_offsets_[b + 2] = g.down.offset;
        expert_strides_[b + 0] = g.gate.stride;
        expert_strides_[b + 1] = g.up.stride;
        expert_strides_[b + 2] = g.down.stride;
        expert_shards_[b + 0] = g.gate.shard;
        expert_shards_[b + 1] = g.up.shard;
        expert_shards_[b + 2] = g.down.shard;
    }
}

ExpertCache::~ExpertCache() {
    for (io::FileId f : shard_files_) {
        if (f != io::kInvalidFile) backend_.close(f);
    }
}

bool ExpertCache::valid() const {
    for (const auto& l : layers_) {
        if (l && l->valid()) return true;
    }
    return false;
}

LayerCache* ExpertCache::layer_cache(int32_t layer) {
    if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return nullptr;
    return layers_[static_cast<size_t>(layer)].get();
}

uint64_t ExpertCache::expert_offset(int32_t layer, int32_t which) const {
    size_t idx = static_cast<size_t>(layer) * 3 + static_cast<size_t>(which);
    return idx < expert_offsets_.size() ? expert_offsets_[idx] : 0;
}

uint64_t ExpertCache::expert_stride(int32_t layer, int32_t which) const {
    size_t idx = static_cast<size_t>(layer) * 3 + static_cast<size_t>(which);
    return idx < expert_strides_.size() ? expert_strides_[idx] : 0;
}

io::FileId ExpertCache::expert_file(int32_t layer, int32_t which) const {
    size_t idx = static_cast<size_t>(layer) * 3 + static_cast<size_t>(which);
    if (idx >= expert_shards_.size()) return io::kInvalidFile;
    const int32_t s = expert_shards_[idx];
    if (s < 0 || static_cast<size_t>(s) >= shard_files_.size()) return io::kInvalidFile;
    return shard_files_[static_cast<size_t>(s)];
}

Remap ExpertCache::ensure(int32_t layer, const int32_t* expert_ids, size_t n) {
    Remap out;
    out.slot_of_selected.assign(n, kNoSlot);
    LayerCache* lc = layer_cache(layer);
    if (!lc || !lc->valid()) return out;

    ++clock_;

    // Assign slots and collect the reads needed for misses.
    //
    // Uncached reads must be sector-aligned in offset AND length, but GGUF aligns
    // tensor data to general.alignment (32 by default). So every slice is read into
    // an aligned staging buffer covering the widened range, and its interior is
    // copied into the slot. This copy is what the io_staging floor line pays for;
    // submitting the raw offsets simply fails, which is how this was found -- 912
    // misses, zero bytes read.
    struct Pending {
        int32_t  slot = kNoSlot;
        int32_t  which = 0;
        uint8_t* staging = nullptr;   // aligned, owned here
        uint64_t head = 0;            // bytes of padding before the wanted data
        uint8_t* dst = nullptr;
        uint64_t len = 0;             // wanted bytes
        uint32_t span = 0;            // widened bytes actually read
    };
    std::vector<io::ReadRequest> reqs;
    std::vector<Pending> pending;
    const io::Alignment al = backend_.alignment(
        expert_file(layer, 0) != io::kInvalidFile ? expert_file(layer, 0) : io::kInvalidFile);
    auto free_staging = [&pending]() {
        for (Pending& q : pending) {
            if (q.staging) { mem::aligned_free_host(q.staging, q.span); q.staging = nullptr; }
        }
    };

    for (size_t i = 0; i < n; ++i) {
        const int32_t eid = expert_ids[i];
        const bool was_resident = lc->find(eid) != kNoSlot;
        const int32_t slot = lc->acquire(eid, clock_);
        if (slot == kNoSlot) {
            // S24: unwind THIS call's pins and staging before bailing, or every
            // retry ratchets the pin count up until the layer wedges permanently.
            for (size_t j = 0; j < i; ++j) lc->unpin(out.slot_of_selected[j]);
            free_staging();
            return out;
        }

        out.slot_of_selected[i] = slot;
        lc->pin(slot);

        if (was_resident) continue;

        // Three slices per expert. The slab slot holds them back to back in the
        // same order the arenas expect.
        uint8_t* dst = static_cast<uint8_t*>(lc->slot_ptr(slot));
        if (!dst) return out;

        // gate/up/down have DIFFERENT per-expert strides -- on M3 UD-Q2_K_XL gate
        // and up are IQ2_XS (5,455,872 B) while down is IQ3_XXS (7,225,344 B).
        // Deriving a stride by dividing the slot evenly would read wrong ranges
        // and silently return garbage weights.
        uint64_t cursor = 0;
        for (int which = 0; which < 3; ++which) {
            const uint64_t stride = expert_stride(layer, which);
            if (stride == 0) { free_staging(); return out; }
            const io::FileId fid = expert_file(layer, which);
            if (fid == io::kInvalidFile) { free_staging(); return out; }

            const uint64_t want = expert_offset(layer, which) + static_cast<uint64_t>(eid) * stride;
            const uint64_t lo = (want / al.offset) * al.offset;
            const uint64_t head = want - lo;
            uint64_t span = head + stride;
            span = ((span + al.length - 1) / al.length) * al.length;

            uint8_t* stage = static_cast<uint8_t*>(
                mem::aligned_alloc_host(static_cast<size_t>(span), al.memory));
            if (!stage) { free_staging(); return out; }

            io::ReadRequest r;
            r.file   = fid;
            r.offset = lo;
            r.length = static_cast<uint32_t>(span);
            r.dst    = stage;
            r.tag    = static_cast<uint64_t>(pending.size());
            reqs.push_back(r);

            Pending q;
            q.slot = slot; q.which = which; q.staging = stage;
            q.head = head; q.dst = dst + cursor; q.len = stride;
            q.span = static_cast<uint32_t>(span);
            pending.push_back(q);
            cursor += stride;
        }
    }

    // Submit and drain. Submission can be partial when the queue is full.
    size_t submitted = 0;
    size_t completed = 0;
    std::vector<io::Completion> comps(reqs.size());
    bool failed = false;

    while (completed < reqs.size()) {
        if (submitted < reqs.size()) {
            size_t took = backend_.submit(reqs.data() + submitted, reqs.size() - submitted);
            submitted += took;
            if (took == 0 && backend_.in_flight() == 0) { failed = true; break; }
        }
        if (submitted == completed) continue;
        size_t got = backend_.poll(comps.data(), comps.size(), 1);
        for (size_t i = 0; i < got; ++i) {
            const size_t qi = static_cast<size_t>(comps[i].tag);
            if (qi >= pending.size()) { failed = true; continue; }
            Pending& q = pending[qi];
            if (!comps[i].ok(q.span)) {
                // A short read or EIO must never reach mul_mat_id: a partially
                // filled slot produces plausible text from garbage weights.
                lc->mark_failed(q.slot);
                failed = true;
            } else {
                std::memcpy(q.dst, q.staging + q.head, static_cast<size_t>(q.len));
            }
            bytes_read_ += comps[i].bytes;
        }
        completed += got;
        // S24 sibling (the worse one): a dead backend returning 0 completions
        // left failed==false, marking every slot valid with bytes never copied
        // in -- garbage weights presented as good.
        if (got == 0) { failed = true; break; }
    }

    free_staging();

    if (!failed) {
        for (size_t i = 0; i < n; ++i) lc->mark_valid(out.slot_of_selected[i]);
        out.complete = true;
    } else {
        // T21: every failure return must unwind ITS pins, or each faulted call
        // permanently retires n_expert_used slots (acquire skips refcount>0)
        // and the layer wedges. The header's kNoSlot contract is honoured at
        // the same time: failed entries report no slot.
        for (size_t i = 0; i < n; ++i) {
            if (out.slot_of_selected[i] != kNoSlot) {
                lc->unpin(out.slot_of_selected[i]);
                out.slot_of_selected[i] = kNoSlot;
            }
        }
    }
    return out;
}

void ExpertCache::release(int32_t layer, const Remap& r) {
    LayerCache* lc = layer_cache(layer);
    if (!lc) return;
    for (int32_t s : r.slot_of_selected) {
        if (s != kNoSlot) lc->unpin(s);
    }
}

uint64_t ExpertCache::hits() const {
    uint64_t h = 0;
    for (const auto& l : layers_) if (l) h += l->hits();
    return h;
}

uint64_t ExpertCache::misses() const {
    uint64_t m = 0;
    for (const auto& l : layers_) if (l) m += l->misses();
    return m;
}

double ExpertCache::hit_rate() const {
    const uint64_t h = hits(), m = misses();
    return (h + m) ? static_cast<double>(h) / static_cast<double>(h + m) : 0.0;
}

std::string ExpertCache::report() const {
    std::ostringstream o;
    size_t live = 0, total_slots = 0;
    for (const auto& l : layers_) {
        if (!l || !l->valid()) continue;
        ++live;
        total_slots += l->n_slots();
    }
    o << "expert cache: " << live << " layers, " << total_slots << " slots, "
      << "routed hit rate " << (hit_rate() * 100.0) << "% ("
      << hits() << " hit / " << misses() << " miss), "
      << (static_cast<double>(bytes_read_) / 1e9) << " GB read";
    return o.str();
}

}  // namespace dray::cache
