#include "backend/stream_buffer.h"

// A (2026-08-22): the fast CPU matmul kernels gate on buffer-type IDENTITY with
// ggml_backend_cpu_repack_buffer_type(), which a streaming engine can never
// satisfy. The vendored fork exposes two entry points so we can repack the
// bytes ourselves and declare this buffer acceptable.
extern "C" {
bool ggml_cpu_repack_data_in_place(const struct ggml_tensor* t, void* data, size_t size);
void* ggml_cpu_repack_traits_for(const struct ggml_tensor* t);
void ggml_cpu_repack_accept_buft(ggml_backend_buffer_type_t buft);
}

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <unordered_set>
#include <list>
#include <set>
#include <sstream>
#include <unordered_map>

#include "ggml-backend-impl.h"
#include "ggml.h"

#include "mem/pool_arena.h"
#include "gguf.h"

namespace dray::backend {

namespace {
constexpr size_t kBufAlign = 64;   // ggml's host tensor alignment
}

// ---------------------------------------------------------------------------

// Eviction priority. Falls straight out of reads-per-token: an unconditional
// weight is read EVERY token, a routed expert only when selected (k/n_experts, so
// 10/512 on Qwen3.8). Evicting an unconditional weight to house an expert trades a
// 1.0 for a 0.02 -- which is what pure LRU does, and why this cap was doing ~50
// GB/token against the ~31.7 the RAM curve predicts.
enum class Prio : uint8_t {
    RoutedExpert  = 0,   // evict these first
    Unconditional = 1,   // ~51x more valuable per byte on Qwen3.8
};

struct Resident {
    void*    mem = nullptr;
    uint64_t bytes = 0;
    uint32_t refs = 0;
    bool     pinned = false;
    // A: bytes were rewritten into ggml-cpu's interleaved layout so the
    // optimised matmul kernels accept them. Such a tensor no longer matches
    // the file byte-for-byte, so self_check must skip it and SAY it skipped.
    bool     repacked = false;
    Prio     prio = Prio::Unconditional;
    // Which ledger line this memory is charged to, so releasing it credits the
    // same category that reserving it debited. Getting this wrong would drift the
    // Accountant silently, which is the failure the single-ledger change exists to
    // prevent.
    mem::Category cat = mem::Category::ExpertCache;
    // Offset from the allocation base to the tensor's first byte. Non-zero when the
    // read was widened to device alignment and landed directly in this buffer, with
    // no staging copy. `mem`+`bytes` describe the ALLOCATION (what to free and what
    // is charged); the tensor points at mem+head.
    uint32_t head = 0;
    // For compacted expert regions: the exact expert ordering this layout holds.
    // A compacted region is ONLY valid for the routing decision that produced it,
    // so reuse is gated on the ordering matching, not on recency. Empty means a
    // whole tensor, valid for any routing.
    std::vector<int32_t> uniq;
};

struct Streamer::Impl {
    mem::Accountant&  acct;
    const plan::Plan& plan;
    Config            cfg;

    std::unique_ptr<io::Backend>  io;
    std::vector<io::FileId>       shards;
    uint32_t                      align = 4096;

    std::unordered_map<std::string, Source>            sources;   // by tensor name
    std::unordered_map<const ggml_tensor*, Source>     bound;     // by tensor
    std::unordered_map<const ggml_tensor*, Resident>   resident;
    std::list<const ggml_tensor*>                      lru;       // front = newest

    // NEITHER the budget NOR the bytes in use are stored. Both derive from the
    // Accountant (cache_budget, cache_used), so they cannot drift from the ledger.
    // A stored copy is updated at a different instant than the ledger is, and the
    // two then disagree exactly when it matters: at the cap, where make_room
    // reported room while every reservation was being refused.

    // STATIC PINNING, because the access pattern is cyclic.
    //
    // The graph sweeps every layer once per token, so each weight is touched
    // exactly once per cycle. When the working set exceeds the cache, LRU evicts
    // every entry before it is reused: hit rate is 0%, and reordering WHICH entry
    // is evicted cannot change that. Measured directly -- adding priority eviction
    // left bytes-streamed byte-identical at 594.007 GB.
    //
    // Pinning a fixed subset instead converts that 0% into cache/working-set. On
    // Qwen3.8 at a 12 GiB cap: ~8.9 GB pinned against a 36.9 GB unconditional set,
    // so ~24% of the stream stops being re-read every token.
    //
    // This is the static-placement-beats-dynamic-caching result from the prior art
    // (~11% vs 100% on a balanced model), arrived at from the other direction.
    // Likewise derived: see static_allowance().
    uint64_t static_used = 0;
    // Room the current layer is guaranteed. Prefetch may never encroach on it:
    // in-flight reads hold memory that eviction cannot reclaim (they are not in the
    // LRU yet), so unbounded prefetch fills the budget and starves the very node it
    // was meant to help.
    uint64_t churn_reserve = 0;
    // A: repack whole 2D weights on materialisation so the optimised kernels
    // accept them (DRAY_NO_REPACK=1 disables, for bisecting).
    bool repack_on = true;
    // Where does the streaming path spend its time? Accumulated nanoseconds
    // inside the two hot callbacks, reported at close. If these dominate the
    // wall clock, the 3x is ours and fixable; if they do not, it is elsewhere.
    // Effective queue depth: the drive sustains 2.75 GB/s during K3 decode where
    // calibration says 6.8 at QD16. Continuous, so not an overlap problem. If
    // the mean batch is small the deep-queue claim in read_batch is not what
    // the hot path delivers.
    uint64_t depth_sum = 0;
    uint64_t depth_n = 0;
    uint64_t depth_max = 0;
    uint64_t depth_min = ~0ull;
    // Service-time histogram, microseconds: <200, <500, <1000, <2000, >=2000
    uint64_t lat_bucket[5] = {0,0,0,0,0};
    uint64_t lat_sum_us = 0;
    uint64_t lat_n = 0;
    uint64_t lat_bytes = 0;
    // Harvesting cadence. If polls are rare and each returns a pile, the 26.5 ms
    // is us not looking, not the drive being slow.
    uint64_t pump_calls = 0;
    uint64_t pump_harvested = 0;
    uint64_t pump_gap_us = 0;
    uint64_t pump_last_ns = 0;
    uint64_t gap_bucket[5] = {0,0,0,0,0};   // <100us, <1ms, <5ms, <20ms, >=20ms
    uint64_t gap_max_us = 0;
    uint64_t gap_long_us = 0;               // total time spent in gaps >= 5ms
    uint64_t in_pump_ns = 0;                // time INSIDE pump, i.e. blocked in poll
    uint64_t memcpy_ns = 0;
    uint64_t memcpy_bytes = 0;
    uint64_t stage_alloc_ns = 0;
    uint64_t stage_alloc_n = 0;
    // TIME-WEIGHTED depth. The submit-sampled figure is biased high: it records
    // in_flight() at the one moment we are pushing work in. This integrates
    // depth over elapsed time instead, which is what Little's law needs.
    uint64_t depth_area_ns = 0;   // sum of in_flight * nanoseconds
    uint64_t depth_t_last = 0;
    uint64_t depth_t0 = 0;
    size_t   depth_cur = 0;
    uint64_t depth_low = 0;   // submits that found fewer than 4 in flight
    uint64_t io_batches = 0;
    uint64_t io_slices = 0;
    uint64_t io_batch_max = 0;
    uint64_t ns_materialise = 0;
    uint64_t ns_release = 0;
    uint64_t n_materialise = 0;
    uint64_t repack_tried = 0;    // 2D whole tensors reaching maybe_repack
    uint64_t repack_done = 0;     // ...that actually got interleaved
    uint64_t repack_no_traits = 0;// ...declined for lack of an optimal repack
    // Widest single batched union region (uniq(n_seq) x slot): the measured
    // admission floor. Certain death below 1x (B=38), proven clean at 2x (B=32).
    uint64_t batch_region_bound = 0;
    // Speculative reads are capped explicitly, not just checked against the budget.
    //
    // Three claimants share the budget: statically pinned tensors, in-flight
    // prefetch, and the working set of the node being computed. Only the last one
    // MUST succeed, and it is the only one that cannot wait. Prefetch memory is not
    // in the LRU, so make_room cannot reclaim it -- an unbounded prefetch therefore
    // starves the node it exists to help, which is exactly what happened.
    uint64_t inflight_bytes = 0;
    uint64_t inflight_cap = 0;

    // Per ids-tensor state for expert compaction.
    //
    // build_moe_ffn hands the SAME selected_experts tensor to three separate
    // MUL_MAT_ID nodes (gate, up, down). The ids must therefore be rewritten to
    // compact indices exactly once, and all three weight tensors must lay their
    // compact regions out in the SAME order, or nodes two and three index a
    // different permutation than node one built.
    //
    // `remapped` is a copy of what we wrote into the ids buffer. Before reusing a
    // mapping we compare the buffer against it: equal means this is still the same
    // routing decision, different means the router has produced fresh ids (next
    // token) and the mapping must be rebuilt. Without that check the second token
    // would be remapped on top of the first token's remapping.
    // PRIVATE IDS. The shared selected_experts tensor is never written.
    //
    // An earlier design rewrote it in place to compact indices, which requires
    // EVERY consumer of those ids to be compaction-aware. Three were found at
    // three different indexed axes -- MUL_MAT_ID (dim 2), ADD_ID (dim 1), and
    // GET_ROWS on the per-expert scale vector inside build_lora_mm_id -- each one
    // only after assuming the set was complete. The set is not enumerable from
    // here: it depends on which architecture is loaded.
    //
    // So the weight tensor gets its own ids instead. node->src[2] is repointed at
    // this struct just before the node computes; the original keeps real expert
    // ids, so scales and biases stay correct by construction rather than by
    // remembering to handle them.
    // PER-NODE, not shared. A single shared ids tensor is only safe if exactly one
    // node computes between one materialise and the next -- an assumption about
    // ggml_backend_sched batching that this engine has already been burned by once.
    // Each compacted node owns its ids outright, so the question cannot arise.
    // unordered_map gives stable element addresses, which node->src must rely on.
    struct PrivIds {
        ggml_tensor          t{};
        std::vector<int32_t> buf;
        // The ids tensor the GRAPH originally supplied. We overwrite node->src with
        // our own and never restore it, and llama.cpp reuses graphs across decode
        // steps -- so on the next token node->src already points at OUR tensor, and
        // deriving the mapping from it would remap an already-remapped set: the
        // router asks for experts 5,12,33 and we would fetch 0,1,2. First token
        // correct, every later token wrong, which is the observed signature.
        ggml_tensor*         orig = nullptr;
    };
    std::unordered_map<const ggml_tensor*, PrivIds> priv_by_node;

    // PHASE A of the I/O pipeline: fused per-layer expert reads.
    //
    // The three MUL_MAT_ID tensors of a layer (gate/up/down) share one ids tensor,
    // so the moment ONE of them derives the routing, the other two regions can be
    // submitted unwaited -- all ~48 expert reads of the layer in flight together,
    // and up/down stream while gate computes. A region sits here from submission
    // until its own node adopts it; it is NOT in the LRU, so it is unreclaimable
    // while pending -- bounded by two sibling regions (~100 MB on K3), covered by
    // the churn reserve.
    // PHASE C: early unlock. The router's ids tensor is itself a graph node; the
    // instant its release callback fires, the layer's expert addresses exist --
    // several small ops before the first MUL_MAT_ID asks for them. This maps each
    // ids tensor to the expert tensors it unlocks, learned at the first
    // compaction (token 1 runs sync; every later token submits early).
    struct ExpertTrio {
        ggml_tensor* w[3] = {nullptr, nullptr, nullptr};
        int64_t      n_expert = 0;
    };
    std::unordered_map<const ggml_tensor*, ExpertTrio> ids_trio;
    uint64_t early_unlocks = 0;
    uint64_t ring_promotions = 0;

    // The two hit rates, Invariant 6. NEVER one symbol: h_routed = routed bytes
    // served from RAM over routed bytes needed (feeds the cost model; whole-file
    // denominators flatter thrashing configs). h_bytes = ALL active bytes served
    // from RAM over all needed (the RAM-vs-disk crossover). Denominator zero
    // prints unknown, never a default.
    uint64_t routed_needed = 0, routed_read = 0;
    uint64_t uncond_needed = 0, uncond_read = 0;

    // Routing-skew histogram (decode picks only). Whether a frequency-kept expert
    // cache can beat plain recency is a property of THIS distribution -- measured,
    // never assumed (the model was trained for balance; invariant 8).
    std::unordered_map<const ggml_tensor*, std::vector<uint32_t>> expert_hist;
    uint64_t hist_selections = 0;

    // The frequency expert cache (owner design, evidence: top-10% experts take
    // 36%/75%/88% of picks on testbed/K3/GLM). Per-expert slots admitted from
    // compaction reads into GENUINELY free budget only -- never by evicting,
    // never into the churn reserve. A hit is a memcpy instead of a disk read.
    // Admit-only first increment; frequency eviction earns code after this is
    // measured. Slots are separate from regions and from the LRU.
    struct ESlot { void* mem = nullptr; uint64_t bytes = 0; };  // self-sized: at
    // teardown the ggml tensors are already gone, so nb[2] must not be consulted
    std::unordered_map<const ggml_tensor*, std::unordered_map<int32_t, ESlot>> eslots;
    uint64_t eslot_hits = 0, eslot_bytes = 0;
    // THE PLAN-TIME LINE ITEM (the law all three speculative features derived):
    // slots may hold at most the past-knee surplus, budget - uncond - churn -
    // ring, computed once at construct. A pinned uncond byte saves exactly one
    // read per token, so pinning outranks slots per byte until the whole set
    // fits; below the knee the pool is zero and the cache is silent. Bounded
    // this way, make_room during admission is SAFE: the evictable mass always
    // covers the request, and slots can never ratchet in_use past budget.
    uint64_t eslot_pool = 0;
    uint32_t eslot_admits_pass = 0;   // per-token admission budget (storm guard)

    struct PendingRegion {
        uint8_t*              mem = nullptr;
        uint64_t              bytes = 0;
        std::vector<int32_t>  uniq;    // the routing this layout was built for
        std::vector<uint64_t> tags;    // outstanding reads, routed by pump()
        std::vector<size_t>   miss;    // region slots that came from DISK, for
                                       // honest bytes accounting and admission
    };
    std::unordered_map<const ggml_tensor*, PendingRegion> pending_regions;
    // Sibling lookup: "blk.7.ffn_gate_exps.weight" -> the up/down tensors.
    std::unordered_map<std::string, ggml_tensor*> by_name;

    // PHASE B: the unconditional-stream ring.
    //
    // 91% of bytes per token are the same addresses in the same order every token
    // -- a zero-branch FIFO stream read today as a stop-and-wait gather. The ring
    // is a fixed arena that segments stream through in graph order: produced ahead
    // by pump_ring() whenever a callback gives us the CPU, consumed by bring_in as
    // a lookup instead of a read. Consumption order equals production order, so a
    // FIFO byte-ring has no fragmentation and cannot deadlock: full ring, producer
    // waits; consumer (compute) drains it.
    //
    // The stream is CYCLIC -- after the last uncond tensor of a pass, the next
    // pass's first is already known -- so the producer wraps across token
    // boundaries and the drive need never idle. A stale segment is byte-correct by
    // definition (weights are immutable); desync costs duplicate reads, never
    // wrong data.
    struct RingSeg {
        ggml_tensor*          t = nullptr;   // null = wrap padding
        uint64_t              pos = 0;       // arena offset
        uint64_t              span = 0;      // widened to device alignment
        uint32_t              head = 0;      // file-alignment lead-in
        std::vector<uint64_t> tags;          // outstanding chunk reads
        bool                  consumed = false;
        bool                  complete = false;  // every chunk submitted
        // SUBMITTED IS NOT LANDED. `complete` says every chunk was handed to the
        // backend; it says nothing about whether the bytes arrived. Retention
        // promoted a segment into the cache on `complete` alone, so a short read
        // or an EIO -- which storage.h calls expected at this read volume, not
        // exceptional -- could install partly-unwritten arena bytes as a resident
        // weight and serve them as a RAM hit on every subsequent token, with no
        // failure counted and nothing tainted. This records the answer to the
        // question that actually matters (2026-08-24 audit).
        bool                  reads_ok = true;
    };
    uint8_t*  ring = nullptr;
    uint64_t  ring_bytes = 0;
    uint64_t  ring_head_off = 0;             // next placement offset
    uint64_t  ring_used = 0;                 // reserved, incl. wrap padding
    std::deque<RingSeg> ring_q;              // FIFO, front = oldest
    std::vector<ggml_tensor*> stream_list;   // streamed uncond tensors, graph order
    size_t    ring_cursor = 0;               // modular over stream_list
    bool      ring_active = false;
    // Why the producer stopped, per call site -- 64 ring-fed twice in a row with
    // two different theories is two theories too many. Indexed: 0=depth 1=lap
    // 2=full-at-wrap 3=full 4=submit-refused 5=produced-nothing-else.
    uint64_t  ring_stop[6] = {};
    uint64_t  ring_segs = 0, ring_pops = 0, ring_calls = 0;
    // What pass 1 ACTUALLY consumed, in consumption order. The stream list is
    // built from this, not from creation order: the GGUF carries tensors the graph
    // never references (blk.0.attn_res_score.weight on K3), and one such segment
    // at the FIFO front wedged the entire ring -- 0 pops in 136k calls.
    std::vector<ggml_tensor*>        consumed_order;
    std::unordered_set<ggml_tensor*> consumed_seen;
    uint32_t  pass_count = 0;                // forward passes seen (embed lookups)
    uint64_t  ring_hits = 0;

    ggml_backend_buffer_type buft{};
    void* fake_base = nullptr;

    // Shared landing zone for the loader's direct reads into streamed tensors,
    // sized to the largest single tensor in the model. The loader is sequential so
    // sharing is safe, and everything written here is discarded -- we already know
    // where those bytes live on disk.
    //
    // Poisoned rather than zeroed: if a weight is ever computed against without
    // being materialised, the result should be obviously broken rather than a
    // plausible-looking zero tensor.
    uint8_t* scratch = nullptr;
    // Grown lazily on a failure whose tensor exceeds scratch_bytes: the fast-load
    // scratch is slot-sized (~22 MB on K3), and pointing a 0.96 GB plain MUL_MAT
    // at it read 40x past the end (SIGSEGV in ggml_vec_dot on Linux, where the
    // smaller budget made the failure reachable). Raw OS allocation, outside the
    // cap ledger: a dying run may briefly overshoot the cap, which is the honest
    // price of never corrupting memory. Freed at teardown.
    uint8_t* emergency = nullptr;
    uint64_t emergency_bytes = 0;
    uint64_t scratch_bytes = 0;

    // PREFETCH.
    //
    // Reading a tensor and then computing with it leaves the drive idle for the
    // whole compute step: 206 GB moved in 126 s is 1.63 GB/s against a calibrated
    // 6.63, and the gap is not queue depth inside a read, it is the absence of any
    // overlap between reading and computing.
    //
    // Weight access is almost perfectly predictable -- the graph sweeps layers in
    // order, and every unconditional weight is touched once per token -- so the
    // next tensors can be read while the current node computes. `order` is the
    // sequence tensors were created in, which is model build order, i.e. layer
    // order; `cursor` is how far the current node has got through it.
    // Tensor names that arrived with no disk source, reported once each. See
    // init_tensor: a real weight here is uninitialised memory, not a curiosity.
    std::set<std::string> unsourced;

    std::vector<ggml_tensor*> order;
    size_t cursor = 0;
    uint64_t prefetch_hits = 0;    // needed and already in flight or resident
    uint64_t prefetch_issued = 0;

    // ONE registry for every read in flight, keyed by a globally unique tag.
    //
    // Prefetch and batched expert reads share a single completion queue, so they
    // must share a single tag space. An earlier version had batch reads tag
    // requests 0,1,2... and treat a completion's tag as an index into their own
    // array, while prefetch used a global counter: a prefetch completion arriving
    // during a batch poll was consumed as if it were a batch entry, and the waiter
    // for it then blocked forever. Low CPU, low disk, no progress.
    struct InFlight {
        uint8_t* stage = nullptr;
        uint64_t head = 0;
        uint32_t span = 0;
        void*    mem = nullptr;    // destination: slot for prefetch, caller's for batch
        uint64_t bytes = 0;
        bool     prefetch = false; // prefetch entries become resident on settle
        const ggml_tensor* owner = nullptr;   // prefetch only
        bool     done = false;
        bool     ok = false;
        uint64_t got = 0;      // bytes the device actually returned
        // Per-request service time. Four hypotheses about the missing bandwidth
        // died today for want of this number (2026-08-24).
        uint64_t t_submit_ns = 0;
        int      status = 0;
    };
    std::unordered_map<uint64_t, InFlight> pending;          // by tag
    std::unordered_map<const ggml_tensor*, uint64_t> prefetch_tag;
    uint64_t next_tag = 1;

    // Weights the node currently being computed depends on. Never evicted while
    // set. Replaces refcounting, which leaked whenever a materialise failed.
    std::vector<ggml_tensor*> current;   // mutable: bring_in rewrites ->data
    // Atomic: read cross-thread by /health's engine_tainted while the decode
    // thread increments (swarm S4). Relaxed is enough: monotonic flag semantics.
    std::atomic<uint64_t> failures{0};
    uint64_t compacted = 0;   // MUL_MAT_ID nodes served with only their selected experts
    uint32_t ring_idle_backoff = 0;   // S22: sleeps the producer after a fruitless lap

    // Tier 2: the ONE metal mapping over the pool arena. Metal refuses to
    // no-copy-wrap the same pages twice (nil buffer, and get_base crashes), so
    // the first alloc creates the master and every later alloc returns a view
    // sharing its context with free/clear neutered. Views outliving the master
    // is safe by construction: nothing executes after llama_model_free.
    ggml_backend_buffer_t metal_master = nullptr;

    // One contiguous VA arena behind alloc_acct: block sizes repeat (uniform
    // slot classes), commit tracks live bytes, and any future GPU backend
    // registers ONE range instead of chasing thousands of mallocs.
    mem::PoolArena pool;

    Impl(mem::Accountant& a, const plan::Plan& p, Config c) : acct(a), plan(p), cfg(c) {
        const char* v = std::getenv("DRAY_NO_POOL");
        if (!(v && v[0] == '1')) {
            // 2x cap: VA is free and exact-size reuse means fragmentation is
            // bounded, but 1x would make an unlucky size mix into a hard wall.
            pool.init(cfg.cap * 2);
        }
    }
};
// THE ONLY WAY THIS FILE ALLOCATES.
//
// Invariant 1 says every allocation goes through one accounted allocator, and it
// was not being honoured: the streamer tracked its own `in_use` against its own
// `budget` while the Accountant -- which owns the cap -- was told about the
// sentinel and nothing else. Slots and per-read staging were invisible to it, so
// the cap bound on our estimate of memory rather than on memory. Measured on
// Qwen3.8 at a 12 GiB cap: 13.57 GB private bytes.
//
// Reserving BEFORE allocating is the point. A reservation that fails means the
// caller must evict or wait, never proceed.
static bool maybe_repack(Streamer::Impl& im, ggml_tensor* t);   // defined below

static void* alloc_acct(Streamer::Impl& im, mem::Category cat, uint64_t bytes, uint32_t align) {
    if (bytes == 0) return nullptr;
    if (!im.acct.reserve(cat, bytes)) return nullptr;
    void* p = im.pool.alloc(bytes, align);
    if (!p) {
        // Pool disabled or VA exhausted: plain host memory is correct, merely
        // outside the registrable range. Counted, so a Metal-day audit sees it.
        if (im.pool.enabled()) im.pool.note_fallback();
        p = mem::aligned_alloc_host(static_cast<size_t>(bytes), align);
    }
    if (!p) im.acct.release(cat, bytes);
    return p;
}

static void free_acct(Streamer::Impl& im, mem::Category cat, void* p, uint64_t bytes) {
    if (!p) return;
    if (im.pool.contains(p)) im.pool.free(p, bytes);
    else                     mem::aligned_free_host(p, bytes);
    im.acct.release(cat, bytes);
}

// The cache gets whatever the cap has left after every non-cache category --
// floor, KV, compute scratch, and staging currently in flight. Derived rather than
// stored, so it cannot drift from the ledger the way a cached copy did.
// Live from the ledger, never mirrored. A local copy of this figure is updated at
// a different instant than the Accountant is -- when a destination slot is charged
// but its Resident entry not yet registered, the two differ by exactly that slot,
// and make_room then reports room while reservations are being refused.
static uint64_t cache_used(const Streamer::Impl& im) {
    return im.acct.used_in(mem::Category::ExpertCache);
}

static uint64_t cache_budget(const Streamer::Impl& im) {
    const uint64_t nc = im.acct.non_cache();
    const uint64_t cap = im.acct.cap();
    return cap > nc ? cap - nc : 0;
}

// How much of the cache budget may be held permanently. Derived for the same
// reason: the floor is charged during init_tensor, AFTER construction, so a value
// computed once would let pinning size itself against a ceiling that no longer
// exists. Router gates alone are 1.5-2.4 GB on these models.
static uint64_t static_allowance(const Streamer::Impl& im) {
    const uint64_t b = cache_budget(im);
    return b > im.churn_reserve ? b - im.churn_reserve : 0;
}


// Defined below; init_tensor needs it to source the floor from disk itself.
static bool read_exact(Streamer::Impl& im, const Source& s, void* dst, uint64_t bytes);

// ---------------------------------------------------------------------------
// buffer type / buffer callbacks

static Streamer::Impl* impl_of(ggml_backend_buffer_type_t buft) {
    return static_cast<Streamer::Impl*>(buft->context);
}
static Streamer::Impl* impl_of(ggml_backend_buffer_t buf) {
    return static_cast<Streamer::Impl*>(buf->context);
}

static const char* dray_buft_name(ggml_backend_buffer_type_t) { return "dray_stream"; }
static size_t dray_buft_alignment(ggml_backend_buffer_type_t) { return kBufAlign; }
static size_t dray_buft_max_size(ggml_backend_buffer_type_t) { return SIZE_MAX; }
// Host-accessible: the CPU backend must be able to compute directly against the
// pointers materialise() installs. Saying otherwise makes ggml_backend_sched treat
// every weight as living on a foreign backend and try to copy the whole model into
// a CPU buffer, which dies in graph_reserve.
//
// The cost of saying host is that the loader reads file bytes STRAIGHT into
// tensor->data, bypassing set_tensor. Streamed tensors therefore need a real
// writable landing zone during load -- see `scratch` -- which is shared by all of
// them because the loader is sequential and we discard what it writes.
static bool   dray_buft_is_host(ggml_backend_buffer_type_t) { return true; }

static void dray_buffer_free(ggml_backend_buffer_t) {}

static void* dray_buffer_get_base(ggml_backend_buffer_t buf) {
    // The allocator assigns tensor->data = base + offset before calling
    // init_tensor. We overwrite data there, so this address is never dereferenced
    // -- but it must be non-null and aligned or the allocator's arithmetic trips.
    return impl_of(buf)->fake_base;
}

// On the metal path the buffers carry Metal's context, not ours, so the hook
// resolves the engine through this stash (single engine per process by design;
// serve is one engine, run is one engine).
static Streamer::Impl* g_metal_impl = nullptr;

static enum ggml_status dray_buffer_init_tensor(ggml_backend_buffer_t buf, ggml_tensor* t) {
    Streamer::Impl* im = g_metal_impl ? g_metal_impl : impl_of(buf);

    // VIEWS FIRST. init_tensor is not only called at load: ggml_gallocr_init_tensor
    // calls ggml_backend_view_init for every view whose source lives in our buffer,
    // on EVERY graph allocation. That has already set
    //     t->data = t->view_src->data + t->view_offs
    // and then dispatches here -- so the old code overwrote a correct alias with
    // fresh uninitialised memory, and registered a permanently-pinned entry keyed
    // by a per-graph tensor pointer. Two consequences: the view stopped aliasing
    // its weight (reading uninitialised bytes, not even the 0xA5 poison, so it
    // failed plausibly rather than loudly), and resident entries accumulated every
    // decode -- 746 -> 751 -> 754 across successive runs, unaccounted against the
    // cap that Invariant 1 exists to hold.
    if (t->view_src != nullptr) {
        return GGML_STATUS_SUCCESS;
    }

    auto it = im->sources.find(t->name);
    if (it == im->sources.end()) {
        // Not a tensor we have a disk source for (llama.cpp creates a few of its
        // own). Give it real memory so it behaves normally.
        // NO DISK SOURCE. llama.cpp creates a few tensors of its own, and those are
        // fine here. A real WEIGHT reaching this path is silent corruption: it gets
        // memory and is never read, so on the no-read load path it holds whatever
        // the allocator returned. That is the "!!!! with zero reported failures"
        // failure mode, and nothing downstream can detect it.
        //
        // So say so, loudly and once per name. Anything listed here that looks like
        // a model weight is a classifier gap, not a curiosity.
        if (im->unsourced.insert(t->name).second) {
            std::fprintf(stderr, "[dray] NO SOURCE: %s (%.3f MB) -- allocated, never read\n",
                         t->name, ggml_nbytes(t) / 1e6);
            std::fflush(stderr);
        }
        void* m = alloc_acct(*im, mem::Category::Misc, ggml_nbytes(t), kBufAlign);
        t->data = m;
        Resident r; r.mem = m; r.bytes = ggml_nbytes(t); r.pinned = true;
        r.cat = mem::Category::Misc;
        im->resident[t] = r;
        return m ? GGML_STATUS_SUCCESS : GGML_STATUS_ALLOC_FAILED;
    }

    Source s = it->second;
    im->bound[t] = s;
    im->by_name[t->name] = t;   // sibling lookup for fused per-layer expert reads

    if (s.pinned) {
        // Mandatory floor: resident for the life of the run. Router gates in
        // particular MUST be resident -- a non-resident gate has to be read before
        // you know which experts to read, adding a serialized round trip per layer.
        // Floor tensors are charged to their real category, so the startup report
        // shows where a user's 4 GB actually went rather than lumping them in cache.
        const mem::Category fcat = s.is_router_gate ? mem::Category::RouterGates
                                                    : mem::Category::NormsAndBiases;
        void* m = alloc_acct(*im, fcat, ggml_nbytes(t), kBufAlign);
        if (!m) return GGML_STATUS_ALLOC_FAILED;

        // READ IT HERE, from disk, rather than waiting for set_tensor to hand us
        // bytes. This file's whole premise is that we know where every tensor lives
        // and fetch it ourselves; relying on set_tensor quietly made the floor
        // depend on llama.cpp reading the model first. On the no-read path nothing
        // calls set_tensor, so the router gates and norms held uninitialised memory
        // and the model emitted "!!!!" with zero reported failures -- silent
        // corruption produced by an inconsistency between two load paths.
        if (!read_exact(*im, s, m, ggml_nbytes(t))) {
            free_acct(*im, fcat, m, ggml_nbytes(t));
            ++im->failures;
            return GGML_STATUS_FAILED;
        }
        t->data = m;
        Resident r; r.mem = m; r.bytes = ggml_nbytes(t); r.pinned = true;
        r.cat = fcat;
        r.repacked = maybe_repack(*im, t);
        im->resident[t] = r;
        return GGML_STATUS_SUCCESS;
    }

    // Record creation order. llama.cpp builds tensors layer by layer, so this is
    // the order the graph will want them in, which is what prefetch walks ahead of.
    im->order.push_back(t);

    // Streamed: no storage now, and none until the node that needs it runs.
    //
    // data must still be NON-NULL: ggml_backend_tensor_set asserts
    // `tensor->data != NULL && "tensor not allocated"` before it ever reaches our
    // set_tensor, so a null here aborts the load. The sentinel is never
    // dereferenced -- our set_tensor drops the bytes, and materialise() installs a
    // real pointer before any node computes.
    t->data = im->scratch;
    return GGML_STATUS_SUCCESS;
}

static void dray_buffer_set_tensor(ggml_backend_buffer_t buf, ggml_tensor* t,
                                  const void* data, size_t offset, size_t size) {
    // Deliberately a no-op for EVERY tensor we own, on both load paths.
    //
    // We already know where every byte lives on disk: the floor was read in
    // init_tensor, and streamed tensors are read when a node needs them. Accepting
    // these writes would materialise the whole model, which is the thing this
    // project exists to avoid -- and treating the two load paths differently is
    // what produced uninitialised router gates on the no-read path.
    (void)buf; (void)t; (void)data; (void)offset; (void)size;
}

static void dray_buffer_get_tensor(ggml_backend_buffer_t buf, const ggml_tensor* t,
                                  void* data, size_t offset, size_t size) {
    Streamer::Impl* im = impl_of(buf);
    auto r = im->resident.find(t);
    if (r != im->resident.end() && r->second.mem) {
        std::memcpy(data, static_cast<const uint8_t*>(r->second.mem) + offset, size);
        return;
    }
    // Zeros presented as data, silently, was the worst available behaviour: any
    // llama.cpp read-back of a streamed (non-resident) weight got fabricated
    // bytes with no failure counted. Still zero-fill -- callers do not check --
    // but count it and say so once, so a run that hit this path can never again
    // look clean (2026-08-24 audit).
    ++im->failures;
    if (im->failures <= 8) {
        std::fprintf(stderr,
                     "[dray] GET_TENSOR MISS: %s read back before materialisation; "
                     "returning zeros and counting a failure\n",
                     t->name);
    }
    std::memset(data, 0, size);
}

static void dray_buffer_clear(ggml_backend_buffer_t, uint8_t) {}

#if defined(__APPLE__)
static void dray_metal_noop_clear(ggml_backend_buffer_t, uint8_t) {
    // A real clear would memset the whole mapping and commit the entire 2x-cap
    // reservation. Weights are never cleared on the metadata load path anyway.
}
static void dray_metal_noop_free(ggml_backend_buffer_t) {
    // Views share the master mapping's context; only the master frees it.
}
#endif

static ggml_backend_buffer_t dray_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
#if defined(__APPLE__)
    // TIER 2 (DRAY_METAL=1): hand llama a REAL Metal buffer mapping the pool
    // arena no-copy. Its buffer type is Metal's mapped type, so the scheduler
    // routes matmuls to the GPU; every pointer we later install is inside the
    // mapped range (that is what PoolArena and the arena unification are FOR),
    // so per-node containment resolution finds it. The buffer object needs two
    // lies to be safe: clear neutered (above), and size reported as what llama
    // asked for -- its layout math runs against a range we never let execute
    // before repointing. Falls through to the portable path when the env is
    // unset, the pool is off, or no GPU device exists (CPU-only build).
    {
        const char* mv = std::getenv("DRAY_METAL");
        Streamer::Impl* im = impl_of(buft);
        if (mv && mv[0] == '1' && im->pool.enabled()) {
            if (im->metal_master) {
                // A view: same context, same iface, no ownership.
                ggml_backend_buffer_t v = ggml_backend_buffer_init(
                    im->metal_master->buft, im->metal_master->iface,
                    im->metal_master->context, size);
                v->iface.free_buffer = dray_metal_noop_free;
                v->iface.clear = dray_metal_noop_clear;
                return v;
            }
            ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
            if (gpu) {
                ggml_backend_buffer_t b = ggml_backend_dev_buffer_from_host_ptr(
                    gpu, im->pool.base(),
                    static_cast<size_t>(im->pool.span()),
                    static_cast<size_t>(im->pool.span()));
                if (b) {
                    b->iface.clear = dray_metal_noop_clear;
                    b->iface.init_tensor = dray_buffer_init_tensor;
                    b->size = size;
                    im->metal_master = b;
                    g_metal_impl = im;
                    return b;
                }
            }
        }
    }
#endif
    static ggml_backend_buffer_i iface = {};
    iface.free_buffer   = dray_buffer_free;
    iface.get_base      = dray_buffer_get_base;
    iface.init_tensor   = dray_buffer_init_tensor;
    iface.set_tensor    = dray_buffer_set_tensor;
    iface.get_tensor    = dray_buffer_get_tensor;
    iface.clear         = dray_buffer_clear;
    return ggml_backend_buffer_init(buft, iface, impl_of(buft), size);
}

// ---------------------------------------------------------------------------

Streamer::Streamer(mem::Accountant& acct, const plan::Plan& p, Config cfg)
    : impl_(new Impl(acct, p, cfg)) {
    Impl& im = *impl_;

    im.io = io::make_backend(cfg.queue_depth);
    if (!im.io) { error_ = "no storage backend"; return; }

    for (const std::string& sp : p.shard_paths) {
        io::FileId f = im.io->open(sp);
        if (f == io::kInvalidFile) { error_ = "cannot open shard uncached: " + sp; return; }
        im.shards.push_back(f);
    }
    if (im.shards.empty()) { error_ = "plan carries no shards"; return; }
    // Pass-4b: the max across ALL shards, not shard 0's -- shards of one model
    // share a directory in practice, but Invariant 5 says discovered, not
    // assumed, and the conservative max is correct for every member.
    im.align = 0;
    for (io::FileId sf : im.shards) {
        const uint64_t a = im.io->alignment(sf).max();
        if (a > im.align) im.align = a;
    }
    if (im.align == 0) im.align = 4096;

    // Every tensor's true location, from the GGUF tensor table. Router gates and
    // norms are pinned; everything else streams, including attention and shared
    // experts -- at a 4 GB cap those do not fit either.
    for (const plan::TensorInfo& t : p.tensors) {
        Source s;
        s.shard  = t.shard;
        s.offset = t.offset;
        s.bytes  = t.bytes;
        s.pinned = (t.cls == plan::TensorClass::RouterGate ||
                    t.cls == plan::TensorClass::NormOrBias);
        s.is_router_gate = (t.cls == plan::TensorClass::RouterGate);
        s.disk_stride = t.disk_stride;
        // Routed experts are compacted; the token embedding is row-sliced. Neither
        // ever needs its full size resident, so neither bounds the minimum cap.
        s.sliceable = (t.cls == plan::TensorClass::RoutedExpert) ||
                      (t.name.find("token_embd") != std::string::npos);
        s.routed = (t.cls == plan::TensorClass::RoutedExpert);   // I2: class matters per-lever
        im.sources[t.name] = s;
    }

    im.fake_base = mem::aligned_alloc_host(1 << 16, kBufAlign);
    if (!im.fake_base) { error_ = "cannot allocate base sentinel"; return; }

    // Sentinel, not a landing zone. This was sized to the largest tensor (1.6 GiB
    // on Qwen3.8) only because llama.cpp's file loader wrote bytes straight into
    // tensor->data. On the metadata path the loader never reads or writes anything,
    // so this only has to be a non-null address satisfying ggml's "tensor not
    // allocated" assertions, plus a poison pattern.
    //
    // That 1.6 GiB is exactly what made a 4 GiB cap infeasible: on top of a ~2.3 GiB
    // floor it consumed the entire budget before a single weight could be cached.
    //
    // Still sized to the largest tensor when the fallback file path is selected,
    // because that path really does write here.
    // Big enough that a failed node reading slot 0 (see the zeroed private ids on
    // the failure path) stays inside it: one expert stride across gate/up/down,
    // with headroom. The slow-load path additionally needs it to be the LANDING
    // ZONE for llama.cpp's own reads, which must fit the largest whole tensor.
    uint64_t widest = 8ull << 20;
    for (const plan::LayerSlotClass& sc : p.slot_classes) {
        widest = std::max(widest, sc.slot_bytes * 2);
    }
    if (cfg.slow_load) {
        for (const plan::TensorInfo& t : p.tensors) widest = std::max(widest, t.bytes);
    }
    im.scratch_bytes = widest;
    // From the pool when possible: the poison scratch is READ BY COMPUTE on
    // failure paths, so on a GPU-mapped build its pointer must resolve inside
    // the one registered range. Raw host memory stays the fallback.
    im.scratch = static_cast<uint8_t*>(im.pool.alloc(im.scratch_bytes, kBufAlign));
    if (!im.scratch) {
        im.scratch = static_cast<uint8_t*>(
            mem::aligned_alloc_host(static_cast<size_t>(im.scratch_bytes), kBufAlign));
    }
    if (!im.scratch) { error_ = "cannot allocate sentinel"; return; }
    std::memset(im.scratch, 0xA5, static_cast<size_t>(im.scratch_bytes));
    // Pass-4b: the reserve result was DISCARDED -- a failed reservation left
    // the sentinel's bytes outside the ledger (Invariant 1). Near-dead path,
    // but honesty is not sized by reachability.
    if (!acct.reserve(mem::Category::IoStaging, im.scratch_bytes)) {
        if (im.pool.contains(im.scratch)) im.pool.free(im.scratch, im.scratch_bytes);
        else mem::aligned_free_host(im.scratch, im.scratch_bytes);
        im.scratch = nullptr;
        error_ = "cap cannot hold the sentinel buffer";
        return;
    }

    // The budget is whatever the cap has left after every non-cache category, read
    // live from the Accountant. It is NOT stored: the floor is charged during
    // init_tensor, long after this point, and a stored copy would not see it.
    if (cache_budget(im) == 0) { error_ = "cap leaves nothing for the paging cache"; return; }

    // Reserve churn space for the experts of the layer being computed, plus a
    // little slack, and let everything else be pinned permanently. One layer needs
    // n_expert_used slots across gate/up/down; the widest layer sets the bound.
    // Sized for what this mode actually materialises. With compaction that is
    // n_expert_used slots across gate/up/down; without it, whole routed tensors,
    // which on K3 are ~2 GB each. Reserving only the compacted figure while
    // materialising whole tensors starves eviction: static pinning fills the
    // budget, nothing is evictable, and the fallback fails.
    // Batch (measured 2026-08-19, the cap-independent single-failure at wide
    // distinct widths): a batched node's working set is the UNION of the
    // width's selections, E*(1-(1-k/E)^n_seq) experts, not n_expert_used --
    // sizing churn for one sequence let static pinning eat every budget at
    // every cap and starve the 2.5 GB regions batch actually materialises.
    // The formula reduces EXACTLY to n_expert_used at width 1.
    const uint32_t nsq = cfg.n_seq ? cfg.n_seq : 1;
    uint64_t uniq_w = p.n_expert_used;
    if (nsq > 1 && p.n_experts > 0 && p.n_expert_used > 0) {
        const double uE = static_cast<double>(p.n_experts);
        const double uk = static_cast<double>(p.n_expert_used);
        uniq_w = static_cast<uint64_t>(std::ceil(
            uE * (1.0 - std::pow(1.0 - uk / uE, static_cast<double>(nsq)))));
    }
    uint64_t churn = 0;
    for (const plan::LayerSlotClass& sc : p.slot_classes) {
        churn = std::max<uint64_t>(churn, sc.slot_bytes * uniq_w * 3ull);
        // The single widest union region: admission's empirical floor. B=38
        // died with budget at 0.71x this; B=32 ran clean at 2.07x.
        im.batch_region_bound = std::max<uint64_t>(im.batch_region_bound,
                                                   sc.slot_bytes * uniq_w);
    }
    if (cfg.no_compact) {
        for (const plan::TensorInfo& t : p.tensors) {
            if (t.cls == plan::TensorClass::RoutedExpert) churn = std::max(churn, t.bytes);
        }
    }
    churn = churn * 2 + (256ull << 20);   // two layers in flight, plus slack

    // AND at least the largest tensor that must be materialised WHOLE.
    //
    // Sizing the reserve only from expert slots makes this scale wrongly with the
    // cap: static pinning grows to fill whatever budget exists while the reserve
    // stays constant, so the evictable remainder shrinks as the cap RISES. K3 at
    // 32 GiB pinned 29.27 of 30.27 GB, leaving 0.05 GB against a 0.96 GB
    // output.weight -- a configuration that works at 12 GiB and fails at 32, which
    // is the opposite of what anyone would predict.
    //
    // Experts are compacted and embedding rows are sliced, so neither needs its
    // full size; everything else does, and output.weight is the biggest of them.
    uint64_t widest_whole = 0;
    for (const plan::TensorInfo& t : p.tensors) {
        if (t.cls == plan::TensorClass::RoutedExpert) continue;   // compacted
        if (t.cls == plan::TensorClass::RouterGate) continue;     // pinned floor
        if (t.cls == plan::TensorClass::NormOrBias) continue;     // pinned floor
        if (t.name.find("token_embd") != std::string::npos) continue;   // row-sliced
        widest_whole = std::max(widest_whole, t.bytes);
    }
    churn = std::max(churn, widest_whole + (widest_whole / 4));   // + headroom

    im.churn_reserve = churn;
    // Prefetch gets a slice of the churn reserve, never all of it: whatever it
    // holds is unreclaimable until its read lands, so the rest must stay free for
    // the node being computed.
    im.inflight_cap = churn / 2;

    // PHASE B ring: an accounted, fixed line item -- never carved from the cache at
    // runtime, which is what killed the first prefetch. Default 2x the widest whole
    // tensor (the lump must fit with lookahead room behind it), capped at a quarter
    // of the total cap; DRAY_RING_MB overrides, 0 disables. IoStaging category:
    // cache_budget() derives from cap minus non-cache, so the budget shrinks by
    // exactly this amount automatically.
    {
        // MEASURED, not derived: at an 8 GiB cap on K3, 768 MB/1280 MB/2048 MB rings
        // gave 24.5/24.5/25.0 s/tok and 528/533/539 GB. Bytes fall monotonically as
        // the ring shrinks (returned pinning) while overlap does not degrade -- the
        // widest tensor simply takes the sync path via the span > ring/2 guard, once
        // per token. So the ring should be SMALL: cap/8, floored at 512 MB. The old
        // cap/4 default cost ~1.3 GB/token for nothing.
        uint64_t want = cfg.cap / 8;
        if (want < (512ull << 20)) want = 512ull << 20;
        if (want > widest_whole * 2 + (256ull << 20)) want = widest_whole * 2 + (256ull << 20);

        // INVERSE scaling near the knee. The ring is a flow-through window sized by
        // bandwidth x stall, not by RAM -- but it must also never exceed the stream
        // it serves. As the cap approaches the unconditional set, pinning absorbs
        // the stream and the remainder shrinks; a 2.2 GB window over a ~3 GB/token
        // stream just displaces ~1.7 GB/token of pinning (measured, K3 at 56 GiB).
        // So cap the ring at the estimated streamed remainder.
        {
            uint64_t uncond = 0;
            for (const plan::TensorInfo& t : p.tensors) {
                if (t.cls != plan::TensorClass::UnconditionalBulk) continue;
                if (t.name.find("token_embd") != std::string::npos) continue;
                uncond += t.bytes;
            }
            const uint64_t budget_est =
                cfg.cap > p.floor.total() ? cfg.cap - p.floor.total() : 0;
            const uint64_t pinnable = budget_est > churn ? budget_est - churn : 0;
            const uint64_t remainder = uncond > pinnable ? uncond - pinnable : 0;
            if (want > remainder) {
                want = remainder < (512ull << 20) ? (512ull << 20) : remainder;
            }
        }

        if (const char* v = std::getenv("DRAY_RING_MB")) {
            want = std::strtoull(v, nullptr, 10) << 20;
        }
        if (want > cfg.cap / 4) want = cfg.cap / 4;
        want = (want / im.align) * im.align;
        if (want >= (64ull << 20)) {
            im.ring = static_cast<uint8_t*>(
                alloc_acct(im, mem::Category::IoStaging, want, im.align));
            if (im.ring) im.ring_bytes = want;
        }

        // Slot pool: the past-knee surplus, sized here and never renegotiated.
        {
            uint64_t uncond_total = 0;
            for (const plan::TensorInfo& ti : p.tensors) {
                if (ti.cls != plan::TensorClass::UnconditionalBulk) continue;
                if (ti.name.find("token_embd") != std::string::npos) continue;
                uncond_total += ti.bytes;
            }
            uint64_t routed_whole = 0;
            for (const plan::TensorInfo& ti : p.tensors) {
                if (ti.cls == plan::TensorClass::RoutedExpert && ti.bytes > routed_whole)
                    routed_whole = ti.bytes;
            }
            const uint64_t be =
                cfg.cap > p.floor.total() ? cfg.cap - p.floor.total() : 0;
            // Reserve the whole-expert FALLBACK too: churn_reserve excludes routed
            // tensors from its widest scan, and a compact failure falls back to
            // materialising one whole (2.5 GB on K3) -- slots must never eat that
            // headroom (measured: in_use 54.87/54.89, 0.67 GB evictable, refused).
            const uint64_t reserved = uncond_total + churn + im.ring_bytes +
                                      routed_whole + routed_whole / 4;
            // Batch: eslot admission is single-stream-only (measured, see the
            // admission site), so a committed pool under batch is budget that
            // can never earn a hit -- and at caps where the unconditional set
            // fully fits, that dead commitment strangled make_room (M3 28G
            // B=16: budget 12.30, in_use 12.21, need 0.70, nothing evictable;
            // NO_ESLOTS cleared it). No width, no pool.
            im.eslot_pool = cfg.n_seq > 1 ? 0
                          : (be > reserved ? be - reserved : 0);
        }
    }
    // Static pinning takes everything except the current layer's reserve. It does
    // NOT set aside room for prefetch: doing so cost 1.3 GB of pinning and made the
    // run slower on both time and bytes (see start_prefetch). Pinning wins below the
    // knee; prefetch gets whatever genuinely remains free, which is little here and
    // a lot once the cap clears the knee.
    // static_allowance() derives from the same ledger; nothing to assign here.

    im.buft.iface.get_name       = dray_buft_name;
    im.buft.iface.alloc_buffer   = dray_buft_alloc;
    im.buft.iface.get_alignment  = dray_buft_alignment;
    im.buft.iface.get_max_size   = dray_buft_max_size;
    im.buft.iface.get_alloc_size = nullptr;   // defaults to ggml_nbytes
    im.buft.iface.is_host        = dray_buft_is_host;
    im.buft.device  = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    // A: opt this buffer type into the optimised matmul kernels. We guarantee
    // the bytes behind matching tensors are repacked (see maybe_repack).
    // MEASURED NEGATIVE (2026-08-22): repacking on the streaming path costs
    // ~20% (3.85 vs 3.2 s/tok on a 27B, two runs each) because the transform
    // is paid on EVERY materialisation while only 74 of 545 2D tensors in a
    // UD quant even have an optimal repack type. OFF by default; enable with
    // DRAY_REPACK=1 to reproduce. The machinery stays because install-time
    // repacking (pay once, on disk) is the design that would actually win.
    im.repack_on = [] {
        const char* v = std::getenv("DRAY_REPACK");
        return v && v[0] == '1';
    }();
    if (im.repack_on) ggml_cpu_repack_accept_buft(&im.buft);
    im.buft.context = &im;
}

// A: rewrite a freshly-read whole tensor into the interleaved layout the
// optimised CPU kernels want. Only 2D weights (GGML_OP_MUL_MAT); the 3D
// expert slabs that mul_mat_id consumes carry our own slot stride and are left
// alone until that path is proven separately. Returns true if it repacked.
static bool maybe_repack(Streamer::Impl& im, ggml_tensor* t) {
    if (!im.repack_on || !t || !t->data) return false;
    if (ggml_n_dims(t) != 2) return false;
    ++im.repack_tried;
    // The kernels read tensor->extra to find their traits; a buffer that
    // repacks must set it, or dispatch silently declines and the repack is
    // wasted work (measured: 2.8 vs 2.6 s/tok, i.e. nothing, before this).
    void* traits = ggml_cpu_repack_traits_for(t);
    if (!traits) { ++im.repack_no_traits; return false; }
    if (!ggml_cpu_repack_data_in_place(t, t->data, ggml_nbytes(t))) return false;
    t->extra = traits;
    ++im.repack_done;
    return true;
}

static bool settle_tags(Streamer::Impl& im, std::vector<uint64_t>& tags,
                        bool* backend_dead = nullptr);
static uint8_t* poison_for(Streamer::Impl& im, const ggml_tensor* t);

Streamer::~Streamer() {
    if (!impl_) return;
    Impl& im = *impl_;
    // Teardown frees route by origin: pool blocks go back to the pool (which
    // dies with Impl anyway), plain blocks to the host allocator. Freeing a
    // pool block with the host deallocator is heap corruption -- the sibling-
    // path rule again.
    auto free_any = [&im](void* p, uint64_t bytes) {
        if (im.pool.contains(p)) im.pool.free(p, bytes);
        else                     mem::aligned_free_host(p, bytes);
    };
    for (auto& kv : im.resident) {
        if (kv.second.mem) free_any(kv.second.mem, kv.second.bytes);
    }
    // Pending sibling regions have DMA in flight into their memory: every tag must
    // complete before the buffer may be released, success or not.
    for (auto& kv : im.pending_regions) {
        // Pass-4b teardown: a FAILED settle means the backend died with reads
        // outstanding into this region -- the F2 rule applies in the dtor too:
        // leak the memory, never free under DMA, even on the way out.
        bool dead_pr = false;
        const bool ok_pr = settle_tags(im, kv.second.tags, &dead_pr);
        // H15: a completed-but-FAILED read is fully landed -- freeing is safe;
        // only a dead backend (reads still in flight) forces the leak.
        if (ok_pr || !dead_pr) {
            if (kv.second.mem) free_any(kv.second.mem, kv.second.bytes);
        } else if (kv.second.mem) {
            std::fprintf(stderr, "[dray] FATAL: teardown leaking a pending region "
                                 "(backend died with reads outstanding)\n");
        }
    }
    im.pending_regions.clear();
    // Same rule for the ring: outstanding chunk DMA lands in the arena.
    bool ring_dead = false;
    for (auto& sg : im.ring_q) {
        bool d1 = false;
        settle_tags(im, sg.tags, &d1);
        ring_dead = ring_dead || d1;
    }
    const bool ring_clean = !ring_dead;   // H15: leak only on a genuinely dead backend
    im.ring_q.clear();
    if (im.ring && ring_clean) free_any(im.ring, im.ring_bytes);
    else if (im.ring) std::fprintf(stderr, "[dray] FATAL: teardown leaking the ring "
                                           "(backend died with reads outstanding)\n");
    if (im.emergency) free_any(im.emergency, im.emergency_bytes);
    for (auto& tw : im.eslots) {
        for (auto& kv : tw.second) {
            if (kv.second.mem) free_any(kv.second.mem, kv.second.bytes);
        }
    }
    im.eslots.clear();
    // Compact expert regions live in `resident` like any other materialised
    // tensor and were freed by the loop above; there is no separate pool.
    if (im.fake_base) mem::aligned_free_host(im.fake_base, 1 << 16);
    if (im.scratch && im.pool.contains(im.scratch)) im.pool.free(im.scratch, im.scratch_bytes);
    else if (im.scratch) mem::aligned_free_host(im.scratch, im.scratch_bytes);
    if (im.io) {
        for (io::FileId f : im.shards) im.io->close(f);
    }
}

ggml_backend_buffer_type_t Streamer::buft() { return &impl_->buft; }

// Reads [offset, offset+bytes) of a shard into dst, widening to the device's
// alignment and slicing the interior out. GGUF aligns tensor data to
// general.alignment (32 by default), which is never the device's I/O alignment.
// Defined below read_batch: a single-slice batch, so it shares the one tag space
// and the one completion dispatcher. It previously submitted with tag 0 and polled
// the queue itself, which meant a prefetch completion arriving here was consumed
// and lost, and its waiter blocked forever.

// Evicts least-recently-used unpinned, unreferenced tensors until `need` fits.
// Reads many slices as ONE batch: all requests submitted before any is waited on,
// so the drive sees a deep queue instead of a stream of single round trips.
//
// This is the difference between the calibrated QD1 and QD16 figures on this drive
// -- 0.5-3.4 GB/s against 6.6 -- and it is why the engine was achieving ~1.5 GB/s
// at 25% utilisation while the drive was idle waiting for the next request. An
// earlier version read one expert at a time, so K3 (top-16 across gate/up/down)
// issued 48 serialised round trips per layer.
struct Slice {
    Source   src;
    uint8_t* dst = nullptr;
    uint64_t len = 0;
};

// Harvests whatever has completed and routes each completion to its owner by tag.
// The ONLY place completions are consumed, so no caller can swallow another's.
static bool make_room(Streamer::Impl& im, uint64_t need);
static bool settle_tags(Streamer::Impl& im, std::vector<uint64_t>& tags,
                        bool* backend_dead);

static size_t pump(Streamer::Impl& im, size_t min_complete) {
    if (im.io->in_flight() == 0) return 0;
    {
        const uint64_t now = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        if (im.pump_last_ns) {
            const uint64_t g = (now - im.pump_last_ns) / 1000;
            im.pump_gap_us += g;
            im.gap_bucket[g < 100 ? 0 : g < 1000 ? 1 : g < 5000 ? 2 : g < 20000 ? 3 : 4]++;
            if (g > im.gap_max_us) im.gap_max_us = g;
            if (g >= 5000) im.gap_long_us += g;
        }
        im.pump_last_ns = now;
        ++im.pump_calls;
    }
    std::vector<io::Completion> comps(64);
    const auto pp0 = std::chrono::steady_clock::now();
    const size_t got = im.io->poll(comps.data(), comps.size(), min_complete);
    im.in_pump_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - pp0).count();
    im.pump_harvested += got;
    for (size_t i = 0; i < got; ++i) {
        auto it = im.pending.find(comps[i].tag);
        if (it == im.pending.end()) continue;   // already settled
        Streamer::Impl::InFlight& p = it->second;
        p.done = true;
        if (p.t_submit_ns) {
            const uint64_t now_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            const uint64_t us = (now_ns - p.t_submit_ns) / 1000;
            im.lat_sum_us += us; ++im.lat_n; im.lat_bytes += p.span;
            im.lat_bucket[us < 200 ? 0 : us < 500 ? 1 : us < 1000 ? 2 : us < 2000 ? 3 : 4]++;
        }
        // Short at EOF is legal: the span is widened to device alignment and the
        // last tensor in a shard runs past the end. The interior must be covered.
        p.ok = comps[i].status == 0 &&
               static_cast<uint64_t>(comps[i].bytes) >= p.head + p.bytes;
        // Keep what the device actually returned, so a failure can report the short
        // read rather than just asserting one happened.
        p.got    = static_cast<uint64_t>(comps[i].bytes);
        p.status = comps[i].status;
        if (p.ok && p.mem) {
            const auto c0 = std::chrono::steady_clock::now();
            std::memcpy(p.mem, p.stage + p.head, static_cast<size_t>(p.bytes));
            im.memcpy_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - c0).count();
            im.memcpy_bytes += p.bytes;
        }
    }
    return got;
}

// Registers one read and submits it. Returns 0 if it could not be started.
static uint64_t submit_one(Streamer::Impl& im, const Source& s, void* dst, uint64_t bytes,
                           bool prefetch, const ggml_tensor* owner) {
    if (s.shard < 0 || static_cast<size_t>(s.shard) >= im.shards.size()) return 0;
    const io::FileId f = im.shards[static_cast<size_t>(s.shard)];
    const io::Alignment al = im.io->alignment(f);
    const uint64_t lo = (s.offset / al.offset) * al.offset;
    const uint64_t head = s.offset - lo;
    uint64_t span = head + bytes;
    span = ((span + al.length - 1) / al.length) * al.length;

    // Staging counts against the cap like anything else -- it was never accounted,
    // and with a deep queue it is not small. But unlike cache, staging is TRANSIENT
    // AND MANDATORY: without it no read can proceed, so a refused reservation must
    // reclaim rather than fail. Once accounting was switched on, reads began
    // failing outright at the cap with the cache still holding 9.89 of 10.21 GB.
    const auto sa0 = std::chrono::steady_clock::now();
    uint8_t* stage = static_cast<uint8_t*>(
        alloc_acct(im, mem::Category::IoStaging, span, al.memory));
    im.stage_alloc_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - sa0).count();
    ++im.stage_alloc_n;
    if (!stage) {
        // T23: an earlier version pumped completions here hoping "completed
        // reads still holding staging" would free room. They cannot: pump()
        // only marks entries done -- staging is freed by each tag's OWNER
        // (settle/discard), none of which can run between iterations of a
        // loop on this single-threaded path. The ledger state was identical
        // on every retry, so each pass was a blocking poll for nothing.
        // Evicting cache is the one lever that actually moves the sum.
        if (make_room(im, span)) {
            stage = static_cast<uint8_t*>(
                alloc_acct(im, mem::Category::IoStaging, span, al.memory));
        }
    }
    if (!stage) return 0;

    const uint64_t tag = im.next_tag++;
    io::ReadRequest r{f, lo, static_cast<uint32_t>(span), stage, tag};
    if (im.io->submit(&r, 1) != 1) {
        free_acct(im, mem::Category::IoStaging, stage, span);
        return 0;
    }
    // TRUE queue depth, sampled where the drive sees it. read_batch is NOT the
    // place to measure this: submit_layer_siblings issues the other two tensors
    // of a trio outside it, unwaited, so counting slices per read_batch call
    // undercounts what is actually in flight (2026-08-24).
    {
        const size_t nf = im.io->in_flight();
        {
            const uint64_t t = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            if (im.depth_t_last) im.depth_area_ns += im.depth_cur * (t - im.depth_t_last);
            else im.depth_t0 = t;
            im.depth_t_last = t;
            im.depth_cur = nf;
        }
        im.depth_sum += nf;
        ++im.depth_n;
        if (nf > im.depth_max) im.depth_max = nf;
        if (nf < im.depth_min) im.depth_min = nf;
        if (nf < 4) ++im.depth_low;
    }

    im.pending[tag] = {stage, head, static_cast<uint32_t>(span), dst, bytes,
                       prefetch, owner, false, false, 0,
                       (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch()).count()};
    return tag;
}

static void discard(Streamer::Impl& im, uint64_t tag) {
    auto it = im.pending.find(tag);
    if (it == im.pending.end()) return;
    free_acct(im, mem::Category::IoStaging, it->second.stage, it->second.span);
    im.pending.erase(it);
}

// Reads many slices as ONE batch: every request submitted before any is waited on,
// so the drive sees a deep queue instead of a stream of single round trips. That
// is the calibrated difference between QD1 (0.5-3.4 GB/s) and QD16 (6.6).
static bool read_batch(Streamer::Impl& im, const std::vector<Slice>& slices) {
    if (slices.empty()) return true;
    ++im.io_batches;
    im.io_slices += slices.size();
    if (slices.size() > im.io_batch_max) im.io_batch_max = slices.size();

    std::vector<uint64_t> tags;
    tags.reserve(slices.size());
    bool failed = false;

    for (const Slice& sl : slices) {
        // Respect the backend's depth: beyond it submit() refuses, so drain first.
        while (im.io->in_flight() >= im.io->max_in_flight()) {
            if (pump(im, 1) == 0) { failed = true; break; }
        }
        if (failed) break;
        const uint64_t tag = submit_one(im, sl.src, sl.dst, sl.len, false, nullptr);
        if (tag == 0) { failed = true; break; }
        tags.push_back(tag);
    }

    // F2: settle ALL submitted tags UNCONDITIONALLY before any discard --
    // success or failure, every submitted read has DMA outstanding into its
    // staging buffer, and discard() frees that buffer for immediate reuse.
    // The old shape skipped this wait when `failed` was set mid-submission,
    // freeing under DMA (the no-cancel contract, storage.h and D8; T23's
    // deletion of submit_one's drain loop made the path reachable). A failed
    // batch still waits for its own reads; only the RESULT is failed.
    for (;;) {
        bool outstanding = false;
        for (uint64_t t : tags) {
            auto it = im.pending.find(t);
            if (it != im.pending.end() && !it->second.done) { outstanding = true; break; }
        }
        if (!outstanding) break;
        if (pump(im, 1) == 0) {
            // Backend dead with reads outstanding. Freeing under possible DMA
            // trades a loud failure for silent corruption, so the staging is
            // LEAKED deliberately: pending entries are dropped without
            // free_acct, the bytes stay charged to IoStaging forever, and the
            // ledger honestly shows the cost of a dead backend.
            std::fprintf(stderr,
                         "[dray] FATAL: backend dead with reads outstanding; "
                         "leaking %zu staging buffers (no-cancel contract)\n",
                         tags.size());
            for (uint64_t t : tags) im.pending.erase(t);
            return false;
        }
    }

    for (uint64_t t : tags) {
        auto it = im.pending.find(t);
        if (it == im.pending.end() || !it->second.ok) failed = true;
        discard(im, t);
    }
    return !failed;
}

static bool read_exact(Streamer::Impl& im, const Source& s, void* dst, uint64_t bytes) {
    const std::vector<Slice> one{{s, static_cast<uint8_t*>(dst), bytes}};
    return read_batch(im, one);
}

// Reads a whole tensor with NO staging buffer and NO copy.
//
// The destination is allocated at the ALIGNMENT-WIDENED span and read into
// directly; the tensor then points at base+head. The padding costs at most two
// alignment units (~8 KB) against a tensor of up to 1.6 GB.
//
// Staging a whole tensor meant holding it twice -- output.weight is 1.14 GB, so a
// read needed 2.28 GB of transient space against a 0.85 GB churn reserve, and
// simply could not be satisfied at this cap. It also copied every byte read, which
// is pure memory bandwidth on the hot path for no benefit.
//
// Compaction still stages: its slices must land contiguously at an exact stride,
// and arbitrary file offsets cannot be made to line up with that. Those slices are
// megabytes, not gigabytes.
static void* alloc_and_read_direct(Streamer::Impl& im, mem::Category cat,
                                   const Source& s, uint64_t bytes,
                                   uint64_t* out_alloc, uint32_t* out_head) {
    auto bail = [&](const char* why) -> void* {
        if (im.failures < 8) {
            std::fprintf(stderr, "[dray] DIRECT %s: shard=%d bytes=%llu inflight=%zu/%zu\n",
                         why, s.shard, (unsigned long long)bytes,
                         im.io->in_flight(), im.io->max_in_flight());
            std::fflush(stderr);
        }
        return nullptr;
    };

    if (s.shard < 0 || static_cast<size_t>(s.shard) >= im.shards.size()) return bail("BAD SHARD");
    const io::FileId f = im.shards[static_cast<size_t>(s.shard)];
    const io::Alignment al = im.io->alignment(f);

    const uint64_t lo = (s.offset / al.offset) * al.offset;
    const uint64_t head = s.offset - lo;
    uint64_t span = head + bytes;
    span = ((span + al.length - 1) / al.length) * al.length;

    uint8_t* base = static_cast<uint8_t*>(alloc_acct(im, cat, span, al.memory));
    if (!base) return bail("ALLOC");

    // The op pool IS the queue-depth gate, so a full pool makes submit refuse. Drain
    // and retry rather than failing the read: the staging path already does this,
    // and a transient queue-full is not a reason to abandon a weight the graph needs.
    uint64_t tag = 0;
    for (int attempt = 0; attempt < 64; ++attempt) {
        tag = im.next_tag++;
        io::ReadRequest r{f, lo, static_cast<uint32_t>(span), base, tag};
        // stage and mem both null: pump has nothing to copy, discard nothing to free.
        im.pending[tag] = {nullptr, head, static_cast<uint32_t>(span),
                           nullptr, bytes, false, nullptr, false, false};
        if (im.io->submit(&r, 1) == 1) break;
        im.pending.erase(tag);
        tag = 0;
        if (im.io->in_flight() == 0) break;   // not queue pressure; genuinely refused
        if (pump(im, 1) == 0) break;
    }
    if (tag == 0) {
        free_acct(im, cat, base, span);
        return bail("SUBMIT");
    }
    bool backend_dead = false;
    for (;;) {
        auto it = im.pending.find(tag);
        if (it == im.pending.end() || it->second.done) break;
        // Pass-4b: a dead backend here means the read may STILL be writing into
        // `base` -- fabricating done/ok and freeing below was the third
        // free-under-DMA site of the F2 class (the first two were fixed, this
        // one was not enumerated). Same remedy: leak deliberately and loudly.
        if (pump(im, 1) == 0) { backend_dead = true; break; }
    }
    if (backend_dead) {
        std::fprintf(stderr,
                     "[dray] FATAL: backend dead with a direct read outstanding; "
                     "leaking its staging (no-cancel contract)\n");
        im.pending.erase(tag);
        return nullptr;
    }
    auto it = im.pending.find(tag);
    const bool ok = (it != im.pending.end()) && it->second.ok;
    if (!ok && im.failures < 8) {
        // Name the reason. "read failed" without the numbers has cost hours on this
        // codebase every time; status and the short-read arithmetic identify it at
        // a glance.
        std::fprintf(stderr,
            "[dray] DIRECT READ FAIL shard=%d off=%llu lo=%llu head=%llu bytes=%llu "
            "span=%u got=%lld status=%d\n",
            s.shard, (unsigned long long)s.offset, (unsigned long long)lo,
            (unsigned long long)head, (unsigned long long)bytes, (unsigned)span,
            it == im.pending.end() ? -1LL : (long long)it->second.got,
            it == im.pending.end() ? -1 : it->second.status);
        std::fflush(stderr);
    }
    im.pending.erase(tag);
    if (!ok) { free_acct(im, cat, base, span); return nullptr; }

    *out_alloc = span;
    *out_head  = static_cast<uint32_t>(head);
    return base;
}

// Evicts by PRIORITY first, recency second: every routed-expert region goes before
// any unconditional weight is touched. Without this the cache spends its budget on
// bytes worth 0.02 reads/token while evicting bytes worth 1.0.
static bool make_room(Streamer::Impl& im, uint64_t need) {
    for (int pass = 0; pass < 2 && cache_used(im) + need > cache_budget(im); ++pass) {
        const Prio evictable = (pass == 0) ? Prio::RoutedExpert : Prio::Unconditional;
    while (cache_used(im) + need > cache_budget(im) && !im.lru.empty()) {
        bool evicted = false;
        for (auto it = im.lru.rbegin(); it != im.lru.rend(); ++it) {
            auto r = im.resident.find(*it);
            if (r == im.resident.end()) continue;
            if (r->second.pinned) continue;
            if (r->second.prio != evictable) continue;   // lower-value class first
            if (std::find(im.current.begin(), im.current.end(), *it) != im.current.end()) {
                continue;  // needed by the node being computed right now
            }
            free_acct(im, r->second.cat, r->second.mem, r->second.bytes);
            const ggml_tensor* victim = *it;
            im.resident.erase(r);
            im.lru.erase(std::next(it).base());
            // Back to the sentinel, not null: ggml asserts on a null data pointer
            // in several places, and an evicted tensor may still be inspected
            // before it is next materialised.
            const_cast<ggml_tensor*>(victim)->data = im.scratch;
            evicted = true;
            break;
        }
        if (!evicted) break;   // nothing left in this class; try the next one
    }
    }
    return cache_used(im) + need <= cache_budget(im);
}

// Waits for a specific prefetched tensor, then turns it into a resident entry.
static bool settle_prefetch(Streamer::Impl& im, ggml_tensor* t) {
    auto pt = im.prefetch_tag.find(t);
    if (pt == im.prefetch_tag.end()) return false;
    const uint64_t tag = pt->second;

    for (;;) {
        auto it = im.pending.find(tag);
        if (it == im.pending.end()) { im.prefetch_tag.erase(pt); return false; }
        if (it->second.done) break;
        if (pump(im, 1) == 0) { it->second.done = true; it->second.ok = false; break; }
    }

    auto it = im.pending.find(tag);
    const bool ok = it != im.pending.end() && it->second.ok;
    void* mem = (it != im.pending.end()) ? it->second.mem : nullptr;
    const uint64_t bytes = (it != im.pending.end()) ? it->second.bytes : 0;
    discard(im, tag);
    im.prefetch_tag.erase(pt);
    im.inflight_bytes -= std::min(im.inflight_bytes, bytes);

    if (!ok) {
        if (mem) { free_acct(im, mem::Category::ExpertCache, mem, bytes); }
        t->data = poison_for(im, t);
        return false;
    }
    Resident nr; nr.mem = mem; nr.bytes = bytes; nr.pinned = false;
    nr.prio = Prio::Unconditional;
    if (im.static_used + bytes <= static_allowance(im)) {
        nr.pinned = true;
        im.static_used += bytes;
    }
    t->data = mem;
    nr.repacked = maybe_repack(im, t);
    im.resident[t] = nr;
    im.lru.push_front(t);
    return true;
}

// Starts a read for `t` without waiting. Returns false if it could not be started,
// which is not an error -- the tensor is simply read synchronously when needed.
static bool start_prefetch(Streamer::Impl& im, ggml_tensor* t) {
    if (!t || im.resident.count(t) || im.prefetch_tag.count(t)) return false;
    auto b = im.bound.find(t);
    if (b == im.bound.end()) return false;
    if (im.io->in_flight() >= im.io->max_in_flight()) return false;

    const uint64_t bytes = ggml_nbytes(t);

    // PREFETCH NEVER EVICTS, which makes it inactive below the knee. Measured, not
    // assumed. Qwen3.8 UD-IQ1_S, 12 GiB cap, 4 tokens:
    //
    //   batched only             31.5 s/tok   206.3 GB
    //   + prefetch, no evict     35.2 s/tok   208.8 GB   304/304 hits
    //   + prefetch, may evict    36.7 s/tok   211.5 GB   673/673 hits
    //
    // The hit rate is perfect and it loses on BOTH axes. Every byte prefetch holds
    // is a byte not permanently pinned, and the +5.2 GB matches exactly the 1.3 GB
    // of lost pinning re-read across 4 tokens. Below the knee the cache is far
    // smaller than the working set, so there is no cold data to displace: holding a
    // byte beats overlapping a read.
    //
    // The mechanism stays because it is correct and costs nothing when it cannot
    // run: above the knee, where slack exists, it should pay. It self-limits --
    // this guard is simply unsatisfiable while the cache is saturated.
    if (cache_used(im) + bytes + im.churn_reserve > cache_budget(im)) return false;
    if (im.inflight_bytes + bytes > im.inflight_cap) return false;

    void* mem = alloc_acct(im, mem::Category::ExpertCache, bytes, kBufAlign);
    if (!mem) return false;

    const uint64_t tag = submit_one(im, b->second, mem, bytes, true, t);
    if (tag == 0) { free_acct(im, mem::Category::ExpertCache, mem, bytes); return false; }

    im.inflight_bytes += bytes;
    im.prefetch_tag[t] = tag;
    ++im.prefetch_issued;
    return true;
}

// Everything the ring will stream, in graph order: bound, not floor, not
// row-sliced, not an expert tensor, not statically pinned during pass 1. Built at
// the SECOND forward pass, once pinning has claimed its prefix -- the split
// between pinned and streamed is settled by then and never changes.
static void build_stream_list(Streamer::Impl& im) {
    im.stream_list.clear();
    for (ggml_tensor* t : im.consumed_order) {
        auto b = im.bound.find(t);
        if (b == im.bound.end()) continue;
        if (b->second.pinned || b->second.sliceable) continue;
        if (t->ne[2] > 1) continue;              // routed experts: compacted path
        // NO resident filter. The producer skips resident tensors per lap, which
        // adapts every cycle; filtering at build time forced dynamic admission to
        // exist, and admission APPENDED tensors out of consumption order -- the
        // producer then tripped its lap check chronically and the drive idled.
        // Measured: 19.9 -> 28.9 s/tok at n=16 from that one commit, bytes
        // identical. Order is the ring's load-bearing invariant; nothing may
        // insert into the list except this build, in consumed_order.
        im.stream_list.push_back(t);
    }
    im.ring_cursor = 0;
    im.ring_active = !im.stream_list.empty();
    std::fprintf(stderr, "[dray] RING stream_list=%zu of consumed=%zu (pass %u)\n",
                 im.stream_list.size(), im.consumed_order.size(), im.pass_count);
    std::fflush(stderr);
}

// The producer. Reclaims consumed segments from the front, then submits ahead in
// stream order until the queue is deep or the ring is full. Called from the
// callbacks (end of materialise, release): between callbacks the kernel keeps the
// already-submitted DMA flowing, and callbacks are milliseconds apart.
static void pump_ring(Streamer::Impl& im) {
    if (!im.ring || !im.ring_active) return;
    // S22: once the cache has absorbed the whole stream, every call walked the
    // entire stream_list to discover nothing streamable -- thousands of no-op
    // scans per token. A fruitless FULL lap arms a 256-call backoff; after an
    // eviction the producer wakes at most one backoff late and streams the
    // same bytes it always would (difftest-gated: byte-identical).
    if (im.ring_idle_backoff > 0) { --im.ring_idle_backoff; return; }
    ++im.ring_calls;

    // Free the front -- but NEVER a segment the node being computed right now is
    // using: its compute runs after this materialise returns. One-node batching
    // guarantees the next materialise call means the previous compute finished.
    //
    // RETENTION (consume != free): before a consumed segment's bytes are recycled,
    // promote them into the cache IF the budget has genuinely free room -- never
    // by evicting, and never into the churn reserve. Above the knee this converts
    // the ring's one pass over the churn set into residency, and the producer's
    // resident-check then stops re-streaming those tensors: the ring self-empties
    // as the cache absorbs its stream. At tight caps the room test fails on every
    // call and this is a no-op. One memcpy per tensor per RUN, not per token.
    static const bool no_retain = [] {
        const char* v = std::getenv("DRAY_NO_RETAIN");
        return v && v[0] == '1';
    }();
    while (!im.ring_q.empty() && im.ring_q.front().consumed &&
           im.ring_q.front().tags.empty() &&
           std::find(im.current.begin(), im.current.end(), im.ring_q.front().t) ==
               im.current.end()) {
        Streamer::Impl::RingSeg& fr = im.ring_q.front();
        // fr.reads_ok, not just fr.complete: promoting a segment whose reads
        // failed installs partly-unwritten arena bytes as a permanent cache hit.
        if (!no_retain && fr.t && fr.complete && fr.reads_ok && !im.resident.count(fr.t)) {
            const uint64_t nb = ggml_nbytes(fr.t);
            if (cache_used(im) + nb + im.churn_reserve <= cache_budget(im)) {
                void* mem = alloc_acct(im, mem::Category::ExpertCache, nb, kBufAlign);
                if (mem) {
                    std::memcpy(mem, im.ring + fr.pos + fr.head, static_cast<size_t>(nb));
                    Resident nr;
                    nr.mem = mem; nr.bytes = nb; nr.pinned = false;
                    nr.prio = Prio::Unconditional;
                    nr.cat = mem::Category::ExpertCache;
                    // Promoted bytes come from the ring, which holds the file
                    // layout. Stale repack traits here would tell the kernels to
                    // read it as interleaved.
                    fr.t->extra = nullptr;
                    im.resident[fr.t] = nr;
                    im.lru.push_front(fr.t);
                    ++im.ring_promotions;
                }
            }
        }
        im.ring_used -= fr.span;
        im.ring_q.pop_front();
        ++im.ring_pops;
    }
    if (im.stream_list.empty()) return;

    // Leave depth for unlock-triggered expert batches: they have the nearest
    // deadline and must not queue behind the speculative stream.
    const size_t cap_depth = im.io->max_in_flight();
    const size_t depth_target = cap_depth > 16 ? cap_depth - 16 : cap_depth;

    size_t guard = 0;
    bool did_work = false;
    while (guard++ < im.stream_list.size()) {
        // HARVEST FIRST, non-blocking. An op counts as in-flight until polled, so
        // without this the depth check saturates after the first top-up and never
        // clears: measured 64 ring-fed across 8 tokens -- ~6-8 segments (48 chunks)
        // per token, then every pump broke on depth for the rest of the pass while
        // the drive sat idle. Harvesting here is what turns the thousands of
        // callbacks per layer into a continuously refilled window.
        pump(im, 0);
        if (im.io->in_flight() >= depth_target) { ++im.ring_stop[0]; break; }
        ggml_tensor* t = im.stream_list[im.ring_cursor % im.stream_list.size()];

        // A full lap: the next tensor is already queued and unconsumed. Deep enough.
        bool queued = false;
        for (const auto& sg : im.ring_q) {
            if (sg.t == t && !sg.consumed) { queued = true; break; }
        }
        if (queued) { ++im.ring_stop[1]; break; }

        auto b = im.bound.find(t);
        if (b == im.bound.end() || im.resident.count(t)) { ++im.ring_cursor; continue; }

        const uint64_t nb   = ggml_nbytes(t);
        const uint64_t lo   = (b->second.offset / im.align) * im.align;
        const uint32_t head = static_cast<uint32_t>(b->second.offset - lo);
        const uint64_t span = ((head + nb + im.align - 1) / im.align) * im.align;
        // Too big to pipeline (would monopolise the ring): old path handles it.
        if (span > im.ring_bytes / 2) { ++im.ring_cursor; continue; }

        // Wrap: segments must be contiguous, so pad the tail and place at 0.
        if (im.ring_head_off + span > im.ring_bytes) {
            const uint64_t pad = im.ring_bytes - im.ring_head_off;
            if (im.ring_used + pad + span > im.ring_bytes) { ++im.ring_stop[2]; break; }
            Streamer::Impl::RingSeg ps;
            ps.pos = im.ring_head_off; ps.span = pad; ps.consumed = true;
            im.ring_q.push_back(std::move(ps));
            im.ring_used += pad;
            im.ring_head_off = 0;
        }
        if (im.ring_used + span > im.ring_bytes) { ++im.ring_stop[3]; break; }

        // Submit in <=8 MiB chunks: per calibration, moderate requests at depth
        // beat one giant request, and no single request occupies the drive long
        // enough to delay an expert batch.
        Streamer::Impl::RingSeg sg;
        sg.t = t; sg.pos = im.ring_head_off; sg.span = span; sg.head = head;
        sg.complete = true;
        const io::FileId f = im.shards[static_cast<size_t>(b->second.shard)];
        const uint64_t interior = head + nb;
        bool fail = false;
        for (uint64_t off = 0; off < span && !fail; off += (8ull << 20)) {
            const uint32_t clen =
                static_cast<uint32_t>(std::min<uint64_t>(8ull << 20, span - off));
            // What this chunk must actually deliver; the widened tail past EOF may
            // legally come up short.
            const uint64_t need =
                off >= interior ? 0 : std::min<uint64_t>(clen, interior - off);
            const uint64_t tag = im.next_tag++;
            io::ReadRequest r{f, lo + off, clen, im.ring + sg.pos + off, tag};
            im.pending[tag] = {nullptr, 0, clen, nullptr, need, false, nullptr,
                               false, false};
            if (im.io->submit(&r, 1) != 1) { im.pending.erase(tag); fail = true; sg.complete = false; }
            else sg.tags.push_back(tag);
        }
        if (fail && sg.tags.empty()) { ++im.ring_stop[4]; break; }
        im.ring_used += span;
        im.ring_head_off = (im.ring_head_off + span) % im.ring_bytes;
        im.ring_q.push_back(std::move(sg));
        did_work = true;
        ++im.ring_cursor;
        ++im.ring_segs;
        if (fail) { ++im.ring_stop[4]; break; }
    }
    // A complete lap that submitted nothing (everything resident or oversized):
    // sleep. Depth/lap/full breaks are NOT fruitless -- work is in flight.
    if (!did_work && guard > im.stream_list.size()) {
        im.ring_idle_backoff = 256;
    }
}

// The consumer: a lookup instead of a read. Failure falls back to the sync path,
// which re-reads -- correct, only slower.
static bool ring_consume(Streamer::Impl& im, ggml_tensor* t, uint64_t* streamed) {
    if (!im.ring || !im.ring_active) return false;
    for (auto& sg : im.ring_q) {
        if (sg.t != t || sg.consumed) continue;
        // A partially submitted segment holds valid bytes only where chunks were
        // actually issued: serving it would be silent corruption. Settle what is
        // outstanding, mark consumed, and let the sync path read properly.
        // Three-arg settle: false can mean "a read landed with a bad status"
        // (safe, memory is ours again) or "the backend died with reads
        // OUTSTANDING" (the arena bytes are still a DMA target). Only the
        // out-param distinguishes them, and this site used to pass nullptr --
        // letting pump_ring recycle a segment the kernel could still write
        // (2026-08-24 audit). On death, deactivate the ring: the single reclaim
        // site is gated on ring_active, so nothing is ever reused, and the
        // destructor already knows how to leak a dead ring deliberately.
        bool ring_backend_dead = false;
        const bool settled_ok = settle_tags(im, sg.tags, &ring_backend_dead);
        if (ring_backend_dead) {
            std::fprintf(stderr,
                         "[dray] FATAL: backend dead with ring reads outstanding; "
                         "ring disabled and its memory retired (no-cancel contract)\n");
            im.ring_active = false;
        }
        const bool ok = settled_ok && sg.complete;
        sg.consumed = true;
        if (!settled_ok) sg.reads_ok = false;   // retention must not promote this
        if (!ok) return false;
        // Raw file bytes, NOT the interleaved layout maybe_repack may have left
        // traits for. extra is set once at init_tensor and was never cleared, so
        // the repacked-matmul kernel could dispatch over non-interleaved data
        // (2026-08-24 audit). Clearing it here is free when repack is off.
        t->extra = nullptr;
        t->data = im.ring + sg.pos + sg.head;
        *streamed += ggml_nbytes(t);
        im.uncond_needed += ggml_nbytes(t);
        im.uncond_read += ggml_nbytes(t);
        ++im.ring_hits;
        return true;
    }
    return false;
}

static bool bring_in(Streamer::Impl& im, ggml_tensor* t, uint64_t* streamed) {
    // Pass-1 recording: the graph declares its real working set by consuming it.
    if (im.ring && !im.ring_active) {
        auto b0 = im.bound.find(t);
        if (b0 != im.bound.end() && !b0->second.pinned && !b0->second.sliceable &&
            t->ne[2] <= 1 && im.consumed_seen.insert(t).second) {
            im.consumed_order.push_back(t);
        }
    }
    auto r = im.resident.find(t);
    // R13: a COMPACTED partial region (uniq non-empty: only the router-selected
    // experts, remapped) must never be served as the whole tensor -- a plain
    // MUL_MAT reading it would see k experts where n_experts belong. Both known
    // triggers are near-dead, but this file exists to prevent exactly that.
    // T20: and the refusal must RECLAIM, not orphan -- falling through to a
    // whole-tensor read that copy-assigns over the entry left the old region
    // permanently charged and invisible to make_room (t sits in im.current).
    if (r != im.resident.end() && r->second.mem && !r->second.uniq.empty()) {
        free_acct(im, r->second.cat, r->second.mem, r->second.bytes);
        im.resident.erase(r);
        im.lru.remove(t);
        r = im.resident.end();
    }
    if (r != im.resident.end() && r->second.mem && r->second.uniq.empty()) {
        // A pending ring segment for a tensor served from residency would sit
        // unconsumed at the FIFO front forever and wedge reclaim -- the
        // attn_res_score failure over again, reached via LRU instead of via an
        // unused tensor. Settle its DMA and discard it; the bytes are a transient
        // duplicate, and the producer skips resident tensors on the next lap.
        if (im.ring_active) {
            for (auto& sg : im.ring_q) {
                if (sg.t == t && !sg.consumed) {
                    // The bool was discarded here, which is how a failed read
                    // reached retention looking clean. And the dead-backend case
                    // must retire the ring, not just this segment: outstanding
                    // DMA does not care which segment the caller was touching.
                    bool rbd = false;
                    if (!settle_tags(im, sg.tags, &rbd)) sg.reads_ok = false;
                    if (rbd) {
                        std::fprintf(stderr,
                                     "[dray] FATAL: backend dead with ring reads outstanding; "
                                     "ring disabled and its memory retired (no-cancel contract)\n");
                        im.ring_active = false;
                    }
                    sg.consumed = true;
                    break;
                }
            }
        }
        t->data = static_cast<uint8_t*>(r->second.mem) + r->second.head;
        // H8: the RAM-hit sibling of the miss-path split -- a routed tensor's
        // hits and misses must land in ONE class or h_routed lies (Metal's
        // forced whole-tensor mode printed a hard 0% no matter the pinning).
        if (t->ne[2] > 1) im.routed_needed += ggml_nbytes(t);
        else              im.uncond_needed += ggml_nbytes(t);   // served from RAM: read 0
        return true;
    }
    // Already being read by prefetch: wait for that read rather than starting a
    // second one. This is where the overlap pays off -- by now it has usually
    // landed while an earlier node was computing.
    if (im.prefetch_tag.count(t)) {
        ++im.prefetch_hits;
        if (settle_prefetch(im, t)) { *streamed += ggml_nbytes(t); return true; }
        return false;
    }
    // The ring: by now the bytes are usually already in the arena, and this is a
    // pointer assignment. On failure fall through to the synchronous read.
    if (ring_consume(im, t, streamed)) return true;


    auto b = im.bound.find(t);
    if (b == im.bound.end()) return t->data != nullptr;   // not ours

    const uint64_t bytes = ggml_nbytes(t);

    // On ANY failure below, data must be left pointing at the poisoned landing
    // zone -- never at null. A null here is dereferenced by the CPU backend and
    // access-violates; the poison at least fails loudly in the output instead.
    auto fail = [&](const char* why) {
        if (im.failures < 8) {
            std::fprintf(stderr,
                "[dray] MATERIALISE FAIL %s: %s  need=%.2fGB in_use=%.2fGB budget=%.2fGB lru=%zu\n",
                t->name, why, bytes / 1e9, cache_used(im) / 1e9, cache_budget(im) / 1e9, im.lru.size());
            std::fflush(stderr);
        }
        t->data = poison_for(im, t);
        return false;
    };

    // Before giving up, settle anything prefetch is holding: its memory is
    // unreclaimable while in flight, but once landed it becomes ordinary resident
    // memory that make_room can evict. The node being computed always wins.
    if (!make_room(im, bytes) && !im.prefetch_tag.empty()) {
        std::vector<ggml_tensor*> waiting;
        waiting.reserve(im.prefetch_tag.size());
        for (const auto& kv : im.prefetch_tag) waiting.push_back(const_cast<ggml_tensor*>(kv.first));
        for (ggml_tensor* w : waiting) settle_prefetch(im, w);
    }
    if (!make_room(im, bytes)) return fail("no room");
    uint64_t alloc_bytes = 0;
    uint32_t head = 0;
    void* m = alloc_and_read_direct(im, mem::Category::ExpertCache, b->second, bytes,
                                    &alloc_bytes, &head);
    if (!m) return fail("read");

    Resident nr; nr.mem = m; nr.bytes = alloc_bytes; nr.head = head;
    nr.refs = 0; nr.pinned = false;
    // Whole-tensor materialisation is an unconditional weight: attention, shared
    // experts, embedding, lm_head. Read every token, so worth ~51x a routed byte.
    nr.prio = Prio::Unconditional;

    // Claim static residency while there is room. First-come is deliberate and
    // matches the access order, so the pinned set is a contiguous prefix of the
    // layer sweep rather than a random scatter -- and because every weight is read
    // exactly once per cycle, WHICH ones are pinned does not affect the hit rate,
    // only how many.
    if (im.static_used + alloc_bytes <= static_allowance(im)) {
        nr.pinned = true;
        im.static_used += alloc_bytes;
    }
    t->data = static_cast<uint8_t*>(m) + head;
    nr.repacked = maybe_repack(im, t);
    im.resident[t] = nr;
    im.lru.push_front(t);
    *streamed += bytes;
    // Pass-4b: a whole-tensor read of a ROUTED (expert-fused, ne[2]>1) tensor
    // reaches here via the compaction-decline fallthrough; charging it to the
    // unconditional class conflated the two hit rates the readout exists to
    // keep apart. Bytes-read is unchanged; only the class split is.
    if (t->ne[2] > 1) {
        im.routed_needed += bytes;
        im.routed_read += bytes;
    } else {
        im.uncond_needed += bytes;
        im.uncond_read += bytes;
    }
    return true;
}

// MUL_MAT_ID reads only n_expert_used of n_expert experts, but the fused tensor
// covers all of them -- materialising it whole reads ~4.7 GiB per MoE layer to use
// ~98 MB. On Qwen3.8 that is the difference between ~450 GB and ~9 GB per token.
//
// So: read only the selected experts into a compact region with the SAME stride,
// and rewrite the ids in place to 0..k-1. ne[2] never has to change, because
// remapped ids are always in range. The ids tensor is already computed by the time
// this node runs -- the router is earlier in the same graph.
// The ids the GRAPH meant, not the ones we may have installed on a previous token.
static ggml_tensor* original_ids(Streamer::Impl& im, ggml_tensor* node, ggml_tensor* ids) {
    auto it = im.priv_by_node.find(node);
    if (it != im.priv_by_node.end() && ids == &it->second.t && it->second.orig) {
        return it->second.orig;
    }
    return ids;
}

// Returns the compact ordering for this ids tensor, rewriting the ids buffer to
// compact indices on first use. Every weight tensor sharing these ids must lay its
// compact region out in exactly this order.
// The ids tensor is a VIEW: selected_experts comes from ggml_argsort_top_k, which
// views the first n_expert_used entries of each row of an [n_expert, n_tokens]
// argsort result. So nb[1] is the FULL row (512*4 bytes on Qwen3.8), not
// ne[0]*nb[0], and the elements are not contiguous.
//
// Treating it as a flat array read the wrong ids and, worse, wrote n_expert_used *
// n_tokens contiguous int32s straight through the view into neighbouring memory.
// With a single token there is one row, so a contiguous walk is accidentally
// correct -- which is exactly why one token worked and five faulted.
static inline int32_t* ids_at(ggml_tensor* ids, int64_t i0, int64_t i1) {
    return reinterpret_cast<int32_t*>(static_cast<char*>(ids->data) +
                                      i1 * ids->nb[1] + i0 * ids->nb[0]);
}

// Derives the compact ordering and the remapped indices from the ORIGINAL ids.
// Reads only -- the shared tensor is never written.
static bool derive_ids_map(ggml_tensor* ids, int64_t n_expert,
                           std::vector<int32_t>* uniq, std::vector<int32_t>* remapped) {
    if (!ids->data) return false;
    const int64_t n0 = ids->ne[0], n1 = ids->ne[1];
    if (n0 <= 0 || n1 <= 0) return false;

    uniq->clear();
    remapped->assign(static_cast<size_t>(n0 * n1), 0);

    size_t k = 0;
    for (int64_t i1 = 0; i1 < n1; ++i1) {
        for (int64_t i0 = 0; i0 < n0; ++i0, ++k) {
            const int32_t e = *ids_at(ids, i0, i1);
            if (e < 0 || e >= n_expert) return false;   // out of range: never guess
            auto f = std::find(uniq->begin(), uniq->end(), e);
            if (f == uniq->end()) { uniq->push_back(e); f = uniq->end() - 1; }
            (*remapped)[k] = static_cast<int32_t>(f - uniq->begin());
        }
    }
    return !uniq->empty();
}

// Builds the private compact-ids tensor and repoints the node at it. The original
// ids keep real expert ids, so every other consumer of them stays correct without
// this code having to know they exist.
static void install_private_ids(Streamer::Impl& im, ggml_tensor* node, ggml_tensor* ids,
                                const std::vector<int32_t>& remapped, int src_index = 2) {
    Streamer::Impl::PrivIds& p = im.priv_by_node[node];
    if (p.orig == nullptr || ids != &p.t) p.orig = ids;

    // Copy the original's metadata so type/shape/op all match exactly, then give it
    // contiguous strides over our own buffer. The original is a strided VIEW over an
    // argsort result; ours does not have to be, as long as nb[] describes it.
    // A CLEAN tensor, not a doctored copy of a view.
    //
    // Copying *ids brought its op (GGML_OP_VIEW), its src[] pointing at the argsort
    // result, and its flags along with the shape -- then nulling view_src and buffer
    // left an object claiming to be a view of nothing. Proven to be the fault: with
    // an identity mapping, where the remapped ids are numerically identical to the
    // originals, installing this tensor still produced wrong output, and skipping the
    // install produced text matching the reference exactly.
    // SAME STRIDES AS THE ORIGINAL, not a re-laid-out contiguous copy.
    //
    // The MoE ids are a strided view of an argsort result: ne[0] is n_expert_used but
    // nb[1] is a full n_expert row. Handing back a contiguous tensor with the same
    // ne[] and different nb[] SHOULD be equivalent -- mul_mat_id reads through nb[] --
    // and demonstrably is not. GET_ROWS ids are already contiguous, we do not change
    // their layout, and those work; MUL_MAT_ID ids are the only layout we alter, and
    // those are the ones that break.
    //
    // So mirror the original byte layout exactly and change only the VALUES. The
    // buffer is sized from the original stride and the gaps are left as they are.
    const size_t words = static_cast<size_t>(ggml_nbytes(ids) / sizeof(int32_t)) + 4;
    p.buf.assign(words, 0);
    p.t = ggml_tensor{};
    p.t.type = ids->type;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        p.t.ne[i] = ids->ne[i];
        p.t.nb[i] = ids->nb[i];
    }
    p.t.op = GGML_OP_NONE;
    p.t.buffer   = ids->buffer;
    p.t.data     = p.buf.data();
    p.t.view_src = nullptr;
    p.t.view_offs = 0;
    // Write the remapped values at the ORIGINAL strided positions.
    for (int64_t i1 = 0; i1 < ids->ne[1]; ++i1) {
        for (int64_t i0 = 0; i0 < ids->ne[0]; ++i0) {
            const size_t k = static_cast<size_t>(i1 * ids->ne[0] + i0);
            if (k >= remapped.size()) continue;
            *reinterpret_cast<int32_t*>(static_cast<char*>(p.t.data) +
                                        i1 * p.t.nb[1] + i0 * p.t.nb[0]) = remapped[k];
        }
    }

    // Read the installed tensor back through its OWN strides and compare against the
    // original read through ITS strides. If these disagree the tensor is not what we
    // think we built, whatever the field values look like.
    static const bool vtrace = [] {
        const char* v = std::getenv("DRAY_TRACE_COMPACT");
        return v && v[0] == '1';
    }();
    static int vcount = 0;
    if (vtrace && vcount < 4) {
        ++vcount;
        int bad = 0;
        for (int64_t i1 = 0; i1 < ids->ne[1]; ++i1) {
            for (int64_t i0 = 0; i0 < ids->ne[0]; ++i0) {
                const int32_t got = *reinterpret_cast<int32_t*>(
                    static_cast<char*>(p.t.data) + i1 * p.t.nb[1] + i0 * p.t.nb[0]);
                const size_t k = static_cast<size_t>(i1 * ids->ne[0] + i0);
                if (k >= remapped.size() || got != remapped[k]) ++bad;
            }
        }
        std::fprintf(stderr, "[dray] PRIVIDS ne=[%lld,%lld] nb=[%zu,%zu] buf=%zu bad=%d src=%d\n",
                     (long long)p.t.ne[0], (long long)p.t.ne[1], p.t.nb[0], p.t.nb[1],
                     p.buf.size(), bad, src_index);
        std::fflush(stderr);
    }

    // MUL_MAT_ID takes its ids at src[2], GET_ROWS at src[1].
    node->src[src_index] = &p.t;
}

// ROW SLICING for GET_ROWS -- the same idea as expert compaction, one node type over.
//
// token_embd.weight is 1.09 GiB on Qwen3.8 and the graph materialises ALL of it to
// read the rows of the tokens actually being processed: ~7.5 KB each. That single
// tensor is what made a small cap impossible -- at a 4 GB cap the whole cache budget
// was 0.18 GB and this allocation alone needed 1.14 GB.
//
// Rows are contiguous and uniformly strided by nb[1], exactly as experts are by
// nb[2], so the identical trick applies: read only the selected rows into a compact
// region and give the node PRIVATE ids holding compact indices. The shared ids keep
// real token ids, so anything else reading them stays correct without knowing.
//
// ggml_get_rows indexes by the id VALUE and does not range-check against ne[1] at
// compute time, so a compact buffer of k rows addressed by ids in 0..k-1 is in
// bounds by construction.
static bool compact_rows(Streamer::Impl& im, ggml_tensor* node, ggml_tensor* w,
                         ggml_tensor* ids, uint64_t* streamed) {
    if (!w || !ids || ids->type != GGML_TYPE_I32) return false;

    auto b = im.bound.find(w);
    if (b == im.bound.end()) return false;          // not ours
    if (w->ne[1] <= 1 || w->ne[2] > 1) return false;   // a plain 2-D table only

    const int64_t n_rows  = w->ne[1];
    const uint64_t stride = static_cast<uint64_t>(w->nb[1]);
    if (stride == 0) return false;

    std::vector<int32_t> uniq, remapped;
    if (!derive_ids_map(ids, n_rows, &uniq, &remapped)) return false;

    // Not worth it if the "slice" is most of the table: fall back to the whole
    // tensor, which is cacheable and reusable across tokens.
    if (static_cast<int64_t>(uniq.size()) * 4 >= n_rows) return false;

    auto ex = im.resident.find(w);
    if (ex != im.resident.end()) {
        if (ex->second.uniq == uniq) {
            w->extra = nullptr;   // compacted slab is not the repacked layout
            w->data = ex->second.mem;
            install_private_ids(im, node, ids, remapped, 1);
            return true;
        }
        free_acct(im, ex->second.cat, ex->second.mem, ex->second.bytes);
        im.resident.erase(ex);
        im.lru.remove(w);
    }

    const uint64_t need = static_cast<uint64_t>(uniq.size()) * stride;
    if (!make_room(im, need)) return false;

    uint8_t* mem = static_cast<uint8_t*>(
        alloc_acct(im, mem::Category::ExpertCache, need, kBufAlign));
    if (!mem) return false;

    std::vector<Slice> slices(uniq.size());
    for (size_t k = 0; k < uniq.size(); ++k) {
        slices[k].src = b->second;
        slices[k].src.offset += static_cast<uint64_t>(uniq[k]) * stride;
        slices[k].dst = mem + k * stride;
        slices[k].len = stride;
    }
    if (!read_batch(im, slices)) {
        free_acct(im, mem::Category::ExpertCache, mem, need);
        return false;
    }
    *streamed += need;

    Resident nr; nr.mem = mem; nr.bytes = need; nr.pinned = false;
    nr.prio = Prio::RoutedExpert;   // valid only for this token's ids, like a compact
    nr.uniq = uniq;
    im.resident[w] = nr;
    im.lru.push_front(w);
    w->extra = nullptr;   // freshly compacted bytes; drop any stale repack traits
    w->data = mem;
    install_private_ids(im, node, ids, remapped, 1);
    return true;
}

// Materialises only the experts this node selects, into a compact region with the
// same stride, ordered by the shared ids map.
// Pumps completions until every tag in `tags` has landed, then discards them.
// Returns false if any read failed or the queue went quiet with tags outstanding.
// A poison buffer guaranteed to cover this tensor's FULL extent. The fixed
// scratch is slot-sized on the fast-load path; a failing whole tensor larger
// than it gets a lazily grown emergency buffer instead (raw OS memory, outside
// the cap ledger -- the run is dying, and a brief overshoot beats corruption).
// If even that allocation fails there is no safe way to let the node run, and a
// clean fatal beats undefined behaviour.
static uint8_t* poison_for(Streamer::Impl& im, const ggml_tensor* t) {
    const uint64_t need = ggml_nbytes(t);
    if (need <= im.scratch_bytes) return im.scratch;
    if (need > im.emergency_bytes) {
        if (im.emergency && im.pool.contains(im.emergency)) im.pool.free(im.emergency, im.emergency_bytes);
        else if (im.emergency) mem::aligned_free_host(im.emergency, im.emergency_bytes);
        // Pool first, same reason as the scratch: emergency memory is read by
        // compute, so it must live in the registrable range when one exists.
        im.emergency = static_cast<uint8_t*>(im.pool.alloc(need, kBufAlign));
        if (!im.emergency) {
            im.emergency = static_cast<uint8_t*>(mem::aligned_alloc_host(
                static_cast<size_t>(need), kBufAlign));
        }
        im.emergency_bytes = im.emergency ? need : 0;
        if (im.emergency) std::memset(im.emergency, 0xA5, static_cast<size_t>(need));
    }
    if (!im.emergency) {
        std::fprintf(stderr,
                     "[dray] FATAL: %s failed to materialise and no safe poison "
                     "buffer of %llu bytes could be allocated\n",
                     t->name, static_cast<unsigned long long>(need));
        std::abort();
    }
    return im.emergency;
}

static bool settle_tags(Streamer::Impl& im, std::vector<uint64_t>& tags,
                        bool* backend_dead) {
    if (backend_dead) *backend_dead = false;
    bool ok = true;
    for (;;) {
        bool all = true;
        for (uint64_t t : tags) {
            auto it = im.pending.find(t);
            if (it != im.pending.end() && !it->second.done) { all = false; break; }
        }
        if (all) break;
        if (pump(im, 1) == 0) {
            // F2 (same shape as read_batch): a dead backend with reads
            // outstanding must LEAK the staging, never free under DMA.
            // I1: THE line H15 forgot -- without it the out-param stayed false,
            // both dtor consumers computed constants, and the teardown guard
            // was disarmed in the direction of always-free. A fix that adds a
            // flag and never sets it is worse than no fix: it retires the
            // FATAL message that would have said so.
            if (backend_dead) *backend_dead = true;
            std::fprintf(stderr,
                         "[dray] FATAL: backend dead with reads outstanding; "
                         "leaking %zu staging buffers (no-cancel contract)\n",
                         tags.size());
            for (uint64_t t : tags) im.pending.erase(t);
            tags.clear();
            return false;
        }
    }
    for (uint64_t t : tags) {
        auto it = im.pending.find(t);
        if (it == im.pending.end() || !it->second.ok) ok = false;
        discard(im, t);
    }
    tags.clear();
    return ok;
}

static void* eslot_get(Streamer::Impl& im, const ggml_tensor* w, int32_t e) {
    auto t = im.eslots.find(w);
    if (t == im.eslots.end()) return nullptr;
    auto s = t->second.find(e);
    return s == t->second.end() ? nullptr : s->second.mem;
}

// Admission is the retention rule again: only genuinely free room, never the
// churn reserve, never by evicting. Under heavy skew the hot experts recur every
// few tokens, so admit-on-read fills with hot slots quickly without a policy.
static void eslot_admit(Streamer::Impl& im, const ggml_tensor* w, int32_t e,
                        const void* src, uint64_t stride) {
    // ON by default, bounded by the plan-time pool (DRAY_NO_ESLOTS disables).
    // History in git: free-room-only found zero by decode; UNBOUNDED
    // eviction-assisted admission ratcheted in_use past budget. The pool bound is
    // what makes make_room here compose: slots stay below the evictable mass.
    static const bool off = [] {
        const char* v = std::getenv("DRAY_NO_ESLOTS");
        return v && v[0] == '1';
    }();
    if (off || eslot_get(im, w, e)) return;
    // Pool full: replace the coldest slot OF THIS TENSOR if the newcomer is
    // measurably hotter. Admit-only filled the pool with a cross-section and
    // measured h_routed=4% at 3.4% coverage -- rank-blind, exactly as predicted.
    // Per-tensor confinement keeps the victim scan tiny; the 2x hysteresis stops
    // equal-count churn. Eviction frees the SAME bytes the newcomer needs, so
    // the pool bound is preserved by construction.
    if (im.eslot_bytes + stride > im.eslot_pool) {
        auto hw = im.expert_hist.find(w);
        if (hw == im.expert_hist.end()) return;
        auto tv = im.eslots.find(w);
        if (tv == im.eslots.end() || tv->second.empty()) return;
        const auto& hist = hw->second;
        auto count_of = [&](int32_t id) -> uint32_t {
            return (id >= 0 && static_cast<size_t>(id) < hist.size())
                       ? hist[static_cast<size_t>(id)] : 0;
        };
        int32_t victim = -1;
        uint32_t vcount = UINT32_MAX;
        for (const auto& kv : tv->second) {
            const uint32_t c = count_of(kv.first);
            if (c < vcount) { vcount = c; victim = kv.first; }
        }
        if (victim < 0 || count_of(e) < vcount * 2) return;
        auto vs = tv->second.find(victim);
        free_acct(im, mem::Category::ExpertCache, vs->second.mem, vs->second.bytes);
        im.eslot_bytes -= vs->second.bytes;
        tv->second.erase(vs);
    }
    // STORM GUARD: filling a 10 GB pool in one token cost 105 s (900 admissions,
    // each a make_room plus a 10 MB memcpy, on the decode path). Amortise: at
    // most 32 admissions per token, and only experts the histogram has seen
    // before (the doorkeeper) -- under 75-88% skew the hot ones return within a
    // few tokens, the cold ones never earn a slot.
    if (im.eslot_admits_pass >= 32) return;
    {
        auto hw = im.expert_hist.find(w);
        if (hw == im.expert_hist.end()) return;
        if (e < 0 || static_cast<size_t>(e) >= hw->second.size()) return;
        if (hw->second[static_cast<size_t>(e)] < 2) return;
    }
    if (cache_used(im) + stride + im.churn_reserve > cache_budget(im)) {
        if (!make_room(im, stride + im.churn_reserve)) return;
        if (cache_used(im) + stride + im.churn_reserve > cache_budget(im)) return;
    }
    void* m = alloc_acct(im, mem::Category::ExpertCache, stride, kBufAlign);
    if (!m) return;
    std::memcpy(m, src, static_cast<size_t>(stride));
    im.eslots[w][e] = Streamer::Impl::ESlot{m, stride};
    im.eslot_bytes += stride;
    ++im.eslot_admits_pass;
}

// Submits one sibling tensor's compact region for `uniq` WITHOUT waiting. Any
// refusal (no room, queue full, alloc fail) just means the sibling reads
// synchronously when its own node arrives -- never an error, only lost overlap.
static void submit_sibling_region(Streamer::Impl& im, ggml_tensor* w,
                                  const std::vector<int32_t>& uniq) {
    if (!w || im.pending_regions.count(w)) return;
    auto b = im.bound.find(w);
    if (b == im.bound.end() || w->ne[2] <= 1) return;
    auto ex = im.resident.find(w);
    if (ex != im.resident.end() && ex->second.uniq == uniq) return;   // reusable as is

    // Strides DIFFER across gate/up/down -- always this tensor's own nb[2].
    const uint64_t stride = static_cast<uint64_t>(w->nb[2]);
    const uint64_t need   = static_cast<uint64_t>(uniq.size()) * stride;
    if (!make_room(im, need)) return;
    uint8_t* mem = static_cast<uint8_t*>(
        alloc_acct(im, mem::Category::ExpertCache, need, kBufAlign));
    if (!mem) return;

    Streamer::Impl::PendingRegion pr;
    pr.mem = mem; pr.bytes = need; pr.uniq = uniq;
    bool fail = false;
    for (size_t k = 0; k < uniq.size() && !fail; ++k) {
        // Frequency cache first: a hit is a memcpy at RAM speed, no disk request,
        // and no tag to wait on at adoption.
        if (const void* c = eslot_get(im, w, uniq[k])) {
            std::memcpy(mem + k * stride, c, static_cast<size_t>(stride));
            ++im.eslot_hits;
            continue;
        }
        pr.miss.push_back(k);
        Source s = b->second;
        s.offset += static_cast<uint64_t>(uniq[k]) *
                    (s.disk_stride ? s.disk_stride : stride);
        while (im.io->in_flight() >= im.io->max_in_flight()) {
            if (pump(im, 1) == 0) { fail = true; break; }
        }
        if (fail) break;
        const uint64_t tag = submit_one(im, s, mem + k * stride, stride, false, nullptr);
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
        settle_tags(im, pr.tags, &dead);
        if (dead) {
            std::fprintf(stderr, "[dray] FATAL: backend dead with region reads outstanding; "
                                 "leaking the region (no-cancel contract)\n");
            return;
        }
        free_acct(im, mem::Category::ExpertCache, mem, need);
        return;
    }
    im.pending_regions[w] = std::move(pr);
}

// From any one of a layer's expert tensors, submit the other two. The suffixes are
// the ggml MoE convention shared by every target model, not an architecture branch.
static void submit_layer_siblings(Streamer::Impl& im, ggml_tensor* w,
                                  const std::vector<int32_t>& uniq) {
    static const char* kinds[3] = {"_gate_exps", "_up_exps", "_down_exps"};
    const char* mine = nullptr;
    for (const char* k : kinds) if (std::strstr(w->name, k)) { mine = k; break; }
    if (!mine) return;
    for (const char* k : kinds) {
        if (k == mine) continue;
        std::string n(w->name);
        n.replace(n.find(mine), std::strlen(mine), k);
        auto it = im.by_name.find(n);
        if (it != im.by_name.end()) submit_sibling_region(im, it->second, uniq);
    }
}

static bool compact_experts(Streamer::Impl& im, ggml_tensor* node, ggml_tensor* w,
                            ggml_tensor* ids, uint64_t* streamed) {
    if (!w || !ids || ids->type != GGML_TYPE_I32) return false;

    auto b = im.bound.find(w);
    if (b == im.bound.end()) return false;          // weights not ours
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
    static const bool compact_all = [] {
        const char* v = std::getenv("DRAY_COMPACT_ALL");
        return v && v[0] == '1';
    }();
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

    // PHASE C bookkeeping and the skew histogram, decode only. Registering here
    // rather than at load is what keeps this model-agnostic: the graph declares
    // which ids tensor unlocks which experts by consuming them together.
    // Decode vs prefill: ne[1]==1 was only ever a proxy. A batched decode step
    // carries ne[1]==B<=n_seq; prefill chunks carry hundreds. The bound is the
    // admitted width, and at n_seq==1 it reduces to the old test exactly.
    const int64_t decode_w = static_cast<int64_t>(im.cfg.n_seq ? im.cfg.n_seq : 1);
    if (ids->ne[1] <= decode_w) {
        auto& trio = im.ids_trio[ids];
        trio.n_expert = n_expert;
        bool known = false;
        for (int s = 0; s < 3; ++s) if (trio.w[s] == w) { known = true; break; }
        if (!known) {
            for (int s = 0; s < 3; ++s) if (!trio.w[s]) { trio.w[s] = w; break; }
        }
        auto& h = im.expert_hist[w];
        if (h.empty()) h.resize(static_cast<size_t>(n_expert), 0);
        for (int32_t e : uniq) {
            if (e >= 0 && e < n_expert) { ++h[static_cast<size_t>(e)]; ++im.hist_selections; }
        }
    }

    // Reuse only if the region holds exactly this ordering.
    static const bool no_reuse = [] {
        const char* v = std::getenv("DRAY_NO_REUSE");
        return v && v[0] == '1';
    }();
    auto ex = im.resident.find(w);
    if (ex != im.resident.end()) {
        if (!no_reuse && ex->second.uniq == uniq) {
            w->extra = nullptr;   // compacted slab is not the repacked layout
            w->data = ex->second.mem;
            im.routed_needed += ex->second.bytes;   // served from RAM: read 0
            if (!compact_all) install_private_ids(im, node, ids, remapped);
            return true;
        }
        free_acct(im, ex->second.cat, ex->second.mem, ex->second.bytes);
        im.resident.erase(ex);
        im.lru.remove(w);
    }

    // PHASE A: get the whole layer's ~48 reads in flight together. Submit the two
    // sibling regions first (unwaited), then handle this tensor -- whose own reads
    // overlap the siblings', and whose wait loop routes their completions to the
    // pending registry via pump().
    static const bool no_fuse = [] {
        const char* v = std::getenv("DRAY_NO_FUSE");
        return v && v[0] == '1';
    }();
    // DECODE ONLY. During prefill the ids carry every prompt token, so uniq grows
    // toward n_expert and each region toward the whole tensor -- two unwaited,
    // unreclaimable sibling regions then eat exactly the churn slack the current
    // tensor needs (measured: blk.5 gate failed with 107 LRU entries, all pinned,
    // nothing evictable, at a cap that worked unfused). At decode ne[1]<=n_seq
    // caps the trio at ~3 x uniq(n_seq) x stride -- which is precisely what the
    // width-scaled churn reserve funds, so batch fusion is inside the reserve by
    // construction.
    if (!no_fuse && ids->ne[1] <= static_cast<int64_t>(im.cfg.n_seq ? im.cfg.n_seq : 1))
        submit_layer_siblings(im, w, uniq);

    // Adopt a region a sibling's call already submitted for this routing.
    auto pr = im.pending_regions.find(w);
    if (pr != im.pending_regions.end()) {
        if (pr->second.uniq == uniq) {
            bool dead = false;
            const bool ok = settle_tags(im, pr->second.tags, &dead);
            uint8_t* mem = pr->second.mem;
            const uint64_t need_p = pr->second.bytes;
            const std::vector<size_t> miss = std::move(pr->second.miss);
            im.pending_regions.erase(pr);
            if (dead) {
                std::fprintf(stderr, "[dray] FATAL: backend dead with region reads outstanding; "
                                     "leaking the region (no-cancel contract)\n");
                return false;   // caller falls back; the memory is never reused
            }
            if (!ok) {
                free_acct(im, mem::Category::ExpertCache, mem, need_p);
                return false;
            }
            // Bytes-read counts DISK bytes only: cache-hit slots were memcpy'd at
            // submission and never touched the drive. h_routed splits the same way.
            const uint64_t miss_bytes = static_cast<uint64_t>(miss.size()) * stride;
            *streamed += miss_bytes;
            im.routed_needed += need_p;
            im.routed_read += miss_bytes;
            for (size_t k : miss) {
                eslot_admit(im, w, uniq[k], mem + k * stride, stride);
            }
            if (!compact_all) install_private_ids(im, node, ids, remapped);
            Resident nr; nr.mem = mem; nr.bytes = need_p; nr.pinned = false;
            nr.uniq = uniq; nr.prio = Prio::RoutedExpert;
            im.resident[w] = nr;
            im.lru.push_front(w);
            w->data = mem;
            return true;
        }
        // Routing changed before the region was consumed (abort/edge path): the
        // layout is for the wrong experts. Drain the DMA, free, fall through --
        // unless the backend died, in which case the DMA never drained and the
        // region must be leaked, not freed.
        bool dead = false;
        settle_tags(im, pr->second.tags, &dead);
        if (dead) {
            std::fprintf(stderr, "[dray] FATAL: backend dead with region reads outstanding; "
                                 "leaking the region (no-cancel contract)\n");
            im.pending_regions.erase(pr);
        } else {
            free_acct(im, mem::Category::ExpertCache, pr->second.mem, pr->second.bytes);
            im.pending_regions.erase(pr);
        }
    }

    static const bool trace = [] {
        const char* v = std::getenv("DRAY_TRACE_COMPACT");
        return v && v[0] == '1';
    }();
    static int traced = 0;
    if (trace && ids->ne[1] == 1 && traced < 20) {
        ++traced;
        std::fprintf(stderr, "[dray] COMPACT %s ids.ne=[%lld,%lld,%lld] nb=[%zu,%zu] uniq=%zu first=%d last=%d\n",
                     w->name, (long long)ids->ne[0], (long long)ids->ne[1],
                     (long long)ids->ne[2], ids->nb[0], ids->nb[1],
                     uniq.size(), uniq.empty() ? -1 : uniq.front(),
                     uniq.empty() ? -1 : uniq.back());
        std::fflush(stderr);
    }

    const uint64_t need = static_cast<uint64_t>(uniq.size()) * stride;
    if (!make_room(im, need)) {
        if (im.failures < 8) {
            std::fprintf(stderr,
                "[dray] COMPACT FAIL %s: no room  uniq=%zu need=%.2fGB in_use=%.2fGB budget=%.2fGB lru=%zu\n",
                w->name, uniq.size(), need / 1e9, cache_used(im) / 1e9, cache_budget(im) / 1e9, im.lru.size());
            std::fflush(stderr);
        }
        return false;
    }
    uint8_t* mem = static_cast<uint8_t*>(
        alloc_acct(im, mem::Category::ExpertCache, need, kBufAlign));
    if (!mem) return false;

    // Frequency cache first, then ONE batch for the misses, so the drive sees
    // every remaining request in flight together.
    std::vector<size_t> miss_k;
    miss_k.reserve(uniq.size());
    for (size_t k = 0; k < uniq.size(); ++k) {
        if (const void* c = eslot_get(im, w, uniq[k])) {
            std::memcpy(mem + k * stride, c, static_cast<size_t>(stride));
            ++im.eslot_hits;
        } else {
            miss_k.push_back(k);
        }
    }
    const uint64_t dstride = b->second.disk_stride ? b->second.disk_stride : stride;
    std::vector<Slice> slices(miss_k.size());
    for (size_t i = 0; i < miss_k.size(); ++i) {
        const size_t k = miss_k[i];
        slices[i].src = b->second;
        slices[i].src.offset += static_cast<uint64_t>(uniq[k]) * dstride;
        slices[i].dst = mem + k * stride;
        slices[i].len = stride;
    }
    if (!slices.empty() && !read_batch(im, slices)) {
        free_acct(im, mem::Category::ExpertCache, mem, need);
        return false;
    }
    const uint64_t miss_bytes = static_cast<uint64_t>(miss_k.size()) * stride;
    *streamed += miss_bytes;
    im.routed_needed += static_cast<uint64_t>(uniq.size()) * stride;
    im.routed_read   += miss_bytes;
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
            eslot_admit(im, w, uniq[k], mem + k * stride, stride);
        }
    }

    // Does read_batch actually deliver the bytes compact_experts asked for? The
    // verify subcommand proved a DIFFERENT implementation in main.cpp, never this
    // one, and identity compaction being wrong points here.
    if (trace) {
        for (size_t k = 0; k < uniq.size() && k < 2; ++k) {
            Source one = b->second;
            one.offset += static_cast<uint64_t>(uniq[k]) * stride;
            uint64_t ab = 0; uint32_t hd = 0;
            void* chk = alloc_and_read_direct(im, mem::Category::Misc, one, stride, &ab, &hd);
            if (chk) {
                const bool same = std::memcmp(static_cast<uint8_t*>(chk) + hd,
                                              mem + k * stride, static_cast<size_t>(stride)) == 0;
                std::fprintf(stderr, "[dray] READBATCH slot %zu expert %d: %s\n",
                             k, uniq[k], same ? "MATCHES direct read" : "*** MISMATCH ***");
                std::fflush(stderr);
                free_acct(im, mem::Category::Misc, chk, ab);
            }
        }
    }

    // Under identity compaction the remapped ids are numerically identical to the
    // originals, so a private tensor is semantically a no-op. Skipping it isolates
    // "the private ids tensor is wrong" from "the compact buffer is wrong".
    if (!compact_all) install_private_ids(im, node, ids, remapped);

    Resident nr; nr.mem = mem; nr.bytes = need; nr.pinned = false; nr.uniq = uniq;
    // A compacted region is valid for ONE routing decision, so it is the first
    // thing that should go when room is needed.
    nr.prio = Prio::RoutedExpert;
    im.resident[w] = nr;
    im.lru.push_front(w);
    w->extra = nullptr;   // freshly compacted bytes; drop any stale repack traits
    w->data = mem;
    return true;
}

// Does this node need anything from us? ggml executes NODE BY NODE with a full
// ggml_backend_synchronize after every node for which the eval callback answers
// true (ggml-backend.cpp, the callback_eval branch). Answering true for every
// node -- which this engine did until 2026-08-22 -- means one thread-pool
// barrier per node, ~4300 per token, and it is the whole of the 3x gap against
// stock: it also explains why threads never helped and why 22 of them
// collapsed (barrier cost scales with pool size). Nodes with no disk-backed
// source need nothing from us and must be batched by ggml instead.
bool Streamer::needs(ggml_tensor* node) {
    if (!node || !impl_) return false;
    Impl& im = *impl_;
    // Claiming every node costs a thread-pool barrier each. DRAY_FAST_NODES=1
    // opts into the predicate below, which is now CORRECT on all five
    // architectures in archgate -- the earlier divergence was a real bug, found
    // by reading rather than guessing: a skipped node never gets its VIEWS
    // re-pointed, and ggml fixes a view's data pointer at graph-allocation time.
    // See the K3 ssm_a note in materialise().
    //
    // It is worth 11-15%%, not the 42%% first reported: that figure came from the
    // BROKEN version, which was fast partly because it skipped work it needed to
    // do. Claiming view nodes takes back about half the skips (122B: 152,295 ->
    // 78,903 claims, 2.7 -> 2.4 s/token; 27B: 65,399 -> 36,771, 2.6 -> 2.2).
    // Re-verified 2026-08-24 after the instrumentation edits: archgate 5/5 with
    // the flag ON, so the correctness claim is current, not inherited.
    static const bool fast = [] {
        const char* v = std::getenv("DRAY_FAST_NODES");
        return v && v[0] == '1';
    }();
    if (!fast) return true;
    // Loud once per run. The flag is opt-in because archgate can only vouch for
    // the architectures it has: an untested graph shape could still reach a
    // skipped node through a path the predicate below does not follow.
    static bool warned = false;
    if (!warned) {
        warned = true;
        std::fprintf(stderr,
                     "\n*** DRAY_FAST_NODES=1: experimental node skipping ***\n"
                     "    Verified on the five archgate architectures (2026-08-24).\n"
                     "    Check YOUR model with scripts/archgate.ps1 before believing a token.\n\n");
    }

    // EXACT TEST, replacing the op enumeration that broke DeepSeek: a node needs
    // us if and only if it reads a tensor allocated in OUR buffer type. Buffer
    // ownership is a fact ggml already tracks; "which op kinds can reach our
    // memory" was a guess, and a new architecture falsified it. Views are
    // followed to their root because a view of a streamed weight is a streamed
    // weight.
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        for (ggml_tensor* v = node->src[i]; v; v = v->view_src) {
            if (v->buffer && v->buffer->buft == &im.buft) return true;
        }
    }
    for (ggml_tensor* v = node; v; v = v->view_src) {
        if (v->buffer && v->buffer->buft == &im.buft) return true;
    }
    if (im.priv_by_node.count(node)) return true;
    // HYPOTHESIS UNDER TEST: a node whose src is a VIEW must claim even if the
    // view root is not ours. materialise re-points views EVERY node, because
    // ggml fixes a view's data pointer at graph-allocation time (this is the K3
    // ssm_a bug documented below). A skipped node never gets that repointing.
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (node->src[i] && node->src[i]->view_src) return true;
    }
    return false;
}

bool Streamer::materialise(ggml_tensor* node) {
    const auto t_enter = std::chrono::steady_clock::now();
    struct TimeIt {
        Streamer::Impl* im; std::chrono::steady_clock::time_point t0;
        ~TimeIt() {
            im->ns_materialise += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count();
            ++im->n_materialise;
        }
    } timeit{impl_.get(), t_enter};
    if (!node) return true;
    Impl& im = *impl_;
    ++nodes_;

    // DRAY_TRACE=1 prints every node as it is materialised, unbuffered, so the
    // last line before a fault names the node that faulted. Guessing at which node
    // crashed from an exit code cost several wrong diagnoses.
    static const bool trace = [] {
        const char* v = std::getenv("DRAY_TRACE");
        return v && v[0] == '1';
    }();
    if (trace) {
        std::fprintf(stderr, "[dray] node=%s op=%s ne2=%lld in_use=%.2fGB\n",
                     node->name, ggml_op_name(node->op),
                     static_cast<long long>(node->ne[2]), cache_used(im) / 1e9);
        std::fflush(stderr);
    }

    // ONLY MUL_MAT_ID is compacted, and only its src[0].
    //
    // ADD_ID (bias, indexed at dim 1) and the GET_ROWS on the per-expert scale
    // vector inside build_lora_mm_id also consume the ids -- but because the shared
    // ids are never modified, they simply see real expert ids and index whole
    // tensors correctly. They need no special handling, which is the point: the
    // previous design required knowing about every ids consumer, and that set is
    // not knowable from here.
    //
    // Bisect switch. DRAY_NO_COMPACT=1 materialises whole expert tensors
    // instead: much slower, but it isolates a fault to compaction vs paging.
    // cfg.no_compact ALSO gates here: the Metal interlock forces whole-tensor
    // mode through the config, and a gate that only read the env made that
    // force vacuous (swarm finding R4 -- the sibling-path law, again).
    static const bool env_no_compact = [] {
        const char* v = std::getenv("DRAY_NO_COMPACT");
        return v && v[0] == '1';
    }();
    const bool no_compact = env_no_compact || im.cfg.no_compact;

    // Separately gateable from expert compaction: they are different mechanisms on
    // different node types, and a bisect that cannot tell them apart wastes a cycle.
    static const bool no_rowslice = [] {
        const char* v = std::getenv("DRAY_NO_ROWSLICE");
        return v && v[0] == '1';
    }();

    // Forward-pass boundary: the embedding lookup is the first op of every pass.
    // The ring activates at pass 2, once static pinning has claimed its prefix in
    // pass 1 and the pinned/streamed split is settled for the life of the run.
    if (node->op == GGML_OP_GET_ROWS && node->src[0]) {
        auto bs = im.bound.find(node->src[0]);
        if (bs != im.bound.end() && bs->second.sliceable) {
            ++im.pass_count;
            im.eslot_admits_pass = 0;
            if (im.ring && !im.ring_active && im.pass_count == 2) build_stream_list(im);
        }
    }

    // GET_ROWS on a streamed table: read only the rows this node asks for. Without
    // this, token_embd.weight (1.09 GiB) is materialised whole every token to read
    // a handful of ~7.5 KB rows, and it alone sets the floor under the minimum cap.
    if (node->op == GGML_OP_GET_ROWS && node->src[0] && node->src[1] &&
        im.bound.count(node->src[0]) && !no_compact && !no_rowslice) {
        ggml_tensor* w = node->src[0];
        im.current.clear();
        im.current.push_back(w);
        if (compact_rows(im, node, w, original_ids(im, node, node->src[1]), &bytes_streamed_)) {
            ++im.compacted;
            return true;
        }
        // Declined (too many rows, or no room). F9: the NODE may still hold a
        // previous token's private compact ids (never restored); reading the
        // whole table through them is rows 0..k-1, fluent garbage. Restore the
        // original ids before the fallthrough.
        {
            auto pv = im.priv_by_node.find(node);
            if (pv != im.priv_by_node.end() && node->src[1] == &pv->second.t &&
                pv->second.orig != nullptr) {
                node->src[1] = pv->second.orig;
            }
        }
        if (bring_in(im, w, &bytes_streamed_)) return true;

        // Failed outright. Same guard as the MUL_MAT_ID path: give the node private
        // ids that are all zero so every lookup lands in the first row and cannot
        // read past the poison buffer. Real token ids reach 151,936 against an 8 MB
        // sentinel, which is an access violation rather than a wrong answer.
        {
            // F10: route through original_ids -- src[1] may already BE the
            // private tensor, and install would zero its own source.
            ggml_tensor* oids = original_ids(im, node, node->src[1]);
            std::vector<int32_t> zeros(
                static_cast<size_t>(ggml_nelements(oids)), 0);
            install_private_ids(im, node, oids, zeros, 1);
        }
        w->data = im.scratch;
        ++im.failures;
        return true;
    }

    // The disk_stride refusal below lives inside the !no_compact branch, so with
    // compaction disabled -- by the env lever, or by the Metal interlock that
    // forces whole-tensor mode -- a repacked companion fell straight through to
    // the generic path and was read CONTIGUOUSLY. Its experts are interleaved on
    // disk, so that read succeeds at full length and assembles a tensor out of
    // the wrong bytes: exactly the silent-wrong-weights outcome the guard was
    // written to prevent, reachable by going around it (2026-08-24 audit).
    if (no_compact && node->op == GGML_OP_MUL_MAT_ID && node->src[0]) {
        auto bsrc = im.bound.find(node->src[0]);
        if (bsrc != im.bound.end() && bsrc->second.disk_stride) {
            std::fprintf(stderr,
                "[dray] REFUSED: %s is expert-interleaved on disk (a repacked "
                "companion), and compaction is disabled, so it cannot be read "
                "whole. Drop DRAY_NO_COMPACT, or use the original model file.\n",
                node->src[0]->name);
            std::fflush(stderr);
            ++im.failures;
            node->src[0]->data = poison_for(im, node->src[0]);
            return false;
        }
    }
    if (node->op == GGML_OP_MUL_MAT_ID && node->src[0] && node->src[2] &&
        im.bound.count(node->src[0]) && !no_compact) {
        ggml_tensor* w = node->src[0];
        im.current.clear();
        im.current.push_back(w);
        if (compact_experts(im, node, w, original_ids(im, node, node->src[2]), &bytes_streamed_)) {
            ++im.compacted;
            return true;
        }
        // Compaction declined. The SHARED ids were never touched -- but the NODE
        // may still point at a previous token's private compact ids (install is
        // never restored), and a whole tensor read through 0..k-1 ids is rows
        // 0..k-1 of the full tensor: fluent garbage with failures==0, the one
        // failure this file exists to prevent (F9). Restore the original ids
        // BEFORE the fallthrough; then the whole tensor is correctly indexable
        // -- expensive but right, and a tight cap keeps working.
        //
        // EXCEPT under a repacked companion: there the tensor's bytes are
        // expert-interleaved on disk, so a contiguous whole read would assemble
        // plausible garbage. Refuse loudly instead -- fluent output from a
        // misassembled tensor is the one failure this engine must never ship.
        {
            // F9: the restore itself.
            {
                auto pv = im.priv_by_node.find(node);
                if (pv != im.priv_by_node.end() && node->src[2] == &pv->second.t &&
                    pv->second.orig != nullptr) {
                    node->src[2] = pv->second.orig;
                }
            }
            auto bsrc = im.bound.find(w);
            if (bsrc != im.bound.end() && bsrc->second.disk_stride) {
                std::fprintf(stderr,
                    "[dray] REPACK: whole-tensor fallback impossible for %s "
                    "(expert-interleaved on disk); failing the node instead\n",
                    w->name);
            } else if (bring_in(im, w, &bytes_streamed_)) {
                return true;
            }
        }

        // Failed. Make the node structurally incapable of reading out of bounds
        // before the abort takes effect: point the weights at the poison buffer AND
        // give the node private ids that are all zero, so every lookup lands in the
        // first stride. Leaving real expert ids against a small poison buffer is
        // what turned a materialise failure into an access violation.
        //
        // The run aborts regardless -- this only guarantees the in-flight node
        // cannot fault before it does.
        if (node->src[2]) {
            // F10: same self-alias guard as the GET_ROWS twin.
            ggml_tensor* oids = original_ids(im, node, node->src[2]);
            std::vector<int32_t> zeros(
                static_cast<size_t>(ggml_nelements(oids)), 0);
            install_private_ids(im, node, oids, zeros);
        }
        w->data = poison_for(im, w);
        ++im.failures;
        return true;   // see the cb_eval comment: the return value controls batching
    }

    // No refcounts. An earlier version incremented on materialise and decremented
    // in the post-callback, but the post-callback only fires when the ask returns
    // true -- so any failure leaked a reference, eviction starved, and every
    // subsequent tensor failed to materialise. Instead: protect exactly the
    // tensors this node needs, for the duration of this node.
    im.current.clear();
    // VIEWS OVER STREAMED WEIGHTS.
    //
    // A src may be a view whose view_src is a weight we own. ggml_backend_view_init
    // fixes the view's data pointer from its parent at GRAPH-ALLOCATION time, which
    // is before anything is materialised -- so the view captures the poison sentinel
    // and keeps it. The parent is never a direct src of any node, so walking srcs
    // alone never materialises it and never notices.
    //
    // This is what broke Kimi K3. Its only use of ssm_a is
    //     A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head_kda, 1)
    // and ggml_reshape_3d returns a view. ssm_a was therefore never read from disk
    // on any layer of any token, and the KDA gate computed against 0xA5 -- fluent,
    // confidently wrong output with zero reported failures, because every mechanism
    // that could have caught it (failure counters, floor integrity, byte
    // verification) was looking at tensors that WERE materialised.
    //
    // Qwen3.8 never reshapes a streamed weight, which is why it was unaffected and
    // why "it works on the other model" proved nothing.
    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> views;   // (view, parent)
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        ggml_tensor* s = node->src[i];
        if (!s) continue;
        if (im.bound.find(s) != im.bound.end()) { im.current.push_back(s); continue; }
        // Follow the view chain to its root; reshapes can nest.
        ggml_tensor* root = s->view_src;
        while (root && im.bound.find(root) == im.bound.end() && root->view_src) {
            root = root->view_src;
        }
        if (root && im.bound.find(root) != im.bound.end()) {
            im.current.push_back(root);
            views.emplace_back(s, root);
        }
    }

    bool ok = true;
    for (ggml_tensor* s : im.current) {
        if (!bring_in(im, s, &bytes_streamed_)) { ok = false; ++im.failures; }
    }
    // Re-point every view at where its parent actually landed. Must run AFTER
    // bring_in, which is what moves the parent.
    for (const auto& vp : views) {
        if (vp.second->data) {
            vp.first->data = static_cast<uint8_t*>(vp.second->data) + vp.first->view_offs;
        }
    }

    // Keep the drive fed: the ring producer replaces the old per-tensor prefetch,
    // whose admission guard was unsatisfiable below the knee (304 issues in 44,655
    // nodes). The ring's memory is its own line item, so the guard cannot starve.
    pump_ring(im);
    return ok;
}

void Streamer::release(ggml_tensor* node) {
    const auto t_enter_rel = std::chrono::steady_clock::now();
    struct RelTimeIt {
        Streamer::Impl* im; std::chrono::steady_clock::time_point t0;
        ~RelTimeIt() {
            im->ns_release += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count();
        }
    } reltimeit{impl_.get(), t_enter_rel};
    // Protection is implicit in Impl::current, which the next materialise()
    // replaces. But a release callback is CPU time the drive can use.
    if (!impl_) return;
    Impl& im = *impl_;

    // PHASE C: if this node IS a router's ids tensor, its data was computed an
    // instant ago -- the earliest moment the layer's expert addresses exist.
    // Submit all three regions unwaited; the MUL_MAT_ID nodes a few small ops
    // later adopt them mid-flight. Refusals degrade to the Phase A path.
    static const bool no_early = [] {
        const char* v = std::getenv("DRAY_NO_EARLY");
        return v && v[0] == '1';
    }();
    if (!no_early && node && !im.ids_trio.empty() &&
        node->ne[1] <= static_cast<int64_t>(im.cfg.n_seq ? im.cfg.n_seq : 1)) {
        auto it = im.ids_trio.find(node);
        if (it != im.ids_trio.end()) {
            std::vector<int32_t> uniq, remapped;
            if (derive_ids_map(node, it->second.n_expert, &uniq, &remapped)) {
                for (int s = 0; s < 3; ++s) {
                    if (it->second.w[s]) submit_sibling_region(im, it->second.w[s], uniq);
                }
                ++im.early_unlocks;
            }
        }
    }

    pump_ring(im);
}

Streamer::SelfCheck Streamer::self_check(size_t max_tensors) {
    SelfCheck out;
    if (!impl_) return out;
    Impl& im = *impl_;

    // Re-read through the UNCACHED backend on FRESH handles, not through fopen.
    //
    // Invariant 2 is not negotiable and this runs on every start: buffered reads of
    // up to 32 resident tensors would dirty hundreds of MB of the user's page cache
    // each time (K3's router gates alone are 2.2 GB across 92 tensors). An earlier
    // version of this check used fopen/fread and did exactly that.
    //
    // Fresh handles rather than the streamer's own: it keeps the open/offset path
    // independently exercised. It is weaker than comparing against a completely
    // different I/O API -- a bug inside read_batch itself would be invisible here --
    // and that is the deliberate trade for not touching the page cache. The
    // differential compaction test covers the read path's logic separately.
    std::vector<io::FileId> refs;
    refs.reserve(im.plan.shard_paths.size());
    for (const std::string& sp : im.plan.shard_paths) {
        refs.push_back(im.io->open(sp));
    }

    std::vector<uint8_t> buf;
    for (const auto& kv : im.resident) {
        if (out.checked >= max_tensors) break;
        const Resident& r = kv.second;
        if (!r.pinned || !r.mem || r.bytes == 0) continue;
        if (r.repacked) { ++out.skipped_repacked; continue; }   // no longer file-shaped
        auto b = im.bound.find(kv.first);
        if (b == im.bound.end()) continue;          // no disk source to compare to
        Source s = b->second;
        if (s.shard < 0 || static_cast<size_t>(s.shard) >= refs.size()) continue;
        if (refs[static_cast<size_t>(s.shard)] == io::kInvalidFile) continue;

        // Point the read at the fresh handle by temporarily swapping the shard
        // table entry, so read_exact's alignment widening is reused verbatim.
        const io::FileId saved = im.shards[static_cast<size_t>(s.shard)];
        im.shards[static_cast<size_t>(s.shard)] = refs[static_cast<size_t>(s.shard)];
        buf.assign(static_cast<size_t>(r.bytes), 0);
        const bool got = read_exact(im, s, buf.data(), r.bytes);
        im.shards[static_cast<size_t>(s.shard)] = saved;
        if (!got) continue;

        ++out.checked;
        if (std::memcmp(buf.data(), r.mem, buf.size()) != 0) ++out.mismatched;
    }

    for (io::FileId f : refs) if (f != io::kInvalidFile) im.io->close(f);
    return out;
}

uint64_t Streamer::rebudget_against_rss(uint64_t rss) {
    if (!impl_ || rss == 0) return impl_ ? cache_budget(*impl_) : 0;
    Impl& im = *impl_;

    // Reconcile the ledger against the OS, then let the derived budget do the rest.
    //
    // Whatever the process holds beyond what the ledger explains is real resident
    // memory -- llama.cpp's KV and compute buffers, the vocab, allocator overhead,
    // fragmentation -- and it counts against the cap like anything else. Recording
    // it shrinks cache_budget() automatically, because that is defined as cap minus
    // every non-cache category.
    //
    // charge_unreserved SETS rather than accumulates, so calling this on a timer
    // tracks the figure instead of compounding it.
    const uint64_t explained = im.acct.used() - im.acct.unreserved();
    im.acct.charge_unreserved(rss > explained ? rss - explained : 0);

    // Hand back anything now over the line. Pinned entries are not evictable, so
    // this can fail to reach the target -- report_over_cap() is what tells the user
    // that happened rather than leaving them to infer it from RSS.
    //
    // FIRST, give the shrink teeth: static pins were claimed against the
    // LOAD-TIME allowance, and when the budget contracts under them they become
    // an over-claim nothing can evict -- measured on Linux (heavier process
    // floor): 3.27 GB pinned of 2.39 allowed, make_room impotent, output.weight
    // homeless, and the poison path reached. Demote static pins (never the
    // mandatory floor, whose Source carries pinned=true) back to evictable LRU
    // until the claim fits the allowance it now has.
    while (im.static_used > static_allowance(im)) {
        bool demoted = false;
        for (auto& kv : im.resident) {
            if (!kv.second.pinned) continue;
            auto b = im.bound.find(kv.first);
            if (b == im.bound.end()) continue;   // NO-SOURCE entry: see below
            if (b->second.pinned) continue;      // floor: never
            // The two `continue`s above are load-bearing in different ways. A
            // floor tensor must never be demoted. A NO-SOURCE entry (created by
            // init_tensor for llama-owned tensors) is marked pinned but never
            // added to static_used and never entered the LRU -- demoting it used
            // to subtract bytes that were never added, clamping static_used to 0
            // after a few iterations while the REAL static pins survived
            // untouched. The loop then reported success with the over-claim
            // intact, which on a tight-RSS machine sends output.weight to the
            // poison path: the exact failure this function exists to prevent
            // (2026-08-24 audit).
            kv.second.pinned = false;
            im.static_used -= std::min(im.static_used, kv.second.bytes);
            demoted = true;
            break;
        }
        if (!demoted) break;   // only floor remains; nothing more to give back
    }
    while (cache_used(im) > cache_budget(im) && !im.lru.empty()) {
        if (!make_room(im, 0)) break;
    }
    return cache_budget(im);
}

std::string Streamer::accountant_report() const {
    if (!impl_) return "";
    std::string r = impl_->acct.report();
    const uint64_t unres = impl_->acct.unreserved();
    if (unres) {
        std::ostringstream o;
        o << "  unreserved (llama.cpp buffers, vocab, allocator overhead)  "
          << (unres / 1e9) << " GB";
        r += std::string("\n") + o.str();
    }
    return r;
}

// I2: ONE traversal for size and name, so the pair can never drift again --
// H9 fixed the drift, H14 reintroduced it one commit later by widening one
// sibling. And TWO predicates, not one: routed experts are compacted under
// NO_ROWSLICE alone (the gates differ, stream_buffer's own dispatch is the
// authority), so counting them "whole" there refused caps that would run.
static void largest_whole_read(const Streamer::Impl& im, uint64_t* bytes,
                               std::string* name) {
    static const bool env_no_compact = [] {
        const char* v = std::getenv("DRAY_NO_COMPACT");
        return v && v[0] == '1';
    }();
    static const bool env_no_rowslice = [] {
        const char* v = std::getenv("DRAY_NO_ROWSLICE");
        return v && v[0] == '1';
    }();
    const bool no_compact = env_no_compact || im.cfg.no_compact;
    *bytes = 0;
    name->clear();
    for (const auto& kv : im.sources) {
        if (kv.second.pinned) continue;          // floor is resident, not cached
        if (kv.second.sliceable) {
            const bool whole = kv.second.routed ? no_compact
                                                : (no_compact || env_no_rowslice);
            if (!whole) continue;
        }
        if (kv.second.bytes > *bytes) { *bytes = kv.second.bytes; *name = kv.first; }
    }
}

uint64_t Streamer::churn_reserve_bytes() const {
    return impl_ ? impl_->churn_reserve : 0;
}

uint64_t Streamer::batch_region_bytes() const {
    return impl_ ? impl_->batch_region_bound : 0;
}

uint64_t Streamer::largest_streamed_bytes() const {
    if (!impl_) return 0;
    uint64_t b = 0;
    std::string n;
    largest_whole_read(*impl_, &b, &n);
    return b;
}

std::string Streamer::largest_streamed_name() const {
    if (!impl_) return "";
    uint64_t b = 0;
    std::string n;
    largest_whole_read(*impl_, &b, &n);
    return n;
}

bool Streamer::over_cap() const {
    if (!impl_) return false;
    return impl_->acct.used() > impl_->acct.cap();
}

bool Streamer::aborted() const { return impl_ && impl_->failures > 0; }

bool Streamer::abort_cb(void* p) {
    auto* s = static_cast<Streamer*>(p);
    return s && s->aborted();
}

uint64_t Streamer::failures() const  { return impl_ ? impl_->failures.load(std::memory_order_relaxed) : 0; }
uint64_t Streamer::routed_bytes_read() const { return impl_ ? impl_->routed_read : 0; }
std::string Streamer::io_describe() const {
    return impl_ && impl_->io ? impl_->io->describe() : std::string("no backend");
}
uint64_t Streamer::routed_bytes_needed() const { return impl_ ? impl_->routed_needed : 0; }
uint64_t Streamer::uncond_bytes_needed() const { return impl_ ? impl_->uncond_needed : 0; }
uint64_t Streamer::uncond_bytes_read() const   { return impl_ ? impl_->uncond_read : 0; }
uint64_t Streamer::compacted() const { return impl_ ? impl_->compacted : 0; }

std::string Streamer::report() const {
    const Impl& im = *impl_;
    std::ostringstream o;
    o << "streamer: budget " << (cache_budget(im) / 1e9) << " GB, in use "
      << (cache_used(im) / 1e9) << " GB, " << im.resident.size() << " tensors resident, "
      << (bytes_streamed_ / 1e9) << " GB streamed over " << nodes_ << " nodes, "
      << im.compacted << " expert-compacted, "
      << im.prefetch_hits << "/" << im.prefetch_issued << " prefetch hit, "
      << "pool " << (im.pool.enabled() ? im.pool.committed() / 1e9 : -1.0) << " GB committed "
      << im.pool.fallbacks() << " fallbacks, "
      << im.ring_hits << " ring-fed (front="
      << (im.ring_q.empty() ? "none" : (im.ring_q.front().t ? im.ring_q.front().t->name : "pad"))
      << (im.ring_q.empty() ? "" : (im.ring_q.front().consumed ? "/consumed" : "/UNCONSUMED"))
      << ", " << im.ring_segs << " segs, " << im.ring_pops
      << " pops, " << im.ring_calls << " calls; stops d=" << im.ring_stop[0]
      << " lap=" << im.ring_stop[1] << " wrapfull=" << im.ring_stop[2]
      << " full=" << im.ring_stop[3] << " refuse=" << im.ring_stop[4] << "), "
      << (im.static_used / 1e9) << " GB pinned static of "
      << (static_allowance(im) / 1e9) << " GB allowed, "
      << im.early_unlocks << " early-unlocked, " << im.ring_promotions << " promoted, "
      << im.eslot_hits << " eslot hits (" << (im.eslot_bytes / 1e9) << " of "
      << (im.eslot_pool / 1e9) << " GB pool)";

    // I/O FORENSICS, off unless asked for (DRAY_IO_STATS=1). These counters
    // located the bandwidth defect on 2026-08-24 and are worth keeping, but they
    // add 250 characters to a line that was already long. Default output is what
    // it was before that investigation.
    static const bool io_stats = [] {
        const char* v = std::getenv("DRAY_IO_STATS");
        return v && v[0] == '1';
    }();
    if (io_stats) {
      o << ", cb " << (im.ns_materialise / 1000000)
      << "ms mat/" << (im.ns_release / 1000000) << "ms rel over " << im.n_materialise
      << " calls, io " << im.io_batches << " batches/" << im.io_slices
      << " slices (max " << im.io_batch_max << "), qdepth mean "
      << (im.depth_n ? (im.depth_sum / im.depth_n) : 0) << " max " << im.depth_max
      << " min " << (im.depth_min == ~0ull ? 0 : im.depth_min)
      << " (" << im.depth_low << " submits found <4 in flight)";
    if (im.depth_t_last > im.depth_t0) {
        o << ", time-weighted depth "
          << (im.depth_area_ns / (im.depth_t_last - im.depth_t0));
    }
    if (im.lat_n) {
        o << ", read latency mean " << (im.lat_sum_us / im.lat_n) << "us over "
          << im.lat_n << " reads (" << (im.lat_bytes / im.lat_n / 1024) << " KiB mean): <200us "
          << im.lat_bucket[0] << ", <500 " << im.lat_bucket[1] << ", <1ms " << im.lat_bucket[2]
          << ", <2ms " << im.lat_bucket[3] << ", slower " << im.lat_bucket[4]
          << "; polls " << im.pump_calls << " mean gap "
          << (im.pump_calls > 1 ? im.pump_gap_us / (im.pump_calls - 1) : 0) << "us, "
          << (im.pump_calls ? im.pump_harvested / im.pump_calls : 0) << " harvested each"
          << "; gap buckets <100us " << im.gap_bucket[0] << " <1ms " << im.gap_bucket[1]
          << " <5ms " << im.gap_bucket[2] << " <20ms " << im.gap_bucket[3]
          << " longer " << im.gap_bucket[4] << ", max " << im.gap_max_us << "us, "
          << (im.gap_long_us / 1000) << "ms total spent in gaps over 5ms, "
          << (im.in_pump_ns / 1000000) << "ms BLOCKED INSIDE poll"
          << "; staging memcpy " << (im.memcpy_ns / 1000000) << "ms for "
          << (im.memcpy_bytes / 1000000) << " MB on the poll thread"
          << ", staging alloc " << (im.stage_alloc_ns / 1000000) << "ms over "
          << im.stage_alloc_n << " reads";
    }
    }   // end DRAY_IO_STATS
    // Repack is off by default and measured as a 20% loss on the streaming
    // path; its counters are noise unless someone has deliberately enabled it.
    if (im.repack_on) {
        o << ", repack " << im.repack_done << "/" << im.repack_tried
          << " (" << im.repack_no_traits << " no-traits)";
    }
    o << ", h_routed=";
    if (im.routed_needed) {
        o << static_cast<int>(100.0 * (1.0 - static_cast<double>(im.routed_read) /
                                             static_cast<double>(im.routed_needed)) + 0.5) << "%";
    } else {
        o << "unknown";
    }
    o << " h_bytes=";
    if (im.routed_needed + im.uncond_needed) {
        const double read_t = static_cast<double>(im.routed_read + im.uncond_read);
        const double need_t = static_cast<double>(im.routed_needed + im.uncond_needed);
        o << static_cast<int>(100.0 * (1.0 - read_t / need_t) + 0.5) << "%";
    } else {
        o << "unknown";
    }
    // Routing skew, measured: share of decode picks landing on each tensor's
    // top-10% most-picked experts, averaged. Uniform routing scores ~0.10 plus
    // sparsity noise; a frequency cache only earns its complexity well above that.
    if (!im.expert_hist.empty()) {
        double share_sum = 0.0;
        uint64_t picks_sum = 0;
        size_t counted = 0;
        for (const auto& kv : im.expert_hist) {
            const auto& h = kv.second;
            uint64_t tot = 0;
            for (uint32_t c : h) tot += c;
            if (tot == 0) continue;
            std::vector<uint32_t> s(h.begin(), h.end());
            std::sort(s.begin(), s.end(), [](uint32_t a, uint32_t b) { return a > b; });
            const size_t top = s.size() >= 10 ? s.size() / 10 : 1;
            uint64_t t10 = 0;
            for (size_t i = 0; i < top; ++i) t10 += s[i];
            share_sum += static_cast<double>(t10) / static_cast<double>(tot);
            picks_sum += tot;
            ++counted;
        }
        if (counted) {
            o << ", skew: top-10% experts take "
              << static_cast<int>(100.0 * share_sum / counted + 0.5)
              << "% of picks (" << (picks_sum / counted) << " picks/tensor, "
              << counted << " tensors)";
        }
    }
    if (im.failures) {
        o << "  [" << im.failures << " MATERIALISE FAILURES -- output is not trustworthy]";
    }
    return o.str();
}

MergedMetadata merge_shard_metadata(const plan::Plan& p, std::string* error) {
    MergedMetadata out;
    if (p.shard_paths.empty()) { if (error) *error = "no shards"; return out; }

    out.gguf = gguf_init_empty();
    if (!out.gguf) { if (error) *error = "gguf_init_empty failed"; return out; }

    // One ggml context holding metadata-only tensors for every shard. no_alloc, so
    // these carry shape and type but no data -- which is all llama.cpp needs to
    // build the model; the bytes arrive later, from the streamer.
    struct ggml_init_params ip = {};
    ip.mem_size   = static_cast<size_t>(p.tensors.size() + 64) * ggml_tensor_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) {
        gguf_free(out.gguf); out.gguf = nullptr;
        if (error) *error = "ggml_init failed";
        return out;
    }

    const size_t n_meta = p.n_meta_shards ? p.n_meta_shards : p.shard_paths.size();
    for (size_t s = 0; s < n_meta; ++s) {
        struct ggml_context* shard_ctx = nullptr;
        struct gguf_init_params gp = {};
        gp.no_alloc = true;
        gp.ctx = &shard_ctx;
        struct gguf_context* g = gguf_init_from_file(p.shard_paths[s].c_str(), gp);
        if (!g) {
            if (error) *error = "cannot reopen shard for metadata: " + p.shard_paths[s];
            free_merged_metadata(out);
            return out;
        }
        // Shard 0 carries the model metadata; later shards carry only tensors.
        if (s == 0) gguf_set_kv(out.gguf, g);

        for (struct ggml_tensor* t = ggml_get_first_tensor(shard_ctx); t != nullptr;
             t = ggml_get_next_tensor(shard_ctx, t)) {
            struct ggml_tensor* copy = ggml_new_tensor(out.ctx, t->type, GGML_MAX_DIMS, t->ne);
            if (!copy) {
                if (error) *error = "out of metadata context space";
                gguf_free(g); ggml_free(shard_ctx);
                free_merged_metadata(out);
                return out;
            }
            ggml_set_name(copy, ggml_get_name(t));
            gguf_add_tensor(out.gguf, copy);
            ++out.n_tensors;
        }
        gguf_free(g);
        ggml_free(shard_ctx);
    }

    // Drop the split keys: there is one merged description now, and leaving them
    // makes llama.cpp hunt for shard files it will never open.
    for (const char* k : { "split.no", "split.count", "split.tensors.count" }) {
        const int64_t id = gguf_find_key(out.gguf, k);
        if (id >= 0) gguf_remove_key(out.gguf, k);
    }
    return out;
}

void free_merged_metadata(MergedMetadata& m) {
    if (m.gguf) { gguf_free(m.gguf); m.gguf = nullptr; }
    if (m.ctx)  { ggml_free(m.ctx);  m.ctx  = nullptr; }
    m.n_tensors = 0;
}

}  // namespace dray::backend
