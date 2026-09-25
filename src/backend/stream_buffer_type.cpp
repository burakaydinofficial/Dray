#include "backend/stream_buffer_type.h"

#include <cstdint>

#include "backend/streamer_impl.h"

namespace dray::backend {

namespace {

Streamer::Impl* impl_of(ggml_backend_buffer_type_t buft) {
    return static_cast<Streamer::Impl*>(buft->context);
}
Streamer::Impl* impl_of(ggml_backend_buffer_t buf) {
    return static_cast<Streamer::Impl*>(buf->context);
}

// On the metal path the buffers carry Metal's context, not ours, so the hook
// resolves the engine through this stash (single engine per process by design;
// serve is one engine, run is one engine).
Streamer::Impl* g_metal_impl = nullptr;

const char* dray_buft_name(ggml_backend_buffer_type_t) { return "dray_stream"; }
size_t dray_buft_alignment(ggml_backend_buffer_type_t) { return kHostAlign; }
size_t dray_buft_max_size(ggml_backend_buffer_type_t) { return SIZE_MAX; }
// Host-accessible: the CPU backend must be able to compute directly against the
// pointers materialise() installs. Saying otherwise makes ggml_backend_sched treat
// every weight as living on a foreign backend and try to copy the whole model into
// a CPU buffer, which dies in graph_reserve.
//
// The cost of saying host is that the loader reads file bytes STRAIGHT into
// tensor->data, bypassing set_tensor. Streamed tensors therefore need a real
// writable landing zone during load -- the sentinel (PoisonBuffers) -- shared by
// all of them because the loader is sequential and we discard what it writes.
bool dray_buft_is_host(ggml_backend_buffer_type_t) { return true; }

void dray_buffer_free(ggml_backend_buffer_t) {}

void* dray_buffer_get_base(ggml_backend_buffer_t buf) {
    // The allocator assigns tensor->data = base + offset before calling
    // init_tensor. We overwrite data there, so this address is never dereferenced
    // -- but it must be non-null and aligned or the allocator's arithmetic trips.
    return impl_of(buf)->fake_base;
}

enum ggml_status dray_buffer_init_tensor(ggml_backend_buffer_t buf, ggml_tensor* t) {
    Streamer::Impl* im = g_metal_impl ? g_metal_impl : impl_of(buf);
    return im->on_init_tensor(t);
}

void dray_buffer_set_tensor(ggml_backend_buffer_t buf, ggml_tensor* t,
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

void dray_buffer_get_tensor(ggml_backend_buffer_t buf, const ggml_tensor* t,
                           void* data, size_t offset, size_t size) {
    impl_of(buf)->on_get_tensor(t, data, offset, size);
}

void dray_buffer_clear(ggml_backend_buffer_t, uint8_t) {}

#if defined(__APPLE__)
void dray_metal_noop_clear(ggml_backend_buffer_t, uint8_t) {
    // A real clear would memset the whole mapping and commit the entire 2x-cap
    // reservation. Weights are never cleared on the metadata load path anyway.
}
void dray_metal_noop_free(ggml_backend_buffer_t) {
    // Views share the master mapping's context; only the master frees it.
}
#endif

ggml_backend_buffer_t dray_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
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
        Streamer::Impl* im = impl_of(buft);
        if (im->flags.metal && im->mem.arena().enabled()) {
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
                    gpu, im->mem.arena().base(),
                    static_cast<size_t>(im->mem.arena().span()),
                    static_cast<size_t>(im->mem.arena().span()));
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

}  // namespace

void init_stream_buffer_type(Streamer::Impl& im) {
    im.buft.iface.get_name       = dray_buft_name;
    im.buft.iface.alloc_buffer   = dray_buft_alloc;
    im.buft.iface.get_alignment  = dray_buft_alignment;
    im.buft.iface.get_max_size   = dray_buft_max_size;
    im.buft.iface.get_alloc_size = nullptr;   // defaults to ggml_nbytes
    im.buft.iface.is_host        = dray_buft_is_host;
    im.buft.device  = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
}

}  // namespace dray::backend
