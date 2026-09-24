// Install-time repack: rewrite the routed experts expert-major into a companion
// file so one expert's gate/up/down sit adjacent and 4096-aligned on disk.
//
// WHY: decode issues ~3 x k x n_moe_layers scattered reads per token at the
// GGUF's tensor-major layout. Expert-major flips the request shape: same bytes,
// same logical tensors (memory layout untouched -- ggml never knows), but each
// expert's slices are neighbours and every start is aligned, so nothing widens
// and the drive sees fewer, larger, aligned requests. The engine reads the
// companion through the SAME compact path via Source::disk_stride.
//
// The companion is ADDITIVE in v1: originals stay (uncond still reads from
// them), costing extra disk. An installer that relocates uncond too and drops
// the originals is the v2 that makes repack a replacement, not a copy.
#pragma once

#include <string>

namespace dray::plan { struct Plan; }

namespace dray::tools {

// Writes <out_dir>/repack.bin + <out_dir>/repack.json. Returns 0 on success.
// Buffered I/O by design: this is a one-time installer writing hundreds of GB,
// not the engine's read path; Invariant 2 governs runs, not installs.
int repack_write(const std::string& model_gguf, const std::string& out_dir);

// If <dir>/repack.json exists, rewrites the plan's routed tensors to source from
// the companion (appended as one more shard) and sets disk_stride. Returns true
// if a map was found and applied; false (untouched plan) otherwise.
bool repack_apply(dray::plan::Plan* p, const std::string& dir, std::string* err);

}  // namespace dray::tools
