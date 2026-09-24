# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Dray is

An inference endpoint that runs frontier-scale MoE models on one workstation by keeping most of the model on
SSD and streaming what the forward pass actually needs. Only a small mandatory floor is truly resident; every
weight streams when the RAM cap is tight.

Three constraints define it:

- **Small.** The original target was ~5-6k lines; the measured tree is ~14k, and the
  overage is bought deliberately: the server/job layer, the honesty instrumentation,
  and three platform I/O backends. The part that was the point stays true: anything
  model-specific is ~100 lines, and adding a model that reuses existing mixer classes
  is zero lines of engine code.
- **Small RAM is the design center.** A 4–16 GB cap on a 16–32 GB machine, leaving the system usable. RAM is a
  quality dial, never a gate.
- **Slow, honestly, and provably at the hardware limit.** For K3 the realistic unit is **seconds per token**.
  That is stated up front. What is not acceptable is leaving the user unable to tell whether the engine or the
  drive is the limit.

Never report an optimistic figure, never quietly drop a measurement because it looks bad.

## Status

Built and measured. NINE models generate correct text through one arch-blind
planner: the 1B-7B testbed (control), GLM-5.2 744B, Qwen3.8 2.4T, Kimi K3 2.8T,
MiniMax-M3 429B, Qwen3.8-27B (dense), Qwen3.6 35B-A3B, Qwen3.5 122B-A10B and
DeepSeek V4 Flash 284B (hyper-connections + MLA).

**The 2026-08-23 headline, which reframes the project:** cost tracks ACTIVE
bytes per token, not parameter count. Qwen3.6 35B-A3B costs 287 MiB/token at
1.5 s/token; Qwen3.5 122B-A10B 872 MiB at 2.5; DeepSeek V4 Flash 284B 1.18-1.93
GB. Kimi K3 costs 54.5 GB/token for scale. All three sparse models run correct
text on a THREE GiB cap. Batched at width 32 with varied prompts: 35B-A3B 0.213
GB/token (0 bytes at 28 GiB, fully resident), 122B-A10B 0.406, DeepSeek 0.892.
The 122B clears width 96 on 28 GiB. Prefill: a 6,594-token prompt takes 22
seconds with --gpu against 47 minutes on CPU. Resident mode (opt-out, when the
model fits the cap) reaches parity with stock llama.cpp; --force-stream pins the
streaming path and every gate uses it.

Correctness is sealed on Windows/Linux/macOS at identical node counts; the
reference performance numbers are Windows (DECISIONS.md, with commits). The
OpenAI-compatible server ships with background jobs and crash-resume. Earlier
batch figures for the large flagships (2026-08-19, same conditions): K3 5.45x at
width 32, Qwen 3.77x at 16, GLM 2.55x at 44 with `--kv q4`. The MULTIPLIER rises
with routing sparsity but SHRINKS as absolute cost falls, because there is less
left to share once a token costs under a gigabyte; read the absolute column.
Over-width runs refuse AT LOAD with the required cap priced. Width is bound by
per-sequence state (KV + recurrent), not the weight cache -- which is why the
122B reaches width 96 (176 MiB per sequence) where K3 stops at 44 (~1 GB). Cohort rotation (`--rotate`, opt-in, write-honest,
`--state-dir` explicit) serves deeper queues; mid-generation parking is guarded off —
the fork's llama_state_seq restore is measurably nondeterministic (minimal repro in
DECISIONS). The original build-order paragraph is preserved in git history; its
M3-first plan was overtaken by events — K3 support arrived via an upstream community PR
carried in the vendored fork, and GLM-5.2 joined as the small flagship. MiniMax M3 is
measured (2026-08-20: correct text at 8 GiB, 4.61 GiB/token — the profile kv_defaults
mechanism supplies five indexer keys the released conversion lacks); Qwen3.8's q1-dtype
branch reconversion remains open.

Models live outside the tree — `D:\Models\unsloth\<repo>`, never in-repo (see `.gitignore`).

## Toolchain

C++20, CMake. **llama.cpp** vendored as a submodule of our own fork, pinned to a SHA, linked directly — no FFI,
because the scheduler repoints `ggml_tensor` data on the hot path. ggml alone is not enough: the tokenizer,
sampler and per-model graphs are all llama.cpp. Compute is **CPU-only for v1**
(`-DGGML_METAL=OFF -DGGML_CUDA=OFF`); the cache slab is host memory.

```bash
cmake --preset dev                      # configure
cmake --build --preset dev -j           # build
ctest --preset dev                      # full suite
ctest --preset dev -R graph_reuse       # one test
ctest --preset dev -R graph_reuse -V --output-on-failure
```

Per-platform I/O backends belong in presets, not `if(WIN32)` scattered through `CMakeLists.txt`.

### The gates

Three run before every commit, on the testbed model (fast, one small model).
The last two are slower and run before shipping rather than before each commit:

```bash
scripts/difftest.ps1     # compaction ON vs OFF, byte-identical text
scripts/residenttest.ps1 # resident vs streaming allocation, byte-identical text
scripts/batchdiff.ps1    # within-batch identity, and compact ON vs OFF
scripts/archgate.ps1     # ARCHITECTURE DIVERSITY -- slow, before shipping
scripts/clonegate.ps1    # DOES THIS BUILD FOR A STRANGER -- before any push
                         # that touches the submodule or build files
```

`archgate` is the newest and exists because the other three cannot catch what
they cannot see: they all passed while a node-batching optimisation produced
fluent, wrong text on DeepSeek V4 Flash, whose hyper-connection weights arrive
through graph shapes the other models never build. It runs a short greedy
generation on every architecture present and compares against a recorded
prefix. It takes minutes per model, so it is not pre-commit -- but nothing that
touches the streamer's node handling, materialisation, graph interception, THE
I/O BACKEND OR THREADING ships without it. Absent models are reported as
skipped, never silently passed.

The I/O and threading clause was added on 2026-08-24 after a completion-
harvester thread passed difftest and residenttest on the testbed and then
failed with MATERIALISE FAIL on the 35B -- a model archgate covers and the
fast gates do not. RUN IT WITH THE FLAG ON: a feature behind an env var is
not exercised by a default-configuration gate, which is how that attempt got
as far as it did.

`clonegate` answers the one question a local build cannot: it clones this repo
into a scratch directory, fetches the submodule from its REMOTE, builds, and
checks that a binary exists and is newer than 30 minutes -- the artifact, not
the exit code. It was written after the tree pinned a fork commit that lacked
three functions the engine calls, so every local build passed and a clone would
have died at the link. It also refuses immediately if the submodule has
uncommitted changes, which is that same defect one step earlier.

Any measurement or gate must be the only thing using the drive. A download
running alongside a timing run makes the wall clock meaningless (bytes stay
valid; they are deterministic).

**Running the Linux gate from Windows: use PowerShell, not Git Bash.** Git
Bash rewrites Unix absolute paths into Windows ones, so `wsl -- /home/you/bin/
cmake` silently becomes `C:/Program Files/Git/home/you/bin/cmake`, the command
is not found, and the failure still surfaces as exit 0 through the pipeline.
That produced a reported-green Linux build on 2026-08-23 that had compiled
nothing -- the giveaway was a binary three days old. Check the artifact's
timestamp, not the exit code:

```powershell
wsl -d Ubuntu-22.04 -- /home/you/tools/cmake/bin/cmake --build /home/you/dray-build -j 8
wsl -d Ubuntu-22.04 -- env BIN=/home/you/dray-build/bin/dray bash /mnt/d/Projects/AI/dray/scripts/linux-gate.sh /mnt/d/Models/.../model.gguf 8G
```

## Invariants

1. **Resident memory never exceeds the configured cap, and the cap means *total* resident bytes.** The
   mandatory floor and KV reservation are computed at load and subtracted to yield the cache budget, never
   added on top. Every allocation goes through one accounted allocator; nothing on the decode path calls a
   general-purpose allocator. llama.cpp's own KV, graph scratch and vocab are inside the cap and must be
   intercepted or pre-reserved. The default cap is polite and never auto-expands.
2. **Reads never *retain* OS page cache, and tensor data is never mmap'd.** ("Retain", not "populate" —
   `RWF_DONTCACHE` instantiates folios and prunes them.) mmap is fine for the GGUF header and tensor table.
3. **Never assume bits-per-weight, anywhere.** Slot sizes, coalescing widths and every byte figure come from
   the GGUF tensor table. This is what makes all quant tiers work with no per-quant code.
4. **A model name appears only in the registry.** No branching on architecture identity anywhere else.
5. **Alignment, page size, sector size, block size and queue depth are discovered at runtime**, never
   hardcoded — the drive population is too varied for defaults.
6. **The readout is measured, live, honest and durable.** Seconds per token by default; ETAs disclose the rate
   window they used; counters are integer or float64, never float32; anything unobtainable is reported
   *unknown*, never defaulted.
7. **Admission is serialized by default.** Batching past ~8 dissolves the streaming premise.
8. **Policy results are not generalized across models** — gains are strongly model-dependent.

## Architecture

1. **Residency planner** — classifies every tensor, reserves the floor and the KV ceiling for the requested
   `n_ctx` up front, refuses admission if it does not fit. The cache never shrinks after load. At small caps
   this component decides almost all achieved performance.
2. **Expert cache** — arena, index, refcounts, eviction. Slots for a layer live in **one contiguous per-layer
   arena of uniform slot size**, so that layer's `ffn_*_exps` is a valid ggml tensor with `nb[2] = slot_size`;
   a per-token remap tensor maps router-chosen ids to slot indices for `ggml_mul_mat_id`. This is forced —
   `mul_mat_id` indexes all experts by one uniform stride, so scattered slots cannot be expressed by
   repointing `tensor->data`. It is also why slots are per-layer size classes.
3. **Prefetch scheduler** — router output and speculation → coalesced ordered reads → queue sized by
   calibration. Prefill runs a different policy from decode.
4. **Storage backend** — platform-specific uncached async reads behind one interface. Its submission and
   completion shape sets the threading model for everything above it and must be decided first.
5. **llama.cpp integration** — a custom ggml buffer type over host memory (the
   streamer repoints tensor data per node via cb_eval; Metal wraps the same arena
   on Apple).
6. **Job server / CLI** — the public protocol is an **OpenAI-compatible API only** (owner,
   2026-08-14), in the spirit of llama-server: no interface, no dashboard, ever. Underneath
   it remains a job engine — a multi-day generation cannot survive an HTTP request, so long
   runs are jobs with checkpoint/resume, surfaced through the OpenAI protocol (streaming;
   reconnectable request ids).
7. **Persistence** — checkpoints and resume. Cannot be retrofitted: every I/O primitive above is a *read*.

## Adding a model must be additive

Most of a model is data: arch id, per-layer schedule, expert counts, top-k, policy defaults. Only a genuinely
new mixer class or routing scheme is code. The three targets need **two state classes** — sliceable KV (M3)
and opaque fixed-size recurrent state (K3's KDA and Qwen3.8's Gated DeltaNet share it).

```
src/models/registry.cpp     # the ONLY file that lists models — one line each
src/models/profile.h
src/models/minimax_m3.cpp   # the one profile that exists; qwen38/kimi_k3 run
                            # today with NO entry (arch-blind planner) and get
                            # one only if they ever need profile-expressed
                            # behaviour. src/state/ is future work for sliced
                            # recurrent checkpoints; whole-state snapshots
                            # cover resume today.
```

A new model reusing existing mixer classes is a profile entry and nothing else. If a model needs behaviour the
profile cannot express, extend the schema or add a hook interface — never add a branch on the name.

## The cost model

```
per_token_disk_time = misses_per_token × bytes_per_miss / realized_read_bandwidth
    misses_per_token = k_routed × n_moe_layers × (1 − h_routed)
```

Only `n_moe_layers` is fixed by architecture. When proposing a change, state which term it moves.

`realized_read_bandwidth` is **measured on this drive, at the engine's own block size and queue depth**, never
a vendor sequential rating. Decode is a large-block *gather*, so `fio --rw=read` is the wrong profile for it —
but it is the *right* profile for prefill, which is a single ordered scan of the whole model.

Two distinct hit rates, never one symbol: **`h_routed`** = cache bytes ÷ routed-expert bytes (feeds the model
above; never use a whole-file denominator, which flatters thrashing configs), and **`h_bytes`** = fraction of
total active bytes per token served from RAM (what the RAM-vs-disk regime crossover requires, since active
bytes include the unconditional stream). They diverge sharply — at the knee `h_routed` is 0 while `h_bytes` is
~87–90%.

**Below the knee, cache *policy* is irrelevant and cache *size* is everything.** A forward pass sweeps every
layer once, so a working set larger than the cache is a cyclic scan — the classic LRU pathology, where every
entry is evicted before its reuse and the hit rate is exactly 0%. Measured: adding value-per-byte eviction
priority changed bytes-read by nothing at all (594.007 GB, byte-identical); pinning a fixed prefix instead took
it to 483.884 GB. Reordering evictions and fetching early are both worthless here; only "how much you never
evict" moves the number. The fill order above applies *above* the knee, and the RAM curve's assumption that
cached bytes stay cached holds only there.

**Wall time cannot settle a question this size on this hardware** — the same configuration measures ~10% apart
run to run. Conclusions come from bytes-read, which is deterministic.

## Working notes

The verified measurements behind all of this — per-model GGUF splits, mandatory-floor arithmetic, per-platform
I/O facts and traps, prior-art numbers, the eviction hypothesis and its falsification plan, multi-day operating
requirements, and the list of open decisions — are kept in project memory rather than here. Consult them before
changing anything quantitative; several are non-obvious and expensive to rediscover.

## Rule scope

`~/.claude/CLAUDE.md` targets TypeScript/React; its styling, JSX and `useEffect` rules do not apply. Its process
rules do: no commits without an explicit request, and no running tests, benchmarks or servers without asking.
That matters more than usual here — a run means hours of disk-saturating I/O, the drive's thermal state carries
between runs so back-to-back results are not comparable, and sustained reads at this volume are not
endurance-free.
