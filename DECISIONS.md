# dray lab notebook

What this engine does, what was measured, what turned out to be wrong, and what is
still open. Organised by topic, current state first. Every figure states its
conditions; where an earlier figure was withdrawn, the withdrawal is recorded so a
reader who met the old number knows why it moved.

How to read the numbers:

- **Bytes are the evidence; seconds are supporting.** Bytes read per token are
  deterministic for a given build, cap and prompt. Wall time on this machine varies
  about 10% run to run, so timings are quoted as interleaved runs (A/B/B/A or three
  per build) with the spread shown.
- **The power plan is part of the measurement.** On the test laptop, Windows
  "High performance" versus "Balanced" moves K3 by about 25%. Timings are only
  compared within one plan, and newer entries name it.
- **A measurement needs the drive to itself.** A run that shared the machine with
  other heavy work is not evidence, even when its pairs look self-consistent.

---

## The test machine

- 22-thread laptop CPU, 79.7 GB RAM, RTX 4070 Laptop GPU (8 GB, used only with
  `--gpu`), WD_BLACK SN850X 4 TB (PCIe 4.0 x4). Windows 11 unless noted.
- Drive calibration by `dray calibrate`: 6.82 GB/s sequential (2 MiB x QD16),
  6.63 GB/s gather (1 MiB x QD16). On Kimi K3's own fourteen shards: 2 MiB random
  reads at 4.00 GB/s at QD1, 5.88 at QD4, 6.03 at QD16, 6.11 at QD64. Large-block
  gather is ~97% of sequential on this drive; queue depth, not access pattern, is
  what matters (QD1 loses 2-13x).

---

## Current state (2026-09-27)

**Kimi K3 2.8T (UD-IQ1_S, 594 GB), 9 GiB cap, 16 tokens, High performance plan.**
Each step below is one change, measured A/B against the build before it, with
identical generated text:

| change | decode s/token | GB read (16 tokens + prefill) |
|---|---|---|
| start of 2026-09-26 (CPU-only build) | 13.0-13.2 | 989 |
| ggml scheduler metadata sized on demand | 11.9-12.4 | 934 |
| the ring streams by tensor class, not shape | 10.6 | 933 |
| expert reads land in place (no bounce buffer) | 8.8-9.0 | 917 |
| dead expert regions reused, not freed | 8.6-8.7 | 917 |

About 34% faster than the start of those two days. At 8.7 s/token K3 reads roughly
51 GB per decode token, about **5.9 GB/s against a drive that calibrates at
6.7-6.9 GB/s** -- close to 90% of the hardware. At this cap the remaining lever is
fewer bytes per token (more RAM), not better I/O scheduling.

The README's per-model performance table predates this work (older build,
Balanced plan) and understates the current engine; it will be re-measured as a
whole rather than patched row by row.

---

## Memory: the cap means total resident bytes

The cap covers everything the process holds: the mandatory floor (router gates,
norms, KV, recurrent state, scratch), I/O staging, the weight cache, and llama.cpp's
own buffers. The ledger is reconciled against the OS (the worse of resident set and
commit charge) at load and every token; whatever the ledger cannot explain is
charged as "unreserved" and comes out of the cache.

**Why the reconciliation exists.** An early Qwen3.8 claim of correct text at a
4 GiB cap was wrong: the ledger was not counting I/O staging, cache slots or
llama.cpp's buffers, and peak commit on a "successful" 12 GiB run was 13.59 GB.
With everything counted, 4 GiB cannot hold the 1.09 GiB embedding; the measured
minimum was 5 GiB. Zero reported failures and fluent output were both true and
meaningless -- a counter that only counts what it knows cannot see what it never
knew about.

**3.5 GB of the cap was scheduler metadata nobody touched (2026-09-26).** Sampling
commit after each load step showed creating the llama context committed 4.78 GB and
touched 0.68. ggml's scheduler pre-allocated split-copy metadata for one split per
graph node (~20 KB per node, plus backend-id arrays at 61x the graph size); a CPU
graph has one split. At K3's graph budget of 164,672 nodes that was ~3.5 GB of
commit, never touched, which Windows charges and the cap therefore took from the
cache. The vendored fork now sizes it on demand. K3 at 9 GiB: cache budget 1.17 ->
4.66 GB, 989 -> 934 GB read. (Linux and macOS never saw it: untouched memory is free
there.)

**Measured minimum caps** (correct text required; below them the engine refuses at
load and names a cap that works): Qwen3.6 35B-A3B 3 GiB, Qwen3.5 122B-A10B 3 GiB,
DeepSeek V4 Flash 3 GiB, Qwen3.8-27B dense 4 GiB (1-bit: 3 GiB), MiniMax-M3 4 GiB,
GLM-5.2 3 GiB, Qwen3.8 2.4T 5 GiB, Kimi K3 9 GiB. K3's floor is set by
`output.weight` (918 MiB), which must be materialised whole.

---

## Correctness: the failure mode is fluent wrong text

Almost every serious bug in this engine produced plausible output rather than an
error. The gates exist because of specific escapes:

- **Expert compaction and graph reuse.** Compaction repoints a MUL_MAT_ID node at a
  private remapped ids tensor. llama.cpp reuses graphs across decode steps, so an
  early version derived the second token's mapping from its own remapped ids: first
  token right, every later one wrong. Found only by running the same model with and
  without the streamer (`difftest`). The shared router ids are never modified.
- **Views over streamed weights.** ggml fixes a view's data pointer at graph
  allocation time; a weight reached only through a reshape (K3's `ssm_a`) was never
  materialised and computed against poison. Every view is now re-pointed after its
  parent lands.
- **Skipping graph nodes.** Letting ggml batch nodes that touch no streamed weight is
  worth 11-15% (27B: 2.6 -> 2.2 s/token; 122B: 2.7 -> 2.4). The first version skipped
  nodes whose views needed re-pointing and produced fluent nonsense on DeepSeek V4
  while three models and four gates passed; its advertised 42% gain came partly from
  skipping work it needed. It ships as an opt-in (`DRAY_FAST_NODES=1`), verified on
  five architectures.

**The gates.** `difftest` (compaction on/off, byte-identical text), `residenttest`
(llama's allocator vs the streaming path), `batchdiff` (within-batch identity), `golden`
(25 cases: every command and lever, text and counters), `serve-smoke` (every server
route, including crash-resume), `archgate` (short greedy generations on five
architectures: dense, gated delta-net, sparse MoE, hyper-connections + MLA; run with
each streamer lever on and off), `clonegate` (a fresh clone from the remote builds and
passes its tests -- a submodule pin that existed only locally once built everywhere
except for anyone else). Unit tests cover the I/O scheduler against a scripted fake
backend, and each new test is checked by injecting the bug it targets.

---

## The I/O path

**The reads only progressed between graph nodes (found 2026-08-24).** The thread
that submitted and collected reads was the thread that computed, so during every
matmul nothing was harvested or submitted. On a K3 run about 91 of 121 seconds had
nobody polling the completion queue; realised bandwidth was 1.8 GB/s against a drive
that serves 6 GB/s on the same files. Earlier explanations (shallow batching, ring
starvation, block size) were each measured and ruled out on the way.

**A dedicated I/O thread (2026-09-26).** One thread owns the storage backend and
nothing else calls it; it keeps the queue full, collects completions and copies
staged bytes into place while the CPU computes. It changes when a read lands, never
what is read, so output and bytes are identical with it on or off
(`DRAY_IO_THREAD=0` restores the old path). An earlier attempt added a second
consumer inside the Windows backend and corrupted reads; the rule now is one
consumer. Verified with ThreadSanitizer on Linux. K3 12.25 GiB: ~10% faster on the
Balanced plan, ~6% on High performance.

**Reads a node is waiting on go first.** The scheduler keeps two queues: urgent reads
(anything a node will wait on) are submitted before ring read-ahead.

**The ring streams by class, not shape (2026-09-26).** The ring pre-reads the
unconditional weights in last pass's order. It excluded any 3-D tensor as "an expert
tensor", which dropped K3's 3-D attention and conv weights into synchronous reads
(~190 per token). The planner's tensor class is now the only filter. K3 9 GiB:
12.3 -> 10.6 s/token.

**Expert reads land in place (2026-09-27).** Unbuffered reads need aligned offsets,
lengths and buffers, so every expert slice used to be read into a bounce buffer and
copied into its region -- 220 GB of copying over one 16-token K3 run. Every expert
stride on the target models is a multiple of 4096 bytes, so all experts of a tensor
share one offset remainder; regions now place their data at that remainder and each
expert's aligned middle is read straight into its slot, with only the sub-4 KiB edges
staged (2.9 GB copied instead of 220). K3 9 GiB: 10.8 -> 8.9 s/token, 932 -> 917 GB.

**Dead expert regions are reused (2026-09-27).** Where allocating a region would
evict one anyway, a dead region of exactly the right size is handed over instead of
being freed and re-committed (about 4,400 per 16-token K3 run). K3 9 GiB: 8.9 -> 8.7
s/token. A first version also reused regions while there was free room, discarding
cached experts for nothing; a pressure test caught it as a cap breach.

**What did not pay.** Predicting the next layer's experts from its router a layer
early works (73% of picks on one model, 61% on K3), but reading the guesses cost
more bytes than it saved, and at tight caps there is no room to hold them.
Install-time repacking into interleaved layouts is correct and neutral on this
drive. Growing the ring beyond its default buys nothing: it is consumer-limited.

---

## Compute

- **One thread-pool barrier per claimed node.** With an eval callback installed,
  ggml runs the graph node by node with a full synchronise after each claimed node.
  The default claims every node -- the only shape proven on every architecture --
  and pays for it; `DRAY_FAST_NODES=1` claims only nodes that can reach streamed
  memory (11-15%, see Correctness). The cost vanishes into disk time once a model
  streams more than a few GB per token.
- **Thread oversubscription.** ggml's pool spin-waits; with the engine's own threads
  on top, requesting all 22 threads ran about 40x slower. The pool is clamped to
  (cores - 9); decode defaults to 4 threads (flat above that, bandwidth-bound),
  prefill to the clamp.
- **Resident mode.** When the whole model fits the cap, llama.cpp allocates it
  natively. It is correct and much faster than streaming the same model, but still
  21-46% slower than stock llama.cpp (see the README table); if a model fits, use
  llama.cpp. `--force-stream` pins the streaming path, and every gate uses it.
- **For sparse models the disk is free.** Qwen3.5-122B-A10B runs at 2.7 s/token
  reading 872 MiB per token at 12 GiB and at 2.7 s/token reading nothing at 56 GiB.
  Cost tracks active bytes per token, not parameter count; below roughly a gigabyte
  per token, compute is the limit.

---

## GPU (`--gpu`, opt-in at build and run time)

- **Decode loses** on every model measured (15 of 15 pairs): weights live in host
  memory, so offloaded decode drags them across PCIe every token.
- **Prefill wins** by 12-72x: a 6,594-token prompt on a 27B takes 22 s instead of
  47 minutes. The KV cache stays in RAM and weights never move to VRAM; ggml offloads
  only large-batch ops, so prefill goes to the GPU and decode does not.
- **It raises the minimum cap**, because the Vulkan compute buffer is inside it.
- **Vulkan is a build-time opt-in** (2026-09-26): merely compiling the Vulkan backend
  in cost K3 about 11% on CPU runs (ggml initialises it eagerly), so the default
  build is CPU-only and `--gpu` refuses on it with instructions.

---

## Batching

One decode step reads the union of every sequence's expert selections, so
concurrent sequences share most of the stream. Measured at 28 GiB, 4k context,
distinct prompts, decode bytes only (full table in the README):

- Multiplier rises with routing sparsity and falls as absolute cost falls: K3 5.45x
  at width 32 (5.98 GB/token); Qwen3.5 122B 2.15x (0.406 GB/token); Qwen3.6 35B at
  28 GiB is fully resident and reads nothing.
- **Width is bound by per-sequence state** (KV plus recurrent), not the weight cache:
  about 1 GB per sequence on K3 (two thirds of the budget at width 32), 176 MiB on
  the 122B, which clears width 96. Over-width runs are refused at load with the
  required cap quoted.
- **The table is a best case.** Sequences diverge as they generate, so the shared
  union widens (GLM width 44: 2.40 GB/token at 6 tokens, 2.64 at 18). Similar
  prompts share more: templated prompts measured 34% cheaper than varied ones at the
  same width, and identical prompts reached 20.5x on K3. Varied-prompt figures at
  widths the table does not list are unmeasured, not extrapolated.
- **Joint batched prefill** cut prefill bytes 40% at width 32 on K3.
- **Cohort rotation** (`--rotate`, opt-in) serves more prompts than the cap funds by
  running cohorts to completion, parking state to a directory the user chooses; it is
  the engine's only sustained-write feature and reports its write bytes.
  Mid-generation parking is refused: llama.cpp's per-sequence state restore was
  measured nondeterministic.

---

## KV cache quantisation

`--kv q4|q8|f16` quantises only the attention KV cache. It pays most on attention-state
models (GLM-5.2 saves 3.8 GiB at 32k context and batches from width 32 to 44) and
least on recurrent-state models, whose fixed state no knob touches (K3 keeps 443 MiB).
A dense 27B holds its full 262,144-token context in 11 GiB with q4. Quality is
measured per model, not assumed. `--kv q4` does not work on Kimi K3 at all
(quantised KV forces flash attention, and context creation fails).

---

## Platforms

- **Windows** is the reference for performance numbers.
- **Linux** (Ubuntu, g++, `O_DIRECT`): correctness sealed with node counts identical
  to Windows; two bugs only Linux exposed (a poison buffer too small for plain
  MUL_MAT, unevictable static pins after a budget shrink) were fixed for all
  platforms. Linux timings were taken in a VM and are indicative only.
- **macOS** (Apple silicon, `F_NOCACHE` thread pool): correctness sealed; GLM-5.2
  (744B) answered correctly on a fanless MacBook Air from a USB SSD in 3 GiB,
  saturating the link (0.41 GB/s of a 0.40 GB/s calibrated ceiling). `F_NOCACHE`
  does not evict existing page cache, so a model written to the Mac must be purged
  before measuring, and any ceiling above the physical link is contamination.
  Metal over the streamed weights is correct in whole-tensor mode but not yet faster;
  it needs windowed materialisation.
- CI builds and tests Windows, macOS, Linux and a ThreadSanitizer job on every push.

---

## Corrections worth knowing

- **"Resident mode reaches parity with stock llama.cpp"** -- it does not; stock is
  21-46% faster when a model fits.
- **"Kimi K3 runs at 8 GiB at 16.1 s/token, 6 GiB minimum"** -- not reproducible once
  memory accounting became complete; the minimum is 9 GiB.
- **"Qwen3.8 at 4 GiB"** -- the ledger was not counting everything; the minimum is 5 GiB.
- **"Node skipping is worth 42%"** -- measured on a broken version; the correct
  version is worth 11-15%.
- **The K3 slowdown after its first published figure** was two things, neither in
  the streamer: the scheduler metadata above (bytes) and the unused Vulkan backend
  compiled in (time).
- **Direct expert reads were first timed while other work shared the machine**; those
  figures were withdrawn and re-measured (the ones above).
- **Batch figures** first divided prefill-inclusive bytes by decode steps, and were
  first measured with identical prompts; both were corrected (decode-only, distinct
  prompts).

---

## Open questions

- `output.weight` (963 MB on K3) is still read whole every token; chunking the final
  projection would lower the K3 floor.
- At small caps K3 is near the drive's bandwidth; further gains need fewer bytes
  per token.
- The checkpoint compatibility stamp is a fixed string where a build identity belongs.
- Sampled (non-greedy) resume is approximate: the RNG state does not survive a
  restart.
- A VRAM pool for per-sequence state would move the batch-width bound off RAM.
- Metal needs windowed materialisation to reach its measured 4.1x MoE prefill.

---

## Standing design decisions

- **Storage has no `cancel()`.** The portable macOS backend cannot cancel a read, so
  the interface forbids it: memory with a read outstanding is never freed, and if the
  backend dies with reads in flight their buffers are leaked, loudly, rather than
  reused under DMA.
- **A failed read fails the token.** A partially filled weight is never computed on;
  the run is marked untrustworthy.
- **Classify by the planner's tensor class, never by shape or bits-per-weight.**
  Every size comes from the GGUF tensor table.
- **Private ids for compaction.** The router's ids are never modified, so every other
  consumer of them stays correct without the engine knowing about it.
- **One consumer of the storage backend.** Only the I/O thread submits and harvests.
- **`--cap 16G` means GiB.** The readout prints both units.
