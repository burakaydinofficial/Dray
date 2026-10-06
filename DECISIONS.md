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

**Since then (2026-10-05).** Every CPU run had built a ggml thread pool per graph
node (see Compute); with that fixed the testbed decodes about 2x faster and
Qwen3.6 35B-A3B about 4x, and `--gpu` decode no longer trails the CPU (see GPU).
K3's minimum cap is 5 GiB (see Memory). K3's speed after the thread-pool fix has
not been measured yet: that needs a quiet machine.

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

**The expert frequency cache is evictable and bounded by bytes.** It holds hot
experts in RAM between tokens, gives memory back to the main cache under pressure
(coldest first), and admits by bytes rather than a count, so it stays on at every
batch width. Flash Next at width 2: -46% decode bytes; MiniMax-M3 at 28 GiB and width
16, which used to fail, completes.

**Measured minimum caps** (correct text required; below them the engine refuses at
load and names a cap that works): Qwen3.6 35B-A3B 3 GiB, Qwen3.5 122B-A10B 3 GiB,
DeepSeek V4 Flash 3 GiB, Qwen3.8-27B dense 4 GiB (1-bit: 3 GiB), MiniMax-M3 4 GiB,
GLM-5.2 3 GiB, Qwen3.8 2.4T 5 GiB, Kimi K3 5 GiB. K3's floor is set by
`output.weight` (918 MiB), which must be materialised whole.

**An optional buffer never decides whether a run fits (2026-10-05).** K3 at 5 GiB
was refused -- 891 MiB of cache against `output.weight`'s 918 -- while a 640 MiB
read-ahead ring sat in the same cap. The ring only overlaps reads with compute (the
output is identical without it), so admission now frees it before refusing,
re-measures, and refuses only if the cache still falls short, saying the ring was
already off. K3 at 5 GiB and 4k context: correct text over 32 tokens, 4.6 GB
resident, ~56 GB read per token; 4 GiB stays refused (508 MiB of cache). The 1B-7B
testbed's minimum moves from 400 to 300 MiB. Configurations that ran before are
untouched. Going below 5 GiB on K3 means splitting the final projection, which llama
builds separately in every model's graph code.

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
(llama's allocator vs the streaming path, with the same kernels on both legs; the
repacked kernels resident mode uses by default are compared and reported, never
required), `batchdiff` (within-batch identity), `golden` (every command and lever,
text and counters), `rotategate` (parking mid-generation vs running straight
through), `serve-smoke` (every server route, including crash-resume), `archgate` (short greedy generations on five
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
  memory (11-15%, measured before the pool fix below; see Correctness). The cost vanishes into disk time once a model
  streams more than a few GB per token.
- **Thread oversubscription.** ggml's pool spin-waits; with the engine's own threads
  on top, requesting all 22 threads ran about 40x slower. The pool is clamped to
  (cores - 9); decode defaults to 4 threads, prefill to the clamp.
- **One thread pool, not one per node.** Until 2026-10-05 every CPU run built and
  joined a fresh ggml thread pool for each claimed node: without `--gpu` the engine
  listed the CPU as an offload device, llama created a second CPU backend for it,
  and the scheduler ran every node there -- a backend that never received the
  persistent pools. A CPU-sample trace of 48 testbed tokens counted 73,372 thread
  starts in 22 s. Fixed (an empty device list), same machine, interleaved, text
  identical:

  | model | before | after |
  |---|---|---|
  | 1B-7B testbed, 2 GiB, 4 threads | 2.4-3.0 tok/s | 4.9-5.3 tok/s |
  | Qwen3.6 35B-A3B, 12 GiB, 4 threads | 0.5-1.1 tok/s | 3.7-4.1 tok/s |
  | Qwen3.6 35B-A3B, 12 GiB, 1 thread | 2.5-2.9 tok/s | 2.7-3.0 tok/s |

  The earlier finding that one decode thread beat four on small-active models was
  this bug (more threads meant more threads created per node); with the pool in
  effect four threads win again, and the default stays 4. ggml's default polling
  is kept: workers that sleep instead were faster on a quiet machine and half
  speed on a loaded one.
- **Resident mode.** When the whole model fits the cap, llama.cpp allocates it
  natively. It is correct and much faster than streaming the same model. It measured
  21-46% slower than stock llama.cpp, but that was before the thread-pool fix above,
  which also let it reach llama's repacked kernels for the first time (the same
  duplicate backend had kept every layer in a plain buffer); to be re-measured. If a
  model fits, use llama.cpp. `--force-stream` pins the streaming path, and every gate uses it.
- **For sparse models the disk is free.** Qwen3.5-122B-A10B runs at 2.7 s/token
  reading 872 MiB per token at 12 GiB and at 2.7 s/token reading nothing at 56 GiB.
  Cost tracks active bytes per token, not parameter count; below roughly a gigabyte
  per token, compute is the limit.

---

## GPU (`--gpu`, opt-in at build and run time)

The GPU is used for prefill only: weights stay in host memory, the KV cache stays in
RAM, and decode stays on the CPU (llama turns op offload off for a batch in which
every token is from a different sequence).

- **Streamed `--gpu` computed on garbage until 2026-09-30.** ggml's scheduler copies a
  GPU split's weights straight from host memory *before* the eval callback that
  streams them in runs, so a weight not yet loaded went to the GPU as the poison
  sentinel: wrong text with exit 0 on every streamed model, and a crash on a
  512-expert model. The fork now calls back around each such copy and the engine
  loads the weight first. `scripts/gpugate.ps1` proves streamed `--gpu` bit-identical
  to llama.cpp's own GPU path (resident mode, repacked kernels and Vulkan fusion off on
  both legs; not the CPU, whose q8 activations flip near-ties). Every earlier GPU
  prefill figure is withdrawn (see Corrections).
- **Copies read only the routed experts.** The scheduler copies only the experts a
  split routes to; the copy now reads only those (whole tensors when every routed
  tensor fits the cache). GPU prefill reads the same bytes as CPU prefill: Qwen3.8
  Flash Next 547 -> 325.6 GB for a 2.3k-token prompt (CPU 325.5).
- **Copies land in pinned memory and read ahead.** Unpinned source memory costs a
  staging memcpy and a device synchronise per copy. Routed experts now land in host
  memory from the GPU device's own host buffer type, one slot per expert kind, and a
  layer's other two expert tensors (same routing) are read ahead while one is copied.
  Qwen3.6 35B-A3B at 12 GiB, 2.3k-token prompt, same bytes and text: 123/94 s ->
  34/34 s. The slots (562 MiB there) are charged to the cap.
- **First corrected figure.** Same model and cap, 2.3k-token prompt, default settings
  (the automatic VRAM limit settled on 1024-token chunks), High performance plan:
  generation starts at 16.8 s with `--gpu` against 133.9 s on CPU, both including
  about 5 s of load -- roughly 10 s of prefill against 129 s, reading 66.6 GB against
  105.6. Copies then run near the laptop's PCIe rate, so the remaining cost is bytes
  over PCIe, which fall with chunk size. Qwen3.8-27B dense at 16 GiB, same prompt:
  32-33 s against 707 s (about 25x on prefill), but 71.1 GB read against 54.5 -- a
  dense model's every weight is copied per chunk, while the CPU path keeps part of
  it cached.
- **A hard VRAM limit, like the cap.** `gpu.vram_cap` (system settings, a size) or
  `--vram-cap`; unset, a quarter of the card (half on a card of at most 2 GiB). The
  prefill chunk is halved until the GPU context fits, or the run is refused with the
  VRAM it needs.
- **Context-sized work stays with a KV cache kept in RAM** when VRAM requires it:
  attention and sparse-attention indexer scores against every cached key are sized by
  the context, reserved for a full one. With the fork's opt-in scheduler rule those
  ops run on the CPU beside the cache; it costs about 25% at 8k context, so it is used
  only when it is what makes a chunk fit (Flash Next, 2 x 600k context in 1.94 GiB:
  chunk 32 -> 512).
- **It raises the minimum cap**, because the pinned copy slots are inside it.
- **Vulkan only wakes when asked.** Merely initialising the Vulkan backend costs CPU
  decode 25-30% (Flash Next: 1.83 vs 2.23 s/token, same bytes; not power, buffers or
  I/O -- the same work takes more CPU time). Vulkan is a build-time opt-in, and a
  Vulkan build without `--gpu` now disables it for its own process before any ggml
  call.
- **`--gpu` decode no longer goes to the GPU and back.** Until 2026-10-05 a one-token
  per-head norm in every gated-deltanet layer had 32 rows, which Vulkan's offload
  rule reads as a batch; with host weights it went to the GPU and back in every such
  layer of every token (Qwen3.6-35B-A3B: 61 graph splits per decode token, 14.5% of
  the main thread waiting on fences). The fork now turns op offload off for decode
  batches. 35B, 12 GiB: `--gpu` decode 2.9-3.0 -> 3.7-4.2 tok/s, against 3.6-3.9
  without `--gpu`; text identical; a 1852-token prefill still offloads.

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
  rotating cohorts, parking state to a directory the user chooses; it is the engine's
  only sustained-write feature and reports its write bytes. A span shorter than the
  generation parks sequences mid-generation and resumes them later. That was refused
  while identical reruns diverged; restore has since measured bit-exact, and
  `scripts/rotategate.ps1` now requires parked runs to match running straight through,
  token for token: six prompts at width 2, parked every 4 tokens, again, and every 5,
  on the 1B-7B testbed (attention, 0.035 GB parked per run) and Qwen3.8-27B
  (recurrent, 2.84 GB). A restore that flips the sign of ~1000 cached values runs
  without error and changes one sequence at its 15th token; the gate fails it.

**Serving.** `serve` runs a continuous-batching scheduler: `serve.max_parallel` slots,
each with its own KV funded at load inside the cap, one decode step carrying every
generating slot plus a bounded share of prompt tokens. With no slot decoding, a prompt
goes in full-size steps -- each step re-reads the experts its tokens route to, so a
4.4k-token prompt in 256-token steps cost Flash Next 922 s on CPU. Conversations are
reused by token prefix regardless of connection: kept in their slot (attention KV cut
back to the shared prefix, recurrent state restored from checkpoints taken just before
the prompt end), or parked whole in a pool outside the slots with a token budget,
idle expiry and LRU. A restored conversation answers exactly what it would have
without leaving its slot. Tool calls and reasoning go through the model's own chat
template (llama.cpp's chat library).

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

- **"`--gpu` prefill is 12-72x faster; a 6,594-token prompt takes 22 s"** --
  withdrawn: every streamed `--gpu` run computed on garbage until 2026-09-30 (see
  GPU). Correct GPU prefill figures are being re-measured.
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
- GPU prefill still runs one split per host-resident weight, each synchronised;
  grouping weights per split under the VRAM limit is the next lever.

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
