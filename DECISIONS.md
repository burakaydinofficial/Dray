# Dead expert regions are reused, not freed and reallocated (2026-09-27)

After direct reads, the K3 release callbacks (Phase C, where a layer's expert
regions are allocated) were the next visible cost: at 9 GiB a 16-token run
evicts ~6,300 regions (~259 GB), each a free (decommit) and a fresh commit.
ResidencyCache::take_region now hands a dead routed-expert region of exactly
the needed size to the new owner instead: removed from the cache and pointed at
the sentinel like any eviction, its ledger charge carried over unchanged, its
pages never decommitted. It is used ONLY where a fresh allocation would have
had to evict -- the first version also took regions while there was free room,
throwing away reusable cached experts for nothing; test_pressure caught it (a
smaller cache at its rebudget, then a cap breach). Never pinned, protected,
unconditional, differently sized or wrongly aligned regions
(test_region_reuse, mutation-checked).

  K3 9 GiB, 16 tokens, High performance, neighbour agent active, interleaved:
  direct            8.9  8.8  9.0 s/tok   917.03 GB
  direct + reuse    8.7  8.6  8.7 s/tok   916.6-917.03 GB
  4,432 regions reused in place per run; text identical in all runs.

A small gain (~2.5%), but outside the run-to-run range. Confirmed on GLM 5.3 Flash
(8 GiB, 32 tokens, same conditions, built on the GLM-capable fork base that is
not pinned yet), 3 interleaved runs each, mean rate incl. prefill:
  staged 2.8 2.8 2.8 | direct 1.9 2.2 2.2 | direct + reuse 1.9 1.8 1.8 s/tok
  routed wait 28-29 s -> 8.4-9.0 s; 100.256 GB and identical text in all nine. Also new under
DRAY_IO_STATS: time inside Phase C, region room/allocation time, evictions
and the time spent freeing them.

---

# Expert reads land in place, without a bounce buffer (2026-09-27)

The I/O thread copied 220 GB of staged expert reads into their regions over one
K3 run (9 GiB, 16 tokens) -- ~13.8 GB per token of pure memory traffic. The
staging existed because unbuffered reads need aligned offsets, lengths and buffers, while an expert's bytes must land at the
region's stride.

But every expert stride on K3 and GLM 5.3 Flash is a multiple of 4096 (K3
6,451,200 = 1575 x 4096), so all experts of a tensor share one offset
remainder h. A region now places its data at mem + h
(ExpertCompactor::alloc_region), and IoScheduler::submit_exact reads each
expert's aligned middle straight into its slot; only the two sub-4 KiB edges go
through staging. Unaligned strides, unaligned destinations and slices without
an aligned middle fall back to one staged read, exactly as before.

  K3, 9 GiB, 16 tokens, text identical (deterministic figures only):
  staged   932 GB read   220 GB copied through staging
  direct   916 GB read   2.9 GB copied through staging (edges only)

TIMINGS WITHDRAWN (same day). This entry first quoted decode 17.5/17.4 ->
13.7/15.0 s/tok, routed wait 124.6 -> 34.0 s and memcpy 74 -> 0.8 s. Those
runs (11:21-11:45) overlapped other agents working on the same machine
(owner, 12:35), which breaks the exclusive-drive rule invisibly; they are not
evidence. The bytes and the copy volume above are deterministic and stand; the
speed-up was re-measured the same afternoon (below). The commit message of
56d64df carries the withdrawn figures.

  RE-MEASURED, K3 9 GiB, 16 tokens, High performance plan, the neighbour agent
  still active (small bursts) -- so three interleaved runs each, ranges shown:
  staged (abebf8f)   10.3  11.4  10.8 s/tok   919-934 GB (varies run to run)
  direct (56d64df)    8.9   8.8   9.0 s/tok   917.03 GB
  -18% decode; text identical in all runs. The staged build's bytes vary
  because in-flight staging holds cache room for a timing-dependent time;
  direct reads hold almost none, so their bytes are near-deterministic.

Fewer bytes too: in-flight staging no longer holds ~6.5 MB per expert read, so
the cache keeps that room. Found on the way, both fixed: on_get_tensor and
self_check read a resident tensor from mem instead of mem + head, so any whole
tensor placed at a widened span read back shifted (self_check would have
reported false mismatches; get_tensor served wrong bytes).

---

# The ring skipped 3-D weights that are not experts: K3 -14% (2026-09-26)

After the scheduler fix, per decode token on K3 (9 GiB, 16 tokens minus a
2-token run, High performance plan): 2.74 s waiting on routed experts, 2.72
s on SYNCHRONOUS reads of unconditional tensors, 0.32 s on the ring, ~5.3 s
compute. The synchronous reads were ~300 tensors/token, 1.24 GB: output.weight
(963 MB, larger than half the ring by design), and ~190 tensors the ring
never streamed -- K3's MLA attn_k_b/attn_v_b and KDA ssm_conv1d_{q,k,v}.

Cause: the ring built its stream list with a SHAPE test, ne[2] > 1 meaning
"expert tensor, compaction's job". Those weights are 3-D but unconditional,
so they fell outside both the ring and compaction. The planner's class
(Source::sliceable covers routed experts and the row-sliced embedding) is
now the only filter -- class, never shape.

  K3, 9 GiB, 16 tokens, High performance plan, ABBA, text identical:
  shape test   12.4  12.2 s/tok   933.8 GB   stream list 1,460   sync reads 7,193
  class        10.6  10.6 s/tok   932.8 GB   stream list 1,715   sync reads 2,809

Remaining synchronous waits: the prefill pass (runs before the ring has a
list) and output.weight once per token.

---

# 3.5 GB of the cap was ggml scheduler metadata nobody touched (2026-09-26)

Corrects the entry below: the K3 bytes regression since 08-14 was NOT
honest accounting.

Found by asking where K3's memory goes at 12.25 GiB. The ledger showed 4.19
GB "unreserved" (commit the engine did not allocate), with RSS 1 GB under
the ledger total. Commit sampled after each load step: creating the llama
context committed 4.78 GB and touched 0.68. Real buffers (KV 54 MiB,
recurrent state 443 MiB, compute 574 MiB) explain ~1.1 GB. Halving the
graph node budget freed 1.83 GB -- ~22 KB per node.

The cause: ggml_backend_sched_new sized its split-copy context for one
split per node with 30 inputs each (+ backend-id arrays at 61x the graph),
~20 KB per node, malloc'd up front. A CPU graph has ONE split. At K3's node
budget of 164,672 that is ~3.5 GB, never touched -- Windows charges it as
commit, and the cap binds on the worse of commit and RSS, so it came out
of the cache. (Linux/macOS never see it: untouched memory is free there,
and our commit figure is unknown on them.) K3 got that node budget from
fork commit 16cb54a (08-20), which "fixed" K3's dead budget branch: 20,584
nodes -> 164,672. The bisect shows the step exactly there (9 GiB cache
4.36 -> 1.18 GB between points 144 and 145). That commit's comment called
the headroom "cheap in metadata (~1 MB per 2048 nodes)"; it was ~40 MB.

Fix, in the fork (6a80411ef): the split context is a chain of blocks sized
on demand (first block = previous graph's high-water mark), and the id
arrays grow with the graph copy. The node budget itself is unchanged.

  K3, 9 GiB, 16 tokens, High performance plan, ABBA, text identical:
  before  13.0  13.2 s/tok   989.07 GB   cache budget 1.17 GB
  after   11.9  12.4 s/tok   933.76 GB   cache budget 4.66 GB   RSS 8.56 GB

933.8 GB is below the 08-14 headline build at the same cap (938.1): the
bytes regression is gone. At 12.25 GiB the budget rose 4.22 -> 7.72 GB.
Gates: golden text identical (26 cases; 9 counter files read equal or
fewer bytes), fast gates, server smoke, archgate 5/5; --gpu on the testbed
and the 35B (844 splits) byte-identical to the unfixed build.

---

# The I/O thread lands, and the K3 slowdown was two things, neither the streamer (2026-09-26)

**K3 since the 08-14 headline.** A clean A/B (old/new/new/old, idle drive, 9 GiB,
16 tokens) gave 7596a8e 17.2/17.3 s/tok against today's 20.6/19.7, and 938 vs
989 GB. A bisect over the 250 commits between, each point probed first and
then run at a cap giving ~4.3 GB of cache (a given cap buys each commit a
different budget), plus drift controls, separated two causes:

- BYTES: first written as "honest accounting" -- WRONG, see the entry above.
  The cap did shrink the cache (9 GiB left 1.13 GB where 08-14 had 4.36), and
  at an equal budget the bytes matched (941.6 vs 938.1 GB), but what took the
  budget was ~3.5 GB of never-touched scheduler metadata, now fixed.
- TIME (~11%): the Vulkan backend COMPILED IN, never used. Every build before
  8583e9c (which made Vulkan an option) ran 18.1-18.8; every Vulkan-carrying
  build 19.6-20.9. Same code, CPU-only vs Vulkan-carrying, ABBA at 12.25 GiB:
  18.3/18.5 vs 20.6/20.0, bytes and text identical. ggml registers the Vulkan
  backend eagerly (ggml_vk_instance_init), waking the GPU driver before --gpu
  is checked; removing the duplicate CPU backend llama.cpp made did NOT help.
  (Laptop with a discrete GPU; the cost may differ elsewhere.) Fixed in d51cb7e:
  Vulkan is a build-time opt-in, and --gpu refuses on a CPU-only binary.

CPU-only today (18.4) matches 08-19 code at the same point in the session
(18.4): nothing else in those commits slowed K3. The published head-to-head
(8 GiB, 16.1 s/token) is not reproducible on current code; correcting it is
the owner's call.

**The I/O thread (8550714, 41ac5bb, a97a356).** The fix the 08-24 attempt
failed at, rebuilt on two rules: ONE consumer of the backend (only the I/O
thread calls submit/poll -- that attempt added a second harvester inside the
Windows backend), and the thread changes WHEN, never WHAT (staging, the
ledger and every decision stay on the compute thread). Proven before any
threading with a scripted fake backend and mutation tests; ThreadSanitizer
on the Linux VM reports nothing over five runs, and flags an injected
unlocked insert at once. Golden identical with the thread on (twice) and
off; archgate 5/5 both ways.

K3, 12.25 GiB (4.2 GB cache), 16 tokens, CPU-only build, same binary, ABBA,
Windows "Balanced" power plan:

  thread off   17.9  18.8 s/tok   940.82  940.91 GB
  thread on    16.4  16.6 s/tok   940.87  940.83 GB     text identical

~10% faster decode, same bytes. Read latency 18.6 -> 11.5 ms (collected
promptly); the 55-61 s of staging copies moved off the compute thread.
What remains is not the device: with the thread, the long poll gaps are the
I/O thread IDLE -- nothing queued. CORRECTION (same day): not because the
ring is arena-bound, as first written. Step 2 bounded read-ahead by chunks
outstanding until CONSUMED, and on K3 the ring now stops on that bound on
every call (stops d=544784, full=0; the old code stopped on a full arena).
That cap throttles look-ahead far below the arena; the other part is that
the next layer's experts are not yet known. Both are decision-side.
DRAY_IO_THREAD=0 restores the old path.

Re-measured the same day in the Windows "High performance" power plan (the
row above was Balanced; timings are only comparable within one plan):

  thread off   13.6  13.0 s/tok   940.82  940.84 GB
  thread on    12.4  12.5 s/tok   940.82  940.89 GB     text identical

~6% faster in that plan: with the CPU clocked higher, less of each token was
waiting on uncollected reads to begin with. The plan itself moves K3 by
~25%, which is why every timing here now names it.

---

# The audit-fix arc: 21 agents, ~110 findings, 13 batches (2026-08-24)

The owner asked for a swarm review with NO self-filtering -- false positives
explicitly wanted. Twenty finder agents on narrow slices plus a synthesis pass
produced 461 raw findings, merged to ~110. Thirteen fix batches followed, each
gated by the full suite before commit (5872f00 through 505d64b). What was
fixed, compressed:

SERVED WRONG ANSWERS AS GOOD ONES. The non-streaming endpoint returned 200 for
aborted/tainted runs (the streaming path had the check; the second call site
did not). engine_generate checked llama_decode's return before asking the
streamer whether it aborted, so r.aborted was dead on the path that fires.
materialise()'s contract said "returns false on failure, callers must not
compute" while two paths returned true and both callers ignored the result.
Ring retention promoted segments whose reads FAILED into the cache (complete
meant submitted, not landed). get_tensor zero-filled on a miss and called it
success. A truncated shard warned and proceeded.

CORRUPTION AND CRASH PATHS. The IOCP op pool had no double-reap guard -- the
latent defect that corrupted reads when a harvester thread was attempted, and
present without one. Five hot-path sites conflated "read failed" with "backend
died with DMA outstanding" and freed memory the kernel could still write.
rebudget_against_rss demoted NO-SOURCE entries and clamped static_used to zero
while the real over-claim survived. gguf_get_arr_data was called before the
type check (process abort on a hostile header); block_count and blk.<N> sized
vectors unbounded from untrusted metadata; layer_of wrapped its accumulator;
duplicate names across shards double-counted then GGML_ABORTed.

GATES THAT CERTIFIED NOTHING. archgate ran its control model resident (no
--force-stream), exited 0 having checked zero architectures, deleted absent
models' expectations on -Record, passed unrecorded models, collapsed newlines,
compared culture-sensitively. No gate checked its binary existed (stale
$LASTEXITCODE passes), none cleared ambient DRAY_* (an inherited
NO_COMPACT=1 made difftest compare identical-to-identical), and nothing
anywhere passed --no-tests=error, so a vanished test suite was green. The
matrix harness recorded crashes as REFUSED, never looked for taint, and could
not distinguish resident cells from streaming ones.

HONESTY AND BOUNDS. classify() missed fused expert tensors (real in the fork:
qwen3next, qwen35moe, --fuse-gate-up-exps). verify's second half could not
fail the command (added that day, never wired to the verdict -- mine).
PrefillActivation was priced but never reserved, and every floor reserve()
result was discarded. Stops did not survive the resume boundary, falsifying
the README's greedy-identical guarantee; run --stop was parsed and dropped;
--kv typos priced f16; CLI numbers went through atoi ("--ctx 32k" became 32);
serve cast unclamped doubles (UB at 1e20). Shutdown-cancelled jobs reported
terminal "cancelled" then silently re-ran at boot; the worker could not be
stopped by shutdown; the job queue was unbounded; the API key compare leaked
timing.

DELIBERATELY NOT FIXED, with reasons: the checkpoint compat stamp hardcodes
"dray-jobs-v1" where a build id belongs (needs an owner decision about what
invalidates checkpoints); sampled-resume RNG divergence (already disclosed in
the README as approximate); deeper HTTP hardening (loopback-only, unpublished);
repack.json validation (the tool is experimental and off); the ~60 low and
cosmetic findings in the report's tail. The full synthesis lives in the session
task output; the raw findings in the workflow journal.

THE PATTERN, stated once for whoever reads this next: almost every serious
defect was a rule the author already knew, implemented correctly once, and
missed at the second site -- the second call path, the second gate, the second
branch. The audit's value was not discovering new principles; it was finding
every place an existing principle had one uncovered instance.

---

# The batch table is complete, 35B to 2.8T (2026-08-23)

Every model in the collection now has a batch figure. Width 32, varied
prompts, 4k ctx, --kv q4, decode only:

  model                     cap      GB/token   vs single
  Qwen3.6 35B-A3B           8 GiB    0.213      cheapest streaming
  Qwen3.6 35B-A3B           28 GiB   0 BYTES    fully resident
  Qwen3.5 122B-A10B         28 GiB   0.406      2.15x
  DeepSeek V4 Flash 284B    28 GiB   0.892      1.75x
  MiniMax-M3 429B           28 GiB   1.62       2.50x
  GLM-5.2 744B (width 44)   28 GiB   2.40       2.55x
  Kimi K3 2.8T              28 GiB   5.98       5.45x

Two things this table says that the individual runs did not. First, the
MULTIPLIER SHRINKS as models get cheaper -- K3 gains 5.45x from batching, the
122B only 2.15x -- because there is less left to share once a token already
costs under a gigabyte. The absolute column is what a user feels; the
multiplier column is what the architecture allows. Second, at 28 GiB the 35B
is entirely resident and decode reads NOTHING from disk: 32 sequences served
with zero streaming, which is the regime where this engine contributes only
its cap and gets out of the way.

---

# Width scaling on the 122B, and the cost of my own test prompts (2026-08-23)

Qwen3.5-122B-A10B batches far wider than the older flagships because its
per-sequence state is about 176 MiB (27 MiB of KV at 4k plus 149 MiB
recurrent) against the 0.7-1.0 GB that stopped K3 at width 44. All runs 28
GiB, 4k ctx, --kv q4, decode only, 12 tokens per sequence:

  width   templated prompts   varied prompts
  32      0.266 GB/token      0.406 GB/token
  64      0.188               unmeasured
  96      0.139               unmeasured

Width 96 clears without refusal: 96 sequences at once from a 122B model on 28
GiB. Genuine width scaling is the 0.266 -> 0.139 column, a 1.9x improvement
from 32 to 96.

AND THE TRAP I WALKED INTO TWICE. First attempt at width 64 used a prompt
file padded by repeating a 44-line list, so 20 of 64 were literal duplicates
-- the same correlated-ceiling error that once made K3 look like 20.5x.
Regenerated 96 unique prompts and re-ran. But unique STRINGS are not diverse
ROUTING: those 96 came from 6 templates over 16 subjects, and the control run
proves it matters -- same width 32, varied prompts 0.406 vs templated 0.266,
a 34% premium for prompt homogeneity alone.

So the published table keeps the VARIED figure (0.406 at width 32) and the
wider entries are recorded here as templated, with the varied equivalents
left UNMEASURED rather than scaled by the observed 1.53x ratio. A projection
is not a measurement (Invariant 8).

---

# The sparse models batch best too: 406 MB/token (2026-08-23)

The batch table had no entry for the models that turned out to be the most
interesting ones. Qwen3.5-122B-A10B, 28 GiB, width 32, distinct prompts,
--kv q4, decode only: 0.406 GB/token-aggregate against 0.872 single-stream,
a 2.15x multiplier and FOUR TIMES cheaper than the best figure previously in
that table (MiniMax-M3 at 1.62).

The multiplier is modest because the single-stream cost is already low --
there is less to share when a token only touches a gigabyte. The ABSOLUTE
number is what matters: 406 MB per token for a 122B model serving 32
sequences at once. Width should also go much further than the older
flagships allowed, because per-sequence state here is about 176 MiB (27 MiB
of KV at 4k plus 149 MiB recurrent) rather than the 0.7-1.0 GB that made
width 32 eat two thirds of a 28 GiB budget on K3. A width-64 run is
measuring that now.

---

# Batch figures are a BEST case, not a steady state (2026-08-23)

The short-run audit reached the batch table expecting the same answer it got
for single-stream -- that the published figures were pessimistic. The opposite
is true, and the table needed a caveat rather than a victory lap.

GLM-5.2, 28 GiB, width 44, --kv q4, distinct prompts:
   6 tokens/sequence: 2.400 GB/token-aggregate  (the published figure)
  18 tokens/sequence: 2.640 GB/token-aggregate  (+10%)

WHY, and it is structural rather than a caching artefact: batching pays off
because sequences share expert selections within a step. As generation
proceeds the sequences DIVERGE -- different topics, different routing -- so
the union per step widens and the shared fraction decays. Cache ramp effects
push the other way and lose.

So batch figures behave opposite to single-stream figures under longer runs,
and the README now says so: the table is the best case, budget ~10% above it.
This is the third time this week that a measurement I expected to confirm
something corrected it instead.

---

# ATTEMPTED THE FIX, IT FAILED, REVERTED (2026-08-24)

Owner: "so there is nothing you can do here remaining?" There was, and my
reason for not doing it was drawn in the wrong place. I said a harvester thread
needs locking across 42 im.pending sites -- true of a STREAMER-level thread,
false of a BACKEND-level one. The IO layer has a closed interface
(submit/poll/in_flight) and already maintains a ready queue that poll() drains,
so a thread that pulls IOCP packets and parks them there touches no engine
state. About 30 lines in storage_win.cpp, behind DRAY_IO_THREAD=1.

Built it. First result: correct on the 1B testbed, STALLED on K3. Cause found
by reading: poll() drains the ready queue ONCE at entry, then waits on the
kernel -- so packets the harvester parks DURING that wait are never seen, the
loop spins until the port empties, and it returns zero completions. Fixed by
re-draining each pass.

Second result: still wrong. Exit 1 with MATERIALISE FAIL on Qwen3.6-35B-A3B.
The harvester and the reaping paths race somewhere I did not localise --
reap_locked and the close/cancel path both consume packets, and two consumers
of one completion port need more care than a mutex around the ready list.

REVERTED. A flag that corrupts reads when enabled is worse than no flag, and
the gates prove text identity on short single-threaded runs, which is not
evidence of race freedom. The diagnosis stands and is unaffected; what is now
also known is that the completion side alone is NOT a 30-line change, and that
the first two attempts fail in ways the existing gates do not catch.

For whoever builds it properly: the two consumers must be reconciled at the
port, not at the ready list. And the submission half -- keeping reads flowing
DURING compute -- still needs streamer state and is the larger half of the win.

---

# ROOT CAUSE: the I/O path only runs between compute nodes, and that is 90% dark

Found, and it reverses a conclusion I published twice. I "ruled out" harvesting
delay using a MEAN poll gap of 2 ms. The mean was worthless. The distribution:

  gaps <100us  30,678
  gaps <1ms    11,086
  gaps <5ms     9,392
  gaps <20ms    2,223
  gaps longer   2,043     max 319,793 us
  TOTAL TIME IN GAPS OVER 5 ms: 108,534 ms of a ~121 s run

Most polls cluster tightly, and the long tail owns NINETY PERCENT OF THE WALL
CLOCK. For 108 of 121 seconds nothing polls the completion queue.

WHY: pump() is called only from materialise() and release(), which run between
compute nodes. The thread that drives I/O is the thread that computes. While a
matmul runs -- or while the completion path memcpys 51 GB, which is the same
thread again -- no completion is harvested and no read is submitted.

THAT EXPLAINS EVERY CONFUSING NUMBER. in_flight() counts submitted-but-
unharvested, so it reads 44 while the device may have finished long ago:
not a deep queue, a pile of unclaimed completions. Per-request "latency" of
26.7 ms is time-until-noticed, not service time. Little's law failed to close
because both inputs measured our absence rather than the drive.

And the drive is clean: calibrated on K3's OWN fourteen shards it serves 2 MiB
random reads at 6.03 GB/s at QD16. We get 1.8.

WHICH SETTLES THE OWNER'S PROPOSAL. A separate loader with its own thread is
exactly the right shape, and the reason is now precise rather than intuitive:
not that batching is smarter, but that I/O progress currently cannot happen
during compute at all. Any design that gives submission and completion their
own thread recovers this; the ring geometry and priority ordering are
refinements on top.

CHECKED BEFORE BELIEVING IT, because the gap timer measures the interval
between pump() ENTRIES and pump(im,1) BLOCKS inside poll -- a 320 ms "gap"
could have been the thread sitting in the completion wait, which would be
legitimate. Instrumented the time spent inside poll: 17,290 ms. So of 108,534
ms in long gaps, only 17 s is waiting on the device and ROUGHLY 91 SECONDS OF A
121 SECOND RUN IS TRUE ABSENCE. The conclusion survives its own audit.

COST OF MY ERROR, recorded because it repeated: I dismissed this with an
average, twice, when the distribution was one counter away.

---

# The drive is exonerated ON ITS OWN FILES: 6.0 GB/s where we get 1.8

Owner proposed a smart loader: a deep ring of large blocks, filled by a
separate component that reads sequentially or in parallel-sequential order
from a priority list. Before endorsing a rewrite, the premise needed testing --
my calibration ceiling came from a 21 GB model, and random reads across THAT
span may sit in the drive's mapping cache in a way random reads across 594 GB
do not.

Calibrated K3 itself. Gather, its own fourteen shards:
  2 MiB  QD1 4.00   QD4 5.88   QD16 6.03   QD32 6.10   QD64 6.11 GB/s
  512 KiB QD16 6.11 (chosen operating point)

So the span is NOT the problem and scattered access is NOT the problem. This
drive serves 2 MiB random reads across 594 GB at 6 GB/s. The engine, reading
2.2 MiB at a time-weighted depth of 44, achieves 1.8. ROUGHLY 3.3x IS LOST
INSIDE OUR SOFTWARE PATH -- not the device, not the access pattern, not the
queue depth, all three now measured and cleared.

WHAT THAT MEANS FOR THE PROPOSAL. It is well aimed. Its strongest single idea
is recognising that below the knee, decode is a SCAN executed as a gather:
h_bytes is 4% at these caps, so the model is re-read almost entirely every
token, and the project already knows scans want sequential I/O (it says so
about prefill) while decode does not exploit it. Decoupling the loader also
removes the measured 10.6 s of memcpy from the thread that polls and submits,
and allows coalescing adjacent slices.

TWO CONSTRAINTS THE DESIGN MUST RESPECT. 256 MB blocks would park a routed
expert read behind bulk transfer it does not need, so the loader needs
preemption or a separate small-request lane -- close to what "priority order"
already implies. And the horizon still binds: routed experts cannot be fetched
before the router picks them, so lookahead only applies to the unconditional
set, which at these caps is most of the bytes anyway.

Also done here: K3 now has its own calibration file, so its readout stops
borrowing a ceiling from another model.

---

# AUDIT: I had not tested properly, and the reconciliation was wrong

Owner asked how sure I was and whether every hypothesis was properly tested.
Honest answer was no: two of six were controlled measurements, two were
moderate, three were weak. Redone:

RING CAPACITY -- was tested by changing the CAP from 9 to 40 GiB, which moves
the cache budget, pinning, slot pool and ring size together. Retested with the
single variable the code already exposes (DRAY_RING_MB), same cap: 256 MB
gives full=102,736 and 1.70 GB/s; 2,048 MB gives full=130,143 and 1.81 GB/s.
Eight times the memory, 6% throughput (noise), and MORE full stops. Conclusion
unchanged but now actually established: the ring is consumer-limited, not
capacity-limited, and a bigger ring simply fills and stays full.

QUEUE DEPTH -- was sampled at submit time, which records in_flight() at the one
moment we are pushing work in. I predicted that biased HIGH. It biases LOW:
integrating depth over elapsed time gives 44, not 24, because the queue keeps
filling after each submit and stays deep through compute, which is exactly when
nothing samples it.

WHICH BREAKS MY HEADLINE. I wrote that Little's law "closes" at depth 23 and
26.5 ms giving 1.9 GB/s against an observed 2.0-2.75. With the correct depth it
predicts 44 / 26.7 ms x 2.2 MiB = 3.6 GB/s against 1.8 observed. The model does
NOT close; it now over-predicts by a factor of two. The earlier agreement was
arithmetic on a wrong number landing near the right answer by luck.

So the honest state is worse than yesterday's entry claimed: depth is healthy
(44 of a 64 cap), latency is 26.7 ms, and those two facts together should give
roughly twice the throughput actually observed. Something is consuming
in-flight slots without delivering bytes, or the mean request size is not what
the counter reports. Both are checkable, neither is checked.

---

# SUPERSEDED: 26.5 ms per read, and Little's law closes

Owner, fairly: "I do not think you are capable of fixing it." My record on this
problem today was one confident claim retracted and three more killed. But I
had also written "needs a profiler", and that was giving up early -- the
obvious measurement I never took was PER-REQUEST SERVICE TIME, which needs no
tool I lack.

Taken: K3 at a 9 GiB cap, 22,697 reads averaging 2,203 KiB, MEAN LATENCY 26,555
MICROSECONDS. Distribution: 245 reads under 200 us, 274 under 500, one under a
millisecond, and 22,177 SLOWER THAN 2 ms. A 2.2 MiB read on a drive that
calibrates at 6.8 GB/s should take about 320 us.

AND IT RECONCILES. Little's law: depth 23 divided by 26.5 ms, times 2.2 MiB,
is 1.9 GB/s. The observed rate is 2.0-2.75. So the engine is LATENCY-BOUND, not
bandwidth-bound, and every earlier hypothesis was looking at the wrong axis.
To reach 6.8 GB/s at this latency the queue would need to be about 82 deep; the
backend caps at 64 and we run at 23.

THE CAVEAT THAT DECIDES THE FIX, and I am not skipping it this time: that timer
measures submit until WE OBSERVE completion, and completions are only harvested
inside pump(), which is called from eight places, all of them on the node path.
A read that finished in 300 us but is not polled for 26 ms records 26 ms. So
this number is device time PLUS harvesting delay, and the two have very
different fixes -- a slow device would need different I/O, while slow
harvesting needs a completion thread or more frequent polling, which is cheap.

Separated, by instrumenting the poll cadence: 59,670 polls at a mean gap of
2,030 us, most returning nothing. We look every 2 ms and the read still is not
ready, so the 26.5 ms is NOT harvesting delay. And calibrate reaches 6.8 GB/s
THROUGH THE SAME BACKEND, which exonerates the I/O layer and implicates how the
streamer uses it.

ONE MEASURED CONTRIBUTOR, and it is real but not sufficient: the completion
path memcpys every read from its staging buffer to the destination, INSIDE
pump(), on the same thread that polls and submits. K3 at a 9 GiB cap spends
10,814 ms moving 51,165 MB that way, out of a 110 s run. Ten percent of wall
goes to copying at 4.7 GB/s, and during every one of those milliseconds nothing
is submitted, so the queue drains and depth falls -- which is the shape of the
whole problem.

The project already knows this pattern is bad and removed it elsewhere: the
whole-tensor path is documented as "NO staging buffer and NO copy" for exactly
this reason. The compaction slice path still stages, because its slices must
land contiguously at an exact stride and arbitrary file offsets cannot be made
to line up with that.

HONEST LIMIT: 10% of wall does not explain 2.75 GB/s against 6.8. It is a
contributor with a measured size, not the answer. STAGING ALLOCATION ALSO KILLED: 96 ms across 22,697 reads, four microseconds
each, 0.08% of the run. It is pooled. Invariant 1 technically forbids it on the
decode path and it should still be moved to a freelist for principle, but it
costs nothing measurable and is not the answer.

WHERE THIS LINE ENDS, after six hypotheses in one day. Ruled out: shallow
batching (depth is 23), per-layer drain as a defect (inherent to routing), ring
starvation (cap-independent), block size (2.2 MiB mean), harvesting delay
(polls every 2 ms), staging allocation (96 ms). Sized: staging memcpy on the
poll thread, 10.6 s of 126 s. Unexplained: the rest of the gap between 2.75 and
6.8 GB/s.

What the numbers say structurally, for whoever picks this up: at depth 23 and
26.5 ms service time, Little's law gives exactly the throughput observed. The
ceiling needs either depth near 82 at this latency, or the latency down to 7 ms
at this depth. Depth is bounded because expert reads cannot start before the
router picks them and the ring is already full of unconsumed prefetch, so the
lever is latency, and latency is now the thing to profile per request inside
the backend rather than at its edges. That is a real tool problem, not another
afternoon of counters.

---

# RETRACTED: queue depth is 22, not 3. The slow reads are still unexplained

The entry below claimed the effective queue depth is 3 and called it the
largest defect in the project. THAT DIAGNOSIS IS WRONG and this correction
sits above it because a wrong diagnosis in this notebook is worse than none.

I instrumented read_batch and counted slices per call: mean 2.9. But
submit_layer_siblings issues the OTHER TWO TENSORS of an expert trio outside
read_batch, unwaited, by design -- the comment in compact_experts says so
plainly ("sibling regions first (unwaited), then handle this tensor -- whose
own reads overlap the siblings'"). Counting slices per read_batch call
therefore measures one third of one tensor's reads and misses the rest.

Measured where the drive actually sees it, by sampling io->in_flight() at
every submit: K3 runs at QDEPTH MEAN 22, MAX 48, against a backend cap of 64.
The queue is deep. The design works.

WHAT REMAINS TRUE AND UNEXPLAINED: K3 still realises 2.75 GB/s while the drive
calibrates at 6.6-6.8 GB/s at QD16 with 1-2 MiB blocks, and the disk counter
shows it continuously busy. Deep queue, large blocks, busy drive, one third of
the rate. Candidates not yet tested: reads scatter across FOURTEEN shard files
where calibrate sweeps one; every read lands in a staging buffer and is then
copied into the slab, so the memory traffic is doubled; completion handling
cost per request. The gap is real -- 46.22 GB at 6.8 GB/s would be 6.8 s of a
22.7 s token -- but its cause is now genuinely unknown rather than wrongly
assigned.

REFINED, with the queue watched properly: K3 runs at mean depth 23, MAX 50,
MIN 1, and 1,995 separate submits found FEWER THAN FOUR reads in flight. The
queue does not sit deep; it OSCILLATES -- a burst at each layer, then a drain
while that layer computes. A mean of 23 hides that a large share of submits
happen against a nearly empty queue, and the disk counter cannot see the gaps
because they are shorter than its 400 ms sampling.

AND THE DRAIN MAY BE FUNDAMENTAL, which is the part that matters. This engine
cannot prefetch layer N+1's experts until layer N's router has chosen them --
the one-mixer-block horizon this notebook has documented from the start. So a
queue drain at every layer boundary is not a bug to fix, it is the dependency
chain of MoE decode. What CAN fill those gaps is the unconditional stream,
which is knowable arbitrarily far ahead; the ring exists to do exactly that and
is evidently not keeping up on K3. That is the concrete thing to investigate
next, and it is a smaller and better-defined target than "make the queue deep".

INVESTIGATED, AND THE HYPOTHESIS FAILED. K3's ring reports 136,196 calls with
128,372 stops on "full" at a 9 GiB cap -- 94% blocked -- which looked like a
capacity starvation: no room to run ahead against a 594 GB model. Tested by
raising the cap to 40 GiB. The stops barely moved: full 108,117 plus wrapfull
27,121, still about 99% of calls. Four times the budget bought nothing.

Which means "full" is not starvation. A full ring is prefetch sitting AHEAD of
consumption, waiting for compute to drain it -- the healthy state, not the
broken one. The unconditional stream is evidently well fed; it is the routed
expert reads, which cannot be known ahead, that arrive on demand and bursty.

SO THE ROOT CAUSE OF 2.75 GB/s IS STILL UNKNOWN. Four hypotheses tested and
killed today: shallow batching (queue is 23 deep, retracted), drain gaps as a
bug (they are inherent to MoE routing), ring starvation (cap-independent), and
block size (6.15 MiB slots, far above where it matters). What is left needs a
profiler rather than another guess: per-request latency inside the backend,
cross-file behaviour over fourteen shards, and the staging-buffer copy. I am
stopping this line here rather than publishing a fifth theory -- one retraction
in a day is enough, and the instrumentation now in the readout (qdepth mean/
max/min, submits-below-4, realised bandwidth against ceiling) means the next
attempt starts from measurements instead of intuition.

LESSON, and it is the same one twice in one day: I instrumented the function
whose name matched my hypothesis instead of the layer that owns the truth. The
first version of the M3 sampling made the identical error, reading a finished
run as an idle drive.

---

# SUPERSEDED: effective queue depth is 3, not 64 (2026-08-24)

The owner asked whether this project is in good shape. It is not, and this is
the largest single reason.

CHAIN OF MEASUREMENT. K3 realises 2.0-2.75 GB/s while the drive calibrates at
6.7-6.9 GB/s for both sequential and gather. Sampling the disk counter 40
times during a K3 decode: NEVER idle, 0 of 40 samples below 0.3 GB/s, steady
2.75 mean. So reads are continuous, not fast-with-gaps -- this is not a
failure to overlap compute, the reads themselves are slow. K3 slots are 6.15
MiB, far above the size where block size would explain it. Instrumented
read_batch: 709 batches for 2,089 slices, MEAN 2.9 SLICES PER BATCH, max 16,
against a backend configured for depth 64.

That number matches the calibration table exactly: 512 KiB at QD1 is 2.73
GB/s, which is what we get. read_batch itself is written correctly -- it
submits every slice before waiting on any, and its comment cites the QD1 to
QD16 difference as the whole point. The CALLERS defeat it by handing it three
slices at a time.

WHAT IT IS WORTH. K3 moves 46.22 GB per token and the drive is busy the whole
time at 2.75 GB/s, which is 16.8 s of the 22.7 s token. At the calibrated 6.8
GB/s the same bytes take 6.8 s. That is roughly a 2x improvement on the
flagship, and it applies to every model whose bytes dominate. It dwarfs the
11-15% node-skipping question I spent hours on.

WHY IT WENT UNSEEN -- CORRECTED, because the first version of this entry
blamed the engine for something it does correctly. The K3 readout says, in
plain text: "no drive calibration; limiter verdicts unknown. run: dray
calibrate -m <model>" and "stream 1.87 GB/s over the run, unknown of ceiling
unknown". It reported the realised rate, knew it could not judge it, and said
how to fix that. It refused to guess, which is the behaviour this project
wants.

The REAL gap is narrower and is a design choice, not a lapse: calibration is
stored per MODEL FILE (model.gguf.lzcal.json), but the ceiling it measures is
a property of the DRIVE. Calibrating the 35B teaches the engine nothing about
K3 on the same disk, and a sweep is expensive enough that most models never
get one -- so most runs show "ceiling unknown" and the limiter verdict, the
feature that exists precisely to catch what took three days to find by hand,
stays dark. Sharing calibration per volume would light it up for every model
at the cost of one sweep.

DONE, same day. A model with no calibration of its own now borrows one from
any .lzcal.json on the SAME VOLUME (walking up from the model directory and
one level into each sibling, because models live at repo/quant/file.gguf so a
neighbour's calibration is never in a shared parent's own file list). The
borrow is DISCLOSED, never silent: the readout names the file it came from and
says to re-calibrate this model to remove it.

With that, K3 now prints on its own, in one line, what took three days to find
by hand:

  stream  1.66 GB/s over the run, 24.8% of ceiling 6.70 GB/s

That line is the feature the project always intended -- "tells you whether the
engine or the disk is the limit" -- and it had been dark for every model whose
own sweep nobody had run.

UNIVERSAL, and the shape is exact. Instrumented three models:
  35B-A3B  @3 GiB  292 batches /   572 slices  mean 1.96  max 8
  122B     @3 GiB  342 batches /   678 slices  mean 1.98  max 8
  K3       @9 GiB  709 batches / 2,089 slices  mean 2.90  max 16
The MAX equals experts-used-per-layer in every case. So a batch is at most ONE
LAYER of expert reads, and averages two -- there is no cross-layer batching and
no lookahead deep enough to keep the queue full. The backend is configured for
64 and never sees more than 16.

WHO IT ACTUALLY COSTS, stated carefully because the readout percentage invites
over-reading. That line is bytes divided by WALL against the drive ceiling, so
it measures UTILISATION, not read efficiency: the 35B at 8 GiB shows 1.6% of
ceiling because it is compute-bound and barely reads, which is expected and
fine. The models that lose are the ones where the drive is continuously busy
and still slow -- K3 sampled 40 times mid-decode was never idle at 2.75 GB/s
against a 6.8 GB/s ceiling. So the fix is worth about 2x on K3, GLM, Qwen 2.4T
and M3, and roughly nothing on the sparse models, whose bottleneck is compute.

SCOPE, now measured rather than assumed. Sampling the disk counter DURING
decode, with the process confirmed alive so a finished run cannot masquerade
as an idle drive (the first M3 attempt made exactly that mistake and its
22 zeros were simply the run having ended):

  K3  @9 GiB   40 samples, 0 idle, mean 2.75 GB/s  = 40% of a 6.8 ceiling
  M3  @8 GiB   30 samples, 1 idle, mean 1.17 GB/s  = 17% of ceiling

Both are CONTINUOUSLY BUSY AND SLOW, which is the read-bound signature. The
sparse models are the opposite: measured directly, the 122B costs the same 2.7
s/token whether it streams 872 MiB or nothing at all, so its drive is idle
capacity and no queue fix touches it.

So the repair is worth roughly 2x on K3 and potentially more on M3, applies to
GLM and Qwen 2.4T by the same reasoning (large bytes per token, same shallow
batching), and is worth NOTHING on the 35B, 122B, DeepSeek and the dense 27B.
Anyone benchmarking the fix on a sparse model will see no change and conclude
it failed.

SUPERSEDED: this entry's premise (shallow batching) was retracted, and the real
cause is above -- the I/O path only runs between compute nodes. Do not chase
"batch across layers" from here; the horizon forbids it for routed experts
anyway. The live target is keeping submission and completion running DURING
compute, and the completion half was attempted and reverted on 2026-08-24.

---

# The llama.cpp baseline, and a "parity" claim that was not true (2026-08-24)

The owner asked for stock scores alongside ours. Stock can only load three of
the seven here (79.7 GB of RAM): the 27B at 16 GB, the 35B at 21, the 122B at
39. DeepSeek at 90 GB, M3 at 143, GLM at 217 and K3 at 594 CANNOT BE RUN by
stock on this machine at all, which is both the honest entry for those rows
and the entire argument for this project.

Where a fair fight exists, with load excluded from both sides (our windowed
steady-state against llama-bench tg32):

  model              ours resident   stock    prompt: ours CPU / stock
  qwen3.6 35B-A3B    7.5 tok/s       9.10     15.3 / 25.7 tok/s
  qwen3.5 122B-A10B  2.7 tok/s       3.35     4.0 / 6.45
  qwen3.8-27B dense  1.2 tok/s       1.75     2.5 / 3.29

STOCK WINS EVERY CELL, by 21% to 46% on generation and 30% to 70% on prompt.

AND THE README WAS WRONG. It claimed resident mode reached "parity" with
stock. It does not, and I made that claim twice -- first from 1.1 against 1.43
tok/s, then in the README -- by gesturing at load and prefill overhead instead
of measuring the windowed rate, which is the number llama-bench reports. The
file now carries the measured table and says stock is faster. The
recommendation was always right (if it fits, use llama.cpp); the number
attached to it was flattering.

What this does NOT change: the streaming path is a different regime, and on
the four models stock cannot load the comparison is not 20% or 3x, it is
infinite.

---

PLACEHOLDER_NEVER_MATCHES: seven models x cap x context x GPU, measured

Every cell a real run; scripts/matrix-results.csv carries all 145 rows and
scripts/matrix.ps1 reproduces them. DECODE, varied cap, 4k context:

  model                cap    s/token CPU   s/token GPU   bytes/token
  qwen36-35b-a3b       3 GiB  1.9           REFUSED       575 MiB
  qwen36-35b-a3b       8 GiB  1.7           1.8           415 MiB
  qwen36-35b-a3b       28 GiB 5.7 tok/s     3.1 tok/s     0 (resident)
  qwen35-122b-a10b     3 GiB  2.6           REFUSED       2.50 GiB
  qwen35-122b-a10b     12 GiB 2.3           2.6           872 MiB
  qwen35-122b-a10b     28 GiB 2.6           2.7           360 MiB
  deepseek-v4-flash    3 GiB  4.6           REFUSED       5.80 GiB
  deepseek-v4-flash    8 GiB  3.9           4.5           1.93 GiB
  deepseek-v4-flash    24 GiB 3.8           3.9           1.56 GiB
  minimax-m3           8 GiB  6.8           REFUSED       4.77 GiB
  minimax-m3           28 GiB 5.3           5.5           3.30 GiB
  glm-5.2-744b         8 GiB  8.6           REFUSED       12.80 GiB
  glm-5.2-744b         28 GiB 8.3           8.6           5.45 GiB
  kimi-k3-2.8t         9 GiB  25.8          REFUSED       46.22 GiB
    (matrix cells are 32 generated tokens except K3 at 16, where loading 594 GB
     still dilutes the mean; the same config over 32 tokens measures 22.7)
  kimi-k3-2.8t         28 GiB 24.3          26.1          27.22 GiB

PREFILL, 6,594-token prompt at 32k context, tokens/second:

  qwen36-35b-a3b       8 GiB   12.8 CPU    343.4 GPU    27x
  qwen36-35b-a3b       28 GiB  15.3 CPU    180.7 GPU    12x
  qwen35-122b-a10b     12 GiB  4.0 CPU     280.6 GPU    70x
  qwen35-122b-a10b     28 GiB  4.0 CPU     289.2 GPU    72x
  qwen38-27b-dense     8 GiB   2.5 CPU     (see note)
  deepseek-v4-flash    24 GiB  2.6 CPU

FOUR THINGS THE TABLE SAYS AT ONCE.

1. --gpu RAISES THE FLOOR, on every single model. 35B needs 4 GiB instead of
   3, the 27B 5 instead of 4, M3 9 instead of 8, K3 10 instead of 9. The
   Vulkan compute buffer is inside the cap. For a tool whose point is leaving
   the machine usable, that is a real cost.
2. GPU DECODE ALWAYS LOSES, without exception across 15 comparable pairs.
   Worst case the 35B at 28 GiB: 5.7 tok/s CPU against 3.1 GPU.
3. GPU PREFILL ALWAYS WINS, by 12x to 72x. The split is not a preference, it
   is a property: prefill is a batched matmul, decode is not.
4. MORE RAM IS NOT ALWAYS FASTER. The 122B decodes at 2.3 s/token with 12 GiB
   and 2.6 with 28. Bytes fall from 872 to 360 MiB and the wall gets WORSE,
   because the disk was never the limit for that model and a bigger cache
   costs more bookkeeping. Buying RAM helps only where bytes bind.

---

# A published K3 figure does not reproduce, and the drive is not the reason

The performance matrix ran K3 and got REFUSED where the README promised 8
GiB. Chased it rather than filing it.

WHAT IS TRUE NOW, measured 2026-08-24 on the current build:
  8 GiB, ctx 4096: REFUSED (output.weight needs 918 MiB whole, cap leaves 150)
  8 GiB, ctx 2048: REFUSED (same, 206 MiB left)
  9 GiB, ctx 4096: RUNS, correct text, 46.22 GB/token, 22.7 s/token steady

The README claimed a 6 GiB minimum and 8 GiB at 16.1 s/token. Neither holds.
The refusal is pre-existing engine behaviour, not something this week's
planner change introduced -- the same message appeared on the 27B and the
122B days earlier. The likely history is the one this project already
documented for Qwen3.8 (4 GiB became 5 GiB): the figure predates the
resident-byte ledger counting everything it now counts.

AND THE DRIVE IS NOT THE EXCUSE. Calibrated within the hour: 6.7-6.9 GB/s at
the chosen operating points, sequential AND gather, QD 16-64. K3 realises
46.2 GB in 22.7 s, which is 2.0 GB/s. The engine leaves roughly two thirds of
this drive unused on that workload. That is consistent with the shallow-queue
symptom seen under batch prefill (645 MB/s against the same 6.8 GB/s ceiling)
and is now the most valuable open performance lead in the project -- bigger
than the 11-15% node-skipping question, on the models where bytes actually
dominate.

SEPARATELY: --kv q4 does not work on Kimi K3 at all. Quantised KV forces
flash attention and llama fails to create the context. Undocumented until
now; the matrix found it by trying.

---

# The node-skipping bug was VIEWS, and the 42% was never real (2026-08-23)

The owner asked what "risky" meant, which was the right question -- it meant I
had not read the code. Reading it took twenty minutes and answered everything.

THE COUPLING IS ONE VARIABLE. im.current is a one-node protection window:
materialise clears it and pushes what this node needs; exactly two readers
consult it, make_room when evicting and pump_ring when retiring. No fork
change is involved anywhere; this is one file of ours.

THE BUG WAS NOT THAT WINDOW, IT WAS VIEWS. ggml fixes a view's data pointer at
GRAPH-ALLOCATION time, so materialise re-points every view on every node --
the hazard already documented in this file as the Kimi K3 ssm_a case, sitting
about forty lines below the code I was speculating about. A skipped node never
gets that repointing and reads a stale address. Claim any node with a view
source and all FIVE architectures match with DRAY_FAST_NODES=1, including
the two that previously diverged and failed.

AND THE PAYOFF WAS INFLATED BY THE BUG. The 42% (4.0 -> 2.3 s/token on the
27B) came from the BROKEN version, which was fast partly because it skipped
work it needed to do. Corrected:

  27B    2.6 -> 2.2 s/token   claims 65,399 -> 36,771    15%
  122B   2.7 -> 2.4 s/token   claims 152,295 -> 78,903   11%

So the standing recommendation changes twice over: the correctness objection
is GONE (found, understood, fixed, gated on five architectures), and the
reward is 11-15% rather than 42%. Whether that justifies making it default is
the owner's call, but it is now a decision about 13%, not about risk.

TWO LESSONS, both mine. "It might be risky" is what you say when you have not
read the code; the file documented both the failed refcount attempt AND the
view hazard in comments adjacent to the bug. And a speedup measured on a
broken version is not a speedup.

---

# For sparse models the DISK IS FREE, and that decides the rewrite question

Measured to settle whether reclaiming the node-skipping 42% is worth doing.
Qwen3.5-122B-A10B, streaming path PINNED with --force-stream in both legs so
allocation mode is not a variable, 32 tokens:

  cap     streamed/token   mean rate
  12 GiB  872 MiB          2.7 s/token
  56 GiB  0 B              2.7 s/token

IDENTICAL. Reading 872 MiB per token costs no measurable wall time against
reading nothing at all. The drive is entirely hidden behind compute and
per-node overhead at this model's byte rate, so for the sparse models the
wall clock is 100% engine, 0% disk.

This inverts the project's founding assumption for this class of model. The
cost model in CLAUDE.md prices a token as misses x bytes / bandwidth, which
is exactly right for K3 (54.5 GB/token, genuinely disk-bound at ~3.4 GB/s
realized) and exactly wrong for a model that moves under a gigabyte. The
knee has a second side nobody had measured: below some byte rate the disk
stops being the limit and OUR OWN per-node barrier is.

CONSEQUENCE FOR THE OPEN DECISION: the 42% that node-skipping was worth is
not a fraction of a fraction here -- it would come off the entire wall clock,
taking the 122B from 2.7 s/token toward 1.6. That is the strongest argument
yet for making the streamer state machines independent of callback cadence,
and it applies precisely to the models that turned out to be the interesting
ones. The risk is unchanged and still the real objection: silent wrong output
in the hot path, invisible to three of five gates.

---

# The repo only built on this machine, and nobody would have known (2026-08-23)

The worst defect found this week, and it was invisible to every gate: the
parent tree pinned the fork at 16cb54a4c while src/backend/stream_buffer.cpp
called ggml_cpu_repack_accept_buft, _traits_for and _data_in_place -- three
functions that existed ONLY in the working copy of the submodule. A clone
would configure, compile every translation unit, and fail at LINK. The calls
are emitted whatever the feature defaults to, so turning repack off did not
hide the dependency.

Every local build passed, including a deliberate from-scratch one, because a
from-scratch build still uses the working tree. Nothing short of an actual
clone could catch it.

FIXED: fork commit c7dba9019 pushed to dray-k3, submodule pointer moved,
and then VERIFIED the way the claim deserved -- a real clone into a scratch
directory, submodule fetched from GitHub rather than from disk, 253/253
targets linked, 12/12 tests, artifact timestamped. Not an exit code.

THE PATTERN, three instances in one day, all the same shape: a gate that
verified nothing (verify skipped every shard but one), a Linux build that
compiled nothing (Git Bash rewrote the path, failure surfaced as exit 0), and
a repo that built nowhere but here. Each time the exit code said fine and the
ARTIFACT said otherwise -- an empty comparison count, a three-day-old binary,
a submodule SHA. Check the artifact.

---

# verify now checks the half that mattered more (2026-08-23)

verify compared routed expert slices and nothing else. That is the sampled
half: hundreds of thousands of slices, of which a token reads six per layer.
The UNCONDITIONAL set -- attention projections, shared experts, embedding,
output -- is read on EVERY token, so a bad offset there corrupts every token
rather than the ones that happen to route through a bad expert. It was
unchecked.

A second pass now compares a 4 MB prefix of every unconditional tensor
against a buffered reference (a prefix suffices: these are contiguous reads,
so an offset or alignment error shows in the first bytes, and the bound keeps
output.weight from costing a gigabyte of memcmp). The verdict line now covers
both classes.

DeepSeek V4 Flash: 881 unconditional tensors and 516 expert slices compared,
ZERO mismatches. Which also retires the last doubt about that model's garbage
output having any I/O cause -- both halves of the file read byte-identical.

---

# Short-run audit: the old numbers hold where it matters (2026-08-23)

The admission ramp finding raised a fair worry -- if a 16-token run reports a
cold cache, is every published figure here pessimistic? Audited by re-running
GLM-5.2 at 8 GiB for 128 tokens against its recorded 8.6 s/token short-run
figure.

RESULT: 7.9 s/token, 12.80 GB/token, h_routed 5.7%. Slightly better, not
materially. The reason is the knee, which this notebook already documented:
at 8 GiB against 217 GB of routed experts the cache cannot hold a useful
fraction of the working set, so it never fills and there is no ramp to miss.
The ramp matters only where the cache CAN hold a meaningful share -- DeepSeek
at 24 GiB, where 16 tokens showed 37% and 128 tokens showed 50%.

So: the 8 GiB flagship figures stand. High-cap figures on sparse models should
be taken past 64 tokens. Both rules now stated rather than assumed.

Also measured: Qwen3.6 35B-A3B minimum cap is 3 GiB with correct text,
completing that row.

---

# The sparse-MoE end of the collection, measured (2026-08-23)

With the engine correct again, the three sparse MoE models measured this week
are the cheapest things this project has ever run, and they invert the usual
intuition that bigger costs more:

  model                    disk    cap     GB/token   s/token
  Qwen3.6 35B-A3B          21 GB   12 GiB  0.287      1.5
  Qwen3.5 122B-A10B        39 GB   12 GiB  0.872      2.5
  DeepSeek V4 Flash 284B   90 GB   8 GiB   1.93       6.3
  DeepSeek V4 Flash 284B   90 GB   40 GiB  1.18       --
  (for scale) Kimi K3 2.8T 594 GB  8 GiB   54.5       16.1

Minimum caps with correct text: 35B-A3B unmeasured, 122B-A10B 3 GiB, DeepSeek
V4 Flash 3 GiB. A 284B model on three gigabytes.

What sets the cost is ACTIVE bytes per token, not parameter count: 8 of 256
experts over 48 layers is about a gigabyte, whatever the total happens to be.
That is the whole argument for this engine stated in one table -- disk size is
nearly free, sparsity is what you pay for, and the models the industry ships
now are getting sparser, not denser.

Also closed: the DeepSeek 8 GiB segfault was never a separate defect. It was
the same node-skipping predicate, and the same revert fixed it -- 8 GiB now
runs clean at 1.93 GB/token with a 32.9% hit rate.

---

# The node-skipping optimisation is DEAD, and the reason is structural (2026-08-23)

Third attempt, third failure, and this one settles it. The predicate was
rewritten to ask an EXACT question -- does this node read a tensor allocated
in our buffer type, following view chains to the root -- rather than
enumerating which op kinds can reach our memory. Buffer ownership is a fact
ggml already tracks, so this should have been airtight.

ARCHGATE SAYS NO: with DRAY_FAST_NODES=1, Qwen3.6-35B-A3B DIVERGED and
DeepSeek V4 Flash FAILED OUTRIGHT, while the testbed and the dense 27B
matched. Two of four architectures.

THE STRUCTURAL REASON, which is worth more than the 42%: the one-node-per-call
shape does not merely guarantee that a weight is materialised before it is
read. It BRACKETS THE STREAMER'S OWN STATE MACHINES -- the ring stream list,
the LRU and eviction ordering, the per-token expert compaction and its ids
remap, and the release() that closes each node. Those advance per CLAIMED
node, so skipping a node that touches none of our memory still perturbs the
sequence they expect. No predicate over node contents can fix that, because
the bug is not about node contents.

So the flag stays opt-in, now prints an unmissable warning naming the two
architectures it is known to break, and the 4.0 -> 2.3 s/token it buys stays
unclaimed. To collect it properly the streamer's state machines would have to
be made independent of the callback cadence -- a real piece of work, filed
rather than faked.

And archgate earned its existence on its first real use: it caught in one run
what three gates on three models missed twice.

---

# DeepSeek V4 Flash MEASURED, and a methodology error found (2026-08-23)

With the predicate off, the model is correct and the numbers are the best any
flagship has posted here. Qwen-style prompt, 4k ctx, --kv q4, 16 tokens:

  cap     GB/token   text
  12 GiB  1.84       coherent
  24 GiB  1.56       coherent
  40 GiB  1.18       coherent

Against the rest of the collection that is remarkable: K3 (2.8T) costs 54.5
GB/token, the dense 27B costs 12.93 at its floor, and this 284B model costs
1.18-1.84. Sparsity is why -- 6 experts of 256 per layer, so a token touches
~2 GB of the 83.6 GB of experts. Its floor is 362 MiB and MLA keeps KV at 96
MiB for 4k context. Fixing the predicate also restored the cache: h_routed
went from 3.2% to 37.2%.

METHODOLOGY ERROR, and it is not confined to this model: the streamer admits
at most 32 experts per token (the storm guard, which exists because filling a
10 GB pool in one token cost 105 s). DeepSeek touches 258 expert-instances
per token, so a 16 GB pool needs ~2,000 admissions -- at least 63 tokens --
before the cache reaches steady state. Every 16-token measurement therefore
reports a COLD cache. Proof from the runs above: at a 24 GiB cap the budget
is 23.63 GB but in-use stopped at 9.97 GB, byte-identical to the 12 GiB run.
Short runs UNDERSTATE MoE caching, and any figure in this notebook taken with
a handful of tokens should be read as an upper bound on cost, not a
steady-state result. MEASURED: at 128 tokens the cache reached 19.58 GB of a 23.63 GB budget and
h_routed rose to 49.7% (h_bytes 78.3%), against 9.97 GB and 37.2% at 16
tokens. The ramp is real and it is slow.

Qwen3.5-122B-A10B, added the same day, is cheaper still: 8 of 256 experts
over 48 layers puts a token at 1.09 GB cold, its floor is 448 MiB and its KV
is 27 MiB at 4k. ENGINE-MEASURED: minimum cap 3 GiB with correct text (2 GiB
refuses cleanly, naming output.weight at 409 MiB as the obstacle), and at 12
GiB over 64 tokens it costs 872 MiB/token at 2.5 s/token -- the cheapest and
fastest flagship this project has measured. A 122B model on a 3 GiB budget.

---

# DeepSeek V4 Flash works -- and MY optimisation was the bug (2026-08-23)

The garbage was not DeepSeek and not the fork. It was the node-batching
predicate I shipped yesterday. With DRAY_NEEDS_ALL=1 -- the old
claim-every-node shape -- the model answered "The capital of France is" with
" Paris." immediately. The predicate is now OFF BY DEFAULT and DeepSeek is
correct out of the box.

WHY IT ESCAPED: the predicate answers "does this node touch a streamed
weight?" and I proved it with difftest, residenttest and batchdiff on three
models. DeepSeek V4 reaches its HYPER-CONNECTION weights (hc_attn_base,
hc_attn_scale, hc_mixes) through node shapes the predicate misjudges;
following the full view chain and removing the view family from the
allowlist was NOT enough, so the residual is something else in the same
family. An optimisation that must enumerate "which node kinds can reach our
memory" is only as correct as the enumeration, and a new architecture
falsifies it.

COST OF SAFETY, stated plainly: claiming every node costs a thread-pool
barrier each -- 4.0 vs 2.3 s/token on the 27B, the 42% measured yesterday.
That gain now lives behind DRAY_FAST_NODES=1 with a warning to check
your own model's output first. Wrong tokens are not worth 42%.

WHAT THE GATE SUITE LEARNED: three models and four gates all passed while a
fourth architecture produced fluent nonsense. Architecture diversity is a
GATE DIMENSION, not a nice-to-have -- DeepSeek (hyper-connections + MLA)
belongs in the pre-commit set, and the next optimisation of this kind does
not ship until it runs there.

CLOSED 2026-08-24, and it was never an allocation-failure path. The crash was
the node-skipping predicate, the same defect that produced the fluent-wrong
text; the same revert fixed both. Retested afterwards: NO_COMPACT at a 48 GiB
cap exits 0, produces text, reports no taint, and sits at 48.86 GB in use
against a 51.5 GB cap. Nothing here needed a refusal because nothing here
failed to allocate.

---

# DeepSeek V4 Flash: BROKEN, and the search so far (2026-08-23)

The model whose geometry fits this engine best -- 284B total, 6 of 256
experts per token, MLA giving 96 MiB of KV at 4k, a 362 MiB floor, 1.96
GB/token cold, plan-feasible at 4 GiB -- does not work. Recorded before it
is fixed because a negative this size belongs in the notebook immediately.

SYMPTOMS: segfault at an 8 GiB cap; at 12 and 24 GiB it runs but emits
FLUENT, DETERMINISTIC, SEMANTICALLY WRONG text ("The capital of France is"
-> "DOS as a common shorthand for"); h_routed 3.2% where the arithmetic
says ~28% at 24 GiB.

RULED OUT so far: fusion, early-unlock, eslots and row-slicing (all five
configurations produce BYTE-IDENTICAL wrong output); the floor (integrity
check passes, 32 tensors match the file); and the I/O path -- verify now
compares 516 expert slices across ALL shards and finds ZERO mismatches, so
the bytes we read are the bytes on disk.

TOOL FIXED ON THE WAY: verify opened only the shard named on the command
line, so on this model it skipped all 129 routed tensors and reported
"INCONCLUSIVE -- nothing compared". It now walks every shard via
plan.shard_paths. A verifier that silently verifies nothing is worse than no
verifier.

NO REFERENCE AVAILABLE: stock llama.cpp cannot run this model on this
machine at all (90 GB against 79.7 GB of RAM), which is exactly the regime
this engine exists for -- and exactly why the bug has to be found from the
inside.

RESOLVED: compaction was not the culprit. The predicate was. See the
views entry above -- ggml fixes a view's data pointer at graph-allocation time,
so a skipped node reads a stale address, and claiming any node with a view
source makes all five architectures match.

---

# Prefill, answered properly (2026-08-23): 47 minutes to 22 seconds

The owner: "prefill still looks very slow, no way to improve this further?"
There was, and the reason it had not shown up before is a bug in the flag I
added for it: n_ubatch -- the PHYSICAL batch, the thing actually computed in
one graph -- was never set and stayed at llama's 512 no matter what
--prefill-chunk said. So raising the chunk could not do anything, in either
direction. With n_ubatch following the chunk, measured on a 6,594-token
prompt (Qwen3.8-27B):

            chunk 512     chunk 2048
  CPU       47m23s        44m03s
  GPU       43s           23s        (287 tokens/second)

Chunk size barely moves CPU and nearly halves GPU, which is the expected
shape: a bigger physical batch amortises the PCIe weight transfer, the same
cost that makes GPU decode lose. So the default now follows the device --
2048 under --gpu, 512 on CPU where the win is small and the buffer costs
capped RAM -- and --prefill-chunk still overrides for small cards (1.3 GiB
of VRAM at chunk 64, measured).

END TO END, no flags but --gpu: a 6,594-token prompt went from 47 minutes to
22 SECONDS across today's work (thread split, KV kept off the GPU, weights
kept out of VRAM, ubatch, chunk default). That is the difference between
"long prompts are impractical" and "long prompts are free".

---

# Seamless (2026-08-22): --gpu becomes a switch worth flipping -- 29x prefill

The GPU work from yesterday was measured against a crippled CPU (4 threads)
and a bad configuration (weights and KV drifting to VRAM). Both fixed, and
the result is the largest single number this project has produced:

  834-token prefill, Qwen3.8-27B:  CPU 4m21s   GPU 7-9s     ~29x
  decode, same model:              CPU 1.4 tok/s  GPU 1.1     -21%

THREE changes made it seamless rather than a tuning exercise. (1)
n_gpu_layers = 0 under --gpu: weights NEVER move to VRAM, which is both the
project's premise and the fix for resident mode trying to push a 17.5 GB
model into an 8 GB card. (2) offload_kqv = false: the KV cache stays in CPU
RAM, so attention does not follow it into VRAM (that was the 852-split
graph behind yesterday's slow GPU decode) and the cache stays inside the
accounted cap -- the reservation-release special case is deleted. (3)
Nothing else: ggml already offloads only ops above a batch-size threshold
(32 by default in the Vulkan backend), so PREFILL offloads and DECODE does
not, with no phase detection of ours.

A TRAP CAUGHT BY THE GATES, worth recording: n_gpu_layers = 0 was first set
unconditionally, and residenttest diverged -- on a CPU-only run it changes
llama's preferred buffer type and therefore its kernels. It is now scoped to
GPU runs. The first hypothesis (thread-count reduction order) was wrong, and
pinning --threads in the gate did not fix it; the gate was right and the
reasoning was wrong.

The trade to state plainly: --gpu is transformative for prompts and costs
21% on generation. Break-even is roughly prompt > two thirds of output
length, which nearly every real workload clears.

---

# Seamless (2026-08-22): prefill and decode get different thread counts

"There must be ways to make this seamless" -- and the first one costs the
user nothing. llama has ALWAYS had separate n_threads and n_threads_batch;
this engine set both to the same number. They are different workloads:
decode is bandwidth-bound and measured flat from 4 threads up, while prefill
is compute-bound over 512-token chunks and scales. Defaults now differ --
decode 4, prefill the oversubscription ceiling (cores - 9) -- and an
explicit --threads still sets both, because someone measuring wants one
variable rather than two.

MEASURED: 834-token prefill on the 27B, 7m02s to 5m01s. 29% off the slowest
part of a long-prompt run, with no flag to discover and no behaviour to
understand. Gates identical.

---

# A DONE (2026-08-22): two root causes, both fixed -- barriers and oversubscription

The 81x thread collapse is a SECOND, independent defect, and the barrier fix
did not touch it: after cutting barriers 86%, 22 threads still measured 61
s/token. Resident mode collapses identically (1.1 -> 50.7), which rules out
the streaming path entirely. The cause, measured directly: this process runs
31 threads when 22 are requested -- ggml's pool plus our IOCP completions,
main, and backend -- on a 22-core machine, and ggml's pool SPIN-waits, so
the spinners starve the workers. Stock llama.cpp does not collapse because
it adds no threads of its own.

FIXED: the engine clamps the requested pool to (cores - 9) and says so.
22 requested becomes 13 used, and the rate goes from 45-61 s/token to 1.2 --
a 40x rescue for the most natural thing a user can type. Both defects are
now one-line-explainable: too many barriers, then too many threads.

---

# A: ROOT CAUSE FOUND AND PARTLY FIXED (2026-08-22) -- one barrier per node

The 3x against stock is not kernels, not I/O, and not our callback body. It
is ggml-backend.cpp's callback_eval branch: when an eval callback is
installed, the graph is executed NODE BY NODE, with a full
ggml_backend_synchronize -- a thread-pool barrier -- after every node the
callback claims. This engine claimed EVERY node, so it paid ~4,300 barriers
per token. That single fact explains all three open symptoms: the 3x
penalty, the flat thread scaling, and the 81x collapse at 22 threads
(barrier cost scales with pool size).

PROOF it is not the alternatives: at a 40 GiB cap with ZERO bytes streamed
per token, the streaming path still runs 4.0 s/token against resident's
1.2. Our callback body accounts for 0.7 s/token of that (measured
 directly: 5.6 s over 34,623 calls). The rest is barriers.

FIX, SHIPPED: the callback now answers FALSE for nodes that cannot touch a
streamed weight, letting ggml batch them into one compute call. 34,623
calls per 8 tokens became 15,300, and 4.0 s/token became 3.1-3.3. All gates
identical: difftest, residenttest, batchdiff (both checks), suite 12/12.

AND THE FAILURE THAT SHAPED IT: the first version used the obvious
predicate -- "no disk-backed source" -- which is 16% faster and WRONG. Both
difftest and residenttest diverged. The shipped version is therefore a
NARROW ALLOWLIST of elementwise/normalisation ops (add, mul, rms_norm,
soft_max, rope, unary, cont, reshape, permute, transpose, view) that read
activations from llama's own compute buffer and can never reach ours. It is
proven by the gates rather than by argument, and widening it is a
gate-verified exercise, not a judgement call.

EXTENDED, same method, each step gate-verified: a second op tranche
(flash_attn_ext, cpy, scale, set_rows, div, concat, sqr, sum_rows, clamp)
took calls to 11,844 and the rate to ~3.0. Then the decisive one: a bound
tensor that is resident AND PINNED can never be evicted, so its data
pointer stays valid for a whole batch and its node needs nothing from us.
With that, calls fall to 4,835 and the rate to 2.3-2.4 s/token across three
runs -- 42% off the original 4.0, with difftest, residenttest and both
batchdiff checks identical throughout.

---

# A RESOLVED (2026-08-22): the 3x was never the repack kernels

A of the mandate, and the answer is a negative result plus a corrected
diagnosis -- which is worth more than the fix I set out to build.

WHAT WAS BUILT: the vendored fork now exposes three entry points
(ggml_cpu_repack_accept_buft, ggml_cpu_repack_traits_for,
ggml_cpu_repack_data_in_place) so a streaming buffer can opt into ggml-cpu's
interleaved matmul kernels -- which is otherwise IMPOSSIBLE, because
supports_op tests POINTER IDENTITY against ggml_backend_cpu_repack_buffer_type
and get_tensor_traits reads tensor->extra, neither of which any foreign
buffer can satisfy. The streamer repacks whole 2D weights on materialisation
and sets extra.

WHAT IT MEASURED: a 20% LOSS. 3.9/3.8 s/token with repack against 3.2/3.2
without, two runs each on the 27B. Two reasons, both instrumented rather
than guessed: the transform is paid on EVERY materialisation while the
streaming path re-reads tensors constantly, and only 74 of 545 2D tensors
in this UD quant have an optimal repack type at all (471 declined -- ggml's
table covers Q4_0/Q4_K/Q5_K/Q6_K/Q2_K/IQ4_NL/MXFP4/Q8_0, and dynamic quants
lean on types outside it). Default is OFF; DRAY_REPACK=1 reproduces.

THE CORRECTED DIAGNOSIS, which is the real finding: stock llama.cpp faces
the SAME 14% repack coverage, so repack was never the source of its
advantage. Resident mode -- which simply lets llama allocate -- runs 1.1
tok/s against stock's 1.43 measured on tg8, i.e. at parity once our load and
prefill are accounted for. THEREFORE the 3x penalty lives in the streaming
path's per-node materialisation itself (the cb_eval round trip, the resident
map, the accounting), not in kernel selection. That also fits the two
unexplained symptoms: no thread scaling, and the 81x collapse at 22 threads.

WHERE THIS LEAVES THE PROJECT: models that fit run at stock speed (resident
mode, B). Models that do not fit pay the streaming overhead, and there it is
masked by I/O -- K3 spends ~2 s of a ~16 s token computing. The machinery
stays in the tree because INSTALL-TIME repacking (pay the transform once, on
disk, next to the existing repack tool) is the design that could still win;
it needs an ISA stamp, since interleave width is CPU-specific.

---

# DONE (2026-08-22): resident mode is correct, 3x faster, and the default

B of the owner's two-item mandate, complete. The garbage had a precise
cause and it was ours: the fast load path hands llama a NO-OP tensor loader
(`auto no_load = [](ggml_tensor*, void*) {}`) because in streaming mode
weights are materialised on demand and must never be read at load time. With
no streamer filling them, resident tensors stayed uninitialised -- hence
"!!!!!!!!". Resident mode now uses llama's real loader with DIRECT_IO, so
Invariant 2 still holds (no retained page cache, no mmap of tensor data).
Two further fixes were needed on the way: our cb_eval must not repoint
tensors llama owns, and the floor self-check must report NOT APPLICABLE
rather than fail when we own nothing to inspect.

MEASURED, Qwen3.8-27B UD-Q4_K_XL at a 20 GiB cap: resident 1.2 s/token,
streaming 3.6 -- the predicted 3x, with IDENTICAL generated text on both the
27B and the testbed model. Stock llama-bench measured 0.58-0.76 s/token on
the same model, so resident mode closes most of the gap that the control
exposed; what remains is prefill/load amortisation in a short run.

NEW GATE, and it is the reason this could flip: scripts/residenttest.ps1
asserts resident text == streaming text VERBATIM, refuses to pass if the
resident leg did not actually enter resident mode (the vacuous-gate trap),
and VOIDs on taint or empty output. All four gates green together: difftest,
residenttest, batchdiff (within-batch and compact), suite 12/12.

DEFAULT IS NOW OPT-OUT as the owner asked: if the whole model fits the cache
budget, llama allocates it natively and reaches its repacked-weight kernels.
--force-stream pins the streaming path and is MANDATORY in every gate and
measurement -- without it, a fitting model silently stops testing this
engine at all.

---

# Shipped partial (2026-08-22): --force-stream lands, resident mode does NOT

Owner: "getting out of the way should be opt out -- I do not want software
getting out of the way when we are testing some cases." Both halves matter
and only one is done.

DONE, and it caught a trap: --force-stream pins the streaming path, and
EVERY gate now passes it -- difftest, batchdiff (both PowerShell and POSIX),
linux-gate. The testbed model FITS an 8 GiB cap, so an opt-out resident mode
would have silently turned this project's primary correctness gate into a
test of llama's allocator instead of our engine. The knob is censused, so
any run that pulls it says so in its own log.

NOT DONE, and honestly labelled: resident mode itself. The idea is sound --
when the whole model fits, hand allocation to llama, reach its
repacked-weight kernels, stop paying our 3x. The implementation produces
FLUENT GARBAGE ("!!!!!!!!"). Two causes found and fixed (our cb_eval was
still repointing tensors llama owns; the floor self-check has nothing to
inspect when we own no tensors and must say NOT APPLICABLE rather than
fail), and it is STILL wrong -- llama evidently still allocates into our
registered buffer type, which nothing then fills. So resident mode is
--resident/DRAY_RESIDENT, OPT-IN, marked experimental, and the default
remains the streaming path every measurement here used. It becomes the
owner's opt-out default the day a gate proves it byte-correct, not before:
shipping a mode that emits garbage would betray the one rule this project
does not bend.

---

# MEASURED (2026-08-22): the control the project never had -- we are 3x slower than stock

Built stock llama-bench from the vendored fork and ran the SAME model on
the SAME CPU. Qwen3.8-27B UD-Q4_K_XL, tg8:

  threads   stock llama.cpp   dray        gap
  4         0.70 s/tok        2.2 s/tok      3.1x slower
  12        0.58 s/tok        3.7 s/tok      6.4x slower
  22        0.76 s/tok        62 s/tok       81x slower

TWO SEPARATE DEFECTS, now distinguishable for the first time:

(1) A ~3x COMPUTE-PATH PENALTY at every thread count. Stock runs this model
at 0.58-0.76 s/token; we run it at 2.2 with the model fully resident and
zero disk activity. Prime suspect remains our custom ggml buffer type
bypassing ggml-cpu's optimized quantized-matmul path (the extra-buffer-type
repack that interleaves Q4_K for SIMD). If that is it, fixing it is worth
~3x on EVERY compute-bound configuration, on every model.

(2) A CATASTROPHIC THREAD PATHOLOGY that is OURS alone. Stock merely
plateaus with threads (1.43 -> 1.71 -> 1.32 tok/s across 4/12/22 -- this
workload is memory-bandwidth bound, so flat scaling is NORMAL and expected).
We go 2.2 -> 3.7 -> 62 s/tok. An 81x gap at 22 threads is contention in our
own structures (accountant atomics, streamer locks, per-node cb_eval), not
physics. This likely also explains the shallow I/O queues under batch (645
MB/s against a 6.7 GB/s drive).

The honest framing for the post: the streaming architecture is sound and its
I/O numbers stand, but the compute path currently costs 3x what stock pays,
and that is a bug with a name, not a tax on the design.

---

# MEASURED (2026-08-21): the engine is bound by FIXED PER-TOKEN OVERHEAD, not compute

The most consequential finding of the campaign, and it invalidates every
compute-bound number this notebook published before today. Chased from the
owner asking why a 27B felt slow.

FACT 1 -- llama's n_threads default is FOUR (GGML_DEFAULT_N_THREADS) and
this engine never set it. Every compute-bound measurement ever taken here
used 4 threads of a 22-core machine.

FACT 2 -- raising it makes things WORSE, catastrophically. Qwen3.6-35B-A3B,
everything resident, zero bytes from disk:
  4 threads 2.2 s/tok | 6 -> 2.3 | 8 -> 2.3 | 12 -> 3.4 | 22 -> 62.0
Dense Qwen3.8-27B: 4 -> 2.2, 12 -> 3.7. ENGINE-WIDE, not MoE-specific.

FACT 3 -- the smoking gun: a DENSE 27B (27B active) and a 35B-A3B (~3B
active) take EXACTLY the same 2.2 s/token at 4 threads. Two workloads whose
arithmetic differs 9x cost identical time. The time is therefore NOT
compute; it is fixed per-token overhead in our own path. The 7B testbed
runs 1.0 s/tok, which no CPU explanation fits either.

WHAT THIS RETRACTS: "the 27B is compute-bound at 1.5 s/token" was wrong --
it is OVERHEAD-bound. "MoE is 9x less efficient per active parameter" was
wrong -- both models are pinned at the same overhead floor. The GPU prefill
win (5.4x) was measured against a 4-thread CPU and needs re-measuring.
I/O-bound flagship numbers (K3, GLM, Qwen 2.4T batch curves) are unaffected:
those wait on the drive, which dwarfs the overhead.

LEADS, in order of suspicion: (1) our custom ggml buffer type may bypass
ggml-cpu's optimized quantized-matmul kernels (the "extra buffer type"
repack path), forcing a generic scalar route -- a stock llama-cli reference
build is being measured to settle it; (2) per-node cb_eval work serializing
on the scheduler thread while the compute pool idles; (3) contention in the
accountant/streamer atomics at high thread counts, which would also explain
the shallow I/O queues seen under batch (645 MB/s against a 6.7 GB/s drive).

SHIPPED NOW: --threads (-t), defaulting to the MEASURED optimum of 4 rather
than the core count, with the pathology in the code comment. Raising the
default to hardware concurrency would have shipped a 28x regression.

---

# Measured + shipped (2026-08-21): the VRAM floor is 1.3 GiB, and state moves off RAM

Owner question: can 1.5 GB of VRAM run this, for old laptops (GTX 950M
class)? MEASURED YES -- 1.3 GiB, with the desktop baseline subtracted
honestly (peak 3,555 MiB minus 2,259 MiB already in use):

  prefill chunk   compute buffer   ours (VRAM)   wall, 834 tok
  512             1,500 MiB        1.75 GiB      1m21s
  128             1,121 MiB        1.35 GiB      1m29s
  64              1,136 MiB        1.30 GiB      1m33s   (CPU: 7m19s)

The buffer does NOT scale linearly with the chunk -- it floors near 1.1 GiB
-- but small chunks cost ~15% wall for ~450 MiB, and chunk 64 still beats
CPU prefill 4.7x. --prefill-chunk (16..8192, default 512) ships so low-VRAM
cards can take the prefill win.

AND THE ACCOUNTING WAS WRONG, now fixed: under --gpu llama puts the KV
cache AND recurrent state in VRAM (log: "Vulkan0 KV buffer", "Vulkan0 RS
buffer") while our weights stay in the host arena. The RAM floor was still
reserving both, starving the weight cache of memory nothing was using. The
engine now RELEASES that reservation on GPU consent, says so loudly, and
re-reconciles -- measured on a 27B at 4 GiB/8k ctx: 0.32 GB handed back to
the cache budget. The disclosure names the remaining honesty gap: VRAM is
NOT bounded by --cap. That is the dual-cap Phase C shape appearing for
free; capping it is future work, stated as unmeasured.

---

# Measured (2026-08-21): GPU decode LOSES on this architecture, and the reason is exact

The owner asked what --gpu does for the compute-bound 27B. It works
(single-stream survives what the batch experiment crashed on), stays
polite (VRAM peak 3.9 GiB, 1.5 GiB compute buffer -- a 2-4 GiB budget
would hold it), produces coherent text, and is SLOWER: 2.6 s/token mean vs
2.0 on CPU, same cap, same prompt, same token count.

The mechanism is in the log: 852 graph splits (CPU: 1). Weights live in
OUR host arena by design, so every offloaded op drags its weights across
PCIe -- ~16 GiB per token for a dense model. That transfer costs more than
the compute it saves. The finding generalizes to both regimes: dense models
are compute-bound but pay per-token transfer; giant MoE models are I/O-bound
so GPU compute buys nothing. GPU offload of host-resident weights does not
pay for DECODE on either.

PREFILL, MEASURED, AND IT WINS DECISIVELY. Same model, same 20 GiB cap,
same 834-token prompt, -n 1: CPU 7m19s, GPU 1m21s -- 5.4x, with a VRAM
peak of 3,840 MiB (a 4 GiB budget holds it). The mechanism is the mirror
of the decode loss: a 512-token chunk pays the PCIe weight transfer ONCE
and does 512 tokens of compute with it, so the split overhead that sinks
decode is amortized into irrelevance while the GPU's arithmetic advantage
(~28 TFLOP per chunk against a couple hundred GFLOPS effective on 22
cores) lands in full.

THE ARCHITECTURE IS NOW SETTLED BY MEASUREMENT, not theory: weights stay
host-resident; DECODE stays on CPU (GPU loses 30%); PREFILL goes to the
GPU under a small explicit VRAM budget (GPU wins 5.4x). That is exactly
the dual-cap Phase A shape, vindicated in its proper scope after the batch
crash muddied it. Practical consequence: CPU prefill costs ~0.5 s per
PROMPT token on a dense 27B, so an 8k-token prompt is ~70 minutes of
waiting; on GPU it is ~13. Long-prompt work -- RAG, document analysis,
agentic context -- moves from impractical to routine.

---

# Measured (2026-08-21, owner-triggered): the 27B is COMPUTE-bound, not I/O-bound

The owner asked why a 27B was slow, observing it should not be. The answer
is that the published floor number was the wrong number to publish alone,
and the truth is better for the engine than the floor implied. Steady-state
decode, Qwen3.8-27B UD-Q4_K_XL, 4k ctx, --kv q4, 32 tokens:

  cap     streamed/token   steady s/tok   h_bytes
  20 GiB  0 B              1.5            97%   (fully resident: ZERO disk)
  12 GiB  4.74 GiB         1.8            57%
  8 GiB   8.74 GiB         2.0            35%
  4 GiB   12.93 GiB        ~3.9 (earlier) low   (the floor, published alone)

TWO conclusions, both worth more than the floor. (1) A 27B dense at Q4 on
this 22-core CPU costs ~1.5 s/token in PURE COMPUTE with the entire model
resident and no disk activity at all. That is the machine's floor for this
model class; no engine can go below it. The model is not slow because of
streaming. (2) The streaming is nearly FREE where compute can hide it:
halving residency (8 GiB, 8.74 GiB/token off disk) costs only 33% over the
fully-resident case, because reads overlap decode compute. The engine earns
its keep at the LOW end -- it turns "impossible" into "33% slower".

The reporting failure recorded: publishing 3.9 s/token at the 4 GiB floor
without the resident-cap number made a compute-bound model look
I/O-crippled. Dense models are compute-bound above modest caps; the giant
MoE flagships (K3, Qwen3.8 2.4T) are I/O-bound because ~30B active
parameters ride on a 594 GB file. Both numbers ship from now on.

---

# Correction + measured (2026-08-21): plan ignored --kv; and what KV quant buys the flagships

Self-inflicted, caught within the hour by the owner asking why the big
models had not used it: --kv reached engine_open but NOT cmd_plan, which
calls build_plan directly -- so `plan --kv q4` printed the f16 floor,
silently. The instant-truth command was lying about the exact feature that
changes its answer most. Fixed; the flag now reaches both paths.

With it fixed, what quantization actually buys each flagship (32k ctx,
plan arithmetic from each model's own tensor table):

  model         KV f16     KV q4      recurrent state (NOT quantizable)
  GLM-5.2       5.25 GiB   1.48 GiB   none -- pure attention KV
  Qwen3.8 2.4T  3.00 GiB   864 MiB    568 MiB fixed
  Kimi K3       975 MiB    274 MiB    443 MiB fixed

The lesson is architectural: --kv only touches the attention KV cache, so
the payoff is inverted from intuition. GLM-5.2, the SMALLEST flagship, wins
most (3.8 GiB saved at 32k) because its state is pure KV; K3, the largest,
wins least in absolute terms and keeps a 443 MiB recurrent floor no knob
can touch. Consequence for batching, where per-sequence state is the width
bound: q4 should widen GLM most of all -- MEASURED AND CONFIRMED. GLM-5.2
at 28 GiB, 44 distinct prompts, --kv q4: clean, zero failures, 2.400
GB/token-aggregate. That is 2.55x single-stream (6.12), against 1.93x at
the f16-bound width of 32 -- quantizing the cache bought 12 more sequences
AND a third off the per-token bill. The rule for the docs: on
attention-state models, quantize KV to buy batch width; on recurrent-state
models (K3, Qwen3.8 2.4T), only RAM buys width.

---

# Measured (2026-08-21, owner-triggered): --kv quantization, and 262k context in 11 GiB

The owner measured q4/q4 KV as near-lossless on Qwen3.8-27B and asked for
it. Shipped as --kv q4|q8|f16 (default f16, so every prior number stands;
difftest byte-identical). The floor is priced at what the run will actually
allocate -- the plan report names the cache type on its own kv line -- and
the engine sets llama's K/V types with flash attention, which a quantized V
cache requires. Quality is NOT generalized (Invariant 8): near-lossless is
the owner's measurement on THIS model; other models are unmeasured and the
docs say so.

ENGINE-VERIFIED ladders, correct text at every rung, Qwen3.8-27B UD-Q4_K_XL:

  context     f16 floor    q4 floor
  4k          4 GiB        4 GiB      (weights dominate; KV is noise here)
  32k         6 GiB        4 GiB      (1.9 GiB of KV becomes 612 MiB)
  128k        13 GiB       6 GiB
  262k        28 GiB       11 GiB     (the full native window, laptop RAM)

The headline the collection did not have before: a 27B with its ENTIRE
262,144-token native context in 11 GiB, at ~4 s/token, on one NVMe.

---

# Measured (2026-08-21): the sixth model is DENSE -- a 27B in 4 GiB, and RAM buys context

Qwen3.8-27B (UD-Q4_K_XL, 16.34 GiB, 64 layers: 48 Gated DeltaNet + 16
attention, NO experts) -- the arch-blind planner's first dense model, and
it classified all 866 tensors correctly on contact: 100% unconditional
bulk, zero routed, floor = norms + KV (68 KB/token from the 16 attention
layers) + a fixed 150 MiB DeltaNet state. ENGINE-VERIFIED context ladder,
correct text at every rung: 4k ctx -> 4 GiB; 32k -> 6 GiB; 128k -> 13 GiB;
native 262k -> 28 GiB. At the 4 GiB floor: ~3.9 s/token wall, 12.93
GiB/token -- and the dense scan rides the drive's SEQUENTIAL class (3+
GB/s realized vs ~1.3 for MoE gather), so a model that streams 79% of
itself per token still walks at whiteboard speed. Two engine improvements
purchased by the descent, both filed: plan feasibility does not model the
largest-whole or churn constraints (plan said 1 GiB; the engine's honest
floor is 4 -- the refusal machinery caught it, three times, pricing each
rung); and the refusal's --cap suggestion underestimates because it omits
its own churn-reserve growth. The 1-bit variant (UD-IQ1_S, 6.19 GB)
floors at 3 GiB with correct text: 3.33 GiB/token, ~1 s/token at
sequential rates -- a 27B on three gigabytes at a second per token, the
collection's on-ramp number.

---

# Measured (2026-08-21): Qwen3.8 single-stream at 8 GiB -- the chart's last gap

The flagship that never got its canonical wall number (its milestones were
correctness-shaped: the 5 GiB min-cap, the cross-platform seal). Standard
protocol, 8 GiB/512 ctx/24 tokens: wall 5m42s = 14.3 s/token including
prefill, 35.73 GiB/token -- the price of 508 GB run that deep below its
knee -- zero failures, cap held, coherent reasoning-model output. The
single-stream chart now reads: M3 ~4.8, GLM 8.6, Qwen 14.3, K3 16.1
s/token, all at 8 GiB, all on one protocol.

---

# Measured (2026-08-20 night): dual-cap Phase A graded -- a precise negative

The experiment ran on the resident RTX 4070 via a Vulkan build variant
(DRAY_VULKAN_BUILD opt-in, mirroring the Metal pattern; the default
build remains CPU-only and byte-identical -- our own CMake force-pin
caught the first attempt at sneaking the backend in, which is what it is
for). Facts: the backend engages (device found; testbed runs to coherent
text); on the M3 16G B=8 leg the scheduler split the graph 1,234 ways
(vs 1 on CPU) ping-ponging ops over host-resident weights, reserved a
2.2 GB Vulkan compute buffer, peaked at 6,962 MiB VRAM -- IMPOLITE by any
budget -- and crashed with an access violation: the streamer REPOINTS
tensor->data per node while the multi-backend sched assumes stable
storage for its copies. VERDICT: transient-upload offload conflicts
structurally with the repointing streamer; Phase A as designed is not the
cheap phase on this engine. The dual-cap design stands, but its first
buildable increment is now Phase C (KV/state pool in VRAM -- llama-owned
memory the streamer never touches, moving the batch width bound onto the
second budget) or a repoint-aware offload in the fork (pin the per-op
source before submit). Recorded with the experiment's artifacts; nothing
promised.

---

# Measured (2026-08-20 night): M3 batches -- the lower-budget answer, and three bugs paid

The owner's first night question answered in bytes. M3 DISTINCT-prompt
curves (4k ctx, decode-only): at 16 GiB -- B=1 4.048, B=8 2.924 (1.38x),
B=16 2.768 GB/token (1.46x, inside the warned band, clean); at 28 GiB --
B=16 2.263 GB/token (1.79x, clean); B=32 is past this cap's wall at 4k ctx
(budget after 32 sequences' state holds barely one layer's region trio,
nothing evictable -- the failure is measured, the warn band now covers it).
THE LEVER, MEASURED: at ctx 2048 the same B=32 runs CLEAN at 1.617
GB/token (2.50x) -- the machine's best absolute batch number. The wall was
per-sequence state; context is the dial that moves it. M3's multipliers are
modest exactly as its geometry predicts (k/E 0.031, heavy cache coverage of
a 143 GB model); its ABSOLUTE numbers are the machine's best: 2.26 GB/token
batched. Three bugs paid for the curves: (1) fork graph-node budget --
MSA reserve graphs scale with KV streams; width >= 16 died at context
creation (GGML_ASSERT obj_new); fixed with per-stream headroom, and the
same function's K3 branch turned out to be DEAD CODE (its custom budget
unconditionally clobbered by the else-chain -- every K3 run to date
survived on the generic budget by luck; now honored). (2) The committed
eslot pool under batch is budget that can never earn a hit -- at caps where
the unconditional set fully fits it strangled make_room (need 0.70, lru=421
unevictable); no width, no pool. (3) Batch eslot admission regated to
single-stream on the three-scale verdict: testbed -22%, K3 nil, M3 fatal; the
properly EVICTABLE pool is the filed fix.

---

# Design (2026-08-20, owner): the dual-cap machine -- a polite VRAM budget beside the RAM one

The owner's framing, which is exactly this project's: the user grants TWO
explicit budgets (--cap 16G --vram-cap 3G), both hard, both accounted, both
honest, and keeps using the computer. Design consequences, in value order:

PHASE A -- GPU compute for prefill, zero resident VRAM weights. The decisive
asymmetry: in our regime DECODE is disk-bound (bytes/token over drive
bandwidth), so GPU decode compute buys almost nothing below the knee -- but
PREFILL is compute-and-bandwidth heavy at large ne[1] and measured at 72% of
batch wall. llama's scheduler can run big-batch matmuls on GPU with
on-the-fly weight upload from our host arena (weights never resident in
VRAM; transient op buffers only). VRAM cost = scheduler reserve +
activations, bounded and measurable against the vram cap. This phase is
cheap, targets the measured bottleneck, and leaves the CPU/default path
byte-identical (GPU strictly opt-in; difftest gates it).

PHASE B (later) -- VRAM as a third cache tier for HOT experts. The skew
measurement motivates it (M3: top-10% of experts take 73% of picks; 2-4 GB
of hot experts could serve most routed hits at PCIe rates instead of NVMe).
But expert-tier residency forces per-layer split buffers, router-time
dispatch, and GPU-side FFN compute to avoid pointless copy-backs -- a real
research project, not a phase. Recorded, not promised.

PHASE C (middle) -- per-sequence KV/state in VRAM. Width is bound by state
(67% of the RAM budget at K3-32), so a VRAM state pool moves the batch width
bound onto the second budget and frees RAM for the weight cache. Attention
computes where KV lives. Moderate cost, measurable win for the batch story.

Invariants extend, not bend: a second Accountant for VRAM (every byte
through it, cap means cap), Invariant 2 untouched (weights still uncached
reads, never mmap), readout gains a VRAM ledger line, unmeasured renders
unknown. Tonight's experiment grades Phase A's premise on the resident RTX
4070 (8 GB): Vulkan build variant, M3 prefill CPU vs GPU-offload, measuring
prefill wall, VRAM peak, cap behaviour, and text coherence (GEMM-class
divergence expected and judged, not asserted).

---

# Measured (2026-08-20): MiniMax-M3 -- the fifth architecture, first light to correct text in one night

The owner's pre-post mandate, closed. The 143 GB UD-Q2_K_XL was fetched
byte-exact (142,966,273,184 B, matching the profile's verified total); the
arch-blind planner classified all 948 tensors correctly on first contact
(routed 133,857,017,856 B exact, the three slot tiers as documented, KV
122,880 B/token exact); the one load failure was the released conversion
LACKING five indexer hyperparameter keys the fork's loader requires --
closed by the new profile kv_defaults mechanism (values verified
field-by-field against the model's published config; generic --override-kv
CLI wins per key; malformed overrides fail the load loudly). First
generation: CORRECT TEXT (" Paris." and coherent continuation), exit 0,
RSS 6.70 of 8 GiB cap, ledger accounting 99.5% -- which also settles the
profile's flagged unknown: the MSA indexer cache lands in the unreserved
reconcile (1.36 GB with llama's buffers) and the cap holds. Measured at 8
GiB: 4.61 GiB/token, knee 8.71 GiB, ~4.8 s/token wall including load --
the lightest flagship per token. Routing skew: top-10% of experts take 73%
of picks. M3's sliceable-KV state class arrives with it. Night extras: the
compact ON/OFF gate passes on M3 (byte-identical text at 8 GiB), and the
minimum cap is 4 GiB -- measured by the admission's own pricing (3 GiB
refuses naming 4G; 4 GiB serves correct text), the refusal machinery
quoting its price exactly.

---

# Measured (2026-08-20): CI green five-for-five on its fourth run ever

The maiden campaign, one layer per run, each failure banked: run 1 --
environment rot (a Visual Studio generator pin the image had outgrown;
ubuntu-22.04 runners shutdown-signalled mid-job by retirement brownouts);
run 2 -- a resource bug (bare -j is UNLIMITED make parallelism; cc1plus
OOM-killed on 16 GB runners); run 3 -- a real portability defect (the
churn-reserve width formula used std::ceil/std::pow with <cmath> never
included; MSVC forgave it transitively, g++ did not -- the first off-MSVC
compile of the entire batch arc, failing exactly where the release audit
predicted); run 4 -- all five jobs green: windows, windows-preset (the
documented build path), macos (Apple silicon), linux, and linux-tsan,
whose clean pass is ThreadSanitizer's first-ever verdict on this engine.
Release blockers A1 (CI proven) and A2 (HEAD sealed cross-platform at
compile+suite level) both close. CI now guards every push.

---

# Decision (2026-08-20, owner): private and public repositories stay separate

The release-audit residue (an abbreviation inside an archived-evidence path
in one superseded blob; the pre-scrub comparison vocabulary in historical
blobs; two deleted transfer scripts naming a private VM address) is
ACCEPTED in the private repository, whose history -- including every
commit SHA that DECISIONS cites -- stays intact and unrewritten. The
public repository, when created, is born from a fresh export of the
current tree: no inherited blobs, the never-name rule met at the
strongest standard (nothing googlable, not merely nothing named) from its
first commit. No filter-repo rewrite ever runs on the private history.

---

# Consolidated (2026-08-19): throughput and the memory ledger, owner-framed

Two owner corrections worth recording as method. (1) GB/token is the COST;
the performance is tokens per unit time. Steady-state aggregate throughput,
derived as measured decode bytes/token over the measured realized batch
bandwidth (1.2-1.4 GB/s, labeled derivation): K3 B=32 ~4.6 s/token-agg
(~13 tok/min, from ~25 s/token single); Qwen B=16 ~4.1 s/token-agg; GLM
B=32 ~2.4 s/token-agg (~25 tok/min). Measured run walls (n=6,
prefill-heavy, understate steady state): K3 27.6 s/tok single, 12.6
s/tok-agg at B=32 with joint prefill. Realized bandwidth is ~20% of the
drive's 6.6 GB/s sequential ceiling, so every figure carries large
I/O-depth headroom. (2) The RAM at high width is SEQUENCE STATE, not
weight cache: at 4k ctx per-seq state is K3 0.59 GB (mostly fixed
recurrent, ctx-independent), Qwen 1.00 GB (same class), GLM 0.70 GB (true
KV, scales with ctx). At K3 B=32 that is 18.9 of 28 GiB -- 67% of the whole
budget -- leaving ~5-7 GB of cache after the floor, which is exactly where
the admission wall and the union-region warning live. Levers: smaller --ctx
helps GLM ~linearly and K3/Qwen only marginally (their recurrent floor does
not shrink); each extra GiB of cap buys ~1.7 K3 or ~1.4 GLM sequences.

---

# Shipped (2026-08-19): cohort rotation v1 -- run-to-completion cohorts, parking guarded

dray batch --rotate SPAN --state-dir DIR serves MORE prompts than the
funded width in cohorts, each running to completion before the next loads.
Opt-in; state-dir explicit (rotation writes; off-device is wise); writes
reported at MB/token amortized. Boundaries measured on testbed: cohort slot
recycling is DETERMINISTIC, 7/8 byte-identical to split unrotated
baselines, the eighth a single near-tie flip of the documented
deterministic layout class. The llama_state_seq park/restore roundtrip is
NONDETERMINISTIC (identical reruns diverge; set_data reports success), so
span < max_tokens is REFUSED with the defect named -- mid-generation
parking waits on a fork-side investigation of multi-stream state
serialization, now queued. Rotation is an isolated new engine function;
every existing path difftest/batchdiff-proven untouched.

---

# Measured (2026-08-19): all three flagships batched -- the machine's menu is complete

Qwen3.8 closes the trio (28G/4k, distinct, decode-split): B=1 19.972
GB/token; B=8 7.408 (2.70x); B=16 5.300 (3.77x) CLEAN -- the optimum; B=24
REFUSED AT LOAD, priced ('one batch-24 union region needs 3055 MiB, cache
has 348; try --cap 31G') -- the 1 GB/seq recurrent state is Qwen's width
tax, exactly as its geometry said. THE MEASURED MENU at a 28 GiB cap, one
protocol, no projections: K3 2.8T -- optimum B=32, 5.98 GB/token, 5.45x
(the multiplier king: sparsest routing, slowest union growth); Qwen3.8
2.4T -- optimum B=16, 5.30 GB/token, 3.77x (the middle on every axis);
GLM-5.2 744B -- optimum B=32, 3.17 GB/token, 1.93x (the absolute king: the
cap caches 71% of it, so little disk remains to share -- and little is
needed). The architecture law the trio demonstrates: batch multiplier
rises with routing sparsity and falls with cache coverage; absolute
throughput does the opposite. Every number decode-only, every refusal
priced, every width admitted or refused at load.

---

# Measured (2026-08-19): GLM-5.2 batch curve -- small multiplier, best absolute number

Same protocol as K3 (28G/4k, distinct prompts, decode-split), all widths
clean, no warnings: B=1 6.116 GB/token; B=8 4.445 (1.38x); B=16 3.687
(1.66x); B=32 3.172 (1.93x). The multiplier is the trio's smallest, as the
geometry said it would be -- dense routing (k/E ~0.05) saturates the union
fast, and the cap already absorbs ~71% of a 217 GB model, leaving less disk
to share. But the ABSOLUTE number wins: 3.17 GB/token at B=32 is the
fastest batch configuration measured on any model tonight, beating K3's
optimum nearly 2:1. Measured vs the cold projection (2.84x/4.77x at 8/32):
the cold multipliers do not survive contact with a warm cache -- the cache
eats the shareable stream first. Qwen3.8 queued on the same protocol.

---

# Measured (2026-08-19): joint batched prefill -- prefill bytes -40%, wall -21%

The 72%-of-wall item, measured on the crown config (B=32/28G distinct):
prefill 2963 -> 1767 GB, total wall 3074 -> 2415 s, decode untouched (6.15
vs 6.08 GB/token, noise). All 320 prompt tokens ride ONE shared chunk, but
the one-sweep ideal (~600 GB) is not reached: 1767 is almost exactly 3x one
sweep, the signature of the whole-tensor scan re-reading each 5.9 GB routed
tensor in windows because it exceeds the 5.25 GB cache remainder at this
cap. The residual is cap-dependent -- at caps where a routed tensor fits
the remainder, joint prefill should approach one true sweep; unmeasured,
labeled as such. Correctness: within-batch and compact ON/OFF identical,
texts byte-identical to sequential prefill on the gate model. Two traps on
the way in, both caught by gates: llama_get_logits_ith takes the batch
token index, not a dense logits ordinal (fail-fast crash, caught by
batchdiff VOID); and a pipeline exit code masked that VOID as a pass --
gates assert PRESENT pass strings, re-learned.

---

# Measured (2026-08-19): batch fusion lands harmless; the 4x wall hypothesis is FALSIFIED

The ne[1]<=n_seq gate generalization (98f469c) re-measured on the crown
config, B=32/28G distinct: decode bytes 6.083 GB/token vs 5.983 unfused
(noise), wall 2980 s vs 3074 (-3%). The predicted ~4x wall gain DID NOT
APPEAR at K3 scale, and the run says why: 72% of total bytes are PREFILL
(2963 of ~4130 GB) -- 32 sequential near-full-model scans that stay
correctly unfused per the Phase A hazard -- so the observed 1.2-1.4 GB/s
lived mostly in prefill, and decode fusion could only touch the remainder.
The testbed eslot win (-22% bytes) also does not transfer: a 28 GiB pool
holds no useful fraction of a 393-expert union, so cross-step reuse is
testbed-scale physics, not K3-scale. The gates stay because they are correct
and harmless (identity proven, single-stream byte-identical); the REAL
batch wall item is JOINT BATCHED PREFILL -- one shared scan instead of B
sequential ones, already recorded as future work, now with its cost
quantified: it is 72% of the run.

---

# Final (2026-08-19): the sweep closes -- 28 GiB optimum B=32 at 5.45x, wall priced at load

The complete distinct-prompt curve at 28 GiB/4k, decode-only: B=1 32.62
GB/token; B=8 10.95 (2.98x); B=16 8.18 (3.99x); B=20 7.39 (4.41x); B=32
5.98 (5.45x) -- monotone, all clean, zero failures. Above it, admission now
refuses AT LOAD: the working-set check (this commit) catches what the
output.weight check could not -- B=38 had passed with a 1.81 GB cache
budget and died mid-run on a 2.54 GB union region; it now refuses in a
minute, pricing the next width honestly ("working set needs 20187 MiB; try
--cap 46G"). THE OPERATIONAL ANSWER: at 28 GiB run batch 32, 5.45x the
bytes-throughput of single-stream; wider widths are a rerun away at the cap
the refusal message quotes. Wall figures still carry the fusion-under-batch
~4x headroom (next item). Nine distinct-prompt runs + four correlated, ~25
TB read total, two engine bugs and two accounting artifacts found, fixed,
and documented on the way to the number.

---

# Measured (2026-08-19): the churn fix dissolves the boundary -- B=32 clean at 5.45x

With the width-scaled churn reserve (5b2daf1), the widths the old reserve
refused now run clean at the ORIGINAL 28 GiB: B=20 at 7.392 GB/token
(4.41x), B=32 at 5.983 GB/token (5.45x), zero failures, no breach. The
distinct curve is monotone through 32 -- 2.98x, 3.99x, 4.41x, 5.45x -- so
the earlier "cap-bound optimum at B=16" conclusion is WITHDRAWN: the (16,20]
boundary was the single-sequence churn sizing, not capacity. What remains
cap-bound is admission itself (per-seq state funds ~46 at 28 GiB); a B=44
leg probes the curve at that wall. Cache contribution visible at B=32: step
191.5 GB against a 294 GB union cold bound.

---

# Final (2026-08-19): B=20 also refuses; the 28 GiB optimum is B=16 at 3.99x

The boundary probe: B=20 completed two clean steps (143.8 GB/step, trending
7.19 GB/token) then hit the same single materialise failure as 24 and 32 --
refused, flagged, nonzero. The pressure boundary at 28 GiB/4k ctx lies in
(16, 20]; the operational recommendation is B=16, 3.99x decode-bytes
efficiency on realistic mixed prompts, with over-width runs failing LOUDLY
rather than degrading. The sweep is complete: correlated ceiling 7.11x /
20.5x, distinct floor 2.98x / 3.99x, boundary mapped, optimum
cap-bound. Sweep total ~11 TB read across nine runs, all decode/prefill
split, all gates green or honestly failed.

---

# Measured (2026-08-19): the DISTINCT-prompt curve, and the real optimum at 28 GiB

The number the batching feature ships on -- 32 genuinely different prompts,
decorrelated routing, decode-only split accounting, K3 2.8T at 28 GiB/4k
ctx: B=1 32.62 GB/token; B=8 10.95 (2.98x, clean); B=16 8.18 GB/token =
3.99x, CLEAN -- the measured optimum; B=24 and B=32 both REFUSED with one
materialise failure under peak union pressure, flagged NOT TRUSTWORTHY and
exited nonzero exactly as the honesty machinery is built to do. B=24's
partial first step trended better (6.47 GB/token) before failing: the curve
is still improving where the cap gives out, so the optimum is CAP-BOUND,
not sharing-bound -- a larger cap moves both the boundary and the optimum
up. Correlated-ceiling reference points (identical prompts): B=8 7.11x,
B=32 20.5x -- the envelope brackets any real workload. Wall figures still
carry the ~4x fusion-under-batch headroom; bytes are the verdict.

---

# Correction (2026-08-19): the B=32 thrash NEVER EXISTED, and all figures are CEILINGS

The split B=32 run: prefill 2942.7 GB (32 sequential ~92 GB sweeps), decode
305.5 GB = 50.9 GB/step -- FAR UNDER the 294 GB cold bound -- and 1.591
GB/token aggregate, 20.5x against B=1. The "intra-step thrash" was entirely
the J7 prefill-in-numerator artifact; the earlier sentence defending it
("too large to be prefill accounting") was written without subtracting 32
prefills and stands corrected. B=32 is the BEST point measured, not a
regression. TWO caveats bound every batch figure in this file, both raised
by the owner from the running machine: (1) all runs used B IDENTICAL greedy
prompts, so router selections were fully correlated and the union collapsed
toward one sequence's -- these are the CORRELATED CEILING; real mixed
workloads decorrelate toward the independence model and a distinct-prompt
sweep measures that floor next. (2) Observed disk read during batch was
1.2-1.4 GB/s against the 6.6 calibrated ceiling: the decode overlap
machinery (fusion/early-unlock) is gated to single-token ids and turns OFF
under batch, so every wall figure carries ~4x headroom; extending fusion to
batched ids is the next engineering item. Bytes conclusions are unaffected
by (2).

---

# Decision (2026-08-19, owner): cohort rotation -- optional, write-honest, off-device

Direction ratified for the next batching phase: more logical sequences than
the cap holds, served by rotating cohorts -- decode a RAM-resident cohort
for a fixed span, park its per-sequence state to disk via the existing
snapshot machinery, load the next cohort. Constraints set by the owner and
quantified here: (1) OPT-IN ONLY (explicit --rotate + span; default engine
write behaviour unchanged; flag censused) because rotation introduces
sustained writes where the engine's premise has been read-dominance -- at
0.6 GB state and span 64 the cost is ~9.4 MB written per aggregate token,
which is ~9.4 TB per million-token job, a real TBW bite; span scales it
inversely. (2) --state-dir may target a DIFFERENT device, so the model
drive stays effectively read-only for life and a cheap scratch SSD absorbs
the writes. (3) The honesty contract extends to WRITES: bytes-written
reported per run with the same rigor as bytes-read, projected against span
before a long job commits. Design only -- nothing implemented; the B-sweep
finding the aggregate optimum runs as this is written.

---

# Measured (2026-08-19): K3 decode-only split -- batch-8 is 7.11x, not 3.06x

The re-measurement the J7 correction promised, with prefill and decode
reported separately (28G cap, 4k ctx, n=8, zero failures, no breach):
B=1 prefill 96.1 GB + decode 260.9 GB = 32.617 GB/token decode-only;
B=8 prefill 639.9 GB + decode 293.6 GB = 36.705 GB/step = 4.588 GB/token
aggregate. TRUE decode bytes-efficiency at batch 8: 32.617/4.588 = 7.11x --
the prefill-inclusive accounting had reported 3.06x and swallowed most of
the win. The step tells the mechanism plainly: eight sequences' expert
unions cost ~12% more bytes per step than one sequence's selections. The
visible next frontier is the PREFILL: 80 GB per sequence, sequential and
unshared (8 near-full sweeps at B=8); a batched joint prefill would amortize
it the same way and is recorded as future work, not promised.

---

# Correction (2026-08-19, J7): the K3 batch figures below are PREFILL-INCLUSIVE

The batch adversary caught the accounting: the benchmark divided total bytes
(B prefills included) by decode-only steps, inflating every per-step figure
by a term that GROWS with B. The 3.06x at B=8 is therefore a LOWER BOUND on
true decode bytes-efficiency, and part of the 3.06-vs-3.75 shortfall was
this artifact, not physics. cmd_batch now reports prefill and decode bytes
separately; a decode-only B=8 re-measurement replaces these figures when it
lands. The B=32 thrash finding stands (its 1.8x-over-cold-bound is too large
to be prefill accounting). Also corrected same-day, from the same adversary:
the cells guard defended a unified KV pool this build never creates (each
sequence owns a PRIVATE ring; the guard is per-sequence now, with ctx_wall
reported honestly); the union-feasibility column could not fire on the case
it was added for and double-spent the cap -- replaced with a disclosure
column and a measured-anchor footnote, because a fake-precise model is worse
than an honest "measure"; batch prefill gained per-sequence admission; the
cap verdict uses the LATCHED breach signal and prints CAP BREACH; the batch
loop rebinds the cap against RSS per step; batchdiff no longer fails open at
width 1 and greps for taint.

---

# Measured (2026-08-19): K3 batched at frontier scale, and the projection corrected

The number the feature exists for, on K3 2.8T / 594 GB / 28 GiB cap / 4k ctx,
zero failures, cap held, all through cmd_batch's one code path: B=1 reads
44.57 GB/token; B=8 reads 116.6 GB/step = 14.58 GB/token aggregate, a
measured 3.06x bytes-efficiency against the cold-bound prediction of 3.75x.
AND THE CORRECTION: B=32 REGRESSES to 16.91 GB/token (541 GB/step against a
294 GB cold bound) -- at that width the per-step expert-union working set
(~49% of all experts per layer) exceeds the cap and the cache thrashes
WITHIN a single step, evict-and-reread, the below-knee pathology compressed
into one decode. The projection's feasibility column counted only per-seq KV
state; it must ALSO fund the per-step union working set, and the plan table
carries that correction now as a modeling lesson (Invariant 8 both ways: the
measurement graded the projection and failed one column of it). At 28 GiB
the bytes-optimal batch is ~8, not 32. Wall (noisy, secondary): aggregate
~2x at B=8.

---

# Measured (2026-08-19): batched decode -- correct, and the sharing is real

Owner-mandated production batching, phases 1-4 landed together. The planner
funds B sequences (KV and recurrent state are per-sequence in the floor;
admission refuses what the cap cannot hold); `dray batch` decodes N
prompts in LOCKSTEP through one streamer; and the gates, all green on testbed
8G/512ctx/24tok: WITHIN-BATCH identity (B copies of a greedy prompt,
byte-identical outputs at B=2 and B=4 -- the [k,B] compaction paths three
audits called untested, now proven consistent under a differential);
cross-batch-size identity (B=2 text == B=4 text); compact ON vs OFF under
batch IDENTICAL (same kernels both legs, no numerics excuse -- the
batch-difftest, scripted as scripts/batchdiff.ps1). THE MEASUREMENT: batch 2
and batch 4 read IDENTICAL total bytes, 13.558 GB, 0.565 GB/step at both
widths -- per-token aggregate halves 0.282 -> 0.141 GB/tok. The
unconditional stream is fully shared and routed-union overlap cost nothing
measurable: the sharing thesis, deterministic and exact. One honest caveat,
documented not hidden: batch output diverges from the SINGLE-stream control
in the last bits (GEMV at B=1 vs GEMM at B>1, accumulation order, greedy
flips on near-ties) -- the same documented class as the repack difference,
batch-size-invariant, judged on coherence like difftest's reference half.
The frontier-scale measurement (K3, where the projection says 6x at B=32/28G)
remains queued: hours of drive time, needs the machine free.

---

# Measured (2026-08-19): the quarantine round-trip, end to end

The battery the process rule demanded, run before any pass seven: a job
driven to quarantine at ceiling 1 (durable counter through hard kills), the
poll answering "quarantined" ACROSS a restart, DELETE removing the
.job.failed AND discarding the pinned blob (1 -> 0 on disk, the I4 hash
wiring proven), 404 after, and one more restart answering 404 -- the
resurrection loop pass six confirmed is measured dead. testbed 8G,
exit-verified drains throughout.

---

# Sixth pass (2026-08-19, I1-I6 + M1/M2): the remediation needed remediating

Six confirmed, all inside the pass-5 delta itself -- the synthesis's verdict
that "the remediation has a higher defect density per line changed than the
code it remediated" is recorded here verbatim because it is the important
sentence. I1 (critical): H15 added a backend_dead flag and NEVER SET IT --
the teardown DMA guard was disarmed in the always-free direction, the exact
opposite of its stated property; one line fixes it, and the lesson is that a
fix which adds a flag nobody sets retires the very FATAL that would have
reported the gap. I2: H14 widened bytes() and not name() one commit after H9
unified them -- both now share ONE traversal and TWO class-correct predicates
(routed experts are whole only under NO_COMPACT; token_embd under either
lever), so the pair structurally cannot drift. I3: the spent-budget test now
runs BEFORE the pending decode, and the cancel path never checkpoints a
pending token past the client's budget. I4: quarantine records carry their
hash, DELETE reaches .job.failed, and the reaper knows the two new terminal
words. I5: a TRANSIENT deferral hands its durably-counted attempt back the
same durable way -- the engine never ran, so nothing was attempted. I6:
permanently-unusable blobs are unlinked BY PATH (discard-by-index was a
guaranteed no-op after load de-indexed). M1/M2: the two remaining knobs
validate like their siblings. Pass counts: 20, 61, 39, 10, 21, 6.

PROCESS RULE, adopted: no further review pass until the lifecycle battery
covers the paths that pass touched. The battery, not the next reviewer, is
the regression gate for this subsystem -- the calibration measured WHY
(reviewers miss quiet arithmetic at 3/10) and passes 4-6 demonstrated it
(every delta seeded its successor's findings).

---

# Measured (2026-08-18): the job lifecycle, exercised end to end at last

The executable coverage pass five demanded, run as a three-leg battery on
testbed 8G. LEG A (reset-on-progress): one 400-token greedy job through THREE
graceful restarts -- the sidecar showed attempts=0 after every one (tokens
11 -> 67 -> 131 across stops), completed 400/400, never approached
quarantine; yesterday's code counted 1, 2, 3 and would have quarantined a
healthy job on its fourth restart. LEG C (permanent-unusable): a byte
corrupted inside the snapshot blob -> "checkpoint unusable" logged, restart
from prompt, completed with a clean un-duplicated prefix. LEG B
(no-progress bound): hard kills with checkpointing suppressed climbed the
DURABLY persisted counter 1 -> 2 -> quarantined exactly at the ceiling; the
one residual -- the quarantine poll record died with its process -- is
closed in the same commit (.job.failed files register at scan, so the
operator's poll survives restarts).

---

# Fifth pass (2026-08-18, H1-H21): severity converged, the lifecycle had not

Zero blind-confirmed criticals -- the first pass to clear that bar -- but 21
findings, 11 of them in the job checkpoint/resume lifecycle, which had zero
executable coverage and had been patched by review for three passes: exactly
the quiet-arithmetic class the calibration below proved reviewers miss. All
21 are fixed in the H-commit (this one): TRANSIENT honoured by its caller
instead of only its author; the unusable-checkpoint hash discarded instead
of parked (which had been swallowing the restarted run's first real
checkpoint and burning the time trigger); attempts decided before the
publish that persists them, reset with all five progress counters, written
durably at scan, bounded by a validated ceiling; the quarantine branch
registers a validated id with the counter seeded; the resume clamp is exact
(room+1), a spent budget is a clean finish, and the stop exit outranks the
clamp; hits and misses of a routed tensor land in one class on both paths;
name and size of the largest whole read use one filter; OOM on restore is
transient; whole-mode respects NO_ROWSLICE; teardown leaks only on a
genuinely dead backend; the Host gate folds case like its neighbour; the
cancel 415 that broke protocol-legal SDK calls is gone; and difftest
compares VERBATIM and case-sensitively -- the collapsing compare had been
hiding exactly the trailing-newline class the calibration flagged, and the
strengthened gate still reads IDENTICAL, which is now a stronger sentence
than it was yesterday. Correction per G9's own rule: the earlier header's
"52-agent" figure was a run-record number, not an artifact number; the
artifact enumerates its findings and this file now cites only those. The
calibration triage landed in 5ac5974 and f841bcf (an earlier note credited
712bccf, which carried the pass-4b live set).

---

# The skeptics, measured (2026-08-18): recall 7/10 on seeded defects

The review pipeline was calibrated the only way this project accepts: ten
defects seeded into a COPY of the tree, each modeled on a real bug class
from this repository's history, ground truth held outside the copy, the
SAME lens prompts run blind. RECALL 7/10. Found: the flattering hit-rate
denominator, the settle_tags DMA reversion (filed critical, twice,
independently), the fabricated zero, the inverted ledger gate, the
alignment-class confusion, fsync deleted under a comment claiming it, and
the keyless Host-gate skip. MISSED, all three: the resume off-by-one, the
lock-scope move, the stop-boundary >= to > -- every silent one-character
change with no contradicting comment or name/content mismatch to smell.
The measured shape: these reviewers catch LIES and MISMATCHES reliably and
miss QUIET ARITHMETIC; the compensating control for the missed class is
executable tests and byte-differential gates, which is what the difftest
already is. Severity calibration: one underclassification (alignment filed
minor), two defensible escalations. The calibration lenses also filed ~34
non-seed findings; the ones describing real-tree code were triaged into the
pass-4b live set (commit 712bccf).

---

# Fourth pass (2026-08-18, G1-G10): not converged, and the miss was measured

The bounded fourth adversary over the F-diff confirmed 10 findings (1
critical, 6 major, 3 minor) -- the loop had NOT converged, and the critical
was the fix session's own: F4 sized its allowance from an empty context (16
bytes, "0 MiB reserved" printed in the battery logs unread) and exempted
every checkpoint blob from the ledger. Six of seven root causes were
HALF-LANDED fixes: the right mechanism at one call site, missing at its
sibling -- the failure shape this file already names. All ten are fixed and
gated (commit 1fd89b0); the byte-identity measurement below survives G3/G4
because the battery's stop fired mid-loop, not at the resume boundary --
which is exactly the difference between a measurement and a proof of
absence. Pass confirmations: 20, 61, 39, 10.

---

# Measured (2026-08-18): resume identity holds through TWO graceful interruptions

The crash-resume flagship claim now has its strongest possible demonstration.
One 300-token greedy background job was interrupted twice by graceful
shutdown (/admin/shutdown -> cancel-all -> final safepoint -> drain), with
the sidecar inspected between stops: absolute tokens_done climbed 17 -> 81
across sessions (F5), the attempts counter incremented (F15), and the
twice-resumed completion was BYTE-IDENTICAL to an uninterrupted control over
1,326 characters (F5+F6+F8+T1 jointly). A stop sequence persisted in the
sidecar fired AFTER a resume and trimmed at the exact boundary (F7). testbed
8G, CKPT_EVERY=8, exit-verified drains. This is the measurement behind
README's "greedy jobs resume with text identical to an uninterrupted run".

---

# Third adversarial pass (2026-08-18, F1-F16): the release sweep's own fixes audited

The off-repo report (final-adversary-2026-08-18.md) enumerates F1-F16, two of
them critical, plus its filed minors. (G9/H11: two earlier versions of this
sentence quoted counts the artifact does not contain -- in the paragraph
written to fix exactly that habit, twice.)
The two criticals: F1 -- src/models/ was swallowed by a day-one gitignore
pattern and had NEVER been committed, so no fresh clone ever configured on
any platform (fixed; a real clone-to-tmp configure now passes, the first in
the repository's history); F2 -- the T23 fix deleted a loop that was
secretly read_batch's in-flight barrier, reintroducing a free-under-DMA the
previous sweep had refuted BECAUSE that loop existed (fixed: unconditional
settle before discard, loud deliberate leak on a dead backend). F3-F16 are
fixed in the F-commit trail (shutdown race term, checkpoint floor allowance,
absolute resume counts, cancel-path text completeness, stop persistence and
delivery, compaction-decline id restore, admin auth/Host/CT gates, prefill
progress frames, created on all chunks, durable DELETE + bounded resume
attempts, vendored licence texts + THIRD_PARTY.md).

KNOWN RESIDUALS, open by name: partial-stop holdback across pieces (a stop
prefix already streamed before the match cannot be unsent); the ne[1]>1
compaction gates remain ctest-unreachable (multi-token ids exist in tests
only at ne[1]==1; difftest's real prefill covers the path per commit); the
expert-cache secondary failure returns still hold pins (the module is dead
code per A6 and its deletion is the recorded intent); T14's httplib bump and
T17's testability refactor stay deferred as before. The header below
overstated closure and stands corrected by this one; its "every actionable
T-finding closed" claim was written by the fixer, which is the lesson this
file keeps teaching.

---

# Release-sweep fixes complete (2026-08-18): every actionable T-finding closed

Count reconciliation, recorded once so three numbers stop competing: the two
review passes filed 81 RAW findings, which dedupe to 59 DISTINCT root causes
(the "59 of 61" phrasing below counted two later-split clusters separately);
the release sweep then confirmed 52 of its own, T1-T28 after merging. The
review reports stay off-repo because they quote the prior-art project this
repository must never name; the IN-REPO audit trail is the commit history --
every fix commit names its finding (grep the log for Rd, Sd+, Td+), and
this file records the campaign arcs. D1 (the "no fork repo exists" decision)
and the ranked open-concerns block are both SUPERSEDED where they conflict
with newer sections: the fork exists (gitmodules names it), CI has five jobs
(their first runs await the public remote), and C1/V1/V3/P1/P2 were retired
by the measured sections above them.

T1-T14, T16, T18-T28 are fixed and gated (commit trail below this file's
history). Two items are DEFERRED BY DESIGN, not forgotten: the wholesale
httplib upgrade (T14; its unbounded-chunked-body hole is closed by a 411 at
pre-routing, and the bump gets its own battery-gated pass), and the
server-into-library testability refactor (T17; structural, zero behaviour
change, wrong to rush against a release). The limiter verdict exists, cap
breaches are visible live, shutdown preserves checkpoints, and the tree
carries no secrets. This header supersedes the hold verdict below.

---

# Correction (2026-08-18, same day): the release sweep falsified the closure claim

A third swarm pass over the post-campaign tree confirmed the fixes were NOT
all sound: 12 of the 81 were partial, vacuous, or introduced regressions --
including the worst finding in the project's history, T1: four individually
correct fixes jointly making graceful shutdown DELETE the checkpoint it
existed to protect. The section below stands as written, because corrections
are recorded here, never rewritten -- the release-sweep report and the T-fix
commits are the current truth, and the lesson is this header: a campaign that
verifies its own fixes needs a fresh adversary, not its own checklist.

---

# Swarm campaign closed (2026-08-18): 59 of 61 sweep findings resolved

Of the 81 confirmed findings across both swarm passes: 57 fixed and gated
(see the commit trail), S38 resolved by design (excising the reference's
dead branches would contradict its deliberate byte-freeze; the deadness is
documented in place and the stale co-maintenance rule is retired), and S20 -- the
last one -- closed the same day: make_gguf_split writes two real GGUF shards (full KV
block copied via gguf_set_kv so the shards cannot drift, split.no/count/tensors.count,
llama naming), and test_two_shard proves the plan spans both files at verified offsets
AND that the streamer, asked for a second-shard expert, returns bytes only that file
could contain. ALL 81 CONFIRMED FINDINGS ARE RESOLVED.

---

# Status sweep (2026-08-18, swarm S32/S33)

The "Open concerns (ranked)" block and several per-section status lines below predate
the last three days of work and are SUPERSEDED where they conflict with newer sections
above them. Newest sections are prepended, so read this file top-down: the first
statement about a topic is the current one. Specifically retired: V2 ("snapshot store
unreachable" -- it ships as the crash-resume mechanism; its buffered I/O is now tracked
as swarm R12), C3, A1-A3, R3/R4, G3/G4 (all overtaken by the server, persistence, and
swarm-fix sections above). The swarm review reports live outside the repo; their
confirmed findings and fix commits are recorded in the sections above.

---

# The compaction bug, and the harness that found it (2026-08-13)

**Both large models now generate coherent text at a 12 GiB cap.** Qwen3.8: *" Paris. The
capital of Germany is Berlin. The capital of Italy is Rome."* K3: *" Paris. The capital of
France is Paris."* Before the fix they produced `!!!!` and `The The.F/The}`.

**The bug.** `compact_experts` takes its ids from `node->src[2]`. We overwrite that src
with a private remapped tensor and never restore it, and llama.cpp reuses graphs across
decode steps — so from the second token on, the mapping was derived from our OWN
already-remapped ids. The router asks for experts 5, 12, 33; we remap to 0, 1, 2; next
token we read 0, 1, 2 as the router's choice. First token correct, every later token
compounding. Fixed by recording the graph's original ids per node.

**Why it survived.** Every check looked at something that was genuinely correct: failure
counters (zero), floor integrity (passing), `verify` (2,868 slices byte-identical — of a
*different* implementation), a `read_batch` check added while hunting this (matches: we
read exactly the experts we asked for, having asked for the wrong ones), an ids readback
(self-consistent), and `test_compaction.cpp` (passes, with a negative control — but it
builds one ids tensor and never runs a second token, which is the only place this appears).

Three confident wrong diagnoses preceded the right one: copied `op`/`src` fields, strided
vs contiguous ids layout, and shared `priv_ids` state. Each was rebuilt and each left the
bug in place. The changes are kept where they are better hygiene; none was the fault.

**What actually found it: a reference.** `--no-stream` loads through llama.cpp normally.
On the 1B-7B testbed (4 GB, runs both ways) the two paths differ in exactly one variable, so any
divergence is ours — and iteration drops from ~4 minutes to seconds. The bug fell out in a
handful of runs after a day of reasoning backwards from garbage output. **Models that fit
in RAM are useless as a target and indispensable as a control**; conflating those two cost
most of a day.

**Known limit of the harness.** llama.cpp assigns quantised weights to `CPU_REPACK`, which
interleaves them at load and dispatches a different matmul kernel — 192 tensors on testbed.
We cannot use it: repacking needs the whole tensor resident at load. So the two paths run
different kernels and diverge in the last bits, which flips a greedy token choice around
step 15. Both paths are individually deterministic. The pass criterion is therefore
"coherent and semantically equivalent", not bit-identical, unless repack is disabled in
the reference build.

---

# Correction: the 4 GiB result was wrong (2026-08-13)

**Claimed:** Qwen3.8 UD-IQ1_S generating correct text at a 4 GiB total resident cap.
**Actual:** at 4 GiB the cache gets 0.18 GB and the run dies materialising a 1.09 GiB
embedding. The claim was measured while the ledger counted almost none of what the
process held.

Four things were uncounted or mismeasured, all fixed now: per-read I/O staging, cache
slot allocations, llama.cpp's own buffers, and the check read working set rather than
commit charge. Peak commit on that "successful" 12 GiB run was 13.59 GB against a
12.88 GB cap.

Per-category ledger at a 4 GB cap, which is what the claim should have been checked
against:

| item | GB | |
|---|---|---|
| router gates | 1.44 | 36% of the cap, mandatory-resident by design |
| recurrent state | 0.55 | mandatory |
| compute scratch | 0.14 | **under-reserved**: llama.cpp actually allocates 0.576 |
| KV + norms + staging | 0.09 | |
| unreserved | ~1.90 | llama.cpp buffers, vocab, model metadata |
| **total** | **4.12** | 0.18 left for cache |

**Measured minimum after the fixes below: 5 GiB.** 4 GiB is refused with an actionable
message; 5 GiB runs with 1.25 GB of cache, 6 GiB with 2.32, 8 GiB with 4.47, 12 GiB with
8.74 — all producing correct text, all within cap on peak commit.

Two structural facts this exposes:

- **`token_embd.weight` (1.09 GiB) was materialised whole to read one row.** FIXED: rows
  are contiguous and strided by `nb[1]` exactly as experts are by `nb[2]`, so the same
  compaction applies. Measured at 5 GiB over 2 tokens: 155.235 → 151.802 GB, a 3.433 GB
  saving against 3 forward passes × 1.14 GB = 3.42 GB predicted.
- **`output.weight` (1.09 GiB) genuinely needs every row** to produce logits, so it is a
  hard lower bound on cache size unless the final projection is chunked.

The lesson is the one that keeps repeating: **a counter that only counts what it knows
about cannot detect what it never knew about.** Zero reported failures and fluent output
were true and meaningless. The check that would have caught this is comparing the ledger
against the OS, which is now done every token.

---

# TIER 2 SHIPPED (2026-08-16 night, 8f5f353): correct Metal compute over streamed weights

Working, measured, honest: DRAY_METAL=1 on a GGML_METAL build maps the pool
arena into Metal no-copy, llama offloads all layers, the full streaming
pipeline (cb_eval, repointing, ring) runs unchanged, and the text is
CPU-IDENTICAL in whole-tensor mode (forced, loudly -- Metal MUL_MAT_ID
assumes canonical strides and reads compacted nb[2] layouts at the wrong
stride: "Takd" vs correct; an upstreamable kernel patch would lift this).
Default builds/runs bit-identical, gated 7/7 + difftest both platforms.

Speed, honestly: engine prefill metal 3.13 s vs cpu 2.95 s at 441 tokens --
NO gain through our engine yet, and the mechanism is named: cb_eval gates
materialisation per node, so the sched splits per node and Metal pays encode
overhead per command buffer, forfeiting the whole-graph 4.1x that llama-cli
proves this hardware has (600.3 vs 147.4 t/s). Recovering it = windowed
materialisation (materialise a window of nodes ahead, let Metal encode the
window) -- the next Apple session's single design problem. Also noted: ctx
2048 at cap 3G aborts BOTH backends (pre-existing scratch/batch limit, not
Tier 2; reproduce with a 1761-token prompt).

---

# Tier 2 blueprint: Metal over the pool arena, source-validated (2026-08-16)

Prerequisite SHIPPED: PoolArena (17b67e1) puts every hot-path allocation in
one page-aligned VA range; testbed carries 2.22 GB committed, 0 fallbacks.

Pin-verified mechanisms (third_party/llama.cpp, exact SHA): (1)
ggml_metal_buffer_map(dev, ptr, size, max) wraps external host memory no-copy,
exposed as the device standard .buffer_from_host_ptr; (2) it chunks the range
into MTLBuffers with containment resolution, so tensor->data REPOINTED inside
the mapped range resolves without re-registration -- our per-token repointing
survives untouched.

The integration, refined after the buft survey (2026-08-16 night): every
pointer compute can meet now lives in THE pool range (regions and ring were
already in via alloc_acct; scratch and emergency routed in tonight, raw-host
fallback kept, frees origin-routed; fake_base stays raw BECAUSE it is never
executed-from -- tensors are repointed before any node runs). TWO TRAPS for
the implementing session, found by reading, cheaper than crashing: (1) do NOT
return a real Metal buffer from dray_buft_alloc -- ggml_backend_buffer_clear
would memset the whole mapping and commit the entire 2x-cap reservation; keep
our custom iface/fake_base buffer and register the arena mapping SEPARATELY
with the Metal device. (2) sched routes by buffer type, so the tensors' buft
must read as Metal's mapped type while keeping our iface -- the exact seam to
resolve on a machine that can run the result. Mac-day confirmations required before believing it: (a) the
lid-open Metal probe (remote/clamshell sessions wedge -- measured), (b)
repointed-resolution smoke, (c) difftest vs the CPU build for kernel-numerics
drift, (d) one-node-per-batch behaviour under the Metal sched. Until a Mac can
run (a), this stays a blueprint, deliberately unwritten past the prerequisite:
code that cannot even be compile-tested is risk pretending to be progress.

RESOLVED (2026-08-16, owner console mailbox): METAL WORKS AND ALWAYS DID.
Measured on the M1 Air, testbed, ~600-token prompt, single turn:
Metal 600.3 t/s vs CPU 147.4 t/s prefill = 4.1x ON MOE PREFILL (MUL_MAT_ID on
GPU) -- against the +13% BLAS ceiling. Tier 2's payoff is real and measured.
The "hang" was never Metal: this pin's REWRITTEN llama-cli (cli_context/ui::)
runs a chat UI that waits at user_turn::read_input on a prompt nobody can see
(stack-sampled), ignores -no-cnv for exit purposes, spams its spinner through
fflush (the gigabytes, the warm core, the false "it computes" reads), and
writes output to /dev/tty PAST every redirect (the zero-byte files). Batch use
needs: -st, a pty (script -q), and locale-aware number parsing (600,3 = 600.3
under tr_TR). Four wrong theories died before the stack sample named the truth;
the lesson is the oldest one here: sample the process, never theorize heat.

The (a) investigation, one night of it (2026-08-16): llama-cli -ngl 99 on the
M1 Air over ssh never finishes MODEL LOAD. Three theories killed by
experiment: NOT conversation-mode EOF looping (-no-cnv valid in the pin, same
hang); NOT runtime Metal-kernel compilation (rebuilt with
GGML_METAL_EMBED_LIBRARY=ON, xcrun present, same hang, 1h25m and 8.6 GB of
spinner output); NOT memory pressure (-ngl 10 = ~600 MB wired, same hang,
swap calm). Also learned: a WARM machine proves nothing here -- the progress
SPINNER thread pegs a core while the loader blocks, so heat reads as work.
One theory left standing: Metal initialization blocks without a console
session / awake display. Discriminator: the same command in the Mac's own
Terminal; if it hangs even there, this M1/macOS-26/pin combination cannot run
this Metal build at all and the blueprint validates on some other Mac.

---

# Mac closure campaign (final Mac access, everything banked)

| item | result |
|---|---|
| snaptest on macOS | ROUND-TRIP IDENTICAL, stamp refusal correct |
| crash-resume on macOS | killed @120/200, resumed @ckpt-120, completed 200 |
| completion cleanup | verified (finished job deletes its sidecar) |
| internal-SSD ceiling | 3.48 GB/s gather (the good-drive Mac projection) |
| alignment discovery | F_NOCACHE + st_blksize 4096, threadpool QD 64 |
| iq1_s Metal kernels | present in vendored tree (static check) |
| CI | macos-latest (Apple silicon) job added -- Mac coverage forever |
| Metal MoE prefill multiple | INCONCLUSIVE: llama-cli -ngl 99 wedges on this M1 in a remote/clamshell session (Metal device acquisition stall, a known headless-macOS class); CPU legs ran fine both attempts. Kernels statically present; the multiple needs a Mac with its lid open. RE-CONFIRMED via clean scp-delivered script (no shell mangling possible): the -ngl 99 leg still wedges indefinitely while trivial ssh commands succeed -- the stall is real platform behavior in this remote session, not tooling. |

---

# Mac measured (2026-08-15, purged, honest)

| number | value |
|---|---|
| link ceiling (our calibrator, cold cache) | 0.40 GB/s |
| GLM-5.2 @ 3 GiB decode | ~64 s/token, ~26 GB/token |
| implied realized | 0.41 GB/s = 100% of the link |
| F_NOCACHE retention | 0.69 GB held vs ~145 GB streamed: CLEAN |

The engine saturates what the USB drive physically offers, the page-cache
claim is measured not assumed, and the text stays correct. On a faster
enclosure or the internal SSD the same engine simply inherits the better
link -- that is the whole design, demonstrated on a fanless laptop.

---

# macOS measurement protocol (2026-08-15, learned the honest way)

F_NOCACHE is NOT O_DIRECT: it stops NEW page-cache retention but neither
evicts existing pages nor bypasses them on read. Writing a model onto the Mac
leaves it (partially) cache-resident, and the calibrator then reported 28 GB/s
-- RAM speed, 30x the USB link, an impossible and therefore invalid number.
PROTOCOL: on macOS, purge the cache (sudo purge, or reboot) between writing a
model and measuring it; treat any ceiling above the physical link as
contamination, never as good news. Invariant 2 note: no-RETAIN holds on
macOS reads; no-CONSUME of pre-existing residency does not exist there.
Mac timing therefore still open: purged calibrate + decode-only rates.

---

# Apple demo banked (2026-08-15 evening): 744B on a MacBook Air in 3 GiB

GLM-5.2 UD-IQ1_S on an M1 Air (16 GB, fanless), streaming from a SanDisk
Extreme over USB, HFS+, 3 GiB cap: correct text (" the city of Paris"),
exit 0, all 1,809 tensors, first try. Wall 285 s for load+prefill+4 tokens --
a demo figure, NOT a measurement; the Mac timing session (SSD ceiling
calibration, decode-only rates, F_NOCACHE hygiene) remains future work.
The model lives at /Volumes/Extreme SSD/models/ (6 shards verified).

---

# Platform matrix complete (2026-08-15): three platforms, two ISAs, one engine

| platform | ISA | status |
|---|---|---|
| Windows 11 | x86_64 | measured (the reference numbers) |
| Ubuntu 26.04 | x86_64 | measured, Linux-on-Hyper-V labeled |
| macOS 26.4 (M1 Air) | arm64 | correctness sealed: gate IDENTICAL, 23,350 nodes matching both others |

The macOS gate ran the F_NOCACHE thread pool -- written as that platform's
permanent backend before ever touching it -- on NEON kernels, at a 2 GiB cap,
on a 16 GB machine with the owner's applications open. Zero new source errors
on Apple clang 21 (the gcc strictness tax covered it). The environment lives
entirely under /tmp (4.4 GB used of a 64 GB allowance): portable cmake/ninja,
source, model -- nothing installed user- or system-scoped. arm64 performance
awaits nothing but interest; correctness awaits nothing at all.

---

# Native Linux (2026-08-15): MEASURED, with the two bugs only Linux could find

The VM's heavier process floor reached what Windows arithmetic never could:
output.weight failed mid-run and the slot-sized fast-load poison scratch let a
plain MUL_MAT read 40x past it (SIGSEGV, gdb-backtraced). Two fixes, both
gated IDENTICAL and both improving Windows: poison_for() guarantees the poison
contract at every tensor size (debt 17 CLOSED, verified before/after on the
same trigger), and rebudget now DEMOTES static pins when the allowance shrinks
under them (3.27 GB pinned of 2.39 allowed was unevictable by construction).

Linux-on-Hyper-V table (11 vCPU, in-VM ceiling 6.33 GB/s gather, balloon
warmed, fixed VHDX on the same SN850X):

| point | result |
|---|---|
| GLM-5.2 @ 3 GiB | correct text (minimum cap reproduced natively) |
| GLM-5.2 @ 8 GiB | 19.3 s/tok |
| K3 @ 8 GiB | 20.2 s/tok, trustworthy, coherent (was: segfault) |

Windows equivalents are faster (16.1 K3 @ 8 GiB) on 2x the cores without the
virtualization tax; the Linux numbers are labeled indicative, correctness is
absolute. Cross-platform greedy text can diverge at knife-edge logits (MSVC vs
gcc kernels differ in last bits -- the known difftest caveat, now observed
cross-platform at GLM 3 GiB); within-platform gates are the correctness bar.

(previous section:) correctness sealed on real ext4 + O_DIRECT

Ubuntu 26.04 VM (Hyper-V, 11 vCPU, dynamic RAM to 32 GB, fixed 1 TB VHDX on the
same SN850X), g++ 15.2, native build: 6/6 unit tests, testbed compact ON vs OFF
IDENTICAL. Node count matches Windows exactly (23,350); bytes differ by ~0.7%
(10.95 vs 11.03 GB) because the slot-pool budget derives from process RSS/commit,
which is platform-real, not platform-identical -- the pre-eslot engine matched to
the digit. Timing and calibration wait for a quiet disk (transfer in flight);
all Linux performance numbers will be labeled Linux-on-Hyper-V.

---

# Queue (2026-08-15 morning, supersedes everything below where they conflict)

**Done in the overnight autonomy session** (each commit gated; see git log from
7522625 onward): the 5d5de2f order-invariant regression found, bisected,
fixed-by-deletion, and REVERIFIED at 8/28 GiB; slot pool landed lawfully (pool
line item + storm guard + doorkeeper; warms slowly by design); OpenAI-compatible
server SHIPPED and smoke-tested (serve subcommand, SSE, serialized admission,
honest /health); CI workflow; ARM64 presets (marked untested); K3 minimum cap
MEASURED = 6 GiB; GLM-5.2 minimum MEASURED = 3 GiB (beats the 4 GB stretch);
install-time repack v1 built, gated, and measured NEUTRAL on this drive (kept,
documented, companions deleted); snapshot_cache test reconciled -- suite 6/6;
licence recommendation prepared (MIT, owner decides).

**Next, in order:**
1. Long-generation steady-state: one multi-hundred-token K3/GLM job -- measures
   the slot pool warm curve and h_routed at steady state (the product use case).
2. cmd_run <-> server engine consolidation (flagged in both files; daylight task).
3. Persistence wiring: snapshot store/load integration test on testbed, then jobs
   with checkpoint/resume surfaced through the server ids.
4. DONE 2026-08-15 ~06:15: verify/stream references now uncached and independent
   (read_reference_uncached; 192 slices 0 mismatches; difftest IDENTICAL).
5. MUL_MAT poison-path hardening; llama.cpp allocation interception.
6. Prefill wave partitioning; thermal/endurance counters.

**Apple acceleration (designed 2026-08-15, awaits a Mac session):** prefill is the
target (decode stays disk-bound). Tier 1: GGML_BLAS+Accelerate routes prefill GEMMs to
AMX -- build flag + counting BLAS dequant scratch under the cap. Tier 2: Metal via
zero-copy no-copy buffers over our PAGE-ALIGNED ARENAS -- the single-allocator
invariant means every repointed tensor resolves to (arena buffer, offset), which is
exactly the shape Metal needs; verify one-node batching and IQ1 kernel coverage.
CPU-only remains the v1 policy; this is the v2 Apple story.

MEASURED 2026-08-15 (M1 Air, testbed, 511-token prefill, warm, load-subtracted):
CPU 5.4 s vs BLAS/AMX 4.7 s -- +13%, NOT several-fold, and the reason is
structural: ggml routes only plain MUL_MAT through BLAS, never MUL_MAT_ID, and
on an MoE model the expert matmuls ARE the prefill. Tier 1 is therefore cheap
but marginal FOR OUR MODEL CLASS; the several-fold intuition was a dense-model
number. Real Apple prefill gains live in Tier 2 (Metal has MUL_MAT_ID
kernels), behind the region-pooling refactor. BLAS compact gate: IDENTICAL.
Unreserved-growth check inconclusive (report line not captured) -- rerun with
the accountant report if Tier 1 is ever revisited.

**Blocked on hardware/owner:** macOS backend (machine); io_uring (real Linux);
ARM64 validation (machine); winrun repo name + publish (owner); the post (owner
voice; all numbers ready); licence decision (owner); M3 upstream reconversion;
Qwen UD-Q1_0 re-download when the q1 branch lands.

---

# Repack verdict (2026-08-15): correct everywhere, neutral on this drive

Paired same-n runs, text identity required: GLM-5.2 16 GiB n=32 -- 6.5 s/tok and
251.095 GB, IDENTICAL with and without the companion; K3 8 GiB n=32 -- 16.8 s/tok
and 1,798.73 GB, IDENTICAL. The pipeline (ring + fused/early expert batches at
depth) already saturates what this NVMe needs, so converting request shape buys
nothing here. The feature ships correct and gated (--repack DIR; whole-tensor
fallback refused loudly under a companion) and is documented as neutral on this
hardware -- request-overhead-dominated drives (QLC, DRAM-less, USB) are where it
would earn its disk, and that claim awaits such a drive, per invariant 8.
Companions deleted (derived; one command regenerates).

---

# Measured minimum caps (2026-08-14): the memory targets, settled

Probed by ladder, 4 tokens each, correct text required; refusals carry the
actionable message naming a cap that works.

| model | total params | checkpoint | MINIMUM cap | refused at |
|---|---|---|---|---|
| GLM-5.2   | 744B | 202 GiB | **3 GiB** | 2 GiB |
| Qwen3.8   | 2.4T | 508 GB  | **5 GiB** | 4 GiB |
| Kimi K3   | 2.8T | 594 GB  | **6 GiB** | 5 GiB |

Against the owner targets (must <8 / 6 very good / 4 happy stretch): all three
clear "must" with room; K3 lands exactly on "very good"; GLM beats the happy
stretch at 3 GiB -- a 744B frontier model under the polite default cap. The
floors differ for structural reasons the ledger names: K3 carries 2.20 GB of
router gates plus 0.63 GB of recurrent state; GLM carries neither at that
scale.

---

# GLM-5.2: fourth architecture, zero model code (2026-08-14)

unsloth/GLM-5.2-GGUF UD-IQ1_S, 202 GiB, 76 MoE layers, 256 experts top-8. `plan`
read everything from the tensor table; the registry was never touched; correct text
on the first execution. Unconditional bulk is 14.06 GiB because Unsloth's dynamic
method keeps attention high-precision -- which is why 1-bit output stays coherent.

64 tokens, Balanced mode, peaks sampled externally:

| cap | s/token | peak | bytes/token |
|---|---|---|---|
| 8 GiB  | 8.6 | 8.03 GB  | 15.8 GB |
| 16 GiB | **5.8** | 16.07 GB | 7.0 GB (knee at 15.04 GiB, plan predicted 5.85) |

Routing skew (share of decode picks on top-10% experts; uniform ~10%): testbed 36%,
K3 75%, **GLM-5.2 88%** -- the frequency-kept expert cache is evidence-backed on
every model measured, and it is the next lever: at 16 GiB nearly all remaining
traffic is routed experts.

The K3 56 GiB diagnosis is corrected by counters: all uncond resident by pass 2
(0 eligible misses); near-knee traffic is ~9.7 GB/token of routed experts + the
~8 s compute floor. Top-of-dial gains belong to the expert cache, not the ring.

---

# Measured head-to-head (2026-08-14)

Same workstation (22 cores, 79.7 GB RAM, WD_BLACK SN850X PCIe 4.0 x4, calibrated
6.73 GB/s gather at 1 MiB x QD16), same Balanced power mode, same prompt, 64 generated
tokens, Kimi K3 both sides. Peak memory sampled externally every 10 s on both. The
reference engine runs through a ~750-line Windows I/O port (published) whose primitives
are identical to ours; its dense-set prefetch ring is held at 1 slot at this
configuration by its own budget rule.

|                    | dray            | reference engine     |
|--------------------|--------------------|----------------------|
| s/token (64, incl prefill) | **15.9** (16.1, 16.2 on reruns; 16.1 REVERIFIED on HEAD 7522625 after the order-invariant fix) | 33.73 |
| peak committed (sampled) | **8.12 GB** (8 GiB cap enforced) | 8.45 GB |
| bytes/token        | 54.5 GB            | 136.8 GB             |
| checkpoint on disk | **594 GB** (1-bit) | 1,563 GB (4-bit)     |
| realized GB/s (avg)| 3.43               | **4.06**             |

**The honest decomposition:** the reference engine realizes BETTER average bandwidth
than we do — its packed sequential dense-set layout is well engineered. We are 2.12x faster only
because the 1-bit checkpoint moves 2.51x fewer bytes. On a disk-bound workload,
checkpoint size is speed. Quantisation quality differs (1-bit vs 4-bit) and is
disclosed, not measured here. Both engines produced correct, on-topic text for the
same prompt; ours ran at lower sampled peak memory.

**Second tier — the reference engine at its BEST documented configuration**
(double-buffered ring, read overlap active, 10 of 93 layers pinned, 13.16 GB dense
set + 10 GB expert cache) against ours at an equal cap:

|                    | dray @ 28 GiB   | reference @ best cfg |
|--------------------|--------------------|----------------------|
| s/token (64, incl prefill) | **14.0**   | 29.49                |
| peak committed (sampled) | 28.06 GB     | 29.37 GB             |
| bytes/token        | 34.2 GB            | (dense set+experts, overlap on) |

**The ratio is scale-invariant: 2.12x at 8 GB, 2.11x at ~29 GB — with their prefetch
crippled in neither the second tier.** Extra memory helps both engines differently:
ours eliminates reads (22.6 GB pinned → 54.5 → 34.2 GB/token, entering compute-bound
territory around 14 s), theirs overlaps reads it still performs (33.73 → 29.49).

Evidence archived off-repo (run logs, the engine's own run-report JSON,
tokenizer/config files). With both tiers measured, the 1.45 TB reference checkpoint
is no longer needed.

Our own pipeline work the same day: 25.3 → 15.9 s/token (−37%) from fused per-layer
expert reads + a FIFO byte-ring streaming the unconditional set in pass-1 consumption
order. Remaining measured headroom to the 8.5 s/token drive floor: ring-vs-pin split,
expert unlock at router release, install-time repacking.

---

# Requirements (binding)

Non-negotiable, in the owner's words where it matters.

**Purpose**
1. **Run on very narrow RAM. The single most important point. Not negotiable.**
2. **Never refuse a model for not fitting.** Not fitting is the input, not an error.
3. Models that already fit in RAM are irrelevant — do not validate against them.
4. No excuses about llama.cpp not supporting something: use ggml, else fork, else write
   from scratch. Do not panic-hack it either.
5. Must actually RUN the models, not benchmark them.
6. Seconds per token is fine. Being as efficient as possible is not, so that users know
   this is as fast as it gets.

**Engineering**
7. C++20 + CMake. Cross-platform: mac, linux, windows, amd64, arm.
    **Release gate (owner, 2026-08-14): the engine is not released publicly until
    Linux and macOS support exist.** It will never be Windows-only in public.
8. **Must not consume or dirty the OS page cache**; must not trigger paging.
9. Must reach the drive's full throughput.
10. Very small project. Modular: adding a model is additive plus one or two lines.
11. Support every quantisation of a model where it costs no extra complexity.
12. No size threshold for ignoring tensors — compute per model.
13. Default cap 4 GiB, polite, never auto-expanding.
    **Memory targets (owner, 2026-08-14): everything must run below 8 GB. 6 GB is
    very good. 4 GB with frontier models is the happy stretch result — not a gate.**
    The polite 4 GiB default stays; each model's measured minimum cap is stated,
    never implied.

**Discipline**
14. **No patches. The final implementation must be correct, not "it somehow works."**
15. Verify properly. A single-token run is not verification.
16. External feedback is not bible — evaluate it.
17. Carry no obsolete content in docs.
18. Report the drive, the measured per-class bandwidth, and bytes-read-per-token. A
    benchmark that names the CPU but not the disk is the failure mode to avoid.
19. Credit Unsloth prominently; never imply we did the quantisation.
20. Never name the prior-art project anywhere in this repository.

---

# Open concerns (ranked)

**Correctness — unresolved**
- **C1. K3 emits degenerate text** ("The the of the"). Not a regression (Qwen3.8 is fine on
  the same build), not the floor (self-check passes), no reported failures. Untested
  candidates: our compaction against K3's KDA/MLA paths, an incomplete open-PR
  implementation, or genuine 1-bit degradation.
- **C2. self_check is weaker than it was.** It now re-reads through the same uncached path
  it is checking, to honour the page-cache invariant. A bug inside `read_batch` would be
  invisible to it.
- **C3. The snapshot cache is unwired and untested.** Its test suite is deliberately
  skipped because implementation and test were written against different contracts.

**Invariant violations — known**
- **V1. The cap does not bind on total resident bytes.** Staging buffers are never
  accounted at all, and the RSS rebudget fires once at load — before the cache fills — so
  it cannot catch the overshoot that appears during generation. Measured 13.57 GB private
  bytes against a 12.88 GB cap.
- **V2. `snapshot_cache.cpp` uses buffered `fopen`/`fread`**, which would dirty the page
  cache. Not yet reachable (unwired), but it is there.
- **V3. `verify` and `stream` read their references buffered.** Opt-in diagnostics rather
  than the hot path, but they still touch the page cache.

**Performance**
- **P1. ~1.5 GB/s against a calibrated 6.63.** Whole-tensor reads are one request each and
  serialised with compute; only expert slices are batched. The fix is structural, see P2.
- **P2. Prefetch is built but loses, and the design is wrong rather than the idea.**
  Measured on Qwen3.8 at 12 GiB, 4 tokens: batched-only 206.3 GB; prefetch without eviction
  208.8 GB (304/304 hits); prefetch with eviction 211.5 GB (673/673 hits). A perfect hit
  rate that still costs bytes, because every byte it holds comes out of `static_budget` and
  the +5.2 GB matches the displaced pinning re-read across 4 tokens exactly.
  **Do not conclude that prefetching cannot pay.** It competes with the cache only because
  it allocates per tensor from the same pool. A fixed small **ring** (one slot computing,
  one reading) with a dedicated reader thread costs a constant and never touches pinning;
  that is the shape to build. Prior art on the same access pattern reports 1.70x from
  exactly that structure, beating four times the memory without overlap.
- **P4. Measurement discipline.** Wall time varies ~10% run to run here — the same config
  gave 31.5 and 34.9 s/tok. Bytes-read is deterministic and is the only metric that has
  resolved anything. Timing was quoted as evidence once in this session and should not have
  been.
- **P3. The readout's counters are not fed from the streamer** — `h_routed` and `h_bytes`
  print "unknown", and fraction-of-ceiling is never computed from real traffic.

**Coverage**
- **G1. M3 cannot load at all.** Its published GGUFs predate llama.cpp's MiniMax Sparse
  Attention and lack the four indexer weight tensors per layer that it requires. Stock
  llama.cpp fails identically. Needs an upstream reconversion.
- **G2. K3 support is an open upstream PR**, not merged.
- **G3. The 4 GiB cap is verified on Qwen3.8 only.**
- **G4. Only the Windows I/O backend exists.** Linux and macOS are stubs; nothing has been
  built or run on either, nor on ARM.

**Architecture specified but not built**
- **A1.** Job API for multi-day runs. **A2.** Crash resume and checkpointing.
  **A3.** Snapshot cache wiring. **A4.** A separate prefill policy. **A5.** Thermal and
  endurance counters. **A6.** `expert_cache.cpp` is now dead code, superseded by the
  streamer, and should go.

**Process risk**
- **R1.** The llama.cpp fork carries two loader patches on top of an open PR branch; every
  upstream move is a rebase.
- **R2.** We depend on `ggml-backend-impl.h`, a private header, and will break on churn.
- **R3.** No CI. **R4.** No licence chosen.

---

# Correctness posture (2026-08-13)

**Six of the seven bugs in the streaming backend were the same mistake**: assuming a
ggml/llama.cpp interface's semantics instead of reading the source. `tensor_buft_overrides`
is a regex not a glob; the loader writes straight into `tensor->data` for host buffers;
the eval callback's return value controls *batching*, so returning false makes the next
materialise evict a node still in flight; the MoE ids tensor is a strided VIEW over an
argsort result; short reads at EOF are legal; and an optional tensor absent from metadata
must be reported missing rather than invented.

**Every one of them produced fluent output rather than an error.** Two reached the user
labelled as working. The habits that actually caught them, in order of value:

1. **Print the faulting node** (`DRAY_TRACE`). Found the strided-view bug in one run
   after three wrong diagnoses reasoned from plausible mechanisms.
2. **Name the failure reason, not just the count.** "90 failures" made "no room" and
   "read" equally likely; the tensor names said *shard boundary* immediately.
3. **A negative control in every differential test.** `verify` compared 2,868 slices and
   missed the EOF bug because it *skipped* spans exceeding the file — it passed by not
   looking where the bug was.
4. **Make the wrong thing impossible, not merely fixed.** Rewriting the shared ids in
   place required knowing every consumer of them; three were found at three different
   indexed axes before the design changed to a private ids tensor, after which the
   remaining consumers became correct without any code knowing they exist.

Guard now in place: `Streamer::self_check()` re-reads a sample of resident floor tensors
through a buffered path and compares them against memory, and `run` refuses to generate
on mismatch. A counter that only counts what it knows went wrong cannot detect what it
never knew about.

---

# Measured findings (this machine, 2026-08-12)

**Drive calibration — WD_BLACK SN850X 4 TB, PCIe 4.0 x4.** Measured by
`dray calibrate`, 2.59 GiB read in 0.53 s.

| class | ceiling | at |
|---|---|---|
| stream (sequential extents) | **6.82 GB/s** | 2 MiB x QD 16 |
| gather (8 MiB extents, random offsets) | **6.63 GB/s** | 1 MiB x QD 16 |

Two results worth carrying into the design notes:

1. **Large-block gather is ~97% of stream on this drive**, not the ~40% the
   literature's "2-5 GB/s under MoE load vs 12.4 GB/s sequential rating" implied.
   That field figure was probably measured at smaller extents or shallower queues.
   Our own extents are 5.4-10 MB (real M3 expert slices), which is squarely in the
   regime where the penalty has already vanished. The doc's insistence on never
   using a vendor sequential rating still stands; the *size* of the gather penalty
   on modern NVMe at these extents looks much smaller than assumed.
2. **QD 1 collapses to 0.50-3.42 GB/s** across every block size — a 2-13x loss.
   Deep queues are doing the work, exactly as designed, and this is the single
   strongest argument against any mmap-based path (a page fault is QD 1 per thread).

Caveat the tool prints itself: the gather extent size (8 MiB) is assumed, not taken
from the model. Re-run once the scheduler feeds it real per-layer slot sizes.

**Expert slice verification — PASS on all three models.** `dray verify` reads real
expert slices through the uncached IOCP path and compares them byte-for-byte against
a buffered reference at the same offsets.

| model | slices compared | mismatches |
|---|---|---|
| M3 UD-Q2_K_XL shard 2 | 684 | 0 |
| K3 UD-IQ1_S shard 3 | 1,080 | 0 |
| Qwen3.8 UD-IQ1_S shard 3 | 1,104 | 0 |

**2,868 slices, zero mismatches**, across three architectures with 128 / 896 / 512
experts per layer and different quant types. This validates the planner's tensor
offsets, the per-tensor strides (gate/up/down differ), the alignment widening and
interior slicing, and the Windows `FILE_FLAG_NO_BUFFERING` + IOCP backend.

**Streaming hot path — RUNS on the real models.** `dray stream` drives the actual
decode path (slab, LRU, refcounts, multi-shard uncached reads, remap) with the matmul
omitted, and spot-checks cached slot contents against the file.

| model | shards | layers | fetches | read | GB/token | slot checks |
|---|---|---|---|---|---|---|
| M3 UD-Q2_K_XL | 4 | 57 | 912 | 15.93 GB | **3.98** | 4, 0 wrong |
| K3 UD-IQ1_S | 14 | 92 | 2,914 | 19.28 GB | **9.64** | 2, 0 wrong |

Both land within ~5% of the predicted cold figures (4.18 and 9.50 GB/token), which is
independent confirmation that the whole cost model is sound.

**But the engine is the limit, not the drive: ~0.79–0.89 GB/s against a calibrated
6.63 GB/s gather ceiling — about 12–13%.** That is the fraction-of-ceiling metric doing
exactly its job on its first real use. The cause is structural and known: `ensure()` is
synchronous per layer, so effective queue depth is k×3 (12 on M3) rather than the 64 the
drive wants, there is no cross-layer pipelining, and every slice allocates and frees a
staging buffer. **The prefetch scheduler is now the single highest-value piece of work**,
and its payoff is bounded and knowable: up to ~7× on these numbers.

**Bug found by this run.** `ensure()` submitted raw GGUF offsets. Uncached reads need
sector-aligned offset *and* length, but GGUF aligns tensor data to `general.alignment`
(32 by default), so every submit was rejected: 912 misses, zero bytes read. Slices are
now read into an aligned staging buffer covering the widened range and their interior
copied into the slot — which is what the `io_staging` floor line was always paying for.

**FIXED — ExpertCache now addresses sharded models.** It previously held a single
`io::FileId` while every target ships 4-14 shards and `TensorInfo::offset` is
within-shard, so it would have read the right offset in the wrong file, silently.
`TensorInfo` now carries a shard index, `Plan` carries the shard path list, and the
cache opens one handle per shard. Original description: It holds a single
`io::FileId`, but every target ships 4–14 shards and a layer's tensors can live in a
different shard from its neighbours. `plan::TensorInfo::offset` is within-shard, so
the cache would read the right offset in the wrong file — silently, producing garbage
weights. `verify` sidesteps this by testing one shard at a time, which is why it
passes. Fix: carry a shard index on `TensorInfo`, a path list on `Plan`, and open one
`FileId` per shard. Mechanical but not yet done, and it blocks any real streaming run.

**Planner cross-check.** On the real 4-shard M3 the planner independently reproduced
the separately-measured tensor split byte-for-byte: routed 133,857,017,856 B, router
gates 179,306,496 B, gates+norms+bulk 9,100,952,064 B, and exactly the three
non-uniform slot classes 18,137,088 / 20,938,752 / 24,477,696 B.

---

# Decisions log

Reversible calls made without you, with reasoning and how to reverse. Newest last.
Design rationale and verified measurements live in project memory; this file is only
the record of choices made during unattended work.

---

### D1 — llama.cpp pinned at `84e908c625fb60992b4cdef8180fb12fa9b4c4bf`
Vendored at `third_party/llama.cpp` as a plain clone (not yet a submodule — no fork
repo exists to point one at). This is the SHA the state-API verification was done
against, so `llama_memory_hybrid::state_write` coverage is known-good here.
**Reverse:** `git -C third_party/llama.cpp checkout <other-sha>`, then rebuild.

### D2 — `LLAMA_BUILD_TOOLS=OFF` in our CMakeLists
At this SHA the unified `llama` app links `llama-server-impl`, so building tools
with the server disabled fails at link. Our engine needs only `llama` + `ggml`.
The reference CLI is built separately in `build/llamacpp` with `LLAMA_BUILD_SERVER=ON`.
**Reverse:** flip both flags on in the top-level `CMakeLists.txt`.

### D3 — llama.cpp built as shared libraries (upstream default)
`llama.dll`, `ggml*.dll` sit next to the exe. Left as-is rather than forcing static,
because forcing static on Windows also means rebuilding the CPU backend variants.
**Reverse:** `-DBUILD_SHARED_LIBS=OFF` and rebuild. Note the exe then needs no DLLs
beside it, which is nicer for distribution — worth doing before any release.

### D4 — Bare `G` suffix on `--cap` means GiB, not GB
"64 GB of RAM" almost always means 64 GiB, and the doc flags the GiB/GB ambiguity as
a trap. The readout prints both units so the choice is never load-bearing.
**Reverse:** `parse_size()` in `src/main.cpp`.

### D5 — Expert slot layout: three back-to-back regions per layer
A layer's routed experts live in three fused GGUF tensors (`ffn_{gate,up,down}_exps`),
each shaped `[.., .., n_expert]`, and `ggml_mul_mat_id` indexes dim 2 by a uniform
stride from one base pointer. So each of the three needs its own uniformly-strided
arena; one expert is three non-adjacent slices, not one block. A slab slot therefore
holds gate|up|down back to back, and slot index *i* means the same expert in all three.
**Consequence caught during implementation:** the three strides DIFFER (on M3
UD-Q2_K_XL gate/up are IQ2_XS at 5,455,872 B, down is IQ3_XXS at 7,225,344 B), so
deriving a stride by dividing the slot evenly reads wrong ranges and returns garbage
weights silently. Strides are stored per layer per tensor.
**Reverse:** `src/cache/expert_cache.{h,cpp}`.

### D6 — Uniform slots-per-layer distribution
The cache budget is divided evenly across MoE layers rather than skew-weighted.
Expert access follows deterministic layer order, so every MoE layer is touched
exactly once per token and none inherently deserves more slots. Skew-aware
distribution is downstream of the first utilisation trace, which does not exist yet.
Floor of `n_expert_used` slots per layer, or a single token cannot be served.
**Reverse:** `ExpertCache` constructor.

### D7 — Read failures fail the token, they do not degrade it
A short read or `EIO` marks the slot invalid and returns `Remap::complete = false`.
Callers must refuse to compute rather than feed a partially filled slot to
`mul_mat_id`, which would produce plausible text from the wrong weights — the same
class of silent corruption the checkpoint compatibility stamp exists to prevent.
**Reverse:** `ExpertCache::ensure()`.

### D8b — Per-token expert streaming is NOT wired into llama.cpp's graph tonight
Scoped, not attempted. What tonight delivers instead: the full streaming machinery
(planner, slab, uncached backend, remap, cache) built and tested, plus a proof
harness that reads real M3 expert slices from disk into slab slots and validates
`mul_mat_id` against a directly-read reference. What it does not deliver: llama.cpp
computing a real forward pass out of our slab.

The integration point is identified and is better than expected.
`llm_graph_context::build_moe_ffn` (src/llama-graph.cpp:1915) already takes
`gate_exps` / `up_exps` / `down_exps` as parameters **and** takes
`selected_experts_in` — so a caller can supply both slab-backed expert tensors
(`ne[2] = n_slots`) and a pre-remapped selection of slot indices. That is exactly
the shape `cache::Remap` produces. The remaining work is substituting those
arguments at the model-build site, where `hparams.n_expert` and the tensor's
`ne[2]` must be allowed to differ.

Not attempted tonight because it is real surgery in a codebase newly met, with a
throttled CPU making each compile-debug cycle expensive, and a wrong version fails
*silently* — plausible text from the wrong weights. Better done deliberately.
**Reverse:** nothing to reverse; this is work not yet started.

Note: `third_party/llama.cpp` carries its own `CLAUDE.md` pointing at `AGENTS.md`.
Not binding while we only link it, but it becomes binding the moment we patch it.

### D8 — Storage interface has no `cancel()`
The portable macOS floor is a pool of blocking `pread` and genuinely cannot cancel.
Rather than expose an operation one backend cannot honour, the interface forbids it,
which makes it the cache's job never to evict a slot with a read outstanding
(enforced by refcount in `LayerCache::acquire`).
**Reverse:** would require adding cancellation to every backend, including one that
cannot implement it. Prefer keeping the restriction.
