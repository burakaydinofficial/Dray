# dray

Run frontier-scale MoE models from an SSD, on a polite slice of RAM.

Kimi K3 is 2.8T parameters; GLM-5.2 is 744B. At Unsloth's dynamic 1-bit they are 594 GB
and 202 GB on disk, which fits a cheap NVMe drive and no amount of RAM you own. dray
keeps a small mandatory floor resident, leaves everything else on disk, and streams what
the forward pass actually asks for.

It is slow, honestly, and provably at the hardware limit. For K3 the honest unit is
**seconds per token**. That is the deal, and it is stated up front rather than
discovered: the tool measures your drive, reports its realized bandwidth per traffic
class, and tells you whether the engine or the disk is the limit.

## When NOT to use this

If the model fits in your RAM, use llama.cpp. It is FASTER than we are there,
measured on this machine with load excluded from both sides (our windowed
steady-state against llama-bench generation):

| model | dray resident | stock llama.cpp | stock is |
|---|---|---|---|
| Qwen3.6 35B-A3B | 7.5 tok/s | 9.1 | 21% faster |
| Qwen3.5 122B-A10B | 2.7 tok/s | 3.35 | 24% faster |
| Qwen3.8-27B dense | 1.2 tok/s | 1.75 | 46% faster |

An earlier version of this file called that parity. It is not parity, and the
claim was made by waving at load overhead instead of measuring the windowed
rate. When the model fits, you gain a memory cap you do not need and pay 20 to
45% for it.

This engine earns its keep on ONE thing: models that do not fit. A 594 GB
checkpoint under an 8 GiB cap is not slower elsewhere, it is impossible
elsewhere. When the model does not fit, the streaming path costs roughly 2x
what resident allocation would (a thread-pool barrier per intercepted graph
node, which is the price of repointing weights mid-forward-pass), and that
penalty disappears into disk time as soon as a configuration streams more than
a few GB per token -- K3 at ~16 s/token spends about 2 of those seconds
computing.

The rule: does it fit? Use llama.cpp. Does it not fit? That is what this is.

## A surprise worth knowing before you buy a faster drive

For sparse models the DISK IS FREE. Measured on Qwen3.5-122B-A10B with the
streaming path pinned in both legs: at a 12 GiB cap it reads 872 MiB per token
and runs at 2.7 s/token; at 56 GiB it reads NOTHING and runs at 2.7 s/token.
Identical. The 35B-A3B behaves the same way (479 MiB/token at 6 GiB, zero at 28
GiB, 1.6 s/token both). The drive is entirely hidden behind compute.

So the cost model this project is built on -- misses x bytes / bandwidth -- is
right in shape for Kimi K3 and wrong for anything moving under a gigabyte per
token. Below some byte rate the engine is the limit and the SSD is idle
capacity you already paid for. If your model is in that regime, a faster drive
buys nothing.

AND ON THE LARGE MODELS THE DRIVE IS NOT THE LIMIT EITHER, which is worse.
Calibrated on Kimi K3's own fourteen shards this NVMe serves 2 MiB random reads
at 6.03 GB/s. The engine achieves 1.8. The reason, measured 2026-08-24: the I/O
path only runs between compute nodes, because the thread that submits and
harvests reads is the thread that computes. Of a 121 second run about 91
seconds has nobody polling the completion queue. So a faster drive buys nothing
here either, until that is fixed. It is written up in DECISIONS with the
instrumentation to judge any attempt.

## Measured status (2026-08-23)

Every number below is from DECISIONS.md, the lab notebook this repository
carries; where a figure has a producing commit recorded there, the notebook
names it.

| model | disk | cap | decode | bytes/token |
|---|---|---|---|---|
| Qwen3.6 35B-A3B | 21 GB | **3 GiB** | 1.9 s/token | 575 MiB |
| Qwen3.6 35B-A3B | 21 GB | 8 GiB | 1.7 s/token | 415 MiB |
| Qwen3.6 35B-A3B | 21 GB | 28 GiB | 5.7 tok/s | 0 (resident) |
| Qwen3.5 122B-A10B | 39 GB | **3 GiB** | 2.6 s/token | 2.50 GiB |
| Qwen3.5 122B-A10B | 39 GB | 12 GiB | 2.3 s/token | 872 MiB |
| Qwen3.5 122B-A10B | 39 GB | 28 GiB | 2.6 s/token | 360 MiB |
| DeepSeek V4 Flash 284B | 90 GB | **3 GiB** | 4.6 s/token | 5.80 GiB |
| DeepSeek V4 Flash 284B | 90 GB | 8 GiB | 3.9 s/token | 1.93 GiB |
| DeepSeek V4 Flash 284B | 90 GB | 24 GiB | 3.8 s/token | 1.56 GiB |
| Qwen3.8-27B dense | 16 GB | **4 GiB** | 3.3 s/token | 12.74 GiB |
| Qwen3.8-27B dense | 16 GB | 8 GiB | 2.6 s/token | 8.74 GiB |
| Qwen3.8-27B dense | 16 GB | 20 GiB | 1.2 tok/s | 0 (resident) |
| MiniMax-M3 429B | 143 GB | 8 GiB | 6.8 s/token | 4.77 GiB |
| MiniMax-M3 429B | 143 GB | 28 GiB | 5.3 s/token | 3.30 GiB |
| GLM-5.2 744B | 217 GB | 8 GiB | 8.6 s/token | 12.80 GiB |
| GLM-5.2 744B | 217 GB | 28 GiB | 8.3 s/token | 5.45 GiB |
| Kimi K3 2.8T | 594 GB | **9 GiB** | 25.8 s/token (22.7 over 32 tokens) | 46.22 GiB |
| Kimi K3 2.8T | 594 GB | 28 GiB | 24.3 s/token | 27.22 GiB |

Bold caps are the measured minimum that still produces correct text. All rows:
4k context, CPU only, `--kv q4` (K3 uses f16, the only setting it accepts), 32
generated tokens, one methodology and one build. Reproduce with
`scripts/matrix.ps1`; every cell including 32k context and GPU is in
`scripts/matrix-results.csv`. Earlier versions of this table mixed figures taken
from different builds and prompts, which is why several rows moved.

**Prefill is a different workload** and gets its own column. 6,594-token prompt,
tokens per second:

| model | cap | CPU | with `--gpu` |
|---|---|---|---|
| Qwen3.6 35B-A3B | 8 GiB | 12.8 | **343.4** |
| Qwen3.6 35B-A3B | 28 GiB | 15.3 | **180.7** |
| Qwen3.5 122B-A10B | 12 GiB | 4.0 | **280.6** |
| Qwen3.5 122B-A10B | 28 GiB | 4.0 | **289.2** |
| Qwen3.8-27B dense | 8 GiB | 2.5 | REFUSED |
| DeepSeek V4 Flash 284B | 24 GiB | 2.6 | REFUSED |

A REFUSED cell means the engine would not admit that configuration and said so
at load with the cap it needed, not that it went unmeasured. At 32k context the
GPU path needs more than the dense 27B or DeepSeek were given here, which is
the same floor-raising effect described below.

GPU decode LOSES on every model measured, in all fifteen comparable pairs, so
`--gpu` is worth flipping for prompt-heavy work and not otherwise. It also
RAISES the minimum cap, because the Vulkan compute buffer sits inside your
budget: the 35B needs 4 GiB instead of 3, the 27B 5 instead of 4, K3 10 instead
of 9. It is opt-in twice -- at build time and at run time (see Build).

**Batched decode (2026-08-19, sparse models 2026-08-23).** One decode step reads the union of every
sequence's expert selections, so sequences share most of the stream. Measured
at a 28 GiB cap, 4k context, genuinely distinct prompts, decode bytes only:

| model | best width | decode GB/token | vs single stream |
|---|---|---|---|
| Kimi K3 2.8T | 32 | 5.98 (from 32.62) | **5.45x** |
| Qwen3.8 2.4T | 16 | 5.30 (from 19.97) | 3.77x |
| GLM-5.2 744B | 32 | 3.17 (from 6.12) | 1.93x |
| GLM-5.2 744B | 44 (`--kv q4`) | **2.40** | **2.55x** |
| MiniMax-M3 429B | 32 (2k ctx) | 1.62 (from 4.05) | 2.50x |
| Qwen3.5 122B-A10B | 32 (`--kv q4`) | **0.406** (from 0.872) | 2.15x |
| DeepSeek V4 Flash 284B | 32 (`--kv q4`) | 0.892 (from 1.56) | 1.75x |
| Qwen3.6 35B-A3B | 32 @ 8 GiB (`--kv q4`) | **0.213** | cheapest streaming batch |
| Qwen3.6 35B-A3B | 32 @ 28 GiB | **0 bytes** | whole model resident; decode never reads disk |

The 122B also clears width 96 on the same 28 GiB cap -- 96 sequences at once,
where K3 stopped at 44 -- because its per-sequence state is ~176 MiB rather
than ~1 GB. Those wider runs measured 0.188 and 0.139 GB/token, but on a
more homogeneous prompt set; a control at width 32 showed prompt similarity
alone is worth 34%, so the varied-prompt figures at 64 and 96 are unmeasured
rather than estimated. See DECISIONS.

Similar-content workloads share more: with identical prompts K3 measured
20.5x at width 32. These figures are decode-only at 6 tokens per sequence;
longer generations cost MORE per token, not less, because sequences diverge
and the shared expert union widens (GLM at width 44 measured 2.40 GB/token
at 6 tokens and 2.640 at 18). Read the table as the best case and budget
about 10% above it for real work. Widths the cap cannot serve are refused at load with the
required cap quoted -- never a corrupted run. The RAM that limits width is
per-sequence state (KV plus recurrent), not the weight cache: at width 32 and
4k context it is two thirds of the 28 GiB budget. More prompts than the cap
funds can rotate through in cohorts (`--rotate`), with state parked to a
directory you choose -- the engine's only sustained-write feature, opt-in,
and its write bytes are reported like every read.

Correctness is sealed on **Windows, Linux, and macOS (Apple silicon)** with identical
node counts on all three; performance figures above are Windows, on the drive named in
DECISIONS.md. A 744B model has answered correctly on a fanless MacBook Air from a
pocket SSD in 3 GiB of RAM.

> **Correction (2026-08-24).** Earlier versions of this table claimed Kimi K3 at a
> 6 GiB minimum and 8 GiB/16.1 s per token. Neither reproduces on the current
> build: 8 GiB is REFUSED at both 4k and 2k context because output.weight needs
> 918 MiB whole and the cap leaves about 200 MiB of cache. The measured floor is
> 9 GiB, where K3 answers correctly at 22.7 s/token and 46.2 GB/token. The
> earlier figures were taken before the resident-byte ledger reached its current
> honesty, the same way the Qwen3.8 note below describes. Also note `--kv q4` does
> NOT work on K3 at all: quantised KV forces flash attention and context creation
> fails outright.
>
> The speed gap is not the drive. Calibrated the same day, this NVMe delivers
> 6.7-6.9 GB/s at the chosen operating points for both sequential and gather, yet
> K3 realises about 2.0 GB/s (46.2 GB in 22.7 s). The engine is leaving roughly
> two thirds of the drive unused on that workload, which is an open lead, not a
> property of the hardware.

> **Correction (2026-08-13).** An earlier version of this file claimed a 4 GiB cap for
> Qwen3.8. That claim was wrong: it was measured before the resident-byte ledger counted
> I/O staging, cache slots, or llama.cpp's own buffers. With everything counted, 4 GiB
> cannot materialise the 1.09 GiB embedding. The run reported zero failures and correct
> text throughout, which is exactly why it went unnoticed: the engine could only count
> what it knew about. The measured minimum is 5 GiB. This correction stays here because
> the lesson is the project's whole method.

## Build

```bash
git clone --recursive <this repo>
cmake --preset dev          # needs Ninja (or pass -G yourself)
cmake --build --preset dev -j
ctest --preset dev
# binary: build/dray/bin/dray (DLLs colocated on Windows)
```

C++20, CMake, CPU-only compute by default. llama.cpp is vendored as a pinned submodule
(the true fork delta is two loader commits, ~47 lines; the rest is an upstream Kimi-K3
PR that has not merged yet). The dev preset is portable (Windows and macOS); a
linux preset and untested ARM64 crosses exist beside it.
`-DDRAY_METAL_BUILD=ON` builds the experimental Metal path, runtime-gated by
`DRAY_METAL=1`.

`--gpu` needs the Vulkan backend, which the default build leaves out: configure with
`-DDRAY_VULKAN_BUILD=ON` (Vulkan SDK installed), or on Windows run
`scripts\build.ps1 -Vulkan`. It is left out because an unused backend is not free --
ggml initialises it at startup, which wakes the GPU driver before `--gpu` is ever
checked. Measured on Kimi K3, same code, same bytes and text: 18.4 s/token CPU-only
against 20.3 with Vulkan compiled in and never used (a laptop with a discrete GPU; the
cost may differ elsewhere). A binary without it refuses `--gpu` rather than quietly
running on the CPU.

## Use

```bash
# What would this model cost me, and what would more RAM buy? (instant, no load)
dray plan -m model-00001-of-00014.gguf --cap 8G

# What can this drive actually do, per traffic class?
dray calibrate -m model.gguf

# Generate, with the live honest readout
dray run -m model.gguf --cap 8G -p "The capital of France is" -n 32

# OpenAI-compatible server: /v1/chat/completions (SSE), /v1/models, /health,
# background jobs with crash-resume via --jobs-dir
dray serve -m model.gguf --cap 8G --port 8080 --jobs-dir ./jobs

# Batched decode: N prompts share one expert stream (see the measured table)
dray batch -m model.gguf --cap 28G --ctx 4096 --batch 32 --prompts jobs.txt -n 200

# More prompts than the cap funds: rotate cohorts, state parked to --state-dir
# (rotation writes; a different device than the model is wise)
dray batch -m model.gguf --cap 28G --batch 32 --prompts many.txt -n 200 \
    --rotate 200 --state-dir E:/scratch
```

The server is deliberately boring: OpenAI protocol only, one generation at a time
(admission is serialized by design), background jobs that checkpoint and survive a
kill. Greedy jobs resume with text identical to an uninterrupted run; sampled jobs
resume plausibly on a disclosed seed. A vanished streaming client stops the disk
within one token (non-streaming requests run to completion; the job API is the
right tool for long work). Failures and taint are reported on every response
surface, because serving
fluent text from wrong weights is worse than not serving.

## The honesty contract

- The resident-byte cap is total process bytes, bound against `max(RSS, commit)`
  every token (commit charge on Windows; RSS on POSIX, where the pool keeps it
  exact-or-conservative). cmd_run prints the per-category ledger at every close
  and a loud CAP BREACH line the moment the cap is exceeded; serve exposes the
  same live through /health, including mid-generation.
- The readout reports byte rates per traffic class from the engine's own counters. A
  quantity that was not measured renders as `unknown`, never as zero.
- Weight reads never retain OS page cache, and tensor data is never mmap'd.
  (One disclosed exception: job checkpoint files use buffered writes, noted in
  place in the code.)
- Wall time is noise; bytes are the decisive metric. Conclusions in DECISIONS.md come
  from bytes-read, which is deterministic.

## Design notes

DECISIONS.md is the project's lab notebook: every measurement with its commit, every
correction with its cause, every open question with the experiment that would settle
it. CLAUDE.md holds the architecture and its eight invariants. Models never live in
this repository.

## Licence

MIT. See LICENSE. Vendored third-party components and their licences are
listed in THIRD_PARTY.md.

## Experimental levers

Every `DRAY_*` variable gates measured behaviour and is named in the run's
own log (an `env:` line in the plan output and the JSONL context), so a
measurement is reproducible from its output. Bisect switches: `NO_FUSE`,
`NO_EARLY`, `NO_RETAIN`, `NO_ESLOTS`, `NO_COMPACT`, `NO_REUSE`, `NO_POOL`,
`NO_ROWSLICE`, `COMPACT_ALL`, `SLOW_LOAD`. Backend opt-ins: `URING` (Linux),
`METAL` (macOS, experimental), `ALLOW_DEGRADED` (accept a backend that cannot
do uncached reads; the cap becomes unenforceable and the run says so).
Tuning: `RING_MB`, `CKPT_EVERY`,
`CKPT_SECONDS`, `RESUME_ATTEMPTS`. Diagnostics: `TRACE`, `TRACE_COMPACT`,
`ROTATE_UNSAFE` (bypasses the rotation span guard and keeps parked-state
copies for byte comparison; the guarded regime is measurably
nondeterministic and this knob exists to diagnose it, not to use it). The
canonical list lives in `src/config/env_report.cpp` and a census test keeps
it honest.
