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

If the model fits in your RAM, use llama.cpp. Measured 2026-10-06 on this machine,
both built from the same llama.cpp commit, load excluded from both sides (our
windowed decode rate against llama-bench tg32):

| model | dray resident | stock llama.cpp, same 4 threads | stock at its default 16 threads |
|---|---|---|---|
| Qwen3.6 35B-A3B | 10.4 tok/s | 10.40 | 12.20 |
| Qwen3.8-27B dense | 1.9 tok/s | 1.88 | 2.51 |

At the same thread count resident mode now matches stock. Stock at its own default
is still faster, by 17% and 32%; our best on the 35B is 11.4 tok/s at 8 threads,
still behind. When the model fits, you gain a memory cap you do not need and give
up that margin. (Until 2026-10-05 the gap was 21-46% at stock's settings: every CPU
run here built a fresh thread pool per graph node, and resident mode never reached
llama's repacked kernels. Both are fixed.)

This engine earns its keep on ONE thing: models that do not fit. A 594 GB
checkpoint under a 5 GiB cap is not slower elsewhere, it is impossible
elsewhere. When the model does not fit, the streaming path costs more compute
than resident allocation would (a thread-pool barrier per intercepted graph node,
which is the price of repointing weights mid-forward-pass), and that penalty
disappears into disk time as soon as a configuration streams more than a few GB
per token.

The rule: does it fit? Use llama.cpp. Does it not fit? That is what this is.

## A surprise worth knowing before you buy a faster drive

For sparse models the disk is nearly free. Qwen3.6 35B-A3B with the streaming path
pinned in both legs (2026-10-06): at a 3 GiB cap it reads about 420 MB per token
more than at 28 GiB, and decodes at 2.6 tok/s against 3.0. The drive adds about
13%; compute is the rest. (Before the 2026-10-05 thread-pool fix, compute was slow
enough to hide the drive completely: Qwen3.5-122B-A10B ran at 2.7 s/token reading
872 MiB per token or nothing.)

So the cost model this project is built on -- misses x bytes / bandwidth -- is
right in shape for Kimi K3 and wrong for anything moving under a gigabyte per
token. Below some byte rate the engine is the limit and the SSD is idle
capacity you already paid for. If your model is in that regime, a faster drive
buys little.

ON THE LARGE MODELS THE DRIVE IS THE LIMIT, as it should be. Calibrated on Kimi
K3's own fourteen shards this NVMe serves 2 MiB random reads at 6.03 GB/s. At a
9 GiB cap K3 reads 50.6 GB per decode token at 8.3 s/token (2026-10-06, 30 tokens,
load and prompt excluded): 6.1 GB/s, the drive's measured ceiling. In August the
engine reached 1.8 GB/s, because reads only progressed between compute nodes; an
I/O thread has owned the device since 2026-09-26 (DECISIONS). Here more RAM (fewer
bytes per token) or a faster drive is what buys speed.

## Measured status (2026-10-06)

Re-measured on 2026-10-06 after the 10-05 fixes (a thread pool per graph node on
every CPU run; see DECISIONS.md), two full passes on an otherwise idle machine,
High performance power plan. Both passes are shown; where they agree, one value.

| model | disk | cap | decode (pass 1 / pass 2) | projected bytes/token |
|---|---|---|---|---|
| Qwen3.6 35B-A3B | 21 GB | **3 GiB** | 2.8 tok/s | 559 MiB |
| Qwen3.6 35B-A3B | 21 GB | 8 GiB | 2.9 / 3.1 tok/s | 399 MiB |
| Qwen3.6 35B-A3B | 21 GB | 28 GiB | 7.3 / 8.1 tok/s | 0 (resident) |
| DeepSeek V4 Flash 284B | 90 GB | **3 GiB** | 1.5 s/token | 5.46 GiB |
| DeepSeek V4 Flash 284B | 90 GB | 8 GiB | 1.0 tok/s | 1.92 GiB |
| DeepSeek V4 Flash 284B | 90 GB | 24 GiB | 1.0 / 1.1 tok/s | 1.55 GiB |
| Qwen3.8-27B dense | 16 GB | **4 GiB** | 2.4 s/token | 12.07 GiB |
| Qwen3.8-27B dense | 16 GB | 8 GiB | 2.0 s/token | 8.07 GiB |
| Qwen3.8-27B dense | 16 GB | 20 GiB | 1.6 / 1.7 tok/s | 0 (resident) |
| MiniMax-M3 429B | 143 GB | 8 GiB | 1.9 / 2.0 s/token | 3.98 GiB |
| MiniMax-M3 429B | 143 GB | 28 GiB | 1.5 s/token | 3.27 GiB |
| Kimi K3 2.8T | 594 GB | **5 GiB** | 27.4 s/token (see below) | 49.32 GiB |
| Kimi K3 2.8T | 594 GB | 9 GiB | 9.8 / 9.9 s/token | 45.32 GiB |
| Kimi K3 2.8T | 594 GB | 28 GiB | 7.7 s/token | 26.32 GiB |

All rows: 4k context, CPU only, `--kv q4` (K3 uses f16, the only setting it
accepts), 32 generated tokens, one build. "Decode" is the run's own mean rate
including the short prompt. Bold caps are the smallest measured that produce correct
text; some may now run lower (the read-ahead ring yields to a tight cap since
2026-10-05) and have not been probed. Reproduce with `scripts/matrix.ps1 -GpuModes
off -Contexts 4096`.

**Bytes/token is the planner's projection** for that cap (the `plan` output), which
assumes perfect eviction; measured traffic is higher. K3 at 5 GiB projects 49.3 GiB
and measured 1,860 GB over the prompt and 32 tokens, 52-54 GiB per forward pass.

**K3 at 5 GiB** is the floor since 2026-10-05: the read-ahead ring gives its memory
to the cache, and the run says so; 4.6 GB resident. Five of six runs completed with
correct text (26.9-27.4 s/token); one exited with an error after generating text,
and that run's log was not kept -- the harness now keeps every failed run's output.

Against the previous table (2026-08-23, Balanced plan, older build): the 35B at 3 GiB
went from 1.9 s/token to 2.8 tok/s, DeepSeek at 8 GiB from 3.9 s/token to 1.0 tok/s,
MiniMax-M3 at 8 GiB from 6.8 to 1.9 s/token, and K3 at 9 GiB from 25.8 to 9.8
s/token. The Qwen3.5 122B-A10B and GLM-5.2 rows of that table are not repeated: those
models are no longer on this machine.

**Prefill is a different workload**, and `--gpu` exists for it.

> **Correction (2026-09-30).** This section used to show `--gpu` prefill at
> 180-343 tokens/second (12x-72x CPU). Those figures are WITHDRAWN: on the
> streaming path every `--gpu` run computed on garbage. ggml's scheduler copies
> a GPU split's weights straight from host memory before our streaming callback
> runs, so weights not yet loaded went to the GPU as poison -- wrong text, exit
> 0, no failure counted -- and no gate ran `--gpu`. Fixed on 2026-09-30 (a copy
> callback in our llama.cpp fork loads each weight before it is copied), and
> `scripts/gpugate.ps1` now proves `--gpu` bit-identical to llama.cpp's own GPU
> path on every architecture that fits resident here.
>
> Measured 2026-10-06 (High performance plan, idle machine, 8k context), on a fixed
> 3,264-token prompt -- the first 9,000 characters of this README as of 2026-10-01 --
> with defaults: the automatic 1.94 GiB VRAM limit chooses the chunk. Seconds until
> generation starts, both including ~5 s of load:
>
> | model | cap | CPU | `--gpu` | GB read CPU / GPU |
> |---|---|---|---|---|
> | Qwen3.6 35B-A3B | 12 GiB | 106.1 / 106.7 | **36.9-42.6** (1024-token chunks) | 100.4 / 62.2 |
> | Qwen3.8-27B dense | 16 GiB | 553.3 | **47.6 / 49.5** (512-token chunks) | 35.1 / 43.8 |
>
> The dense model's GPU path reads more: every weight is copied per chunk, while
> the CPU path keeps part of the model cached. The 2026-10-01 figures this replaces
> (CPU 133.9 and 707.1 s, GPU 16.8 and 33 s) were taken on a shorter prompt -- that
> day's README -- and the CPU ones with a fresh thread pool per graph node; they are
> withdrawn rather than compared. On this prompt the 2026-10-01 build takes 55-56 s
> for the 35B with `--gpu`, the current one 37-43 s.

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
and its write bytes are reported like every read. A span shorter than the
generation parks sequences mid-generation and resumes them later, token for
token identical to running them straight through (`scripts/rotategate.ps1`,
attention KV and recurrent state).

Correctness is sealed on **Windows, Linux, and macOS (Apple silicon)** with identical
node counts on all three; performance figures above are Windows, on the drive named in
DECISIONS.md. A 744B model has answered correctly on a fanless MacBook Air from a
pocket SSD in 3 GiB of RAM.

> **Correction (2026-08-24).** Earlier versions of this table claimed Kimi K3 at a
> 6 GiB minimum and 8 GiB/16.1 s per token. Neither reproduces on the current
> build: 8 GiB is REFUSED at both 4k and 2k context because output.weight needs
> 918 MiB whole and the cap leaves about 200 MiB of cache. The measured floor was
> 9 GiB, where K3 answers correctly at 22.7 s/token and 46.2 GB/token (since
> 2026-10-05 it is 5 GiB: a fork fix returned 3.5 GB of unused scheduler memory,
> and the read-ahead ring now yields to output.weight instead of refusing). The
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

The server is deliberately boring: OpenAI protocol only -- tool calls and
reasoning_content included, through the model's own chat template, so agents
such as Cline work -- `--parallel N` requests generating together (each with
its own context, all funded inside `--cap` at load; the rest queue), and
background jobs that checkpoint and survive a kill. Greedy jobs resume with
text identical to an uninterrupted run; sampled jobs resume plausibly on a
disclosed seed. A vanished streaming client stops the disk
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

DECISIONS.md is the project's lab notebook: what was measured and under which
conditions, every correction with its cause, and the questions still open. CLAUDE.md
holds the architecture and its eight invariants. Models never live in this
repository.

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
`CKPT_SECONDS`, `RESUME_ATTEMPTS`. Diagnostics: `TRACE`, `TRACE_COMPACT`. The
canonical list lives in `src/config/env_report.cpp` and a census test keeps
it honest.
