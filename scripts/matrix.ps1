# Performance matrix: model x RAM cap x context x GPU, prefill and decode kept
# SEPARATE because they are different workloads with opposite hardware
# preferences (prefill is compute-bound and offloads well; decode is not).
#
# Appends one CSV row per cell to scripts/matrix-results.csv so a long run can
# be read while it is still going, and so a killed run loses one cell, not all
# of them. Re-running skips cells already present unless -Force.
#
# Every cell is a real run. Refusals record the cap the engine asked for; a
# crash records FAILED with its exit code; a run that emitted taint markers
# records TAINTED and its number is discarded; a run whose readout could not be
# parsed records UNPARSED. Nothing is estimated, and an absent row means it was
# not run.
#
# The note column carries the allocation path -- resident or streaming. A cell
# where the model fits the cap measures llama.cpp's allocator rather than this
# engine, which is worth reporting but is not the same quantity as the streaming
# rows beside it.
#
# ROWS RECORDED BEFORE 2026-08-24 CARRY NO MODE LABEL. In those, the resident
# cells are the ones whose bytes_per_token is 0; everything else streamed.

param(
    [string]$Only = "",          # substring filter on model name
    [switch]$Force,              # re-run cells already in the CSV
    [int]$DecodeTokens = 32,
    [string]$PrefillFile = ""    # defaults to the 6,594-token prompt
)
$ErrorActionPreference = "Continue"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$bin  = Join-Path $root "build\dray\bin\dray.exe"
$csv  = Join-Path $PSScriptRoot "matrix-results.csv"
if (-not $PrefillFile) { $PrefillFile = Join-Path $root "build\prompt_8000tok.txt" }
# Refuse to produce a results file at all if the inputs are missing. A missing
# binary or prompt file used to write a full column of "the engine refused"
# rows -- fabricated measurements, in the file the README quotes.
if (-not (Test-Path $bin))         { Write-Host "*** binary not found at $bin -- build first ***"; exit 3 }
if (-not (Test-Path $PrefillFile)) { Write-Host "*** prefill prompt not found at $PrefillFile ***"; exit 3 }
# Every cell must record which allocation path produced it. Cells where the
# model fits the cap take the resident path and measure llama's allocator, not
# this engine; they used to land in the same columns as streaming rows with
# nothing to tell them apart.
Get-ChildItem Env: | Where-Object { $_.Name -like 'DRAY_*' } |
    ForEach-Object { Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue }

# cap list is per model: the floor, a small budget, a working budget, and a
# generous one. No point offering a 28 GiB cap to a model whose floor is 6.
$models = @(
  @{ n="qwen38-27b-dense"; p="D:\Models\unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-Q4_K_XL.gguf";
     caps=@("4G","8G","20G"); ctxs=@(4096,32768) },
  @{ n="qwen36-35b-a3b";   p="D:\Models\unsloth\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf";
     caps=@("3G","8G","28G"); ctxs=@(4096,32768) },
  @{ n="qwen35-122b-a10b"; p="D:\Models\unsloth\Qwen3.5-122B-A10B-GGUF\Qwen3.5-122B-A10B-UD-Q2_K_XL.gguf";
     caps=@("3G","12G","28G"); ctxs=@(4096,32768) },
  @{ n="deepseek-v4-flash"; p="D:\Models\unsloth\DeepSeek-V4-Flash-0731-GGUF\UD-Q2_K_XL\DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf";
     caps=@("3G","8G","24G"); ctxs=@(4096,32768) },
  @{ n="minimax-m3";       p="D:\Models\unsloth\MiniMax-M3-GGUF\UD-Q2_K_XL\MiniMax-M3-UD-Q2_K_XL-00001-of-00004.gguf";
     caps=@("8G","28G"); ctxs=@(4096) },
  @{ n="glm-5.2-744b";     p="D:\Models\unsloth\GLM-5.2-GGUF\UD-IQ1_S\GLM-5.2-UD-IQ1_S-00001-of-00006.gguf";
     caps=@("8G","28G"); ctxs=@(4096) },
  @{ n="kimi-k3-2.8t";     p="D:\Models\unsloth\Kimi-K3-GGUF\UD-IQ1_S\Kimi-K3-UD-IQ1_S-00001-of-00014.gguf";
     caps=@("9G","28G"); ctxs=@(4096); kv="f16" }   # q4 fails: quantised KV forces
                                                    # flash attention and K3 cannot
                                                    # create a context with it
)

if (-not (Test-Path $csv)) {
    "model,cap,ctx,gpu,phase,metric,value,note" | Set-Content $csv -Encoding utf8
}
$done = @{}
Get-Content $csv | Select-Object -Skip 1 | ForEach-Object {
    $f = $_ -split ','
    if ($f.Count -ge 5) { $done["$($f[0])|$($f[1])|$($f[2])|$($f[3])|$($f[4])"] = $true }
}

function Emit($m, $cap, $ctx, $gpu, $phase, $metric, $value, $note) {
    "$m,$cap,$ctx,$gpu,$phase,$metric,$value,$note" | Add-Content $csv -Encoding utf8
    Write-Host "  $m $cap ctx$ctx gpu=$gpu $phase : $metric=$value $note"
}

foreach ($mm in $models) {
    if ($Only -and $mm.n -notlike "*$Only*") { continue }
    if (-not (Test-Path $mm.p)) { Write-Host "$($mm.n): ABSENT, skipped"; continue }
    foreach ($cap in $mm.caps) {
      foreach ($ctx in $mm.ctxs) {
        foreach ($gpu in @("off","on")) {
          $gpuArgs = if ($gpu -eq "on") { @("--gpu") } else { @() }
          $kv = if ($mm.kv) { $mm.kv } else { "q4" }

          # ---- DECODE: short prompt, so the wall is dominated by generation
          $key = "$($mm.n)|$cap|$ctx|$gpu|decode"
          if ($Force -or -not $done.ContainsKey($key)) {
            $a = @("run","-m",$mm.p,"--cap",$cap,"--ctx","$ctx","--kv",$kv,
                   "-n","$DecodeTokens","-p","Explain how a refrigerator moves heat from inside to outside.") + $gpuArgs
            $out = & $bin @a 2>&1
            if ($LASTEXITCODE -ne 0) {
                # A crash is not a refusal. This used to emit REFUSED on ANY
                # nonzero exit whether or not the word appeared, so a segfault, an
                # OOM and a genuine admission refusal all landed in the published
                # CSV as "the engine declined this cap" (2026-08-24 audit).
                $r = ($out | Select-String "REFUSED" | Select-Object -First 1).Line
                if ($r) {
                    Emit $mm.n $cap $ctx $gpu "decode" "status" "REFUSED" ($r -replace ',',';')
                } else {
                    $tail = ($out | Select-Object -Last 1) -replace ',',';'
                    Emit $mm.n $cap $ctx $gpu "decode" "status" "FAILED" "exit $LASTEXITCODE : $tail"
                }
            } elseif ($out | Select-String "NOT TRUSTWORTHY|CAP BREACH" -Quiet) {
                # Every correctness gate refuses a tainted run; the harness whose
                # output populates the README did not even look.
                Emit $mm.n $cap $ctx $gpu "decode" "status" "TAINTED" "run produced taint markers; number discarded"
            } else {
                # Which allocation path produced this number. A cell where the
                # model fits the cap runs resident and is measuring llama.cpp's
                # allocator, not the streaming engine -- a legitimate thing to
                # report, but not the same quantity as the streaming rows beside
                # it, and previously indistinguishable from them.
                $mode = if ($out | Select-String "resident mode:" -Quiet) { "resident" } else { "streaming" }
                $rate  = ($out | Select-String -Pattern "mean rate\s+([0-9.]+) (s/tok|tok/s)" | Select-Object -First 1).Matches
                $bytes = ($out | Select-String -Pattern "total per token\s+([0-9.]+) (MiB|GiB|B)" | Select-Object -First 1).Matches
                if (-not $rate) {
                    # An unparsed readout is a HOLE, not an absent row. The header
                    # claims "an absent row means it was not run"; a format change
                    # used to make a run that happened look like one that did not.
                    Emit $mm.n $cap $ctx $gpu "decode" "status" "UNPARSED" "$mode ; ran clean but no rate line matched"
                }
                if ($rate)  { Emit $mm.n $cap $ctx $gpu "decode" $rate.Groups[2].Value $rate.Groups[1].Value $mode }
                if ($bytes) { Emit $mm.n $cap $ctx $gpu "decode" "bytes_per_token_$($bytes.Groups[2].Value)" $bytes.Groups[1].Value $mode }
            }
          }

          # ---- PREFILL: long prompt, one token out, so the wall is the prompt
          $key = "$($mm.n)|$cap|$ctx|$gpu|prefill"
          if (($ctx -ge 8192) -and ($Force -or -not $done.ContainsKey($key))) {
            $a = @("run","-m",$mm.p,"--cap",$cap,"--ctx","$ctx","--kv",$kv,
                   "-n","1","--prompt-file",$PrefillFile) + $gpuArgs
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $out = & $bin @a 2>&1
            $sw.Stop()
            if ($LASTEXITCODE -ne 0) {
                # This branch did not even look for the word before writing it.
                $r = ($out | Select-String "REFUSED" | Select-Object -First 1).Line
                if ($r) {
                    Emit $mm.n $cap $ctx $gpu "prefill" "status" "REFUSED" ($r -replace ',',';')
                } else {
                    Emit $mm.n $cap $ctx $gpu "prefill" "status" "FAILED" "exit $LASTEXITCODE"
                }
            } elseif ($out | Select-String "NOT TRUSTWORTHY|CAP BREACH" -Quiet) {
                Emit $mm.n $cap $ctx $gpu "prefill" "status" "TAINTED" "run produced taint markers; number discarded"
            } else {
                $tk = ($out | Select-String -Pattern "prefill: (\d+) tokens" | Select-Object -First 1).Matches
                $n  = if ($tk) { [int]$tk.Groups[1].Value } else { 0 }
                $s  = [math]::Round($sw.Elapsed.TotalSeconds,1)
                Emit $mm.n $cap $ctx $gpu "prefill" "wall_s" $s "$n tokens incl load"
                if ($n -gt 0) { Emit $mm.n $cap $ctx $gpu "prefill" "tok_per_s" ([math]::Round($n/$s,1)) "" }
            }
          }
        }
      }
    }
}
Write-Host "matrix: done, results in $csv"
