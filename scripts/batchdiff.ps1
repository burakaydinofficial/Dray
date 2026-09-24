# Batched-decode correctness gate (production batching, phase 3).
#
# Three checks, in decreasing strength:
#   1. WITHIN-BATCH identity: B copies of one greedy prompt must produce B
#      byte-identical outputs. Same kernels, same step -- any difference is
#      cross-sequence contamination in the [k,B] compaction paths. HARD FAIL.
#   2. Compact ON vs OFF under batch: same kernels both legs, so identity is
#      required exactly as in difftest. HARD FAIL.
#   (A single-stream control comparison is deliberately NOT here: batch>1
#   runs GEMM where batch=1 runs GEMV, last bits differ, greedy flips on
#   near-ties -- the documented repack class. Judge such comparisons by
#   coherence manually; this script asserts only what MUST be exact.)
#   J6: requires -Batch >= 2 -- at 1, check 1 is vacuous by construction.
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [string]$Cap = "8G",
    [int]$Tokens = 24,
    [ValidateRange(2, 4096)][int]$Batch = 2,
    [string]$Prompt = "The capital of France is"
)
$ErrorActionPreference = "Continue"
$bin = Join-Path $PSScriptRoot "..\build\dray\bin\dray.exe"
# A missing binary must VOID, not pass: PowerShell leaves $LASTEXITCODE at its
# previous value when a command cannot be launched, so a stale 0 would let both
# legs capture the same launcher error and compare IDENTICAL (2026-08-24 audit).
if (-not (Test-Path $bin)) { Write-Host "*** binary not found at $bin -- build first. VOID ***"; exit 3 }
$out = Join-Path $env:TEMP "dray-batchdiff"
New-Item -ItemType Directory -Force -Path $out | Out-Null

function SeqTexts([string]$path) {
    $raw = [System.IO.File]::ReadAllText($path)
    $texts = @()
    $parts = [regex]::Split($raw, '--- seq \d+ \(\d+ tokens[^)]*\) ---\r?\n')
    for ($i = 1; $i -lt $parts.Count; $i++) {
        $t = $parts[$i]
        $cut = $t.IndexOf("--- seq"); if ($cut -lt 0) { $cut = $t.IndexOf("batch summary") }
        if ($cut -gt 0) { $t = $t.Substring(0, $cut) }
        $texts += ($t -replace "`r", '').Trim("`n")
    }
    return ,$texts
}

$f1 = Join-Path $out "on.txt"
& $bin batch -m $Model --cap $Cap --force-stream --ctx 512 --batch $Batch -n $Tokens -p $Prompt *> $f1
if ($LASTEXITCODE -ne 0) { Write-Host "*** batch (compact on) exited $LASTEXITCODE -- VOID ***"; exit 3 }
if (Select-String -Path $f1 -Pattern "NOT TRUSTWORTHY|CAP BREACH" -Quiet) { Write-Host "*** compact-on leg TAINTED -- VOID ***"; exit 3 }
$env:DRAY_NO_COMPACT = '1'
$f2 = Join-Path $out "off.txt"
& $bin batch -m $Model --cap $Cap --force-stream --ctx 512 --batch $Batch -n $Tokens -p $Prompt *> $f2
$code2 = $LASTEXITCODE
Remove-Item Env:\DRAY_NO_COMPACT -ErrorAction SilentlyContinue
if ($code2 -ne 0) { Write-Host "*** batch (compact off) exited $code2 -- VOID ***"; exit 3 }
if (Select-String -Path $f2 -Pattern "NOT TRUSTWORTHY|CAP BREACH" -Quiet) { Write-Host "*** compact-off leg TAINTED -- VOID ***"; exit 3 }

$on = SeqTexts $f1
$off = SeqTexts $f2
if ($on.Count -ne $Batch -or [string]::IsNullOrWhiteSpace($on[0])) {
    Write-Host "*** expected $Batch sequences with text; got $($on.Count) -- VOID ***"; exit 3
}
$within = $true
for ($i = 1; $i -lt $on.Count; $i++) { if ($on[$i] -cne $on[0]) { $within = $false } }
Write-Host ("within-batch identity  : {0}" -f $(if ($within) { "IDENTICAL (pass)" } else { "*** DIFFER -- CONTAMINATION ***" }))
$across = $true
for ($i = 0; $i -lt $on.Count; $i++) { if ($off[$i] -cne $on[$i]) { $across = $false } }
Write-Host ("compact ON vs OFF      : {0}" -f $(if ($across) { "IDENTICAL (pass)" } else { "*** DIFFER -- BUG ***" }))
if (-not ($within -and $across)) { exit 1 }
exit 0
