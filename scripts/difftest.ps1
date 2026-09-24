# Differential correctness check.
#
# The engine's failure mode is fluent, confident, wrong output with zero reported
# failures -- seven bugs this way, several of which passed every internal check.
# Counters cannot catch a weight that was never read or an expert that was read
# from the wrong slot. Only comparing against a known-good answer can.
#
# Two comparisons, in decreasing strength:
#
#   1. --no-stream (llama.cpp holds the weights) vs streaming.
#      Ground truth, but only on a model small enough to hold, and NOT bit-exact:
#      llama.cpp repacks quantised weights into an interleaved layout we cannot use
#      (it needs the whole tensor at load), so the kernels differ in the last bits.
#      Judge this one on coherence and meaning, not equality.
#
#   2. compaction ON vs OFF, both streaming.
#      Weaker as evidence but works on ANY model, needs no reference, and IS
#      expected to be exact: compaction changes which bytes are fetched, never the
#      arithmetic. Any difference here is a bug, full stop.
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [string]$Cap = "8G",
    [int]$Tokens = 32,
    [string]$Prompt = "The capital of France is",
    [switch]$WithReference   # only meaningful for models that fit in the cap
)

# NOT "Stop": redirecting a native command's stderr in PowerShell 5.1 wraps each
# line in an ErrorRecord and trips a terminating error, even on exit code 0. llama.cpp
# writes its whole banner to stderr, so "Stop" here fails every run before it starts.
$ErrorActionPreference = "Continue"
$bin = Join-Path $PSScriptRoot "..\build\dray\bin\dray.exe"
# A missing binary must VOID, not pass: PowerShell leaves $LASTEXITCODE at its
# previous value when a command cannot be launched, so a stale 0 would let both
# legs capture the same launcher error and compare IDENTICAL (2026-08-24 audit).
if (-not (Test-Path $bin)) { Write-Host "*** binary not found at $bin -- build first. VOID ***"; exit 3 }
$out = Join-Path $env:TEMP "dray-difftest"
New-Item -ItemType Directory -Force -Path $out | Out-Null

function Generated([string]$path) {
    # The emitted text is whatever follows the last llama.cpp banner line.
    # H19: verbatim except CR stripping. Collapsing all whitespace hid exactly
    # the trailing-newline/stop-boundary class the seeded calibration proved
    # reviewers miss, and -eq compared case-insensitively on top of it -- the
    # reference-platform gate was strictly weaker than linux-gate.sh's verbatim
    # compare with nothing recording the weakening.
    $all = Get-Content $path -Raw
    $idx = $all.LastIndexOf("MiB")
    if ($idx -ge 0) { $all = $all.Substring($idx + 3) }
    ($all -replace "`r", '').Trim("`n")
}

function Run([string]$tag, [hashtable]$envs, [string[]]$extra) {
    # CLEAR EVERY LEVER FIRST. Twenty-seven DRAY_* variables are censused, and
    # several change what this gate is comparing: an ambient DRAY_NO_COMPACT=1
    # makes BOTH legs run with compaction off, so they match and the gate reports
    # a pass having verified nothing. DRAY_FAST_NODES and
    # DRAY_ALLOW_DEGRADED are equally load-bearing. A gate must define its own
    # environment, not inherit the operator's (2026-08-24 audit).
    Get-ChildItem Env: | Where-Object { $_.Name -like 'DRAY_*' } |
        ForEach-Object { Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue }
    foreach ($k in $envs.Keys) { Set-Item -Path "Env:$k" -Value $envs[$k] }
    $f = Join-Path $out "$tag.txt"
    # --force-stream is MANDATORY here: the testbed model fits an 8 GiB cap, so
    # without it resident mode would hand allocation to llama and this gate
    # would silently stop testing the streaming engine entirely.
    $args = @("run", "-m", $Model, "--cap", $Cap, "--force-stream", "--ctx", "512", "-n", "$Tokens", "-p", $Prompt) + $extra
    & $bin @args *> $f
    $code = $LASTEXITCODE
    $bad = Select-String -Path $f -Pattern "NOT TRUSTWORTHY|CAP BREACH" -Quiet
    foreach ($k in $envs.Keys) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
    if ($bad) {
        Write-Host ("*** $tag is TAINTED (NOT TRUSTWORTHY/CAP BREACH in output) -- run is VOID ***")
        exit 3
    }
    if ($code -ne 0) {
        Write-Host ("*** $tag exited $code -- run is VOID, not comparable ***")
        Write-Host (Get-Content $f -Tail 3 | Out-String)
        exit 3
    }
    $t = Generated $f
    if ([string]::IsNullOrWhiteSpace($t)) {
        Write-Host ("*** $tag produced NO generated text -- two empty runs would 'match'; refusing (swarm R5) ***")
        exit 3
    }
    return $t
}

Write-Host "model : $Model"
Write-Host "cap   : $Cap, $Tokens tokens"
Write-Host ""

$onText  = Run "compact_on"  @{}                          @()
$offText = Run "compact_off" @{ DRAY_NO_COMPACT = '1' } @()

$exact = ($onText -ceq $offText)   # H19: case-SENSITIVE
Write-Host ("compaction ON vs OFF : {0}" -f $(if ($exact) { "IDENTICAL (pass)" } else { "*** DIFFER -- BUG ***" }))
if (-not $exact) {
    Write-Host "  ON  : $onText"
    Write-Host "  OFF : $offText"
}

if ($WithReference) {
    $refText = Run "reference" @{} @("--no-stream")
    Write-Host ""
    Write-Host "reference (llama.cpp) : $refText"
    Write-Host "streamed              : $onText"
    Write-Host ""
    if ($refText -ceq $onText) {
        Write-Host "identical -- stronger than required (repack must be off in this build)"
    } else {
        Write-Host "differ: expected on quantised models, llama.cpp repacks and we cannot."
        Write-Host "Judge by meaning. Wrong-expert corruption looks like word salad, not paraphrase."
    }
}

if (-not $exact) { exit 1 }
exit 0
