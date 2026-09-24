# Resident-mode correctness gate. Resident mode hands allocation to llama when
# the whole model fits the cap, so it is a SECOND code path through the engine
# and needs its own proof before it can ever be a default.
#
# The check: greedy text from resident mode must match greedy text from the
# streaming path exactly. Same weights, same sampler, same prompt -- the only
# difference is who allocated the memory, which must not change a single token.
# (This is a stricter bar than difftest's compaction check and deliberately so:
# a mode that silently produces different text is the failure this project
# refuses to ship. The first implementation emitted "!!!!!!!!".)
#
# VOIDs on nonzero exit, taint, or empty text; FAILS on any divergence.
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [string]$Cap = "8G",
    [int]$Tokens = 24,
    [string]$Prompt = "The capital of France is"
)
$ErrorActionPreference = "Continue"
$bin = Join-Path $PSScriptRoot "..\build\dray\bin\dray.exe"
# A missing binary must VOID, not pass: PowerShell leaves $LASTEXITCODE at its
# previous value when a command cannot be launched, so a stale 0 would let both
# legs capture the same launcher error and compare IDENTICAL (2026-08-24 audit).
if (-not (Test-Path $bin)) { Write-Host "*** binary not found at $bin -- build first. VOID ***"; exit 3 }
$out = Join-Path $env:TEMP "dray-residenttest"
New-Item -ItemType Directory -Force -Path $out | Out-Null

function Leg([string]$tag, [string[]]$extra) {
    # Clear every DRAY_* lever: an ambient one changes what both legs do, and
    # two legs perturbed identically still compare IDENTICAL (2026-08-24 audit).
    Get-ChildItem Env: | Where-Object { $_.Name -like 'DRAY_*' } |
        ForEach-Object { Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue }
    $f = Join-Path $out "$tag.txt"
    # --threads 4 PINS both phases: this gate isolates the ALLOCATION PATH, so
    # every other variable must be held constant. Thread count changes matmul
    # reduction order, and the two paths parallelise differently, so at the
    # default split (4 decode / 13 prefill) they legitimately diverge in the
    # last bits -- documented, not a bug, but it would mask a real one here.
    $a = @("run", "-m", $Model, "--cap", $Cap, "--threads", "4", "--ctx", "512", "-n", "$Tokens", "-p", $Prompt) + $extra
    & $bin @a *> $f
    if ($LASTEXITCODE -ne 0) { Write-Host "*** $tag exited $LASTEXITCODE -- VOID ***"; exit 3 }
    if (Select-String -Path $f -Pattern "NOT TRUSTWORTHY|CAP BREACH|REFUSED" -Quiet) {
        Write-Host "*** $tag TAINTED -- VOID ***"; exit 3
    }
    # Generated text = everything after the last readout line (same "MiB" anchor
    # difftest uses), CR-stripped, edge newlines trimmed. Verbatim otherwise.
    $raw = [System.IO.File]::ReadAllText($f) -replace "`r", ''
    $lines = $raw -split "`n"
    $p = -1
    for ($i = 0; $i -lt $lines.Count; $i++) { if ($lines[$i] -match "MiB") { $p = $i } }
    if ($p -lt 0) { Write-Host "*** $tag has no readout anchor -- VOID ***"; exit 3 }
    $text = ($lines[($p + 1)..($lines.Count - 1)] -join "`n").Trim("`n")
    if ([string]::IsNullOrWhiteSpace($text)) { Write-Host "*** $tag produced NO text -- VOID ***"; exit 3 }
    return $text
}

$streamText = Leg "stream" @("--force-stream")
$residentText = Leg "resident" @("--resident")

# The resident leg must actually have taken the resident path, or this gate is
# vacuous -- exactly the trap --force-stream exists to prevent elsewhere.
$rf = Join-Path $out "resident.txt"
if (-not (Select-String -Path $rf -Pattern "resident mode:" -Quiet)) {
    Write-Host "*** resident leg did NOT enter resident mode (model too big for this cap?) -- VOID ***"
    exit 3
}

if ($streamText -ceq $residentText) {
    Write-Host "resident vs streaming : IDENTICAL (pass)"
    exit 0
} else {
    Write-Host "resident vs streaming : *** DIFFER -- BUG ***"
    Write-Host "--- streaming ---"; Write-Host $streamText.Substring(0, [Math]::Min(200, $streamText.Length))
    Write-Host "--- resident ---";  Write-Host $residentText.Substring(0, [Math]::Min(200, $residentText.Length))
    exit 1
}
