# Rotation gate: does mid-generation parking change a single token?
#
# `batch --rotate SPAN` serves more prompts than the funded width in cohorts. A
# SPAN below -n parks each live sequence's KV/recurrent state to --state-dir
# after SPAN tokens and restores it on the cohort's next turn. Until 2026-10-05
# that was refused: on 08-19 identical reruns diverged (DECISIONS). It no longer
# reproduces (09-29), and this gate is what keeps it that way.
#
# Per model, four legs over the same prompts at the same width:
#   control  --rotate N      cohorts run to completion: nothing is parked
#   park     --rotate 4      parked and restored every 4 tokens
#   rerun    --rotate 4      the same command again (the 08-19 failure mode)
#   offset   --rotate 5      different park boundaries
# Every sequence's text must be identical across all four. Same width, same
# cohorts, same slots, so the only difference is the park/restore roundtrip.
# A park leg that wrote no state VOIDS the run (it would prove nothing).
#
# Two state classes, both required where present: attention KV (the testbed)
# and opaque recurrent state (Qwen3.8-27B, Gated DeltaNet). Absent models are
# SKIPPED, never passed. Exit 0 = every checked model passed; 1 = a mismatch;
# 3 = void.
param(
    [string]$BinDir = (Join-Path $PSScriptRoot "..\build\dray\bin"),
    [string]$Exe = "dray.exe",
    [int]$Tokens = 16,
    [string]$Only = ""          # substring filter on model name
)
$ErrorActionPreference = "Continue"
$bin = Join-Path $BinDir $Exe
if (-not (Test-Path $bin)) { Write-Host "*** binary not found at $bin -- build first. VOID ***"; exit 3 }
# One directory per run: two runs sharing one once deleted each other's parked
# state mid-leg (2026-10-05). Removed after a pass, kept for inspection otherwise.
$work = Join-Path $env:TEMP ("dray-rotategate-" + $PID)
New-Item -ItemType Directory -Force -Path $work | Out-Null

$ambient = Get-ChildItem Env: | Where-Object { $_.Name -match '^(DRAY|DRAY)_' }
if ($ambient) { Write-Host ("ambient levers (apply to every leg): " + (($ambient | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ", ")) }

# Six distinct prompts at width 2: three cohorts, so every cohort is parked and
# restored while the others run.
$prompts = Join-Path $work "prompts.txt"
@(
    "The capital of France is",
    "The largest planet in the solar system is",
    "Water boils at a temperature of",
    "The author of Hamlet is",
    "The chemical symbol for gold is",
    "The tallest mountain on Earth is"
) | Set-Content -Path $prompts -Encoding utf8
$nPrompts = 6

$models = @(
    @{ name = "testbed-1b7b"; cap = "4G"; state = "attention KV"
       path = "D:\Models\testbed\OLMoE\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf" },
    @{ name = "qwen38-27b";   cap = "12G"; state = "recurrent (Gated DeltaNet) + attention"
       path = "D:\Models\unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-Q4_K_XL.gguf" }
)

function SeqTexts([string]$path) {
    $raw = [System.IO.File]::ReadAllText($path)
    $texts = @()
    $parts = [regex]::Split($raw, '--- seq \d+ \(\d+ tokens[^)]*\) ---\r?\n')
    for ($i = 1; $i -lt $parts.Count; $i++) {
        $t = $parts[$i]
        $cut = $t.IndexOf("--- seq"); if ($cut -lt 0) { $cut = $t.IndexOf("rotate summary") }
        if ($cut -gt 0) { $t = $t.Substring(0, $cut) }
        $texts += ($t -replace "`r", '').Trim("`n")
    }
    return ,$texts
}

# Returns the leg's sequence texts; "FAIL" when parking or restoring itself
# failed (that is the feature under test); $null after printing why it is void.
function Run-Leg($m, [string]$tag, [int]$span) {
    $out = Join-Path $work "$($m.name)-$tag.txt"
    # stdout only: the texts are parsed from it. stderr (the streamer report,
    # progress lines) once landed inside the last sequence's text when the two
    # streams were captured together, failing a run whose tokens all matched.
    $err = Join-Path $work "$($m.name)-$tag.err"
    $state = Join-Path $work "$($m.name)-$tag-state"
    & $bin batch -m $m.path --cap $m.cap --force-stream --ctx 512 --batch 2 --prompts $prompts `
        -n $Tokens --rotate $span --state-dir $state > $out 2> $err
    $parkfail = Select-String -Path $out -Pattern "state (restore|park) failed" | Select-Object -First 1
    if ($parkfail) {
        Write-Host ("  {0,-8}: --rotate {1,-2} FAILED: {2}" -f $tag, $span, $parkfail.Line.Trim())
        return "FAIL"
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host ("  {0,-8}: exited {1} -- VOID" -f $tag, $LASTEXITCODE)
        Get-Content $err -Tail 3 | ForEach-Object { Write-Host "            $_" }
        return $null
    }
    if (Select-String -Path $out, $err -Pattern "NOT TRUSTWORTHY|CAP BREACH|ERROR: " -Quiet) {
        Write-Host ("  {0,-8}: tainted or a sequence failed -- VOID" -f $tag); return $null
    }
    $w = Select-String -Path $out -Pattern 'state bytes WRITTEN ([\d.]+) GB' | Select-Object -First 1
    if (-not $w) { Write-Host ("  {0,-8}: no rotate summary -- VOID" -f $tag); return $null }
    $written = [double]$w.Matches[0].Groups[1].Value
    if ($span -lt $Tokens -and $written -le 0) {
        Write-Host ("  {0,-8}: parked nothing (0 bytes written) -- proves nothing, VOID" -f $tag); return $null
    }
    $texts = SeqTexts $out
    if ($texts.Count -ne $nPrompts -or ($texts | Where-Object { [string]::IsNullOrWhiteSpace($_) })) {
        Write-Host ("  {0,-8}: expected {1} sequences with text, got {2} -- VOID" -f $tag, $nPrompts, $texts.Count)
        return $null
    }
    Write-Host ("  {0,-8}: --rotate {1,-2} state written {2:N3} GB" -f $tag, $span, $written)
    return ,$texts
}

$checked = 0; $failed = 0; $void = 0; $skipped = @()
foreach ($m in $models) {
    if ($Only -and $m.name -notlike "*$Only*") { continue }
    if (-not (Test-Path $m.path)) { Write-Host ("{0,-14}: SKIPPED (model absent)" -f $m.name); $skipped += $m.name; continue }
    Write-Host ("{0,-14}: {1}, cap {2}, {3} prompts at width 2, {4} tokens" -f $m.name, $m.state, $m.cap, $nPrompts, $Tokens)
    $legs = [ordered]@{}
    $ok = $true
    $legfail = $false
    foreach ($leg in @(@("control", $Tokens), @("park", 4), @("rerun", 4), @("offset", 5))) {
        $t = Run-Leg $m $leg[0] $leg[1]
        if ($t -is [string]) { $ok = $false; $legfail = $true; break }
        if ($null -eq $t) { $ok = $false; break }
        $legs[$leg[0]] = $t
    }
    if ($legfail) { Write-Host ("{0,-14}: FAIL  parking or restoring state failed" -f $m.name); $failed++; continue }
    if (-not $ok) { $void++; continue }
    $checked++
    $diffs = 0
    foreach ($tag in @("park", "rerun", "offset")) {
        for ($i = 0; $i -lt $nPrompts; $i++) {
            if ($legs[$tag][$i] -cne $legs["control"][$i]) {
                $diffs++
                Write-Host ("  *** seq {0}: {1} differs from control" -f $i, $tag)
                Write-Host ("      control: {0}" -f ($legs["control"][$i] -replace "`n", ' '))
                Write-Host ("      {0,-7}: {1}" -f $tag, ($legs[$tag][$i] -replace "`n", ' '))
            }
        }
    }
    if ($diffs -eq 0) { Write-Host ("{0,-14}: PASS  all {1} sequences identical across control, park, rerun, offset" -f $m.name, $nPrompts) }
    else { Write-Host ("{0,-14}: FAIL  {1} mismatches" -f $m.name, $diffs); $failed++ }
}
Write-Host ("models checked: {0}, failed: {1}, void: {2}, skipped: {3}{4}" -f $checked, $failed, $void, $skipped.Count,
    $(if ($skipped.Count) { " (" + ($skipped -join ", ") + ")" } else { "" }))
if ($failed -gt 0) { Write-Host "outputs kept in $work"; exit 1 }
if ($void -gt 0 -or $checked -eq 0) { Write-Host "outputs kept in $work"; exit 3 }
Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
exit 0
