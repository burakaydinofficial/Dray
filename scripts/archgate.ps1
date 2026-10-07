# Architecture-diversity gate.
#
# WHY THIS EXISTS (2026-08-23): difftest, residenttest and batchdiff all passed
# on three models while a node-batching optimisation produced fluent, wrong text
# on DeepSeek V4 Flash -- whose hyper-connection weights arrive through node
# shapes the optimisation's predicate misjudged. Four green gates and a silently
# broken architecture. Correctness on ONE graph shape is not correctness.
#
# So this gate runs a short greedy generation on every architecture present on
# this machine and asserts the output still starts with what it produced when
# known good. It is SLOW (minutes per model, real disk) and is therefore not a
# pre-commit gate -- run it before shipping anything that touches the streamer's
# node handling, materialisation, or graph interception.
#
# A model that is absent is SKIPPED and reported as skipped, never silently
# passed. Exit 0 = every present model matched; 1 = a mismatch; 3 = void.

param(
    [string]$Cap = "12G",
    [int]$Tokens = 8,
    [switch]$Record,         # write current outputs as the new expectations
    [string]$Only = ""       # substring filter on model name
)
$ErrorActionPreference = "Continue"
$bin = Join-Path $PSScriptRoot "..\build\dray\bin\dray.exe"
# A missing binary must VOID, not pass. PowerShell leaves $LASTEXITCODE at its
# previous value when a command cannot be launched, so if that value happened to
# be 0 the exit-code guard below would pass, both legs would capture the same
# launcher error text, and the gate would report IDENTICAL. Same shape as the
# Linux build that reported green having compiled nothing (2026-08-24 audit).
if (-not (Test-Path $bin)) {
    Write-Host "*** binary not found at $bin -- build first. VOID ***"
    exit 3
}
$expectFile = Join-Path $PSScriptRoot "archgate-expected.json"
$prompt = "The capital of France is"

# One entry per ARCHITECTURE FAMILY, not per model -- the point is graph shape
# coverage. Add a row whenever a genuinely new mixer, attention or routing
# scheme lands.
$models = @(
    @{ name = "testbed-1b7b";  why = "dense-ish MoE control";
       path = "D:\Models\testbed\OLMoE\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf" },
    @{ name = "qwen38-27b";    why = "DENSE, gated deltanet + attention";
       path = "D:\Models\unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-Q4_K_XL.gguf" },
    @{ name = "qwen36-35b-a3b"; why = "sparse MoE, 256 experts top-8";
       path = "D:\Models\unsloth\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf" },
    @{ name = "deepseek-v4-flash"; why = "HYPER-CONNECTIONS + MLA, 256 experts top-6";
       path = "D:\Models\unsloth\DeepSeek-V4-Flash-0731-GGUF\UD-Q2_K_XL\DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf" },
    @{ name = "glm-5.3-flash"; why = "glm5next, 288 experts top-8 (new base only)";
       path = "D:\Models\unsloth\GLM-5.3-Flash-GGUF\UD-IQ1_M\GLM-5.3-Flash-UD-IQ1_M-00001-of-00003.gguf" },
    @{ name = "qwen38-flash-next"; why = "qwen4exp, 512 experts top-10 (new base only)";
       path = "D:\Models\unsloth\Qwen3.8-Flash-Next-GGUF\UD-Q4_K_XL\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf" }
)

$expected = @{}
if (Test-Path $expectFile) {
    (Get-Content $expectFile -Raw | ConvertFrom-Json).PSObject.Properties | ForEach-Object {
        $expected[$_.Name] = $_.Value
    }
}

$fail = 0; $ran = 0; $skipped = @()
$results = @{}

foreach ($m in $models) {
    if ($Only -and $m.name -notlike "*$Only*") { continue }
    if (-not (Test-Path $m.path)) { $skipped += $m.name; continue }
    $f = Join-Path $env:TEMP "archgate-$($m.name).txt"
    # --force-stream is MANDATORY here. Without it, any model that fits the cap
    # takes the resident path (engine.cpp: "EVERY correctness gate and
    # measurement must use it, or it silently stops testing this engine"), and
    # the 4.2 GB testbed control fits a 12 GiB cap comfortably -- so the row this
    # gate calls its control was exercising llama's allocator, not the streamer
    # the gate exists to protect (2026-08-24 audit).
    & $bin run -m $m.path --cap $Cap --threads 4 --ctx 512 --kv q4 --force-stream `
        -n $Tokens -p $prompt *> $f
    if ($LASTEXITCODE -ne 0) {
        Write-Host "$($m.name) : *** exited $LASTEXITCODE -- VOID *** ($($m.why))"
        exit 3
    }
    if (Select-String -Path $f -Pattern "NOT TRUSTWORTHY|CAP BREACH|REFUSED" -Quiet) {
        Write-Host "$($m.name) : *** TAINTED -- VOID ***"; exit 3
    }
    $raw = [System.IO.File]::ReadAllText($f) -replace "`r", ''
    $lines = $raw -split "`n"
    $p = -1
    for ($i = 0; $i -lt $lines.Count; $i++) { if ($lines[$i] -match "MiB") { $p = $i } }
    if ($p -lt 0) { Write-Host "$($m.name) : *** no readout anchor -- VOID ***"; exit 3 }
    # Keep newlines. difftest was deliberately changed from a space-join to this
    # because collapsing whitespace hid the trailing-newline and stop-boundary
    # class of difference; this gate had quietly reintroduced the weaker form.
    $text = ($lines[($p + 1)..($lines.Count - 1)] -join "`n").Trim("`n")
    if ([string]::IsNullOrWhiteSpace($text)) { Write-Host "$($m.name) : *** NO text -- VOID ***"; exit 3 }
    $results[$m.name] = $text
    $ran++

    if ($Record) { Write-Host "$($m.name) : recorded"; continue }
    # A model present but unrecorded is an UNKNOWN, not a pass. It used to
    # `continue` without touching $fail, so an architecture whose expectation had
    # been lost (see -Record above) sailed through this gate forever.
    if (-not $expected.ContainsKey($m.name) -or
        [string]::IsNullOrEmpty([string]$expected[$m.name])) {
        Write-Host "$($m.name) : *** NO EXPECTATION STORED -- cannot judge (run -Record) *** output: $($text.Substring(0,[Math]::Min(60,$text.Length)))"
        $fail++
        continue
    }
    # Ordinal, not culture-sensitive. The default StartsWith(String) overload
    # does a linguistic comparison, which can treat differing byte sequences as
    # equal -- the opposite of what a byte-identity gate wants.
    if ($text.StartsWith([string]$expected[$m.name], [System.StringComparison]::Ordinal)) {
        Write-Host "$($m.name) : MATCHES ($($m.why))"
    } else {
        Write-Host "$($m.name) : *** DIVERGED -- BUG *** ($($m.why))"
        Write-Host "   expected start : $($expected[$m.name])"
        Write-Host "   got            : $($text.Substring(0,[Math]::Min(80,$text.Length)))"
        $fail++
    }
}

if ($Record) {
    # MERGE, do not replace. Rebuilding the file from only the models present at
    # record time silently deleted the golden prefixes of absent ones -- and a
    # model with no stored expectation does not increment $fail below, so the
    # erased entry then passed forever. Recording on a machine missing two
    # models used to destroy two gates (2026-08-24 audit).
    $store = @{}
    foreach ($k in $expected.Keys) { $store[$k] = $expected[$k] }
    foreach ($k in $results.Keys) {
        $t = $results[$k]
        $store[$k] = $t.Substring(0, [Math]::Min(40, $t.Length))
    }
    if ($store.Count -eq 0) {
        Write-Host "*** nothing recorded and nothing previously stored -- refusing to write an empty expectations file ***"
        exit 3
    }
    $store | ConvertTo-Json | Set-Content $expectFile -Encoding utf8
    Write-Host "recorded $($results.Count) fresh, kept $($store.Count - $results.Count) existing, total $($store.Count)"
    exit 0
}

if ($skipped.Count -gt 0) { Write-Host "SKIPPED (absent): $($skipped -join ', ')" }
Write-Host "architectures checked: $ran, diverged: $fail"
if ($fail -gt 0) { exit 1 }
# Checking nothing is not passing. The header promises absent models are
# "reported as skipped, never silently passed" -- but the exit code did not
# distinguish 0 checked from 5 checked, so on any machine without the model
# tree (including the scratch clone clonegate builds) this gate was a green
# no-op to anything reading its status (2026-08-24 audit).
if ($ran -eq 0) {
    Write-Host "clone gate : *** NOTHING CHECKED -- no model present, this is not a pass ***"
    exit 3
}
exit 0
