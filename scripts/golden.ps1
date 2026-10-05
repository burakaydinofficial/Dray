# Golden-output harness: proves a change did not change behaviour.
#
# Runs a fixed suite on the testbed model with one binary and records, per case:
#   <case>.out   stdout (plan report + generated text; deterministic)
#   <case>.sig   exit code + the deterministic streamer counters (bytes, nodes,
#                compaction, unlocks, eslot hits, hit rates)
# The suite covers every command (plan, run, batch, rotate, snaptest, refusal)
# and one run per streamer lever (NO_FUSE, NO_EARLY, NO_RETAIN, NO_ESLOTS,
# NO_REUSE, COMPACT_ALL, NO_ROWSLICE, FAST_NODES, RING_MB=0, NO_POOL, IO_THREAD=0):
# a lever not exercised by a gate is a path nobody checked.
#
#   golden.ps1 -OutDir before            # with the binary you trust
#   golden.ps1 -OutDir after             # after the change
#   golden.ps1 -Compare before,after     # byte-for-byte; exit 1 on any difference
#
# Record the baseline TWICE and compare the two first: anything that differs
# between identical binaries is noise, not signal. Wall-clock lines and the
# memory ledger (which follows RSS) are already excluded for that reason.
param(
    [string]$BinDir = (Join-Path $PSScriptRoot "..\build\dray\bin"),
    [string]$OutDir,
    [string[]]$Compare,
    [string]$Model = "D:\Models\testbed\OLMoE\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf",
    [string]$Exe = "dray.exe",
    [string]$EnvPrefix = "DRAY_",
    [string[]]$Only            # run just these case names
)

$ErrorActionPreference = "Continue"

if ($Compare) {
    $a = $Compare[0]; $b = $Compare[1]; $bad = 0
    foreach ($f in (Get-ChildItem $a -File | Where-Object { $_.Name -like '*.out' -or $_.Name -like '*.sig' })) {
        $o = Join-Path $b $f.Name
        if (-not (Test-Path $o)) { Write-Host "MISSING  $($f.Name)"; $bad++; continue }
        if ((Get-FileHash $f.FullName).Hash -ne (Get-FileHash $o).Hash) {
            Write-Host "DIFFERS  $($f.Name)"; $bad++
            Compare-Object (Get-Content $f.FullName) (Get-Content $o) | Select-Object -First 6 | Format-Table -AutoSize | Out-String | Write-Host
        } else { Write-Host "same     $($f.Name)" }
    }
    if ($bad) { Write-Host "GOLDEN: $bad difference(s)"; exit 1 }
    Write-Host "GOLDEN: identical"; exit 0
}

$bin = Join-Path $BinDir $Exe
if (-not (Test-Path $bin)) { Write-Host "no binary at $bin"; exit 3 }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$m = $Model
$p = "The capital of France is"

$cases = [ordered]@{
    "plan"          = @{ a = @("plan", "-m", $m, "--cap", "4G", "--ctx", "2048") }
    "plan_batch"    = @{ a = @("plan", "-m", $m, "--cap", "4G", "--ctx", "512", "--batch", "8") }
    "run_stream"    = @{ a = @("run", "-m", $m, "--cap", "4G", "--force-stream", "--ctx", "512", "-n", "24", "-p", $p) }
    "run_tight"     = @{ a = @("run", "-m", $m, "--cap", "2G", "--force-stream", "--ctx", "512", "-n", "24", "-p", $p) }
    "run_nocompact" = @{ a = @("run", "-m", $m, "--cap", "8G", "--force-stream", "--ctx", "512", "-n", "16", "-p", $p); env = @{ DRAY_NO_COMPACT = "1" } }
    "run_resident"  = @{ a = @("run", "-m", $m, "--cap", "8G", "--ctx", "512", "-n", "24", "-p", $p) }
    "run_temp"      = @{ a = @("run", "-m", $m, "--cap", "4G", "--force-stream", "--ctx", "512", "-n", "24", "--temp", "--seed", "7", "-p", $p) }
    "run_kvq4"      = @{ a = @("run", "-m", $m, "--cap", "4G", "--force-stream", "--ctx", "1024", "--kv", "q4", "-n", "16", "-p", $p) }
    "run_stop"      = @{ a = @("run", "-m", $m, "--cap", "4G", "--force-stream", "--ctx", "512", "-n", "32", "--stop", "Euro", "-p", $p) }
    "run_refused"   = @{ a = @("run", "-m", $m, "--cap", "256M", "--force-stream", "--ctx", "512", "-n", "8", "-p", $p) }
    "batch4"        = @{ a = @("batch", "-m", $m, "--cap", "4G", "--force-stream", "--ctx", "512", "--batch", "4", "-n", "12", "-p", $p) }
    "snaptest"      = @{ a = @("snaptest", "-m", $m, "--cap", "4G", "--force-stream", "--ctx", "512", "-p", $p) }
    "rotate"        = @{ a = @("batch", "-m", $m, "--cap", "4G", "--force-stream", "--ctx", "512", "--batch", "2", "--prompts", "$PSScriptRoot\golden-prompts.txt", "-n", "10", "--rotate", "10", "--state-dir", "$env:TEMP\dray-golden-state") }
    "batch_prompts" = @{ a = @("batch", "-m", $m, "--cap", "4G", "--force-stream", "--ctx", "512", "--batch", "4", "--prompts", "$PSScriptRoot\golden-prompts.txt", "-n", "10", "--stop", "Paris") }
}
# One case per streamer lever, so every path the streamer can take has a baseline.
$levers = [ordered]@{
    "lv_nofuse" = @{ NO_FUSE = "1" };      "lv_noearly" = @{ NO_EARLY = "1" }
    "lv_noretain" = @{ NO_RETAIN = "1" };  "lv_noeslots" = @{ NO_ESLOTS = "1" }
    "lv_noreuse" = @{ NO_REUSE = "1" };    "lv_compactall" = @{ COMPACT_ALL = "1" }
    "lv_norowslice" = @{ NO_ROWSLICE = "1" }; "lv_fastnodes" = @{ FAST_NODES = "1" }
    "lv_noring" = @{ RING_MB = "0" };      "lv_nopool" = @{ NO_POOL = "1" }
    "lv_noiothread" = @{ IO_THREAD = "0" }
}
foreach ($ln in $levers.Keys) {
    $e = @{}; foreach ($k in $levers[$ln].Keys) { $e["$EnvPrefix$k"] = $levers[$ln][$k] }
    $cases[$ln] = @{ a = @("run", "-m", $m, "--cap", $(if ($ln -eq "lv_compactall") { "8G" } else { "2G" }), "--force-stream", "--ctx", "512", "-n", "16", "-p", $p); env = $e }
}
# The NO_COMPACT case predates the prefix parameter.
$cases["run_nocompact"].env = @{ "${EnvPrefix}NO_COMPACT" = "1" }

foreach ($name in @($cases.Keys)) {
    if ($Only -and ($Only -notcontains $name)) { continue }
    $c = $cases[$name]
    $saved = @{}
    if ($c.env) { foreach ($k in $c.env.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $c.env[$k]) } }
    $out = Join-Path $OutDir "$name.out"
    $err = Join-Path $OutDir "$name.stderr.log"
    # Start-Process redirects natively: PowerShell 5.1's own 2> wraps long
    # native lines as ErrorRecords and re-encodes stdout, both of which corrupt
    # the comparison.
    $argline = ($c.a | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
    $proc = Start-Process -FilePath $bin -ArgumentList $argline -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $code = $proc.ExitCode
    if ($c.env) { foreach ($k in $c.env.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) } }
    # Wall-clock lines are noise by the project's own rule ("bytes decide").
    # The memory ledger follows RSS, so stdout is cut at its header.
    $keep = @(); foreach ($l in (Get-Content $out)) { if ($l -match '^resident memory ledger') { break }; if ($l -notmatch '^\s*wall\s') { $keep += $l } }
    $keep | Set-Content $out
    # Deterministic counters only: bytes, nodes, compaction, unlocks, eslot hits,
    # hit rates. Budget and RSS figures move with the OS and are excluded.
    $sig = @("exit=$code")
    $line = (Select-String -Path $err -Pattern 'streamed over' | Select-Object -Last 1).Line
    if ($line) {
        foreach ($rx in 'streamed over \d+ nodes', '[\d.]+ GB streamed', '\d+ expert-compacted', '\d+ early-unlocked', '\d+ eslot hits', 'h_routed=\S+', 'h_bytes=\S+') {
            $mm = [regex]::Match($line, $rx); if ($mm.Success) { $sig += $mm.Value }
        }
    }
    foreach ($ln in (Select-String -Path $err -Pattern 'REFUSED|NOT TRUSTWORTHY|CAP BREACH|SNAPSHOT ROUND-TRIP|STAMP REFUSAL|decode bytes|prefill bytes')) {
        $sig += ($ln.Line -replace '\s+', ' ').Trim()
    }
    $sig | Set-Content (Join-Path $OutDir "$name.sig")
    Write-Host ("{0,-14} exit={1}" -f $name, $code)
}
# stderr logs are for humans, not for comparison
Get-ChildItem $OutDir -Filter *.stderr.log | ForEach-Object { Move-Item $_.FullName ($_.FullName -replace '\.stderr\.log$', '.log.txt') -Force }
