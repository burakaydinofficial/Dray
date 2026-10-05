# Parallel serving gate: many clients at once through the scheduler.
#
#   8 different prompts fired together at a server with --parallel 4, so four
#   run while four wait for a slot; then each prompt again ALONE, as the serial
#   reference. Checks: every request answers 200 with a clean finish; /health
#   shows the load while it runs; and each concurrent answer against its
#   serial reference.
#
# Batched math differs from single-stream math in the last bits (GEMM vs GEMV
# accumulation order -- the documented batch class, see batchdiff.ps1), and the
# batch a request lands in depends on arrival timing. So an answer that differs
# from its serial reference is REPORTED with the length of the shared prefix
# and judged by eye; only failures, errors and empty answers fail the gate.
#
#   scripts/serve-parallel.ps1 [-Parallel 4] [-Model <gguf>]

param(
    [string]$BinDir = (Join-Path $PSScriptRoot "..\build\dray\bin"),
    [int]$Port = 18091,
    [int]$Parallel = 4,
    [int]$Tokens = 24,
    [string]$Model = "D:\Models\testbed\OLMoE\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"
)

$ErrorActionPreference = "Continue"
$bin = Join-Path $BinDir "dray.exe"
if (-not (Test-Path $bin)) { Write-Host "serve-parallel: no binary at $bin"; exit 2 }
$work = Join-Path $env:TEMP "dray-serve-parallel"
if (Test-Path $work) { Remove-Item -Recurse -Force $work }
New-Item -ItemType Directory -Force $work | Out-Null
Get-ChildItem Env: | Where-Object { $_.Name -like "DRAY_*" } | ForEach-Object { Remove-Item "Env:$($_.Name)" }

$prompts = @(
    "What is the capital of France?",
    "Name the largest planet in the solar system.",
    "At what temperature does water boil at sea level?",
    "Who wrote Hamlet?",
    "What gas do plants absorb from the air?",
    "How many continents are there?",
    "What is the chemical symbol for gold?",
    "Which ocean is the largest?"
)
$base = "http://127.0.0.1:$Port"

$srv = Start-Process -FilePath $bin -NoNewWindow -PassThru `
    -ArgumentList "serve -m `"$Model`" --cap 6G --force-stream --ctx 512 --port $Port --parallel $Parallel" `
    -RedirectStandardError (Join-Path $work "serve.log.txt") -RedirectStandardOutput (Join-Path $work "serve.out.txt")
$up = $false
for ($i = 0; $i -lt 600; $i++) {
    Start-Sleep -Milliseconds 500
    try { $h = Invoke-RestMethod "$base/health" -TimeoutSec 2; if ($h.status) { $up = $true; break } } catch {}
    if ($srv.HasExited) { break }
}
if (-not $up) { Write-Host "serve-parallel: server did not come up"; try { $srv.Kill() } catch {}; exit 2 }

function Body($p) {
    return (@{ messages = @(@{ role = "user"; content = $p }); max_tokens = $Tokens; temperature = 0 } | ConvertTo-Json -Depth 5 -Compress)
}
function Fire($i, $p, $tag) {
    $bf = Join-Path $work "$tag-$i.req.json"
    Set-Content -Path $bf -Value (Body $p) -Encoding ascii
    $of = Join-Path $work "$tag-$i.json"
    return Start-Process -FilePath "curl.exe" -NoNewWindow -PassThru `
        -ArgumentList "-s -o `"$of`" -w %{http_code} -H `"Content-Type: application/json`" --data-binary `"@$bf`" $base/v1/chat/completions" `
        -RedirectStandardOutput (Join-Path $work "$tag-$i.code.txt")
}

# --- all at once
$procs = @()
for ($i = 0; $i -lt $prompts.Count; $i++) { $procs += Fire $i $prompts[$i] "par" }
Start-Sleep -Milliseconds 700
$peak = 0; $waited = 0
for ($k = 0; $k -lt 40; $k++) {
    try {
        $h = Invoke-RestMethod "$base/health" -TimeoutSec 5
        if ($h.requests.active -gt $peak) { $peak = $h.requests.active }
        if ($h.requests.waiting -gt $waited) { $waited = $h.requests.waiting }
    } catch {}
    if (($procs | Where-Object { -not $_.HasExited }).Count -eq 0) { break }
    Start-Sleep -Milliseconds 500
}
$procs | ForEach-Object { $_.WaitForExit() }

# --- each alone
for ($i = 0; $i -lt $prompts.Count; $i++) { (Fire $i $prompts[$i] "ser").WaitForExit() }

try { Invoke-RestMethod -Method Post "$base/admin/shutdown" -ContentType "application/json" -Body "{}" -TimeoutSec 10 | Out-Null } catch {}
$srv.WaitForExit(120000) | Out-Null
$clean = $srv.HasExited
if (-not $clean) { try { $srv.Kill() } catch {} }

# --- judge
$fail = 0; $same = 0
for ($i = 0; $i -lt $prompts.Count; $i++) {
    $pc = (Get-Content (Join-Path $work "par-$i.code.txt") -Raw).Trim()
    $sc = (Get-Content (Join-Path $work "ser-$i.code.txt") -Raw).Trim()
    $pj = $null; $sj = $null
    try { $pj = Get-Content (Join-Path $work "par-$i.json") -Raw | ConvertFrom-Json } catch {}
    try { $sj = Get-Content (Join-Path $work "ser-$i.json") -Raw | ConvertFrom-Json } catch {}
    $pt = if ($pj) { $pj.choices[0].message.content } else { "" }
    $st = if ($sj) { $sj.choices[0].message.content } else { "" }
    $pf = if ($pj) { $pj.choices[0].finish_reason } else { "" }
    $ok = ($pc -eq "200") -and ($sc -eq "200") -and $pt -and ($pf -eq "stop" -or $pf -eq "length")
    if (-not $ok) { $fail++ }
    if ($pt -eq $st) {
        $same++
        $verdict = "identical"
    } else {
        $n = 0; while ($n -lt [Math]::Min($pt.Length, $st.Length) -and $pt[$n] -eq $st[$n]) { $n++ }
        $verdict = "differs after $n chars"
    }
    $show = ($pt -replace "\s+", " ")
    if ($show.Length -gt 60) { $show = $show.Substring(0, 60) }
    Write-Host ("  [{0}] http {1}/{2} {3,-7} {4,-24} {5}" -f $i, $pc, $sc, $pf, $verdict, $show)
}
Write-Host "  /health under load: peak active $peak, peak waiting $waited (slots: $Parallel)"
Write-Host "  identical to serial: $same of $($prompts.Count)"
if (-not $clean) { Write-Host "  server did not shut down within 2 min (killed)"; $fail++ }
if ($fail -gt 0 -or $peak -lt 2) {
    Write-Host "serve-parallel : FAIL ($fail failed request(s); peak active $peak)"
    exit 1
}
Write-Host "serve-parallel : PASS (all $($prompts.Count) answered; ran $peak at once)"
exit 0
