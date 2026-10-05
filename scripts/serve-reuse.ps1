# Conversation reuse gate: does a chat's next turn, which continues a kept
# conversation instead of prefilling it again, answer what a fresh server does?
#
#   Server A: turn 1, turn 2, turn 3 of one conversation (each request carries
#   the whole history, as every chat client sends it). Turns 2 and 3 should
#   reuse the kept conversation -- /health's kv_reuse must show it.
#   Server B, fresh for each: turn 2 alone, turn 3 alone -- nothing to reuse.
#
# Attention models reuse by cutting the KV back; recurrent/hybrid models by
# restoring a checkpoint. A reused prefix is prefilled in different chunks than
# a whole prompt, and chunked math can differ in the last bits (the batch
# class), so a difference is REPORTED with the shared prefix length and judged
# by eye; the gate FAILS only on errors or when no reuse happened at all.
#
#   scripts/serve-reuse.ps1 [-Model <gguf>] [-Cap 6G]

param(
    [string]$BinDir = (Join-Path $PSScriptRoot "..\build\dray\bin"),
    [int]$Port = 18093,
    [string]$Cap = "6G",
    [int]$Tokens = 16,
    [string]$Model = "D:\Models\testbed\OLMoE\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf",
    # Extra server flags for both servers (e.g. "--config-dir DIR" for a variant).
    [string]$ServeArgs = "",
    # Where to keep the results (default: %TEMP%\dray-serve-reuse).
    [string]$WorkDir = ""
)

$ErrorActionPreference = "Continue"
$bin = Join-Path $BinDir "dray.exe"
if (-not (Test-Path $bin)) { Write-Host "serve-reuse: no binary at $bin"; exit 2 }
$work = if ($WorkDir) { $WorkDir } else { Join-Path $env:TEMP "dray-serve-reuse" }
if (Test-Path $work) { Remove-Item -Recurse -Force $work }
New-Item -ItemType Directory -Force $work | Out-Null
Get-ChildItem Env: | Where-Object { $_.Name -like "DRAY_*" } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
$base = "http://127.0.0.1:$Port"

function Start-Server($tag, $parallel = 2) {
    $p = Start-Process -FilePath $bin -NoNewWindow -PassThru `
        -ArgumentList "serve -m `"$Model`" --cap $Cap --force-stream --ctx 1024 --port $Port --parallel $parallel $ServeArgs" `
        -RedirectStandardError (Join-Path $work "serve-$tag.log.txt") -RedirectStandardOutput (Join-Path $work "serve-$tag.out.txt")
    for ($i = 0; $i -lt 1200; $i++) {
        Start-Sleep -Milliseconds 500
        try { $h = Invoke-RestMethod "$base/health" -TimeoutSec 2; if ($h.status) { return $p } } catch {}
        if ($p.HasExited) { break }
    }
    Write-Host "serve-reuse: server $tag did not come up"; try { $p.Kill() } catch {}; exit 2
}
function Stop-Server($p) {
    try { Invoke-RestMethod -Method Post "$base/admin/shutdown" -ContentType "application/json" -Body "{}" -TimeoutSec 10 | Out-Null } catch {}
    $p.WaitForExit(300000) | Out-Null
    if (-not $p.HasExited) { $script:unclean++; try { $p.Kill() } catch {} }
}
$script:unclean = 0
function Ask($messages, $tag) {
    $bf = Join-Path $work "$tag.req.json"
    $body = @{ messages = $messages; max_tokens = $Tokens; temperature = 0 } | ConvertTo-Json -Depth 6 -Compress
    Set-Content -Path $bf -Value $body -Encoding utf8
    $of = Join-Path $work "$tag.json"
    $code = & curl.exe -s -o $of -w "%{http_code}" -H "Content-Type: application/json" --data-binary "@$bf" "$base/v1/chat/completions"
    if ($code -ne "200") { Write-Host "  $tag : HTTP $code"; return $null }
    return (Get-Content $of -Raw | ConvertFrom-Json).choices[0].message.content
}

$u1 = "Name three primary colors."
$u2 = "Which of them is the colour of the sky?"
$u3 = "And which one is the colour of grass when mixed with yellow?"

# --- server A: the whole conversation
$a = Start-Server "A"
$a1 = Ask @(@{ role = "user"; content = $u1 }) "A-t1"
$h2 = @(@{ role = "user"; content = $u1 }, @{ role = "assistant"; content = $a1 }, @{ role = "user"; content = $u2 })
$a2 = Ask $h2 "A-t2"
$h3 = $h2 + @(@{ role = "assistant"; content = $a2 }, @{ role = "user"; content = $u3 })
$a3 = Ask $h3 "A-t3"
$before = $null
try { $before = (Invoke-RestMethod "$base/health" -TimeoutSec 30).kv_reuse } catch {}
# Edited history: the client rewrites the assistant's turn-2 answer, so the
# prompt diverges INSIDE the kept conversation. Attention models cut the KV
# back; recurrent models must restore a checkpoint taken before turn 2's end.
$hE = $h2 + @(@{ role = "assistant"; content = "Blue." }, @{ role = "user"; content = $u3 })
$aE = Ask $hE "A-tE"
$reuse = $null
try { $reuse = (Invoke-RestMethod "$base/health" -TimeoutSec 30).kv_reuse } catch {}
Stop-Server $a

# --- server B, fresh per turn: nothing to reuse
$b = Start-Server "B2"; $r2 = Ask $h2 "B-t2"; Stop-Server $b
$b = Start-Server "B3"; $r3 = Ask $h3 "B-t3"; Stop-Server $b
$b = Start-Server "BE"; $rE = Ask $hE "B-tE"; Stop-Server $b

# --- the pool: ONE slot, two conversations. Y's turn parks X; X's next turn
# must come back from the pool and answer what a fresh server does.
$x1u = "List two planets."
$y1u = "Name a famous painter."
$x2u = "Which of them is bigger?"
$p = Start-Server "P" 1
$x1 = Ask @(@{ role = "user"; content = $x1u }) "P-x1"
$null = Ask @(@{ role = "user"; content = $y1u }) "P-y1"
$hx2 = @(@{ role = "user"; content = $x1u }, @{ role = "assistant"; content = $x1 }, @{ role = "user"; content = $x2u })
$x2 = Ask $hx2 "P-x2"
$pool = $null
try { $pool = (Invoke-RestMethod "$base/health" -TimeoutSec 30).kv_pool } catch {}
Stop-Server $p
$b = Start-Server "BP" 1; $rx2 = Ask $hx2 "BP-x2"; Stop-Server $b
# The control that decides whether the pool's restore is EXACT: the same
# continuation with X never leaving its slot. Must equal the pooled answer; a
# difference from the FRESH server is the decode-vs-prefill class, not the pool.
$c = Start-Server "SP" 1
$cx1 = Ask @(@{ role = "user"; content = $x1u }) "SP-x1"
$cx2 = Ask $hx2 "SP-x2"
Stop-Server $c

$fail = 0
$pr = if ($pool) { $pool.restored } else { 0 }
$pp = if ($pool) { $pool.parked } else { 0 }
Write-Host "  pool (1 slot): parked $pp, restored $pr"
if ($pr -lt 1) { Write-Host "  the pool never restored a conversation"; $fail++ }
if ($x2 -and $rx2) {
    if ($x2 -eq $rx2) { $v = "identical" } else {
        $n = 0; while ($n -lt [Math]::Min($x2.Length, $rx2.Length) -and $x2[$n] -eq $rx2[$n]) { $n++ }
        $v = "differs after $n chars"
    }
    Write-Host ("  {0,-7}: restored vs fresh -> {1}   [{2}]" -f "pooled", $v, (($x2 -replace "\s+", " ")))
} else { $fail++; Write-Host "  pooled : missing answer" }
if ($cx1 -ne $x1) { Write-Host "  control: turn 1 differs from the pooled run's turn 1 (cannot compare)"; $fail++ }
elseif ($cx2 -eq $x2) { Write-Host "  pooled : restored vs never-left-the-slot -> identical (the restore is exact)" }
else { Write-Host "  pooled : restored vs never-left-the-slot -> DIFFERS: the restore is not exact"; $fail++ }
foreach ($t in @(@("turn 2", $a2, $r2), @("turn 3", $a3, $r3), @("edited", $aE, $rE))) {
    $name = $t[0]; $x = $t[1]; $y = $t[2]
    if (-not $x -or -not $y) { $fail++; Write-Host "  $name : missing answer"; continue }
    if ($x -eq $y) { $v = "identical" } else {
        $n = 0; while ($n -lt [Math]::Min($x.Length, $y.Length) -and $x[$n] -eq $y[$n]) { $n++ }
        $v = "differs after $n chars"
    }
    Write-Host ("  {0,-7}: reused vs fresh -> {1}   [{2}]" -f $name, $v, (($x -replace "\s+", " ")))
}
$rq = if ($reuse) { $reuse.requests } else { 0 }
$rt = if ($reuse) { $reuse.tokens } else { 0 }
$eq = if ($reuse -and $before) { $reuse.tokens - $before.tokens } else { 0 }
Write-Host "  kv_reuse on server A: $rq request(s), $rt prompt tokens not prefilled again ($eq of them by the edited turn)"
Select-String -Path (Join-Path $work "serve-A.log.txt") -Pattern "conversation reuse" | Select-Object -First 1 | ForEach-Object { Write-Host "  $($_.Line)" }
if ($script:unclean -gt 0) { Write-Host "  $script:unclean server(s) did not shut down (killed)"; $fail++ }
if ($eq -le 0) { Write-Host "  the edited turn reused nothing: the cut / checkpoint path was not exercised"; $fail++ }
if ($fail -gt 0 -or $rq -lt 3) { Write-Host "serve-reuse : FAIL"; exit 1 }
Write-Host "serve-reuse : PASS"
exit 0
