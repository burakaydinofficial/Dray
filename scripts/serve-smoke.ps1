# Server smoke harness: every route, the background job lifecycle, crash-resume
# across a KILLED process (resumed text must equal the uninterrupted text), and
# graceful shutdown, on the testbed model. Writes normalized results (no ids, no
# times) to <OutDir>\serve.txt, so two binaries can be diffed line for line:
#
#   serve-smoke.ps1 -OutDir before
#   serve-smoke.ps1 -OutDir after
#   Compare-Object (Get-Content before\serve.txt) (Get-Content after\serve.txt)
param(
    [string]$BinDir = (Join-Path $PSScriptRoot "..\build\dray\bin"),
    [string]$OutDir,
    [int]$Port = 18089,
    [string]$Name = "dray",
    [string]$Model = "D:\Models\testbed\OLMoE\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"
)

$ErrorActionPreference = "Continue"
$bin = Join-Path $BinDir "$Name.exe"
$m = $Model
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$jobs = Join-Path $env:TEMP "dray-smoke-jobs"
Remove-Item -Recurse -Force $jobs -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $jobs | Out-Null
$base = "http://127.0.0.1:$Port"
$lines = New-Object System.Collections.Generic.List[string]
function Rec($k, $v) { $lines.Add("${k}: $v") }

[Environment]::SetEnvironmentVariable("$($Name.ToUpper())_CKPT_EVERY", "4")
function Start-Server($tag) {
    $a = "serve -m `"$m`" --cap 4G --force-stream --ctx 512 --port $Port --jobs-dir `"$jobs`""
    $p = Start-Process -FilePath $bin -ArgumentList $a -NoNewWindow -PassThru `
        -RedirectStandardError (Join-Path $OutDir "serve-$tag.log.txt") `
        -RedirectStandardOutput (Join-Path $OutDir "serve-$tag.out.log.txt")
    for ($i = 0; $i -lt 360; $i++) {
        Start-Sleep -Milliseconds 500
        $code = & curl.exe -s -o NUL -w "%{http_code}" "$base/health"
        if ($code -eq "200") { return $p }
        if ($p.HasExited) { break }
    }
    throw "server did not come up ($tag)"
}
function Post($path, $obj, [switch]$Raw) {
    $f = Join-Path $env:TEMP "dray-smoke-body.json"
    if ($Raw) { [IO.File]::WriteAllText($f, $obj) } else { [IO.File]::WriteAllText($f, ($obj | ConvertTo-Json -Depth 8 -Compress)) }
    $o = Join-Path $env:TEMP "dray-smoke-resp.txt"
    $code = & curl.exe -s -o $o -w "%{http_code}" -H "Content-Type: application/json" --data-binary "@$f" "$base$path"
    return @{ code = $code; body = [IO.File]::ReadAllText($o) }
}
function Get-Json($path) {
    $o = Join-Path $env:TEMP "dray-smoke-get.txt"
    $code = & curl.exe -s -o $o -w "%{http_code}" "$base$path"
    return @{ code = $code; json = ([IO.File]::ReadAllText($o) | ConvertFrom-Json) }
}
function Wait-Job($id, $minTokens = -1) {
    for ($i = 0; $i -lt 1200; $i++) {
        $r = Get-Json "/v1/responses/$id"
        $s = $r.json.status
        if ($minTokens -ge 0) {
            if ($s -eq "in_progress" -and $r.json.usage.completion_tokens -ge $minTokens) { return $r.json }
        }
        if ($s -in @("completed", "failed", "cancelled", "quarantined")) { return $r.json }
        Start-Sleep -Milliseconds 500
    }
    throw "job $id did not settle"
}
$msg = @(@{ role = "user"; content = "The capital of France is" })

$p = Start-Server "a"
$h = Get-Json "/health"
Rec "health.code" $h.code; Rec "health.status" $h.json.status; Rec "health.model" $h.json.model
$mo = Get-Json "/v1/models"
Rec "models.code" $mo.code; Rec "models.id" $mo.json.data[0].id; Rec "models.ctx" $mo.json.data[0].context_length

$r = Post "/v1/chat/completions" @{ messages = $msg; temperature = 0; max_tokens = 12 }
$j = $r.body | ConvertFrom-Json
Rec "chat.code" $r.code; Rec "chat.content" ($j.choices[0].message.content | ConvertTo-Json -Compress)
Rec "chat.finish" $j.choices[0].finish_reason; Rec "chat.usage" ($j.usage | ConvertTo-Json -Compress)
Rec "chat.template" $j.$Name.chat_template

# Streaming: collect deltas from the SSE frames.
$f = Join-Path $env:TEMP "dray-smoke-body.json"
[IO.File]::WriteAllText($f, (@{ messages = $msg; temperature = 0; max_tokens = 12; stream = $true } | ConvertTo-Json -Depth 8 -Compress))
$sse = & curl.exe -s -N -H "Content-Type: application/json" --data-binary "@$f" "$base/v1/chat/completions"
$acc = ""; $fin = ""; $frames = 0
foreach ($l in ($sse -split "`n")) {
    if ($l -match '^data: (.*)$') {
        $d = $Matches[1].Trim(); if ($d -eq "[DONE]") { Rec "sse.done" "yes"; continue }
        $frames++; $o = $d | ConvertFrom-Json
        if ($o.choices[0].delta.content) { $acc += $o.choices[0].delta.content }
        if ($o.choices[0].finish_reason) { $fin = $o.choices[0].finish_reason }
    }
}
Rec "sse.content" ($acc | ConvertTo-Json -Compress); Rec "sse.finish" $fin
Rec "sse.matches_chat" ($acc -eq $j.choices[0].message.content)

$r = Post "/v1/chat/completions" @{ messages = $msg; tools = @(@{ type = "function" }) }
Rec "tools.code" $r.code
$r = Post "/v1/chat/completions" "{not json" -Raw
Rec "badjson.code" $r.code
$r = Post "/v1/chat/completions" @{ temperature = 0 }
Rec "nomessages.code" $r.code
$r = Get-Json "/v1/responses/does-not-exist"
Rec "unknownjob.code" $r.code

# Background job, uninterrupted: the reference text for the resume check.
$r = Post "/v1/chat/completions" @{ messages = $msg; temperature = 0; max_tokens = 40; background = $true }
Rec "bg.submit.code" $r.code
$id = ($r.body | ConvertFrom-Json).id
$done = Wait-Job $id
Rec "bg.status" $done.status; Rec "bg.tokens" $done.usage.completion_tokens
Rec "bg.text" ($done.output_text | ConvertTo-Json -Compress)
$reference = $done.output_text

# Crash-resume: same job, kill the server mid-generation, restart, finish.
$r = Post "/v1/chat/completions" @{ messages = $msg; temperature = 0; max_tokens = 40; background = $true }
$id2 = ($r.body | ConvertFrom-Json).id
$mid = Wait-Job $id2 16
Rec "crash.midstatus" $mid.status
Stop-Process -Id $p.Id -Force
Start-Sleep -Seconds 2
$sidecars = @(Get-ChildItem $jobs -Filter *.job).Count
Rec "crash.sidecars_after_kill" $sidecars
$p = Start-Server "b"
$fin2 = Wait-Job $id2
Rec "resume.status" $fin2.status; Rec "resume.tokens" $fin2.usage.completion_tokens
Rec "resume.text" ($fin2.output_text | ConvertTo-Json -Compress)
Rec "resume.identical_to_uninterrupted" ($fin2.output_text -eq $reference)

# Cancel a queued job, then delete it.
$r = Post "/v1/chat/completions" @{ messages = $msg; temperature = 0; max_tokens = 200; background = $true }
$id3 = ($r.body | ConvertFrom-Json).id
Start-Sleep -Seconds 2
$c = Post "/v1/responses/$id3/cancel" @{}
Rec "cancel.code" $c.code
$cj = Wait-Job $id3
Rec "cancel.status" $cj.status
$o = Join-Path $env:TEMP "dray-smoke-del.txt"
$dc = & curl.exe -s -o $o -w "%{http_code}" -X DELETE "$base/v1/responses/$id3"
Rec "delete.code" $dc
Rec "delete.then_get" (Get-Json "/v1/responses/$id3").code

# Graceful shutdown.
$s = Post "/admin/shutdown" @{}
Rec "shutdown.code" $s.code
$p.WaitForExit(120000) | Out-Null
Rec "shutdown.exited" $p.HasExited
Rec "shutdown.exit_code" $p.ExitCode
Rec "shutdown.sidecars_left" (@(Get-ChildItem $jobs -Filter *.job).Count)

$lines | Set-Content (Join-Path $OutDir "serve.txt")
$lines | ForEach-Object { Write-Host $_ }
