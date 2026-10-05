# GPU gate: does --gpu on the streaming path compute what llama.cpp computes?
#
# WHY THIS EXISTS (2026-09-30): ggml's scheduler copies a GPU split's weights
# straight from host memory BEFORE any eval callback runs, so every streamed
# weight not yet materialised went to the GPU as poison. --gpu produced garbage
# on every streamed model -- exit 0, no failure counted -- for as long as the
# upstream copy path existed, and no gate ever ran --gpu. Fixed by the fork's
# copy callback (Streamer::copy_begin); this gate keeps it fixed.
#
# THE REFERENCE is the same GPU with llama.cpp owning the weights (resident
# mode; its repacked CPU kernels off via DRAY_NO_REPACK, and Vulkan kernel
# fusion off on both legs -- see Run-Leg): same device, same
# kernels, same graph -- only how the weights arrive
# differs, which is exactly what broke. Streaming must be BIT-IDENTICAL to it.
# (Not the CPU: CPU matmuls quantize activations to q8 and the GPU does not, so
# GPU and CPU legitimately flip near-ties -- measured on the 35B, full precision
# included. CPU agreement is reported, never required.)
#
# Models too big to be resident here get no reference: they must run clean (no
# failure, text produced) and their CPU agreement is reported. The prompt must
# exceed the GPU offload batch, or no GPU split is ever built.
#
# Needs a Vulkan build (-BinDir). Absent models are SKIPPED, a model this build
# cannot load is UNSUPPORTED, too little free memory SKIPS -- none of them pass.
# Exit 0 = every checked model passed; 1 = a failure; 3 = void.
#
#   scripts/gpugate.ps1 -BinDir build\dray-vk\bin

param(
    [string]$BinDir = (Join-Path $PSScriptRoot "..\build\dray-vk\bin"),
    [int]$Tokens = 12
)

$ErrorActionPreference = "Continue"
$bin = Join-Path $BinDir "dray.exe"
if (-not (Test-Path $bin)) { Write-Host "*** binary not found at $bin -- build with -Vulkan first. VOID ***"; exit 3 }
if (-not (Test-Path (Join-Path $BinDir "ggml-vulkan.dll"))) { Write-Host "*** $BinDir is not a Vulkan build. VOID ***"; exit 3 }

$work = Join-Path $env:TEMP "dray-gpugate"
if (Test-Path $work) { Remove-Item -Recurse -Force $work }
New-Item -ItemType Directory -Force $work | Out-Null
Get-ChildItem Env: | Where-Object { $_.Name -like "DRAY_*" -or $_.Name -like "GGML_VK_*" } | ForEach-Object { Remove-Item "Env:$($_.Name)" }

# Long enough for several prefill chunks; fixed text, so the three legs compare.
$promptFile = Join-Path $work "prompt.txt"
$readme = Get-Content -Raw (Join-Path $PSScriptRoot "..\README.md")
[IO.File]::WriteAllText($promptFile, $readme.Substring(0, [Math]::Min(9000, $readme.Length)))

# cap: the streaming leg's cap. resident: a cap the whole model fits (the
# reference); absent = too big to be resident on this machine.
$models = @(
    @{ name = "testbed-1b7b";      cap = "8G";  resident = "8G";  path = "D:\Models\testbed\OLMoE\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf" },
    @{ name = "qwen38-27b";        cap = "16G"; resident = "22G"; path = "D:\Models\unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-Q4_K_XL.gguf" },
    @{ name = "qwen36-35b-a3b";    cap = "12G"; resident = "28G"; path = "D:\Models\unsloth\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf" },
    @{ name = "deepseek-v4-flash"; cap = "24G"; path = "D:\Models\unsloth\DeepSeek-V4-Flash-0731-GGUF\UD-Q2_K_XL\DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf" },
    @{ name = "qwen38-flash-next"; cap = "36G"; kv = "q8"; path = "D:\Models\unsloth\Qwen3.8-Flash-Next-GGUF\UD-Q4_K_XL\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf" }
)

function Run-Leg($m, $leg, $cap, [switch]$Gpu, [switch]$Resident) {
    $out = Join-Path $work "$($m.name)-$leg.out"; $err = Join-Path $work "$($m.name)-$leg.err"
    $args = @("run", "-m", $m.path, "--cap", $cap, "--ctx", "8192", "--prefill-chunk", "512",
              "-n", "$Tokens", "--prompt-file", $promptFile)
    if (-not $Resident) { $args += "--force-stream" }
    # The reference runs without llama's repacked CPU kernels, which the
    # streaming buffer never reaches: same kernels on both legs, both devices.
    if ($Resident) { $env:DRAY_NO_REPACK = "1" } else { Remove-Item Env:DRAY_NO_REPACK -ErrorAction SilentlyContinue }
    # And without Vulkan's kernel fusion on either leg: the streaming path runs a
    # GPU split node by node (the eval callback claims nodes), so it never fuses,
    # while resident mode runs whole splits and does -- fused kernels round
    # differently (measured: testbed resident fused "> 9", unfused "> 24" ==
    # streaming). With fusion off both run the same kernels.
    $env:GGML_VK_DISABLE_FUSION = "1"; $env:GGML_VK_DISABLE_MULTI_ADD = "1"
    if ($m.kv) { $args += @("--kv", $m.kv) }
    if ($Gpu) { $args += "--gpu" }
    & $bin @args > $out 2> $err
    $code = $LASTEXITCODE
    Remove-Item Env:DRAY_NO_REPACK, Env:GGML_VK_DISABLE_FUSION, Env:GGML_VK_DISABLE_MULTI_ADD -ErrorAction SilentlyContinue
    $text = $null
    $lines = @(Get-Content $out)
    $at = ($lines | Select-String -Pattern "^what more RAM would buy" | Select-Object -First 1)
    if ($at) { $text = ($lines[$at.LineNumber..($lines.Count - 1)] | Where-Object { $_ -notmatch "^\s" }) -join "`n" }
    $fails = (Select-String -Path $err -Pattern "MATERIALISE FAIL|GET_TENSOR MISS" | Measure-Object).Count
    $unsupported = [bool](Select-String -Path $err -Pattern "unknown model architecture" -Quiet)
    $wasResident = [bool](Select-String -Path $err -Pattern "^resident mode:" -Quiet)
    return @{ code = $code; text = $text; fails = $fails; unsupported = $unsupported; resident = $wasResident }
}
function Clean($r) { return $r.code -eq 0 -and $r.fails -eq 0 -and $r.text }
function Agree($a, $b) {
    if ($a -eq $b) { return "identical" }
    $n = 0; while ($n -lt [Math]::Min($a.Length, $b.Length) -and $a[$n] -eq $b[$n]) { $n++ }
    return "differs after $n chars"
}

$fail = 0; $ran = 0; $skipped = @()
foreach ($m in $models) {
    if (-not (Test-Path $m.path)) { $skipped += $m.name; continue }
    $need = [Math]::Max([int]$m.cap.TrimEnd('G'), $(if ($m.resident) { [int]$m.resident.TrimEnd('G') } else { 0 })) + 4
    $free = [int]((Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue / 1024)
    if ($free -lt $need) { Write-Host ("{0,-18}: SKIPPED, {1} GB free, needs {2}" -f $m.name, $free, $need); $skipped += $m.name; continue }
    $gpu = Run-Leg $m "gpu-stream" $m.cap -Gpu
    if ($gpu.unsupported) { Write-Host ("{0,-18}: UNSUPPORTED by this build's llama.cpp base" -f $m.name); $skipped += $m.name; continue }
    $ran++
    $cpu = Run-Leg $m "cpu-stream" $m.cap
    $why = @()
    if (-not (Clean $gpu)) { $why += "gpu streaming: exit $($gpu.code), $($gpu.fails) failures" }
    if (-not (Clean $cpu)) { $why += "cpu streaming: exit $($cpu.code), $($cpu.fails) failures" }
    $verdict = "no resident reference (too big here)"
    if ($m.resident) {
        $ref = Run-Leg $m "gpu-resident" $m.resident -Gpu -Resident
        if (-not $ref.resident) { $why += "reference did not run resident at $($m.resident)" }
        elseif (-not (Clean $ref)) { $why += "gpu resident: exit $($ref.code), $($ref.fails) failures" }
        elseif ($gpu.text -ne $ref.text) { $why += "GPU streaming text differs from GPU resident: $(Agree $gpu.text $ref.text)" }
        else { $verdict = "BIT-IDENTICAL to GPU resident" }
    }
    if ($why.Count -gt 0) {
        $fail++
        Write-Host ("{0,-18}: FAIL  {1}" -f $m.name, ($why -join "; "))
        Write-Host "    gpu streaming : $($gpu.text)"
        if ($ref) { Write-Host "    gpu resident  : $($ref.text)" }
    } else {
        Write-Host ("{0,-18}: PASS  {1}; vs CPU: {2} (reported)" -f $m.name, $verdict, (Agree $gpu.text $cpu.text))
    }
    $ref = $null
}
Write-Host "models checked: $ran, failed: $fail, skipped: $($skipped.Count) $(if ($skipped) { '(' + ($skipped -join ', ') + ')' })"
if ($ran -eq 0) { Write-Host "*** nothing ran: VOID ***"; exit 3 }
if ($fail -gt 0) { exit 1 }
exit 0
