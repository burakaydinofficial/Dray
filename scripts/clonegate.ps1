# Clone gate: does this repository build for someone who is not you?
#
# WHY THIS EXISTS (2026-08-23). The tree pinned the vendored fork at one commit
# while src/backend/stream_buffer.cpp called three functions that existed only
# in the working copy of the submodule. Every local build passed -- INCLUDING a
# deliberate from-scratch one, because from-scratch still compiles your working
# tree. A clone would have configured, compiled every translation unit, and died
# at the link. Nothing in difftest, residenttest, batchdiff or archgate can see
# that class of defect, because they all run against the binary you already have.
#
# It is the third false green of that day and they shared a shape: the exit code
# said fine while the ARTIFACT said otherwise (an empty comparison count, a
# three-day-old binary, a stale submodule SHA). So this gate checks artifacts.
#
# Run it before pushing anything that touches the submodule, the build files, or
# a symbol that crosses into the fork.
#
# Exit 0 = a stranger can build this. 1 = they cannot. 3 = the gate itself broke.

param(
    [string]$Scratch = "",
    [switch]$KeepOnFailure   # leave the clone behind so you can inspect it
)
$ErrorActionPreference = "Continue"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
if (-not $Scratch) { $Scratch = Join-Path ([System.IO.Path]::GetTempPath()) "dray-clonegate" }

function Fail([string]$msg, [int]$code = 1) {
    Write-Host "clone gate : *** $msg ***"
    if (-not $KeepOnFailure) { Remove-Item -Recurse -Force $Scratch -EA SilentlyContinue }
    else { Write-Host "  clone left at $Scratch" }
    exit $code
}

# Uncommitted submodule work is the exact defect this gate was written for, so
# say it plainly before spending minutes on a compile that would prove it anyway.
Push-Location $repo
$subDirty = git -C third_party/llama.cpp status --porcelain 2>$null
Pop-Location
if ($subDirty) {
    Write-Host "clone gate : *** SUBMODULE HAS UNCOMMITTED CHANGES ***"
    Write-Host "  A clone cannot see these. Commit and push them to the fork branch,"
    Write-Host "  then move the submodule pointer, or this tree builds only here:"
    $subDirty -split "`n" | Select-Object -First 8 | ForEach-Object { Write-Host "    $_" }
    exit 1
}

Remove-Item -Recurse -Force $Scratch -EA SilentlyContinue
git clone --recursive --depth 1 $repo $Scratch 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Fail "clone failed" 3 }

# The submodule must resolve from its REMOTE, not from a local path that only
# exists on this machine.
$sha = (git -C (Join-Path $Scratch "third_party/llama.cpp") rev-parse HEAD 2>$null)
if (-not $sha) { Fail "submodule did not check out in the clone" }
Write-Host "clone gate : submodule resolved at $($sha.Substring(0,9))"

Push-Location $Scratch
. .\scripts\msvc-env.ps1 | Out-Null
& $DRAY_CMAKE -S . -B build-clonegate -G Ninja -DCMAKE_BUILD_TYPE=Release `
    -DGGML_VULKAN=OFF -DGGML_CUDA=OFF "-DCMAKE_MAKE_PROGRAM=$DRAY_NINJA" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Pop-Location; Fail "configure failed in the clone" }
& $DRAY_CMAKE --build build-clonegate -j 12 2>&1 | Select-Object -Last 3 | Out-Null
$buildCode = $LASTEXITCODE
Pop-Location

# ARTIFACT, not exit code. This is the whole point of the gate.
$bin = Join-Path $Scratch "build-clonegate/bin/dray.exe"
if (-not (Test-Path $bin)) { Fail "build reported $buildCode but produced NO BINARY" }
$age = (Get-Date) - (Get-Item $bin).LastWriteTime
if ($age.TotalMinutes -gt 30) { Fail "binary exists but is $([int]$age.TotalMinutes) min old -- nothing was built" }

Push-Location $Scratch
& $DRAY_CTEST --test-dir build-clonegate --no-tests=error 2>&1 | Select-Object -Last 2 | Out-Null
$testCode = $LASTEXITCODE
Pop-Location
if ($testCode -ne 0) { Fail "clone builds but its tests fail" }

Remove-Item -Recurse -Force $Scratch -EA SilentlyContinue
Write-Host "clone gate : PASS - a fresh clone configures, links a binary, and passes its tests"
exit 0
