# Export a CLEAN-HISTORY copy of the current tree for the public repository
# (owner decision 2026-08-20: private history stays private and unrewritten;
# the public repo is born from the current tree with a single initial commit).
#
#   .\scripts\export-public.ps1 -Target D:\Projects\AI\dray-public
#
# What it does, exactly:
#   1. `git archive HEAD` -> the target dir: tracked files only, so nothing
#      gitignored, nothing untracked, and no .git history can leak.
#   2. Re-adds the llama.cpp fork as a submodule pinned to the SAME SHA the
#      private tree pins (git archive cannot carry submodule gitlinks).
#   3. Creates the single initial commit. Nothing is pushed -- inspect first,
#      then: git remote add origin <public-url>; git push -u origin master.
#
# Release-zip note (measured 2026-08-21): dray.exe links the CRT
# dynamically (MSVCP140/VCRUNTIME140*). Windows release zips should bundle
# those three DLLs from the VS redist folder, or the release notes must name
# the VC++ x64 redistributable.

param(
    [Parameter(Mandatory = $true)][string]$Target
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (Test-Path $Target) { throw "target exists: $Target -- refusing to overwrite" }

# The submodule pin, read from the private tree before anything else.
$subLine = (git submodule status third_party/llama.cpp).Trim()
$subSha = $subLine.Split(' ')[0].TrimStart('-', '+')
$subUrl = (git config -f .gitmodules submodule."third_party/llama.cpp".url)
$subBranch = (git config -f .gitmodules submodule."third_party/llama.cpp".branch)
if (-not $subSha -or -not $subUrl) { throw "could not read submodule pin" }
Write-Host "fork pin: $subUrl @ $subSha (branch $subBranch)"

New-Item -ItemType Directory -Path $Target | Out-Null
# No `git archive | tar` pipe: PowerShell 5.1 pipelines are text-shaped and
# corrupt binary streams. Write the tar, extract it, delete it.
$tarTmp = Join-Path $env:TEMP "dray-export.tar"
git archive HEAD -o $tarTmp
if ($LASTEXITCODE -ne 0) { throw "git archive failed" }
tar -xf $tarTmp -C $Target
if ($LASTEXITCODE -ne 0) { throw "tar extraction failed" }
Remove-Item $tarTmp

Push-Location $Target
try {
    # Native git writes progress to stderr; under ErrorActionPreference=Stop
    # PowerShell 5.1 converts that into a terminating error mid-script. Git
    # success is judged by exit codes here, not by stream silence.
    $ErrorActionPreference = "Continue"
    git init -q
    if ($LASTEXITCODE -ne 0) { throw "git init failed" }
    # The archive leaves third_party/llama.cpp as an empty dir; the submodule
    # must be a real gitlink at the SAME pin.
    if (Test-Path third_party/llama.cpp) { Remove-Item -Recurse -Force third_party/llama.cpp }
    git submodule add -b $subBranch $subUrl third_party/llama.cpp 2>&1 | Out-Null
    Push-Location third_party/llama.cpp
    git checkout -q $subSha
    Pop-Location
    git add -A
    git -c user.name="Burak Aydın" commit -q -m "dray: initial public release

Run frontier-scale MoE models from an SSD, on a polite slice of RAM.
Measured status, design notes and the lab notebook are in-repo: README.md,
CLAUDE.md, DECISIONS.md."
    Write-Host "exported: $(git rev-parse --short HEAD) in $Target"
    Write-Host "next: git remote add origin <public-url>; git push -u origin master; tag; release"
} finally {
    Pop-Location
}
