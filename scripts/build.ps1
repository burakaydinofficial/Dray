# Configure and build dray. Imports the MSVC environment first, so this works
# from a plain PowerShell prompt with nothing installed beyond Visual Studio.
#
#   .\scripts\build.ps1              # configure if needed, then build (CPU-only)
#   .\scripts\build.ps1 -Vulkan      # build with the Vulkan backend, for --gpu
#   .\scripts\build.ps1 -Fresh       # wipe the build dir first
#   .\scripts\build.ps1 -Target dray
#   .\scripts\build.ps1 -Test        # build, then run ctest

param(
    [switch]$Fresh,
    [switch]$Test,
    [switch]$Vulkan,           # compile the Vulkan backend in (costs ~11% on CPU runs)
    [string]$Target = "",
    [string]$BuildDir = "build/dray",
    [string]$Config = "RelWithDebInfo"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'msvc-env.ps1')
Set-Location $root

# -Vulkan needs the SDK; find it even when this shell predates its install (the
# installer sets machine env, not session env).
if ($Vulkan -and -not $env:VULKAN_SDK -and (Test-Path 'C:\VulkanSDK')) {
    $sdk = Get-ChildItem 'C:\VulkanSDK' -Directory | Sort-Object Name -Descending | Select-Object -First 1
    if ($sdk) { $env:VULKAN_SDK = $sdk.FullName }
}
$want = if ($Vulkan) { "ON" } else { "OFF" }

if ($Fresh -and (Test-Path $BuildDir)) {
    Write-Host "removing $BuildDir"
    Remove-Item -Recurse -Force $BuildDir
}

# CMake remembers options per build dir, so an existing dir keeps whatever it was
# configured with. Reconfigure whenever the cached choice differs from this call's.
$cache = Join-Path $BuildDir 'CMakeCache.txt'
$have = if (Test-Path $cache) {
    (Select-String -Path $cache -Pattern '^DRAY_VULKAN_BUILD:BOOL=(\w+)' |
        Select-Object -First 1 | ForEach-Object { $_.Matches[0].Groups[1].Value })
} else { $null }
if (-not (Test-Path (Join-Path $BuildDir 'build.ninja')) -or $have -ne $want) {
    Write-Host "--- configure (Vulkan $want) ---"
    # Arg array rather than backtick continuations: a mangled continuation silently
    # passes a literal '$Config' and the failure surfaces much later as a ninja
    # lexing error, which is a miserable thing to debug.
    $cfgArgs = @(
        '-S', '.',
        '-B', $BuildDir,
        '-G', 'Ninja',
        "-DCMAKE_MAKE_PROGRAM=$DRAY_NINJA",
        "-DCMAKE_BUILD_TYPE=$Config",
        "-DDRAY_VULKAN_BUILD=$want"
    )
    & $DRAY_CMAKE @cfgArgs
    if ($LASTEXITCODE -ne 0) { throw "configure failed" }
}

Write-Host "--- build ---"
$sw = [Diagnostics.Stopwatch]::StartNew()
if ($Target) {
    & $DRAY_CMAKE --build $BuildDir --target $Target
} else {
    & $DRAY_CMAKE --build $BuildDir
}
$code = $LASTEXITCODE
Write-Host ("--- build exit {0} in {1:N1} min ---" -f $code, $sw.Elapsed.TotalMinutes)
if ($code -ne 0) { throw "build failed" }

if ($Test) {
    Write-Host "--- ctest ---"
    Push-Location $BuildDir
    # --no-tests=error: ctest's legacy default is to print "No tests were found"
    # and exit 0. Any accident that drops the tests subdirectory (a BUILD_TESTING
    # cache entry, a moved CMakeLists) would otherwise turn this gate, and
    # clonegate's "clone builds but its tests fail" check, into unconditional
    # greens (2026-08-24 audit).
    & $DRAY_CTEST --output-on-failure --no-tests=error
    $tc = $LASTEXITCODE
    Pop-Location
    if ($tc -ne 0) { throw "tests failed" }
}
