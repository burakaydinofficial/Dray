# Configure and build dray. Imports the MSVC environment first, so this works
# from a plain PowerShell prompt with nothing installed beyond Visual Studio.
#
#   .\scripts\build.ps1              # configure if needed, then build
#   .\scripts\build.ps1 -Fresh       # wipe the build dir first
#   .\scripts\build.ps1 -Target dray
#   .\scripts\build.ps1 -Test        # build, then run ctest

param(
    [switch]$Fresh,
    [switch]$Test,
    [string]$Target = "",
    [string]$BuildDir = "build/dray",
    [string]$Config = "RelWithDebInfo"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'msvc-env.ps1')
Set-Location $root

# GPU runtime opt-in needs the backend compiled in; find the Vulkan SDK even
# when this shell predates its install (the installer sets machine env, not
# session env). Absent SDK = CPU-only build, silently correct.
if (-not $env:VULKAN_SDK -and (Test-Path 'C:\VulkanSDK')) {
    $sdk = Get-ChildItem 'C:\VulkanSDK' -Directory | Sort-Object Name -Descending | Select-Object -First 1
    if ($sdk) { $env:VULKAN_SDK = $sdk.FullName }
}

if ($Fresh -and (Test-Path $BuildDir)) {
    Write-Host "removing $BuildDir"
    Remove-Item -Recurse -Force $BuildDir
}

if (-not (Test-Path (Join-Path $BuildDir 'build.ninja'))) {
    Write-Host "--- configure ---"
    # Arg array rather than backtick continuations: a mangled continuation silently
    # passes a literal '$Config' and the failure surfaces much later as a ninja
    # lexing error, which is a miserable thing to debug.
    $cfgArgs = @(
        '-S', '.',
        '-B', $BuildDir,
        '-G', 'Ninja',
        "-DCMAKE_MAKE_PROGRAM=$DRAY_NINJA",
        "-DCMAKE_BUILD_TYPE=$Config"
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
