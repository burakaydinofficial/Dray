# Imports the MSVC x64 build environment into the current PowerShell session and
# exposes $DRAY_CMAKE / $DRAY_NINJA. Nothing is installed: Visual Studio 2022
# Community ships CMake and Ninja under its own root, so the build has no external
# dependencies beyond VS itself.
#
#   . .\scripts\msvc-env.ps1

$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) { throw "vswhere not found - is Visual Studio installed?" }

$vsRoot = & $vswhere -latest -products * -property installationPath
if (-not $vsRoot) { throw "no Visual Studio installation found" }

$vcvars = Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

# vcvars only exports into cmd, so run it there and copy the resulting environment.
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        Set-Item -Path ("env:" + $matches[1]) -Value $matches[2] -ErrorAction SilentlyContinue
    }
}

$global:DRAY_CMAKE = Join-Path $vsRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$global:DRAY_NINJA = Join-Path $vsRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
$global:DRAY_CTEST = Join-Path $vsRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe'

if (-not (Test-Path $DRAY_CMAKE)) { throw "bundled cmake not found at $DRAY_CMAKE" }
if (-not (Test-Path $DRAY_NINJA)) { throw "bundled ninja not found at $DRAY_NINJA" }

# Do NOT pipe `cl` through 2>&1 here: in Windows PowerShell 5.1 redirecting a
# native command's stderr wraps each line in an ErrorRecord and sets $? to false
# even on success, which makes this script look like it failed.
$clPath = (Get-Command cl -ErrorAction SilentlyContinue).Source
Write-Host "MSVC env ready: $clPath"
