
param(
    [string]$Config = "Release",
    [string]$BuildDir = ""
)

$ErrorActionPreference = "Stop"

if (-not $BuildDir) {
    $BuildDir = Join-Path "build" $Config
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found. Install Visual Studio Build Tools with the C++ workload."
}

$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) {
    throw "Visual Studio Build Tools with the C++ workload are not installed."
}

$systemRoot = if ($env:SystemRoot) { $env:SystemRoot } else { "C:\Windows" }
$system32 = Join-Path $systemRoot "System32"
if ($env:PATH -notlike "*$system32*") {
    $env:PATH = "$system32;$env:PATH"
}

$cmdExe = $env:ComSpec
if (-not $cmdExe -or -not (Test-Path $cmdExe)) {
    $cmdExe = Join-Path $system32 "cmd.exe"
}

$devCmd = Join-Path $vsPath "Common7\Tools\VsDevCmd.bat"
$envLines = & $cmdExe /c "`"$devCmd`" -arch=x64 -host_arch=x64 >nul 2>&1 && set"
foreach ($line in $envLines) {
    if ($line -match "^([^=]+)=(.*)$") {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}

# The NVRHI pin needs one local change - the binding-layout limit raised to 16 for the ray-tracing
# pipeline - which the repository carries as third_party/nvrhi-max-binding-layouts.patch. The patch is
# applied for the duration of this build and removed again before the script ends, so the checked-out
# dependency always stays exactly what the submodule records. Any other build path (an IDE, a manual
# cmake invocation) has to apply the patch itself.
$nvrhiDir = Join-Path $PSScriptRoot "third_party\nvrhi"
$nvrhiPatch = Join-Path $PSScriptRoot "third_party\nvrhi-max-binding-layouts.patch"
$nvrhiPatchedHere = $false

if ((Test-Path $nvrhiPatch) -and (Test-Path (Join-Path $nvrhiDir "include\nvrhi\nvrhi.h")))
{
    & git -C $nvrhiDir apply --check --reverse $nvrhiPatch 2>$null
    if ($LASTEXITCODE -eq 0)
    {
        Write-Host "NVRHI patch is already applied, leaving it in place" -ForegroundColor Yellow
    }
    else
    {
        & git -C $nvrhiDir apply $nvrhiPatch
        if ($LASTEXITCODE -ne 0) { throw "Failed to apply $nvrhiPatch to third_party/nvrhi." }
        $nvrhiPatchedHere = $true
        Write-Host "Applied the NVRHI patch for this build" -ForegroundColor Yellow
    }
}

$exitCode = 0

$cmakeArgs = @("-B", $BuildDir, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=$Config", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")

cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { $exitCode = $LASTEXITCODE }

if ($exitCode -eq 0)
{
    cmake --build $BuildDir
    if ($LASTEXITCODE -ne 0) { $exitCode = $LASTEXITCODE }
}

if ($exitCode -ne 0)
{
    if ($nvrhiPatchedHere)
    {
        & git -C $nvrhiDir apply --reverse $nvrhiPatch
        if ($LASTEXITCODE -eq 0) { Write-Host "Reverted the NVRHI patch" -ForegroundColor Yellow }
    }
    exit $exitCode
}

$srcRoot = Join-Path $PSScriptRoot "vkpt\Source"
$gameDir = Join-Path $BuildDir "id1"
if (-not (Test-Path $gameDir)) { New-Item -ItemType Directory -Path $gameDir -Force | Out-Null }

$matYaml = Join-Path $srcRoot "materials.yaml"
if (Test-Path $matYaml) {
    $matDir = Join-Path $gameDir "materials"
    if (-not (Test-Path $matDir)) { New-Item -ItemType Directory -Path $matDir -Force | Out-Null }
    Copy-Item $matYaml (Join-Path $matDir "materials.yaml") -Force
}

foreach ($sub in @("textures", "progs", "mdl_skins")) {
    $src = Join-Path $srcRoot $sub
    if (Test-Path $src) {
        $dst = Join-Path $gameDir $sub
        if (-not (Test-Path $dst)) { New-Item -ItemType Directory -Path $dst -Force | Out-Null }
        Copy-Item -Path (Join-Path $src "*") -Destination $dst -Recurse -Force
    }
}

foreach ($f in @("BlueNoise_LDR_RGBA_128.ktx2", "WaterNormal_n.ktx2")) {
    $src = Join-Path $srcRoot $f
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $gameDir $f) -Force
    }
}

& (Join-Path $PSScriptRoot "build_shaders.ps1") -DestDir (Join-Path $gameDir "shaders")
if ($LASTEXITCODE -ne 0) { $exitCode = $LASTEXITCODE }

if ($nvrhiPatchedHere)
{
    & git -C $nvrhiDir apply --reverse $nvrhiPatch
    if ($LASTEXITCODE -ne 0) { throw "Could not revert $nvrhiPatch; third_party/nvrhi is left patched." }
    Write-Host "Reverted the NVRHI patch, third_party/nvrhi is clean again" -ForegroundColor Yellow
}

exit $exitCode
