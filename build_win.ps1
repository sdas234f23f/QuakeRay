
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

$cmakeArgs = @("-B", $BuildDir, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=$Config", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")

cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build $BuildDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

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

# All of the shaders, every time: the generator decides what to build by what it
# remembers changing, and it has let a change in a header the shaders include
# (CloudLayer.h and the rest) go by without rebuilding them -- stale .spv files in
# the game folder with a build that reported success, more than once. A minute of
# shader work per build is what that costs, and it is worth it.
& (Join-Path $PSScriptRoot "build_shaders.ps1") -Rebuild -DestDir (Join-Path $gameDir "shaders")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

exit 0
