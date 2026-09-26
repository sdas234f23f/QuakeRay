<#
.SYNOPSIS
    Bundles a Release build of QuakeRay into a distributable ZIP archive.

.DESCRIPTION
    Packages quakeray.exe, the runtime DLLs, and the id1 runtime assets
    (materials, mdl_skins, progs, shaders, textures, and the BlueNoise /
    WaterNormal KTX2 files) plus documentation into a single ZIP ready for
    distribution.

    The engine binary is taken from the Release configuration, which is built
    without debug information. Debug artifacts (.pdb/.ilk/.map) are never
    included.

    Original game data (PAK0.PAK / PAK1.PAK, music) is NOT bundled and must be
    supplied by the end user from their own copy of Quake.

.PARAMETER Config
    Build configuration directory under build\ to bundle from.
    Defaults to "Release".

.PARAMETER Version
    Version string used for the archive name. If empty, it is read from
    Quake\quakedef.h (ENGINE_VERSION + ENGINE_VER_PATCH).

.PARAMETER BuildDir
    Optional explicit build directory. Defaults to build\<Config>.

.PARAMETER OutDir
    Directory (relative to the repo root) where the ZIP is written.
    Defaults to "dist".

.EXAMPLE
    .\bundle_release.ps1
    .\bundle_release.ps1 -Config Release -Version 0.10.0
#>
param(
    [string]$Config  = "Release",
    [string]$Version = "",
    [string]$BuildDir = "",
    [string]$OutDir  = "dist"
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot

if (-not $BuildDir) {
    $BuildDir = Join-Path $repoRoot "build\$Config"
}
$BuildDir = (Resolve-Path $BuildDir).Path

$exe = Join-Path $BuildDir "quakeray.exe"
if (-not (Test-Path $exe)) {
    throw "quakeray.exe not found in $BuildDir. Build the Release configuration first: .\build_win.ps1 Release"
}

$gameDir = Join-Path $BuildDir "id1"
if (-not (Test-Path $gameDir)) {
    throw "Runtime assets not found in $gameDir. Run .\build_win.ps1 Release to deploy them."
}

# Resolve the version from the engine header when not supplied explicitly.
if (-not $Version) {
    $qdef  = Join-Path $repoRoot "Quake\quakedef.h"
    $text  = Get-Content $qdef -Raw
    $maj   = [regex]::Match($text, 'ENGINE_VERSION\s+([0-9]+(?:\.[0-9]+)?)').Groups[1].Value
    $patch = [regex]::Match($text, 'ENGINE_VER_PATCH\s+([0-9]+)').Groups[1].Value
    if (-not $maj) { throw "Could not read ENGINE_VERSION from $qdef" }
    $Version = "$maj.$patch"
}

$distDir  = Join-Path $repoRoot $OutDir
$rootName = "QuakeRay-$Version-win64"
$zipPath  = Join-Path $distDir "$rootName.zip"
$stage    = Join-Path $distDir $rootName

# Reset the staging directory.
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage -Force | Out-Null

# 1) Engine binary.
Copy-Item $exe (Join-Path $stage "quakeray.exe") -Force
Write-Host "Added quakeray.exe"

# 2) Runtime DLLs only (no debug artifacts).
$dlls = Get-ChildItem $BuildDir -File -Filter "*.dll"
foreach ($dll in $dlls) {
    Copy-Item $dll.FullName (Join-Path $stage $dll.Name) -Force
}
Write-Host "Added $($dlls.Count) DLL(s)"

# 3) id1 runtime assets.
$stageId1 = Join-Path $stage "id1"
New-Item -ItemType Directory -Path $stageId1 -Force | Out-Null

foreach ($sub in @("materials", "mdl_skins", "progs", "shaders", "textures")) {
    $src = Join-Path $gameDir $sub
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $stageId1 $sub) -Recurse -Force
        Write-Host "Added id1\$sub"
    }
    else {
        Write-Warning "Skipped id1\$sub (not found in $gameDir)"
    }
}

foreach ($f in @("BlueNoise_LDR_RGBA_128.ktx2", "WaterNormal_n.ktx2")) {
    $src = Join-Path $gameDir $f
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $stageId1 $f) -Force
        Write-Host "Added id1\$f"
    }
    else {
        Write-Warning "Skipped id1\$f (not found in $gameDir)"
    }
}

# 4) Documentation / license.
foreach ($d in @("readme.md", "changelog.md", "LICENSE.txt")) {
    $src = Join-Path $repoRoot $d
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $stage $d) -Force
        Write-Host "Added $d"
    }
}

# 5) Create the ZIP with a top-level QuakeRay-<version>-win64 folder.
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$zipFs = [System.IO.Compression.ZipFile]::Open($zipPath, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($file in (Get-ChildItem $stage -Recurse -File)) {
        $rel       = $file.FullName.Substring($stage.Length + 1).Replace('\', '/')
        $entryName = "$rootName/$rel"
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $zipFs,
            $file.FullName,
            $entryName,
            [System.IO.Compression.CompressionLevel]::Optimal
        ) | Out-Null
    }
}
finally {
    $zipFs.Dispose()
}

# 6) Clean up staging.
Remove-Item $stage -Recurse -Force

Write-Host ""
Write-Host "Created $zipPath"
