
param(
    [switch]$Rebuild,
    [switch]$GenCommon,
    [string]$DestDir = ""
)

$ErrorActionPreference = "Stop"

$shaderSrc = Join-Path $PSScriptRoot "vkpt\Source\Shaders"
$shaderOut = Join-Path $PSScriptRoot "vkpt\Build"
if (-not $DestDir) { $DestDir = Join-Path $PSScriptRoot "build\Debug\id1\shaders" }
$destDir   = $DestDir

# The generator rebuilds a shader when a file it depends on is newer than the .spv
# it produced, and the headers the shaders include (CloudLayer.h and the rest, which
# sit next to the sources) have been outside that check: a change in one of them has
# shipped stale .spv files more than once, with a build that reported success. When
# any header in the source folder is newer than the oldest built shader, throw them
# all away and let the generator build them again. The same is done for every build
# that asks for a rebuild.
$headers = @(Get-ChildItem -Path (Join-Path $shaderSrc "*.h") -ErrorAction SilentlyContinue)
$built   = @(Get-ChildItem -Path (Join-Path $shaderOut "*.spv") -ErrorAction SilentlyContinue)
if ($headers.Count -gt 0 -and $built.Count -gt 0)
{
    $newestHeader = ($headers | Sort-Object LastWriteTime -Descending)[0]
    $oldestBuild  = ($built   | Sort-Object LastWriteTime)[0]

    if ($newestHeader.LastWriteTime -gt $oldestBuild.LastWriteTime)
    {
        Write-Host "Shader header $($newestHeader.Name) is newer than the built shaders: rebuilding all of them." -ForegroundColor Yellow
        Remove-Item (Join-Path $shaderOut "*.spv") -Force
    }
}
if ($Rebuild -and $built.Count -gt 0)
{
    Remove-Item (Join-Path $shaderOut "*.spv") -Force
}

if ($env:VULKAN_SDK) {
    $sdkBin = Join-Path $env:VULKAN_SDK "Bin"
    if (Test-Path (Join-Path $sdkBin "glslc.exe")) {
        $env:PATH = "$sdkBin;$env:PATH"
    }
}
if (-not (Get-Command glslc -ErrorAction SilentlyContinue)) {
    throw "glslc not found. Install the Vulkan SDK or set VULKAN_SDK."
}

if ($GenCommon) {
    Push-Location (Join-Path $PSScriptRoot "vkpt\Source\Generated")
    try {
        python GenerateShaderCommon.py --path .
        if ($LASTEXITCODE -ne 0) { throw "GenerateShaderCommon.py failed (exit $LASTEXITCODE)." }
    }
    finally {
        Pop-Location
    }
}

$genArgs = @()
if ($GenCommon) { $genArgs += "-gencomm" }
if ($Rebuild)  { $genArgs += "-rebuild" }
$genArgs += "-psout"

Push-Location $shaderSrc
try {
    python GenerateShaders.py @genArgs
    if ($LASTEXITCODE -ne 0) { throw "GenerateShaders.py failed (exit $LASTEXITCODE)." }
}
finally {
    Pop-Location
}

if (-not (Test-Path $destDir)) {
    New-Item -ItemType Directory -Path $destDir -Force | Out-Null
}

$srcNames = @(Get-ChildItem -Path (Join-Path $shaderOut "*.spv") | Select-Object -ExpandProperty Name)

$stale = Get-ChildItem -Path (Join-Path $destDir "*.spv") | Where-Object { $srcNames -notcontains $_.Name }
foreach ($f in $stale) {
    Write-Host "Removing stale shader: $($f.Name)" -ForegroundColor Yellow
    Remove-Item $f.FullName -Force
}

$copied = Copy-Item -Path (Join-Path $shaderOut "*.spv") -Destination $destDir -Force -PassThru
Write-Host "Deployed $($copied.Count) shader(s) to $destDir" -ForegroundColor Green
