
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

if ($env:VULKAN_SDK) {
    $sdkBin = Join-Path $env:VULKAN_SDK "Bin"
    if (Test-Path (Join-Path $sdkBin "glslc.exe")) {
        $env:PATH = "$sdkBin;$env:PATH"
    }
}
if (-not (Get-Command glslc -ErrorAction SilentlyContinue)) {
    throw "glslc not found. Install the Vulkan SDK or set VULKAN_SDK."
}

# dxc ships with the Vulkan SDK; the Windows SDK also has one. It is only required while HLSL
# shaders exist, i.e. from the first ported file until the GLSL sources are gone.
$hlslSources = @(Get-ChildItem -Path $shaderSrc -Filter "*.hlsl" -Recurse -ErrorAction SilentlyContinue)
if ($hlslSources.Count -gt 0 -and -not (Get-Command dxc -ErrorAction SilentlyContinue)) {
    throw ("dxc not found, but $($hlslSources.Count) HLSL shader file(s) are present. " +
           "Install the Vulkan SDK or set VULKAN_SDK.")
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

$srcFiles = @(Get-ChildItem -Path (Join-Path $shaderOut "*.spv") -ErrorAction SilentlyContinue)
if ($srcFiles.Count -eq 0) {
    throw "No SPIR-V was produced in $shaderOut; $destDir is left untouched."
}

$srcNames = @($srcFiles | Select-Object -ExpandProperty Name)

$stale = Get-ChildItem -Path (Join-Path $destDir "*.spv") | Where-Object { $srcNames -notcontains $_.Name }
foreach ($f in $stale) {
    Write-Host "Removing stale shader: $($f.Name)" -ForegroundColor Yellow
    Remove-Item $f.FullName -Force
}

$copied = Copy-Item -Path (Join-Path $shaderOut "*.spv") -Destination $destDir -Force -PassThru
Write-Host "Deployed $($copied.Count) shader(s) to $destDir" -ForegroundColor Green
