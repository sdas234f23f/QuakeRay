<#
.SYNOPSIS
    Regenerates the FidelityFX Denoiser shader blobs (sdk/generated/denoiser/*.h).

.DESCRIPTION
    Offline equivalent of the upstream CMake helper `compile_shaders_with_depfile()`
    (sdk/include/FidelityFX/gpu/CMakeCompileShaders.txt) for the Vulkan/GLSL backend.

    For every shader in sdk/src/backends/vk/shaders/denoiser it runs four permutations:
      <name>              -DFFX_HALF=0
      <name>_wave64       -DFFX_HALF=0
      <name>_16bit        -DFFX_HALF=1
      <name>_wave64_16bit -DFFX_HALF=1
    so 8 shaders x 4 = 32 compilations. Each compilation expands
    -DFFX_DENOISER_OPTION_INVERTED_DEPTH={0,1} into two blob headers (36 unique files, the rest are
    content-hash deduplicated).

    The generated headers are committed to the repository; run this script only after updating the
    vendored SDK or changing the GLSL sources.

.PARAMETER ToolsPath
    Folder containing FidelityFX_SC.exe and glslangValidator.exe (SDK: sdk/tools/binary_store).
    Falls back to $env:FFX_SC_PATH.

.PARAMETER OutputPath
    Where the headers are written. Defaults to <sdk root>/generated/denoiser.

.PARAMETER Force
    Clear an existing output folder before generating. Required when the default output folder
    already holds the committed headers, because FidelityFX_SC merges into existing permutation
    tables instead of rewriting them.

.EXAMPLE
    .\gen_denoiser_blobs.ps1 -ToolsPath D:\FidelityFX-SDK\sdk\tools\binary_store -Force
#>

param(
    [string]$ToolsPath = $env:FFX_SC_PATH,
    [string]$OutputPath,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$sdkRoot = Join-Path (Split-Path -Parent $PSScriptRoot) 'sdk'   # ...\FidelityFX_SDK_1.1.4\sdk
if (-not (Test-Path "$sdkRoot\include\FidelityFX\host\ffx_denoiser.h")) {
    throw "Unexpected folder layout: '$sdkRoot' does not look like the SDK root."
}
if (-not $OutputPath) { $OutputPath = "$sdkRoot\generated\denoiser" }

if (-not $ToolsPath -or -not (Test-Path "$ToolsPath\FidelityFX_SC.exe")) {
    throw "FidelityFX_SC.exe not found. Pass -ToolsPath <FidelityFX-SDK>/sdk/tools/binary_store or set FFX_SC_PATH."
}
$sc = (Resolve-Path "$ToolsPath\FidelityFX_SC.exe").Path
$toolsDir = Split-Path -Parent $sc

if (-not (Test-Path $OutputPath)) { New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null }
$OutputPath = (Resolve-Path $OutputPath).Path

$existing = @(Get-ChildItem "$OutputPath\*.h" -ErrorAction SilentlyContinue)
if ($existing.Count -gt 0 -and -not $Force) {
    throw "$OutputPath already contains $($existing.Count) generated headers. Regenerating into a dirty folder is not supported (FidelityFX_SC appends to existing permutation tables); pass -Force to clear it first."
}
if ($existing.Count -gt 0) { $existing | Remove-Item -Force }

$shaderDir = "$sdkRoot\src\backends\vk\shaders\denoiser"
$shaders = Get-ChildItem "$shaderDir\*.glsl" | Sort-Object Name
if ($shaders.Count -eq 0) { throw "No GLSL shaders found in '$shaderDir'." }

$gpuPath = "$sdkRoot\include\FidelityFX\gpu"
$baseArgs = @(
    '-compiler=glslang', '-e', 'CS', '--target-env', 'vulkan1.2', '-S', 'comp', '-Os', '-DFFX_GLSL=1',
    '-reflection', '-deps=gcc', '-DFFX_GPU=1',
    # '{0,1}' is FidelityFX_SC's own permutation syntax: one invocation emits both variants.
    '-DFFX_DENOISER_OPTION_INVERTED_DEPTH={0,1}',
    "-I$gpuPath", "-I$gpuPath\denoiser"
)

$variants = @(
    @{ Suffix = '';              Half = 0 },
    @{ Suffix = '_wave64';       Half = 0 },
    @{ Suffix = '_16bit';        Half = 1 },
    @{ Suffix = '_wave64_16bit'; Half = 1 }
)

$ok = 0
$fail = 0
foreach ($shader in $shaders) {
    $target = $shader.BaseName
    foreach ($v in $variants) {
        $scArgs = @($baseArgs) + @("-name=$target$($v.Suffix)", "-DFFX_HALF=$($v.Half)", "-output=$OutputPath", $shader.FullName)
        Push-Location $toolsDir
        try {
            & $sc @scArgs | Out-Null
        } finally {
            Pop-Location
        }
        if ($LASTEXITCODE -eq 0) {
            $ok++
        } else {
            $fail++
            Write-Warning "FAILED: $target$($v.Suffix) (exit $LASTEXITCODE)"
        }
    }
}

# Dependency files reference absolute paths of the machine that generated them; drop them.
Get-ChildItem "$OutputPath\*.d" -ErrorAction SilentlyContinue | Remove-Item -Force

$count = (Get-ChildItem "$OutputPath\*.h" -File).Count
Write-Host "compilations: ok=$ok fail=$fail -> $count headers in $OutputPath"
if ($fail -ne 0) { exit 1 }
