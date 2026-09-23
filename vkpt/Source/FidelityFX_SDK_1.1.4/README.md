# FidelityFX Denoiser (SDK 1.1.4) — vendored subset for QuakeRay

This folder contains a **subset** of the AMD FidelityFX SDK, taken from tag `v1.1.4`
(`https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK`), used by QuakeRay to run the
analytical **FidelityFX Denoiser** (`ffx_denoiser`, "DNSR") through the **Vulkan** backend.

License: MIT (see `LICENSE.txt`, `Copyright (C) 2024 Advanced Micro Devices, Inc.`).

The pre-existing `src/ffx-api` and `src/PrebuiltSignedDLL` folders are *not* part of this vendor
drop; they came with the earlier FSR integration and use the newer `ffx_api` interface.

## What was taken

| Path | Content |
| --- | --- |
| `sdk/include/FidelityFX/host/*` | host headers: `ffx_denoiser.h`, `ffx_interface.h`, `ffx_types.h`, `ffx_error.h`, `ffx_assert.h`, `ffx_util.h`, `ffx_message.h`, `ffx_breadcrumbs.h` |
| `sdk/include/FidelityFX/host/backends/vk/ffx_vk.h` | Vulkan backend interface |
| `sdk/include/FidelityFX/gpu/**` | shared GPU headers (`ffx_core*.h`, `ffx_common_types.h`, portability shims) + `denoiser/` callbacks and resource ids |
| `sdk/include/FidelityFX/gpu/CMakeCompileShaders.txt` | upstream shader-compilation helper (kept as the reference for the offline generator below) |
| `sdk/src/components/denoiser/**` | the effect itself: `ffx_denoiser.cpp`, `ffx_denoiser_private.h` |
| `sdk/src/backends/vk/ffx_vk.cpp` | Vulkan backend implementation |
| `sdk/src/backends/vk/CMakeShadersDenoiser.txt` | Vulkan-specific shader compiler arguments |
| `sdk/src/backends/vk/shaders/denoiser/*.glsl` | 8 GLSL compute shaders (prepare, tile classification, 3 soft-shadow filters, prefilter/reproject/resolve reflections) |
| `sdk/src/backends/shared/ffx_shader_blobs.{h,cpp}` | shader-blob registry, entry point `ffxGetPermutationBlobByIndex` |
| `sdk/src/backends/shared/blob_accessors/ffx_denoiser_shaderblobs.{h,cpp}` | denoiser blob accessor |
| `sdk/src/shared/ffx_{assert,message,breadcrumbs_list,object_management}.{h,cpp}` | SDK support code |
| `sdk/generated/denoiser/*.h` | **generated** shader blobs (68 headers, see below) |

## What was intentionally left out

* `sdk/src/backends/vk/FrameInterpolationSwapchain/**` — it pulls in the FSR3 host side
  (`FidelityFX/host/ffx_fsr3.h`) and the prebuilt FSR3/DX12 signed DLLs. `ffxGetInterfaceVK()`
  references exactly one symbol from it (`ffxSetFrameGenerationConfigToSwapchainVK`), so
  `vkpt/Source/FfxDenoiser/ffx_vk_swapchain_stub.cpp` provides a stub instead. The denoiser never
  calls that callback (context creation validates only `fpGetSDKVersion`,
  `fpGetDeviceCapabilities`, `fpCreateBackendContext`, `fpDestroyBackendContext`).
* Every other effect (FSR1/2/3, CACAO, SSSR, Brixelizer, LPM, ...) — the build defines only
  `FFX_DENOISER`, and `ffx_shader_blobs.cpp` gates the effect includes on
  `defined(FFX_<EFFECT>) || defined(FFX_ALL)`.
* `sdk/tools/binary_store/*` (shader compiler + `glslangValidator.exe`, ~4.9 MB) — only needed to
  *regenerate* blobs, see below.

## Generated shader blobs

`sdk/generated/denoiser/*.h` is **not** checked into the upstream repository; upstream generates it
at build time with `FidelityFX_SC.exe`. The 68 headers here were generated once, offline, and are
committed so that a normal QuakeRay build needs no shader compiler.

To regenerate (only needed when the GLSL sources change or the SDK is updated):

```powershell
# FidelityFX_SC.exe + glslangValidator.exe ship in the SDK's sdk/tools/binary_store folder.
.\tools\gen_denoiser_blobs.ps1 -ToolsPath <path\to\FidelityFX-SDK>\sdk\tools\binary_store
```

The script mirrors upstream `compile_shaders_with_depfile()`: for each shader in
`src/backends/vk/shaders/denoiser/` it runs four permutations
(`-name=<shader>` / `_wave64` / `_16bit` / `_wave64_16bit` with `-DFFX_HALF=0/1`) with

```
-compiler=glslang -e CS --target-env vulkan1.2 -S comp -Os -DFFX_GLSL=1
-reflection -deps=gcc -DFFX_GPU=1
-DFFX_DENOISER_OPTION_INVERTED_DEPTH=0 -DFFX_DENOISER_OPTION_INVERTED_DEPTH=1
-I<sdk>/include/FidelityFX/gpu -I<sdk>/include/FidelityFX/gpu/denoiser
```

i.e. 8 shaders x 4 permutations = 32 compilations. The host code selects a blob via
`ffxGetPermutationBlobByIndex()` using the device's wave size and 16-bit-support capability, so all
four variants must be present.

### Reproducibility note

`FidelityFX_SC.exe` emits its permutation tables in filesystem-enumeration order, which is not
stable between runs. Only the 4 `ffx_denoiser_shadows_tile_classification_pass_*_permutations.h`
headers are affected (the inverted-depth option is used only by
`ffx_denoiser_shadows_tileclassification.h`; every other pass produces one distinct blob per
variant and content-hash deduplication hides the ordering). Two clean runs were verified to agree on
the `IndirectionTable` + `PermutationInfo` pair for every entry, i.e. the key -> blob mapping is
identical and the difference is limited to the order of the `#include` lines and of the table
entries. Regenerating therefore may show up as a small diff — check the mapping, not the hash.
The script clears the output folder first (with `-Force`) because generating into a folder that
already holds headers makes the compiler *merge* into the existing tables.

## Build integration

`vkpt/CMakeLists.txt` defines a static library target `ffx_denoiser` from nine translation units:

```
sdk/src/backends/vk/ffx_vk.cpp
sdk/src/backends/shared/ffx_shader_blobs.cpp
sdk/src/backends/shared/blob_accessors/ffx_denoiser_shaderblobs.cpp
sdk/src/components/denoiser/ffx_denoiser.cpp
sdk/src/shared/ffx_{assert,breadcrumbs_list,message,object_management}.cpp
vkpt/Source/FfxDenoiser/ffx_vk_swapchain_stub.cpp
```

with `FFX_DENOISER` defined and these folders on the include path:

```
sdk/include, sdk/src/shared, sdk/src/backends/shared, sdk/src/components, sdk/generated/denoiser
```

`vkpt` links the target privately. The only external dependency is `vulkan-1` (via
`Vulkan::Vulkan`); nothing else from the SDK is needed, and no `FFX_BUILD_AS_DLL` is defined, so
the whole set links into one module without export/import annotations. The target carries its own
`/W3 /WX-` settings so SDK warnings cannot break the host build's `/WX`.

Verified: the target compiles clean with MSVC 19.44 and a link test that takes the addresses of
`ffxDenoiserContextCreate`, `ffxDenoiserContextDispatchShadows` and `ffxGetInterfaceVK` resolves
against the produced `ffx_denoiser.lib` + `vulkan-1.lib`.

Upstream targets Vulkan 1.3.250+ headers; `vkpt/CMakeLists.txt` warns at configure time if the
resolved `VK_HEADER_VERSION` is older.

