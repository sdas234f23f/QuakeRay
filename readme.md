# QuakeRay engine

QuakeRay is a ray tracing engine for Quake 1, with Q2RTX-style partial path tracing features and a Vulkan backend.

## Features
* Q2RTX-style ray tracing and partial path tracing with ReSTIR direct light sampling
* ASVGF denoiser
* RT Global Illumination
* NEE (Next Event Estimation) for the sun, emissives and dynamic lights.
* per-BSP-cluster light lists.
* Animated light entities (`rt_light_styles`) make their own fixture flicker, in accordance with the original light style, to preserve the original Quake 1 lighting design.
* Full material system with per-brush and per-model metalness/roughness, normal map strength and texture-driven gloss maps, plus ray-traced water with animated wave normals and refraction.

## Graphics
* Procedural "physical" sky
* God rays (volumetric sun shafts)
* Volumetric fog
* Bloom
* Post-processing: chromatic aberration, and a configurable LUT for color grading
* Dynamic HDR Tone mapping and exposure control

## Roadmap

* In-game light editor for emissive surfaces and dynamic lights.
* Arcane Dimensions support (the original Quake 1 expansion pack)
* Quake Remastered (2021) support (the official remaster of Quake 1)
* Mixed rasterization and ray tracing for better performance on older GPUs
* Full physically correct path tracing
* Shader effects: explosion, fire, smoke, etc.
* DirectX 12
* FSR 4
* DLSS

## Definitions

See [changelog.md](changelog.md).

## Build

The project is built with CMake and Ninja using the MSVC compiler from Visual Studio Build Tools.

### Windows

Prerequisites:

* [Git for Windows](https://github.com/git-for-windows/git/releases)
* [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) with the "Desktop development with C++" workload
* [CMake](https://cmake.org/download/) 3.20 or newer
* [Ninja](https://ninja-build.org/)
* [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) (with `glslc`; the shaders are compiled from `vkpt/Source/Shaders`)
* GPU with ray tracing support

Steps:

1. Clone the repository: `git clone --recursive https://github.com/sdas234f23f/QuakeRay.git`
2. (Re)build the SPIR-V shaders — optional: `build_win.ps1` 
3. Configure and build: `.\build_win.ps1 Debug`
4. Run the game: `build\Debug\quakeray.exe`

## More information
1. [changelog.md](changelog.md)
