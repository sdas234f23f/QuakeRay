# QuakeRay engine

QuakeRay is a ray tracing engine for Quake 1, with Q2RTX-style partial path tracing features and a Vulkan backend.

## Features
* Q2RTX-style ray tracing and partial path tracing with ReSTIR direct light sampling
* ASVGF denoiser
* RT Global Illumination
* Dynamic Texture Area Lights (**DTAL**)
* FSR 2.0 and 3.1 support
* True Light Mode (opt-in): all light sources are **DTAL**, which means all emissive textures are actual light sources
* NEE (Next Event Estimation) for the sun, emissive and dynamic lights
* Material system
* per-BSP-cluster light lists (legacy)

## Graphics
* Procedural "physical" sky
* God rays (volumetric sun shafts)
* Volumetric fog
* Bloom
* Post-processing: chromatic aberration, and a configurable LUT for color grading
* Dynamic HDR Tone mapping and exposure control

## Roadmap
* Light and material editor
* Arcane Dimensions support (the original Quake 1 expansion pack)
* Quake Remastered (2021) support (the official remaster of Quake 1)
* Mixed rasterization and ray tracing for better performance on older GPUs
* Full physically correct path tracing
* Shader effects: explosion, fire, smoke, etc.
* DirectX 12
* FSR 4
* DLSS

## Definitions

* **ASVGF (Adaptive Spatio-Temporal Variance-Guided Filtering)** - the denoiser of the renderer, ported from Q2RTX: 
a temporal pass accumulates the lighting with the frames before it, and an a-trous (wavelet) pass filters it with weights guided by the variance of the sample and by depth, normal and colour, 
so a tap on other geometry cannot smear. Direct, indirect (at a third of the resolution, as luma spherical harmonics in YCoCg) and specular light are filtered apart from one another, 
and a gradient pass shortens the history wherever it stopped matching the frame
* **TAL (Texture Area Light)** - Q2RTX's name for a light cut out of a surface: the light is a polygon of the face with the face's own uvs,
and it samples the emission mask of the texture at the point it picks, so a face bright in its center and dark around it lights the scene from its lit part alone
* **DTAL (Dynamic Texture Area Light)** - QuakeRay's implementation of that idea, and the difference is the word dynamic: 
a Q2RTX TAL is a polygon of a face of the world, while a DTAL is any emissive surface of this engine, the moving ones included. 
A face of the world or a brush entity is stored and re-read every frame, following its light styles and animated frames, 
while an alias model is built from the triangles of the pose it draws, so its light follows the animation, the pose and the movement of the entity

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
