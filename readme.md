# QuakeRay engine

QuakeRay is Ray Tracing engine for Quake 1, with a Q2RTX-style partial path traced features and a Vulkan backend.

## Features
### Path traced renderer
* Q2RTX-style ray tracing with ReSTIR direct light sampling.
* ASVGF denoiser
* RT Global Illumination
* Dynamic Texture Area Lights (***DTAL**).
* FSR 2.0 and 3.1 support
* True Light Mode (opt-in): all light sources are ***DTAL**, which means all emissive textures are actual light sources. (Light System: new)
* NEE (Next Event Estimation) denoiser for the sun, emissive and dynamic lights.
* Full material system with per-brush and per-model metalness/roughness, normal map strength and texture-driven gloss maps.
* per-BSP-cluster light lists (legacy)

## Graphics
* Procedural sky: a colour of its own, with procedural clouds that keep their own color too, and a sun disc
* God rays (volumetric sun shafts)
* Volumetric fog
* Bloom
* Post-processing: chromatic aberration, and a configurable LUT for color grading
* Dynamic HDR Tone mapping and exposure control

## Roadmap
* Light and material editor: dynamically edit the materials and lights in the game, with live updates to the ray-traced scene.
* Arcane Dimensions support (the original Quake 1 expansion pack)
* Quake Remastered (2021) support (the official remaster of Quake 1)
* Mixed rasterization and ray tracing for better performance on older GPUs (the current renderer is RT only, so it is limited to GPUs with ray tracing support).
* Full physically correct path tracing.
* Shader effects: explosion, fire, smoke, etc.
* DirectX 12
* FSR 4
* DLSS

## Definitions

* **ASVGF (Adaptive Spatio-Temporal Variance-Guided Filtering)** - the denoiser of the renderer, ported from Q2RTX: 
a temporal pass accumulates the lighting with the frames before it, and an a-trous (wavelet) pass filters it with weights guided by the variance of the sample and by depth, normal and colour, 
so a tap on other geometry cannot smear. Direct, indirect (at a third of the resolution, as luma spherical harmonics in YCoCg) and specular light are filtered apart from one another, 
and a gradient pass shortens the history wherever it stopped matching the frame.
* *TAL (Texture Area Light) - Q2RTX's name for a light cut out of a surface: the light is a polygon of the face with the face's own uvs,
and it samples the emission mask of the texture at the point it picks, so a face bright in its center and dark around it lights the scene from its lit part alone.
* **DTAL (Dynamic Texture Area Light) - QuakeRay's implementation of that idea, and the difference is the word dynamic: 
a Q2RTX TAL is a polygon of a face of the world, while a DTAL is any emissive surface of this engine, the moving ones included. 
A face of the world or a brush entity is stored and re-read every frame, following its light styles and animated frames, 
while an alias model is built from the triangles of the pose it draws, so its light follows the animation, the pose and the movement of the entity.

## Changelog

See [changelog.md](changelog.md).

## Material editor

`qr_light_editor_start` turns the view over to a free camera; the player stands where he stood.
Aim with the crosshair — a face under it is picked out by an outline — and fire to select it and
open the material panel on the right edge of the screen. The panel edits the `materials.yaml`
parameters of the picked texture, every animation frame of it at once (medkits, blinking buttons),
and the change is on screen the same frame: the material is re-synthesized and the traced world —
which bakes a material's texture indices when it is uploaded — is asked to re-upload itself, lights
included. 

* `WASD` + mouse: fly; `Shift`: faster; jump / movedown: up / down; `~`: console; `Esc`: exit.
* `Apply` — writes all materials back to the `materials/*.yaml` files they were loaded from
  (newly created materials go to `materials/materials.yaml`).
* `Cancel` — reverts the live materials to the values the files hold.
* `Exit` (or `qr_light_editor_stop`) — closes the editor, restores the player's view, discards
  whatever was not applied.
* The panel owns the mouse while open: sliders with editable values, checkboxes, hex colour
  fields with an inline colour editor, texture path fields with a file dialog, and a scrollbar.
* Written files win only over the loose files on disk: a `materials.yaml` packed into a mounted
  `.pkz` is read in preference to the written one, so keep the runtime materials loose while
  editing.

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