# QuakeRay

QuakeRay **0.9.0** adds a path tracing renderer to id Software's [Quake](https://en.wikipedia.org/wiki/Quake_(video_game)).

The renderer is a Q2RTX-style ray tracer (ported from [Q2RTX](https://github.com/NVIDIA/Q2RTX)) — it is **vendored into this repository** in the `vkpt/` folder (source + shaders + KTX/FidelityFX) and built as a static library linked straight into `quakeray.exe`. There is no external renderer library dependency.

QuakeRay is based on the [vkQuake](https://github.com/Novum/vkQuake) — a port of QuakeSpasm to Vulkan API.

## What is implemented

* Q2RTX-style path traced lighting: NEE direct light (per-BSP-cluster light lists, light-selection CDF + adaptive shadow statistics), NEE indirect light with a second diffuse bounce, and explicit sun/sky light sampling combined with the traced bounce by multiple importance sampling
* Per-BSP-cluster light lists — the world model's BSP leaves are used as clusters and the PVS is used for cluster visibility, exactly like Q2RTX (no distance-based cutoffs, occluded lights excluded per cluster)
* ASVGF denoiser (Q2RTX), checkerboard rendering, TAAU upscaler by default; FSR 2/FSR 3.1/DLSS upscalers are also available in the video menu
* Reflection/refraction, god rays (volumetric sunlight), fog volumes, procedural sky
* Q2RTX-style materials: `.mat` definitions + `.pkz` archives mounted as native search paths, automatic detection from HD texture pack suffixes (`_gloss`, `_luma`, `_glow`)
* Emissive surfaces: Q2RTX `.mat` emissives and classic fullbright textures (buttons, switches, light panels, runes, lava) emit light and tint nearby walls with their color
* Dynamic lights (torches, muzzle flashes, explosions) and map `light` entities as RT light sources
* `rt_debugflags` diagnostic views (raw unfiltered direct/indirect/specular, gradients, etc.)

The classic (non-RT) engine lighting is fully disabled in the ray-traced renderer — the ray tracer produces all the lighting (Q2RTX model). The classic renderer fallback is still available via `rt_classic_render 1`.

## Changelog

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

1. Clone the repository:

   ```
   git clone https://github.com/sdas234f23f/vkquake-ray.git
   ```

2. (Re)build the SPIR-V shaders — optional: `build_win.ps1` already builds and deploys them (see step 3), so you only need this when iterating on `vkpt/Source/Shaders` on their own:

   ```
   .\build_shaders.ps1
   ```

   This compiles `vkpt/Source/Shaders` with `glslc` and deploys the SPIR-V into `build\Debug\id1\shaders` (`-DestDir <dir>` to deploy somewhere else). Pass `-Rebuild` to ignore the shader cache and recompile everything, and `-GenCommon` when the generated shader-common headers changed.

3. Configure and build:

   ```
   .\build_win.ps1 Debug
   ```

   Debug builds go to `build\Debug` (the default build dir for the given configuration). Pass an explicit directory as a second argument only if you know you want a different one.

   (or with plain CMake: `cmake -B build\Debug -G Ninja -DCMAKE_BUILD_TYPE=Debug` + `cmake --build build\Debug`; use `-DCMAKE_BUILD_TYPE=Release` and `build\Release` for a release build).

   The build then deploys the ray-traced game data into `build\<Config>\id1`: the material definitions (`vkpt/Source/materials.yaml` → `id1/materials/materials.yaml`), `vkpt/Source/textures`, `vkpt/Source/progs` and `vkpt/Source/mdl_skins`, the blue noise table and the water normal map, and the SPIR-V shaders into `id1/shaders`.

4. Run the game:

   ```
   build\Debug\quakeray.exe
   ```

   `SDL2.dll` and all codec DLLs are copied next to `quakeray.exe` automatically during the build. The renderer is compiled into the executable — no external renderer DLL is needed. The `.spv` shaders and the blue noise texture are loaded from the game data (`id1/shaders/`, `id1/BlueNoise_LDR_RGBA_128.ktx2`).

## Ray tracing settings

Everything is exposed as console variables; run `cvarlist rt_` in the console for the full list. The ones that change the look most are:

* `rt_classic_render 0` — `1` falls back to the classic (non-ray-traced) raster renderer
* `rt_brightness 1.0` — overall brightness of the ray-traced image
* `rt_sun 1` with `rt_sun_pitch 140` / `rt_sun_yaw 120` — the sun (on/off) and its direction
* `rt_sky 1`, `rt_sky_brightness 1.0`, `rt_physical_sky 1` — sky intensity and sky model
* `rt_sky_ambient_lod 4` — mip level the ambient sky light is read from; lower is more directional, `10` is a flat wash
* `rt_sky_nee 1` — sample the sky as an explicit light; `0` restores the pre-NEE result
* `rt_indir2bounces 0` — second diffuse bounce (its own NEE and sun sample are skipped when they cannot change the result)
* `rt_emis_light_intensity 1.0` — how much light the emissive (luma-masked) surfaces emit
* `rt_light_styles 1` with `rt_light_styles_reach 48` — animated light entities make their own fixture flicker; the reach (Quake units, measured from the surface centre to the light) keeps the flicker on the fixture instead of every surface that light happens to illuminate, `-1` removes the limit
* `rt_stats 0` — on-screen ray statistics (rays per second, per-category ray counts)
* `rt_pass_stats 0` — on-screen GPU timing per render pass, next to the ray statistics
* `rt_debugflags 0` — diagnostic views (raw direct/indirect/specular, gradients, ...)

## Game data

Quake 1 game files (`id1/`) are required (registered or shareware). HD texture packs can be used through `.pkz` archives or `.mat` material definitions (see `Tools/` for converters), and the ray-traced material overrides are deployed into the build's game dir by `build_win.ps1` (`id1/materials/materials.yaml` plus the `id1/textures`, `id1/progs` and `id1/mdl_skins` folders) — nothing has to be packed by hand.

