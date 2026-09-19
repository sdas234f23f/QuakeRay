# QuakeRay

QuakeRay **0.10.0** adds a path tracing renderer to id Software's [Quake](https://en.wikipedia.org/wiki/Quake_(video_game)).

The renderer is a Q2RTX-style ray tracer (ported from [Q2RTX](https://github.com/NVIDIA/Q2RTX)) — it is **vendored into this repository** in the `vkpt/` folder (source + shaders + KTX/FidelityFX) and built as a static library linked straight into `quakeray.exe`. There is no external renderer library dependency.

QuakeRay is based on the [vkQuake](https://github.com/Novum/vkQuake) — a port of QuakeSpasm to Vulkan API.

## Features

### Path traced renderer

* Q2RTX-style path traced lighting: NEE direct light (per-BSP-cluster light lists, light-selection CDF + adaptive shadow statistics), NEE indirect light with a second diffuse bounce, and explicit sun/sky light sampling combined with the traced bounce by multiple importance sampling
* Per-BSP-cluster light lists — the world model's BSP leaves are used as clusters and the PVS is used for cluster visibility, exactly like Q2RTX (no distance-based cutoffs, occluded lights excluded per cluster). They are not rebuilt from scratch every frame: `rt_cluster_incremental` (on by default) reuses the lists as they stand while the frame registers the same lights and none of them has walked out of the quantum its slots were granted in, and a frame in which lights did move hands back only their own slots and gives them out again from where those lights stand now; `0` is Q2RTX's behaviour and composes the whole map on any change, which is the switch to try if a moving light ever leaves a spot of its own light behind
* Indirect lighting in four levels (`rt_gi_level`, also in the video menu): off, half-resolution, one bounce, and a second diffuse bounce; how far sun light reaches into an indirect bounce is bounded by `rt_sun_bounce_range` and scaled by `rt_sun_bounce_scale`
* Animated light entities (`rt_light_styles`) make their own fixture flicker, kept on the fixture by `rt_light_styles_reach`
* Dynamic lights (torches, muzzle flashes, explosions) and map `light` entities as RT light sources, each class with its own intensity and radius knob
* Point-of-interest lighting: lights near pickups, weapons, keys, ammo, armour and triggers are force-included in the light lists, so items never stand in the dark
* Emissive surfaces: Q2RTX `.mat` emissives and classic fullbright textures (buttons, switches, light panels, runes, lava) emit light and tint nearby walls with their color
* Emissive geometry sampled as textured area lights with a per-surface CDF, with its own intensity, blend mode, screen-color ceiling, sharp mask and mip boost knobs
* A flashlight (`rt_flashlight`) with configurable muzzle offsets
* ASVGF denoiser (Q2RTX), firefly suppression (`rt_antifirefly`) and a roughness floor for specular sampling (`rt_roughmin`)
* `rt_debugflags` diagnostic views (raw unfiltered direct/indirect/specular, gradients, etc.)

### Materials and surfaces

* Q2RTX-style materials: `.mat` definitions + `.pkz` archives mounted as native search paths, automatic detection from HD texture pack suffixes (`_gloss`, `_luma`, `_glow`)
* `materials/*.yaml` (loose files or inside a `.pkz`) plus a per-map `materials/<mapname>.yaml` override list, re-read on map change, and an `rt_mat` console command
* Per-brush and per-model metalness/roughness, normal map strength and texture-driven gloss maps
* Ray-traced water with animated wave normals (speed, strength, sharpness, scale) and refraction with its own index of refraction; separate glass refraction, and a refraction depth for the multi-bounce reflection/refraction pass with an early-out for pixels that cannot see through
* Split-path (reflection/refraction) surfaces resolve their two halves on alternating checkerboard fields and reconstruct the missing field from the four neighbours — Q2RTX's checkerboard composition
* Classic turbulent-surface warp on lava and teleport textures, evaluated per hit in the ray tracer (`rt_turb_warp`), and the mirrored RT portal effect on teleport surfaces (`rt_teleport_portals`)
* Alpha-cutout and alpha-blend materials

### Sky, atmosphere and effects

* Procedural sky with a physical sky model, cloud layer (on/off, coverage, density, speed, composite color), a tintable sun disc, sun presets, and NEE sampling of the sky as an explicit light
* Sun with its own intensity, pitch, yaw and preset, coloured by the sky and through the same radiometric fixup as the other lights
* God rays — volumetric sun shafts (`rt_godrays`, `gr_intensity`) marched through a shadow map
* Volumetric fog in three modes (off / simple / sky) with scattering, distance falloff, ambient term and an anisotropy-controlled artificial light source
* Classic level fog from the map's `fog` key and the `fog <density> <r> <g> <b>` console command (what Arcane Dimensions' `globalfog.qc` drives at runtime), applied with the original `exp(-(density * distance)²)` falloff, its `skyfog` sky blend, and an on/off switch in `rt_level_fog`
* Media-aware camera: underwater/slime/lava tint, acid fog, and a damage-driven screen flash with its transition
* Bloom (`rt_bloom`) with its own multiplier for emissive surfaces
* Tone mapping: overall brightness, exposure bias in EV, contrast as a mix of the fixed and the auto-exposure adapted curve
* Screen effects: CRT curvature, chromatic aberration, screen waves, and the "vintage" modes (off / CRT / 200p / 480p / 720p) that render at low internal resolution

### Image quality and video

* Upscalers in the video menu: AMD FSR 3.1 (the default, Quality preset), AMD FSR 2.0, Nvidia DLSS (needs a build configured with `RG_WITH_NVIDIA_DLSS`), and Q2RTX's temporal TAAU upscaler, used together with a nearest-neighbour fallback when all three are off; the FSR path sharpens with AMD CAS
* Render scale, sharpening (none / naive / AMD CAS), texture filtering (smooth / classic), particle style (none / circle / classic)
* A minimal ray-traced HUD with configurable padding, and separate viewmodel FOV/width scaling
* Field of view, vertical sync, max FPS, on-screen FPS
* On-screen diagnostics: one `rt_stats` command with three panels (ray counters, per-pass GPU timings, CPU frame profile), plus the debug views above

### Engine

* Quake 1 game data (registered or shareware) and its two mission packs, plus mods through gamedirs and `.pak` files
* Map formats: BSP29, BSP2 (`2PSB`/`BSP2`) and Quake 64 BSP, LordHavoc-style coloured lightmaps (`.lit`) and external entity files (`.ent`)
* Raised limits for large mods: edicts up to 32000, 2048 models, sounds and particle types, 64 light styles, 64 KB network messages
* vkQuake's QuakeC extensions (extra builtins and autocvars, gated by `pr_checkextension`)
* Networking: UDP client/server, dedicated server, co-op and deathmatch; demo recording/playback and savegames
* Audio: SDL2 output, CD music, and MP3/Ogg Vorbis/FLAC/Opus/WAV/XMP/UMX music files
* Input: keyboard, mouse and SDL2 game controllers (with `gamecontrollerdb.txt` loaded from the base dir)
* Windowed and fullscreen video modes with resolution and display selection

The ray tracer is the only renderer — there is no classic (non-RT) fallback and no renderer switch. The engine's own lighting (lightmaps and dynamic-light marking) no longer runs: the ray tracer produces all the lighting, the Q2RTX way.

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
   git clone https://github.com/sdas234f23f/QuakeRay.git
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

* `rt_brightness 1.0` — overall brightness of the ray-traced image
* `rt_exposure_bias -2.8` — exposure in EV, a power-of-two factor applied inside the tone curve (Q2RTX's exposure bias). The ray-traced image is rendered bright and pulled down here, which is what makes the noise in dark areas fade out instead of turning into visible grain
* `rt_contrast 0.6` — mixes the fixed tone curve with the auto-exposure adapted one (`0` keeps the fixed curve, `1` is the adapted curve alone)
* `rt_sun 1` with `rt_sun_pitch 140` / `rt_sun_yaw 120` — the sun's intensity and its direction. The sun is coloured by the sky (`rt_sky_color`) and goes through the same radiometric fixup as the dlights and the world lights, at a hundredth of its strength (a sun emits from no area and lights the whole sky, where those lights have an area to pay for), so `1` is a usable daylight, larger values overdrive it and `0` turns it off; the indirect sun (`rt_sun_bounce_range` / `rt_sun_bounce_scale`) and the god rays both scale with it, so `rt_sun 10` is the daylight this setting produced before the fraction was cut a tenth further
* `gr_intensity 1` with `rt_godrays 1` — strength of the volumetric sun shafts and their on/off switch: `2` doubles the shafts, `0` takes them out of the frame along with the shadow map they are marched through (as in Q2RTX). The name is Q2RTX's `gr_intensity` and means the same, but the two are not interchangeable: upstream scales its accumulated sun-disc radiance with it, this one scales the directional light colour
* `rt_sky 1`, `rt_sky_brightness 1.0`, `rt_physical_sky 1` — sky intensity and sky model
* `rt_sky_color 32 0 64` — colour of the sky as `<r> <g> <b>` in `0-255`; it tints the drawn sky (the sun disc drawn in it included) and colours the light the sky casts (sun, sky ambient, god rays). It is a command rather than a cvar, so `cvarlist` does not list it, and running it without arguments prints the current value; the value it stores is archived, so it survives a restart. `rt_sky_color "32 0 64"` and `rt_sky_color 32,0,64` work as well
* `rt_sky_clouds_color 0 0 0` — colour the procedural clouds are composited over the sky with, as `<r> <g> <b>` in `0-255`; it is the same kind of command as `rt_sky_color` (`rt_sky_clouds_color 0,0,0` and a bare `rt_sky_clouds_color` work too). The clouds themselves are shaped and animated by `rt_sky_clouds` (on/off), `rt_sky_cloud_coverage`, `rt_sky_cloud_density` and `rt_sky_cloud_speed`
* `rt_sky_ambient_lod 4` — mip level the ambient sky light is read from; lower is more directional, `10` is a flat wash
* `rt_sky_nee 1` — sample the sky as an explicit light; `0` restores the pre-NEE result
* `rt_gi_level 1` — indirect lighting level (Q2RTX's `pt_num_bounce_rays`): `0` turns it off, `0.5` traces it at half resolution, `1` is one indirect bounce and `2` adds the diffuse second bounce; the menu cycles the same four levels
* `rt_sun_bounce_range 2000` — how far the sun reaches into an indirect bounce, in Quake units. A bounce ray longer than this receives no sun light and does not trace its sun shadow ray either, which is what keeps indirect sunlight out of dark corners; smaller values are cheaper and dimmer, `0` turns indirect sunlight off. It is Q2RTX's `pt_sun_bounce_range`, default included; upstream's reference-accumulation mode uses `10000`
* `rt_sun_bounce_scale 1.0` — multiplier on the sun's contribution to an indirect bounce, applied on top of the distance falloff above (Q2RTX's `sun_bounce`); `1.0` is the physical value
* `rt_nee_samples 1` — next-event light samples per pixel in the direct pass, `1` (the Q2RTX count) or `2`; both estimators are unbiased, so `2` only trades shadow rays for a quieter image
* `rt_indir2bounces 0` — legacy switch for the second diffuse bounce, kept for old configs; the GI level above now selects it
* `rt_denoiser 1` — ASVGF reconstruction of the lighting channels (`0` composites the raw ReSTIR output)
* `rt_no_textures 0` — `1` swaps the diffuse albedo for a fixed value, i.e. "no textures"
* `rt_emis_light_intensity 1.0` — how much light the emissive (luma-masked) surfaces emit
* `rt_light_color 255 255 255` — tint multiplied into every light source (sun, dynamic, world, emissive), as `<r> <g> <b>` in `0-255`. It is a command like `rt_sky_color` above, so `cvarlist` does not list it and a bare `rt_light_color` prints the current value; `rt_light_color "255 255 255"` and `255,255,255` work too, and the value is archived
* `rt_globallight 255 255 255` — colour a light starts from before its own colour and the tint above are applied, as `<r> <g> <b>` in `0-255`; same command form, and `rt_globallight_mult` is still the separate intensity multiplier
* `rt_light_styles 1` with `rt_light_styles_reach 48` — animated light entities make their own fixture flicker; the reach (Quake units, measured from the surface centre to the light) keeps the flicker on the fixture instead of every surface that light happens to illuminate, `-1` removes the limit
* `rt_cluster_incremental 1` with `rt_cluster_dlights 1` — the two switches of the per-BSP-cluster light lists described at the top of this file. `rt_cluster_incremental 0` is Q2RTX's all-or-nothing behaviour and composes every list of the map on any change to the lights, `1` rebuilds the slots of the lights that changed and leaves the rest of the scene alone — a light that appears takes the place of one that left, so a light starting or stopping no longer composes the map either. `rt_cluster_dlights 0` keeps the moving emitters — dlights, sprites and moving emissive models — out of the lists altogether: they are still traced and still light the scene, only the lists are not offered them, so a lava ball starting to glow or a sprite leaving the screen cannot disturb the composition of the frame. Both are A/B switches for the same symptom, a `clusters` row of `rt_stats 3` that jumps to a fifth of a second when something starts to emit light, and both are meant to go away with the light module that builds the lists in the renderer from the light set
* `rt_turb_warp 1` — amplitude of the classic texture warp on lava and teleport surfaces (`0` freezes them; water and slime use the RT water waves instead)
* `rt_teleport_portals 0` — the RT portal effect on teleport surfaces is off, so they render as ordinary surfaces; `1` re-enables the mirrored destination
* `rt_stats 1,2,3` — the on-screen readout, one panel per number: `1` ray statistics (rays per second, per-category counts), `2` GPU timing per render pass, `3` CPU frame profile. The panels stack in one block at one font, each after a blank line; `rt_stats 1 2` and `rt_stats 1 2 3` work the same, `rt_stats 0` hides the readout, and a bare `rt_stats` prints which panels are on. The setting is archived, so the choice survives a restart
* `rt_stats_dump` — writes every number of the current frame into `qperfdump.log` in the game directory, one `section name value` line per number so runs can be diffed. The file is appended to, not overwritten, and the write happens on a separate thread, so taking a dump does not disturb the frame rate it measures
* `rt_debugflags 0` — diagnostic views (raw direct/indirect/specular, gradients, ...)

## Game data

Quake 1 game files (`id1/`) are required (registered or shareware). HD texture packs can be used through `.pkz` archives or `.mat` material definitions, and the ray-traced material overrides are deployed into the build's game dir by `build_win.ps1` (`id1/materials/materials.yaml` plus the `id1/textures`, `id1/progs` and `id1/mdl_skins` folders) — nothing has to be packed by hand.

