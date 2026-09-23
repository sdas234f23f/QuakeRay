# QuakeRay engine

QuakeRay is Ray Tracing engine for Quake 1, with a Q2RTX-style partial path traced features and a Vulkan backend.

## Features

### Path traced renderer

* Ray tracing with ReSTIR direct light sampling
* FSR 2.0 and 3.1 support
* TAL (Texture Area Lights) system: all emissive surfaces are sampled as textured area lights with a per-surface light, with its own intensity, blend mode, screen-color ceiling, sharp mask and mip boost knobs. A light reads the same emission mask the visible surface does, in the point it samples, so a face bright in its centre and dark around it lights the scene from its lit part alone — through the light styles and the animated frames as well.
* True Light Mode (opt-in): All light sources are TAL, which means all emissive textures are actual light sources.
* Q2RTX-style path traced lighting.
* ASVGF denoiser.
* RT Global Illumination
* NEE (Next Event Estimation) for the sun, emissives and dynamic lights.
* per-BSP-cluster light lists.
* Animated light entities (`rt_light_styles`) make their own fixture flicker, in accordance with the original light style, to preserve the original Quake 1 lighting design.
* Full material system with per-brush and per-model metalness/roughness, normal map strength and texture-driven gloss maps, plus ray-traced water with animated wave normals and refraction.

## Graphics

* Dynamic HDR Tone mapping: overall brightness, exposure bias in EV, contrast as a mix of the fixed and the auto-exposure adapted curve
* Procedural sky: a colour of its own, with a volumetric cloud layer in a colour of its own — the sun goes behind the clouds, and its light and its god rays go out under them — and a sun disc. Both the layer and its shadow are resolved at a quality level of their own (`rt_sky_clouds_quality`)
* God rays — volumetric sun shafts, resolved at a quality level of their own (`rt_sky_godrays_quality`)
* Volumetric fog
* Bloom
* Post-processing: chromatic aberration, and a configurable LUT for colour grading

## Roadmap

* In-game light editor for emissive surfaces and dynamic lights.
* Arcane Dimensions support (the original Quake 1 expansion pack)
* Quake Remastered (2021) support (the official remaster of Quake 1)
* Mixed rasterization and ray tracing for better performance on older GPUs (the current renderer is RT only, so it is limited to GPUs with ray tracing support).
* Full physically correct path tracing.

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

5. (Optional) Package a release — needs a Release build (`.\build_win.ps1 Release`):

   ```
   .\bundle_release.ps1
   ```

   Writes `dist\QuakeRay-<version>-win64.zip`: the Release `quakeray.exe`, the runtime DLLs, the `id1` runtime assets (`materials`, `mdl_skins`, `progs`, `shaders`, `textures` and the blue noise / water normal KTX2 tables) and `readme.md`, `changelog.md` and `LICENSE.txt`. The version in the archive name is read from `ENGINE_VERSION` / `ENGINE_VER_PATCH` (`Quake\quakedef.h`) unless `-Version` passes one in; debug artifacts are never included, and the original game data is not bundled.

## Ray tracing settings

Everything is exposed as console variables; run `cvarlist rt_` in the console for the full list. The ones that change the look most are:

* `rt_brightness 1.0` — overall brightness of the ray-traced image
* `rt_exposure_bias -2.8` — exposure in EV, a power-of-two factor applied inside the tone curve (Q2RTX's exposure bias). The ray-traced image is rendered bright and pulled down here, which is what makes the noise in dark areas fade out instead of turning into visible grain
* `rt_contrast 0.6` — mixes the fixed tone curve with the auto-exposure adapted one (`0` keeps the fixed curve, `1` is the adapted curve alone)
* `rt_sky_sun 1` with `rt_sky_sun_pitch 140` / `rt_sky_sun_yaw 120` — the sun's intensity and its direction. The sun takes its colour from `rt_sky_sun_color` and goes through the same radiometric fixup as the dlights and the world lights, at a hundredth of its strength (a sun emits from no area and lights the whole sky, where those lights have an area to pay for), so `1` is a usable daylight, larger values overdrive it and `0` turns it off; the indirect sun (`rt_sky_sun_bounce_range` / `rt_sky_sun_bounce_scale`) and the god rays both scale with it, so `rt_sky_sun 10` is the daylight this setting produced before the fraction was cut a tenth further. With `rt_physical_sky 0` there is no sun at all — the classic sky is a picture the engine draws and lights nothing — so no directional light is uploaded and this setting has nothing to act on
* `rt_sky_sun_color 255 255 255` — colour of the sun as `<r> <g> <b>` in `0-255`, decoupled from the sky: it colours the directional light and everything that reads it, which is the indirect sun, the god rays, the fog's sunlit shafts (`rt_volume_lintensity`) and the sun proxy a rasterized sky hands the god rays — and it is the colour the disc of the sun is drawn in inside the procedural sky, so a sky tinted dark still has a sun in it. It is the same kind of command as `rt_sky_color` below, so `rt_sky_sun_color 255,255,255` and a bare `rt_sky_sun_color` work too and the value is archived; `rt_sky_sun_preset` writes it
* `rt_sky_sun_edit 0` — place the sun by aiming at it. While it is `1` the sun follows the crosshair — the centre of the screen, so the view vector of the frame with bob and weapon kick included — and a press of the fire button leaves it there and takes the mode off, which is why that press is swallowed instead of shooting. It writes `rt_sky_sun_pitch` and `rt_sky_sun_yaw` as it goes and nothing else, and every part of the sun reads them: the directional light and its shadow map, the indirect sun, the god rays, the fog's sunlit shafts and the disc the procedural sky draws. It is a mode rather than a setting — not archived, never on after a restart — it says what it is doing and warns when `rt_sky_sun` is `0`, and leaving it prints the pitch and the yaw the sun was left at
* `rt_sky_godrays_intensity 1` with `rt_sky_godrays 1` — strength of the volumetric sun shafts and their on/off switch: `2` doubles the shafts, `0` takes them out of the frame along with the shadow map they are marched through (as in Q2RTX). It is Q2RTX's `gr_intensity` under this engine's `rt_` naming and it means the same, but the two values are not interchangeable: upstream scales its accumulated sun-disc radiance with it, this one scales the directional light colour. Where a cloud of the procedural sky stands between the frame and the sun, the shafts are put out with the sunlight they are made of; `rt_sky_godrays_quality 2` is how finely they are drawn — the map of the world they are traced through is `1024` a side at `0` and doubles with every level to `16384` at `4`, so the level is what the crispness of the edges of a shaft costs, and the video menu's `God rays` row is the switch and the row under it is this setting. The shafts are the light of the procedural sky scattered through the air: with `rt_physical_sky 0` there is no sun to scatter and the switch has nothing to draw, whatever it says
* `rt_sky 1`, `rt_sky_brightness 1.0`, `rt_physical_sky 1` — sky intensity, sky brightness and sky model: `rt_physical_sky 1` draws the procedural sky — a colour of its own (`rt_sky_color`), the sun's disc, and the volumetric cloud layer of `rt_sky_clouds` in front of both — and `0` gives the classic sky back whole: the texture the engine draws, with nothing of the procedural sky over it — no colour and no tint of `rt_sky_color`, no sun disc, no clouds and no cloud shadow, and no sun light, no shafts and no sunlit fog with them — which is the path `rt_sky` (intensity) belongs to. The procedural sky is scaled once, by `rt_sky_brightness` and by the global `rt_brightness` it is multiplied with
* `rt_sky_color 32 0 64` — colour of the sky as `<r> <g> <b>` in `0-255`. The procedural sky is drawn in exactly this colour, with nothing mixed into it — no atmosphere and no tint — so the sky looks the way it is set and the ambient light it casts is that same colour. The sun in that sky — its light and the disc it is drawn as — is the separate `rt_sky_sun_color` above, and the clouds are the separate `rt_sky_clouds_color` below. It is a command rather than a cvar, so `cvarlist` does not list it, and running it without arguments prints the current value; the value it stores is archived, so it survives a restart. `rt_sky_color "32 0 64"` and `rt_sky_color 32,0,64` work as well. It is the colour of the *procedural* sky: the classic sky of `rt_physical_sky 0` is a texture the engine draws and is not tinted by it
* `rt_sky_clouds_color 0 0 0` with `rt_sky_clouds_alpha 1.0` — colour the procedural clouds are drawn in, as `<r> <g> <b>` in `0-255`, and the opacity they are composited over the sky with. The clouds are independent of the sky and may be darker than it (the default black is exactly that), or lighter; `rt_sky_clouds_alpha 0` takes them out of the sky and out of the light of the world with them. The colour is the same kind of command as `rt_sky_color` (`rt_sky_clouds_color 0,0,0` and a bare `rt_sky_clouds_color` work too)
* `rt_sky_clouds 1`, `rt_sky_clouds_coverage 0.2`, `rt_sky_clouds_density 0.8`, `rt_sky_clouds_speed 0.3`, `rt_sky_clouds_height 1400`, `rt_sky_clouds_thickness 900` — the cloud layer itself: the on/off switch, how much of the sky is cloud, how much light a cloud holds back, how fast the layer drifts, and where the base of it stands over the eye and how deep it is. The slab lies in the world's horizontal plane — the game world stands `z` up, which is the axis the eye looks up along — so what the eye sees is the underside of a layer that recedes to the horizon, rather than a dome wrapped around the eye. The layer is a volume the procedural sky and the sun are drawn behind and through rather than a picture of clouds: what the sky composites is the light that scattered in the cloud towards the eye over the fraction of the sky that got past it, so the sun goes *behind* the clouds — a cloud over the sun covers it, with the light scattering around its edge. The layer then shades the world: the column of cloud the sunlight crosses over a spot of the ground is what dims the sun that lights it and the god rays that would come through it, so a cloud that covers the sun in the frame puts out the sunlight and the shafts under it, in the same cloud. `rt_sky_clouds_density` is the opacity of the cloud: `8` optical depths per layer thickness at `1`, so a column that is cloud over a tenth of the layer passes about half of the light at `1` and a column that is cloud over the whole depth of it is opaque long before the default `0.8`. There is nothing to turn the shadow off with — it is what the layer being drawn means — and switching the clouds off, or taking them out of the sky with `rt_sky_clouds_alpha 0`, takes the shadow with them. `rt_sky_clouds_quality 2` is how finely the layer and its shadow are resolved rather than what they look like: the level doubles the cubemap the layer is drawn into (`128` a side at `0` to `2048` at `4`), the map of the shadow it casts (`512` a side at `0` to `8192` at `4`, which is a texel every `8` metres of ground at `0` and one every two metres at the default `2`), how old that map is allowed to get before it is filled again (`4` frames at the two lowest levels, `2` at the default, `1` from `ultra` on) and the steps the layer is marched in (`40` along the view ray and `6` toward the sun at `0`, `72` and `8` at `4`), so the level is what the clouds cost — the maps of the top level come to about `340` MB — and the video menu's `Volumetric clouds` row is the switch and the row under it is this setting
* `rt_sky_ambient_lod 4` — mip level the ambient sky light is read from; lower is more directional, `10` is a flat wash
* `rt_sky_nee 1` — sample the sky as an explicit light; `0` restores the pre-NEE result
* `rt_gi_level 1` — indirect lighting level (Q2RTX's `pt_num_bounce_rays`): `0` turns it off, `0.5` traces it at half resolution, `1` is one indirect bounce and `2` adds the diffuse second bounce; the menu cycles the same four levels
* `rt_sky_sun_bounce_range 2000` — how far the sun reaches into an indirect bounce, in Quake units. A bounce ray longer than this receives no sun light and does not trace its sun shadow ray either, which is what keeps indirect sunlight out of dark corners; smaller values are cheaper and dimmer, `0` turns indirect sunlight off. It is Q2RTX's `pt_sun_bounce_range`, default included; upstream's reference-accumulation mode uses `10000`
* `rt_sky_sun_bounce_scale 1.0` — multiplier on the sun's contribution to an indirect bounce, applied on top of the distance falloff above (Q2RTX's `sun_bounce`); `1.0` is the physical value
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
* `rt_viewm_scale 0.32` — the size the weapon in hand takes up in the world, one factor on the whole model of it. The weapon is scaled about the eye and not about its own zero, and the setting is both halves of that: the model is drawn `0.32` times smaller and its zero is carried `0.32` times closer to the camera, so the picture of the weapon in the frame is the one it always had while the room it takes up in the world shrinks with the factor. At the default of `0.32` the room it takes up in the world is under a third of what it was, and what that comes to is a weapon that is out of the walls: the barrels of `id1` reach from `19` to `54` units forward of the eye (`v_shot2` to `v_rock2`), and under a third of that is `6` to `18` units where the player's own hull keeps that eye `16` units off a flat wall, so the weapon of the fist is no longer drawn through the wall in front of it — which is the whole of what the setting is for, the ray-traced renderer tracing the weapon as geometry of the same scene the world is traced in and the depth range the classic renderer hid that with having no equivalent in a traced scene. The axe is the one weapon left out of that room: its head is swung over most of `75` units of the room the player stands in, `25` of them at `0.32`, which is more than the `16` units that separate the eye from a flat wall, and the swing of it is where a weapon is still drawn through one; `0.1` brings that under `8` units and puts every weapon of the fist out of reach of a wall altogether. The one thing the weapon can not be drawn past is the near clip distance, which is a distance of the world as well: `GL_GetCameraNear` reads `4` units at the field of view of a weapon, `0.5` of them under the floor of it, so a weapon drawn `0.32` times closer than that stands inside the near plane and is clipped away — the whole distance is taken from the same factor, the `4` units and the `0.5` under them going to `1.28` and `0.16`, which is what keeps a scaled weapon in view. What comes with that is the near plane of the whole scene, the traced frame carrying one near distance and not two: a scaled frame starts its primary rays `1.28` units from the camera instead of `4`, its projection, its FSR2 setup and its volumetric pass read the same number, and the geometry the near plane used to cut away around the player — the floor at his feet, the wall at his shoulder, the weapon's own barrel — is traced. The factor composes with `rt_viewm_fovscale` (`1.2`) and `rt_viewm_wide` (`1.05`), which shape the weapon across the view while this one shapes it away from the player: the length of a weapon and everything that follows from it, the clip included, come from this factor alone. One thing of the weapon does not follow it, and that is the light the weapon makes rather than the weapon: the flash of a shot is a dynamic light placed off the camera by `rt_muzzleoffs_x`, `rt_muzzleoffs_y` and `rt_muzzleoffs_z` (`0`, `-30`, `100` — the reach of the flash rather than the muzzle itself), and those are distances of the world read from the camera, so they do not shrink with the weapon; a weapon of `0.32` whose flash is wanted at its own muzzle again wants the three read as `0`, `-9.6`, `32`, and a flash wanted to light the room ahead can be left as it is. The setting is archived, so a choice survives a restart, and `rt_viewm_scale 1` is the weapon as it always was.

## Game data

Quake 1 game files (`id1/`) are required (registered or shareware). HD texture packs can be used through `.pkz` archives or `.mat` material definitions, and the ray-traced material overrides are deployed into the build's game dir by `build_win.ps1` (`id1/materials/materials.yaml` plus the `id1/textures`, `id1/progs` and `id1/mdl_skins` folders) — nothing has to be packed by hand.

