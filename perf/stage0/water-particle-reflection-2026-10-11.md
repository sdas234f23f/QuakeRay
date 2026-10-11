# Water composites the traced particle layer — implementation and acceptance (2026-10-11)

Scope: make the traced particle stand-ins visible at water surfaces. The reflect/refract raygen
writes the composite mask (`framebufQ2GlassFilter`) only for `MEDIA_TYPE_GLASS` primary hits,
so the traced particle layer (`framebufQ2ParticleLayer`) was composited at glass panes but
never at water (`CmCheckerboard.comp.hlsl:76-80`, mask `abs >= 4`), even though water segments
always traced the stand-ins (`RaygenPrimary.hlsli:1034/1084`). Water and glass are the owner's
"always traced and shown" surfaces (owner, 2026-10-11).

Change: the mask branch accepts `MEDIA_TYPE_WATER` beside `MEDIA_TYPE_GLASS`
(`RaygenPrimary.hlsli:369`). The write, the raster discard (`RsParticle.frag.hlsl:44-51`,
`RsWorld.frag.hlsl:61-70`) and the composite therefore activate at water pixels; nothing else
in the chain changes.

## Revision and inputs

- Build with the change: `.\build_win.ps1 Debug` (exit 0), `quakeray.exe` SHA256
  `C56F9486A0D51C0A9A714FED6E177B12B6D47A8A5D1A0836B52E36406CCA4DB`; baseline without the
  change: `3F117E6E…D991` (the gate revision, `946ed997`).
- Scene: `qr_start_water.sav` `A2A93A56…5DFE` (`start`, owner-made at the water), torch (mdl)
  and its particle flame on the far wall over a dark pool; `rt_water_normstren 0.1` (owner, for
  a clear surface).
- Stand: `%LOCALAPPDATA%\Temp\opencode\run_water_shot.ps1` (load save, menu-fixed, per arm:
  static screenshot, then rocket salvos into the water with four screenshots, arms
  `rt_glass_particles 1/0`, config backed up and restored).
- Artifacts: `wg-before-1` / `wg-after-1` (10 PNGs each, `compare-water-flame.png`, diff images),
  `wg-diag-1` (pending), `wg-baseline-1` (the earlier `qr_ad_swamp` attempt, dark scene — the
  known light-load bug, kept only as history).
- Gate state in both sessions: open on the water view (rays > 0), capture ~5.3k proxies
  (the torch flame's FTE particles); `rt_glass_particles 0` closes the gate by design.

## Findings

1. Before the change, the flame's traced image is absent from the water in both consumer states
   (`rt_glass_particles 1` and `0` look the same at the water: no reflected flame; only the
   flare's light pool on the wall). This is the defect the owner observed live ("факелы…
   не применяется искажение воды").
2. After the change, with `rt_glass_particles 1` the flame's reflected image appears in the
   water below the torch (a wavy column), traced through the water surface; with
   `rt_glass_particles 0` it disappears. The same-session pair isolates the layer as the source
   (`wg-after-1`: `diff-on-off.png`, crops 0084/0089).
3. The owner confirmed live on the changed build: particle reflections appear at water for
   captured particles ("я вижу отражение частиц"), and smoke sprites still do not reflect —
   see the follow-up below.
4. The change does not touch glass: panes keep their mask, discard and composite (the branch is
   the same, the flag set widened).
5. Remaining exclusions (pre-existing, documented capture exceptions): the engine smoke path
   (`r_smoke`, `SMOKE` state, never a capture candidate), the per-channel FTE blends
   (`BM_BLENDCOLOUR`, `BM_SUBTRACT`, `BM_INVMODC` -> `rejBlend`) and the line sparks
   (`BEF_LINES` -> `rejLines`). The owner's requirement "all particles must reflect" (widened
   from FTE to every particle path on 2026-10-11) is the follow-up item that lifts these
   exceptions: the layer compositing needs per-channel (multiply, add) accumulation to express
   the per-channel blends, the lines need a segment proxy kind, and the smoke needs a capture
   branch plus its procedural look in the traced hit.

## Limitations

- Underwater camera view untested (the save looks from above the surface).
- The glass-blur mode (`glassBlur != 0`) still writes no water mask (the whole branch is behind
  `glassBlur == 0`, same as glass).
- The glass denoiser (`rt_glass_denoise`) now sees water pixels as pane-like mask values; the
  owner's config has it off. A denoiser-on check is pending.
- The gate's 2-3 frame water entry window becomes observable now (the first frames of a newly
  visible water surface still use the raster copy); closing it map-wide costs the `start` win
  because the map contains liquids anywhere.
- `qr_ad_swamp` renders without its loaded light (known engine bug; a `restart` is the
  workaround) — the record uses `qr_start_water` instead.

## Changed files

- `renderer/Source/Shaders/RaygenPrimary.hlsli`: the mask branch accepts water.
