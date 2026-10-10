# Glass stand-in capture gate — implementation and acceptance (2026-10-11)

Scope: stop paying the per-frame particle "glass stand-in" (proxy) capture, upload and BLAS
build when no reflected/refracted particle work can use the result, without ever stopping the
tracing the feature exists for: water and glass keep their stand-ins. The path runs for every
`PARTICLE_SPRITE` raster upload — classic legacy vertices, compact points and FTE triangles —
through `RasterizedDataCollector::CaptureParticleProxies` / `CaptureParticlePointProxies`,
then feeds `sky.particleProxies` -> `RhiAccelStructs::BuildParticleProxies` (128 B/proxy
`writeBuffer` + a per-frame AABB BLAS rebuild + one TLAS instance), read by the Q2
reflect/refract raygen (`ParticleProxies.hlsli`, calls at `RaygenPrimary.hlsli:1034/1084`).
Before this work `rt_glass_particles` gated only the consumer flag; the capture ran for every
frame. Owner queue item "glass gate" (`docs/particle-plan.md`).

New cvar `rt_particle_proxy_gate` (archived, default `1`):

- `0` = unconditional capture (the previous behaviour; A/B arm),
- `1` = auto: capture while `rt_glass_particles` is on and **either** signal holds in the last
  completed frame — (a) the reflect/refract pass traced at least one ray
  (`RAY_STATS_CATEGORY_REFLECTION_REFRACTION > 0`), or (b) the submitted instance set contains
  a `PT_GLASS` instance (`RhiAccelStructs::HasGlassInstances()`),
- `2` = never (diagnostic).

Signal mechanics (the important part):

- The ray mark is always-on: `rayStatsMark` (`RaygenCommon.hlsli`) stores a plain `1` when the
  debug counters are off and keeps the atomic count when the stats view is on; both
  reflect/refract trace sites carry it (`RaygenPrimary.hlsli:622`, `:1024`). The host reads the
  category before resetting it (`VulkanDevice.cpp:1219-1224`); the ring gives the signal a 2–3
  frame lag. Without this mark the counter would only tick under the debug stats view.
- The glass signal is the last `BuildTopLevel`'s `PT_GLASS` presence (reset per build,
  `RhiAccelStructs.cpp:1826`), i.e. one frame behind the capture and map-scoped for static
  glass. It is read through the same upload field that arms the frame's only live mask writer
  (`RaygenPrimary.hlsli:369-381`), so the gate cannot close while a pane can compose the layer.
- Guarantees: a glass map keeps the capture on (no pane loses its traced layer, no entry pop);
  water/mirror/acid segments keep the capture on while the pass traces them (the owner rule:
  water and glass are always traced). The only window is the first 2–3 frames of a newly visible
  water/mirror surface (ring lag) — nothing that displays the layer today is affected (finding 5).
- Closed state is self-consistent: no proxies -> `BuildParticleProxies` skipped ->
  `GetParticleProxyBuffer` null -> `glassParticles` forced `0` -> no query, no raster discard,
  no layer composite (the chain is verified end to end in Findings/Falsifiers).

## Revision history and inputs

- v2 (this record's main set): build `.\build_win.ps1 Debug` (exit 0), `quakeray.exe` SHA256
  `3F117E6E…FD991`. Change: the signal above; plus review fixes (bool-parity for
  `rt_glass_particles`, gate state in the capture diagnostic, `qr_audit_max.cfg` pin,
  ARCHITECTURE wording).
- v1 (superseded; commit `7bbcc9c1`): the signal was `PT_GLASS` only; measured on
  `AFAAF7BC…C9F` (`start` 70.3 -> 87.7 fps) and re-verified on `F3A83B5C…FBF`. Replaced after
  the owner ruled that water and glass must always be traced: the v1 auto closed the capture on
  the water scene `ad_swampy` (no `PT_GLASS` anywhere), which stops the stand-ins water
  segments trace. v2 keeps every v1 guarantee and adds the traced-ray arm; on the owner scenes
  the v1 behaviour is unchanged (rays 0 and no glass there).
- Saves: `qr_ad_start.sav` `729D29F9…D4CA` (`start`), `qr_fuma_start.sav` `0B820D4F…98F4`
  (`ad_tfuma`), `qr_ad_swamp.sav` `9318AE31…FC13` (`ad_swampy`, water control).
- Harness: `%LOCALAPPDATA%\Temp\opencode\run_tf_attribution.ps1` (fixture
  `tests/perf/qr_audit_max.cfg` — now pins `rt_particle_proxy_gate 1` — + `rt_upscale_fsr31 3`
  + `r_tasks 1`; menu-fixed; F10 `toggleconsole;quit`; 15 s capture; window-maxima medians from
  `stats-*.dump`).
- Demo: `perf/stage0/run_points_demo_ab.ps1 -Demos ad_particle_heavy -Arms points_on
  -ExtraCvars 'r_tasks 1; rt_particle_proxy_gate 1'` (closed arm) and `… 'r_tasks 1;
  rt_particle_proxy_gate 0'` (open arm) — every arm pins the cvar (it is archived; the runner
  does not sandbox `config.cfg`).
- Artifacts: `%LOCALAPPDATA%\Temp\opencode\gg4-start-auto-1`, `gg4-start-forced-1`,
  `gg4-tfuma-auto-1`, `gg4-swampy-auto-1`; demo blocks in `build/Debug/ad/benchmark.log`
  (01:55:11 closed, plus the same-session open block). Earlier sets: `gg-*`/`gg2-*` (v1 and its
  re-verification), `gg3-*` (v2 signal check; timing contaminated by a concurrent host-side
  disk scan and marked unusable for fps, its gate prints still valid).

## Arms and results

`start` (task mode, 15 s; v2):

| Arm | fps | frame | parts | setup | upload | convert | capture proxies (med) | rays_refl_refr | fte | verts | live |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| forced `rt_particle_proxy_gate 0` | 75.6 | 9.84 | 3.85 | 2.18 | 2.10 | 2.75 | 11856 | 0 | 6174 | 24696 | 5081–6174 |
| auto (default `1`) | 91.3 | 8.10 | 2.14 | 1.80 | 0.26 | 0.97 | 0 | 0 | 6186 | 24744 | 5706–5784 |

`ad_tfuma` (task mode, 15 s; v2 auto): fps 50.2, frame 14.07, parts 1.20, setup 3.90,
upload 0.23, convert 0.67, capture 0 (`proxies=0`), `rays_refl_refr` 0, fte 3139, verts 12560,
live 3011–3101.

`ad_swampy` (water control; v2 auto): the stderr transitions read
`closed (refl/refr traced last frame: no, glass instances: absent)` then
`open (refl/refr traced last frame: yes, glass instances: absent)` — the gate opens once the
water segments trace, and the capture stays on: fps 31.0, frame 22.70, parts 0.20, setup 7.19,
upload 0.15, convert 0.16, capture samples up to 10 (~1 live particle), `rays_refl_refr` 46013.

Demo `ad_particle_heavy` (4K, task mode, 69.3 s, v2; same-session pair, every arm pinned):
closed arm (pinned `1`) 71.0 fps, 4924 frames, `gate=closed` in every capture line, zero
proxies, `particles upload` 0.32, `fte convert` 1.12, `frame` 10.02, TLAS instances 7,
`interrupted=0`; open arm (pinned `0`) 66.9 fps, 4640 frames, `gate=open` in every line
(capture 3.3–15.5k proxies), `particles upload` 1.14, `fte convert` 1.92, `frame` 10.45, TLAS
instances 8. Closed/open deltas: fps +6.1%, upload −0.82 ms, convert −0.80 ms (upload inside
the convert bracket).

## Findings

1. The gate engages exactly on the content: `start` and `ad_tfuma` print the closed transition
   (`refl/refr traced last frame: no, glass instances: absent`) and their capture samples
   collapse from 11.9k / 6k proxies to zero, while `ad_swampy` (the only owner scene where the
   pass traces rays at the saved viewpoint — water) opens and keeps capturing. The water
   tracing the owner requires is therefore never stopped; the v1 auto would have closed there.
2. The saving is exactly the capture + build path; the particle stream is untouched
   (`raster_upload_bytes` final totals comparable, `particles_dropped=0`, `rays_particle`
   unchanged, FTE buffer sizes unchanged).
3. Paired `start` deltas (v2): upload 2.10 -> 0.26 ms, convert 2.75 -> 0.97 ms (upload sits
   inside the convert bracket), particles 3.85 -> 2.14 ms, setup 2.18 -> 1.80 ms, fps 75.6 ->
   91.3 (+20.8%). v1 measured the same shape on its own pair (70.3 -> 87.7); the absolute
   levels differ by run-to-run machine state, the deltas reproduce.
4. `ad_tfuma` auto: closed with slots 0.23 / 0.67 / setup 3.90 — consistent with v1's auto run
   (0.23 / 0.69 / 4.08).
5. Water and mirrors today trace the stand-ins but do not display them: the layer's only
   consumer (`CmCheckerboard.comp.hlsl:76-80`) reads `framebufQ2GlassFilter.a` and composites
   only at `abs >= 4`, whose only live writer is the `MEDIA_TYPE_GLASS` pane branch
   (`RaygenPrimary.hlsli:367-381`); water is `MEDIA_TYPE_WATER` and mirrors are `REFLECT`
   (`VertexCollector.cpp:121-158`), and the payload branches (`glassFilter.x/.z/.w`) have no
   writers. The owner has confirmed water and glass must always be traced and shown; the mask
   writer for water/mirrors is a separate defect item. When it lands, the gate's 2–3 frame water
   entry window (Limitations) becomes visible and must be revisited (a CPU media arm or a
   shorter-lag signal are the known options).
6. Demo (v2, quiet machine, same-session pair): closed 71.0 fps / open 66.9 fps (+6.1%
   closed), upload 1.14 -> 0.32, convert 1.92 -> 1.12, TLAS 8 -> 7, with the `gate=open/closed`
   witness matching each arm. Across v1 sessions the same demo measured 34.1–53.4 fps with the
   same in-frame slots, so the fps column is a direction (session-limited); the slots, the
   `gate=` witness and the TLAS instance count are the durable evidence.
7. The gate state is now readable from the artifacts: the 120-frame capture diagnostic carries
   `gate=open/closed` (`VulkanDevice.cpp:833-839`) besides the stderr transition line, and the
   shared audit fixture pins the cvar so an inherited archive value cannot silently turn a run
   into the forced arm.

## Falsifiers checked

- Gate never engages: refuted — zero proxies in every auto sample with the explicit transition
  print.
- Wrong close while reflective/refractive work is traced: refuted — `ad_swampy` opens while the
  water traces 46k rays (the v1 signal failed exactly this and was replaced).
- Half state or holes: refuted by construction (buffer-driven flag chain) and by the runs
  (no validation messages, no asserts, particles keep drawing).
- Byte/counter fallout: `raster_upload_bytes`, `particles_dropped`, `rays_particle`, FTE
  reallocation sizes unchanged between paired arms; the byte dump columns are cumulative, so
  final totals were compared (R2 review).
- Config inheritance: the fixture and every demo arm pin `rt_particle_proxy_gate`; the line was
  removed from `build/Debug/ad/config.cfg` after the runs.

## Limitations

- No glass content exists in the current build (only commented `material_glass` examples in the
  materials yaml), so the auto-open arm and the pane transitions remain unexercised; the future
  glass scene checklist is in Owner checks. The v2 signal is a strict superset of v1's on glass
  maps, so no regression is expected there.
- The water/mirror display defect (finding 5) is a separate item; until it is fixed, closing or
  opening the gate has no visible effect at water/mirror pixels.
- Water/mirror entry window: 2–3 frames of missing stand-ins at the moment a water/mirror
  surface first becomes visible (the stats ring lag); invisible today, to revisit with the
  display fix.
- Gate decision lag is by construction: one frame behind the instance signal, 2–3 frames behind
  the ray signal; UI-only/paused frames run no capture and keep the last flag harmlessly.
- Demo fps is session-limited (v2 closed 71.0 against v1's 41.4–53.4 across sessions at the same
  slots); the fps column is reported as a direction.
- Capture samples are sampled every 120th frame; medians in this record use the per-line
  extraction convention of the analysis script (raw per-sample medians can differ by ~1%).
- Visual claims (no holes, unchanged look) are code-path arguments, not captures — no visual
  frames were recorded in these runs.

## Owner checks

- `start` / `fuma`: torch and FTE look unchanged with the default gate; the frame is faster.
- Water (hub pool, `ad_swampy`): the gate stays open (the stderr line and `gate=open` in the
  capture line); once the water display defect is fixed, particles must appear refracted in the
  water exactly as they do behind a glass pane.
- Future glass scene: capture non-zero while the pane is visible; first frame after a load may
  show the raster copy sharp (one frame, by design); dynamic glass entering/leaving view;
  `rt_glass_particles 1/0` and `rt_particle_proxy_gate 0/1/2` behave as documented.

## Review outcomes (independent reviewers on v1)

- R1 (code): no blockers; fixed the `rt_glass_particles` predicate parity (now `CVAR_TO_BOOL`
  on both sides), the ARCHITECTURE slot wording (gate in `setup`, capture in
  `particles_upload`, build in `qrDrawFrame`), and recorded the out-of-range gate values
  behaviour (0/1/2 are the modes; other values fall to auto except fractional <1 truncation).
- R2 (evidence): every number in the v1 record reproduced exactly (no refutations); fixed the
  wording/provenance items this revision absorbs (extraction convention, tail ranges, demo
  invocation with explicit pins, the unverifiable whitespace claim removed, visual claims moved
  to code arguments, byte claims switched to final totals).
- R3 (design): the single-consumer analysis, the mask chain, the closed-state consistency and
  the transition table all confirmed; carried into this revision: the fixture pin, the gate
  state witness, the future-mask-writer caveat (now largely defused — a future water/mirror mask
  writer also traces rays, so the gate stays open), and the owner-decision list (glass scene
  checks; changelog).

## Changed files

- `renderer/Include/qray/qray.h`, `renderer/Source/qray.cpp`: `qrSetParticleProxyGate`.
- `renderer/Source/VulkanDevice.h/.cpp`: policy resolution (rays ∪ glass), collector flag,
  transition print, gate state in the capture diagnostic.
- `renderer/Source/RasterizedDataCollector.h/.cpp`: `SetParticleProxyCaptureEnabled` + the two
  early-outs.
- `renderer/Source/RayStats.h`, `renderer/Source/Shaders/RaygenCommon.hlsli`,
  `renderer/Source/Shaders/RaygenPrimary.hlsli`: the always-on reflect/refract ray mark.
- `Quake/gl_rmain.c`: per-frame call from `R_SetupViewBeforeMark` (`CVAR_TO_BOOL` parity).
- `Quake/gl_vidsdl.c`: `rt_particle_proxy_gate` cvar + bench witness.
- `tests/perf/qr_audit_max.cfg`: the cvar pin.
- `docs/particle-plan.md`, `ARCHITECTURE.md`, `changelog.md`: queue/stage note, particles row,
  Unreleased entry.
