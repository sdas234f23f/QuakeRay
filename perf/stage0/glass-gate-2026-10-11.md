# Glass stand-in capture gate — implementation and acceptance (2026-10-11)

Scope: stop paying the per-frame particle "glass stand-in" (proxy) capture, upload and BLAS
build when no reflective/refractive work can consume the result. The path runs for every
`PARTICLE_SPRITE` raster upload — classic legacy vertices, compact points and FTE triangles —
through `RasterizedDataCollector::CaptureParticleProxies` / `CaptureParticlePointProxies`,
then feeds `sky.particleProxies` -> `RhiAccelStructs::BuildParticleProxies` (128 B/proxy
`writeBuffer` + a per-frame AABB BLAS rebuild + one TLAS instance) -> the Q2 reflect/refract
raygen (`ParticleProxies.hlsli`). Until this change `rt_glass_particles` gated only the
consumer flag, not the capture or the build. Owner queue item "glass gate"
(`docs/particle-plan.md`).

New cvar `rt_particle_proxy_gate` (archived, default `1`):

- `0` = unconditional capture (the previous behaviour; A/B arm),
- `1` = auto: capture while the last completed frame's submitted instance set contains
  `PT_GLASS` glass instances (`RhiAccelStructs::HasGlassInstances()`), and while
  `rt_glass_particles` is on,
- `2` = never (diagnostic).

Auto is map/instance-scoped, not per-pixel: on maps whose current material set has no
`PT_GLASS` instances (every owner scene measured today) the gate closes for the whole frame and
the capture collapses to two early-outs. The closed state is self-consistent by construction:
empty proxy list -> `BuildParticleProxies` not called -> `GetParticleProxyBuffer` null ->
`glassParticles` forced `0` in the frame uniform -> no shader query, no raster discard, no
layer composite.

## Revision and inputs

- Build: `.\build_win.ps1 Debug` (exit 0). Main evidence: `quakeray.exe` SHA256
  `AFAAF7BC3E96391EA0990FAB33EA9A761AD47FA084FFA00F08FE925ADCD82C9F`
  (source `de7c133e` plus the gate change; the committed tree differs only by one whitespace
  line in a bench witness call). Final artifact re-verification on the committed tree:
  `F3A83B5C37B40D738D9091E420AF298FA18451B089E172C9E80F73BC2B101FBF`.
- Saves: `qr_ad_start.sav` `729D29F9…D4CA` (`start`), `qr_fuma_start.sav` `0B820D4F…98F4`
  (`ad_tfuma`), `qr_ad_swamp.sav` `9318AE31…FC13` (`ad_swampy`, water control).
- Harness: `%LOCALAPPDATA%\Temp\opencode\run_tf_attribution.ps1` (fixture `qr_audit_max.cfg` +
  `rt_upscale_fsr31 3` + `r_tasks 1`, menu-fixed, F10 `toggleconsole;quit` exit, 15 s capture,
  window-maxima medians from `stats-*.dump`).
- Demo: `perf/stage0/run_points_demo_ab.ps1 -Demos ad_particle_heavy -Arms points_on`
  (`r_tasks 1` in both arms; the runner's own cfg, owner 4K window, `interrupted=0` in all runs).
- Artifacts: `%LOCALAPPDATA%\Temp\opencode\gg-start-forced-1`, `gg-start-auto-1`,
  `gg-tfuma-forced-1`, `gg-tfuma-auto-1`, `gg-swampy-auto-1`; demo blocks in
  `build/Debug/ad/benchmark.log` (01:18:39 auto, 01:19:57 forced, plus the reversed control).

## Arms and results

`start` (task mode, 15 s):

| Arm | fps | frame | parts | setup | upload | convert | capture proxies (med) | rays_refl_refr | fte | verts | live |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| forced `rt_particle_proxy_gate 0` | 70.3 | 10.41 | 3.84 | 2.19 | 1.97 | 2.77 | 11744 | 0 | 6251 | 25004 | 5569–5894 |
| auto (default `1`) | 87.7 | 8.43 | 2.10 | 1.85 | 0.27 | 0.97 | 0 | 0 | 6139 | 24556 | 5882–6217 |

`ad_tfuma` (task mode, 15 s):

| Arm | fps | frame | parts | setup | upload | convert | capture proxies (med) | rays_refl_refr | fte | verts | live |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| forced | 46.9 | 15.42 | 1.99 | 4.29 | 1.05 | 1.44 | 6098 | 0 | 3169 | 12676 | 3075–3143 |
| auto | 46.8 | 15.83 | 1.20 | 4.08 | 0.23 | 0.69 | 0 | 0 | 3121 | 12484 | 3156–3042 |

`ad_swampy` (control; the only scene where the reflect/refract pass traces rays at the saved
viewpoint — water):

| Arm | fps | frame | parts | setup | upload | convert | capture proxies | rays_refl_refr | fte | verts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| auto | 28.5 | 25.98 | 0.22 | 7.21 | 0.18 | 0.18 | 0 | 46021 | 1 | 23 |

Demo `ad_particle_heavy` (4K, task mode, 69.4 s per run; five runs, all `interrupted=0`):

| Run | gate | fps | frames | upload | convert | TLAS instances | capture proxies |
|---|---|---:|---:|---:|---:|---:|---:|
| 01:18:39 unpinned | auto (closed) | 41.4 | 2872 | 0.19 | 1.00 | 7 | 0 |
| 01:19:57 forced | open | 34.1 | 2363 | 1.09 | 1.88 | 8 | up to 14704 |
| 01:22:53 reversed control | open | 34.1 | 2362 | 1.10 | 1.87 | 8 | up to 14710 |
| 01:24:11 reversed control, "auto" slot | open (inherited `0`) | 36.5 | 2528 | 1.09 | 1.87 | 8 | up to 15440 |
| 01:26:23 pinned `rt_particle_proxy_gate 1` | auto (closed) | 53.4 | 3704 | 0.19 | 0.95 | 7 | 0 |

Slot reproducibility across the five runs is tight — upload `1.09/1.10/1.09` open vs `0.19/0.19`
closed, convert `1.87–1.88` open vs `0.95–1.00` closed, TLAS `8` open vs `7` closed, capture
samples zero vs ~6–15k proxies. The fps column separates the arms in this sample set
(closed 41.4/53.4 vs open 34.1/34.1/36.5) but carries large session-to-session variance (closed
arm spread 41.4 -> 53.4 at the same slot values), so it is reported as a direction, not a
magnitude.

Harness note (found by the reversed-order control): `rt_particle_proxy_gate` is an archived cvar
and the demo runner does not sandbox `config.cfg`, so an unpinned arm inherits the previous
arm's value at exit — the reversed "auto" slot above ran with the capture open because the
preceding forced arm had persisted `0`. Every arm must pin the cvar explicitly; the line was
removed from `build/Debug/ad/config.cfg` after the runs.

## Final artifact re-verification

The committed tree rebuilds to `F3A83B5C…`; the `start` pair re-run on that binary reproduces
the effect (task mode, 15 s): auto (closed, capture samples zero, transition print present)
fps 82.2, particles 2.12, setup 1.85, upload 0.28, convert 1.00 — versus forced (capture 11.5k
proxies med) fps 67.8, particles 3.63, setup 2.26, upload 1.91, convert 2.58. Artifacts
`gg2-start-auto-1` / `gg2-start-forced-1`.

## Findings

1. The gate engages and the capture disappears: on `start` and `ad_tfuma` the capture samples
   collapse from 11.7k / 6.1k proxies to zero, and `stderr.log` of every auto run carries the
   transition print `qray: particle proxy gate closed (glass instances absent)`. `start`
   keeps the same save/viewpoint and the same particle population class (6251 -> 6139 live FTE,
   within new-session simulation drift; the same run has `particles_dropped=0`). `ad_swampy`
   has one FTE particle, so its capture is zero in both states by content; the scene's value
   is the water-rays case in finding 4.
2. The saving is exactly the capture + build path, not the particle stream: `raster upload`
   bytes and the FTE vertex/index sizes (`Reallocating FTE particle vertex buffer (2343 KB)`)
   are unchanged, `rays_particle` (smoke-light rays) is unchanged, and the raster particles
   still draw (no discard without a proxy buffer).
3. Paired deltas scale with the captured triangle count and with the frame budget:
   `start` (11.7k captured triangles, 14.2 ms frame) — upload `1.97 -> 0.27`, convert
   `2.77 -> 0.97`, particles `3.84 -> 2.10`, setup `2.19 -> 1.85`, fps `70.3 -> 87.7`
   (+24.8%). `ad_tfuma` (6.1k triangles, 21.3 ms frame) — upload `1.05 -> 0.23`, convert
   `1.44 -> 0.69`, particles `1.99 -> 1.20`, setup `4.29 -> 4.08`, fps inside the run
   spread (46.9 vs 46.8). The upload slot sits inside the FTE convert bracket, so the two
   slot deltas do not add; `particles` (the branch total) is the additive estimate
   (-1.74 ms on `start`, -0.79 ms on `ad_tfuma`).
4. The gate does not touch the runtime decision to trace: on `ad_swampy` the reflect/refract
   pass still traces (46021 rays med) while the gate is closed. That is deliberate — the
   particle layer is composited only where the glass mask is >= 4
   (`CmCheckerboard.comp.hlsl`), which in the current tree is written only by the normal-map
   glass branch; water/mirror pixels traverse the stand-ins today but never display the
   layer. Closing the capture there removes work that had no consumer, and the raster copy
   (which is what the player actually sees on water) is untouched.
5. The demo runs repeat the same shape at 4K: upload `1.09 -> 0.19`, convert `1.88 -> 1.00`,
   capture samples zero vs ~6–15k proxies, and one fewer TLAS instance (`8 -> 7`); the slot
   values are reproducible across sessions while the fps column is not (closed 41.4/53.4 vs
   open 34.1/34.1/36.5 — see the demo table note). The in-frame `frame` slot is flat
   (9.15–9.94 ms) in all five runs, so the fps spread is a pacing/session effect, not a
   CPU-frame effect; the demo claim rests on the deterministic slot/witness deltas.

## Falsifiers checked

- Gate never engages: refuted — zero proxies in every auto-run sample and the explicit
  transition print.
- Half state (consumer on without a buffer, or buffer without the consumer): refuted by
  construction (`particleProxyActive` reset per `BuildTopLevel`; the uniform flag requires a
  non-null buffer) and by the runs — no validation messages, no asserts, no raster holes
  (particles keep drawing; only the traced copy is gone).
- Wrong close on glass content: not reachable on the current owner content — the maps have no
  `PT_GLASS` instances (the only authored pane material is `textures/glass2: mirror: true`,
  a `PT_OPAQUE` reflect surface), which the transition print confirms (`absent`). The open
  path is exercised by the `rt_particle_proxy_gate 0` arms instead.
- Byte/counter fallout: `raster_upload_bytes`, `particles_dropped`, `rays_particle` and the
  FTE buffer reallocation sizes are unchanged between paired arms.

## Limitations

- The "gate must stay open on a real glass pane" arm cannot be run on the owner saves: the
  current material set contains no `PT_GLASS` instances, and the only recorded glass-pane
  scene belonged to an older worktree (`material_glass` on `textures/glass2`). When authored
  glass returns, the owner check is: particles behind the pane still refract/displace with
  `rt_glass_particles 1`, and `rt_particle_proxy_gate 0` / `1` / `2` behave as documented.
- Gate state uses the previous frame's instance list (one frame of lag on map/teleport
  transitions, where the capture runs one extra or one fewer frame).
- The composite-mask semantics (`glassFilter.x/.z/.w` have no writers; water/mirror layers
  are traced but never composited) are recorded here as the reason the swampy close is
  visually inert; if those writers are ever added, the gate signal must be extended to those
  surfaces, otherwise their reflections would lose particles.
- Demo fps is session/pacing-limited (five 69.4 s playbacks; closed 41.4–53.4 vs open
  34.1–36.5 at flat in-frame slots); the slot deltas, the TLAS witness and the capture samples
  are the durable demo evidence. The demo runner does not sandbox `config.cfg` and the gate
  cvar is archived — pin `rt_particle_proxy_gate` in every arm.

## Owner checks

- `start`: the torch/FTE look is unchanged with the default `rt_particle_proxy_gate 1`
  (capture skipped; raster particles untouched). No popping at cluster boundaries expected —
  the lighting path is not modified.
- Any scene with glass/mirror/water: particles must look exactly as before; if a glass pane
  is ever authored again, toggle `rt_glass_particles 1/0` and `rt_particle_proxy_gate 1/0` to
  see the traced layer appear/disappear.

## Changed files

- `renderer/Include/qray/qray.h`, `renderer/Source/qray.cpp`: `qrSetParticleProxyGate`.
- `renderer/Source/VulkanDevice.h/.cpp`: policy resolution, collector flag, transition print.
- `renderer/Source/RasterizedDataCollector.h/.cpp`: `SetParticleProxyCaptureEnabled` + the two
  early-outs.
- `Quake/gl_rmain.c`: per-frame call from `R_SetupViewBeforeMark`.
- `Quake/gl_vidsdl.c`: `rt_particle_proxy_gate` cvar + bench witness.
- `docs/particle-plan.md`, `ARCHITECTURE.md`: queue/stage note and the particles row update.
