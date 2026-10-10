# Particle rendering — improvement plan

Planning draft, corrected after the external review and the verification round. Fixes nothing by
itself; every stage is gated by measurements taken first. Current architecture:
`docs/particle-current.md`; reference: `docs/particle-modern.md`; shortfalls:
`docs/particle-comparison.md`.

## 1. Goal and target scale

Goal: remove the particle bottleneck on the owner's scenes without losing the lighting feature, then
scale to the content that actually exists.

- **T1 committed: 100k rendered particles** on one owner-named scene, <=1 cluster evaluation and
  <=1 budgeted ray per particle, upload <=2.4 MB/frame at a 24 B record.
- **T2 stress: 262,144 particles** (the FTE compiled cap) with a loud, counted overflow policy.
- "Millions" is out of scope; shipped content expresses bursts <=1024 and per-map `particlemax`
  <=2048 as authored data, and the raster collector drops silently at 87381 3-vert particles
  (65535 FTE quads) when otherwise empty. Measure the actual counts in Stage 0 before defending T1.

## 2. Owner decision register

| ID | Decision | Recommendation |
|---|---|---|
| G0 | Confirm T1 scene and numbers; separate synthetic stress scenario | AD `start` saved viewpoint + dense-trail demo + a synthetic 100k test |
| D1 | Identity-map volume policy (start: 6125 visleafs -> 6125 identity clusters, cluster 0 shared) | Bake an R16_UINT volume by painting `leaf_cluster`; keep the CPU resolve behind a flag; 64 u is lossy and must be measured |
| D2 | Resolve removal first? | Stage 0 decides; if resolve is not dominant, prioritize conversion/upload instead of the volume |
| A1 | Volume accuracy contract | Paint `leaf_cluster` per texel (never position->cell arithmetic; tfuma counterexample leaf 12278 -> cluster 1766 vs arithmetic 3118); ambiguous texels follow the CPU rule; keep 0 where the CPU returns 0 (cluster 0 has no PVS row; the sun is ray-gated) |
| A2 | Volume generation | Its own map generation bumped where the clusters rebuild; never `listGeneration` (that ticks per composition) |
| A3 | Volume memory/bounds | 64 u base (start ~195 KiB, tfuma ~2.73 MiB); pin origin/dims/sampling in the upload; 32 u optional |
| L1 | Lighting end state | One lighting result per particle for small sprites (all vertices of a small sprite evaluate its center; a literal single invocation is not available at vs_6_2 and remains the compute-pass follow-up for the T1 scale); large FTE sprites keep a spatially varying evaluation (per vertex, cheap once the volume exists) and must not jump dark/light across cluster boundaries; cluster sample from the volume/table; the ray budget is per particle |
| L2 | RT ray budget | Hard per-frame cap on particle-path rays with a reproducible spend order (cluster-keyed where per-particle identity does not exist; the FTE legacy stream has none); <=1 sun/shadow ray per particle at its center, distance-faded, temporally reused keyed on (generation, light revision, cluster) once a GPU-visible revision exists |
| L3 | TLAS membership | Engine particles stay out; they cast no shadows and add no GI; AD sprites and entity beams are a separate case, do not generalize |
| L4 | DTAL compatibility gate | Particle/smoke consumers must use `q2SampleClusterLights` (fast + tail) -> `sampleLightNee` -> divide by `lightPdf * memberPdf`; parity matrices with `rt_dtal_groups {0,1}` x `rt_cluster_sampling {0,1}` |
| D4 | Geometry/draw shape | Classic triangle `draw(3, N)`; FTE geometry varies (line sparks 2, fan/clipped 3, billboards 4); smoke 6 — a per-instance vertex buffer, not a universal quad |
| D5 | Budgets and cvars | Separate budgets for pools, uploads, beams, decals, smoke; `rt_particles_gpu`, `rt_particle_volume` documented; loud overflow |
| D6 | FTE end state | Simulation/PSET/dlights stay CPU; FTE geometry converges in the deferred stage; classic gets GPU sim first |
| D7 | `SMOKE_CLUSTER_SCAN` | Keep 16 until the GPU pass is measured; after dtal choose fast/tail/group sampling deliberately |
| D8 | Legacy-path removal | Define a parity suite (demos, maps, vendor matrix, counters) and thresholds; not just "two releases" |
| E1 | ssqc pointparticles distance gate (owner, 2026-10-10) | ON by default, radius 2048 (`r_part_emit_distance`, 0=off); deterministic multiplicative count scaling with a hard drop at R; scope = `CL_ParseParticles` point branch only; the visual wall at R is accepted (upstream parity), and the Stage-1 "no visual change" gate does not apply to this deliberate visual change |

## 3. Stages

### Stage 0 — measurement (mandatory gate)
- Record demos at a fixed `setpos`: `ad_particle_heavy`, `ad_particle_idle`, plus a synthetic
  particle test; commit control configs under `perf/`; keep demos/logs out of git.
- Counters: live counts per path (classic/FTE/smoke) and dropped counts; upload bytes; per-resolve
  ns + cache hit/miss; vertex-ray counter (append a trailing `raysParticle` field to `QrFrameStats`
  or bump `QR_API_VERSION`; do not grow `raysPerCategory[5]` in place); GPU particle pass timer.
- Split `RT_PROF_PARTICLES` and add sub-slots; dump `rs_particles` (r_part.c:985); note that classic
  and smoke simulation run outside the draw bracket (host.c:1008-1011).
- Ablations: `r_particles {0,2}` x `r_particle_lighting {0,1}` x `r_smoke {0,1}` x
  `r_fteparticles {0,1}`, plus `-particles`/`r_part_maxparticles` (reinitialization scenario;
  live resize does not exist, r_part_fte.c:3383-3404). Account for fallbacks: `r_fteparticles 0`
  moves effects to classic, `r_particles 0` stops FTE sim and sky weather, `r_smoke 0` changes
  classic trail substitution (r_part.c:675-685).
- Statistics: report p50/p95 per frame once true per-frame instrumentation exists; the current CSV
  holds window maxima only.
- Exit gate: a signed attribution table (resolve vs expansion vs upload vs FTE conversion vs smoke
  vs sim) with build hash, pinned config, map/content hashes, and the capture interval.

### Stage 1 — repair before redesign
- FTE conversion and scratch memset bounded by live counts (`cl_numstrisvert`/`cl_numstrisidx`), not
  capacity (r_part_fte.c:6838-6857); keep the empty-scenetri early-out.
- Overflow policy: recycle-oldest with counters and a rate-limited warning; remove the silent return
  and the Debug-only assert path (RasterizedDataCollector.cpp:221-231).
- If Stage 0 says the resolve dominates: land the point-cluster cache (uncommitted, gl_rlight.c) with
  a cvar and hit/miss counters, before the dtal merge (gl_rlight.c is dtal-owned); otherwise revert
  it. The cache is not correctness-preserving at 16 u cell boundaries and must not ship silently.
- Gate: particle slot down without visual change on `start` and `ad_tfuma`.

### Stage 2 — compact points + vertex-shader expansion (particle-owned, no set-6 changes)
- `QrParticlePoint { float position[3]; uint32_t packedColor; float size; uint32_t cluster; }`
  = 24 B -> 2.4 MB at 100k, a 10x cut vs 240 B classic; `flags` has no Stage-2 consumer; stable ID
  appends a fifth word or packs into a spare bit field in a later stage; `cluster` is required.
- Transport: a per-instance vertex buffer consumed by `draw(3, N)`; the cluster volume lives in a
  particle-owned descriptor set as a sampled image. **Set 6 stays dtal-owned** (bindings 0-8 legacy,
  9-11 DTAL members/tail); no additions to set 6.
- Classic first; keep per-vertex ray behavior for parity until Stage 4.
- Gate: <=5% fps / <=10% slot regression on both demos; upload-byte counter; caps raised loudly.

### Stage 2 — implementation notes (pending measurement)
- The new public API: `QrParticlePoint` (24 B: position, packedColor, size, cluster),
  `QrParticleUploadInfo` and `qrUploadParticles`. Classic sprites upload one point per particle
  through a per-instance buffer; the new `RsParticlePoints.vert.hlsl` expands the point into the
  triangle, replacing the 3-vertex `QrVertex` transport.
- `r_particles_points` (default `1`, archived) selects the compact-point transport for classic
  triangles; `0` restores the legacy transport for the A/B arm. `QUAD_PARTICLES` always falls back
  to legacy.
- FTE and smoke keep the legacy `QrVertex` path; the traced glass stand-ins of the classic sprites
  are kept for the new path.
- The gate above ran on the Stage-0 demos on 2026-10-10; the result is at the end of this section.
- Preliminary reading, not the gate: a diagnostic run of the owner's `start1` save at the benchmark
  viewpoint (3840x2160, FSR 2, vsync off, dtal master) shows `particles_classic = 0` at rest -
  the classic transport changes nothing on that scene - while the FTE path carries the bucket:
  `particles_fte` about 7-8k live, `fte convert` averaging 5.7-6.3 ms, the per-vertex
  `particles resolve` 3.6-3.9 ms and `particles upload` 1.4-1.5 ms per frame, with `rays_particle`
  10-17k per window. The automated salvo that was meant to spawn classic particles never
  registered (the scripted attack did not fire), so the classic figure is an idle figure. This
  points at the FTE conversion/resolve as the measured hot spots and at the owner's recorded demos
  as the decisive Stage-2 exercise.
- Runtime check (stock `demo1`, both transports, 74 s each, 3840x2160): the compact-point path ran
  with classic sprites in the capture and left no crash, and the frame held within noise -
  `frame` 15.77 vs 15.80 ms, `particles fill` 0.10 vs 0.10 ms, `particles upload` 0.01 vs 0.02 ms
  (points on vs off). A confirmation run saw the classic capture reach about 1090 sprites and still
  left no point-pipeline or validation warning. The stock demo carries too few classic particles on
  average for the transport win to show; the gate still awaits the owner's dense demos.
- Stage-0 ablation session on the re-recorded heavy demo (2026-10-08, build `691e02e1`):
  `perf/stage0/attribution-2026-10-08.md`. Particle-attributable cost about 4.7 ms of a 25 ms
  frame; the lighting/resolve path about 2.6 ms; the FTE conversion about 2.8 ms; a comparable
  effect-driven share sits outside the particle slots (removed by `r_fteparticles 0`). The classic
  and smoke paths are empty on this content (`points_off` and `r_smoke 0` are no-ops). The
  point-cluster cache is worth about 1.2-1.7 ms of the heavy demo's frame (62% hits, a hit about
  63 ns against a miss's 194 ns) and stays on; two collapse-looking runs did not survive a
  controlled diagnostic and are recorded as contamination.
- Stage-2 gate (2026-10-10, build `ef41fdad` with the restored resolve slot feed): the points A/B on
  both owner demos passes the <=5% fps / <=10% slot criterion; the largest paired delta is 1.6%,
  and the classic path is empty on both demos, so the reading is a no-regression check - the
  transport's live-sprite evidence stays the stock `demo1` capture. Method, table and the
  foreground-contamination caveat: `perf/stage0/stage2-gate-2026-10-10.md`. The same build restored
  the `cpu.particles_resolve_ms` feed the merge had dropped, and a `r_tasks 1` smoke ran the heavy
  demo at 41.1 fps with the feed alive and no crash.
- Point-cluster cache thread safety (2026-10-10): the one shared table was the race - under
  `r_tasks 1` the particle task, the entity brush chains and the alias paths resolve from different
  workers at once. It is now one table per thread with the map identity kept in the table and
  per-thread hit/miss/ns counters, summed by `RT_PointClusterCacheStats` on reads that happen after
  the frame's join (`Quake/gl_rlight.c`); serial numbers are unchanged, the task-mode counters now
  add up correctly.

### Stage 3 — cluster volume, remove CPU resolves (after dtal; accuracy contract A1-A3)
- R16_UINT volume painted from `leaf_cluster`; 64 u base; own map generation; keep 0 semantics.
- Replace every CPU resolve for classic/FTE/smoke with a volume sample plus the CPU-cluster fallback
  behind a cvar.
- Known precision: one 64 u texel on start holds clusters 2 and 4; tfuma engine cells are ~358 u.
  Mismatch counters are required; the visual tolerance is an owner decision.
- Gate: parity on `start` (identity, 6125 clusters) and `ad_tfuma` (grid, 18806 visleafs -> 7936
  clusters); the particle-attributable share of `cpu.particles_resolve_ms` must collapse onto the
  per-map non-particle floor (measured with `r_particles 0`: start ~0.3 ms, tfuma ~1.8-2.0 ms,
  swamp ~3.2-3.4 ms), or a particle-only counter must carry the gate; volume-vs-CPU mismatch
  counters and the one-shot bake report are required evidence.

### Stage 4 — per-particle lighting, DTAL gate, ray budget
- One cluster/light evaluation per particle for small sprites; <=1 budgeted ray per particle under a
  hard per-frame cap; delete the per-vertex loop where the sprite size allows it
  (RsParticle.vert.hlsl:34).
- Large-sprite constraint (owner, 2026-10-08): FTE carries many large sprites and they must not
  jump dark/light as they cross cluster boundaries, especially near torches. Large sprites keep the
  spatially varying cluster/light evaluation - per vertex, which becomes a texture fetch once the
  Stage-3 volume exists - while the shadow/occlusion rays stay bounded per particle (L2) rather
  than per vertex. The volume sample must not step visibly between neighbouring samples. The
  dark smoke next to torches is parked (owner, 2026-10-10): not touched further - the particle lighting path is to be reworked under a world-space radiance cache.
- DTAL gate (L4): the smoke decoder currently misreads groups and drops them; implement the direct-
  pass pattern and prove parity with `rt_dtal_groups`/`rt_cluster_sampling` matrices.
- Gate: bounded ray counter; visual parity; the large-sprite lighting does not pop (owner check on
  the torch scenes); `rt_bench` baselines unchanged or better.
- Owner decision (2026-10-10): Stage 4 is split and ordered `4b` -> the planned queue (glass gate,
  compact transport, distance culling). Glass gate done 2026-10-11 (`rt_particle_proxy_gate`,
  measured on the owner scenes; `perf/stage0/glass-gate-2026-10-11.md`); compact transport is next. `4b` = DTAL consumer correctness plus the dark-smoke
  defect, and it starts with a scene cross-check (`rt_dtal_groups {0,1}` x `rt_particle_volume
  {0,1}` arms on a torch-view smoke scene) to separate the DTAL-group drop from the volume-lost
  class before choosing the fix shape. The GPU volume (`4a`), the L1 scale semantics (`4c`, gated
  on the synthetic 100k scene) and the L2 budget/temporal reuse (`4d`, needs stable particle IDs
  and a GPU-visible light revision) are backlog items: on the owner scenes the remaining particle
  lighting measures <=0.11 ms GPU and <=0.4 ms CPU with no fps effect, so they do not gate
  owner-visible work.
- L1 contract (owner, 2026-10-10): "one evaluation per particle" means one lighting result per
  small sprite (the center is evaluated and replicated to its vertices), not one shader
  invocation; the per-invocation reading is the compute-pass follow-up.
- L2 contract (owner, 2026-10-10): the gate is a hard per-frame cap with a reproducible spend
  order (cluster-keyed), not "by particle index"; stable IDs and a GPU-visible revision are
  prerequisites of the temporal reuse only.
- Owner decision (2026-10-10, after the cross-check): the dark-smoke work is closed and not
  touched further; particle lighting is to be reworked under a world-space radiance cache
  (particles sample cached lighting instead of evaluating lights and tracing per-particle rays;
  the id Tech 8 froxel irradiance / SHaRC direction; FSR Radiance Cache is not available on
  Vulkan yet). Findings the rework inherits from the `start` brazier cross-check: the consumer's
  weight ignores direction, so a one-sided emitter-card light can win at a point its hemisphere
  excludes, `sampleLight` returns `dw=0` and the puff goes black; a single candidate is tested
  (no fallback); and the inline visibility `RayQuery` has no alpha any-hit, so emitter geometry in
  the path is opaque to it. Cross-check runs live in `%LOCALAPPDATA%\Temp\opencode` (`cc-*`,
  `cc2-*`, `fl-*`, `fl2-*`, `fl3-*`, `fl5-*`).
- Known issue before using `rays_particle` as a proxy metric (Stage 4 verification, 2026-10-10):
  the current `RtQ2ReflectRefract.rgen` blob declares `smokeRayStats` at set 7 binding 0 while the
  pass binds the bindless cubemap table there (ray stats are at set 11); the pass interface comment
  is stale. Resolve/verify before counting proxy rays.

### Pointparticle distance gate (owner item, 2026-10-10)
- Client-side gate in `CL_ParseParticles`: the message `count` scales linearly from 1 at `R/2` to 0 at `R`, and the message is dropped at `d >= R`; `R = r_part_emit_distance` (default 2048, 0 = off).
- Only messages that would spawn are gated (`efnum` valid, `count > 0`); the latch `r_vieworg_valid` disables gating until the first rendered frame so signon/load packets are not culled against a zero origin.
- Scope: the point branch only. Trails, per-entity `emiteffectnum`/model emitters, CSQC-local effects and the rain path are untouched.
- Counters: `particles_emit_culled` / `particles_emit_faded` (reset at bench start and per idle
  window, so they are run totals under `rt_bench`; dumped and added to the bench settings witness).
- Known behavior: recipes whose per-message effective count stays at or below 1 (a literal
  `count 1` and `countabsolute` parts) reduce only at the hard wall at R; multi-count recipes fade
  inside [R/2, R). The wall drop removes the whole message including its dlight/sound side effects.

### Glass stand-in capture gate (owner item, 2026-10-11)
- New cvar `rt_particle_proxy_gate` (archived, default `1`): `0` = unconditional capture (the
  previous behaviour), `1` = auto, `2` = never. Auto captures while `rt_glass_particles` is on
  and the last completed frame either traced a reflect/refract ray (always-on mark in the
  raygen + the stats ring read) or submitted a `PT_GLASS` instance.
- Owner rule (2026-10-11): water and glass are always traced. The glass arm keeps the capture on
  for any map with `PT_GLASS` content (no pane entry pop); the ray arm keeps it on while water,
  mirrors or acid trace, so the stand-ins are never removed from a surface the pass is working
  on. The only window is the first 2-3 frames of a newly visible water/mirror surface (stats
  ring lag).
- Scope: the `PARTICLE_SPRITE` capture paths (`CaptureParticleProxies` /
  `CaptureParticlePointProxies`) only; the raster upload, the FTE transport and the
  reflect/refract dispatch are untouched. The closed state is self-consistent: no proxies -> no
  BLAS/TLAS instance -> `glassParticles` forced 0 (no query, no raster discard, no composite).
- Measured (2026-10-11, `perf/stage0/glass-gate-2026-10-11.md`): `start` closes (rays 0, no
  glass) and removes 11.9k proxies per frame — upload 2.10 -> 0.26, convert 2.75 -> 0.97,
  particles 3.85 -> 2.14, setup 2.18 -> 1.80 ms and fps 75.6 -> 91.3 (task mode); `ad_tfuma`
  closes with slots 0.23/0.67; `ad_swampy` (water) opens and keeps capturing (`rays_refl_refr`
  46k); the 4K demo runs closed at 71.0 fps with zero proxies and the `gate=closed` line.
- Known defect (separate item, owner-confirmed): the layer composites only at `MEDIA_TYPE_GLASS`
  pixels (`CmCheckerboard` mask >= 4, written only by the pane branch), so water and mirrors
  trace the stand-ins but do not show them today; when their mask writer lands, revisit the
  2-3 frame water entry window and let the checks cover the refracted particles.

### Stage 5 — GPU simulation (classic only, optional until measured)
- Ping-pong state, spawn ring, append/compaction, indirect draw; bench-freeze mode for determinism
  (fixed seed, fixed step, culling off); CPU remains the authority for dlights and PSET effects.
- Gate: demo/bench determinism, 1M stress bounded with loud overflow, FTE untouched.

### Stage 6 — deferred convergence and quality (named future work)
- FTE geometry/lighting convergence, smoke on the volume, culling, sorting/transparency, soft
  particles, particles in the TLAS, froxel ambient; each enters only when a measurement names it.
- Voxel smoke: the `feature/voxel-smoke` branch overlaps r_smoke.c, gl_vidsdl.c, render.h, qray.h,
  VulkanDevice.*, NvrhiFrameSkeleton.*, RhiRasterOverlayPass.*; serialize rebases with Stage 0
  counters and the smoke work.

## 4. Interfaces (sketch, corrected names)

```cpp
struct QrParticlePoint { float position[3]; uint32_t packedColor; float size; uint32_t cluster; };
struct QrParticleUploadInfo { const QrParticlePoint *pPoints; uint32_t count;
                              QrMaterial material; uint32_t pipelineState; QrFloat4D smokeLook; };
QrResult qrUploadParticles(QrInstance, const QrParticleUploadInfo *);
struct QrClusterVolumeUploadInfo { QrFloat3D worldMins; float invCellSize; QrExtent3D dims;
                                   const uint16_t *pCells; uint32_t generation; };
QrResult qrUploadClusterVolume(QrInstance, const QrClusterVolumeUploadInfo *);
```

`qray.h` appends; it never renumbers existing bindings, struct fields or enum values. If
`QrFrameStats` must grow, append a trailing field or bump `QR_API_VERSION` together with
`RayStats.h`, `ShaderCommonC.h`, `GenerateShaderCommon.py` and `VulkanDevice.cpp`.

## 5. Branch order and ownership

1. `bugfix/present-wait-removal` (commit 50910a89, Swapchain.cpp only) to master.
2. `refactor/dtal-cluster-dedup` to master (owns gl_rlight.c, ClusterLightLists.*, LightManager.*,
   r_world.c cluster building and the set-6 layout).
3. Particle work rebased: Stage 0 touches host.c, gl_vidsdl.c, glquake.h, RayStats, generated
   bindings, qray.h and the dumps; Stage 1 touches gl_rlight.c (coordinate with dtal); Stage 2
   touches qray.h/qray.cpp, VulkanDevice.*, the collector and r_part*; Stage 3+ follow dtal
   interfaces; set-6 changes are serialized and currently avoided.
4. Docs and changelog per stage; update `perf/README.md` with the canonical demos.

## 6. Regression guard and evidence

- `perf/compare_benchmark.ps1` today: default tolerance 20%, compares block averages, and silently
  skips metrics missing from the baseline. Strengthen before Stage 2: explicit metric list, p95
  support, fail on missing columns, and a pinned settings witness in `benchmark.log`.
- Evidence manifest per stage: commit, pinned config, map/content hashes, GPU/driver, internal
  resolution, demo hash, capture interval. No stage is accepted on owner observation alone.

## 7. Risks

- The "40 ms" figure is a window maximum; the plan must not promise numbers Stage 0 has not
  confirmed.
- Volume precision on identity maps and near solid/leaf boundaries; mismatch counters and an owner
  tolerance decision are required.
- DTAL consumer correctness (empty samples for groups on the current particle path) is a hard gate.
- Dual paths drift; bound with counters, parity demos and removal criteria.
- Driver hazards: push-constant 128 B floor, R16_UINT sampling, per-slot ring sync, AMD TDR history.
