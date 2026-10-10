/*
Copyright (C) 1996-2001 Id Software, Inc.
Copyright (C) 2002-2009 John Fitzgibbons and others
Copyright (C) 2007-2008 Kristian Duske
Copyright (C) 2010-2014 QuakeSpasm developers
Copyright (C) 2016 Axel Gneiting

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

*/
// gl_vidsdl.c -- SDL vid component

#include "quakedef.h"
#include "cfgfile.h"
#include "bgmusic.h"
#include "palette.h"
#include "qr_gui.h"
#include "cursor.h"
#include "rt_material.h"
#include "rt_lights.h"
#include "qr_editor.h"
#include "photocam.h"
#include "../shared/rt_frame_policy.h"
#include "../shared/rt_prof_window.h"
#include "SDL.h"
#include "SDL_syswm.h"
#include <time.h> // for the timestamp of the frame rt_stats_dump appends

extern float rt_viewmodel_depth_near;
extern float rt_viewmodel_depth_far;

#define MAX_MODE_LIST  600 // johnfitz -- was 30
#define MAX_BPPS_LIST  5
#define MAX_RATES_LIST 20
#define MAXWIDTH       10000
#define MAXHEIGHT      10000

#define MAX_SWAP_CHAIN_IMAGES 8
#define REQUIRED_COLOR_BUFFER_FEATURES                                                                                             \
	(VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | \
	 VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)

#define DEFAULT_REFRESHRATE 60

typedef struct
{
	int width;
	int height;
	int refreshrate;
} vmode_t;

static vmode_t *modelist = NULL;
static int      nummodes;

static qboolean vid_initialized = false;
static qboolean has_focus = true;

static SDL_Window   *draw_context;
static SDL_SysWMinfo sys_wm_info;

static qboolean vid_locked = false; // johnfitz
static qboolean vid_changed = false;

static void VID_Menu_Init (void); // johnfitz
static void VID_Menu_f (void);    // johnfitz
static void VID_MenuDraw (cb_context_t *cbx);
static void VID_MenuKey (int key);
static void VID_Restart (qboolean set_mode);
static void VID_Restart_f (void);
static void RT_Fog_Cmd (void);

static void ClearAllStates (void);

viddef_t        vid; // global video state
modestate_t     modestate = MS_UNINIT;
extern qboolean scr_initialized;
extern cvar_t   r_particles, host_maxfps, r_gpulightmapupdate;
extern cvar_t   scr_showfps, scr_fov;

//====================================

// johnfitz -- new cvars
static cvar_t                   vid_fullscreen = {"vid_fullscreen", "0", CVAR_ARCHIVE}; // QuakeSpasm, was "1"
static cvar_t                   vid_width = {"vid_width", "-1", CVAR_ARCHIVE};         // RT: set to "-1", so we have 
static cvar_t                   vid_height = {"vid_height", "-1", CVAR_ARCHIVE};       //     desktop resolution at the first time
static cvar_t                   vid_refreshrate = {"vid_refreshrate", "60", CVAR_ARCHIVE};
cvar_t                          vid_vsync = {"vid_vsync", "2", CVAR_ARCHIVE};
static cvar_t                   vid_maxframelatency = {"vid_maxframelatency", "0", CVAR_ROM};

int                             vid_display_refresh = 0;
static cvar_t                   vid_desktopfullscreen = {"vid_desktopfullscreen", "0", CVAR_ARCHIVE}; // QuakeSpasm
static cvar_t                   vid_borderless = {"vid_borderless", "0", CVAR_ARCHIVE};               // QuakeSpasm
static cvar_t                   vid_palettize = {"vid_palettize", "0", CVAR_ARCHIVE};
cvar_t                          vid_filter = {"vid_filter", "1", CVAR_ARCHIVE};
cvar_t                          vid_gamma = {"gamma", "1", CVAR_ARCHIVE};       // johnfitz -- moved here from view.c
cvar_t                          vid_contrast = {"contrast", "1", CVAR_ARCHIVE}; // QuakeSpasm, MarkV
cvar_t                          r_usesops = {"r_usesops", "1", CVAR_ARCHIVE};   // johnfitz

task_handle_t prev_end_rendering_task = INVALID_TASK_HANDLE;
atomic_uint32_t rt_end_task_running;

// RT
#define CVAR_DEF_LIST( CVAR_DEF_T ) \
	\
	CVAR_DEF_T (rt_enable_pvs, "0") \
	/* No pass reads maxBounceShadows. Kept as the engine-side setter of the
	   public field, and old configs keep loading. */ \
	CVAR_DEF_T (rt_shadowrays, "2") \
	/* Kept only so old configs load without an unknown-cvar warning: the GI level
	   below decides the bounce count (see GL_EndRenderingTask). */ \
	CVAR_DEF_T (rt_indir2bounces, "0") \
	CVAR_DEF_T (rt_gi_level, "1") \
	CVAR_DEF_T (rt_sky_sun_bounce_range, "2000") \
	CVAR_DEF_T (rt_sky_sun_bounce_scale, "1.0") \
	CVAR_DEF_T (rt_sky_godrays, "1") \
	CVAR_DEF_T (rt_sky_godrays_intensity, "1") /* Q2RTX's gr_intensity: strength of the sun shafts */ \
	/* Quality of the volumetric sun shafts: 0 low, 1 medium, 2 high, 3 ultra, 4 extreme (anything \
	   above is read as extreme). The shafts are traced through a shadow map of the world, and the \
	   level is the resolution that map is drawn at: it is what decides how crisp the edges of the \
	   shafts are, and what they cost. */ \
	CVAR_DEF_T (rt_sky_godrays_quality, "2") \
	CVAR_DEF_T (rt_sky_godrays_sky_threshold, "0.75") \
	CVAR_DEF_T (rt_denoiser, "1") \
	CVAR_DEF_T (rt_no_textures, "0") \
	/* No pass reads forceAntiFirefly: CmQ2Adapter's anti-firefly is not gated by
	   it. Kept as the setter of the public QrDrawFrameInfo field. */ \
	CVAR_DEF_T (rt_antifirefly, "1") \
	CVAR_DEF_T (rt_roughmin, "0.02") \
    \
	CVAR_DEF_T (rt_dlight_intensity, "1.0") \
	CVAR_DEF_T (rt_dlight_radius, "0.1") \
	\
	CVAR_DEF_T (rt_emis_light_intensity, "1.0") \
	\
	CVAR_DEF_T (rt_elight_normaliz, "100") \
	CVAR_DEF_T (rt_elight_default, "200") \
	CVAR_DEF_T (rt_elight_default_mdl, "1000") \
	CVAR_DEF_T (rt_elight_threshold, "-1") \
    CVAR_DEF_T (rt_elight_radius, "0.01") \
    \
	/* How far a light of the map itself is heard from where it stands, in metres. It is also the
	   reach the top-up pass looks around a cluster with, so it bounds the lights a cluster takes
	   beyond its own PVS. A setting of zero turns the pass off and falls back to the reach the
	   moving lights are held to. */ \
	CVAR_DEF_T (rt_light_reach_static, "25") \
	/* Reach of a light of a moving entity, in metres: the distance it is promised not to reach
	   past, and so the reason a torch or a flame reaches a few rooms instead of every list of
	   the map. */ \
	CVAR_DEF_T (rt_light_reach_dynamic, "10") \
	/* Kept only so old configs load without an unknown-cvar warning: the reaches are
	   rt_light_reach_static and rt_light_reach_dynamic now. */ \
	CVAR_DEF_T (rt_light_reach, "25") \
	CVAR_DEF_T (rt_light_reach_max, "10") \
	CVAR_DEF_T (rt_cluster_dlights, "1") \
	/* 1 takes back only the slots of the lights that moved and hands them out again from where \
	   those lights stand now: every other list of the scene keeps what the composition left, so \
	   a moving lamp costs the lists by the rooms it covers rather than by the map. 0 keeps the \
	   lists all or nothing, a light that moves takes every list of the scene with it, which is \
	   what a lava ball was measured to cost. */ \
	CVAR_DEF_T (rt_cluster_incremental, "1") \
	CVAR_DEF_T (rt_particle_resolve_cache, "1") \
	CVAR_DEF_T (rt_cluster_sampling, "0") \
	CVAR_DEF_T (rt_cluster_assert, "0") \
	CVAR_DEF_T (rt_truelight, "1") \
	CVAR_DEF_T (rt_materials_only, "0") \
	CVAR_DEF_T (rt_light_styles, "1") \
	CVAR_DEF_T (rt_light_styles_reach, "48") \
	/* 1 splits a water or animated chain of one world texture only when something the upload \
	   of a batch actually reads differs (entity, model, materials, alpha, alpha test, zbias, \
	   warp/water/acid/animated), so the surfaces of that texture share one uploaded geometry; \
	   the lightmap page is not a reason to split, because the ray traced upload never reads it \
	   -- it names a texture of the rasterizer's lightmap pass. 0 keeps the historical test, \
	   whose alpha comparison has its sign inverted and which reads no other state. */ \
	CVAR_DEF_T (rt_world_batch_merge, "1") \
	CVAR_DEF_T (rt_brush_persistent, "0") \
	/* 0 uploads the map's lights one call at a time, for measuring the batched path. */ \
	CVAR_DEF_T (rt_wmodel_lights_batch, "1") \
	/* DTAL: 1 lights an alias model from the triangles of the pose it draws, when its material
	   is a light and carries an emissive mask; 0 keeps the fake dlight for it. The per-model cap
	   and the frame budget bound a crowd of glowing models (see RT_AddAliasEmissiveLights). */ \
	CVAR_DEF_T (rt_model_lights, "1") \
	CVAR_DEF_T (rt_dtal_model_maxpolys, "8") \
	CVAR_DEF_T (rt_dtal_model_budget, "512") \
	CVAR_DEF_T (rt_dtal_model_minarea, "0") \
	\
	CVAR_DEF_T (rt_poi_distthresh, "2") \
	CVAR_DEF_T (rt_poi_distthresh_super, "3") \
	CVAR_DEF_T (rt_poi_trigger, "1") \
	CVAR_DEF_T (rt_poi_func, "1") \
	CVAR_DEF_T (rt_poi_weapon, "1") \
	CVAR_DEF_T (rt_poi_ammo, "1") \
	CVAR_DEF_T (rt_poi_pwrup, "1") \
	CVAR_DEF_T (rt_poi_health, "1") \
	CVAR_DEF_T (rt_poi_armor, "1") \
	CVAR_DEF_T (rt_poi_key, "1") \
	\
	CVAR_DEF_T (rt_sky_sun, "1") \
	CVAR_DEF_T (rt_sky_sun_color, "255 255 255") \
	CVAR_DEF_T (rt_sky_sun_size, "1.0") \
	CVAR_DEF_T (rt_sky_sun_pitch, "140") \
	CVAR_DEF_T (rt_sky_sun_yaw, "120") \
	CVAR_DEF_T (rt_sky_sun_preset, "0") \
	CVAR_DEF_T (rt_flashlight, "0") \
	CVAR_DEF_T (rt_dlightspot_intensity, "1") \
	\
	CVAR_DEF_T (rt_muzzleoffs_x, "0") \
	CVAR_DEF_T (rt_muzzleoffs_y, "-30") \
	CVAR_DEF_T (rt_muzzleoffs_z, "100") \
	\
	CVAR_DEF_T (rt_sky, "1") \
	CVAR_DEF_T (rt_sky_ambient_lod, "4") \
	CVAR_DEF_T (rt_sky_nee, "1") \
	CVAR_DEF_T (rt_physical_sky, "1") \
	CVAR_DEF_T (rt_physical_sun, "0") \
	CVAR_DEF_T (rt_sky_color, "255 255 255") \
	CVAR_DEF_T (rt_sky_brightness, "1.0") \
	CVAR_DEF_T (rt_sky_light_mult, "1.0") \
	CVAR_DEF_T (rt_brightness, "1.0") \
	CVAR_DEF_T (rt_light_color, "255 255 255") \
	CVAR_DEF_T (rt_sky_clouds, "1") \
	CVAR_DEF_T (rt_sky_clouds_quality, "2") \
	CVAR_DEF_T (rt_sky_clouds_color, "0 0 0") \
	CVAR_DEF_T (rt_sky_clouds_alpha, "1.0") \
	CVAR_DEF_T (rt_sky_clouds_coverage, "0.2") \
	CVAR_DEF_T (rt_sky_clouds_density, "0.8") \
	CVAR_DEF_T (rt_sky_clouds_speed, "0.3") \
	CVAR_DEF_T (rt_sky_clouds_height, "140000") \
	CVAR_DEF_T (rt_sky_clouds_thickness, "90000") \
	\
	CVAR_DEF_T (rt_brush_metal, "0.0") \
	CVAR_DEF_T (rt_brush_rough, "1.0") \
	CVAR_DEF_T (rt_model_metal, "0.0") \
	CVAR_DEF_T (rt_model_rough, "1.0") \
    \
	CVAR_DEF_T (rt_normalmap_stren, "1") \
	CVAR_DEF_T (rt_emis_mapboost, "30") \
	CVAR_DEF_T (rt_emis_maxscrcolor, "4") \
	CVAR_DEF_T (rt_emis_fullbright_dflt, "255") \
	CVAR_DEF_T (rt_emis_sharpmask, "1") \
	CVAR_DEF_T (rt_emis_blend, "1") \
	CVAR_DEF_T (rt_emis_blendstr, "1") \
	CVAR_DEF_T (rt_tal_selflit, "6") \
	CVAR_DEF_T (rt_dtal_minarea, "0") \
	CVAR_DEF_T (rt_dtal_maxpolys, "64") \
	CVAR_DEF_T (rt_dtal_clearance, "1") \
	CVAR_DEF_T (rt_dtal_groups, "0") \
	CVAR_DEF_T (rt_dtal_spacing, "128") \
    \
	CVAR_DEF_T (rt_reflrefr_depth, "2") \
	CVAR_DEF_T (rt_refr_glass, "1.52") \
	CVAR_DEF_T (rt_refr_water, "1.33") \
	CVAR_DEF_T (rt_glass_shadows, "1") \
	CVAR_DEF_T (rt_glass_denoise, "0") \
	CVAR_DEF_T (rt_glass_particles, "1") \
	CVAR_DEF_T (rt_particle_proxy_gate, "1") \
	\
	CVAR_DEF_T (rt_volume_type, "2") \
	/* The screen-space volumetric these parameterise is gone on the Q2RTX core:
	   CmPrepareFinal's applyVolumetrics returns at coreQ2RTX != 0, and
	   Volumetric::ProcessScattering has no caller. No output of either renderer
	   depends on them; kept as the setters of the public volumetric params. */ \
	CVAR_DEF_T (rt_volume_far, "1000") \
	CVAR_DEF_T (rt_volume_scatter, "0.3") \
	CVAR_DEF_T (rt_volume_ambient, "2.0") \
	CVAR_DEF_T (rt_volume_lintensity, "250") \
	CVAR_DEF_T (rt_volume_lassymetry, "0.0") \
    \
	CVAR_DEF_T (rt_water_speed, "0.4") \
	CVAR_DEF_T (rt_water_normstren, "1") \
	CVAR_DEF_T (rt_water_normsharp, "5") \
	CVAR_DEF_T (rt_water_scale, "1") \
	CVAR_DEF_T (rt_water_color, "171 193 210") \
	CVAR_DEF_T (rt_water_acidcolor, "122 143 21") \
	CVAR_DEF_T (rt_turb_warp, "1") \
	\
	CVAR_DEF_T (rt_portal_twirl, "1") \
	CVAR_DEF_T (rt_teleport_portals, "0") \
    \
	CVAR_DEF_T (rt_sharpen, "2") \
	CVAR_DEF_T (rt_sharpen_strength, "0.5") \
	CVAR_DEF_T (rt_renderscale, "0") \
	CVAR_DEF_T (rt_upscale_fsr31, "2") \
	CVAR_DEF_T (rt_upscale_dlss, "0") \
	\
	/* No pass reads the *SensitivityToChange fields: the ASVGF port is driven by
	   the depth gradient mode alone. Kept as the setters of the public
	   illumination params. */ \
	CVAR_DEF_T (rt_sensit_dir, "0.4") \
	CVAR_DEF_T (rt_sensit_indir, "0.06") \
	CVAR_DEF_T (rt_sensit_spec, "0.03") \
	\
	CVAR_DEF_T (rt_globallight_mult, "10") \
	CVAR_DEF_T (rt_globallight, "255 255 255") \
	\
	CVAR_DEF_T (rt_bloom_intensity, "0.1") \
	CVAR_DEF_T (rt_bloom_quality, "2") \
	CVAR_DEF_T (rt_bloom_threshold, "6.0") \
	CVAR_DEF_T (rt_bloom_knee, "0.5") \
	CVAR_DEF_T (rt_bloom_scatter, "0.7") \
	CVAR_DEF_T (rt_bloom_radius, "0.04") \
	CVAR_DEF_T (rt_bloom_emis_mult, "50") \
	CVAR_DEF_T (rt_bloom, "1") \
	\
	CVAR_DEF_T (rt_dof_near, "0.8") \
	\
	CVAR_DEF_T (rt_exposure_bias, "0") \
	CVAR_DEF_T (rt_exposure_speed_up, "3.0") \
	CVAR_DEF_T (rt_exposure_speed_down, "1.0") \
	CVAR_DEF_T (rt_exposure_low_percentile, "70") \
	CVAR_DEF_T (rt_exposure_high_percentile, "90") \
	CVAR_DEF_T (rt_exposure_min_luminance, "0.02") \
	CVAR_DEF_T (rt_exposure_max_luminance, "1.0") \
	CVAR_DEF_T (rt_local_exposure, "0.1") \
	CVAR_DEF_T (rt_tonemap_power, "0.9") \
	CVAR_DEF_T (rt_contrast, "0.9") \
	CVAR_DEF_T (rt_tonemap, "1") \
	\
	CVAR_DEF_T (rt_ef_crt, "0") \
	CVAR_DEF_T (rt_vignette, "0.5") \
	CVAR_DEF_T (rt_vignette_start, "0.45") \
	CVAR_DEF_T (rt_vignette_end, "1.0") \
	CVAR_DEF_T (rt_vignette_roundness, "0.35") \
	CVAR_DEF_T (rt_filmgrain, "0.5") \
	CVAR_DEF_T (rt_filmgrain_size, "1") \
	CVAR_DEF_T (rt_ef_chraber, "0.3") \
	CVAR_DEF_T (rt_ef_waves_stren, "1") \
	CVAR_DEF_T (rt_ef_damage, "1") \
	CVAR_DEF_T (rt_ef_damage_strength, "0.5") \
	CVAR_DEF_T (rt_ef_liquid, "1") \
	CVAR_DEF_T (rt_ef_liquid_strength, "0.51") \
	\
	CVAR_DEF_T (rt_viewm_fovscale, "1.2") \
	CVAR_DEF_T (rt_viewm_wide, "1.05") \
	CVAR_DEF_T (rt_viewm_scale, "0.32") \
	CVAR_DEF_T (rt_viewm_normalize, "1") \
	\
	CVAR_DEF_T (rt_hud_minimal, "1") \
	CVAR_DEF_T (rt_hud_padding, "8") \
	\
	CVAR_DEF_T (rt_debugflags, "0") \
	CVAR_DEF_T (rt_dtal_debug, "0") \
	CVAR_DEF_T (rt_q2_depthgrad, "1") \
	CVAR_DEF_T (rt_q2_lightstats, "1") \
	CVAR_DEF_T (rt_reflrefr_earlyout, "1") \
	CVAR_DEF_T (rt_nee_samples, "1") \
	CVAR_DEF_T (rt_restir, "0") \
	CVAR_DEF_T (rt_restir_candidates, "8") \
	CVAR_DEF_T (rt_stats_panels, "0") \
	CVAR_DEF_T (rt_stats_interval, "0.2") \
	CVAR_DEF_T (rt_end_task_delay_ms, "0") \
	CVAR_DEF_T (rt_worldcensus, "0") \
	CVAR_DEF_T (rt_worldlights_stats, "0") \
	CVAR_DEF_T (rt_particle_volume, "1") \
	CVAR_DEF_T (rt_particle_volume_check, "0") \
	CVAR_DEF_T (rt_worldclusters_grid, "1")



#define CVAR_DEF_T(name, default_value) cvar_t name = {#name, default_value, CVAR_ARCHIVE};
    CVAR_DEF_LIST (CVAR_DEF_T)
#undef CVAR_DEF_T

cvar_t rt_light_report_filter = {"rt_light_report_filter", "", 0};
cvar_t r_particles_overflow = {"r_particles_overflow", "0", CVAR_ARCHIVE};

// The sun editor is a mode, not a setting: it must not come back on after a
// restart, which would grab the sun without anybody asking for it.
cvar_t rt_sky_sun_edit = {"rt_sky_sun_edit", "0", 0};


/*
================
RT frame profiler -- rt_stats 3

Times the CPU side of the frame, which the GPU timestamps of panel 2 do not
cover: the geometry marking chain, the per-pass scene submission and the main
thread's wait for the task graph. The results are drawn on screen by
RT_StatsDrawGui, and rt_prof_report, which it reads, is rebuilt every
`rt_stats_interval` seconds (at most a fifth of a second, the rate the ImGui
overlay refreshes at; the setting may only make the readout faster) by
RT_Prof_Update; rt_stats_dump writes one snapshot to qperfdump.log.

================
*/
double           rt_prof_ms[RT_PROF_COUNT];
rt_prof_report_t rt_prof_report;

int      rt_particles_classic;
int      rt_particles_fte;
int      rt_particles_vertices;
int      rt_particles_smoke;
int      rt_particles_dropped;
int      rt_particles_emit_culled;
int      rt_particles_emit_faded;
uint64_t rt_particles_volume_samples;
uint64_t rt_particles_volume_mismatch;
uint64_t rt_particles_volume_lost;
uint64_t rt_particles_volume_extra;
uint64_t rt_fte_convert_bytes;
uint64_t rt_particle_upload_bytes;

static uint64_t rt_particle_cache_hits_last;
static uint64_t rt_particle_cache_misses_last;
static double   rt_particle_cache_total_ms_last;

static double   rt_prof_frame_start;
static rt_prof_window_t rt_prof_window;
static qboolean rt_prof_capture_frame;
static double   rt_prof_sum[RT_PROF_COUNT];
static double   rt_renderer_cpu_sum[QR_CPU_PASS_COUNT];
static float    rt_renderer_cpu_max[QR_CPU_PASS_COUNT];
static int      rt_renderer_cpu_samples;

/* rt_bench: the same slots, summed over a whole demo run instead of maximized over one
   `rt_stats_interval` window. The on-screen panel wants the worst frame of the window; a
   benchmark wants the cost of an average frame, so every slot is added up and the longest
   one kept as well.
   The run is started by CL_Bench_f when the timedemo clock starts and reported when the demo
   ends, and the profiler is forced on for its duration whatever rt_stats asks for. */
static qboolean rt_bench_active;
static qboolean rt_bench_interrupted;
static int      rt_bench_frames;
static double   rt_bench_start_time;
static double   rt_bench_frame_min;
static double   rt_bench_sum[RT_PROF_COUNT];
static double   rt_bench_max[RT_PROF_COUNT];
static double   rt_bench_host_sum[RT_HOST_SPEED_COUNT];
static int      rt_bench_host_frames;
static double   rt_bench_frame_slots[RT_PROF_COUNT];
static atomic_uint32_t rt_prof_spin;

static void RT_Prof_Lock (void)
{
	uint32_t expected = 0;
	while (!Atomic_CompareExchangeUInt32 (&rt_prof_spin, &expected, 1))
		expected = 0;
}

static void RT_Prof_Unlock (void)
{
	Atomic_StoreUInt32 (&rt_prof_spin, 0);
}

static void RT_Prof_WindowLock (void *unused)
{
	RT_Prof_Lock ();
}

static void RT_Prof_WindowUnlock (void *unused)
{
	RT_Prof_Unlock ();
}

static const rt_prof_window_sync_t rt_prof_window_sync = {NULL, RT_Prof_WindowLock, RT_Prof_WindowUnlock};

static void RT_Prof_ClearWindow (void *unused)
{
	memset (rt_prof_ms, 0, sizeof (rt_prof_ms));
	memset (rt_prof_sum, 0, sizeof (rt_prof_sum));
	memset (rt_renderer_cpu_sum, 0, sizeof (rt_renderer_cpu_sum));
	memset (rt_renderer_cpu_max, 0, sizeof (rt_renderer_cpu_max));
	rt_renderer_cpu_samples = 0;
}

static void RT_Prof_ResetWindow (void *unused)
{
	RT_Prof_ClearWindow (NULL);
	rt_prof_report.valid = false;
}

static double   rt_bench_last_frame_end;
static QrFrameStats rt_bench_renderer_stats;
static uint32_t     rt_end_task_serial;
static uint32_t     rt_end_task_current_serial;
static uint32_t     rt_end_task_finished_serial;
static double       rt_end_task_finished_ms;
static QrFrameStats rt_end_task_finished_stats;
static uint32_t     rt_bench_pending_end_serial;
static int          rt_bench_pending_end_index;

#define RT_BENCH_FRAME_CAPACITY 32768

typedef struct
{
	double       timeMs;
	double       intervalMs;
	double       hostIntervalMs;
	double       clientTime;
	double       slots[RT_PROF_COUNT];
	QrFrameStats renderer;
	int          keyGame;
	int          paused;
	int          signon;
} rt_bench_frame_t;

static rt_bench_frame_t *rt_bench_frame_samples;
static int              rt_bench_frame_sample_count;
static unsigned         rt_bench_capture_id;

/* What the results screen of the benchmark menu shows, filled by RT_Bench_Report. */
rt_bench_result_t rt_bench_result;

qboolean RT_Bench_Active (void)
{
	return rt_bench_active;
}

void RT_Bench_Start (void)
{
	if (!rt_bench_frame_samples)
		rt_bench_frame_samples = Mem_Alloc (RT_BENCH_FRAME_CAPACITY * sizeof (*rt_bench_frame_samples));

	rt_bench_active = true;
	rt_bench_interrupted = false;
	rt_bench_frames = 0;
	rt_bench_start_time = Sys_DoubleTime ();
	rt_bench_frame_min = 0.0;
	rt_bench_last_frame_end = 0.0;
	rt_bench_frame_sample_count = 0;
	RT_Prof_Lock ();
	memset (rt_bench_frame_slots, 0, sizeof (rt_bench_frame_slots));
	memset (&rt_bench_renderer_stats, 0, sizeof (rt_bench_renderer_stats));
	memset (rt_bench_sum, 0, sizeof (rt_bench_sum));
	memset (rt_bench_max, 0, sizeof (rt_bench_max));
	memset (rt_bench_host_sum, 0, sizeof (rt_bench_host_sum));
	rt_end_task_finished_serial = 0;
	memset (&rt_end_task_finished_stats, 0, sizeof (rt_end_task_finished_stats));
	rt_bench_pending_end_serial = 0;
	RT_Prof_Unlock ();
	rt_bench_host_frames = 0;
	rt_cluster_cache_hits = 0;
	rt_cluster_cache_misses = 0;
	rt_cluster_miss_set = 0;
	rt_cluster_miss_move = 0;
	rt_cluster_miss_other = 0;
	rt_cluster_pub_skips = 0;
	rt_cluster_pub_copies = 0;
	rt_cluster_last_pub_mask = 0;
	rt_particles_dropped = 0;
	rt_particles_emit_culled = 0;
	rt_particles_emit_faded = 0;
	RT_ClusterVolumeResetVerify ();
	rt_fte_convert_bytes = 0;
	rt_particle_upload_bytes = 0;
}

void RT_Bench_Stop (void)
{
	rt_bench_active = false;
}

void RT_Bench_Interrupt (void)
{
	if (rt_bench_active)
		rt_bench_interrupted = true;
}

qboolean RT_Bench_Interrupted (void)
{
	return rt_bench_interrupted;
}

static void RT_Bench_Slot (int slot, double ms)
{
	if (!rt_bench_active)
		return;

	rt_bench_sum[slot] += ms;
	rt_bench_frame_slots[slot] += ms;
	if (ms > rt_bench_max[slot])
		rt_bench_max[slot] = ms;

	if (slot == RT_PROF_FRAME && (rt_bench_frames == 0 || ms < rt_bench_frame_min))
		rt_bench_frame_min = ms;
}

void RT_Bench_HostFrame (void)
{
	if (!rt_bench_active)
		return;

	for (int i = 0; i < RT_HOST_SPEED_COUNT; i++)
		rt_bench_host_sum[i] += rt_host_speeds_ms[i];

	++rt_bench_host_frames;
}

double RT_Prof_Begin (void)
{
	return (RT_StatsPanel (RT_STATS_PROFILE) || rt_bench_active) ? Sys_DoubleTime () : 0.0;
}

void RT_Prof_End (int slot, double start)
{
	if (start == 0.0)
		return;

	const double ms = (Sys_DoubleTime () - start) * 1000.0;
	RT_Prof_Lock ();
	RT_Bench_Slot (slot, ms);
	rt_prof_sum[slot] += ms;
	if (ms > rt_prof_ms[slot])
		rt_prof_ms[slot] = ms;
	RT_Prof_Unlock ();
}

void RT_Prof_Sample (int slot, double ms)
{
	RT_Prof_Lock ();
	RT_Bench_Slot (slot, ms);
	rt_prof_sum[slot] += ms;
	if (ms > rt_prof_ms[slot])
		rt_prof_ms[slot] = ms;
	RT_Prof_Unlock ();
}

void RT_Prof_FrameStart (void)
{
	GL_SynchronizeEndRenderingTask ();
	const qboolean enabled = RT_StatsPanel (RT_STATS_PROFILE);
	rt_prof_capture_frame = enabled || rt_bench_active;
	const double now = Sys_DoubleTime ();
	RT_ProfWindowBeginFrame (&rt_prof_window, &rt_prof_window_sync, enabled, rt_prof_capture_frame, now, RT_Prof_ResetWindow, NULL);
	if (!rt_prof_capture_frame)
		return;

	rt_prof_frame_start = now;
	if (rt_bench_active)
	{
		RT_Prof_Lock ();
		memset (rt_bench_frame_slots, 0, sizeof (rt_bench_frame_slots));
		memset (&rt_bench_renderer_stats, 0, sizeof (rt_bench_renderer_stats));
		RT_Prof_Unlock ();
	}

	RT_PointClusterCacheStats (NULL, NULL, &rt_particle_cache_total_ms_last, NULL);
}

void RT_Prof_FrameEnd (void)
{
	if (!rt_prof_capture_frame)
		return;

	const uint32_t end_serial = rt_end_task_current_serial;
	rt_end_task_current_serial = 0;

	RT_Prof_End (RT_PROF_FRAME, rt_prof_frame_start);

	double cacheTotalMs;
	RT_PointClusterCacheStats (NULL, NULL, &cacheTotalMs, NULL);
	RT_Prof_Sample (RT_PROF_PARTICLES_RESOLVE, cacheTotalMs - rt_particle_cache_total_ms_last);
	rt_particle_cache_total_ms_last = cacheTotalMs;

	RT_ProfWindowEndFrame (&rt_prof_window, &rt_prof_window_sync);
	if (rt_bench_active)
	{
		const double now = Sys_DoubleTime ();
		RT_Prof_Lock ();
		if (rt_bench_frame_sample_count < RT_BENCH_FRAME_CAPACITY)
		{
			rt_bench_frame_t *sample = &rt_bench_frame_samples[rt_bench_frame_sample_count++];
			sample->timeMs = (now - rt_bench_start_time) * 1000.0;
			sample->intervalMs = rt_bench_last_frame_end > 0.0 ? (now - rt_bench_last_frame_end) * 1000.0 : 0.0;
			sample->hostIntervalMs = host_rawframetime * 1000.0;
			sample->clientTime = cl.time;
			memcpy (sample->slots, rt_bench_frame_slots, sizeof (sample->slots));
			sample->renderer = rt_bench_renderer_stats;
			sample->keyGame = key_dest == key_game;
			sample->paused = sv.paused || cl.paused;
			sample->signon = cls.signon;

			if (end_serial != 0)
			{
				sample->slots[RT_PROF_DRAWFRAME] = 0.0;
				memset (&sample->renderer, 0, sizeof (sample->renderer));

				if (RT_EndFrameConsumesEarlyResult (end_serial, rt_end_task_finished_serial))
				{
					sample->slots[RT_PROF_DRAWFRAME] += rt_end_task_finished_ms;
					sample->renderer = rt_end_task_finished_stats;
					rt_end_task_finished_serial = 0;
				}
				else
				{
					rt_bench_pending_end_serial = end_serial;
					rt_bench_pending_end_index = rt_bench_frame_sample_count - 1;
				}
			}
		}
		for (int i = 0; i < RT_PROF_COUNT; ++i)
			if (rt_bench_frame_slots[i] > rt_bench_max[i])
				rt_bench_max[i] = rt_bench_frame_slots[i];
		RT_Prof_Unlock ();
		rt_bench_last_frame_end = now;
		++rt_bench_frames;
	}
}

static void RT_Prof_RecordRenderer (void)
{
	QrFrameStats stats = {0};

	if (qrGetFrameStatsEx (vulkan_globals.instance, &stats) != QR_SUCCESS || !stats.cpuTimingValid)
		return;

	RT_Prof_Lock ();
	if (rt_bench_active)
		rt_bench_renderer_stats = stats;

	++rt_renderer_cpu_samples;
	for (int i = 0; i < QR_CPU_PASS_COUNT; i++)
	{
		rt_renderer_cpu_sum[i] += stats.cpuPassMs[i];
		if (stats.cpuPassMs[i] > rt_renderer_cpu_max[i])
			rt_renderer_cpu_max[i] = stats.cpuPassMs[i];
	}
	RT_Prof_Unlock ();
}

typedef struct
{
	uint32_t serial;
	double ms;
	qboolean measured;
	const QrFrameStats *stats;
} rt_prof_end_result_t;

static void RT_Prof_RecordEndResult (void *context)
{
	const rt_prof_end_result_t *result = context;
	if (!result->measured)
		return;

	const double ms = result->ms;
	const uint32_t serial = result->serial;
	const QrFrameStats *stats = result->stats;

	rt_prof_sum[RT_PROF_DRAWFRAME] += ms;
	if (ms > rt_prof_ms[RT_PROF_DRAWFRAME])
		rt_prof_ms[RT_PROF_DRAWFRAME] = ms;

	if (stats)
	{
		++rt_renderer_cpu_samples;
		for (int i = 0; i < QR_CPU_PASS_COUNT; i++)
		{
			rt_renderer_cpu_sum[i] += stats->cpuPassMs[i];
			if (stats->cpuPassMs[i] > rt_renderer_cpu_max[i])
				rt_renderer_cpu_max[i] = stats->cpuPassMs[i];
		}
	}

	if (rt_bench_active)
	{
		rt_bench_sum[RT_PROF_DRAWFRAME] += ms;
		if (ms > rt_bench_max[RT_PROF_DRAWFRAME])
			rt_bench_max[RT_PROF_DRAWFRAME] = ms;

		if (RT_EndTaskResultMergesNow (rt_bench_pending_end_serial, serial))
		{
			rt_bench_frame_t *sample = &rt_bench_frame_samples[rt_bench_pending_end_index];

			sample->slots[RT_PROF_DRAWFRAME] += ms;
			if (stats)
				sample->renderer = *stats;
			rt_bench_pending_end_serial = 0;
		}
		else
		{
			rt_end_task_finished_serial = serial;
			rt_end_task_finished_ms = ms;
			if (stats)
				rt_end_task_finished_stats = *stats;
			else
				memset (&rt_end_task_finished_stats, 0, sizeof (rt_end_task_finished_stats));
		}
	}
}

static void RT_Prof_EndTaskRecord (uint32_t serial, double start, const QrFrameStats *stats)
{
	rt_prof_end_result_t result = {
		.serial = serial,
		.ms = start != 0.0 ? (Sys_DoubleTime () - start) * 1000.0 : 0.0,
		.measured = start != 0.0,
		.stats = stats,
	};
	RT_ProfWindowRecordEnd (&rt_prof_window, &rt_prof_window_sync, RT_Prof_RecordEndResult, &result);
}

static void RT_Prof_PublishWindow (void *unused, unsigned frames, double elapsed, unsigned publication)
{
	float fps = (float)(frames / elapsed);
	if (fps < 0.1f)
		fps = 0.1f;

	memset (&rt_prof_report, 0, sizeof (rt_prof_report));
	rt_prof_report.windowId = publication;
	rt_prof_report.frames = frames;
	rt_prof_report.fps = fps;
	rt_prof_report.frameMs = (float)rt_prof_ms[RT_PROF_FRAME];
	rt_prof_report.waitMs = (float)rt_prof_ms[RT_PROF_WAIT];

	for (int i = 0; i < RT_PROF_COUNT; ++i)
	{
		rt_prof_report.ms[i] = (float)rt_prof_ms[i];
		rt_prof_report.averageMs[i] = (float)(rt_prof_sum[i] / frames);
	}
	for (int i = 0; i < QR_CPU_PASS_COUNT; ++i)
	{
		rt_prof_report.rendererAverageMs[i] = rt_renderer_cpu_samples > 0
		    ? (float)(rt_renderer_cpu_sum[i] / rt_renderer_cpu_samples) : 0.0f;
		rt_prof_report.rendererMaxMs[i] = rt_renderer_cpu_max[i];
	}
	rt_prof_report.rendererSamples = rt_renderer_cpu_samples;

	rt_prof_report.clusterCacheHits = rt_cluster_cache_hits;
	rt_prof_report.clusterCacheMisses = rt_cluster_cache_misses;
	rt_prof_report.clusterMissSet = rt_cluster_miss_set;
	rt_prof_report.clusterMissMove = rt_cluster_miss_move;
	rt_prof_report.clusterMissOther = rt_cluster_miss_other;
	rt_prof_report.clusterDirty = rt_cluster_last_dirty;
	rt_prof_report.clusterMoveFootprint = rt_cluster_last_move_footprint;
	rt_prof_report.clusterGrants = rt_cluster_last_grants;
	rt_prof_report.clusterDenied = rt_cluster_last_denied;
	rt_prof_report.clusterGated = rt_cluster_last_gated;
	rt_prof_report.clusterLights = rt_cluster_last_lights;
	rt_prof_report.clusterAttempts = rt_cluster_last_attempts;
	rt_prof_report.clusterDropped = rt_cluster_last_dropped;
	rt_prof_report.clusterPublicationMask = rt_cluster_last_pub_mask;
	rt_prof_report.clusterPublicationSkips = rt_cluster_pub_skips;
	rt_prof_report.clusterPublicationCopies = rt_cluster_pub_copies;
	rt_prof_report.particlesClassic = rt_particles_classic;
	rt_prof_report.particlesFte = rt_particles_fte;
	rt_prof_report.particlesVertices = rt_particles_vertices;
	rt_prof_report.particlesSmoke = rt_particles_smoke;
	rt_prof_report.particlesDropped = rt_particles_dropped;
	rt_prof_report.particlesEmitCulled = rt_particles_emit_culled;
	rt_prof_report.particlesEmitFaded = rt_particles_emit_faded;
	RT_ClusterVolumeVerifyStats (&rt_particles_volume_samples, &rt_particles_volume_mismatch,
		&rt_particles_volume_lost, &rt_particles_volume_extra);
	rt_prof_report.volumeVerifySamples = rt_particles_volume_samples;
	rt_prof_report.volumeVerifyMismatch = rt_particles_volume_mismatch;
	rt_prof_report.volumeVerifyLost = rt_particles_volume_lost;
	rt_prof_report.volumeVerifyExtra = rt_particles_volume_extra;
	rt_prof_report.fteConvertBytes = rt_fte_convert_bytes;
	rt_prof_report.particleUploadBytes = rt_particle_upload_bytes;

	uint64_t particleCacheHits, particleCacheMisses;
	double   particleCacheAvgNs;
	RT_PointClusterCacheStats (&particleCacheHits, &particleCacheMisses, NULL, &particleCacheAvgNs);
	rt_prof_report.particleResolveCacheHits = particleCacheHits - rt_particle_cache_hits_last;
	rt_prof_report.particleResolveCacheMisses = particleCacheMisses - rt_particle_cache_misses_last;
	rt_prof_report.particleResolveCacheAvgNs = particleCacheAvgNs;
	rt_particle_cache_hits_last = particleCacheHits;
	rt_particle_cache_misses_last = particleCacheMisses;
	rt_prof_report.valid = true;
}

void RT_Prof_Update (void)
{
	if (!RT_StatsPanel (RT_STATS_PROFILE))
		return;

	double interval = CLAMP (0.05, CVAR_TO_FLOAT (rt_stats_interval), 0.2);
	if (!(interval >= 0.05 && interval <= 0.2))
		interval = 0.2;
	if (!RT_ProfWindowTryPublish (&rt_prof_window, &rt_prof_window_sync, Sys_DoubleTime (), interval,
		RT_Prof_PublishWindow, RT_Prof_ClearWindow, NULL))
		return;

	if (!rt_bench_active)
	{
		rt_cluster_cache_hits = 0;
		rt_cluster_cache_misses = 0;
		rt_cluster_miss_set = 0;
		rt_cluster_miss_move = 0;
		rt_cluster_miss_other = 0;
		rt_cluster_pub_skips = 0;
		rt_cluster_pub_copies = 0;
		rt_cluster_last_pub_mask = 0;
		rt_particles_dropped = 0;
		rt_particles_emit_culled = 0;
		rt_particles_emit_faded = 0;
		RT_ClusterVolumeResetVerify ();
		rt_fte_convert_bytes = 0;
		rt_particle_upload_bytes = 0;
	}
}

/*
================
RT_Bench_Report

Appends the accumulated profile of a demo run to benchmark.log in the game directory. The
header names the demo, the frame count, the wall time and the settings that shape the host
frame, so two runs can only differ in what the code does; every slot follows with its average
and its longest time, and the cluster counters close the run. Play the same demo before and
after a change and diff the two blocks, which is what the benchmark is for.
================
*/
#define RT_BENCH_FILE "benchmark.log"

static void RT_Bench_Setting (FILE *f, const char *name)
{
	const cvar_t *var = Cvar_FindVar (name);

	// a name that no longer exists is marked instead of printed empty
	fprintf (f, " %s=%s", name, var ? var->string : "?");
}

static void RT_Bench_Column (FILE *f, const char *prefix, const char *name)
{
	fprintf (f, ",%s", prefix);
	for (; *name; ++name)
		fputc (*name == ' ' ? '_' : *name, f);
	fputs ("_ms", f);
}

static void RT_Bench_WriteFrames (FILE *log, const char *demo)
{
	char stamp[32];
	char name[80];
	char path[MAX_OSPATH];
	const time_t now = time (NULL);
	struct tm *local = localtime (&now);
	FILE *f;

	if (local)
		strftime (stamp, sizeof (stamp), "%Y%m%d-%H%M%S", local);
	else
		q_strlcpy (stamp, "unknown", sizeof (stamp));
	q_snprintf (name, sizeof (name), "benchmark-frames-%s-%u.csv", stamp, ++rt_bench_capture_id);
	q_snprintf (path, sizeof (path), "%s/%s", com_gamedir, name);
	f = fopen (path, "w");
	if (!f)
	{
		Con_Printf ("rt_bench: could not write frame samples to %s\n", path);
		return;
	}

	fprintf (f, "# rt_bench_frames demo=%s frames=%d samples=%d dropped=%d\n", demo ? demo : "?",
	         rt_bench_frames, rt_bench_frame_sample_count, rt_bench_frames - rt_bench_frame_sample_count);
	fputs ("frame,time_ms,interval_ms,host_interval_ms,client_time,key_game,paused,signon,gpu_valid,gpu.frame_ms", f);
	for (int i = 0; i < RT_PROF_COUNT; ++i)
		RT_Bench_Column (f, "cpu.", RT_ProfSlotName (i));
	for (int i = 0; i < QR_CPU_PASS_COUNT; ++i)
		RT_Bench_Column (f, "cpu.draw.", qrGetCpuPassName (i));
	for (int i = 0; i < QR_GPU_PASS_COUNT; ++i)
		RT_Bench_Column (f, "gpu.", qrGetGpuPassName (i));
	fputs (",calls_geometry,calls_raster,calls_lights,ui_only\n", f);

	for (int i = 0; i < rt_bench_frame_sample_count; ++i)
	{
		const rt_bench_frame_t *sample = &rt_bench_frame_samples[i];
		const QrFrameStats *stats = &sample->renderer;
		fprintf (f, "%d,%.4f,%.4f,%.4f,%.6f,%d,%d,%d,%u,%.4f", i, sample->timeMs,
		         sample->intervalMs, sample->hostIntervalMs, sample->clientTime,
		         sample->keyGame, sample->paused, sample->signon, stats->gpuTimingValid, stats->gpuFrameMs);
		for (int j = 0; j < RT_PROF_COUNT; ++j)
			fprintf (f, ",%.4f", sample->slots[j]);
		for (int j = 0; j < QR_CPU_PASS_COUNT; ++j)
			fprintf (f, ",%.4f", stats->cpuPassMs[j]);
		for (int j = 0; j < QR_GPU_PASS_COUNT; ++j)
			fprintf (f, ",%.4f", stats->gpuPassMs[j]);
		fprintf (f, ",%u,%u,%u,%u\n", stats->apiCallsGeometry, stats->apiCallsRasterized,
		         stats->apiCallsLights, stats->renderedUiOnly);
	}
	fclose (f);
	fprintf (log, "frame_samples file=%s samples=%d dropped=%d\n", name,
	         rt_bench_frame_sample_count, rt_bench_frames - rt_bench_frame_sample_count);
}

qboolean RT_Bench_Report (const char *demo)
{
	RT_Prof_Lock ();
	char        path[MAX_OSPATH];
	char        stamp[32];
	time_t      now;
	struct tm  *local;
	FILE       *f;
	const int    frames = rt_bench_frames > 0 ? rt_bench_frames : 1;
	const double seconds = rt_bench_frames > 0 ? (Sys_DoubleTime () - rt_bench_start_time) : 0.0;

	if (!RT_ReportTakeRun (&rt_bench_active))
	{
		RT_Prof_Unlock ();
		return false;
	}

	/* A run that ended before a single frame was finished has nothing to report and no result
	   to show; it is not a measurement of zero frames. */
	if (rt_bench_frames == 0)
	{
		rt_bench_result.valid = false;
		RT_Prof_Unlock ();
		return false;
	}

	double slotSum[RT_PROF_COUNT];
	double slotMax[RT_PROF_COUNT];
	double hostSum[RT_HOST_SPEED_COUNT];
	const int interrupted = rt_bench_interrupted;
	const int clusterHits = rt_cluster_cache_hits;
	const int clusterMisses = rt_cluster_cache_misses;
	const int clusterSet = rt_cluster_miss_set;
	const int clusterMove = rt_cluster_miss_move;
	const int clusterOther = rt_cluster_miss_other;

	memcpy (slotSum, rt_bench_sum, sizeof (slotSum));
	memcpy (slotMax, rt_bench_max, sizeof (slotMax));
	memcpy (hostSum, rt_bench_host_sum, sizeof (hostSum));
	RT_Prof_Unlock ();

	const double frameAvg = slotSum[RT_PROF_FRAME] / frames;
	const double frameMin = rt_bench_frame_min > 0.0 ? rt_bench_frame_min : frameAvg;
	const double frameMax = slotMax[RT_PROF_FRAME];

	rt_bench_result.valid = true;
	q_strlcpy (rt_bench_result.demo, (demo && demo[0]) ? demo : "?", sizeof (rt_bench_result.demo));
	rt_bench_result.frames = rt_bench_frames;
	rt_bench_result.seconds = seconds;
	rt_bench_result.frameAvgMs = frameAvg;
	rt_bench_result.frameMinMs = frameMin;
	rt_bench_result.frameMaxMs = frameMax;
	rt_bench_result.fpsAvg = frameAvg > 0.0 ? 1000.0 / frameAvg : 0.0;
	rt_bench_result.fpsMin = frameMax > 0.0 ? 1000.0 / frameMax : 0.0;
	rt_bench_result.fpsMax = frameMin > 0.0 ? 1000.0 / frameMin : 0.0;

	q_snprintf (path, sizeof (path), "%s/%s", com_gamedir, RT_BENCH_FILE);

	now = time (NULL);
	local = localtime (&now);
	if (local)
		strftime (stamp, sizeof (stamp), "%Y-%m-%d %H:%M:%S", local);
	else
		stamp[0] = 0;

	f = fopen (path, "a");

	if (!f)
	{
		Con_Printf ("rt_bench: could not write %s\n", path);
		return true; // the result itself is there, only the log file is not
	}

	fprintf (f, "# rt_bench %s demo=%s frames=%d seconds=%.2f fps=%.1f interrupted=%d\n", stamp,
	         (demo && demo[0]) ? demo : "?", frames, seconds,
	         seconds > 0.0 ? frames / seconds : 0.0, interrupted ? 1 : 0);

	for (int i = 0; i < RT_PROF_COUNT; i++)
		fprintf (f, "cpu.slot    %-17s avg_ms=%.2f max_ms=%.2f\n", RT_ProfSlotName (i),
		         slotSum[i] / frames, slotMax[i]);

	fprintf (f, "cpu.main    %-17s avg_ms=%.2f\n", "frame minus wait",
	         (slotSum[RT_PROF_FRAME] - slotSum[RT_PROF_WAIT]) / frames);

	static const char *const hostPasses[RT_HOST_SPEED_COUNT] = {"tot", "server", "gfx", "snd"};
	const int                hostFrames = rt_bench_host_frames > 0 ? rt_bench_host_frames : 1;

	for (int i = 0; i < RT_HOST_SPEED_COUNT; i++)
		fprintf (f, "host.pass   %-17s avg_ms=%.2f\n", hostPasses[i], hostSum[i] / hostFrames);

	fprintf (f, "cpu.cluster %-17s hits=%d misses=%d set=%d move=%d other=%d pubskips=%d pubcopies=%d\n", "lists",
	         clusterHits, clusterMisses, clusterSet,
	         clusterMove, clusterOther, rt_cluster_pub_skips, rt_cluster_pub_copies);

	fprintf (f, "settings");
	RT_Bench_Setting (f, "host_maxfps");
	RT_Bench_Setting (f, "host_timescale");
	RT_Bench_Setting (f, "fov");
	RT_Bench_Setting (f, "vid_maxframelatency");
	RT_Bench_Setting (f, "r_drawviewmodel");
	RT_Bench_Setting (f, "r_enhancedmodels");
	RT_Bench_Setting (f, "r_simd");
	RT_Bench_Setting (f, "r_lerpmodels");
	RT_Bench_Setting (f, "r_smoke");
	RT_Bench_Setting (f, "rt_enable_pvs");
	RT_Bench_Setting (f, "sv_novis");
	RT_Bench_Setting (f, "rt_truelight");
	RT_Bench_Setting (f, "rt_world_batch_merge");
	RT_Bench_Setting (f, "rt_brush_persistent");
	RT_Bench_Setting (f, "rt_wmodel_lights_batch");
	RT_Bench_Setting (f, "rt_cluster_incremental");
	RT_Bench_Setting (f, "rt_cluster_sampling");
	RT_Bench_Setting (f, "rt_cluster_dlights");
	RT_Bench_Setting (f, "rt_light_reach_static");
	RT_Bench_Setting (f, "rt_light_reach_dynamic");
	RT_Bench_Setting (f, "rt_light_styles");
	RT_Bench_Setting (f, "r_particles");
	RT_Bench_Setting (f, "r_fteparticles");
	RT_Bench_Setting (f, "rt_model_lights");
	RT_Bench_Setting (f, "rt_dtal_model_maxpolys");
	RT_Bench_Setting (f, "rt_dtal_model_budget");
	RT_Bench_Setting (f, "rt_dtal_model_minarea");
	RT_Bench_Setting (f, "rt_dtal_minarea");
	RT_Bench_Setting (f, "rt_dtal_maxpolys");
	RT_Bench_Setting (f, "rt_dtal_clearance");
	RT_Bench_Setting (f, "rt_dtal_groups");
	RT_Bench_Setting (f, "rt_dtal_spacing");
	RT_Bench_Setting (f, "rt_shadowrays");
	RT_Bench_Setting (f, "rt_sky_godrays");
	RT_Bench_Setting (f, "rt_sky_godrays_intensity");
	RT_Bench_Setting (f, "rt_sky_godrays_quality");
	RT_Bench_Setting (f, "rt_sky_godrays_sky_threshold");
	RT_Bench_Setting (f, "rt_physical_sky");
	RT_Bench_Setting (f, "rt_physical_sun");
	RT_Bench_Setting (f, "rt_sky_sun_size");
	RT_Bench_Setting (f, "rt_sky_clouds");
	RT_Bench_Setting (f, "rt_sky_clouds_coverage");
	RT_Bench_Setting (f, "rt_sky_clouds_density");
	RT_Bench_Setting (f, "rt_sky_clouds_speed");
	RT_Bench_Setting (f, "rt_sky_clouds_quality");
	RT_Bench_Setting (f, "rt_sky_clouds_height");
	RT_Bench_Setting (f, "rt_sky_clouds_thickness");
	RT_Bench_Setting (f, "rt_denoiser");
	RT_Bench_Setting (f, "rt_reflrefr_depth");
	RT_Bench_Setting (f, "rt_glass_shadows");
	RT_Bench_Setting (f, "rt_glass_denoise");
	RT_Bench_Setting (f, "rt_glass_particles");
	RT_Bench_Setting (f, "rt_particle_proxy_gate");
	RT_Bench_Setting (f, "rt_bloom");
	RT_Bench_Setting (f, "rt_bloom_quality");
	RT_Bench_Setting (f, "rt_local_exposure");
	RT_Bench_Setting (f, "rt_gi_level");
	RT_Bench_Setting (f, "rt_nee_samples");
	RT_Bench_Setting (f, "rt_restir");
	RT_Bench_Setting (f, "rt_restir_candidates");
	RT_Bench_Setting (f, "rt_renderscale");
	RT_Bench_Setting (f, "rt_upscale_fsr31");
	RT_Bench_Setting (f, "rt_upscale_dlss");
	RT_Bench_Setting (f, "rt_stats_panels");
	RT_Bench_Setting (f, "r_particles");
	RT_Bench_Setting (f, "r_particle_lighting");
	RT_Bench_Setting (f, "r_fteparticles");
	RT_Bench_Setting (f, "r_smoke");
	RT_Bench_Setting (f, "rt_particle_resolve_cache");
	RT_Bench_Setting (f, "r_particles_points");
	RT_Bench_Setting (f, "r_part_emit_distance");
	RT_Bench_Setting (f, "rt_particle_volume");
	RT_Bench_Setting (f, "rt_particle_volume_check");
	fprintf (f, " vid=%dx%d@%d vsync=%d version=%s\n", vid.width, vid.height, vid_display_refresh,
	         (int)vid_vsync.value, ENGINE_VER_STRING);

	RT_Bench_WriteFrames (f, demo);
	fclose (f);

	Con_Printf ("rt_bench: %d frames, %.2f s, %.1f fps -> %s\n", rt_bench_frames, seconds,
	            seconds > 0.0 ? frames / seconds : 0.0, path);

	for (int i = 0; i < RT_PROF_COUNT; i++)
		Con_Printf ("  %-17s avg %.2f ms, max %.2f ms\n", RT_ProfSlotName (i),
		            slotSum[i] / frames, slotMax[i]);

	for (int i = 0; i < RT_HOST_SPEED_COUNT; i++)
		Con_Printf ("  host %-12s avg %.2f ms\n", hostPasses[i], hostSum[i] / hostFrames);

	return true;
}


/*
================
RT_StatsPanel

One bit per panel number, so the readouts share a single switch and a single
line in the config file. The command is the only writer of the cvar, which is
why the value can be read straight from here.
================
*/
qboolean RT_StatsPanel (int panel)
{
	if (panel < RT_STATS_RAYS || panel > RT_STATS_PROFILE)
		return false;

	return (CVAR_TO_UINT32 (rt_stats_panels) & (1u << (panel - 1))) != 0;
}

/*
================
RT_StatsPanelsFixup -- keeps the panels cvar a level

The command writes the panels of a level, but the cvar is archived, and a value
written before the command took levels can name a panel without the ones below
it: an old "rt_stats 3" was the CPU panel alone, a mask of 4. The readout has no
such state any more, so the panels below a set one are folded in --- the smallest
level that shows everything the value asked for --- and the value is written
back, which is what makes the config, the dump and the benchmark log name a
level too. A value that is not one of the eight masks at all (a huge or negative
number typed in the console) folds into the level of no panels, which is off.
================
*/
static void RT_StatsPanelsFixup (cvar_t *var)
{
	const float    raw = var->value;
	const unsigned mask = (raw > 0.0f && raw < 8.0f) ? (unsigned)raw : 0u;
	const unsigned level = mask | (mask >> 1) | (mask >> 2);

	if (mask != level)
		Cvar_SetValueQuick (var, (float)level);

	RT_StatsGuiReset ();
}

/*
================
RT_ProfSlotName

Labels of the profile slots, shared by the on-screen panel and the dump so both
spell the same measurement the same way.
================
*/
const char *RT_ProfSlotName (int slot)
{
	static const struct
	{
		int         slot;
		const char *name;
	} names[] = {
		{ RT_PROF_SETUP, "setup" },
		{ RT_PROF_MARK, "mark" },
		{ RT_PROF_EFRAGS, "efrags" },
		{ RT_PROF_CULL, "cull" },
		{ RT_PROF_CHAIN, "chain" },
		{ RT_PROF_WORLD, "world" },
		{ RT_PROF_SKY, "sky" },
		{ RT_PROF_ENTS, "ents" },
		{ RT_PROF_ENTS_ALIAS, "ents alias" },
		{ RT_PROF_ENTS_BRUSH, "ents brush" },
		{ RT_PROF_ENTS_SPRITE, "ents sprite" },
		{ RT_PROF_BRUSH_LIGHTMARK, "brush lightmark" },
		{ RT_PROF_ALPHA, "alpha" },
		{ RT_PROF_PARTICLES, "particles" },
		{ RT_PROF_VIEWMODEL, "viewmodel" },
		{ RT_PROF_VIEWMODEL_DRAW, "vm draw" },
		{ RT_PROF_ELIGHTS, "elights" },
		{ RT_PROF_WMODEL_LIGHTS, "wmodel lights" },
		{ RT_PROF_TELEPORTS, "teleports" },
		{ RT_PROF_CLUSTERS, "clusters" },
		{ RT_PROF_CLUSTERS_LISTS, "clust lists" },
		{ RT_PROF_CLUSTERS_RESOLVE, "clust resolve" },
		{ RT_PROF_CLUSTERS_VIS, "clust vis" },
		{ RT_PROF_CLUSTERS_TOPUP, "clust topup" },
		{ RT_PROF_CLUSTERS_MARK, "clust mark" },
		{ RT_PROF_CLUSTERS_GRID, "clust grid" },
		{ RT_PROF_CLUSTERS_FILL, "clust fill" },
		{ RT_PROF_CLUSTERS_TAIL, "clust tail" },
		{ RT_PROF_CLUSTERS_UPLOAD, "clust upload" },
		{ RT_PROF_CLUSTERS_PUBLISH, "clust publish" },
		{ RT_PROF_DRAWFRAME, "qrDrawFrame" },
		{ RT_PROF_WAIT, "wait" },
		{ RT_PROF_FRAME, "frame" },
		{ RT_PROF_BRUSH_CHAIN, "brush chain" },
		{ RT_PROF_BRUSH_LIGHTS, "brush lights" },
		{ RT_PROF_BRUSH_PACK, "brush pack" },
		{ RT_PROF_BRUSH_UPLOAD, "brush upload" },
		{ RT_PROF_ALIAS_POSE, "alias pose" },
		{ RT_PROF_ALIAS_LIGHTS, "alias lights" },
		{ RT_PROF_ALIAS_UPLOAD, "alias upload" },
		{ RT_PROF_BRUSH_MATRIX, "brush matrix" },
		{ RT_PROF_BRUSH_STYLES, "brush styles" },
		{ RT_PROF_BRUSH_CLUSTER, "brush cluster" },
		{ RT_PROF_PARTICLES_SIM, "particles sim" },
		{ RT_PROF_PARTICLES_RESOLVE, "particles resolve" },
		{ RT_PROF_PARTICLES_FILL, "particles fill" },
		{ RT_PROF_PARTICLES_UPLOAD, "particles upload" },
		{ RT_PROF_FTE_CONVERT, "fte convert" },
	};

	int i;

	for (i = 0; i < (int)countof (names); i++)
		if (names[i].slot == slot)
			return names[i].name;

	return "?";
}

/*
================
RT_StatsCapture

Reads both sources of the readout in one go, so that the panel and the dump
report the same frame instead of being a pass apart.
================
*/
void RT_StatsCapture (rt_stats_snapshot_t *snap)
{
	memset (snap, 0, sizeof (*snap));
	snap->panels = CVAR_TO_UINT32 (rt_stats_panels);

	if (vulkan_globals.instance != NULL)
		snap->haveGpu = (qrGetFrameStatsEx (vulkan_globals.instance, &snap->gpu) == QR_SUCCESS);

	if (rt_prof_report.valid)
	{
		snap->profile = rt_prof_report;
		snap->haveProfile = true;
	}
}


/*
================
RT_StatsPrintPanels -- state line printed by rt_stats after a change or a query
================
*/
static void RT_StatsPrintPanels (const char *prefix)
{
	char on[8];
	int  i, n = 0;

	for (i = RT_STATS_RAYS; i <= RT_STATS_PROFILE; i++)
		if (RT_StatsPanel (i))
			on[n++] = (char)('0' + i);
	on[n] = 0;

	Con_Printf ("%s showing %s   (1 = ray counters, 2 = + GPU pass timings, 3 = + CPU profile, 0 = off)\n",
	            prefix, n ? on : "nothing");
}

/*
================
RT_Stats_f -- rt_stats 1, 2 or 3

The argument is the level of the readout, and each level stands for the panels
of the ones below it: 1 is the ray counters, 2 adds the GPU pass timings and 3
adds the CPU frame profile, which is all of it; 0 hides the readout. Without an
argument the panels in force are printed, and anything else prints the usage.
================
*/
static void RT_Stats_f (void)
{
	const char *arg;
	int         level;

	if (Cmd_Argc () < 2)
	{
		RT_StatsPrintPanels ("rt_stats is");
		return;
	}

	arg = Cmd_Argv (1);

	if (Cmd_Argc () > 2 || arg[0] == 0 || arg[1] != 0 || arg[0] < '0' || arg[0] > '0' + RT_STATS_PROFILE)
	{
		Con_Printf ("rt_stats: expected one level 0 to %d: 1 = ray counters, 2 = + GPU pass timings, 3 = + CPU profile, 0 = off\n",
		            (int)RT_STATS_PROFILE);
		return;
	}

	level = arg[0] - '0';

	// a level is the panels up to it, so the bits below the level's own are set too
	Cvar_SetValueQuick (&rt_stats_panels, (float)((1u << level) - 1u));
	RT_StatsPrintPanels ("rt_stats is");
}

/*
================
rt_stats_dump

The panel on screen is meant to be read at a glance, which is exactly what makes
it hard to compare runs: the numbers change while you look at them. This writes
one frame of the same readout into qperfdump.log in the game directory, with one
"section name value" line per number so the files can be diffed.

The snapshot is taken on the calling thread, since it is a copy of two small
structs, and everything else -- formatting and the write itself -- happens on a
detached thread. A dump therefore cannot show up in the frame it measures.
================
*/
#define RT_STATS_DUMP_FILE "qperfdump.log"

typedef struct
{
	rt_stats_snapshot_t snap;
	char                path[MAX_OSPATH];
	char                stamp[32];
} rt_stats_dump_job_t;

static SDL_mutex         *rt_stats_dump_mutex;
static rt_stats_dump_job_t rt_stats_dump_job;
static qboolean           rt_stats_dump_busy;

static void RT_StatsDumpWrite (FILE *f, const rt_stats_dump_job_t *job)
{
	const rt_stats_snapshot_t *snap = &job->snap;
	int i;

	fprintf (f, "# rt_stats_dump %s panels %u\n", job->stamp, snap->panels);

	if (snap->haveGpu)
	{
		fprintf (f, "%-11s %-17s %.2f\n", "gpu.frame", "ms", snap->gpu.gpuFrameMs);

		if (snap->gpu.gpuTimingValid)
			for (i = 0; i < QR_GPU_PASS_COUNT; i++)
				fprintf (f, "%-11s %-17s %.2f\n", "gpu.pass", qrGetGpuPassName (i), snap->gpu.gpuPassMs[i]);
		else
			fprintf (f, "%-11s %-17s %s\n", "gpu.pass", "timings", "unavailable, the GPU timer queries did not run");

		fprintf (f, "%-11s %-17s %u\n", "gpu.rays", "total", snap->gpu.raysTotal);
		fprintf (f, "%-11s %-17s %u\n", "gpu.rays", "primary", snap->gpu.raysPerCategory[0]);
		fprintf (f, "%-11s %-17s %u\n", "gpu.rays", "refl refr", snap->gpu.raysPerCategory[1]);
		fprintf (f, "%-11s %-17s %u\n", "gpu.rays", "indirect", snap->gpu.raysPerCategory[2]);
		fprintf (f, "%-11s %-17s %u\n", "gpu.rays", "shadow dir", snap->gpu.raysPerCategory[3]);
		fprintf (f, "%-11s %-17s %u\n", "gpu.rays", "shadow ind", snap->gpu.raysPerCategory[4]);
		fprintf (f, "%-11s %-17s %u\n", "gpu.calls", "rg entry points", snap->gpu.apiCalls);
		fprintf (f, "%-11s %-17s %u\n", "gpu.calls", "geometry", snap->gpu.apiCallsGeometry);
		fprintf (f, "%-11s %-17s %u\n", "gpu.calls", "raster", snap->gpu.apiCallsRasterized);
		fprintf (f, "%-11s %-17s %u\n", "gpu.calls", "lights", snap->gpu.apiCallsLights);
		fprintf (f, "%-11s %-17s %u\n", "gpu.calls", "other",
			snap->gpu.apiCalls - snap->gpu.apiCallsGeometry - snap->gpu.apiCallsRasterized - snap->gpu.apiCallsLights);
	}
	else
	{
		fprintf (f, "%-11s %-17s %s\n", "gpu", "unavailable", "the backend returned no frame stats");
	}

	fputc ('\n', f);

	if (snap->haveProfile)
	{
		fprintf (f, "%-11s %-17s %u\n", "cpu.window", "id", snap->profile.windowId);
		fprintf (f, "%-11s %-17s %u\n", "cpu.window", "frames", snap->profile.frames);
		fprintf (f, "%-11s %-17s %i\n", "cpu.window", "renderer samples", snap->profile.rendererSamples);
		fprintf (f, "%-11s %-17s %.2f\n", "cpu.frame", "ms", snap->profile.frameMs);
		fprintf (f, "%-11s %-17s %.2f\n", "cpu.main", "ms", snap->profile.frameMs - snap->profile.waitMs);
		fprintf (f, "%-11s %-17s %.2f\n", "cpu.wait", "ms", snap->profile.waitMs);
		fprintf (f, "%-11s %-17s %.1f\n", "fps", "-", snap->profile.fps);

		for (i = 0; i < RT_PROF_COUNT; i++)
			fprintf (f, "%-11s %-17s %.2f\n", "cpu.slot", RT_ProfSlotName (i), snap->profile.ms[i]);

		for (i = 0; i < RT_PROF_COUNT; i++)
			fprintf (f, "%-11s %-17s %.2f\n", "cpu.avg", RT_ProfSlotName (i), snap->profile.averageMs[i]);

		if (snap->profile.rendererSamples > 0)
			for (i = 0; i < QR_CPU_PASS_COUNT; i++)
				fprintf (f, "%-11s %-17s avg_ms=%.3f max_ms=%.3f\n", "cpu.draw", qrGetCpuPassName (i),
				         snap->profile.rendererAverageMs[i], snap->profile.rendererMaxMs[i]);

		fprintf (f, "%-11s %-17s %i\n", "cpu.cluster", "cache hits", snap->profile.clusterCacheHits);
		fprintf (f, "%-11s %-17s %i\n", "cpu.cluster", "cache misses", snap->profile.clusterCacheMisses);
		fprintf (f, "%-11s %-17s %i\n", "cpu.cluster", "miss set", snap->profile.clusterMissSet);
		fprintf (f, "%-11s %-17s %i\n", "cpu.cluster", "miss move", snap->profile.clusterMissMove);
		fprintf (f, "%-11s %-17s %i\n", "cpu.cluster", "miss other", snap->profile.clusterMissOther);
		fprintf (f, "%-11s %-17s %i\n", "cpu.cluster", "grants", snap->profile.clusterGrants);
		fprintf (f, "%-11s %-17s %i\n", "cpu.cluster", "denied", snap->profile.clusterDenied);
		fprintf (f, "%-11s %-17s %i\n", "cpu.cluster", "gated", snap->profile.clusterGated);
		fprintf (f, "%-11s %-17s %i\n", "cpu.cluster", "lights", snap->profile.clusterLights);
		fprintf (f, "%-11s %-17s %i\n", "cpu.cluster", "lights add", snap->profile.clusterAttempts);
		fprintf (f, "%-11s %-17s %i\n", "cpu.cluster", "lights drop", snap->profile.clusterDropped);
		fprintf (f, "%-11s %-17s %i\n", "cpu.cluster", "pub skips", snap->profile.clusterPublicationSkips);
		fprintf (f, "%-11s %-17s %i\n", "cpu.cluster", "pub copies", snap->profile.clusterPublicationCopies);
		fprintf (f, "%-11s %-17s %i\n", "cpu.cluster", "pub mask", snap->profile.clusterPublicationMask);
	}
	else
	{
		fprintf (f, "%-11s %-17s %s\n", "cpu", "unavailable", "the profiler is off, rt_stats 3 turns it on");
	}

	fputc ('\n', f);
	fputc ('\n', f);
}

static int RT_StatsDumpThread (void *unused)
{
	rt_stats_dump_job_t job;
	FILE               *f;

	SDL_LockMutex (rt_stats_dump_mutex);
	job = rt_stats_dump_job;
	SDL_UnlockMutex (rt_stats_dump_mutex);

	f = fopen (job.path, "a");

	if (f)
	{
		RT_StatsDumpWrite (f, &job);
		fclose (f);
	}

	// The flag stays set until the file is closed, so a second dump issued while this one is
	// being written is refused instead of appending to the same file from two threads.
	SDL_LockMutex (rt_stats_dump_mutex);
	rt_stats_dump_busy = false;
	SDL_UnlockMutex (rt_stats_dump_mutex);

	return 0;
}

static void RT_StatsDump_f (void)
{
	SDL_Thread *thread;
	time_t      now;
	struct tm  *local;

	if (!rt_stats_dump_mutex)
		rt_stats_dump_mutex = SDL_CreateMutex ();

	if (!rt_stats_dump_mutex)
	{
		Con_Printf ("rt_stats_dump: could not create the writer lock\n");
		return;
	}

	SDL_LockMutex (rt_stats_dump_mutex);

	if (rt_stats_dump_busy)
	{
		SDL_UnlockMutex (rt_stats_dump_mutex);
		Con_Printf ("rt_stats_dump: the previous dump is still being written\n");
		return;
	}

	RT_StatsCapture (&rt_stats_dump_job.snap);
	q_snprintf (rt_stats_dump_job.path, sizeof (rt_stats_dump_job.path), "%s/" RT_STATS_DUMP_FILE, com_gamedir);

	now = time (NULL);
	local = localtime (&now);
	if (local)
		strftime (rt_stats_dump_job.stamp, sizeof (rt_stats_dump_job.stamp), "%Y-%m-%d %H:%M:%S", local);
	else
		rt_stats_dump_job.stamp[0] = 0;

	rt_stats_dump_busy = true;

	SDL_UnlockMutex (rt_stats_dump_mutex);

	thread = SDL_CreateThread (RT_StatsDumpThread, "rt_stats_dump", NULL);

	if (!thread)
	{
		SDL_LockMutex (rt_stats_dump_mutex);
		rt_stats_dump_busy = false;
		SDL_UnlockMutex (rt_stats_dump_mutex);
		Con_Printf ("rt_stats_dump: could not start the writer thread\n");
		return;
	}

	SDL_DetachThread (thread);

	Con_Printf ("rt_stats_dump: appending the frame to %s\n", rt_stats_dump_job.path);

	if (!RT_StatsPanel (RT_STATS_RAYS))
		Con_Printf ("rt_stats_dump: the ray counters are counted when rt_stats 1 or higher is on\n");
}


#define RT_STATS_RECORD_SECONDS     30.0
#define RT_STATS_RECORD_MAX_SAMPLES 700

typedef struct
{
	double              time;
	rt_stats_snapshot_t snap;
} rt_stats_record_sample_t;

typedef struct
{
	char                      path[MAX_OSPATH];
	char                      stamp[32];
	unsigned                  panels;
	double                    interval;
	double                    duration;
	int                       count;
	rt_stats_record_sample_t *samples;
} rt_stats_record_job_t;

static rt_stats_record_sample_t *rt_stats_record_samples;
static int                       rt_stats_record_capacity;
static int                       rt_stats_record_count;
static double                    rt_stats_record_start;
static qboolean                  rt_stats_record_active;
static SDL_mutex                *rt_stats_record_mutex;
static qboolean                  rt_stats_record_writing;
static rt_stats_record_job_t     rt_stats_record_job;

qboolean RT_StatsRecording (void)
{
	return rt_stats_record_active;
}

static void RT_StatsRecordColName (char *out, size_t size, const char *name)
{
	size_t i;

	for (i = 0; name[i] != 0 && i + 1 < size; i++)
		out[i] = (name[i] == ' ' || name[i] == '/') ? '_' : name[i];
	out[i] = 0;
}

static void RT_StatsRecordField (FILE *f, int *first)
{
	if (!*first)
		fputc (',', f);
	*first = 0;
}

static void RT_StatsRecordWrite (FILE *f, const rt_stats_record_job_t *job)
{
	char name[64];
	int  i, j, first;

	fprintf (f, "# rt_stats_dump_start %s panels %u interval %.3f samples %d duration %.2f\n",
	         job->stamp, job->panels, job->interval, job->count, job->duration);

	fputs ("t,fps,gpu.frame_ms", f);
	for (i = 0; i < QR_GPU_PASS_COUNT; i++)
		fprintf (f, ",gpu.%s_ms", qrGetGpuPassName (i));

	fputs (",cpu.frame_ms,cpu.main_ms,cpu.wait_ms", f);
	for (i = 0; i < RT_PROF_COUNT; i++)
	{
		if (i == RT_PROF_FRAME || i == RT_PROF_WAIT)
			continue;

		RT_StatsRecordColName (name, sizeof (name), RT_ProfSlotName (i));
		fprintf (f, ",cpu.%s_ms", name);
	}

	fputs (",cpu.qrDrawFrame_avg_ms", f);
	for (i = 0; i < QR_CPU_PASS_COUNT; i++)
	{
		RT_StatsRecordColName (name, sizeof (name), qrGetCpuPassName (i));
		fprintf (f, ",cpu.draw.%s_avg_ms,cpu.draw.%s_max_ms", name, name);
	}

	fputs (",clust_cache_hits,clust_cache_misses,clust_miss_set,clust_miss_move,clust_miss_other,"
	       "clust_grants,clust_denied,clust_gated,clust_lights,clust_attempts,clust_dropped,"
	       "clust_dirty_max,clust_move_footprint,clust_pub_mask,clust_pub_skips,clust_pub_copies",
	       f);
	fputs (",rays_total,rays_primary,rays_refl_refr,rays_indirect,rays_shadow_dir,rays_shadow_ind,calls,calls_geometry,calls_raster,calls_lights,calls_other,rays_particle,"
	       "particles_classic,particles_fte,particles_vertices,particles_smoke,particles_dropped,"
	       "fte_convert_bytes,particle_upload_bytes,particles_cache_hits,particles_cache_misses,"
	       "particles_cache_avg_ns,raster_upload_bytes,raster_upload_dropped_batches,"
	       "cpu.window_id,cpu.window_frames,cpu.renderer_samples,particles_emit_culled,particles_emit_faded,"
	       "particles_volume_samples,particles_volume_mismatch,particles_volume_lost,particles_volume_extra\n", f);

	for (i = 0; i < job->count; i++)
	{
		const rt_stats_record_sample_t *sample = &job->samples[i];
		const rt_stats_snapshot_t      *snap = &sample->snap;
		const rt_prof_report_t         *rep = &snap->profile;

		first = 1;

		RT_StatsRecordField (f, &first);
		fprintf (f, "%.3f", sample->time);

		RT_StatsRecordField (f, &first);
		if (snap->haveGpu)
			fprintf (f, "%.1f", snap->gpu.fpsX10 / 10.0f);
		else if (snap->haveProfile)
			fprintf (f, "%.1f", rep->fps);

		RT_StatsRecordField (f, &first);
		if (snap->haveGpu && snap->gpu.gpuTimingValid)
			fprintf (f, "%.2f", snap->gpu.gpuFrameMs);

		for (j = 0; j < QR_GPU_PASS_COUNT; j++)
		{
			RT_StatsRecordField (f, &first);
			if (snap->haveGpu && snap->gpu.gpuTimingValid)
				fprintf (f, "%.2f", snap->gpu.gpuPassMs[j]);
		}

		RT_StatsRecordField (f, &first);
		if (snap->haveProfile)
			fprintf (f, "%.2f", rep->frameMs);

		RT_StatsRecordField (f, &first);
		if (snap->haveProfile)
			fprintf (f, "%.2f", rep->frameMs - rep->waitMs);

		RT_StatsRecordField (f, &first);
		if (snap->haveProfile)
			fprintf (f, "%.2f", rep->waitMs);

		for (j = 0; j < RT_PROF_COUNT; j++)
		{
			if (j == RT_PROF_FRAME || j == RT_PROF_WAIT)
				continue;

			RT_StatsRecordField (f, &first);
			if (snap->haveProfile)
				fprintf (f, "%.2f", rep->ms[j]);
		}

		RT_StatsRecordField (f, &first);
		if (snap->haveProfile)
			fprintf (f, "%.3f", rep->averageMs[RT_PROF_DRAWFRAME]);
		for (j = 0; j < QR_CPU_PASS_COUNT; j++)
		{
			RT_StatsRecordField (f, &first);
			if (snap->haveProfile && rep->rendererSamples > 0)
				fprintf (f, "%.3f", rep->rendererAverageMs[j]);
			RT_StatsRecordField (f, &first);
			if (snap->haveProfile && rep->rendererSamples > 0)
				fprintf (f, "%.3f", rep->rendererMaxMs[j]);
		}

		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%i", rep->clusterCacheHits);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%i", rep->clusterCacheMisses);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%i", rep->clusterMissSet);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%i", rep->clusterMissMove);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%i", rep->clusterMissOther);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%i", rep->clusterGrants);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%i", rep->clusterDenied);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%i", rep->clusterGated);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%i", rep->clusterLights);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%i", rep->clusterAttempts);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%i", rep->clusterDropped);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%i", rep->clusterDirty);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%i", rep->clusterMoveFootprint);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%i", rep->clusterPublicationMask);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%i", rep->clusterPublicationSkips);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%i", rep->clusterPublicationCopies);

		RT_StatsRecordField (f, &first);
		if (snap->haveGpu) fprintf (f, "%u", snap->gpu.raysTotal);
		RT_StatsRecordField (f, &first);
		if (snap->haveGpu) fprintf (f, "%u", snap->gpu.raysPerCategory[0]);
		RT_StatsRecordField (f, &first);
		if (snap->haveGpu) fprintf (f, "%u", snap->gpu.raysPerCategory[1]);
		RT_StatsRecordField (f, &first);
		if (snap->haveGpu) fprintf (f, "%u", snap->gpu.raysPerCategory[2]);
		RT_StatsRecordField (f, &first);
		if (snap->haveGpu) fprintf (f, "%u", snap->gpu.raysPerCategory[3]);
		RT_StatsRecordField (f, &first);
		if (snap->haveGpu) fprintf (f, "%u", snap->gpu.raysPerCategory[4]);
		RT_StatsRecordField (f, &first);
		if (snap->haveGpu) fprintf (f, "%u", snap->gpu.apiCalls);
		RT_StatsRecordField (f, &first);
		if (snap->haveGpu) fprintf (f, "%u", snap->gpu.apiCallsGeometry);
		RT_StatsRecordField (f, &first);
		if (snap->haveGpu) fprintf (f, "%u", snap->gpu.apiCallsRasterized);
		RT_StatsRecordField (f, &first);
		if (snap->haveGpu) fprintf (f, "%u", snap->gpu.apiCallsLights);
		RT_StatsRecordField (f, &first);
		if (snap->haveGpu) fprintf (f, "%u",
			snap->gpu.apiCalls - snap->gpu.apiCallsGeometry - snap->gpu.apiCallsRasterized - snap->gpu.apiCallsLights);
		RT_StatsRecordField (f, &first);
		if (snap->haveGpu) fprintf (f, "%u", snap->gpu.raysParticle);

		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%i", rep->particlesClassic);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%i", rep->particlesFte);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%i", rep->particlesVertices);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%i", rep->particlesSmoke);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%i", rep->particlesDropped);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%llu", (unsigned long long)rep->fteConvertBytes);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%llu", (unsigned long long)rep->particleUploadBytes);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%llu", (unsigned long long)rep->particleResolveCacheHits);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%llu", (unsigned long long)rep->particleResolveCacheMisses);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%.0f", rep->particleResolveCacheAvgNs);
		RT_StatsRecordField (f, &first);
		if (snap->haveGpu) fprintf (f, "%llu", (unsigned long long)snap->gpu.rasterUploadBytes);
		RT_StatsRecordField (f, &first);
		if (snap->haveGpu) fprintf (f, "%u", snap->gpu.rasterUploadDroppedBatches);

		for (j = 0; j < 3; j++)
		{
			RT_StatsRecordField (f, &first);
			if (snap->haveProfile)
			{
				const unsigned values[3] = {rep->windowId, rep->frames, (unsigned)rep->rendererSamples};
				fprintf (f, "%u", values[j]);
			}
		}

		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%i", rep->particlesEmitCulled);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%i", rep->particlesEmitFaded);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%llu", (unsigned long long)rep->volumeVerifySamples);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%llu", (unsigned long long)rep->volumeVerifyMismatch);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%llu", (unsigned long long)rep->volumeVerifyLost);
		RT_StatsRecordField (f, &first);
		if (snap->haveProfile) fprintf (f, "%llu", (unsigned long long)rep->volumeVerifyExtra);

		fputc ('\n', f);
	}
}

static int RT_StatsRecordThread (void *unused)
{
	rt_stats_record_job_t job;
	FILE                 *f;

	SDL_LockMutex (rt_stats_record_mutex);
	job = rt_stats_record_job;
	SDL_UnlockMutex (rt_stats_record_mutex);

	f = fopen (job.path, "w");

	if (f)
	{
		RT_StatsRecordWrite (f, &job);
		fclose (f);
	}

	SDL_LockMutex (rt_stats_record_mutex);
	rt_stats_record_writing = false;
	SDL_UnlockMutex (rt_stats_record_mutex);

	return 0;
}

static void RT_StatsRecordFinish (qboolean auto_stop)
{
	SDL_Thread *thread;
	char        filestamp[32];
	time_t      now;
	struct tm  *local;
	double      interval;

	if (!rt_stats_record_active)
		return;

	if (!rt_stats_record_mutex)
		rt_stats_record_mutex = SDL_CreateMutex ();

	if (!rt_stats_record_mutex)
	{
		Con_Printf ("rt_stats_dump_end: could not create the writer lock\n");
		rt_stats_record_active = false;
		return;
	}

	interval = CLAMP (0.05, CVAR_TO_FLOAT (rt_stats_interval), 0.2);
	if (!(interval >= 0.05 && interval <= 0.2))
		interval = 0.2;

	now = time (NULL);
	local = localtime (&now);

	SDL_LockMutex (rt_stats_record_mutex);

	if (local)
	{
		strftime (rt_stats_record_job.stamp, sizeof (rt_stats_record_job.stamp), "%Y-%m-%d %H:%M:%S", local);
		strftime (filestamp, sizeof (filestamp), "%Y%m%d-%H%M%S", local);
	}
	else
	{
		rt_stats_record_job.stamp[0] = 0;
		q_strlcpy (filestamp, "unknown", sizeof (filestamp));
	}

	q_snprintf (rt_stats_record_job.path, sizeof (rt_stats_record_job.path), "%s/stats-%s.dump", com_gamedir,
	            filestamp);
	rt_stats_record_job.panels = CVAR_TO_UINT32 (rt_stats_panels);
	rt_stats_record_job.interval = interval;
	rt_stats_record_job.count = rt_stats_record_count;
	rt_stats_record_job.samples = rt_stats_record_samples;
	rt_stats_record_job.duration = rt_stats_record_count > 0 ? rt_stats_record_samples[rt_stats_record_count - 1].time : 0.0;
	rt_stats_record_writing = true;

	SDL_UnlockMutex (rt_stats_record_mutex);

	rt_stats_record_active = false;

	thread = SDL_CreateThread (RT_StatsRecordThread, "rt_stats_record", NULL);

	if (!thread)
	{
		SDL_LockMutex (rt_stats_record_mutex);
		rt_stats_record_writing = false;
		SDL_UnlockMutex (rt_stats_record_mutex);
		Con_Printf ("rt_stats_dump_end: could not start the writer thread\n");
		return;
	}

	SDL_DetachThread (thread);

	Con_Printf ("rt_stats_dump_end%s: %d samples over %.1f s -> %s\n",
	            auto_stop ? " (the 30 s are up)" : "", rt_stats_record_count, rt_stats_record_job.duration,
	            rt_stats_record_job.path);
}

void RT_StatsRecordSample (const rt_stats_snapshot_t *snap)
{
	rt_stats_record_sample_t *sample;
	double                    now, elapsed;

	if (!rt_stats_record_active || rt_stats_record_samples == NULL)
		return;

	now = Sys_DoubleTime ();
	elapsed = now - rt_stats_record_start;

	if (rt_stats_record_count < rt_stats_record_capacity)
	{
		sample = &rt_stats_record_samples[rt_stats_record_count];
		sample->time = elapsed;
		sample->snap = *snap;
		rt_stats_record_count++;
	}

	if (elapsed >= RT_STATS_RECORD_SECONDS)
		RT_StatsRecordFinish (true);
}

static void RT_StatsDumpStart_f (void)
{
	if (rt_stats_record_active)
	{
		Con_Printf ("rt_stats_dump_start: already recording\n");
		return;
	}

	if (rt_stats_record_mutex)
	{
		qboolean writing;

		SDL_LockMutex (rt_stats_record_mutex);
		writing = rt_stats_record_writing;
		SDL_UnlockMutex (rt_stats_record_mutex);

		if (writing)
		{
			Con_Printf ("rt_stats_dump_start: the previous recording is still being written\n");
			return;
		}
	}

	if (rt_stats_record_samples == NULL)
	{
		rt_stats_record_capacity = RT_STATS_RECORD_MAX_SAMPLES;
		rt_stats_record_samples = Mem_Alloc (sizeof (rt_stats_record_sample_t) * rt_stats_record_capacity);
	}

	rt_stats_record_count = 0;
	rt_stats_record_start = Sys_DoubleTime ();
	rt_stats_record_active = true;

	Con_Printf ("rt_stats_dump_start: recording the readout for up to %.0f s; rt_stats_dump_end writes stats-<datetime>.dump\n",
	            RT_STATS_RECORD_SECONDS);
}

static void RT_StatsDumpEnd_f (void)
{
	if (!rt_stats_record_active)
	{
		Con_Printf ("rt_stats_dump_end: not recording\n");
		return;
	}

	RT_StatsRecordFinish (false);
}


#define RT_LIGHT_REPORT_FILE "qlightreport.log"

// Writes the same emissive stats + cluster report as rt_light_report to qlightreport.log
// in the game directory, so the verdict of every light can be read back from a file.
// Takes the same arguments as rt_light_report (line count / texture filter).
void RT_LightReportDump_f (void)
{
	char       path[MAX_OSPATH];
	char       stamp[32];
	time_t     now;
	struct tm *local;
	FILE      *f;

	q_snprintf (path, sizeof (path), "%s/" RT_LIGHT_REPORT_FILE, com_gamedir);
	f = fopen (path, "a");

	if (!f)
	{
		Con_Printf ("rt_light_report_dump: could not open %s\n", path);
		return;
	}

	now = time (NULL);
	local = localtime (&now);
	if (local)
		strftime (stamp, sizeof (stamp), "%Y-%m-%d %H:%M:%S", local);
	else
		stamp[0] = 0;

	fprintf (f, "# rt_light_report_dump %s\n", stamp);
	RT_ClusterLightDumpHeader (f);

	// Mirror every report line into the file in addition to the console.
	rt_light_report_file = f;
	RT_LightReport_f ();
	rt_light_report_file = NULL;

	fputc ('\n', f);
	fclose (f);

	Con_Printf ("rt_light_report_dump: appended the report to %s\n", path);
}


static qboolean request_shaders_reload = false;

/*
================
VID_Gamma_Init -- call on init
================
*/
static void VID_Gamma_Init (void)
{
	Cvar_RegisterVariable (&vid_gamma);
	Cvar_RegisterVariable (&vid_contrast);
}

/*
======================
VID_GetCurrentWidth
======================
*/
static int VID_GetCurrentWidth (void)
{
	int w = 0, h = 0;
	SDL_GetWindowSize (draw_context, &w, &h);
	return w;
}

/*
=======================
VID_GetCurrentHeight
=======================
*/
static int VID_GetCurrentHeight (void)
{
	int w = 0, h = 0;
	SDL_GetWindowSize (draw_context, &w, &h);
	return h;
}

/*
====================
VID_GetCurrentRefreshRate
====================
*/
static int VID_GetCurrentRefreshRate (void)
{
	SDL_DisplayMode mode;
	int             current_display;

	current_display = SDL_GetWindowDisplayIndex (draw_context);

	if (0 != SDL_GetCurrentDisplayMode (current_display, &mode))
		return DEFAULT_REFRESHRATE;

	return mode.refresh_rate;
}

/*
====================
VID_GetCurrentBPP
====================
*/
static int VID_GetCurrentBPP (void)
{
	const Uint32 pixelFormat = SDL_GetWindowPixelFormat (draw_context);
	return SDL_BITSPERPIXEL (pixelFormat);
}

/*
====================
VID_GetFullscreen

returns true if we are in regular fullscreen or "desktop fullscren"
====================
*/
static qboolean VID_GetFullscreen (void)
{
	return (SDL_GetWindowFlags (draw_context) & SDL_WINDOW_FULLSCREEN) != 0;
}

/*
====================
VID_GetDesktopFullscreen

returns true if we are specifically in "desktop fullscreen" mode
====================
*/
static qboolean VID_GetDesktopFullscreen (void)
{
	return (SDL_GetWindowFlags (draw_context) & SDL_WINDOW_FULLSCREEN_DESKTOP) == SDL_WINDOW_FULLSCREEN_DESKTOP;
}

/*
====================
VID_GetWindow

used by pl_win.c
====================
*/
void *VID_GetWindow (void)
{
	return draw_context;
}

/*
====================
VID_HasMouseOrInputFocus
====================
*/
qboolean VID_HasMouseOrInputFocus (void)
{
	return (SDL_GetWindowFlags (draw_context) & (SDL_WINDOW_MOUSE_FOCUS | SDL_WINDOW_INPUT_FOCUS)) != 0;
}

/*
====================
VID_IsMinimized
====================
*/
qboolean VID_IsMinimized (void)
{
	const Uint32 flags = SDL_GetWindowFlags (draw_context);

	return !(flags & SDL_WINDOW_SHOWN) || (flags & SDL_WINDOW_MINIMIZED);
}

static const char *VID_VsyncModeName (int mode)
{
	switch (mode)
	{
	case VID_VSYNC_ON:        return "vsync";
	case VID_VSYNC_ADAPTIVE:  return "adaptive";
	case VID_VSYNC_FREESYNC:  return "freesync";
	default:                  return "off";
	}
}

static QrPresentMode VID_PresentMode (void)
{
	switch ((int)vid_vsync.value)
	{
	case VID_VSYNC_ON:        return QR_PRESENT_MODE_VSYNC;
	case VID_VSYNC_ADAPTIVE:  return QR_PRESENT_MODE_ADAPTIVE;
	case VID_VSYNC_FREESYNC:  return QR_PRESENT_MODE_VSYNC;
	default:                  return QR_PRESENT_MODE_MAILBOX;
	}
}

static void VID_Vsync_f (cvar_t *var)
{
	Con_Printf ("Video: vsync mode is %s\n", VID_VsyncModeName ((int)var->value));
}

static void VID_CloudsQuality_f (cvar_t *var)
{
	const int quality = (int)CLAMP (0.0f, var->value, (float)QR_SKY_CLOUDS_MAX_QUALITY);
	if (var->value != (float)quality)
	{
		Cvar_SetValueQuick (var, (float)quality);
	}
}

/*
================
VID_SDL2_GetDisplayMode

Returns a pointer to a statically allocated SDL_DisplayMode structure
if there is one with the requested params on the default display.
Otherwise returns NULL.

This is passed to SDL_SetWindowDisplayMode to specify a pixel format
with the requested bpp. If we didn't care about bpp we could just pass NULL.
================
*/
static SDL_DisplayMode *VID_SDL2_GetDisplayMode (int width, int height, int refreshrate)
{
	static SDL_DisplayMode mode;
	const int              sdlmodes = SDL_GetNumDisplayModes (0);
	int                    i;

	for (i = 0; i < sdlmodes; i++)
	{
		if (SDL_GetDisplayMode (0, i, &mode) != 0)
			continue;

		if (mode.w == width && mode.h == height && SDL_BITSPERPIXEL (mode.format) >= 24 && mode.refresh_rate == refreshrate)
		{
			return &mode;
		}
	}
	return NULL;
}

/*
================
VID_ValidMode
================
*/
static qboolean VID_ValidMode (int width, int height, int refreshrate, qboolean fullscreen)
{
	// ignore width / height / bpp if vid_desktopfullscreen is enabled
	if (fullscreen && vid_desktopfullscreen.value)
		return true;

	if (width < 320)
		return false;

	if (height < 200)
		return false;

	if (fullscreen && VID_SDL2_GetDisplayMode (width, height, refreshrate) == NULL)
		return false;

	return true;
}

/*
================
VID_SetMode
================
*/
static qboolean VID_SetMode (int width, int height, int refreshrate, qboolean fullscreen)
{
	int    temp;
	Uint32 flags;
	char   caption[50];
	int    previous_display;

	// so Con_Printfs don't mess us up by forcing vid and snd updates
	temp = scr_disabled_for_loading;
	scr_disabled_for_loading = true;

	CDAudio_Pause ();
	BGM_Pause ();

	q_snprintf (caption, sizeof (caption), "QuakeRay " ENGINE_VER_STRING);

	/* Create the window if needed, hidden */
	if (!draw_context)
	{
		flags = SDL_WINDOW_HIDDEN | SDL_WINDOW_VULKAN;

		if (vid_borderless.value)
			flags |= SDL_WINDOW_BORDERLESS;
		else if (!fullscreen)
			flags |= SDL_WINDOW_RESIZABLE;

		draw_context = SDL_CreateWindow (caption, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, flags);
		if (!draw_context)
			Sys_Error ("Couldn't create window: %s", SDL_GetError ());

		SDL_VERSION (&sys_wm_info.version);
		if (!SDL_GetWindowWMInfo (draw_context, &sys_wm_info))
			Sys_Error ("Couldn't get window wm info: %s", SDL_GetError ());

		previous_display = -1;
	}
	else
	{
		previous_display = SDL_GetWindowDisplayIndex (draw_context);
	}

	/* Ensure the window is not fullscreen */
	if (VID_GetFullscreen ())
	{
		if (SDL_SetWindowFullscreen (draw_context, 0) != 0)
			Sys_Error ("Couldn't set fullscreen state mode: %s", SDL_GetError ());
	}

	/* Set window size and display mode */
	SDL_SetWindowSize (draw_context, width, height);
	if (previous_display >= 0)
		SDL_SetWindowPosition (draw_context, SDL_WINDOWPOS_CENTERED_DISPLAY (previous_display), SDL_WINDOWPOS_CENTERED_DISPLAY (previous_display));
	else
		SDL_SetWindowPosition (draw_context, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
	SDL_SetWindowDisplayMode (draw_context, VID_SDL2_GetDisplayMode (width, height, refreshrate));
	SDL_SetWindowBordered (draw_context, vid_borderless.value ? SDL_FALSE : SDL_TRUE);

	/* Make window fullscreen if needed, and show the window */

	if (fullscreen)
	{
		const Uint32 flag = vid_desktopfullscreen.value ? SDL_WINDOW_FULLSCREEN_DESKTOP : SDL_WINDOW_FULLSCREEN;
		if (SDL_SetWindowFullscreen (draw_context, flag) != 0)
			Sys_Error ("Couldn't set fullscreen state mode: %s", SDL_GetError ());
	}

	SDL_ShowWindow (draw_context);
	SDL_RaiseWindow (draw_context);

	vid.width = VID_GetCurrentWidth ();
	vid.height = VID_GetCurrentHeight ();
	vid.conwidth = vid.width & 0xFFFFFFF8;
	vid.conheight = vid.conwidth * vid.height / vid.width;

	modestate = VID_GetFullscreen () ? MS_FULLSCREEN : MS_WINDOWED;

	CDAudio_Resume ();
	BGM_Resume ();
	scr_disabled_for_loading = temp;

	// fix the leftover Alt from any Alt-Tab or the like that switched us away
	ClearAllStates ();

	vid.recalc_refdef = 1;

	// no pending changes
	vid_changed = false;

	SCR_UpdateRelativeScale ();

	return true;
}

/*
===================
VID_Changed_f -- kristian -- notify us that a value has changed that requires a vid_restart
===================
*/
static void VID_Changed_f (cvar_t *var)
{
	vid_changed = true;
}

/*
================
VID_Test -- johnfitz -- like vid_restart, but asks for confirmation after switching modes
================
*/
static void VID_Test (void)
{
	int old_width, old_height, old_refreshrate, old_fullscreen;

	if (vid_locked || !vid_changed)
		return;
	//
	// now try the switch
	//
	old_width = VID_GetCurrentWidth ();
	old_height = VID_GetCurrentHeight ();
	old_refreshrate = VID_GetCurrentRefreshRate ();
	old_fullscreen = VID_GetFullscreen () ? 1 : 0;
	VID_Restart (true);

	// pop up confirmation dialoge
	if (!SCR_ModalMessage ("Would you like to keep this\nvideo mode? (y/n)\n", 5.0f))
	{
		// revert cvars and mode
		Cvar_SetValueQuick (&vid_width, old_width);
		Cvar_SetValueQuick (&vid_height, old_height);
		Cvar_SetValueQuick (&vid_refreshrate, old_refreshrate);
		Cvar_SetValueQuick (&vid_fullscreen, old_fullscreen);
		VID_Restart (true);
	}
}

/*
================
VID_Unlock -- johnfitz
================
*/
static void VID_Unlock (void)
{
	vid_locked = false;
	VID_SyncCvars ();
}

/*
================
VID_Lock -- ericw

Subsequent changes to vid_* mode settings, and vid_restart commands, will
be ignored until the "vid_unlock" command is run.

Used when changing gamedirs so the current settings override what was saved
in the config.cfg.
================
*/
void VID_Lock (void)
{
	vid_locked = true;
}

//==============================================================================
//
//	Vulkan Stuff
//
//==============================================================================

static void RT_PrintMessage (const char *pMessage, void *pUserData)
{
	Con_Warning (pMessage);
}

static void RT_LoadFile (const char *pFilePath, void *pUserData, const void **ppOutData,
                         uint32_t *pOutDataSize, void **ppOutFileUserHandle)
{
	char        name[MAX_OSPATH];
	const char *p = pFilePath;
	byte       *data;
	int         i;

	(void)pUserData;

	if (ppOutData)
		*ppOutData = NULL;
	if (pOutDataSize)
		*pOutDataSize = 0;
	if (ppOutFileUserHandle)
		*ppOutFileUserHandle = NULL;

	if (!p || !p[0])
		return;

	if (!q_strncasecmp (p, RT_OVERRIDEN_FOLDER, sizeof (RT_OVERRIDEN_FOLDER) - 1))
		p += sizeof (RT_OVERRIDEN_FOLDER) - 1;
	while (*p == '/' || *p == '\\')
		p++;

	for (i = 0; p[i] && i < (int)sizeof (name) - 1; i++)
		name[i] = (p[i] == '\\') ? '/' : p[i];
	name[i] = 0;

	data = COM_LoadFile (name, NULL);
	if (!data)
		return;

	if (ppOutData)
		*ppOutData = data;
	if (pOutDataSize)
		*pOutDataSize = (uint32_t)com_filesize;
	if (ppOutFileUserHandle)
		*ppOutFileUserHandle = data;
}

static void RT_FreeFile (void *pFileUserHandle, void *pUserData)
{
	(void)pUserData;

	if (pFileUserHandle)
		Mem_Free (pFileUserHandle);
}

static void RT_ReloadShaders (void)
{
	request_shaders_reload = true;
}

// A color setting is a console command plus an archived cvar of the same name,
// so one setting describes a color completely. It cannot be a plain cvar:
// Cvar_Command takes a single token, so "rt_sky_color 32 0 64" would never reach
// three channels. The command writes the cvar, so the value is archived with the
// rest of the config.
typedef struct
{
	cvar_t  *cvar;     // its name is also the name of the command
	vec3_t   fallback; // used while the cvar holds something unparsable
	vec3_t   value;
	qboolean dirty;    // `value` is stale until RT_ColorGet re-parses the cvar
} rt_color_t;

typedef enum
{
	RT_COLOR_SKY,
	RT_COLOR_SUN,
	RT_COLOR_CLOUDS,
	RT_COLOR_LIGHT,
	RT_COLOR_GLOBALLIGHT,
	RT_COLOR_WATER,
	RT_COLOR_ACID,

	RT_COLOR_COUNT
} rt_color_index_t;

static rt_color_t rt_colors[RT_COLOR_COUNT] = {
	[RT_COLOR_SKY]         = {.cvar = &rt_sky_color,        .fallback = {32 / 255.0f, 0.0f, 64 / 255.0f}, .dirty = true},
	[RT_COLOR_SUN]         = {.cvar = &rt_sky_sun_color,        .fallback = {1.0f, 1.0f, 1.0f},                .dirty = true},
	[RT_COLOR_CLOUDS]      = {.cvar = &rt_sky_clouds_color, .fallback = {0.0f, 0.0f, 0.0f},                .dirty = true},
	[RT_COLOR_LIGHT]       = {.cvar = &rt_light_color,      .fallback = {1.0f, 1.0f, 1.0f},                .dirty = true},
	[RT_COLOR_GLOBALLIGHT] = {.cvar = &rt_globallight,      .fallback = {1.0f, 1.0f, 1.0f},                .dirty = true},
	[RT_COLOR_WATER]       = {.cvar = &rt_water_color,      .fallback = {171 / 255.0f, 193 / 255.0f, 210 / 255.0f}, .dirty = true},
	[RT_COLOR_ACID]        = {.cvar = &rt_water_acidcolor,  .fallback = {122 / 255.0f, 143 / 255.0f, 21 / 255.0f}, .dirty = true},
};

static qboolean RT_ColorParse (const char *s, float *out)
{
	char buf[64];
	int  i;

	for (i = 0; s[i] && i < (int)sizeof (buf) - 1; i++)
	{
		buf[i] = (s[i] == ',') ? ' ' : s[i];
	}
	buf[i] = '\0';

	float r, g, b;

	if (3 != sscanf (buf, "%f %f %f", &r, &g, &b))
	{
		return false;
	}

	out[0] = CLAMP (0, r, 255) / 255.0f;
	out[1] = CLAMP (0, g, 255) / 255.0f;
	out[2] = CLAMP (0, b, 255) / 255.0f;
	return true;
}

static rt_color_t *RT_ColorFind (const char *name)
{
	for (size_t i = 0; i < countof (rt_colors); i++)
	{
		if (!q_strcasecmp (name, rt_colors[i].cvar->name))
		{
			return &rt_colors[i];
		}
	}

	return NULL;
}

// Fired by Cvar_SetQuick on every real change, so the getters below need no test of
// their own.
static void RT_ColorChanged_f (cvar_t *var)
{
	for (size_t i = 0; i < countof (rt_colors); i++)
	{
		if (rt_colors[i].cvar == var)
		{
			rt_colors[i].dirty = true;
			return;
		}
	}
}

// The getters run once per light per frame, which rules out comparing the string on
// every call -- and a pointer comparison would not be enough either, because
// Cvar_SetQuick reallocates only when the length changes (cvar.c:399) and memcpys
// into the existing buffer otherwise (cvar.c:404), which leaves the pointer alone. So
// the cvar's callback reports the change instead: the parse, and with it the
// complaint about a value that does not parse, happens once per change rather than on
// every call.
static void RT_ColorGet (rt_color_t *c, float *out)
{
	if (c->dirty)
	{
		if (Tasks_IsWorker ())
		{
			VectorCopy (c->value, out);
			return;
		}

		c->dirty = false;

		if (!RT_ColorParse (c->cvar->string, c->value))
		{
			Con_Printf ("%s: expected <r 0..255> <g 0..255> <b 0..255>, got \"%s\"\n", c->cvar->name, c->cvar->string);
			VectorCopy (c->fallback, c->value);
		}
	}

	VectorCopy (c->value, out);
}

void RT_ColorsRefresh (void)
{
	float color[3];

	for (size_t i = 0; i < countof (rt_colors); i++)
		RT_ColorGet (&rt_colors[i], color);
}

void RT_GetSkyColor (float color[3])
{
	RT_ColorGet (&rt_colors[RT_COLOR_SKY], color);
}

void RT_GetSunColor (float color[3])
{
	RT_ColorGet (&rt_colors[RT_COLOR_SUN], color);
}

void RT_GetWaterColor (float color[3])
{
	RT_ColorGet (&rt_colors[RT_COLOR_WATER], color);
}

void RT_GetAcidColor (float color[3])
{
	RT_ColorGet (&rt_colors[RT_COLOR_ACID], color);
}

#define RAD2DEG(a) ((a) / M_PI_DIV_180)

// The crosshair is the view vector of the frame, bob and weapon kick included,
// and the sun goes exactly where it points. The light, the god rays and the sky
// read rt_sky_sun_pitch/rt_sky_sun_yaw through AngleVectors and take -forward as the
// direction toward the sun, so the sun's sine of pitch is the view vector's own
// z -- which is already -sin of the view pitch -- and its yaw is the view yaw
// turned around.
void RT_UpdateSunEditor (void)
{
	float pitch, yaw;

	if (!CVAR_TO_BOOL (rt_sky_sun_edit) || !CVAR_TO_BOOL (rt_physical_sun))
	{
		return;
	}

	pitch = RAD2DEG (asin (CLAMP (-1.0f, vpn[2], 1.0f)));
	yaw = anglemod (RAD2DEG (atan2 (vpn[1], vpn[0])) + 180.0f);

	Cvar_SetValueQuick (&rt_sky_sun_pitch, pitch);
	Cvar_SetValueQuick (&rt_sky_sun_yaw, yaw);
}

void RT_GetSkyCloudsColor (float color[3])
{
	RT_ColorGet (&rt_colors[RT_COLOR_CLOUDS], color);
}

void RT_GetLightColor (float color[3])
{
	RT_ColorGet (&rt_colors[RT_COLOR_LIGHT], color);
}

void RT_GetGlobalLightColor (float color[3])
{
	RT_ColorGet (&rt_colors[RT_COLOR_GLOBALLIGHT], color);
}

static void RT_ColorPrint (rt_color_t *c)
{
	float color[3];

	RT_ColorGet (c, color);
	Con_Printf ("current: %d %d %d\n", (int)(color[0] * 255 + 0.5f), (int)(color[1] * 255 + 0.5f), (int)(color[2] * 255 + 0.5f));
	Con_Printf ("usage: <r 0..255> <g 0..255> <b 0..255>\n");
	Con_Printf ("       (\"r g b\" and r,g,b are accepted too)\n");
}

static void RT_ColorSet (rt_color_t *c, const float color[3])
{
	Cvar_Set (c->cvar->name, va ("%d %d %d",
		(int)(CLAMP (0, color[0], 1.0f) * 255 + 0.5f),
		(int)(CLAMP (0, color[1], 1.0f) * 255 + 0.5f),
		(int)(CLAMP (0, color[2], 1.0f) * 255 + 0.5f)));

	c->dirty = true;
}

static void RT_Color (void)
{
	rt_color_t *c = RT_ColorFind (Cmd_Argv (0));
	float       color[3];

	if (!c)
	{
		return;
	}

	if (Cmd_Argc () == 4)
	{
		RT_VEC3_SET (
			color,
			CLAMP (0, strtof (Cmd_Argv (1), NULL), 255) / 255.0f,
			CLAMP (0, strtof (Cmd_Argv (2), NULL), 255) / 255.0f,
			CLAMP (0, strtof (Cmd_Argv (3), NULL), 255) / 255.0f );
	}
	else if (Cmd_Argc () == 2)
	{
		if (!RT_ColorParse (Cmd_Argv (1), color))
		{
			Con_Printf ("invalid color '%s'\n", Cmd_Argv (1));
			RT_ColorPrint (c);
			return;
		}
	}
	else
	{
		RT_ColorPrint (c);
		return;
	}

	RT_ColorSet (c, color);
	RT_ColorPrint (c);
}

// Color settings are commands backed by archivable cvars: the command parses the
// "r g b" form, the cvar stores the value and writes it to config.cfg. The two
// registrations get in each other's way -- Cmd_AddCommand2 rejects a name that is
// already a registered var, and Cvar_RegisterVariable rejects a name that is already
// a src_command command (Cmd_Exists ignores non-console commands) -- so the commands
// are added as client commands, from VID_Init, before the RT cvars are registered.
static void RT_ColorInit (void)
{
	for (size_t i = 0; i < countof (rt_colors); i++)
	{
		Cmd_AddCommand2 (rt_colors[i].cvar->name, RT_Color, src_client);
	}
}

/*
===============
GL_InitInstance
===============
*/
static void GL_InitInstance (void)
{
	SDL_SysWMinfo wmInfo;
	SDL_VERSION (&wmInfo.version);
	SDL_GetWindowWMInfo (draw_context, &wmInfo);

	QrWin32SurfaceCreateInfo win32Info = {.hinstance = wmInfo.info.win.hinstance, .hwnd = wmInfo.info.win.window};

	const char pShaderPath[] = RT_OVERRIDEN_FOLDER "shaders/";
	const char pBlueNoisePath[] = RT_OVERRIDEN_FOLDER "BlueNoise_LDR_RGBA_128.png";
	const char pWaterTexturePath[] = RT_OVERRIDEN_FOLDER "WaterNormal_n.png";

	QrInstanceCreateInfo info = {
		.pAppName = "QuakeRay",
		.pAppGUID = "8d1f551a-b0e4-4365-985c-5e1182f3c54a",

		.pWin32SurfaceInfo = &win32Info,
		
		.pfnPrint = RT_PrintMessage,

		.pShaderFolderPath = pShaderPath,
		.pBlueNoiseFilePath = pBlueNoisePath,

		.pfnOpenFile = RT_LoadFile,
		.pfnCloseFile = RT_FreeFile,

		.primaryRaysMaxAlbedoLayers = 2,
		.indirectIlluminationMaxAlbedoLayers = 1,
		.rayCullBackFacingTriangles = 1,
		.allowGeometryWithSkyFlag = 1,

		.rasterizedMaxVertexCount = 1 << 18,
		.rasterizedMaxIndexCount = 1 << 19,

		.rasterizedVertexColorGamma = true,

		.rasterizedSkyCubemapSize = 256,

		.maxTextureCount = 4096,
		.textureSamplerForceMinificationFilterLinear = true,
		.textureSamplerForceNormalMapFilterLinear = true,

		.pOverridenTexturesFolderPath = RT_OVERRIDEN_FOLDER "mat",
		.pOverridenTexturesFolderPathDeveloper = RT_OVERRIDEN_FOLDER "matdev",

		.originalAlbedoAlphaTextureIsSRGB = true,
	    .originalRoughnessMetallicEmissionTextureIsSRGB = false,
	    .originalNormalTextureIsSRGB = false,

		.overridenAlbedoAlphaTextureIsSRGB = true,
		.overridenRoughnessMetallicEmissionTextureIsSRGB = false,
		.overridenNormalTextureIsSRGB = false,

		.pWaterNormalTexturePath = pWaterTexturePath,

		// The shadow map the sun shafts are traced through is sized for this level
		// (ShadowMap::SetQuality); every frame carries the level again, so the cvar
		// may be changed at any time.
		.godRaysQuality = CVAR_TO_UINT32 (rt_sky_godrays_quality),
	};

	QrResult r = qrCreateInstance (&info, &vulkan_globals.instance);
	QR_CHECK (r);

	RT_MAT_Init ();
	RT_LIGHT_Init ();

	QR_Editor_Init (); // qr light editor console commands
	PhotoCam_Init ();

	QR_GUI_Init (VID_GetWindow (), (void *)(intptr_t) vulkan_globals.instance, NULL, 0);

	Cmd_AddCommand ("rt_pfnreloadshaders", RT_ReloadShaders);
	Cmd_AddCommand ("rt_light_report", RT_LightReport_f);
	Cmd_AddCommand ("rt_light_report_dump", RT_LightReportDump_f);
	Cmd_AddCommand ("rt_cluster_lists", RT_ClusterLists_f);
	Cmd_AddCommand ("rt_dtal_rebuild", RT_DtalRebuild_f);
	Cmd_AddCommand ("dlightspot", RT_DlightSpot_f);
	Cmd_AddCommand ("fog", RT_Fog_Cmd);
	Cmd_AddCommand ("rt_stats", RT_Stats_f);
	Cmd_AddCommand ("rt_stats_dump", RT_StatsDump_f);
	Cmd_AddCommand ("rt_stats_dump_start", RT_StatsDumpStart_f);
	Cmd_AddCommand ("rt_stats_dump_end", RT_StatsDumpEnd_f);


    vulkan_globals.primary_cb_context.batch_indices = Mem_Alloc (sizeof (uint32_t) * MAX_BATCH_INDICES);
	vulkan_globals.primary_cb_context.batch_verts = Mem_Alloc (sizeof (QrVertex) * MAX_BATCH_VERTS);
	vulkan_globals.primary_cb_context.batch_verts_count = 0;
	vulkan_globals.primary_cb_context.batch_indices_count = 0;
	for (int i = 0; i < CBX_NUM; i++)
	{
		vulkan_globals.secondary_cb_contexts[i].batch_indices = Mem_Alloc (sizeof (uint32_t) * MAX_BATCH_INDICES);
		vulkan_globals.secondary_cb_contexts[i].batch_verts = Mem_Alloc (sizeof (QrVertex) * MAX_BATCH_VERTS);
		vulkan_globals.secondary_cb_contexts[i].batch_verts_count = 0;
		vulkan_globals.secondary_cb_contexts[i].batch_indices_count = 0;
	}
}

/*
=================
GL_BeginRenderingTask
=================
*/
void GL_BeginRenderingTask (void *unused)
{
	QrStartFrameInfo info = {
		.presentMode = VID_PresentMode (),
		.maxFrameLatency = (uint32_t)CLAMP (0, (int)vid_maxframelatency.value, 1),
		.requestShaderReload = request_shaders_reload,
	};

	QrResult r = qrStartFrame (vulkan_globals.instance, &info);
	QR_CHECK (r);

	request_shaders_reload = false;

	{
		cb_context_t *cbx = &vulkan_globals.primary_cb_context;
		cbx->current_canvas = CANVAS_INVALID;
	}

	for (int cbx_index = 0; cbx_index < CBX_NUM; ++cbx_index)
	{
		cb_context_t *cbx = &vulkan_globals.secondary_cb_contexts[cbx_index];
		cbx->current_canvas = CANVAS_INVALID;

		GL_SetCanvas (cbx, CANVAS_NONE);
	}
}

/*
=================
GL_SynchronizeEndRenderingTask
=================
*/
void GL_SynchronizeEndRenderingTask (void)
{
	if (prev_end_rendering_task != INVALID_TASK_HANDLE)
	{
		Task_Join (prev_end_rendering_task, SDL_MUTEX_MAXWAIT);
		prev_end_rendering_task = INVALID_TASK_HANDLE;
	}
}

/*
=================
GL_BeginRendering
=================
*/
qboolean GL_BeginRendering (qboolean use_tasks, task_handle_t *begin_rendering_task, int *x, int *y, int *width, int *height)
{
	if (!use_tasks)
		GL_SynchronizeEndRenderingTask ();

	if (vid.restart_next_frame)
	{
		VID_Restart (false);
		vid.restart_next_frame = false;
	}

	*x = *y = 0;
	*width = vid.width;
	*height = vid.height;

	if (use_tasks)
		*begin_rendering_task = Task_AllocateAndAssignFunc (GL_BeginRenderingTask, NULL, 0);
	else
		GL_BeginRenderingTask (NULL);

	return true;
}

static QrRenderSharpenTechnique GetSharpenTechniqueFromCvar ()
{
	int t = CVAR_TO_INT32 (rt_sharpen);

	switch (t)
	{
	case 2:
		return QR_RENDER_SHARPEN_TECHNIQUE_AMD_CAS;
	case 1:
		return QR_RENDER_SHARPEN_TECHNIQUE_NAIVE;
	default:
		return QR_RENDER_SHARPEN_TECHNIQUE_NONE;
	}
}

static void UpscaleCvarsToQray (QrDrawFrameRenderResolutionParams *pDst)
{
	int nvDlss = CVAR_TO_INT32 (rt_upscale_dlss);
	int amdFsr31 = CVAR_TO_INT32 (rt_upscale_fsr31);

	switch (nvDlss)
	{
	case 1:
		// start with Quality
		pDst->upscaleTechnique = QR_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS;
		pDst->resolutionMode = QR_RENDER_RESOLUTION_MODE_QUALITY;
		break;
	case 2:
		pDst->upscaleTechnique = QR_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS;
		pDst->resolutionMode = QR_RENDER_RESOLUTION_MODE_BALANCED;
		break;
	case 3:
		pDst->upscaleTechnique = QR_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS;
		pDst->resolutionMode = QR_RENDER_RESOLUTION_MODE_PERFORMANCE;
		break;
	case 4:
		pDst->upscaleTechnique = QR_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS;
		pDst->resolutionMode = QR_RENDER_RESOLUTION_MODE_ULTRA_PERFORMANCE;
		break;

	case 5:
		// use DLSS with rt_renderscale
		pDst->upscaleTechnique = QR_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS;
		pDst->resolutionMode = QR_RENDER_RESOLUTION_MODE_CUSTOM;
		break;

	default:
		nvDlss = 0;
		break;
	}

	switch (amdFsr31)
	{
	case 1:
		pDst->upscaleTechnique = QR_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3;
		pDst->resolutionMode = QR_RENDER_RESOLUTION_MODE_NATIVE_AA;
		break;
	case 2:
		pDst->upscaleTechnique = QR_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3;
		pDst->resolutionMode = QR_RENDER_RESOLUTION_MODE_QUALITY;
		break;
	case 3:
		pDst->upscaleTechnique = QR_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3;
		pDst->resolutionMode = QR_RENDER_RESOLUTION_MODE_BALANCED;
		break;
	case 4:
		pDst->upscaleTechnique = QR_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3;
		pDst->resolutionMode = QR_RENDER_RESOLUTION_MODE_PERFORMANCE;
		break;
	case 5:
		pDst->upscaleTechnique = QR_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3;
		pDst->resolutionMode = QR_RENDER_RESOLUTION_MODE_ULTRA_PERFORMANCE;
		break;

	case 6:
		pDst->upscaleTechnique = QR_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3;
		pDst->resolutionMode = QR_RENDER_RESOLUTION_MODE_CUSTOM;
		break;

	default:
		amdFsr31 = 0;
		break;
	}

	// both disabled
	if (nvDlss == 0 && amdFsr31 == 0)
	{
		pDst->upscaleTechnique = QR_RENDER_UPSCALE_TECHNIQUE_NEAREST;
		pDst->resolutionMode = QR_RENDER_RESOLUTION_MODE_CUSTOM;
	}

	pDst->sharpenTechnique = GetSharpenTechniqueFromCvar ();
}

static const char *GetUpscalerOptionName (int i, QrRenderUpscaleTechnique technique)
{
	if (!qrIsRenderUpscaleTechniqueAvailable (vulkan_globals.instance, technique))
	{
		return "Not Available";
	}

    switch (i)
    {
	case 0:
		return "Off";
    case 1:
		return (technique == QR_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3) ? "Native AA" : "Quality";
	case 2:
		return (technique == QR_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3) ? "Quality" : "Balanced";
	case 3:
		return (technique == QR_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3) ? "Balanced" : "Performance";
	case 4:
		return (technique == QR_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3) ? "Performance" : "Ultra Performance";
	case 5:
		return "Ultra Performance";
	default:
		return "Custom";
    }
}

typedef struct end_rendering_parms_s
{
	uint32_t serial;
	uint32_t delay_ms;
	float    vid_width;
	float    vid_height;
} end_rendering_parms_t;

#define DEG2RAD(a) ((a)*M_PI_DIV_180)
#define FROMCOLOR255(a) {((a)[0] / 255.0f), ((a)[1] / 255.0f), ((a)[2] / 255.0f)}

extern float  GL_GetCameraNear (float radfovx, float radfovy);
extern float  GL_GetCameraFar (void);
extern float  r_fovx, r_fovy;
extern cvar_t r_fastsky;
extern float  skyflatcolor[3];
extern float  skyfog;
extern float    rt_dmg_value;
extern qboolean rt_dmg_inthisframe;
extern QrMediaType rt_cameramedia;
extern qboolean rt_lavaeffects;
extern float rt_ef_damage_pulse;
extern float rt_ef_liquid_pulse;
extern float rt_ef_pickup_pulse;

static void ResolutionToQray (QrDrawFrameRenderResolutionParams *dst, const QrExtent2D winsize)
{
	if (CVAR_TO_INT32 (rt_renderscale) > 0)
	{
		float scale = (float)CVAR_TO_INT32 (rt_renderscale) / 100.0f;
		scale = CLAMP (scale, 0.2f, 1.0f);

		dst->customRenderSize.width = (uint32_t)(scale * winsize.width);
		dst->customRenderSize.height = (uint32_t)(scale * winsize.height);
		dst->pPixelizedRenderSize = NULL;

		return;
	}

	dst->customRenderSize = winsize;
	dst->pPixelizedRenderSize = NULL;
}

/*
=================
GL_EndRenderingTask
=================
*/
static void GL_EndRenderingTask (end_rendering_parms_t *parms)
{
	Atomic_StoreUInt32 (&rt_end_task_running, 1);

	if (parms->delay_ms > 0)
		SDL_Delay (parms->delay_ms);

	const QrExtent2D winsize = {.width = parms->vid_width, .height = parms->vid_height};
	const qboolean editor_active = QR_Editor_Active ();

	QrDrawFrameRenderResolutionParams resolution_params = {0};
	ResolutionToQray (&resolution_params, winsize);
	UpscaleCvarsToQray (&resolution_params);

	const float q2_lightstats_value = CVAR_TO_FLOAT (rt_q2_lightstats);
	const uint32_t q2_lightstats_mode = (q2_lightstats_value < 0.0f || q2_lightstats_value > 4.0f) ? 1u : (uint32_t)q2_lightstats_value;

	// Single- vs double-sample NEE in the direct pass. The estimator divides by
	// the count, so both values are unbiased; the range cannot go above 2 because
	// a third sample's RNG salt would collide with the sun disk stream.
	const uint32_t nee_samples = (CVAR_TO_FLOAT (rt_nee_samples) < 1.5f) ? 1u : 2u;

	// Q2RTX-style global illumination level (pt_num_bounce_rays): 0 disables the
	// indirect pass, 1 mirrors the historical single-bounce behavior and 2 adds a
	// second indirect bounce.
	const float gi_level = CLAMP (0.0f, CVAR_TO_FLOAT (rt_gi_level), 2.0f);

	QrDrawFrameIlluminationParams illum_params = {
	    .maxBounceShadows = CVAR_TO_UINT32 (rt_shadowrays),
		// The level alone decides the bounce count, like Q2RTX pt_num_bounce_rays.
		// rt_indir2bounces is kept for config compatibility but no longer forces the
		// second bounce on at every level, which made medium and high identical.
		.enableSecondBounceForIndirect = (gi_level >= 1.5f),
		.cellWorldSize = METRIC_TO_QUAKEUNIT(2.0f),
		.directDiffuseSensitivityToChange = CVAR_TO_FLOAT (rt_sensit_dir),
		.indirectDiffuseSensitivityToChange = CVAR_TO_FLOAT (rt_sensit_indir),
		.specularSensitivityToChange = CVAR_TO_FLOAT (rt_sensit_spec),
		.polygonalLightSpotlightFactor = 2.0f,
		.q2DepthGradMode = CVAR_TO_UINT32 (rt_q2_depthgrad) != 0,
		.q2LightStatsMode = q2_lightstats_mode,
		.reflRefrEarlyOut = CVAR_TO_BOOL (rt_reflrefr_earlyout),
		.neeLightSamples = nee_samples,
		.restirEnabled = CVAR_TO_BOOL (rt_restir) ? 1u : 0u,
		.restirCandidates = (uint32_t)CLAMP (1.0f, CVAR_TO_FLOAT (rt_restir_candidates), 64.0f),
		.giBounceRays = gi_level,
		// Q2RTX pt_sun_bounce_range / sun_bounce: how far the sun reaches into an
		// indirect bounce (game units, 0 turns indirect sunlight off) and a
		// multiplier on what it delivers there. Both only affect the indirect
		// pass, and the shadow ray a bounce casts for the sun is skipped once the
		// distance falloff is zero.
		.sunBounceRange = CVAR_TO_FLOAT (rt_sky_sun_bounce_range),
		.sunBounceScale = CVAR_TO_FLOAT (rt_sky_sun_bounce_scale),
		.denoiserEnabled = CVAR_TO_BOOL (rt_denoiser),
		// Q2RTX writes 2 through its "textures" toggle (flt_fixed_albedo ~1),
		// so the unchecked state uses the same flat albedo value.
		.fixedAlbedo = CVAR_TO_BOOL (rt_no_textures) ? 2.0f : 0.0f,
		.lightUniqueIdIgnoreFirstPersonViewerShadows = NULL,
	};

	QrDrawFrameBloomParams bloom_params = {
		.bloomIntensity = 0.0f,
		.inputThreshold = 0.0f,
		.bloomEmissionMultiplier = CVAR_TO_FLOAT (rt_bloom_emis_mult),
	};

	// Exposure bias is in EV and darkens the image when negative; contrast blends
	// the adaptive tone curve with Reinhard (Q2RTX tm_exposure_bias / tm_reinhard).
	QrDrawFrameTonemappingParams tonemap_params = {
		.minLogLuminance = -3.9f,
		.maxLogLuminance = -2.8f,
		.luminanceWhitePoint = 10.0f,
		.exposureBias = CLAMP (-3.0f, CVAR_TO_FLOAT (rt_exposure_bias), 3.0f),
		.tonemapPower = CLAMP (0.0f, CVAR_TO_FLOAT (rt_tonemap_power), 1.0f),
		.exposureSpeedUp = CVAR_TO_FLOAT (rt_exposure_speed_up),
		.exposureSpeedDown = CVAR_TO_FLOAT (rt_exposure_speed_down),
		.exposureLowPercentile = CVAR_TO_FLOAT (rt_exposure_low_percentile),
		.exposureHighPercentile = CVAR_TO_FLOAT (rt_exposure_high_percentile),
		.minAdaptedLuminance = CVAR_TO_FLOAT (rt_exposure_min_luminance),
		.maxAdaptedLuminance = CVAR_TO_FLOAT (rt_exposure_max_luminance),
		.tonemapType = (uint32_t)CLAMP (0, CVAR_TO_INT32 (rt_tonemap), 4),
	};

	vec3_t water_color;
	vec3_t acid_color;

	RT_GetWaterColor (water_color);
	RT_GetAcidColor (acid_color);

	// a zero or a hostile value here would poison the refraction of every glass
	// surface with a division by zero, so the uniform gets a sane index
	float refr_glass = CVAR_TO_FLOAT (rt_refr_glass);

	if (!(refr_glass >= 1.0f))
		refr_glass = 1.0f;
	else if (refr_glass > 5.0f)
		refr_glass = 5.0f;

	QrDrawFrameReflectRefractParams refl_refr_params = {
		.maxReflectRefractDepth = CVAR_TO_UINT32 (rt_reflrefr_depth),
		.typeOfMediaAroundCamera = rt_cameramedia,
		.indexOfRefractionGlass = refr_glass,
		.indexOfRefractionWater = CVAR_TO_FLOAT (rt_refr_water),
		.waterWaveSpeed = METRIC_TO_QUAKEUNIT (CVAR_TO_FLOAT (rt_water_speed)),
		.waterWaveNormalStrength = CVAR_TO_FLOAT (rt_water_normstren),
		.turbWarpStrength = editor_active ? 0.0f : CVAR_TO_FLOAT (rt_turb_warp),
		.waterColor = RT_VEC3 (water_color),
		.acidColor = RT_VEC3 (acid_color),
		.waterWaveTextureDerivativesMultiplier = CVAR_TO_FLOAT (rt_water_normsharp),
		.waterTextureAreaScale = METRIC_TO_QUAKEUNIT (CVAR_TO_FLOAT (rt_water_scale)),
		.portalNormalTwirl = CVAR_TO_BOOL (rt_portal_twirl),
		.glassShadows = CVAR_TO_BOOL (rt_glass_shadows),
		.glassDenoise = CVAR_TO_BOOL (rt_glass_denoise),
		.glassParticles = CVAR_TO_BOOL (rt_glass_particles),
	};
	// because 1 quake unit is not 1 meter
	refl_refr_params.waterColor.data[0] = powf (refl_refr_params.waterColor.data[0], 1.0f / METRIC_TO_QUAKEUNIT (1.0f));
	refl_refr_params.waterColor.data[1] = powf (refl_refr_params.waterColor.data[1], 1.0f / METRIC_TO_QUAKEUNIT (1.0f));
	refl_refr_params.waterColor.data[2] = powf (refl_refr_params.waterColor.data[2], 1.0f / METRIC_TO_QUAKEUNIT (1.0f));
	refl_refr_params.acidColor.data[0] = powf (refl_refr_params.acidColor.data[0], 1.0f / METRIC_TO_QUAKEUNIT (1.0f));
	refl_refr_params.acidColor.data[1] = powf (refl_refr_params.acidColor.data[1], 1.0f / METRIC_TO_QUAKEUNIT (1.0f));
	refl_refr_params.acidColor.data[2] = powf (refl_refr_params.acidColor.data[2], 1.0f / METRIC_TO_QUAKEUNIT (1.0f));


	float skyMult = 1.0f / CLAMP (0.02f, RT_Luminance (skyflatcolor), 1.0f);
	skyMult *= CVAR_TO_FLOAT (rt_sky);

	const float skyBrightness = RT_SKY_RADIANCE_SCALE * CVAR_TO_FLOAT (rt_sky_brightness) * CVAR_TO_FLOAT (rt_brightness);

	const qboolean materials_only = CVAR_TO_BOOL (rt_materials_only);

	const int usePhysicalSky = CVAR_TO_BOOL (rt_physical_sky) != 0;

	vec3_t sky_base_color;

	if (usePhysicalSky)
	{
		// The procedural sky is a flat color and nothing else, so it is given
		// exactly the color it is set to; skyBrightness reaches it once, as the
		// multiplier the shader paints the whole sky with.
		RT_GetSkyColor (sky_base_color);
	}
	else
	{
		VectorCopy (skyflatcolor, sky_base_color);
		VectorScale (sky_base_color, skyBrightness, sky_base_color);
	}

	if (materials_only)
	{
		sky_base_color[0] = sky_base_color[1] = sky_base_color[2] = 0.0f;
	}

	vec3_t sun_disc_color;
	RT_GetSunColor (sun_disc_color);

	QrDrawFrameSkyParams sky_params = {
		.skyType = CVAR_TO_BOOL (r_fastsky) ? QR_SKY_TYPE_COLOR
		         : usePhysicalSky ? QR_SKY_TYPE_PROCEDURAL
		         : QR_SKY_TYPE_RASTERIZED_GEOMETRY,
		.skyColorDefault = RT_VEC3 (sky_base_color),
		.sunDiscColor = RT_VEC3 (sun_disc_color),
		.sunDiscSize = CVAR_TO_FLOAT (rt_sky_sun_size),
		.skyColorMultiplier = materials_only ? 0.0f : (usePhysicalSky ? skyBrightness : skyMult * skyBrightness),
		// The procedural sky has no tint strength any more -- its color is its
		// own -- so the slot carries the opacity the clouds are composited with
		// instead (rt_sky_clouds_alpha); see QrDrawFrameSkyParams.
		.skyColorSaturation = CVAR_TO_FLOAT (rt_sky_clouds_alpha),
		.skyAmbientLod = CVAR_TO_FLOAT (rt_sky_ambient_lod),
		.skyLightMultiplier = CVAR_TO_FLOAT (rt_sky_light_mult),
		.skyNee = CVAR_TO_FLOAT (rt_sky_nee) > 0.0f,
		.skyViewerPosition = RT_VEC3 (r_origin),
		// The shafts are the light of the procedural sky scattered through the air:
		// with the classic sky (rt_physical_sky 0) there is no sun to scatter and
		// no shafts, whatever rt_sky_godrays says.
		.godRaysEnabled = CVAR_TO_BOOL (rt_sky_godrays),
		.godRaysIntensity = CVAR_TO_FLOAT (rt_sky_godrays_intensity),
		.godRaysQuality = CVAR_TO_UINT32 (rt_sky_godrays_quality),
		.godRaysFromSkyTexture = 0,
		.godRaysSkyDirection = {{0.0f, 0.0f, 0.0f}},
		.godRaysSkyColor = {{0.0f, 0.0f, 0.0f}},
	};

	if (usePhysicalSky)
	{
		sky_params.skyCloudsQuality = (uint32_t)CLAMP (0.0f, CVAR_TO_FLOAT (rt_sky_clouds_quality), (float)QR_SKY_CLOUDS_MAX_QUALITY);

		float *c = &sky_params.skyCubemapRotationTransform.matrix[0][0];
		// the cloud settings ride in the otherwise unused rotation matrix of this
		// sky type (keeps the public QrDrawFrameSkyParams layout unchanged)
		RT_GetSkyCloudsColor (c);
		c[3] = CVAR_TO_FLOAT (rt_sky_clouds_coverage);
		c[4] = CVAR_TO_FLOAT (rt_sky_clouds_density);
		c[5] = CVAR_TO_FLOAT (rt_sky_clouds_speed);
		c[6] = CVAR_TO_BOOL (rt_sky_clouds) ? 1.0f : 0.0f;
		c[7] = CVAR_TO_FLOAT (rt_sky_clouds_height);
		c[8] = CVAR_TO_FLOAT (rt_sky_clouds_thickness);
	}
	else if (!CVAR_TO_BOOL (r_fastsky) && !CVAR_TO_BOOL (rt_physical_sun))
	{
		vec3_t brightest_dir, brightest_color;

		// The shafts of a classic sky are not drawn any more: rt_sky_godrays is read
		// only for the procedural sky, and with rt_physical_sky 0 nothing on this
		// path reaches the frame. What is sent here is the aim a shaft *would* have
		// over a sky that is a picture -- the brightest point of the texture,
		// dressed as the sun -- kept for the record and for the day the shafts are
		// wanted over one again; the lookup behind it is done at load time.
		if (Sky_GetBrightestPoint (cl.time, brightest_dir, brightest_color))
		{
			// The brightest point of a rasterized sky stands in for the sun, so
			// it takes the sun's color like the directional light does -- and
			// the same fraction of the light fixup.
			RT_APPLY_SUN_COLOR (brightest_color);
			RT_FIXUP_LIGHT_INTENSITY (brightest_color, true);
			VectorScale (brightest_color, RT_SUN_LIGHT_INTENSITY_SCALE, brightest_color);
			VectorScale (brightest_color, RT_SKY_RADIANCE_SCALE * CLAMP (0.0f, CVAR_TO_FLOAT (rt_sky_brightness), 10.0f), brightest_color);

			sky_params.godRaysFromSkyTexture = 1;
			RT_VEC3_SET (sky_params.godRaysSkyDirection.data, brightest_dir[0], brightest_dir[1], brightest_dir[2]);
			RT_VEC3_SET (sky_params.godRaysSkyColor.data, brightest_color[0], brightest_color[1], brightest_color[2]);
		}
	}

	vec3_t volume_light_angles;
	vec3_t volume_light_color;

	RT_VEC3_SET (volume_light_angles, CVAR_TO_FLOAT (rt_sky_sun_pitch), CVAR_TO_FLOAT (rt_sky_sun_yaw), 0);

	if (CVAR_TO_BOOL (rt_physical_sun) && CVAR_TO_BOOL (rt_sky_sun))
		RT_GetSunColor (volume_light_color);
	else
		volume_light_color[0] = volume_light_color[1] = volume_light_color[2] = 0.0f;

	VectorScale (volume_light_color, CVAR_TO_FLOAT (rt_volume_lintensity) * CVAR_TO_FLOAT (rt_brightness), volume_light_color);
	RT_APPLY_LIGHT_TINT (volume_light_color);

	vec3_t volume_ambient_color;
	VectorScale (skyflatcolor, CVAR_TO_FLOAT (rt_volume_ambient) * CVAR_TO_FLOAT (rt_brightness), volume_ambient_color);
	RT_APPLY_LIGHT_TINT (volume_ambient_color);

	if (materials_only)
	{
		volume_light_color[0] = volume_light_color[1] = volume_light_color[2] = 0.0f;
		volume_ambient_color[0] = volume_ambient_color[1] = volume_ambient_color[2] = 0.0f;
	}

	QrDrawFrameVolumetricParams volumetric_params = {
		.enable = !materials_only && CVAR_TO_UINT32 (rt_volume_type) != 0,
		.useSimpleDepthBased = CVAR_TO_UINT32 (rt_volume_type) == 1,
		.volumetricFar = CVAR_TO_FLOAT (rt_volume_far),
		.ambientColor = RT_VEC3 (volume_ambient_color),
		.scaterring = materials_only ? 0.0f : CVAR_TO_FLOAT (rt_volume_scatter),
		.sourceColor = RT_VEC3 (volume_light_color),
		.sourceDirection = RT_AnglesToDir (volume_light_angles),
		.sourceAssymetry = CVAR_TO_FLOAT (rt_volume_lassymetry),
	};

	QrDrawFrameTexturesParams texture_params = {
		.dynamicSamplerFilter = CVAR_TO_INT32 (vid_filter) == 1 ? QR_SAMPLER_FILTER_NEAREST : QR_SAMPLER_FILTER_LINEAR,
		.normalMapStrength = CVAR_TO_FLOAT (rt_normalmap_stren),
		.emissionMapBoost = CVAR_TO_FLOAT (rt_emis_mapboost) * CVAR_TO_FLOAT (rt_emis_light_intensity),
		.emissionMaxScreenColor = CVAR_TO_FLOAT (rt_emis_maxscrcolor),
		.emissionSharpMask = CVAR_TO_FLOAT (rt_emis_sharpmask),
		.talSelfLitOffset = CVAR_TO_FLOAT (rt_tal_selflit),
		.minRoughness = CVAR_TO_FLOAT (rt_roughmin),
		.emissionBlendMode = CVAR_TO_UINT32 (rt_emis_blend),
		.emissionBlendStrength = CVAR_TO_FLOAT (rt_emis_blendstr),
	};

	for (int i = 0; i < MAX_LIGHTSTYLES; i++)
		texture_params.lightStyleScales[i] = (float)d_lightstylevalue[i] * (1.0f / 256.0f);

	// Classic level fog: the worldspawn "fog" key and the `fog` console
	// command, which Arcane Dimensions also uses to drive its dynamic fog. The
	// color is passed as it is, without an sRGB decoding, the same way the
	// classic renderer blended it into the framebuffer and the same way
	// Sky_DrawSky above hands it to the sky. The density is divided by the 64
	// the classic renderer scaled it with, so that a density of 0.05, a
	// mid-range value for the shipped maps, fades the far plane into the fog
	// instead of everything. A density of 0 draws no level fog.
	float level_fog_color[4];
	Fog_GetColor (level_fog_color);

	const qboolean level_fog_active = Fog_Enabled () && Fog_GetDensity () > 0;

	QrDrawFrameLevelFogParams level_fog_params = {
		.color = RT_VEC3 (level_fog_color),
		.density = (level_fog_active && !materials_only) ? Fog_GetDensity () / 64.0f : 0.0f,
		.skyBlend = (level_fog_active && !materials_only) ? skyfog : 0.0f,
	};

	QrPostEffectCRT crt_effect = {
		.isActive = CVAR_TO_BOOL (rt_ef_crt),
	};

	QrPostEffectColorTint tint_quad = {
		.isActive = true,
		.transitionDurationIn = 1.0f,
		.transitionDurationOut = 1.0f,
		.intensity = 4.0f,
		.color = {0.25f, 0.0f, 1.0f},
	};
	QrPostEffectColorTint tint_invuln = {
		.isActive = true,
		.transitionDurationIn = 1.0f,
		.transitionDurationOut = 1.0f,
		.intensity = 4.0f,
		.color = {1.0f, 0.0f, 0.0f},
	};
	QrPostEffectColorTint tint_lava = {
		.isActive = true,
		.transitionDurationIn = 0.05f,
		.transitionDurationOut = 0.5f,
		.intensity = 10.0f,
		.color = {1.0f, 0.1f, 0.0f},
	};
	static QrPostEffectColorTint tint_effect = {0}; // static, so prev state's transition durations are preserved
	tint_effect.isActive = false;
	if (cl.stats[STAT_HEALTH] > 0)
	{
	    if (cl.items & IT_QUAD) tint_effect = tint_quad;
	    else if (cl.items & IT_INVULNERABILITY) tint_effect = tint_invuln;
	    else if (rt_lavaeffects) tint_effect = tint_lava;
	}
	rt_dmg_inthisframe = false;

	static QrPostEffectsBloomParams bloom_effect = {0};
	bloom_effect.intensity = CLAMP (0.0f, CVAR_TO_FLOAT (rt_bloom_intensity), 0.2f);
	bloom_effect.isActive = bloom_effect.intensity > 0.0f;
	bloom_effect.threshold = CLAMP (0.0f, CVAR_TO_FLOAT (rt_bloom_threshold), 20.0f);
	bloom_effect.knee = CLAMP (0.0f, CVAR_TO_FLOAT (rt_bloom_knee), 1.0f);
	bloom_effect.scatter = CLAMP (0.0f, CVAR_TO_FLOAT (rt_bloom_scatter), 1.0f);
	bloom_effect.radius = CLAMP (0.005f, CVAR_TO_FLOAT (rt_bloom_radius), 0.15f);
	bloom_effect.quality = (uint32_t)CLAMP (0.0f, CVAR_TO_FLOAT (rt_bloom_quality), 2.0f);

	const float viewmodel_scale = CVAR_TO_FLOAT (rt_viewm_scale) > 0.0f ? CVAR_TO_FLOAT (rt_viewm_scale) : 1.0f;

	float dof_focus = 24.0f * viewmodel_scale;
	float dof_radius = 48.0f;
	if (rt_viewmodel_depth_far > rt_viewmodel_depth_near)
	{
		dof_focus = rt_viewmodel_depth_far * 0.8f;
		if (dof_focus <= rt_viewmodel_depth_near)
			dof_focus = rt_viewmodel_depth_far * 1.02f;

		const float dof_near = CLAMP (0.0f, rt_viewmodel_depth_near, dof_focus * 0.95f);
		dof_radius = 48.0f / q_max (1.0f - dof_near / dof_focus, 0.05f);
	}

	QrPostEffectsNearDofParams near_dof_effect = {
		.strength = CLAMP (0.0f, CVAR_TO_FLOAT (rt_dof_near), 1.0f),
		.focusDistance = dof_focus,
		.maxRadius = dof_radius,
	};

	static QrPostEffectsSharpenParams sharpen_effect = {0};
	sharpen_effect.strength = CLAMP (0.0f, CVAR_TO_FLOAT (rt_sharpen_strength), 1.0f);
	sharpen_effect.isActive = sharpen_effect.strength > 0.0f;

	const float suit_feedback = (cl.stats[STAT_HEALTH] > 0) ? rt_ef_suit_pulse : 0.0f;

	static QrPostEffectsGameplayFeedback feedback_effect = {0};
	feedback_effect.damage = q_max (rt_ef_damage_pulse, rt_ef_lowhealth_pulse) * CLAMP (0.0f, CVAR_TO_FLOAT (rt_ef_damage_strength), 1.0f);
	feedback_effect.liquid = rt_ef_liquid_pulse * CLAMP (0.0f, CVAR_TO_FLOAT (rt_ef_liquid_strength), 1.0f);
	feedback_effect.pickup = rt_ef_pickup_pulse * 0.00083f;
	feedback_effect.pickupHeight = 0.14f;
	feedback_effect.pickupColor = (QrFloat3D){{1.0f, 0.831373f, 0.482353f}};
	feedback_effect.aberration = CLAMP (0.0f, CVAR_TO_FLOAT (rt_ef_chraber), 1.0f);
	feedback_effect.suit = suit_feedback;

	QrPostEffectsVignetteParams vignette_effect = {
		.intensity = CLAMP (0.0f, CVAR_TO_FLOAT (rt_vignette) + suit_feedback * 0.25f, 1.0f),
		.start = CLAMP (0.0f, CVAR_TO_FLOAT (rt_vignette_start), 0.99f),
		.end = CLAMP (0.01f, CVAR_TO_FLOAT (rt_vignette_end), 2.0f),
		.roundness = CLAMP (0.0f, CVAR_TO_FLOAT (rt_vignette_roundness), 1.0f),
	};

	QrPostEffectsFilmGrainParams filmgrain_effect = {
		.intensity = CLAMP (0.0f, CVAR_TO_FLOAT (rt_filmgrain), 1.0f),
		.size = CLAMP (0.25f, CVAR_TO_FLOAT (rt_filmgrain_size), 8.0f),
	};

    QrPostEffectRadialBlur radial_effect = {
		.isActive = (cl.items & (IT_QUAD | IT_INVULNERABILITY)) && cl.stats[STAT_HEALTH] > 0,
		.transitionDurationIn = 1.0f,
		.transitionDurationOut = 2.0f,
	};

	QrPostEffectWaves waves_effect = {
		.isActive = rt_cameramedia != QR_MEDIA_TYPE_VACUUM,
		.transitionDurationIn = 0.1f,
		.transitionDurationOut = 0.75f,
		.amplitude = CVAR_TO_FLOAT (rt_ef_waves_stren) * 0.01f,
		.speed = 1.0f,
		.xMultiplier = 0.5f,
	};

	QrDrawFrameDebugParams debug_params = {
		.drawFlags = CVAR_TO_UINT32 (rt_debugflags),
	};
	debug_params.drawFlags |= QR_DEBUG_DRAW_Q2RTX_CORE_BIT;
	if (RT_StatsPanel (RT_STATS_RAYS))
	{
		debug_params.drawFlags |= QR_DEBUG_DRAW_STATS_BIT;
	}
	if (RT_StatsPanel (RT_STATS_PASSES) || rt_bench_active)
	{
		debug_params.drawFlags |= QR_DEBUG_DRAW_PASS_STATS_BIT;
	}

	float cameranear = GL_GetCameraNear (DEG2RAD (r_fovx), DEG2RAD (r_fovy));
	float camerafar = GL_GetCameraFar ();

	// The light editor's world is frozen: the traced water warp and the cloud
	// drift follow this clock, so it takes the held client time while the
	// editor runs instead of the wall clock.
	const double frame_time = editor_active ? (double)cl.time : (double)SDL_GetTicks () / 1000.0;

	QrDrawFrameInfo info = {
		.worldUpVector = {0, 0, 1},
		.fovYRadians = DEG2RAD (r_fovy),
		.cameraNear = cameranear,
		.cameraFar = camerafar,
		.rayLength = 10000.0f,
		.rayCullMaskWorld = QR_DRAW_FRAME_RAY_CULL_WORLD_0_BIT | QR_DRAW_FRAME_RAY_CULL_WORLD_1_BIT | QR_DRAW_FRAME_RAY_CULL_SKY_BIT,
		.disableRayTracedGeometry = false,
		.disableRasterization = false,
		.currentTime = frame_time,
		.disableEyeAdaptation = false,
		.forceAntiFirefly = CVAR_TO_BOOL (rt_antifirefly),
		.pRenderResolutionParams = &resolution_params,
		.pIlluminationParams = &illum_params,
		.pVolumetricParams = &volumetric_params,
		.pBloomParams = &bloom_params,
		.pTonemappingParams = &tonemap_params,
		.pReflectRefractParams = &refl_refr_params,
		.pSkyParams = &sky_params,
		.pTexturesParams = &texture_params,
		.pLevelFogParams = &level_fog_params,
		.postEffectParams =
			{
				.pChromaticAberration = NULL,
				.pWaves = (!editor_active && CVAR_TO_INT32(r_waterwarp) == 1) ? &waves_effect : NULL,
				.pColorTint = (cl.intermission || editor_active) ? NULL : &tint_effect,
				.pCRT = &crt_effect,
				.pRadialBlur = (cl.intermission || editor_active) ? NULL : &radial_effect,
				.pBloom = &bloom_effect,
				.pNearDof = (cl.intermission || editor_active) ? NULL : &near_dof_effect,
				.pSharpen = &sharpen_effect,
				.pVignette = &vignette_effect,
				.pFilmGrain = &filmgrain_effect,
				.localExposure = CLAMP (0.0f, CVAR_TO_FLOAT (rt_local_exposure), 1.0f),
				.pGameplayFeedback = (cl.intermission || editor_active) ? NULL : &feedback_effect,
			},
		.pDebugParams = &debug_params,
		.renderUiOnly = RT_ShouldRenderUiOnly (cl.worldmodel != NULL, cls.signon == SIGNONS),
		.enableCpuProfiling = RT_StatsPanel (RT_STATS_PROFILE) || rt_bench_active,
	};
	memcpy (info.view, vulkan_globals.view_matrix, 16 * sizeof(float));

	double prof_start = RT_Prof_Begin ();
	QrResult r = qrDrawFrame (vulkan_globals.instance, &info);
	QR_CHECK (r);

	if (parms->serial != 0)
	{
		QrFrameStats stats = {0};
		const qboolean stats_valid = (r == QR_SUCCESS && info.enableCpuProfiling) && qrGetFrameStatsEx (vulkan_globals.instance, &stats) == QR_SUCCESS && stats.cpuTimingValid;

		RT_Prof_EndTaskRecord (parms->serial, prof_start, stats_valid ? &stats : NULL);
	}
	else
	{
		RT_Prof_End (RT_PROF_DRAWFRAME, prof_start);
		if (r == QR_SUCCESS && info.enableCpuProfiling)
			RT_Prof_RecordRenderer ();
	}

	Atomic_StoreUInt32 (&rt_end_task_running, 0);
}

/*
=================
GL_EndRendering
=================
*/
task_handle_t GL_EndRendering (qboolean use_tasks, qboolean swapchain)
{
	if (use_tasks)
	{
		if (++rt_end_task_serial == 0)
			rt_end_task_serial = 1;
		rt_end_task_current_serial = rt_end_task_serial;

		RT_ProfWindowSubmitEnd (&rt_prof_window, &rt_prof_window_sync);
	}
	else
		rt_end_task_current_serial = 0;

	end_rendering_parms_t parms = {
		.serial = rt_end_task_current_serial,
		.delay_ms = use_tasks ? (uint32_t)q_max (0.0f, CVAR_TO_FLOAT (rt_end_task_delay_ms)) : 0,
		.vid_width = (float)vid.width,
		.vid_height = (float)vid.height,
	};
	task_handle_t end_rendering_task = INVALID_TASK_HANDLE;
	if (use_tasks)
		end_rendering_task = Task_AllocateAndAssignFunc ((task_func_t)GL_EndRenderingTask, &parms, sizeof (parms));
	else
		GL_EndRenderingTask (&parms);
	return end_rendering_task;
}

/*
=================
GL_WaitForDeviceIdle
=================
*/
void GL_WaitForDeviceIdle (void)
{
	assert(!Tasks_IsWorker());
	GL_SynchronizeEndRenderingTask ();
}

/*
=================
VID_Shutdown
=================
*/
void VID_Shutdown (void)
{
	if (vid_initialized)
	{
		Cursor_Shutdown ();

		if (vulkan_globals.instance != QR_NULL_HANDLE)
		{
		    QR_GUI_Shutdown ();
		    RT_MAT_Shutdown ();
		    RT_LIGHT_Shutdown ();
		    QrResult r = qrDestroyInstance (vulkan_globals.instance);
			QR_CHECK (r);

			Mem_Free (vulkan_globals.primary_cb_context.batch_indices);
			Mem_Free (vulkan_globals.primary_cb_context.batch_verts);
			for (int i = 0; i < CBX_NUM; i++)
			{
				Mem_Free (vulkan_globals.secondary_cb_contexts[i].batch_indices);
				Mem_Free (vulkan_globals.secondary_cb_contexts[i].batch_verts);
			}
		}

		SDL_QuitSubSystem (SDL_INIT_VIDEO);
		draw_context = NULL;
		PL_VID_Shutdown ();
	}
}

/*
===================================================================

MAIN WINDOW

===================================================================
*/

/*
================
ClearAllStates
================
*/
static void ClearAllStates (void)
{
	Key_ClearStates ();
	IN_ClearStates ();
}

//==========================================================================
//
//  COMMANDS
//
//==========================================================================

/*
=================
VID_DescribeCurrentMode_f
=================
*/
static void VID_DescribeCurrentMode_f (void)
{
	if (draw_context)
		Con_Printf (
			"%dx%dx%d %dHz %s\n", VID_GetCurrentWidth (), VID_GetCurrentHeight (), VID_GetCurrentBPP (), VID_GetCurrentRefreshRate (),
			VID_GetFullscreen () ? "fullscreen" : "windowed");
}

/*
=================
VID_DescribeModes_f -- johnfitz -- changed formatting, and added refresh rates after each mode.
=================
*/
static void VID_DescribeModes_f (void)
{
	int i;
	int lastwidth, lastheight, count;

	lastwidth = lastheight = count = 0;

	for (i = 0; i < nummodes; i++)
	{
		if (lastwidth != modelist[i].width || lastheight != modelist[i].height)
		{
			if (count > 0)
				Con_SafePrintf ("\n");
			Con_SafePrintf ("   %4i x %4i : %i", modelist[i].width, modelist[i].height, modelist[i].refreshrate);
			lastwidth = modelist[i].width;
			lastheight = modelist[i].height;
			count++;
		}
	}
	Con_Printf ("\n%i modes\n", count);
}

//==========================================================================
//
//  INIT
//
//==========================================================================

/*
=================
VID_InitModelist
=================
*/
static void VID_InitModelist (void)
{
	const int sdlmodes = SDL_GetNumDisplayModes (0);
	int       i;

	modelist = Mem_Realloc (modelist, sizeof (vmode_t) * sdlmodes);
	nummodes = 0;
	for (i = 0; i < sdlmodes; i++)
	{
		SDL_DisplayMode mode;

		if (SDL_GetDisplayMode (0, i, &mode) == 0)
		{
			modelist[nummodes].width = mode.w;
			modelist[nummodes].height = mode.h;
			modelist[nummodes].refreshrate = mode.refresh_rate;
			nummodes++;
		}
	}
}

static void RT_SunEditChanged_f (cvar_t *var)
{
	(void)var;

	if (CVAR_TO_BOOL (rt_sky_sun_edit))
	{
		Con_Printf ("Sun editor: the sun follows the crosshair, press fire to leave it there.\n");

		if (!CVAR_TO_BOOL (rt_physical_sky))
		{
			Con_Printf ("Sun editor: rt_physical_sky is 0, and the classic sky has no sun to place -- set rt_physical_sky 1 first.\n");
		}
		else if (CVAR_TO_FLOAT (rt_sky_sun) <= 0.0f)
		{
			Con_Printf ("Sun editor: rt_sky_sun is 0, so there is no sun to see -- set rt_sky_sun 1 first.\n");
		}
	}
	else
	{
		Con_Printf ("Sun placed: rt_sky_sun_pitch %s, rt_sky_sun_yaw %s\n", rt_sky_sun_pitch.string, rt_sky_sun_yaw.string);
	}

	QR_Editor_SunPlacement (CVAR_TO_BOOL (rt_sky_sun_edit));
}

static void RT_SunPreset_f (cvar_t *var)
{
	const int preset = CLAMP (0, CVAR_TO_INT32 (rt_sky_sun_preset), 7);

	if (preset == 0)
	{
		return;
	}

	static const int presets[8][3] = {
		{0, 0, 0},
		{255, 214, 163},
		{255, 235, 200},
		{255, 246, 230},
		{255, 178, 92},
		{200, 216, 255},
		{168, 118, 218},
		{140, 180, 255},
	};

	Cvar_Set ("rt_sky_sun_color", va ("%d %d %d", presets[preset][0], presets[preset][1], presets[preset][2]));
}

extern atomic_uint32_t rt_require_static_submit;

static void RT_LightStylesChanged_f (cvar_t *var)
{
	(void)var;
	Atomic_StoreUInt32 (&rt_require_static_submit, true);
}

static void RT_BrushPersistentChanged_f (cvar_t *var)
{
	(void)var;
	Atomic_StoreUInt32 (&rt_require_static_submit, true);
}

static void RT_EmissiveLimitsChanged_f (cvar_t *var)
{
	(void)var;
	Atomic_StoreUInt32 (&rt_require_static_submit, true);
}

// Both diagnostics run on the current map, so they do not wait for a map reload.
static void RT_WorldCensusChanged_f (cvar_t *var)
{
	(void)var;
	RT_WorldCensus ();
}

static void RT_WorldLightsStatsChanged_f (cvar_t *var)
{
	(void)var;
	RT_UploadWorldLights ();
}

static QrFogVolume rt_fog_volumes[QR_MAX_FOG_VOLUMES];

static void RT_Fog_ParsePoint (const char *s, float *out)
{
	if (!strcmp (s, "here"))
	{
		VectorCopy (r_origin, out);
		return;
	}

	if (3 != sscanf (s, "%f,%f,%f", &out[0], &out[1], &out[2]))
	{
		Con_Printf ("invalid coordinates '%s'\n", s);
	}
}

static uint32_t RT_Fog_ParseSoftFace (const char *s)
{
	if (!strcmp (s, "xa")) return 1;
	if (!strcmp (s, "xb")) return 2;
	if (!strcmp (s, "ya")) return 3;
	if (!strcmp (s, "yb")) return 4;
	if (!strcmp (s, "za")) return 5;
	if (!strcmp (s, "zb")) return 6;
	return 0;
}

static const char *RT_Fog_SoftFaceName (uint32_t softface)
{
	static const char *names[] = {"none", "xa", "xb", "ya", "yb", "za", "zb"};
	if (softface > 6)
	{
		softface = 0;
	}
	return names[softface];
}

static void RT_Fog_PrintVolume (int index, const QrFogVolume *vol)
{
	Con_Printf ("fog -v %d -a %.2f,%.2f,%.2f -b %.2f,%.2f,%.2f -c %.2f,%.2f,%.2f -d %.0f -f %s\n",
	            index,
	            vol->pointA.data[0], vol->pointA.data[1], vol->pointA.data[2],
	            vol->pointB.data[0], vol->pointB.data[1], vol->pointB.data[2],
	            vol->color.data[0], vol->color.data[1], vol->color.data[2],
	            vol->halfExtinctionDistance,
	            RT_Fog_SoftFaceName (vol->softface));
}

static void RT_Fog_Cmd (void)
{
	const int argc = Cmd_Argc ();
	if (argc <= 1)
	{
		Con_Printf ("usage: fog -v <index> -a <x,y,z|here> -b <x,y,z|here> -c <r,g,b> -d <distance> -f <none|xa|xb|ya|yb|za|zb>\n");
		Con_Printf ("       fog <density> <r> <g> <b>                      set the level fog\n");
		return;
	}

	// The name is shared with the classic Quake fog command, which is also how
	// mods set fog at runtime ("fog <density> <r> <g> <b>", as Arcane
	// Dimensions does). Every option of the volume editor starts with a dash,
	// so a first argument that does not is left to the classic handler.
	if (Cmd_Argv (1)[0] != '-')
	{
		Fog_FogCommand_f ();
		return;
	}

	int          index = -1;
	QrFogVolume *vol   = NULL;

	for (int i = 1; i < argc; i++)
	{
		const char *arg = Cmd_Argv (i);

		if (!strcmp (arg, "-h"))
		{
			Con_Printf ("Set parameters of a Q2RTX-style fog volume.\n");
			return;
		}
		else if (!strcmp (arg, "-v") && i + 1 < argc)
		{
			index = atoi (Cmd_Argv (++i));
			if (index < 0 || index >= QR_MAX_FOG_VOLUMES)
			{
				Con_Printf ("invalid volume index '%d'\n", index);
				return;
			}
			vol = &rt_fog_volumes[index];
		}
		else if (!strcmp (arg, "-a") && i + 1 < argc)
		{
			if (!vol) goto no_volume;
			RT_Fog_ParsePoint (Cmd_Argv (++i), vol->pointA.data);
		}
		else if (!strcmp (arg, "-b") && i + 1 < argc)
		{
			if (!vol) goto no_volume;
			RT_Fog_ParsePoint (Cmd_Argv (++i), vol->pointB.data);
		}
		else if (!strcmp (arg, "-c") && i + 1 < argc)
		{
			if (!vol) goto no_volume;
			RT_Fog_ParsePoint (Cmd_Argv (++i), vol->color.data);
		}
		else if (!strcmp (arg, "-d") && i + 1 < argc)
		{
			if (!vol) goto no_volume;
			vol->halfExtinctionDistance = atof (Cmd_Argv (++i));
		}
		else if (!strcmp (arg, "-f") && i + 1 < argc)
		{
			if (!vol) goto no_volume;
			vol->softface = RT_Fog_ParseSoftFace (Cmd_Argv (++i));
		}
		else if (!strcmp (arg, "-p"))
		{
			if (!vol) goto no_volume;
			RT_Fog_PrintVolume (index, vol);
		}
		else if (!strcmp (arg, "-r"))
		{
			if (!vol) goto no_volume;
			memset (vol, 0, sizeof (*vol));
		}
		else if (!strcmp (arg, "-R"))
		{
			memset (rt_fog_volumes, 0, sizeof (rt_fog_volumes));
		}
		else
		{
			Con_Printf ("unknown fog option '%s'\n", arg);
			return;
		}
	}

	qrSetFogVolumes (vulkan_globals.instance, QR_MAX_FOG_VOLUMES, rt_fog_volumes);
	return;

no_volume:
	Con_Printf ("volume not specified\n");
}

static void RT_ParticleResolveCacheChanged_f (cvar_t *var)
{
	RT_PointClusterCacheSetEnabled (CVAR_TO_BOOL (*var));
}

/*
===================
VID_Init
===================
*/
void VID_Init (void)
{
	static char vid_center[] = "SDL_VIDEO_CENTERED=center";
	int         p, width, height, refreshrate;
	int         display_width, display_height, display_refreshrate;
	qboolean    fullscreen;
	const char *read_vars[] = {"vid_fullscreen",        "vid_width",    "vid_height", "vid_refreshrate", "vid_vsync",
	                           "vid_desktopfullscreen", "vid_borderless"};
#define num_readvars (sizeof (read_vars) / sizeof (read_vars[0]))

	Cvar_RegisterVariable (&vid_fullscreen);  // johnfitz
	Cvar_RegisterVariable (&vid_width);       // johnfitz
	Cvar_RegisterVariable (&vid_height);      // johnfitz
	Cvar_RegisterVariable (&vid_refreshrate); // johnfitz
	Cvar_RegisterVariable (&vid_vsync);       // johnfitz
	Cvar_SetCallback (&vid_vsync, VID_Vsync_f);
	Cvar_RegisterVariable (&vid_maxframelatency);
	Cvar_RegisterVariable (&vid_filter);
	Cvar_RegisterVariable (&vid_desktopfullscreen); // QuakeSpasm
	Cvar_RegisterVariable (&vid_borderless);        // QuakeSpasm
	Cvar_RegisterVariable (&vid_palettize);

	Cvar_SetCallback (&vid_fullscreen, VID_Changed_f);
	Cvar_SetCallback (&vid_width, VID_Changed_f);
	Cvar_SetCallback (&vid_height, VID_Changed_f);
	Cvar_SetCallback (&vid_refreshrate, VID_Changed_f);
	Cvar_SetCallback (&vid_desktopfullscreen, VID_Changed_f);
	Cvar_SetCallback (&vid_borderless, VID_Changed_f);

	// RT
	{
		RT_ColorInit ();

#define CVAR_DEF_T(name, default_value) Cvar_RegisterVariable (&name);
		CVAR_DEF_LIST (CVAR_DEF_T)
#undef CVAR_DEF_T

		/* Read-only: the incremental composition is the only mode the renderer selects. The
		   legacy non-incremental path (0) composes every list of the scene on every frame, and
		   the light a scene keeps was measured to flicker in it; an engine-side Cvar_SetROM is
		   the only way to select it. A saved 0 in a configuration is ignored. */
		rt_cluster_incremental.flags |= CVAR_ROM;

		Cvar_SetROM ("rt_globallight_mult", "10");

		Cvar_RegisterVariable (&rt_light_report_filter);
		Cvar_RegisterVariable (&rt_sky_sun_edit);
		Cvar_RegisterVariable (&r_particles_overflow);

		Cvar_SetCallback (&rt_particle_resolve_cache, RT_ParticleResolveCacheChanged_f);
		RT_ParticleResolveCacheChanged_f (&rt_particle_resolve_cache);

		// The panels cvar is archived and used to hold any set of panels; the command
		// takes a level now, so a value left by the old form --- or typed by hand --- is
		// folded into the panels of the level it asks for as it is set.
		Cvar_SetCallback (&rt_stats_panels, RT_StatsPanelsFixup);

		// The color settings are read per light, so they watch their cvar instead of
		// comparing its string on every read. Registered after the cvars, which is
		// where VID_Init has them.
		for (size_t i = 0; i < countof (rt_colors); i++)
		{
			Cvar_SetCallback (rt_colors[i].cvar, RT_ColorChanged_f);
		}
	}

	Cvar_SetCallback (&rt_sky_sun_preset, RT_SunPreset_f);
	Cvar_SetCallback (&rt_sky_clouds_quality, VID_CloudsQuality_f);
	VID_CloudsQuality_f (&rt_sky_clouds_quality);
	Cvar_SetCallback (&rt_sky_sun_edit, RT_SunEditChanged_f);
	Cvar_SetCallback (&rt_light_styles, RT_LightStylesChanged_f);
	Cvar_SetCallback (&rt_light_styles_reach, RT_LightStylesChanged_f);
	Cvar_SetCallback (&rt_brush_persistent, RT_BrushPersistentChanged_f);
	Cvar_SetCallback (&rt_dtal_minarea, RT_EmissiveLimitsChanged_f);
	Cvar_SetCallback (&rt_dtal_maxpolys, RT_EmissiveLimitsChanged_f);
	Cvar_SetCallback (&rt_dtal_clearance, RT_EmissiveLimitsChanged_f);
	Cvar_SetCallback (&rt_dtal_groups, RT_EmissiveLimitsChanged_f);
	Cvar_SetCallback (&rt_dtal_spacing, RT_EmissiveLimitsChanged_f);
	Cvar_SetCallback (&rt_worldcensus, RT_WorldCensusChanged_f);
	Cvar_SetCallback (&rt_worldlights_stats, RT_WorldLightsStatsChanged_f);

	Cmd_AddCommand ("vid_unlock", VID_Unlock);     // johnfitz
	Cmd_AddCommand ("vid_restart", VID_Restart_f); // johnfitz
	Cmd_AddCommand ("vid_test", VID_Test);         // johnfitz
	Cmd_AddCommand ("vid_describecurrentmode", VID_DescribeCurrentMode_f);
	Cmd_AddCommand ("vid_describemodes", VID_DescribeModes_f);

#ifdef _DEBUG
	Cmd_AddCommand ("create_palette_octree", CreatePaletteOctree_f);
#endif

	putenv (vid_center); /* SDL_putenv is problematic in versions <= 1.2.9 */

	if (SDL_InitSubSystem (SDL_INIT_VIDEO) < 0)
		Sys_Error ("Couldn't init SDL video: %s", SDL_GetError ());

	{
		SDL_DisplayMode mode;
		if (SDL_GetDesktopDisplayMode (0, &mode) != 0)
			Sys_Error ("Could not get desktop display mode: %s\n", SDL_GetError ());

		display_width = mode.w;
		display_height = mode.h;
		display_refreshrate = mode.refresh_rate;
	}

	if (CFG_OpenConfig ("config.cfg") == 0)
	{
		CFG_ReadCvars (read_vars, num_readvars);
		CFG_CloseConfig ();
	}
	CFG_ReadCvarOverrides (read_vars, num_readvars);

	VID_InitModelist ();

	width = (int)vid_width.value;
	height = (int)vid_height.value;
	refreshrate = (int)vid_refreshrate.value;
	fullscreen = (int)vid_fullscreen.value;

	if (COM_CheckParm ("-current"))
	{
		width = display_width;
		height = display_height;
		refreshrate = display_refreshrate;
		fullscreen = true;
	}
	else
	{
		p = COM_CheckParm ("-width");
		if (p && p < com_argc - 1)
		{
			width = atoi (com_argv[p + 1]);

			if (!COM_CheckParm ("-height"))
				height = width * 3 / 4;
		}

		p = COM_CheckParm ("-height");
		if (p && p < com_argc - 1)
		{
			height = atoi (com_argv[p + 1]);

			if (!COM_CheckParm ("-width"))
				width = height * 4 / 3;
		}

		p = COM_CheckParm ("-refreshrate");
		if (p && p < com_argc - 1)
			refreshrate = atoi (com_argv[p + 1]);

		if (COM_CheckParm ("-window") || COM_CheckParm ("-w"))
			fullscreen = false;
		else if (COM_CheckParm ("-fullscreen") || COM_CheckParm ("-f"))
			fullscreen = true;
	}

	if (width <= 0 || height <= 0)
	{
		width = display_width;
		height = display_height;
		refreshrate = display_refreshrate;
		fullscreen = false;
	}

	if (!VID_ValidMode (width, height, refreshrate, fullscreen))
	{
		width = (int)vid_width.value;
		height = (int)vid_height.value;
		refreshrate = (int)vid_refreshrate.value;
		fullscreen = (int)vid_fullscreen.value;
	}

	if (!VID_ValidMode (width, height, refreshrate, fullscreen))
	{
		width = 640;
		height = 480;
		refreshrate = display_refreshrate;
		fullscreen = false;
	}

	vid_initialized = true;

	vid.colormap = host_colormap;
	vid.fullbright = 256 - LittleLong (*((int *)vid.colormap + 2048));

	VID_SetMode (width, height, refreshrate, fullscreen);

	// set window icon
	PL_SetWindowIcon ();

	Con_Printf ("\nRay tracing Initialization\n");
	GL_InitInstance ();
	Cursor_Init ();

	// johnfitz -- removed code creating "glquake" subdirectory

	vid_menucmdfn = VID_Menu_f; // johnfitz
	vid_menudrawfn = VID_MenuDraw;
	vid_menukeyfn = VID_MenuKey;

	VID_Gamma_Init (); // johnfitz
	VID_Menu_Init ();  // johnfitz

	// QuakeSpasm: current vid settings should override config file settings.
	// so we have to lock the vid mode from now until after all config files are read.
	vid_locked = true;
}

/*
===================
VID_Restart
===================
*/
static void VID_Restart (qboolean set_mode)
{
	GL_SynchronizeEndRenderingTask ();

	int      width, height, refreshrate;
	qboolean fullscreen;

	width = (int)vid_width.value;
	height = (int)vid_height.value;
	refreshrate = (int)vid_refreshrate.value;
	fullscreen = vid_fullscreen.value ? true : false;

	//
	// validate new mode
	//
	if (set_mode && !VID_ValidMode (width, height, refreshrate, fullscreen))
	{
		Con_Printf ("%dx%d %dHz %s is not a valid mode\n", width, height, refreshrate, fullscreen ? "fullscreen" : "windowed");
		return;
	}

	scr_initialized = false;

	GL_WaitForDeviceIdle ();

	//
	// set new mode
	//
	if (set_mode)
		VID_SetMode (width, height, refreshrate, fullscreen);

	// conwidth and conheight need to be recalculated
	if (vid.width > 0 && vid.height > 0)
	{
		vid.conwidth = (scr_conwidth.value > 0) ? (int)scr_conwidth.value : (scr_conscale.value > 0) ? (int)(vid.width / scr_conscale.value) : vid.width;
		vid.conwidth = CLAMP (320, vid.conwidth, vid.width);
		vid.conwidth &= 0xFFFFFFF8;
		vid.conheight = vid.conwidth * vid.height / vid.width;
	}
	//
	// keep cvars in line with actual mode
	//
	VID_SyncCvars ();

	Cursor_Init ();

	Con_Printf ("Video: %dx%d at %d Hz, vsync %s\n", vid.width, vid.height, vid_display_refresh,
	            VID_VsyncModeName ((int)vid_vsync.value));

	//
	// update mouse grab
	//
	if (key_dest == key_console || key_dest == key_menu)
	{
		if (modestate == MS_WINDOWED)
			IN_Deactivate (true);
		else if (modestate == MS_FULLSCREEN)
			IN_Activate ();
	}

	SCR_UpdateRelativeScale ();

	scr_initialized = true;
}

/*
===================
VID_Restart_f -- johnfitz -- change video modes on the fly
===================
*/
static void VID_Restart_f (void)
{
	if (vid_locked || !vid_changed)
		return;
	VID_Restart (true);
}

/*
===================
VID_Toggle
new proc by S.A., called by alt-return key binding.
===================
*/
void VID_Toggle (void)
{
	qboolean toggleWorked;
	Uint32   flags = 0;

	S_ClearBuffer ();

	if (!VID_GetFullscreen ())
	{
		flags = vid_desktopfullscreen.value ? SDL_WINDOW_FULLSCREEN_DESKTOP : SDL_WINDOW_FULLSCREEN;
	}

	toggleWorked = SDL_SetWindowFullscreen (draw_context, flags) == 0;
	if (toggleWorked)
	{
		modestate = VID_GetFullscreen () ? MS_FULLSCREEN : MS_WINDOWED;

		VID_SyncCvars ();

		Cursor_Init ();

		// update mouse grab
		if (key_dest == key_console || key_dest == key_menu)
		{
			if (modestate == MS_WINDOWED)
				IN_Deactivate (true);
			else if (modestate == MS_FULLSCREEN)
				IN_Activate ();
		}
	}
}

#define UPSCALER_OFF  0
#define UPSCALER_FSR31 1
#define UPSCALER_DLSS 2

static int GetUpscalerDefaultQuality (int type)
{
	return (type == UPSCALER_FSR31) ? 2 : 1;
}

// For settings that are not applied during vid_restart
typedef struct
{
	int host_maxfps;
	int r_particles;
	int vid_filter;
	int upscaler_type;
	int upscaler_quality;
} vid_menu_settings_t;

static vid_menu_settings_t menu_settings;

/*
================
VID_SyncCvars -- johnfitz -- set vid cvars to match current video mode
================
*/
void VID_SyncCvars (void)
{
	if (draw_context)
	{
		if (!VID_GetDesktopFullscreen ())
		{
			Cvar_SetValueQuick (&vid_width, VID_GetCurrentWidth ());
			Cvar_SetValueQuick (&vid_height, VID_GetCurrentHeight ());
		}
		Cvar_SetValueQuick (&vid_refreshrate, VID_GetCurrentRefreshRate ());
		Cvar_SetQuick (&vid_fullscreen, VID_GetFullscreen () ? "1" : "0");
		// don't sync vid_desktopfullscreen, it's a user preference that
		// should persist even if we are in windowed mode.
	}

	vid_display_refresh = VID_GetCurrentRefreshRate ();

	menu_settings.host_maxfps = CLAMP (0, host_maxfps.value, 1000);
	menu_settings.r_particles = CLAMP (0, (int)r_particles.value, 2);
	{
		int fsr31 = CVAR_TO_INT32 (rt_upscale_fsr31);
		int dlss = CVAR_TO_INT32 (rt_upscale_dlss);
		if (fsr31 > 0)      { menu_settings.upscaler_type = UPSCALER_FSR31; menu_settings.upscaler_quality = CLAMP (0, fsr31, 6); }
		else if (dlss > 0)  { menu_settings.upscaler_type = UPSCALER_DLSS;  menu_settings.upscaler_quality = CLAMP (0, dlss, 4); }
		else                { menu_settings.upscaler_type = UPSCALER_OFF;   menu_settings.upscaler_quality = 0; }
	}
	menu_settings.vid_filter = CLAMP (0, (int)vid_filter.value, 1);

	vid_changed = false;
}

//==========================================================================
//
//  NEW VIDEO MENU -- johnfitz
//
//==========================================================================

enum
{
	VID_OPT_MODE,
	VID_OPT_REFRESHRATE,
	VID_OPT_APPLY,

	VID_OPT_UPSCALER,
	VID_OPT_UPSCALER_QUALITY,


	VID_OPT_VSYNC,
	VID_OPT_MAX_FPS,

	VID_OPT_MATERIALS_ONLY,


	VID_OPT_FOV,
	VID_OPT_SHOWFPS,


	VID_OPT_DENOISER,
	VID_OPT_TEXTURES,

	VIDEO_OPTIONS_ITEMS
};

static int video_options_cursor = 0;

typedef struct
{
	int width, height;
} vid_menu_mode;

// TODO: replace these fixed-length arrays with hunk_allocated buffers
static vid_menu_mode vid_menu_modes[MAX_MODE_LIST];
static int           vid_menu_nummodes = 0;

static int vid_menu_rates[MAX_RATES_LIST];
static int vid_menu_numrates = 0;

/*
================
VID_Menu_Init
================
*/
static void VID_Menu_Init (void)
{
	int i, j, h, w;

	for (i = 0; i < nummodes; i++)
	{
		w = modelist[i].width;
		h = modelist[i].height;

		for (j = 0; j < vid_menu_nummodes; j++)
		{
			if (vid_menu_modes[j].width == w && vid_menu_modes[j].height == h)
				break;
		}

		if (j == vid_menu_nummodes)
		{
			vid_menu_modes[j].width = w;
			vid_menu_modes[j].height = h;
			vid_menu_nummodes++;
		}
	}
}

/*
================
VID_Menu_RebuildRateList

regenerates rate list based on current vid_width, vid_height
================
*/
static void VID_Menu_RebuildRateList (void)
{
	int i, j, r;

	vid_menu_numrates = 0;

	for (i = 0; i < nummodes; i++)
	{
		// rate list is limited to rates available with current width/height
		if (modelist[i].width != vid_width.value || modelist[i].height != vid_height.value)
			continue;

		r = modelist[i].refreshrate;

		for (j = 0; j < vid_menu_numrates; j++)
		{
			if (vid_menu_rates[j] == r)
				break;
		}

		if (j == vid_menu_numrates)
		{
			vid_menu_rates[j] = r;
			vid_menu_numrates++;
		}
	}

	// if there are no valid fullscreen refreshrates for this width/height, just pick one
	if (vid_menu_numrates == 0)
	{
		Cvar_SetValue ("vid_refreshrate", (float)modelist[0].refreshrate);
		return;
	}

	// if vid_refreshrate is not in the new list, change vid_refreshrate
	for (i = 0; i < vid_menu_numrates; i++)
		if (vid_menu_rates[i] == (int)(vid_refreshrate.value))
			break;

	if (i == vid_menu_numrates)
		Cvar_SetValue ("vid_refreshrate", (float)vid_menu_rates[0]);
}

/*
================
VID_Menu_ChooseNextMode

chooses next resolution in order, then updates vid_width and
vid_height cvars, then updates refreshrate lists
================
*/
static void VID_Menu_ChooseNextMode (int dir)
{
	if (vid_menu_nummodes)
	{
		int i;
		for (i = 0; i < vid_menu_nummodes; i++)
		{
			if (vid_menu_modes[i].width == vid_width.value && vid_menu_modes[i].height == vid_height.value)
				break;
		}

		if (i == vid_menu_nummodes) // can't find it in list, so it must be a custom windowed res
		{
			i = 0;
		}
		else
		{
			i = CLAMP (0, i + dir, vid_menu_nummodes - 1);
		}

		Cvar_SetValueQuick (&vid_width, (float)vid_menu_modes[i].width);
		Cvar_SetValueQuick (&vid_height, (float)vid_menu_modes[i].height);
		VID_Menu_RebuildRateList ();
	}
}

static void VID_Menu_ChooseNextVsync (int dir)
{
	int mode = (int)vid_vsync.value + dir;

	if (mode < VID_VSYNC_OFF)
		mode = VID_VSYNC_FREESYNC;
	else if (mode > VID_VSYNC_FREESYNC)
		mode = VID_VSYNC_OFF;

	Cvar_SetValueQuick (&vid_vsync, (float)mode);
}

/*
================
VID_Menu_ChooseNextMaxFPS
================
*/
static void VID_Menu_ChooseNextMaxFPS (int dir)
{
	menu_settings.host_maxfps = CLAMP (0, ((menu_settings.host_maxfps + (dir * 10)) / 10) * 10, 1000);
}

/*
================
VID_Menu_ChooseNextRate

chooses next refresh rate in order, then updates vid_refreshrate cvar
================
*/
static void VID_Menu_ChooseNextRate (int dir)
{
	int i;

	// no fullscreen rates for the current size (custom windowed mode, etc.)
	if (vid_menu_numrates <= 0)
		return;

	for (i = 0; i < vid_menu_numrates; i++)
	{
		if (vid_menu_rates[i] == vid_refreshrate.value)
			break;
	}

	if (i == vid_menu_numrates) // can't find it in list
	{
		i = 0;
	}
	else
	{
		i += dir;
		if (i >= vid_menu_numrates)
			i = 0;
		else if (i < 0)
			i = vid_menu_numrates - 1;
	}

	Cvar_SetValue ("vid_refreshrate", (float)vid_menu_rates[i]);
}


static void VID_Menu_ChooseNextAA (int vidopt, int dir)
{
	QrBool32 fsr31_ok = qrIsRenderUpscaleTechniqueAvailable (vulkan_globals.instance, QR_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3);
	QrBool32 dlss_ok = qrIsRenderUpscaleTechniqueAvailable (vulkan_globals.instance, QR_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS);

	const int prev_type = menu_settings.upscaler_type;
	const int maxq_fsr31 = fsr31_ok ? 6 : 0;
	const int maxq_dlss = dlss_ok ? 4 : 0;

	if (vidopt == VID_OPT_UPSCALER)
	{
		do {
			menu_settings.upscaler_type += dir < 0 ? -1 : 1;
			if (menu_settings.upscaler_type < 0) menu_settings.upscaler_type = UPSCALER_DLSS;
			if (menu_settings.upscaler_type > UPSCALER_DLSS) menu_settings.upscaler_type = UPSCALER_OFF;
		} while (
			(menu_settings.upscaler_type == UPSCALER_FSR31 && !fsr31_ok) ||
			(menu_settings.upscaler_type == UPSCALER_DLSS  && !dlss_ok));

		if (menu_settings.upscaler_type != prev_type)
		{
			menu_settings.upscaler_quality = (menu_settings.upscaler_type == UPSCALER_OFF) ? 0 : GetUpscalerDefaultQuality (menu_settings.upscaler_type);
		}
	}
	else if (vidopt == VID_OPT_UPSCALER_QUALITY)
	{
		int maxq;
		switch (menu_settings.upscaler_type)
		{
		case UPSCALER_FSR31: maxq = maxq_fsr31; break;
		case UPSCALER_DLSS:  maxq = maxq_dlss; break;
		default:             maxq = 0; break;
		}
		if (maxq > 0)
		{
			menu_settings.upscaler_quality += dir < 0 ? -1 : 1;
			menu_settings.upscaler_quality = CLAMP (1, menu_settings.upscaler_quality, maxq);
		}
	}
}


/*
================
VID_Menu_StepFloatCvar -- step a float cvar in 0.1 increments, clamped
================
*/
static void VID_Menu_StepFloatCvar (cvar_t *var, float step, float minval, float maxval)
{
	float v = CLAMP (minval, var->value + step, maxval);

	// snap to the displayed precision so the value cannot drift
	v = floorf (v * 10.0f + 0.5f) / 10.0f;

	Cvar_SetValueQuick (var, v);
}

static void VID_Menu_Adjust (int dir)
{
	switch (video_options_cursor)
	{
	case VID_OPT_MODE:
		VID_Menu_ChooseNextMode (-dir);
		break;
	case VID_OPT_REFRESHRATE:
		VID_Menu_ChooseNextRate (-dir);
		break;
	case VID_OPT_VSYNC:
		VID_Menu_ChooseNextVsync (dir);
		break;
	case VID_OPT_MAX_FPS:
		VID_Menu_ChooseNextMaxFPS (dir);
		Cvar_SetValueQuick (&host_maxfps, menu_settings.host_maxfps);
		break;
	case VID_OPT_UPSCALER:
	case VID_OPT_UPSCALER_QUALITY:
		VID_Menu_ChooseNextAA (video_options_cursor, dir);
		{
			int q = menu_settings.upscaler_quality;
			if (menu_settings.upscaler_type != UPSCALER_OFF && q < 1)
				q = GetUpscalerDefaultQuality (menu_settings.upscaler_type);
			Cvar_SetValueQuick (&rt_upscale_fsr31, (menu_settings.upscaler_type == UPSCALER_FSR31) ? q : 0);
			Cvar_SetValueQuick (&rt_upscale_dlss, (menu_settings.upscaler_type == UPSCALER_DLSS) ? q : 0);
		}
		break;
	case VID_OPT_MATERIALS_ONLY:
		Cvar_SetValueQuick (&rt_materials_only, !CVAR_TO_BOOL (rt_materials_only));
		break;
	case VID_OPT_FOV:
		VID_Menu_StepFloatCvar (&scr_fov, dir * 5.0f, 60.0f, 140.0f);
		break;
	case VID_OPT_SHOWFPS:
		Cvar_SetValueQuick (&scr_showfps, !CVAR_TO_BOOL (scr_showfps));
		break;
	case VID_OPT_DENOISER:
		Cvar_SetValueQuick (&rt_denoiser, !CVAR_TO_BOOL (rt_denoiser));
		break;
	case VID_OPT_TEXTURES:
		Cvar_SetValueQuick (&rt_no_textures, !CVAR_TO_BOOL (rt_no_textures));
		break;
	}
}

/*
================
VID_MenuKey
================
*/
static void VID_MenuKey (int key)
{
	if (key == K_MOUSE2 || key == K_ESCAPE || key == K_BBUTTON || key == K_BACKSPACE)
	{
		VID_SyncCvars (); // sync cvars before leaving menu. FIXME: there are other ways to leave menu
		S_LocalSound ("misc/menu1.wav");
		M_Menu_Options_f ();

		return;
	}

	switch (key)
	{
	case K_UPARROW:
		S_LocalSound ("misc/menu1.wav");
		video_options_cursor--;
		if (video_options_cursor < 0)
			video_options_cursor = VIDEO_OPTIONS_ITEMS - 1;
		break;

	case K_DOWNARROW:
		S_LocalSound ("misc/menu1.wav");
		video_options_cursor++;
		if (video_options_cursor >= VIDEO_OPTIONS_ITEMS)
			video_options_cursor = 0;
		break;

	case K_ENTER:
	case K_KP_ENTER:
	case K_MOUSE1:
		m_entersound = true;
		if (video_options_cursor == VID_OPT_APPLY)
		{
			Cbuf_AddText ("vid_restart\n");
		}
		else
		{
			VID_Menu_Adjust (1);
		}
		break;

	case K_LEFTARROW:
		S_LocalSound ("misc/menu3.wav");
		VID_Menu_Adjust (-1);
		break;

	case K_RIGHTARROW:
		S_LocalSound ("misc/menu3.wav");
		VID_Menu_Adjust (1);
		break;

	default:
		break;
	}
}

void M_Menu_Video_f (void)
{
	VID_Menu_f ();
}

void M_Video_Draw (cb_context_t *cbx)
{
	VID_MenuDraw (cbx);
}

void M_Video_Key (int key)
{
	VID_MenuKey (key);
}

/*
================
VID_MenuDraw
================
*/
static void VID_MenuDraw (cb_context_t *cbx)
{
	int         i, y;
	int         row_y[VIDEO_OPTIONS_ITEMS];
	qpic_t     *p;
	const char *title;

	y = 4;

	// plaque
	p = Draw_CachePic ("gfx/qplaque.lmp");
	M_DrawTransPic (cbx, 16, y, p);

	// p = Draw_CachePic ("gfx/vidmodes.lmp");
	p = Draw_CachePic ("gfx/p_option.lmp");
	M_DrawPic (cbx, (320 - p->width) / 2, y, p);

	y += 28;

	// title
	title = "Video Options";
	M_PrintWhite (cbx, (320 - 8 * strlen (title)) / 2, y, title);

	y += 12;

	// options
	for (i = 0; i < VIDEO_OPTIONS_ITEMS; i++)
	{
		switch (i)
		{
		case VID_OPT_MODE:
			M_Print (cbx, 16, y, "        Video mode");
			M_Print (cbx, 184, y, va ("%ix%i", (int)vid_width.value, (int)vid_height.value));
			break;
		case VID_OPT_REFRESHRATE:
			M_Print (cbx, 16, y, "      Refresh rate");
			M_Print (cbx, 184, y, va ("%i", (int)vid_refreshrate.value));
			break;
		case VID_OPT_APPLY:
			M_Print (cbx, 16, y, "             Apply");
			break;


		case VID_OPT_UPSCALER:
			y += 8; // separate

			M_Print (cbx, 16, y, "          Upscaler");
			{
				const char *name = "Off";
				switch (menu_settings.upscaler_type)
				{
				case UPSCALER_FSR31: name = "AMD FSR 3.1"; break;
				case UPSCALER_DLSS:  name = "Nvidia DLSS"; break;
				}
				M_Print (cbx, 184, y, name);
			}
			break;
		case VID_OPT_UPSCALER_QUALITY:
			M_Print (cbx, 16, y, "            Preset");
			{
				QrRenderUpscaleTechnique tech;
				int q = menu_settings.upscaler_quality;
				switch (menu_settings.upscaler_type)
				{
				case UPSCALER_FSR31: tech = QR_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3; break;
				case UPSCALER_DLSS:  tech = QR_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS; break;
				default:             tech = QR_RENDER_UPSCALE_TECHNIQUE_NEAREST; q = 0; break;
				}
				if (q < 1 && menu_settings.upscaler_type != UPSCALER_OFF)
					q = GetUpscalerDefaultQuality (menu_settings.upscaler_type);
				M_Print (cbx, 184, y, GetUpscalerOptionName (q, tech));
			}
			break;


		case VID_OPT_VSYNC:
			M_Print (cbx, 16, y, "     Vertical sync");
			M_Print (cbx, 184, y, VID_VsyncModeName ((int)vid_vsync.value));
			break;
		case VID_OPT_MAX_FPS:
			M_Print (cbx, 16, y, "           Max FPS");
			if (menu_settings.host_maxfps <= 0)
				M_Print (cbx, 184, y, "no limit");
			else
				M_Print (cbx, 184, y, va ("%d", menu_settings.host_maxfps));
			break;


		case VID_OPT_MATERIALS_ONLY:
			M_Print (cbx, 16, y, "  Materials only");
			M_Print (cbx, 184, y, CVAR_TO_BOOL (rt_materials_only) ? "on" : "off");
			break;


		case VID_OPT_FOV:
			M_Print (cbx, 16, y, "     Field of view");
			M_Print (cbx, 184, y, va ("%d", (int)scr_fov.value));
			break;
		case VID_OPT_SHOWFPS:
			M_Print (cbx, 16, y, "       Display FPS");
			M_DrawCheckbox (cbx, 184, y, (int)scr_showfps.value);
			break;


		case VID_OPT_DENOISER:
			M_Print (cbx, 16, y, "          Denoiser");
			M_DrawCheckbox (cbx, 184, y, CVAR_TO_BOOL (rt_denoiser));
			break;
		case VID_OPT_TEXTURES:
			M_Print (cbx, 16, y, "          Textures");
			M_DrawCheckbox (cbx, 184, y, !CVAR_TO_BOOL (rt_no_textures));
			break;
		}

		row_y[i] = y;

		y += 8;
	}

	for (i = 0; i < VIDEO_OPTIONS_ITEMS; i++)
		M_Mouse_UpdateCursor (&video_options_cursor, 16, 320, row_y[i], 8, i);

	M_DrawCharacter (cbx, 168, row_y[video_options_cursor], 12 + ((int)(realtime * 4) & 1));
}

/*
================
VID_Menu_f
================
*/
static void VID_Menu_f (void)
{
	IN_Deactivate (modestate == MS_WINDOWED);
	key_dest = key_menu;
	m_state = m_video;
	m_entersound = true;

	// set all the cvars to match the current mode when entering the menu
	VID_SyncCvars ();

	// set up bpp and rate lists based on current cvars
	VID_Menu_RebuildRateList ();
}

/*
==============================================================================

SCREEN SHOTS

==============================================================================
*/

/*
==================
SCR_ScreenShot_f -- asks the renderer for a PNG of the next frame in com_gamedir
==================
*/
void SCR_ScreenShot_f (void)
{
	static int	screenshotNumber;
	char		name[MAX_OSPATH];
	char		pathname[MAX_OSPATH];
	int			i;
	QrResult	result;

	if (vulkan_globals.instance == NULL)
	{
		Con_Printf ("screenshot: the renderer is not initialized\n");
		return;
	}

	Sys_mkdir (com_gamedir);

	for (i = screenshotNumber; i < screenshotNumber + 10000; i++)
	{
		q_snprintf (name, sizeof (name), "screenshot%04d.png", i);

		if (!COM_FileExists (name, NULL))
		{
			break;
		}
	}

	screenshotNumber = i + 1;

	q_snprintf (pathname, sizeof (pathname), "%s/%s", com_gamedir, name);

	result = qrRequestScreenshot (vulkan_globals.instance, pathname);

	if (result == QR_SUCCESS)
	{
		Con_Printf ("screenshot: %s\n", pathname);
	}
	else
	{
		Con_Printf ("screenshot: failed (%s)\n", qrGetResultDescription (result));
	}
}

void VID_FocusGained (void)
{
	has_focus = true;
}

void VID_FocusLost (void)
{
	has_focus = false;
}
