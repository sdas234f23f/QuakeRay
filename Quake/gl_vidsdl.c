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
#include "resource.h"
#include "palette.h"
#include "rt_material.h"
#include "SDL.h"
#include "SDL_syswm.h"
#include <time.h> // for the timestamp of the frame rt_stats_dump appends

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
static cvar_t                   vid_vsync = {"vid_vsync", "0", CVAR_ARCHIVE};
static cvar_t                   vid_desktopfullscreen = {"vid_desktopfullscreen", "0", CVAR_ARCHIVE}; // QuakeSpasm
static cvar_t                   vid_borderless = {"vid_borderless", "0", CVAR_ARCHIVE};               // QuakeSpasm
static cvar_t                   vid_palettize = {"vid_palettize", "0", CVAR_ARCHIVE};
cvar_t                          vid_filter = {"vid_filter", "1", CVAR_ARCHIVE};
cvar_t                          vid_gamma = {"gamma", "1", CVAR_ARCHIVE};       // johnfitz -- moved here from view.c
cvar_t                          vid_contrast = {"contrast", "1", CVAR_ARCHIVE}; // QuakeSpasm, MarkV
cvar_t                          r_usesops = {"r_usesops", "1", CVAR_ARCHIVE};   // johnfitz

task_handle_t prev_end_rendering_task = INVALID_TASK_HANDLE;

// RT
#define CVAR_DEF_LIST( CVAR_DEF_T ) \
	\
	CVAR_DEF_T (rt_enable_pvs, "0") \
	CVAR_DEF_T (rt_shadowrays, "2") \
	CVAR_DEF_T (rt_indir2bounces, "0") \
	CVAR_DEF_T (rt_gi_level, "1") \
	CVAR_DEF_T (rt_sun_bounce_range, "2000") \
	CVAR_DEF_T (rt_sun_bounce_scale, "1.0") \
	CVAR_DEF_T (rt_godrays, "1") \
	CVAR_DEF_T (gr_intensity, "1") /* Q2RTX's god ray strength, hence the name */ \
	CVAR_DEF_T (rt_denoiser, "1") \
	CVAR_DEF_T (rt_no_textures, "0") \
	CVAR_DEF_T (rt_antifirefly, "1") \
	CVAR_DEF_T (rt_roughmin, "0.02") \
    \
	CVAR_DEF_T (rt_dlight_intensity, "3.0") \
	CVAR_DEF_T (rt_dlight_radius, "0.1") \
	\
	CVAR_DEF_T (rt_plight_intensity, "3.0") \
	CVAR_DEF_T (rt_plight_radius, "0.02") \
	\
	CVAR_DEF_T (rt_wlight_intensity, "3.0") \
	CVAR_DEF_T (rt_wlight_radius, "0.01") \
	\
	CVAR_DEF_T (rt_emis_light_intensity, "1.0") \
	\
	CVAR_DEF_T (rt_elight_normaliz, "100") \
	CVAR_DEF_T (rt_elight_default, "200") \
	CVAR_DEF_T (rt_elight_default_mdl, "1000") \
	CVAR_DEF_T (rt_elight_threshold, "-1") \
    CVAR_DEF_T (rt_elight_radius, "0.01") \
    \
	CVAR_DEF_T (rt_light_reach, "25") \
	/* Reach of a light of a moving entity, in metres: the distance it is promised not to reach
	   past, and so the reason a torch or a flame reaches a few rooms instead of every list of
	   the map. The lights of the map itself state no reach and keep the one their leaf gives
	   them. */ \
	CVAR_DEF_T (rt_light_reach_max, "10") \
	CVAR_DEF_T (rt_cluster_dlights, "1") \
	/* 1 takes back only the slots of the lights that moved and hands them out again from where \
	   those lights stand now: every other list of the scene keeps what the composition left, so \
	   a moving lamp costs the lists by the rooms it covers rather than by the map. 0 keeps the \
	   lists all or nothing, a light that moves takes every list of the scene with it, which is \
	   what a lava ball was measured to cost. */ \
	CVAR_DEF_T (rt_cluster_incremental, "1") \
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
	/* 0 uploads the map's lights one call at a time, for measuring the batched path. */ \
	CVAR_DEF_T (rt_wmodel_lights_batch, "1") \
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
	CVAR_DEF_T (rt_sun, "1") \
	CVAR_DEF_T (rt_sun_pitch, "140") \
	CVAR_DEF_T (rt_sun_yaw, "120") \
	CVAR_DEF_T (rt_sun_preset, "0") \
	CVAR_DEF_T (rt_flashlight, "0") \
	\
	CVAR_DEF_T (rt_muzzleoffs_x, "0") \
	CVAR_DEF_T (rt_muzzleoffs_y, "-30") \
	CVAR_DEF_T (rt_muzzleoffs_z, "100") \
	\
	CVAR_DEF_T (rt_sky, "1") \
	CVAR_DEF_T (rt_sky_tint, "1.0") \
	CVAR_DEF_T (rt_sky_ambient_lod, "4") \
	CVAR_DEF_T (rt_sky_nee, "1") \
	CVAR_DEF_T (rt_physical_sky, "1") \
	CVAR_DEF_T (rt_sky_color, "32 0 64") \
	CVAR_DEF_T (rt_sky_brightness, "1.0") \
	CVAR_DEF_T (rt_brightness, "1.0") \
	CVAR_DEF_T (rt_light_color, "255 255 255") \
	CVAR_DEF_T (rt_sky_clouds, "1") \
	CVAR_DEF_T (rt_sky_clouds_color, "0 0 0") \
	CVAR_DEF_T (rt_sky_cloud_coverage, "0.2") \
	CVAR_DEF_T (rt_sky_cloud_density, "0.8") \
	CVAR_DEF_T (rt_sky_cloud_speed, "0.3") \
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
    \
	CVAR_DEF_T (rt_reflrefr_depth, "2") \
	CVAR_DEF_T (rt_refr_glass, "1.52") \
	CVAR_DEF_T (rt_refr_water, "1.33") \
	\
	CVAR_DEF_T (rt_volume_type, "2") \
	CVAR_DEF_T (rt_volume_far, "1000") \
	CVAR_DEF_T (rt_volume_scatter, "0.3") \
	CVAR_DEF_T (rt_volume_ambient, "2.0") \
	CVAR_DEF_T (rt_volume_lintensity, "250") \
	CVAR_DEF_T (rt_volume_lassymetry, "0.0") \
	CVAR_DEF_T (rt_level_fog, "1") \
    \
	CVAR_DEF_T (rt_water_aciddensity, "25") \
	CVAR_DEF_T (rt_water_speed, "0.4") \
	CVAR_DEF_T (rt_water_normstren, "1") \
	CVAR_DEF_T (rt_water_normsharp, "5") \
	CVAR_DEF_T (rt_water_scale, "1") \
	CVAR_DEF_T (rt_turb_warp, "1") \
	\
	CVAR_DEF_T (rt_portal_twirl, "1") \
	CVAR_DEF_T (rt_teleport_portals, "0") \
    \
	CVAR_DEF_T (rt_sharpen, "0") \
	CVAR_DEF_T (rt_renderscale, "0") \
	CVAR_DEF_T (rt_vintage, "0") \
	CVAR_DEF_T (rt_upscale_fsr2, "0") \
	CVAR_DEF_T (rt_upscale_fsr31, "2") \
	CVAR_DEF_T (rt_upscale_dlss, "0") \
	\
	CVAR_DEF_T (rt_sensit_dir, "0.4") \
	CVAR_DEF_T (rt_sensit_indir, "0.06") \
	CVAR_DEF_T (rt_sensit_spec, "0.03") \
	\
	CVAR_DEF_T (rt_globallight_mult, "5") \
	CVAR_DEF_T (rt_globallight, "255 255 255") \
	\
	CVAR_DEF_T (rt_bloom_intensity, "1") \
	CVAR_DEF_T (rt_bloom_emis_mult, "50") \
	CVAR_DEF_T (rt_bloom, "0") \
	\
	CVAR_DEF_T (rt_exposure_bias, "-2.8") \
	CVAR_DEF_T (rt_contrast, "0.6") \
	\
	CVAR_DEF_T (rt_ef_crt, "0") \
	CVAR_DEF_T (rt_ef_chraber, "0.3") \
	CVAR_DEF_T (rt_ef_waves_stren, "1") \
	\
	CVAR_DEF_T (rt_viewm_fovscale, "1.2") \
	CVAR_DEF_T (rt_viewm_wide, "1.05") \
	\
	CVAR_DEF_T (rt_hud_minimal, "1") \
	CVAR_DEF_T (rt_hud_padding, "8") \
	\
	CVAR_DEF_T (rt_debugflags, "0") \
	CVAR_DEF_T (rt_debugemissive, "0") \
	CVAR_DEF_T (rt_q2_depthgrad, "1") \
	CVAR_DEF_T (rt_q2_lightstats, "1") \
	CVAR_DEF_T (rt_reflrefr_earlyout, "1") \
	CVAR_DEF_T (rt_nee_samples, "1") \
	CVAR_DEF_T (rt_stats_panels, "0") \
	CVAR_DEF_T (rt_worldcensus, "0") \
	CVAR_DEF_T (rt_worldlights_stats, "0") \
	CVAR_DEF_T (rt_worldclusters_grid, "1") \
	\
	CVAR_DEF_T (_rt_firsttime, "1")



#define CVAR_DEF_T(name, default_value) cvar_t name = {#name, default_value, CVAR_ARCHIVE};
    CVAR_DEF_LIST (CVAR_DEF_T)
#undef CVAR_DEF_T

cvar_t rt_light_report_filter = {"rt_light_report_filter", "", 0};


enum
{
	RT_VINTAGE_OFF,
	RT_VINTAGE_CRT,
	RT_VINTAGE_200,
	RT_VINTAGE_480,
	RT_VINTAGE_720,

	RT_VINTAGE__COUNT
};


/*
================
RT frame profiler -- rt_stats 3

Times the CPU side of the frame, which the GPU timestamps of panel 2 do not
cover: the geometry marking chain, the per-pass scene submission and the main
thread's wait for the task graph. The results are drawn on screen once a second
by SCR_DrawRTStats, and rt_stats_dump writes one snapshot to qperfdump.log.

The slots are written from worker threads without synchronization, so taking the
report and the reset that follows it can race a running task; for a diagnostic
tool the worst case is a garbage value in one window.
================
*/
double           rt_prof_ms[RT_PROF_COUNT];
rt_prof_report_t rt_prof_report;

static double   rt_prof_frame_start;
static double   rt_prof_window_start;
static int      rt_prof_frames;
static qboolean rt_prof_active;

double RT_Prof_Begin (void)
{
	return RT_StatsPanel (RT_STATS_PROFILE) ? Sys_DoubleTime () : 0.0;
}

void RT_Prof_End (int slot, double start)
{
	if (start == 0.0)
		return;

	const double ms = (Sys_DoubleTime () - start) * 1000.0;
	if (ms > rt_prof_ms[slot])
		rt_prof_ms[slot] = ms;
}

void RT_Prof_Sample (int slot, double ms)
{
	if (ms > rt_prof_ms[slot])
		rt_prof_ms[slot] = ms;
}

void RT_Prof_FrameStart (void)
{
	if (!RT_StatsPanel (RT_STATS_PROFILE))
		return;

	rt_prof_frame_start = Sys_DoubleTime ();
	++rt_prof_frames;
}

void RT_Prof_FrameEnd (void)
{
	if (!RT_StatsPanel (RT_STATS_PROFILE))
		return;

	RT_Prof_End (RT_PROF_FRAME, rt_prof_frame_start);
}

void RT_Prof_Update (void)
{
	if (!RT_StatsPanel (RT_STATS_PROFILE))
	{
		if (!rt_prof_active)
			return;

		rt_prof_active = false;
		rt_prof_report.valid = false;
		memset (rt_prof_ms, 0, sizeof (rt_prof_ms));
		rt_prof_frames = 0;
		rt_cluster_cache_hits = 0;
		rt_cluster_cache_misses = 0;
		rt_cluster_miss_set = 0;
		rt_cluster_miss_move = 0;
		rt_cluster_miss_other = 0;
		return;
	}

	const double now = Sys_DoubleTime ();

	if (!rt_prof_active)
	{
		// start a fresh window, so switching the profiler on never shows stale numbers
		rt_prof_active = true;
		rt_prof_report.valid = false;
		rt_prof_window_start = now;
		rt_prof_frames = 0;
		rt_cluster_cache_hits = 0;
		rt_cluster_cache_misses = 0;
		rt_cluster_miss_set = 0;
		rt_cluster_miss_move = 0;
		rt_cluster_miss_other = 0;
		return;
	}

	const double elapsed = now - rt_prof_window_start;
	if (elapsed < 1.0)
		return;

	rt_prof_window_start = now;

	float fps = (float)(rt_prof_frames / elapsed);
	if (fps < 0.1f)
		fps = 0.1f;

	memset (&rt_prof_report, 0, sizeof (rt_prof_report));
	rt_prof_report.fps = fps;
	rt_prof_report.frameMs = (float)rt_prof_ms[RT_PROF_FRAME];
	rt_prof_report.waitMs = (float)rt_prof_ms[RT_PROF_WAIT];

	for (int i = 0; i < RT_PROF_COUNT; ++i)
		rt_prof_report.ms[i] = (float)rt_prof_ms[i];

	rt_prof_report.clusterCacheHits = rt_cluster_cache_hits;
	rt_prof_report.clusterCacheMisses = rt_cluster_cache_misses;
	rt_prof_report.clusterMissSet = rt_cluster_miss_set;
	rt_prof_report.clusterMissMove = rt_cluster_miss_move;
	rt_prof_report.clusterMissOther = rt_cluster_miss_other;
	rt_prof_report.clusterGrants = rt_cluster_last_grants;
	rt_prof_report.clusterDenied = rt_cluster_last_denied;
	rt_prof_report.clusterGated = rt_cluster_last_gated;
	rt_prof_report.clusterLights = rt_cluster_last_lights;
	rt_prof_report.clusterAttempts = rt_cluster_last_attempts;
	rt_prof_report.clusterDropped = rt_cluster_last_dropped;
	rt_prof_report.valid = true;

	rt_cluster_cache_hits = 0;
	rt_cluster_cache_misses = 0;
	rt_cluster_miss_set = 0;
	rt_cluster_miss_move = 0;
	rt_cluster_miss_other = 0;
	memset (rt_prof_ms, 0, sizeof (rt_prof_ms));
	rt_prof_frames = 0;
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
		{ RT_PROF_CLUSTERS_FILL, "clust fill" },
		{ RT_PROF_CLUSTERS_UPLOAD, "clust upload" },
		{ RT_PROF_DRAWFRAME, "rgDrawFrame" },
		{ RT_PROF_WAIT, "wait" },
		{ RT_PROF_FRAME, "frame" },
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
		snap->haveGpu = (rgGetFrameStatsEx (vulkan_globals.instance, &snap->gpu) == RG_SUCCESS);

	if (rt_prof_report.valid)
	{
		snap->profile = rt_prof_report;
		snap->haveProfile = true;
	}
}


/*
================
RT_StatsPrintPanels -- state line shared by rt_stats and its old spellings
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

	Con_Printf ("%s showing %s   (1 = ray counters, 2 = GPU pass timings, 3 = CPU profile)\n",
	            prefix, n ? on : "nothing");
}

/*
================
RT_Stats_f -- rt_stats 1,2,3

Replaces the three readouts that used to be switched one by one: the argument
lists the panels to show, so "rt_stats 1,2,3" shows all of them, "rt_stats 2"
only the GPU timings and "rt_stats 0" hides the readout. Without an argument the
current selection is printed.
================
*/
static void RT_Stats_f (void)
{
	unsigned int mask = 0;
	qboolean     invalid = false;
	int          i;

	if (Cmd_Argc () < 2)
	{
		RT_StatsPrintPanels ("rt_stats is");
		return;
	}

	for (i = 1; i < Cmd_Argc (); i++)
	{
		const char *arg = Cmd_Argv (i);

		for (; *arg; arg++)
		{
			const int panel = *arg - '0';

			if (panel < 0 || panel > RT_STATS_PROFILE)
				invalid = true;
			else if (panel > 0)
				mask |= 1u << (panel - 1);
		}
	}

	if (invalid)
	{
		Con_Printf ("rt_stats: expected the panels 1, 2 and 3, like \"rt_stats 1,2,3\"\n");
		return;
	}

	Cvar_SetValueQuick (&rt_stats_panels, (float)mask);
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
			for (i = 0; i < RG_GPU_PASS_COUNT; i++)
				fprintf (f, "%-11s %-17s %.2f\n", "gpu.pass", rgGetGpuPassName (i), snap->gpu.gpuPassMs[i]);
		else
			fprintf (f, "%-11s %-17s %s\n", "gpu.pass", "timings", "not collected, rt_stats 2 was off");

		fprintf (f, "%-11s %-17s %u\n", "gpu.rays", "total", snap->gpu.raysTotal);
		fprintf (f, "%-11s %-17s %u\n", "gpu.rays", "primary", snap->gpu.raysPerCategory[0]);
		fprintf (f, "%-11s %-17s %u\n", "gpu.rays", "refl refr", snap->gpu.raysPerCategory[1]);
		fprintf (f, "%-11s %-17s %u\n", "gpu.rays", "indirect", snap->gpu.raysPerCategory[2]);
		fprintf (f, "%-11s %-17s %u\n", "gpu.rays", "shadow dir", snap->gpu.raysPerCategory[3]);
		fprintf (f, "%-11s %-17s %u\n", "gpu.rays", "shadow ind", snap->gpu.raysPerCategory[4]);
		fprintf (f, "%-11s %-17s %u\n", "gpu.calls", "rg entry points", snap->gpu.apiCalls);
	}
	else
	{
		fprintf (f, "%-11s %-17s %s\n", "gpu", "unavailable", "the backend returned no frame stats");
	}

	fputc ('\n', f);

	if (snap->haveProfile)
	{
		fprintf (f, "%-11s %-17s %.2f\n", "cpu.frame", "ms", snap->profile.frameMs);
		fprintf (f, "%-11s %-17s %.2f\n", "cpu.main", "ms", snap->profile.frameMs - snap->profile.waitMs);
		fprintf (f, "%-11s %-17s %.2f\n", "cpu.wait", "ms", snap->profile.waitMs);
		fprintf (f, "%-11s %-17s %.1f\n", "fps", "-", snap->profile.fps);

		for (i = 0; i < RT_PROF_COUNT; i++)
			fprintf (f, "%-11s %-17s %.2f\n", "cpu.slot", RT_ProfSlotName (i), snap->profile.ms[i]);

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

	if (!RT_StatsPanel (RT_STATS_RAYS) || !RT_StatsPanel (RT_STATS_PASSES))
		Con_Printf ("rt_stats_dump: ray counters need rt_stats 1 and the GPU timings need rt_stats 2\n");
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
	return !(SDL_GetWindowFlags (draw_context) & SDL_WINDOW_SHOWN);
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

static void RT_ReloadShaders (void)
{
	request_shaders_reload = true;
}

static vec3_t rt_water_color = {171 / 255.0f, 193 / 255.0f, 210 / 255.0f};
static void RT_WaterColor(void)
{
	if (Cmd_Argc () != 4)
	{
		Con_Printf ("current: %d %d %d\n", (int)(rt_water_color[0] * 255), (int)(rt_water_color[1] * 255), (int)(rt_water_color[2] * 255));
		Con_Printf ("usage: <r 0..255> <g 0..255> <b 0..255>\n");
		return;
	}

	RT_VEC3_SET (
		rt_water_color, 
		strtof (Cmd_Argv (1), NULL) / 255.0f, 
		strtof (Cmd_Argv (2), NULL) / 255.0f, 
		strtof (Cmd_Argv (3), NULL) / 255.0f );
}

static vec3_t rt_acid_color = {0 / 255.0f, 169 / 255.0f, 145 / 255.0f};
static void RT_AcidColor(void)
{
	if (Cmd_Argc () != 4)
	{
		Con_Printf ("current: %d %d %d\n", (int)(rt_acid_color[0] * 255), (int)(rt_acid_color[1] * 255), (int)(rt_acid_color[2] * 255));
		Con_Printf ("usage: <r 0..255> <g 0..255> <b 0..255>\n");
		return;
	}
	
	RT_VEC3_SET (
		rt_acid_color, 
		strtof (Cmd_Argv (1), NULL) / 255.0f, 
		strtof (Cmd_Argv (2), NULL) / 255.0f, 
		strtof (Cmd_Argv (3), NULL) / 255.0f );
}

// A colour setting is a console command plus an archived cvar of the same name,
// so one setting describes a colour completely. It cannot be a plain cvar:
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
	RT_COLOR_CLOUDS,
	RT_COLOR_LIGHT,
	RT_COLOR_GLOBALLIGHT,

	RT_COLOR_COUNT
} rt_color_index_t;

static rt_color_t rt_colors[RT_COLOR_COUNT] = {
	[RT_COLOR_SKY]         = {.cvar = &rt_sky_color,        .fallback = {32 / 255.0f, 0.0f, 64 / 255.0f}, .dirty = true},
	[RT_COLOR_CLOUDS]      = {.cvar = &rt_sky_clouds_color, .fallback = {0.0f, 0.0f, 0.0f},                .dirty = true},
	[RT_COLOR_LIGHT]       = {.cvar = &rt_light_color,      .fallback = {1.0f, 1.0f, 1.0f},                .dirty = true},
	[RT_COLOR_GLOBALLIGHT] = {.cvar = &rt_globallight,      .fallback = {1.0f, 1.0f, 1.0f},                .dirty = true},
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
		c->dirty = false;

		if (!RT_ColorParse (c->cvar->string, c->value))
		{
			Con_Printf ("%s: expected <r 0..255> <g 0..255> <b 0..255>, got \"%s\"\n", c->cvar->name, c->cvar->string);
			VectorCopy (c->fallback, c->value);
		}
	}

	VectorCopy (c->value, out);
}

void RT_GetSkyColor (float color[3])
{
	RT_ColorGet (&rt_colors[RT_COLOR_SKY], color);
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

// Colour settings are commands backed by archivable cvars: the command parses the
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

#ifdef RG_USE_SURFACE_WIN32
	RgWin32SurfaceCreateInfo win32Info = {.hinstance = wmInfo.info.win.hinstance, .hwnd = wmInfo.info.win.window};
#elif RG_USE_SURFACE_XLIB
	RgXlibSurfaceCreateInfo x11Info = {.dpy = wmInfo.info.x11.display, .window = wmInfo.info.x11.window};
#endif

	const char pShaderPath[] = RT_OVERRIDEN_FOLDER "shaders/";
	const char pBlueNoisePath[] = RT_OVERRIDEN_FOLDER "BlueNoise_LDR_RGBA_128.ktx2";
	const char pWaterTexturePath[] = RT_OVERRIDEN_FOLDER "WaterNormal_n.ktx2";

	RgInstanceCreateInfo info = {
		.pAppName = "QuakeRay",
		.pAppGUID = "8d1f551a-b0e4-4365-985c-5e1182f3c54a",

#ifdef RG_USE_SURFACE_WIN32
		.pWin32SurfaceInfo = &win32Info,
#elif RG_USE_SURFACE_XLIB
		.pXlibSurfaceCreateInfo = &x11Info,
#endif
		
		.pfnPrint = RT_PrintMessage,

		.pShaderFolderPath = pShaderPath,
		.pBlueNoiseFilePath = pBlueNoisePath,

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
	};

	RgResult r = rgCreateInstance (&info, &vulkan_globals.instance);
	RG_CHECK (r);

	RT_MAT_Init ();

	Cmd_AddCommand ("rt_pfnreloadshaders", RT_ReloadShaders);
	Cmd_AddCommand ("rt_water_color", RT_WaterColor);
	Cmd_AddCommand ("rt_water_acidcolor", RT_AcidColor);
	Cmd_AddCommand ("rt_light_report", RT_LightReport_f);
	Cmd_AddCommand ("rt_light_report_dump", RT_LightReportDump_f);
	Cmd_AddCommand ("fog", RT_Fog_Cmd);
	Cmd_AddCommand ("rt_stats", RT_Stats_f);
	Cmd_AddCommand ("rt_stats_dump", RT_StatsDump_f);
	Cvar_SetValueQuick (&_rt_firsttime, 0);


    vulkan_globals.primary_cb_context.batch_indices = Mem_Alloc (sizeof (uint32_t) * MAX_BATCH_INDICES);
	vulkan_globals.primary_cb_context.batch_verts = Mem_Alloc (sizeof (RgVertex) * MAX_BATCH_VERTS);
	vulkan_globals.primary_cb_context.batch_verts_count = 0;
	vulkan_globals.primary_cb_context.batch_indices_count = 0;
	for (int i = 0; i < CBX_NUM; i++)
	{
		vulkan_globals.secondary_cb_contexts[i].batch_indices = Mem_Alloc (sizeof (uint32_t) * MAX_BATCH_INDICES);
		vulkan_globals.secondary_cb_contexts[i].batch_verts = Mem_Alloc (sizeof (RgVertex) * MAX_BATCH_VERTS);
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
	RgStartFrameInfo info = {
		.requestVSync = CVAR_TO_BOOL (vid_vsync),
		.requestShaderReload = request_shaders_reload,
	};

	RgResult r = rgStartFrame (vulkan_globals.instance, &info);
	RG_CHECK (r);

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

static RgRenderSharpenTechnique GetSharpenTechniqueFromCvar ()
{
	int vintage = CVAR_TO_INT32 (rt_vintage);
	int t = CVAR_TO_INT32 (rt_sharpen);

	switch (t)
	{
	case 2:
		return RG_RENDER_SHARPEN_TECHNIQUE_AMD_CAS;
	case 1:
		return RG_RENDER_SHARPEN_TECHNIQUE_NAIVE;
	default:
		// to accentuate a chunky look, because of the linear (not nearest) downscale mode
		if (vintage == RT_VINTAGE_200 || vintage == RT_VINTAGE_480)
		{
			return RG_RENDER_SHARPEN_TECHNIQUE_AMD_CAS;
		}
		return RG_RENDER_SHARPEN_TECHNIQUE_NONE;
	}
}

static void UpscaleCvarsToRtgl (RgDrawFrameRenderResolutionParams *pDst)
{
	int nvDlss = CVAR_TO_INT32 (rt_upscale_dlss);
	int amdFsr = CVAR_TO_INT32 (rt_upscale_fsr2);
	int amdFsr31 = CVAR_TO_INT32 (rt_upscale_fsr31);

	switch (nvDlss)
	{
	case 1:
		// start with Quality
		pDst->upscaleTechnique = RG_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS;
		pDst->resolutionMode = RG_RENDER_RESOLUTION_MODE_QUALITY;
		break;
	case 2:
		pDst->upscaleTechnique = RG_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS;
		pDst->resolutionMode = RG_RENDER_RESOLUTION_MODE_BALANCED;
		break;
	case 3:
		pDst->upscaleTechnique = RG_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS;
		pDst->resolutionMode = RG_RENDER_RESOLUTION_MODE_PERFORMANCE;
		break;
	case 4:
		pDst->upscaleTechnique = RG_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS;
		pDst->resolutionMode = RG_RENDER_RESOLUTION_MODE_ULTRA_PERFORMANCE;
		break;

	case 5:
		// use DLSS with rt_renderscale
		pDst->upscaleTechnique = RG_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS;
		pDst->resolutionMode = RG_RENDER_RESOLUTION_MODE_CUSTOM;
		break;

	default:
		nvDlss = 0;
		break;
	}

	switch (amdFsr)
	{
	case 1:
		pDst->upscaleTechnique = RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR2;
		pDst->resolutionMode = RG_RENDER_RESOLUTION_MODE_QUALITY;
		break;
	case 2:
		pDst->upscaleTechnique = RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR2;
		pDst->resolutionMode = RG_RENDER_RESOLUTION_MODE_BALANCED;
		break;
	case 3:
		pDst->upscaleTechnique = RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR2;
		pDst->resolutionMode = RG_RENDER_RESOLUTION_MODE_PERFORMANCE;
		break;
	case 4:
		pDst->upscaleTechnique = RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR2;
		pDst->resolutionMode = RG_RENDER_RESOLUTION_MODE_ULTRA_PERFORMANCE;
		break;

	case 5:
		// use FSR2 with rt_renderscale
		pDst->upscaleTechnique = RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR2;
		pDst->resolutionMode = RG_RENDER_RESOLUTION_MODE_CUSTOM;
		break;

	default:
		amdFsr = 0;
		break;
	}

	switch (amdFsr31)
	{
	case 1:
		pDst->upscaleTechnique = RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3;
		pDst->resolutionMode = RG_RENDER_RESOLUTION_MODE_NATIVE_AA;
		break;
	case 2:
		pDst->upscaleTechnique = RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3;
		pDst->resolutionMode = RG_RENDER_RESOLUTION_MODE_QUALITY;
		break;
	case 3:
		pDst->upscaleTechnique = RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3;
		pDst->resolutionMode = RG_RENDER_RESOLUTION_MODE_BALANCED;
		break;
	case 4:
		pDst->upscaleTechnique = RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3;
		pDst->resolutionMode = RG_RENDER_RESOLUTION_MODE_PERFORMANCE;
		break;
	case 5:
		pDst->upscaleTechnique = RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3;
		pDst->resolutionMode = RG_RENDER_RESOLUTION_MODE_ULTRA_PERFORMANCE;
		break;

	default:
		amdFsr31 = 0;
		break;
	}

	// both disabled
	if (nvDlss == 0 && amdFsr == 0 && amdFsr31 == 0)
	{
		pDst->upscaleTechnique = RG_RENDER_UPSCALE_TECHNIQUE_NEAREST;
		pDst->resolutionMode = RG_RENDER_RESOLUTION_MODE_CUSTOM;
	}

	if (amdFsr)
	{
		pDst->sharpenTechnique = RG_RENDER_SHARPEN_TECHNIQUE_AMD_CAS;
	}
	else
	{
		pDst->sharpenTechnique = GetSharpenTechniqueFromCvar ();
	}
}

static const char *GetUpscalerOptionName (int i, RgRenderUpscaleTechnique technique)
{
	if (!rgIsRenderUpscaleTechniqueAvailable (vulkan_globals.instance, technique))
	{
		return "Not Available";
	}

    switch (i)
    {
	case 0:
		return "Off";
    case 1:
		return (technique == RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3) ? "Native AA" : "Quality";
	case 2:
		return (technique == RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3) ? "Quality" : "Balanced";
	case 3:
		return (technique == RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3) ? "Balanced" : "Performance";
	case 4:
		return (technique == RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3) ? "Performance" : "Ultra Performance";
	case 5:
		return "Ultra Performance";
	default:
		return "Custom";
    }
}

static const char* GetVintageOptionName(int vintage)
{
	switch (vintage)
	{
	case RT_VINTAGE_OFF:
		return "Off";
	case RT_VINTAGE_CRT:
		return "CRT";
	case RT_VINTAGE_200:
		return "320x200";
	case RT_VINTAGE_480:
		return "640x480";
	case RT_VINTAGE_720:
		return "1024x768";
	default:
		return "Custom";
	}
}

typedef struct end_rendering_parms_s
{
	float   vid_width;
	float   vid_height;
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
extern RgMediaType rt_cameramedia;
extern qboolean rt_lavaeffects;

static void ResolutionToRtgl (RgDrawFrameRenderResolutionParams *dst, const RgExtent2D winsize, RgExtent2D *storage)
{
	const float aspect = (float)winsize.width / (float)winsize.height;

	if (CVAR_TO_INT32 (rt_renderscale) > 0)
	{
		float scale = (float)CVAR_TO_INT32 (rt_renderscale) / 100.0f;
		scale = CLAMP (scale, 0.2f, 1.0f);

		dst->customRenderSize.width = (uint32_t)(scale * winsize.width);
		dst->customRenderSize.height = (uint32_t)(scale * winsize.height);
		dst->pPixelizedRenderSize = NULL;

		return;
	}
	else
	{
		if (CVAR_TO_INT32 (rt_vintage) != RT_VINTAGE_OFF)
		{
			uint32_t h_pixelized = 0;
			uint32_t h_render = 0;

			switch (CVAR_TO_INT32 (rt_vintage))
			{
			case RT_VINTAGE_200:
				h_pixelized = 200;
				h_render = 400;
				break;

			case RT_VINTAGE_480:
				h_pixelized = 480;
				h_render = 600;
				break;

			case RT_VINTAGE_CRT:
				h_pixelized = 480;
				h_render = 480;
				break;

			case RT_VINTAGE_720:
				h_pixelized = 720;
				h_render = 720;
				break;

			default:
				Cvar_SetValueQuick (&rt_vintage, 0);
				dst->customRenderSize = winsize;
				dst->pPixelizedRenderSize = NULL;
				return;
			}

			assert (h_render > 0 && h_pixelized > 0);
			
			storage->height = h_pixelized;
			storage->width = (uint32_t)(h_pixelized * aspect);
			dst->pPixelizedRenderSize = storage;
			dst->customRenderSize.height = h_render;
			dst->customRenderSize.width = (uint32_t)(h_render * aspect);

			return;
		}
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
	RgExtent2D       pixstorage = {0};
	const RgExtent2D winsize = {.width = parms->vid_width, .height = parms->vid_height};

	RgDrawFrameRenderResolutionParams resolution_params = {0};
	ResolutionToRtgl (&resolution_params, winsize, &pixstorage);
	UpscaleCvarsToRtgl (&resolution_params);

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

	RgDrawFrameIlluminationParams illum_params = {
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
		.giBounceRays = gi_level,
		// Q2RTX pt_sun_bounce_range / sun_bounce: how far the sun reaches into an
		// indirect bounce (game units, 0 turns indirect sunlight off) and a
		// multiplier on what it delivers there. Both only affect the indirect
		// pass, and the shadow ray a bounce casts for the sun is skipped once the
		// distance falloff is zero.
		.sunBounceRange = CVAR_TO_FLOAT (rt_sun_bounce_range),
		.sunBounceScale = CVAR_TO_FLOAT (rt_sun_bounce_scale),
		.denoiserEnabled = CVAR_TO_BOOL (rt_denoiser),
		// Q2RTX writes 2 through its "textures" toggle (flt_fixed_albedo ~1),
		// so the unchecked state uses the same flat albedo value.
		.fixedAlbedo = CVAR_TO_BOOL (rt_no_textures) ? 2.0f : 0.0f,
		.lightUniqueIdIgnoreFirstPersonViewerShadows = NULL,
	};

	RgDrawFrameBloomParams bloom_params = {
		.bloomIntensity = !CVAR_TO_BOOL (rt_bloom) ? 0 : CVAR_TO_FLOAT (rt_bloom_intensity),
		.inputThreshold = 0.0f,
		.bloomEmissionMultiplier = CVAR_TO_FLOAT (rt_bloom_emis_mult),
	};

	// Exposure bias is in EV and darkens the image when negative; contrast blends
	// the adaptive tone curve with Reinhard (Q2RTX tm_exposure_bias / tm_reinhard).
	RgDrawFrameTonemappingParams tonemap_params = {
		.minLogLuminance = -3.9f,
		.maxLogLuminance = -2.8f,
		.luminanceWhitePoint = 10.0f,
		.exposureBias = CLAMP (-5.0f, CVAR_TO_FLOAT (rt_exposure_bias), 0.0f),
		.contrast = CLAMP (0.0f, CVAR_TO_FLOAT (rt_contrast), 1.0f),
	};

	RgDrawFrameReflectRefractParams refl_refr_params = {
		.maxReflectRefractDepth = CVAR_TO_UINT32 (rt_reflrefr_depth),
		.typeOfMediaAroundCamera = rt_cameramedia,
		.indexOfRefractionGlass = CVAR_TO_FLOAT (rt_refr_glass),
		.indexOfRefractionWater = CVAR_TO_FLOAT (rt_refr_water),
		.waterWaveSpeed = METRIC_TO_QUAKEUNIT (CVAR_TO_FLOAT (rt_water_speed)),
		.waterWaveNormalStrength = CVAR_TO_FLOAT (rt_water_normstren),
		.turbWarpStrength = CVAR_TO_FLOAT (rt_turb_warp),
		.waterColor = RT_VEC3 (rt_water_color),
		.acidColor = RT_VEC3 (rt_acid_color),
		.acidDensity = CVAR_TO_FLOAT(rt_water_aciddensity),
		.waterWaveTextureDerivativesMultiplier = CVAR_TO_FLOAT (rt_water_normsharp),
		.waterTextureAreaScale = METRIC_TO_QUAKEUNIT (CVAR_TO_FLOAT (rt_water_scale)),
		.portalNormalTwirl = CVAR_TO_BOOL (rt_portal_twirl),
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

	const float skyBrightness = CVAR_TO_FLOAT (rt_sky_brightness) * CVAR_TO_FLOAT (rt_brightness);

	const qboolean materials_only = CVAR_TO_BOOL (rt_materials_only);

	const int usePhysicalSky = CVAR_TO_BOOL (rt_physical_sky) != 0;

	vec3_t sky_base_color;

	if (usePhysicalSky)
	{
		RT_GetSkyColor (sky_base_color);
	}
	else
	{
		VectorCopy (skyflatcolor, sky_base_color);
	}
	VectorScale (sky_base_color, skyBrightness, sky_base_color);
	RT_APPLY_SKY_COLOR (sky_base_color);

	if (materials_only)
	{
		sky_base_color[0] = sky_base_color[1] = sky_base_color[2] = 0.0f;
	}

	RgDrawFrameSkyParams sky_params = {
		.skyType = CVAR_TO_BOOL (r_fastsky) ? RG_SKY_TYPE_COLOR
		         : usePhysicalSky ? RG_SKY_TYPE_PROCEDURAL
		         : RG_SKY_TYPE_RASTERIZED_GEOMETRY,
		.skyColorDefault = RT_VEC3 (sky_base_color),
		.skyColorMultiplier = materials_only ? 0.0f : (usePhysicalSky ? skyBrightness : skyMult * skyBrightness),
		.skyColorSaturation = CVAR_TO_FLOAT (rt_sky_tint),
		.skyAmbientLod = CVAR_TO_FLOAT (rt_sky_ambient_lod),
		.skyNee = CVAR_TO_FLOAT (rt_sky_nee) > 0.0f,
		.skyViewerPosition = RT_VEC3 (r_origin),
		.godRaysEnabled = CVAR_TO_BOOL (rt_godrays),
		.godRaysIntensity = CVAR_TO_FLOAT (gr_intensity),
	};

	if (usePhysicalSky)
	{
		float *c = &sky_params.skyCubemapRotationTransform.matrix[0][0];
		RT_GetSkyCloudsColor (c);
		c[3] = CVAR_TO_FLOAT (rt_sky_cloud_coverage);
		c[4] = CVAR_TO_FLOAT (rt_sky_cloud_density);
		c[5] = CVAR_TO_FLOAT (rt_sky_cloud_speed);
		c[6] = CVAR_TO_BOOL (rt_sky_clouds) ? 1.0f : 0.0f;
		c[7] = c[8] = 0.0f;
	}

	// The volume light is the sky seen from the sun's direction, so it is
	// sunlight: with rt_sun 0 there is nothing to scatter and only the ambient
	// term below still lights the fog. The legacy rt_volume_lpitch/lyaw +
	// skyflatcolor fallback that used to run here is gone: its colour was the
	// sky texture's average rather than rt_sky_color, which made the shafts an
	// order of magnitude brighter with the sun off.
	vec3_t volume_light_angles;
	vec3_t volume_light_color;

	RT_VEC3_SET (volume_light_angles, CVAR_TO_FLOAT (rt_sun_pitch), CVAR_TO_FLOAT (rt_sun_yaw), 0);

	if (CVAR_TO_BOOL (rt_sun))
		RT_GetSkyColor (volume_light_color);
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

	RgDrawFrameVolumetricParams volumetric_params = {
		.enable = !materials_only && CVAR_TO_UINT32 (rt_volume_type) != 0,
		.useSimpleDepthBased = CVAR_TO_UINT32 (rt_volume_type) == 1,
		.volumetricFar = CVAR_TO_FLOAT (rt_volume_far),
		.ambientColor = RT_VEC3 (volume_ambient_color),
		.scaterring = materials_only ? 0.0f : CVAR_TO_FLOAT (rt_volume_scatter),
		.sourceColor = RT_VEC3 (volume_light_color),
		.sourceDirection = RT_AnglesToDir (volume_light_angles),
		.sourceAssymetry = CVAR_TO_FLOAT (rt_volume_lassymetry),
	};

	RgDrawFrameTexturesParams texture_params = {
		.dynamicSamplerFilter = CVAR_TO_INT32 (vid_filter) == 1 ? RG_SAMPLER_FILTER_NEAREST : RG_SAMPLER_FILTER_LINEAR,
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

	RgDrawFrameLensFlareParams lens_flare_params = {
		.lensFlareBlendFuncSrc = RG_BLEND_FACTOR_SRC_ALPHA,
		.lensFlareBlendFuncDst = RG_BLEND_FACTOR_ONE,
	};

	// Classic level fog: the worldspawn "fog" key and the `fog` console
	// command, which Arcane Dimensions also uses to drive its dynamic fog. The
	// color is passed as it is, without an sRGB decoding, the same way the
	// classic renderer blended it into the framebuffer and the same way
	// Sky_DrawSky above hands it to the sky. The density is divided by the 64
	// the classic renderer scaled it with, so that a density of 0.05, a
	// mid-range value for the shipped maps, fades the far plane into the fog
	// instead of everything. rt_level_fog 0 ignores the level's fog.
	float level_fog_color[4];
	Fog_GetColor (level_fog_color);

	const qboolean level_fog_active = CVAR_TO_BOOL (rt_level_fog) && Fog_GetDensity () > 0;

	RgDrawFrameLevelFogParams level_fog_params = {
		.color = RT_VEC3 (level_fog_color),
		.density = (level_fog_active && !materials_only) ? Fog_GetDensity () / 64.0f : 0.0f,
		.skyBlend = (level_fog_active && !materials_only) ? skyfog : 0.0f,
	};

	RgPostEffectCRT crt_effect = {
		.isActive = CVAR_TO_BOOL (rt_ef_crt) || CVAR_TO_INT32 (rt_vintage) == RT_VINTAGE_CRT,
	};

	RgPostEffectChromaticAberration chromatic_aberration_effect = {
		.isActive = CVAR_TO_FLOAT (rt_ef_chraber) > 0.0f,
		.transitionDurationIn = 0,
		.transitionDurationOut = 0,
		.intensity = CVAR_TO_FLOAT (rt_ef_chraber),
	};

	RgPostEffectColorTint tint_quad = {
		.isActive = true,
		.transitionDurationIn = 1.0f,
		.transitionDurationOut = 1.0f,
		.intensity = 4.0f,
		.color = {0.25f, 0.0f, 1.0f},
	};
	RgPostEffectColorTint tint_invuln = {
		.isActive = true,
		.transitionDurationIn = 1.0f,
		.transitionDurationOut = 1.0f,
		.intensity = 4.0f,
		.color = {1.0f, 0.0f, 0.0f},
	};
	RgPostEffectColorTint tint_lava = {
		.isActive = true,
		.transitionDurationIn = 0.05f,
		.transitionDurationOut = 0.5f,
		.intensity = 10.0f,
		.color = {1.0f, 0.1f, 0.0f},
	};
	RgPostEffectColorTint tint_radsuit = {
		.isActive = true,
		.transitionDurationIn = 1.0f,
		.transitionDurationOut = 1.0f,
		.intensity = 1.0f,
		.color = {0.2f, 1.0f, 0.4f},
	};
	RgPostEffectColorTint tint_bonus = {
		.isActive = true,
		.transitionDurationIn = 0.0f,
		.transitionDurationOut = 0.7f,
		.intensity = 0.5f,
		.color = {0.85f, 0.72f, 0.27f},
	};
	RgPostEffectColorTint tint_damage = {
		.isActive = true,
		.transitionDurationIn = 0.0f,
		.transitionDurationOut = 0.2f + rt_dmg_value * 0.8f,
		.intensity = 1.0f,
		.color = FROMCOLOR255 (cl.cshifts[CSHIFT_DAMAGE].destcolor),
	};

	static RgPostEffectColorTint tint_effect = {0}; // static, so prev state's transition durations are preserved
	tint_effect.isActive = false;
	if (cl.stats[STAT_HEALTH] > 0)
	{
	    if (cl.items & IT_QUAD) tint_effect = tint_quad;
	    else if (cl.items & IT_INVULNERABILITY) tint_effect = tint_invuln;
	    else if (rt_lavaeffects) tint_effect = tint_lava;
	    else if (cl.items & IT_SUIT) tint_effect = tint_radsuit;
	    else if (rt_dmg_inthisframe) tint_effect = tint_damage;
	    else if (cl.cshifts[CSHIFT_BONUS].percent > 0) tint_effect = tint_bonus;
	}
	rt_dmg_inthisframe = false;

    RgPostEffectRadialBlur radial_effect = {
		.isActive = (cl.items & (IT_QUAD | IT_INVULNERABILITY)) && cl.stats[STAT_HEALTH] > 0,
		.transitionDurationIn = 1.0f,
		.transitionDurationOut = 2.0f,
	};

	RgPostEffectWaves waves_effect = {
		.isActive = rt_cameramedia != RG_MEDIA_TYPE_VACUUM,
		.transitionDurationIn = 0.1f,
		.transitionDurationOut = 0.75f,
		.amplitude = CVAR_TO_FLOAT (rt_ef_waves_stren) * 0.01f,
		.speed = 1.0f,
		.xMultiplier = 0.5f,
	};

	RgDrawFrameDebugParams debug_params = {
		.drawFlags = CVAR_TO_UINT32 (rt_debugflags),
	};
	debug_params.drawFlags |= RG_DEBUG_DRAW_Q2RTX_CORE_BIT;
	if (RT_StatsPanel (RT_STATS_RAYS))
	{
		debug_params.drawFlags |= RG_DEBUG_DRAW_STATS_BIT;
	}
	if (RT_StatsPanel (RT_STATS_PASSES))
	{
		debug_params.drawFlags |= RG_DEBUG_DRAW_PASS_STATS_BIT;
	}

	float cameranear = GL_GetCameraNear (DEG2RAD (r_fovx), DEG2RAD (r_fovy));
	float camerafar = GL_GetCameraFar ();

	RgDrawFrameInfo info = {
		.worldUpVector = {0, 0, 1},
		.fovYRadians = DEG2RAD (r_fovy),
		.cameraNear = cameranear,
		.cameraFar = camerafar,
		.rayLength = 10000.0f,
		.rayCullMaskWorld = RG_DRAW_FRAME_RAY_CULL_WORLD_0_BIT | RG_DRAW_FRAME_RAY_CULL_WORLD_1_BIT | RG_DRAW_FRAME_RAY_CULL_SKY_BIT,
		.disableRayTracedGeometry = false,
		.disableRasterization = false,
		.currentTime = (double)SDL_GetTicks () / 1000.0,
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
		.pLensFlareParams = &lens_flare_params,
		.pLevelFogParams = &level_fog_params,
		.postEffectParams =
			{
				.pChromaticAberration = &chromatic_aberration_effect,
				.pWaves = CVAR_TO_INT32(r_waterwarp) == 1 ? &waves_effect : NULL,
				.pColorTint = cl.intermission ? NULL : &tint_effect ,
				.pCRT = &crt_effect,
				.pRadialBlur = cl.intermission ? NULL : &radial_effect,
			},
		.pDebugParams = &debug_params,
	};
	memcpy (info.view, vulkan_globals.view_matrix, 16 * sizeof(float));

	double prof_start = RT_Prof_Begin ();
	RgResult r = rgDrawFrame (vulkan_globals.instance, &info);
	RT_Prof_End (RT_PROF_DRAWFRAME, prof_start);
	RG_CHECK (r);
}

/*
=================
GL_EndRendering
=================
*/
task_handle_t GL_EndRendering (qboolean use_tasks, qboolean swapchain)
{
	end_rendering_parms_t parms = {
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
		if (vulkan_globals.instance != RG_NULL_HANDLE)
		{
		    RT_MAT_Shutdown ();
		    RgResult r = rgDestroyInstance (vulkan_globals.instance);
			RG_CHECK (r);

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

static void RT_SunPreset_f (cvar_t *var)
{
	const int preset = CLAMP (0, CVAR_TO_INT32 (rt_sun_preset), 7);

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

	Cvar_Set ("rt_sky_color", va ("%d %d %d", presets[preset][0], presets[preset][1], presets[preset][2]));
}

extern atomic_uint32_t rt_require_static_submit;

static void RT_LightStylesChanged_f (cvar_t *var)
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

static RgFogVolume rt_fog_volumes[RG_MAX_FOG_VOLUMES];

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

static void RT_Fog_PrintVolume (int index, const RgFogVolume *vol)
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
	RgFogVolume *vol   = NULL;

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
			if (index < 0 || index >= RG_MAX_FOG_VOLUMES)
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

	rgSetFogVolumes (vulkan_globals.instance, RG_MAX_FOG_VOLUMES, rt_fog_volumes);
	return;

no_volume:
	Con_Printf ("volume not specified\n");
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

		Cvar_RegisterVariable (&rt_light_report_filter);

		// The colour settings are read per light, so they watch their cvar instead of
		// comparing its string on every read. Registered after the cvars, which is
		// where VID_Init has them.
		for (size_t i = 0; i < countof (rt_colors); i++)
		{
			Cvar_SetCallback (rt_colors[i].cvar, RT_ColorChanged_f);
		}
	}

	Cvar_SetCallback (&rt_sun_preset, RT_SunPreset_f);
	Cvar_SetCallback (&rt_light_styles, RT_LightStylesChanged_f);
	Cvar_SetCallback (&rt_light_styles_reach, RT_LightStylesChanged_f);
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
	vid.conwidth = (scr_conwidth.value > 0) ? (int)scr_conwidth.value : (scr_conscale.value > 0) ? (int)(vid.width / scr_conscale.value) : vid.width;
	vid.conwidth = CLAMP (320, vid.conwidth, vid.width);
	vid.conwidth &= 0xFFFFFFF8;
	vid.conheight = vid.conwidth * vid.height / vid.width;
	//
	// keep cvars in line with actual mode
	//
	VID_SyncCvars ();

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
#define UPSCALER_FSR2 1
#define UPSCALER_FSR31 2
#define UPSCALER_DLSS 3

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
	int rt_vintage;
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

	menu_settings.host_maxfps = CLAMP (0, host_maxfps.value, 1000);
	menu_settings.r_particles = CLAMP (0, (int)r_particles.value, 2);
	{
		int fsr2 = CVAR_TO_INT32 (rt_upscale_fsr2);
		int fsr31 = CVAR_TO_INT32 (rt_upscale_fsr31);
		int dlss = CVAR_TO_INT32 (rt_upscale_dlss);
		if (fsr31 > 0)      { menu_settings.upscaler_type = UPSCALER_FSR31; menu_settings.upscaler_quality = CLAMP (0, fsr31, 5); }
		else if (fsr2 > 0)  { menu_settings.upscaler_type = UPSCALER_FSR2;  menu_settings.upscaler_quality = CLAMP (0, fsr2, 4); }
		else if (dlss > 0)  { menu_settings.upscaler_type = UPSCALER_DLSS;  menu_settings.upscaler_quality = CLAMP (0, dlss, 4); }
		else                { menu_settings.upscaler_type = UPSCALER_OFF;   menu_settings.upscaler_quality = 0; }
	}
	menu_settings.rt_vintage = CLAMP (0, CVAR_TO_INT32 (rt_vintage), RT_VINTAGE__COUNT - 1);
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
	// VID_OPT_REFRESHRATE,
	VID_OPT_APPLY,


	VID_OPT_BLOOM,
	VID_OPT_EXPOSURE_BIAS,
	VID_OPT_CONTRAST,
	VID_OPT_VSYNC,
	VID_OPT_MAX_FPS,


	VID_OPT_UPSCALER,
	VID_OPT_UPSCALER_QUALITY,

	VID_OPT_FILTER,
	VID_OPT_PARTICLES,
	VID_OPT_VOLUMETRICS,
	VID_OPT_MATERIALS_ONLY,
	VID_OPT_RENDER_SCALE,

	VID_OPT_NEXT_PAGE, // last row of the first page


	// second page
	VID_OPT_FOV,
	VID_OPT_SHOWFPS,


	VID_OPT_GI_LEVEL,
	VID_OPT_GODRAYS,
	VID_OPT_REFLECT,
	VID_OPT_DENOISER,
	VID_OPT_TEXTURES,

	VID_OPT_BACK,

	VIDEO_OPTIONS_ITEMS
};

// The menu canvas is a fixed 640x200 without scrolling, so the rows are split
// into two pages, in enum order.
#define VID_OPT_PAGE_COUNT 2

static int VID_MenuRowPage (int vidopt)
{
	return (vidopt <= VID_OPT_NEXT_PAGE) ? 0 : 1;
}

static int video_options_cursor = 0;
static int video_options_page = 0;

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
	int i;

	if (vid_menu_nummodes)
	{
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
VID_Menu_ChooseNextParticles
================
*/
static void VID_Menu_ChooseNextParticles (int dir)
{
	if (dir > 0)
	{
		if (menu_settings.r_particles == 0)
			menu_settings.r_particles = 2;
		else if (menu_settings.r_particles == 2)
			menu_settings.r_particles = 1;
		else
			menu_settings.r_particles = 0;
	}
	else
	{
		if (menu_settings.r_particles == 0)
			menu_settings.r_particles = 1;
		else if (menu_settings.r_particles == 2)
			menu_settings.r_particles = 0;
		else
			menu_settings.r_particles = 2;
	}
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
	RgBool32 fsr2_ok = rgIsRenderUpscaleTechniqueAvailable (vulkan_globals.instance, RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR2);
	RgBool32 fsr31_ok = rgIsRenderUpscaleTechniqueAvailable (vulkan_globals.instance, RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3);
	RgBool32 dlss_ok = rgIsRenderUpscaleTechniqueAvailable (vulkan_globals.instance, RG_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS);

	const int prev_type = menu_settings.upscaler_type;
	const int prev_quality = menu_settings.upscaler_quality;
	const int prev_vint = menu_settings.rt_vintage;
	const int maxq_fsr2 = fsr2_ok ? 4 : 0;
	const int maxq_fsr31 = fsr31_ok ? 5 : 0;
	const int maxq_dlss = dlss_ok ? 4 : 0;

	if (vidopt == VID_OPT_UPSCALER)
	{
		do {
			menu_settings.upscaler_type += dir < 0 ? -1 : 1;
			if (menu_settings.upscaler_type < 0) menu_settings.upscaler_type = UPSCALER_DLSS;
			if (menu_settings.upscaler_type > UPSCALER_DLSS) menu_settings.upscaler_type = UPSCALER_OFF;
		} while (
			(menu_settings.upscaler_type == UPSCALER_FSR2  && !fsr2_ok) ||
			(menu_settings.upscaler_type == UPSCALER_FSR31 && !fsr31_ok) ||
			(menu_settings.upscaler_type == UPSCALER_DLSS  && !dlss_ok));

		if (menu_settings.upscaler_type != prev_type)
		{
			menu_settings.upscaler_quality = (menu_settings.upscaler_type == UPSCALER_OFF) ? 0 : GetUpscalerDefaultQuality (menu_settings.upscaler_type);
			menu_settings.rt_vintage = 0;
		}
	}
	else if (vidopt == VID_OPT_UPSCALER_QUALITY)
	{
		int maxq;
		switch (menu_settings.upscaler_type)
		{
		case UPSCALER_FSR2:  maxq = maxq_fsr2; break;
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
	else if (vidopt == VID_OPT_RENDER_SCALE)
	{
		menu_settings.rt_vintage += dir < 0 ? -1 : 1;
	}

	menu_settings.rt_vintage = CLAMP (0, menu_settings.rt_vintage, RT_VINTAGE__COUNT - 1);

	if (vidopt == VID_OPT_UPSCALER || vidopt == VID_OPT_UPSCALER_QUALITY)
	{
		if (menu_settings.upscaler_type != prev_type || menu_settings.upscaler_quality != prev_quality)
			menu_settings.rt_vintage = 0;
	}
	else if (vidopt == VID_OPT_RENDER_SCALE)
	{
		if (menu_settings.rt_vintage != prev_vint)
		{
			menu_settings.upscaler_type = UPSCALER_OFF;
			menu_settings.upscaler_quality = 0;
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

/*
================
VID_Menu_GetGiLevelName -- Q2RTX pt_num_bounce_rays as a word
================
*/
static const char *VID_Menu_GetGiLevelName (void)
{
	const float v = CVAR_TO_FLOAT (rt_gi_level);

	if (v < 0.25f)
		return "off";
	if (v < 0.75f)
		return "low";
	if (v < 1.5f)
		return "medium";

	return "high";
}

/*
================
VID_Menu_StepGiLevel -- cycle through the Q2RTX gi levels
================
*/
static void VID_Menu_StepGiLevel (float dir)
{
	static const float levels[] = { 0.0f, 0.5f, 1.0f, 2.0f };
	const int numlevels = (int)(sizeof (levels) / sizeof (levels[0]));
	const float cur = CVAR_TO_FLOAT (rt_gi_level);

	int   idx = 2; // medium
	float best = 1e9f;

	for (int i = 0; i < numlevels; i++)
	{
		const float d = fabsf (levels[i] - cur);

		if (d < best)
		{
			best = d;
			idx = i;
		}
	}

	idx = CLAMP (0, idx + ((dir > 0.0f) ? 1 : -1), numlevels - 1);

	Cvar_SetValueQuick (&rt_gi_level, levels[idx]);
}

/*
================
VID_Menu_StepReflDepth -- cycle through the Q2RTX reflection depths
================
*/
static void VID_Menu_StepReflDepth (float dir)
{
	static const float depths[] = { 0.0f, 1.0f, 2.0f, 4.0f, 8.0f };
	const int numdepths = (int)(sizeof (depths) / sizeof (depths[0]));
	const float cur = CVAR_TO_FLOAT (rt_reflrefr_depth);

	int   idx = 2; // 2 bounces
	float best = 1e9f;

	for (int i = 0; i < numdepths; i++)
	{
		const float d = fabsf (depths[i] - cur);

		if (d < best)
		{
			best = d;
			idx = i;
		}
	}

	idx = CLAMP (0, idx + ((dir > 0.0f) ? 1 : -1), numdepths - 1);

	Cvar_SetValueQuick (&rt_reflrefr_depth, depths[idx]);
}

/*
================
VID_Menu_SetPage -- switch page, keeping the cursor on a visible row
================
*/
static void VID_Menu_SetPage (int page)
{
	int i;

	video_options_page = CLAMP (0, page, VID_OPT_PAGE_COUNT - 1);

	for (i = 0; i < VIDEO_OPTIONS_ITEMS; i++)
	{
		if (VID_MenuRowPage (i) == video_options_page)
		{
			video_options_cursor = i;
			break;
		}
	}
}

/*
================
VID_MenuKey
================
*/
static void VID_MenuKey (int key)
{
	{
		qboolean leave = false;

		if (key == K_ESCAPE || key == K_BACKSPACE)
		{
			leave = true;
		}

		if (key == K_ENTER || key == K_KP_ENTER)
		{
			if (video_options_cursor == VID_OPT_BACK)
			{
				m_entersound = true;
				leave = true;
			}
		}

		if (leave)
		{
			VID_SyncCvars (); // sync cvars before leaving menu. FIXME: there are other ways to leave menu
			S_LocalSound ("misc/menu1.wav");
			M_Menu_Options_f ();

			return;
		}
	}

	switch (key)
	{
	case K_UPARROW:
		S_LocalSound ("misc/menu1.wav");
		do
		{
			video_options_cursor--;
			if (video_options_cursor < 0)
				video_options_cursor = VIDEO_OPTIONS_ITEMS - 1;
		} while (VID_MenuRowPage (video_options_cursor) != video_options_page);
		break;

	case K_DOWNARROW:
		S_LocalSound ("misc/menu1.wav");
		do
		{
			video_options_cursor++;
			if (video_options_cursor >= VIDEO_OPTIONS_ITEMS)
				video_options_cursor = 0;
		} while (VID_MenuRowPage (video_options_cursor) != video_options_page);
		break;

	case K_LEFTARROW:
		S_LocalSound ("misc/menu3.wav");
		switch (video_options_cursor)
		{
		case VID_OPT_MODE:
			VID_Menu_ChooseNextMode (1);
			break;
		//case VID_OPT_REFRESHRATE:
		//	VID_Menu_ChooseNextRate (1);
		//	break;
		case VID_OPT_VSYNC:
			Cbuf_AddText ("toggle vid_vsync\n"); // kristian
			break;
		case VID_OPT_MAX_FPS:
			VID_Menu_ChooseNextMaxFPS (-1);
			Cvar_SetValueQuick (&host_maxfps, menu_settings.host_maxfps);
			break;
		case VID_OPT_BLOOM:
			Cvar_SetValueQuick (&rt_bloom, !CVAR_TO_BOOL (rt_bloom));
			break;
		case VID_OPT_EXPOSURE_BIAS:
			VID_Menu_StepFloatCvar (&rt_exposure_bias, -0.1f, -5.0f, 0.0f);
			break;
		case VID_OPT_CONTRAST:
			VID_Menu_StepFloatCvar (&rt_contrast, -0.1f, 0.0f, 1.0f);
			break;
		case VID_OPT_UPSCALER:
		case VID_OPT_UPSCALER_QUALITY:
		case VID_OPT_RENDER_SCALE:
			VID_Menu_ChooseNextAA (video_options_cursor, -1);
			{
				int q = menu_settings.upscaler_quality;
				if (menu_settings.upscaler_type != UPSCALER_OFF && q < 1)
					q = GetUpscalerDefaultQuality (menu_settings.upscaler_type);
				Cvar_SetValueQuick (&rt_upscale_fsr2, (menu_settings.upscaler_type == UPSCALER_FSR2) ? q : 0);
				Cvar_SetValueQuick (&rt_upscale_fsr31, (menu_settings.upscaler_type == UPSCALER_FSR31) ? q : 0);
				Cvar_SetValueQuick (&rt_upscale_dlss, (menu_settings.upscaler_type == UPSCALER_DLSS) ? q : 0);
			}
			Cvar_SetValueQuick (&rt_vintage, menu_settings.rt_vintage);
			break;
		case VID_OPT_FILTER:
			menu_settings.vid_filter = (menu_settings.vid_filter == 0) ? 1 : 0;
			Cvar_SetValueQuick (&vid_filter, menu_settings.vid_filter);
			break;
		case VID_OPT_PARTICLES:
			VID_Menu_ChooseNextParticles (-1);
			Cvar_SetValueQuick (&r_particles, menu_settings.r_particles);
			break;
		case VID_OPT_VOLUMETRICS:
			int newval = (CVAR_TO_UINT32 (rt_volume_type) + 2) % 3;
			Cvar_SetValueQuick (&rt_volume_type, newval);
			break;
		case VID_OPT_MATERIALS_ONLY:
			Cvar_SetValueQuick (&rt_materials_only, !CVAR_TO_BOOL (rt_materials_only));
			break;
		case VID_OPT_NEXT_PAGE:
			VID_Menu_SetPage (1);
			break;
		case VID_OPT_BACK:
			VID_Menu_SetPage (0);
			break;
		case VID_OPT_FOV:
			VID_Menu_StepFloatCvar (&scr_fov, -5.0f, 60.0f, 140.0f);
			break;
		case VID_OPT_SHOWFPS:
			Cvar_SetValueQuick (&scr_showfps, !CVAR_TO_BOOL (scr_showfps));
			break;
		case VID_OPT_GI_LEVEL:
			VID_Menu_StepGiLevel (-1.0f);
			break;
		case VID_OPT_GODRAYS:
			Cvar_SetValueQuick (&rt_godrays, !CVAR_TO_BOOL (rt_godrays));
			break;
		case VID_OPT_REFLECT:
			VID_Menu_StepReflDepth (-1.0f);
			break;
		case VID_OPT_DENOISER:
			Cvar_SetValueQuick (&rt_denoiser, !CVAR_TO_BOOL (rt_denoiser));
			break;
		case VID_OPT_TEXTURES:
			Cvar_SetValueQuick (&rt_no_textures, !CVAR_TO_BOOL (rt_no_textures));
			break;
		default:
			break;
		}
		break;

	case K_RIGHTARROW:
		S_LocalSound ("misc/menu3.wav");
		switch (video_options_cursor)
		{
		case VID_OPT_MODE:
			VID_Menu_ChooseNextMode (-1);
			break;
		//case VID_OPT_REFRESHRATE:
		//	VID_Menu_ChooseNextRate (-1);
		//	break;
		case VID_OPT_VSYNC:
			Cbuf_AddText ("toggle vid_vsync\n");
			break;
		case VID_OPT_MAX_FPS:
			VID_Menu_ChooseNextMaxFPS (1);
			Cvar_SetValueQuick (&host_maxfps, menu_settings.host_maxfps);
			break;
		case VID_OPT_BLOOM:
			Cvar_SetValueQuick (&rt_bloom, !CVAR_TO_BOOL (rt_bloom));
			break;
		case VID_OPT_EXPOSURE_BIAS:
			VID_Menu_StepFloatCvar (&rt_exposure_bias, 0.1f, -5.0f, 0.0f);
			break;
		case VID_OPT_CONTRAST:
			VID_Menu_StepFloatCvar (&rt_contrast, 0.1f, 0.0f, 1.0f);
			break;
		case VID_OPT_UPSCALER:
		case VID_OPT_UPSCALER_QUALITY:
		case VID_OPT_RENDER_SCALE:
			VID_Menu_ChooseNextAA (video_options_cursor, 1);
			{
				int q = menu_settings.upscaler_quality;
				if (menu_settings.upscaler_type != UPSCALER_OFF && q < 1)
					q = GetUpscalerDefaultQuality (menu_settings.upscaler_type);
				Cvar_SetValueQuick (&rt_upscale_fsr2, (menu_settings.upscaler_type == UPSCALER_FSR2) ? q : 0);
				Cvar_SetValueQuick (&rt_upscale_fsr31, (menu_settings.upscaler_type == UPSCALER_FSR31) ? q : 0);
				Cvar_SetValueQuick (&rt_upscale_dlss, (menu_settings.upscaler_type == UPSCALER_DLSS) ? q : 0);
			}
			Cvar_SetValueQuick (&rt_vintage, menu_settings.rt_vintage);
			break;
		case VID_OPT_FILTER:
			menu_settings.vid_filter = (menu_settings.vid_filter == 0) ? 1 : 0;
			Cvar_SetValueQuick (&vid_filter, menu_settings.vid_filter);
			break;
		case VID_OPT_PARTICLES:
			VID_Menu_ChooseNextParticles (1);
			Cvar_SetValueQuick (&r_particles, menu_settings.r_particles);
			break;
		case VID_OPT_VOLUMETRICS:
			int newval = (CVAR_TO_UINT32 (rt_volume_type) + 1) % 3;
			Cvar_SetValueQuick (&rt_volume_type, newval);
			break;
		case VID_OPT_MATERIALS_ONLY:
			Cvar_SetValueQuick (&rt_materials_only, !CVAR_TO_BOOL (rt_materials_only));
			break;
		case VID_OPT_NEXT_PAGE:
			VID_Menu_SetPage (1);
			break;
		case VID_OPT_BACK:
			VID_Menu_SetPage (0);
			break;
		case VID_OPT_FOV:
			VID_Menu_StepFloatCvar (&scr_fov, 5.0f, 60.0f, 140.0f);
			break;
		case VID_OPT_SHOWFPS:
			Cvar_SetValueQuick (&scr_showfps, !CVAR_TO_BOOL (scr_showfps));
			break;
		case VID_OPT_GI_LEVEL:
			VID_Menu_StepGiLevel (1.0f);
			break;
		case VID_OPT_GODRAYS:
			Cvar_SetValueQuick (&rt_godrays, !CVAR_TO_BOOL (rt_godrays));
			break;
		case VID_OPT_REFLECT:
			VID_Menu_StepReflDepth (1.0f);
			break;
		case VID_OPT_DENOISER:
			Cvar_SetValueQuick (&rt_denoiser, !CVAR_TO_BOOL (rt_denoiser));
			break;
		case VID_OPT_TEXTURES:
			Cvar_SetValueQuick (&rt_no_textures, !CVAR_TO_BOOL (rt_no_textures));
			break;
		default:
			break;
		}
		break;

	case K_ENTER:
	case K_KP_ENTER:
		m_entersound = true;
		switch (video_options_cursor)
		{
		//case VID_OPT_MODE:
		//	VID_Menu_ChooseNextMode (1);
		//	break;
		//case VID_OPT_BPP:
		//	VID_Menu_ChooseNextBpp ();
		//	break;
		//case VID_OPT_REFRESHRATE:
		//	VID_Menu_ChooseNextRate (1);
		//	break;
		case VID_OPT_MODE:
		case VID_OPT_APPLY:
			Cbuf_AddText ("vid_restart\n");
			break;
		case VID_OPT_BLOOM:
			Cvar_SetValueQuick (&rt_bloom, !CVAR_TO_BOOL (rt_bloom));
			break;
		case VID_OPT_VSYNC:
			Cbuf_AddText ("toggle vid_vsync\n");
			break;
		case VID_OPT_NEXT_PAGE:
			VID_Menu_SetPage (1);
			break;
		case VID_OPT_BACK:
			VID_Menu_SetPage (0);
			break;
		case VID_OPT_SHOWFPS:
			Cvar_SetValueQuick (&scr_showfps, !CVAR_TO_BOOL (scr_showfps));
			break;
		case VID_OPT_GODRAYS:
			Cvar_SetValueQuick (&rt_godrays, !CVAR_TO_BOOL (rt_godrays));
			break;
		case VID_OPT_DENOISER:
			Cvar_SetValueQuick (&rt_denoiser, !CVAR_TO_BOOL (rt_denoiser));
			break;
		case VID_OPT_TEXTURES:
			Cvar_SetValueQuick (&rt_no_textures, !CVAR_TO_BOOL (rt_no_textures));
			break;
		}
		break;

	default:
		break;
	}
}

/*
================
VID_MenuDraw
================
*/
static void VID_MenuDraw (cb_context_t *cbx)
{
	int         i, y;
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
	title = (video_options_page == 0) ? "Video Options" : "Video Options (2/2)";
	M_PrintWhite (cbx, (320 - 8 * strlen (title)) / 2, y, title);

	y += 12;

	// options
	for (i = 0; i < VIDEO_OPTIONS_ITEMS; i++)
	{
		if (VID_MenuRowPage (i) != video_options_page)
			continue;

		switch (i)
		{
		case VID_OPT_MODE:
			M_Print (cbx, 16, y, "        Video mode");
			M_Print (cbx, 184, y, va ("%ix%i", (int)vid_width.value, (int)vid_height.value));
			break;
		//case VID_OPT_REFRESHRATE:
		//	M_Print (cbx, 16, y, "      Refresh rate");
		//	M_Print (cbx, 184, y, va ("%i", (int)vid_refreshrate.value));
		//	break;
		case VID_OPT_APPLY:
			M_Print (cbx, 16, y, "             Apply");
			break;


		case VID_OPT_BLOOM:
			M_Print (cbx, 16, y, "             Bloom");
			M_DrawCheckbox (cbx, 184, y, CVAR_TO_BOOL (rt_bloom));
			break;
		case VID_OPT_EXPOSURE_BIAS:
			M_Print (cbx, 16, y, "     Exposure bias");
			M_Print (cbx, 184, y, va ("%+.1f EV", CVAR_TO_FLOAT (rt_exposure_bias)));
			break;
		case VID_OPT_CONTRAST:
			M_Print (cbx, 16, y, "          Contrast");
			M_Print (cbx, 184, y, va ("%d%%", (int)(CVAR_TO_FLOAT (rt_contrast) * 100.0f + 0.5f)));
			break;
		case VID_OPT_VSYNC:
			M_Print (cbx, 16, y, "     Vertical sync");
			M_DrawCheckbox (cbx, 184, y, (int)vid_vsync.value);
			break;
		case VID_OPT_MAX_FPS:
			M_Print (cbx, 16, y, "           Max FPS");
			if (menu_settings.host_maxfps <= 0)
				M_Print (cbx, 184, y, "no limit");
			else
				M_Print (cbx, 184, y, va ("%d", menu_settings.host_maxfps));
			break;


		case VID_OPT_UPSCALER:
			y += 8; // separate

			M_Print (cbx, 16, y, "          Upscaler");
			{
				const char *name = "Off";
				switch (menu_settings.upscaler_type)
				{
				case UPSCALER_FSR2:  name = "AMD FSR 2.0"; break;
				case UPSCALER_FSR31: name = "AMD FSR 3.1"; break;
				case UPSCALER_DLSS:  name = "Nvidia DLSS"; break;
				}
				M_Print (cbx, 184, y, name);
			}
			break;
		case VID_OPT_UPSCALER_QUALITY:
			M_Print (cbx, 16, y, "            Preset");
			{
				RgRenderUpscaleTechnique tech;
				int q = menu_settings.upscaler_quality;
				switch (menu_settings.upscaler_type)
				{
				case UPSCALER_FSR2:  tech = RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR2; break;
				case UPSCALER_FSR31: tech = RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR3; break;
				case UPSCALER_DLSS:  tech = RG_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS; break;
				default:             tech = RG_RENDER_UPSCALE_TECHNIQUE_NEAREST; q = 0; break;
				}
				if (q < 1 && menu_settings.upscaler_type != UPSCALER_OFF)
					q = GetUpscalerDefaultQuality (menu_settings.upscaler_type);
				M_Print (cbx, 184, y, GetUpscalerOptionName (q, tech));
			}
			break;


		case VID_OPT_FILTER:
			M_Print (cbx, 16, y, " Texture filtering");
			M_Print (cbx, 184, y, (menu_settings.vid_filter == 0) ? "smooth" : "classic");
			break;
		case VID_OPT_PARTICLES:
			M_Print (cbx, 16, y, "         Particles");
			M_Print (cbx, 184, y, (menu_settings.r_particles == 0) ? "none" : ((menu_settings.r_particles == 2) ? "classic" : "circle"));
			break;
		case VID_OPT_VOLUMETRICS:
			M_Print (cbx, 16, y, "       Volumetrics");
			M_Print (cbx, 184, y, CVAR_TO_UINT32 (rt_volume_type) == 2 ? "sky" : CVAR_TO_UINT32 (rt_volume_type) == 1 ? "simple" : "off");
			break;
		case VID_OPT_MATERIALS_ONLY:
			M_Print (cbx, 16, y, "  Materials only");
			M_Print (cbx, 184, y, CVAR_TO_BOOL (rt_materials_only) ? "on" : "off");
			break;
		case VID_OPT_RENDER_SCALE:
			M_Print (cbx, 16, y, "           Vintage");
			M_Print (cbx, 184, y, GetVintageOptionName (menu_settings.rt_vintage));
			break;


		case VID_OPT_NEXT_PAGE:
			y += 8; // separate

			M_Print (cbx, 16, y, "         Next page");
			break;


		case VID_OPT_FOV:
			M_Print (cbx, 16, y, "     Field of view");
			M_Print (cbx, 184, y, va ("%d", (int)scr_fov.value));
			break;
		case VID_OPT_SHOWFPS:
			M_Print (cbx, 16, y, "       Display FPS");
			M_DrawCheckbox (cbx, 184, y, (int)scr_showfps.value);
			break;


		case VID_OPT_GI_LEVEL:
			y += 8; // separate

			M_Print (cbx, 16, y, " Indirect lighting");
			M_Print (cbx, 184, y, VID_Menu_GetGiLevelName ());
			break;
		case VID_OPT_GODRAYS:
			M_Print (cbx, 16, y, "          God rays");
			M_DrawCheckbox (cbx, 184, y, CVAR_TO_BOOL (rt_godrays));
			break;
		case VID_OPT_REFLECT:
			{
				const int depth = (int)(CVAR_TO_FLOAT (rt_reflrefr_depth) + 0.5f);
				char      label[32];

				if (depth <= 0)
					q_strlcpy (label, "off", sizeof (label));
				else if (depth == 1)
					q_strlcpy (label, "1 bounce", sizeof (label));
				else
					q_snprintf (label, sizeof (label), "%d bounces", depth);

				M_Print (cbx, 16, y, "       Reflections");
				M_Print (cbx, 184, y, label);
			}
			break;
		case VID_OPT_DENOISER:
			M_Print (cbx, 16, y, "          Denoiser");
			M_DrawCheckbox (cbx, 184, y, CVAR_TO_BOOL (rt_denoiser));
			break;
		case VID_OPT_TEXTURES:
			M_Print (cbx, 16, y, "          Textures");
			M_DrawCheckbox (cbx, 184, y, !CVAR_TO_BOOL (rt_no_textures));
			break;


		case VID_OPT_BACK:
			y += 8; // separate

			M_Print (cbx, 16, y, "              Back");
			break;
		}

		if (video_options_cursor == i)
			M_DrawCharacter (cbx, 168, y, 12 + ((int)(realtime * 4) & 1));

		y += 8;
	}
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
SCR_ScreenShot_f -- not implemented, the game takes no screen grabs
==================
*/
void SCR_ScreenShot_f (void)
{
	Con_Printf ("SCR_ScreenShot_f: Not implemented\n");
}

void VID_FocusGained (void)
{
	has_focus = true;
}

void VID_FocusLost (void)
{
	has_focus = false;
}
