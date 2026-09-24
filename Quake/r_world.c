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
// r_world.c: world model rendering

#include "quakedef.h"
#include "atomics.h"

extern cvar_t gl_fullbrights;
extern cvar_t r_drawflat;
extern cvar_t r_oldskyleaf;
extern cvar_t r_showtris;
extern cvar_t r_simd;
extern cvar_t gl_zfix;
extern cvar_t r_gpulightmapupdate;

extern cvar_t rt_brush_metal;
extern cvar_t rt_brush_rough;
extern cvar_t rt_enable_pvs;
extern cvar_t rt_reflrefr_depth;
extern cvar_t rt_teleport_portals;
extern cvar_t rt_wlight_intensity, rt_wlight_radius;
extern cvar_t rt_emis_light_intensity;
extern cvar_t rt_cluster_dlights;
extern cvar_t rt_light_styles;
extern cvar_t rt_light_styles_reach;
extern cvar_t rt_wmodel_lights_batch;
extern cvar_t rt_world_batch_merge;
extern cvar_t rt_truelight;
extern cvar_t rt_materials_only;
extern cvar_t rt_debugemissive;
extern cvar_t rt_light_report_filter;
extern cvar_t rt_worldcensus;
extern cvar_t rt_worldlights_stats;
extern cvar_t rt_worldclusters_grid;

cvar_t r_parallelmark = {"r_parallelmark", "1", CVAR_NONE};

byte *SV_FatPVS (vec3_t org, qmodel_t *worldmodel);

static int world_texstart[NUM_WORLD_CBX];
static int world_texend[NUM_WORLD_CBX];

extern RgVertex *rtallbrushvertices;

#define MAX_WORLDLIGHTS_COUNT 8192

/* Polygons are emitted per texture repetition over the glow extents (see RT_AddEmissiveLight),
   so a face contributes as many lights as it has repetitions of the glow. This is not a
   fidelity limit but a hang guard: texture coordinates of a badly scaled surface can span
   hundreds of repetitions, and every one of them would cost a clip, a light and a cluster
   registry slot. Faces above it keep the whole-surface light. */
#define RT_MAX_EMISSIVE_POLYS_PER_FACE 64

/* A masked light has to keep the surface's own polygons, so a face with more corners than a light
   carries is cut into convex pieces instead of being replaced by a square (see RT_SplitUvPolygon).
   RT_MAX_UV_SPLIT_VERTS is how many corners a face may have and still be cut, and
   RT_UV_SPLIT_PIECE_VERTS how many corners each piece takes: six at a time leaves a convex
   remainder, so the largest accepted face comes apart in exactly the sixteen pieces
   RT_MAX_UV_SPLIT_PIECES allows, which is also what bounds the lights one face can add. A face
   past the split keeps the square, and its emission is then the mean of its mask. */
#define RT_MAX_UV_SPLIT_VERTS   68
#define RT_UV_SPLIT_PIECE_VERTS 6
#define RT_MAX_UV_SPLIT_PIECES  16

static RgTexturedAreaLightUploadInfo rt_wldlights_emissive[MAX_WORLDLIGHTS_COUNT];
static int                           rt_wldlights_emissive_count = 0;

static const msurface_t *rt_wldlights_emissive_surf[MAX_WORLDLIGHTS_COUNT];
static gltexture_t      *rt_wldlights_emissive_tex[MAX_WORLDLIGHTS_COUNT];

/* The map's lights as handed to the renderer, and the center each one is placed by. */
static RgTexturedAreaLightUploadInfo rt_wldlights_emissive_upload[MAX_WORLDLIGHTS_COUNT];
static vec3_t                        rt_wldlights_emissive_center[MAX_WORLDLIGHTS_COUNT];

/* Whether a style of a stored light is accepted by the reach test of RT_SurfaceLightStyleScale.
   The answer only changes with the light list itself or with the reach cvar, never per frame:
   the elight table is filled at map load and is not touched again. */
static byte     rt_wldlights_style_accepted[MAX_WORLDLIGHTS_COUNT][MAXLIGHTMAPS];
static qboolean rt_wldlights_style_accepted_dirty = true;
static float    rt_wldlights_style_accepted_reach = 0.0f;

typedef struct rt_emis_stats_s
{
	int surfaces;
	int no_material;
	int no_color;
	int style_off;
	int degenerate;
	int frame_off;
	int static_queued;
	int static_dropped;
	int dynamic;
	int glow_lights;
	int glow_faces;
	int glow_fallback;
} rt_emis_stats_t;

static rt_emis_stats_t rt_emis_stats;

#define RT_EMIS_SKIP_NAMES 8
static char rt_emis_skip_texture[RT_EMIS_SKIP_NAMES][32];
static char rt_emis_skip_reason[RT_EMIS_SKIP_NAMES][24];
static int  rt_emis_skip_count[RT_EMIS_SKIP_NAMES];
static int  rt_emis_skip_num;

static void RT_EmisNoteSkip (const char *texture, const char *reason)
{
	for (int i = 0; i < rt_emis_skip_num; i++)
	{
		if (!strcmp (rt_emis_skip_texture[i], texture) && !strcmp (rt_emis_skip_reason[i], reason))
		{
			rt_emis_skip_count[i]++;
			return;
		}
	}

	if (rt_emis_skip_num >= RT_EMIS_SKIP_NAMES)
		return;

	q_snprintf (rt_emis_skip_texture[rt_emis_skip_num], sizeof (rt_emis_skip_texture[0]), "%s", texture);
	q_snprintf (rt_emis_skip_reason[rt_emis_skip_num], sizeof (rt_emis_skip_reason[0]), "%s", reason);
	rt_emis_skip_count[rt_emis_skip_num] = 1;
	rt_emis_skip_num++;
}

#define RT_EMIS_WATCH_MAX 16
typedef struct rt_emis_watch_s
{
	char name[32];
	int  surfaces;
	int  lights;
	int  no_material;
	int  no_color;
	int  style_off;
	int  degenerate;
	int  frame_off;
	int  hist_frames;
	int  hist_lit;
	int  hist_dark;
	int  hist_style_off;
	int  hist_frame_off;
	int  hist_min_lights;
	int  hist_max_lights;
	float min_style_scale;
	int   style_count;
	byte  styles[MAXLIGHTMAPS];
} rt_emis_watch_t;

static rt_emis_watch_t rt_emis_watch[RT_EMIS_WATCH_MAX];
static int             rt_emis_watch_num;
static char            rt_emis_watch_filter[64];

static rt_emis_watch_t *RT_EmisWatch (const char *texture)
{
	const char *filter = rt_light_report_filter.string;

	if (!filter[0] || !texture)
		return NULL;

	if (strcmp (filter, rt_emis_watch_filter))
	{
		memset (rt_emis_watch, 0, sizeof (rt_emis_watch));
		rt_emis_watch_num = 0;
		q_snprintf (rt_emis_watch_filter, sizeof (rt_emis_watch_filter), "%s", filter);
	}

	if (!strstr (texture, filter))
		return NULL;

	for (int i = 0; i < rt_emis_watch_num; i++)
	{
		if (!strcmp (rt_emis_watch[i].name, texture))
			return &rt_emis_watch[i];
	}

	if (rt_emis_watch_num >= RT_EMIS_WATCH_MAX)
		return NULL;

	rt_emis_watch_t *watch = &rt_emis_watch[rt_emis_watch_num++];
	memset (watch, 0, sizeof (*watch));
	watch->hist_min_lights = 0x7FFFFFFF;
	watch->min_style_scale = 1.0f;
	q_snprintf (watch->name, sizeof (watch->name), "%s", texture);
	return watch;
}

static void RT_EmisWatchFrameEnd (void)
{
	for (int i = 0; i < rt_emis_watch_num; i++)
	{
		rt_emis_watch_t *w = &rt_emis_watch[i];

		if (w->surfaces > 0)
		{
			w->hist_frames++;
			/* A frame that carries no light of its own is a dark frame: the surface is still
			   there and still visible, it just does not glow on that frame. Counting it as lit
			   would hide the blinks of a lamp behind a steady "lit" history. */
			if (w->lights > 0 && w->frame_off == 0)
				w->hist_lit++;
			else
				w->hist_dark++;
			if (w->style_off > 0)
				w->hist_style_off++;
			if (w->frame_off > 0)
				w->hist_frame_off++;
			if (w->lights < w->hist_min_lights)
				w->hist_min_lights = w->lights;
			if (w->lights > w->hist_max_lights)
				w->hist_max_lights = w->lights;
		}

		w->surfaces = 0;
		w->lights = 0;
		w->no_material = 0;
		w->no_color = 0;
		w->style_off = 0;
		w->degenerate = 0;
		w->frame_off = 0;
		w->min_style_scale = 1.0f;
		w->style_count = 0;
	}
}


// RT: remove SIMD here, as the culling is not requires
#undef USE_SIMD
#undef USE_SSE2

/*
===============
mark_surfaces_state_t
===============
*/
typedef struct
{
#if defined(USE_SIMD)
	__m128 frustum_px[4];
	__m128 frustum_py[4];
	__m128 frustum_pz[4];
	__m128 frustum_pd[4];
	__m128 vieworg_px;
	__m128 vieworg_py;
	__m128 vieworg_pz;
	int    frustum_ofsx[4];
	int    frustum_ofsy[4];
	int    frustum_ofsz[4];
#endif
	byte *vis;
} mark_surfaces_state_t;
mark_surfaces_state_t mark_surfaces_state;

//==============================================================================
//
// SETUP CHAINS
//
//==============================================================================

/*
================
R_ClearTextureChains -- ericw

clears texture chains for all textures used by the given model, and also
clears the lightmap chains
================
*/
void R_ClearTextureChains (qmodel_t *mod, texchain_t chain)
{
	int i;

	// set all chains to null
	for (i = 0; i < mod->numtextures; i++)
	{
		if (mod->textures[i])
		{
			mod->textures[i]->texturechains[chain] = NULL;
			mod->textures[i]->chain_size[chain] = 0;
		}
	}
}

/*
================
R_ChainSurface -- ericw -- adds the given surface to its texture chain
================
*/
void R_ChainSurface (msurface_t *surf, texchain_t chain)
{
	surf->texturechains[chain] = surf->texinfo->texture->texturechains[chain];
	surf->texinfo->texture->texturechains[chain] = surf;
	surf->texinfo->texture->chain_size[chain] += 1;
}

/*
================
R_BackFaceCull -- johnfitz -- returns true if the surface is facing away from vieworg
================
*/
static inline qboolean R_BackFaceCull (msurface_t *surf)
{
	double dot;

	if (surf->plane->type < 3)
		dot = r_refdef.vieworg[surf->plane->type] - surf->plane->dist;
	else
		dot = DotProduct (r_refdef.vieworg, surf->plane->normal) - surf->plane->dist;

	if ((dot < 0) ^ !!(surf->flags & SURF_PLANEBACK))
		return true;

	return false;
}

/*
===============
R_SetupWorldCBXTexRanges
===============
*/
void R_SetupWorldCBXTexRanges (qboolean use_tasks)
{
	memset (world_texstart, 0, sizeof (world_texstart));
	memset (world_texend, 0, sizeof (world_texend));

	const int num_textures = cl.worldmodel->numtextures;
	if (!use_tasks)
	{
		world_texstart[0] = 0;
		world_texend[0] = num_textures;
		return;
	}

	int total_world_surfs = 0;
	for (int i = 0; i < num_textures; ++i)
	{
		texture_t *t = cl.worldmodel->textures[i];
		if (!t || !t->texturechains[chain_world] || t->texturechains[chain_world]->flags & (SURF_DRAWTURB | SURF_DRAWTILED | SURF_NOTEXTURE))
			continue;
		total_world_surfs += t->chain_size[chain_world];
	}

	const int num_surfs_per_cbx = (total_world_surfs + NUM_WORLD_CBX - 1) / NUM_WORLD_CBX;
	int       current_cbx = 0;
	int       num_assigned_to_cbx = 0;
	for (int i = 0; i < num_textures; ++i)
	{
		texture_t *t = cl.worldmodel->textures[i];
		if (!t || !t->texturechains[chain_world] || t->texturechains[chain_world]->flags & (SURF_DRAWTURB | SURF_DRAWTILED | SURF_NOTEXTURE))
			continue;
		assert (current_cbx < NUM_WORLD_CBX);
		world_texend[current_cbx] = i + 1;
		num_assigned_to_cbx += t->chain_size[chain_world];
		if (num_assigned_to_cbx >= num_surfs_per_cbx)
		{
			current_cbx += 1;
			if (current_cbx < NUM_WORLD_CBX)
			{
				world_texstart[current_cbx] = i + 1;
			}
			num_assigned_to_cbx = 0;
		}
	}
}

#ifdef USE_SSE2
/*
===============
R_BackFaceCullSIMD

Performs backface culling for 32 planes
===============
*/
static FORCE_INLINE uint32_t R_BackFaceCullSIMD (soa_plane_t *planes)
{
	__m128 px = mark_surfaces_state.vieworg_px;
	__m128 py = mark_surfaces_state.vieworg_py;
	__m128 pz = mark_surfaces_state.vieworg_pz;

	uint32_t activelanes = 0;
	for (int plane_index = 0; plane_index < 4; ++plane_index)
	{
		soa_plane_t *plane = planes + plane_index;

		__m128 v0 = _mm_mul_ps (_mm_loadu_ps ((*plane) + 0), px);
		__m128 v1 = _mm_mul_ps (_mm_loadu_ps ((*plane) + 4), px);

		v0 = _mm_add_ps (v0, _mm_mul_ps (_mm_loadu_ps ((*plane) + 8), py));
		v1 = _mm_add_ps (v1, _mm_mul_ps (_mm_loadu_ps ((*plane) + 12), py));

		v0 = _mm_add_ps (v0, _mm_mul_ps (_mm_loadu_ps ((*plane) + 16), pz));
		v1 = _mm_add_ps (v1, _mm_mul_ps (_mm_loadu_ps ((*plane) + 20), pz));

		__m128 pd0 = _mm_loadu_ps ((*plane) + 24);
		__m128 pd1 = _mm_loadu_ps ((*plane) + 28);

		uint32_t plane_lanes = (uint32_t)(_mm_movemask_ps (_mm_cmplt_ps (pd0, v0)) | (_mm_movemask_ps (_mm_cmplt_ps (pd1, v1)) << 4));
		activelanes |= plane_lanes << (plane_index * 8);
	}
	return activelanes;
}

/*
===============
R_CullBoxSIMD

Performs frustum culling for 32 bounding boxes
===============
*/
static FORCE_INLINE uint32_t R_CullBoxSIMD (soa_aabb_t *boxes, uint32_t activelanes)
{
	for (int frustum_index = 0; frustum_index < 4; ++frustum_index)
	{
		if (activelanes == 0)
			break;

		int    ofsx = mark_surfaces_state.frustum_ofsx[frustum_index];
		int    ofsy = mark_surfaces_state.frustum_ofsy[frustum_index];
		int    ofsz = mark_surfaces_state.frustum_ofsz[frustum_index];
		__m128 px = mark_surfaces_state.frustum_px[frustum_index];
		__m128 py = mark_surfaces_state.frustum_py[frustum_index];
		__m128 pz = mark_surfaces_state.frustum_pz[frustum_index];
		__m128 pd = mark_surfaces_state.frustum_pd[frustum_index];

		uint32_t frustum_lanes = 0;
		for (int boxes_index = 0; boxes_index < 4; ++boxes_index)
		{
			soa_aabb_t *box = boxes + boxes_index;
			__m128      v0 = _mm_mul_ps (_mm_loadu_ps ((*box) + ofsx), px);
			__m128      v1 = _mm_mul_ps (_mm_loadu_ps ((*box) + ofsx + 4), px);
			v0 = _mm_add_ps (v0, _mm_mul_ps (_mm_loadu_ps ((*box) + ofsy), py));
			v1 = _mm_add_ps (v1, _mm_mul_ps (_mm_loadu_ps ((*box) + ofsy + 4), py));
			v0 = _mm_add_ps (v0, _mm_mul_ps (_mm_loadu_ps ((*box) + ofsz), pz));
			v1 = _mm_add_ps (v1, _mm_mul_ps (_mm_loadu_ps ((*box) + ofsz + 4), pz));
			frustum_lanes |= (uint32_t)(_mm_movemask_ps (_mm_cmplt_ps (pd, v0)) | (_mm_movemask_ps (_mm_cmplt_ps (pd, v1)) << 4)) << (boxes_index * 8);
		}
		activelanes &= frustum_lanes;
	}

	return activelanes;
}
#endif // defined(USE_SSE2)

#if defined(USE_SIMD)
/*
===============
R_MarkVisSurfacesSIMD
===============
*/
void R_MarkVisSurfacesSIMD (qboolean *use_tasks)
{
	double prof_start = RT_Prof_Begin ();

	msurface_t  *surf;
	unsigned int i, k;
	unsigned int numleafs = cl.worldmodel->numleafs;
	unsigned int numsurfaces = cl.worldmodel->numsurfaces;
	uint32_t    *vis = (uint32_t *)mark_surfaces_state.vis;
	uint32_t    *surfvis = (uint32_t *)cl.worldmodel->surfvis;
	soa_aabb_t  *leafbounds = cl.worldmodel->soa_leafbounds;

	// iterate through leaves, marking surfaces
	for (i = 0; i < numleafs; i += 32)
	{
		uint32_t mask = vis[i / 32];
		if (mask == 0)
			continue;

		mask = R_CullBoxSIMD (&leafbounds[i / 8], mask);
		while (mask != 0)
		{
			const int j = FindFirstBitNonZero (mask);
			mask &= ~(1u << j);

			mleaf_t *leaf = &cl.worldmodel->leafs[1 + i + j];
			if (leaf->contents != CONTENTS_SKY || r_oldskyleaf.value)
			{
				unsigned int nummarksurfaces = leaf->nummarksurfaces;
				int         *marksurfaces = leaf->firstmarksurface;
				for (k = 0; k < nummarksurfaces; ++k)
				{
					unsigned int index = marksurfaces[k];
					surfvis[index / 32] |= 1u << (index % 32);
				}
			}

			// add static models
			if (leaf->efrags)
				R_StoreEfrags (&leaf->efrags);
		}
	}

	uint32_t brushpolys = 0;
	for (i = 0; i < numsurfaces; i += 32)
	{
		uint32_t mask = surfvis[i / 32];
		if (mask == 0)
			continue;

		mask &= R_BackFaceCullSIMD (&cl.worldmodel->soa_surfplanes[i / 8]);
		while (mask != 0)
		{
			const int j = FindFirstBitNonZero (mask);
			mask &= ~(1u << j);

			surf = &cl.worldmodel->surfaces[i + j];
			++brushpolys;
			R_ChainSurface (surf, chain_world);
			if (surf->texinfo->texture->warpimage)
				Atomic_StoreUInt32 (&surf->texinfo->texture->update_warp, true);
		}
	}

	Atomic_AddUInt32 (&rs_brushpolys, brushpolys); // count wpolys here
	R_SetupWorldCBXTexRanges (*use_tasks);

	RT_Prof_End (RT_PROF_MARK, prof_start);
}

/*
===============
R_MarkLeafsSIMD
===============
*/
void R_MarkLeafsSIMD (int index, void *unused)
{
	double prof_start = RT_Prof_Begin ();

	unsigned int     j;
	unsigned int     first_leaf = index * 32;
	atomic_uint32_t *surfvis = (atomic_uint32_t *)cl.worldmodel->surfvis;
	soa_aabb_t      *leafbounds = cl.worldmodel->soa_leafbounds;
	uint32_t        *vis = (uint32_t *)mark_surfaces_state.vis;

	uint32_t *mask = &vis[index];
	if (*mask == 0)
	{
		RT_Prof_End (RT_PROF_MARK, prof_start);
		return;
	}

	*mask = R_CullBoxSIMD (&leafbounds[index * 4], *mask);

	uint32_t mask_iter = *mask;
	while (mask_iter != 0)
	{
		const int i = FindFirstBitNonZero (mask_iter);

		mleaf_t *leaf = &cl.worldmodel->leafs[1 + first_leaf + i];
		if (leaf->contents != CONTENTS_SKY || r_oldskyleaf.value)
		{
			unsigned int nummarksurfaces = leaf->nummarksurfaces;
			int         *marksurfaces = leaf->firstmarksurface;
			for (j = 0; j < nummarksurfaces; ++j)
			{
				unsigned int surf_index = marksurfaces[j];
				Atomic_OrUInt32 (&surfvis[surf_index / 32], 1u << (surf_index % 32));
			}
		}
		const uint32_t bit_mask = ~(1u << i);
		if (!leaf->efrags)
		{
			*mask &= bit_mask;
		}
		mask_iter &= bit_mask;
	}

	RT_Prof_End (RT_PROF_MARK, prof_start);
}

/*
===============
R_BackfaceCullSurfacesSIMD
===============
*/
void R_BackfaceCullSurfacesSIMD (int index, void *unused)
{
	double prof_start = RT_Prof_Begin ();

	uint32_t   *surfvis = (uint32_t *)cl.worldmodel->surfvis;
	msurface_t *surf;

	uint32_t *mask = &surfvis[index];
	if (*mask == 0)
	{
		RT_Prof_End (RT_PROF_CULL, prof_start);
		return;
	}

	*mask &= R_BackFaceCullSIMD (&cl.worldmodel->soa_surfplanes[index * 4]);

	uint32_t mask_iter = *mask;
	while (mask_iter != 0)
	{
		const int i = FindFirstBitNonZero (mask_iter);

		surf = &cl.worldmodel->surfaces[(index * 32) + i];
		if (surf->lightmaptexturenum >= 0)
			Atomic_StoreUInt32 (&lightmaps[surf->lightmaptexturenum].modified, true);
		if (surf->texinfo->texture->warpimage)
			Atomic_StoreUInt32 (&surf->texinfo->texture->update_warp, true);

		const uint32_t bit_mask = ~(1u << i);
		mask_iter &= bit_mask;
	}

	RT_Prof_End (RT_PROF_CULL, prof_start);
}

/*
===============
R_StoreLeafEFrags
===============
*/
void R_StoreLeafEFrags (void *unused)
{
	double prof_start = RT_Prof_Begin ();

	unsigned int i;
	unsigned int numleafs = cl.worldmodel->numleafs;
	uint32_t    *vis = (uint32_t *)mark_surfaces_state.vis;
	for (i = 0; i < numleafs; i += 32)
	{
		uint32_t mask = vis[i / 32];
		while (mask != 0)
		{
			const int j = FindFirstBitNonZero (mask);
			mask &= ~(1u << j);
			mleaf_t *leaf = &cl.worldmodel->leafs[1 + i + j];
			R_StoreEfrags (&leaf->efrags);
		}
	}

	RT_Prof_End (RT_PROF_EFRAGS, prof_start);
}

/*
===============
R_ChainVisSurfaces
===============
*/
void R_ChainVisSurfaces (qboolean *use_tasks)
{
	double prof_start = RT_Prof_Begin ();

	unsigned int i;
	msurface_t  *surf;
	unsigned int numsurfaces = cl.worldmodel->numsurfaces;
	uint32_t    *surfvis = (uint32_t *)cl.worldmodel->surfvis;
	uint32_t     brushpolys = 0;
	for (i = 0; i < numsurfaces; i += 32)
	{
		uint32_t mask = surfvis[i / 32];
		while (mask != 0)
		{
			const int j = FindFirstBitNonZero (mask);
			mask &= ~(1u << j);
			surf = &cl.worldmodel->surfaces[i + j];
			++brushpolys;
			R_ChainSurface (surf, chain_world);
		}
	}

	Atomic_AddUInt32 (&rs_brushpolys, brushpolys); // count wpolys here
	R_SetupWorldCBXTexRanges (*use_tasks);

	RT_Prof_End (RT_PROF_CHAIN, prof_start);
}
#endif // defined(USE_SIMD)

/*
===============
R_MarkVisSurfaces
===============
*/
void R_MarkVisSurfaces (qboolean *use_tasks)
{
	double prof_start = RT_Prof_Begin ();

	int         i, j;
	msurface_t *surf;
	mleaf_t    *leaf;
	uint32_t    brushpolys = 0;
	uint32_t   *vis = (uint32_t *)mark_surfaces_state.vis;

	leaf = &cl.worldmodel->leafs[1];
	for (i = 0; i < cl.worldmodel->numleafs; i++, leaf++)
	{
        if (CVAR_TO_BOOL (rt_enable_pvs))
        {
		    if (!(vis[i / 32] & 1u << i % 32))
                continue;

            if (R_CullBox (leaf->minmaxs, leaf->minmaxs + 3))
                continue;
        }

        if (r_oldskyleaf.value || leaf->contents != CONTENTS_SKY)
        {
            for (j = 0; j < leaf->nummarksurfaces; j++)
            {
                surf = &cl.worldmodel->surfaces[leaf->firstmarksurface[j]];

                if (surf->visframe == r_visframecount)
                    continue;

                surf->visframe = r_visframecount;

                if (CVAR_TO_BOOL (rt_enable_pvs))
                {		    
                    if (R_BackFaceCull (surf))
                        continue;
                }

                ++brushpolys;
                R_ChainSurface (surf, chain_world);
                if (surf->texinfo->texture->warpimage)
                    Atomic_StoreUInt32 (&surf->texinfo->texture->update_warp, true);
            }
        }

        // add static models
        if (leaf->efrags)
            R_StoreEfrags (&leaf->efrags);
    }

	Atomic_AddUInt32 (&rs_brushpolys, brushpolys); // count wpolys here
	R_SetupWorldCBXTexRanges (*use_tasks);

	RT_Prof_End (RT_PROF_MARK, prof_start);
}

/*
===============
R_MarkSurfacesPrepare
===============
*/
static void R_MarkSurfacesPrepare (void *unused)
{
	int      i;
	qboolean nearwaterportal;
	int      numleafs = cl.worldmodel->numleafs;

	// check this leaf for water portals
	// TODO: loop through all water surfs and use distance to leaf cullbox
	nearwaterportal = false;
	for (i = 0; i < r_viewleaf->nummarksurfaces; i++)
		if (cl.worldmodel->surfaces[r_viewleaf->firstmarksurface[i]].flags & SURF_DRAWTURB)
			nearwaterportal = true;

	// choose vis data
	if (!CVAR_TO_BOOL (rt_enable_pvs) || r_viewleaf->contents == CONTENTS_SOLID || r_viewleaf->contents == CONTENTS_SKY)
		mark_surfaces_state.vis = Mod_NoVisPVS (cl.worldmodel);
	else if (nearwaterportal)
		mark_surfaces_state.vis = SV_FatPVS (r_origin, cl.worldmodel);
	else
		mark_surfaces_state.vis = Mod_LeafPVS (r_viewleaf, cl.worldmodel);

	uint32_t *vis = (uint32_t *)mark_surfaces_state.vis;
	if ((numleafs % 32) != 0)
		vis[numleafs / 32] &= (1u << (numleafs % 32)) - 1;

	r_visframecount++;

	// set all chains to null
	for (i = 0; i < cl.worldmodel->numtextures; i++)
		if (cl.worldmodel->textures[i])
		{
			cl.worldmodel->textures[i]->texturechains[chain_world] = NULL;
			cl.worldmodel->textures[i]->chain_size[chain_world] = 0;
		}

#if defined(USE_SIMD)
	if (use_simd)
	{
		memset (cl.worldmodel->surfvis, 0, (cl.worldmodel->numsurfaces + 31) / 8);
		for (int frustum_index = 0; frustum_index < 4; ++frustum_index)
		{
			mplane_t *p = frustum + frustum_index;
			byte      signbits = p->signbits;
			__m128    vplane = _mm_loadu_ps (p->normal);
			mark_surfaces_state.frustum_ofsx[frustum_index] = signbits & 1 ? 0 : 8;   // x min/max
			mark_surfaces_state.frustum_ofsy[frustum_index] = signbits & 2 ? 16 : 24; // y min/max
			mark_surfaces_state.frustum_ofsz[frustum_index] = signbits & 4 ? 32 : 40; // z min/max
			mark_surfaces_state.frustum_px[frustum_index] = _mm_shuffle_ps (vplane, vplane, _MM_SHUFFLE (0, 0, 0, 0));
			mark_surfaces_state.frustum_py[frustum_index] = _mm_shuffle_ps (vplane, vplane, _MM_SHUFFLE (1, 1, 1, 1));
			mark_surfaces_state.frustum_pz[frustum_index] = _mm_shuffle_ps (vplane, vplane, _MM_SHUFFLE (2, 2, 2, 2));
			mark_surfaces_state.frustum_pd[frustum_index] = _mm_shuffle_ps (vplane, vplane, _MM_SHUFFLE (3, 3, 3, 3));
		}
		__m128 pos = _mm_loadu_ps (r_refdef.vieworg);
		mark_surfaces_state.vieworg_px = _mm_shuffle_ps (pos, pos, _MM_SHUFFLE (0, 0, 0, 0));
		mark_surfaces_state.vieworg_py = _mm_shuffle_ps (pos, pos, _MM_SHUFFLE (1, 1, 1, 1));
		mark_surfaces_state.vieworg_pz = _mm_shuffle_ps (pos, pos, _MM_SHUFFLE (2, 2, 2, 2));
	}
#endif
}

/*
===============
R_MarkSurfaces -- johnfitz -- mark surfaces based on PVS and rebuild texture chains
===============
*/
void R_MarkSurfaces (qboolean use_tasks, task_handle_t before_mark, task_handle_t *store_efrags, task_handle_t *cull_surfaces, task_handle_t *chain_surfaces)
{
	if (use_tasks)
	{
		task_handle_t prepare_mark = Task_AllocateAndAssignFunc (R_MarkSurfacesPrepare, NULL, 0);
		Task_AddDependency (before_mark, prepare_mark);
		Task_Submit (prepare_mark);
#if defined(USE_SIMD)
		if (use_simd)
		{
			if (r_parallelmark.value)
			{
				unsigned int  numleafs = cl.worldmodel->numleafs;
				task_handle_t mark_surfaces = Task_AllocateAndAssignIndexedFunc (R_MarkLeafsSIMD, (numleafs + 31) / 32, NULL, 0);
				Task_AddDependency (prepare_mark, mark_surfaces);
				Task_Submit (mark_surfaces);

				*store_efrags = Task_AllocateAndAssignFunc (R_StoreLeafEFrags, NULL, 0);
				Task_AddDependency (mark_surfaces, *store_efrags);

				unsigned int numsurfaces = cl.worldmodel->numsurfaces;
				*cull_surfaces = Task_AllocateAndAssignIndexedFunc (R_BackfaceCullSurfacesSIMD, (numsurfaces + 31) / 32, NULL, 0);
				Task_AddDependency (mark_surfaces, *cull_surfaces);

				*chain_surfaces = Task_AllocateAndAssignFunc ((task_func_t)R_ChainVisSurfaces, &use_tasks, sizeof (qboolean));
				Task_AddDependency (*cull_surfaces, *chain_surfaces);
			}
			else
			{
				task_handle_t mark_surfaces = Task_AllocateAndAssignFunc ((task_func_t)R_MarkVisSurfacesSIMD, &use_tasks, sizeof (qboolean));
				Task_AddDependency (prepare_mark, mark_surfaces);
				*store_efrags = mark_surfaces;
				*chain_surfaces = mark_surfaces;
				*cull_surfaces = mark_surfaces;
			}
		}
		else
#endif
		{
			task_handle_t mark_surfaces = Task_AllocateAndAssignFunc ((task_func_t)R_MarkVisSurfaces, &use_tasks, sizeof (qboolean));
			Task_AddDependency (prepare_mark, mark_surfaces);
			*store_efrags = mark_surfaces;
			*chain_surfaces = mark_surfaces;
			*cull_surfaces = mark_surfaces;
		}
	}
	else
	{
		R_MarkSurfacesPrepare (NULL);
		// iterate through leaves, marking surfaces
#if defined(USE_SIMD)
		if (use_simd)
		{
			R_MarkVisSurfacesSIMD (&use_tasks);
		}
		else
#endif
			R_MarkVisSurfaces (&use_tasks);
	}
}

//==============================================================================
//
// VBO SUPPORT
//
//==============================================================================

static int R_NumTriangleIndicesForSurf (int vertcount)
{
	return q_max (0, 3 * (vertcount - 2));
}

/*
================
R_TriangleIndicesForSurf

Writes out the triangle indices needed to draw s as a triangle list.
The number of indices it will write is given by R_NumTriangleIndicesForSurf.
================
*/
static void R_TriangleIndicesForSurf (int basevert, int vertcount, uint32_t *dest)
{
	int i;
	for (i = 2; i < vertcount; i++)
	{
		*dest++ = basevert + i;
		*dest++ = basevert + i - 1;
		*dest++ = basevert;
	}
}

static void RT_ClearBatch (cb_context_t *cbx)
{
	cbx->batch_verts_count = 0;
	cbx->batch_indices_count = 0;
}

RgTransform RT_GetBrushModelMatrix (entity_t *e)
{
	if (e == NULL)
	{
		const static RgTransform identity = RT_TRANSFORM_IDENTITY;
		return identity;
	}

	vec3_t e_angles;
	VectorCopy (e->angles, e_angles);
	e_angles[0] = -e_angles[0]; // stupid quake bug

	float model_matrix[16];
	IdentityMatrix (model_matrix);
	R_RotateForEntity (model_matrix, e->origin, e_angles);

	return RT_GetModelTransform (model_matrix);
}

static qboolean  RT_FindNearestTeleport (const RgGeometryUploadInfo *info, uint8_t *result, qboolean *potentially_mirror);
static RgFloat3D ApplyTransform (const RgTransform *transform, const vec3_t v);
static gltexture_t *RT_CanonicalLightTex (texture_t *base, int alt);
static void         RT_KeepEmissiveLightColor (vec3_t color);

typedef struct rt_uploadsurf_state_t
{
	int          entuniqueid;
	entity_t    *ent;
	qmodel_t    *model;
	msurface_t  *surf;
	gltexture_t *diffuse_tex;
	gltexture_t *light_tex;
	gltexture_t *lightmap_tex;
	qboolean     alpha_test;
	float        alpha;
	qboolean     use_zbias;
	qboolean     is_warp;
	qboolean     is_water;
	qboolean     is_acid;
	qboolean     is_teleport;
	qboolean     is_animated;
} rt_uploadsurf_state_t;

static void RT_EmitEmissiveWireTriangle (const RgFloat3D *p0, const RgFloat3D *p1, const RgFloat3D *p2)
{
	const static uint32_t tri_indices[6] = {0, 1, 1, 2, 2, 0};
	const uint32_t        cyan        = RT_PackColorToUint32 (0, 255, 255, 255);

	RgVertex vertices[3] = {0};

	vertices[0].position[0] = p0->data[0];
	vertices[0].position[1] = p0->data[1];
	vertices[0].position[2] = p0->data[2];
	vertices[1].position[0] = p1->data[0];
	vertices[1].position[1] = p1->data[1];
	vertices[1].position[2] = p1->data[2];
	vertices[2].position[0] = p2->data[0];
	vertices[2].position[1] = p2->data[1];
	vertices[2].position[2] = p2->data[2];

	for (int i = 0; i < 3; i++)
		vertices[i].packedColor = cyan;

	RgRasterizedGeometryUploadInfo info = {
		.renderType = RG_RASTERIZED_GEOMETRY_RENDER_TYPE_DEFAULT,
		.vertexCount = 3,
		.pVertices = vertices,
		.indexCount = countof (tri_indices),
		.pIndices = tri_indices,
		.transform = RT_TRANSFORM_IDENTITY,
		.color = RT_COLOR_WHITE,
		.material = RG_NO_MATERIAL,
		.pipelineState = RG_RASTERIZED_GEOMETRY_STATE_FORCE_LINE_LIST,
		.blendFuncSrc = 0,
		.blendFuncDst = 0,
	};

	RgResult r = rgUploadRasterizedGeometry (vulkan_globals.instance, &info, NULL, NULL);
	RG_CHECK (r);
}

typedef struct
{
	uint32_t packed;
	float    reach;
	qboolean light_styles;
	qboolean set;
} rt_surfacepack_entry_t;

/* Faces of brush entities: the face cannot be found by index, and its reach answer follows the
   entity, so the transform it was computed for is kept next to it. */
typedef struct
{
	const entity_t   *ent;
	const msurface_t *surf;
	const texture_t  *tex;
	RgTransform       transform;
	uint32_t          packed;
	float             reach;
	qboolean          light_styles;
	qboolean          set;
} rt_surfacepackent_entry_t;

#define RT_SURFACEPACK_ENT_SIZE 512

static rt_surfacepack_entry_t    *rt_surfacepack;
static rt_surfacepackent_entry_t  rt_surfacepack_ent[RT_SURFACEPACK_ENT_SIZE];
static const qmodel_t            *rt_surfacepack_model;
static int                        rt_surfacepack_size;

/*
================
RT_SurfacePacksReset

Every face of the world packs the same light styles each time it is drawn: the face does not
move, the lights its reach test reads change only with the map, and the test itself looks at
nothing but where those lights stand. The answers are therefore kept per face until the faces
themselves go, which RT_BrushClusterCacheReset marks off the render tasks.
================
*/
static void RT_SurfacePacksReset (void)
{
	Mem_Free (rt_surfacepack);
	rt_surfacepack = NULL;
	rt_surfacepack_model = NULL;
	rt_surfacepack_size = 0;
	memset (rt_surfacepack_ent, 0, sizeof (rt_surfacepack_ent));

	const qmodel_t *model = cl.worldmodel;

	if (!model || !model->surfaces || model->numsurfaces <= 0)
		return;

	rt_surfacepack = Mem_Alloc (sizeof (rt_surfacepack[0]) * (size_t) model->numsurfaces);
	rt_surfacepack_model = model;
	rt_surfacepack_size = model->numsurfaces;
}

static uint32_t RT_SurfacePackLightStyles (
	const rt_uploadsurf_state_t *s, const RgVertex *verts, int num_verts, const RgTransform *transform, qboolean light_styles, float reach)
{
	uint32_t packed = 0;
	int      slot   = 0;

	gltexture_t *light_tex = RT_CanonicalLightTex (s->surf->texinfo->texture, 0);

	if (!light_tex || !light_tex->rtlightstyles || !light_styles)
		return 0;

	vec3_t accum = {0.0f, 0.0f, 0.0f};
	for (int i = 0; i < num_verts; i++)
		VectorAdd (accum, verts[i].position, accum);
	vec3_t center;
	VectorScale (accum, 1.0f / num_verts, center);
	const RgFloat3D world_center = ApplyTransform (transform, center);

	for (int i = 0; i < MAXLIGHTMAPS && s->surf->styles[i] != 255; i++)
	{
		const int style = s->surf->styles[i];

		if (reach >= 0.0f)
		{
			const float dist = RT_NearestStyledLightDistance (style, world_center.data);

			if (dist < 0.0f || dist > reach)
				continue;
		}

		packed |= (uint32_t)(style + 1) << (slot * 8);
		if (++slot == 4)
			break;
	}

	return packed;
}

static uint32_t RT_PackSurfaceLightStyles (const rt_uploadsurf_state_t *s, const RgVertex *verts, int num_verts, const RgTransform *transform)
{
	const qboolean light_styles = CVAR_TO_BOOL (rt_light_styles);
	const float    reach        = CVAR_TO_FLOAT (rt_light_styles_reach);

	rt_surfacepack_entry_t *entry = NULL;

	/* Only a face of the world that stands still is answered out of the table: the reach of a
	   face of a moving model moves with it. */
	if (rt_surfacepack && s->surf && !s->ent && s->model == rt_surfacepack_model)
	{
		const int index = (int) (s->surf - rt_surfacepack_model->surfaces);

		if (index >= 0 && index < rt_surfacepack_size)
			entry = &rt_surfacepack[index];
	}

	if (entry && entry->set && entry->light_styles == light_styles && entry->reach == reach)
		return entry->packed;

	if (!entry && s->surf && s->ent && s->model && s->model->surfaces && transform)
	{
		/* Direct mapped, as the brush cluster cache is: a collision only costs the lookup. The
		   faces of a model are visited in order, so a stride index keeps them in step, and the
		   entity moves the window so that two models do not share it. */
		const size_t               index = (((size_t) (s->surf - s->model->surfaces)) + ((uintptr_t) s->ent >> 4)) % RT_SURFACEPACK_ENT_SIZE;
		rt_surfacepackent_entry_t *ententry = &rt_surfacepack_ent[index];

		if (ententry->set && ententry->ent == s->ent && ententry->surf == s->surf &&
		    ententry->tex == s->surf->texinfo->texture &&
		    ententry->light_styles == light_styles && ententry->reach == reach &&
		    memcmp (&ententry->transform, transform, sizeof (RgTransform)) == 0)
			return ententry->packed;

		const uint32_t entpacked = RT_SurfacePackLightStyles (s, verts, num_verts, transform, light_styles, reach);

		ententry->ent          = s->ent;
		ententry->surf         = s->surf;
		ententry->tex          = s->surf->texinfo->texture;
		ententry->transform    = *transform;
		ententry->packed       = entpacked;
		ententry->reach        = reach;
		ententry->light_styles = light_styles;
		ententry->set          = true;

		return entpacked;
	}

	const uint32_t packed = RT_SurfacePackLightStyles (s, verts, num_verts, transform, light_styles, reach);

	if (entry)
	{
		entry->packed       = packed;
		entry->reach        = reach;
		entry->light_styles = light_styles;
		entry->set          = true;
	}

	return packed;
}

static void RT_FlushBatch (cb_context_t *cbx, const rt_uploadsurf_state_t *s, uint32_t *brushpasses)
{
	if (cbx->batch_verts_count == 0 || cbx->batch_indices_count == 0)
	{
		return;
	}

    const RgVertex *vertices = cbx->batch_verts;
	const uint32_t *indices = cbx->batch_indices;
	const int       num_surf_verts = cbx->batch_verts_count;
	const int       num_surf_indices = cbx->batch_indices_count;

	// i.e. uploaded once at the level load
    const qboolean is_static_geom = (s->model == cl.worldmodel) && !s->is_warp && !s->is_animated;

#if 0
	float constant_factor = 0.0f, slope_factor = 0.0f;
	if (use_zbias)
	{
		if (vulkan_globals.depth_format == VK_FORMAT_D32_SFLOAT_S8_UINT || vulkan_globals.depth_format == VK_FORMAT_D32_SFLOAT)
		{
			constant_factor = -4.f;
			slope_factor = -0.125f;
		}
		else
		{
			constant_factor = -1.f;
			slope_factor = -0.25f;
		}
	}
	vkCmdSetDepthBias (cbx->cb, constant_factor, 0.0f, slope_factor);
#endif

	gltexture_t *diffuse_tex = r_lightmap_cheatsafe ? NULL : s->diffuse_tex;

	const qboolean is_teleport_portal =
		s->is_teleport && CVAR_TO_BOOL (rt_teleport_portals);

	if (is_teleport_portal && CVAR_TO_INT32 (rt_reflrefr_depth) > 0)
	{
		diffuse_tex = NULL;
	}

	float alpha = CLAMP (0.0f, s->alpha, 1.0f);
	uint8_t portalindex = 0;

	qboolean is_mirror = diffuse_tex && diffuse_tex->rtmirror;
	qboolean rasterize = (alpha < 1.0f) && !s->is_warp;

	if (rasterize)
	{
		// worldmodel must be uploaded only once
		assert (!is_static_geom);

		RgRasterizedGeometryUploadInfo info = {
			.renderType = RG_RASTERIZED_GEOMETRY_RENDER_TYPE_DEFAULT,
			.vertexCount = num_surf_verts,
			.pVertices = vertices,
			.indexCount = num_surf_indices,
			.pIndices = indices,
			.transform = RT_GetBrushModelMatrix (s->ent),
			.color = RT_COLOR_WHITE,
			.material = diffuse_tex ? diffuse_tex->rtmaterial : greytexture->rtmaterial,
			.pipelineState = RG_RASTERIZED_GEOMETRY_STATE_DEPTH_TEST,
			.blendFuncSrc = 0,
			.blendFuncDst = 0,
		};

		if (s->alpha_test)
		{
			info.pipelineState |= RG_RASTERIZED_GEOMETRY_STATE_ALPHA_TEST;
		}

		if (alpha < 1.0f)
		{
			info.color.data[3] = alpha;
			info.pipelineState |= RG_RASTERIZED_GEOMETRY_STATE_BLEND_ENABLE;
			info.blendFuncSrc = RG_BLEND_FACTOR_SRC_ALPHA;
			info.blendFuncDst = RG_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		}
		else
		{
			info.pipelineState |= RG_RASTERIZED_GEOMETRY_STATE_DEPTH_WRITE;
		}

		RgResult r = rgUploadRasterizedGeometry (vulkan_globals.instance, &info, NULL, NULL);
		RG_CHECK (r);
	}
	else
	{
		RgGeometryUploadInfo info = {
			.uniqueID = RT_GetBrushSurfUniqueId (s->entuniqueid, s->model, s->surf, 0),
			.flags = 
			    (is_mirror ? RG_GEOMETRY_UPLOAD_REFL_REFR_ALBEDO_MULTIPLY_BIT : 0) |
			    (is_teleport_portal ? RG_GEOMETRY_UPLOAD_REFL_REFR_ALBEDO_ADD_BIT : 0) |
			    // water and slime already churn through the RT wave normals
			    (s->is_warp && !s->is_water && !s->is_acid ? RG_GEOMETRY_UPLOAD_TURB_WARP_BIT : 0) |
                RG_GEOMETRY_UPLOAD_GENERATE_NORMALS_BIT,
			.geomType = is_static_geom ? RG_GEOMETRY_TYPE_STATIC : RG_GEOMETRY_TYPE_DYNAMIC,
			.passThroughType = 
			    is_mirror ? RG_GEOMETRY_PASS_THROUGH_TYPE_MIRROR :
			    s->is_water ? RG_GEOMETRY_PASS_THROUGH_TYPE_WATER_REFLECT_REFRACT :
			    s->is_acid ? RG_GEOMETRY_PASS_THROUGH_TYPE_ACID_REFLECT_REFRACT :
			    is_teleport_portal ? RG_GEOMETRY_PASS_THROUGH_TYPE_PORTAL :
			    // A fence texture keeps its alpha only in the traced path: the rasterized
			    // one is reserved for translucent surfaces, which a fence is not.
			    s->alpha_test ? RG_GEOMETRY_PASS_THROUGH_TYPE_ALPHA_TESTED :
		        RG_GEOMETRY_PASS_THROUGH_TYPE_OPAQUE,
			.visibilityType = RG_GEOMETRY_VISIBILITY_TYPE_WORLD_0,
			.vertexCount = num_surf_verts,
			.pVertices = vertices,
			.indexCount = num_surf_indices,
			.pIndices = indices,
			.layerColors =
				{
					RT_COLOR_WHITE,
				},
			.layerBlendingTypes =
				{
					RG_GEOMETRY_MATERIAL_BLEND_TYPE_OPAQUE,
				},
			.geomMaterial =
				{
					diffuse_tex ? diffuse_tex->rtmaterial : greytexture->rtmaterial,
				},
			.defaultRoughness = CVAR_TO_FLOAT (rt_brush_rough),
			.defaultMetallicity = CVAR_TO_FLOAT (rt_brush_metal),
			.defaultEmission = 0,
			.transform = RT_GetBrushModelMatrix (s->ent),
		};

		if (is_teleport_portal)
		{
			qboolean portal_is_mirror = false;

			if (RT_FindNearestTeleport (&info, &portalindex, &portal_is_mirror))
			{
				if (portal_is_mirror)
				{
					info.passThroughType = RG_GEOMETRY_PASS_THROUGH_TYPE_MIRROR;
				}
				else
				{
				    info.pPortalIndex = &portalindex;
				}
			}
		}

		RgResult r = rgUploadGeometry (vulkan_globals.instance, &info);
		RG_CHECK (r);
	}

	RT_ClearBatch (cbx);
	++(*brushpasses);
}

static void RT_TexturedAreaLightCenter (const RgTexturedAreaLightUploadInfo *lt, vec3_t out)
{
	float s = 0.0f, t = 0.0f;
	const int n = lt->numVerts;

	for (int i = 0; i < n; i++)
	{
		s += lt->uvVerts[i].data[0];
		t += lt->uvVerts[i].data[1];
	}

	const float inv = n > 0 ? 1.0f / (float) n : 0.0f;
	s *= inv;
	t *= inv;

	out[0] = lt->A.data[0] * s + lt->B.data[0] * t + lt->C.data[0];
	out[1] = lt->A.data[1] * s + lt->B.data[1] * t + lt->C.data[1];
	out[2] = lt->A.data[2] * s + lt->B.data[2] * t + lt->C.data[2];
}

static void RT_EmitEmissiveWirePolygon (const RgTexturedAreaLightUploadInfo *lt)
{
	RgFloat3D wv[MAX_TEXTURED_AREA_LIGHT_VERTS];
	const int n = lt->numVerts;

	if (n < 3)
		return;

	for (int i = 0; i < n; i++)
	{
		const float s = lt->uvVerts[i].data[0];
		const float t = lt->uvVerts[i].data[1];
		wv[i].data[0] = lt->A.data[0] * s + lt->B.data[0] * t + lt->C.data[0];
		wv[i].data[1] = lt->A.data[1] * s + lt->B.data[1] * t + lt->C.data[1];
		wv[i].data[2] = lt->A.data[2] * s + lt->B.data[2] * t + lt->C.data[2];
	}

	for (int i = 1; i < n - 1; i++)
		RT_EmitEmissiveWireTriangle (&wv[0], &wv[i], &wv[i + 1]);
}

static void RT_ScaleEmissiveLightColor (vec3_t color)
{
	VectorScale (color, RT_EMIS_INTENSITY_TO_RAW (CVAR_TO_FLOAT (rt_emis_light_intensity)), color);
	RT_FIXUP_LIGHT_INTENSITY (color, true);
}

/* rt_truelight 0 renders the original Quake light sources only (legacy entity lights,
   classic dlights, and model/sprite light_color spheres) and disables textured area
   lights -- the emissive-texture (TAL) lights -- entirely. rt_truelight 1 (default) and
   2 keep them on. rt_materials_only always keeps them on, as they are what that mode
   previews. */
static qboolean RT_AllowTexturedAreaLights (void)
{
	return CVAR_TO_BOOL (rt_materials_only) || CVAR_TO_FLOAT (rt_truelight) > 0.0f;
}

static void RT_UploadEmissiveLight (const RgTexturedAreaLightUploadInfo *light_info, qboolean is_static_geom,
                                    const msurface_t *surf, gltexture_t *light_tex)
{
	if (!is_static_geom)
	{
		rt_emis_stats.dynamic++;

		RgTexturedAreaLightUploadInfo li = *light_info;
		RT_ScaleEmissiveLightColor (li.color.data);

		/* Same clamp as a stored light gets: without it a light dimmed to nothing by its
		   animation frame falls below the keep threshold of the light manager and is dropped
		   from the light array, which costs the cluster lists an id they name. */
		RT_KeepEmissiveLightColor (li.color.data);

		RgResult r = rgUploadTexturedAreaLight (vulkan_globals.instance, &li);
		RG_CHECK (r);

		vec3_t center;
		RT_TexturedAreaLightCenter (&li, center);
		/* The geometry moved to get here, so the light is only promised the reach of a light of
		   a moving entity. */
		if (CVAR_TO_FLOAT (rt_cluster_dlights) != 0)
			RT_ClusterLightAdd (li.uniqueID, center, RT_ClusterLightReach ());

		if (CVAR_TO_BOOL (rt_debugemissive))
		{
			RT_EmitEmissiveWirePolygon (&li);
		}
	}
	else if (rt_wldlights_emissive_count < MAX_WORLDLIGHTS_COUNT)
	{
		const int index = rt_wldlights_emissive_count++;
		rt_wldlights_emissive[index]      = *light_info;
		rt_wldlights_emissive_surf[index] = surf;
		rt_wldlights_emissive_tex[index]  = light_tex;
		/* The stored light does not move, so the center it is placed by is derived here and
		   not once per frame. */
		RT_TexturedAreaLightCenter (&rt_wldlights_emissive[index], rt_wldlights_emissive_center[index]);
		rt_emis_stats.static_queued++;
	}
	else
	{
		rt_emis_stats.static_dropped++;

		static qboolean warned = false;
		if (!warned)
		{
			warned = true;
			Con_DWarning ("RT: emissive world lights exceeded MAX_WORLDLIGHTS_COUNT (%i)\n",
			              MAX_WORLDLIGHTS_COUNT);
		}
	}
}

static float RT_SurfaceLightStyleScale (const msurface_t *surf, const vec3_t center)
{
	const float reach = CVAR_TO_FLOAT (rt_light_styles_reach);
	float       scale = 1.0f;
	qboolean    dims  = false;

	for (int i = 0; i < MAXLIGHTMAPS && surf->styles[i] != 255; i++)
	{
		const int style = surf->styles[i];
		/* The style value is 8.8 fixed point. Normalised here so that the dimmest style
		   wins by comparison, exactly as the shader picks it for the visible material. */
		const float value = (float)d_lightstylevalue[style] * (1.0f / 256.0f);

		if (value >= 255.5f * (1.0f / 256.0f))
			continue;

		if (reach >= 0.0f)
		{
			const float dist = RT_NearestStyledLightDistance (style, center);

			if (dist < 0.0f || dist > reach)
				continue;
		}

		if (!dims || value < scale)
			scale = value;
		dims = true;
	}

	return dims ? scale : 1.0f;
}

/*
=================
RT_BuildWorldLightStyleAcceptance

Runs the reach test of RT_SurfaceLightStyleScale once per stored light and keeps the answer.
=================
*/
static void RT_BuildWorldLightStyleAcceptance (void)
{
	const float reach = CVAR_TO_FLOAT (rt_light_styles_reach);

	memset (rt_wldlights_style_accepted, 0, sizeof (rt_wldlights_style_accepted));

	for (int i = 0; i < rt_wldlights_emissive_count; i++)
	{
		const msurface_t *surf = rt_wldlights_emissive_surf[i];

		for (int k = 0; k < MAXLIGHTMAPS && surf->styles[k] != 255; k++)
		{
			if (reach < 0.0f)
			{
				/* No reach limit, every style of the surface is accepted. */
				rt_wldlights_style_accepted[i][k] = 1;
				continue;
			}

			const float dist = RT_NearestStyledLightDistance (surf->styles[k], rt_wldlights_emissive_center[i]);

			rt_wldlights_style_accepted[i][k] = (dist >= 0.0f && dist <= reach);
		}
	}

	rt_wldlights_style_accepted_dirty = false;
	rt_wldlights_style_accepted_reach = reach;
}

/* RT_SurfaceLightStyleScale for a stored world light: the reach test is the cached answer, the
   style value is read per frame because it is what animates. */
static float RT_WorldLightStyleScale (int index)
{
	const msurface_t *surf     = rt_wldlights_emissive_surf[index];
	const byte       *accepted = rt_wldlights_style_accepted[index];
	float             scale    = 1.0f;
	qboolean          dims     = false;

	for (int k = 0; k < MAXLIGHTMAPS && surf->styles[k] != 255; k++)
	{
		const float value = (float)d_lightstylevalue[surf->styles[k]] * (1.0f / 256.0f);

		if (value >= 255.5f * (1.0f / 256.0f))
			continue;

		if (!accepted[k])
			continue;

		if (!dims || value < scale)
			scale = value;
		dims = true;
	}

	return dims ? scale : 1.0f;
}

static qboolean RT_IsStaticWorldSurface (const rt_uploadsurf_state_t *s)
{
	return s->model == cl.worldmodel && !s->is_warp && !s->is_animated;
}

typedef struct rt_emissive_params_s
{
	RgMaterial material;
	float      meanEmiss;
	vec3_t     color;
	/* The texture glows over part of itself, so the light is built from polygons over those
	   extents instead of from the whole surface. */
	qboolean   glow;
	float      glow_uvmin[2];
	float      glow_uvmax[2];
	float      glow_mean;
} rt_emissive_params_t;

static qboolean RT_EmissiveLightParamsForTex(gltexture_t *light_tex, rt_emissive_params_t *p)
{
	if (!light_tex || !light_tex->rtislight)
		return false;

	if (!light_tex->rthaslightcolor && VectorLength (light_tex->rtemissivecolor) <= 0.0005f)
		return false;

	/* Whether a mask is there is a question about the material, not about how bright the mask is:
	   a mask that is black everywhere is still there, and the light built from it emits nothing
	   instead of glowing evenly over its whole polygon. */
	const qboolean has_mask = light_tex->rtemissivetex;

	/* The light has to glow where the surface glows, so a masked texture sends its material along
	   and the shader reads the mask at the point it sampled. Only a texture with no mask is
	   uniform over its polygon, and only there does meanEmiss describe the emission. */
	p->material  = has_mask ? light_tex->rtmaterial : RG_NO_MATERIAL;
	p->meanEmiss = has_mask ? light_tex->rtemissivemean : 1.0f;
	p->glow      = false;
	p->glow_mean = p->meanEmiss;

	if (light_tex->rthaslightcolor)
	{
		VectorCopy (light_tex->rtlightcolor, p->color);
	}
	else if (has_mask)
	{
		const float meanBase = light_tex->rtemissivemeanbase > 1e-6f ? light_tex->rtemissivemeanbase : 1e-6f;
		VectorScale (light_tex->rtemissivecolor, 1.0f / meanBase, p->color);
	}
	else
	{
		VectorCopy (light_tex->rtemissivecolor, p->color);
	}

	if (has_mask && light_tex->rtemissiveglowtex && light_tex->rtemisglowfrac > 1e-6f)
	{
		p->glow          = true;
		p->glow_uvmin[0] = light_tex->rtemisuvmin[0];
		p->glow_uvmin[1] = light_tex->rtemisuvmin[1];
		p->glow_uvmax[0] = light_tex->rtemisuvmax[0];
		p->glow_uvmax[1] = light_tex->rtemisuvmax[1];
		/* Area mean over the glow extents: the polygon lights cover only those, so they carry
		   the emission the whole surface would have had over the glowing part. */
		p->glow_mean     = light_tex->rtemissiveglow;
	}

	return true;
}

/*
=================
RT_CanonicalLightTex

The frame of an animated texture the emissive light (and its light styles) is built from, or NULL
when the texture cannot host a light. The light is collected anew every frame
(RT_CollectWorldEmissiveLights, RT_AddEmissiveLight), so the frame the light comes from must not
depend on cl.time: with R_TextureAnimation every animated texture in the map steps to the next
frame at the same instant (relative = (int)(cl.time * 10) % anim_total), so a time-dependent pick
makes the whole set of emissive lights change colour, change size or drop out together, which
shifts the light array and, with it, every cluster light list. See plan.md, the section on the flicker
of the whole scene.

alt selects the alternate animation the entity is currently in (ent->frame), which is a state change
and not an animation step, so it is kept. The frame inside that cycle is the first one that can host
a light, in ring order from the base texture.
=================
*/
static gltexture_t *RT_CanonicalLightTex (texture_t *base, int alt)
{
	if (alt && base && base->alternate_anims)
		base = base->alternate_anims;

	if (!base || !base->gltexture)
		return NULL;

	rt_emissive_params_t params;
	gltexture_t         *fallback = base->gltexture;

	for (texture_t *f = base; f; f = f->anim_next)
	{
		gltexture_t *ft = f->gltexture;

		if (ft && RT_EmissiveLightParamsForTex (ft, &params))
			return ft;

		if (ft && ft->rthasmaterial && !fallback->rthasmaterial)
			fallback = ft;

		if (!f->anim_next || f->anim_next == base)
			break;
	}

	return fallback;
}

/*
=================
RT_AnimatedLightTex

The frame of an animated texture the surface currently shows, resolved at cl.time, or the base
frame when the current frame cannot host a light. It is the time-dependent counterpart of
RT_CanonicalLightTex: it is read only to re-derive a stored light's emission per frame, never to
pick the light's identity or geometry, so the time dependency does not shift the light array.
=================
*/
static gltexture_t *RT_AnimatedLightTex (texture_t *base)
{
	if (!base || !base->gltexture)
		return NULL;

	gltexture_t *frame = R_TextureAnimation (base, 0)->gltexture;

	return (frame && frame->rthasmaterial) ? frame : base->gltexture;
}

static float RT_UvPolyArea (const RgFloat2D *p, int n)
{
	float a = 0.0f;

	for (int i = 0; i < n; i++)
	{
		const RgFloat2D p0 = p[i];
		const RgFloat2D p1 = p[(i + 1) % n];

		a += p0.data[0] * p1.data[1] - p1.data[0] * p0.data[1];
	}

	return 0.5f * a;
}

static int RT_ClipUvPolyEdge (const RgFloat2D *in, int n, int axis, float limit, qboolean keep_greater, RgFloat2D *out)
{
	int m = 0;

	for (int i = 0; i < n; i++)
	{
		const RgFloat2D a  = in[i];
		const RgFloat2D b  = in[(i + 1) % n];
		const float     da = a.data[axis] - limit;
		const float     db = b.data[axis] - limit;
		const qboolean  ia = keep_greater ? (da >= 0.0f) : (da <= 0.0f);
		const qboolean  ib = keep_greater ? (db >= 0.0f) : (db <= 0.0f);

		if (ia)
		{
			if (m >= MAX_TEXTURED_AREA_LIGHT_VERTS)
				return 0;
			out[m++] = a;
		}

		if (ia != ib)
		{
			const float t = da / (da - db);
			RgFloat2D   p;

			p.data[0] = a.data[0] + t * (b.data[0] - a.data[0]);
			p.data[1] = a.data[1] + t * (b.data[1] - a.data[1]);

			if (m >= MAX_TEXTURED_AREA_LIGHT_VERTS)
				return 0;
			out[m++] = p;
		}
	}

	return m;
}

/* Sutherland-Hodgman against one repetition of the glow extents. The result stays in the uv
   space of the surface fit, so A/B/C map it back to world without any world-space vertices. */
static int RT_ClipUvPolyToRect (const RgFloat2D *in, int n, const float *uvmin, const float *uvmax, RgFloat2D *out)
{
	RgFloat2D tmp[MAX_TEXTURED_AREA_LIGHT_VERTS];
	int       m;

	m = RT_ClipUvPolyEdge (in, n, 0, uvmin[0], true, tmp);
	if (m < 3)
		return 0;
	m = RT_ClipUvPolyEdge (tmp, m, 0, uvmax[0], false, out);
	if (m < 3)
		return 0;
	m = RT_ClipUvPolyEdge (out, m, 1, uvmin[1], true, tmp);
	if (m < 3)
		return 0;

	return RT_ClipUvPolyEdge (tmp, m, 1, uvmax[1], false, out);
}

typedef struct rt_uv_piece_s
{
	RgFloat2D verts[MAX_TEXTURED_AREA_LIGHT_VERTS];
	int       count;
} rt_uv_piece_t;

/*
=================
RT_SplitUvPolygon

Cuts a uv polygon into convex pieces of at most MAX_TEXTURED_AREA_LIGHT_VERTS corners, and returns
0 when there is no way to do it: a face whose uv collapses (no fit), or one with more corners than
pieces can be cut from. A piece is a run of consecutive corners of the input in winding order, so it
lies in the plane of the face and carries its uv: the pieces are the face, edges and all. That is
what a light reading the mask needs, since a square standing in for a face would read the mask over
a shape the surface does not have.
=================
*/
static int RT_SplitUvPolygon (const RgFloat2D *uv, int n, rt_uv_piece_t *out, int out_max)
{
	RgFloat2D rest[RT_MAX_UV_SPLIT_VERTS];
	int       num = 0;

	if (n < 3 || n > RT_MAX_UV_SPLIT_VERTS || out_max < 1)
		return 0;

	for (int i = 0; i < n; i++)
		rest[i] = uv[i];

	/* A run of consecutive corners of a convex polygon is convex, and so is what is left after the
	   chord from the first to the last of them is cut off; six at a time therefore cuts a face of
	   any size into a handful of pieces. */
	while (n > MAX_TEXTURED_AREA_LIGHT_VERTS)
	{
		if (num >= out_max || n < RT_UV_SPLIT_PIECE_VERTS + 2)
			return 0;

		rt_uv_piece_t *piece = &out[num++];

		piece->count = RT_UV_SPLIT_PIECE_VERTS;

		for (int i = 0; i < piece->count; i++)
			piece->verts[i] = rest[i];

		/* The remainder is the first corner, the last corner of the piece, and every corner after
		   it. It is read and written from the front, and the write never reaches the read. */
		int m = 1;

		for (int i = RT_UV_SPLIT_PIECE_VERTS - 1; i < n; i++)
			rest[m++] = rest[i];

		n = m;
	}

	if (num >= out_max)
		return 0;

	out[num].count = n;

	for (int i = 0; i < n; i++)
		out[num].verts[i] = rest[i];

	return num + 1;
}

static int RT_EmissiveGlowPolygons (const rt_uploadsurf_state_t *s, const RgFloat2D *surfuv, int vertcount,
                                    const vec3_t A, const vec3_t B, const rt_emissive_params_t *params,
                                    const RgTexturedAreaLightUploadInfo *base,
                                    RgTexturedAreaLightUploadInfo *out, int out_max)
{
	float uvmin[2] = {surfuv[0].data[0], surfuv[0].data[1]};
	float uvmax[2] = {surfuv[0].data[0], surfuv[0].data[1]};

	for (int i = 1; i < vertcount; i++)
	{
		for (int k = 0; k < 2; k++)
		{
			if (surfuv[i].data[k] < uvmin[k])
				uvmin[k] = surfuv[i].data[k];
			if (surfuv[i].data[k] > uvmax[k])
				uvmax[k] = surfuv[i].data[k];
		}
	}

	/* One unit of uv is one repetition of the texture, so the glow extents have to be cut out of
	   every repetition the face covers. */
	const int tileminx = (int)floor (uvmin[0]);
	const int tilemaxx = (int)fmax ((double)ceil (uvmax[0]) - 1.0, (double)tileminx);
	const int tileminy = (int)floor (uvmin[1]);
	const int tilemaxy = (int)fmax ((double)ceil (uvmax[1]) - 1.0, (double)tileminy);

	const int tiles = (tilemaxx - tileminx + 1) * (tilemaxy - tileminy + 1);

	if (tiles <= 0 || tiles > RT_MAX_EMISSIVE_POLYS_PER_FACE)
		return 0;

	vec3_t ab;
	CrossProduct (A, B, ab);

	int num = 0;

	for (int ty = tileminy; ty <= tilemaxy && num < out_max; ty++)
	{
		for (int tx = tileminx; tx <= tilemaxx && num < out_max; tx++)
		{
			const float uvmin_t[2] = {(float)tx + params->glow_uvmin[0], (float)ty + params->glow_uvmin[1]};
			const float uvmax_t[2] = {(float)tx + params->glow_uvmax[0], (float)ty + params->glow_uvmax[1]};
			RgFloat2D   clipped[MAX_TEXTURED_AREA_LIGHT_VERTS];
			const int   n = RT_ClipUvPolyToRect (surfuv, vertcount, uvmin_t, uvmax_t, clipped);

			if (n < 3)
				continue;

			const float uvare = fabs (RT_UvPolyArea (clipped, n));

			if (uvare <= 1e-9f)
				continue;

			RgTexturedAreaLightUploadInfo li = *base;

			for (int i = 0; i < n; i++)
				li.uvVerts[i] = clipped[i];

			li.numVerts  = n;
			li.area      = uvare * VectorLength (ab);
			li.meanEmiss = params->glow_mean;
			li.uniqueID  = RT_GetBrushSurfUniqueId (s->entuniqueid, s->model, s->surf, (uint64_t)(num + 1));

			out[num++] = li;
		}
	}

	return num;
}

static void RT_AddEmissiveLight (const rt_uploadsurf_state_t *s)
{
	if (!RT_AllowTexturedAreaLights ())
		return;

	gltexture_t *light_tex = s->light_tex ? s->light_tex : s->diffuse_tex;
	rt_emis_watch_t *watch = RT_EmisWatch (light_tex ? light_tex->name : NULL);

	rt_emis_stats.surfaces++;
	if (watch)
		watch->surfaces++;

	rt_emissive_params_t params;

	if (!RT_EmissiveLightParamsForTex (light_tex, &params))
	{
		if (!light_tex || !light_tex->rtislight)
		{
			rt_emis_stats.no_material++;
			if (watch)
				watch->no_material++;
			RT_EmisNoteSkip (light_tex ? light_tex->name : "<no texture>", "no light material");
		}
		else
		{
			rt_emis_stats.no_color++;
			if (watch)
				watch->no_color++;
			RT_EmisNoteSkip (light_tex->name, "no light color");
		}
		return;
	}

	const qboolean is_static_geom = RT_IsStaticWorldSurface (s);

	/* Whether the light reads a mask shapes it (cut into pieces of the face, or a square that
	   carries no mask), so it is asked of the canonical frame and not of the animation. */
	const qboolean light_masked = (params.material != RG_NO_MATERIAL);

	/* Emission follows the animation, identity does not. s->diffuse_tex is the frame the
	   visible pass renders of this surface, so a surface that steps to another frame steps its
	   light with it. Deliberately no fallback to the base frame the light was picked from: the
	   base is the lit frame of a blink, and reading it back would leave the off frame glowing.
	   A frame that cannot light for itself only dims the light -- dropping it would take an id
	   out of every cluster list naming it. */
	gltexture_t *frame_tex = s->diffuse_tex;

	if (!is_static_geom && frame_tex && frame_tex != light_tex)
	{
		rt_emissive_params_t frame_params;

		if (RT_EmissiveLightParamsForTex (frame_tex, &frame_params))
		{
			params.material  = frame_params.material;
			params.meanEmiss = frame_params.meanEmiss;
			VectorCopy (frame_params.color, params.color);
		}
		else
		{
			rt_emis_stats.frame_off++;
			if (watch)
				watch->frame_off++;
			RT_EmisNoteSkip (frame_tex->name, "dark animation frame");
			VectorCopy (vec3_origin, params.color);
		}
	}

	const RgTransform transf = RT_GetBrushModelMatrix (s->ent);
	const int        vertcount = s->surf->numedges;
	const RgVertex  *verts = rtallbrushvertices + s->surf->vbo_firstvert;

	if (vertcount < 3)
	{
		rt_emis_stats.degenerate++;
		if (watch)
			watch->degenerate++;
		return;
	}

	vec3_t accum_center = {0, 0, 0};
	vec3_t accum_normal = {0, 0, 0};
	float  total_area = 0.0f;
	int    num_tris = 0;

	float ss = 0.0f, st = 0.0f, tt = 0.0f, ssum = 0.0f, tsum = 0.0f;
	vec3_t sx = {0, 0, 0}, tx = {0, 0, 0}, psum = {0, 0, 0};

	for (int i = 0; i < vertcount; i++)
	{
		RgFloat3D    p = ApplyTransform (&transf, verts[i].position);
		const float  su = verts[i].texCoord[0];
		const float  tv = verts[i].texCoord[1];

		ss += su * su;
		st += su * tv;
		tt += tv * tv;
		ssum += su;
		tsum += tv;

		VectorMA (sx, su, p.data, sx);
		VectorMA (tx, tv, p.data, tx);
		VectorAdd (psum, p.data, psum);

		if (i >= 2)
		{
			const RgFloat3D p0 = ApplyTransform (&transf, verts[0].position);
			const RgFloat3D p1 = ApplyTransform (&transf, verts[i - 1].position);
			const RgFloat3D p2 = p;

			vec3_t e1, e2, cross;
			VectorSubtract (p1.data, p0.data, e1);
			VectorSubtract (p2.data, p0.data, e2);
			CrossProduct (e1, e2, cross);

			const float area = 0.5f * VectorLength (cross);
			if (area <= 0.0f)
				continue;

			vec3_t tri_center;
			VectorAdd (p0.data, p1.data, tri_center);
			VectorAdd (tri_center, p2.data, tri_center);
			VectorScale (tri_center, 1.0f / 3.0f, tri_center);

			VectorMA (accum_center, area, tri_center, accum_center);
			VectorAdd (accum_normal, cross, accum_normal);
			total_area += area;
			num_tris++;
		}
	}

	if (num_tris == 0 || total_area <= 0.0f)
	{
		rt_emis_stats.degenerate++;
		if (watch)
			watch->degenerate++;
		return;
	}

	if (light_tex->rtlightstyles && CVAR_TO_BOOL (rt_light_styles))
	{
		vec3_t center;
		VectorScale (accum_center, 1.0f / total_area, center);

		const float style_scale = RT_SurfaceLightStyleScale (s->surf, center);

		if (watch)
		{
			if (style_scale < watch->min_style_scale)
				watch->min_style_scale = style_scale;
			if (watch->style_count == 0)
			{
				for (int i = 0; i < MAXLIGHTMAPS && s->surf->styles[i] != 255; i++)
					watch->styles[watch->style_count++] = s->surf->styles[i];
			}
		}

		if (!is_static_geom)
		{
			VectorScale (params.color, style_scale, params.color);

			if (style_scale <= 0.0f)
			{
				rt_emis_stats.style_off++;
				if (watch)
					watch->style_off++;
				RT_EmisNoteSkip (light_tex->name, "lightstyle off");
				return;
			}
		}
	}

	vec3_t normal;
	{
		vec3_t accum_normal_dir;
		VectorCopy (accum_normal, accum_normal_dir);
		VectorNormalize (accum_normal_dir);

		if (s->surf->plane != NULL)
		{
			vec3_t pn;
			VectorCopy (s->surf->plane->normal, pn);
			if (s->surf->flags & SURF_PLANEBACK)
			{
				pn[0] = -pn[0];
				pn[1] = -pn[1];
				pn[2] = -pn[2];
			}
			normal[0] = transf.matrix[0][0] * pn[0] + transf.matrix[0][1] * pn[1] + transf.matrix[0][2] * pn[2];
			normal[1] = transf.matrix[1][0] * pn[0] + transf.matrix[1][1] * pn[1] + transf.matrix[1][2] * pn[2];
			normal[2] = transf.matrix[2][0] * pn[0] + transf.matrix[2][1] * pn[1] + transf.matrix[2][2] * pn[2];
			VectorNormalize (normal);
		}
		else
		{
			VectorCopy (accum_normal_dir, normal);
		}
	}

	qboolean fit_ok = false;
	vec3_t   A = {0, 0, 0}, B = {0, 0, 0}, C = {0, 0, 0};

	{
		float m[3][3] = {
			{ ss, st, ssum },
			{ st, tt, tsum },
			{ ssum, tsum, (float) vertcount },
		};
		float rhs[3][3] = {
			{ sx[0], sx[1], sx[2] },
			{ tx[0], tx[1], tx[2] },
			{ psum[0], psum[1], psum[2] },
		};

		fit_ok = true;
		for (int col = 0; col < 3 && fit_ok; col++)
		{
			int piv = col;
			for (int row = col + 1; row < 3; row++)
			{
				if (fabs (m[row][col]) > fabs (m[piv][col]))
					piv = row;
			}
			if (fabs (m[piv][col]) < 1e-9f)
			{
				fit_ok = false;
				break;
			}
			if (piv != col)
			{
				for (int k = 0; k < 3; k++)
				{
					float tmp = m[col][k];
					m[col][k] = m[piv][k];
					m[piv][k] = tmp;
					tmp = rhs[col][k];
					rhs[col][k] = rhs[piv][k];
					rhs[piv][k] = tmp;
				}
			}
			const float inv = 1.0f / m[col][col];
			for (int k = 0; k < 3; k++)
			{
				m[col][k] *= inv;
				rhs[col][k] *= inv;
			}
			for (int row = 0; row < 3; row++)
			{
				if (row == col)
					continue;
				const float factor = m[row][col];
				if (fabs (factor) < 1e-10f)
					continue;
				for (int k = 0; k < 3; k++)
				{
					m[row][k] -= factor * m[col][k];
					rhs[row][k] -= factor * rhs[col][k];
				}
			}
		}

		if (fit_ok)
		{
			A[0] = rhs[0][0]; A[1] = rhs[0][1]; A[2] = rhs[0][2];
			B[0] = rhs[1][0]; B[1] = rhs[1][1]; B[2] = rhs[1][2];
			C[0] = rhs[2][0]; C[1] = rhs[2][1]; C[2] = rhs[2][2];
		}
	}

	RgTexturedAreaLightUploadInfo light_info = {0};
	light_info.uniqueID  = RT_GetBrushSurfUniqueId (s->entuniqueid, s->model, s->surf, 0);
	light_info.material  = params.material;
	light_info.meanEmiss = params.meanEmiss;
	light_info.area      = total_area;
	light_info.fit       = 0;
	light_info.isStatic  = is_static_geom ? 1 : 0;
	VectorCopy (normal, light_info.normal.data);

	qboolean poly_ok = false;

	/* A light reading the mask has to keep the surface's own geometry, so the surface's uv is
	   kept aside whether or not a single polygon can hold the face. */
	const qboolean uv_ok = (vertcount <= RT_MAX_UV_SPLIT_VERTS);
	RgFloat2D      surfuv[RT_MAX_UV_SPLIT_VERTS];

	if (uv_ok)
	{
		for (int i = 0; i < vertcount; i++)
		{
			surfuv[i].data[0] = verts[i].texCoord[0];
			surfuv[i].data[1] = verts[i].texCoord[1];
		}
	}

	if (fit_ok && vertcount <= MAX_TEXTURED_AREA_LIGHT_VERTS)
	{
		for (int i = 0; i < vertcount; i++)
			light_info.uvVerts[i] = surfuv[i];

		light_info.numVerts = vertcount;
		VectorCopy (A, light_info.A.data);
		VectorCopy (B, light_info.B.data);
		VectorCopy (C, light_info.C.data);
		poly_ok = true;
	}

	/* A masked surface too complex for one light is cut into the convex pieces it is made of,
	   each with the uv of its own corners, and each becomes a light of its own. Cutting is
	   preferred over the square below because the square would read the mask over a shape the
	   surface does not have, and it is preferred over dropping the light because the surface
	   does glow, only not evenly. */
	if (!poly_ok && fit_ok && uv_ok && light_masked)
	{
		rt_uv_piece_t pieces[RT_MAX_UV_SPLIT_PIECES];
		const int     num = RT_SplitUvPolygon (surfuv, vertcount, pieces, RT_MAX_UV_SPLIT_PIECES);

		if (num > 0)
		{
			vec3_t ab;
			CrossProduct (A, B, ab);

			for (int i = 0; i < num; i++)
			{
				RgTexturedAreaLightUploadInfo piece = light_info;

				for (int k = 0; k < pieces[i].count; k++)
					piece.uvVerts[k] = pieces[i].verts[k];

				piece.numVerts = pieces[i].count;
				piece.area     = (float) fabs (RT_UvPolyArea (pieces[i].verts, pieces[i].count)) * VectorLength (ab);
				piece.uniqueID = RT_GetBrushSurfUniqueId (s->entuniqueid, s->model, s->surf, (uint64_t) (i + 1));
				piece.fit      = 1;
				VectorCopy (A, piece.A.data);
				VectorCopy (B, piece.B.data);
				VectorCopy (C, piece.C.data);
				VectorCopy (params.color, piece.color.data);

				RT_UploadEmissiveLight (&piece, is_static_geom, s->surf, light_tex);
			}

			if (watch)
				watch->lights += num;

			return;
		}
	}

	/* What is left is a face no single polygon holds: one with more corners than a light carries,
	   or one whose uv collapses (no fit), and a masked one with more corners than can be cut up.
	   A square over the surface's own area keeps the light's energy, which is all an unmasked
	   texture needs. */
	if (!poly_ok)
	{
		/* The square stands for the surface and carries a uv of its own, so the mask cannot be
		   read through it: a sample of it would land on a point of the texture the surface does
		   not cover, and a face whose uv the light cannot follow, or one larger than the cut
		   takes, would flicker with that sample instead of glowing evenly. It keeps the mean of
		   the mask instead, which is the emission an unmasked light carries as well. */
		light_info.material = RG_NO_MATERIAL;

		VectorScale (accum_center, 1.0f / total_area, accum_center);

		const float L = sqrt (total_area);

		vec3_t axis = {0, 0, 1};
		if (fabs (normal[2]) > 0.9f)
			RT_VEC3_SET (axis, 1, 0, 0);

		vec3_t u, v;
		CrossProduct (axis, normal, u);
		VectorNormalize (u);
		CrossProduct (normal, u, v);

		vec3_t corner;
		VectorMA (accum_center, -0.5f * L, u, corner);
		VectorMA (corner, -0.5f * L, v, corner);

		light_info.numVerts = 4;
		light_info.uvVerts[0].data[0] = 0.0f; light_info.uvVerts[0].data[1] = 0.0f;
		light_info.uvVerts[1].data[0] = 1.0f; light_info.uvVerts[1].data[1] = 0.0f;
		light_info.uvVerts[2].data[0] = 1.0f; light_info.uvVerts[2].data[1] = 1.0f;
		light_info.uvVerts[3].data[0] = 0.0f; light_info.uvVerts[3].data[1] = 1.0f;

		VectorScale (u, L, light_info.A.data);
		VectorScale (v, L, light_info.B.data);
		VectorCopy (corner, light_info.C.data);
	}

	light_info.fit = fit_ok ? 1 : 0;

	VectorCopy (params.color, light_info.color.data);

	if (watch)
		watch->lights++;

	if (params.glow)
	{
		if (poly_ok)
		{
			RgTexturedAreaLightUploadInfo polys[RT_MAX_EMISSIVE_POLYS_PER_FACE];
			const int num = RT_EmissiveGlowPolygons (s, light_info.uvVerts, light_info.numVerts, A, B, &params,
			                                        &light_info, polys, RT_MAX_EMISSIVE_POLYS_PER_FACE);

			if (num > 0)
			{
				rt_emis_stats.glow_faces++;
				rt_emis_stats.glow_lights += num;

				for (int i = 0; i < num; i++)
					RT_UploadEmissiveLight (&polys[i], is_static_geom, s->surf, light_tex);

				return;
			}
		}

		/* No glow polygon survived (too many repetitions, or a surface that has no uv fit):
		   the whole surface keeps the light instead. */
		rt_emis_stats.glow_fallback++;
	}

	RT_UploadEmissiveLight (&light_info, is_static_geom, s->surf, light_tex);
}

/*
================
RT_OwnedFacesReset / RT_FaceOwnedBySubmodel

Whether a face of the world belongs to an inline submodel (a door, a platform) is a property
of the BSP, so it is answered once per map instead of walking the submodel table once per
face: the collector asks it for every face, and the world of ad_mountain has 393 submodels
over 25k faces. Like the surface packs, this table is sized off the render tasks by
RT_BrushClusterCacheReset; the fallback keeps the answer when it was not built.
================
*/
static uint8_t        *rt_ownedfaces;
static const qmodel_t *rt_ownedfaces_model;
static int             rt_ownedfaces_count;

static void RT_OwnedFacesReset (void)
{
	Mem_Free (rt_ownedfaces);
	rt_ownedfaces = NULL;
	rt_ownedfaces_model = NULL;
	rt_ownedfaces_count = 0;

	const qmodel_t *model = cl.worldmodel;

	if (!model || !model->surfaces || !model->submodels || model->numsubmodels <= 0 || model->numsurfaces <= 0)
		return;

	rt_ownedfaces = Mem_Alloc ((size_t) model->numsurfaces);

	for (int j = 1; j < model->numsubmodels; j++)
	{
		const dmodel_t *sm    = &model->submodels[j];
		int             first = sm->firstface;
		int             last  = sm->firstface + sm->numfaces;

		if (first < 0)
			first = 0;
		if (last > model->numsurfaces)
			last = model->numsurfaces;
		if (last > first)
			memset (rt_ownedfaces + first, 1, (size_t) (last - first));
	}

	rt_ownedfaces_model = model;
	rt_ownedfaces_count = model->numsurfaces;
}

static qboolean RT_FaceOwnedBySubmodel (const qmodel_t *model, int surfindex)
{
	if (!model)
		return false;

	if (rt_ownedfaces && model == rt_ownedfaces_model && surfindex >= 0 && surfindex < rt_ownedfaces_count)
		return rt_ownedfaces[surfindex] != 0;

	for (int j = 1; j < model->numsubmodels; j++)
	{
		const dmodel_t *sm = &model->submodels[j];

		if (surfindex >= sm->firstface && surfindex < sm->firstface + sm->numfaces)
			return true;
	}

	return false;
}

static void RT_CollectWorldEmissiveLights (void)
{
	qmodel_t *model = cl.worldmodel;

	for (int i = 0; i < model->numsurfaces; i++)
	{
		if (RT_FaceOwnedBySubmodel (model, i))
			continue;

		msurface_t *surf = &model->surfaces[i];
		texture_t  *t    = surf->texinfo->texture;

		if (!t || !t->gltexture)
			continue;

		if (surf->flags & (SURF_DRAWSKY | SURF_NOTEXTURE))
			continue;

		gltexture_t *light_tex = RT_CanonicalLightTex (t, 0);

		if (!light_tex)
			continue;

		rt_uploadsurf_state_t state = {
			.entuniqueid = ENT_UNIQUEID_WORLD,
			.ent = NULL,
			.model = model,
			.surf = surf,
			.diffuse_tex = light_tex,
			.light_tex = light_tex,
			.lightmap_tex = (surf->lightmaptexturenum >= 0) ? lightmaps[surf->lightmaptexturenum].texture : greytexture,
			.alpha = 1.0f,
		};

		RT_AddEmissiveLight (&state);
	}
}

/*
================
RT_RecollectWorldEmissiveLights

Rebuilds the world's emissive light list from the current gltexture state
without re-uploading the world geometry (the qr light editor calls it after a
material was re-synthesized in place, so edits of is_light / light_color /
light_brightness / the emissive mask reach the lights of the next frame).
================
*/
void RT_RecollectWorldEmissiveLights (void)
{
	rt_wldlights_emissive_count = 0;
	/* The list is about to be collected again, so the cached reach answers for
	   the old entries are void. */
	rt_wldlights_style_accepted_dirty = true;

	RT_CollectWorldEmissiveLights ();
}

#define RT_BRUSHCLUSTER_CACHE_SIZE 256

typedef struct
{
	const entity_t *ent;
	int             vbo_firstvert;
	vec3_t          origin;
	int             cluster;
} rt_brushcluster_cacheentry_t;

static rt_brushcluster_cacheentry_t rt_brushcluster_cache[RT_BRUSHCLUSTER_CACHE_SIZE];

void RT_BrushClusterCacheReset (void)
{
	memset (rt_brushcluster_cache, 0, sizeof (rt_brushcluster_cache));

	/* New faces and a new light set: the per surface answers of the old map are void. Both
	   tables are sized here, where no render task runs, and never during a frame. */
	RT_SurfacePacksReset ();
	RT_OwnedFacesReset ();
}

static int RT_ResolveBrushSurfCluster (const rt_uploadsurf_state_t *s, const RgVertex *verts, int numverts)
{
	const size_t idx = ((size_t) s->ent / sizeof (void *) + (size_t) s->surf->vbo_firstvert) % RT_BRUSHCLUSTER_CACHE_SIZE;
	rt_brushcluster_cacheentry_t *ce = &rt_brushcluster_cache[idx];

	if (ce->ent == s->ent && ce->vbo_firstvert == s->surf->vbo_firstvert &&
		VectorCompare (ce->origin, s->ent->origin))
	{
		return ce->cluster;
	}

	vec3_t centroid = {0, 0, 0};

	for (int i = 0; i < numverts; i++)
	{
		centroid[0] += verts[i].position[0];
		centroid[1] += verts[i].position[1];
		centroid[2] += verts[i].position[2];
	}

	if (numverts > 0)
		VectorScale (centroid, 1.0f / (float) numverts, centroid);
	VectorAdd (centroid, s->ent->origin, centroid);

	ce->ent = s->ent;
	ce->vbo_firstvert = s->surf->vbo_firstvert;
	VectorCopy (s->ent->origin, ce->origin);
	ce->cluster = RT_ResolvePointCluster (centroid);

	return ce->cluster;
}

static void RT_BatchSurface (cb_context_t *cbx, const rt_uploadsurf_state_t *s, uint32_t *brushpasses)
{
	int num_surf_verts = s->surf->numedges;
	int num_surf_indices = R_NumTriangleIndicesForSurf (num_surf_verts);

	if (s->model != cl.worldmodel)
		RT_AddEmissiveLight (s);

	if (cbx->batch_indices_count + num_surf_indices > MAX_BATCH_INDICES ||
		cbx->batch_verts_count + num_surf_verts > MAX_BATCH_VERTS)
	{
		RT_FlushBatch (cbx, s, brushpasses);
	}

	R_TriangleIndicesForSurf (cbx->batch_verts_count, num_surf_verts, &cbx->batch_indices[cbx->batch_indices_count]);
	RgVertex *batch_verts = &cbx->batch_verts[cbx->batch_verts_count];
	memcpy (batch_verts, rtallbrushvertices + s->surf->vbo_firstvert, sizeof (RgVertex) * num_surf_verts);

	if (s->ent && s->model != cl.worldmodel)
	{
		const uint32_t cluster = (uint32_t) RT_ResolveBrushSurfCluster (s, batch_verts, num_surf_verts);

		for (int i = 0; i < num_surf_verts; i++)
			batch_verts[i].cluster = cluster;
	}

	{
		const RgTransform transform = RT_GetBrushModelMatrix (s->ent);
		const uint32_t    packed_styles = RT_PackSurfaceLightStyles (s, batch_verts, num_surf_verts, &transform);

		if (packed_styles != 0)
			for (int i = 0; i < num_surf_verts; i++)
				batch_verts[i].lightStyles = packed_styles;
	}

	cbx->batch_indices_count += num_surf_indices;
	cbx->batch_verts_count += num_surf_verts;
}

/*
================
GL_WaterAlphaForEntitySurface -- ericw

Returns the water alpha to use for the entity and surface combination.
================
*/
float GL_WaterAlphaForEntitySurface (entity_t *ent, msurface_t *s)
{
	float entalpha;
	if (r_lightmap_cheatsafe)
		entalpha = 1;
	else if (ent == NULL || ent->alpha == ENTALPHA_DEFAULT)
		entalpha = GL_WaterAlphaForSurface (s);
	else
		entalpha = ENTALPHA_DECODE (ent->alpha);
	return entalpha;
}

/*
================
R_DrawTextureChains_ShowTris -- johnfitz
================
*/
void R_DrawTextureChains_ShowTris (cb_context_t *cbx, qmodel_t *model, texchain_t chain)
{
	int         i;
	msurface_t *s;
	texture_t  *t;
	float       color[] = {1.0f, 1.0f, 1.0f};
	const float alpha = 1.0f;

    const static RgTransform tr = RT_TRANSFORM_IDENTITY;

	for (i = 0; i < model->numtextures; i++)
	{
		t = model->textures[i];
		if (!t)
			continue;

		for (s = t->texturechains[chain]; s; s = s->texturechains[chain])
			DrawGLPoly (
				cbx, RT_UNIQUEID_DONTCARE,
				s->polys, color, alpha, 
				&tr, NULL, 
				CVAR_TO_BOOL (r_showtris) ? DRAW_GL_POLY_TYPE_SHOWTRI : DRAW_GL_POLY_TYPE_SHOWTRI_NODEPTH);
	}
}

/*
================
RT_UploadStatesMatch

True when two consecutive surfaces of a world texture chain produce the same upload, so
that the second one continues the batch of the first.

Everything the upload of the batch reads is compared. Left out are the surface, whose
vertex range, unique id and packed light styles travel per surface inside the batch, and
the lightmap page, which the upload does not read: it names a texture of the rasterizer's
lightmap pass, and no per surface state is taken from it.
================
*/
static qboolean RT_UploadStatesMatch (const rt_uploadsurf_state_t *cur, const rt_uploadsurf_state_t *last)
{
	if (cur->is_teleport || last->is_teleport)
	{
		// the portal of a teleport surface is resolved into the uploaded geometry
		return false;
	}

	return cur->entuniqueid == last->entuniqueid &&
	       cur->ent == last->ent &&
	       cur->model == last->model &&
	       cur->diffuse_tex == last->diffuse_tex &&
	       cur->light_tex == last->light_tex &&
	       cur->alpha_test == last->alpha_test &&
	       cur->alpha == last->alpha &&
	       cur->use_zbias == last->use_zbias &&
	       cur->is_warp == last->is_warp &&
	       cur->is_water == last->is_water &&
	       cur->is_acid == last->is_acid &&
	       cur->is_animated == last->is_animated;
}

/*
================
RT_ShouldSplitBatch

Decides whether the surface being batched opens a new upload.

rt_world_batch_merge 0 keeps the split that history left in these two chains: a new
lightmap page, and -- a comparison whose sign was never fixed -- an unchanged alpha, which
cuts a separate upload for nearly every water and animated surface of a map although their
vertex data, material and flags are the same. With it 1 the batch is split only when
something the upload carries really differs, so the surfaces of one texture share one
upload.
================
*/
static qboolean RT_ShouldSplitBatch (const rt_uploadsurf_state_t *cur, const rt_uploadsurf_state_t *last, qboolean legacy_split)
{
	if (CVAR_TO_BOOL (rt_world_batch_merge))
		return !RT_UploadStatesMatch (cur, last);

	return legacy_split;
}

/*
================
R_DrawTextureChains_Water -- johnfitz
================
*/
void R_DrawTextureChains_Water (cb_context_t *cbx, qmodel_t *model, entity_t *ent, texchain_t chain, int entuniqueid)
{
	int                   i;
	msurface_t           *s;
	texture_t            *t;
	rt_uploadsurf_state_t last_state = {0};

	uint32_t brushpasses = 0;
	for (i = 0; i < model->numtextures; ++i)
	{
		t = model->textures[i];

		if (!t || !t->texturechains[chain] || !(t->texturechains[chain]->flags & SURF_DRAWTURB))
			continue;
		
		RT_ClearBatch (cbx);

		/* All three name a property of the texture: R_TextureAnimation is read at frame 0 of
		   the cycle and cl.time is fixed for the frame, so they do not vary between the
		   surfaces of this chain. */
		gltexture_t *anim_tex  = R_TextureAnimation (t, 0)->gltexture;
		gltexture_t *surf_tex  = (anim_tex && anim_tex->rthasmaterial) ? anim_tex : t->gltexture;
		gltexture_t *light_tex = RT_CanonicalLightTex (t, 0);

		for (s = t->texturechains[chain]; s; s = s->texturechains[chain])
		{
			if (model != cl.worldmodel)
			{
				// ericw -- this is copied from R_DrawSequentialPoly.
				// If the poly is not part of the world we have to
				// set this flag
				Atomic_StoreUInt32 (&t->update_warp, true); // FIXME: one frame too late!
			}

			rt_uploadsurf_state_t cur_state = {
				.entuniqueid = entuniqueid,
				.ent = ent,
				.model = model,
				.surf = s,
				.diffuse_tex = surf_tex,
				.light_tex = light_tex ? light_tex : surf_tex,
				.lightmap_tex = (s->lightmaptexturenum >= 0) ? lightmaps[s->lightmaptexturenum].texture : greytexture,
				.alpha_test = false,
				.alpha = GL_WaterAlphaForEntitySurface (ent, s),
				.use_zbias = false,
				.is_warp = true,
				.is_water = s->flags & SURF_DRAWWATER,
				.is_acid = s->flags & SURF_DRAWSLIME,
				.is_teleport = (s->flags & SURF_DRAWTELE),
			};

			if (RT_ShouldSplitBatch (&cur_state, &last_state,
			                         cur_state.lightmap_tex != last_state.lightmap_tex ||
			                         fabsf(cur_state.alpha - last_state.alpha) < 0.001f))
			{
				RT_FlushBatch (cbx, &last_state, &brushpasses);
			}

			RT_BatchSurface (cbx, &cur_state, &brushpasses);
			last_state = cur_state;
		}

		RT_FlushBatch (cbx, &last_state, &brushpasses);
	}

	Atomic_AddUInt32 (&rs_brushpasses, brushpasses);
}

void R_DrawTextureChains_Animated (cb_context_t *cbx, qmodel_t *model)
{
	int                   i;
	msurface_t           *s;
	texture_t            *t;
	rt_uploadsurf_state_t last_state = {0};

	uint32_t brushpasses = 0;
	for (i = 0; i < model->numtextures; ++i)
	{
		t = model->textures[i];

		if (!t || !t->texturechains[chain_world] || !t->anim_total)
			continue;
		if (t->texturechains[chain_world]->flags & (SURF_DRAWTURB | SURF_DRAWTILED | SURF_NOTEXTURE))
			continue;

		RT_ClearBatch (cbx);

		gltexture_t *diffuse_tex = R_TextureAnimation (t, 0)->gltexture;
		if (!diffuse_tex)
			diffuse_tex = t->gltexture;

		gltexture_t *light_tex = t->gltexture;
		if (diffuse_tex->rthasmaterial)
			light_tex = diffuse_tex;

		const qboolean alpha_test = (t->texturechains[chain_world]->flags & SURF_DRAWFENCE) != 0;

		for (s = t->texturechains[chain_world]; s; s = s->texturechains[chain_world])
		{
			if (s->flags & SURF_DRAWSKY)
				continue;

			rt_uploadsurf_state_t cur_state = {
				.entuniqueid = ENT_UNIQUEID_WORLD,
				.ent = NULL,
				.model = model,
				.surf = s,
				.diffuse_tex = diffuse_tex,
				.light_tex = light_tex,
				.lightmap_tex = (s->lightmaptexturenum >= 0) ? lightmaps[s->lightmaptexturenum].texture : greytexture,
				.alpha_test = alpha_test,
				.alpha = 1.0f,
				.use_zbias = false,
				.is_warp = false,
				.is_water = false,
				.is_acid = false,
				.is_teleport = false,
				.is_animated = true,
			};

			if (RT_ShouldSplitBatch (&cur_state, &last_state, cur_state.lightmap_tex != last_state.lightmap_tex))
			{
				RT_FlushBatch (cbx, &last_state, &brushpasses);
			}

			RT_BatchSurface (cbx, &cur_state, &brushpasses);
			last_state = cur_state;
		}

		RT_FlushBatch (cbx, &last_state, &brushpasses);
	}

	Atomic_AddUInt32 (&rs_brushpasses, brushpasses);
}

/*
================
R_DrawTextureChains_Multitexture
================
*/
void R_DrawTextureChains_Multitexture (
	cb_context_t *cbx, qmodel_t *model, entity_t *ent, texchain_t chain, const float alpha, int texstart, int texend, int entuniqueid)
{
	int                   i;
	msurface_t           *s;
	texture_t            *t;
	qboolean              use_zbias = (gl_zfix.value && model != cl.worldmodel);
	int                   ent_frame = ent != NULL ? ent->frame : 0;
	rt_uploadsurf_state_t last_state = {0};
	
	uint32_t brushpasses = 0;
	for (i = texstart; i < texend; ++i)
	{
		t = model->textures[i];

		if (!t || !t->texturechains[chain] || t->texturechains[chain]->flags & (SURF_DRAWTURB | SURF_DRAWTILED | SURF_NOTEXTURE))
			continue;

		if (ent == NULL && t->anim_total != 0)
			continue;

		RT_ClearBatch (cbx);

		qboolean alpha_test = (t->texturechains[chain]->flags & SURF_DRAWFENCE) != 0;
		gltexture_t *diffuse_tex = R_TextureAnimation (t, ent_frame)->gltexture;

		gltexture_t *light_tex = RT_CanonicalLightTex (t, ent_frame);

		for (s = t->texturechains[chain]; s; s = s->texturechains[chain])
		{
			if (s->flags & SURF_DRAWSKY)
				continue;

			rt_uploadsurf_state_t cur_state = {
				.entuniqueid = entuniqueid,
				.ent = ent,
				.model = model,
				.surf = s,
				.diffuse_tex = diffuse_tex,
				.light_tex = light_tex,
				.lightmap_tex = (s->lightmaptexturenum >= 0) ? lightmaps[s->lightmaptexturenum].texture : greytexture,
				.alpha_test = alpha_test,
				.alpha = alpha,
				.use_zbias = use_zbias,
				.is_warp = false,
				.is_water = false,
				.is_acid = false,
				.is_teleport = false,
			};

			if (cur_state.lightmap_tex != last_state.lightmap_tex)
			{
				RT_FlushBatch (cbx, &last_state, &brushpasses);
			}

			RT_BatchSurface (cbx, &cur_state, &brushpasses);
			last_state = cur_state;
		}

		RT_FlushBatch (cbx, &last_state, &brushpasses);
	}

	Atomic_AddUInt32 (&rs_brushpasses, brushpasses);
}

/*
=============
R_DrawWorld -- johnfitz -- rewritten
=============
*/
void R_DrawTextureChains (cb_context_t *cbx, qmodel_t *model, entity_t *ent, texchain_t chain, int entuniqueid)
{
	float entalpha;

	if (ent != NULL)
		entalpha = ENTALPHA_DECODE (ent->alpha);
	else
		entalpha = 1;

	R_DrawTextureChains_Multitexture (cbx, model, ent, chain, entalpha, 0, model->numtextures, entuniqueid);
}

/*
=============
R_DrawWorld -- ericw -- moved from R_DrawTextureChains, which is no longer specific to the world.
=============
*/
void R_DrawWorld (cb_context_t *cbx)
{
	rt_wldlights_emissive_count = 0;
	/* The list is about to be collected again, so the cached reach answers for the old
	   entries are void. */
	rt_wldlights_style_accepted_dirty = true;

	memset (&rt_emis_stats, 0, sizeof (rt_emis_stats));
	rt_emis_skip_num = 0;
	RT_EmisWatchFrameEnd ();

	RT_CollectWorldEmissiveLights ();

	if (!r_drawworld_cheatsafe)
		return;

	R_BeginDebugUtilsLabel (cbx, "World");
	R_DrawTextureChains_Multitexture (cbx, cl.worldmodel, NULL, chain_world, 1, 0, cl.worldmodel->numtextures, ENT_UNIQUEID_WORLD);

	R_EndDebugUtilsLabel (cbx);
}

/*
=============
R_DrawWorld_Water -- ericw -- moved from R_DrawTextureChains_Water, which is no longer specific to the world.
=============
*/
void R_DrawWorld_Water (cb_context_t *cbx)
{
	if (!r_drawworld_cheatsafe)
		return;

	R_BeginDebugUtilsLabel (cbx, "Water");
	R_DrawTextureChains_Water (cbx, cl.worldmodel, NULL, chain_world, ENT_UNIQUEID_WORLD);
	R_EndDebugUtilsLabel (cbx);
}

void R_DrawWorld_Animated (cb_context_t *cbx)
{
	if (!r_drawworld_cheatsafe)
		return;

	R_BeginDebugUtilsLabel (cbx, "Animated");
	R_DrawTextureChains_Animated (cbx, cl.worldmodel);
	R_EndDebugUtilsLabel (cbx);
}

/*
=============
R_DrawWorld_ShowTris -- ericw -- moved from R_DrawTextureChains_ShowTris, which is no longer specific to the world.
=============
*/
void R_DrawWorld_ShowTris (cb_context_t *cbx)
{
	if (!r_drawworld_cheatsafe)
		return;

	R_DrawTextureChains_ShowTris (cbx, cl.worldmodel, chain_world);
}



/*
=============
RT_RegisterWorldModelLight

A light of the map itself: it stands where it stands every frame, so it is held to the reach of
rt_light_reach rather than to the cap the moving lights are registered with. The leaf it resolved
into is where its list starts, not how far the light is heard: the PVS of that leaf is as wide as
the doorways of the map make it, and a light given the whole of it fills the lists of areas it
only sees into, where the lights standing there are then the ones the pass has to drop.
=============
*/
static void RT_RegisterWorldModelLight (const RgTexturedAreaLightUploadInfo *lt, vec3_t center)
{
	const float nudge = 2.0f;
	const vec3_t origin = {
		center[0] + nudge * lt->normal.data[0],
		center[1] + nudge * lt->normal.data[1],
		center[2] + nudge * lt->normal.data[2],
	};

	RT_ClusterLightAdd (lt->uniqueID, origin, RT_ClusterLightReachStatic ());

	if (CVAR_TO_BOOL (rt_debugemissive))
	{
		RT_EmitEmissiveWirePolygon (lt);
	}
}

/* The light manager drops a light whose colour channels sum to less than 0.0001
   (IsColorTooDim in vkpt). A light style or a dark animation frame switching the light off must
   not push the colour below that: a dropped id leaves a hole in every cluster list naming it,
   which is the whole-scene flicker. Keep the faintest possible emission instead. */
#define RT_MIN_EMISSIVE_COLOR_SUM 0.0002f

static void RT_KeepEmissiveLightColor (vec3_t color)
{
	float sum = 0.0f;

	for (int i = 0; i < 3; i++)
		if (color[i] > 0.0f)
			sum += color[i];

	if (sum < RT_MIN_EMISSIVE_COLOR_SUM)
	{
		const float c = RT_MIN_EMISSIVE_COLOR_SUM * (1.0f / 3.0f);
		color[0] = color[1] = color[2] = c;
	}
}

void RT_UploadAllWorldModelLights (void)
{
	if (!RT_AllowTexturedAreaLights ())
		return;

	if (rt_wldlights_style_accepted_dirty ||
	    rt_wldlights_style_accepted_reach != CVAR_TO_FLOAT (rt_light_styles_reach))
	{
		RT_BuildWorldLightStyleAcceptance ();
	}

	for (int i = 0; i < rt_wldlights_emissive_count; i++)
	{
		RgTexturedAreaLightUploadInfo *li = &rt_wldlights_emissive_upload[i];
		const msurface_t *surf = rt_wldlights_emissive_surf[i];
		gltexture_t *light_tex = rt_wldlights_emissive_tex[i];

		*li = rt_wldlights_emissive[i];

		/* Animated world textures (teleporters and the like) light the level from the frame
		   they currently show. Re-derive the emission from that frame every frame. The light's
		   identity and geometry come from the surface and stay fixed, so this never moves the
		   light in the array or in the cluster lists; only its colour and mean emission change.
		   A frame that cannot host a light dims the light instead of dropping it. */
		gltexture_t *cur_tex = RT_AnimatedLightTex (surf->texinfo->texture);
		if (cur_tex && cur_tex != light_tex)
		{
			rt_emissive_params_t params;
			if (RT_EmissiveLightParamsForTex (cur_tex, &params))
			{
				/* The mask the shader samples belongs to the frame the surface shows, so the
				   material moves with it; the geometry stays where the surface is. */
				li->material  = params.material;
				li->meanEmiss = params.meanEmiss;
				VectorCopy (params.color, li->color.data);
			}
			else
			{
				VectorCopy (vec3_origin, li->color.data);
			}
		}

		if (light_tex && light_tex->rtlightstyles && CVAR_TO_BOOL (rt_light_styles))
		{
			const float style_scale = RT_WorldLightStyleScale (i);

			/* A switched-off style dims the light's emission to zero instead of dropping it
			   from the light array: a missing id leaves a hole in every cluster list naming it.
			   The scale goes on the color, which is the radiance the shader reads of a light
			   with a mask and without one alike. meanEmiss is not it: for a masked light the
			   shader never reads meanEmiss, so a style written there would not dim the light. */
			VectorScale (li->color.data, style_scale, li->color.data);
		}

		RT_ScaleEmissiveLightColor (li->color.data);

		/* The light manager drops a light whose colour channels sum to less than its keep
		   threshold, so a light switched off by its style or by a dark animation frame is
		   clamped to a barely-there emission instead of being dropped. */
		RT_KeepEmissiveLightColor (li->color.data);
	}

	if (CVAR_TO_BOOL (rt_wmodel_lights_batch))
	{
		/* The per-light cost is the call, not the light: hand the whole map over in one. */
		RgResult r = rgUploadTexturedAreaLights (vulkan_globals.instance, rt_wldlights_emissive_upload,
		                                         (uint32_t) rt_wldlights_emissive_count);
		RG_CHECK (r);

		for (int i = 0; i < rt_wldlights_emissive_count; i++)
		{
			RT_RegisterWorldModelLight (&rt_wldlights_emissive_upload[i], rt_wldlights_emissive_center[i]);
		}
	}
	else
	{
		for (int i = 0; i < rt_wldlights_emissive_count; i++)
		{
			const RgTexturedAreaLightUploadInfo *lt = &rt_wldlights_emissive_upload[i];

			RgResult r = rgUploadTexturedAreaLight (vulkan_globals.instance, lt);
			RG_CHECK (r);

			RT_RegisterWorldModelLight (lt, rt_wldlights_emissive_center[i]);
		}
	}
}



typedef struct rt_teleport_s
{
	vec3_t   a;
	vec3_t   b;
	float    b_angle;
	qboolean potentially_mirror;
} rt_teleport_t;

rt_teleport_t *rt_teleports = NULL;
int            rt_teleports_count = 0;


struct rt_triggerteleport_t
{
	char target[128];
	char model[128];
};
struct rt_infoteleportdestination_t
{
	char   targetname[128];
	float  angle;
	vec3_t origin;
};
struct rt_parsetriggers_result_t
{
	struct rt_triggerteleport_t         *trigs;
	int                                  trigs_count;
	struct rt_infoteleportdestination_t *dsts;
	int                                  dsts_count;
};

static struct rt_parsetriggers_result_t ParseTeleportTriggers (void)
{
	struct rt_parsetriggers_result_t result = {0};

	if (!cl.worldmodel)
	{
		return result;
	}

	const char *data = cl.worldmodel->entities;
	if (!data)
	{
		return result;
	}


	char key[128], value[4096];


    #define STRUCT_STATE_STRUCT_STARTED			1
    #define STRUCT_STATE_CLASSNAME_TRIGGER		2
    #define STRUCT_STATE_CLASSNAME_DESTINATION	4
    #define STRUCT_STATE_TARGET					8
    #define STRUCT_STATE_TARGETNAME				16
    #define STRUCT_STATE_MODEL					32
    #define STRUCT_STATE_ANGLE					64
    #define STRUCT_STATE_ORIGIN					128
	int structstate = 0;

	struct
	{
		struct rt_triggerteleport_t tr;
		struct rt_infoteleportdestination_t dst;
	} structvalues = {0};
	

	while (1)
	{
		data = COM_Parse (data);
		if (!data)
			return result; // error

	    if (com_token[0] == '{')
	    {
			memset (&structvalues, 0, sizeof (structvalues));
			structstate = STRUCT_STATE_STRUCT_STARTED;
            continue;
	    }
	    else if (com_token[0] == '}')
		{
			if (structstate & STRUCT_STATE_STRUCT_STARTED)
			{
				if (structstate & STRUCT_STATE_CLASSNAME_TRIGGER)
				{
					result.trigs = Mem_Realloc (result.trigs, sizeof (*result.trigs) * (result.trigs_count + 1));
					result.trigs[result.trigs_count] = structvalues.tr;
					result.trigs_count++;
				}
				else if (structstate & STRUCT_STATE_CLASSNAME_DESTINATION)
				{
					result.dsts = Mem_Realloc (result.dsts, sizeof (*result.dsts) * (result.dsts_count + 1));
					result.dsts[result.dsts_count] = structvalues.dst;
					result.dsts_count++;
				}
			}

			structstate = 0; // end of struct
			continue;
		}

		if (com_token[0] == '_')
			q_strlcpy (key, com_token + 1, sizeof (key));
		else
			q_strlcpy (key, com_token, sizeof (key));
		while (key[0] && key[strlen (key) - 1] == ' ') // remove trailing spaces
			key[strlen (key) - 1] = 0;
		data = COM_Parse (data);
		if (!data)
			return result; // error
		q_strlcpy (value, com_token, sizeof (value));

		
		if (strcmp (key, "classname") == 0)
		{
			if (strcmp (value, "trigger_teleport") == 0)
			{
				structstate |= STRUCT_STATE_CLASSNAME_TRIGGER;
			}
			else if (strcmp (value, "info_teleport_destination") == 0)
			{
				structstate |= STRUCT_STATE_CLASSNAME_DESTINATION;
			}
		}
		else if (strcmp (key, "origin") == 0)
		{
			vec3_t tmpvec;
			int    components = sscanf (value, "%f %f %f", &tmpvec[0], &tmpvec[1], &tmpvec[2]);

			if (components == 3)
			{
				structvalues.dst.origin[0] = tmpvec[0];
				structvalues.dst.origin[1] = tmpvec[1];
				structvalues.dst.origin[2] = tmpvec[2];
				structstate |= STRUCT_STATE_ORIGIN;
			}
		}
		else if (strcmp (key, "angle") == 0)
		{
			float tmp;
			int   components = sscanf (value, "%f", &tmp);

			if (components ==1)
			{
				structvalues.dst.angle = tmp;
				structstate |= STRUCT_STATE_ANGLE;
			}
		}
		else if (strcmp (key, "model") == 0)
		{
			q_strlcpy (structvalues.tr.model, value, sizeof (structvalues.tr.model));
			structstate |= STRUCT_STATE_MODEL;
		}
		else if (strcmp (key, "target") == 0)
		{
			q_strlcpy (structvalues.tr.target, value, sizeof (structvalues.tr.target));
			structstate |= STRUCT_STATE_TARGET;
		}
		else if (strcmp (key, "targetname") == 0)
		{
			q_strlcpy (structvalues.dst.targetname, value, sizeof (structvalues.dst.targetname));
			structstate |= STRUCT_STATE_TARGETNAME;
		}
	}
}

#define RG_MAX_PORTALS 62

void RT_ParseTeleports (void)
{
	rt_teleports_count = 0;
	
	struct rt_parsetriggers_result_t r = ParseTeleportTriggers ();
	if (r.trigs_count == 0 || r.dsts_count == 0)
	{
		return;
	}

	for (int i = 0; i < r.trigs_count; i++)
	{
		for (int o = 0; o < r.dsts_count; o++)
		{
			const struct rt_triggerteleport_t         *in = &r.trigs[i];
			const struct rt_infoteleportdestination_t *out = &r.dsts[o];

			// if found a match between trigger and destination
			if (strncmp (in->target, out->targetname, sizeof (in->target)) == 0)
			{
				qmodel_t *mod = Mod_ForName (in->model, false);

				if (mod)
				{
					rt_teleport_t *entry;
					{
						rt_teleports = Mem_Realloc (rt_teleports, sizeof (*rt_teleports) * (rt_teleports_count + 1));
						entry = &rt_teleports[rt_teleports_count];
						memset (entry, 0, sizeof (*entry));
						rt_teleports_count++;
					}


					// trigger position
					VectorAdd (mod->mins, mod->maxs, entry->a);
					VectorScale (entry->a, 0.5f, entry->a);


					// destination position
					VectorCopy (out->origin, entry->b);
					entry->b_angle = out->angle;


					entry->potentially_mirror = false;
				}

				break;
			}
		}
	}

	if (rt_teleports_count > RG_MAX_PORTALS)
	{
		rt_teleports_count = RG_MAX_PORTALS;
		Con_Warning ("Too many teleports to render, limit is 62");
	}

	Mem_Free (r.trigs);
	Mem_Free (r.dsts);
}


static RgFloat3D ApplyTransform (const RgTransform *transform, const vec3_t v)
{
	RgFloat3D r = {0};
	for (int i = 0; i < 3; i++)
	{
		r.data[i] = 
			transform->matrix[i][0] * v[0] + 
			transform->matrix[i][1] * v[1] + 
			transform->matrix[i][2] * v[2] + 
			transform->matrix[i][3];
	}
	return r;
}


static float DistanceSqr (const vec3_t a, const vec3_t b)
{
	vec3_t delta;
	VectorSubtract (a, b, delta);

	return DotProduct (delta, delta);
}


static qboolean RT_FindNearestTeleport (const RgGeometryUploadInfo *info, uint8_t *result, qboolean *potentially_mirror)
{
	vec3_t emin = {FLT_MAX, FLT_MAX, FLT_MAX};
	vec3_t emax = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

	for (uint32_t i = 0; i < info->vertexCount; i++)
	{
		RgFloat3D v = ApplyTransform (&info->transform, info->pVertices[i].position);

		for (int k = 0; k < 3; k++)
		{
			emin[k] = q_min (v.data[k], emin[k]);
			emax[k] = q_max (v.data[k], emax[k]);
		}
	}

	vec3_t center;
	VectorAdd (emin, emax, center);
	VectorScale (center, 0.5f, center);

	int nearest = -1;
	float nearest_dist = FLT_MAX;

	for (int i = 0; i < rt_teleports_count; i++)
	{
		float d = DistanceSqr (rt_teleports[i].a, center);

		if (d < nearest_dist)
		{
			nearest = i;
			nearest_dist = d;
		}
	}

	if (nearest < 0)
	{
		return false;
	}

	assert (nearest <= RG_MAX_PORTALS);

	*result = (uint8_t)nearest;
	*potentially_mirror = rt_teleports[nearest].potentially_mirror;
	return true;
}


void RT_UploadAllTeleports ()
{
	assert (rt_teleports_count >= 0 && rt_teleports_count <= RG_MAX_PORTALS);

	if (!CVAR_TO_BOOL (rt_teleport_portals))
	{
		return;
	}

	const vec3_t outoffset = {0, 0, 64};

	for (int i = 0; i < rt_teleports_count; i++)
	{
		const rt_teleport_t *tele = &rt_teleports[i];

		vec3_t forward, right, up;
		{
			vec3_t out_angles = {0, tele->b_angle, 0};
			AngleVectors (out_angles, forward, right, up);
		}

		RgPortalUploadInfo info = 
		{
			.portalIndex = (uint8_t)i,
			.inPosition = RT_VEC3 (tele->a),
			.outPosition = RT_VEC3 (tele->b),
			.outDirection = RT_VEC3 (forward),
			.outUp = RT_VEC3 (up),
		};

		VectorAdd (info.outPosition.data, outoffset, info.outPosition.data);

		RgResult r = rgUploadPortal (vulkan_globals.instance, &info);
		RG_CHECK (r);
	}
}


/*
=================
RT_EmissiveLightTex

The texture that makes a surface emissive, or NULL. Mirrors the gate inside
RT_CollectWorldEmissiveLights. The out parameters are optional.
=================
*/
static gltexture_t *RT_EmissiveLightTex (msurface_t *surf, RgMaterial *material, float *meanEmiss, vec3_t color)
{
	if (!surf || !surf->texinfo || (surf->flags & (SURF_DRAWSKY | SURF_NOTEXTURE)))
		return NULL;

	texture_t *t = surf->texinfo->texture;

	if (!t || !t->gltexture)
		return NULL;

	gltexture_t *light_tex = RT_CanonicalLightTex (t, 0);

	rt_emissive_params_t p;

	if (!light_tex || !RT_EmissiveLightParamsForTex (light_tex, &p))
		return NULL;

	if (material)
		*material = p.material;
	if (meanEmiss)
		*meanEmiss = p.meanEmiss;
	if (color)
		VectorCopy (p.color, color);

	return light_tex;
}

static qboolean RT_SurfaceOwnedBySubmodel (const qmodel_t *model, int surfindex)
{
	return RT_FaceOwnedBySubmodel (model, surfindex);
}

/*
=================
RT_WorldCensus

Reports the BSP tables the renderer has to own once the light polygons (and later the
cluster light lists) move into it, so the payload of rgUploadWorldLights can be sized
before any of it is written. Read only: nothing is uploaded and nothing is built.
Enable with rt_worldcensus 1 and reload the map.
=================
*/
void RT_WorldCensus (void)
{
	qmodel_t *model = cl.worldmodel;

	if (!CVAR_TO_BOOL (rt_worldcensus) || !model)
		return;

	// Mod_DecompressVis uses the same row size, so this is the number S4 has to match.
	const int pvs_rowbytes = (model->numleafs + 31) / 8;

	int leaves_pvs = 0, leaves_no_pvs = 0, pvs_last_byte = 0;

	for (int i = 0; i < model->numleafs; i++)
	{
		const byte *row = model->leafs[i].compressed_vis;
		if (!row)
		{
			leaves_no_pvs++;
			continue;
		}

		leaves_pvs++;

		const int end = (int)(row - model->visdata) + pvs_rowbytes;

		if (end > pvs_last_byte)
			pvs_last_byte = end;
	}

	int all_verts = 0, all_light_faces = 0, masked_light_faces = 0;
	int inline_models = 0, inline_faces = 0, inline_verts = 0, inline_light_faces = 0;
	int world_only_faces = 0, world_only_light_faces = 0;

	for (int i = 0; i < model->numsurfaces; i++)
	{
		msurface_t  *surf      = &model->surfaces[i];
		gltexture_t *light_tex = RT_EmissiveLightTex (surf, NULL, NULL, NULL);

		all_verts += surf->numedges;

		if (light_tex)
		{
			all_light_faces++;

			// has_mask of RT_EmissiveLightParamsForTex: the texture carries a mask, bright or not.
			if (light_tex->rtemissivetex)
				masked_light_faces++;
		}

		const qboolean owned_by_submodel = RT_SurfaceOwnedBySubmodel (model, i);

		if (owned_by_submodel)
		{
			inline_faces++;
			inline_verts += surf->numedges;

			if (light_tex)
				inline_light_faces++;
		}
		else
		{
			world_only_faces++;

			if (light_tex)
				world_only_light_faces++;
		}
	}

	for (int i = 1; i < MAX_MODELS; i++)
	{
		const qmodel_t *m = cl.model_precache[i];

		if (m && m->name[0] == '*' && m->type == mod_brush)
			inline_models++;
	}

	Con_Printf ("world census: %s\n", model->name);
	Con_Printf ("  bsp: %i leafs (clusters), %i submodels, %i surfaces, %i edges, %i vertexes\n",
		model->numleafs, model->numsubmodels, model->numsurfaces, model->numedges, model->numvertexes);
	Con_Printf ("  faces: %i stand in the world, %i belong to %i inline models; %i RgVertex for all of them\n",
		world_only_faces, inline_faces, inline_models, all_verts);
	Con_Printf ("  emissive faces: %i world + %i inline = %i total, %i of them carry a mask (%i inline vertexes)\n",
		world_only_light_faces, inline_light_faces, all_light_faces, masked_light_faces, inline_verts);
	Con_Printf ("  pvs: %i bytes per row (%i leafs), %i leafs have a row, %i have none; visdata spans %i bytes (%.1f MB)\n",
		pvs_rowbytes, model->numleafs, leaves_pvs, leaves_no_pvs, pvs_last_byte, (float)pvs_last_byte / 1048576.0f);
	Con_Printf ("  light budget: %i static polygons against a cap of %i in the engine, %i free\n",
		all_light_faces, MAX_WORLDLIGHTS_COUNT, MAX_WORLDLIGHTS_COUNT - all_light_faces);
}

/*
=================
World clusters

The renderer builds its cluster light lists, and the statistics of those lists, on tables of a
fixed size: a cluster index past that size lands outside the offsets, outside the lists and
outside the statistics buffer, and the atomic write of the statistics is what turns a light of a
moving entity into a shifting picture all over the map. BSP leaf indices do reach that size - the
AD maps ship up to 134394 leafs against a table of 8192 - so the leaf indices are folded into a
cluster space that fits here, once per map load, and every surface, brush model and light is
translated into it.

Two tables can be built, and the leaf count alone decides which one:

  identity  every leaf is a cluster. The surface, brush model and light indices are the leaf
            indices they always were, the map's own visdata is the PVS the renderer reads, and
            nothing about the picture changes.
  grid      the leafs are folded into uniform cells of the map box, at most 8191 of them. A cell
            carries the union of the PVS rows of the leafs inside it, so a light in a cell reaches
            everything any of those leafs reached. The union is wider than the row of a single
            leaf, which is why the grid only takes over when the identity table cannot be built.

A PVS row names its cluster with bit + 1 - R_MarkLeafsSIMD and the renderer's own walk read it
that way - so a row stands for the clusters 1..n and not for 0..n-1. The cell index is the bit
index, which keeps the projection a copy of the bits when the clusters happen to be leafs.
=================
*/
#define RT_WORLD_CLUSTER_MAX 8192
// A row names its cluster with bit + 1, so the last index of the table stays unused.
#define RT_WORLD_CLUSTER_CELLS (RT_WORLD_CLUSTER_MAX - 1)

typedef struct
{
	int      num_leafs;
	int32_t *leaf_cluster; // leaf index -> cluster index
	int      is_grid;
	// The map the tables were built from: a new map needs new tables, and the callers of the
	// mapping run before the upload that would otherwise build them.
	qmodel_t *model;

	int        num_clusters;
	RgFloat3D *cluster_mins;
	RgFloat3D *cluster_maxs;
	uint8_t   *cluster_flags;
	int32_t   *vis_offsets;
	uint32_t   pvs_row_bytes;

	// Only the grid publishes its own rows; the identity table leaves the map's visdata in place.
	uint8_t *vis_data;
	int      vis_data_size;

	// Bit c of byte c/8 is 1 when a sun ray from cluster c can still reach the sky.
	uint8_t *sky_vis;

	float grid_mins[3];
	float grid_cell_size[3];
	float grid_inv_cell[3];
	int   grid_dims[3];
} rt_worldclusters_t;

static rt_worldclusters_t rt_worldclusters;

static void RT_AllocClusterTables (int num_clusters)
{
	rt_worldclusters.num_clusters  = num_clusters;
	rt_worldclusters.cluster_mins  = (RgFloat3D *)Mem_Alloc (sizeof (RgFloat3D) * num_clusters);
	rt_worldclusters.cluster_maxs  = (RgFloat3D *)Mem_Alloc (sizeof (RgFloat3D) * num_clusters);
	rt_worldclusters.cluster_flags = (uint8_t *)Mem_Alloc (sizeof (uint8_t) * num_clusters);
	rt_worldclusters.vis_offsets   = (int32_t *)Mem_Alloc (sizeof (int32_t) * num_clusters);
}

void RT_FreeWorldClusters (void)
{
	Mem_Free (rt_worldclusters.leaf_cluster);
	Mem_Free (rt_worldclusters.cluster_mins);
	Mem_Free (rt_worldclusters.cluster_maxs);
	Mem_Free (rt_worldclusters.cluster_flags);
	Mem_Free (rt_worldclusters.vis_offsets);
	Mem_Free (rt_worldclusters.vis_data);
	Mem_Free (rt_worldclusters.sky_vis);

	memset (&rt_worldclusters, 0, sizeof (rt_worldclusters));
}

/*
=================
RT_MapWorldCluster

The one place a leaf index becomes a cluster index, for surfaces, brush models and lights alike.
Both tables are built from the same leaf table, and a request that arrives before one exists
builds it, so no caller has to care about the map load order.
=================
*/
int RT_MapWorldCluster (int leaf_index)
{
	if (rt_worldclusters.model != cl.worldmodel || !rt_worldclusters.leaf_cluster)
		RT_BuildWorldClusters ();

	if (leaf_index <= 0 || leaf_index > rt_worldclusters.num_leafs)
		return 0;

	return rt_worldclusters.leaf_cluster[leaf_index];
}

static int RT_GridCellOfPoint (const float *p)
{
	int cell = 0;

	for (int a = 0; a < 3; a++)
	{
		float f = (p[a] - rt_worldclusters.grid_mins[a]) * rt_worldclusters.grid_inv_cell[a];

		if (!(f > 0.0f))
			f = 0.0f;
		else if (f >= (float)rt_worldclusters.grid_dims[a])
			f = (float)rt_worldclusters.grid_dims[a] - 1.0f;

		cell = cell * rt_worldclusters.grid_dims[a] + (int)f;
	}

	return cell;
}

static void RT_GridCellBounds (int cell, float *mins, float *maxs)
{
	const int dim_y = rt_worldclusters.grid_dims[1];
	const int dim_z = rt_worldclusters.grid_dims[2];
	const int idx[3] = { cell / (dim_y * dim_z), (cell / dim_z) % dim_y, cell % dim_z };

	for (int a = 0; a < 3; a++)
	{
		mins[a] = rt_worldclusters.grid_mins[a] + (float)idx[a] * rt_worldclusters.grid_cell_size[a];
		maxs[a] = mins[a] + rt_worldclusters.grid_cell_size[a];
	}
}

/* The row of a leaf names a leaf with bit + 1 - the renderer reads its own walk that way - so the
   row of a cluster names the cluster with the same offset: the bit of a cluster is its index
   minus one. The map stores a row as runs, and the rows of the maps that need the grid are
   millions of bits with a few hundred of them set, so the runs are followed rather than the row
   expanded first: a run carries no bits at all. */
static void RT_ProjectVisRow (const uint8_t *p_compressed, int in_avail, int leaf_row_bytes, int num_leafs,
	const int32_t *p_leaf_cluster, uint8_t *p_cluster_row)
{
	// A byte of the row never costs more than two, and the lump ends after the last row of the map.
	int in_limit = leaf_row_bytes * 2;

	if (in_limit > in_avail)
		in_limit = in_avail;

	if (in_limit <= 0)
		return;

	int in = 0;

	for (int b = 0; b < leaf_row_bytes && in < in_limit; )
	{
		const uint8_t bits = p_compressed[in++];

		if (bits != 0)
		{
			for (int k = 0; k < 8; k++)
			{
				const int leaf = (b << 3) + k + 1;
				int32_t   cluster;

				if (!(bits & (1u << k)) || leaf > num_leafs)
					continue;

				cluster = p_leaf_cluster[leaf] - 1;

				if (cluster < 0)
					continue;

				p_cluster_row[cluster >> 3] |= (uint8_t)(1u << (cluster & 7));
			}

			b++;
			continue;
		}

		if (in >= in_limit)
			return;

		// The count of a run covers the byte the run starts at, as Mod_DecompressVis expands it,
		// so the row advances by the count alone; a run that names more than the row holds is cut
		// the same way the renderer cuts it.
		int run = p_compressed[in++];

		if (run > leaf_row_bytes - b)
			run = leaf_row_bytes - b;

		b += run;
	}
}

// A run of zero bytes is what the renderer reads as a run count, so a row is left uncompressed.
static int RT_CompressVisRow (const uint8_t *p_row, int row_bytes, uint8_t *p_out)
{
	int out = 0;

	for (int i = 0; i < row_bytes; )
	{
		if (p_row[i] != 0)
		{
			p_out[out++] = p_row[i++];
			continue;
		}

		int run = 0;

		while (i + run < row_bytes && run < 255 && p_row[i + run] == 0)
			run++;

		p_out[out++] = 0;
		p_out[out++] = (uint8_t)run;
		i += run;
	}

	return out;
}

static void RT_BuildWorldClustersIdentity (qmodel_t *model)
{
	const int num_leafs = model->numleafs;
	// Cluster i is leaf i, and cluster 0 is reserved for the solid leaf and geometry that has no
	// leaf of its own, so the table holds one cluster per leaf plus the reserved cluster 0.
	const int num_clusters = num_leafs + 1;
	const int row_bytes = (num_leafs + 31) / 8;

	rt_worldclusters.num_leafs = num_leafs;
	rt_worldclusters.leaf_cluster = (int32_t *)Mem_Alloc (sizeof (int32_t) * num_clusters);
	rt_worldclusters.pvs_row_bytes = (uint32_t)row_bytes;
	rt_worldclusters.model = model;

	RT_AllocClusterTables (num_clusters);

	/* Cluster 0 is the hole: the whole map, solid, no PVS row. An invalid or solid leaf maps to
	   it, and it never names itself in a row. */
	rt_worldclusters.leaf_cluster[0] = 0;
	VectorCopy (model->mins, rt_worldclusters.cluster_mins[0].data);
	VectorCopy (model->maxs, rt_worldclusters.cluster_maxs[0].data);
	rt_worldclusters.cluster_flags[0] = RG_WORLD_CLUSTER_SOLID_BIT;
	rt_worldclusters.vis_offsets[0] = -1;

	for (int i = 1; i <= num_leafs; i++)
	{
		const mleaf_t *leaf = &model->leafs[i];
		int32_t        offset = -1;

		rt_worldclusters.leaf_cluster[i] = i;
		VectorCopy (leaf->minmaxs, rt_worldclusters.cluster_mins[i].data);
		VectorCopy (leaf->minmaxs + 3, rt_worldclusters.cluster_maxs[i].data);
		rt_worldclusters.cluster_flags[i] = (leaf->contents == CONTENTS_SOLID) ? RG_WORLD_CLUSTER_SOLID_BIT : 0;

		if (leaf->compressed_vis)
		{
			offset = (int32_t)(leaf->compressed_vis - model->visdata);

			if ((int)offset + row_bytes > rt_worldclusters.vis_data_size)
				rt_worldclusters.vis_data_size = (int)offset + row_bytes;
		}

		rt_worldclusters.vis_offsets[i] = offset;
	}
}

// The number of cells an axis would be cut into at this scale, the product of the three being
// the cell count of the whole grid.
static int RT_GridCellsAt (const float *ext, float scale, int *dims)
{
	int cells = 1;

	for (int a = 0; a < 3; a++)
	{
		int n = (int)floor (ext[a] * scale);

		if (n < 1)
			n = 1;
		else if (n > RT_WORLD_CLUSTER_CELLS)
			n = RT_WORLD_CLUSTER_CELLS;

		dims[a] = n;
		cells *= n;
	}

	return cells;
}

static void RT_BuildWorldClustersGrid (qmodel_t *model, const float *map_mins, const float *map_maxs, const int *dims)
{
	const int num_leafs = model->numleafs;
	const int num_cells = dims[0] * dims[1] * dims[2];
	// A row names its cluster with bit + 1, so cluster 0 stays empty and the cells start at 1.
	const int num_clusters = num_cells + 1;
	const int cluster_row_bytes = (num_clusters + 7) / 8;
	const int leaf_row_bytes = (num_leafs + 31) / 8;

	int     *cell_counts = (int *)Mem_Alloc (sizeof (int) * num_cells);
	int     *cell_start = (int *)Mem_Alloc (sizeof (int) * (num_cells + 1));
	int     *members = (int *)Mem_Alloc (sizeof (int) * num_leafs);
	int     *filled = (int *)Mem_Alloc (sizeof (int) * num_cells);
	uint8_t *row = (uint8_t *)Mem_Alloc (cluster_row_bytes);

	rt_worldclusters.num_leafs = num_leafs;
	rt_worldclusters.leaf_cluster = (int32_t *)Mem_Alloc (sizeof (int32_t) * (num_leafs + 1));
	rt_worldclusters.leaf_cluster[0] = 0;
	rt_worldclusters.is_grid = 1;
	rt_worldclusters.pvs_row_bytes = (uint32_t)cluster_row_bytes;
	rt_worldclusters.model = model;
	// A cell of n rows needs at most two bytes per byte of a row, and there are num_cells of them.
	rt_worldclusters.vis_data = (uint8_t *)Mem_Alloc ((size_t)num_clusters * 2 * cluster_row_bytes + 16);

	for (int a = 0; a < 3; a++)
	{
		const float ext = map_maxs[a] - map_mins[a];

		rt_worldclusters.grid_mins[a] = map_mins[a];
		rt_worldclusters.grid_dims[a] = dims[a];
		rt_worldclusters.grid_cell_size[a] = ext / (float)dims[a];
		rt_worldclusters.grid_inv_cell[a] = (ext > 0.0f) ? (float)dims[a] / ext : 0.0f;
	}

	RT_AllocClusterTables (num_clusters);

	/* An empty cell keeps the whole map in its bounds and is marked solid: the top-up pass skips
	   it and no light can stand in it, so it is a hole at the edge of the grid and not a room. */
	for (int c = 0; c < num_clusters; c++)
	{
		for (int a = 0; a < 3; a++)
		{
			rt_worldclusters.cluster_mins[c].data[a] = map_maxs[a];
			rt_worldclusters.cluster_maxs[c].data[a] = map_mins[a];
		}

		rt_worldclusters.cluster_flags[c] = RG_WORLD_CLUSTER_SOLID_BIT;
	}

	rt_worldclusters.vis_offsets[0] = -1;

	for (int i = 1; i <= num_leafs; i++)
	{
		const mleaf_t *leaf = &model->leafs[i];
		float          center[3];
		int            cell;

		// A leaf sits in the cell its centre falls into, so the members of a cell are neighbours.
		for (int a = 0; a < 3; a++)
			center[a] = 0.5f * (leaf->minmaxs[a] + leaf->minmaxs[a + 3]);

		cell = RT_GridCellOfPoint (center);
		rt_worldclusters.leaf_cluster[i] = cell + 1;

		if (leaf->contents == CONTENTS_SOLID)
			continue;

		cell_counts[cell]++;

		for (int a = 0; a < 3; a++)
		{
			if (leaf->minmaxs[a] < rt_worldclusters.cluster_mins[cell + 1].data[a])
				rt_worldclusters.cluster_mins[cell + 1].data[a] = leaf->minmaxs[a];

			if (leaf->minmaxs[a + 3] > rt_worldclusters.cluster_maxs[cell + 1].data[a])
				rt_worldclusters.cluster_maxs[cell + 1].data[a] = leaf->minmaxs[a + 3];
		}
	}

	int running = 0;

	for (int cell = 0; cell < num_cells; cell++)
	{
		cell_start[cell] = running;
		running += cell_counts[cell];
	}

	cell_start[num_cells] = running;

	for (int i = 1; i <= num_leafs; i++)
	{
		int cell;

		if (model->leafs[i].contents == CONTENTS_SOLID)
			continue;

		cell = rt_worldclusters.leaf_cluster[i] - 1;
		members[cell_start[cell] + filled[cell]] = i;
		filled[cell]++;
	}

	/* The row of a cell is the union of the rows of the leafs inside it, and the union is what a
	   light standing in the cell hands out. */
	int vis_data_used = 0;

	for (int cell = 0; cell < num_cells; cell++)
	{
		qboolean have_row = false;

		memset (row, 0, cluster_row_bytes);

		for (int m = cell_start[cell]; m < cell_start[cell + 1]; m++)
		{
			const uint8_t *compressed = model->leafs[members[m]].compressed_vis;
			int            offset;

			if (compressed == NULL || model->visdata == NULL)
				continue;

			offset = (int)(compressed - model->visdata);

			// What a walk may read: the row and the rows that follow it, up to the end of the lump.
			RT_ProjectVisRow (compressed, model->visdatasize - offset, leaf_row_bytes, num_leafs,
				rt_worldclusters.leaf_cluster, row);
			have_row = true;
		}

		if (!have_row)
		{
			// A light standing here hands out nothing and is left to the top-up pass, which is
			// what the renderer does with every cluster that carries no row.
			rt_worldclusters.vis_offsets[cell + 1] = -1;
			continue;
		}

		rt_worldclusters.vis_offsets[cell + 1] = vis_data_used;
		vis_data_used += RT_CompressVisRow (row, cluster_row_bytes, rt_worldclusters.vis_data + vis_data_used);
	}

	rt_worldclusters.vis_data_size = vis_data_used + cluster_row_bytes;

	// A cell no leaf lives in is solid, and its box is the one the top-up pass measures against.
	for (int cell = 0; cell < num_cells; cell++)
	{
		if (cell_counts[cell] > 0)
		{
			rt_worldclusters.cluster_flags[cell + 1] = 0;
			continue;
		}

		RT_GridCellBounds (cell, rt_worldclusters.cluster_mins[cell + 1].data,
			rt_worldclusters.cluster_maxs[cell + 1].data);
	}

	// Cluster 0 is the one no row can name: the whole map in its bounds, and solid.
	for (int a = 0; a < 3; a++)
	{
		rt_worldclusters.cluster_mins[0].data[a] = map_mins[a];
		rt_worldclusters.cluster_maxs[0].data[a] = map_maxs[a];
	}

	Mem_Free (cell_counts);
	Mem_Free (cell_start);
	Mem_Free (members);
	Mem_Free (filled);
	Mem_Free (row);
}

void RT_BuildWorldClusters (void)
{
	qmodel_t *model = cl.worldmodel;
	vec3_t    map_mins, map_maxs, ext;
	qboolean  have_bounds = false;
	float     scale = 1.0f;
	int       num_cells, dims[3];

	RT_FreeWorldClusters ();

	if (!model || !model->leafs || model->numleafs < 2)
		return;

	/* The identity table needs one cluster per leaf plus the reserved cluster 0, so it fits the
	   renderer's table when the leaves do. */
	if (model->numleafs < RT_WORLD_CLUSTER_MAX || !CVAR_TO_BOOL (rt_worldclusters_grid))
	{
		if (model->numleafs >= RT_WORLD_CLUSTER_MAX)
		{
			/* Leaf indices go to the renderer as they are, and the renderer's tables hold
			   RT_WORLD_CLUSTER_MAX of them, so what lies past that is not addressed at all. */
			Con_DWarning ("RT: %i leafs over the renderer's %i clusters, rt_worldclusters_grid is off\n",
				model->numleafs, RT_WORLD_CLUSTER_MAX);
		}

		RT_BuildWorldClustersIdentity (model);
		return;
	}

	/* The grid has to hold the leafs a light or a view can stand in, and the solid leafs reach
	   all around the map, so the open ones are what its box is measured from. Leaf 0 is the solid
	   leaf and is skipped; the real leafs are leafs[1..numleafs]. */
	for (int i = 1; i <= model->numleafs; i++)
	{
		const mleaf_t *leaf = &model->leafs[i];

		if (leaf->contents == CONTENTS_SOLID)
			continue;

		for (int a = 0; a < 3; a++)
		{
			if (!have_bounds || leaf->minmaxs[a] < map_mins[a])
				map_mins[a] = leaf->minmaxs[a];
			if (!have_bounds || leaf->minmaxs[a + 3] > map_maxs[a])
				map_maxs[a] = leaf->minmaxs[a + 3];
		}

		have_bounds = true;
	}

	if (!have_bounds)
	{
		for (int a = 0; a < 3; a++)
		{
			map_mins[a] = model->mins[a];
			map_maxs[a] = model->maxs[a];
		}
	}

	for (int a = 0; a < 3; a++)
		ext[a] = map_maxs[a] - map_mins[a];

	/* The finest grid whose cells fit the table, the same count per axis: cells the same size in
	   every direction is what the reach test of the top-up pass assumes. */
	const double volume = (double)ext[0] * (double)ext[1] * (double)ext[2];

	if (volume > 1.0)
		scale = (float)pow ((double)RT_WORLD_CLUSTER_CELLS / volume, 1.0 / 3.0);

	if (!(scale > 0.0f) || scale > 1.0e4f)
		scale = 1.0f;

	num_cells = RT_GridCellsAt (ext, scale, dims);

	while (num_cells > RT_WORLD_CLUSTER_CELLS)
	{
		scale *= 0.9f;
		num_cells = RT_GridCellsAt (ext, scale, dims);
	}

	RT_BuildWorldClustersGrid (model, map_mins, map_maxs, dims);

	Con_Printf ("RT: %i leafs folded into %i clusters over a %ix%ix%i grid, %i bytes of PVS rows\n",
		model->numleafs, num_cells + 1, dims[0], dims[1], dims[2], rt_worldclusters.vis_data_size);
}

/*
=================
RT_BuildClusterSkyVisibility

The per-cluster sky visibility of Q2RTX's compute_sky_visibility: a cluster keeps tracing its
sun ray only when the sky can be seen from it at all. The sun direction is a runtime cvar here,
so no direction is tested; instead the map's own PVS row of every leaf whose cluster holds a sky
surface is unioned, which marks exactly the clusters the sky is visible from - PVS is symmetric,
so "C sees K" is the same bit as "K sees C", and a sun ray from C that leaves through the sky of
K needs exactly that. A row of the map names a leaf by its index, as Mod_DecompressVis expands
it, and the cluster table names that leaf, so the union lands in the cluster the renderer asks
about (surf.cluster) in both tables. The rows of the grid table are projected instead and carry
the renderer's own +1 offset, so they are deliberately not read here. Every doubt (sky that
resolved into no cluster, a leaf with no row of its own, no row data at all, a row that decodes
short) marks everything, so a cluster only stops tracing when the sky is provably out of its
reach.
=================
*/
static void RT_BuildClusterSkyVisibility (void)
{
	qmodel_t *model = rt_worldclusters.model;
	const int num_clusters = rt_worldclusters.num_clusters;
	const int num_leafs = rt_worldclusters.num_leafs;
	// The rows the map carries: Mod_DecompressVis gives a row this many bytes in every table.
	const int leaf_row_bytes = (num_leafs + 31) / 8;
	const int num_bytes = (num_clusters + 7) / 8;
	uint8_t *has_sky;
	qboolean any_sky = false, any_cluster = false, everything = false;

	if (!model || num_clusters <= 0 || num_leafs <= 0 || !rt_worldclusters.leaf_cluster)
		return;

	rt_worldclusters.sky_vis = (uint8_t *)Mem_Alloc (num_bytes);
	memset (rt_worldclusters.sky_vis, 0, num_bytes);
	// Cluster 0 is where geometry with no leaf of its own lands; it keeps tracing, as Q2RTX
	// keeps tracing for an invalid cluster.
	rt_worldclusters.sky_vis[0] |= 1;

	has_sky = (uint8_t *)Mem_Alloc (num_bytes);
	memset (has_sky, 0, num_bytes);

	for (int i = 0; i < model->numsurfaces; i++)
	{
		msurface_t *surf = &model->surfaces[i];
		int         c;

		if (!(surf->flags & SURF_DRAWSKY))
			continue;

		any_sky = true;
		c = RT_MapWorldCluster (RT_GetSurfaceCluster (model, surf));

		if (c <= 0 || c >= num_clusters)
			continue;

		has_sky[c >> 3] |= (uint8_t)(1u << (c & 7));
		any_cluster = true;
	}

	// A map with no sky surface at all, or sky that resolved into no cluster, says nothing
	// about where the sky is seen from: every cluster keeps tracing.
	if (!any_sky || !any_cluster)
		everything = true;

	if (any_cluster && !everything)
	{
		uint8_t *row = (uint8_t *)Mem_Alloc (leaf_row_bytes);

		if (!model->visdata || model->visdatasize <= 0)
		{
			// Without rows of its own the map is one the tables say sees everything.
			everything = true;
		}

		for (int leaf = 1; !everything && leaf <= num_leafs; leaf++)
		{
			const int      c = rt_worldclusters.leaf_cluster[leaf];
			const uint8_t *prow;
			int            offset, in, in_limit, b;

			// Only the leaves of a cluster that holds sky hand out the sky.
			if (c <= 0 || c >= num_clusters || !(has_sky[c >> 3] & (1u << (c & 7))))
				continue;

			prow = model->leafs[leaf].compressed_vis;

			if (prow == NULL)
			{
				everything = true;
				break;
			}

			/* The RLE of Mod_DecompressVis, followed the way Mod_DecompressVis follows it: a
			   nonzero byte is eight bits, a zero byte a run count that covers the byte it
			   starts at, and a byte of the row never costs more than two of the data. */
			offset = (int)(prow - model->visdata);
			in_limit = leaf_row_bytes * 2;

			if (offset < 0 || offset >= model->visdatasize)
			{
				everything = true;
				break;
			}

			if (in_limit > model->visdatasize - offset)
				in_limit = model->visdatasize - offset;

			memset (row, 0, leaf_row_bytes);

			for (in = 0, b = 0; b < leaf_row_bytes && in < in_limit; )
			{
				const uint8_t bits = prow[in++];

				if (bits != 0)
				{
					row[b++] = bits;
					continue;
				}

				if (in >= in_limit)
					break;

				{
					int run = prow[in++];

					if (run > leaf_row_bytes - b)
						run = leaf_row_bytes - b;

					b += run;
				}
			}

			// A row that decodes short of its length is corrupt; trust nothing.
			if (b < leaf_row_bytes)
			{
				everything = true;
				break;
			}

			// Bit b of the row names leaf b + 1, and the cluster table names that leaf.
			for (b = 0; b < leaf_row_bytes * 8 && b < num_leafs; b++)
			{
				int target;

				if (!(row[b >> 3] & (1u << (b & 7))))
					continue;

				target = rt_worldclusters.leaf_cluster[b + 1];

				if (target > 0 && target < num_clusters)
					rt_worldclusters.sky_vis[target >> 3] |= (uint8_t)(1u << (target & 7));
			}
		}

		Mem_Free (row);
	}

	if (everything)
		memset (rt_worldclusters.sky_vis, 0xFF, num_bytes);

	if (CVAR_TO_BOOL (rt_worldlights_stats))
	{
		int traced = 0, sky_clusters = 0;

		for (int i = 0; i < num_clusters; i++)
		{
			if (rt_worldclusters.sky_vis[i >> 3] & (1u << (i & 7)))
				traced++;

			if (has_sky[i >> 3] & (1u << (i & 7)))
				sky_clusters++;
		}

		Con_Printf ("sky visibility: %i of %i clusters trace the sun (%i clusters hold sky%s)\n",
			traced, num_clusters, sky_clusters, everything ? ", every cluster traces" : "");
	}

	Mem_Free (has_sky);
}

/*
=================
RT_UploadWorldLights

Builds the world tables once per map load and hands them to the renderer: the clusters with
their bounds, the compressed PVS and every emissive face with its corners. The clusters are the
leafs themselves while the map fits the renderer's table and a grid of cells over them when it
does not, and each face carries the cluster of the surface it was grown from. The face gate is
the same one RT_WorldCensus counts and RT_CollectWorldEmissiveLights collects, so the renderer
can grow light polygons out of these faces.
=================
*/
typedef struct
{
	RgWorldLightFace *faces;
	RgVertex         *face_vertices;

	int num_clusters;
	int num_faces;
	int num_face_vertices;
} rt_worldlights_t;

static rt_worldlights_t rt_worldlights;

static void RT_FreeWorldLights (void)
{
	Mem_Free (rt_worldlights.faces);
	Mem_Free (rt_worldlights.face_vertices);

	memset (&rt_worldlights, 0, sizeof (rt_worldlights));
}

void RT_UploadWorldLights (void)
{
	qmodel_t *model = cl.worldmodel;

	RT_FreeWorldLights ();

	if (!model || !model->leafs || !model->surfaces || model->numleafs < 2 || !vulkan_globals.instance)
		return;

	// The cluster tables are what turns a leaf index into the index the renderer indexes with.
	RT_BuildWorldClusters ();

	// Which of those clusters can see the sky at all: the renderer skips the sun shadow ray
	// of the ones that cannot (Q2RTX's sky_visibility).
	RT_BuildClusterSkyVisibility ();

	// First pass: how many faces and corners there are, so the tables can be sized once.
	for (int i = 0; i < model->numsurfaces; i++)
	{
		msurface_t *surf = &model->surfaces[i];

		if (!RT_EmissiveLightTex (surf, NULL, NULL, NULL) || !surf->polys || surf->numedges < 3)
			continue;

		rt_worldlights.num_faces++;
		rt_worldlights.num_face_vertices += surf->numedges;
	}

	rt_worldlights.num_clusters = rt_worldclusters.num_clusters;

	if (rt_worldlights.num_faces > 0)
	{
		rt_worldlights.faces = (RgWorldLightFace *)Mem_Alloc (sizeof (RgWorldLightFace) * rt_worldlights.num_faces);
		rt_worldlights.face_vertices = (RgVertex *)Mem_Alloc (sizeof (RgVertex) * rt_worldlights.num_face_vertices);
	}

	// Second pass: the faces themselves, with their corners in world space.
	int face_index = 0, vertex_index = 0;

	for (int i = 0; i < model->numsurfaces; i++)
	{
		msurface_t *surf = &model->surfaces[i];
		RgMaterial  material;
		float       meanEmiss;
		vec3_t      color;

		if (!RT_EmissiveLightTex (surf, &material, &meanEmiss, color) || !surf->polys || surf->numedges < 3)
			continue;

		RgWorldLightFace *face = &rt_worldlights.faces[face_index++];
		const qboolean    owned_by_submodel = RT_SurfaceOwnedBySubmodel (model, i);

		face->uniqueID    = RT_GetBrushSurfUniqueId (ENT_UNIQUEID_WORLD, model, surf, 0);
		face->firstVertex = (uint32_t)vertex_index;
		face->numVertices = (uint32_t)surf->numedges;
		face->cluster     = (uint32_t)RT_MapWorldCluster (RT_GetSurfaceCluster (model, surf));
		// has_mask of RT_EmissiveLightParamsForTex: only a part of the texture glows.
		face->flags     = (material != RG_NO_MATERIAL) ? RG_WORLD_LIGHT_FACE_MASKED_BIT : 0;
		face->material  = material;
		face->meanEmiss = meanEmiss;
		// Inline models (doors, platforms) still move, so the host keeps their corners
		// up to date and the renderer must not bake them as static.
		face->isStatic = owned_by_submodel ? 0 : 1;

		if (owned_by_submodel)
			face->flags |= RG_WORLD_LIGHT_FACE_INLINE_MODEL_BIT;

		VectorCopy (color, face->color.data);

		for (int v = 0; v < surf->numedges; v++)
		{
			const float *srcv = surf->polys->verts[v];
			RgVertex    *dstv = &rt_worldlights.face_vertices[vertex_index++];

			dstv->position[0] = srcv[0];
			dstv->position[1] = srcv[1];
			dstv->position[2] = srcv[2];
			dstv->texCoord[0] = srcv[3];
			dstv->texCoord[1] = srcv[4];
			dstv->packedColor = RT_PACKED_COLOR_WHITE;
			dstv->cluster     = face->cluster;
		}
	}

	// The grid carries its own rows; the identity table leaves the map's visdata in place, and
	// RT_BuildWorldClusters measured the span the renderer walks in it.
	const uint8_t *pvis_data = model->visdata;
	uint32_t       vis_data_size = (uint32_t)rt_worldclusters.vis_data_size;

	if (rt_worldclusters.vis_data)
		pvis_data = rt_worldclusters.vis_data;

	RgWorldLightsUploadInfo info =
	{
		.flags           = CVAR_TO_BOOL (rt_worldlights_stats) ? RG_WORLD_LIGHTS_UPLOAD_PRINT_STATS_BIT : 0,
		.numClusters     = (uint32_t)rt_worldlights.num_clusters,
		.pClusterMins    = rt_worldclusters.cluster_mins,
		.pClusterMaxs    = rt_worldclusters.cluster_maxs,
		.pClusterFlags   = rt_worldclusters.cluster_flags,
		.numFaces        = (uint32_t)rt_worldlights.num_faces,
		.pFaces          = rt_worldlights.faces,
		.numFaceVertices = (uint32_t)rt_worldlights.num_face_vertices,
		.pFaceVertices   = rt_worldlights.face_vertices,
		.pVisData        = pvis_data,
		.visDataSize     = vis_data_size,
		.pvsRowBytes     = rt_worldclusters.pvs_row_bytes,
		.pVisOffsets     = rt_worldclusters.vis_offsets,
		.pClusterSkyVisibility = rt_worldclusters.sky_vis,
	};

	RgResult r = rgUploadWorldLights (vulkan_globals.instance, &info);
	RG_CHECK (r);
}

void RT_PrintEmissiveStats (void)
{
	RT_LightReportPrint ("emissive pass: %i surfaces considered -> %i static world lights baked (whole map), %i entity lights uploaded (all passes)\n",
		rt_emis_stats.surfaces, rt_emis_stats.static_queued, rt_emis_stats.dynamic);
	RT_LightReportPrint ("rejected: %i no light material, %i no light color, %i lightstyle off, %i degenerate, %i of a dark animation frame\n",
		rt_emis_stats.no_material, rt_emis_stats.no_color, rt_emis_stats.style_off, rt_emis_stats.degenerate, rt_emis_stats.frame_off);

	if (rt_emis_stats.glow_lights || rt_emis_stats.glow_fallback)
		RT_LightReportPrint ("partial-glow textures: %i faces built %i polygon lights, %i faces fell back to the whole surface\n",
			rt_emis_stats.glow_faces, rt_emis_stats.glow_lights, rt_emis_stats.glow_fallback);

	if (rt_emis_stats.static_dropped || rt_wldlights_emissive_count >= MAX_WORLDLIGHTS_COUNT)
		RT_LightReportPrint ("WARNING: the static world-light list is full (%i/%i), %i dropped - "
			"raise MAX_WORLDLIGHTS_COUNT or reduce emissive surfaces\n",
			rt_wldlights_emissive_count, MAX_WORLDLIGHTS_COUNT, rt_emis_stats.static_dropped);

	for (int i = 0; i < rt_emis_skip_num; i++)
		RT_LightReportPrint ("  skipped %-16s x%-5i (%s)\n", rt_emis_skip_texture[i], rt_emis_skip_count[i], rt_emis_skip_reason[i]);

	if (rt_emis_skip_num >= RT_EMIS_SKIP_NAMES)
		RT_LightReportPrint ("  ... more rejected textures not listed\n");

	for (int i = 0; i < rt_emis_watch_num; i++)
	{
		const rt_emis_watch_t *w = &rt_emis_watch[i];
		RT_LightReportPrint ("  <%s> visible %i -> lights %i (rejected %i no material, %i no color, %i style off, %i degenerate, %i dark frame)\n",
			w->name, w->surfaces, w->lights, w->no_material, w->no_color, w->style_off, w->degenerate, w->frame_off);

		if (w->hist_frames > 0)
			RT_LightReportPrint ("       since the filter was set: %i frames visible, lit %i, dark %i, %i with a lightstyle reject, %i on a dark animation frame, lights %i..%i%s\n",
				w->hist_frames, w->hist_lit, w->hist_dark, w->hist_style_off, w->hist_frame_off,
				(w->hist_min_lights > w->hist_max_lights) ? 0 : w->hist_min_lights, w->hist_max_lights,
				(w->hist_lit > 0 && w->hist_dark > 0) ? "   <-- INTERMITTENT" : "");

		if (w->style_count > 0 || w->min_style_scale < 1.0f)
		{
			RT_LightReportPrint ("       lightstyle: scale %.2f on styles", w->min_style_scale);
			for (int j = 0; j < w->style_count; j++)
				RT_LightReportPrint (" %i(value %i)", w->styles[j], d_lightstylevalue[w->styles[j]]);
			if (w->style_count == 0)
				RT_LightReportPrint (" (none: surface is unlit by style)");
			RT_LightReportPrint ("%s\n", (w->min_style_scale <= 0.0f) ? "  <-- LIGHT DROPPED WHILE THE STYLE IS DARK" : "");
		}
	}

	if (rt_light_report_filter.string[0] && rt_emis_watch_num == 0)
		RT_LightReportPrint ("  (no visible surface matched \"%s\" this frame)\n", rt_light_report_filter.string);
}

void RT_LightReport_f (void)
{
	char filter[64] = "";

	if (Cmd_Argc () > 1)
	{
		const char *a1 = Cmd_Argv (1);
		if (a1[0] >= '0' && a1[0] <= '9')
		{
			if (Cmd_Argc () > 2)
				q_snprintf (filter, sizeof (filter), "%s", Cmd_Argv (2));
		}
		else
		{
			q_snprintf (filter, sizeof (filter), "%s", a1);
		}
	}

	Cvar_Set ("rt_light_report_filter", filter);

	RT_PrintEmissiveStats ();
	RT_LightReportPrint ("\n");
	RT_ClusterLightReport_f ();
}
