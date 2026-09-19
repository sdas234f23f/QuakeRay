/*
Copyright (C) 1996-2001 Id Software, Inc.
Copyright (C) 2002-2009 John Fitzgibbons and others
Copyright (C) 2010-2014 QuakeSpasm developers

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
// r_light.c

#include "quakedef.h"

int r_dlightframecount;

extern cvar_t r_flatlightstyles; // johnfitz
extern cvar_t r_lerplightstyles;
extern cvar_t r_gpulightmapupdate;

extern SDL_mutex *lightcache_mutex;

/*
==================
R_AnimateLight
==================
*/
void R_AnimateLight (void)
{
	int    i, j, k, n;
	double f;

	//
	// light animations
	// 'm' is normal light, 'a' is no light, 'z' is double bright
	i = f = cl.time * 10;
	for (j = 0; j < MAX_LIGHTSTYLES; j++)
	{
		if (!cl_lightstyle[j].length)
		{
			d_lightstylevalue[j] = 256;
			continue;
		}
		// johnfitz -- r_flatlightstyles
		if (r_flatlightstyles.value == 2)
			k = n = cl_lightstyle[j].peak - 'a';
		else if (r_flatlightstyles.value == 1)
			k = n = cl_lightstyle[j].average - 'a';
		else
		{
			k = cl_lightstyle[j].map[i % cl_lightstyle[j].length] - 'a';
			n = cl_lightstyle[j].map[(i + 1) % cl_lightstyle[j].length] - 'a';
		}
		if (!r_gpulightmapupdate.value || !r_lerplightstyles.value || (r_lerplightstyles.value < 2 && abs (n - k) >= ('m' - 'a') / 2))
			n = k;
		d_lightstylevalue[j] = (k + (n - k) * (f - i)) * 22;
		// johnfitz
	}
}

/*
=============================================================================

DYNAMIC LIGHTS

=============================================================================
*/

/*
=============
R_MarkLights -- johnfitz -- rewritten to use LordHavoc's lighting speedup
=============
*/
void R_MarkLights (dlight_t *light, int num, mnode_t *node)
{
	mplane_t    *splitplane;
	msurface_t  *surf;
	vec3_t       impact;
	float        dist, l, maxdist;
	unsigned int i;
	int          j, s, t;

start:

	if (node->contents < 0)
		return;

	splitplane = node->plane;
	if (splitplane->type < 3)
		dist = light->origin[splitplane->type] - splitplane->dist;
	else
		dist = DotProduct (light->origin, splitplane->normal) - splitplane->dist;

	if (dist > light->radius)
	{
		node = node->children[0];
		goto start;
	}
	if (dist < -light->radius)
	{
		node = node->children[1];
		goto start;
	}

	maxdist = light->radius * light->radius;
	// mark the polygons
	surf = cl.worldmodel->surfaces + node->firstsurface;
	for (i = 0; i < node->numsurfaces; i++, surf++)
	{
		for (j = 0; j < 3; j++)
			impact[j] = light->origin[j] - surf->plane->normal[j] * dist;
		// clamp center of light to corner and check brightness
		l = DotProduct (impact, surf->texinfo->vecs[0]) + surf->texinfo->vecs[0][3] - surf->texturemins[0];
		s = l + 0.5;
		if (s < 0)
			s = 0;
		else if (s > surf->extents[0])
			s = surf->extents[0];
		s = l - s;
		l = DotProduct (impact, surf->texinfo->vecs[1]) + surf->texinfo->vecs[1][3] - surf->texturemins[1];
		t = l + 0.5;
		if (t < 0)
			t = 0;
		else if (t > surf->extents[1])
			t = surf->extents[1];
		t = l - t;
		// compare to minimum light
		if ((s * s + t * t + dist * dist) < maxdist)
		{
			if (surf->dlightframe != r_dlightframecount) // not dynamic until now
			{
				surf->dlightbits[num >> 5] = 1U << (num & 31);
				surf->dlightframe = r_dlightframecount;
			}
			else // already dynamic
				surf->dlightbits[num >> 5] |= 1U << (num & 31);
		}
	}

	if (node->children[0]->contents >= 0)
		R_MarkLights (light, num, node->children[0]);
	if (node->children[1]->contents >= 0)
		R_MarkLights (light, num, node->children[1]);
}

/*
=============================================================================
LIGHT SAMPLING

=============================================================================
*/

static void InterpolateLightmap (vec3_t color, msurface_t *surf, int ds, int dt)
{
	byte *lightmap;
	int   maps, line3, dsfrac = ds & 15, dtfrac = dt & 15, r00 = 0, g00 = 0, b00 = 0, r01 = 0, g01 = 0, b01 = 0, r10 = 0, g10 = 0, b10 = 0, r11 = 0, g11 = 0,
					 b11 = 0;
	int scale;
	line3 = ((surf->extents[0] >> 4) + 1) * 3;

	lightmap = surf->samples + ((dt >> 4) * ((surf->extents[0] >> 4) + 1) + (ds >> 4)) * 3; // LordHavoc: *3 for color

	for (maps = 0; maps < MAXLIGHTMAPS && surf->styles[maps] != 255; maps++)
	{
		scale = d_lightstylevalue[surf->styles[maps]];
		r00 += lightmap[0] * scale;
		g00 += lightmap[1] * scale;
		b00 += lightmap[2] * scale;
		r01 += lightmap[3] * scale;
		g01 += lightmap[4] * scale;
		b01 += lightmap[5] * scale;
		r10 += lightmap[line3 + 0] * scale;
		g10 += lightmap[line3 + 1] * scale;
		b10 += lightmap[line3 + 2] * scale;
		r11 += lightmap[line3 + 3] * scale;
		g11 += lightmap[line3 + 4] * scale;
		b11 += lightmap[line3 + 5] * scale;
		lightmap += ((surf->extents[0] >> 4) + 1) * ((surf->extents[1] >> 4) + 1) * 3; // LordHavoc: *3 for colored lighting
	}

	color[0] = ((((((((r11 - r10) * dsfrac) >> 4) + r10) - ((((r01 - r00) * dsfrac) >> 4) + r00)) * dtfrac) >> 4) + ((((r01 - r00) * dsfrac) >> 4) + r00)) *
	           (1.f / 256.f);
	color[1] = ((((((((g11 - g10) * dsfrac) >> 4) + g10) - ((((g01 - g00) * dsfrac) >> 4) + g00)) * dtfrac) >> 4) + ((((g01 - g00) * dsfrac) >> 4) + g00)) *
	           (1.f / 256.f);
	color[2] = ((((((((b11 - b10) * dsfrac) >> 4) + b10) - ((((b01 - b00) * dsfrac) >> 4) + b00)) * dtfrac) >> 4) + ((((b01 - b00) * dsfrac) >> 4) + b00)) *
	           (1.f / 256.f);
}

/*
=============
RecursiveLightPoint -- johnfitz -- replaced entire function for lit support via lordhavoc
=============
*/
int RecursiveLightPoint (lightcache_t *cache, mnode_t *node, vec3_t rayorg, vec3_t start, vec3_t end, float *maxdist)
{
	float  front, back, frac;
	vec3_t mid;

loc0:
	if (node->contents < 0)
		return false; // didn't hit anything

	// calculate mid point
	if (node->plane->type < 3)
	{
		front = start[node->plane->type] - node->plane->dist;
		back = end[node->plane->type] - node->plane->dist;
	}
	else
	{
		front = DotProduct (start, node->plane->normal) - node->plane->dist;
		back = DotProduct (end, node->plane->normal) - node->plane->dist;
	}

	// LordHavoc: optimized recursion
	if ((back < 0) == (front < 0))
	//		return RecursiveLightPoint (cache, node->children[front < 0], rayorg, start, end, maxdist);
	{
		node = node->children[front < 0];
		goto loc0;
	}

	frac = front / (front - back);
	mid[0] = start[0] + (end[0] - start[0]) * frac;
	mid[1] = start[1] + (end[1] - start[1]) * frac;
	mid[2] = start[2] + (end[2] - start[2]) * frac;

	// go down front side
	if (RecursiveLightPoint (cache, node->children[front < 0], rayorg, start, mid, maxdist))
		return true; // hit something
	else
	{
		unsigned int i;
		int          ds, dt;
		msurface_t  *surf;

		surf = cl.worldmodel->surfaces + node->firstsurface;
		for (i = 0; i < node->numsurfaces; i++, surf++)
		{
			float  sfront, sback, dist;
			vec3_t raydelta;

			if (surf->flags & SURF_DRAWTILED)
				continue; // no lightmaps

			// ericw -- added double casts to force 64-bit precision.
			// Without them the zombie at the start of jam3_ericw.bsp was
			// incorrectly being lit up in SSE builds.
			ds = (int)((double)DoublePrecisionDotProduct (mid, surf->texinfo->vecs[0]) + surf->texinfo->vecs[0][3]);
			dt = (int)((double)DoublePrecisionDotProduct (mid, surf->texinfo->vecs[1]) + surf->texinfo->vecs[1][3]);

			if (ds < surf->texturemins[0] || dt < surf->texturemins[1])
				continue;

			ds -= surf->texturemins[0];
			dt -= surf->texturemins[1];

			if (ds > surf->extents[0] || dt > surf->extents[1])
				continue;

			if (surf->plane->type < 3)
			{
				sfront = rayorg[surf->plane->type] - surf->plane->dist;
				sback = end[surf->plane->type] - surf->plane->dist;
			}
			else
			{
				sfront = DotProduct (rayorg, surf->plane->normal) - surf->plane->dist;
				sback = DotProduct (end, surf->plane->normal) - surf->plane->dist;
			}
			VectorSubtract (end, rayorg, raydelta);
			dist = sfront / (sfront - sback) * VectorLength (raydelta);

			if (!surf->samples)
			{
				// We hit a surface that is flagged as lightmapped, but doesn't have actual lightmap info.
				// Instead of just returning black, we'll keep looking for nearby surfaces that do have valid samples.
				// This fixes occasional pitch-black models in otherwise well-lit areas in DOTM (e.g. mge1m1, mge4m1)
				// caused by overlapping surfaces with mixed lighting data.
				const float nearby = 8.f;
				dist += nearby;
				*maxdist = q_min (*maxdist, dist);
				continue;
			}

			if (dist < *maxdist)
			{
				cache->surfidx = surf - cl.worldmodel->surfaces + 1;
				cache->ds = ds;
				cache->dt = dt;
			}
			else
			{
				cache->surfidx = -1;
			}

			return true; // success
		}

		// go down back side
		return RecursiveLightPoint (cache, node->children[front >= 0], rayorg, mid, end, maxdist);
	}
}

/*
=============
R_LightPoint -- johnfitz -- replaced entire function for lit support via lordhavoc
=============
*/
int R_LightPoint (vec3_t p, lightcache_t *cache, vec3_t *lightcolor)
{
	vec3_t end;
	float  maxdist = 8192.f; // johnfitz -- was 2048

	if (!cl.worldmodel->lightdata)
	{
		(*lightcolor)[0] = (*lightcolor)[1] = (*lightcolor)[2] = 255;
		return 255;
	}

	end[0] = p[0];
	end[1] = p[1];
	end[2] = p[2] - maxdist;

	(*lightcolor)[0] = (*lightcolor)[1] = (*lightcolor)[2] = 0;

	SDL_mutex *mtx = cache->mutex ? cache->mutex : lightcache_mutex;
	SDL_LockMutex (mtx);
	if (!cache || cache->surfidx <= 0 // no cache or pitch black
	    || cache->surfidx > cl.worldmodel->numsurfaces || fabsf (cache->pos[0] - p[0]) >= 1.f || fabsf (cache->pos[1] - p[1]) >= 1.f ||
	    fabsf (cache->pos[2] - p[2]) >= 1.f)
	{
		cache->surfidx = 0;
		VectorCopy (p, cache->pos);
		RecursiveLightPoint (cache, cl.worldmodel->nodes, p, p, end, &maxdist);
	}

	if (cache && cache->surfidx > 0)
		InterpolateLightmap (*lightcolor, cl.worldmodel->surfaces + cache->surfidx - 1, cache->ds, cache->dt);
	SDL_UnlockMutex (mtx);

	return (((*lightcolor)[0] + (*lightcolor)[1] + (*lightcolor)[2]) * (1.0f / 3.0f));
}



extern cvar_t rt_elight_normaliz, rt_elight_default, rt_elight_default_mdl, rt_elight_radius, rt_elight_threshold;
extern cvar_t rt_truelight;
extern cvar_t rt_materials_only;
extern cvar_t rt_poi_trigger, rt_poi_func, rt_poi_weapon, rt_poi_pwrup, rt_poi_armor, rt_poi_key, rt_poi_health, rt_poi_ammo;
extern cvar_t rt_poi_distthresh, rt_poi_distthresh_super;
extern cvar_t rt_light_reach;
extern cvar_t rt_light_reach_max;
extern cvar_t rt_cluster_incremental;
extern cvar_t rt_light_report_filter;


static qboolean StartsWith (const char *val, const char *begin)
{
	return strncmp (val, begin, strlen (begin)) == 0;
}


// look https://www.gamers.org/dEngine/quake/QDP/qmapspec.html#2.3.1
// for classnames

static qboolean IsClassname_Light (const char *classname)
{
	return StartsWith (classname, "light");
}

static qboolean IsClassname_LightWithModel (const char *classname)
{
	// For example,
	//    "light_fluoro"
	//    "light_fluorospark"
	//    "light_globe"
	//    "light_torch_small_walltorch"
	//    "light_flame_small_yellow"
	//    "light_flame_large_yellow"
	//    "light_flame_small_white"
	// but not just "light"
	
	return StartsWith (classname, "light_");
}

static qboolean IsClassname_Offsetted (const char *classname)
{
	// to prevent light source being inside the flame model
	return strcmp (classname, "light_torch_small_walltorch") == 0;
}

static qboolean IsClassname_PointOfInterest (const char *classname, qboolean *out_superimportant)
{
	if (CVAR_TO_BOOL (rt_poi_trigger))
	{
		if (StartsWith (classname, "trigger") || strcmp (classname, "info_teleport_destination") == 0)
		{
			*out_superimportant = true;
			return true;
		}
	}

	if (CVAR_TO_BOOL (rt_poi_func))
	{
		if (StartsWith (classname, "func"))
		{
			*out_superimportant = true;
			return true;
		}
	}

	if (CVAR_TO_BOOL (rt_poi_weapon))
	{
		if (strcmp (classname, "item_weapon") == 0 ||
			StartsWith(classname, "weapon"))
		{
			return true;
		}
	}

	if (StartsWith (classname, "item"))
	{
		if (CVAR_TO_BOOL (rt_poi_pwrup))
		{
			if (StartsWith (classname, "item_artifact"))
			{
				return true;
			}
		}

		if (CVAR_TO_BOOL (rt_poi_armor))
		{
			if (StartsWith (classname, "item_armor"))
			{
				return true;
			}
		}

		if (CVAR_TO_BOOL (rt_poi_key))
		{
			if (strcmp (classname, "item_sigil") == 0 || 
				StartsWith (classname, "item_key"))
			{
				*out_superimportant = true;
				return true;
			}
		}

		if (CVAR_TO_BOOL (rt_poi_ammo))
		{
			if (strcmp (classname, "item_cells") == 0 || 
				strcmp (classname, "item_rockets") == 0 || 
				strcmp (classname, "item_shells") == 0 ||
			    strcmp (classname, "item_spikes") == 0)
			{
				return true;
			}
		}

		if (CVAR_TO_BOOL (rt_poi_health))
		{
			if (strcmp (classname, "item_health") == 0)
			{
				return true;
			}
		}
	}

	return false;
}



typedef struct rt_poi_s
{
	vec3_t origin;
	qboolean is_super_imporant;
} rt_poi_t;

rt_poi_t *rt_poi = NULL;
int       rt_poi_count = 0;
int       rt_poi_allocated = 0;

static void RT_ParsePointsOfInterest ()
{
	rt_poi_count = 0;

    char key[128], value[4096];

    if (!cl.worldmodel)
	{
		return;
	}

	const char* data = cl.worldmodel->entities;
	if (!data)
	{
		return;
	}
	
	rt_poi_t cur_values = {0};
	int      cur_state = 0;

    #define CUR_STRUCT_STARTED 1
    #define CUR_IS_POI         2
    #define CUR_FOUND_ORIGIN   4
	    
	while (1)
	{
		data = COM_Parse (data);
		if (!data)
			return; // error

	    if (com_token[0] == '{')
	    {
			memset (&cur_values, 0, sizeof (cur_values));
			cur_state = CUR_STRUCT_STARTED;
            continue;
	    }
	    else if (com_token[0] == '}')
		{
			if ((cur_state & CUR_STRUCT_STARTED) &&
			    (cur_state & CUR_IS_POI) &&
			    (cur_state & CUR_FOUND_ORIGIN))
			{
				if (rt_poi_count >= rt_poi_allocated)
				{
					rt_poi_allocated += 256;
					rt_poi = Mem_Realloc (rt_poi, sizeof (rt_poi_t) * rt_poi_allocated);
				}

				rt_poi[rt_poi_count] = cur_values;
				rt_poi_count++;
			}

			cur_state = 0; // end of struct
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
			return; // error
		q_strlcpy (value, com_token, sizeof (value));

		
		if (strcmp (key, "classname") == 0)
		{
			qboolean is_super = 0;

			if (IsClassname_PointOfInterest (value, &is_super))
			{
				cur_state |= CUR_IS_POI;
				cur_values.is_super_imporant = is_super;
			}
		}
		else if (strcmp (key, "origin") == 0)
		{
			vec3_t tmpvec;
			int    components = sscanf (value, "%f %f %f", &tmpvec[0], &tmpvec[1], &tmpvec[2]);

			if (components == 3)
			{
				cur_values.origin[0] = tmpvec[0];
				cur_values.origin[1] = tmpvec[1];
				cur_values.origin[2] = tmpvec[2];
				cur_state |= CUR_FOUND_ORIGIN;
			}
		}
	}

}

static qboolean IsAroundPOI (vec3_t origin)
{
	float threshold = METRIC_TO_QUAKEUNIT (CVAR_TO_FLOAT (rt_poi_distthresh));
	float threshold_loose = METRIC_TO_QUAKEUNIT (CVAR_TO_FLOAT (rt_poi_distthresh_super));

	threshold *= threshold;
	threshold_loose *= threshold_loose;

	for (int i = 0; i < rt_poi_count;i++)
	{
		const rt_poi_t *src = &rt_poi[i];

		vec3_t v;
		VectorSubtract (src->origin, origin, v);

		float distsq_thresh = src->is_super_imporant ? threshold_loose : threshold;

		if (DotProduct (v, v) < distsq_thresh)
		{
			return true;
		}
	}

	return false;
}



typedef struct rt_elight_s
{
	int      state;
	vec3_t   origin;
	float    intensity;
	int      lightstyle;
	qboolean is_around_poi;
} rt_elight_t;

rt_elight_t *rt_elights = NULL;
int          rt_elights_count = 0;
int          rt_elights_allocated = 0;

/* rt_elights indexed by lightstyle; RT_NearestStyledLightDistance only reads one style. */
static int  rt_styled_elights[MAX_LIGHTSTYLES];
static int  rt_styled_elight_counts[MAX_LIGHTSTYLES];
static int *rt_styled_elight_index = NULL;
static int  rt_styled_elight_revision = -1;
static int  rt_elights_revision = 0;

#define STRUCT_STATE_STRUCT_STARTED       1
#define STRUCT_STATE_FOUND_LIGHTCLASSNAME 2
#define STRUCT_STATE_FOUND_ORIGIN         4
#define STRUCT_STATE_FOUND_INTENSITY      8
#define STRUCT_STATE_FOUND_WITH_MODEL     16
#define STRUCT_STATE_FOUND_LIGHTSTYLE     32
#define STRUCT_STATE_FOUND_APPLY_OFFSET   64

// Parse worldmodel->entities, to find static lights
void RT_ParseElights ()
{
	rt_elights_count = 0;
	rt_elights_revision++;

	RT_ParsePointsOfInterest ();


	char key[128], value[4096];

    if (!cl.worldmodel)
	{
		return;
	}

	const char* data = cl.worldmodel->entities;
	if (!data)
	{
		return;
	}
	
	rt_elight_t struct_values = {0};

	while (1)
	{
		data = COM_Parse (data);
		if (!data)
			return; // error

	    if (com_token[0] == '{')
	    {
			memset (&struct_values, 0, sizeof (struct_values));
			struct_values.state = STRUCT_STATE_STRUCT_STARTED;
            continue;
	    }
	    else if (com_token[0] == '}')
		{
			if ((struct_values.state & STRUCT_STATE_STRUCT_STARTED) &&
			    (struct_values.state & STRUCT_STATE_FOUND_LIGHTCLASSNAME) &&
			    (struct_values.state & STRUCT_STATE_FOUND_ORIGIN))
			{
				struct_values.is_around_poi = IsAroundPOI (struct_values.origin);


				if (rt_elights_count >= rt_elights_allocated)
				{
					rt_elights_allocated += 256;
					rt_elights = Mem_Realloc (rt_elights, sizeof (rt_elight_t) * rt_elights_allocated);
				}
				rt_elights[rt_elights_count] = struct_values;
				rt_elights_count++;
			}

			struct_values.state = 0; // end of struct
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
			return; // error
		q_strlcpy (value, com_token, sizeof (value));

		
		if (strcmp (key, "classname") == 0)
		{
			if (IsClassname_Light (value))
			{
				struct_values.state |= STRUCT_STATE_FOUND_LIGHTCLASSNAME;
			}

			if (IsClassname_LightWithModel (value))
			{
				struct_values.state |= STRUCT_STATE_FOUND_WITH_MODEL;

				if (IsClassname_Offsetted (value))
				{
					struct_values.state |= STRUCT_STATE_FOUND_APPLY_OFFSET;
				}
			}
		}
		else if (strcmp (key, "origin") == 0)
		{
			vec3_t tmpvec;
			int    components = sscanf (value, "%f %f %f", &tmpvec[0], &tmpvec[1], &tmpvec[2]);

			if (components == 3)
			{
				struct_values.origin[0] = tmpvec[0];
				struct_values.origin[1] = tmpvec[1];
				struct_values.origin[2] = tmpvec[2];
				struct_values.state |= STRUCT_STATE_FOUND_ORIGIN;
			}
		}
		else if (strcmp (key, "light") == 0)
		{
			float tmpval = strtof(value, NULL);

		    if (tmpval > 0.0f)
			{
				struct_values.intensity = tmpval;
				struct_values.state |= STRUCT_STATE_FOUND_INTENSITY;
			}
		}
		else if (strcmp (key, "style") == 0)
		{
			int tmpval = strtol (value, NULL, 10);

			if (tmpval >= 0 && tmpval < MAX_LIGHTSTYLES)
			{
				struct_values.lightstyle = tmpval;
				struct_values.state |= STRUCT_STATE_FOUND_LIGHTSTYLE;
			}
		}
	}
}

qboolean RT_AllowFakeLights (void)
{
	return !CVAR_TO_BOOL (rt_materials_only) && CVAR_TO_FLOAT (rt_truelight) < 2;
}

static void RT_BuildStyledLightIndex (void)
{
	int starts[MAX_LIGHTSTYLES + 1];
	int counts[MAX_LIGHTSTYLES];
	int fill[MAX_LIGHTSTYLES];

	memset (counts, 0, sizeof (counts));

	for (int i = 0; i < rt_elights_count; i++)
	{
		const rt_elight_t *src = &rt_elights[i];

		if (src->state & STRUCT_STATE_FOUND_LIGHTSTYLE)
			counts[src->lightstyle]++;
	}

	starts[0] = 0;
	for (int s = 0; s < MAX_LIGHTSTYLES; s++)
	{
		rt_styled_elights[s]       = starts[s];
		rt_styled_elight_counts[s] = counts[s];
		starts[s + 1]              = starts[s] + counts[s];
	}

	const int total = starts[MAX_LIGHTSTYLES];

	if (total > 0)
	{
		rt_styled_elight_index = Mem_Realloc (rt_styled_elight_index, sizeof (int) * total);
	}

	memcpy (fill, starts, sizeof (fill));

	/* Ascending source order: a style's list visits the entities the full scan used to. */
	for (int i = 0; i < rt_elights_count; i++)
	{
		const rt_elight_t *src = &rt_elights[i];

		if (!(src->state & STRUCT_STATE_FOUND_LIGHTSTYLE))
			continue;

		rt_styled_elight_index[fill[src->lightstyle]++] = i;
	}

	rt_styled_elight_revision = rt_elights_revision;
}

float RT_NearestStyledLightDistance (int style, const vec3_t point)
{
	if (style < 0 || style >= MAX_LIGHTSTYLES)
	{
		return -1.0f;
	}

	if (rt_styled_elight_revision != rt_elights_revision)
	{
		RT_BuildStyledLightIndex ();
	}

	const int start = rt_styled_elights[style];
	const int count = rt_styled_elight_counts[style];
	float     nearest = -1.0f;

	for (int k = 0; k < count; k++)
	{
		const rt_elight_t *src = &rt_elights[rt_styled_elight_index[start + k]];

		vec3_t delta;
		VectorSubtract (src->origin, point, delta);

		const float dist = VectorLength (delta);

		if (nearest < 0.0f || dist < nearest)
			nearest = dist;
	}

	return nearest;
}

void RT_UploadAllElights ()
{
	if (CVAR_TO_FLOAT (rt_truelight) > 0 || CVAR_TO_BOOL (rt_materials_only))
	{
		return;
	}

	if (CVAR_TO_FLOAT (rt_elight_normaliz) < 0.5f)
	{
		return;
	}

	for (int i = 0; i < rt_elights_count; i++)
	{
		const rt_elight_t *src = &rt_elights[i];

		assert (src->state & STRUCT_STATE_STRUCT_STARTED);
		assert (src->state & STRUCT_STATE_FOUND_LIGHTCLASSNAME);
		assert (src->state & STRUCT_STATE_FOUND_ORIGIN);

		float quake_intensity;
		if (src->state & STRUCT_STATE_FOUND_INTENSITY)
		{
			quake_intensity = src->intensity;
		}
		else
		{
			quake_intensity = src->state & STRUCT_STATE_FOUND_WITH_MODEL ? CVAR_TO_FLOAT (rt_elight_default_mdl) : CVAR_TO_FLOAT (rt_elight_default);
		}

		qboolean accept = 
			quake_intensity >= CVAR_TO_FLOAT (rt_elight_threshold) &&
			CVAR_TO_FLOAT (rt_elight_threshold) >= 0;

		if (src->state & STRUCT_STATE_FOUND_WITH_MODEL)
		{
			accept = true;
		}

	    if ((src->state & STRUCT_STATE_FOUND_LIGHTSTYLE) && src->lightstyle > 0)
		{
			accept = true;
		}

		if (src->is_around_poi)
		{
			accept = true;
		}

		if (accept)
		{
			float intens = quake_intensity / CVAR_TO_FLOAT (rt_elight_normaliz);

			if (src->state & STRUCT_STATE_FOUND_LIGHTSTYLE)
			{
				float ls = (float)d_lightstylevalue[src->lightstyle] / 256.0f;
				intens *= CLAMP (0.0f, ls, 1.0f);
			}

			vec3_t color;
			RT_INIT_DEFAULT_LIGHT_COLOR (color);
			VectorScale (color, intens, color);
			RT_FIXUP_LIGHT_INTENSITY (color, true);

			RgSphericalLightUploadInfo info = {
				.uniqueID = (uint64_t)UINT16_MAX + i,
				.color = {color[0], color[1], color[2]},
				.position = {src->origin[0], src->origin[1], src->origin[2]},
				.radius = METRIC_TO_QUAKEUNIT (CVAR_TO_FLOAT (rt_elight_radius)),
			};

			// offset up a bit, so light is not inside the model itself
			if (src->state & STRUCT_STATE_FOUND_APPLY_OFFSET)
			{
				info.position.data[2] += METRIC_TO_QUAKEUNIT (0.75f);
			}

			RgResult r = rgUploadSphericalLight (vulkan_globals.instance, &info);
			RG_CHECK (r);

			RT_ClusterLightAdd (info.uniqueID, info.position.data, RT_ClusterLightReach ());
		}
	}
}

#define RT_CLUSTER_MAX_LIGHTS    1024

typedef struct rt_cluster_light_s
{
	uint64_t uniqueID;
	vec3_t   origin;
	float    reach;   /* Quake units, zero when the light states no reach of its own */
} rt_cluster_light_t;

static rt_cluster_light_t rt_cluster_lights[RT_CLUSTER_MAX_LIGHTS];
static int rt_cluster_light_count;
static qboolean rt_cluster_dropped_warned;

/* Slot a light's id was last registered at; the id comparison is what validates it. */
#define RT_CLUSTER_UID_HINTS 4096
static uint16_t rt_cluster_uid_hint[RT_CLUSTER_UID_HINTS];

static uint32_t RT_ClusterUidHint (uint64_t uniqueID)
{
	return (uint32_t) ((uniqueID * 0x9E3779B97F4A7C15ull) >> 52) & (RT_CLUSTER_UID_HINTS - 1);
}

typedef struct rt_light_diag_s
{
	uint64_t uniqueID;
	vec3_t   origin;
	qboolean resolved;
	int      granted;
	int      denied;
} rt_light_diag_t;

static rt_light_diag_t rt_light_diag[RT_CLUSTER_MAX_LIGHTS];

static int rt_light_diag_count;
static int rt_light_diag_unresolved;
static int rt_light_diag_granted;
static int rt_light_diag_denied;

static vec3_t rt_cluster_vieworg;

/* Mirror of the renderer's accounting, refreshed after every upload: the composition lives there
   now, and these counters are the only numbers the host still has on it. */
int rt_cluster_cache_hits;    /* frames that found the light set unchanged and reused the lists */
int rt_cluster_cache_misses;  /* frames that composed the lists again */
int rt_cluster_miss_set;      /* compositions set off by lights appearing or disappearing */
int rt_cluster_miss_move;     /* compositions set off by lights that moved, changed their reach,
                                 or resolved into a different leaf */
int rt_cluster_miss_other;    /* compositions with neither of those: a new map, a new top-up
                                 reach, or a frame the incremental path turned down */
int rt_cluster_last_grants;   /* (light, cluster) pairs the last composition granted */
int rt_cluster_last_denied;   /* pairs it refused because the cluster had filled its slots */
int rt_cluster_last_gated;    /* pairs it left out because the cluster stood beyond the light's reach */
int rt_cluster_reg_attempts;  /* RT_ClusterLightAdd calls of the frame */
int rt_cluster_reg_dropped;   /* additions refused by RT_CLUSTER_MAX_LIGHTS */
int rt_cluster_last_lights;   /* lights in the registry of the last frame that uploaded */
int rt_cluster_last_attempts; /* additions that frame attempted */
int rt_cluster_last_dropped;  /* additions that frame lost to the cap */

void RT_ClusterLightListsReset (void)
{
	rt_cluster_light_count = 0;
	rt_light_diag_count = 0;
	rt_light_diag_unresolved = 0;
	rt_light_diag_granted = 0;
	rt_light_diag_denied = 0;
	rt_cluster_reg_attempts = 0;
	rt_cluster_reg_dropped = 0;
	VectorCopy (r_refdef.vieworg, rt_cluster_vieworg);
}

/* The reach a light of a moving entity is registered with, from rt_light_reach_max: the distance
   the host promises such a light does not reach past, in Quake units. A light that moves is what
   makes the lists rebuild, so this is what keeps one entity from reaching every list of the map.
   A light of the map itself is registered with zero instead, and reaches wherever its own leaf
   sees, which is right for it: it stands where it stands every frame. */
float RT_ClusterLightReach (void)
{
	return METRIC_TO_QUAKEUNIT (CVAR_TO_FLOAT (rt_light_reach_max));
}

void RT_ClusterLightAdd (uint64_t uniqueID, const vec3_t origin, float reach)
{
	rt_cluster_reg_attempts++;

	if (rt_cluster_light_count >= RT_CLUSTER_MAX_LIGHTS)
	{
		rt_cluster_reg_dropped++;

		if (!rt_cluster_dropped_warned)
		{
			Con_DWarning ("RT: light count exceeded RT_CLUSTER_MAX_LIGHTS (%i), "
				"some lights will not be sampled by the RT renderer.\n",
				RT_CLUSTER_MAX_LIGHTS);
			rt_cluster_dropped_warned = true;
		}
		return;
	}

	/* A light that registers twice in one frame keeps its slot, but not the position or the
	   reach it was first seen at: a flame that is drawn by two passes, or a light that the
	   frame registers again after it moved, must be handed to the lists where it stands now. */
	/* The slot only says where to look; the comparison below says whether to trust it. */
	const uint32_t hint = RT_ClusterUidHint (uniqueID);

	if (rt_cluster_uid_hint[hint] < rt_cluster_light_count &&
	    rt_cluster_lights[rt_cluster_uid_hint[hint]].uniqueID == uniqueID)
	{
		const int i = rt_cluster_uid_hint[hint];

		VectorCopy (origin, rt_cluster_lights[i].origin);
		rt_cluster_lights[i].reach = reach;
		VectorCopy (origin, rt_light_diag[i].origin);
		return;
	}

	for (int i = 0; i < rt_cluster_light_count; i++)
	{
		if (rt_cluster_lights[i].uniqueID == uniqueID)
		{
			VectorCopy (origin, rt_cluster_lights[i].origin);
			rt_cluster_lights[i].reach = reach;
			VectorCopy (origin, rt_light_diag[i].origin);
			rt_cluster_uid_hint[hint] = (uint16_t) i;
			return;
		}
	}

	rt_cluster_uid_hint[hint] = (uint16_t) rt_cluster_light_count;

	rt_cluster_lights[rt_cluster_light_count].uniqueID = uniqueID;
	VectorCopy (origin, rt_cluster_lights[rt_cluster_light_count].origin);
	rt_cluster_lights[rt_cluster_light_count].reach = reach;

	rt_light_diag[rt_cluster_light_count].uniqueID = uniqueID;
	VectorCopy (origin, rt_light_diag[rt_cluster_light_count].origin);
	rt_light_diag[rt_cluster_light_count].resolved = false;
	rt_light_diag[rt_cluster_light_count].granted = 0;
	rt_light_diag[rt_cluster_light_count].denied = 0;
	rt_light_diag_count = rt_cluster_light_count + 1;

	rt_cluster_light_count++;
}

static mleaf_t *RT_ResolveLightLeaf (const vec3_t origin, qmodel_t *wm)
{
	static const vec3_t probeDirs[6] = {
		{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1},
	};
	static const float probeDists[3] = {2.0f, 8.0f, 24.0f};

	mleaf_t *leaf = Mod_PointInLeaf ((float *)origin, wm);
	if (leaf && leaf != wm->leafs && leaf->contents != CONTENTS_SOLID)
		return leaf;

	for (int d = 0; d < 3; d++)
	{
		for (int i = 0; i < 6; i++)
		{
			vec3_t probe;
			probe[0] = origin[0] + probeDists[d] * probeDirs[i][0];
			probe[1] = origin[1] + probeDists[d] * probeDirs[i][1];
			probe[2] = origin[2] + probeDists[d] * probeDirs[i][2];
			leaf = Mod_PointInLeaf (probe, wm);
			if (leaf && leaf != wm->leafs && leaf->contents != CONTENTS_SOLID)
				return leaf;
		}
	}

	return NULL;
}

int RT_ResolvePointCluster (const vec3_t p)
{
	mleaf_t *leaf = RT_ResolveLightLeaf (p, cl.worldmodel);
	if (!leaf)
		return 0;

	return RT_MapWorldCluster ((int)(leaf - cl.worldmodel->leafs));
}

/* Sources of the frame, handed to the renderer. It composes the per-cluster lists out of them,
   so the slot table, the PVS cache and the top-up grid that used to live here are gone: what is
   left on this side is the registry and the two things the renderer cannot derive from the map,
   the leaf each origin resolved into and the reach of the light. */
static RgClusterLightSource rt_cluster_sources[RT_CLUSTER_MAX_LIGHTS];

/* The leaf a light last resolved into. The walk is a function of the origin and the map, so a
   light that stands where it stood last frame resolves to the leaf it resolved to then, which is
   most of them: the map's lights, the entity lights of a world that is not moving and the dlights
   of a paused one. The identity is the light's id, never the slot, because a slot is reused by
   whatever registers into it next. */
typedef struct rt_leaf_cache_s
{
	uint64_t uniqueID;
	vec3_t   origin;
	int      leafIndex;  /* -1 when that origin resolved to no leaf at all */
	qboolean valid;
} rt_leaf_cache_t;

static rt_leaf_cache_t rt_leaf_cache[RT_CLUSTER_MAX_LIGHTS];

/* What the cached leaves are leaves of. A new level can land on the very addresses the old one
   had, so the map is identified by its parse as well, which every map load does afresh. */
static qboolean  rt_leaf_cache_map_valid;
static qmodel_t *rt_leaf_cache_wm;
static mleaf_t  *rt_leaf_cache_leafs;
static int       rt_leaf_cache_numleafs;
static int       rt_leaf_cache_revision;

void RT_ClusterLightListsUpload (void)
{
	qmodel_t *wm = cl.worldmodel;
	if (!wm || wm->type != mod_brush || !wm->leafs || wm->numleafs < 2)
		return;

	/* The registry is what the lists are built from, so how many lights were accepted, how many
	   additions were attempted and how many fell off the cap explain most of the churn. */
	rt_cluster_last_lights = rt_cluster_light_count;
	rt_cluster_last_attempts = rt_cluster_reg_attempts;
	rt_cluster_last_dropped = rt_cluster_reg_dropped;

	if (!rt_leaf_cache_map_valid || rt_leaf_cache_wm != wm || rt_leaf_cache_leafs != wm->leafs ||
	    rt_leaf_cache_numleafs != wm->numleafs || rt_leaf_cache_revision != rt_elights_revision)
	{
		rt_leaf_cache_map_valid = true;
		rt_leaf_cache_wm = wm;
		rt_leaf_cache_leafs = wm->leafs;
		rt_leaf_cache_numleafs = wm->numleafs;
		rt_leaf_cache_revision = rt_elights_revision;

		for (int i = 0; i < RT_CLUSTER_MAX_LIGHTS; i++)
			rt_leaf_cache[i].valid = false;
	}

	const double prof_resolve = RT_Prof_Begin ();

	for (int li = 0; li < rt_cluster_light_count; li++)
	{
		rt_leaf_cache_t *cached = &rt_leaf_cache[li];
		int              leafIndex;

		if (cached->valid && cached->uniqueID == rt_cluster_lights[li].uniqueID &&
		    VectorCompare (cached->origin, rt_cluster_lights[li].origin))
		{
			leafIndex = cached->leafIndex;
		}
		else
		{
			mleaf_t *leaf = RT_ResolveLightLeaf (rt_cluster_lights[li].origin, wm);
			leafIndex = leaf ? (int)(leaf - wm->leafs) : -1;

			cached->valid = true;
			cached->uniqueID = rt_cluster_lights[li].uniqueID;
			VectorCopy (rt_cluster_lights[li].origin, cached->origin);
			cached->leafIndex = leafIndex;
		}

		rt_cluster_sources[li].uniqueID = rt_cluster_lights[li].uniqueID;
		VectorCopy (rt_cluster_lights[li].origin, rt_cluster_sources[li].origin.data);
		rt_cluster_sources[li].cluster = (leafIndex >= 0)
			? (uint32_t)RT_MapWorldCluster (leafIndex)
			: (uint32_t)RG_CLUSTER_LIGHT_NO_CLUSTER;
		rt_cluster_sources[li].reach = rt_cluster_lights[li].reach;

		rt_light_diag[li].resolved = (leafIndex >= 0);

		if (leafIndex < 0)
			rt_light_diag_unresolved++;
	}

	if (prof_resolve != 0.0)
		RT_Prof_End (RT_PROF_CLUSTERS_RESOLVE, prof_resolve);

	const RgClusterLightSourcesUploadInfo info = {
		.numLights = (uint32_t)rt_cluster_light_count,
		.pLights = rt_cluster_sources,
		.topUpReach = METRIC_TO_QUAKEUNIT (CVAR_TO_FLOAT (rt_light_reach)),
		.allowIncremental = CVAR_TO_BOOL (rt_cluster_incremental) ? 1 : 0,
	};

	RgResult r = rgUploadClusterLightSources (vulkan_globals.instance, &info);
	RG_CHECK (r);

	RgClusterLightStats st;
	memset (&st, 0, sizeof (st));

	if (rgGetClusterLightStats (vulkan_globals.instance, &st) == RG_SUCCESS)
	{
		rt_cluster_last_grants = (int)st.grants;
		rt_cluster_last_denied = (int)st.denied;
		rt_cluster_last_gated = (int)st.reachGated;
		rt_light_diag_granted = (int)st.grants;
		rt_light_diag_denied = (int)st.denied;

		if (st.reusedFrames)
			rt_cluster_cache_hits++;

		if (st.composedFrames)
		{
			rt_cluster_cache_misses++;

			if (st.addedSources || st.removedSources)
				rt_cluster_miss_set++;
			else if (st.movedSources)
				rt_cluster_miss_move++;
			else
				rt_cluster_miss_other++;
		}

		/* The passes run inside the renderer, so their cost is reported rather than measured here:
		   vis is the PVS walk, topup the reach pass, fill the compaction of the slots into the
		   lists, upload the publication of them to the light manager. totalMs is the whole of
		   SetSources, the publication included, so the row that carries it holds the four rows
		   below it and says what the lists cost on their own rather than inside the whole
		   bracket. */
		RT_Prof_Sample (RT_PROF_CLUSTERS_LISTS, st.totalMs);
		RT_Prof_Sample (RT_PROF_CLUSTERS_VIS, st.visMs);
		RT_Prof_Sample (RT_PROF_CLUSTERS_TOPUP, st.topUpMs);
		RT_Prof_Sample (RT_PROF_CLUSTERS_FILL, st.fillMs);
		RT_Prof_Sample (RT_PROF_CLUSTERS_UPLOAD, st.publishMs);
	}

	/* Per-light slot accounting, already back on the order this frame registered the lights in,
	   so that rt_light_report can print it next to the light it belongs to. */
	static uint32_t diagGranted[RT_CLUSTER_MAX_LIGHTS];
	static uint32_t diagDenied[RT_CLUSTER_MAX_LIGHTS];
	uint32_t        diagCount = 0;

	if (rgGetClusterLightGrants (vulkan_globals.instance, diagGranted, diagDenied,
			(uint32_t)countof (diagGranted), &diagCount) == RG_SUCCESS)
	{
		const int n = (diagCount < (uint32_t)rt_light_diag_count) ? (int)diagCount : rt_light_diag_count;

		for (int i = 0; i < n; i++)
		{
			rt_light_diag[i].granted = (int)diagGranted[i];
			rt_light_diag[i].denied = (int)diagDenied[i];
		}
	}
}


static void RT_FormatLightId (char *out, size_t outSize, uint64_t uid)
{
	const int      type      = (int)(uid >> 60);
	const int      triangle  = (int)((uid >> 48) & 0xFFFull);
	const int      surfindex = (int)((uid >> 32) & 0xFFFFull);
	const unsigned ent       = (unsigned)(uid & 0xFFFFFFFFull);
	const char    *texname   = NULL;

	if (type == 1 && ent == ENT_UNIQUEID_WORLD && cl.worldmodel && cl.worldmodel->type == mod_brush &&
		surfindex < cl.worldmodel->numsurfaces)
	{
		msurface_t *surf = &cl.worldmodel->surfaces[surfindex];
		if (surf->texinfo && surf->texinfo->texture)
			texname = surf->texinfo->texture->name;
	}

	if (type == 1 && texname)
		q_snprintf (out, outSize, "world  surf %-4i %s", surfindex, texname);
	else if (type == 1)
		q_snprintf (out, outSize, "brush  ent %-5u surf %-4i tri %i", ent, surfindex, triangle);
	else if (type == 2)
		q_snprintf (out, outSize, "alias  ent %u", ent);
	else if (type == 3)
		q_snprintf (out, outSize, "sprite ent %u", ent);
	else
		q_snprintf (out, outSize, "type %i uid %016llx", type, (unsigned long long)uid);
}

static float RT_LightDiagDist (const rt_light_diag_t *d)
{
	const float dx = d->origin[0] - rt_cluster_vieworg[0];
	const float dy = d->origin[1] - rt_cluster_vieworg[1];
	const float dz = d->origin[2] - rt_cluster_vieworg[2];
	return sqrtf (dx * dx + dy * dy + dz * dz);
}

/* Slots rt_light_report prints of one cluster list. It follows the renderer's
   Q2_LIGHT_LIST_MAX_PER_CELL, which is 128, so a whole list is printed; a longer list is still
   reported as truncated rather than silently cut. */
#define RT_CLUSTER_REPORT_SLOTS 128


void RT_ClusterLightReport_f (void)
{
	const int maxLines = (Cmd_Argc () > 1) ? atoi (Cmd_Argv (1)) : 64;

	if (rt_light_diag_count <= 0)
	{
		Con_Printf ("RT lights: no cluster light state yet - load a map and look at the world first.\n");
		return;
	}

	Con_Printf ("RT lights: %i registered, %i dropped (no open leaf), %i cluster slots granted, %i denied\n",
		rt_cluster_light_count, rt_light_diag_unresolved, rt_light_diag_granted, rt_light_diag_denied);

	int      viewCluster = -1;
	int      viewFill = 0;
	uint64_t viewUids[RT_CLUSTER_REPORT_SLOTS];
	uint32_t viewCount = 0;

	if (cl.worldmodel && cl.worldmodel->type == mod_brush)
	{
		mleaf_t *viewleaf = Mod_PointInLeaf (rt_cluster_vieworg, cl.worldmodel);

		if (viewleaf && viewleaf != cl.worldmodel->leafs)
		{
			viewCluster = (int)(viewleaf - cl.worldmodel->leafs);

			if (rgGetClusterLightList (vulkan_globals.instance, (uint32_t)viewCluster,
					viewUids, (uint32_t)countof (viewUids), &viewCount) != RG_SUCCESS)
			{
				viewCount = 0;
			}

			viewFill = (int)viewCount;
		}
	}

	if (viewCluster < 0)
	{
		Con_Printf ("camera cluster: unavailable (camera is not in the world)\n");
	}
	else
	{
		Con_Printf ("camera cluster %i: %i lights sampled%s\n", viewCluster, viewFill,
			(viewFill >= RT_CLUSTER_REPORT_SLOTS) ? "  *** the read stopped at the slot count above ***" : "");
	}

	Con_Printf ("%-3s %5s %5s %8s  %-34s %s\n", "cls", "pvs", "no!", "dist", "light", "verdict");

	int shown = 0;

	const char *filter = rt_light_report_filter.string;
	static int  order[RT_CLUSTER_MAX_LIGHTS];
	int         order_num = 0;

	for (int li = 0; li < rt_light_diag_count; li++)
	{
		if (filter[0])
		{
			char probe[64];
			RT_FormatLightId (probe, sizeof (probe), rt_light_diag[li].uniqueID);
			if (!strstr (probe, filter))
				continue;
		}
		order[order_num++] = li;
	}


	for (int i = 1; i < order_num; i++)
	{
		const int key = order[i];
		const float keydist = RT_LightDiagDist (&rt_light_diag[key]);
		int j = i - 1;
		while (j >= 0 && RT_LightDiagDist (&rt_light_diag[order[j]]) > keydist)
		{
			order[j + 1] = order[j];
			j--;
		}
		order[j + 1] = key;
	}

	if (filter[0])
	{
		Con_Printf ("filter \"%s\": %i of %i registered lights match\n", filter, order_num, rt_light_diag_count);
		if (order_num == 0)
			Con_Printf ("  (this texture registered no light at all - see the emissive pass above)\n");
	}

	for (int oi = 0; oi < order_num; oi++)
	{
		const rt_light_diag_t *d = &rt_light_diag[order[oi]];

		qboolean inView = false;
		for (int s = 0; s < viewFill; s++)
		{
			if (viewUids[s] == d->uniqueID)
			{
				inView = true;
				break;
			}
		}

		const float dist = RT_LightDiagDist (d);

		const char *verdict;
		if (!d->resolved)
			verdict = "DROPPED: no open leaf for the light origin";
		else if (!d->granted && d->denied)
			verdict = "DROPPED: every cluster was full";
		else if (!d->granted)
			verdict = "DROPPED: no cluster accepted it";
		else if (!inView)
			verdict = "ok, but camera cluster does not sample it";
		else
			verdict = "ok";

		if (shown >= maxLines)
			break;
		shown++;

		char id[64];
		RT_FormatLightId (id, sizeof (id), d->uniqueID);
		Con_Printf ("%-3s %5i %5i %8.0f  %-34s %s\n", inView ? "yes" : "-", d->granted, d->denied, dist, id, verdict);
	}

	if (shown < order_num)
		Con_Printf ("... %i more matches (rt_light_report <line count>%s to print more)\n",
			order_num - shown, filter[0] ? " <texture>" : "");
}
