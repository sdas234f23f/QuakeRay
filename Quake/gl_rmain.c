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
// r_main.c

#include "quakedef.h"
#include "tasks.h"
#include "atomics.h"
#include "rt_lights.h"

// The editor's GUI depends on this task when tasks are on (see gl_screen.c).
task_handle_t rt_editor_draw_done_task = INVALID_TASK_HANDLE;
#include "qr_editor.h"
#include "photocam.h"
#include "observer.h"

int r_visframecount; // bumped when going to a new PVS
int r_framecount;    // used for dlight push checking
atomic_uint32_t rt_require_static_submit;
atomic_uint32_t rt_require_world_light_recollect;

mplane_t frustum[4];

QrMediaType rt_cameramedia = QR_MEDIA_TYPE_VACUUM;
qboolean    rt_lavaeffects = false;

// johnfitz -- rendering statistics
atomic_uint32_t rs_brushpolys, rs_aliaspolys, rs_skypolys, rs_particles, rs_fogpolys;
atomic_uint32_t rs_dynamiclightmaps, rs_brushpasses, rs_aliaspasses, rs_skypasses;
float           rt_world_draw_ms;

//
// view origin
//
vec3_t vup;
vec3_t vpn;
vec3_t vright;
vec3_t r_origin;
qboolean r_vieworg_valid;

float r_fovx, r_fovy; // johnfitz -- rendering fov may be different becuase of r_waterwarp

//
// screen size info
//
refdef_t r_refdef;

mleaf_t *r_viewleaf, *r_oldviewleaf;

int d_lightstylevalue[256]; // 8.8 fraction of base light value

cvar_t r_drawentities = {"r_drawentities", "1", CVAR_NONE};
cvar_t r_drawviewmodel = {"r_drawviewmodel", "1", CVAR_NONE};
cvar_t r_speeds = {"r_speeds", "0", CVAR_NONE};
cvar_t r_pos = {"r_pos", "0", CVAR_NONE};
cvar_t r_fullbright = {"r_fullbright", "0", CVAR_NONE};
cvar_t r_lightmap = {"r_lightmap", "0", CVAR_NONE};
cvar_t r_wateralpha = {"r_wateralpha", "1", CVAR_ARCHIVE};
cvar_t r_dynamic = {"r_dynamic", "0", CVAR_ARCHIVE}; // RT: was 1, but there are significant performance drops e.g. on E1M6 when lavaballs fly and touch a lot of surfaces
#if defined(USE_SIMD)
cvar_t r_simd = {"r_simd", "1", CVAR_ARCHIVE};
#endif

cvar_t gl_finish = {"gl_finish", "0", CVAR_NONE};
cvar_t gl_polyblend = {"gl_polyblend", "1", CVAR_NONE};
cvar_t gl_nocolors = {"gl_nocolors", "0", CVAR_NONE};

// johnfitz -- new cvars
cvar_t r_flatlightstyles = {"r_flatlightstyles", "0", CVAR_NONE};
cvar_t r_lerplightstyles = {"r_lerplightstyles", "1", CVAR_ARCHIVE}; // 0=off; 1=skip abrupt transitions; 2=always lerp
cvar_t gl_fullbrights = {"gl_fullbrights", "1", CVAR_ARCHIVE};
cvar_t gl_farclip = {"gl_farclip", "16384", CVAR_ARCHIVE};
cvar_t r_oldskyleaf = {"r_oldskyleaf", "0", CVAR_NONE};
cvar_t r_drawworld = {"r_drawworld", "1", CVAR_NONE};
cvar_t r_showtris = {"r_showtris", "0", CVAR_NONE};
cvar_t r_showbboxes = {"r_showbboxes", "0", CVAR_NONE};
cvar_t r_lerpmodels = {"r_lerpmodels", "1", CVAR_NONE};
cvar_t r_lerpmove = {"r_lerpmove", "1", CVAR_NONE};
cvar_t r_nolerp_list = {
	"r_nolerp_list",
	"progs/flame.mdl,progs/flame2.mdl,progs/braztall.mdl,progs/brazshrt.mdl,progs/longtrch.mdl,progs/flame_pyre.mdl,progs/v_saw.mdl,progs/"
	"v_xfist.mdl,progs/h2stuff/newfire.mdl",
	CVAR_NONE};

extern cvar_t r_vfog;
// johnfitz

cvar_t gl_zfix = {"gl_zfix", "1", CVAR_ARCHIVE}; // QuakeSpasm z-fighting fix

cvar_t r_lavaalpha = {"r_lavaalpha", "0", CVAR_NONE};
cvar_t r_telealpha = {"r_telealpha", "0", CVAR_NONE};
cvar_t r_slimealpha = {"r_slimealpha", "0", CVAR_NONE};

float map_wateralpha, map_lavaalpha, map_telealpha, map_slimealpha;
float map_fallbackalpha;

qboolean r_drawworld_cheatsafe, r_fullbright_cheatsafe, r_lightmap_cheatsafe; // johnfitz

cvar_t r_gpulightmapupdate = {"r_gpulightmapupdate", "0", CVAR_NONE};

cvar_t r_tasks = {"r_tasks", "0", CVAR_NONE};

extern cvar_t rt_dlight_intensity;
extern cvar_t rt_dlight_radius;
extern cvar_t rt_flashlight;
extern cvar_t rt_dlightspot_intensity;
extern cvar_t rt_sky_sun;
extern cvar_t rt_sky_sun_pitch;
extern cvar_t rt_sky_sun_yaw;
extern cvar_t rt_physical_sun;
extern cvar_t rt_materials_only;
extern cvar_t rt_cluster_dlights;
extern cvar_t rt_particle_proxy_gate;
extern cvar_t rt_glass_particles;
extern cvar_t rt_viewm_scale;

/*
=================
R_CullBox -- johnfitz -- replaced with new function from lordhavoc

Returns true if the box is completely outside the frustum
=================
*/
qboolean R_CullBox (vec3_t emins, vec3_t emaxs)
{
	int       i;
	mplane_t *p;
	byte      signbits;
	float     vec[3];
	for (i = 0; i < 4; i++)
	{
		p = frustum + i;
		signbits = p->signbits;
		vec[0] = ((signbits % 2) < 1) ? emaxs[0] : emins[0];
		vec[1] = ((signbits % 4) < 2) ? emaxs[1] : emins[1];
		vec[2] = ((signbits % 8) < 4) ? emaxs[2] : emins[2];
		if (p->normal[0] * vec[0] + p->normal[1] * vec[1] + p->normal[2] * vec[2] < p->dist)
			return true;
	}
	return false;
}
/*
===============
R_CullModelForEntity -- johnfitz -- uses correct bounds based on rotation
===============
*/
qboolean R_CullModelForEntity (entity_t *e)
{
	vec3_t mins, maxs;

	if (e->angles[0] || e->angles[2]) // pitch or roll
	{
		VectorAdd (e->origin, e->model->rmins, mins);
		VectorAdd (e->origin, e->model->rmaxs, maxs);
	}
	else if (e->angles[1]) // yaw
	{
		VectorAdd (e->origin, e->model->ymins, mins);
		VectorAdd (e->origin, e->model->ymaxs, maxs);
	}
	else // no rotation
	{
		VectorAdd (e->origin, e->model->mins, mins);
		VectorAdd (e->origin, e->model->maxs, maxs);
	}

	return R_CullBox (mins, maxs);
}

/*
===============
R_RotateForEntity -- johnfitz -- modified to take origin and angles instead of pointer to entity
===============
*/
#define DEG2RAD(a) ((a)*M_PI_DIV_180)
void R_RotateForEntity (float matrix[16], vec3_t origin, vec3_t angles)
{
	float translation_matrix[16];
	TranslationMatrix (translation_matrix, origin[0], origin[1], origin[2]);
	MatrixMultiply (matrix, translation_matrix);

	float rotation_matrix[16];
	RotationMatrix (rotation_matrix, DEG2RAD (angles[1]), 0, 0, 1);
	MatrixMultiply (matrix, rotation_matrix);
	RotationMatrix (rotation_matrix, DEG2RAD (-angles[0]), 0, 1, 0);
	MatrixMultiply (matrix, rotation_matrix);
	RotationMatrix (rotation_matrix, DEG2RAD (angles[2]), 1, 0, 0);
	MatrixMultiply (matrix, rotation_matrix);
}

//==============================================================================
//
// SETUP FRAME
//
//==============================================================================

int SignbitsForPlane (mplane_t *out)
{
	int bits, j;

	// for fast box on planeside test

	bits = 0;
	for (j = 0; j < 3; j++)
	{
		if (out->normal[j] < 0)
			bits |= 1 << j;
	}
	return bits;
}

/*
===============
TurnVector -- johnfitz

turn forward towards side on the plane defined by forward and side
if angle = 90, the result will be equal to side
assumes side and forward are perpendicular, and normalized
to turn away from side, use a negative angle
===============
*/
#define DEG2RAD(a) ((a)*M_PI_DIV_180)
void TurnVector (vec3_t out, const vec3_t forward, const vec3_t side, float angle)
{
	float scale_forward, scale_side;

	scale_forward = cos (DEG2RAD (angle));
	scale_side = sin (DEG2RAD (angle));

	out[0] = scale_forward * forward[0] + scale_side * side[0];
	out[1] = scale_forward * forward[1] + scale_side * side[1];
	out[2] = scale_forward * forward[2] + scale_side * side[2];
}

/*
===============
R_SetFrustum -- johnfitz -- rewritten
===============
*/
void R_SetFrustum (float fovx, float fovy)
{
	int i;

	TurnVector (frustum[0].normal, vpn, vright, fovx / 2 - 90); // right plane
	TurnVector (frustum[1].normal, vpn, vright, 90 - fovx / 2); // left plane
	TurnVector (frustum[2].normal, vpn, vup, 90 - fovy / 2);    // bottom plane
	TurnVector (frustum[3].normal, vpn, vup, fovy / 2 - 90);    // top plane

	for (i = 0; i < 4; i++)
	{
		frustum[i].type = PLANE_ANYZ;
		frustum[i].dist = DotProduct (r_origin, frustum[i].normal); // FIXME: shouldn't this always be zero?
		frustum[i].signbits = SignbitsForPlane (&frustum[i]);
	}
}

#define NEARCLIP 4
float GL_GetCameraNear (float radfovx, float radfovy)
{
	const float w = 1.0f / tanf (radfovx * 0.5f);
	const float h = 1.0f / tanf (radfovy * 0.5f);

    // reduce near clip distance at high FOV's to avoid seeing through walls
    const float d = 12.f * q_min (w, h);

    // The weapon is drawn smaller and closer by rt_viewm_scale, and the near clip
    // distance is where "closer" ends: the weapon keeps its picture on screen only if
    // the clip comes along with it, by the same factor.
    const float viewmscale = CVAR_TO_FLOAT (rt_viewm_scale) > 0 ? CVAR_TO_FLOAT (rt_viewm_scale) : 1.0f;

    return CLAMP (0.5f * viewmscale, d * viewmscale, NEARCLIP * viewmscale);
}

float GL_GetCameraFar (void)
{
	return gl_farclip.value;
}

#define INVERSE_ZDEPTH 0

/*
=============
GL_FrustumMatrix
=============
*/
static void GL_FrustumMatrix (float matrix[16], float radfovx, float radfovy)
{
	const float w = 1.0f / tanf (radfovx * 0.5f);
	const float h = 1.0f / tanf (radfovy * 0.5f);

	const float n = GL_GetCameraNear (radfovx, radfovy);
	const float f = GL_GetCameraFar ();

	memset (matrix, 0, 16 * sizeof (float));

	// First column
	matrix[0 * 4 + 0] = w;

	// Second column
	matrix[1 * 4 + 1] = -h;

#if INVERSE_ZDEPTH
	// Third column
	matrix[2 * 4 + 2] = n / (f - n);
	matrix[2 * 4 + 3] = -1.0f;

	// Fourth column
	matrix[3 * 4 + 2] = (f * n) / (f - n);
#else
	// Third column
	matrix[2 * 4 + 2] = f / (n - f);
	matrix[2 * 4 + 3] = -1.0f;

	// Fourth column
	matrix[3 * 4 + 2] = (f * n) / (n - f);
#endif
}

/*
=============
R_SetupMatrices
=============
*/
static void R_SetupMatrices ()
{
	// Projection matrix
	GL_FrustumMatrix (vulkan_globals.projection_matrix, DEG2RAD (r_fovx), DEG2RAD (r_fovy));

	// View matrix
	float rotation_matrix[16];
	RotationMatrix (vulkan_globals.view_matrix, -M_PI / 2.0f, 1.0f, 0.0f, 0.0f);
	RotationMatrix (rotation_matrix, M_PI / 2.0f, 0.0f, 0.0f, 1.0f);
	MatrixMultiply (vulkan_globals.view_matrix, rotation_matrix);
	RotationMatrix (rotation_matrix, DEG2RAD (-r_refdef.viewangles[2]), 1.0f, 0.0f, 0.0f);
	MatrixMultiply (vulkan_globals.view_matrix, rotation_matrix);
	RotationMatrix (rotation_matrix, DEG2RAD (-r_refdef.viewangles[0]), 0.0f, 1.0f, 0.0f);
	MatrixMultiply (vulkan_globals.view_matrix, rotation_matrix);
	RotationMatrix (rotation_matrix, DEG2RAD (-r_refdef.viewangles[1]), 0.0f, 0.0f, 1.0f);
	MatrixMultiply (vulkan_globals.view_matrix, rotation_matrix);

	float translation_matrix[16];
	TranslationMatrix (translation_matrix, -r_refdef.vieworg[0], -r_refdef.vieworg[1], -r_refdef.vieworg[2]);
	MatrixMultiply (vulkan_globals.view_matrix, translation_matrix);

	// View projection matrix
	memcpy (vulkan_globals.view_projection_matrix, vulkan_globals.projection_matrix, 16 * sizeof (float));
	MatrixMultiply (vulkan_globals.view_projection_matrix, vulkan_globals.view_matrix);
}

/*
=============
R_SetupContext
=============
*/
static void R_SetupContext (cb_context_t *cbx)
{
	GL_Viewport (
		cbx, glx + r_refdef.vrect.x, gly + glheight - r_refdef.vrect.y - r_refdef.vrect.height, r_refdef.vrect.width, r_refdef.vrect.height, 0.0f, 1.0f);
}

static void RT_UploadSunLight (void)
{
	if (CVAR_TO_BOOL (rt_materials_only) || !CVAR_TO_BOOL (rt_physical_sun))
	{
		return;
	}

	if (CVAR_TO_FLOAT (rt_sky_sun) > 0.001f)
	{
		vec3_t angles = {CVAR_TO_FLOAT (rt_sky_sun_pitch), CVAR_TO_FLOAT (rt_sky_sun_yaw), 0};

		vec3_t forward, right, up;
		AngleVectors (angles, forward, right, up);

		vec3_t color;
		RT_GetSunColor (color);
		VectorScale (color, CVAR_TO_FLOAT (rt_sky_sun), color);
		// The sun is a light source like every other one, so it needs the same
		// radiometric fixup the world and dlight sources get. Without it its 0..1
		// colour reached the shading orders of magnitude below them, which is
		// why rt_sky_sun only did anything from ~10^4 up.
		RT_FIXUP_LIGHT_INTENSITY (color, true);
		// A sun emits from no area and covers the whole sky, so it takes that
		// fixup at a fraction of its strength (RT_SUN_LIGHT_INTENSITY_SCALE) --
		// otherwise rt_sky_sun 1 overdrives the scene. The god rays read this colour,
		// so they follow the sun too, including the fraction.
		VectorScale (color, RT_SUN_LIGHT_INTENSITY_SCALE, color);

		QrDirectionalLightUploadInfo info = {
			.uniqueID = (uint64_t)UINT32_MAX + 1,
			.color = {color[0], color[1], color[2]},
			.direction = {forward[0], forward[1], forward[2]},
			.angularDiameterDegrees = 0.05f,
		};

		QrResult r = qrUploadDirectionalLight (vulkan_globals.instance, &info);
		QR_CHECK (r);
	}
}

static void RT_UploadAllDlights ()
{
	if (CVAR_TO_BOOL (rt_materials_only))
	{
		return;
	}

	if (RT_AllowFakeLights ())
	{
	for (int i = 0; i < MAX_DLIGHTS; i++)
	{
		const dlight_t *l = &cl_dlights[i];

		if (l->die < cl.time || !l->radius)
		{
			continue;
		}

		if (l->key > 0 && l->key < cl.num_entities)
		{
			entity_t *src = &cl.entities[l->key];
			if (src->model && (src->model->flags & MF_RT_LUMA))
			{
				continue;
			}
		}

		/* A legacy dlight belongs to the entity that asked for it (a lava ball, a
		   monster): its model names the emitter a lights.yaml entry may override. */
		const char *light_name = NULL;

		if (l->key > 0 && l->key < cl.num_entities && cl.entities[l->key].model)
			light_name = cl.entities[l->key].model->name;

		rt_emitter_light_t light;

		memset (&light, 0, sizeof (light));
		light.name = light_name ? light_name : "";
		light.uniqueID = (uint64_t)i;
		light.kind = RT_LIGHT_KIND_DLIGHT;
		VectorCopy (l->origin, light.position);
		VectorCopy (l->color, light.color);
		light.intensity = CVAR_TO_FLOAT (rt_dlight_intensity);
		light.radius = CVAR_TO_FLOAT (rt_dlight_radius);
		light.style = -1;

		/* A spot is one whose editor property gave it a beam; anything else keeps the
		   spherical path, a spot whose beam is still empty included. */
		if (l->type == DLIGHT_TYPE_SPOT && l->angleOuter > 0.0f && DotProduct (l->dir, l->dir) > 0.0f)
		{
			light.spot = true;
			VectorCopy (l->dir, light.direction);
			light.angleInner = l->angleInner;
			light.angleOuter = l->angleOuter;
		}

		RT_LIGHT_Emit (&light);
	}
	}

	// The lights the light editor authored (its Custom tab). Their file is read
	// once at world load and they do not move on their own, so the renderer keeps
	// their slots: its light lists change only when the editor moves or edits one.
	{
		int                custom_count = 0;
		rt_custom_light_t *custom = RT_CustomLights (&custom_count);

		for (int i = 0; i < custom_count; i++)
		{
			const rt_custom_light_t *l = &custom[i];
			float                    intensity = (l->intensity > 0.0f) ? l->intensity : 1.0f;
			vec3_t                   position, color;
			uint64_t                 uid = (uint64_t)UINT32_MAX + 1 + (uint64_t)i;

			VectorCopy (l->origin, position);
			if (l->has_offset)
			{
				position[0] += l->offset[0];
				position[1] += l->offset[1];
				position[2] += l->offset[2];
			}

			VectorCopy (l->color, color);
			if (l->style > 0 && l->style < RT_CUSTOM_STYLE_COUNT)
				intensity *= CLAMP (0.0f, (float)d_lightstylevalue[l->style] / 256.0f, 1.0f);
			VectorScale (color, intensity, color);
			RT_FIXUP_LIGHT_INTENSITY (color, true);

			QrSphericalLightUploadInfo info = {
				.uniqueID = uid,
				.color = {color[0], color[1], color[2]},
				.position = {position[0], position[1], position[2]},
				.radius = METRIC_TO_QUAKEUNIT (l->radius),
			};

			if (l->spot && (l->dir[0] != 0.0f || l->dir[1] != 0.0f || l->dir[2] != 0.0f))
			{
				vec3_t direction;
				float  angleInner, angleOuter;

				VectorCopy (l->dir, direction);
				VectorNormalize (direction);
				angleInner = DEG2RAD (q_min (l->angle_inner, l->angle_outer));
				angleOuter = DEG2RAD (q_max (l->angle_inner, l->angle_outer));

				QrSpotLightUploadInfo spot = {
					.uniqueID = info.uniqueID,
					.color = {info.color.data[0], info.color.data[1], info.color.data[2]},
					.position = {info.position.data[0], info.position.data[1], info.position.data[2]},
					.direction = {direction[0], direction[1], direction[2]},
					.radius = info.radius,
					.angleOuter = angleOuter,
					.angleInner = angleInner,
				};

				QrResult r = qrUploadSpotLight (vulkan_globals.instance, &spot);
				QR_CHECK (r);
			}
			else
			{
				QrResult r = qrUploadSphericalLight (vulkan_globals.instance, &info);
				QR_CHECK (r);
			}

			RT_TRACK_Light (info.position.data, info.radius, info.color.data,
			                uid, RT_LIGHT_KIND_CUSTOM, "");

			if (CVAR_TO_FLOAT (rt_cluster_dlights) != 0)
			{
				const float power = (info.color.data[0] * 0.2125f + info.color.data[1] * 0.7154f +
				                     info.color.data[2] * 0.0721f) *
				                    info.radius * info.radius;

				RT_ClusterLightAddPower (uid, position, RT_ClusterLightReach (), power);
			}
		}
	}

	if (CVAR_TO_FLOAT (rt_flashlight) > 0.1f)
	{
		vec3_t pos;
		VectorCopy (r_origin, pos);
		VectorMA (pos, METRIC_TO_QUAKEUNIT (-0.3f), vup, pos);
		VectorMA (pos, METRIC_TO_QUAKEUNIT (-0.4f), vright, pos);

		vec3_t color;
		RT_INIT_DEFAULT_LIGHT_COLOR (color);
		VectorScale (color, CVAR_TO_FLOAT (rt_flashlight), color);
		RT_FIXUP_LIGHT_INTENSITY (color, true);

		QrSpotLightUploadInfo info = {
			.uniqueID = (uint64_t)UINT32_MAX + 0,
			.color = {color[0], color[1], color[2]},
			.position = {pos[0], pos[1], pos[2]},
			.direction = {vpn[0], vpn[1], vpn[2]},
			.radius = METRIC_TO_QUAKEUNIT (0.1f),
			.angleOuter = DEG2RAD (30),
			.angleInner = 0,
		};

		QrResult r = qrUploadSpotLight (vulkan_globals.instance, &info);
		QR_CHECK (r);
	}

	if (QR_Editor_TorchOn ())
	{
		vec3_t position, color;

		QR_Editor_TorchOrigin (position);

		color[0] = 1.0f;
		color[1] = 0.84f;
		color[2] = 0.62f;
		VectorScale (color, CVAR_TO_FLOAT (rt_dlight_intensity), color);
		RT_FIXUP_LIGHT_INTENSITY (color, true);

		QrSphericalLightUploadInfo info = {
			.uniqueID = (uint64_t)UINT32_MAX + 1 + RT_CUSTOM_LIGHTS_MAX,
			.color = {color[0], color[1], color[2]},
			.position = {position[0], position[1], position[2]},
			.radius = METRIC_TO_QUAKEUNIT (CVAR_TO_FLOAT (rt_dlight_radius)),
		};

		QrResult r = qrUploadSphericalLight (vulkan_globals.instance, &info);
		QR_CHECK (r);

		if (CVAR_TO_FLOAT (rt_cluster_dlights) != 0)
		{
			const float power = (info.color.data[0] * 0.2125f + info.color.data[1] * 0.7154f +
			                     info.color.data[2] * 0.0721f) *
			                    info.radius * info.radius;

			RT_ClusterLightAddPower (info.uniqueID, position, RT_ClusterLightReach (), power);
		}
	}

	RT_UploadSunLight ();
}

/*
================
RT_DlightSpot_f

Places a spot dlight at the crosshair, pointing along the view. The editor's spot property
is what will create these lights; until it is there, this is how one is made and seen.

dlightspot <outer_deg> [inner_deg] [distance] [strength]
================
*/
void RT_DlightSpot_f (void)
{
	if (Cmd_Argc () < 2)
	{
		Con_Printf ("usage: %s <outer_deg> [inner_deg] [distance] [strength]\n", Cmd_Argv (0));
		return;
	}

	float       outerDeg = (float) atof (Cmd_Argv (1));
	float       innerDeg = (Cmd_Argc () >= 3) ? (float) atof (Cmd_Argv (2)) : 0.0f;
	const float dist     = (Cmd_Argc () >= 4) ? (float) atof (Cmd_Argv (3)) : 48.0f;
	const float strength = (Cmd_Argc () >= 5) ? (float) atof (Cmd_Argv (4)) : CVAR_TO_FLOAT (rt_dlightspot_intensity);

	/* The comparisons read as they do so that a nan fails them: atof takes nan and inf, and a
	   nan edge or origin would poison every cell that samples the light. */
	if (!(outerDeg >= 0.1f && outerDeg <= 89.9f) ||
	    !(innerDeg >= 0.0f) || !(innerDeg <= outerDeg) ||
	    !(dist >= 1.0f && dist <= 4096.0f) ||
	    !(strength >= 0.0f && strength <= 1000.0f))
	{
		Con_Printf ("usage: %s <outer_deg> [inner_deg] [distance] [strength]\n", Cmd_Argv (0));
		return;
	}

	/* The cone edge is a smoothstep, and one with equal edges is undefined, so the inner
	   angle stays strictly inside the outer one. */
	if (innerDeg > outerDeg * 0.999f)
	{
		innerDeg = outerDeg * 0.999f;
	}

	/* DLIGHT_KEY_TEST is a key no emitter makes, so no effect can take the slot back, and
	   every run of the command reuses it: one command owns one light. */
	dlightspot_t *dl = CL_AllocDlightSpot (DLIGHT_KEY_TEST);

	VectorMA (r_origin, dist, vpn, dl->origin);
	VectorCopy (vpn, dl->dir);
	VectorScale (dl->color, strength, dl->color);
	dl->angleOuter = DEG2RAD (outerDeg);
	dl->angleInner = DEG2RAD (innerDeg);
	dl->radius     = 200;
	dl->decay      = 0;
	dl->die        = cl.time + 3600;

	Con_Printf ("dlightspot: outer %.1f deg, inner %.1f deg, %.0f units ahead, strength %.2f\n", outerDeg, innerDeg, dist, strength);
}

/*
===============
R_SetupViewBeforeMark
===============
*/
void R_SetupViewBeforeMark (void *unused)
{
	double prof_start = RT_Prof_Begin ();

	R_AnimateLight ();

	// build the transformation matrix for the given view angles
	VectorCopy (r_refdef.vieworg, r_origin);
	r_vieworg_valid = true;
	AngleVectors (r_refdef.viewangles, vpn, vright, vup);

	// The sun editor follows the crosshair, which is this very vector.
	RT_UpdateSunEditor ();

	// current viewleaf
	r_oldviewleaf = r_viewleaf;
	r_viewleaf = Mod_PointInLeaf (r_origin, cl.worldmodel);

	V_SetContentsColor (r_viewleaf->contents);
	V_CalcBlend ();

	// johnfitz -- calculate r_fovx and r_fovy here
	r_fovx = r_refdef.fov_x;
	r_fovy = r_refdef.fov_y;

	{
		int		 contents = Mod_PointInLeaf (r_origin, cl.worldmodel)->contents;
		qboolean forced = M_ForcedUnderwater ();
		double	 warp_time = forced ? realtime : cl.time;

		rt_lavaeffects = false;

		if (contents == CONTENTS_WATER || forced)
		{
			rt_cameramedia = QR_MEDIA_TYPE_WATER;
		}
		else if (contents == CONTENTS_LAVA)
		{
			rt_cameramedia = QR_MEDIA_TYPE_WATER;
			rt_lavaeffects = true;
		}
		else if (contents == CONTENTS_SLIME)
		{
			rt_cameramedia = QR_MEDIA_TYPE_ACID;
		}
		else
		{
			rt_cameramedia = QR_MEDIA_TYPE_VACUUM;
		}

		if (QR_Editor_Active ())
		{
			rt_cameramedia = QR_MEDIA_TYPE_VACUUM;
			rt_lavaeffects = false;
		}

		const float liquid_target = (!forced && !QR_Editor_Active () &&
									 (contents == CONTENTS_WATER || contents == CONTENTS_LAVA || contents == CONTENTS_SLIME)) ?
										1.0f :
										0.0f;
		const float liquid_rate = (liquid_target > rt_ef_liquid_pulse) ? 0.15f : 0.25f;
		rt_ef_liquid_pulse += (liquid_target - rt_ef_liquid_pulse) * CLAMP (0.0f, host_frametime / liquid_rate, 1.0f);

		if (rt_cameramedia != QR_MEDIA_TYPE_VACUUM && CVAR_TO_INT32 (r_waterwarp) == 2)
		{
			// variance is a percentage of width, where width = 2 * tan(fov / 2) otherwise the effect is too dramatic at high FOV and too subtle at low FOV.
			// what a mess!
			r_fovx = atan (tan (DEG2RAD (r_refdef.fov_x) / 2) * (0.97 + sin (warp_time * 1.5) * 0.03)) * 2 / M_PI_DIV_180;
			r_fovy = atan (tan (DEG2RAD (r_refdef.fov_y) / 2) * (1.03 - sin (warp_time * 1.5) * 0.03)) * 2 / M_PI_DIV_180;
		}
	}
	// johnfitz

	R_SetFrustum (r_fovx, r_fovy); // johnfitz -- use r_fov* vars
	R_SetupMatrices ();

	// johnfitz -- cheat-protect some draw modes
	r_fullbright_cheatsafe = false;
	r_lightmap_cheatsafe = false;
	r_drawworld_cheatsafe = true;
	if (cl.maxclients == 1)
	{
		if (!r_drawworld.value)
			r_drawworld_cheatsafe = false;
		if (r_lightmap.value)
			r_lightmap_cheatsafe = true;
		else if (r_fullbright.value)
			r_fullbright_cheatsafe = true;
	}
	if (!cl.worldmodel->lightdata)
	{
		r_fullbright_cheatsafe = true;
		r_lightmap_cheatsafe = false;
	}
	// johnfitz

	RT_ClusterLightListsReset ();

	RT_UploadAllDlights ();

	qrSetParticleProxyGate (vulkan_globals.instance, (uint32_t)rt_particle_proxy_gate.value,
	                        rt_glass_particles.value != 0.0f ? 1u : 0u);

	RT_Prof_End (RT_PROF_SETUP, prof_start);
}

//==============================================================================
//
// RENDER VIEW
//
//==============================================================================

/*
=============
R_DrawEntitiesOnList
=============
*/
void R_DrawEntitiesOnList (cb_context_t *cbx, qboolean alphapass, int chain, int startedict, int endedict) // johnfitz -- added parameter
{
	int i;

	if (!r_drawentities.value)
		return;

	R_BeginDebugUtilsLabel (cbx, alphapass ? "Entities Alpha Pass" : "Entities");
	// johnfitz -- sprites are not a special case
	for (i = startedict; i < endedict; ++i)
	{
		entity_t *currententity = cl_visedicts[i];

		// johnfitz -- if alphapass is true, draw only alpha entites this time
		// if alphapass is false, draw only nonalpha entities this time
		if ((ENTALPHA_DECODE (currententity->alpha) < 1 && !alphapass) || (ENTALPHA_DECODE (currententity->alpha) == 1 && alphapass))
			continue;

		// johnfitz -- chasecam
		if (currententity == &cl.entities[cl.viewentity])
			currententity->angles[0] *= 0.3;
		// johnfitz

		// spike -- this would be more efficient elsewhere, but its more correct here.
		if (currententity->eflags & EFLAGS_EXTERIORMODEL)
			continue;

		if (!currententity->model)
			continue;

		const int entuniqueid = RT_GetEntityUniqueId (currententity);
		const double prof_entity = RT_Prof_Begin ();

		switch (currententity->model->type)
		{
		case mod_alias:
			R_DrawAliasModel (cbx, currententity, entuniqueid);
			RT_Prof_End (RT_PROF_ENTS_ALIAS, prof_entity);
			break;
		case mod_brush:
			R_DrawBrushModel (cbx, currententity, chain, entuniqueid);
			RT_Prof_End (RT_PROF_ENTS_BRUSH, prof_entity);
			break;
		case mod_sprite:
			R_DrawSpriteModel (cbx, currententity, entuniqueid);
			RT_Prof_End (RT_PROF_ENTS_SPRITE, prof_entity);
			break;
		}
	}
	R_EndDebugUtilsLabel (cbx);
}

/*
=============
R_DrawViewModel -- johnfitz -- gutted
=============
*/
void R_DrawViewModel (cb_context_t *cbx)
{
	const qboolean editor_preview = QR_Editor_ShowViewModel ();

	if (!r_drawentities.value || chase_active.value || PhotoCam_Active () || Observer_Active () ||
	    (QR_Editor_Active () && !editor_preview) || (!editor_preview && !r_drawviewmodel.value))
		return;
	
	if (cl.stats[STAT_HEALTH] <= 0 && !editor_preview)
		return;

	entity_t *currententity = &cl.viewent;
	if (!currententity->model)
		return;

	// johnfitz -- this fixes a crash
	if (currententity->model->type != mod_alias)
		return;
	// johnfitz

	R_BeginDebugUtilsLabel (cbx, "View Model");

	// hack the depth range to prevent view model from poking into walls
	GL_Viewport (
		cbx, glx + r_refdef.vrect.x, gly + glheight - r_refdef.vrect.y - r_refdef.vrect.height, r_refdef.vrect.width, r_refdef.vrect.height, 0.7f, 1.0f);

	R_DrawAliasModel (cbx, currententity, ENT_UNIQUEID_VIEWMODEL);

	GL_Viewport (
		cbx, glx + r_refdef.vrect.x, gly + glheight - r_refdef.vrect.y - r_refdef.vrect.height, r_refdef.vrect.width, r_refdef.vrect.height, 0.0f, 1.0f);

	R_EndDebugUtilsLabel (cbx);
}

/*
================
R_EmitWirePoint -- johnfitz -- draws a wireframe cross shape for point entities
================
*/
void R_EmitWirePoint (cb_context_t *cbx, vec3_t origin)
{
	const int size = 8;

	QrVertex vertices[6] = {0};

	vertices[0].position[0] = origin[0] - size;
	vertices[0].position[1] = origin[1];
	vertices[0].position[2] = origin[2];
	vertices[1].position[0] = origin[0] + size;
	vertices[1].position[1] = origin[1];
	vertices[1].position[2] = origin[2];
	vertices[2].position[0] = origin[0];
	vertices[2].position[1] = origin[1] - size;
	vertices[2].position[2] = origin[2];
	vertices[3].position[0] = origin[0];
	vertices[3].position[1] = origin[1] + size;
	vertices[3].position[2] = origin[2];
	vertices[4].position[0] = origin[0];
	vertices[4].position[1] = origin[1];
	vertices[4].position[2] = origin[2] - size;
	vertices[5].position[0] = origin[0];
	vertices[5].position[1] = origin[1];
	vertices[5].position[2] = origin[2] + size;

	for (int i = 0; i < (int)countof (vertices); i++)
	{
		vertices[i].packedColor = RT_PACKED_COLOR_WHITE;
	}

	QrRasterizedGeometryUploadInfo info = {
		.renderType = QR_RASTERIZED_GEOMETRY_RENDER_TYPE_DEFAULT,
		.vertexCount = countof (vertices),
		.pVertices = vertices,
		.indexCount = 0,
		.pIndices = NULL,
		.transform = RT_TRANSFORM_IDENTITY,
		.color = RT_COLOR_WHITE,
		.material = QR_NO_MATERIAL,
		.pipelineState = QR_RASTERIZED_GEOMETRY_STATE_FORCE_LINE_LIST,
		.blendFuncSrc = 0,
		.blendFuncDst = 0,
	};

	QrResult r = qrUploadRasterizedGeometry (vulkan_globals.instance, &info, NULL, NULL);
	QR_CHECK (r);
}

/*
================
R_EmitWireBox -- johnfitz -- draws one axis aligned bounding box
================
*/
void R_EmitWireBox (cb_context_t *cbx, vec3_t mins, vec3_t maxs)
{
	const static uint32_t box_indices[24] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7, 0, 2, 1, 3, 4, 6, 5, 7};

	QrVertex vertices[8] = {0};

	for (int i = 0; i < 8; ++i)
	{
		vertices[i].position[0] = ((i % 2) < 1) ? mins[0] : maxs[0];
		vertices[i].position[1] = ((i % 4) < 2) ? mins[1] : maxs[1];
		vertices[i].position[2] = ((i % 8) < 4) ? mins[2] : maxs[2];
		vertices[i].packedColor = RT_PACKED_COLOR_WHITE;
	}

	QrRasterizedGeometryUploadInfo info = {
		.renderType = QR_RASTERIZED_GEOMETRY_RENDER_TYPE_DEFAULT,
		.vertexCount = countof (vertices),
		.pVertices = vertices,
		.indexCount = countof (box_indices),
		.pIndices = box_indices,
		.transform = RT_TRANSFORM_IDENTITY,
		.color = RT_COLOR_WHITE,
		.material = QR_NO_MATERIAL,
		.pipelineState = QR_RASTERIZED_GEOMETRY_STATE_FORCE_LINE_LIST,
		.blendFuncSrc = 0,
		.blendFuncDst = 0,
	};

	QrResult r = qrUploadRasterizedGeometry (vulkan_globals.instance, &info, NULL, NULL);
	QR_CHECK (r);
}

/*
================
R_ShowBoundingBoxes -- johnfitz

draw bounding boxes -- the server-side boxes, not the renderer cullboxes
================
*/
void R_ShowBoundingBoxes (cb_context_t *cbx)
{
	extern edict_t *sv_player;
	vec3_t          mins, maxs;
	edict_t        *ed;
	int             i;

	if (!r_showbboxes.value || cl.maxclients > 1 || !r_drawentities.value || !sv.active)
		return;

	R_BeginDebugUtilsLabel (cbx, "show bboxes");

	PR_SwitchQCVM (&sv.qcvm);
	for (i = 0, ed = NEXT_EDICT (qcvm->edicts); i < qcvm->num_edicts; i++, ed = NEXT_EDICT (ed))
	{
		if (ed == sv_player)
			continue; // don't draw player's own bbox

		if (ed->v.mins[0] == ed->v.maxs[0] && ed->v.mins[1] == ed->v.maxs[1] && ed->v.mins[2] == ed->v.maxs[2])
		{
			// point entity
			R_EmitWirePoint (cbx, ed->v.origin);
		}
		else
		{
			// box entity
			VectorAdd (ed->v.mins, ed->v.origin, mins);
			VectorAdd (ed->v.maxs, ed->v.origin, maxs);
			R_EmitWireBox (cbx, mins, maxs);
		}
	}
	PR_SwitchQCVM (NULL);

	R_EndDebugUtilsLabel (cbx);
}


/*
================
R_ShowTris -- johnfitz
================
*/
void R_ShowTris (cb_context_t *cbx)
{
	extern cvar_t r_particles;
	int           i;

	if (r_showtris.value < 1 || r_showtris.value > 2 || cl.maxclients > 1 || !vulkan_globals.non_solid_fill)
		return;

	R_BeginDebugUtilsLabel (cbx, "show tris");
	if (r_drawworld.value)
		R_DrawWorld_ShowTris (cbx);

	if (r_drawentities.value)
	{
		for (i = 0; i < cl_numvisedicts; i++)
		{
			entity_t *currententity = cl_visedicts[i];

			if (currententity == &cl.entities[cl.viewentity]) // chasecam
				currententity->angles[0] *= 0.3;

			switch (currententity->model->type)
			{
			case mod_brush:
				R_DrawBrushModel_ShowTris (cbx, currententity);
				break;
			case mod_alias:
				R_DrawAliasModel_ShowTris (cbx, currententity);
				break;
			case mod_sprite:
				R_DrawSpriteModel_ShowTris (cbx, currententity);
				break;
			default:
				break;
			}
		}

		// viewmodel
		entity_t *currententity = &cl.viewent;
		if (r_drawviewmodel.value && !chase_active.value && cl.stats[STAT_HEALTH] > 0 && !(cl.items & IT_INVISIBILITY) && currententity->model &&
		    currententity->model->type == mod_alias)
		{
			R_DrawAliasModel_ShowTris (cbx, currententity);
		}
	}

	if (r_particles.value)
	{
		R_DrawParticles_ShowTris (cbx);
#ifdef PSET_SCRIPT
		PScript_DrawParticles_ShowTris (cbx);
#endif
	}

	R_EndDebugUtilsLabel (cbx);
}

/*
================
R_DrawWorldTask
================
*/
void R_DrawWorldTask (void *unused)
{
	double prof_start = RT_Prof_Begin ();

	qrBeginDeferredLightUploads (vulkan_globals.instance, 0);

	const qboolean static_submit = Atomic_ExchangeUInt32 (&rt_require_static_submit, false) != 0;
	const qboolean light_recollect = Atomic_ExchangeUInt32 (&rt_require_world_light_recollect, false) != 0;

	if (!static_submit && !light_recollect)
	{
		RT_StaticMovableUpdate ();
		qrEndDeferredLightUploads (vulkan_globals.instance);
		RT_Prof_End (RT_PROF_WORLD, prof_start);
		return;
	}

	if (!static_submit)
	{
		RT_RecollectWorldEmissiveLights ();
		RT_StaticMovableUpdate ();

		qrEndDeferredLightUploads (vulkan_globals.instance);
		RT_Prof_End (RT_PROF_WORLD, prof_start);
		return;
	}

	QrResult r;
	
	r = qrBeginStaticGeometries (vulkan_globals.instance);
	QR_CHECK (r);

	RT_UploadSunLight ();

	cb_context_t *cbx = &vulkan_globals.secondary_cb_contexts[CBX_WORLD_0];
	R_SetupContext (cbx);
	Fog_EnableGFog (cbx);
	R_DrawWorld (cbx);

	RT_StaticMovableUpload (cbx);

	r = qrSubmitStaticGeometries (vulkan_globals.instance);
	QR_CHECK (r);

	Atomic_StoreUInt32 (&rt_require_static_submit, false);
	Atomic_StoreUInt32 (&rt_require_world_light_recollect, false);

	RT_StaticMovableUpdate ();

	qrEndDeferredLightUploads (vulkan_globals.instance);
	RT_Prof_End (RT_PROF_WORLD, prof_start);
}

/*
================
R_DrawSkyAndWaterTask
================
*/
static void R_DrawSkyAndWaterTask (void *unused)
{
	double prof_start = RT_Prof_Begin ();

	R_SetupContext (&vulkan_globals.secondary_cb_contexts[CBX_SKY_AND_WATER]);
	Fog_EnableGFog (&vulkan_globals.secondary_cb_contexts[CBX_SKY_AND_WATER]);
	Sky_DrawSky (&vulkan_globals.secondary_cb_contexts[CBX_SKY_AND_WATER]);
	R_DrawWorld_Water (&vulkan_globals.secondary_cb_contexts[CBX_SKY_AND_WATER]);
	R_DrawWorld_Animated (&vulkan_globals.secondary_cb_contexts[CBX_SKY_AND_WATER]);

	RT_Prof_End (RT_PROF_SKY, prof_start);
}

/*
================
R_DrawEntitiesTask
================
*/
static void R_DrawEntitiesTask (int index, void *unused)
{
	double prof_start = RT_Prof_Begin ();

	qrBeginDeferredLightUploads (vulkan_globals.instance, 1u + (uint32_t)index);

	const int cbx_index = index + CBX_ENTITIES_0;
	R_SetupContext (&vulkan_globals.secondary_cb_contexts[cbx_index]);
	Fog_EnableGFog (&vulkan_globals.secondary_cb_contexts[cbx_index]); // johnfitz
	const int num_edicts_per_cb = (cl_numvisedicts + NUM_ENTITIES_CBX - 1) / NUM_ENTITIES_CBX;
	int       startedict = index * num_edicts_per_cb;
	int       endedict = q_min ((index + 1) * num_edicts_per_cb, cl_numvisedicts);
	R_DrawEntitiesOnList (&vulkan_globals.secondary_cb_contexts[cbx_index], false, index + chain_model_0, startedict, endedict);

	qrEndDeferredLightUploads (vulkan_globals.instance);

	RT_Prof_End (RT_PROF_ENTS, prof_start);
}

/*
================
R_DrawAlphaEntitiesTask
================
*/
static void R_DrawAlphaEntitiesTask (void *unused)
{
	double prof_start = RT_Prof_Begin ();

	R_SetupContext (&vulkan_globals.secondary_cb_contexts[CBX_ALPHA_ENTITIES]);
	Fog_EnableGFog (&vulkan_globals.secondary_cb_contexts[CBX_ALPHA_ENTITIES]);
	qrBeginDeferredLightUploads (vulkan_globals.instance, 1u + NUM_ENTITIES_CBX);
	R_DrawEntitiesOnList (
		&vulkan_globals.secondary_cb_contexts[CBX_ALPHA_ENTITIES], true, chain_alpha_model, 0,
		cl_numvisedicts); // johnfitz -- true means this is the pass for alpha entities
	qrEndDeferredLightUploads (vulkan_globals.instance);

	RT_Prof_End (RT_PROF_ALPHA, prof_start);
}

/*
================
R_DrawParticlesTask
================
*/
static void R_DrawParticlesTask (void *unused)
{
	double prof_start = RT_Prof_Begin ();

	R_SetupContext (&vulkan_globals.secondary_cb_contexts[CBX_PARTICLES]);
	Fog_EnableGFog (&vulkan_globals.secondary_cb_contexts[CBX_PARTICLES]); // johnfitz
	R_DrawSmoke (&vulkan_globals.secondary_cb_contexts[CBX_PARTICLES]);
	R_DrawParticles (&vulkan_globals.secondary_cb_contexts[CBX_PARTICLES]);
#ifdef PSET_SCRIPT
	PScript_DrawParticles (&vulkan_globals.secondary_cb_contexts[CBX_PARTICLES]);
#endif

	RT_Prof_End (RT_PROF_PARTICLES, prof_start);
}

/*
================
R_DrawViewModelTask
================
*/
static void R_DrawViewModelTask (void *unused)
{
	double prof_task = RT_Prof_Begin ();
	double prof_start;

	R_SetupContext (&vulkan_globals.secondary_cb_contexts[CBX_VIEW_MODEL]);

	qrFlushDeferredLightUploads (vulkan_globals.instance);

	prof_start = RT_Prof_Begin ();
	// The editor draws the lights of the frame as wireframes, so their upload
	// (the map light entities) has to happen before the selection draw.
	RT_UploadAllElights (); // RT
	RT_Prof_End (RT_PROF_ELIGHTS, prof_start);

	// Only the draw itself, so that the model upload cost can be told apart from
	// the light uploads that share this task.
	prof_start = RT_Prof_Begin ();
	R_DrawViewModel (&vulkan_globals.secondary_cb_contexts[CBX_VIEW_MODEL]);     // johnfitz -- moved here from R_RenderView
	R_ShowTris (&vulkan_globals.secondary_cb_contexts[CBX_VIEW_MODEL]);          // johnfitz
	R_ShowBoundingBoxes (&vulkan_globals.secondary_cb_contexts[CBX_VIEW_MODEL]); // johnfitz
	RT_Prof_End (RT_PROF_VIEWMODEL_DRAW, prof_start);

	prof_start = RT_Prof_Begin ();
	RT_UploadAllWorldModelLights (); // RT
	RT_Prof_End (RT_PROF_WMODEL_LIGHTS, prof_start);

	prof_start = RT_Prof_Begin ();
	RT_UploadAllTeleports (); // RT
	RT_Prof_End (RT_PROF_TELEPORTS, prof_start);

	RT_Prof_End (RT_PROF_VIEWMODEL, prof_task);
}

static void R_ClusterLightListsPrepareTask (void *unused)
{
	RT_ClusterLightListsPrepare (RT_CLUSTER_SLICES);
}

static void R_ClusterTopUpSliceTask (int index, void *unused)
{
	RT_ClusterLightListsSlice ((uint32_t)index, RT_CLUSTER_SLICES);
}

static void R_ClusterLightListsFinishTask (void *unused)
{
	RT_ClusterLightListsFinish ();
}

static void R_ClusterPublishSliceTask (int index, void *unused)
{
	RT_ClusterLightListsPublishSlice (index, RT_CLUSTER_SLICES);
}

static void R_ClusterLightListsCommitTask (void *unused)
{
	RT_ClusterLightListsCommit ();
}

/*
================
R_RenderView
================
*/
void R_RenderView (qboolean use_tasks, task_handle_t begin_rendering_task, task_handle_t setup_frame_task, task_handle_t draw_done_task)
{
	double time1, time2;

	if (!cl.worldmodel)
		Sys_Error ("R_RenderView: NULL worldmodel");

	rt_editor_draw_done_task = INVALID_TASK_HANDLE;

	RT_ColorsRefresh ();

	// The light editor's list of the frame's lights starts empty every frame; the
	// four upload sites fill it as they go.
	RT_TRACK_BeginFrame ();

	if (Atomic_LoadUInt32 (&rt_require_static_submit) != 0)
		RT_StaticMovablePrepare ();

	time1 = Sys_DoubleTime ();

	// johnfitz -- rendering statistics
	Atomic_StoreUInt32 (&rs_brushpolys, 0u);
	Atomic_StoreUInt32 (&rs_aliaspolys, 0u);
	Atomic_StoreUInt32 (&rs_skypolys, 0u);
	Atomic_StoreUInt32 (&rs_particles, 0u);
	Atomic_StoreUInt32 (&rs_fogpolys, 0u);
	Atomic_StoreUInt32 (&rs_dynamiclightmaps, 0u);
	Atomic_StoreUInt32 (&rs_aliaspasses, 0u);
	Atomic_StoreUInt32 (&rs_skypasses, 0u);
	Atomic_StoreUInt32 (&rs_brushpasses, 0u);

	if (use_tasks)
	{
		task_handle_t before_mark = Task_AllocateAndAssignFunc (R_SetupViewBeforeMark, NULL, 0);
		Task_AddDependency (setup_frame_task, before_mark);
		Task_AddDependency (begin_rendering_task, setup_frame_task);
		Task_AddDependency (begin_rendering_task, before_mark);

		task_handle_t store_efrags = INVALID_TASK_HANDLE;
		task_handle_t cull_surfaces = INVALID_TASK_HANDLE;
		task_handle_t chain_surfaces = INVALID_TASK_HANDLE;
		R_MarkSurfaces (use_tasks, before_mark, &store_efrags, &cull_surfaces, &chain_surfaces);

		task_handle_t draw_world_task = Task_AllocateAndAssignFunc (R_DrawWorldTask, NULL, 0);
		Task_AddDependency (chain_surfaces, draw_world_task);
		Task_AddDependency (begin_rendering_task, draw_world_task);
		Task_AddDependency (draw_world_task, draw_done_task);

		task_handle_t draw_sky_and_water_task = Task_AllocateAndAssignFunc (R_DrawSkyAndWaterTask, NULL, 0);
		Task_AddDependency (store_efrags, draw_sky_and_water_task);
		Task_AddDependency (chain_surfaces, draw_sky_and_water_task);
		Task_AddDependency (begin_rendering_task, draw_sky_and_water_task);
		Task_AddDependency (draw_sky_and_water_task, draw_done_task);

		task_handle_t draw_view_model_task = Task_AllocateAndAssignFunc (R_DrawViewModelTask, NULL, 0);
		Task_AddDependency (before_mark, draw_view_model_task);
		Task_AddDependency (begin_rendering_task, draw_view_model_task);
		Task_AddDependency (draw_view_model_task, draw_done_task);

		task_handle_t cluster_prepare_task = Task_AllocateAndAssignFunc (R_ClusterLightListsPrepareTask, NULL, 0);
		Task_AddDependency (draw_view_model_task, cluster_prepare_task);

		task_handle_t cluster_topup_task =
		    Task_AllocateAndAssignIndexedFunc (R_ClusterTopUpSliceTask, RT_CLUSTER_SLICES, NULL, 0);
		Task_AddDependency (cluster_prepare_task, cluster_topup_task);

		task_handle_t cluster_finish_task = Task_AllocateAndAssignFunc (R_ClusterLightListsFinishTask, NULL, 0);
		Task_AddDependency (cluster_topup_task, cluster_finish_task);

		task_handle_t cluster_publish_task =
		    Task_AllocateAndAssignIndexedFunc (R_ClusterPublishSliceTask, RT_CLUSTER_SLICES, NULL, 0);
		Task_AddDependency (cluster_finish_task, cluster_publish_task);

		task_handle_t cluster_commit_task = Task_AllocateAndAssignFunc (R_ClusterLightListsCommitTask, NULL, 0);
		Task_AddDependency (cluster_publish_task, cluster_commit_task);
		Task_AddDependency (cluster_commit_task, draw_done_task);

		// The editor's GUI reads what the draw tasks uploaded (the tracked lights
		// of the frame), so it has to wait for the task that carries them.
		rt_editor_draw_done_task = draw_view_model_task;

		task_handle_t draw_entities_task = Task_AllocateAndAssignIndexedFunc (R_DrawEntitiesTask, NUM_ENTITIES_CBX, NULL, 0);
		Task_AddDependency (store_efrags, draw_entities_task);
		Task_AddDependency (begin_rendering_task, draw_entities_task);
		Task_AddDependency (draw_entities_task, draw_done_task);

		task_handle_t draw_alpha_entities_task = Task_AllocateAndAssignFunc (R_DrawAlphaEntitiesTask, NULL, 0);
		Task_AddDependency (store_efrags, draw_alpha_entities_task);
		Task_AddDependency (begin_rendering_task, draw_alpha_entities_task);
		Task_AddDependency (draw_alpha_entities_task, draw_done_task);

		task_handle_t draw_particles_task = Task_AllocateAndAssignFunc (R_DrawParticlesTask, NULL, 0);
		Task_AddDependency (before_mark, draw_particles_task);
		Task_AddDependency (begin_rendering_task, draw_particles_task);
		Task_AddDependency (draw_particles_task, draw_done_task);

		task_handle_t update_lightmaps_task = Task_AllocateAndAssignFunc (R_UpdateLightmaps, NULL, 0);
		Task_AddDependency (cull_surfaces, update_lightmaps_task);
		Task_AddDependency (begin_rendering_task, update_lightmaps_task);
		Task_AddDependency (draw_world_task, update_lightmaps_task);
		Task_AddDependency (update_lightmaps_task, draw_done_task);

		Task_AddDependency (draw_world_task, draw_view_model_task);
		Task_AddDependency (draw_sky_and_water_task, draw_view_model_task);
		Task_AddDependency (draw_entities_task, draw_view_model_task);
		Task_AddDependency (draw_alpha_entities_task, draw_view_model_task);
		Task_AddDependency (draw_particles_task, draw_view_model_task);

		Task_AddDependency (draw_world_task, draw_sky_and_water_task);
		Task_AddDependency (draw_world_task, draw_entities_task);
		Task_AddDependency (draw_world_task, draw_alpha_entities_task);
		Task_AddDependency (draw_world_task, draw_particles_task);

		task_handle_t tasks[] = {before_mark,          store_efrags,		                         draw_world_task,     draw_sky_and_water_task,
		                         draw_view_model_task, draw_entities_task, draw_alpha_entities_task, draw_particles_task, update_lightmaps_task,
		                         cluster_prepare_task, cluster_topup_task, cluster_finish_task, cluster_publish_task, cluster_commit_task};
		Tasks_Submit ((sizeof (tasks) / sizeof (task_handle_t)), tasks);
		if (store_efrags != cull_surfaces)
		{
			Task_Submit (cull_surfaces);
			Task_Submit (chain_surfaces);
		}
	}
	else
	{
		R_SetupViewBeforeMark (NULL);
		R_MarkSurfaces (use_tasks, INVALID_TASK_HANDLE, NULL, NULL, NULL); // johnfitz -- create texture chains from PVS
		R_DrawWorldTask (NULL);
		R_DrawSkyAndWaterTask (NULL);
		for (int i = 0; i < NUM_ENTITIES_CBX; ++i)
			R_DrawEntitiesTask (i, NULL);
		R_DrawAlphaEntitiesTask (NULL);
		R_DrawParticlesTask (NULL);
		R_DrawViewModelTask (NULL);
		RT_ClusterLightListsUpload ();
		if (r_gpulightmapupdate.value)
			R_UpdateLightmaps (NULL);
	}

	// johnfitz

	// johnfitz -- modified r_speeds output
	time2 = Sys_DoubleTime ();
	rt_world_draw_ms = (float)((time2 - time1) * 1000.0);
	if (r_pos.value)
		Con_Printf (
			"x %i y %i z %i (pitch %i yaw %i roll %i)\n", (int)cl.entities[cl.viewentity].origin[0], (int)cl.entities[cl.viewentity].origin[1],
			(int)cl.entities[cl.viewentity].origin[2], (int)cl.viewangles[PITCH], (int)cl.viewangles[YAW], (int)cl.viewangles[ROLL]);
	else if (r_speeds.value == 2)
		Con_Printf (
			"%6.3f ms  %4u/%4u wpoly %4u/%4u epoly %3u lmap %4u/%4u sky\n", (time2 - time1) * 1000.0, rs_brushpolys, rs_brushpasses, rs_aliaspolys,
			rs_aliaspasses, rs_dynamiclightmaps, rs_skypolys, rs_skypasses);
	else if (r_speeds.value)
		Con_Printf ("%3i ms  %4i wpoly %4i epoly %3i lmap\n", (int)((time2 - time1) * 1000), rs_brushpolys, rs_aliaspolys, rs_dynamiclightmaps);
	// johnfitz
}
