// qr_editor.c -- qr light editor: realtime material editor for the vkpt renderer.
//
// Console commands: qr_light_editor_start / qr_light_editor_stop.
//
// While the editor runs the view belongs to a free camera (the player stands
// still): aim with the crosshair, fire selects the face under it and opens the
// material panel on the right edge of the screen. The panel is Dear ImGui
// (Quake/qr_gui.cpp), and it edits the materials.yaml parameters of every
// animation frame of the picked texture (medkits, blinking buttons, ...).
// Apply writes the changes back to the materials/*.yaml files, Cancel reverts
// to the values they were loaded from, Exit closes the editor and returns the
// view to the player.
//
// Editing model: the editor mutates the live rt_material_t structs and
// re-synthesizes the affected textures (TexMgr_ReloadImagesForMaterial); world
// geometry and emissive lights are re-uploaded every frame, so the change is
// visible immediately. A full snapshot of both material lists is taken on
// start (and re-taken after Apply) so Cancel/Exit can restore the yaml state.

#include "quakedef.h"
#include "glquake.h"
#include "gl_model.h"
#include "gl_texmgr.h"
#include "gl_heap.h"
#include "rt_material.h"
#include "keys.h"
#include "draw.h"
#include "client.h"
#include "world.h"
#include "console.h"
#include "mathlib.h"
#include "input.h"
#include "vid.h"

#include "SDL.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>
#endif

#include "qr_editor.h"
#include "qr_gui.h"

#include <ctype.h>
#include <math.h>

extern vec3_t     vpn, vright, vup, r_origin; // gl_rmain.c
extern qboolean   keydown[MAX_KEYS];          // keys.c
extern kbutton_t  in_forward, in_back, in_moveleft, in_moveright, in_up, in_down; // cl_input.c

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

enum
{
	QRE_T_FLOAT,
	QRE_T_INT,
	QRE_T_BOOL,
	QRE_T_TEXT,
	QRE_T_COLOR,
};

enum
{
	PARAM_BASE,     // texture_base
	PARAM_NORMALS,  // texture_normals
	PARAM_EMISSIVE, // texture_emissive
	PARAM_MASK,     // texture_mask
	PARAM_GLOSS,    // texture_gloss
	PARAM_BUMP,
	PARAM_ROUGH,
	PARAM_METAL,
	PARAM_EMISF,
	PARAM_SPEC,
	PARAM_BASEF,
	PARAM_LBRIGHT,
	PARAM_LUPOFF,
	PARAM_ETHRESH,
	PARAM_DRAD,
	PARAM_EBLEND,
	PARAM_KIND,
	PARAM_ISLIGHT,
	PARAM_LSTYLES,
	PARAM_METALALPHA,
	PARAM_BSPRAD,
	PARAM_MIRROR,
	PARAM_EXACTN,
	PARAM_FRAST,
	PARAM_CEMIS,
	PARAM_LCOLOR,
	PARAM_COUNT,
};

static const struct qre_param_s
{
	const char *label;
	int         type;
	float       min, max, step;
} qre_params[PARAM_COUNT] = {
	[PARAM_BASE]     = { "texture_base",     QRE_T_TEXT,  0, 0, 0 },
	[PARAM_NORMALS]  = { "texture_normals",  QRE_T_TEXT,  0, 0, 0 },
	[PARAM_EMISSIVE] = { "texture_emissive", QRE_T_TEXT,  0, 0, 0 },
	[PARAM_MASK]     = { "texture_mask",     QRE_T_TEXT,  0, 0, 0 },
	[PARAM_GLOSS]    = { "texture_gloss",    QRE_T_TEXT,  0, 0, 0 },
	[PARAM_BUMP]     = { "bump_scale",       QRE_T_FLOAT, 0, 4, 0.01f },
	[PARAM_ROUGH]    = { "roughness_override", QRE_T_FLOAT, 0, 1, 0.01f },
	[PARAM_METAL]    = { "metalness_factor", QRE_T_FLOAT, 0, 1, 0.01f },
	[PARAM_EMISF]    = { "emissive_factor",  QRE_T_FLOAT, 0, 4, 0.01f },
	[PARAM_SPEC]     = { "specular_factor",  QRE_T_FLOAT, 0, 4, 0.01f },
	[PARAM_BASEF]    = { "base_factor",      QRE_T_FLOAT, 0, 4, 0.01f },
	[PARAM_LBRIGHT]  = { "light_brightness", QRE_T_FLOAT, 0, 1, 0.001f },
	[PARAM_LUPOFF]   = { "light_upoffset",   QRE_T_FLOAT, -64, 64, 0.5f },
	[PARAM_ETHRESH]  = { "color_emissive_threshold", QRE_T_FLOAT, 0, 1, 0.001f },
	[PARAM_DRAD]     = { "default_radiance", QRE_T_FLOAT, 0, 4, 0.01f },
	[PARAM_EBLEND]   = { "emissive_blend",   QRE_T_INT,  -1, 5, 1 },
	[PARAM_KIND]     = { "kind",             QRE_T_INT,   0, 10, 1 },
	[PARAM_ISLIGHT]  = { "is_light",         QRE_T_BOOL,  0, 0, 0 },
	[PARAM_LSTYLES]  = { "light_styles",     QRE_T_BOOL,  0, 0, 0 },
	[PARAM_METALALPHA] = { "metalness_from_normal_alpha", QRE_T_BOOL, 0, 0, 0 },
	[PARAM_BSPRAD]   = { "bsp_radiance",     QRE_T_BOOL,  0, 0, 0 },
	[PARAM_MIRROR]   = { "mirror",           QRE_T_BOOL,  0, 0, 0 },
	[PARAM_EXACTN]   = { "exact_normals",    QRE_T_BOOL,  0, 0, 0 },
	[PARAM_FRAST]    = { "force_rasterize",  QRE_T_BOOL,  0, 0, 0 },
	[PARAM_CEMIS]    = { "color_emissive",   QRE_T_COLOR, 0, 0, 0 },
	[PARAM_LCOLOR]   = { "light_color",      QRE_T_COLOR, 0, 0, 0 },
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#define QRE_GROUP_MAX    12
#define QRE_DIRTY_MAX    16
#define QRE_TOUCHED_MAX  128

static struct
{
	qboolean active;
	qboolean panel_open;

	vec3_t cam_origin;
	vec3_t player_viewangles;

	// the picked / hovered face (hover is updated while flying)
	qmodel_t   *pick_model;
	msurface_t *pick_surf;
	qmodel_t   *hover_model;
	msurface_t *hover_surf;

	// the material group (animation frames) shown by the panel
	rt_material_t *group[QRE_GROUP_MAX];
	int            group_count;

	// snapshot of both material lists for Cancel/Exit
	rt_material_t *snap_global;
	int            snap_global_count;
	rt_material_t *snap_map;
	int            snap_map_count;

	// a material created by the editor for a texture that has none in yaml
	rt_material_t tmp_mat;
	qboolean      tmp_appended;

	// materials waiting for live re-synthesis / all touched this session
	char dirty[QRE_DIRTY_MAX][MAX_QPATH];
	int  dirty_count;
	char touched[QRE_TOUCHED_MAX][MAX_QPATH];
	int  touched_count;
} qre;

// forward declarations (the panel code sits above the apply/save code)
static void     QRE_Apply (void);
static void     QRE_Cancel (void);
static void     QRE_StopEditor (qboolean restore);
static void     QRE_ClosePanel (void);
static void     QRE_SaveMaterials (void);
static qboolean QRE_BrowseTexture (char *out, size_t outsize);

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

static float QRE_CanvasHeight (void)
{
	return 640.0f * (float)glheight / (float)glwidth;
}

static RgFloat4D QRE_Rgba (float r, float g, float b, float a)
{
	RgFloat4D c;

	c.data[0] = r;
	c.data[1] = g;
	c.data[2] = b;
	c.data[3] = a;
	return c;
}

// "textures/+3_med25" -> 3, "textures/_med25" -> -1 (not an animation frame)
static int QRE_FrameDigit (const char *name)
{
	if (!q_strncasecmp (name, "textures/+", 10) && name[10] >= '0' && name[10] <= '9')
		return name[10] - '0';
	return -1;
}

// The animation base of a material name: "textures/+0_med25" -> "textures/_med25".
static void QRE_GroupBaseOf (const char *matname, char *out, size_t outsize)
{
	if (QRE_FrameDigit (matname) >= 0)
	{
		q_snprintf (out, outsize, "textures/%s", matname + 11);
		return;
	}
	q_strlcpy (out, matname, outsize);
}

static qboolean QRE_NameInGroup (const char *matname, const char *groupbase)
{
	char base[MAX_QPATH];

	QRE_GroupBaseOf (matname, base, sizeof (base));
	return !strcmp (base, groupbase);
}

// ---------------------------------------------------------------------------
// Material snapshot (for Cancel/Exit)
// ---------------------------------------------------------------------------

static void QRE_TakeSnapshot (void)
{
	int count;

	RT_MAT_GetList (RT_MAT_LIST_GLOBAL, &count);
	qre.snap_global_count = count;
	if (!qre.snap_global)
		qre.snap_global = (rt_material_t *)Mem_Alloc (4096 * sizeof (rt_material_t));
	memcpy (qre.snap_global, RT_MAT_GetList (RT_MAT_LIST_GLOBAL, NULL), (size_t)count * sizeof (rt_material_t));

	RT_MAT_GetList (RT_MAT_LIST_MAP, &count);
	qre.snap_map_count = count;
	if (!qre.snap_map)
		qre.snap_map = (rt_material_t *)Mem_Alloc (1024 * sizeof (rt_material_t));
	memcpy (qre.snap_map, RT_MAT_GetList (RT_MAT_LIST_MAP, NULL), (size_t)count * sizeof (rt_material_t));
}

static void QRE_RestoreSnapshot (void)
{
	memcpy (RT_MAT_GetList (RT_MAT_LIST_GLOBAL, NULL), qre.snap_global, (size_t)qre.snap_global_count * sizeof (rt_material_t));
	memcpy (RT_MAT_GetList (RT_MAT_LIST_MAP, NULL), qre.snap_map, (size_t)qre.snap_map_count * sizeof (rt_material_t));
}

static void QRE_FreeSnapshot (void)
{
	if (qre.snap_global)
		Mem_Free (qre.snap_global);
	if (qre.snap_map)
		Mem_Free (qre.snap_map);
	qre.snap_global = NULL;
	qre.snap_map = NULL;
}

// ---------------------------------------------------------------------------
// Dirty tracking and live re-synthesis
// ---------------------------------------------------------------------------

static qboolean QRE_NameInList (const char (*list)[MAX_QPATH], int count, const char *name)
{
	int i;

	for (i = 0; i < count; i++)
	{
		if (!strcmp (list[i], name))
			return true;
	}
	return false;
}

static void QRE_MarkDirty (rt_material_t *m)
{
	if (!m || !m->name[0])
		return;

	if (!QRE_NameInList (qre.touched, qre.touched_count, m->name) && qre.touched_count < QRE_TOUCHED_MAX)
		q_strlcpy (qre.touched[qre.touched_count++], m->name, MAX_QPATH);

	if (!QRE_NameInList (qre.dirty, qre.dirty_count, m->name) && qre.dirty_count < QRE_DIRTY_MAX)
		q_strlcpy (qre.dirty[qre.dirty_count++], m->name, MAX_QPATH);
}

static void QRE_FlushDirty (void)
{
	int flushed = 0;

	while (qre.dirty_count > 0 && flushed < 2)
	{
		char name[MAX_QPATH];

		q_strlcpy (name, qre.dirty[--qre.dirty_count], sizeof (name));
		TexMgr_ReloadImagesForMaterial (name);
		flushed++;
	}
}

static void QRE_ReapplyTouched (void)
{
	int i;

	qre.dirty_count = 0;
	for (i = 0; i < qre.touched_count; i++)
	{
		TexMgr_ReloadImagesForMaterial (qre.touched[i]);
	}
	qre.touched_count = 0;
}

// A material created by the editor becomes part of the live global list on the
// first change, so the synthesis (RT_MAT_Find) can see it.
static void QRE_EnsureLive (int g)
{
	if (qre.group[g] != &qre.tmp_mat || qre.tmp_appended)
		return;

	int idx = RT_MAT_AppendGlobal (&qre.tmp_mat);
	if (idx >= 0)
	{
		qre.tmp_appended = true;
		qre.group[g] = RT_MAT_GetList (RT_MAT_LIST_GLOBAL, NULL) + idx;
	}
}

// ---------------------------------------------------------------------------
// Parameter access
// ---------------------------------------------------------------------------

static void QRE_GetColor (const rt_material_t *m, int param, qboolean *enabled, float *rgb)
{
	if (param == PARAM_CEMIS)
	{
		*enabled = m->has_color_emissive;
		VectorCopy (m->color_emissive, rgb);
	}
	else
	{
		*enabled = m->has_light_color;
		VectorCopy (m->light_color, rgb);
	}
}

static void QRE_SetColorEnabled (int g, int param, qboolean enabled)
{
	QRE_EnsureLive (g);
	rt_material_t *m = qre.group[g];

	if (param == PARAM_CEMIS)
		m->has_color_emissive = enabled;
	else
		m->has_light_color = enabled;
	QRE_MarkDirty (m);
}

static void QRE_SetColorChannel (int g, int param, int channel, float value)
{
	QRE_EnsureLive (g);
	rt_material_t *m = qre.group[g];

	if (param == PARAM_CEMIS)
	{
		m->has_color_emissive = true;
		m->color_emissive[channel] = value;
	}
	else
	{
		m->has_light_color = true;
		m->light_color[channel] = value;
	}
	QRE_MarkDirty (m);
}

static float QRE_GetFloat (const rt_material_t *m, int param)
{
	switch (param)
	{
	case PARAM_BUMP:     return m->bump_scale;
	case PARAM_ROUGH:    return m->roughness_override;
	case PARAM_METAL:    return m->metalness_factor;
	case PARAM_EMISF:    return m->emissive_factor;
	case PARAM_SPEC:     return m->specular_factor;
	case PARAM_BASEF:    return m->base_factor;
	case PARAM_LBRIGHT:  return m->light_brightness;
	case PARAM_LUPOFF:   return m->light_upoffset;
	case PARAM_ETHRESH:  return m->color_emissive_threshold;
	case PARAM_DRAD:     return m->default_radiance;
	default:             return 0.0f;
	}
}

static int QRE_GetInt (const rt_material_t *m, int param)
{
	if (param == PARAM_EBLEND)
		return m->emissive_blend;
	if (param == PARAM_KIND)
		return m->kind;
	return 0;
}

static qboolean QRE_GetBool (const rt_material_t *m, int param)
{
	switch (param)
	{
	case PARAM_ISLIGHT:    return m->is_light;
	case PARAM_LSTYLES:    return m->light_styles;
	case PARAM_METALALPHA: return m->metalness_from_normal_alpha;
	case PARAM_BSPRAD:     return m->bsp_radiance;
	case PARAM_MIRROR:     return m->mirror;
	case PARAM_EXACTN:     return m->exact_normals;
	case PARAM_FRAST:      return m->force_rasterize;
	default:               return false;
	}
}

static const char *QRE_GetText (const rt_material_t *m, int param)
{
	switch (param)
	{
	case PARAM_BASE:     return m->filename_base;
	case PARAM_NORMALS:  return m->filename_normals;
	case PARAM_EMISSIVE: return m->filename_emissive;
	case PARAM_MASK:     return m->filename_mask;
	case PARAM_GLOSS:    return m->filename_gloss;
	default:             return "";
	}
}

static void QRE_SetFloat (int g, int param, float value)
{
	QRE_EnsureLive (g);
	rt_material_t *m = qre.group[g];

	switch (param)
	{
	case PARAM_BUMP:     m->bump_scale = value; break;
	case PARAM_ROUGH:    m->roughness_override = value; break;
	case PARAM_METAL:    m->metalness_factor = value; m->has_metalness_factor = true; break;
	case PARAM_EMISF:    m->emissive_factor = value; break;
	case PARAM_SPEC:     m->specular_factor = value; break;
	case PARAM_BASEF:    m->base_factor = value; break;
	case PARAM_LBRIGHT:  m->light_brightness = value; break;
	case PARAM_LUPOFF:   m->light_upoffset = value; break;
	case PARAM_ETHRESH:  m->color_emissive_threshold = value; break;
	case PARAM_DRAD:     m->default_radiance = value; break;
	default:             break;
	}
	QRE_MarkDirty (m);
}

static void QRE_SetInt (int g, int param, int value)
{
	QRE_EnsureLive (g);
	rt_material_t *m = qre.group[g];

	if (param == PARAM_EBLEND)
		m->emissive_blend = value;
	else if (param == PARAM_KIND)
		m->kind = value;
	QRE_MarkDirty (m);
}

static void QRE_SetBool (int g, int param, qboolean value)
{
	QRE_EnsureLive (g);
	rt_material_t *m = qre.group[g];

	switch (param)
	{
	case PARAM_ISLIGHT:    m->is_light = value; break;
	case PARAM_LSTYLES:    m->light_styles = value; break;
	case PARAM_METALALPHA: m->metalness_from_normal_alpha = value; break;
	case PARAM_BSPRAD:     m->bsp_radiance = value; break;
	case PARAM_MIRROR:     m->mirror = value; break;
	case PARAM_EXACTN:     m->exact_normals = value; break;
	case PARAM_FRAST:      m->force_rasterize = value; break;
	default:               break;
	}
	QRE_MarkDirty (m);
}

static void QRE_SetText (int g, int param, const char *value)
{
	QRE_EnsureLive (g);
	rt_material_t *m = qre.group[g];

	switch (param)
	{
	case PARAM_BASE:     q_strlcpy (m->filename_base, value, sizeof (m->filename_base)); break;
	case PARAM_NORMALS:  q_strlcpy (m->filename_normals, value, sizeof (m->filename_normals)); break;
	case PARAM_EMISSIVE: q_strlcpy (m->filename_emissive, value, sizeof (m->filename_emissive)); break;
	case PARAM_MASK:     q_strlcpy (m->filename_mask, value, sizeof (m->filename_mask)); break;
	case PARAM_GLOSS:    q_strlcpy (m->filename_gloss, value, sizeof (m->filename_gloss)); break;
	default:             break;
	}
	QRE_MarkDirty (m);
}

// ---------------------------------------------------------------------------
// Material group resolution (animation frames)
// ---------------------------------------------------------------------------

// Compares frames: the plain base first, then +N in ascending digit order.
static int QRE_CompareMats (const void *a, const void *b)
{
	const rt_material_t *ma = *(rt_material_t *const *)a;
	const rt_material_t *mb = *(rt_material_t *const *)b;
	int da = QRE_FrameDigit (ma->name);
	int db = QRE_FrameDigit (mb->name);

	if (da != db)
		return (da == -1) ? -1 : (db == -1) ? 1 : da - db;
	return strcmp (ma->name, mb->name);
}

// Builds the group of materials for a texture name (all animation frames).
static void QRE_ResolveGroup (const char *texname)
{
	char groupbase[MAX_QPATH];
	int  pass, i, k;

	QRE_GroupBaseOf (texname, groupbase, sizeof (groupbase));

	qre.group_count = 0;
	qre.tmp_appended = false;

	// the map list wins over the global list for duplicate names
	for (pass = 0; pass < 2 && qre.group_count < QRE_GROUP_MAX; pass++)
	{
		int            count;
		rt_material_t *list = RT_MAT_GetList (pass == 0 ? RT_MAT_LIST_MAP : RT_MAT_LIST_GLOBAL, &count);

		for (i = 0; i < count && qre.group_count < QRE_GROUP_MAX; i++)
		{
			qboolean dup = false;

			if (!list[i].valid)
				continue;
			if (!QRE_NameInGroup (list[i].name, groupbase))
				continue;
			for (k = 0; k < qre.group_count; k++)
			{
				if (!strcmp (qre.group[k]->name, list[i].name))
				{
					dup = true;
					break;
				}
			}
			if (dup)
				continue;
			qre.group[qre.group_count++] = &list[i];
		}
	}

	if (qre.group_count == 0)
	{
		// no material authored for this texture: edit a detached default one
		// that joins the global list on the first change
		memset (&qre.tmp_mat, 0, sizeof (qre.tmp_mat));
		qre.tmp_mat.valid = true;
		qre.tmp_mat.bump_scale = 1.0f;
		qre.tmp_mat.emissive_factor = 1.0f;
		qre.tmp_mat.emissive_blend = -1;
		qre.tmp_mat.specular_factor = 1.0f;
		qre.tmp_mat.base_factor = 1.0f;
		qre.tmp_mat.light_brightness = 1.0f;
		qre.tmp_mat.kind = RT_MAT_KIND_REGULAR;
		qre.tmp_mat.light_styles = true;
		qre.tmp_mat.color_emissive_threshold = 0.02f;
		q_strlcpy (qre.tmp_mat.name, groupbase, sizeof (qre.tmp_mat.name));
		qre.group[0] = &qre.tmp_mat;
		qre.group_count = 1;
	}
	else if (qre.group_count > 1)
	{
		qsort (qre.group, (size_t)qre.group_count, sizeof (qre.group[0]), QRE_CompareMats);
	}
}

// ---------------------------------------------------------------------------
// Picking
// ---------------------------------------------------------------------------

static qboolean QRE_PointInPolygon (const glpoly_t *poly, const vec3_t p, const mplane_t *plane)
{
	int ax, ay, i, j, n;
	int inside = 0;

	if (fabsf (plane->normal[0]) >= fabsf (plane->normal[1]) && fabsf (plane->normal[0]) >= fabsf (plane->normal[2]))
	{
		ax = 1; ay = 2;
	}
	else if (fabsf (plane->normal[1]) >= fabsf (plane->normal[2]))
	{
		ax = 0; ay = 2;
	}
	else
	{
		ax = 0; ay = 1;
	}

	n = poly->numverts;
	for (i = 0, j = n - 1; i < n; j = i++)
	{
		float xi = poly->verts[i][ax], yi = poly->verts[i][ay];
		float xj = poly->verts[j][ax], yj = poly->verts[j][ay];

		if (((yi > p[ay]) != (yj > p[ay])) &&
		    (p[ax] < (xj - xi) * (p[ay] - yi) / (yj - yi) + xi))
			inside = !inside;
	}
	return inside != 0;
}

// Finds the surface of a model whose face contains the impact point. Coplanar
// faces are told apart by a polygon test; the nearest plane is the fallback.
static msurface_t *QRE_FindSurface (qmodel_t *model, const vec3_t impact)
{
	int         i;
	msurface_t *best = NULL;
	float       bestd = 1.0f;

	for (i = 0; i < model->nummodelsurfaces; i++)
	{
		msurface_t *s = &model->surfaces[model->firstmodelsurface + i];
		glpoly_t   *p;
		float       d;

		if (!s->texinfo || !s->texinfo->texture)
			continue;
		if (s->flags & (SURF_DRAWSKY | SURF_NOTEXTURE))
			continue;

		d = DotProduct (s->plane->normal, impact) - s->plane->dist;
		if (fabsf (d) > 1.0f)
			continue;

		for (p = s->polys; p; p = p->next)
		{
			if (QRE_PointInPolygon (p, impact, s->plane))
				return s;
		}

		if (fabsf (d) < bestd)
		{
			bestd = fabsf (d);
			best = s;
		}
	}

	return best;
}

static qboolean QRE_TracePick (qmodel_t **out_model, msurface_t **out_surf, gltexture_t **out_glt)
{
	vec3_t    start, end;
	trace_t   tr;
	float     best = 1.0f;
	qmodel_t *bestmodel = NULL;
	vec3_t    bestimpact = { 0, 0, 0 };

	if (!cl.worldmodel)
		return false;

	VectorCopy (r_origin, start);
	VectorMA (start, 8192.0f, vpn, end);

	// the world, including liquid and lava surfaces (they carry materials too)
	memset (&tr, 0, sizeof (tr));
	tr.fraction = 1.0f;
	SV_RecursiveHullCheck (&cl.worldmodel->hulls[0], start, end, &tr,
	                       CONTENTMASK_ANYSOLID |
	                       CONTENTMASK_FROMQ1 (CONTENTS_WATER) |
	                       CONTENTMASK_FROMQ1 (CONTENTS_SLIME) |
	                       CONTENTMASK_FROMQ1 (CONTENTS_LAVA));
	if (!tr.startsolid && tr.fraction < best)
	{
		best = tr.fraction;
		bestmodel = cl.worldmodel;
		VectorCopy (tr.endpos, bestimpact);
	}

	// brush entities (health boxes, buttons, doors, ...)
	{
		vec3_t eimpact, enorm;
		int    entnum;
		float  f = CL_TraceLine (start, end, eimpact, enorm, &entnum);

		if (f < best && entnum >= 0 && entnum < MAX_EDICTS && cl.entities[entnum].model)
		{
			best = f;
			bestmodel = cl.entities[entnum].model;
			VectorCopy (eimpact, bestimpact);
		}
	}

	if (!bestmodel)
		return false;

	*out_surf = QRE_FindSurface (bestmodel, bestimpact);
	if (!*out_surf)
		return false;

	*out_model = bestmodel;
	if (out_glt)
		*out_glt = (*out_surf)->texinfo->texture->gltexture;
	return true;
}

// Hover pick (crosshair, flying) or select pick (fire button).
static void QRE_DoPick (qboolean select)
{
	qmodel_t    *model;
	msurface_t  *surf;
	gltexture_t *glt;

	if (!QRE_TracePick (&model, &surf, &glt))
	{
		qre.hover_model = NULL;
		qre.hover_surf = NULL;
		return;
	}

	if (!select)
	{
		qre.hover_model = model;
		qre.hover_surf = surf;
		return;
	}

	if (!glt)
		return;
	if (!QR_GUI_Ready ())
	{
		Con_Printf ("qr editor: the ImGui panel is not available\n");
		return;
	}

	qre.pick_model = model;
	qre.pick_surf = surf;
	qre.hover_model = NULL;
	qre.hover_surf = NULL;

	{
		char texname[MAX_QPATH];
		char *dot;

		RT_MAT_NormalizeName (glt->name, texname, sizeof (texname));
		dot = strrchr (texname, '.');
		if (dot && !strchr (dot, ':'))
			*dot = '\0';

		QRE_ResolveGroup (texname);
	}

	qre.panel_open = true;

	// the panel owns the mouse: free the cursor, freeze the camera
	IN_Deactivate (true);
	SDL_ShowCursor (SDL_DISABLE);
	QR_GUI_SetMouseCursor (1);
}

// ---------------------------------------------------------------------------
// Selection outline
// ---------------------------------------------------------------------------

static void QRE_EmitOutline (qmodel_t *model, msurface_t *surf, uint32_t color)
{
	const float nudge = 0.35f;
	vec3_t    *verts = NULL;
	int        n = 0;
	int        i, vi, ii;
	RgVertex  *rv;
	uint32_t  *ri;

	// the face outline from the BSP edge list, or the polygon when the face
	// has no edges (should not happen for regular faces)
	if (surf->numedges > 0)
	{
		n = surf->numedges;
		verts = (vec3_t *)RT_AllocScratchMemoryNulled ((size_t)n * sizeof (vec3_t));
		for (i = 0; i < n; i++)
		{
			int e = model->surfedges[surf->firstedge + i];
			if (e >= 0)
				VectorCopy (model->vertexes[model->edges[e].v[0]].position, verts[i]);
			else
				VectorCopy (model->vertexes[model->edges[-e].v[1]].position, verts[i]);
		}
	}
	else if (surf->polys && surf->polys->numverts > 0)
	{
		glpoly_t *p = surf->polys;

		n = p->numverts;
		verts = (vec3_t *)RT_AllocScratchMemoryNulled ((size_t)n * sizeof (vec3_t));
		for (i = 0; i < n; i++)
			VectorCopy (p->verts[i], verts[i]);
	}

	if (!verts || n < 3)
		return;

	rv = (RgVertex *)RT_AllocScratchMemoryNulled ((size_t)n * sizeof (RgVertex));
	ri = (uint32_t *)RT_AllocScratchMemoryNulled ((size_t)n * 2 * sizeof (uint32_t));

	vi = 0;
	ii = 0;
	for (i = 0; i < n; i++)
	{
		VectorMA (verts[i], nudge, surf->plane->normal, rv[vi].position);
		rv[vi].packedColor = color;
		vi++;
	}
	for (i = 0; i < n; i++)
	{
		ri[ii++] = (uint32_t)i;
		ri[ii++] = (uint32_t)((i + 1) % n);
	}

	RgRasterizedGeometryUploadInfo info = {
		.renderType = RG_RASTERIZED_GEOMETRY_RENDER_TYPE_DEFAULT,
		.vertexCount = (uint32_t)vi,
		.pVertices = rv,
		.indexCount = (uint32_t)ii,
		.pIndices = ri,
		.transform = RT_TRANSFORM_IDENTITY,
		.color = RT_COLOR_WHITE,
		.material = RG_NO_MATERIAL,
		.pipelineState = RG_RASTERIZED_GEOMETRY_STATE_FORCE_LINE_LIST | RG_RASTERIZED_GEOMETRY_STATE_DEPTH_TEST,
		.blendFuncSrc = 0,
		.blendFuncDst = 0,
	};

	RgResult r = rgUploadRasterizedGeometry (vulkan_globals.instance, &info, NULL, NULL);
	RG_CHECK (r);
}

void QR_Editor_DrawSelection (cb_context_t *cbx)
{
	(void)cbx;

	if (!qre.active)
		return;

	if (qre.panel_open && qre.pick_surf)
	{
		QRE_EmitOutline (qre.pick_model, qre.pick_surf, RT_PackColorToUint32 (255, 255, 255, 255));
	}
	else if (!qre.panel_open && qre.hover_surf)
	{
		QRE_EmitOutline (qre.hover_model, qre.hover_surf, RT_PackColorToUint32 (255, 214, 64, 255));
	}
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

static void QRE_MoveCamera (void)
{
	vec3_t fwd, right, up;
	float  fm, sm, um, speed;
	int    i;

	AngleVectors (cl.viewangles, fwd, right, up);

	fm = (float)((in_forward.state & 1) - (in_back.state & 1));
	sm = (float)((in_moveright.state & 1) - (in_moveleft.state & 1));
	um = (float)((in_up.state & 1) - (in_down.state & 1));

	speed = 450.0f * (float)host_frametime;
	if (keydown[K_SHIFT])
		speed *= 3.0f;

	for (i = 0; i < 3; i++)
		qre.cam_origin[i] += (fwd[i] * fm + right[i] * sm + up[i] * um) * speed;
}

void QR_Editor_UpdateView (void)
{
	if (!qre.active)
		return;

	if (!qre.panel_open)
		QRE_MoveCamera ();

	VectorCopy (qre.cam_origin, r_refdef.vieworg);
	VectorCopy (cl.viewangles, r_refdef.viewangles);
}

// ---------------------------------------------------------------------------
// Panel (Dear ImGui)
// ---------------------------------------------------------------------------

static void QRE_ParamWidgets (int g)
{
	rt_material_t *m = qre.group[g];
	int            p;

	for (p = 0; p < PARAM_COUNT; p++)
	{
		const char *label = qre_params[p].label;

		switch (qre_params[p].type)
		{
		case QRE_T_TEXT:
		{
			char buf[MAX_QPATH];
			int  res;
			char file[MAX_QPATH];

			q_strlcpy (buf, QRE_GetText (m, p), sizeof (buf));
			res = QR_GUI_TexturePath (label, buf, sizeof (buf));
			if ((res & 1) && strcmp (buf, QRE_GetText (m, p)))
				QRE_SetText (g, p, buf);
			if (res & 2)
			{
				if (QRE_BrowseTexture (file, sizeof (file)))
					QRE_SetText (g, p, file);
			}
			break;
		}
		case QRE_T_FLOAT:
		{
			float value = QRE_GetFloat (m, p);
			if (QR_GUI_SliderFloat (label, &value, qre_params[p].min, qre_params[p].max))
				QRE_SetFloat (g, p, value);
			break;
		}
		case QRE_T_INT:
		{
			int value = QRE_GetInt (m, p);

			if (p == PARAM_KIND)
			{
				static const char *const kinds[] = {
					"REGULAR", "CHROME", "WATER", "LAVA", "SLIME", "GLASS", "SKY", "INVISIBLE", "SCREEN", "CAMERA"
				};
				if (QR_GUI_Combo (label, &value, kinds, (int)countof (kinds)))
					QRE_SetInt (g, p, value);
			}
			else if (p == PARAM_EBLEND)
			{
				static const char *const blends[] = { "cvar", "0", "1", "2", "3", "4", "5" };
				int index = value + 1;
				if (QR_GUI_Combo (label, &index, blends, (int)countof (blends)))
					QRE_SetInt (g, p, index - 1);
			}
			else
			{
				if (QR_GUI_SliderInt (label, &value, (int)qre_params[p].min, (int)qre_params[p].max))
					QRE_SetInt (g, p, value);
			}
			break;
		}
		case QRE_T_BOOL:
		{
			int value = QRE_GetBool (m, p) ? 1 : 0;
			if (QR_GUI_Checkbox (label, &value))
				QRE_SetBool (g, p, value != 0);
			break;
		}
		case QRE_T_COLOR:
		{
			qboolean enabled;
			float    rgb[3];
			float    old_rgb[3];
			int      en;
			int      c;

			QRE_GetColor (m, p, &enabled, rgb);
			VectorCopy (rgb, old_rgb);
			en = enabled ? 1 : 0;

			if (QR_GUI_ColorHex (label, rgb, &en))
			{
				if (!en)
				{
					QRE_SetColorEnabled (g, p, false);
				}
				else
				{
					QRE_SetColorEnabled (g, p, true);
					for (c = 0; c < 3; c++)
						if (rgb[c] != old_rgb[c])
							QRE_SetColorChannel (g, p, c, rgb[c]);
				}
			}
			break;
		}
		default:
			break;
		}
	}
}

static void QRE_BuildPanelGUI (void)
{
	int      panel_w = glwidth * 2 / 5;
	int      g;
	qboolean exit_requested = false;

	if (panel_w > 460)
		panel_w = 460;
	if (panel_w < 320)
		panel_w = 320;

	QR_GUI_BeginPanel ("qr_material_editor", glwidth - panel_w, 0, panel_w, glheight);

	QR_GUI_Label ("MATERIAL EDITOR");
	if (qre.pick_surf && qre.pick_surf->texinfo && qre.pick_surf->texinfo->texture)
	{
		char buf[MAX_QPATH + 16];

		q_snprintf (buf, sizeof (buf), "face: %s", qre.pick_surf->texinfo->texture->name);
		QR_GUI_LabelDim (buf);
	}
	QR_GUI_Spacing ();

	if (QR_GUI_Button ("Apply"))
		QRE_Apply ();
	QR_GUI_SameLine ();
	if (QR_GUI_Button ("Cancel"))
		QRE_Cancel ();
	QR_GUI_SameLine ();
	if (QR_GUI_Button ("Exit"))
		exit_requested = true;

	QR_GUI_Separator ();
	QR_GUI_BeginScroll ();

	for (g = 0; g < qre.group_count; g++)
	{
		if (QR_GUI_Section (qre.group[g]->name, 1))
			QRE_ParamWidgets (g);
	}

	QR_GUI_EndScroll ();
	QR_GUI_EndPanel ();

	if (exit_requested)
		QRE_StopEditor (true);
}

static void QRE_DrawFlyingHint (cb_context_t *cbx)
{
	const RgFloat4D c = QRE_Rgba (0.8f, 0.85f, 0.9f, 0.9f);
	const float H = QRE_CanvasHeight ();

	GL_SetCanvas (cbx, CANVAS_EDITOR);
	Draw_StringScaled (cbx, 8, (int)(H - 46), "qr light editor", 1.0f, &c);
	Draw_StringScaled (cbx, 8, (int)(H - 36), "LMB: select face under crosshair", 1.0f, &c);
	Draw_StringScaled (cbx, 8, (int)(H - 26), "WASD + mouse: fly    Shift: faster    jump/movedown: up/down", 1.0f, &c);
	Draw_StringScaled (cbx, 8, (int)(H - 16), "ESC: exit editor    ~: console", 1.0f, &c);
}

// ---------------------------------------------------------------------------
// Panel: per-frame bookkeeping
// ---------------------------------------------------------------------------

static void QRE_Frame (void)
{
	// the level went away under the editor: drop it (the material snapshot may
	// be stale relative to a freshly loaded map list, so nothing is restored)
	if (cls.state != ca_connected || !cl.worldmodel)
	{
		QRE_StopEditor (false);
		return;
	}

	if (!qre.panel_open)
		QRE_DoPick (false);

	QRE_FlushDirty ();
}

void QR_Editor_DrawPanel (cb_context_t *cbx)
{
	if (!qre.active)
		return;

	QRE_Frame ();

	if (!qre.panel_open)
	{
		QRE_DrawFlyingHint (cbx);
		return;
	}

	// SCR_UpdateScreen can run more than once per host frame
	if (!QR_GUI_BeginFrame ((unsigned int)host_framecount, (float)host_frametime, glx, gly, glwidth, glheight, vid.height))
		return;

	QRE_BuildPanelGUI ();

	QR_GUI_EndFrame ();
}

// ---------------------------------------------------------------------------
// Input hooks
// ---------------------------------------------------------------------------

qboolean QR_Editor_KeyEvent (int key, qboolean down)
{
	if (!qre.active)
		return false;

	// while the panel is open its events are consumed at the SDL level
	// (QR_Editor_GuiProcessEvent); only the flying mode is left here
	if (qre.panel_open)
		return false;

	if (key == K_ESCAPE && down)
	{
		QRE_StopEditor (true);
		return true;
	}

	return false;
}

// Called for every SDL event before the engine handles it.
qboolean QR_Editor_GuiProcessEvent (const void *sdl_event)
{
	const SDL_Event *e = (const SDL_Event *)sdl_event;

	if (!qre.active || !qre.panel_open)
		return false;

	// ESC closes the panel, unless an ImGui text field is editing
	if (e->type == SDL_KEYDOWN && e->key.keysym.sym == SDLK_ESCAPE && !QR_GUI_WantsKeyboard ())
	{
		QRE_ClosePanel ();
		return true;
	}

	return QR_GUI_ProcessEvent (sdl_event) ? true : false;
}

qboolean QR_Editor_TextEntryActive (void)
{
	// while the panel is open SDL text input must stay on for ImGui fields
	return qre.active && qre.panel_open;
}

// ---------------------------------------------------------------------------
// Apply / Cancel / Exit
// ---------------------------------------------------------------------------

static void QRE_ClosePanel (void)
{
	if (!qre.panel_open)
		return;

	qre.panel_open = false;
	qre.pick_model = NULL;
	qre.pick_surf = NULL;

	IN_Activate ();
	SDL_ShowCursor (SDL_ENABLE);
	QR_GUI_SetMouseCursor (0);
}

static void QRE_Apply (void)
{
	QRE_SaveMaterials ();
	QRE_TakeSnapshot (); // Cancel now reverts to the state just saved
	qre.touched_count = 0;
	Con_Printf ("qr editor: materials written to %s/materials/\n", com_gamedir);
}

static void QRE_Cancel (void)
{
	QRE_RestoreSnapshot ();
	QRE_ReapplyTouched (); // put the yaml values back on screen

	qre.tmp_appended = false;

	// the restored lists may no longer contain the edited materials:
	// rebuild the group (a picked texture without a material goes back to the
	// detached defaults)
	if (qre.pick_surf && qre.pick_surf->texinfo && qre.pick_surf->texinfo->texture)
	{
		gltexture_t *glt = qre.pick_surf->texinfo->texture->gltexture;

		if (glt)
		{
			char texname[MAX_QPATH];
			char *dot;

			RT_MAT_NormalizeName (glt->name, texname, sizeof (texname));
			dot = strrchr (texname, '.');
			if (dot && !strchr (dot, ':'))
				*dot = '\0';
			QRE_ResolveGroup (texname);
		}
	}

	Con_Printf ("qr editor: materials reverted to the values from materials.yaml\n");
}

// ---------------------------------------------------------------------------
// Saving materials.yaml
// ---------------------------------------------------------------------------

// Written to the top of materials/materials.yaml. Kept in sync with the header
// of vkpt/Source/materials.yaml, which documents the accepted keys.
static const char *qre_yaml_header =
	"# Global material definitions for the vkpt ray-traced renderer.\n"
	"# `is_light: false` marks a *static* surface (textures/*) whose luma\n"
	"# texture should generate emissive triangle lights. Dynamic surfaces\n"
	"# (progs/* models and sprites) are gated separately by the engine.\n"
	"#\n"
	"# Emissive masks can be authored two ways:\n"
	"#   * `texture_emissive: textures/foo_luma.png` -- a hand-painted mask file:\n"
	"#     its pixels' luminance is the emission. Most precise; keeps the mask\n"
	"#     independent of the diffuse art.\n"
	"#   * `color_emissive: ff0000` -- no mask file: the mask is synthesized from\n"
	"#     the base texture, so only pixels close to that colour glow. The tone\n"
	"#     tolerance is `color_emissive_threshold` (0..1 colour-cube distance\n"
	"#     divided by sqrt(3); default 0.02 = very strict). The synthesized mask is\n"
	"#     white where the pixel matches that colour exactly and decays\n"
	"#     exponentially to black towards the threshold, so the glow fades out\n"
	"#     softly instead of ending in a hard edge. `emissive_factor` scales the\n"
	"#     result. Combine with `is_light: true` to also cast light (otherwise the\n"
	"#     surface only glows).\n"
	"# Precedence: an authored `texture_emissive` always wins -- while the key is\n"
	"# present in an entry, `color_emissive` in that same entry is ignored\n"
	"# entirely (even if the luma file fails to load, which is reported at\n"
	"# startup). The classic fullbright mask is merged rather than replaced, so\n"
	"# its pixels emit in addition to the luma mask.\n"
	"#\n"
	"# `emissive_blend: N` overrides the global `rt_emis_blend` cvar for that\n"
	"# material only; without the key the cvar value is used. It selects how the\n"
	"# screen-space emission of the surface is composited into the HDR image\n"
	"# before tone mapping:\n"
	"#   0 - emission is not composited at all\n"
	"#   1 - emission is used as coverage (this is the cvar default)\n"
	"#   2 - emission is added on top of the image (brightest, keeps saturation)\n"
	"#   3 - overlay, driven by the underlying base color\n"
	"#   4 - overlay, driven by the emission color\n"
	"#   5 - divide (emission acts as a darkener)\n"
	"# Intended for emissive *mirrored* surfaces (stained glass, lit windows):\n"
	"# they read as washed out in the default mode, while the additive mode keeps\n"
	"# them bright and saturated.\n";

static void QRE_WriteColor (FILE *f, const char *key, const vec3_t rgb)
{
	fprintf (f, "    %s: %02x%02x%02x\n", key,
	         (int)(rgb[0] * 255.0f + 0.5f) & 0xff,
	         (int)(rgb[1] * 255.0f + 0.5f) & 0xff,
	         (int)(rgb[2] * 255.0f + 0.5f) & 0xff);
}

// Writes one material entry. Only values that differ from the defaults are
// emitted (plus texture paths and color flags), so the file stays readable and
// matches the style it was loaded from.
static void QRE_WriteMaterial (FILE *f, const rt_material_t *m)
{
	fprintf (f, "  - name: %s\n", m->name);
	if (m->filename_base[0])
		fprintf (f, "    texture_base: %s\n", m->filename_base);
	if (m->filename_normals[0])
		fprintf (f, "    texture_normals: %s\n", m->filename_normals);
	if (m->filename_emissive[0])
		fprintf (f, "    texture_emissive: %s\n", m->filename_emissive);
	if (m->filename_mask[0])
		fprintf (f, "    texture_mask: %s\n", m->filename_mask);
	if (m->filename_gloss[0])
		fprintf (f, "    texture_gloss: %s\n", m->filename_gloss);
	if (m->bump_scale != 1.0f)
		fprintf (f, "    bump_scale: %.6g\n", m->bump_scale);
	if (m->roughness_override != 0.0f)
		fprintf (f, "    roughness_override: %.6g\n", m->roughness_override);
	if (m->has_metalness_factor)
		fprintf (f, "    metalness_factor: %.6g\n", m->metalness_factor);
	if (m->metalness_from_normal_alpha)
		fprintf (f, "    metalness_from_normal_alpha: true\n");
	if (m->emissive_factor != 1.0f)
		fprintf (f, "    emissive_factor: %.6g\n", m->emissive_factor);
	if (m->emissive_blend >= 0)
		fprintf (f, "    emissive_blend: %d\n", m->emissive_blend);
	if (m->specular_factor != 1.0f)
		fprintf (f, "    specular_factor: %.6g\n", m->specular_factor);
	if (m->base_factor != 1.0f)
		fprintf (f, "    base_factor: %.6g\n", m->base_factor);
	if (m->kind != RT_MAT_KIND_REGULAR)
	{
		const char *k = RT_MAT_KindName (m->kind);

		fprintf (f, "    kind: %s\n", k ? k : "REGULAR");
	}
	if (m->is_light)
		fprintf (f, "    is_light: true\n");
	if (!m->light_styles)
		fprintf (f, "    light_styles: false\n");
	if (m->bsp_radiance)
		fprintf (f, "    bsp_radiance: true\n");
	if (m->default_radiance != 0.0f)
		fprintf (f, "    default_radiance: %.6g\n", m->default_radiance);
	if (m->has_color_emissive)
		QRE_WriteColor (f, "color_emissive", m->color_emissive);
	if (m->color_emissive_threshold != 0.02f)
		fprintf (f, "    color_emissive_threshold: %.6g\n", m->color_emissive_threshold);
	if (m->has_light_color)
		QRE_WriteColor (f, "light_color", m->light_color);
	if (m->light_brightness != 1.0f)
		fprintf (f, "    light_brightness: %.6g\n", m->light_brightness);
	if (m->light_upoffset != 0.0f)
		fprintf (f, "    light_upoffset: %.6g\n", m->light_upoffset);
	if (m->mirror)
		fprintf (f, "    mirror: true\n");
	if (m->exact_normals)
		fprintf (f, "    exact_normals: true\n");
	if (m->force_rasterize)
		fprintf (f, "    force_rasterize: true\n");
}

#define QRE_SAVE_FILES_MAX   8
#define QRE_SAVE_MATS_MAX    600

typedef struct qre_save_group_s
{
	char                file[MAX_QPATH];
	const rt_material_t *mats[QRE_SAVE_MATS_MAX];
	int                 count;
} qre_save_group_t;

// Adds a material to its source-file group, deduplicating (file, name) pairs
// (the map list and the global list both carry materials/<map>.yaml entries).
static void QRE_SaveAdd (qre_save_group_t *groups, int *ngroups, const rt_material_t *m)
{
	const char       *file = m->source_file[0] ? m->source_file : "materials/materials.yaml";
	qre_save_group_t *g = NULL;
	int              i;

	for (i = 0; i < *ngroups; i++)
	{
		if (!strcmp (groups[i].file, file))
		{
			g = &groups[i];
			break;
		}
	}
	if (!g)
	{
		if (*ngroups >= QRE_SAVE_FILES_MAX)
			return;
		g = &groups[(*ngroups)++];
		q_strlcpy (g->file, file, sizeof (g->file));
		g->count = 0;
	}

	for (i = 0; i < g->count; i++)
	{
		if (!strcmp (g->mats[i]->name, m->name))
			return; // duplicate (map list wins: it was added first)
	}

	if (g->count < QRE_SAVE_MATS_MAX)
		g->mats[g->count++] = m;
}

static void QRE_SaveMaterials (void)
{
	qre_save_group_t groups[QRE_SAVE_FILES_MAX];
	int              ngroups = 0;
	int              count, i, k;

	// map materials first so they win the duplicate check against the same
	// entries that the global list also carries
	for (k = 0; k < 2; k++)
	{
		rt_material_t *list = RT_MAT_GetList (k == 0 ? RT_MAT_LIST_MAP : RT_MAT_LIST_GLOBAL, &count);

		for (i = 0; i < count; i++)
		{
			if (list[i].valid)
				QRE_SaveAdd (groups, &ngroups, &list[i]);
		}
	}

	for (k = 0; k < ngroups; k++)
	{
		char path[MAX_OSPATH];
		FILE *f;

		q_snprintf (path, sizeof (path), "%s/%s", com_gamedir, groups[k].file);

		f = fopen (path, "w");
		if (!f)
		{
			Con_Printf ("qr editor: cannot write %s\n", path);
			continue;
		}

		if (strstr (groups[k].file, "materials.yaml"))
			fprintf (f, "%s", qre_yaml_header);

		fprintf (f, "materials:\n");
		for (i = 0; i < groups[k].count; i++)
		{
			QRE_WriteMaterial (f, groups[k].mats[i]);
		}

		fclose (f);
		Con_Printf ("qr editor: wrote %d materials to %s\n", groups[k].count, path);
	}
}

// ---------------------------------------------------------------------------
// Texture browse (Win32 open dialog; typed entry elsewhere)
// ---------------------------------------------------------------------------

#ifdef _WIN32
static qboolean QRE_BrowseTexture (char *out, size_t outsize)
{
	char          initdir[MAX_OSPATH];
	char          result[MAX_OSPATH];
	OPENFILENAMEA ofn;
	size_t        glen, i;

	q_snprintf (initdir, sizeof (initdir), "%s/textures", com_gamedir);

	memset (&ofn, 0, sizeof (ofn));
	result[0] = '\0';
	ofn.lStructSize = sizeof (ofn);
	ofn.lpstrFilter = "Images (*.png;*.tga;*.jpg;*.jpeg;*.ktx2)\0*.png;*.tga;*.jpg;*.jpeg;*.ktx2\0All files (*.*)\0*.*\0";
	ofn.lpstrFile = result;
	ofn.nMaxFile = sizeof (result);
	ofn.lpstrInitialDir = initdir;
	ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

	if (!GetOpenFileNameA (&ofn))
		return false;

	// relativize against the gamedir and normalize the separators
	glen = strlen (com_gamedir);
	if (!q_strncasecmp (result, com_gamedir, glen) && (result[glen] == '/' || result[glen] == '\\'))
		q_strlcpy (out, result + glen + 1, outsize);
	else
		q_strlcpy (out, result, outsize);

	for (i = 0; out[i]; i++)
	{
		if (out[i] == '\\')
			out[i] = '/';
	}
	return true;
}
#else
static qboolean QRE_BrowseTexture (char *out, size_t outsize)
{
	(void)out;
	(void)outsize;
	return false;
}
#endif

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

static void QRE_StopEditor (qboolean restore)
{
	if (!qre.active)
		return;

	// revert whatever was not applied, then restore the player's view
	if (restore)
	{
		QRE_RestoreSnapshot ();
		QRE_ReapplyTouched ();
	}

	qre.active = false;
	qre.panel_open = false;
	qre.pick_model = NULL;
	qre.pick_surf = NULL;
	qre.hover_model = NULL;
	qre.hover_surf = NULL;

	VectorCopy (qre.player_viewangles, cl.viewangles);

	QRE_FreeSnapshot ();

	IN_Activate ();
	SDL_ShowCursor (SDL_ENABLE);
	QR_GUI_SetMouseCursor (0);

	Con_Printf ("qr light editor: off\n");
}

static void QR_Editor_Start_f (void)
{
	if (qre.active)
	{
		Con_Printf ("qr light editor: already running\n");
		return;
	}
	if (cls.state != ca_connected || !cl.worldmodel)
	{
		Con_Printf ("qr light editor: a level must be loaded first\n");
		return;
	}

	memset (&qre, 0, sizeof (qre));
	qre.active = true;
	qre.panel_open = false;

	VectorCopy (r_refdef.vieworg, qre.cam_origin);
	VectorCopy (cl.viewangles, qre.player_viewangles);

	QRE_TakeSnapshot ();

	Con_Printf ("qr light editor: on (fly: WASD + mouse; LMB selects a face; ESC exits)\n");
}

static void QR_Editor_Stop_f (void)
{
	QRE_StopEditor (true);
}

void QR_Editor_Init (void)
{
	static qboolean qr_editor_registered = false;
	char            font_path[MAX_OSPATH];

	if (qr_editor_registered)
		return;
	qr_editor_registered = true;

	Cmd_AddCommand ("qr_light_editor_start", QR_Editor_Start_f);
	Cmd_AddCommand ("qr_light_editor_stop", QR_Editor_Stop_f);

	// the font is deployed next to the executable by the build
	q_snprintf (font_path, sizeof (font_path), "%s/fonts/Roboto-Regular.ttf", host_parms->basedir);
	QR_GUI_Init (VID_GetWindow (), vulkan_globals.instance, font_path);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

qboolean QR_Editor_Active (void)
{
	return qre.active;
}

qboolean QR_Editor_PanelOpen (void)
{
	return qre.active && qre.panel_open;
}

qboolean QR_Editor_Flying (void)
{
	return qre.active && !qre.panel_open;
}

void QR_Editor_Pick (void)
{
	if (!qre.active || qre.panel_open)
		return;
	QRE_DoPick (true);
}
