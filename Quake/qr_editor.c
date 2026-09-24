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
// re-synthesizes the affected textures (TexMgr_ReloadImagesForMaterial). That
// replaces the material's RgMaterial, and the traced world bakes a material's
// texture indices when it is uploaded, so after a batch of edits the world is
// asked to re-upload itself — the rt_require_static_submit mechanism the light
// style cvar already uses — which also re-collects its emissive lights. A full
// snapshot of both material lists is taken on start (and re-taken after Apply)
// so Cancel/Exit can restore the yaml state.

#include "quakedef.h"
#include "glquake.h"
#include "gl_model.h"
#include "gl_texmgr.h"
#include "gl_heap.h"
#include "rt_material.h"
#include "keys.h"
#include "client.h"
#include "world.h"
#include "console.h"
#include "mathlib.h"
#include "input.h"
#include "vid.h"
#include "atomics.h"

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
#include <stdarg.h>

extern vec3_t     vpn, vright, vup, r_origin; // gl_rmain.c
extern qboolean   keydown[MAX_KEYS];          // keys.c
extern kbutton_t  in_forward, in_back, in_moveleft, in_moveright, in_up, in_down; // cl_input.c
extern atomic_uint32_t rt_require_static_submit; // gl_rmain.c

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
#define QRE_DIRTY_MAX    64
#define QRE_TOUCHED_MAX  512

static struct
{
	qboolean active;
	qboolean panel_open;

	vec3_t cam_origin;
	vec3_t player_viewangles;

	// the picked / hovered face (hover is updated while flying); ent is NULL
	// for a world face and the brush entity for a model face
	qmodel_t   *pick_model;
	msurface_t *pick_surf;
	entity_t   *pick_ent;
	qmodel_t   *hover_model;
	msurface_t *hover_surf;
	entity_t   *hover_ent;

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

// Editor feedback goes to the ImGui notification line and to the console log.
static void QRE_Notify (const char *fmt, ...)
{
	char    buf[256];
	va_list ap;

	va_start (ap, fmt);
	vsnprintf (buf, sizeof (buf), fmt, ap);
	va_end (ap);

	QR_GUI_Notify (buf);
	Con_Printf ("qr editor: %s\n", buf);
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
		qre.snap_global = (rt_material_t *)Mem_Alloc (RT_MAT_CAP_GLOBAL * sizeof (rt_material_t));
	memcpy (qre.snap_global, RT_MAT_GetList (RT_MAT_LIST_GLOBAL, NULL), (size_t)count * sizeof (rt_material_t));

	RT_MAT_GetList (RT_MAT_LIST_MAP, &count);
	qre.snap_map_count = count;
	if (!qre.snap_map)
		qre.snap_map = (rt_material_t *)Mem_Alloc (RT_MAT_CAP_MAP * sizeof (rt_material_t));
	memcpy (qre.snap_map, RT_MAT_GetList (RT_MAT_LIST_MAP, NULL), (size_t)count * sizeof (rt_material_t));
}

static void QRE_RestoreSnapshot (void)
{
	memcpy (RT_MAT_GetList (RT_MAT_LIST_GLOBAL, NULL), qre.snap_global, (size_t)qre.snap_global_count * sizeof (rt_material_t));
	memcpy (RT_MAT_GetList (RT_MAT_LIST_MAP, NULL), qre.snap_map, (size_t)qre.snap_map_count * sizeof (rt_material_t));

	// The lists may have grown since the snapshot (a material the editor
	// created); the lengths are part of what a snapshot restores.
	RT_MAT_SetListCounts (qre.snap_global_count, qre.snap_map_count);
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
	static qboolean warned = false;

	if (!m || !m->name[0])
		return;

	if (!QRE_NameInList (qre.touched, qre.touched_count, m->name))
	{
		if (qre.touched_count < QRE_TOUCHED_MAX)
			q_strlcpy (qre.touched[qre.touched_count++], m->name, MAX_QPATH);
		else if (!warned)
		{
			warned = true;
			QRE_Notify ("too many materials edited at once; some will not be re-applied");
		}
	}

	if (!QRE_NameInList (qre.dirty, qre.dirty_count, m->name))
	{
		if (qre.dirty_count < QRE_DIRTY_MAX)
			q_strlcpy (qre.dirty[qre.dirty_count++], m->name, MAX_QPATH);
		else
		{
			warned = true;
			QRE_Notify ("too many materials edited at once; some will not be previewed");
		}
	}
}

// Re-synthesizes every dirty material and asks the renderer to re-upload the
// static world: the material handles changed, and the world bakes their texture
// indices at upload time. R_DrawWorldTask then re-uploads the world (and
// re-collects its emissive lights) on the next frame.
static void QRE_FlushDirty (void)
{
	static double last_flush = 0.0;
	double        now = Sys_DoubleTime ();
	int           i;

	if (qre.dirty_count == 0)
		return;

	// While a widget is being dragged the re-synthesis (and the world upload
	// behind it) runs per frame; throttle it then, and apply at once when the
	// drag is over.
	if (QR_GUI_Ready () && QR_GUI_AnyItemActive () && (now - last_flush) < 0.25)
		return;

	last_flush = now;

	for (i = 0; i < qre.dirty_count; i++)
		TexMgr_ReloadImagesForMaterial (qre.dirty[i]);
	qre.dirty_count = 0;

	Atomic_StoreUInt32 (&rt_require_static_submit, true);
}

static void QRE_ReapplyTouched (void)
{
	int i;

	qre.dirty_count = 0;
	for (i = 0; i < qre.touched_count; i++)
	{
		// by texture name: Cancel/Exit drop a material the editor had created,
		// and the texture still has to be re-synthesized without it
		TexMgr_ReloadImagesForTextureName (qre.touched[i]);
	}
	qre.touched_count = 0;

	Atomic_StoreUInt32 (&rt_require_static_submit, true);
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
	else
	{
		QRE_Notify ("cannot create a material: the list is full");
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

// World -> model space for a rigid brush transform (rotation R, translation t):
// local = R^T * (world - t).
static void QRE_WorldToModel (const RgTransform *transform, const vec3_t world, vec3_t out)
{
	vec3_t d;
	int    i;

	for (i = 0; i < 3; i++)
		d[i] = world[i] - transform->matrix[i][3];

	for (i = 0; i < 3; i++)
		out[i] = transform->matrix[0][i] * d[0] + transform->matrix[1][i] * d[1] + transform->matrix[2][i] * d[2];
}

// Finds the surface of a model whose face contains the impact point. Coplanar
// faces are told apart by a polygon test; the nearest plane is the fallback.
// The impact is world space, while a brush entity's planes and polygons are in
// its model space, so the point is transformed first (identity for the world).
static msurface_t *QRE_FindSurface (qmodel_t *model, const vec3_t impact, const RgTransform *transform)
{
	vec3_t      local;
	int         i;
	msurface_t *best = NULL;
	float       bestd = 1.0f;

	QRE_WorldToModel (transform, impact, local);

	for (i = 0; i < model->nummodelsurfaces; i++)
	{
		msurface_t *s = &model->surfaces[model->firstmodelsurface + i];
		glpoly_t   *p;
		float       d;

		if (!s->texinfo || !s->texinfo->texture)
			continue;
		if (s->flags & (SURF_DRAWSKY | SURF_NOTEXTURE))
			continue;

		d = DotProduct (s->plane->normal, local) - s->plane->dist;
		if (fabsf (d) > 1.0f)
			continue;

		for (p = s->polys; p; p = p->next)
		{
			if (QRE_PointInPolygon (p, local, s->plane))
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

static qboolean QRE_TracePick (qmodel_t **out_model, msurface_t **out_surf, entity_t **out_ent, gltexture_t **out_glt)
{
	vec3_t    start, end;
	trace_t   tr;
	float     best = 1.0f;
	qmodel_t *bestmodel = NULL;
	entity_t *bestent = NULL;
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

	// brush entities (health boxes, buttons, doors, ...); their traces are in
	// world space, their model data is not (RT_GetBrushModelMatrix below)
	{
		vec3_t eimpact, enorm;
		int    entnum;
		float  f = CL_TraceLine (start, end, eimpact, enorm, &entnum);

		if (f < best && entnum >= 0 && entnum < MAX_EDICTS &&
		    cl.entities[entnum].model && cl.entities[entnum].model != cl.worldmodel)
		{
			best = f;
			bestent = &cl.entities[entnum];
			bestmodel = bestent->model;
			VectorCopy (eimpact, bestimpact);
		}
	}

	if (!bestmodel)
		return false;

	{
		RgTransform transform = RT_GetBrushModelMatrix (bestent);

		*out_surf = QRE_FindSurface (bestmodel, bestimpact, &transform);
	}
	if (!*out_surf)
		return false;

	*out_model = bestmodel;
	if (out_ent)
		*out_ent = bestent;
	if (out_glt)
		*out_glt = (*out_surf)->texinfo->texture->gltexture;
	return true;
}

// Hover pick (crosshair, flying) or select pick (fire button).
static void QRE_DoPick (qboolean select)
{
	qmodel_t    *model;
	msurface_t  *surf;
	entity_t    *ent;
	gltexture_t *glt;

	if (!QRE_TracePick (&model, &surf, &ent, &glt))
	{
		qre.hover_model = NULL;
		qre.hover_surf = NULL;
		qre.hover_ent = NULL;
		return;
	}

	if (!select)
	{
		qre.hover_model = model;
		qre.hover_surf = surf;
		qre.hover_ent = ent;
		return;
	}

	if (!glt)
		return;
	if (!QR_GUI_Ready ())
	{
		QRE_Notify ("the ImGui panel is not available");
		return;
	}

	qre.pick_model = model;
	qre.pick_surf = surf;
	qre.pick_ent = ent;
	qre.hover_model = NULL;
	qre.hover_surf = NULL;
	qre.hover_ent = NULL;

	{
		char texname[MAX_QPATH];
		char *dot;
		int   i;

		RT_MAT_NormalizeName (glt->name, texname, sizeof (texname));
		dot = strrchr (texname, '.');
		if (dot && !strchr (dot, ':'))
			*dot = '\0';

		QRE_ResolveGroup (texname);

		Con_Printf ("qr editor: picked '%s' (%d material(s) in the group)\n", texname, qre.group_count);
		for (i = 0; i < qre.group_count; i++)
			Con_Printf ("qr editor:   group material '%s'\n", qre.group[i]->name);
	}

	qre.panel_open = true;

	// the panel owns the mouse: free the cursor (keeping its motion events
	// for ImGui), freeze the camera
	IN_FreeCursorForGui ();
	SDL_ShowCursor (SDL_DISABLE);
	QR_GUI_SetMouseCursor (1);
}

// ---------------------------------------------------------------------------
// Selection outline
// ---------------------------------------------------------------------------

static void QRE_EmitOutline (qmodel_t *model, msurface_t *surf, entity_t *ent, uint32_t color)
{
	RgTransform transform = RT_GetBrushModelMatrix (ent);
	vec3_t     *verts;
	RgVertex   *rv;
	uint32_t   *ri;
	byte       *block;
	size_t      verts_bytes, rv_bytes, ri_bytes;
	int         n = 0;
	int         i, j;
	vec3_t      n_world, to_view, first;
	float       nudge = 0.35f;

	// The face outline from the BSP edge list, or the polygon when the face
	// has no edges (should not happen for regular faces).
	if (surf->numedges > 0)
		n = surf->numedges;
	else if (surf->polys && surf->polys->numverts > 0)
		n = surf->polys->numverts;

	if (n < 3)
		return;

	// One scratch allocation for all three arrays: the allocator hands out a
	// single shared buffer, so overlapping allocations would zero or free each
	// other's memory.
	verts_bytes = (size_t)n * sizeof (vec3_t);
	rv_bytes    = (size_t)n * sizeof (RgVertex);
	ri_bytes    = (size_t)n * 2 * sizeof (uint32_t);

	block = (byte *)RT_AllocScratchMemoryNulled (verts_bytes + rv_bytes + ri_bytes);
	verts = (vec3_t *)block;
	rv    = (RgVertex *)(block + verts_bytes);
	ri    = (uint32_t *)(block + verts_bytes + rv_bytes);

	if (surf->numedges > 0)
	{
		for (i = 0; i < n; i++)
		{
			int e = model->surfedges[surf->firstedge + i];
			if (e >= 0)
				VectorCopy (model->vertexes[model->edges[e].v[0]].position, verts[i]);
			else
				VectorCopy (model->vertexes[model->edges[-e].v[1]].position, verts[i]);
		}
	}
	else
	{
		glpoly_t *p = surf->polys;

		for (i = 0; i < n; i++)
			VectorCopy (p->verts[i], verts[i]);
	}

	// The face plane in world space: n_world = R * n (identity for the world).
	for (j = 0; j < 3; j++)
		n_world[j] = transform.matrix[0][j] * surf->plane->normal[0]
		           + transform.matrix[1][j] * surf->plane->normal[1]
		           + transform.matrix[2][j] * surf->plane->normal[2];

	// Nudge the outline off the face towards the viewer: the plane normal of a
	// SURF_PLANEBACK face points away from its visible side, and pushing the
	// line behind the wall would lose it to the traced surface.
	for (j = 0; j < 3; j++)
		first[j] = transform.matrix[0][j] * verts[0][0]
		         + transform.matrix[1][j] * verts[0][1]
		         + transform.matrix[2][j] * verts[0][2]
		         + transform.matrix[j][3];
	VectorSubtract (r_origin, first, to_view);
	if (DotProduct (to_view, n_world) < 0.0f)
		nudge = -nudge;

	// Model -> world (a brush entity carries its own transform), then off the
	// face; the vertices are uploaded in world space.
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < 3; j++)
			rv[i].position[j] = transform.matrix[0][j] * verts[i][0]
			                  + transform.matrix[1][j] * verts[i][1]
			                  + transform.matrix[2][j] * verts[i][2]
			                  + transform.matrix[j][3]
			                  + nudge * n_world[j];
		rv[i].packedColor = color;
	}

	for (i = 0; i < n; i++)
	{
		ri[i * 2 + 0] = (uint32_t)i;
		ri[i * 2 + 1] = (uint32_t)((i + 1) % n);
	}

	// The swapchain render type is the overlay path the engine's own 2D and the
	// ImGui panel use: the lines go over the finished frame, projected with the
	// frame's camera matrices (NULL view projection), and there is no depth
	// buffer to lose them to.
	RgRasterizedGeometryUploadInfo info = {
		.renderType = RG_RASTERIZED_GEOMETRY_RENDER_TYPE_SWAPCHAIN,
		.vertexCount = (uint32_t)n,
		.pVertices = rv,
		.indexCount = (uint32_t)(n * 2),
		.pIndices = ri,
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

void QR_Editor_DrawSelection (cb_context_t *cbx)
{
	(void)cbx;

	if (!qre.active)
		return;

	if (qre.panel_open && qre.pick_surf)
	{
		QRE_EmitOutline (qre.pick_model, qre.pick_surf, qre.pick_ent, RT_PackColorToUint32 (255, 255, 255, 255));
	}
	else if (!qre.panel_open && qre.hover_surf)
	{
		QRE_EmitOutline (qre.hover_model, qre.hover_surf, qre.hover_ent, RT_PackColorToUint32 (255, 214, 64, 255));
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

	// Before the frame renders: a re-synthesis replaces the material handles,
	// and the world's static upload (R_DrawWorldTask) runs later in this frame,
	// so the re-upload the flush asks for happens in the same frame.
	QRE_FlushDirty ();
}

// ---------------------------------------------------------------------------
// Panel (Dear ImGui)
// ---------------------------------------------------------------------------

static void QRE_ParamWidgets (int g)
{
	int p;

	for (p = 0; p < PARAM_COUNT; p++)
	{
		const char   *label = qre_params[p].label;
		// re-read every iteration: the first change of a material that has no
		// yaml entry moves qre.group[g] into the live list (QRE_EnsureLive)
		rt_material_t *m = qre.group[g];

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
				// RT_MAT_KIND_REGULAR is 1, so the combo index is kind - 1
				int index = value - 1;
				if (index < 0 || index >= (int)countof (kinds))
					index = 0;
				if (QR_GUI_Combo (label, &index, kinds, (int)countof (kinds)))
					QRE_SetInt (g, p, index + 1);
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

			if (p == PARAM_CEMIS && m->filename_emissive[0])
				QR_GUI_Tooltip ("color_emissive is ignored while texture_emissive is set");
			break;
		}
		default:
			break;
		}

		// materials.yaml defines these, but no renderer code reads them: the panel
		// edits and saves them, so say what they are worth.
		if (p == PARAM_KIND || p == PARAM_MASK || p == PARAM_BSPRAD || p == PARAM_DRAD)
			QR_GUI_Tooltip ("not read by the renderer (stored in materials.yaml only)");
	}
}

static void QRE_BuildPanelGUI (void)
{
	int      panel_w = glwidth * 11 / 25; // two fifths, plus a tenth
	int      g;
	qboolean exit_requested = false;

	if (panel_w > 506)
		panel_w = 506;
	if (panel_w < 352)
		panel_w = 352;

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
		// the animation frames share their parameter names, so every section
		// needs its own ID scope, or their widgets collide
		QR_GUI_PushID (qre.group[g]->name);
		if (QR_GUI_Section (qre.group[g]->name, 1))
			QRE_ParamWidgets (g);
		QR_GUI_PopID ();
	}

	QR_GUI_EndScroll ();
	QR_GUI_EndPanel ();

	if (exit_requested)
		QRE_StopEditor (true);
}

static void QRE_BuildFlyingOverlay (void)
{
	static const char *const lines[] = {
		"QR LIGHT EDITOR",
		"LMB - select the face under the crosshair",
		"WASD + mouse - fly    Shift - faster    jump/movedown - up/down",
		"Esc - exit the editor    ~ - console",
	};

	QR_GUI_DrawHint (lines, (int)countof (lines));
	QR_GUI_DrawCrosshair ();
}

// ---------------------------------------------------------------------------
// Panel: per-frame bookkeeping
// ---------------------------------------------------------------------------

static void QRE_Frame (void)
{
	static keydest_t prev_key_dest = key_game;

	// the level went away under the editor: drop it (the material snapshot may
	// be stale relative to a freshly loaded map list, so nothing is restored)
	if (cls.state != ca_connected || !cl.worldmodel)
	{
		QRE_StopEditor (false);
		return;
	}

	// While the console is up it owns the input; coming back, the panel needs
	// its free cursor again (the console re-activated the relative mouse mode).
	if (qre.panel_open && prev_key_dest != key_game && key_dest == key_game)
	{
		IN_FreeCursorForGui ();
		SDL_ShowCursor (SDL_DISABLE);
	}
	prev_key_dest = key_dest;

	if (!qre.panel_open)
		QRE_DoPick (false);

	// the normal path flushes before the render (QR_Editor_UpdateView); this
	// covers the frames in which V_CalcRefdef does not run (paused, intermission)
	QRE_FlushDirty ();
}

void QR_Editor_DrawPanel (cb_context_t *cbx)
{
	(void)cbx;

	if (!qre.active)
		return;

	QRE_Frame ();

	// while the console is up it owns the screen; the editor state is kept
	if (key_dest != key_game)
		return;

	// The whole editor interface is ImGui: the panel, the crosshair and the
	// hints. SCR_UpdateScreen can run more than once per host frame.
	if (!QR_GUI_BeginFrame ((unsigned int)host_framecount, (float)host_frametime, glx, gly, glwidth, glheight, vid.height))
		return;

	if (qre.panel_open)
		QRE_BuildPanelGUI ();
	else
		QRE_BuildFlyingOverlay ();

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

	// while the console is up the engine owns the input
	if (key_dest != key_game)
		return false;

	// the console toggle key stays available unless an ImGui text field is
	// editing: the console is the way the editor is driven too
	if ((e->type == SDL_KEYDOWN || e->type == SDL_KEYUP) &&
	    e->key.keysym.scancode == SDL_SCANCODE_GRAVE && !QR_GUI_WantsKeyboard ())
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
	qre.pick_ent = NULL;

	IN_Activate ();
	SDL_ShowCursor (SDL_ENABLE);
	QR_GUI_SetMouseCursor (0);
}

static void QRE_Apply (void)
{
	QRE_SaveMaterials ();
	QRE_TakeSnapshot (); // Cancel now reverts to the state just saved
	qre.touched_count = 0;
	QRE_Notify ("materials written to materials/");
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

	QRE_Notify ("materials reverted to the values from materials.yaml");
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
	"#     surface only glows):\n"
	"#     e.g.  - name: textures/foo\n"
	"#             color_emissive: ff0000\n"
	"#             is_light: true\n"
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
	"# them bright and saturated.\n"
	"#     e.g.  - name: textures/window01_1\n"
	"#             mirror: true\n"
	"#             emissive_blend: 2\n";

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

#define QRE_SAVE_FILES_MAX   32

static const char *QRE_MaterialFile (const rt_material_t *m)
{
	return m->source_file[0] ? m->source_file : "materials/materials.yaml";
}

static qboolean QRE_SameFile (const char *a, const char *b)
{
	// the directory scan and the map name can differ in case on a
	// case-insensitive filesystem; both name the same file
	return !q_strcasecmp (a, b);
}

static qboolean QRE_FileListed (char files[][MAX_QPATH], int nfiles, const char *file)
{
	int i;

	for (i = 0; i < nfiles; i++)
	{
		if (QRE_SameFile (files[i], file))
			return true;
	}
	return false;
}

// A material of the file that the global list does not carry (its map copy is
// then the only one that holds the edit).
static qboolean QRE_NameInGlobalFile (const rt_material_t *list, int count, const char *file, const char *name)
{
	int i;

	for (i = 0; i < count; i++)
	{
		if (list[i].valid && !strcmp (list[i].name, name) && QRE_SameFile (QRE_MaterialFile (&list[i]), file))
			return true;
	}
	return false;
}

// A safety copy of a materials file before the editor rewrites it.
static void QRE_BackupFile (const char *path)
{
	char   bak[MAX_OSPATH];
	FILE  *in, *out;
	char   buf[4096];
	size_t n;

	q_snprintf (bak, sizeof (bak), "%s.bak", path);

	in = fopen (path, "rb");
	if (!in)
		return;

	out = fopen (bak, "wb");
	if (!out)
	{
		fclose (in);
		return;
	}

	while ((n = fread (buf, 1, sizeof (buf), in)) > 0)
		fwrite (buf, 1, n, out);

	fclose (in);
	fclose (out);
}

static void QRE_SaveMaterials (void)
{
	char           files[QRE_SAVE_FILES_MAX][MAX_QPATH];
	int            nfiles = 0;
	rt_material_t *maplist, *glist;
	int            mapcount, gcount;
	int            i, k, f;
	int            written_total = 0;

	maplist = RT_MAT_GetList (RT_MAT_LIST_MAP, &mapcount);
	glist = RT_MAT_GetList (RT_MAT_LIST_GLOBAL, &gcount);

	/* The map materials also live in the global list (the directory scan reads
	   materials/<map>.yaml as a global file), and RT_MAT_Find returns the map
	   copy first, so an edit may sit in the map copy: sync it back. */
	for (i = 0; i < mapcount; i++)
	{
		if (!maplist[i].valid)
			continue;
		for (k = 0; k < gcount; k++)
		{
			if (glist[k].valid && !strcmp (glist[k].name, maplist[i].name) &&
			    QRE_SameFile (QRE_MaterialFile (&glist[k]), QRE_MaterialFile (&maplist[i])))
			{
				glist[k] = maplist[i];
				break;
			}
		}
	}

	/* every distinct source file, of both lists */
	for (k = 0; k < 2; k++)
	{
		rt_material_t *list = (k == 0) ? glist : maplist;
		int            count = (k == 0) ? gcount : mapcount;

		for (i = 0; i < count; i++)
		{
			const char *file;

			if (!list[i].valid)
				continue;
			file = QRE_MaterialFile (&list[i]);
			if (QRE_FileListed (files, nfiles, file))
				continue;
			if (nfiles >= QRE_SAVE_FILES_MAX)
			{
				/* refusing beats writing a file with entries dropped */
				QRE_Notify ("too many materials/*.yaml files; nothing written");
				return;
			}
			q_strlcpy (files[nfiles++], file, MAX_QPATH);
		}
	}

	/* Only files that hold an edited material are rewritten: a file the user did
	   not touch keeps its comments, formatting and keys the loader ignores. */
	{
		int kept = 0;

		for (f = 0; f < nfiles; f++)
		{
			qboolean needed = false;

			for (i = 0; i < gcount && !needed; i++)
			{
				if (glist[i].valid && QRE_SameFile (QRE_MaterialFile (&glist[i]), files[f]) &&
				    QRE_NameInList (qre.touched, qre.touched_count, glist[i].name))
					needed = true;
			}
			for (i = 0; i < mapcount && !needed; i++)
			{
				if (maplist[i].valid && QRE_SameFile (QRE_MaterialFile (&maplist[i]), files[f]) &&
				    QRE_NameInList (qre.touched, qre.touched_count, maplist[i].name))
					needed = true;
			}

			if (needed)
			{
				if (kept != f)
					q_strlcpy (files[kept], files[f], MAX_QPATH);
				kept++;
			}
		}

		nfiles = kept;
	}

	if (nfiles == 0)
	{
		QRE_Notify ("nothing to save");
		return;
	}

	for (f = 0; f < nfiles; f++)
	{
		char  path[MAX_OSPATH];
		FILE *file;
		int   written = 0;

		q_snprintf (path, sizeof (path), "%s/%s", com_gamedir, files[f]);

		QRE_BackupFile (path);

		file = fopen (path, "w");
		if (!file)
		{
			QRE_Notify ("cannot write %s", files[f]);
			continue;
		}

		if (QRE_SameFile (files[f], "materials/materials.yaml"))
			fprintf (file, "%s", qre_yaml_header);

		fprintf (file, "materials:\n");

		/* the global list first; entries of a map file that it does not carry
		   (should not happen, but the editor must not lose data) follow */
		for (i = 0; i < gcount; i++)
		{
			if (glist[i].valid && QRE_SameFile (QRE_MaterialFile (&glist[i]), files[f]))
			{
				QRE_WriteMaterial (file, &glist[i]);
				written++;
			}
		}
		for (i = 0; i < mapcount; i++)
		{
			if (maplist[i].valid && QRE_SameFile (QRE_MaterialFile (&maplist[i]), files[f]) &&
			    !QRE_NameInGlobalFile (glist, gcount, files[f], maplist[i].name))
			{
				QRE_WriteMaterial (file, &maplist[i]);
				written++;
			}
		}

		if (ferror (file) || fflush (file) != 0)
		{
			QRE_Notify ("write error in %s", files[f]);
		}
		fclose (file);

		written_total += written;
	}

	Con_Printf ("qr editor: wrote %d materials to %d file(s)\n", written_total, nfiles);
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
	ofn.lpstrFilter = "Images (*.png;*.tga;*.jpg;*.jpeg)\0*.png;*.tga;*.jpg;*.jpeg\0All files (*.*)\0*.*\0";
	ofn.lpstrFile = result;
	ofn.nMaxFile = sizeof (result);
	ofn.lpstrInitialDir = initdir;
	ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

	if (!GetOpenFileNameA (&ofn))
		return false;

	// The dialog returns backslashes while com_gamedir may carry forward
	// slashes: normalize before comparing, and store forward slashes.
	for (i = 0; result[i]; i++)
	{
		if (result[i] == '\\')
			result[i] = '/';
	}

	glen = strlen (com_gamedir);
	if (!q_strncasecmp (result, com_gamedir, glen) && result[glen] == '/')
	{
		q_strlcpy (out, result + glen + 1, outsize);
	}
	else
	{
		// outside the gamedir the material loader cannot find the file
		QRE_Notify ("the texture must be inside %s", com_gamedir);
		return false;
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
	qre.pick_ent = NULL;
	qre.hover_model = NULL;
	qre.hover_surf = NULL;
	qre.hover_ent = NULL;

	VectorCopy (qre.player_viewangles, cl.viewangles);

	QRE_FreeSnapshot ();

	IN_Activate ();
	SDL_ShowCursor (SDL_ENABLE);
	QR_GUI_SetMouseCursor (0);

	QRE_Notify ("editor closed");
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

void QR_Editor_OnNewMap (void)
{
	if (qre.active)
	{
		Con_Printf ("qr light editor: closed by a map change\n");
		QRE_StopEditor (false);
	}
}

void QR_Editor_Shutdown (void)
{
	QR_GUI_Shutdown ();
}
