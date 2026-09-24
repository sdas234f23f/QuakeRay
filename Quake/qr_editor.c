// qr_editor.c -- qr light editor: realtime material editor for the vkpt renderer.
//
// Console commands: qr_light_editor_start / qr_light_editor_stop.
//
// While the editor runs the view belongs to a free camera (the player stands
// still): aim with the crosshair, fire selects the face under it and opens the
// material panel on the right edge of the screen. The panel edits the
// materials.yaml parameters of every animation frame of the picked texture
// (medkits, blinking buttons, ...). Apply writes the changes back to the
// materials/*.yaml files, Cancel reverts to the values they were loaded from,
// Exit closes the editor and returns the view to the player.
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

#include "SDL.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>
#endif

#include "qr_editor.h"

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
// UI layout (CANVAS_EDITOR units: 640 across the screen, 8x8 glyphs)
// ---------------------------------------------------------------------------

#define QRE_PANEL_X      440   // panel left edge
#define QRE_PANEL_W      200
#define QRE_LABEL_X      444
#define QRE_WIDGET_X     522
#define QRE_WIDGET_MAX   634
#define QRE_ROW_H        11
#define QRE_TEXT_ROW_H   21
#define QRE_HEADER_H     12
#define QRE_CONTENT_TOP  40
#define QRE_SCROLL_X     634
#define QRE_SCROLL_W     6

#define QRE_GROUP_MAX    12
#define QRE_ROWS_MAX     640
#define QRE_DIRTY_MAX    16
#define QRE_TOUCHED_MAX  128

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

enum
{
	ROW_HEADER,
	ROW_FLOAT,
	ROW_INT,
	ROW_BOOL,
	ROW_TEXT,
	ROW_COLOR,
	ROW_CR,   // color editor: red slider
	ROW_CG,   // color editor: green slider
	ROW_CB,   // color editor: blue slider
	ROW_CHEX, // color editor: hex text field
};

typedef struct qre_row_s
{
	int kind;
	int mat;   // index into qre.group
	int param; // PARAM_*
	int sub;   // color rows: 0..2 = R/G/B, 3 = hex
} qre_row_t;

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

	// panel state
	float      scroll;
	int        rows_count;
	qre_row_t  rows[QRE_ROWS_MAX];
	int        active_text_row; // row being typed into, -1 = none
	char       text_buf[MAX_QPATH];
	int        drag_slider_row; // row being slider-dragged, -1 = none
	int        color_expand_mat;
	int        color_expand_param;
	qboolean   drag_scroll;
	float      drag_scroll_grab;
	float      mouse_x, mouse_y; // canvas coords
} qre;

// forward declarations (the panel click handler sits above the apply/save code)
static void    QRE_Apply (void);
static void    QRE_Cancel (void);
static void    QRE_StopEditor (qboolean restore);
static void    QRE_SaveMaterials (void);
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

static void QRE_Truncate (char *out, size_t outsize, const char *in)
{
	if (strlen (in) < outsize)
	{
		q_strlcpy (out, in, outsize);
		return;
	}
	memcpy (out, in, outsize - 1);
	out[outsize - 1] = '\0';
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

// Formats the value of a row for display and for the text field.
static void QRE_FormatValue (const rt_material_t *m, const qre_row_t *row, char *buf, size_t bufsize)
{
	switch (row->kind)
	{
	case ROW_FLOAT:
		q_snprintf (buf, bufsize, "%.4g", QRE_GetFloat (m, row->param));
		break;
	case ROW_INT:
		if (row->param == PARAM_KIND)
		{
			const char *k = RT_MAT_KindName (QRE_GetInt (m, row->param));
			q_snprintf (buf, bufsize, "%s", k ? k : "REGULAR");
		}
		else
			q_snprintf (buf, bufsize, "%d", QRE_GetInt (m, row->param));
		break;
	case ROW_BOOL:
		q_snprintf (buf, bufsize, "%s", QRE_GetBool (m, row->param) ? "on" : "off");
		break;
	case ROW_TEXT:
		QRE_Truncate (buf, bufsize, QRE_GetText (m, row->param));
		break;
	case ROW_COLOR:
	case ROW_CHEX:
	{
		qboolean en;
		float    rgb[3];

		QRE_GetColor (m, row->param, &en, rgb);
		if (!en)
			q_snprintf (buf, bufsize, "------");
		else
			q_snprintf (buf, bufsize, "%02x%02x%02x",
			            (int)(rgb[0] * 255.0f + 0.5f), (int)(rgb[1] * 255.0f + 0.5f), (int)(rgb[2] * 255.0f + 0.5f));
		break;
	}
	case ROW_CR:
	case ROW_CG:
	case ROW_CB:
	{
		qboolean en;
		float    rgb[3];

		QRE_GetColor (m, row->param, &en, rgb);
		q_snprintf (buf, bufsize, "%d", (int)(rgb[row->sub] * 255.0f + 0.5f));
		break;
	}
	default:
		buf[0] = '\0';
		break;
	}
}

static qboolean QRE_ParseHexColor (const char *text, float rgb[3])
{
	int v[6];
	int i;

	if (strlen (text) != 6)
		return false;
	for (i = 0; i < 6; i++)
	{
		char c = text[i];

		if (c >= '0' && c <= '9')       v[i] = c - '0';
		else if (c >= 'a' && c <= 'f')  v[i] = c - 'a' + 10;
		else if (c >= 'A' && c <= 'F')  v[i] = c - 'A' + 10;
		else return false;
	}
	rgb[0] = (float)(v[0] * 16 + v[1]) / 255.0f;
	rgb[1] = (float)(v[2] * 16 + v[3]) / 255.0f;
	rgb[2] = (float)(v[4] * 16 + v[5]) / 255.0f;
	return true;
}

// Commits the text field content to its parameter.
static void QRE_CommitText (void)
{
	const qre_row_t *row;
	rt_material_t   *m;
	qboolean         ok = true;

	if (qre.active_text_row < 0 || qre.active_text_row >= qre.rows_count)
		return;
	row = &qre.rows[qre.active_text_row];
	if (row->mat < 0 || row->mat >= qre.group_count)
		return;
	m = qre.group[row->mat];

	switch (row->kind)
	{
	case ROW_FLOAT:
		QRE_SetFloat (row->mat, row->param, (float)atof (qre.text_buf));
		break;
	case ROW_INT:
		QRE_SetInt (row->mat, row->param, atoi (qre.text_buf));
		break;
	case ROW_BOOL:
		QRE_SetBool (row->mat, row->param, qre.text_buf[0] == '1' || !q_strcasecmp (qre.text_buf, "on") || !q_strcasecmp (qre.text_buf, "true"));
		break;
	case ROW_TEXT:
		QRE_SetText (row->mat, row->param, qre.text_buf);
		break;
	case ROW_COLOR:
	case ROW_CHEX:
	{
		float rgb[3];

		if (QRE_ParseHexColor (qre.text_buf, rgb))
		{
			QRE_EnsureLive (row->mat);
			if (row->param == PARAM_CEMIS)
			{
				m->has_color_emissive = true;
				VectorCopy (rgb, m->color_emissive);
			}
			else
			{
				m->has_light_color = true;
				VectorCopy (rgb, m->light_color);
			}
			QRE_MarkDirty (m);
		}
		else
		{
			Con_Printf ("qr editor: '%s' is not a valid RRGGBB color\n", qre.text_buf);
			ok = false;
		}
		break;
	}
	default:
		ok = false;
		break;
	}

	if (ok)
		qre.active_text_row = -1;
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

	qre.pick_model = model;
	qre.pick_surf = surf;
	qre.hover_model = NULL;
	qre.hover_surf = NULL;

	if (glt)
	{
		char texname[MAX_QPATH];

		RT_MAT_NormalizeName (glt->name, texname, sizeof (texname));
		{
			char *dot = strrchr (texname, '.');
			if (dot && !strchr (dot, ':'))
				*dot = '\0';
		}

		QRE_ResolveGroup (texname);
		qre.panel_open = true;
		qre.scroll = 0;
		qre.active_text_row = -1;
		qre.drag_slider_row = -1;
		qre.color_expand_mat = -1;
		qre.color_expand_param = -1;
		qre.drag_scroll = false;

		// the panel owns the mouse: free the cursor, freeze the camera
		IN_Deactivate (true);
		SDL_ShowCursor (SDL_DISABLE);
	}
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
// Panel: row list
// ---------------------------------------------------------------------------

static float QRE_RowHeight (const qre_row_t *row)
{
	if (row->kind == ROW_TEXT)
		return QRE_TEXT_ROW_H;
	if (row->kind == ROW_HEADER)
		return QRE_HEADER_H;
	return QRE_ROW_H;
}

static int QRE_TypeToRow (int type)
{
	switch (type)
	{
	case QRE_T_FLOAT: return ROW_FLOAT;
	case QRE_T_INT:   return ROW_INT;
	case QRE_T_BOOL:  return ROW_BOOL;
	case QRE_T_TEXT:  return ROW_TEXT;
	case QRE_T_COLOR: return ROW_COLOR;
	default:          return ROW_HEADER;
	}
}

static void QRE_BuildRows (void)
{
	int r = 0;
	int g;

	for (g = 0; g < qre.group_count && r < QRE_ROWS_MAX; g++)
	{
		int p;

		qre.rows[r].kind = ROW_HEADER;
		qre.rows[r].mat = g;
		qre.rows[r].param = -1;
		qre.rows[r].sub = -1;
		r++;

		for (p = 0; p < PARAM_COUNT && r < QRE_ROWS_MAX; p++)
		{
			qre.rows[r].kind = QRE_TypeToRow (qre_params[p].type);
			qre.rows[r].mat = g;
			qre.rows[r].param = p;
			qre.rows[r].sub = -1;
			r++;

			if (qre_params[p].type == QRE_T_COLOR &&
			    qre.color_expand_mat == g && qre.color_expand_param == p && r + 4 <= QRE_ROWS_MAX)
			{
				int c;

				for (c = 0; c < 3; c++)
				{
					qre.rows[r].kind = ROW_CR + c;
					qre.rows[r].mat = g;
					qre.rows[r].param = p;
					qre.rows[r].sub = c;
					r++;
				}
				qre.rows[r].kind = ROW_CHEX;
				qre.rows[r].mat = g;
				qre.rows[r].param = p;
				qre.rows[r].sub = 3;
				r++;
			}
		}
	}

	qre.rows_count = r;
}

static float QRE_ContentHeight (void)
{
	float h = 0;
	int   i;

	for (i = 0; i < qre.rows_count; i++)
		h += QRE_RowHeight (&qre.rows[i]);
	return h;
}

// The row under a canvas y coordinate, or -1.
static int QRE_RowAt (float y)
{
	float cy = QRE_CONTENT_TOP - qre.scroll;
	int   i;

	for (i = 0; i < qre.rows_count; i++)
	{
		float h = QRE_RowHeight (&qre.rows[i]);
		if (y >= cy && y < cy + h)
			return i;
		cy += h;
	}
	return -1;
}

// ---------------------------------------------------------------------------
// Panel: drawing
// ---------------------------------------------------------------------------

static void QRE_DrawButton (cb_context_t *cbx, int x, int y, int w, int h, const char *text)
{
	const float bg[4] = { 0.16f, 0.18f, 0.24f, 1.0f };
	const float border[4] = { 0.4f, 0.42f, 0.5f, 1.0f };
	const RgFloat4D textc = QRE_Rgba (1, 1, 1, 1);

	Draw_FillRGBA (cbx, x, y, w, h, bg);
	Draw_FillRGBA (cbx, x, y, w, 1, border);
	Draw_FillRGBA (cbx, x, y + h - 1, w, 1, border);
	Draw_FillRGBA (cbx, x, y, 1, h, border);
	Draw_FillRGBA (cbx, x + w - 1, y, 1, h, border);
	Draw_StringScaled (cbx, x + (int)((w - 8.0f * strlen (text)) / 2.0f), y + 2, text, 1.0f, &textc);
}

// value is already in slider units (param value for floats/ints, 0..255 for RGB)
static void QRE_DrawSlider (cb_context_t *cbx, const qre_row_t *row, float x, float y, float trackw, float value)
{
	const float track[4] = { 0.22f, 0.23f, 0.3f, 1.0f };
	const float fill[4] = { 0.5f, 0.55f, 0.65f, 1.0f };
	const float knob[4] = { 0.95f, 0.95f, 1.0f, 1.0f };
	float       f;

	if (row->kind == ROW_CR || row->kind == ROW_CG || row->kind == ROW_CB)
		f = value / 255.0f;
	else
		f = (value - qre_params[row->param].min) / (qre_params[row->param].max - qre_params[row->param].min);
	f = CLAMP (0.0f, f, 1.0f);

	Draw_FillRGBA (cbx, (int)x, (int)(y + 4), (int)trackw, 3, track);
	if (f > 0.001f)
		Draw_FillRGBA (cbx, (int)x, (int)(y + 4), (int)(trackw * f), 3, fill);
	Draw_FillRGBA (cbx, (int)(x + trackw * f - 1), (int)(y + 2), 2, 7, knob);
}

static void QRE_DrawCursor (cb_context_t *cbx)
{
	const float c[4] = { 1.0f, 1.0f, 1.0f, 0.95f };
	const float d[4] = { 0.0f, 0.0f, 0.0f, 0.7f };
	int x = (int)qre.mouse_x;
	int y = (int)qre.mouse_y;

	Draw_FillRGBA (cbx, x, y - 1, 9, 2, d);
	Draw_FillRGBA (cbx, x - 1, y, 2, 9, d);
	Draw_FillRGBA (cbx, x, y, 7, 1, c);
	Draw_FillRGBA (cbx, x, y, 1, 7, c);
}

static void QRE_DrawPanelUI (cb_context_t *cbx)
{
	const float H = QRE_CanvasHeight ();
	const float bg[4] = { 0.05f, 0.05f, 0.08f, 0.94f };
	const float border[4] = { 0.32f, 0.34f, 0.42f, 1.0f };
	const RgFloat4D titlec = QRE_Rgba (1.0f, 0.85f, 0.35f, 1.0f);
	const RgFloat4D namec = QRE_Rgba (0.85f, 0.9f, 1.0f, 1.0f);
	char name_trunc[30];
	int  i;
	float y, content_h, maxscroll, view_h;

	GL_SetCanvas (cbx, CANVAS_EDITOR);

	Draw_FillRGBA (cbx, QRE_PANEL_X, 0, QRE_PANEL_W, (int)H, bg);
	Draw_FillRGBA (cbx, QRE_PANEL_X, 0, 1, (int)H, border);
	Draw_FillRGBA (cbx, QRE_PANEL_X, (int)H - 1, QRE_PANEL_W, 1, border);

	Draw_StringScaled (cbx, QRE_LABEL_X, 4, "MATERIAL EDITOR", 1.0f, &titlec);
	if (qre.pick_surf && qre.pick_surf->texinfo && qre.pick_surf->texinfo->texture)
	{
		q_snprintf (name_trunc, sizeof (name_trunc), "face: %s", qre.pick_surf->texinfo->texture->name);
		QRE_Truncate (name_trunc, sizeof (name_trunc), name_trunc);
		Draw_StringScaled (cbx, QRE_LABEL_X, 14, name_trunc, 1.0f, &namec);
	}

	QRE_DrawButton (cbx, 444, 24, 48, 12, "Apply");
	QRE_DrawButton (cbx, 498, 24, 48, 12, "Cancel");
	QRE_DrawButton (cbx, 552, 24, 48, 12, "Exit");

	QRE_BuildRows ();

	view_h = H - QRE_CONTENT_TOP - 4.0f;
	content_h = QRE_ContentHeight ();
	maxscroll = content_h > view_h ? content_h - view_h : 0.0f;
	if (qre.scroll < 0.0f)
		qre.scroll = 0.0f;
	if (qre.scroll > maxscroll)
		qre.scroll = maxscroll;

	y = QRE_CONTENT_TOP - qre.scroll;
	for (i = 0; i < qre.rows_count; i++)
	{
		const qre_row_t *row = &qre.rows[i];
		rt_material_t   *m = qre.group[row->mat];
		const float      rh = QRE_RowHeight (row);
		char             val[64];

		if (y + rh <= QRE_CONTENT_TOP || y >= QRE_CONTENT_TOP + view_h)
		{
			y += rh;
			continue;
		}

		switch (row->kind)
		{
		case ROW_HEADER:
		{
			const float hbg[4] = { 0.11f, 0.13f, 0.18f, 1.0f };
			const RgFloat4D hc = QRE_Rgba (0.5f, 0.85f, 1.0f, 1.0f);
			char header[28];

			Draw_FillRGBA (cbx, QRE_PANEL_X, (int)y, QRE_PANEL_W, (int)rh, hbg);
			QRE_Truncate (header, sizeof (header), m->name);
			Draw_StringScaled (cbx, QRE_LABEL_X, (int)(y + 2), header, 1.0f, &hc);
			break;
		}
		case ROW_FLOAT:
		case ROW_INT:
		{
			const RgFloat4D lc = QRE_Rgba (0.75f, 0.78f, 0.85f, 1.0f);
			const RgFloat4D vc = QRE_Rgba (1, 1, 1, 1);
			float value, value_start;

			QRE_FormatValue (m, row, val, sizeof (val));
			Draw_StringScaled (cbx, QRE_LABEL_X, (int)(y + 1), qre_params[row->param].label, 1.0f, &lc);
			value_start = QRE_WIDGET_MAX - 8.0f * strlen (val);
			if (value_start < QRE_WIDGET_X + 40.0f)
				value_start = QRE_WIDGET_X + 40.0f;
			if (i == qre.active_text_row)
			{
				if (qre.text_buf[0])
					Draw_StringScaled (cbx, (int)value_start, (int)(y + 1), qre.text_buf, 1.0f, &vc);
			}
			else
				Draw_StringScaled (cbx, (int)value_start, (int)(y + 1), val, 1.0f, &vc);

			value = (row->kind == ROW_FLOAT) ? QRE_GetFloat (m, row->param) : (float)QRE_GetInt (m, row->param);
			QRE_DrawSlider (cbx, row, QRE_WIDGET_X + 1.0f, y, value_start - 6.0f - (QRE_WIDGET_X + 1.0f), value);
			break;
		}
		case ROW_BOOL:
		{
			const RgFloat4D lc = QRE_Rgba (0.75f, 0.78f, 0.85f, 1.0f);
			const RgFloat4D vc = QRE_Rgba (1, 1, 1, 1);
			const qboolean on = QRE_GetBool (m, row->param);
			char box[4];

			QRE_FormatValue (m, row, val, sizeof (val));
			Draw_StringScaled (cbx, QRE_LABEL_X, (int)(y + 1), qre_params[row->param].label, 1.0f, &lc);
			q_snprintf (box, sizeof (box), "[%c]", on ? 'x' : ' ');
			Draw_StringScaled (cbx, QRE_WIDGET_X, (int)(y + 1), box, 1.0f, &vc);
			if (i == qre.active_text_row)
			{
				if (qre.text_buf[0])
					Draw_StringScaled (cbx, (int)(QRE_WIDGET_MAX - 8.0f * strlen (qre.text_buf)), (int)(y + 1), qre.text_buf, 1.0f, &vc);
			}
			else
				Draw_StringScaled (cbx, (int)(QRE_WIDGET_MAX - 8.0f * strlen (val)), (int)(y + 1), val, 1.0f, &vc);
			break;
		}
		case ROW_TEXT:
		{
			const RgFloat4D lc = QRE_Rgba (0.75f, 0.78f, 0.85f, 1.0f);
			const RgFloat4D vc = QRE_Rgba (0.9f, 0.93f, 1.0f, 1.0f);
			const RgFloat4D bc = QRE_Rgba (0.7f, 0.75f, 0.85f, 1.0f);
			char shown[28];

			Draw_StringScaled (cbx, QRE_LABEL_X, (int)(y + 1), qre_params[row->param].label, 1.0f, &lc);
			if (i == qre.active_text_row)
			{
				QRE_Truncate (shown, sizeof (shown), qre.text_buf);
				if (shown[0])
					Draw_StringScaled (cbx, QRE_LABEL_X, (int)(y + 11), shown, 1.0f, &vc);
			}
			else
			{
				QRE_FormatValue (m, row, val, sizeof (val));
				QRE_Truncate (shown, sizeof (shown), val);
				if (shown[0])
					Draw_StringScaled (cbx, QRE_LABEL_X, (int)(y + 11), shown, 1.0f, &vc);
			}
			Draw_StringScaled (cbx, QRE_WIDGET_MAX - 24, (int)(y + 11), "...", 1.0f, &bc);
			break;
		}
		case ROW_COLOR:
		{
			const RgFloat4D lc = QRE_Rgba (0.75f, 0.78f, 0.85f, 1.0f);
			const RgFloat4D vc = QRE_Rgba (1, 1, 1, 1);
			qboolean en;
			float    rgb[3];
			char     box[4];
			float    sw[4];

			QRE_GetColor (m, row->param, &en, rgb);
			QRE_FormatValue (m, row, val, sizeof (val));

			Draw_StringScaled (cbx, QRE_LABEL_X, (int)(y + 1), qre_params[row->param].label, 1.0f, &lc);
			q_snprintf (box, sizeof (box), "[%c]", en ? 'x' : ' ');
			Draw_StringScaled (cbx, QRE_WIDGET_X, (int)(y + 1), box, 1.0f, &vc);

			sw[0] = en ? rgb[0] : 0.3f;
			sw[1] = en ? rgb[1] : 0.3f;
			sw[2] = en ? rgb[2] : 0.3f;
			sw[3] = 1.0f;
			Draw_FillRGBA (cbx, QRE_WIDGET_X + 22, (int)(y + 1), 8, 8, sw);

			Draw_StringScaled (cbx, (int)(QRE_WIDGET_MAX - 8.0f * strlen (val)), (int)(y + 1), val, 1.0f, &vc);
			break;
		}
		case ROW_CR:
		case ROW_CG:
		case ROW_CB:
		{
			const RgFloat4D lc = QRE_Rgba (0.75f, 0.78f, 0.85f, 1.0f);
			const RgFloat4D vc = QRE_Rgba (1, 1, 1, 1);
			const char *labels[3] = { "R", "G", "B" };
			qboolean en;
			float    rgb[3];
			float    value_start;

			QRE_GetColor (m, row->param, &en, rgb);
			QRE_FormatValue (m, row, val, sizeof (val));

			Draw_StringScaled (cbx, QRE_LABEL_X, (int)(y + 1), labels[row->sub], 1.0f, &lc);
			value_start = QRE_WIDGET_MAX - 8.0f * strlen (val);
			if (value_start < QRE_WIDGET_X + 40.0f)
				value_start = QRE_WIDGET_X + 40.0f;
			Draw_StringScaled (cbx, (int)value_start, (int)(y + 1), val, 1.0f, &vc);
			QRE_DrawSlider (cbx, row, QRE_WIDGET_X + 1.0f, y, value_start - 6.0f - (QRE_WIDGET_X + 1.0f), rgb[row->sub] * 255.0f);
			break;
		}
		case ROW_CHEX:
		{
			const RgFloat4D lc = QRE_Rgba (0.75f, 0.78f, 0.85f, 1.0f);
			const RgFloat4D vc = QRE_Rgba (1, 1, 1, 1);

			Draw_StringScaled (cbx, QRE_LABEL_X, (int)(y + 1), "hex", 1.0f, &lc);
			if (i == qre.active_text_row)
			{
				if (qre.text_buf[0])
					Draw_StringScaled (cbx, QRE_WIDGET_X, (int)(y + 1), qre.text_buf, 1.0f, &vc);
			}
			else
			{
				QRE_FormatValue (m, row, val, sizeof (val));
				Draw_StringScaled (cbx, QRE_WIDGET_X, (int)(y + 1), val, 1.0f, &vc);
			}
			break;
		}
		default:
			break;
		}

		y += rh;
	}

	// scrollbar
	if (maxscroll > 0.0f)
	{
		const float track[4] = { 0.12f, 0.12f, 0.16f, 1.0f };
		const float thumb[4] = { 0.5f, 0.52f, 0.6f, 1.0f };
		float thumb_h, thumb_y;

		thumb_h = view_h * view_h / content_h;
		if (thumb_h < 16.0f)
			thumb_h = 16.0f;
		thumb_y = QRE_CONTENT_TOP + (qre.scroll / maxscroll) * (view_h - thumb_h);

		Draw_FillRGBA (cbx, QRE_SCROLL_X, QRE_CONTENT_TOP, QRE_SCROLL_W, (int)view_h, track);
		Draw_FillRGBA (cbx, QRE_SCROLL_X + 1, (int)thumb_y, QRE_SCROLL_W - 2, (int)thumb_h, thumb);
	}

	QRE_DrawCursor (cbx);
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
// Panel: interaction
// ---------------------------------------------------------------------------

static qboolean QRE_Hit (float x, float y, int rx, int ry, int rw, int rh)
{
	return x >= rx && x < rx + rw && y >= ry && y < ry + rh;
}

static void QRE_ScrollBy (float dy)
{
	const float H = QRE_CanvasHeight ();
	const float view_h = H - QRE_CONTENT_TOP - 4.0f;
	const float content_h = QRE_ContentHeight ();
	const float maxscroll = content_h > view_h ? content_h - view_h : 0.0f;

	qre.scroll += dy;
	if (qre.scroll < 0.0f)
		qre.scroll = 0.0f;
	if (qre.scroll > maxscroll)
		qre.scroll = maxscroll;
}

static void QRE_OpenTextField (int row)
{
	qre.active_text_row = row;
	QRE_FormatValue (qre.group[qre.rows[row].mat], &qre.rows[row], qre.text_buf, sizeof (qre.text_buf));
}

static void QRE_SliderDrag (int row, float x)
{
	const qre_row_t *r = &qre.rows[row];
	rt_material_t   *m = qre.group[r->mat];
	float            value_start, track_x, track_w, f;
	char             val[64];

	if (r->kind != ROW_FLOAT && r->kind != ROW_INT && r->kind != ROW_CR && r->kind != ROW_CG && r->kind != ROW_CB)
		return;

	QRE_FormatValue (m, r, val, sizeof (val));
	value_start = QRE_WIDGET_MAX - 8.0f * strlen (val);
	if (value_start < QRE_WIDGET_X + 40.0f)
		value_start = QRE_WIDGET_X + 40.0f;

	track_x = QRE_WIDGET_X + 1.0f;
	track_w = value_start - 6.0f - track_x;
	if (track_w < 4.0f)
		track_w = 4.0f;

	f = (x - track_x) / track_w;
	f = CLAMP (0.0f, f, 1.0f);

	if (r->kind == ROW_CR || r->kind == ROW_CG || r->kind == ROW_CB)
	{
		QRE_SetColorChannel (r->mat, r->param, r->sub, f);
	}
	else if (r->kind == ROW_INT)
	{
		float step = qre_params[r->param].step;
		float min = qre_params[r->param].min;
		float max = qre_params[r->param].max;
		int   v = (int)floorf (min + f * (max - min) + 0.5f);

		v = ((int)(v / step)) * ((int)step);
		QRE_SetInt (r->mat, r->param, v);
	}
	else
	{
		float step = qre_params[r->param].step;
		float min = qre_params[r->param].min;
		float max = qre_params[r->param].max;
		float v = min + f * (max - min);

		if (step > 0.0f)
			v = floorf (v / step + 0.5f) * step;
		QRE_SetFloat (r->mat, r->param, v);
	}
}

static void QRE_PanelClick (void)
{
	const float mx = qre.mouse_x;
	const float my = qre.mouse_y;
	const float H = QRE_CanvasHeight ();
	float       view_h, content_h, maxscroll;

	// the buttons
	if (QRE_Hit (mx, my, 444, 24, 48, 12))
	{
		QRE_Apply ();
		return;
	}
	if (QRE_Hit (mx, my, 498, 24, 48, 12))
	{
		QRE_Cancel ();
		return;
	}
	if (QRE_Hit (mx, my, 552, 24, 48, 12))
	{
		QRE_StopEditor (true);
		return;
	}

	view_h = H - QRE_CONTENT_TOP - 4.0f;
	content_h = QRE_ContentHeight ();
	maxscroll = content_h > view_h ? content_h - view_h : 0.0f;

	// the scrollbar
	if (maxscroll > 0.0f && mx >= QRE_SCROLL_X && my >= QRE_CONTENT_TOP && my < QRE_CONTENT_TOP + view_h)
	{
		float thumb_h = view_h * view_h / content_h;
		float thumb_y;

		if (thumb_h < 16.0f)
			thumb_h = 16.0f;
		thumb_y = QRE_CONTENT_TOP + (qre.scroll / maxscroll) * (view_h - thumb_h);

		if (my >= thumb_y && my < thumb_y + thumb_h)
		{
			qre.drag_scroll = true;
			qre.drag_scroll_grab = my - thumb_y;
		}
		else
		{
			qre.scroll = maxscroll * (my - QRE_CONTENT_TOP - thumb_h * 0.5f) / (view_h - thumb_h);
			if (qre.scroll < 0.0f)
				qre.scroll = 0.0f;
			if (qre.scroll > maxscroll)
				qre.scroll = maxscroll;
		}
		return;
	}

	// the rows
	{
		int row = QRE_RowAt (my);
		const qre_row_t *r;
		rt_material_t   *m;
		char             val[64];
		float            value_start;

		if (row < 0)
			return;
		r = &qre.rows[row];
		m = qre.group[r->mat];

		switch (r->kind)
		{
		case ROW_BOOL:
			QRE_SetBool (r->mat, r->param, !QRE_GetBool (m, r->param));
			break;

		case ROW_COLOR:
			// [x] toggles the color, the swatch expands the color editor, the
			// hex value becomes a text field
			if (mx < QRE_WIDGET_X + 20)
			{
				qboolean en;
				float    rgb[3];

				QRE_GetColor (m, r->param, &en, rgb);
				QRE_SetColorEnabled (r->mat, r->param, !en);
			}
			else if (mx >= QRE_WIDGET_X + 20 && mx < QRE_WIDGET_X + 32)
			{
				if (qre.color_expand_mat == r->mat && qre.color_expand_param == r->param)
				{
					qre.color_expand_mat = -1;
					qre.color_expand_param = -1;
				}
				else
				{
					qre.color_expand_mat = r->mat;
					qre.color_expand_param = r->param;
				}
			}
			else
			{
				QRE_OpenTextField (row);
			}
			break;

		case ROW_FLOAT:
		case ROW_INT:
			QRE_FormatValue (m, r, val, sizeof (val));
			value_start = QRE_WIDGET_MAX - 8.0f * strlen (val);
			if (value_start < QRE_WIDGET_X + 40.0f)
				value_start = QRE_WIDGET_X + 40.0f;
			if (mx >= value_start)
			{
				QRE_OpenTextField (row);
			}
			else if (mx >= QRE_WIDGET_X)
			{
				qre.drag_slider_row = row;
				QRE_SliderDrag (row, mx);
			}
			break;

		case ROW_CR:
		case ROW_CG:
		case ROW_CB:
			qre.drag_slider_row = row;
			QRE_SliderDrag (row, mx);
			break;

		case ROW_TEXT:
			if (mx >= QRE_WIDGET_MAX - 24)
			{
				char buf[MAX_QPATH];

				if (QRE_BrowseTexture (buf, sizeof (buf)))
				{
					QRE_SetText (r->mat, r->param, buf);
					qre.active_text_row = -1;
				}
			}
			else
			{
				QRE_OpenTextField (row);
			}
			break;

		case ROW_CHEX:
			QRE_OpenTextField (row);
			break;

		default:
			break;
		}
	}
}

// ---------------------------------------------------------------------------
// Panel: frame bookkeeping
// ---------------------------------------------------------------------------

static void QRE_Frame (void)
{
	int mx, my;

	// the level went away under the editor: drop it (the material snapshot may
	// be stale relative to a freshly loaded map list, so nothing is restored)
	if (cls.state != ca_connected || !cl.worldmodel)
	{
		QRE_StopEditor (false);
		return;
	}

	SDL_GetMouseState (&mx, &my);
	qre.mouse_x = (float)mx * 640.0f / (float)glwidth;
	qre.mouse_y = (float)my * QRE_CanvasHeight () / (float)glheight;

	if (!qre.panel_open)
	{
		QRE_DoPick (false);
	}
	else
	{
		if (qre.drag_scroll && keydown[K_MOUSE1])
		{
			const float H = QRE_CanvasHeight ();
			const float view_h = H - QRE_CONTENT_TOP - 4.0f;
			const float content_h = QRE_ContentHeight ();
			const float maxscroll = content_h > view_h ? content_h - view_h : 0.0f;
			float thumb_h = view_h * view_h / content_h;

			if (thumb_h < 16.0f)
				thumb_h = 16.0f;
			if (maxscroll > 0.0f)
			{
				qre.scroll = maxscroll * (qre.mouse_y - qre.drag_scroll_grab - QRE_CONTENT_TOP) / (view_h - thumb_h);
				if (qre.scroll < 0.0f)
					qre.scroll = 0.0f;
				if (qre.scroll > maxscroll)
					qre.scroll = maxscroll;
			}
		}
		else
		{
			qre.drag_scroll = false;
		}

		if (qre.drag_slider_row >= 0)
		{
			if (keydown[K_MOUSE1])
				QRE_SliderDrag (qre.drag_slider_row, qre.mouse_x);
			else
				qre.drag_slider_row = -1;
		}
	}

	QRE_FlushDirty ();
}

void QR_Editor_DrawPanel (cb_context_t *cbx)
{
	if (!qre.active)
		return;

	QRE_Frame ();

	if (qre.panel_open)
		QRE_DrawPanelUI (cbx);
	else
		QRE_DrawFlyingHint (cbx);
}

// ---------------------------------------------------------------------------
// Input hooks
// ---------------------------------------------------------------------------

qboolean QR_Editor_KeyEvent (int key, qboolean down)
{
	if (!qre.active)
		return false;

	if (qre.panel_open)
	{
		if (!down)
		{
			if (key == K_MOUSE1)
			{
				qre.drag_slider_row = -1;
				qre.drag_scroll = false;
			}
			return true;
		}

		switch (key)
		{
		case K_ESCAPE:
			// close the panel, keep flying
			qre.panel_open = false;
			qre.pick_model = NULL;
			qre.pick_surf = NULL;
			qre.active_text_row = -1;
			qre.drag_slider_row = -1;
			qre.drag_scroll = false;
			qre.color_expand_mat = -1;
			qre.color_expand_param = -1;
			IN_Activate ();
			SDL_ShowCursor (SDL_ENABLE);
			break;

		case K_MOUSE1:
			QRE_PanelClick ();
			break;

		case K_MWHEELUP:
			QRE_ScrollBy (-32.0f);
			break;

		case K_MWHEELDOWN:
			QRE_ScrollBy (32.0f);
			break;

		case K_BACKSPACE:
			if (qre.active_text_row >= 0 && qre.text_buf[0])
				qre.text_buf[strlen (qre.text_buf) - 1] = '\0';
			break;

		case K_ENTER:
		case K_KP_ENTER:
			if (qre.active_text_row >= 0)
				QRE_CommitText ();
			break;

		default:
			break;
		}

		return true;
	}

	// flying: ESC exits the editor instead of opening the menu
	if (key == K_ESCAPE && down)
	{
		QRE_StopEditor (true);
		return true;
	}

	return false;
}

qboolean QR_Editor_CharEvent (int key)
{
	size_t len;

	if (!qre.active || !qre.panel_open || qre.active_text_row < 0)
		return false;
	if (key < 32 || key > 126)
		return false;

	len = strlen (qre.text_buf);
	if (len + 1 < sizeof (qre.text_buf))
	{
		qre.text_buf[len] = (char)key;
		qre.text_buf[len + 1] = '\0';
	}
	return true;
}

qboolean QR_Editor_TextEntryActive (void)
{
	return qre.active && qre.panel_open && qre.active_text_row >= 0;
}

// ---------------------------------------------------------------------------
// Apply / Cancel / Exit
// ---------------------------------------------------------------------------

static void QRE_Apply (void)
{
	QRE_SaveMaterials ();
	QRE_TakeSnapshot (); // Cancel now reverts to the state just saved
	qre.touched_count = 0;
	Con_Printf ("qr editor: materials written to %s/materials/\n", com_gamedir);
}

static void QRE_Cancel (void)
{
	qre.active_text_row = -1;
	qre.drag_slider_row = -1;
	qre.drag_scroll = false;

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

			RT_MAT_NormalizeName (glt->name, texname, sizeof (texname));
			{
				char *dot = strrchr (texname, '.');
				if (dot && !strchr (dot, ':'))
					*dot = '\0';
			}
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
	const char      *file = m->source_file[0] ? m->source_file : "materials/materials.yaml";
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
	qre.active_text_row = -1;
	qre.drag_slider_row = -1;
	qre.drag_scroll = false;

	VectorCopy (qre.player_viewangles, cl.viewangles);

	QRE_FreeSnapshot ();

	IN_Activate ();
	SDL_ShowCursor (SDL_ENABLE);

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

	if (qr_editor_registered)
		return;
	qr_editor_registered = true;

	Cmd_AddCommand ("qr_light_editor_start", QR_Editor_Start_f);
	Cmd_AddCommand ("qr_light_editor_stop", QR_Editor_Stop_f);
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
