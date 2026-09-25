// qr_editor.c -- qr light editor: realtime material editor for the vkpt renderer.
//
// Console commands: qr_material_editor_start / qr_material_editor_stop.
//
// While the editor runs the view belongs to a free camera (the player stands
// still): aim with the crosshair, fire selects the face under it and opens the
// material panel on the right edge of the screen. The panel is Dear ImGui
// (Quake/qr_gui.cpp), and it edits the materials.yaml parameters of every
// animation frame of the picked texture (medkits, blinking buttons, ...).
// The world is frozen while the editor runs (the server is paused and cl.time
// stands still, so nothing animates). Apply writes the session to
// materials.editor.yaml; Exit asks whether to save, and only Save copies the
// session file over <gamedir>/materials.yaml (backing the previous file up as
// backup_materials.yaml) — a mod's materials.yaml overrides the id1 one, both
// because it is loaded after it and because Save writes to the mod's file.
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
#include "rt_lights.h"
#include "keys.h"
#include "client.h"
#include "server.h"
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
// The editor edits what the new light system builds (TAL and the fake dlights
// of materials); the old system has neither, so it refuses to start on it.
extern cvar_t rt_truelight; // gl_vidsdl.c
extern cvar_t rt_dlight_radius, rt_dlight_intensity; // gl_vidsdl.c

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
	PARAM_GLOSS,    // texture_gloss
	PARAM_BUMP,
	PARAM_ROUGH,
	PARAM_METAL,
	PARAM_BASEF,
	PARAM_LBRIGHT,
	PARAM_LUPOFF,
	PARAM_LCOLOR,
	PARAM_ISLIGHT,
	PARAM_LSTYLES,
	PARAM_METALALPHA,
	PARAM_MIRROR,
	PARAM_EXACTN,
	PARAM_FRAST,
	PARAM_CEMIS,
	PARAM_COUNT,
};

static const struct qre_param_s
{
	const char *label;
	int         type;
	float       min, max, step;
	const char *tip;
} qre_params[PARAM_COUNT] = {
	[PARAM_BASE]     = { "texture_base",     QRE_T_TEXT,  0, 0, 0,
	                     "The diffuse texture. NONE keeps the map's own; an authored file replaces it." },
	[PARAM_NORMALS]  = { "texture_normals",  QRE_T_TEXT,  0, 0, 0,
	                     "Per-pixel bump direction. Its alpha channel can drive metalness_from_normal_alpha." },
	[PARAM_EMISSIVE] = { "texture_emissive", QRE_T_TEXT,  0, 0, 0,
	                     "A luma mask image: what is bright in it is what the surface emits. Replaces color_emissive." },
	[PARAM_GLOSS]    = { "texture_gloss",    QRE_T_TEXT,  0, 0, 0,
	                     "White means mirror-smooth, black means rough (roughness = 1 - gloss). Ignored while roughness_override is set." },
	[PARAM_BUMP]     = { "bump_scale",       QRE_T_FLOAT, 0, 4, 0.01f,
	                     "How pronounced the normal map's bumps are. Needs texture_normals." },
	[PARAM_ROUGH]    = { "roughness_override", QRE_T_FLOAT, 0, 1, 0.01f,
	                     "Ignore the gloss map and pin the roughness. 0 leaves it to the gloss; mirror forces 0." },
	[PARAM_METAL]    = { "metalness_factor", QRE_T_FLOAT, 0, 1, 0.01f,
	                     "How metal-like the surface is. With metalness_from_normal_alpha it scales that mask." },
	[PARAM_BASEF]    = { "base_factor",      QRE_T_FLOAT, 0, 4, 0.01f,
	                     "Multiplies the albedo: dims or lifts the whole texture." },
	[PARAM_LBRIGHT]  = { "light_brightness", QRE_T_FLOAT, 0, 5, 0.01f,
	                     "How bright the light the surface casts is. Below 1 the glow dims with it; above 1 only the light grows (the visible glow is already at its maximum)." },
	[PARAM_LUPOFF]   = { "light_upoffset",   QRE_T_FLOAT, -64, 64, 0.5f,
	                     "Lifts the cast light above the model's origin (alias models)." },
	[PARAM_CEMIS]    = { "color_emissive",   QRE_T_BOOL,  0, 0, 0,
	                     "Glow by colour: every block below matches its own colour and carries its own threshold, feather, emissive_factor and blend." },
	[PARAM_ISLIGHT]  = { "is_light",         QRE_T_BOOL,  0, 0, 0,
	                     "The surface casts light into the scene, not only glows (BSP faces; models light from light_color)." },
	[PARAM_LSTYLES]  = { "light_styles",     QRE_T_BOOL,  0, 0, 0,
	                     "Tick to let the map's light styles dim this light; off by default, so a light stays at full brightness unless it asks otherwise." },
	[PARAM_METALALPHA] = { "metalness_from_normal_alpha", QRE_T_BOOL, 0, 0, 0,
	                     "Read metalness from the normal map's alpha channel instead of a flat factor." },
	[PARAM_MIRROR]   = { "mirror",           QRE_T_BOOL,  0, 0, 0,
	                     "Mirror-smooth reflection; roughness is forced to 0." },
	[PARAM_EXACTN]   = { "exact_normals",    QRE_T_BOOL,  0, 0, 0,
	                     "Use the model's own vertex normals instead of generated ones (models only)." },
	[PARAM_FRAST]    = { "force_rasterize",  QRE_T_BOOL,  0, 0, 0,
	                     "Draw the model or sprite with the rasterizer instead of tracing it (models only)." },
	[PARAM_LCOLOR]   = { "light_color",      QRE_T_COLOR, 0, 0, 0,
	                     "The colour of the light the surface casts (needs is_light on BSP faces; models light by themselves)." },
};

// The emissive blend modes as the combos offer them: the index is the mode's
// value plus one, so index 0 is "cvar" (-1) and the rest line up with
// RT_MAT_EmissiveBlendName.
static const char *const qre_emissive_blends[] = {
	"cvar", "off", "normal", "screen", "overlay", "hard light", "colour dodge"
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#define QRE_GROUP_MAX    12
#define QRE_DIRTY_MAX    64
#define QRE_TOUCHED_MAX  512
#define QRE_PREVIEW_SLOTS QRE_GROUP_MAX

// One texture preview: the RgMaterial the panel draws and the pixels the colour
// picker samples, cached under the key of what they were built from, one slot
// per group entry so rebuilding one section never frees a material another
// section's draw command already names.
typedef struct qre_preview_s
{
	char        key[MAX_QPATH * 2 + 32];
	RgMaterial  mat;
	byte       *pixels;
	int         w, h;
	unsigned    last_frame;
} qre_preview_t;

// What the editor edits: the surfaces (materials.yaml) or the dynamic lights
// the emitters cast (lights.yaml). The camera, the picking and the session flow
// (Apply / Cancel / the exit dialog) are shared; the panel and the files differ.
enum
{
	QRE_MODE_MATERIAL = 0,
	QRE_MODE_LIGHT    = 1,
};

static struct
{
	qboolean active;
	qboolean panel_open;

	vec3_t cam_origin;
	vec3_t player_viewangles;

	// the picked / hovered face (hover is updated while flying); ent is NULL
	// for a world face and the brush entity for a model face. A model/sprite
	// pick has no surface: pick_glt is its skin texture and pick_surf is NULL.
	qmodel_t    *pick_model;
	msurface_t  *pick_surf;
	entity_t    *pick_ent;
	gltexture_t *pick_glt;
	qmodel_t    *hover_model;
	msurface_t  *hover_surf;
	entity_t    *hover_ent;
	gltexture_t *hover_glt;

	// the picked texture's normalized material name (what the panel shows and
	// the group was built from; the engine name carries a maps/<map>.bsp: prefix)
	char         pick_name[MAX_QPATH];

	// the material group (animation frames) shown by the panel
	rt_material_t *group[QRE_GROUP_MAX];
	int            group_count;
	// detached defaults for the animation frames a model pick shows before any
	// of them has a yaml entry (each becomes a material on its first change)
	rt_material_t  extra[QRE_GROUP_MAX];
	int            extra_count;

	// snapshot of both material lists for Cancel/Exit
	rt_material_t *snap_global;
	int            snap_global_count;
	rt_material_t *snap_map;
	int            snap_map_count;

	// snapshot of the light overrides, and the light names the session touched
	rt_light_t *snap_lights;
	int         snap_light_count;
	char        light_touched[QRE_TOUCHED_MAX][MAX_QPATH];
	int         light_touched_count;

	// the light editor: the light the crosshair is over, and the one selected
	// (the panel edits the entry of the selected light's emitter)
	rt_tracked_light_t sel_light;
	qboolean           sel_light_valid;
	int                hover_light;

	// the light editor's tabs: 0 = the selected emitter, 1 = the sky, clouds
	// and sun (the global settings)
	int light_tab;

	// which of the two editors this is
	int mode;

	// the session files, resolved on start: <gamedir>/materials.yaml is the file
	// the editor saves to (a mod's file overrides the id1 one), while
	// materials.editor.yaml carries the session until the exit dialog decides and
	// backup_materials.yaml keeps the target as it was before a save
	char target_file[MAX_OSPATH];
	char editor_file[MAX_OSPATH];
	char backup_file[MAX_OSPATH];

	// a material created by the editor for a texture that has none in yaml
	rt_material_t tmp_mat;
	qboolean      tmp_appended;

	// the exit dialog is up: Save/Discard, the editor keeps running until
	// answered; prompt_from_flying is the mode to go back to when dismissed
	qboolean exit_prompt;
	qboolean prompt_from_flying;

	// the pause state the editor found, restored when it closes
	qboolean sv_paused_prev;

	// the base-texture previews of the selected material (the emissive colour
	// picker): one slot per group entry in flight, so a section rebuild never
	// destroys a material an earlier section's draw command still names
	qre_preview_t preview[QRE_PREVIEW_SLOTS];

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
static void     QRE_RequestExit (void);
static qboolean QRE_WriteSession (void);
static qboolean QRE_FileExists (const char *path);
static void     QRE_SessionSave (void);
static void     QRE_SessionDiscard (void);
static qboolean QRE_BrowseTexture (char *out, size_t outsize);

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

// "textures/+3_med25" -> 3, "progs/flame.mdl:frame2" -> 2 (a model or sprite
// skin frame); the ring base and the digit live in rt_material.c, shared with
// the renderer-side group enumeration.
static int QRE_FrameDigit (const char *name)
{
	return RT_MAT_FrameDigit (name);
}

static void QRE_GroupBaseOf (const char *name, char *out, size_t outsize)
{
	RT_MAT_GroupBaseOf (name, out, outsize);
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

// The defaults of a material with no yaml entry: what the detached default is
// built from, and what a parameter of a material the editor created resets to.
static void QRE_InitDefault (rt_material_t *m, const char *name)
{
	memset (m, 0, sizeof (*m));
	m->valid = true;
	m->bump_scale = 1.0f;
	m->emissive_factor = 1.0f;
	m->emissive_blend = -1;
	m->base_factor = 1.0f;
	m->light_brightness = 1.0f;
	m->light_styles = false;
	m->color_emissive_threshold = 0.02f;
	q_strlcpy (m->name, name, sizeof (m->name));
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
// Light snapshot (for Cancel/Exit of the light editor)
// ---------------------------------------------------------------------------

static void QRE_TakeLightSnapshot (void)
{
	rt_light_t *list = RT_LIGHT_List (&qre.snap_light_count);

	if (!qre.snap_lights)
		qre.snap_lights = (rt_light_t *)Mem_Alloc (RT_LIGHT_NAMES_MAX * sizeof (rt_light_t));
	memcpy (qre.snap_lights, list, (size_t)qre.snap_light_count * sizeof (rt_light_t));
}

static void QRE_RestoreLightSnapshot (void)
{
	rt_light_t *list = RT_LIGHT_List (NULL);

	memcpy (list, qre.snap_lights, (size_t)qre.snap_light_count * sizeof (rt_light_t));
	RT_LIGHT_SetCount (qre.snap_light_count);
}

static void QRE_FreeLightSnapshot (void)
{
	if (qre.snap_lights)
		Mem_Free (qre.snap_lights);
	qre.snap_lights = NULL;
	qre.snap_light_count = 0;
	qre.light_touched_count = 0;
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

// The light names the session touched: what the session file writes besides the
// entries the target already carried.
static void QRE_TouchLight (const char *name)
{
	if (!name || !name[0])
		return;
	if (!QRE_NameInList (qre.light_touched, qre.light_touched_count, name) &&
	    qre.light_touched_count < QRE_TOUCHED_MAX)
		q_strlcpy (qre.light_touched[qre.light_touched_count++], name, MAX_QPATH);
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
	// the touched names are kept: they are the session's own list, and the
	// session file is rewritten from them when Cancel reverts the values

	Atomic_StoreUInt32 (&rt_require_static_submit, true);
}

// A material created by the editor becomes part of the live global list on the
// first change, so the synthesis (RT_MAT_Find) can see it. The detached
// defaults are qre.tmp_mat and the qre.extra frames of a model pick.
static qboolean QRE_IsDetached (const rt_material_t *m)
{
	return m == &qre.tmp_mat || (m >= &qre.extra[0] && m < &qre.extra[QRE_GROUP_MAX]);
}

static void QRE_EnsureLive (int g)
{
	rt_material_t *m = qre.group[g];
	int            idx;

	if (!QRE_IsDetached (m))
		return;
	if (m == &qre.tmp_mat && qre.tmp_appended)
		return;

	idx = RT_MAT_AppendGlobal (m);
	if (idx >= 0)
	{
		if (m == &qre.tmp_mat)
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
	(void)param; // only light_color is a single colour now; color_emissive is a list
	*enabled = m->has_light_color;
	VectorCopy (m->light_color, rgb);
}

static void QRE_SetColorEnabled (int g, int param, qboolean enabled)
{
	QRE_EnsureLive (g);
	rt_material_t *m = qre.group[g];

	(void)param;
	m->has_light_color = enabled;
	QRE_MarkDirty (m);
}

static void QRE_SetColorChannel (int g, int param, int channel, float value)
{
	QRE_EnsureLive (g);
	rt_material_t *m = qre.group[g];

	(void)param;
	m->has_light_color = true;
	m->light_color[channel] = value;
	QRE_MarkDirty (m);
}

static float QRE_GetFloat (const rt_material_t *m, int param)
{
	switch (param)
	{
	case PARAM_BUMP:     return m->bump_scale;
	case PARAM_ROUGH:    return m->roughness_override;
	case PARAM_METAL:    return m->metalness_factor;
	case PARAM_BASEF:    return m->base_factor;
	case PARAM_LBRIGHT:  return m->light_brightness;
	case PARAM_LUPOFF:   return m->light_upoffset;
	default:             return 0.0f;
	}
}

static int QRE_GetInt (const rt_material_t *m, int param)
{
	(void)m;
	(void)param;
	return 0; // the per-block controls are edited inside QRE_EmissiveEditor
}

static qboolean QRE_GetBool (const rt_material_t *m, int param)
{
	switch (param)
	{
	case PARAM_ISLIGHT:    return m->is_light;
	case PARAM_CEMIS:      return m->has_color_emissive;
	case PARAM_LSTYLES:    return m->light_styles;
	case PARAM_METALALPHA: return m->metalness_from_normal_alpha;
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
	case PARAM_LBRIGHT:  m->light_brightness = value; break;
	case PARAM_LUPOFF:   m->light_upoffset = value; break;
	default:             break;
	}
	QRE_MarkDirty (m);
}

static void QRE_SetInt (int g, int param, int value)
{
	(void)g;
	(void)param;
	(void)value; // the per-block controls are edited inside QRE_EmissiveEditor
}

static void QRE_SetBool (int g, int param, qboolean value)
{
	QRE_EnsureLive (g);
	rt_material_t *m = qre.group[g];

	switch (param)
	{
	case PARAM_ISLIGHT:    m->is_light = value; break;
	case PARAM_CEMIS:      m->has_color_emissive = value; break;
	case PARAM_LSTYLES:    m->light_styles = value; break;
	case PARAM_METALALPHA: m->metalness_from_normal_alpha = value; break;
	case PARAM_MIRROR:
		m->mirror = value;
		if (value)
			m->roughness_override = 0.0f; // the panel locks the override while mirror is on
		break;
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

// A detached default for one frame name of the picked texture: named exactly as
// the engine names that texture, so the first change to it resolves to it (the
// synthesis looks the texture's own name up).
static void QRE_AddDefaultName (const char *groupbase, const char *name)
{
	char base[MAX_QPATH];
	int  i;

	if (!name[0])
		return;

	RT_MAT_GroupBaseOf (name, base, sizeof (base));
	if (strcmp (base, groupbase))
		return; // not a frame of this ring

	for (i = 0; i < qre.group_count; i++)
	{
		if (!strcmp (qre.group[i]->name, name))
			return; // already shown (authored or added)
	}

	if (qre.extra_count >= QRE_GROUP_MAX)
		return;

	QRE_InitDefault (&qre.extra[qre.extra_count], name);
	qre.group[qre.group_count++] = &qre.extra[qre.extra_count++];
}

// Every frame the engine actually has for the picked texture: the "textures/+N"
// ring and the "progs/x.mdl:frameN" skins of a model or sprite. Without it a
// frame with no yaml entry opened one block named after the ring base -- a name
// no texture resolves to -- so editing it changed nothing on screen.
static void QRE_AddEngineFrames (const char *texname, const char *groupbase)
{
	char names[QRE_GROUP_MAX][MAX_QPATH];
	int  count = TexMgr_CollectGroupNames (texname, names, QRE_GROUP_MAX);
	int  i;

	for (i = 0; i < count && qre.group_count < QRE_GROUP_MAX; i++)
		QRE_AddDefaultName (groupbase, names[i]);
}

// Builds the group of materials for a texture name (all animation frames).
static void QRE_ResolveGroup (const char *texname)
{
	char groupbase[MAX_QPATH];
	int  pass, i, k;

	QRE_GroupBaseOf (texname, groupbase, sizeof (groupbase));

	qre.group_count = 0;
	qre.extra_count = 0;
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
			// A model's own base name is not a texture the engine has: only
			// its :frameN names resolve, so a stale base entry stays hidden
			// (for a texture ring the base is a real material and is shown).
			if (qre.pick_model && qre.pick_model->type != mod_brush &&
			    !q_strcasecmp (list[i].name, groupbase))
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

	// the frames the engine has but no material names yet get their own blocks
	QRE_AddEngineFrames (texname, groupbase);

	if (qre.group_count == 0)
	{
		// no material authored for this texture: edit a detached default one
		// that joins the global list on the first change
		QRE_InitDefault (&qre.tmp_mat, groupbase);
		qre.group[0] = &qre.tmp_mat;
		qre.group_count = 1;
	}
	else if (qre.group_count > 1)
	{
		qsort (qre.group, (size_t)qre.group_count, sizeof (qre.group[0]), QRE_CompareMats);
	}

	// mirror forces roughness_override to 0, and the synthesis gives it the last
	// word: a material loaded with both is shown (and saved) with the override
	// the renderer ignores taken out of the way
	for (i = 0; i < qre.group_count; i++)
	{
		if (qre.group[i]->mirror && qre.group[i]->roughness_override != 0.0f)
			qre.group[i]->roughness_override = 0.0f;
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

// The skin texture a model entity draws: the alias frame at the animation the
// renderer picked (it follows cl.time, which the editor freezes) or the sprite
// frame.
static gltexture_t *QRE_EntityTexture (entity_t *e)
{
	if (!e->model)
		return NULL;

	if (e->model->type == mod_alias)
	{
		aliashdr_t *hdr = (aliashdr_t *)Mod_Extradata (e->model);
		int         anim, skinnum;

		if (!hdr)
			return NULL;

		anim = (int)(cl.time * 10) & 3;
		skinnum = e->skinnum;
		if (skinnum < 0 || skinnum >= hdr->numskins)
			skinnum = 0;
		return hdr->gltextures[skinnum][anim];
	}

	if (e->model->type == mod_sprite)
	{
		mspriteframe_t *frame = R_GetSpriteFrame (e);

		return frame ? frame->gltexture : NULL;
	}

	return NULL;
}

// Moller-Trumbore: the fraction along the ray where it crosses the triangle,
// or -1. The direction is start->end scaled as the ray parameter.
static float QRE_RayTriangle (const vec3_t start, const vec3_t dir,
                              const vec3_t a, const vec3_t b, const vec3_t c)
{
	vec3_t e1, e2, p, t, q;
	float  det, inv, u, v, frac;

	VectorSubtract (b, a, e1);
	VectorSubtract (c, a, e2);
	CrossProduct (dir, e2, p);
	det = DotProduct (e1, p);
	if (fabsf (det) < 1e-8f)
		return -1.0f;
	inv = 1.0f / det;

	VectorSubtract (start, a, t);
	u = DotProduct (t, p) * inv;
	if (u < 0.0f || u > 1.0f)
		return -1.0f;

	CrossProduct (t, e1, q);
	v = DotProduct (dir, q) * inv;
	if (v < 0.0f || u + v > 1.0f)
		return -1.0f;

	frac = DotProduct (e2, q) * inv;
	return (frac >= 0.0f && frac <= 1.0f) ? frac : -1.0f;
}

// A ray against an entity's model bounds (a rigid transform about the origin;
// an alias model's bounds already carry its own scale and scale_origin).
// Returns the fraction along the ray, or -1 when it misses.
static float QRE_TraceEntityBox (entity_t *e, const vec3_t start, const vec3_t end)
{
	float       m[16];
	RgTransform transform;
	vec3_t      mins, maxs, local_start, local_dir;
	float       tmin = 0.0f, tmax = 1.0f;
	int         i;

	if (!e->model || (e->model->type != mod_alias && e->model->type != mod_sprite))
		return -1.0f;

	if (e->model->type == mod_sprite)
	{
		mspriteframe_t *frame = R_GetSpriteFrame (e);
		vec3_t          corners[4];
		vec3_t          dir;
		float           t1, t2;

		if (!frame)
			return -1.0f;

		// the quad the renderer draws, axes and all (R_CreateSpriteVertices)
		R_GetSpriteQuadCorners (e, frame, corners);

		VectorSubtract (end, start, dir);
		t1 = QRE_RayTriangle (start, dir, corners[0], corners[1], corners[2]);
		t2 = QRE_RayTriangle (start, dir, corners[0], corners[2], corners[3]);
		if (t1 < 0.0f || (t2 >= 0.0f && t2 < t1))
			t1 = t2;
		return t1;
	}

	VectorCopy (e->model->mins, mins);
	VectorCopy (e->model->maxs, maxs);

	// an alias model with empty bounds cannot be tested
	if (mins[0] == maxs[0] && mins[1] == maxs[1] && mins[2] == maxs[2])
		return -1.0f;

	IdentityMatrix (m);
	R_RotateForEntity (m, e->origin, e->angles);
	transform = RT_GetModelTransform (m);

	for (i = 0; i < 3; i++)
	{
		vec3_t ds, de;

		VectorSubtract (start, e->origin, ds);
		VectorSubtract (end, e->origin, de);

		// local is the transpose of the rigid rotation applied to the delta
		local_start[i] = transform.matrix[0][i] * ds[0]
		               + transform.matrix[1][i] * ds[1]
		               + transform.matrix[2][i] * ds[2];
		local_dir[i]   = transform.matrix[0][i] * (de[0] - ds[0])
		               + transform.matrix[1][i] * (de[1] - ds[1])
		               + transform.matrix[2][i] * (de[2] - ds[2]);
	}

	for (i = 0; i < 3; i++)
	{
		const float d = local_dir[i];

		if (fabsf (d) < 1e-6f)
		{
			if (local_start[i] < mins[i] || local_start[i] > maxs[i])
				return -1.0f;
		}
		else
		{
			float t1 = (mins[i] - local_start[i]) / d;
			float t2 = (maxs[i] - local_start[i]) / d;

			if (t1 > t2)
			{
				const float t = t1;

				t1 = t2;
				t2 = t;
			}
			if (t1 > tmin)
				tmin = t1;
			if (t2 < tmax)
				tmax = t2;
			if (tmin > tmax)
				return -1.0f;
		}
	}

	return tmin;
}

static qboolean QRE_TracePick (qmodel_t **out_model, msurface_t **out_surf, entity_t **out_ent, gltexture_t **out_glt)
{
	vec3_t    start, end;
	trace_t   tr;
	float     best = 1.0f;
	qmodel_t *bestmodel = NULL;
	entity_t *bestent = NULL;
	qboolean  bestisentity = false;
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

	// alias models and sprites (medkits, items, torches): their textures are
	// model skins, not BSP surfaces, so the traces above cannot see them
	{
		int i;

		for (i = 1; i < cl.num_entities; i++)
		{
			entity_t *e = &cl.entities[i];
			float     f;

			if (!e->model || e->model == cl.worldmodel)
				continue;
			if (e->model->type != mod_alias && e->model->type != mod_sprite)
				continue;
			if (e == &cl.viewent || i == cl.viewentity)
				continue;
			// the renderer draws nothing for these, and a model without a
			// texture has no material to open: neither may take the pick from
			// what stands behind it
			if (e->alpha == ENTALPHA_ZERO || !QRE_EntityTexture (e))
				continue;

			f = QRE_TraceEntityBox (e, start, end);
			if (f >= 0.0f && f < best)
			{
				best = f;
				bestent = e;
				bestmodel = e->model;
				bestisentity = true;
			}
		}
	}

	if (!bestmodel)
		return false;

	if (bestisentity)
	{
		gltexture_t *glt = QRE_EntityTexture (bestent);

		if (!glt)
			return false;

		*out_surf = NULL;
		*out_model = bestmodel;
		if (out_ent)
			*out_ent = bestent;
		if (out_glt)
			*out_glt = glt;
		return true;
	}

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

// ---------------------------------------------------------------------------
// The light editor's picking and wireframes
// ---------------------------------------------------------------------------

#define QRE_LIGHT_WIRE_MAX  96
#define QRE_LIGHT_WIRE_SEGS 12

// The nearest light along the view ray, or -1.
static int QRE_LightUnderCrosshair (void)
{
	const rt_tracked_light_t *lights;
	int   count = 0, i, best = -1;
	float best_t = 1e30f;

	lights = RT_TRACK_Lights (&count);
	for (i = 0; i < count; i++)
	{
		vec3_t oc;
		float  b, c, disc, t;

		if (!lights[i].ready)
			continue;
		VectorSubtract (lights[i].position, r_origin, oc);
		b = DotProduct (oc, vpn);
		c = DotProduct (oc, oc) - lights[i].radius * lights[i].radius;
		disc = b * b - c;
		if (disc < 0.0f)
			continue;
		t = b - sqrtf (disc);
		if (t < 0.0f)
			t = b + sqrtf (disc);
		if (t < 0.0f)
			continue;
		if (t < best_t)
		{
			best_t = t;
			best = i;
		}
	}
	return best;
}

// The cursor mode: the panel is on screen and owns the cursor and the keyboard,
// and the camera stays where it stopped. Tab turns it on and off (from the
// flying side it arrives as a plain key, from the panel side through the SDL
// event hook), and picking a surface or a light turns it on too.
static void QRE_CursorMode (qboolean on)
{
	if (on == qre.panel_open)
		return;

	if (on)
	{
		qre.panel_open = true;

		// free the cursor, keeping its motion events for ImGui
		IN_FreeCursorForGui ();
		SDL_ShowCursor (SDL_DISABLE);
		QR_GUI_SetMouseCursor (1);
	}
	else
	{
		QRE_ClosePanel ();
	}
}

// Hover (flying) or select (fire button) a light by its wireframe.
static void QRE_DoLightPick (qboolean select)
{
	const rt_tracked_light_t *lights;
	int count = 0, index = QRE_LightUnderCrosshair ();

	if (!select)
	{
		qre.hover_light = index;
		return;
	}
	if (index < 0)
		return;

	lights = RT_TRACK_Lights (&count);
	if (index >= count)
		return;

	qre.sel_light = lights[index];
	qre.sel_light_valid = true;
	qre.hover_light = -1;
	q_strlcpy (qre.pick_name, qre.sel_light.name, sizeof (qre.pick_name));
	qre.pick_glt = NULL;
	qre.pick_surf = NULL;
	qre.pick_model = NULL;
	qre.pick_ent = NULL;

	Con_Printf ("qr light editor: picked light '%s' (%s)\n",
	            qre.sel_light.name[0] ? qre.sel_light.name : "(no emitter name)",
	            qre.sel_light.kind == RT_LIGHT_KIND_MATERIAL ? "material" :
	            qre.sel_light.kind == RT_LIGHT_KIND_DLIGHT ? "legacy dlight" : "map light");

	QRE_CursorMode (true);
}

// Keeps the selected light's live values in step with the frame's uploads.
static void QRE_RefreshSelectedLight (void)
{
	const rt_tracked_light_t *lights;
	int count = 0, i;

	if (!qre.sel_light_valid)
		return;

	lights = RT_TRACK_Lights (&count);
	for (i = 0; i < count; i++)
	{
		if (!lights[i].ready)
			continue;
		if (lights[i].uniqueID == qre.sel_light.uniqueID && lights[i].kind == qre.sel_light.kind)
		{
			qre.sel_light = lights[i];
			q_strlcpy (qre.pick_name, qre.sel_light.name, sizeof (qre.pick_name));
			return;
		}
	}
}

// The wireframes of the frame's lights: cyan, the hovered one amber and the
// selected one white, in one line-list upload.
static void QRE_DrawLightWireframes (void)
{
	const rt_tracked_light_t *lights;
	const float               pi = 3.14159265f;
	int                       count = 0, i, seg, axis, drawn = 0;
	RgVertex                 *rv;
	uint32_t                 *ri;
	byte                     *block;
	size_t                    verts_bytes, ri_bytes;

	lights = RT_TRACK_Lights (&count);
	if (count > QRE_LIGHT_WIRE_MAX)
		count = QRE_LIGHT_WIRE_MAX;
	if (count <= 0)
		return;

	verts_bytes = (size_t)count * 3 * (QRE_LIGHT_WIRE_SEGS + 1) * sizeof (RgVertex);
	ri_bytes    = (size_t)count * 3 * QRE_LIGHT_WIRE_SEGS * 2 * sizeof (uint32_t);
	block = (byte *)Mem_Alloc (verts_bytes + ri_bytes);
	rv = (RgVertex *)block;
	ri = (uint32_t *)(block + verts_bytes);

	for (i = 0; i < count; i++)
	{
		const rt_tracked_light_t *l = &lights[i];
		const float               r = l->radius;
		const int                 base = drawn * 3 * (QRE_LIGHT_WIRE_SEGS + 1);
		uint32_t                  color;

		if (!l->ready)
			continue;

		if (qre.sel_light_valid && l->uniqueID == qre.sel_light.uniqueID && l->kind == qre.sel_light.kind)
			color = RT_PackColorToUint32 (255, 255, 255, 255);
		else if (i == qre.hover_light)
			color = RT_PackColorToUint32 (255, 214, 64, 255);
		else
			color = RT_PackColorToUint32 (0, 255, 255, 200);

		for (axis = 0; axis < 3; axis++)
		{
			const int c1 = (axis + 1) % 3;
			const int c2 = (axis + 2) % 3;

			for (seg = 0; seg <= QRE_LIGHT_WIRE_SEGS; seg++)
			{
				const float a = (float)seg / (float)QRE_LIGHT_WIRE_SEGS * 2.0f * pi;
				const int   p = base + axis * (QRE_LIGHT_WIRE_SEGS + 1) + seg;

				rv[p].position[0] = l->position[0];
				rv[p].position[1] = l->position[1];
				rv[p].position[2] = l->position[2];
				rv[p].position[c1] += cosf (a) * r;
				rv[p].position[c2] += sinf (a) * r;
				rv[p].packedColor = color;
			}
		}
		for (axis = 0; axis < 3; axis++)
		{
			for (seg = 0; seg < QRE_LIGHT_WIRE_SEGS; seg++)
			{
				const int p = drawn * 3 * QRE_LIGHT_WIRE_SEGS + axis * QRE_LIGHT_WIRE_SEGS + seg;

				ri[p * 2 + 0] = (uint32_t)(base + axis * (QRE_LIGHT_WIRE_SEGS + 1) + seg);
				ri[p * 2 + 1] = (uint32_t)(base + axis * (QRE_LIGHT_WIRE_SEGS + 1) + seg + 1);
			}
		}
		drawn++;
	}

	if (drawn > 0)
	{
		RgRasterizedGeometryUploadInfo info = {
			.renderType = RG_RASTERIZED_GEOMETRY_RENDER_TYPE_SWAPCHAIN,
			.vertexCount = (uint32_t)(drawn * 3 * (QRE_LIGHT_WIRE_SEGS + 1)),
			.pVertices = rv,
			.indexCount = (uint32_t)(drawn * 3 * QRE_LIGHT_WIRE_SEGS * 2),
			.pIndices = ri,
			.transform = RT_TRANSFORM_IDENTITY,
			.color = RT_COLOR_WHITE,
			.material = RG_NO_MATERIAL,
			.pipelineState = RG_RASTERIZED_GEOMETRY_STATE_FORCE_LINE_LIST,
			.blendFuncSrc = 0,
			.blendFuncDst = 0,
		};
		RgResult res = rgUploadRasterizedGeometry (vulkan_globals.instance, &info, NULL, NULL);

		RG_CHECK (res);
	}

	Mem_Free (block);
}

// Hover pick (crosshair, flying) or select pick (fire button).
static void QRE_DoPick (qboolean select)
{
	qmodel_t    *model;
	msurface_t  *surf;
	entity_t    *ent;
	gltexture_t *glt;

	// the light editor aims at the lights themselves, not at the surfaces
	if (qre.mode == QRE_MODE_LIGHT)
	{
		QRE_DoLightPick (select);
		return;
	}

	if (!QRE_TracePick (&model, &surf, &ent, &glt))
	{
		qre.hover_model = NULL;
		qre.hover_surf = NULL;
		qre.hover_ent = NULL;
		qre.hover_glt = NULL;
		return;
	}

	if (!select)
	{
		qre.hover_model = model;
		qre.hover_surf = surf;
		qre.hover_ent = ent;
		qre.hover_glt = glt;
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
	qre.pick_glt = glt;
	qre.hover_model = NULL;
	qre.hover_surf = NULL;
	qre.hover_ent = NULL;
	qre.hover_glt = NULL;

	{
		char texname[MAX_QPATH];
		char *dot;
		int   i;

		RT_MAT_NormalizeName (glt->name, texname, sizeof (texname));
		dot = strrchr (texname, '.');
		if (dot && !strchr (dot, ':'))
			*dot = '\0';

		q_strlcpy (qre.pick_name, texname, sizeof (qre.pick_name));

		if (qre.mode == QRE_MODE_LIGHT)
		{
			Con_Printf ("qr light editor: picked emitter '%s'\n", texname);
		}
		else
		{
			QRE_ResolveGroup (texname);

			Con_Printf ("qr editor: picked '%s' (%d material(s) in the group)\n", texname, qre.group_count);
			for (i = 0; i < qre.group_count; i++)
				Con_Printf ("qr editor:   group material '%s'\n", qre.group[i]->name);
		}
	}

	// the panel owns the mouse: free the cursor (keeping its motion events
	// for ImGui), freeze the camera
	QRE_CursorMode (true);
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

	// The face plane in world space: the rotation of a row-major RgTransform is
	// applied the same way ApplyTransform (r_world.c) applies it.
	for (j = 0; j < 3; j++)
		n_world[j] = transform.matrix[j][0] * surf->plane->normal[0]
		           + transform.matrix[j][1] * surf->plane->normal[1]
		           + transform.matrix[j][2] * surf->plane->normal[2];

	// Nudge the outline off the face towards the viewer: the plane normal of a
	// SURF_PLANEBACK face points away from its visible side, and pushing the
	// line behind the wall would lose it to the traced surface.
	for (j = 0; j < 3; j++)
		first[j] = transform.matrix[j][0] * verts[0][0]
		         + transform.matrix[j][1] * verts[0][1]
		         + transform.matrix[j][2] * verts[0][2]
		         + transform.matrix[j][3];
	VectorSubtract (r_origin, first, to_view);
	if (DotProduct (to_view, n_world) < 0.0f)
		nudge = -nudge;

	// Model -> world (a brush entity carries its own transform), then off the
	// face; the vertices are uploaded in world space.
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < 3; j++)
			rv[i].position[j] = transform.matrix[j][0] * verts[i][0]
			                  + transform.matrix[j][1] * verts[i][1]
			                  + transform.matrix[j][2] * verts[i][2]
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

// The outline of a model pick: the entity's model bounds as a wire box, drawn
// with the same overlay path as a face outline.
static void QRE_EmitBoxOutline (entity_t *ent, uint32_t color)
{
	static const int edges[12][2] = {
		{ 0, 1 }, { 1, 3 }, { 3, 2 }, { 2, 0 },
		{ 4, 5 }, { 5, 7 }, { 7, 6 }, { 6, 4 },
		{ 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 },
	};
	float       m[16];
	RgTransform transform;
	vec3_t      mins, maxs;
	RgVertex   *rv;
	uint32_t   *ri;
	byte       *block;
	int         i, j;

	if (!ent || !ent->model)
		return;

	if (ent->model->type == mod_sprite)
	{
		mspriteframe_t *frame = R_GetSpriteFrame (ent);
		RgRasterizedGeometryUploadInfo sfinfo;
		vec3_t      corner[4];
		RgVertex   *srv;
		uint32_t   *sri;
		byte       *sblock;
		RgResult    r;
		int         k;

		if (!frame)
			return;

		sblock = (byte *)RT_AllocScratchMemoryNulled (4 * sizeof (RgVertex) + 8 * sizeof (uint32_t));
		srv = (RgVertex *)sblock;
		sri = (uint32_t *)(sblock + 4 * sizeof (RgVertex));

		// the quad the renderer draws, axes and all (R_CreateSpriteVertices)
		R_GetSpriteQuadCorners (ent, frame, corner);

		for (k = 0; k < 4; k++)
		{
			VectorCopy (corner[k], srv[k].position);
			srv[k].packedColor = color;
		}
		for (k = 0; k < 4; k++)
		{
			sri[k * 2 + 0] = (uint32_t)k;
			sri[k * 2 + 1] = (uint32_t)((k + 1) & 3);
		}

		memset (&sfinfo, 0, sizeof (sfinfo));
		sfinfo.renderType = RG_RASTERIZED_GEOMETRY_RENDER_TYPE_SWAPCHAIN;
		sfinfo.vertexCount = 4;
		sfinfo.pVertices = srv;
		sfinfo.indexCount = 8;
		sfinfo.pIndices = sri;
		sfinfo.transform.matrix[0][0] = sfinfo.transform.matrix[1][1] = sfinfo.transform.matrix[2][2] = 1.0f;
		sfinfo.color.data[0] = sfinfo.color.data[1] = sfinfo.color.data[2] = sfinfo.color.data[3] = 1.0f;
		sfinfo.material = RG_NO_MATERIAL;
		sfinfo.pipelineState = RG_RASTERIZED_GEOMETRY_STATE_FORCE_LINE_LIST;

		r = rgUploadRasterizedGeometry (vulkan_globals.instance, &sfinfo, NULL, NULL);
		RG_CHECK (r);
		return;
	}

	VectorCopy (ent->model->mins, mins);
	VectorCopy (ent->model->maxs, maxs);
	if (mins[0] == maxs[0] && mins[1] == maxs[1] && mins[2] == maxs[2])
	{
		mins[0] = mins[1] = mins[2] = -16.0f;
		maxs[0] = maxs[1] = maxs[2] = 16.0f;
	}

	IdentityMatrix (m);
	R_RotateForEntity (m, ent->origin, ent->angles);
	transform = RT_GetModelTransform (m);

	block = (byte *)RT_AllocScratchMemoryNulled (8 * sizeof (RgVertex) + 24 * sizeof (uint32_t));
	rv = (RgVertex *)block;
	ri = (uint32_t *)(block + 8 * sizeof (RgVertex));

	for (i = 0; i < 8; i++)
	{
		vec3_t local;

		local[0] = (i & 1) ? maxs[0] : mins[0];
		local[1] = (i & 2) ? maxs[1] : mins[1];
		local[2] = (i & 4) ? maxs[2] : mins[2];

		for (j = 0; j < 3; j++)
		{
			rv[i].position[j] = transform.matrix[j][0] * local[0]
			                  + transform.matrix[j][1] * local[1]
			                  + transform.matrix[j][2] * local[2]
			                  + transform.matrix[j][3];
		}
		rv[i].packedColor = color;
	}

	for (i = 0; i < 12; i++)
	{
		ri[i * 2 + 0] = (uint32_t)edges[i][0];
		ri[i * 2 + 1] = (uint32_t)edges[i][1];
	}

	RgRasterizedGeometryUploadInfo info = {
		.renderType = RG_RASTERIZED_GEOMETRY_RENDER_TYPE_SWAPCHAIN,
		.vertexCount = 8,
		.pVertices = rv,
		.indexCount = 24,
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

	if (qre.mode == QRE_MODE_LIGHT)
	{
		QRE_DrawLightWireframes ();
		return;
	}

	if (qre.panel_open && (qre.pick_surf || (qre.pick_ent && qre.pick_glt)))
	{
		const uint32_t color = RT_PackColorToUint32 (255, 255, 255, 255);

		if (qre.pick_surf)
			QRE_EmitOutline (qre.pick_model, qre.pick_surf, qre.pick_ent, color);
		else
			QRE_EmitBoxOutline (qre.pick_ent, color);
	}
	else if (!qre.panel_open && qre.hover_surf)
	{
		QRE_EmitOutline (qre.hover_model, qre.hover_surf, qre.hover_ent, RT_PackColorToUint32 (255, 214, 64, 255));
	}
	else if (!qre.panel_open && qre.hover_surf == NULL && qre.hover_ent && qre.hover_glt)
	{
		QRE_EmitBoxOutline (qre.hover_ent, RT_PackColorToUint32 (255, 214, 64, 255));
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

	if (QR_Editor_Flying ())
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

// ---------------------------------------------------------------------------
// Texture preview (the emissive colour picker)
// ---------------------------------------------------------------------------

static void QRE_FreePreview (void)
{
	int i;

	for (i = 0; i < QRE_PREVIEW_SLOTS; i++)
	{
		qre_preview_t *slot = &qre.preview[i];

		if (slot->mat != RG_NO_MATERIAL)
			rgDestroyMaterial (vulkan_globals.instance, slot->mat);
		if (slot->pixels)
			Mem_Free (slot->pixels);
		memset (slot, 0, sizeof (*slot));
	}
}

// The pixels the emissive mask is synthesized from: the author's texture_base
// when the material has one, the engine texture of the picked face otherwise.
// The eyedropper samples this copy, so a picked colour is the colour the
// synthesis will match, and the same pixels become the preview's RgMaterial.
// A slot stays alive while the frame still draws it; a failed load is cached
// too, so a texture that cannot be read is not decoded once per frame.
static qre_preview_t *QRE_PreviewFor (rt_material_t *m)
{
	char          key[MAX_QPATH * 2 + 32];
	qre_preview_t *slot;
	byte         *pixels = NULL;
	int           i, victim = -1;
	int           w = 0, h = 0;

	if (!qre.pick_glt)
		return NULL;

	q_snprintf (key, sizeof (key), "%s|%s|%p", m->name, m->filename_base, (void *)qre.pick_glt);

	for (i = 0; i < QRE_PREVIEW_SLOTS; i++)
	{
		if (!strcmp (key, qre.preview[i].key))
		{
			qre.preview[i].last_frame = (unsigned)host_framecount;
			return (qre.preview[i].mat != RG_NO_MATERIAL) ? &qre.preview[i] : NULL;
		}
	}

	// an empty slot, or the least recently used one this frame has not drawn
	for (i = 0; i < QRE_PREVIEW_SLOTS; i++)
	{
		if (qre.preview[i].last_frame == (unsigned)host_framecount)
			continue;
		if (victim < 0 || qre.preview[i].last_frame < qre.preview[victim].last_frame)
			victim = i;
	}
	if (victim < 0)
		return NULL;

	slot = &qre.preview[victim];

	if (slot->mat != RG_NO_MATERIAL)
	{
		rgDestroyMaterial (vulkan_globals.instance, slot->mat);
		slot->mat = RG_NO_MATERIAL;
	}
	if (slot->pixels)
	{
		Mem_Free (slot->pixels);
		slot->pixels = NULL;
	}
	slot->w = slot->h = 0;
	q_strlcpy (slot->key, key, sizeof (slot->key));
	slot->last_frame = (unsigned)host_framecount;

	if (m->filename_base[0])
		pixels = RT_MAT_LoadTexture (m, RT_MAT_TEX_BASE, &w, &h);
	if (!pixels)
		pixels = TexMgr_LoadRgbaForPreview (qre.pick_glt, &w, &h);

	if (!pixels || w <= 0 || h <= 0)
	{
		if (pixels)
			Mem_Free (pixels);
		return NULL;
	}

	{
		RgMaterialCreateInfo info;

		memset (&info, 0, sizeof (info));
		info.size.width = (uint32_t)w;
		info.size.height = (uint32_t)h;
		info.textures.pDataAlbedoAlpha = pixels;
		info.filter = RG_SAMPLER_FILTER_LINEAR;
		info.addressModeU = RG_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		info.addressModeV = RG_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

		if (rgCreateMaterial (vulkan_globals.instance, &info, &slot->mat) != RG_SUCCESS)
		{
			slot->mat = RG_NO_MATERIAL;
			Mem_Free (pixels);
			return NULL;
		}
	}

	slot->pixels = pixels;
	slot->w = w;
	slot->h = h;
	return slot;
}

// ---------------------------------------------------------------------------
// Per-parameter reset
// ---------------------------------------------------------------------------

// The state a parameter is reset to: the snapshot taken on start (retaken by
// Apply), or the defaults for a material the editor created in this session.
static const rt_material_t *QRE_OriginalOf (const rt_material_t *m, rt_material_t *defbuf)
{
	int i;

	for (i = 0; i < qre.snap_map_count; i++)
	{
		if (!strcmp (qre.snap_map[i].name, m->name))
			return &qre.snap_map[i];
	}
	for (i = 0; i < qre.snap_global_count; i++)
	{
		if (!strcmp (qre.snap_global[i].name, m->name))
			return &qre.snap_global[i];
	}

	QRE_InitDefault (defbuf, m->name);
	return defbuf;
}

static qboolean QRE_ParamChanged (const rt_material_t *m, const rt_material_t *orig, int p)
{
	// the metalness checkbox is the factor's "authored" flag: it is a change
	// by itself, even when the value happens to match
	if (p == PARAM_METAL)
		return m->has_metalness_factor != orig->has_metalness_factor ||
		       m->metalness_factor != orig->metalness_factor;

	if (p == PARAM_CEMIS)
	{
		// every block's colour and tone controls count as one parameter
		return m->has_color_emissive != orig->has_color_emissive ||
		       memcmp (m->color_emissive, orig->color_emissive, sizeof (m->color_emissive)) != 0;
	}

	switch (qre_params[p].type)
	{
	case QRE_T_FLOAT:  return QRE_GetFloat (m, p) != QRE_GetFloat (orig, p);
	case QRE_T_INT:    return QRE_GetInt (m, p) != QRE_GetInt (orig, p);
	case QRE_T_BOOL:   return QRE_GetBool (m, p) != QRE_GetBool (orig, p);
	case QRE_T_TEXT:   return strcmp (QRE_GetText (m, p), QRE_GetText (orig, p)) != 0;
	case QRE_T_COLOR:
	{
		qboolean e1, e2;
		float    c1[3], c2[3];

		QRE_GetColor (m, p, &e1, c1);
		QRE_GetColor (orig, p, &e2, c2);
		return e1 != e2 || c1[0] != c2[0] || c1[1] != c2[1] || c1[2] != c2[2];
	}
	default:
		return false;
	}
}

static void QRE_ResetParam (int g, int p, const rt_material_t *orig)
{
	rt_material_t *m;

	if (p == PARAM_METAL && !orig->has_metalness_factor)
	{
		// QRE_SetFloat would author the factor; the original never had one
		QRE_EnsureLive (g);
		m = qre.group[g];
		m->has_metalness_factor = false;
		m->metalness_factor = orig->metalness_factor;
		QRE_MarkDirty (m);
		return;
	}

	if (p == PARAM_MIRROR)
	{
		// mirror forces roughness_override to 0 while it is on; un-mirroring
		// brings the original override back with it
		QRE_SetBool (g, p, QRE_GetBool (orig, p));
		if (!QRE_GetBool (orig, p))
			QRE_SetFloat (g, PARAM_ROUGH, QRE_GetFloat (orig, PARAM_ROUGH));
		return;
	}

	if (p == PARAM_CEMIS)
	{
		QRE_EnsureLive (g);
		m = qre.group[g];
		m->has_color_emissive = orig->has_color_emissive;
		m->color_emissive_count = orig->color_emissive_count;
		m->emissive_blend = orig->emissive_blend;
		memcpy (m->color_emissive, orig->color_emissive, sizeof (m->color_emissive));
		QRE_MarkDirty (m);
		return;
	}

	switch (qre_params[p].type)
	{
	case QRE_T_FLOAT:
		QRE_SetFloat (g, p, QRE_GetFloat (orig, p));
		break;
	case QRE_T_INT:
		QRE_SetInt (g, p, QRE_GetInt (orig, p));
		break;
	case QRE_T_BOOL:
		QRE_SetBool (g, p, QRE_GetBool (orig, p));
		break;
	case QRE_T_TEXT:
		QRE_SetText (g, p, QRE_GetText (orig, p));
		break;
	case QRE_T_COLOR:
	{
		qboolean enabled;
		float    rgb[3];
		int      c;

		// the channels are restored even behind a disabled colour: enabling it
		// again has to show the original tint, not the last one edited
		QRE_GetColor (orig, p, &enabled, rgb);
		for (c = 0; c < 3; c++)
			QRE_SetColorChannel (g, p, c, rgb[c]);
		QRE_SetColorEnabled (g, p, enabled);
		break;
	}
	default:
		break;
	}
}

// The Emissive section of a material: the eyedropper preview, the colour list
// and the "+" that adds one. The preview comes first so adding colours never
// pushes it off the panel, and a click picks into the last colour — adding a
// colour is the button's job.
static void QRE_EmissiveEditor (int g)
{
	rt_material_t *m = qre.group[g];
	int            ci;

	for (ci = 0; ci < m->color_emissive_count; ci++)
	{
		char     id[32];
		char     label[32];
		qboolean open;

		q_snprintf (id, sizeof (id), "cemis%d", ci);
		q_snprintf (label, sizeof (label), "Colour %d", ci + 1);

		QR_GUI_PushID (id);
		open = QR_GUI_Section (label, 1) ? true : false;

		if (open)
		{
			const int res = QR_GUI_ColorRow ("##color", m->color_emissive[ci].color,
			                                 "The colour this block matches, up to the threshold's distance.");

			if (res & 1)
				QRE_MarkDirty (m);
			if (res & 2)
			{
				int k;

				for (k = ci; k + 1 < m->color_emissive_count; k++)
					m->color_emissive[k] = m->color_emissive[k + 1];
				m->color_emissive_count--;
				if (m->color_emissive_count == 0)
					m->has_color_emissive = false;
				QRE_MarkDirty (m);
				QR_GUI_PopID ();
				break;
			}

			// this block's own preview: the eyedropper picks a colour into it
			{
				qre_preview_t *slot = QRE_PreviewFor (m);

				if (slot)
				{
					float u = 0.5f, v = 0.5f;

					if (QR_GUI_ImagePick ("##color_pick", (int64_t)slot->mat, slot->w, slot->h, &u, &v))
					{
						const int   px = CLAMP (0, (int)(u * (float)slot->w), slot->w - 1);
						const int   py = CLAMP (0, (int)(v * (float)slot->h), slot->h - 1);
						const byte *pix = slot->pixels + ((size_t)py * (size_t)slot->w + (size_t)px) * 4;

						QRE_EnsureLive (g);
						m = qre.group[g];
						m->color_emissive[ci].color[0] = pix[0] / 255.0f;
						m->color_emissive[ci].color[1] = pix[1] / 255.0f;
						m->color_emissive[ci].color[2] = pix[2] / 255.0f;
						QRE_MarkDirty (m);
					}
				}
			}

			if (QR_GUI_SliderFloat ("color_emissive_threshold", &m->color_emissive[ci].threshold, 0.0f, 1.0f,
			                        "How far a pixel's colour may differ from this block's colour and still glow."))
				QRE_MarkDirty (m);
			if (QR_GUI_SliderFloat ("color_emissive_feather", &m->color_emissive[ci].feather, 0.0f, 16.0f,
			                        "Softens this block's mask edge over this many pixels, on both sides of it, without comparing colours."))
				QRE_MarkDirty (m);
			if (QR_GUI_SliderFloat ("emissive_factor", &m->color_emissive[ci].factor, 0.0f, 4.0f,
			                        "Scales this block's emission: below 1 it dims, above 1 it brightens."))
				QRE_MarkDirty (m);
			{
				int idx = m->color_emissive[ci].blend + 1;

				if (QR_GUI_Combo ("emissive_blend", &idx, qre_emissive_blends, (int)countof (qre_emissive_blends),
				                  "How this block's glow is composited into the frame (normal, screen, overlay...)."))
				{
					m->color_emissive[ci].blend = idx - 1;
					QRE_MarkDirty (m);
				}
			}
		}

		QR_GUI_PopID ();
	}

	if (m->color_emissive_count < RT_MAT_MAX_EMISSIVE_COLORS)
	{
		if (QR_GUI_Button ("+"))
		{
			rt_emissive_t *block;

			QRE_EnsureLive (g);
			m = qre.group[g];
			block = &m->color_emissive[m->color_emissive_count];
			memset (block, 0, sizeof (*block));
			block->color[0] = 1.0f;
			block->threshold = (m->color_emissive_threshold > 0.0f) ? m->color_emissive_threshold : 0.02f;
			block->feather = m->color_emissive_feather;
			block->factor = m->emissive_factor;
			block->blend = m->emissive_blend;
			block->has_threshold = block->has_feather = true;
			block->has_factor = block->has_blend = true;
			m->color_emissive_count++;
			m->has_color_emissive = true;
			QRE_MarkDirty (m);
		}
		QR_GUI_Tooltip ("Add a colour block (up to ten)");
	}
}

static void QRE_ParamWidgets (int g)
{
	rt_material_t        defbuf;
	const rt_material_t *orig = QRE_OriginalOf (qre.group[g], &defbuf);
	int                  p;

	for (p = 0; p < PARAM_COUNT; p++)
	{
		const char   *label = qre_params[p].label;
		const char   *tip   = qre_params[p].tip;
		// re-read every iteration: the first change of a material that has no
		// yaml entry moves qre.group[g] into the live list (QRE_EnsureLive)
		rt_material_t *m = qre.group[g];
		// mirror drives roughness on its own (the synthesis gives it the last
		// word): the override is meaningless there, and is locked at 0
		const qboolean mirror_locks_rough = (p == PARAM_ROUGH && m->mirror);

		if (mirror_locks_rough)
			QR_GUI_PushDisabled (1);

		switch (qre_params[p].type)
		{
		case QRE_T_TEXT:
		{
			char buf[MAX_QPATH];
			int  res;
			char file[MAX_QPATH];

			q_strlcpy (buf, QRE_GetText (m, p), sizeof (buf));
			res = QR_GUI_TexturePath (label, buf, sizeof (buf), tip);
			if (res & 1)
			{
				// NONE typed by hand means "no texture", as an empty field does.
				// Normalize before the compare: typing NONE into an empty field
				// is not an edit, and it must not create the material.
				if (!q_strcasecmp (buf, "NONE"))
					buf[0] = '\0';
				if (strcmp (buf, QRE_GetText (m, p)))
					QRE_SetText (g, p, buf);
			}
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
			if (QR_GUI_SliderFloat (label, &value, qre_params[p].min, qre_params[p].max, tip))
			{
				// the panel shows two decimals; a value out of the slider (or
				// typed) is snapped to that grid. A reset does not pass through
				// here, so it restores the snapshot exactly.
				QRE_SetFloat (g, p, roundf (value * 100.0f) / 100.0f);
			}
			break;
		}
		case QRE_T_INT:
		{
			int value = QRE_GetInt (m, p);

			if (QR_GUI_SliderInt (label, &value, (int)qre_params[p].min, (int)qre_params[p].max, tip))
				QRE_SetInt (g, p, value);
			break;
		}
		case QRE_T_BOOL:
		{
			int value = QRE_GetBool (m, p) ? 1 : 0;
			if (QR_GUI_Checkbox (label, &value, tip))
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

			if (QR_GUI_ColorHex (label, rgb, &en, tip))
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

		// Reset one parameter to the state it had when the editor started
		// (or to the defaults, for a material the editor created itself).
		m = qre.group[g];
		if (QR_GUI_ResetButton (label, !mirror_locks_rough && QRE_ParamChanged (m, orig, p)))
			QRE_ResetParam (g, p, orig);

		// The Emissive section sits right under the color_emissive checkbox row
		// (its own row is the one just drawn); without the checkbox there is no
		// section.
		if (p == PARAM_CEMIS && m->has_color_emissive)
		{
			if (QR_GUI_Section ("Emissive", 1))
				QRE_EmissiveEditor (g);
		}

		if (mirror_locks_rough)
			QR_GUI_PopDisabled ();
	}
}

static void QRE_BuildPanelGUI (void)
{
	int      panel_w = glwidth / 4; // a quarter of the screen wide, as asked
	int      g;
	qboolean exit_requested = false;

	// Below this the fixed label column and the browse and reset buttons stop
	// fitting: a quarter of a small window is not worth an unusable panel.
	if (panel_w < 352)
		panel_w = 352;

	QR_GUI_BeginPanel ("qr_material_editor", glwidth - panel_w, 0, panel_w, glheight);

	QR_GUI_Label ("MATERIAL EDITOR");
	if (qre.pick_glt)
	{
		char buf[MAX_QPATH + 16];

		q_snprintf (buf, sizeof (buf), qre.pick_surf ? "face: %s" : "model: %s",
		            qre.pick_name[0] ? qre.pick_name : qre.pick_glt->name);
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
		QRE_RequestExit ();
}

// ---------------------------------------------------------------------------
// The light editor's entries: fields, their originals, and the group
// ---------------------------------------------------------------------------

enum
{
	QRE_LIGHT_F_RADIUS = 0,
	QRE_LIGHT_F_INTENSITY,
	QRE_LIGHT_F_OFFSET,
	QRE_LIGHT_F_COLOR,
	QRE_LIGHT_F_FRAST,
};

// The state the entry had when the editor started (NULL when it had none).
static const rt_light_t *QRE_LightOriginal (const char *name)
{
	int i;

	for (i = 0; i < qre.snap_light_count; i++)
	{
		if (qre.snap_lights[i].valid && !strcmp (qre.snap_lights[i].name, name))
			return &qre.snap_lights[i];
	}
	return NULL;
}

// One field of one entry.
static void QRE_LightApply (rt_light_t *l, int field, float v0, float v1, float v2, qboolean b)
{
	switch (field)
	{
	case QRE_LIGHT_F_RADIUS:    l->radius = v0; l->has_radius = true; break;
	case QRE_LIGHT_F_INTENSITY: l->intensity = v0; l->has_intensity = true; break;
	case QRE_LIGHT_F_OFFSET:
		l->offset[0] = v0; l->offset[1] = v1; l->offset[2] = v2;
		l->has_offset = true;
		break;
	case QRE_LIGHT_F_COLOR:
		if (v0 < 0.0f)
		{
			l->has_color = false; // the colour is switched off
		}
		else
		{
			l->color[0] = v0; l->color[1] = v1; l->color[2] = v2;
			l->has_color = true;
		}
		break;
	case QRE_LIGHT_F_FRAST:     l->force_rasterize = b; break;
	default: break;
	}
	QRE_TouchLight (l->name);
}

// The field of the entry as the editor started with it (the has_ flag too).
static void QRE_LightApplyOriginal (rt_light_t *l, const rt_light_t *orig, int field)
{
	if (!orig)
	{
		// the light had no entry: the field goes back to "not authored"
		switch (field)
		{
		case QRE_LIGHT_F_RADIUS:    l->has_radius = false; break;
		case QRE_LIGHT_F_INTENSITY: l->has_intensity = false; break;
		case QRE_LIGHT_F_OFFSET:    l->has_offset = false; break;
		case QRE_LIGHT_F_COLOR:     l->has_color = false; break;
		case QRE_LIGHT_F_FRAST:     l->force_rasterize = false; break;
		default: break;
		}
	}
	else
	{
		switch (field)
		{
		case QRE_LIGHT_F_RADIUS:    l->has_radius = orig->has_radius; l->radius = orig->radius; break;
		case QRE_LIGHT_F_INTENSITY: l->has_intensity = orig->has_intensity; l->intensity = orig->intensity; break;
		case QRE_LIGHT_F_OFFSET:    l->has_offset = orig->has_offset; VectorCopy (orig->offset, l->offset); break;
		case QRE_LIGHT_F_COLOR:     l->has_color = orig->has_color; VectorCopy (orig->color, l->color); break;
		case QRE_LIGHT_F_FRAST:     l->force_rasterize = orig->force_rasterize; break;
		default: break;
		}
	}
	QRE_TouchLight (l->name);
}

// A light that follows its group writes an edit to every light of the same
// group (the emitter's model, e.g. progs/flame.mdl) that also follows it --
// in the file's list and among the lights of the frame, whose entries a group
// edit creates when they are missing. A light with the flag off is its own.
static void QRE_LightApplyToGroup (const char *source_name, int field, float v0, float v1, float v2, qboolean b)
{
	char group[MAX_QPATH];
	char g2[MAX_QPATH];
	int  count = 0, i;

	RT_MAT_GroupBaseOf (source_name, group, sizeof (group));

	{
		rt_light_t *list = RT_LIGHT_List (&count);

		for (i = 0; i < count; i++)
		{
			if (!list[i].valid || !list[i].group_edit)
				continue;
			RT_MAT_GroupBaseOf (list[i].name, g2, sizeof (g2));
			if (!strcmp (g2, group))
				QRE_LightApply (&list[i], field, v0, v1, v2, b);
		}
	}

	{
		const rt_tracked_light_t *lights = RT_TRACK_Lights (&count);

		for (i = 0; i < count; i++)
		{
			rt_light_t *l;

			if (!lights[i].ready || !lights[i].name[0])
				continue;
			RT_MAT_GroupBaseOf (lights[i].name, g2, sizeof (g2));
			if (strcmp (g2, group))
				continue;

			l = RT_LIGHT_Ensure (lights[i].name);
			if (l && l->group_edit)
				QRE_LightApply (l, field, v0, v1, v2, b);
		}
	}
}

// A light that follows its group resets the field for every follower: each one
// returns to the state the editor started with (its own snapshot entry, or "not
// authored" when it had none), so the group does not keep mixed values.
static void QRE_LightResetField (rt_light_t *self, int field)
{
	char group[MAX_QPATH];
	char g2[MAX_QPATH];
	int  count = 0, i;
	rt_light_t *list;

	if (!self->group_edit)
	{
		QRE_LightApplyOriginal (self, QRE_LightOriginal (self->name), field);
		return;
	}

	list = RT_LIGHT_List (&count);
	RT_MAT_GroupBaseOf (self->name, group, sizeof (group));
	for (i = 0; i < count; i++)
	{
		if (!list[i].valid || !list[i].group_edit)
			continue;
		RT_MAT_GroupBaseOf (list[i].name, g2, sizeof (g2));
		if (!strcmp (g2, group))
			QRE_LightApplyOriginal (&list[i], QRE_LightOriginal (list[i].name), field);
	}
}

// The radius and intensity a light of this kind uses when its field is not
// authored: a material light and a legacy dlight take the dlight settings, a
// map light entity its own.
static float QRE_LightDefaultRadius (int kind)
{
	extern cvar_t rt_elight_radius;

	return CVAR_TO_FLOAT (kind == RT_LIGHT_KIND_MAP ? rt_elight_radius : rt_dlight_radius);
}

static float QRE_LightDefaultIntensity (int kind)
{
	return (kind == RT_LIGHT_KIND_MAP) ? 1.0f : CVAR_TO_FLOAT (rt_dlight_intensity);
}

// The panel's one place to write a field: the group when the light follows it,
// the light itself otherwise.
static void QRE_LightWrite (rt_light_t *l, int field, float v0, float v1, float v2, qboolean b)
{
	if (l->group_edit)
		QRE_LightApplyToGroup (l->name, field, v0, v1, v2, b);
	else
		QRE_LightApply (l, field, v0, v1, v2, b);
}

static qboolean QRE_LightFieldChanged (const rt_light_t *l, const rt_light_t *orig, int field)
{
	switch (field)
	{
	case QRE_LIGHT_F_RADIUS:
		return l->has_radius != (orig && orig->has_radius ? true : false) ||
		       (l->has_radius && orig && orig->radius != l->radius);
	case QRE_LIGHT_F_INTENSITY:
		return l->has_intensity != (orig && orig->has_intensity ? true : false) ||
		       (l->has_intensity && orig && orig->intensity != l->intensity);
	case QRE_LIGHT_F_OFFSET:
		return l->has_offset != (orig && orig->has_offset ? true : false) ||
		       (l->has_offset && orig && memcmp (orig->offset, l->offset, sizeof (l->offset)) != 0);
	case QRE_LIGHT_F_COLOR:
		return l->has_color != (orig && orig->has_color ? true : false) ||
		       (l->has_color && orig && memcmp (orig->color, l->color, sizeof (l->color)) != 0);
	case QRE_LIGHT_F_FRAST:
		return l->force_rasterize != (orig ? orig->force_rasterize : false);
	default:
		return false;
	}
}

// A member of the same group that follows it: the source of the values a light
// joining the group takes.
static rt_light_t *QRE_LightGroupShared (const char *name, const rt_light_t *self)
{
	char group[MAX_QPATH];
	char g2[MAX_QPATH];
	int  count = 0, i;
	rt_light_t *list = RT_LIGHT_List (&count);

	RT_MAT_GroupBaseOf (name, group, sizeof (group));
	for (i = 0; i < count; i++)
	{
		if (!list[i].valid || !list[i].group_edit || &list[i] == self)
			continue;
		if (!RT_LIGHT_HasFields (&list[i]))
			continue; // a member with nothing authored has no values to share
		RT_MAT_GroupBaseOf (list[i].name, g2, sizeof (g2));
		if (!strcmp (g2, group))
			return &list[i];
	}
	return NULL;
}

static void QRE_LightJoinGroup (rt_light_t *l)
{
	rt_light_t *shared = QRE_LightGroupShared (l->name, l);

	if (shared)
	{
		l->has_radius = shared->has_radius;
		l->radius = shared->radius;
		l->has_intensity = shared->has_intensity;
		l->intensity = shared->intensity;
		l->has_offset = shared->has_offset;
		VectorCopy (shared->offset, l->offset);
		l->has_color = shared->has_color;
		VectorCopy (shared->color, l->color);
		l->force_rasterize = shared->force_rasterize;
	}
	l->group_edit = true;
	QRE_TouchLight (l->name);
}

// The light editor's panel: the dlight of the picked emitter. Its fields live in
// lights.yaml (radius, intensity, offset); an emitter without an entry shows the
// global defaults, and authoring a value creates one.
// ---------------------------------------------------------------------------
// The global tab of the light editor: the sky, its clouds and the sun. These
// are engine cvars (the colours are cvars behind a console command, as the rest
// of the renderer uses them), so an edit takes effect at once and is archived
// in the config; Apply and Cancel do not own them.
// ---------------------------------------------------------------------------

enum
{
	QRE_G_BOOL,
	QRE_G_FLOAT,
	QRE_G_INT,
	QRE_G_COLOR,
	QRE_G_BUTTON, // a press sets the cvar in `action` (the row name is its caption)
};

// One row: the cvar, its kind and the range of its slider. A section name opens
// a group; the rows under it belong to it until the next name. A button row
// carries its caption in `name` and the cvar it writes in `action`.
typedef struct
{
	const char *section;
	const char *name;
	int         type;
	float       min, max;
	const char *tip;
	const char *action;
} qre_global_t;

static const qre_global_t qre_globals[] = {
	{ "Sky", "rt_sky",              QRE_G_FLOAT, 0, 8,
	  "Intensity of the sky; the classic sky texture is scaled by it." },
	{ NULL,  "rt_physical_sky",     QRE_G_BOOL,  0, 0,
	  "1 draws the procedural sky (painted in rt_sky_color, with clouds and a sun disc); 0 draws the classic sky texture." },
	{ NULL,  "rt_sky_brightness",   QRE_G_FLOAT, 0, 4,
	  "Brightness of the procedural sky." },
	{ NULL,  "rt_sky_color",        QRE_G_COLOR, 0, 0,
	  "The colour the procedural sky is painted in, and the colour of the light it casts." },
	{ NULL,  "rt_sky_ambient_lod",  QRE_G_INT,   0, 10,
	  "Mip level the ambient sky light is read from: lower is more directional, 10 a flat wash." },
	{ NULL,  "rt_sky_nee",          QRE_G_BOOL,  0, 0,
	  "Sample the sky as an explicit light source." },

	{ "Clouds", "rt_sky_clouds",        QRE_G_BOOL,  0, 0,
	  "Draw the volumetric clouds." },
	{ NULL,  "rt_sky_clouds_color",     QRE_G_COLOR, 0, 0,
	  "The colour the clouds are drawn in; they may be darker than the sky." },
	{ NULL,  "rt_sky_cloud_alpha",      QRE_G_FLOAT, 0, 1,
	  "Opacity the clouds are composited over the sky with; 0 takes them out." },
	{ NULL,  "rt_sky_cloud_coverage",   QRE_G_FLOAT, 0, 1,
	  "How much of the sky the clouds cover." },
	{ NULL,  "rt_sky_cloud_density",    QRE_G_FLOAT, 0, 1,
	  "Sharpness of the cloud contour: 1 a hard edge, 0 a soft one." },
	{ NULL,  "rt_sky_cloud_speed",      QRE_G_FLOAT, 0, 4,
	  "How fast the cloud layer drifts." },

	{ "Sun", "rt_sun",              QRE_G_FLOAT, 0, 10,
	  "Strength of the sun: 1 is a usable daylight, 0 turns it off." },
	{ NULL,  "rt_sun_color",        QRE_G_COLOR, 0, 0,
	  "The colour of the sun: its light, the disc in the procedural sky and everything that reads it (the indirect sun, the god rays, the fog's shafts)." },
	{ NULL,  "rt_sun_pitch",        QRE_G_FLOAT, -180, 180,
	  "The pitch the sun stands at." },
	{ NULL,  "rt_sun_yaw",          QRE_G_FLOAT, -180, 180,
	  "The yaw the sun stands at." },
	{ NULL,  "Set sun position",    QRE_G_BUTTON, 0, 0,
	  "Place the sun by aiming: it follows the crosshair, and the fire button leaves it where it points (that press is swallowed).",
	  "rt_sun_edit" },

	{ "God rays", "rt_godrays",         QRE_G_BOOL,  0, 0,
	  "Draw the sun shafts." },
	{ NULL,  "rt_godrays_intensity",    QRE_G_FLOAT, 0, 4,
	  "Strength of the sun shafts." },
	{ NULL,  "rt_volume_lintensity",    QRE_G_FLOAT, 0, 1000,
	  "Intensity of the light the fog scatters: the shafts that are visible in it." },
	{ NULL,  "rt_volume_lassymetry",    QRE_G_FLOAT, -0.95f, 0.95f,
	  "How much the fog scatters forward (positive) or back (negative); 0 scatters evenly." },
};

static void QRE_GlobalColorGet (const char *name, float rgb[3])
{
	if (!strcmp (name, "rt_sky_color"))
		RT_GetSkyColor (rgb);
	else if (!strcmp (name, "rt_sun_color"))
		RT_GetSunColor (rgb);
	else
		RT_GetSkyCloudsColor (rgb);
}

static void QRE_GlobalColorSet (const char *name, const float rgb[3])
{
	Cvar_Set (name, va ("%d %d %d",
	                    (int)(CLAMP (0.0f, rgb[0], 1.0f) * 255.0f + 0.5f),
	                    (int)(CLAMP (0.0f, rgb[1], 1.0f) * 255.0f + 0.5f),
	                    (int)(CLAMP (0.0f, rgb[2], 1.0f) * 255.0f + 0.5f)));
}

static void QRE_LightGlobalTab (void)
{
	const char *section = NULL;
	int         i;

	QR_GUI_LabelDim ("the light system itself: the sky, its clouds and the sun");
	QR_GUI_Spacing ();

	for (i = 0; i < (int)countof (qre_globals); i++)
	{
		const qre_global_t *g = &qre_globals[i];
		cvar_t             *var;

		if (g->section && (!section || strcmp (section, g->section)))
		{
			section = g->section;
			QR_GUI_Separator ();
			QR_GUI_Label (section);
		}

		if (g->type == QRE_G_BUTTON)
		{
			// a mode rather than a value: the press turns the aiming on and the
			// fire button ends it, so the caption says when it is running
			cvar_t *mode = Cvar_FindVar (g->action);
			char    caption[128];

			if (mode && CVAR_TO_BOOL (*mode))
				q_snprintf (caption, sizeof (caption), "%s (aiming: fire places it)", g->name);
			else
				q_snprintf (caption, sizeof (caption), "%s", g->name);
			if (QR_GUI_Button (caption))
				Cvar_Set (g->action, "1");
			QR_GUI_Tooltip (g->tip);
			continue;
		}

		var = Cvar_FindVar (g->name);

		if (!var)
		{
			QR_GUI_LabelDim (va ("%s: no such cvar", g->name));
			continue;
		}

		switch (g->type)
		{
		case QRE_G_BOOL:
		{
			int value = CVAR_TO_BOOL (*var) ? 1 : 0;

			if (QR_GUI_Checkbox (g->name, &value, g->tip))
				Cvar_Set (g->name, value ? "1" : "0");
			break;
		}
		case QRE_G_FLOAT:
		{
			float value = var->value;

			if (QR_GUI_SliderFloat (g->name, &value, g->min, g->max, g->tip))
				Cvar_Set (g->name, va ("%.4g", value));
			break;
		}
		case QRE_G_INT:
		{
			int value = (int)(var->value + 0.5f);

			if (QR_GUI_SliderInt (g->name, &value, (int)g->min, (int)g->max, g->tip))
				Cvar_Set (g->name, va ("%d", value));
			break;
		}
		case QRE_G_COLOR:
		{
			float rgb[3];
			int   en = 1;

			QRE_GlobalColorGet (g->name, rgb);
			if (QR_GUI_ColorHex (g->name, rgb, &en, g->tip))
				QRE_GlobalColorSet (g->name, rgb);
			break;
		}
		default:
			break;
		}
	}

	QR_GUI_Spacing ();
	QR_GUI_LabelDim ("these live in the config: Apply and Cancel do not own them");
}

static void QRE_BuildLightPanelGUI (void)
{
	int         panel_w = glwidth / 4;
	qboolean    exit_requested = false;
	rt_light_t *light = NULL;
	rt_light_t *inst = NULL;
	rt_light_t *shared = NULL;
	char        ikey[MAX_QPATH];

	if (panel_w < 352)
		panel_w = 352;

	QR_GUI_BeginPanel ("qr_light_editor", glwidth - panel_w, 0, panel_w, glheight);

	QR_GUI_Label ("LIGHT EDITOR");
	QRE_RefreshSelectedLight ();
	if (qre.sel_light_valid && qre.sel_light.name[0])
	{
		// A light that left its group has an entry of its own, keyed by the
		// emitter and the instance id; otherwise the emitter's shared entry is
		// what the panel edits (and an edit of it touches the whole group).
		RT_LIGHT_MakeKey (qre.sel_light.name, qre.sel_light.uniqueID, ikey, sizeof (ikey));
		inst = RT_LIGHT_Find (ikey);
		shared = RT_LIGHT_Ensure (qre.sel_light.name);
		light = inst ? inst : shared;
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

	{
		static const char *const tabs[] = { "Entity", "Global" };

		QR_GUI_Tabs ("light_tabs", tabs, (int)countof (tabs), &qre.light_tab);
		QR_GUI_Spacing ();
	}

	if (qre.light_tab == 1)
	{
		// the global tab: the sky, its clouds and the sun
		QRE_LightGlobalTab ();
		QR_GUI_EndScroll ();
		QR_GUI_EndPanel ();
		if (exit_requested)
			QRE_RequestExit ();
		return;
	}

	if (qre.sel_light_valid)
	{
		char buf[MAX_QPATH + 64];

		q_snprintf (buf, sizeof (buf), "light: %s%s", qre.sel_light.name[0] ? qre.sel_light.name : "(no emitter name)",
		            inst ? "  (own)" : "");
		QR_GUI_LabelDim (buf);
		q_snprintf (buf, sizeof (buf), "at %.0f %.0f %.0f   radius %.1f",
		            qre.sel_light.position[0], qre.sel_light.position[1], qre.sel_light.position[2],
		            qre.sel_light.radius);
		QR_GUI_LabelDim (buf);

		if (qre.sel_light.kind == RT_LIGHT_KIND_MATERIAL)
		{
			rt_material_t *m = qre.sel_light.name[0] ? RT_MAT_Find (qre.sel_light.name) : NULL;

			if (m && m->has_light_color)
				QR_GUI_LabelDim ("casts a dlight (light_color)");
			else
				QR_GUI_LabelDim ("no dlight: the material has no light_color");
		}
		else
		{
			QR_GUI_LabelDim (qre.sel_light.kind == RT_LIGHT_KIND_DLIGHT ? "a legacy dlight" : "a map light entity");
		}
	}
	else
	{
		QR_GUI_LabelDim ("aim at a light wireframe and press the fire button");
	}
	QR_GUI_Spacing ();

	if (light)
	{
		const rt_light_t *orig = QRE_LightOriginal (light->name);
		float             value;

		// group_edit at the very top: whether an edit touches the whole group
		{
			int ge = inst ? 0 : 1;

			if (QR_GUI_Checkbox ("group_edit", &ge,
			                     "Edit every light of this group (the emitter's model) at once. Off gives this light an entry of its own; on drops it and takes the group's values back."))
			{
				if (ge)
				{
					// back into the group: the instance entry goes and the
					// group's values apply again
					RT_LIGHT_Remove (ikey);
					inst = NULL;
					light = RT_LIGHT_Ensure (qre.sel_light.name);
					if (light)
					{
						light->group_edit = true;
						if (!RT_LIGHT_HasFields (light))
							QRE_LightJoinGroup (light);
						else
							QRE_TouchLight (light->name);
					}
				}
				else
				{
					// its own entry: the values in effect move into it and the
					// group stops touching this light
					rt_light_t *own = RT_LIGHT_Ensure (ikey);

					if (own)
					{
						if (light)
						{
							own->has_radius = light->has_radius;
							own->radius = light->radius;
							own->has_intensity = light->has_intensity;
							own->intensity = light->intensity;
							own->has_offset = light->has_offset;
							VectorCopy (light->offset, own->offset);
							own->has_color = light->has_color;
							VectorCopy (light->color, own->color);
							own->force_rasterize = light->force_rasterize;
						}
						own->group_edit = false;
						QRE_TouchLight (own->name);
						inst = own;
						light = own;
					}
				}
			}
		}
		if (QR_GUI_ResetButton ("group_edit", inst != NULL ||
		                        (light->group_edit != (orig ? orig->group_edit : true))))
		{
			// the reset goes back to following the group
			if (inst)
			{
				RT_LIGHT_Remove (ikey);
				inst = NULL;
				light = RT_LIGHT_Ensure (qre.sel_light.name);
			}
			if (light)
			{
				light->group_edit = true;
				if (!RT_LIGHT_HasFields (light))
					QRE_LightJoinGroup (light);
				else
					QRE_TouchLight (light->name);
			}
		}

		value = light->has_radius ? light->radius : QRE_LightDefaultRadius (qre.sel_light.kind);
		if (QR_GUI_SliderFloat ("light_radius", &value, 0.0f, 10.0f,
		                        "The size of the light, in rt_dlight_radius units; the line under it is the radius the renderer draws."))
			QRE_LightWrite (light, QRE_LIGHT_F_RADIUS, value, 0, 0, false);
		if (QR_GUI_ResetButton ("light_radius", QRE_LightFieldChanged (light, orig, QRE_LIGHT_F_RADIUS)))
			QRE_LightResetField (light, QRE_LIGHT_F_RADIUS);
		{
			char buf[64];

			q_snprintf (buf, sizeof (buf), "radius %.1f game units", METRIC_TO_QUAKEUNIT (value));
			QR_GUI_LabelDim (buf);
		}

		value = light->has_intensity ? light->intensity : QRE_LightDefaultIntensity (qre.sel_light.kind);
		if (QR_GUI_SliderFloat ("light_intensity", &value, 0.0f, 8.0f,
		                        "The brightness of the light: a multiplier of its colour."))
			QRE_LightWrite (light, QRE_LIGHT_F_INTENSITY, value, 0, 0, false);
		if (QR_GUI_ResetButton ("light_intensity", QRE_LightFieldChanged (light, orig, QRE_LIGHT_F_INTENSITY)))
			QRE_LightResetField (light, QRE_LIGHT_F_INTENSITY);

		{
			float offs[3];

			offs[0] = light->has_offset ? light->offset[0] : 0.0f;
			offs[1] = light->has_offset ? light->offset[1] : 0.0f;
			offs[2] = light->has_offset ? light->offset[2] : 0.0f;

			if (QR_GUI_Vec3Input ("light_offset", offs, -128.0f, 128.0f,
			                      "The offset of the light from the emitter's pivot point (its origin), X Y Z.")
			    )
				QRE_LightWrite (light, QRE_LIGHT_F_OFFSET, offs[0], offs[1], offs[2], false);
			if (QR_GUI_ResetButton ("light_offset", QRE_LightFieldChanged (light, orig, QRE_LIGHT_F_OFFSET)))
				QRE_LightResetField (light, QRE_LIGHT_F_OFFSET);
		}

		{
			int   en = light->has_color ? 1 : 0;
			float rgb[3];

			VectorCopy (light->has_color ? light->color : vec3_origin, rgb);
			if (QR_GUI_ColorHex ("light_color", rgb, &en,
			                     "An explicit colour of the light, replacing the emitter's own."))
			{
				if (!en)
					QRE_LightWrite (light, QRE_LIGHT_F_COLOR, -1.0f, 0, 0, false);
				else
					QRE_LightWrite (light, QRE_LIGHT_F_COLOR, rgb[0], rgb[1], rgb[2], false);
			}
			if (QR_GUI_ResetButton ("light_color", QRE_LightFieldChanged (light, orig, QRE_LIGHT_F_COLOR)))
				QRE_LightResetField (light, QRE_LIGHT_F_COLOR);
		}

		if (qre.sel_light.kind == RT_LIGHT_KIND_MATERIAL)
		{
			int fr = light->force_rasterize ? 1 : 0;

			if (QR_GUI_Checkbox ("force_rasterize", &fr, "Draw the emitter in the rasterized path."))
				QRE_LightWrite (light, QRE_LIGHT_F_FRAST, 0, 0, 0, fr != 0);
			if (QR_GUI_ResetButton ("force_rasterize", QRE_LightFieldChanged (light, orig, QRE_LIGHT_F_FRAST)))
				QRE_LightResetField (light, QRE_LIGHT_F_FRAST);
		}
	}
	else if (qre.sel_light_valid)
	{
		QR_GUI_LabelDim ("this light has no emitter name: there is nothing to save its fields to");
	}
	else
	{
		QR_GUI_Label ("nothing selected");
	}

	QR_GUI_EndScroll ();
	QR_GUI_EndPanel ();

	if (exit_requested)
		QRE_RequestExit ();
}

static void QRE_BuildFlyingOverlay (void)
{
	static const char *const lines[] = {
		"QR MATERIAL EDITOR",
		"LMB - select the face under the crosshair",
		"WASD + mouse - fly    Shift - faster    jump/movedown - up/down",
		"Tab - the cursor mode (the panel) / fly again",
		"Esc - exit the editor    ~ - console",
	};
	static const char *const light_lines[] = {
		"QR LIGHT EDITOR",
		"LMB - select the emitter under the crosshair",
		"WASD + mouse - fly    Shift - faster    jump/movedown - up/down",
		"Tab - the cursor mode (the panel) / fly again",
		"Esc - exit the editor    ~ - console",
	};
	const char *const *shown = (qre.mode == QRE_MODE_LIGHT) ? light_lines : lines;

	QR_GUI_DrawHint (shown, (int)countof (light_lines));
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

	if (QR_Editor_Flying ())
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

	if (qre.exit_prompt)
	{
		int answer = QR_GUI_Dialog ((qre.mode == QRE_MODE_LIGHT) ? "Save lights?" : "Save materials?",
		                            (qre.mode == QRE_MODE_LIGHT) ? "Save all light changes?" : "Save all materials changes?",
		                            "Save", "Discard");

		if (answer == 1)
		{
			qre.exit_prompt = false;
			QRE_SessionSave ();
		}
		else if (answer == 2)
		{
			qre.exit_prompt = false;
			QRE_SessionDiscard ();
		}
	}
	else if (qre.panel_open)
	{
		if (qre.mode == QRE_MODE_LIGHT)
			QRE_BuildLightPanelGUI ();
		else
			QRE_BuildPanelGUI ();
	}
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

	if (key == K_TAB && down)
	{
		// the other half of the Tab toggle: the panel takes the cursor
		QRE_CursorMode (true);
		return true;
	}

	if (key == K_ESCAPE && down)
	{
		QRE_RequestExit ();
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

	// Tab hands the mouse back to the camera (the flying mode picks with the
	// fire button and flies again); a text field keeps its own Tab
	if (e->type == SDL_KEYDOWN && e->key.keysym.scancode == SDL_SCANCODE_TAB && !QR_GUI_WantsKeyboard ())
	{
		QRE_CursorMode (false);
		return true;
	}

	// ESC dismisses the exit question, or closes the panel, unless an ImGui
	// text field is editing
	if (e->type == SDL_KEYDOWN && e->key.keysym.sym == SDLK_ESCAPE && !QR_GUI_WantsKeyboard ())
	{
		if (qre.exit_prompt)
		{
			qre.exit_prompt = false; // back to editing
			if (qre.prompt_from_flying)
				QRE_ClosePanel ();
		}
		else
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
	qre.pick_glt = NULL;

	QRE_FreePreview ();

	IN_Activate ();
	SDL_ShowCursor (SDL_ENABLE);
	QR_GUI_SetMouseCursor (0);
}

static void QRE_Apply (void)
{
	if (!QRE_WriteSession ())
	{
		QRE_Notify ("nothing to save yet");
		return;
	}

	if (qre.mode == QRE_MODE_LIGHT)
	{
		QRE_TakeLightSnapshot (); // Cancel now reverts to the state just saved
		QRE_Notify ("session written to lights.editor.yaml");
		return;
	}

	QRE_TakeSnapshot (); // Cancel now reverts to the state just saved
	QRE_Notify ("session written to materials.editor.yaml");
}

static void QRE_Cancel (void)
{
	if (qre.mode == QRE_MODE_LIGHT)
	{
		// the light uploads read the live list every frame, so putting the
		// snapshot back is enough
		QRE_RestoreLightSnapshot ();

		if (QRE_FileExists (qre.editor_file) && !QRE_WriteSession ())
			remove (qre.editor_file);

		QRE_Notify ("light overrides reverted to the values from lights.yaml");
		return;
	}

	QRE_RestoreSnapshot ();
	QRE_ReapplyTouched (); // put the yaml values back on screen

	qre.tmp_appended = false;

	// the restored lists may no longer contain the edited materials:
	// rebuild the group (a picked texture without a material goes back to the
	// detached defaults)
	if (qre.pick_glt)
	{
		char texname[MAX_QPATH];
		char *dot;

		RT_MAT_NormalizeName (qre.pick_glt->name, texname, sizeof (texname));
		dot = strrchr (texname, '.');
		if (dot && !strchr (dot, ':'))
			*dot = '\0';
		q_strlcpy (qre.pick_name, texname, sizeof (qre.pick_name));
		QRE_ResolveGroup (texname);
	}

	// the session file no longer matches the lists: put the reverted values
	// there (the session's names stay known until the editor closes, so a later
	// Save still writes what was applied earlier)
	if (QRE_FileExists (qre.editor_file) && !QRE_WriteSession ())
		remove (qre.editor_file);

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
	"#   * `color_emissive:` -- no mask file: each block below matches its own\n"
	"#     colour against the base texture, so only pixels close to it glow. A\n"
	"#     block carries its colour and its own tone controls:\n"
	"#         color_emissive:\n"
	"#           - color: ff0000          # rrggbb\n"
	"#             threshold: 0.02        # 0..1 colour-cube distance / sqrt(3)\n"
	"#             feather: 2             # pixels of edge softening, both sides\n"
	"#             emissive_factor: 1     # scales this block's glow (0..4)\n"
	"#             blend: screen          # cvar | off | normal | screen |\n"
	"#                                    # overlay | hard light | colour dodge\n"
	"#     The synthesized mask is white where the pixel matches the colour\n"
	"#     exactly and decays exponentially to black towards the threshold, so the\n"
	"#     glow fades out softly instead of ending in a hard edge; the feather\n"
	"#     blurs that edge on both sides without comparing colours. Up to ten\n"
	"#     blocks may share one texture; a pixel glows when any of them matches\n"
	"#     it, and the strongest one's blend mode is used there. `emissive_factor`\n"
	"#     scales the result. Combine with `is_light: true` to also cast light\n"
	"#     (otherwise the surface only glows):\n"
	"#     e.g.  - name: textures/foo\n"
	"#             color_emissive:\n"
	"#               - color: ff0000\n"
	"#                 threshold: 0.02\n"
	"#                 feather: 3\n"
	"#                 emissive_factor: 2\n"
	"#                 blend: screen\n"
	"#             is_light: true\n"
	"# The old single-colour keys (`color_emissive: ff0000,00ff00`,\n"
	"# `color_emissive_threshold`, `color_emissive_feather`) are still read: a\n"
	"# block without its own controls inherits them.\n"
	"# Precedence: an authored `texture_emissive` always wins -- while the key is\n"
	"# present in an entry, `color_emissive` in that same entry is ignored\n"
	"# entirely (even if the luma file fails to load, which is reported at\n"
	"# startup). The classic fullbright mask is merged rather than replaced, so\n"
	"# its pixels emit in addition to the luma mask.\n"
	"#\n"
	"# `emissive_blend: screen` (or a number) overrides the global `rt_emis_blend`\n"
	"# cvar for the emission that has no block of its own (a texture_emissive\n"
	"# mask); a colour block's own `blend` wins for its pixels. The names are what\n"
	"# the shader does:\n"
	"#   0 - off: emission is not composited at all\n"
	"#   1 - normal: emission is used as coverage (this is the cvar default)\n"
	"#   2 - screen: added on top of the image (brightest, keeps saturation)\n"
	"#   3 - overlay: the overlay formula, driven by the underlying base color\n"
	"#   4 - hard light: the same formula, driven by the emission color\n"
	"#   5 - colour dodge: base divided by the inverted emission (brightens)\n"
	"# Intended for emissive *mirrored* surfaces (stained glass, lit windows):\n"
	"# they read as washed out in the default mode, while the additive mode keeps\n"
	"# them bright and saturated.\n"
	"#     e.g.  - name: textures/window01_1\n"
	"#             mirror: true\n"
	"#             emissive_blend: screen\n";

static void QRE_WriteColor (FILE *f, const char *key, const vec3_t rgb)
{
	fprintf (f, "    %s: %02x%02x%02x\n", key,
	         (int)(rgb[0] * 255.0f + 0.5f) & 0xff,
	         (int)(rgb[1] * 255.0f + 0.5f) & 0xff,
	         (int)(rgb[2] * 255.0f + 0.5f) & 0xff);
}

// The colour blocks, one YAML mapping each: the colour and the tone controls
// that work for it alone.
static void QRE_WriteEmissiveBlocks (FILE *f, const rt_material_t *m)
{
	int i;

	fprintf (f, "    color_emissive:\n");
	for (i = 0; i < m->color_emissive_count; i++)
	{
		const rt_emissive_t *b = &m->color_emissive[i];

		fprintf (f, "      - color: %02x%02x%02x\n",
		         (int)(b->color[0] * 255.0f + 0.5f) & 0xff,
		         (int)(b->color[1] * 255.0f + 0.5f) & 0xff,
		         (int)(b->color[2] * 255.0f + 0.5f) & 0xff);
		fprintf (f, "        threshold: %.6g\n", b->threshold);
		fprintf (f, "        feather: %.6g\n", b->feather);
		fprintf (f, "        emissive_factor: %.6g\n", b->factor);
		fprintf (f, "        blend: %s\n", RT_MAT_EmissiveBlendName (b->blend));
	}
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
	if (m->filename_gloss[0])
		fprintf (f, "    texture_gloss: %s\n", m->filename_gloss);
	if (m->bump_scale != 1.0f)
		fprintf (f, "    bump_scale: %.6g\n", m->bump_scale);
	// mirror forces roughness_override to 0; the synthesis gives it the last
	// word, so the dead override is not perpetuated by a save
	if (!m->mirror && m->roughness_override != 0.0f)
		fprintf (f, "    roughness_override: %.6g\n", m->roughness_override);
	if (m->has_metalness_factor)
		fprintf (f, "    metalness_factor: %.6g\n", m->metalness_factor);
	if (m->metalness_from_normal_alpha)
		fprintf (f, "    metalness_from_normal_alpha: true\n");
	if (m->emissive_factor != 1.0f)
		fprintf (f, "    emissive_factor: %.6g\n", m->emissive_factor);
	if (m->emissive_blend >= 0)
		fprintf (f, "    emissive_blend: %s\n", RT_MAT_EmissiveBlendName (m->emissive_blend));
	if (m->base_factor != 1.0f)
		fprintf (f, "    base_factor: %.6g\n", m->base_factor);
	if (m->is_light)
		fprintf (f, "    is_light: true\n");
	if (m->light_styles)
		fprintf (f, "    light_styles: true\n");
	if (m->has_color_emissive && m->color_emissive_count > 0)
		QRE_WriteEmissiveBlocks (f, m);
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

static qboolean QRE_FileExists (const char *path)
{
	FILE *f = fopen (path, "rb");

	if (!f)
		return false;
	fclose (f);
	return true;
}

static qboolean QRE_CopyFile (const char *from, const char *to)
{
	FILE    *in = fopen (from, "rb");
	FILE    *out;
	char     buf[8192];
	size_t   n;
	qboolean ok = true;

	if (!in)
		return false;

	out = fopen (to, "wb");
	if (!out)
	{
		fclose (in);
		return false;
	}

	while ((n = fread (buf, 1, sizeof (buf), in)) > 0)
	{
		if (fwrite (buf, 1, n, out) != n)
		{
			ok = false;
			break;
		}
	}
	if (ferror (in))
		ok = false;

	fclose (in);
	if (fflush (out) != 0)
		ok = false;
	fclose (out);
	return ok;
}

// The names the target materials.yaml already carries: the session file is what
// replaces that file when it is saved, so those entries have to be written back
// (with their live, possibly edited values) or the save would drop them.
#define QRE_SESSION_NAMES_MAX 1024

// The touched entry of the current mode, written in the file's own format. A
// name the loader cannot resolve any more is not written at all.
static qboolean QRE_SessionEntryResolves (const char *name)
{
	if (qre.mode == QRE_MODE_LIGHT)
		return RT_LIGHT_HasFields (RT_LIGHT_Find (name)) ? true : false;
	return RT_MAT_Find (name) != NULL;
}

static void QRE_SessionWriteEntry (FILE *f, const char *name)
{
	if (qre.mode == QRE_MODE_LIGHT)
	{
		const rt_light_t *l = RT_LIGHT_Find (name);

		if (l)
			RT_LIGHT_WriteEntry (f, l);
	}
	else
	{
		const rt_material_t *m = RT_MAT_Find (name);

		if (m)
			QRE_WriteMaterial (f, m);
	}
}

// Writes materials.editor.yaml / lights.editor.yaml: the target file's own text
// with the blocks of the touched entries replaced, so comments, formatting and
// keys the loader does not understand survive a save. Entries the target does not
// carry are appended; a target that does not exist gets the standard header.
// Apply writes it, Save copies it over the target, Discard deletes it.
static qboolean QRE_WriteMergedSession (char (*touched)[MAX_QPATH], int touched_count)
{
	FILE    *in;
	FILE    *out;
	char     line[2048];
	char     written[QRE_TOUCHED_MAX];
	qboolean skipping = false;
	qboolean wrote = false;
	int      i;

	if (touched_count <= 0 || touched_count > QRE_TOUCHED_MAX)
		return false;

	memset (written, 0, sizeof (written));

	in = fopen (qre.target_file, "r");
	out = fopen (qre.editor_file, "w");
	if (!out)
	{
		if (in)
			fclose (in);
		QRE_Notify ("cannot write %s", qre.editor_file);
		return false;
	}

	if (!in)
	{
		// a target that does not exist yet: the standard header and the list key
		const qboolean light = (qre.mode == QRE_MODE_LIGHT);

		fprintf (out, "%s", light ? RT_LIGHT_Header () : qre_yaml_header);
		fprintf (out, "%s\n", light ? "lights:" : "materials:");
	}
	else
	{
		while (fgets (line, sizeof (line), in))
		{
			char *p = line;
			char  name[MAX_QPATH];

			while (*p == ' ' || *p == '\t')
				p++;

			if (!strncmp (p, "- name:", 7))
			{
				char *e;
				int   n;

				p += 7;
				while (*p == ' ' || *p == '\t')
					p++;
				e = p;
				while (*e && *e != '\r' && *e != '\n' && *e != ' ' && *e != '\t')
					e++;
				n = (int)(e - p);
				if (n >= MAX_QPATH)
					n = MAX_QPATH - 1;
				memcpy (name, p, (size_t)n);
				name[n] = '\0';
				q_strlwr (name);

				skipping = false;
				for (i = 0; i < touched_count; i++)
				{
					if (!strcmp (touched[i], name) && !written[i] && QRE_SessionEntryResolves (name))
					{
						QRE_SessionWriteEntry (out, name);
						written[i] = 1;
						skipping = true;
						wrote = true;
						break;
					}
				}
				if (skipping)
					continue;
			}
			else if (skipping && (line[0] == ' ' || line[0] == '\t' || line[0] == '\r' || line[0] == '\n'))
			{
				continue; // the body of the block that was replaced
			}
			else
			{
				skipping = false;
			}

			fputs (line, out);
		}
		fclose (in);
	}

	// the touched entries the target did not carry
	for (i = 0; i < touched_count; i++)
	{
		if (!written[i] && QRE_SessionEntryResolves (touched[i]))
		{
			QRE_SessionWriteEntry (out, touched[i]);
			wrote = true;
		}
	}

	if (!wrote)
	{
		fclose (out);
		remove (qre.editor_file); // an empty session is no session
		return false;
	}

	if (ferror (out) || fflush (out) != 0)
	{
		QRE_Notify ("write error in %s", qre.editor_file);
		fclose (out);
		remove (qre.editor_file);
		return false;
	}
	fclose (out);

	Con_Printf ("qr editor: session written to %s\n", qre.editor_file);
	return true;
}

static qboolean QRE_WriteSession (void)
{
	if (qre.mode == QRE_MODE_LIGHT)
		return QRE_WriteMergedSession (qre.light_touched, qre.light_touched_count);
	return QRE_WriteMergedSession (qre.touched, qre.touched_count);
}

// "Save" of the exit dialog: the target is backed up first, then the session
// file becomes the target (a mod's materials.yaml, so it overrides id1's). A
// copy that cannot be made keeps the session file and says so.
static void QRE_SessionSave (void)
{
	const qboolean had_target = QRE_FileExists (qre.target_file);
	const qboolean touched = (qre.mode == QRE_MODE_LIGHT) ? (qre.light_touched_count > 0) : (qre.touched_count > 0);

	if (touched && !QRE_WriteSession ())
	{
		QRE_Notify ("nothing to save");
		return;
	}
	if (!QRE_FileExists (qre.editor_file))
	{
		QRE_Notify ("nothing to save");
		return;
	}

	if (had_target && !QRE_CopyFile (qre.target_file, qre.backup_file))
	{
		QRE_Notify ("cannot write %s; the session is kept", qre.backup_file);
		return;
	}
	if (!QRE_CopyFile (qre.editor_file, qre.target_file))
	{
		QRE_Notify ("cannot write %s; the session is kept", qre.target_file);
		return;
	}

	remove (qre.editor_file);

	QRE_StopEditor (false);
	if (qre.mode == QRE_MODE_LIGHT)
		QRE_Notify (had_target ? "lights.yaml saved; backup_lights.yaml holds the previous file"
		                       : "lights.yaml saved");
	else
		QRE_Notify (had_target ? "materials.yaml saved; backup_materials.yaml holds the previous file"
		                       : "materials.yaml saved");
}

// "Discard": the session file goes away and the original values come back on
// screen; nothing on disk is touched.
static void QRE_SessionDiscard (void)
{
	remove (qre.editor_file);
	QRE_StopEditor (true);
	QRE_Notify ("changes discarded");
}

// Exit (button, Esc, the console command): ask about the session when there is
// one, close straight away when nothing was changed.
static void QRE_RequestExit (void)
{
	const qboolean touched = (qre.mode == QRE_MODE_LIGHT) ? (qre.light_touched_count > 0) : (qre.touched_count > 0);

	if (!touched && !QRE_FileExists (qre.editor_file))
	{
		QRE_StopEditor (true);
		return;
	}

	if (!qre.panel_open)
	{
		qre.prompt_from_flying = true;
		QRE_CursorMode (true);
	}
	qre.exit_prompt = true;
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
		if (qre.mode == QRE_MODE_LIGHT)
		{
			QRE_RestoreLightSnapshot ();
		}
		else
		{
			QRE_RestoreSnapshot ();
			QRE_ReapplyTouched ();
		}
	}

	qre.active = false;
	qre.panel_open = false;
	qre.exit_prompt = false;
	qre.pick_model = NULL;
	qre.pick_surf = NULL;
	qre.pick_ent = NULL;
	qre.pick_glt = NULL;
	qre.hover_model = NULL;
	qre.hover_surf = NULL;
	qre.hover_ent = NULL;
	qre.hover_glt = NULL;

	QRE_FreePreview ();

	// the world runs again: only lift the pause the editor itself set, so a
	// pause toggled meanwhile is left as the player left it; a session the
	// editor did not save goes away with it
	if (sv.paused)
		sv.paused = qre.sv_paused_prev;
	if (!restore)
		remove (qre.editor_file);

	VectorCopy (qre.player_viewangles, cl.viewangles);

	QRE_FreeSnapshot ();
	QRE_FreeLightSnapshot ();

	IN_Activate ();
	SDL_ShowCursor (SDL_ENABLE);
	QR_GUI_SetMouseCursor (0);

	QRE_Notify ("editor closed");
}

static void QRE_StartEditor (int mode)
{
	const char *name = (mode == QRE_MODE_LIGHT) ? "light" : "material";

	if (qre.active)
	{
		Con_Printf ("qr %s editor: already running\n", name);
		return;
	}
	if (cls.state != ca_connected || !cl.worldmodel)
	{
		Con_Printf ("qr %s editor: a level must be loaded first\n", name);
		return;
	}
	if (!sv.active || svs.maxclients > 1 || cls.demoplayback)
	{
		Con_Printf ("qr %s editor: single player only (the world has to be frozen)\n", name);
		return;
	}
	if (CVAR_TO_FLOAT (rt_truelight) != 1.0f)
	{
		Con_Printf ("qr %s editor: rt_truelight must be 1 (the new light system)\n", name);
		return;
	}

	memset (&qre, 0, sizeof (qre));
	qre.active = true;
	qre.panel_open = false;
	qre.mode = mode;

	VectorCopy (r_refdef.vieworg, qre.cam_origin);
	VectorCopy (cl.viewangles, qre.player_viewangles);

	if (mode == QRE_MODE_LIGHT)
	{
		QRE_TakeLightSnapshot ();

		// the light session: the gamedir's lights.yaml is what the light editor
		// saves to, with the session file and the backup the materials use too
		q_snprintf (qre.target_file, sizeof (qre.target_file), "%s/lights.yaml", com_gamedir);
		q_snprintf (qre.editor_file, sizeof (qre.editor_file), "%s/lights.editor.yaml", com_gamedir);
		q_snprintf (qre.backup_file, sizeof (qre.backup_file), "%s/backup_lights.yaml", com_gamedir);
	}
	else
	{
		QRE_TakeSnapshot ();

		// the session files: the target is the gamedir's own materials.yaml (for
		// a mod that is the mod's file, which the loader reads after id1's and
		// lets override it), the session file carries the edits until the exit
		// dialog decides, and the backup keeps the target as it was before a save
		q_snprintf (qre.target_file, sizeof (qre.target_file), "%s/materials.yaml", com_gamedir);
		q_snprintf (qre.editor_file, sizeof (qre.editor_file), "%s/materials.editor.yaml", com_gamedir);
		q_snprintf (qre.backup_file, sizeof (qre.backup_file), "%s/backup_materials.yaml", com_gamedir);
	}

	// a session file left by a crash or a map change belongs to a session that
	// is over: it must not be saved by this one
	remove (qre.editor_file);

	// freeze the world: the server stops thinking and moving, cl.time stops
	// advancing (CL_ReadFromServer) and the frame time handed to the renderer is
	// the held clock, so poses, textures, particles, the water warp and the
	// clouds all stand still. cl.paused is deliberately not touched: the engine
	// skips V_CalcRefdef -- and with it the editor camera -- while it is set.
	qre.sv_paused_prev = sv.paused;
	sv.paused = true;

	Con_Printf ("qr %s editor: on (fly: WASD + mouse; LMB selects a face; ESC exits)\n", name);
}

static void QR_Editor_Start_f (void)
{
	QRE_StartEditor (QRE_MODE_MATERIAL);
}

static void QR_LightEditor_Start_f (void)
{
	QRE_StartEditor (QRE_MODE_LIGHT);
}

static void QR_Editor_Stop_f (void)
{
	if (qre.active)
		QRE_RequestExit ();
}

// Verbose reload diagnostics of the live material editor: logs every texture a
// reload reads (with its file offset) and dumps the synthesized albedo/RME/normal
// of the first reload of a material to <gamedir>/qre_dump.
cvar_t qr_material_editor_debug = { "qr_material_editor_debug", "0", CVAR_NONE };

void QR_Editor_Init (void)
{
	static qboolean qr_editor_registered = false;
	char            font_path[MAX_OSPATH];

	if (qr_editor_registered)
		return;
	qr_editor_registered = true;

	Cvar_RegisterVariable (&qr_material_editor_debug);

	Cmd_AddCommand ("qr_material_editor_start", QR_Editor_Start_f);
	Cmd_AddCommand ("qr_material_editor_stop", QR_Editor_Stop_f);

	Cmd_AddCommand ("qr_light_editor_start", QR_LightEditor_Start_f);
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
	if (!QR_Editor_Flying ())
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
