// Q2RTX-style material definitions (phase 4.5).
// Ported from Q2RTX material.c and adapted to the vkquake host: materials are
// loaded from materials/*.yaml files (global + <map>.yaml) found either on disk
// or inside a mounted .pkz archive, and are used to synthesize the vkpt
// RGBA8 material textures (albedo-alpha, roughness-metallic-emissive, normal).

#ifndef RT_MATERIAL_H
#define RT_MATERIAL_H

#include "quakedef.h"

#define RT_MAT_EMIS_BLEND_MAX 5
/* How many colour_emissive blocks a material may carry: a texture atlas may
   hold several differently coloured emissive regions (window panes, lamps),
   each with its own tone controls. */
#define RT_MAT_MAX_EMISSIVE_COLORS 10

/* One colour_emissive block: a colour and the tone controls that work for it
   alone (they were material-level before). */
typedef struct rt_emissive_s
{
    vec3_t   color;
    float    threshold;      // how far a pixel's colour may differ and still glow
    float    feather;        // pixels of edge softening (0 = a hard mask)
    int      blend;          // how the glow is composited; -1 = the material's / cvar
    /* Set while a file is read: a block that does not carry its own value
       inherits the material-level one (the old single-colour keys). */
    qboolean has_threshold;
    qboolean has_feather;
    qboolean has_blend;
} rt_emissive_t;

/* Capacities of the live material lists: RT_MAT_GetList returns arrays of
   these sizes, and a snapshot of the lists must be allocated for them. */
enum {
    RT_MAT_CAP_GLOBAL = 4096,
    RT_MAT_CAP_MAP    = 1024,
};

typedef struct rt_material_s {
    char name[MAX_QPATH];
    /* The materials/*.yaml file this material was loaded from ("materials/materials.yaml",
       "materials/<map>.yaml", ...). Empty for materials created by the editor at runtime. */
    char source_file[MAX_QPATH];
    char filename_base[MAX_QPATH];
    char filename_normals[MAX_QPATH];
    char filename_emissive[MAX_QPATH];
    char filename_gloss[MAX_QPATH];
    float bump_scale;
    float roughness_override;
    float metalness_factor;
    float emissive_factor;
    /* Overrides the global rt_emis_blend cvar for this material's emission;
       -1 = not authored (use the cvar), 0..RT_MAT_EMIS_BLEND_MAX = mode. */
    int emissive_blend;
    float base_factor;
    qboolean is_light;
    qboolean light_styles;
    qboolean has_metalness_factor;
    qboolean metalness_from_normal_alpha;
    rt_emissive_t color_emissive[RT_MAT_MAX_EMISSIVE_COLORS];
    int    color_emissive_count;
    qboolean has_color_emissive;
    /* The material-level fallbacks the blocks inherit when they do not carry
       their own (the old keys, kept for files written before the blocks). */
    float color_emissive_threshold;
    float color_emissive_feather;
    vec3_t light_color;
    qboolean has_light_color;
    float light_brightness;
    float light_upoffset;
    qboolean mirror;
    qboolean exact_normals;
    qboolean force_rasterize;
    qboolean valid;
} rt_material_t;

void RT_MAT_Init(void);
void RT_MAT_Shutdown(void);

void RT_MAT_ChangeMap(const char *mapname);

void RT_MAT_Reload(void);

rt_material_t *RT_MAT_Find(const char *name);

/* Editor support: the live material lists and their provenance. */

enum {
    RT_MAT_LIST_GLOBAL = 0, /* every materials/*.yaml file (dir scan + pkz) */
    RT_MAT_LIST_MAP    = 1, /* materials/<map>.yaml for the current map only */
};

/* Returns the live array of the requested list. The editor edits these structs in
   place and snapshots them for Cancel. */
rt_material_t *RT_MAT_GetList(int which, int *outCount);

/* Normalizes a texture name the way RT_MAT_Find would ("maps/x.bsp:name" and bare
   names become "textures/name", extension stripped, lowercased). */
void RT_MAT_NormalizeName(const char *name, char *out, size_t outsize);

/* The animation-frame ring a material name belongs to: "textures/+0_med25" ->
   "textures/_med25", "progs/flame.mdl:frame1" -> "progs/flame.mdl". The digit
   of the frame, or -1 when the name is not a frame. */
int  RT_MAT_FrameDigit(const char *name);
void RT_MAT_GroupBaseOf(const char *name, char *out, size_t outsize);

/* Appends a copy of mat to the global list; returns its index or -1 when full. */
int RT_MAT_AppendGlobal(const rt_material_t *mat);

/* Restores the list lengths (the editor truncates what it appended before
   Cancel/Exit). Counts are clamped to the array capacities. */
void RT_MAT_SetListCounts(int globalCount, int mapCount);

/* The current map name ("" when none), as loaded by RT_MAT_ChangeMap. */
const char *RT_MAT_CurrentMap(void);

/* The blend mode of an emissive value: "cvar", "off", "normal", "screen",
   "overlay", "hard light" or "divide" (0..RT_MAT_EMIS_BLEND_MAX). The editor's
   combo boxes show these names and the writer stores them. */
const char *RT_MAT_EmissiveBlendName(int blend);

enum {
    RT_MAT_TEX_BASE,
    RT_MAT_TEX_NORMALS,
    RT_MAT_TEX_EMISSIVE,
    RT_MAT_TEX_GLOSS,
};

byte *RT_MAT_LoadTexture(const rt_material_t *mat, int which, int *outWidth, int *outHeight);

qboolean RT_MAT_Enabled(void);

void RT_MAT_Cmd(void);

#endif
