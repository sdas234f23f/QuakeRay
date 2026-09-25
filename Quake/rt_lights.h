// rt_lights.h -- per-emitter overrides of the dynamic lights (lights.yaml).
//
// What a surface looks like -- and whether it glows -- is materials.yaml; how
// big the light an emitter casts is, how bright it is and where it sits relative
// to the emitter's pivot is lights.yaml. The two files are deliberately apart: a
// material may be shared by many emitters, while a light belongs to one of them,
// and mixing the two would make every window in the map an emitter of its own.
//
// The editor (Quake/qr_editor.c) edits these entries live and persists them
// through the same session flow the materials use.

#ifndef RT_LIGHTS_H
#define RT_LIGHTS_H

#include "quakedef.h"

// The most entries a lights.yaml list (and a session) can hold.
#define RT_LIGHT_NAMES_MAX 256

typedef struct rt_light_s
{
    char     name[MAX_QPATH];  // the emitter the light belongs to (a texture name)
    qboolean valid;
    qboolean has_radius;
    float    radius;           // the size of the light, in the units of rt_dlight_radius
    qboolean has_intensity;
    float    intensity;        // the brightness of the light: a multiplier of its colour
    qboolean has_offset;
    vec3_t   offset;           // the offset from the emitter's pivot point (its origin)
} rt_light_t;

void RT_LIGHT_Init (void);
void RT_LIGHT_Shutdown (void);
void RT_LIGHT_Reload (void);

// The live list the editor edits and snapshots.
rt_light_t *RT_LIGHT_List (int *outCount);
void        RT_LIGHT_SetCount (int count);

// The override of an emitter, or NULL. The name is normalized the way the
// editor normalizes it ("maps/x.bsp:name" -> "textures/name", extension
// stripped, lowercased).
rt_light_t *RT_LIGHT_Find (const char *name);

// Like RT_LIGHT_Find, but an emitter without an entry gets one (with no field
// authored): the editor edits the values in place and the entry joins the file
// when something changes. NULL when the list is full.
rt_light_t *RT_LIGHT_Ensure (const char *name);

// The file format (the editor owns the paths and the session flow).
void RT_LIGHT_WriteEntry (FILE *f, const rt_light_t *l);
// The names a lights.yaml carries ("- name: x" lines; comments and quotes are
// understood). Returns the count.
int  RT_LIGHT_ReadNames (const char *path, char (*names)[MAX_QPATH], int max);
// Writes the named entries from the live list into lights.yaml; an entry with no
// field authored is not written. Returns false when nothing was written (the
// file is not created then, or removed if it existed).
qboolean RT_LIGHT_Write (const char *path, char (*names)[MAX_QPATH], int count);

// True when at least one field is authored (what the writer asks).
qboolean RT_LIGHT_HasFields (const rt_light_t *l);

#endif /* RT_LIGHTS_H */
