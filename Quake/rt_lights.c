// rt_lights.c -- the dynamic-light overrides of lights.yaml (see rt_lights.h).
//
// The files are read once per map (the editor reloads them through its own
// session flow): the base directory's lights.yaml first, then the running
// gamedir's, with a later file replacing the entries of an earlier one -- the
// same precedence the materials files have.

#include "quakedef.h"
#include "rt_material.h"
#include "rt_lights.h"
#include "atomics.h"

#define RT_LIGHT_CAP RT_LIGHT_NAMES_MAX

static rt_light_t rt_lights[RT_LIGHT_CAP];
static int        rt_light_count = 0;
static qboolean   rt_light_initialized = false;

// The lights uploaded in the current frame (see RT_TRACK_Light): the counter is
// atomic because the uploads may come from the render tasks.
static rt_tracked_light_t rt_tracked[RT_TRACKED_LIGHTS_MAX];
static atomic_uint32_t    rt_tracked_count;

void RT_TRACK_BeginFrame(void)
{
    memset(rt_tracked, 0, sizeof(rt_tracked));
    Atomic_StoreUInt32(&rt_tracked_count, 0);
}

void RT_TRACK_Light(const vec3_t position, float radius, const vec3_t color,
                    uint64_t uniqueID, int kind, const char *name)
{
    uint32_t index = Atomic_AddUInt32(&rt_tracked_count, 1);
    rt_tracked_light_t *light;

    if (index >= RT_TRACKED_LIGHTS_MAX)
    {
        return;
    }

    light = &rt_tracked[index];
    VectorCopy(position, light->position);
    light->radius = radius;
    VectorCopy(color, light->color);
    light->uniqueID = uniqueID;
    light->kind = kind;
    q_strlcpy(light->name, name ? name : "", sizeof(light->name));
    // published last: a reader that sees the count (or the ready flag) sees the
    // fields too. The editor's readers also wait for the draw task that fills the
    // list when tasks are on (gl_screen.c), so the frame's list is complete.
    light->ready = 1;
}

const rt_tracked_light_t *RT_TRACK_Lights(int *outCount)
{
    uint32_t count = Atomic_LoadUInt32(&rt_tracked_count);

    if (count > RT_TRACKED_LIGHTS_MAX)
    {
        count = RT_TRACKED_LIGHTS_MAX;
    }
    if (outCount)
    {
        *outCount = (int)count;
    }
    return rt_tracked;
}

static const char *rt_light_header =
    "# Dynamic light overrides for the vkpt ray-traced renderer.\n"
    "# A light belongs to an emitter: the texture a model or a sprite draws (the\n"
    "# same name materials.yaml uses for it), the model of the entity that asked\n"
    "# for a legacy dlight, or the classname of a map light entity -- classname\n"
    "# entries apply where the legacy light system uploads those entities\n"
    "# (rt_truelight 0); the editor itself runs on rt_truelight 1 and shows the\n"
    "# lights that system builds.\n"
    "#   light_radius    -- the size of the light (rt_dlight_radius units)\n"
    "#   light_intensity -- the brightness of the light (a multiplier of its colour)\n"
    "#   light_offset    -- \"x y z\", the offset from the emitter's pivot point\n"
    "#   light_color     -- \"rrggbb\", an explicit colour for the light\n"
    "#   force_rasterize -- draw the emitter in the rasterized path (material lights)\n"
    "#   group_edit      -- true (the default) when an edit of one light of the\n"
    "#                      group (the emitter's model) is written to all of them\n"
    "# An emitter without an entry uses the global rt_dlight_* settings.\n";

// The name the editor and the renderer agree on: the normalized texture name
// with a file extension stripped (a model skin keeps its ":frameN"). An
// instance key ("name#id") keeps its id and normalizes the name before it.
static void rt_light_norm(const char *name, char *out, size_t outsize)
{
    char  buf[MAX_QPATH];
    char *hash;
    char *dot;

    q_strlcpy(buf, name, sizeof(buf));
    hash = strchr(buf, '#');
    if (hash)
    {
        *hash = '\0';
    }

    RT_MAT_NormalizeName(buf, out, outsize);
    dot = strrchr(out, '.');
    if (dot && !strchr(dot, ':'))
    {
        *dot = '\0';
    }

    if (hash)
    {
        size_t len = strlen(out);

        if (len + 1 < outsize)
        {
            q_snprintf(out + len, outsize - len, "#%s", hash + 1);
        }
    }
}

static rt_light_t *rt_light_find_in(const char *normalized)
{
    int i;

    for (i = 0; i < rt_light_count; i++)
    {
        if (rt_lights[i].valid && !strcmp(rt_lights[i].name, normalized))
        {
            return &rt_lights[i];
        }
    }
    return NULL;
}

rt_light_t *RT_LIGHT_List(int *outCount)
{
    if (outCount)
    {
        *outCount = rt_light_count;
    }
    return rt_lights;
}

void RT_LIGHT_SetCount(int count)
{
    if (count < 0)
    {
        count = 0;
    }
    if (count > RT_LIGHT_CAP)
    {
        count = RT_LIGHT_CAP;
    }
    rt_light_count = count;
}

rt_light_t *RT_LIGHT_Find(const char *name)
{
    char normalized[MAX_QPATH];

    if (!rt_light_initialized || !name || !name[0])
    {
        return NULL;
    }
    rt_light_norm(name, normalized, sizeof(normalized));
    return rt_light_find_in(normalized);
}

rt_light_t *RT_LIGHT_Ensure(const char *name)
{
    rt_light_t *light;
    char        normalized[MAX_QPATH];

    if (!rt_light_initialized || !name || !name[0])
    {
        return NULL;
    }

    rt_light_norm(name, normalized, sizeof(normalized));
    light = rt_light_find_in(normalized);
    if (light)
    {
        return light;
    }
    if (rt_light_count >= RT_LIGHT_CAP)
    {
        return NULL;
    }

    light = &rt_lights[rt_light_count++];
    memset(light, 0, sizeof(*light));
    light->valid = true;
    light->group_edit = true; // part of its group until the flag says otherwise
    q_strlcpy(light->name, normalized, sizeof(light->name));
    return light;
}

void RT_LIGHT_MakeKey(const char *name, uint64_t uniqueID, char *out, size_t outsize)
{
    char normalized[MAX_QPATH];

    rt_light_norm(name, normalized, sizeof(normalized));
    q_snprintf(out, outsize, "%s#%llu", normalized, (unsigned long long)uniqueID);
}

rt_light_t *RT_LIGHT_FindInstance(const char *name, uint64_t uniqueID)
{
    char  key[MAX_QPATH];
    rt_light_t *light;

    if (!rt_light_initialized || !name || !name[0])
    {
        return NULL;
    }

    RT_LIGHT_MakeKey(name, uniqueID, key, sizeof(key));
    light = rt_light_find_in(key);
    return light ? light : RT_LIGHT_Find(name);
}

rt_light_t *RT_LIGHT_EnsureInstance(const char *name, uint64_t uniqueID)
{
    char key[MAX_QPATH];

    RT_LIGHT_MakeKey(name, uniqueID, key, sizeof(key));
    return RT_LIGHT_Ensure(key);
}

void RT_LIGHT_Remove(const char *name)
{
    char        normalized[MAX_QPATH];
    rt_light_t *light;
    int         index;

    if (!rt_light_initialized || !name || !name[0])
    {
        return;
    }

    rt_light_norm(name, normalized, sizeof(normalized));
    light = rt_light_find_in(normalized);
    if (!light)
    {
        return;
    }

    index = (int)(light - rt_lights);
    for (; index + 1 < rt_light_count; index++)
    {
        rt_lights[index] = rt_lights[index + 1];
    }
    rt_light_count--;
}

qboolean RT_LIGHT_HasFields(const rt_light_t *l)
{
    return l && (l->has_radius || l->has_intensity || l->has_offset || l->has_color ||
                 l->force_rasterize || !l->group_edit);
}

const char *RT_LIGHT_Header(void)
{
    return rt_light_header;
}

void RT_LIGHT_WriteEntry(FILE *f, const rt_light_t *l)
{
    fprintf(f, "  - name: %s\n", l->name);
    if (l->has_radius)
    {
        fprintf(f, "    light_radius: %.6g\n", l->radius);
    }
    if (l->has_intensity)
    {
        fprintf(f, "    light_intensity: %.6g\n", l->intensity);
    }
    if (l->has_offset)
    {
        fprintf(f, "    light_offset: %.6g %.6g %.6g\n", l->offset[0], l->offset[1], l->offset[2]);
    }
    if (l->has_color)
    {
        fprintf(f, "    light_color: %02x%02x%02x\n",
                (int)(l->color[0] * 255.0f + 0.5f) & 0xff,
                (int)(l->color[1] * 255.0f + 0.5f) & 0xff,
                (int)(l->color[2] * 255.0f + 0.5f) & 0xff);
    }
    if (l->force_rasterize)
    {
        fprintf(f, "    force_rasterize: true\n");
    }
    fprintf(f, "    group_edit: %s\n", l->group_edit ? "true" : "false");
}

// Trims a scalar: leading/trailing space and tab, and a pair of quotes.
static void rt_light_trim(char *text, size_t outsize)
{
    char *start = text;
    char *end;
    size_t len;

    while (*start == ' ' || *start == '\t')
    {
        start++;
    }
    end = start + strlen(start);
    while (end > start && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r' || end[-1] == '\n'))
    {
        end--;
    }
    *end = '\0';

    len = (size_t)(end - start);
    if (len >= 2 && start[0] == '"' && start[len - 1] == '"')
    {
        start++;
        len -= 2;
        start[len] = '\0';
    }
    q_strlcpy(text, start, outsize);
}

// "rrggbb" (a leading # is accepted) -> rgb in 0..1.
static qboolean rt_light_parse_hex(const char *value, vec3_t out)
{
    int i, c;

    while (*value == ' ' || *value == '\t')
    {
        value++;
    }
    if (*value == '#')
    {
        value++;
    }
    if (strlen(value) != 6)
    {
        return false;
    }

    for (i = 0; i < 3; i++)
    {
        int hi = value[i * 2];
        int lo = value[i * 2 + 1];

        if (hi >= '0' && hi <= '9')       hi -= '0';
        else if (hi >= 'a' && hi <= 'f')  hi = hi - 'a' + 10;
        else if (hi >= 'A' && hi <= 'F')  hi = hi - 'A' + 10;
        else return false;

        if (lo >= '0' && lo <= '9')       lo -= '0';
        else if (lo >= 'a' && lo <= 'f')  lo = lo - 'a' + 10;
        else if (lo >= 'A' && lo <= 'F')  lo = lo - 'A' + 10;
        else return false;

        c = hi * 16 + lo;
        out[i] = (float)c / 255.0f;
    }
    return true;
}

static qboolean rt_light_parse_bool(const char *value)
{
    if (!q_strcasecmp(value, "true") || !q_strcasecmp(value, "yes") || !q_strcasecmp(value, "on"))
    {
        return true;
    }
    if (!q_strcasecmp(value, "false") || !q_strcasecmp(value, "no") || !q_strcasecmp(value, "off"))
    {
        return false;
    }
    return atoi(value) != 0;
}

static int rt_light_load_file(const char *path)
{
    FILE       *f = fopen(path, "r");
    char        line[1024];
    rt_light_t *cur = NULL;
    int         loaded = 0;

    if (!f)
    {
        return 0;
    }

    while (fgets(line, sizeof(line), f))
    {
        char *p = line;
        char *colon;

        while (*p == ' ' || *p == '\t')
        {
            p++;
        }
        if (*p == '#' || *p == '\0' || *p == '\r' || *p == '\n')
        {
            continue;
        }

        if (!strncmp(p, "- name:", 7) || !strncmp(p, "name:", 5))
        {
            char name[MAX_QPATH];

            p = strchr(p, ':') + 1;
            q_strlcpy(name, p, sizeof(name));
            rt_light_trim(name, sizeof(name));
            q_strlwr(name);
            if (name[0])
            {
                rt_light_t *light = rt_light_find_in(name);

                if (light)
                {
                    // a later file replaces the fields of the earlier entry
                    memset(light, 0, sizeof(*light));
                }
                else if (rt_light_count < RT_LIGHT_CAP)
                {
                    light = &rt_lights[rt_light_count++];
                    memset(light, 0, sizeof(*light));
                }
                else
                {
                    light = NULL;
                }

                if (light)
                {
                    light->valid = true;
                    light->group_edit = true; // the default: a light follows its group
                    q_strlcpy(light->name, name, sizeof(light->name));
                }
                cur = light;
                loaded++;
            }
            continue;
        }

        if (!cur)
        {
            continue;
        }

        colon = strchr(p, ':');
        if (!colon)
        {
            continue;
        }
        *colon = '\0';
        {
            char key[64];
            char value[256];

            q_strlcpy(key, p, sizeof(key));
            rt_light_trim(key, sizeof(key));
            q_strlcpy(value, colon + 1, sizeof(value));
            rt_light_trim(value, sizeof(value));

            if (!q_strcasecmp(key, "light_radius"))
            {
                cur->radius = (float)atof(value);
                cur->has_radius = true;
            }
            else if (!q_strcasecmp(key, "light_intensity"))
            {
                cur->intensity = (float)atof(value);
                cur->has_intensity = true;
            }
            else if (!q_strcasecmp(key, "light_color"))
            {
                vec3_t rgb;

                if (rt_light_parse_hex(value, rgb))
                {
                    VectorCopy(rgb, cur->color);
                    cur->has_color = true;
                }
            }
            else if (!q_strcasecmp(key, "force_rasterize"))
            {
                cur->force_rasterize = rt_light_parse_bool(value);
            }
            else if (!q_strcasecmp(key, "group_edit"))
            {
                cur->group_edit = rt_light_parse_bool(value);
            }
            else if (!q_strcasecmp(key, "light_offset"))
            {
                float x = 0.0f, y = 0.0f, z = 0.0f;

                if (sscanf(value, "%f %f %f", &x, &y, &z) == 3)
                {
                    cur->offset[0] = x;
                    cur->offset[1] = y;
                    cur->offset[2] = z;
                    cur->has_offset = true;
                }
            }
        }
    }

    fclose(f);
    return loaded;
}

static void rt_light_load_directory(const char *dir)
{
    char path[MAX_OSPATH];

    q_snprintf(path, sizeof(path), "%s/lights.yaml", dir);
    if (Sys_FileTime(path) != -1)
    {
        int loaded = rt_light_load_file(path);

        if (loaded > 0)
        {
            Con_Printf("RT: loaded %d light overrides from %s\n", loaded, path);
        }
    }
}

void RT_LIGHT_Reload(void)
{
    char base[MAX_OSPATH];

    if (!rt_light_initialized)
    {
        return;
    }

    rt_light_count = 0;

    q_snprintf(base, sizeof(base), "%s/id1", com_basedir);
    if (q_strcasecmp(base, com_gamedir))
    {
        rt_light_load_directory(base);
    }
    rt_light_load_directory(com_gamedir);
}

void RT_LIGHT_Init(void)
{
    if (rt_light_initialized)
    {
        return;
    }
    rt_light_initialized = true;
    RT_LIGHT_Reload();
}

void RT_LIGHT_Shutdown(void)
{
    rt_light_initialized = false;
    rt_light_count = 0;
}

int RT_LIGHT_ReadNames(const char *path, char (*names)[MAX_QPATH], int max)
{
    FILE *f = fopen(path, "r");
    char  line[1024];
    int   count = 0;

    if (!f)
    {
        return 0;
    }

    while (count < max && fgets(line, sizeof(line), f))
    {
        char *p = line;
        char  name[MAX_QPATH];
        int   i, dup = 0;

        while (*p == ' ' || *p == '\t')
        {
            p++;
        }
        if (*p == '#' || !strncmp(p, "lights:", 7))
        {
            continue;
        }
        if (strncmp(p, "- name:", 7) != 0 && strncmp(p, "name:", 5) != 0)
        {
            continue;
        }

        p = strchr(p, ':') + 1;
        q_strlcpy(name, p, sizeof(name));
        rt_light_trim(name, sizeof(name));
        q_strlwr(name);
        if (!name[0])
        {
            continue;
        }

        for (i = 0; i < count; i++)
        {
            if (!strcmp(names[i], name))
            {
                dup = 1;
            }
        }
        if (!dup)
        {
            q_strlcpy(names[count++], name, MAX_QPATH);
        }
    }

    fclose(f);
    return count;
}

qboolean RT_LIGHT_Write(const char *path, char (*names)[MAX_QPATH], int count)
{
    FILE *f;
    int   i, written = 0;

    if (!path || !names || count <= 0)
    {
        return false;
    }

    f = fopen(path, "w");
    if (!f)
    {
        Con_DWarning("RT light: cannot write %s\n", path);
        return false;
    }

    fprintf(f, "%s", rt_light_header);
    fprintf(f, "lights:\n");

    for (i = 0; i < count; i++)
    {
        rt_light_t *light = rt_light_find_in(names[i]);

        if (RT_LIGHT_HasFields(light))
        {
            RT_LIGHT_WriteEntry(f, light);
            written++;
        }
    }

    if (written == 0)
    {
        fclose(f);
        remove(path); // an empty file is no file
        return false;
    }

    if (ferror(f) || fflush(f) != 0)
    {
        Con_DWarning("RT light: write error in %s\n", path);
        fclose(f);
        remove(path);
        return false;
    }

    fclose(f);
    Con_Printf("qr editor: wrote %d lights to %s\n", written, path);
    return true;
}
