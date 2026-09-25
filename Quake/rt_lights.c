// rt_lights.c -- the dynamic-light overrides of lights.yaml (see rt_lights.h).
//
// The files are read once per map (the editor reloads them through its own
// session flow): the base directory's lights.yaml first, then the running
// gamedir's, with a later file replacing the entries of an earlier one -- the
// same precedence the materials files have.

#include "quakedef.h"
#include "rt_material.h"
#include "rt_lights.h"

#define RT_LIGHT_CAP 256

static rt_light_t rt_lights[RT_LIGHT_CAP];
static int        rt_light_count = 0;
static qboolean   rt_light_initialized = false;

static const char *rt_light_header =
    "# Dynamic light overrides for the vkpt ray-traced renderer.\n"
    "# A light belongs to an emitter -- the texture a model or a sprite draws\n"
    "# (the same name materials.yaml uses for it):\n"
    "#   light_radius    -- the size of the light (rt_dlight_radius units)\n"
    "#   light_intensity -- the brightness of the light (a multiplier of its colour)\n"
    "#   light_offset    -- \"x y z\", the offset from the emitter's pivot point\n"
    "# An emitter without an entry uses the global rt_dlight_* settings.\n";

// The name the editor and the renderer agree on: the normalized texture name
// with a file extension stripped (a model skin keeps its ":frameN").
static void rt_light_norm(const char *name, char *out, size_t outsize)
{
    char *dot;

    RT_MAT_NormalizeName(name, out, outsize);
    dot = strrchr(out, '.');
    if (dot && !strchr(dot, ':'))
    {
        *dot = '\0';
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
    q_strlcpy(light->name, normalized, sizeof(light->name));
    return light;
}

qboolean RT_LIGHT_HasFields(const rt_light_t *l)
{
    return l && (l->has_radius || l->has_intensity || l->has_offset);
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
                    light->valid = true;
                    q_strlcpy(light->name, name, sizeof(light->name));
                }
                else if (rt_light_count < RT_LIGHT_CAP)
                {
                    light = &rt_lights[rt_light_count++];
                    memset(light, 0, sizeof(*light));
                    light->valid = true;
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
