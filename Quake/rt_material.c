// Q2RTX-style .mat material definitions (phase 4.5).
// Ported from Q2RTX material.c (GPL v2) and adapted to the vkquake host.

#include "quakedef.h"
#include "rt_material.h"
#include "rt_pkz.h"

#include <yaml.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4505)
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STBI_NO_BMP
#define STBI_NO_PSD
#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_PNM
#define STBI_NO_LINEAR
#define STBI_NO_STDIO
#define STBI_MALLOC(sz)		Mem_Alloc (sz)
#define STBI_REALLOC(p, newsz) Mem_Realloc (p, newsz)
#define STBI_FREE(p)		Mem_Free (p)
#include "stb_image.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#define RT_MAT_MAX_MATERIALS 2048
#define RT_MAT_MAX_GLOBAL   RT_MAT_CAP_GLOBAL
#define RT_MAT_MAX_MAP      RT_MAT_CAP_MAP

static rt_material_t rt_global_materials[RT_MAT_MAX_GLOBAL];
static int rt_global_count = 0;
static rt_material_t rt_map_materials[RT_MAT_MAX_MAP];
static int rt_map_count = 0;
static qboolean rt_initialized = false;
static cmd_function_t *rt_mat_cmd = NULL;
static char rt_current_map[MAX_QPATH];

static cvar_t rt_materials = { "rt_materials", "1", CVAR_ARCHIVE };
cvar_t rt_mat_debug = { "rt_mat_debug", "0", 0 };

static byte *rt_load_file(const char *name, int *outLen)
{
    byte *b = RT_PKZ_LoadFile(name, outLen);
    if (b)
    {
        return b;
    }

    unsigned int path_id;
    byte *fs = COM_LoadFile(name, &path_id);
    if (fs)
    {
        if (outLen)
        {
            *outLen = com_filesize;
        }
        return fs;
    }

    if (outLen)
    {
        *outLen = 0;
    }
    return NULL;
}

static void rt_load_file_free(byte *data)
{
    Mem_Free(data);
}

static byte *rt_decode_tga(const byte *buf, int len, int *outW, int *outH)
{
    if (len < 18)
    {
        return NULL;
    }

    const byte id_length = buf[0];
    const byte colormap_type = buf[1];
    const byte image_type = buf[2];
    const int width = buf[12] | (buf[13] << 8);
    const int height = buf[14] | (buf[15] << 8);
    const byte pixel_size = buf[16];
    const byte attributes = buf[17];

    if (colormap_type != 0)
    {
        return NULL;
    }
    if (image_type != 2 && image_type != 10)
    {
        return NULL;
    }
    if (pixel_size != 24 && pixel_size != 32)
    {
        return NULL;
    }
    if (width <= 0 || height <= 0)
    {
        return NULL;
    }

    const qboolean upside_down = !(attributes & 0x20);
    const int numPixels = width * height;
    byte *out = (byte *)Mem_Alloc(numPixels * 4);
    if (!out)
    {
        return NULL;
    }

    int pos = 18 + id_length;
    const int bpp = pixel_size / 8;

    for (int row = height - 1; row >= 0; row--)
    {
        const int realrow = upside_down ? row : height - 1 - row;
        byte *pixbuf = out + realrow * width * 4;

        for (int column = 0; column < width; column++)
        {
            if (image_type == 10)
            {
                if (pos >= len)
                {
                    Mem_Free(out);
                    return NULL;
                }
                const byte packet = buf[pos++];
                const int count = (packet & 0x7f) + 1;
                if (packet & 0x80)
                {
                    if (pos + bpp > len || column + count > width)
                    {
                        Mem_Free(out);
                        return NULL;
                    }
                    const byte r = buf[pos + 2];
                    const byte g = buf[pos + 1];
                    const byte b = buf[pos + 0];
                    const byte a = (bpp == 4) ? buf[pos + 3] : 255;
                    pos += bpp;
                    for (int k = 0; k < count; k++)
                    {
                        *pixbuf++ = r; *pixbuf++ = g; *pixbuf++ = b; *pixbuf++ = a;
                    }
                    column += count - 1;
                }
                else
                {
                    for (int k = 0; k < count; k++)
                    {
                        if (pos + bpp > len || column + k >= width)
                        {
                            Mem_Free(out);
                            return NULL;
                        }
                        *pixbuf++ = buf[pos + 2];
                        *pixbuf++ = buf[pos + 1];
                        *pixbuf++ = buf[pos + 0];
                        *pixbuf++ = (bpp == 4) ? buf[pos + 3] : 255;
                        pos += bpp;
                    }
                    column += count - 1;
                }
            }
            else
            {
                if (pos + bpp > len)
                {
                    Mem_Free(out);
                    return NULL;
                }
                *pixbuf++ = buf[pos + 2];
                *pixbuf++ = buf[pos + 1];
                *pixbuf++ = buf[pos + 0];
                *pixbuf++ = (bpp == 4) ? buf[pos + 3] : 255;
                pos += bpp;
            }
        }
    }

    if (outW) *outW = width;
    if (outH) *outH = height;
    return out;
}

byte *RT_MAT_LoadTexture(const rt_material_t *mat, int which, int *outWidth, int *outHeight)
{
    const char *base = NULL;
    switch (which)
    {
        case RT_MAT_TEX_BASE:     base = mat->filename_base; break;
        case RT_MAT_TEX_NORMALS:  base = mat->filename_normals; break;
        case RT_MAT_TEX_EMISSIVE: base = mat->filename_emissive; break;
        case RT_MAT_TEX_GLOSS:    base = mat->filename_gloss; break;
        default: return NULL;
    }
    if (!base[0])
    {
        return NULL;
    }

    const char *dot = strrchr(base, '.');
    const qboolean hasExt = (dot != NULL && dot[1] != '\0');

    if (hasExt)
    {
        int len = 0;
        byte *file = rt_load_file(base, &len);
        if (!file)
        {
            if (outWidth) *outWidth = 0;
            if (outHeight) *outHeight = 0;
            return NULL;
        }
        byte *decoded = NULL;
        if (!q_strcasecmp(dot, ".tga"))
        {
            decoded = rt_decode_tga(file, len, outWidth, outHeight);
        }
        else
        {
            int w = 0, h = 0, c = 0;
            decoded = stbi_load_from_memory(file, len, &w, &h, &c, 4);
            if (decoded)
            {
                if (outWidth) *outWidth = w;
                if (outHeight) *outHeight = h;
            }
        }
        rt_load_file_free(file);
        if (!decoded)
        {
            if (outWidth) *outWidth = 0;
            if (outHeight) *outHeight = 0;
        }
        return decoded;
    }

    static const char *exts[] = { "tga", "jpg", "png" };
    for (int e = 0; e < 3; e++)
    {
        char name[MAX_QPATH];
        q_snprintf(name, sizeof(name), "%s.%s", base, exts[e]);

        int len = 0;
        byte *file = rt_load_file(name, &len);
        if (!file)
        {
            continue;
        }

        if (e == 0)
        {
            byte *decoded = rt_decode_tga(file, len, outWidth, outHeight);
            rt_load_file_free(file);
            if (decoded)
            {
                return decoded;
            }
        }
        else
        {
            int w = 0, h = 0, comp = 0;
            byte *decoded = stbi_load_from_memory(file, len, &w, &h, &comp, 4);
            rt_load_file_free(file);
            if (decoded)
            {
                if (outWidth) *outWidth = w;
                if (outHeight) *outHeight = h;
                return decoded;
            }
        }
    }

    if (outWidth) *outWidth = 0;
    if (outHeight) *outHeight = 0;
    return NULL;
}

// ---------------------------------------------------------------------------
// .mat parser (ported from Q2RTX material.c load_material_file)
// ---------------------------------------------------------------------------

static void rt_mat_reset(rt_material_t *mat)
{
    memset(mat, 0, sizeof(*mat));
    mat->bump_scale = 1.0f;
    mat->metalness_factor = 0.0f;
    mat->emissive_factor = 1.0f;
    mat->emissive_blend = -1;
    mat->base_factor = 1.0f;
    mat->light_brightness = 1.0f;
    mat->light_styles = false;
    mat->color_emissive_threshold = 0.02f;
}

static qboolean rt_mat_parse_bool(const char *value)
{
    if (!q_strcasecmp(value, "true") || !q_strcasecmp(value, "yes") || !q_strcasecmp(value, "on"))
        return true;
    if (!q_strcasecmp(value, "false") || !q_strcasecmp(value, "no") || !q_strcasecmp(value, "off"))
        return false;
    return atoi(value) != 0;
}

static qboolean rt_mat_parse_hex_color(const char *value, vec3_t out)
{
    int i, c;

    if (!value || strlen(value) != 6)
        return false;

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

        c = (hi << 4) | lo;
        out[i] = (float)c / 255.0f;
    }
    return true;
}

/* Appends an empty block and returns it (NULL when the list is full): the
   mapping form of a colour entry fills the fields itself. */
static rt_emissive_t *rt_mat_append_emissive_block(rt_material_t *mat)
{
    rt_emissive_t *block;

    if (mat->color_emissive_count >= RT_MAT_MAX_EMISSIVE_COLORS)
    {
        return NULL;
    }

    block = &mat->color_emissive[mat->color_emissive_count];
    memset(block, 0, sizeof(*block));
    block->color[0] = block->color[1] = block->color[2] = 1.0f;
    block->threshold = mat->color_emissive_threshold;
    block->feather = mat->color_emissive_feather;
    block->blend = mat->emissive_blend;
    mat->color_emissive_count++;
    mat->has_color_emissive = true;
    return block;
}

// Appends one colour to the material's colour_emissive list ("ff0000", "#ff0000"
// and " ff0000 " all read the same). The list is what a texture atlas with
// several differently coloured emissive regions is authored with. Returns the
// new block or NULL when the list is full or the colour does not parse.
static rt_emissive_t *rt_mat_add_emissive_color(rt_material_t *mat, const char *value)
{
    vec3_t c;
    char   buf[64];
    size_t len;
    rt_emissive_t *block;

    if (!value)
    {
        return NULL;
    }

    while (*value == ' ' || *value == '\t')
    {
        value++;
    }
    if (*value == '#')
    {
        value++;
    }

    len = strlen(value);
    while (len > 0 && (value[len - 1] == ' ' || value[len - 1] == '\t' || value[len - 1] == '\r'))
    {
        len--;
    }
    if (len == 0 || len >= sizeof(buf))
    {
        return NULL;
    }

    memcpy(buf, value, len);
    buf[len] = 0;

    if (!rt_mat_parse_hex_color(buf, c))
    {
        Con_DWarning("RT mat: material '%s': color_emissive '%s' is not rrggbb; ignored\n",
                     mat->name, buf);
        return NULL;
    }

    block = rt_mat_append_emissive_block(mat);
    if (block)
    {
        VectorCopy(c, block->color);
    }
    return block;
}

// The blend modes by name, so the file reads like what the shader does:
// "normal" is the coverage blend, "screen" adds the emission on top, "overlay"
// and "hard light" are the two halves of the overlay formula (the base and the
// emission as the driver), "colour dodge" divides by the inverse.
const char *RT_MAT_EmissiveBlendName(int blend)
{
    switch (blend)
    {
    case -1: return "cvar";
    case 0:  return "off";
    case 1:  return "normal";
    case 2:  return "screen";
    case 3:  return "overlay";
    case 4:  return "hard light";
    case 5:  return "colour dodge";
    default: return "cvar";
    }
}

// A blend mode by number or by any of its aliases; -2 when it is neither.
static int rt_mat_parse_emissive_blend(const char *value)
{
    int v;

    if (!q_strcasecmp(value, "cvar"))                                      return -1;
    if (!q_strcasecmp(value, "off") || !q_strcasecmp(value, "none"))       return 0;
    if (!q_strcasecmp(value, "normal") || !q_strcasecmp(value, "coverage")) return 1;
    if (!q_strcasecmp(value, "screen") || !q_strcasecmp(value, "add") ||
        !q_strcasecmp(value, "additive"))                                  return 2;
    if (!q_strcasecmp(value, "overlay"))                                   return 3;
    if (!q_strcasecmp(value, "hard light") || !q_strcasecmp(value, "hard_light") ||
        !q_strcasecmp(value, "hardlight"))                                 return 4;
    if (!q_strcasecmp(value, "colour dodge") || !q_strcasecmp(value, "color dodge") ||
        !q_strcasecmp(value, "colour_dodge") || !q_strcasecmp(value, "color_dodge") ||
        !q_strcasecmp(value, "dodge") || !q_strcasecmp(value, "divide"))   return 5;

    v = atoi(value);
    if (v < -1 || v > RT_MAT_EMIS_BLEND_MAX)
    {
        return -2;
    }
    return v;
}

static void rt_mat_set_attribute(rt_material_t *mat, const char *key, const char *value)
{
    if (!q_strcasecmp(key, "bump_scale"))
        mat->bump_scale = (float)atof(value);
    else if (!q_strcasecmp(key, "roughness_override"))
        mat->roughness_override = (float)atof(value);
    else if (!q_strcasecmp(key, "metalness_factor"))
    {
        mat->metalness_factor = (float)atof(value);
        mat->has_metalness_factor = true;
    }
    else if (!q_strcasecmp(key, "metalness_from_normal_alpha"))
    {
        mat->metalness_from_normal_alpha = rt_mat_parse_bool(value);
    }
    else if (!q_strcasecmp(key, "emissive_factor"))
        mat->emissive_factor = (float)atof(value);
    else if (!q_strcasecmp(key, "base_factor"))
        mat->base_factor = (float)atof(value);
    else if (!q_strcasecmp(key, "emissive_blend"))
    {
        const int v = rt_mat_parse_emissive_blend(value);

        if (v == -2)
        {
            Con_DWarning("RT mat: material '%s': emissive_blend '%s' is not a mode; ignored\n",
                         mat->name, value);
        }
        else
        {
            mat->emissive_blend = v;
        }
    }
    else if (!q_strcasecmp(key, "is_light"))
        mat->is_light = rt_mat_parse_bool(value);
    else if (!q_strcasecmp(key, "light_styles"))
        mat->light_styles = rt_mat_parse_bool(value);
    else if (!q_strcasecmp(key, "color_emissive"))
    {
        // one or more colours, comma separated: "ff0000,00ff00"
        char  buf[1024];
        char *p, *next;

        q_strlcpy(buf, value, sizeof(buf));
        for (p = buf; p && *p; p = next)
        {
            char *end;

            next = strchr(p, ',');
            if (next)
            {
                *next++ = '\0';
            }
            end = p + strlen(p);
            while (end > p && (end[-1] == ' ' || end[-1] == '\t'))
            {
                *--end = '\0';
            }
            if (*p)
            {
                rt_mat_add_emissive_color(mat, p);
            }
        }
    }
    else if (!q_strcasecmp(key, "color_emissive_threshold"))
        mat->color_emissive_threshold = (float)atof(value);
    else if (!q_strcasecmp(key, "color_emissive_feather"))
    {
        float f = (float)atof(value);

        if (f < 0.0f)
        {
            f = 0.0f;
        }
        if (f > 16.0f)
        {
            f = 16.0f;
        }
        mat->color_emissive_feather = f;
    }
    else if (!q_strcasecmp(key, "light_color"))
        mat->has_light_color = rt_mat_parse_hex_color(value, mat->light_color);
    else if (!q_strcasecmp(key, "light_brightness"))
        mat->light_brightness = (float)atof(value);
    else if (!q_strcasecmp(key, "light_upoffset"))
        mat->light_upoffset = (float)atof(value);
    else if (!q_strcasecmp(key, "mirror"))
        mat->mirror = rt_mat_parse_bool(value);
    else if (!q_strcasecmp(key, "exact_normals"))
        mat->exact_normals = rt_mat_parse_bool(value);
    else if (!q_strcasecmp(key, "force_rasterize"))
        mat->force_rasterize = rt_mat_parse_bool(value);
    else if (!q_strcasecmp(key, "texture_base"))
        q_strlcpy(mat->filename_base, value, sizeof(mat->filename_base));
    else if (!q_strcasecmp(key, "texture_normals"))
        q_strlcpy(mat->filename_normals, value, sizeof(mat->filename_normals));
    else if (!q_strcasecmp(key, "texture_emissive"))
        q_strlcpy(mat->filename_emissive, value, sizeof(mat->filename_emissive));
    else if (!q_strcasecmp(key, "texture_gloss"))
        q_strlcpy(mat->filename_gloss, value, sizeof(mat->filename_gloss));
    else
        Con_DWarning("RT mat: unknown attribute '%s' in material '%s'\n", key, mat->name);
}

static void rt_mat_yaml_scalar(const yaml_node_t *node, char *out, size_t outsize)
{
    if (!node || !node->data.scalar.value || outsize == 0)
    {
        if (outsize > 0)
            out[0] = 0;
        return;
    }

    size_t n = node->data.scalar.length;
    if (n >= outsize)
        n = outsize - 1;
    memcpy(out, node->data.scalar.value, n);
    out[n] = 0;
}

static int rt_mat_parse_yaml(const char *filebuf, int len, const char *file_name, rt_material_t *dest, int max_items)
{
    if (!filebuf || len == 0)
    {
        return 0;
    }

    yaml_parser_t parser;
    yaml_document_t document;
    int count = 0;

    if (!yaml_parser_initialize(&parser))
    {
        return 0;
    }

    yaml_parser_set_input_string(&parser, (const unsigned char *)filebuf, (size_t)len);

    if (yaml_parser_load(&parser, &document))
    {
        yaml_node_t *root = yaml_document_get_root_node(&document);
        if (root && root->type == YAML_MAPPING_NODE)
        {
            for (yaml_node_pair_t *rp = root->data.mapping.pairs.start;
                 rp < root->data.mapping.pairs.top; rp++)
            {
                yaml_node_t *rk = yaml_document_get_node(&document, rp->key);
                if (!rk || rk->type != YAML_SCALAR_NODE || !rk->data.scalar.value)
                    continue;
                if (rk->data.scalar.length != 9 || memcmp(rk->data.scalar.value, "materials", 9) != 0)
                    continue;

                yaml_node_t *seq = yaml_document_get_node(&document, rp->value);
                if (!seq || seq->type != YAML_SEQUENCE_NODE)
                    continue;

                for (yaml_node_item_t *it = seq->data.sequence.items.start;
                     it < seq->data.sequence.items.top && count < max_items; it++)
                {
                    yaml_node_t *item = yaml_document_get_node(&document, *it);
                    if (!item || item->type != YAML_MAPPING_NODE)
                        continue;

                    rt_mat_reset(dest);
                    dest->valid = true;

                    int have_name = 0;
                    for (yaml_node_pair_t *mp = item->data.mapping.pairs.start;
                         mp < item->data.mapping.pairs.top; mp++)
                    {
                        yaml_node_t *mk = yaml_document_get_node(&document, mp->key);
                        yaml_node_t *mv = yaml_document_get_node(&document, mp->value);
                        if (!mk || !mv || mk->type != YAML_SCALAR_NODE)
                            continue;

                        char keybuf[128];
                        rt_mat_yaml_scalar(mk, keybuf, sizeof(keybuf));

                        if (mv->type == YAML_SEQUENCE_NODE)
                        {
                            // a list value: only colour_emissive takes one, so
                            // both "color_emissive: ff0000,00ff00" and a YAML
                            // list of colours author the same thing
                            if (!q_strcasecmp(keybuf, "color_emissive"))
                            {
                                for (yaml_node_item_t *sit = mv->data.sequence.items.start;
                                     sit < mv->data.sequence.items.top; sit++)
                                {
                                    yaml_node_t *sitem = yaml_document_get_node(&document, *sit);
                                    char valbuf[1024];

                                    if (!sitem)
                                        continue;

                                    if (sitem->type == YAML_MAPPING_NODE)
                                    {
                                        // one block with its own tone controls:
                                        //   - { color: ff0000, threshold: 0.02,
                                        //       feather: 2, blend: screen }
                                        rt_emissive_t *block = rt_mat_append_emissive_block(dest);

                                        if (!block)
                                            continue;

                                        for (yaml_node_pair_t *bp = sitem->data.mapping.pairs.start;
                                             bp < sitem->data.mapping.pairs.top; bp++)
                                        {
                                            yaml_node_t *bk = yaml_document_get_node(&document, bp->key);
                                            yaml_node_t *bv = yaml_document_get_node(&document, bp->value);
                                            char kb[64], vb[128];

                                            if (!bk || !bv || bk->type != YAML_SCALAR_NODE ||
                                                bv->type != YAML_SCALAR_NODE)
                                                continue;
                                            rt_mat_yaml_scalar(bk, kb, sizeof(kb));
                                            rt_mat_yaml_scalar(bv, vb, sizeof(vb));

                                            if (!q_strcasecmp(kb, "color") || !q_strcasecmp(kb, "colour"))
                                            {
                                                const char *hex = (vb[0] == '#') ? vb + 1 : vb;

                                                if (!rt_mat_parse_hex_color(hex, block->color))
                                                    Con_DWarning("RT mat: material '%s': colour '%s' is not rrggbb; white is used\n",
                                                                 dest->name, vb);
                                            }
                                            else if (!q_strcasecmp(kb, "threshold"))
                                            {
                                                block->threshold = (float)atof(vb);
                                                block->has_threshold = true;
                                            }
                                            else if (!q_strcasecmp(kb, "feather"))
                                            {
                                                block->feather = CLAMP(0.0f, (float)atof(vb), 16.0f);
                                                block->has_feather = true;
                                            }
                                            else if (!q_strcasecmp(kb, "blend"))
                                            {
                                                const int v = rt_mat_parse_emissive_blend(vb);

                                                if (v == -2)
                                                    Con_DWarning("RT mat: material '%s': blend '%s' is not a mode; the material's is used\n",
                                                                 dest->name, vb);
                                                else
                                                {
                                                    block->blend = v;
                                                    block->has_blend = true;
                                                }
                                            }
                                        }
                                    }
                                    else if (sitem->type == YAML_SCALAR_NODE)
                                    {
                                        rt_mat_yaml_scalar(sitem, valbuf, sizeof(valbuf));
                                        rt_mat_add_emissive_color(dest, valbuf);
                                    }
                                }
                            }
                            continue;
                        }
                        if (mv->type != YAML_SCALAR_NODE)
                            continue;

                        char valbuf[1024];
                        rt_mat_yaml_scalar(mv, valbuf, sizeof(valbuf));

                        if (!q_strcasecmp(keybuf, "name"))
                        {
                            q_strlcpy(dest->name, valbuf, sizeof(dest->name));
                            q_strlwr(dest->name);
                            have_name = 1;
                        }
                        else
                        {
                            rt_mat_set_attribute(dest, keybuf, valbuf);
                        }
                    }

                    // A block that carries no tone controls of its own inherits
                    // the material-level ones (the old single-colour keys), so
                    // after loading every block is complete on its own.
                    for (int c = 0; c < dest->color_emissive_count; c++)
                    {
                        rt_emissive_t *block = &dest->color_emissive[c];

                        if (!block->has_threshold)
                            block->threshold = dest->color_emissive_threshold;
                        if (!block->has_feather)
                            block->feather = dest->color_emissive_feather;
                        if (!block->has_blend)
                            block->blend = dest->emissive_blend;
                        block->has_threshold = block->has_feather = block->has_blend = true;
                    }

                    if (have_name)
                    {
                        q_strlcpy(dest->source_file, file_name, sizeof(dest->source_file));
                        dest++;
                        count++;
                    }
                }
            }
        }
        yaml_document_delete(&document);
    }
    else
    {
        Con_DWarning("RT mat: failed to parse YAML '%s': %s\n",
                     file_name, parser.problem ? parser.problem : "unknown error");
    }

    yaml_parser_delete(&parser);
    return count;
}

static int rt_mat_load_yaml_file(const char *file_name, rt_material_t *dest, int max_items)
{
    int   len = 0;
    char *filebuf = (char *)rt_load_file(file_name, &len);
    int   count;

    if (!filebuf)
        return 0;

    count = rt_mat_parse_yaml(filebuf, len, file_name, dest, max_items);
    rt_load_file_free((byte *)filebuf);
    return count;
}

// A file addressed by an absolute path: the search path cannot reach it. This
// is how the base id1 directory is read while a mod is running.
static int rt_mat_load_abs_file(const char *path, rt_material_t *dest, int max_items)
{
    FILE  *f = fopen(path, "rb");
    long   len;
    char  *buf;
    size_t got;
    int    count;

    if (!f)
        return 0;

    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len <= 0 || len > 64 * 1024 * 1024)
    {
        fclose(f);
        return 0;
    }

    buf = (char *)Mem_Alloc((size_t)len + 1);
    got = fread(buf, 1, (size_t)len, f);
    fclose(f);
    buf[got] = 0;

    count = rt_mat_parse_yaml(buf, (int)got, path, dest, max_items);
    Mem_Free(buf);
    return count;
}

static int rt_mat_load_any(const char *name, rt_material_t *dest, int max_items)
{
    // the directory scan passes an absolute path; the pkz listing and the map
    // file pass a name the search path resolves
    if (name[0] && (name[1] == ':' || name[0] == '/' || name[0] == '\\'))
        return rt_mat_load_abs_file(name, dest, max_items);
    return rt_mat_load_yaml_file(name, dest, max_items);
}

// Every materials/*.yaml of one directory, then its materials.yaml at the root.
// Paths are absolute: the loader reads them itself, so a lower-priority
// directory is reached even when a higher-priority one carries a file of the
// same name (files loaded later override entries by name).
static void rt_mat_load_dir(const char *dir, int (*cb)(const char *name, void *ctx), void *ctx)
{
    char pattern[MAX_OSPATH];
    WIN32_FIND_DATAA fd;
    HANDLE h;

    q_snprintf(pattern, sizeof(pattern), "%s/materials/*.yaml", dir);
    h = FindFirstFileA(pattern, &fd);
    if (h != INVALID_HANDLE_VALUE)
    {
        do
        {
            char path[MAX_OSPATH];

            if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
            {
                continue;
            }
            // the editor's own files: the session file is not a materials file
            // until it is saved, and the backup never is
            if (!q_strcasecmp(fd.cFileName, "materials.editor.yaml") ||
                !q_strcasecmp(fd.cFileName, "backup_materials.yaml"))
            {
                continue;
            }
            q_snprintf(path, sizeof(path), "%s/materials/%s", dir, fd.cFileName);
            if (cb(path, ctx))
            {
                break;
            }
        } while (FindNextFileA(h, &fd));
        FindClose(h);
    }

    // the mod's override file lives at the gamedir root
    q_snprintf(pattern, sizeof(pattern), "%s/materials.yaml", dir);
    if (Sys_FileTime(pattern) != -1)
    {
        cb(pattern, ctx);
    }
}

typedef struct {
    rt_material_t *dest;
    int *count;
    int max;
} rt_mat_load_ctx_t;

static int rt_mat_load_cb(const char *name, void *vctx)
{
    rt_mat_load_ctx_t *ctx = (rt_mat_load_ctx_t *)vctx;
    const int before = *ctx->count;
    int loaded, n, kept = 0;

    if (before >= ctx->max)
    {
        return 1;
    }

    loaded = rt_mat_load_any(name, ctx->dest + before, ctx->max - before);

    /* A later file overrides the entries of an earlier one with the same name.
       The load order is the packaged base, then id1, then the running gamedir;
       this loop is what makes a mod's materials.yaml replace the id1 values it
       names while id1 still supplies everything the mod does not name. */
    for (n = 0; n < loaded; n++)
    {
        rt_material_t *loaded_mat = ctx->dest + before + n;
        int            old = -1, i;

        for (i = 0; i < before; i++)
        {
            if (!q_strcasecmp(ctx->dest[i].name, loaded_mat->name))
            {
                old = i;
                break;
            }
        }

        if (old >= 0)
        {
            ctx->dest[old] = *loaded_mat;
        }
        else
        {
            ctx->dest[before + kept++] = *loaded_mat;
        }
    }
    *ctx->count = before + kept;

    if (loaded > 0)
    {
        Con_Printf("RT: loaded %d materials from %s\n", loaded, name);
    }
    return 0;
}

void RT_MAT_Init(void)
{
    if (rt_initialized)
    {
        return;
    }
    rt_initialized = true;

    Cvar_RegisterVariable(&rt_materials);
    Cvar_RegisterVariable(&rt_mat_debug);

    rt_global_count = 0;
    rt_map_count = 0;

    RT_PKZ_Init();

    rt_mat_load_ctx_t ctx = { rt_global_materials, &rt_global_count, RT_MAT_MAX_GLOBAL };

    // packaged materials first, then id1, then the running gamedir (a mod), so
    // a mod's materials.yaml overrides the id1 entries it names
    RT_PKZ_ListFiles("materials/", ".yaml", rt_mat_load_cb, &ctx);
    {
        char base[MAX_OSPATH];
        q_snprintf(base, sizeof(base), "%s/id1", com_basedir);
        if (q_strcasecmp(base, com_gamedir))
        {
            rt_mat_load_dir(base, rt_mat_load_cb, &ctx);
        }
    }
    rt_mat_load_dir(com_gamedir, rt_mat_load_cb, &ctx);

    rt_mat_cmd = Cmd_AddCommand2("rt_mat", RT_MAT_Cmd, src_command);
}

void RT_MAT_Shutdown(void)
{
    if (!rt_initialized)
    {
        return;
    }
    rt_initialized = false;

    if (rt_mat_cmd)
    {
        Cmd_RemoveCommand(rt_mat_cmd);
        rt_mat_cmd = NULL;
    }
    rt_global_count = 0;
    rt_map_count = 0;
    RT_PKZ_Shutdown();
}

void RT_MAT_ChangeMap(const char *mapname)
{
    if (!rt_initialized)
    {
        return;
    }

    q_strlcpy(rt_current_map, mapname ? mapname : "", sizeof(rt_current_map));

    rt_map_count = 0;

    char name[MAX_QPATH];
    q_snprintf(name, sizeof(name), "materials/%s.yaml", mapname);

    rt_mat_load_ctx_t ctx = { rt_map_materials, &rt_map_count, RT_MAT_MAX_MAP };
    rt_mat_load_cb(name, &ctx);

    // The gamedir's own materials.yaml is the editor's target: it has to beat
    // the map file it was saved from, or a saved edit would be shadowed by that
    // file on the next load of the same map.
    {
        char own[MAX_OSPATH];
        q_snprintf(own, sizeof(own), "%s/materials.yaml", com_gamedir);
        if (Sys_FileTime(own) != -1)
        {
            rt_mat_load_cb(own, &ctx);
        }
    }
}

void RT_MAT_Reload(void)
{
    if (!rt_initialized)
    {
        return;
    }

    rt_global_count = 0;
    rt_map_count = 0;

    rt_mat_load_ctx_t ctx = { rt_global_materials, &rt_global_count, RT_MAT_MAX_GLOBAL };
    RT_PKZ_ListFiles("materials/", ".yaml", rt_mat_load_cb, &ctx);
    {
        char base[MAX_OSPATH];
        q_snprintf(base, sizeof(base), "%s/id1", com_basedir);
        if (q_strcasecmp(base, com_gamedir))
        {
            rt_mat_load_dir(base, rt_mat_load_cb, &ctx);
        }
    }
    rt_mat_load_dir(com_gamedir, rt_mat_load_cb, &ctx);

    if (rt_current_map[0])
    {
        RT_MAT_ChangeMap(rt_current_map);
    }
}

static void rt_mat_normalize_name(const char *name, char *out, size_t outsize)
{
    q_strlcpy(out, name, outsize);

    char *bsp = strstr(out, ".bsp");
    if (bsp)
    {
        char *colon = strchr(bsp, ':');
        if (colon)
        {
            char texname[MAX_QPATH];
            q_strlcpy(texname, colon + 1, sizeof(texname));
            if (texname[0] == '*')
            {
                texname[0] = '#';
            }
            q_snprintf(out, outsize, "textures/%s", texname);
        }
    }

    q_strlwr(out);
}

void RT_MAT_NormalizeName(const char *name, char *out, size_t outsize)
{
    rt_mat_normalize_name(name, out, outsize);
}

// "textures/+3_med25" -> 3, "progs/flame.mdl:frame2" -> 2 (a model or sprite
// skin frame), anything else -> -1
int RT_MAT_FrameDigit(const char *name)
{
    if (!q_strncasecmp(name, "textures/+", 10) && name[10] >= '0' && name[10] <= '9')
    {
        return name[10] - '0';
    }

    {
        const char *p = strstr(name, ":frame");

        if (p && p[6] >= '0' && p[6] <= '9' && (p[7] == '\0' || p[7] == '_'))
        {
            return p[6] - '0';
        }
    }
    return -1;
}

void RT_MAT_GroupBaseOf(const char *name, char *out, size_t outsize)
{
    if (!q_strncasecmp(name, "textures/+", 10) && name[10] >= '0' && name[10] <= '9')
    {
        q_snprintf(out, outsize, "textures/%s", name + 11);
        return;
    }

    {
        const char *p = strstr(name, ":frame");

        if (p && p[6] >= '0' && p[6] <= '9')
        {
            size_t n = (size_t)(p - name);

            if (n >= outsize)
            {
                n = outsize - 1;
            }
            memcpy(out, name, n);
            out[n] = '\0';
            return;
        }
    }

    q_strlcpy(out, name, outsize);
}

static rt_material_t *rt_mat_find_in(const char *name, rt_material_t *first, int count)
{
    char n[MAX_QPATH];
    rt_mat_normalize_name(name, n, sizeof(n));
    char *dot = strrchr(n, '.');
    if (dot && !strchr(dot, ':'))
    {
        *dot = 0;
    }

    for (int i = 0; i < count; i++)
    {
        if (first[i].valid)
        {
            if (!strcmp(first[i].name, n))
            {
                if (CVAR_TO_BOOL (rt_mat_debug))
                {
                    Con_Printf("RT: material lookup '%s' -> '%s' FOUND (emissive=%s gloss=%s)\n",
                               name, n,
                               first[i].filename_emissive[0] ? first[i].filename_emissive : "-",
                               first[i].filename_gloss[0] ? first[i].filename_gloss : "-");
                }
                return &first[i];
            }
        }
    }
    if (CVAR_TO_BOOL (rt_mat_debug))
    {
        Con_Printf("RT: material lookup '%s' -> '%s' NOT FOUND\n", name, n);
    }
    return NULL;
}

rt_material_t *RT_MAT_Find(const char *name)
{
    if (!rt_initialized || !RT_MAT_Enabled())
    {
        return NULL;
    }
    if (!name || !name[0])
    {
        return NULL;
    }

    rt_material_t *m = rt_mat_find_in(name, rt_map_materials, rt_map_count);
    if (m)
    {
        return m;
    }
    return rt_mat_find_in(name, rt_global_materials, rt_global_count);
}

rt_material_t *RT_MAT_GetList(int which, int *outCount)
{
    if (which == RT_MAT_LIST_MAP)
    {
        if (outCount)
            *outCount = rt_map_count;
        return rt_map_materials;
    }

    if (outCount)
        *outCount = rt_global_count;
    return rt_global_materials;
}

int RT_MAT_AppendGlobal(const rt_material_t *mat)
{
    if (!rt_initialized || rt_global_count >= RT_MAT_MAX_GLOBAL)
        return -1;

    rt_global_materials[rt_global_count] = *mat;
    return rt_global_count++;
}

void RT_MAT_SetListCounts(int globalCount, int mapCount)
{
    if (globalCount < 0)
        globalCount = 0;
    if (mapCount < 0)
        mapCount = 0;
    if (globalCount > RT_MAT_MAX_GLOBAL)
        globalCount = RT_MAT_MAX_GLOBAL;
    if (mapCount > RT_MAT_MAX_MAP)
        mapCount = RT_MAT_MAX_MAP;

    /* dropped entries must not stay findable through a stale valid flag */
    for (int i = globalCount; i < rt_global_count; i++)
        rt_global_materials[i].valid = false;
    for (int i = mapCount; i < rt_map_count; i++)
        rt_map_materials[i].valid = false;

    rt_global_count = globalCount;
    rt_map_count = mapCount;
}

const char *RT_MAT_CurrentMap(void)
{
    return rt_current_map;
}

qboolean RT_MAT_Enabled(void)
{
    return rt_materials.value != 0;
}

void RT_MAT_Cmd(void)
{
    if (Cmd_Argc() == 1)
    {
        Con_Printf("rt_materials is %s; %d global, %d map materials\n",
                   RT_MAT_Enabled() ? "ON" : "OFF", rt_global_count, rt_map_count);
        return;
    }

    rt_material_t *m = RT_MAT_Find(Cmd_Argv(1));
    if (!m)
    {
        Con_Printf("no material for '%s'\n", Cmd_Argv(1));
        return;
    }

    Con_Printf("material '%s': base=%s normals=%s emissive=%s gloss=%s "
               "bump=%.2f rough=%.2f metal=%.2f emiss=%.2f base=%.2f emis_blend=%d\n",
               m->name,
               m->filename_base[0] ? m->filename_base : "-",
               m->filename_normals[0] ? m->filename_normals : "-",
               m->filename_emissive[0] ? m->filename_emissive : "-",
               m->filename_gloss[0] ? m->filename_gloss : "-",
               m->bump_scale, m->roughness_override,
               m->metalness_factor, m->emissive_factor, m->base_factor,
               m->emissive_blend);
}
