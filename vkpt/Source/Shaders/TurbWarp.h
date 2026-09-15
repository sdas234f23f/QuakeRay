// The classic "warp" of the turbulent surfaces (lava, teleport): every texture
// axis is shifted by a sine of the other axis, so the pattern swirls in time.
//
// gl_warp.c rasterized the surface texture into an auxiliary image, applying
// exactly this displacement (WARPCALC, kept in the reference compute shader as
// cs_tex_warp.comp). Its sine had an amplitude of 8 and a period of 256 in the
// units of that image, where the texture was sampled at image coordinate / 64:
// the period is therefore two texture tiles and the amplitude an eighth of a
// tile. The phase grows by one radian per second, which is also the unit of
// globalUniform.time.

vec2 getTurbWarpUV(const vec2 texCoord)
{
    const vec2 phase = M_PI * texCoord.yx + globalUniform.time;
    return texCoord + globalUniform.turbWarpStrength * (1.0 / 8.0) * sin(phase);
}

// Texture coordinate of the surface, animated if it is a turbulent one.
vec2 getSurfaceTexCoord(const uint geometryInstanceFlags, const vec2 texCoord)
{
    if ((geometryInstanceFlags & GEOM_INST_FLAG_TURB_WARP) == 0)
    {
        return texCoord;
    }

    return getTurbWarpUV(texCoord);
}
