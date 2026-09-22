// Copyright (c) 2026 QuakeRay contributors
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
//
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
//


// HLSL counterpart of TurbWarp.h. Like the GLSL one it includes nothing itself: the shader has to
// pull in ShaderCommonHLSLFunc.hlsli (globalUniform, GEOM_INST_FLAG_TURB_WARP) and, through it,
// Utils.hlsli (M_PI) first.
//
// Not a single spell had to change. The scalar-vector arithmetic of the phase needs no help: a
// scalar operand promotes to the vector it is mixed with, so `M_PI * texCoord.yx + time` and
// `strength * (1.0 / 8.0) * sin(phase)` are written as in the GLSL original. Only the types spell
// differently: vec2 -> float2.
//
// The texture coordinate is passed through when the instance is not turbulent, so the caller does
// not have to test the flag itself.

#ifndef TURB_WARP_HLSLI_
#define TURB_WARP_HLSLI_
#ifndef DESC_SET_GLOBAL_UNIFORM
    #error DESC_SET_GLOBAL_UNIFORM must be defined
#endif


float2 getTurbWarpUV(const float2 texCoord)
{
    const float2 phase = M_PI * texCoord.yx + globalUniform.time;
    return texCoord + globalUniform.turbWarpStrength * (1.0 / 8.0) * sin(phase);
}

// Texture coordinate of the surface, animated if it is a turbulent one.
float2 getSurfaceTexCoord(const uint geometryInstanceFlags, const float2 texCoord)
{
    if ((geometryInstanceFlags & GEOM_INST_FLAG_TURB_WARP) == 0)
    {
        return texCoord;
    }

    return getTurbWarpUV(texCoord);
}

#endif // TURB_WARP_HLSLI_
