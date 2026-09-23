// Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) 2026 QuakeRay contributors
//
// This file is a port of shader/checkerboard_interleave.comp from Quake 2 RTX (https://github.com/NVIDIA/Q2RTX),
// which is distributed under the terms of the GNU General Public License
// version 2.  It has been adapted to the renderer interface of this project.
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
// Ported from Q2RTX (GPL v2) shader/checkerboard_interleave.comp. Resolves the
// checkerboarded ASVGF output (Q2Color) into the regular-layout PreFinal, which
// the rest of the vkpt post chain (rasterizer, tonemapping) consumes.
//

// HLSL counterpart of CmQ2Interleave.comp.
//
// Spellings that had to change:
//   * gl_GlobalInvocationID -> SV_DispatchThreadID, and the golden's local_size_x/y = 16 become
//     [numthreads(16, 16, 1)] on the entry point
//   * texelFetch(t, p, 0) -> t.Load(int3(p, 0)), imageStore(img, p, v) -> img[p] = v, and
//     mix(a, b, 0.5) -> lerp(a, b, 0.5)
//   * vec4(0) -> (float4)0: the same four-zero vector, and vec4(max(vec3(0.0), rgb), 0) keeps its
//     shape with the renamed types
//   * the golden's two int(globalUniform.renderWidth/Height) conversions keep their spelling as
//     (int)(...), a truncating cast like the GLSL constructor
//
// What did not change: the early out, the checkerboard field/separator helpers, the fourth tap
// whose offset depends on the parity of opos.x, the two edge pixels that are zeroed, the fixed
// 0.25 average of the four taps, and the two components of the mixed color.

#define DESC_SET_FRAMEBUFFERS 0
#define DESC_SET_GLOBAL_UNIFORM 1
#include "ShaderCommonHLSLFunc.hlsli"

float4 safe_load(int2 pos)
{
    if(pos.x < 0 || pos.y < 0 || pos.x >= (int)(globalUniform.renderWidth) || pos.y >= (int)(globalUniform.renderHeight))
        return (float4)0;

    return framebufQ2Color_Sampled.Load(int3( pos, 0 ));
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const int2 opos = int2(dispatchThreadID.xy);

    if (opos.x >= (int)(globalUniform.renderWidth) || opos.y >= (int)(globalUniform.renderHeight))
    {
        return;
    }

    // position of this output pixel in the checkerboarded Q2Color
    const int2 ipos = getCheckerboardPix(opos);

    float4 center = framebufQ2Color_Sampled.Load(int3( ipos, 0 ));

    // Split surfaces (water/glass) contain refl/refr in different checkerboard fields -
    // cross-resolve them from the other parity side.
    float4 color = center;
    if (wasSplit(framebufThroughput_Sampled.Load(int3( ipos, 0 )).a))
    {
        const int sep = getCheckerboardSeparatorX();
        const int other_side_offset = (ipos.x < sep) ? sep : -sep;

        float4 a = safe_load(ipos + int2(other_side_offset, 1));
        float4 b = safe_load(ipos + int2(other_side_offset, -1));
        float4 c = safe_load(ipos + int2(other_side_offset, 0));
        float4 d = safe_load(ipos + int2(other_side_offset + (((opos.x & 1) != 0) ? 1 : -1), 0));

        if(opos.x == 0 || opos.x == (int)(globalUniform.renderWidth) - 1)
        {
            c = (float4)0;
            d = (float4)0;
        }

        float4 neighbors = (a + b + c + d) * 0.25;

        color.rgb = lerp(center.rgb, neighbors.rgb, 0.5);
        color.a = center.a;
    }

    // Write into PreFinal (regular layout), consumed by the rest of the post chain
    framebufPreFinal[opos] = float4(max((float3)0.0, color.rgb), 0);
}
