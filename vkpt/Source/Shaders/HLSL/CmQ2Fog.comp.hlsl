// Copyright (C) 2021, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) 2026 QuakeRay contributors
//
// This file is a port of fog.h from Quake 2 RTX (https://github.com/NVIDIA/Q2RTX),
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
// Q2RTX-style fog volumes, applied to the final HDR image on the new Q2RTX
// core path (before tonemapping). Per pixel the two closest fog volumes are
// found along the camera ray and analytically blended over [0, depth].
//
// HLSL counterpart of CmQ2Fog.comp. Pinned by the pair GLSL/CmQ2Fog.comp <-> this file,
// which pulls in Q2Fog.hlsli and pins its q2FindFogVolumes / q2ApplyFog through this pass.
//
// Spellings that had to change:
//   * layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in
//     -> [numthreads(16, 16, 1)] on main, and gl_GlobalInvocationID -> SV_DispatchThreadID
//   * ivec2/vec2/vec3/vec4/uvec4 -> int2/float2/float3/float4/uint4: the two fog words of
//     q2FindFogVolumes are a uint4 pair on both sides
//   * int(globalUniform.renderWidth) -> (int)globalUniform.renderWidth in the two bounds
//   * texelFetch(t, p, 0) -> t.Load(int3(p, 0)) on the throughput, depth-world and final
//     sampled views; imageStore(framebufFinal, pix, v) -> framebufFinal[pix] = v, the storage
//     image write of the port, keeping the zero alpha of the golden's vec4
//   * getPixelUVWithJitter, getRayDir and getCheckerboardPix keep their names; the golden
//     reaches them through ShaderCommonGLSLFunc.h, this file through its hlsli counterpart
//
// What did not change: the regular-pixel-space view ray, the checkerboarded depth and the
// split-flag early out that skips reflection/refraction pixels, the (depth > MAX_RAY_LENGTH)
// sky extension to 1e6 and the max(depth, 0.0) floor, the empty-pair early out on both
// fog1.w and fog2.w, and the q2ApplyFog call with its t and hdr arguments.

#define DESC_SET_FRAMEBUFFERS 0
#define DESC_SET_GLOBAL_UNIFORM 1
#include "ShaderCommonHLSLFunc.hlsli"
#include "Q2Fog.hlsli"

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const int2 pix = int2(dispatchThreadID.xy);

    if (pix.x >= (int)globalUniform.renderWidth || pix.y >= (int)globalUniform.renderHeight)
    {
        return;
    }

    // view ray (regular pixel space, like CmPrepareFinal.comp)
    const float2 inUV = getPixelUVWithJitter(pix);
    const float3 cameraRayDir = getRayDir(inUV);
    const float3 origin = globalUniform.cameraPosition.xyz;

    const int2 cbPix = getCheckerboardPix(pix);

    // This pass can only trace the primary camera ray; it cannot follow the
    // reflected/refracted path (Q2RTX integrates fog per-ray in the path
    // tracer instead). Skip reflection/refraction pixels - marking them with
    // fog along the primary ray produces artifacts at reflection boundaries.
    const float splitFlag = framebufThroughput_Sampled.Load(int3(cbPix, 0)).a;
    if (splitFlag != 0.0)
    {
        return;
    }

    // distance to the visible surface (checkerboarded depth)
    const float depth = framebufDepthWorld_Sampled.Load(int3(cbPix, 0)).r;
    // for sky pixels extend the fog to the end of the volumes
    const float t = (depth > MAX_RAY_LENGTH) ? 1e6 : max(depth, 0.0);

    uint4 fog1, fog2;
    q2FindFogVolumes(origin, cameraRayDir, 0.0, t, fog1, fog2);

    if (fog1.w == 0u && fog2.w == 0u)
    {
        return;
    }

    const float3 hdr = framebufFinal_Sampled.Load(int3(pix, 0)).rgb;
    framebufFinal[pix] = float4(q2ApplyFog(fog1, fog2, t, hdr), 0);
}
