/*
* Copyright (c) 2026 Sultim Tsyrendashiev
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

// HLSL counterpart of GLSL/RhiDebugTrace.rchit, the closest hit of the RHI debug pair: it reads no
// material, G-buffer or texture, and writes an id-based colour into the payload. The raygen stores
// that colour unchanged; RhiDebugPayload must stay identical in the pair's three files.
//
// The only include is ShaderCommonHLSLFunc.hlsli, and only for the INSTANCE_CUSTOM_INDEX_FLAG_*
// constants and the shared structs, as RtClsOpaque.rchit.hlsl includes it. With no DESC_SET_*
// macro defined the generated header declares no resource for this stage.
//
// Spellings that had to change, relative to RtClsOpaque.rchit.hlsl's measured notes:
//   * gl_InstanceCustomIndexEXT -> InstanceID() and gl_InstanceID -> InstanceIndex(), NOT the
//     other way round: dxc's InstanceID() is the user-provided instance identifier (SPIR-V
//     InstanceCustomIndexKHR, the engine's flags) and InstanceIndex() is the index inside the
//     acceleration structure (SPIR-V InstanceId, the TLAS placement index).
//   * the golden's hitAttributeEXT vec2 becomes the entry's second parameter, a user-defined
//     struct dxc requires; its single float2 read keeps the golden's value at offset 0.
//   * the golden's rayPayloadInEXT global becomes the entry's inout parameter; it is the only
//     payload type of the module, so dxc gives it location 0.
//   * fract() -> frac(), min()/clamp() keep their names, and vec3(...) -> float3(...).

#include "ShaderCommonHLSLFunc.hlsli"

struct RhiDebugPayload
{
    float3 color;
};

struct RhiDebugHitAttributes
{
    float2 baryCoords;
};

// A stable, well separated colour per TLAS instance: golden-ratio steps over the instance index,
// brightened so no channel is near zero.
float3 instanceDebugColor(const uint instanceIndex)
{
    return float3(
        0.25 + 0.75 * frac(float(instanceIndex) * 0.618034),
        0.25 + 0.75 * frac(float(instanceIndex) * 0.381966),
        0.25 + 0.75 * frac(float(instanceIndex) * 0.236068));
}

[shader("closesthit")]
void main(inout RhiDebugPayload payload, in RhiDebugHitAttributes attribs)
{
    // The hit attribute carries (u, v); the third barycentric coordinate is 1 - u - v. The
    // smallest of the three is the distance to the closest triangle edge, and it grows to 1/3 at
    // the centroid.
    const float3 bary = float3(1.0 - attribs.baryCoords.x - attribs.baryCoords.y, attribs.baryCoords.x, attribs.baryCoords.y);
    const float edge = min(bary.x, min(bary.y, bary.z));

    // The custom index carries the engine's instance flags (ASManager), so the debug colours can
    // separate the entities from the world without any G-buffer. A sky instance is not a surface
    // at all - doesPayloadContainHitInfo() of RaygenCommon.hlsli calls a sky hit a miss - so it
    // gets the fixed background colour of RhiDebugTrace.rmiss.
    const uint customIndex = (uint)InstanceID();

    if ((customIndex & INSTANCE_CUSTOM_INDEX_FLAG_SKY) != 0)
    {
        payload.color = float3(0.01, 0.01, 0.02);
        return;
    }

    float3 surfaceColor;

    if ((customIndex & INSTANCE_CUSTOM_INDEX_FLAG_FIRST_PERSON) != 0)
    {
        surfaceColor = float3(1.0, 0.85, 0.15);
    }
    else if ((customIndex & INSTANCE_CUSTOM_INDEX_FLAG_DYNAMIC) != 0)
    {
        surfaceColor = float3(1.0, 0.15, 0.65);
    }
    else
    {
        surfaceColor = instanceDebugColor((uint)InstanceIndex());
    }

    // The facet shade: 0 at the edges, 1 where the centroid reaches the scale, then clamped. The
    // lower bound keeps the darkest part of a surface at 35% instead of black, so the geometry
    // does not vanish into the background. saturate() is the golden's clamp(..., 0.0, 1.0).
    const float facet = saturate(edge * 3.0);
    payload.color = surfaceColor * (0.35 + 0.65 * facet);
}
