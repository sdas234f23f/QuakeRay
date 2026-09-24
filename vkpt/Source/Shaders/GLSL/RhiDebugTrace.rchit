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

#version 460
#extension GL_EXT_ray_tracing : require

// The closest hit of the debug pair: it does not read a single material, G-buffer or texture. It
// writes an id-based colour into the payload, and the raygen stores that colour unchanged. The
// pair's three files must keep RhiDebugPayload identical.
//
// The colour is built from two things the closest hit has without any descriptor:
//   * the instance's custom index, which ASManager fills with the engine's instance flags, so
//     the entities can be told from the world categorically:
//       - first person (the weapon)    -- yellow,
//       - dynamic geometry (entities)  -- magenta,
//       - anything else (static world) -- a stable colour per TLAS instance,
//     and a sky instance is not a surface at all: doesPayloadContainHitInfo() of RaygenCommon.h
//     calls a sky hit a miss, so it gets the same fixed background colour as RhiDebugTrace.rmiss;
//   * the hit's barycentric coordinates, which shade every triangle from its edges (dark)
//     towards its centroid (bright). That is what keeps the world readable where whole
//     instances share one flat colour.
// The only include is ShaderCommonGLSLFunc.h, and only for the INSTANCE_CUSTOM_INDEX_FLAG_*
// constants and the shared structs; with no DESC_SET_* macro defined the header declares no
// resource for this stage.

#include "ShaderCommonGLSLFunc.h"

struct RhiDebugPayload
{
    vec3 color;
};

layout(location = PAYLOAD_INDEX_DEFAULT) rayPayloadInEXT RhiDebugPayload payload;
hitAttributeEXT vec2 inBaryCoords;

// A stable, well separated colour per TLAS instance: golden-ratio steps over the instance index,
// brightened so no channel is near zero.
vec3 instanceDebugColor(const uint instanceIndex)
{
    return vec3(
        0.25 + 0.75 * fract(float(instanceIndex) * 0.618034),
        0.25 + 0.75 * fract(float(instanceIndex) * 0.381966),
        0.25 + 0.75 * fract(float(instanceIndex) * 0.236068));
}

void main()
{
    // The hit attribute carries (u, v); the third barycentric coordinate is 1 - u - v. The
    // smallest of the three is the distance to the closest triangle edge, and it grows to 1/3 at
    // the centroid.
    const vec3 bary = vec3(1.0 - inBaryCoords.x - inBaryCoords.y, inBaryCoords.x, inBaryCoords.y);
    const float edge = min(bary.x, min(bary.y, bary.z));

    // gl_InstanceCustomIndexEXT is the instance's custom index (SPIR-V
    // InstanceCustomIndexKHR), the engine's instance flags, and gl_InstanceID is the TLAS
    // instance index (SPIR-V InstanceId) - the same two values RtClsOpaque.rchit packs into the
    // engine's payload.
    const uint customIndex = uint(gl_InstanceCustomIndexEXT);

    if ((customIndex & INSTANCE_CUSTOM_INDEX_FLAG_SKY) != 0)
    {
        payload.color = vec3(0.01, 0.01, 0.02);
        return;
    }

    vec3 surfaceColor;

    if ((customIndex & INSTANCE_CUSTOM_INDEX_FLAG_FIRST_PERSON) != 0)
    {
        surfaceColor = vec3(1.0, 0.85, 0.15);
    }
    else if ((customIndex & INSTANCE_CUSTOM_INDEX_FLAG_DYNAMIC) != 0)
    {
        surfaceColor = vec3(1.0, 0.15, 0.65);
    }
    else
    {
        surfaceColor = instanceDebugColor(uint(gl_InstanceID));
    }

    // The facet shade: 0 at the edges, 1 where the centroid reaches the scale, then clamped. The
    // lower bound keeps the darkest part of a surface at 35% instead of black, so the geometry
    // does not vanish into the background.
    const float facet = clamp(edge * 3.0, 0.0, 1.0);
    payload.color = surfaceColor * (0.35 + 0.65 * facet);
}
