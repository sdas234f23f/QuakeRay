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

// HLSL counterpart of GLSL/RhiDebugTrace.rgen, the raygen of the RHI debug pair: one primary ray
// per pixel, then the payload colour is stored to the engine's ALBEDO image. The pair is new code
// (not a port of a renderer pass), and its three stages -- RhiDebugTrace.rgen, .rmiss, .rchit --
// own no shared header, so RhiDebugPayload is repeated in all three and must stay identical.
//
// The descriptor binding is frozen to the mapping the RHI pipeline is built for, and it is the
// first three sets of the pass:
//   set 0, binding 0 -- the top-level acceleration structure (declared below, as RaygenCommon.hlsli
//                       declares the engine's own TLAS),
//   set 1, binding 0 -- the engine's global uniform ShGlobalUniform (ShaderCommonHLSL.hlsli
//                       declares it from BINDING_GLOBAL_UNIFORM once DESC_SET_GLOBAL_UNIFORM is
//                       given the value 1),
//   set 2, binding 0 -- the output image, the engine's ALBEDO framebuffer (the generated framebuffer
//                       block declares framebufAlbedo as an R11G11B10_FLOAT RWTexture2D at binding 0
//                       once DESC_SET_FRAMEBUFFERS is given the value 2).
// No other sets, no push constants, no texture table. The macros are per-file defines, exactly as
// in the engine's own shaders. The raygen is the only stage of the pair that binds a resource, so
// the three values above live only in this file; the miss and the closest hit declare none.
#define DESC_SET_TLAS 0
#define DESC_SET_GLOBAL_UNIFORM 1
#define DESC_SET_FRAMEBUFFERS 2

// The accessor header brings in the generated header (the ShGlobalUniform struct, the
// INSTANCE_MASK_* and INSTANCE_CUSTOM_INDEX_FLAG_* constants, the framebuffer block) and defines
// getRayDir() and getPixelUVWithJitter(), which is all this stage needs from the base.
#include "ShaderCommonHLSLFunc.hlsli"

[[vk::binding(BINDING_ACCELERATION_STRUCTURE_MAIN, DESC_SET_TLAS)]] RaytracingAccelerationStructure topLevelAS;

// The pair's payload: one colour. The closest hit fills it with the debug surface colour, the
// miss with the fixed background colour, and this raygen stores it unchanged.
struct RhiDebugPayload
{
    float3 color;
};

[shader("raygeneration")]
void main()
{
    // Every pixel is traced, checkerboarded or not, so the launch index is the output pixel. The
    // ray is reconstructed exactly as RaygenPrimary.hlsli reconstructs the engine's primary ray:
    // the jittered pixel UV, the direction through the inverse projection and the inverse view,
    // the origin from the uniform. DispatchRaysIndex() is the HLSL spelling of gl_LaunchIDEXT and
    // hands back a uint3, so the cast is the golden's ivec2() written the HLSL way.
    const int2 pix = (int2)DispatchRaysIndex().xy;
    const float2 inUV = getPixelUVWithJitter(pix);

    const float3 cameraOrigin = globalUniform.cameraPosition.xyz;
    const float3 cameraRayDir = getRayDir(inUV);

    // The engine's primary ray, inlined: getPrimaryVisibilityCullMask() and
    // getAdditionalRayFlags() of RaygenCommon.hlsli are one line each, but RaygenCommon.hlsli
    // requires the vertex-data, texture, random and light-source sets, which this pass does not
    // have, so the two values are written here instead of including the header. The cull mask is
    // the engine's world mask plus refraction and first-person geometry; the flags cull front
    // faces when the uniform asks for it.
    const uint cullMask = globalUniform.rayCullMaskWorld | INSTANCE_MASK_REFRACT | INSTANCE_MASK_FIRST_PERSON;
    const uint rayFlags = globalUniform.rayCullBackFaces != 0 ? RAY_FLAG_CULL_FRONT_FACING_TRIANGLES : 0;

    RhiDebugPayload payload;
    payload.color = (float3)0.0;

    // traceRayEXT(topLevelAS, flags, mask, sbtOffset, sbtStride, missIndex, origin, tmin,
    // direction, tmax, payloadLocation) becomes TraceRay() with a RayDesc and the payload as an
    // argument, as in RaygenCommon.hlsli. The payload type is the only one of the module, so dxc
    // gives it location 0, which is the PAYLOAD_INDEX_DEFAULT the golden declares.
    RayDesc rayDesc;
    rayDesc.Origin = cameraOrigin;
    rayDesc.TMin = globalUniform.primaryRayMinDist;
    rayDesc.Direction = cameraRayDir;
    rayDesc.TMax = globalUniform.rayLength;

    // missIndex 0: the debug SBT is the pass's own table, with RhiDebugTrace.rmiss as its first
    // miss entry. sbtRecordOffset and sbtRecordStride are 0, so the hit group is the TLAS
    // instance's own contribution (0 for fully opaque geometry, 1 for alpha-tested).
    TraceRay(
        topLevelAS,
        rayFlags,
        cullMask,
        0, 0,     // sbtRecordOffset, sbtRecordStride
        0,        // missIndex
        rayDesc,
        payload);

    // Both the hit and the miss write the payload, so the store cannot read uninitialized data.
    //
    // The row is flipped: the dispatch grid's row 0 is the view's top row (the engine's pixel-to-UV
    // convention, which the ray above must keep bit-for-bit), while the present samples ALBEDO
    // through NVRHI's raster convention, which maps ALBEDO's last row to the screen's top. Storing
    // grid row y at row height-1-y makes the two conventions cancel; without the flip the whole
    // traced frame comes out mirrored vertically on screen.
    framebufAlbedo[int2(pix.x, (int)globalUniform.renderHeight - 1 - pix.y)] = float4(payload.color, 0.0);
}
