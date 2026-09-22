// Hand written counterpart of GLSL/Surface.probe.comp.
//
// Surface.inl has no shader stage of its own, so this pair exists to pin Surface.hlsli: the struct,
// its every member, all three functions of the header and every framebuffer they read are touched,
// so that glslang and dxc both keep them in the module and CheckShaderProperties.py can compare
// both the descriptors and the layout of Surface.
//
// The header is only complete with the defines that RaygenCommon.h sets before including it: the
// global uniform, because CHECKERBOARD_FULL_WIDTH and CHECKERBOARD_FULL_HEIGHT are what makes the
// two fetch functions exist, and the framebuffers, which the fetch functions read.
// FRAMEBUF_IGNORE_ATTACHMENTS has to stay undefined, as in RaygenCommon.h.
//
// Surface is never a buffer in the engine, it is only ever returned by value, so no shipped
// interface depends on its layout; the buffer below exists only so that the checker gets a layout
// for the struct from both compilers, and the golden layout is the std430 one it derives from the
// GLSL half.
//
// The three instances below are kept in a buffer and read back, because a store alone does not keep
// the struct in the module in the same shape on both sides.

#define DESC_SET_GLOBAL_UNIFORM 0
#define DESC_SET_FRAMEBUFFERS   1

#include "ShaderCommonHLSLFunc.hlsli"

#include "Surface.hlsli"

#define PROBE_DESC_SET 8

[[vk::binding(0, PROBE_DESC_SET)]] RWStructuredBuffer<Surface> probeSurfaces;
[[vk::binding(1, PROBE_DESC_SET)]] RWStructuredBuffer<float>   probeOutput;

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float v = 0.0;

    const int2 pix = int2(1, 2);

    // All three functions of the header, both under the gating of Surface.inl
    const Surface fromGbuffer     = fetchGbufferSurface(pix);
    const Surface fromGbufferPrev = fetchGbufferSurface_NoAlbedoViewDir_Prev(pix);

    ShHitInfo h;
    h.albedo                = float3(1.0, 0.0, 0.0);
    h.metallic              = 0.5;
    h.normal                = float3(0.0, 0.0, 1.0);
    h.roughness             = 0.5;
    h.normalGeom            = float3(0.0, 0.0, 1.0);
    h.emission              = 0.25;
    h.hitPosition           = float3(0.0, 1.0, 0.0);
    h.instCustomIndex       = 3u;
    h.geometryInstanceFlags = 0u;
    h.portalIndex           = 0u;
    h.cluster               = 7u;

    const Surface fromHitInfo = hitInfoToSurface_Indirect(h, float3(0.0, 0.0, 1.0));

    // Every member of every instance. The sky branch of fetchGbufferSurface returns before it fills
    // position, instCustomIndex, normalGeom, roughness, normal, toViewerDir and cluster, and
    // fetchGbufferSurface_NoAlbedoViewDir_Prev returns before it fills cluster, so those reads are
    // here only to keep the members in the module, exactly like on the GLSL side.
    v += fromGbuffer.position.x + float(fromGbuffer.instCustomIndex) + fromGbuffer.normalGeom.y;
    v += fromGbuffer.roughness + fromGbuffer.normal.z + fromGbuffer.albedo.x;
    v += float(fromGbuffer.isSky) + fromGbuffer.specularColor.y + fromGbuffer.emission;
    v += fromGbuffer.toViewerDir.z + float(fromGbuffer.cluster);

    v += fromGbufferPrev.position.y + float(fromGbufferPrev.instCustomIndex) + fromGbufferPrev.normalGeom.z;
    v += fromGbufferPrev.roughness + fromGbufferPrev.normal.x + fromGbufferPrev.albedo.y;
    v += float(fromGbufferPrev.isSky) + fromGbufferPrev.specularColor.z + fromGbufferPrev.emission;
    v += fromGbufferPrev.toViewerDir.x + float(fromGbufferPrev.cluster);

    v += fromHitInfo.position.z + float(fromHitInfo.instCustomIndex) + fromHitInfo.normalGeom.x;
    v += fromHitInfo.roughness + fromHitInfo.normal.y + fromHitInfo.albedo.z;
    v += float(fromHitInfo.isSky) + fromHitInfo.specularColor.x + fromHitInfo.emission;
    v += fromHitInfo.toViewerDir.y + float(fromHitInfo.cluster);

    // And the same members again through the buffer, so that the struct itself is part of both
    // modules and gets a layout from both compilers
    probeSurfaces[0] = fromGbuffer;
    probeSurfaces[1] = fromGbufferPrev;
    probeSurfaces[2] = fromHitInfo;

    v += probeSurfaces[0].position.x + float(probeSurfaces[0].instCustomIndex) + probeSurfaces[0].normalGeom.x;
    v += probeSurfaces[0].roughness + probeSurfaces[0].normal.x + probeSurfaces[0].albedo.x;
    v += float(probeSurfaces[0].isSky) + probeSurfaces[0].specularColor.x + probeSurfaces[0].emission;
    v += probeSurfaces[0].toViewerDir.x + float(probeSurfaces[0].cluster);

    v += probeSurfaces[1].position.x + float(probeSurfaces[1].instCustomIndex) + probeSurfaces[1].normalGeom.x;
    v += probeSurfaces[1].roughness + probeSurfaces[1].normal.x + probeSurfaces[1].albedo.x;
    v += float(probeSurfaces[1].isSky) + probeSurfaces[1].specularColor.x + probeSurfaces[1].emission;
    v += probeSurfaces[1].toViewerDir.x + float(probeSurfaces[1].cluster);

    v += probeSurfaces[2].position.x + float(probeSurfaces[2].instCustomIndex) + probeSurfaces[2].normalGeom.x;
    v += probeSurfaces[2].roughness + probeSurfaces[2].normal.x + probeSurfaces[2].albedo.x;
    v += float(probeSurfaces[2].isSky) + probeSurfaces[2].specularColor.x + probeSurfaces[2].emission;
    v += probeSurfaces[2].toViewerDir.x + float(probeSurfaces[2].cluster);

    probeOutput[0] = v;
}
