// Hand written counterpart of GLSL/HitInfo.probe.comp.
//
// HitInfo.inl is the only header of the shader base that reads the geometry, the global uniform and
// the material maps at the same time, and it comes in three mutually exclusive spellings, which
// RaygenCommon.h includes one after another. Neither of the six existing pairs declares that
// combination of sets, so this one declares all three and drives all three spellings of the hit, the
// albedo only shortcut and both entry points of the bounce. What the probe is really there for is
// the descriptor sets and the block layouts: both halves declare the same set indices, touch the same
// resources and call the same helpers, so CheckShaderProperties.py can compare what glslc and dxc
// make of them. Probes are not part of the shader build, so no probe blob is ever shipped.
//
// RayCone.hlsli, TurbWarp.hlsli and BRDF.hlsli are compiled by the ShaderCommonFunc pair already;
// they are included here because HitInfo.hlsli cannot be built without them, and not pinned again
// beyond what it calls through itself.

#define DESC_SET_GLOBAL_UNIFORM 0
#define DESC_SET_VERTEX_DATA    1
#define DESC_SET_TEXTURES       2

#include "ShaderCommonHLSLFunc.hlsli"

// MATERIAL_MAX_ALBEDO_LAYERS is normally defined by the consuming shader, as a specialization
// constant or as 0. RayCone.hlsli requires it.
#define MATERIAL_MAX_ALBEDO_LAYERS 3
#include "RayCone.hlsli"
#include "TurbWarp.hlsli"

// BRDF.hlsli is pulled in for MIN_GGX_ROUGHNESS alone, which HitInfo.hlsli reads at the end of the
// hit. It includes Random.hlsli, whose DESC_SET_RANDOM block has to stay out of this probe: that set
// is pinned by its own pair.
#include "BRDF.hlsli"

// The three spellings of the hit, in the order and with the defines of RaygenCommon.h
#define HITINFO_INL_PRIM
#include "HitInfo.hlsli"
#undef HITINFO_INL_PRIM

#define HITINFO_INL_RFL
#include "HitInfo.hlsli"
#undef HITINFO_INL_RFL

#define HITINFO_INL_INDIR
#include "HitInfo.hlsli"
#undef HITINFO_INL_INDIR

#define PROBE_DESC_SET 8

[[vk::binding(0, PROBE_DESC_SET)]] RWStructuredBuffer<float> probeOutput;

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float v = 0.0;

    const float2 uv = float2(0.5, 0.5);
    const float3 dir = float3(0.0, 0.0, 1.0);

    // DESC_SET_VERTEX_DATA: the buffers are reached through the helpers of VertexData.hlsli only, so
    // every one of them is read straight here, as in VertexData.probe: a buffer that only a helper
    // touches can be dropped from the module, and then the two halves would no longer describe the
    // same set.
    v += float(g_staticVertices[0].cluster);
    v += float(g_dynamicVertices[0].lightStyles);
    v += float(staticIndices[0]);
    v += float(dynamicIndices[0]);
    v += geometryInstances[0].model[0][0];
    v += float(geomIndexPrevToCur[0]);
    v += float(g_dynamicVertices_Prev[0].packedColor);
    v += float(prevDynamicIndices[0]);

    // DESC_SET_TEXTURES, the split table and the size query that the emission sharp mask makes
    v += getTextureSample(0, uv).x;
    v += getTextureSampleLod(1, uv, 0.0).y;
    v += getTextureSampleGrad(2, uv, uv, uv).z;
    v += float(getTextureSize(3, 0).x);

    // The payload every spelling starts from
    ShPayload pl;
    pl.instIdAndIndex = 0u;
    pl.geomAndPrimIndex = 0u;
    pl.baryCoords = (float2)0.0;

    // The cone that the reflection spelling takes
    RayCone rayCone;
    rayCone.width = 0.001;
    rayCone.spreadAngle = 0.01;

    // HITINFO_INL_PRIM: the two auxiliary rays, the depth gradient and the emissive blend code
    float2 motion;
    float motionDepthLinear;
    float3 gradDepth;
    float depthNDC;
    float depthLinear;
    float screenEmission;
    uint emissionBlendCode;

    const ShHitInfo hPrim = getHitInfoPrimaryRay(
        pl, dir, dir, dir, motion, motionDepthLinear, gradDepth, depthNDC, depthLinear, screenEmission, emissionBlendCode);

    v += hPrim.albedo.x + hPrim.metallic + hPrim.normal.y + hPrim.roughness + hPrim.normalGeom.z +
         hPrim.emission + hPrim.hitPosition.x + float(hPrim.instCustomIndex) +
         float(hPrim.geometryInstanceFlags) + float(hPrim.portalIndex) + float(hPrim.cluster);
    v += motion.x + motionDepthLinear + gradDepth.y + depthNDC + depthLinear + screenEmission +
         float(emissionBlendCode);

    // HITINFO_INL_RFL: the same hit through a cone, with the virtual position that keeps the motion
    // of a reflection stable
    float3 virtualPosForMotion = dir;
    float rayLen = 0.0;

    const ShHitInfo hRfl = getHitInfoWithRayCone_ReflectionRefraction(
        pl, rayCone, dir, dir, dir, virtualPosForMotion, rayLen, motion, motionDepthLinear, screenEmission, emissionBlendCode);

    v += hRfl.albedo.y + hRfl.metallic + hRfl.normal.z + hRfl.roughness + hRfl.normalGeom.x +
         hRfl.emission + hRfl.hitPosition.y + float(hRfl.instCustomIndex) +
         float(hRfl.geometryInstanceFlags) + float(hRfl.portalIndex) + float(hRfl.cluster);
    v += virtualPosForMotion.z + rayLen + motion.y + motionDepthLinear + screenEmission +
         float(emissionBlendCode);

    // HITINFO_INL_INDIR: the albedo only shortcut, and the full hit under a bounce mip bias
    v += getHitInfoAlbedoOnly(pl).z;

    const ShHitInfo hIndir = getHitInfoBounce(pl, dir, 0.5, 0.1);

    v += hIndir.albedo.z + hIndir.metallic + hIndir.normal.x + hIndir.roughness + hIndir.normalGeom.y +
         hIndir.emission + hIndir.hitPosition.z + float(hIndir.instCustomIndex) +
         float(hIndir.geometryInstanceFlags) + float(hIndir.portalIndex) + float(hIndir.cluster);

    probeOutput[0] = v;
}
