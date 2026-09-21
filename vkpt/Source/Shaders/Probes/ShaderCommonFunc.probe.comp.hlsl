// Hand written counterpart of GLSL/ShaderCommonFunc.probe.comp.
//
// The generated header has its own probe pair (ShaderCommon.probe.*), which covers the constants,
// the structs and every framebuffer. This pair covers the rest of the shader common surface:
// ShaderCommonHLSLFunc.hlsli against ShaderCommonGLSLFunc.h. Both probes define the same
// descriptor set indices, touch the same resources and call the same helper functions, so
// CheckShaderProperties.py can compare what glslc and dxc make of them.
//
// Every declaration of the header is touched, because dxc drops what stays unused. The two
// probes have to be edited together: a resource that is touched on one side only is reported as
// a mismatch, which is exactly the point.
//
// RayCone.hlsli and Media.hlsli have no probe pair of their own: neither declares a resource, but
// neither can be compiled without the accessors of this layer, so both are pinned from here instead
// (see the blocks before the store, and the same blocks in the GLSL half).

#define DESC_SET_GLOBAL_UNIFORM 0
#define DESC_SET_FRAMEBUFFERS   1
#define DESC_SET_TEXTURES       2
#define DESC_SET_TONEMAPPING    3
#define DESC_SET_LIGHT_SOURCES  4
#define DESC_SET_VOLUMETRIC     5
#define DESC_SET_LENS_FLARES    6
#define DESC_SET_DECALS         7

#include "ShaderCommonHLSLFunc.hlsli"

// MATERIAL_MAX_ALBEDO_LAYERS is normally defined by the consuming shader, as a specialization
// constant or as 0. RayCone.hlsli requires it.
#define MATERIAL_MAX_ALBEDO_LAYERS 3
#include "RayCone.hlsli"
#include "Media.hlsli"

#define PROBE_DESC_SET 8

[[vk::binding(0, PROBE_DESC_SET)]] RWStructuredBuffer<float> probeOutput;

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float v = 0.0;

    const int2 pix = int2(2, 3);
    const float2 uv = float2(0.5, 0.5);
    const float3 dir = float3(0.0, 0.0, 1.0);
    const float3 volUVW = float3(0.5, 0.5, 0.5);

    // ShGlobalUniform. The wrapper declares it, ShaderCommonGLSLFunc.h declares it on the GLSL
    // side; the ray direction helpers below index it.
    v += globalUniform.renderWidth + globalUniform.renderHeight +
         globalUniform.jitterX + globalUniform.jitterY +
         globalUniform.view[0][0] + globalUniform.invView[0][0] + globalUniform.invProjection[0][0];

    // DESC_SET_TEXTURES, the split bindless table
    v += getTextureSample(0, uv).x;
    v += getTextureSampleLod(1, uv, 0.0).y;
    v += getTextureSampleGrad(2, uv, uv, uv).z;

    // Index packing helpers
    int instanceId = 0;
    int instanceCustomIndex = 0;
    int geometryIndex = 0;
    int primitiveIndex = 0;

    unpackInstanceIdAndCustomIndex(packInstanceIdAndCustomIndex(1, 2), instanceId, instanceCustomIndex);
    unpackGeometryAndPrimitiveIndex(packGeometryAndPrimitiveIndex(1, 2), geometryIndex, primitiveIndex);

    v += float(packInstanceIdAndCustomIndex(1, 2)) +
         float(unpackInstanceIdAndCustomIndex(packInstanceIdAndCustomIndex(1, 2)).x) +
         float(packGeometryAndPrimitiveIndex(1, 2)) +
         float(unpackGeometryAndPrimitiveIndex(packGeometryAndPrimitiveIndex(1, 2)).y) +
         float(instanceId + instanceCustomIndex + geometryIndex + primitiveIndex);

    // DESC_SET_TONEMAPPING: a single instance block on the GLSL side, so every access there is
    // `tonemapping.field` and here it is `tonemapping[0].field`.
    v += tonemapping[0].tmExposureBias + tonemapping[0].tmExposureSpeedDown +
         tonemapping[0].tmExposureSpeedUp + tonemapping[0].tmLowPercentile +
         tonemapping[0].tmHighPercentile + tonemapping[0].tmMinLuminance + tonemapping[0].tmMaxLuminance +
         tonemapping[0].tmNoiseBlend + tonemapping[0].tmNoiseStops + tonemapping[0].tmDynRangeStops +
         tonemapping[0].tmReinhard + tonemapping[0].tmKneeStart + tonemapping[0].tmWhitePoint +
         tonemapping[0].tmSlopeBlurSigma + tonemapping[0].frameTime + float(tonemapping[0].resetCurve) +
         tonemapping[0].kneeW + tonemapping[0].kneeA + tonemapping[0].kneeB +
         float(tonemapping[0].histogram[0]) + tonemapping[0].curve[0] + tonemapping[0].normalized[0] +
         tonemapping[0].adaptedLuminance + tonemapping[0].avgLuminance;

    // DESC_SET_LIGHT_SOURCES
    v += lightSources[0].color.x + lightSources_Prev[0].color.x;
    v += float(lightSources_Index_PrevToCur[0]) + float(lightSources_Index_CurToPrev[0]) +
         float(q2LightListOffsets[0]) + float(q2LightListLights[0]) +
         float(talCdf[0]) + float(q2ClusterSkyVis[0]);
    q2LightStats[0] = 1u;

    // DESC_SET_VOLUMETRIC
    g_volumetric[int3(0, 0, 0)] = float4(v, v, v, v);
    g_illuminationVolume[int3(0, 0, 0)] = float4(v, v, v, v);

    v += g_volumetric_Sampled.SampleLevel(g_volumetric_Sampler, volUVW, 0.0).x;
    v += g_volumetric_Sampled_Prev.SampleLevel(g_volumetric_Sampler_Prev, volUVW, 0.0).x;
    v += g_illuminationVolume_Sampled.SampleLevel(g_illuminationVolume_Sampler, volUVW, 0.0).x;

    // DESC_SET_LENS_FLARES
    v += lensFlareCullingInput[0].positionToCheck_X;
    lensFlareDrawCmds[0].lensFlareDrawCmds[0] = lensFlareCullingInput[0];
    lensFlareDrawCmds[0].lensFlareDrawCmdsCount = 1u;

    // DESC_SET_DECALS
    v += decalInstances[0].transform[0][0];
    decalInstances[0].textureAlbedoAlpha = 0u;

    // DESC_SET_FRAMEBUFFERS: the SH accessors
    SH sh = irradianceToSH(float3(v, v, v), dir);

    imageStoreUnfilteredIndirectSH(pix, sh);
    imageStoreIndirAccumSH(pix, sh);
    imageStoreIndirPingSH(pix, sh);
    imageStoreIndirPongSH(pix, sh);

    sh = imageLoadUnfilteredIndirectSH(pix);
    sh = texelFetchUnfilteredIndirectSH(pix);
    sh = texelFetchIndirAccumSH(pix);
    sh = texelFetchIndirAccumSH_Prev(pix);
    sh = texelFetchSH(
        framebufUnfilteredIndirectSH_R_Sampled,
        framebufUnfilteredIndirectSH_G_Sampled,
        framebufUnfilteredIndirectSH_B_Sampled,
        pix);

    accumulateSH(sh, mixSH(sh, sh, 0.5), 0.5);
    multiplySH(sh, 0.5);

    // dxc only counts a resource as used when it is accessed directly, a texture that is
    // nothing but an argument of a helper is dropped from the module. Both probes touch the
    // SH textures explicitly so that the descriptors of both sides can be compared.
    v += framebufUnfilteredIndirectSH_R_Sampled.Load(int3(pix, 0)).x;
    v += framebufUnfilteredIndirectSH_G_Sampled.Load(int3(pix, 0)).y;
    v += framebufUnfilteredIndirectSH_B_Sampled.Load(int3(pix, 0)).z;

    v += framebufIndirAccumSH_R_Sampled.Load(int3(pix, 0)).x;
    v += framebufIndirAccumSH_G_Sampled.Load(int3(pix, 0)).y;
    v += framebufIndirAccumSH_B_Sampled.Load(int3(pix, 0)).z;
    v += framebufIndirAccumSH_R_Prev_Sampled.Load(int3(pix, 0)).x;
    v += framebufIndirAccumSH_G_Prev_Sampled.Load(int3(pix, 0)).y;
    v += framebufIndirAccumSH_B_Prev_Sampled.Load(int3(pix, 0)).z;

    v += getSHColor(sh).x + SHToIrradiance(sh, dir).x;

    // DESC_SET_FRAMEBUFFERS: normals, sky and the checkerboard/reprojection helpers
    imageStoreNormal(pix, texelFetchNormal(pix));
    imageStoreNormalGeometry(pix, texelFetchNormalGeometry(pix));

    v += texelFetchNormal_Prev(pix).x + texelFetchNormalGeometry_Prev(pix).x;
    v += float(texelFetchEncNormal(pix)) + float(texelFetchEncNormalGeometry(pix));

    v += float(textureGatherEncNormalGeometry_Prev(uv).x);
    v += float(textureGatherEncNormalGeometry_Prev(uv).w);

    v += isSkyPix(pix) ? 1.0 : 0.0;
    v += needResolveCheckerboard(pix) ? 1.0 : 0.0;

    v += float(wasOnlyPrimary(v)) + float(wasWithoutSplit(v)) + float(wasSplit(v));

    const int3 renderArea = getCheckerboardedRenderArea(pix);

    v += float(getCheckerboardSeparatorX()) + float(isRegularPixOdd(pix)) + float(isCheckerboardPixOdd(pix));
    v += float(getCheckerboardPix(pix).x) + float(getRegularPixFromCheckerboardPix(pix).x);
    v += float(renderArea.z);

    v += testPixInRenderArea(pix, renderArea) ? 1.0 : 0.0;
    v += testReprojectedDepth(v, v, 0.0) ? 1.0 : 0.0;
    v += testReprojectedNormal(dir, dir) ? 1.0 : 0.0;
    v += testReprojectedNormalEnc(dir, texelFetchEncNormal(pix)) ? 1.0 : 0.0;

    v += getAntilagAlpha(v, 1.0);

    v += getPrevScreenPos(float2(0.0, 0.0), pix).x;
    v += getPrevScreenPos(framebufAlbedo_Sampled, pix).y;

    // DESC_SET_GLOBAL_UNIFORM: the ray direction helpers
    v += getRayDir(uv).x + getRayDirAX(uv).y + getRayDirAY(uv).z + getPixelUVWithJitter(pix).x;

    // RayCone.hlsli: the cone helpers, and the gradient sample that they build out of the split
    // table
    RayCone rayCone;
    rayCone.width = 0.001;
    rayCone.spreadAngle = 0.01;
    propagateRayCone(rayCone, 1.0);

    const float3 worldNormal = float3(0.0, 1.0, 0.0);
    const float3 vertWorldPositions[3] = { float3(0.0, 0.0, 0.0), float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0) };
    const float2 vertTexCoords[3] = { float2(0.0, 0.0), float2(1.0, 0.0), float2(0.0, 1.0) };

    v += getUVDerivativesFromRayCone(rayCone, dir, worldNormal, vertWorldPositions, vertTexCoords).x;
    v += getWaterDerivU(rayCone, dir, worldNormal);

    // `triangle` is a reserved word in HLSL, hence `tri`
    ShTriangle tri;
    tri.positions[0] = vertWorldPositions[0];
    tri.positions[1] = vertWorldPositions[1];
    tri.positions[2] = vertWorldPositions[2];

    for (int layer = 0; layer < MATERIAL_MAX_ALBEDO_LAYERS; layer++)
    {
        // the three GLSL columns, wired into the two rows of the transposed declaration
        tri.layerTexCoord[layer][0] = float3(vertTexCoords[0].x, vertTexCoords[1].x, vertTexCoords[2].x);
        tri.layerTexCoord[layer][1] = float3(vertTexCoords[0].y, vertTexCoords[1].y, vertTexCoords[2].y);
    }

    const DerivativeSet derivSet = getTriangleUVDerivativesFromRayCone(tri, worldNormal, rayCone, dir);
    v += derivSet.u[0] + derivSet.u[1] + derivSet.u[2];

    v += getTextureSampleDerivU(3, uv, 0.01).x;
    v += getTextureSampleDerivSet(3, uv, derivSet, 0).y;

    // Media.hlsli: the medium helpers and the refraction direction, which the helper writes through
    // an out parameter
    v += getIndexOfRefraction(MEDIA_TYPE_WATER) + getIndexOfRefraction(MEDIA_TYPE_ACID) +
         getIndexOfRefraction(MEDIA_TYPE_GLASS) + getIndexOfRefraction(MEDIA_TYPE_VACUUM);
    v += getMediaTransmittance(MEDIA_TYPE_WATER, v).r + getMediaTransmittance(MEDIA_TYPE_ACID, v).g;
    v += getGlowingMediaFog(MEDIA_TYPE_ACID, v).b;

    float3 refractionDir = float3(0.0, 0.0, 0.0);
    if (calcRefractionDirection(1.0, 1.5, dir, worldNormal, refractionDir))
    {
        v += refractionDir.x;
    }

    v += float(getMediaTypeFromFlags(GEOM_INST_FLAG_MEDIA_TYPE_WATER | GEOM_INST_FLAG_MEDIA_TYPE_GLASS |
                                     GEOM_INST_FLAG_MEDIA_TYPE_ACID));
    v += isPortalFromFlags(GEOM_INST_FLAG_PORTAL) ? 1.0 : 0.0;
    v += isRefractFromFlags(GEOM_INST_FLAG_REFRACT) ? 1.0 : 0.0;
    v += isReflectFromFlags(GEOM_INST_FLAG_REFLECT) ? 1.0 : 0.0;

    // Not under any define
    v += rmeEmissionToScreenEmission(v);

    probeOutput[0] = v;
}
