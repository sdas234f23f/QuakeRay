// Copyright (c) 2026 f1ames0ff <f1am3sdev.github@protonmail.com>
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

#ifndef RAYGEN_PRIMARY_HLSLI_
#define RAYGEN_PRIMARY_HLSLI_




#if defined(RAYGEN_PRIMARY_SHADER) && defined(RAYGEN_REFL_REFR_SHADER)
    #error Only one of RAYGEN_PRIMARY_SHADER and RAYGEN_REFL_REFR_SHADER must be defined
#endif
#if !defined(RAYGEN_PRIMARY_SHADER) && !defined(RAYGEN_REFL_REFR_SHADER) && !defined(Q2_REFL_REFR_SHADER)
    #error RAYGEN_PRIMARY_SHADER, RAYGEN_REFL_REFR_SHADER or Q2_REFL_REFR_SHADER must be defined
#endif
#ifndef MATERIAL_MAX_ALBEDO_LAYERS
    #error MATERIAL_MAX_ALBEDO_LAYERS is not defined
#endif



#define DESC_SET_TLAS 0
#define DESC_SET_FRAMEBUFFERS 1
#define DESC_SET_GLOBAL_UNIFORM 2
#define DESC_SET_VERTEX_DATA 3
#define DESC_SET_TEXTURES 4
#define DESC_SET_RANDOM 5
#define DESC_SET_LIGHT_SOURCES 6
#define DESC_SET_CUBEMAPS 7
#define DESC_SET_RENDER_CUBEMAP 8
#define DESC_SET_PORTALS 9
#define DESC_SET_RAY_STATS 11
#define LIGHT_SAMPLE_METHOD (LIGHT_SAMPLE_METHOD_NONE)

#ifndef RAYGEN_PRIMARY_ENTRY_ATTR
    #define RAYGEN_PRIMARY_ENTRY_ATTR [shader("raygeneration")]
#endif

#include "ShaderCommonHLSLFunc.hlsli"
#include "RaygenCommon.hlsli"
#include "Q2Fog.hlsli"
#include "Q2Asvgf.hlsli"
#include "ParticleProxies.hlsli"

#if defined(Q2_REFL_REFR_SHADER)

/* Defined in RayClearance.hlsli, which only the Q2 reflect/refract raygen
   includes: its ray query must not reach the modules that do not use it. */
float traceClearance(float3 origin, float3 direction, float maxDistance, uint cullMask);

#define GLASS_ROUGHNESS_MAX_ANGLE 0.35

/* One GGX micro-normal of a pane for the view direction v, in world space: */
/* roughness is the microfacet distribution the ray leaves along. */
float3 sampleRoughNormal(float3 n, float3 v, float alpha, float2 u)
{
    const float3x3 basis = getONB(n);
    float oneOverPdf;
    const float3 m = sampleGGXVNDF(mul(transpose(basis), v), alpha, u.x, u.y, oneOverPdf);

    return normalize(mul(basis, m));
}

#endif

float2 getMotionVectorForUpscaler(const float2 motionCurToPrev)
{
    return motionCurToPrev;
}

#include "SkyMotion.hlsli"

void storeQ2GBuffer(
    const int2 pix,
    const float3 baseColor, float transparency, const float2 glassParams, const float3 glassColor,
    float metallic, float roughness,
    float depth,
    float halfConeAngle, float distToLight,
    const float3 transparentColor, float transparentAlpha,
    const float4 fogAccum,
    const uint cluster)
{
    if (globalUniform.coreQ2RTX == 0)
    {
        return;
    }

    framebufQ2ViewDepth[pix]                = (float4)depth;
    framebufQ2BaseColor[pix]                = float4(baseColor, transparency);
    /* The metallic buffer's free tail and the bounce throughput's third channel carry the
       pane's glass colour to the reflect/refract pass, which reads it back for the tint. */
    framebufQ2Metallic[pix]                 = float4(metallic, roughness, glassColor.g, glassColor.b);
    framebufQ2BounceThroughput[pix]         = float4(glassParams, glassColor.r, halfConeAngle);
    framebufQ2Transparent[pix]              = float4(transparentColor, transparentAlpha);
    framebufQ2GodRaysThroughputDist[pix]    = float4(1.0, 1.0, 1.0, distToLight);
    framebufQ2RngSeed[pix]                  = (uint4)getRandomSeed(pix, globalUniform.frameId);
    framebufQ2FogAccum[pix]                 = fogAccum;
    framebufQ2Cluster[pix]                  = (uint4)cluster;
}

void storeSky(
    const int2 pix, const float3 rayDir, bool calculateSkyAndStoreToAlbedo, const float3 throughput,
#if defined(RAYGEN_PRIMARY_SHADER)
    float firstHitDepthNDC,
#else
    bool wasSplit,
#endif
    const float4 fogAccum )
{
    framebufIsSky[pix] = (uint4)1;

    {
        float3 albedo;

        if (calculateSkyAndStoreToAlbedo)
        {
            albedo = getSkyPrimary(rayDir);
        }
        else
        {
            albedo = framebufAlbedo[getRegularPixFromCheckerboardPix(pix)].rgb;
        }

        framebufAlbedo[getRegularPixFromCheckerboardPix(pix)] = float4(albedo, 0.0);

        storeQ2GBuffer(pix, albedo, 1.0, (float2)0.0, (float3)1.0, 0.0, 1.0, MAX_RAY_LENGTH * 2.0, 0.0, MAX_RAY_LENGTH * 2.0, albedo, 1.0, fogAccum, ~0u);
    }

    float2 m = getMotionForInfinitePoint(rayDir);
    if (globalUniform.skyType == SKY_TYPE_PROCEDURAL && globalUniform.cloudLayerMotion.w > 0.0)
    {
        float share = renderCubemap.SampleLevel(renderCubemap_Sampler, rayDir, 0.0).a;
        m = lerp(m, getMotionForCloudLayer(rayDir, m), share);
    }

    imageStoreNormal(                       pix, (float3)0.0);
    imageStoreNormalGeometry(               pix, (float3)0.0);
    framebufMetallicRoughness[pix]          = (float4)0.0;
    framebufDepthWorld[pix]                 = (float4)(MAX_RAY_LENGTH * 2.0);
    framebufMotion[pix]                     = float4(m, 0.0, 0.0);
    framebufSurfacePosition[pix]            = (float4)SURFACE_POSITION_INCORRECT;
    framebufVisibilityBuffer[pix]           = (float4)UINT32_MAX;
    framebufViewDirection[pix]              = float4(rayDir, 0.0);
    framebufScreenEmisRT[getRegularPixFromCheckerboardPix( pix )] = (float4)0.0;
    framebufAcidFogRT[getRegularPixFromCheckerboardPix( pix )]    = (float4)0.0;
#if defined(RAYGEN_PRIMARY_SHADER)
    framebufPrimaryToReflRefr[pix]          = uint4(0, 0, PORTAL_INDEX_NONE, 0);
    framebufDepthGrad[pix]                  = (float4)0.0;
    framebufDepthNdc[getRegularPixFromCheckerboardPix(pix)] = (float4)clamp(firstHitDepthNDC, 0.0, 1.0);
    framebufMotionDlss[getRegularPixFromCheckerboardPix(pix)] = float4(getMotionVectorForUpscaler(m), 0.0, 0.0);
    framebufThroughput[pix]                 = float4(throughput, 0.0);
#else
    framebufThroughput[pix]                 = float4(throughput, wasSplit ? 1.0 : -1.0);
#endif
}

uint getNewRayMedia(int i, uint prevMedia, uint geometryInstanceFlags)
{
    if (i == 0 && globalUniform.cameraMediaType != MEDIA_TYPE_VACUUM)
    {
       return MEDIA_TYPE_VACUUM;
    }

    return getMediaTypeFromFlags(geometryInstanceFlags);
}

float3 getWaterNormal(const RayCone rayCone, const float3 rayDir, const float3 normalGeom, const float3 position, bool wasPortal)
{
    const float3x3 basis = getONB(normalGeom);
    const float2 baseUV = float2(dot(position, getColumn(basis, 0)), dot(position, getColumn(basis, 1)));


    float verticality = 1.0 - abs(dot(normalGeom, globalUniform.worldUpVector.xyz));

    float2 flowSpeedVertical = 10 * float2(dot(getColumn(basis, 0), globalUniform.worldUpVector.xyz),
                                           dot(getColumn(basis, 1), globalUniform.worldUpVector.xyz));

    float2 flowSpeedHorizontal = (float2)1.0;


    const float uvScale = 0.05 / globalUniform.waterTextureAreaScale;
    float2 speed0 = uvScale * lerp(flowSpeedHorizontal, flowSpeedVertical, verticality) * globalUniform.waterWaveSpeed;
    float2 speed1 = -0.9 * speed0 * lerp(1.0, -0.1, verticality);


    float derivU = globalUniform.waterTextureDerivativesMultiplier * 0.5 * uvScale * getWaterDerivU(rayCone, rayDir, normalGeom);

    if (wasPortal)
    {
        derivU *= 0.1;
    }


    float2 uv0 = uvScale * baseUV + globalUniform.time * speed0;
    float3 n0 = getTextureSampleDerivU(globalUniform.waterNormalTextureIndex, uv0, derivU).xyz;
    n0.xy = n0.xy * 2.0 - (float2)1.0;


    float2 uv1 = 0.8 * uvScale * baseUV + globalUniform.time * speed1;
    float3 n1 = getTextureSampleDerivU(globalUniform.waterNormalTextureIndex, uv1, derivU).xyz;
    n1.xy = n1.xy * 2.0 - (float2)1.0;


    float2 uv2 = 0.1 * (uvScale * baseUV + speed0 * sin(globalUniform.time * 0.5));
    float3 n2 = getTextureSampleDerivU(globalUniform.waterNormalTextureIndex, uv2, derivU).xyz;
    n2.xy = n2.xy * 2.0 - (float2)1.0;


    const float strength = globalUniform.waterWaveStrength;

    const float3 n = normalize(float3(0, 0, 1) + strength * (0.25 * n0 + 0.2 * n1 + 0.1 * n2));
    return mul(basis, n);
}

float3x3 lookAt(const float3 forward, const float3 worldUp)
{
    float3 right = cross(forward, worldUp);
    float3 up = cross(right, forward);

    return transpose(float3x3(right, up, forward));
}

float raygenPrimaryMod(float x, float y)
{
    return x - y * floor(x / y);
}

float3 getPortalNormal(const float3 baseNormal, const float3 inWorldOffset)
{
    if (globalUniform.twirlPortalNormal == 0)
    {
        return -baseNormal;
    }

    float phaseScale = 3;
    float timeScale = 3;
    float waveScale = 0.01;
    float tm = raygenPrimaryMod(timeScale * globalUniform.time, M_PI * 2);

    const float3x3 inLookAt_Plain = lookAt(-baseNormal, globalUniform.worldUpVector.xyz);
    const float2 localOffset_Plain = float2(dot(inWorldOffset, getColumn(inLookAt_Plain, 0)),
                                            dot(inWorldOffset, getColumn(inLookAt_Plain, 1)));

    float distance = length(localOffset_Plain);
    float angle = atan2(localOffset_Plain.y, localOffset_Plain.x);

    float phase = sin(phaseScale * sqrt(distance) + angle + tm) + 1.0;
    phase *= waveScale;
    phase *= clamp(distance / 20, 0, 1);

    float3 localN = { phase, phase, 1.0 };

    return mul(inLookAt_Plain, normalize(localN));
}

bool isBackface(const float3 normalGeom, const float3 rayDir)
{
    return dot(normalGeom, -rayDir) < 0.0;
}

float3 getNormal(const float3 position, const float3 normalFromMap, const float3 normalGeom, const RayCone rayCone, const float3 rayDir, bool isWater, bool wasPortal)
{
    if (isWater)
    {
        float3 n = normalGeom;

        if (isBackface(normalGeom, rayDir))
        {
            n *= -1;
        }

        return getWaterNormal(rayCone, rayDir, n, position, wasPortal);
    }
    else
    {
        float3 n = normalFromMap;

        if (isBackface(normalGeom, rayDir))
        {
           n *= -1;
        }

        return normalize(n);
    }
}

#if defined(RAYGEN_PRIMARY_SHADER)
float2 rtGlassPaneNormalOct(float3 normal)
{
    normal /= max(abs(normal.x) + abs(normal.y) + abs(normal.z), 0.001);
    float2 oct = normal.xy;
    if (normal.z < 0.0)
    {
        oct = (1.0 - abs(oct.yx)) * float2(oct.x >= 0.0 ? 1.0 : -1.0, oct.y >= 0.0 ? 1.0 : -1.0);
    }
    return oct;
}

RAYGEN_PRIMARY_ENTRY_ATTR
void main()
{
    const int2 regularPix = (int2)DispatchRaysIndex().xy;
    const int2 pix = getCheckerboardPix(regularPix);
    const float2 inUV = getPixelUVWithJitter(regularPix);

    const float3 cameraOrigin = globalUniform.cameraPosition.xyz;
    const float3 cameraRayDir = getRayDir(inUV);
    const float3 cameraRayDirAX = getRayDirAX(inUV);
    const float3 cameraRayDirAY = getRayDirAY(inUV);

    const uint randomSeed = getRandomSeed(pix, globalUniform.frameId);


    const ShPayload primaryPayload = tracePrimaryRay(cameraOrigin, cameraRayDir);
    rayStatsAdd(RAY_STATS_CATEGORY_PRIMARY, 1);
    framebufQ2GlassFilter[pix] = (float4)0.0;
    framebufQ2GlassReflection[regularPix] = (float4)0.0;


    const uint currentRayMedia = globalUniform.cameraMediaType;


    if (!doesPayloadContainHitInfo(primaryPayload))
    {
        float3 throughput = (float3)1.0;

        float4 q2FogAccum = (float4)0;
        if (globalUniform.coreQ2RTX != 0)
        {
            uint4 q2SkyFog1, q2SkyFog2;
            q2FindFogVolumes(cameraOrigin, cameraRayDir, 0.0, 1e6, q2SkyFog1, q2SkyFog2);
            q2FogAccum = q2SegmentFog(q2SkyFog1, q2SkyFog2, 1e6);
        }

        storeSky(pix, cameraRayDir, globalUniform.skyType != SKY_TYPE_RASTERIZED_GEOMETRY, throughput, MAX_RAY_LENGTH * 2.0, q2FogAccum);
        if (primaryPayload.glassFilter.z != 0.0)
        {
            framebufQ2GlassFilter[pix] = float4(primaryPayload.glassTint, 1.0 + primaryPayload.glassFilter.x);
            framebufQ2GlassReflection[regularPix] = float4(0.0, 0.0, 0.0, primaryPayload.glassFilter.w);
        }
        return;
    }


    float2 motionCurToPrev;
    float motionDepthLinearCurToPrev;
    float3 gradDepth;
    float firstHitDepthNDC;
    float firstHitDepthLinear;
    float screenEmission;
    uint emissionBlendCode;
    const ShHitInfo h = getHitInfoPrimaryRay(primaryPayload, cameraOrigin, cameraRayDirAX, cameraRayDirAY, motionCurToPrev, motionDepthLinearCurToPrev, gradDepth, firstHitDepthNDC, firstHitDepthLinear, screenEmission, emissionBlendCode);
    if (globalUniform.glassBlur == 0u && 
        globalUniform.reflectRefractMaxDepth > 0u &&
        (h.geometryInstanceFlags & GEOM_INST_FLAG_MEDIA_TYPE_GLASS) != 0u &&
        (h.geometryInstanceFlags & GEOM_INST_FLAG_IGNORE_REFRACT_AFTER) == 0u)
    {
        const float field = isRegularPixOdd(regularPix) == 0 ? -1.0 : 1.0;
        /* The mask's blue channel carries the pane's own view-space depth (the axis depth the
           raster particles' SV_Position.w interpolates to), which is what lets a raster sprite
           tell whether it stands behind this pane: RsParticle.frag discards the ones that do,
           because the reflect/refract raygen already traced their stand-in. The thickness the
           channel held before moved nowhere else; the (off-by-default) glass denoiser compares
           it only as a pane identity. */
        const float paneViewDepth = -mul(globalUniform.view, float4(h.hitPosition, 1.0)).z;
        framebufQ2GlassFilter[pix] = float4(rtGlassPaneNormalOct(h.normalGeom), paneViewDepth,
                                          field * (4.0 + h.roughness));
        framebufQ2GlassReflection[regularPix] = float4(motionCurToPrev, firstHitDepthLinear, motionDepthLinearCurToPrev);
    }
    if (primaryPayload.glassFilter.z != 0.0 && (primaryPayload.glassDistance < length(h.hitPosition - cameraOrigin) ||
        (h.geometryInstanceFlags & GEOM_INST_FLAG_MEDIA_TYPE_GLASS) != 0u))
    {
        const float field = globalUniform.reflectRefractMaxDepth > 0u && isRegularPixOdd(regularPix) == 0 ? -1.0 : 1.0;
        framebufQ2GlassFilter[pix] = float4(primaryPayload.glassTint, field * (1.0 + primaryPayload.glassFilter.x));
        framebufQ2GlassReflection[regularPix] = float4(0.0, 0.0, 0.0, primaryPayload.glassFilter.w);
    }


    float3 throughput = (float3)1.0;
    throughput *= getMediaTransmittance(currentRayMedia, firstHitDepthLinear);


    framebufIsSky[pix]                      = (uint4)0;
    framebufAlbedo[getRegularPixFromCheckerboardPix(pix)] = float4(h.albedo, 0.0);
    framebufScreenEmisRT[getRegularPixFromCheckerboardPix(pix)] = float4((globalUniform.debugShowFlags & DEBUG_SHOW_FLAG_LUMA) != 0 ? (float3)screenEmission : h.albedo * screenEmission * throughput , 0.0);
    framebufAcidFogRT[getRegularPixFromCheckerboardPix(pix)] = float4(getGlowingMediaFog(currentRayMedia, firstHitDepthLinear), 0);
    imageStoreNormal(                       pix, h.normal);
    imageStoreNormalGeometry(               pix, h.normalGeom);
    framebufMetallicRoughness[pix]          = float4(h.metallic, h.roughness, 0, 0);
    framebufDepthWorld[pix]                 = (float4)firstHitDepthLinear;
    float depthGrad = length(gradDepth.xy);
    if (globalUniform.q2DepthGradMode != 0u)
    {
        depthGrad = 1.0 / max(Q2_DEPTH_GRAD_MIN_STEP, gradDepth.z);
    }
    framebufDepthGrad[pix]                  = (float4)depthGrad;
    framebufMotion[pix]                     = float4(motionCurToPrev, motionDepthLinearCurToPrev, 0.0);
    framebufSurfacePosition[pix]            = float4(h.hitPosition, asfloat(h.instCustomIndex));
    framebufVisibilityBuffer[pix]           = packVisibilityBuffer(primaryPayload);
    framebufViewDirection[pix]              = float4(cameraRayDir, 0.0);
    framebufThroughput[pix]                 = float4(throughput, 0.0);

    framebufPrimaryToReflRefr[pix]          = uint4(h.geometryInstanceFlags, primaryPayload.instIdAndIndex, h.portalIndex, emissionBlendCode);

    framebufDepthNdc[getRegularPixFromCheckerboardPix(pix)] = (float4)clamp(firstHitDepthNDC, 0.0, 1.0);
    framebufMotionDlss[getRegularPixFromCheckerboardPix(pix)] = float4(getMotionVectorForUpscaler(motionCurToPrev), 0.0, 0.0);

    uint4 q2Fog1, q2Fog2;
    q2FindFogVolumes(cameraOrigin, cameraRayDir, 0.0, firstHitDepthLinear, q2Fog1, q2Fog2);
    const float4 q2FogAccum = q2SegmentFog(q2Fog1, q2Fog2, firstHitDepthLinear);
    storeQ2GBuffer(pix, h.albedo, h.transparency, h.glassParams, h.glassColor, h.metallic, h.roughness,
                   firstHitDepthLinear, 0.5 * length(cameraRayDir - cameraRayDirAX), firstHitDepthLinear,
                   (float3)0.0, 0.0, q2FogAccum, h.cluster);
}
#endif


#if defined(RAYGEN_REFL_REFR_SHADER)
RAYGEN_PRIMARY_ENTRY_ATTR
void main()
{
    if (globalUniform.reflectRefractMaxDepth == 0)
    {
        return;
    }


    const int2 regularPix = (int2)DispatchRaysIndex().xy;
    const int2 pix = getCheckerboardPix(regularPix);
    const float2 inUV = getPixelUVWithJitter(regularPix);

    const float3 cameraRayDir = getRayDir(inUV);

    if (isSkyPix(pix))
    {
        return;
    }



    const uint3 primaryToReflRefrBuf        = framebufPrimaryToReflRefr_Sampled.Load(int3(pix, 0)).rgb;
    ShHitInfo h;
    h.albedo                                = framebufAlbedo_Sampled.Load(int3(getRegularPixFromCheckerboardPix(pix), 0)).rgb;
    h.hitPosition                           = framebufSurfacePosition_Sampled.Load(int3(pix, 0)).xyz;
    h.geometryInstanceFlags                 = primaryToReflRefrBuf.r;
    h.portalIndex                           = primaryToReflRefrBuf.b;
    h.normalGeom                            = texelFetchNormalGeometry(pix);
    h.normal                                = texelFetchNormal(pix);
    h.roughness                             = framebufMetallicRoughness_Sampled.Load(int3(pix, 0)).g;
    const float3  motionBuf                 = framebufMotion_Sampled.Load(int3(pix, 0)).rgb;
    float2        motionCurToPrev           = motionBuf.rg;
    float         motionDepthLinearCurToPrev = motionBuf.b;
    float         firstHitDepthLinear        = framebufDepthWorld_Sampled.Load(int3(pix, 0)).r;
    float3        screenEmission            = framebufScreenEmisRT_Sampled.Load(int3(getRegularPixFromCheckerboardPix(pix), 0)).rgb;
    float3        acidFog                   = framebufAcidFogRT_Sampled.Load(int3(getRegularPixFromCheckerboardPix(pix), 0)).rgb;
    float3        throughput                = framebufThroughput_Sampled.Load(int3(pix, 0)).rgb;
    ShPayload currentPayload;
    currentPayload.instIdAndIndex           = primaryToReflRefrBuf.g;

    float4 q2FogAccum = framebufQ2FogAccum_Sampled.Load(int3(pix, 0));



    RayCone rayCone;
    rayCone.width = 0;
    rayCone.spreadAngle = globalUniform.cameraRayConeSpreadAngle;

    float fullPathLength = firstHitDepthLinear;
    float q2LastSegmentLen = 0.0;
    float3 prevHitPosition = h.hitPosition;
    bool wasSplit = false;
    bool wasPortal = false;
    float3 virtualPos = h.hitPosition;
    float3 rayDir = cameraRayDir;
    uint currentRayMedia = globalUniform.cameraMediaType;
    bool hitInfoWasOverwritten = false;


    propagateRayCone(rayCone, firstHitDepthLinear);



    for (int i = 0; i < globalUniform.reflectRefractMaxDepth; i++)
    {
        const uint instIndex = unpackInstanceIdAndCustomIndex(currentPayload.instIdAndIndex).y;


        bool isPixOdd = isCheckerboardPixOdd(pix) != 0;


        uint newRayMedia = getNewRayMedia(i, currentRayMedia, h.geometryInstanceFlags);

        bool isPortal = isPortalFromFlags(h.geometryInstanceFlags) && h.portalIndex != PORTAL_INDEX_NONE;
        bool toRefract = isRefractFromFlags(h.geometryInstanceFlags);
        bool toReflect = isReflectFromFlags( h.geometryInstanceFlags ) &&
                         h.roughness < globalUniform.minRoughness;


        if (!toReflect && !toRefract && !isPortal)
        {
            break;
        }


        const float curIndexOfRefraction = getIndexOfRefraction(currentRayMedia);
        const float newIndexOfRefraction = getIndexOfRefraction(newRayMedia);

        const float3 normal = getNormal(
            h.hitPosition,
            h.normal,
            h.normalGeom,
            rayCone,
            rayDir,
            !isPortal && ( newRayMedia == MEDIA_TYPE_WATER || currentRayMedia == MEDIA_TYPE_WATER ||
                           newRayMedia == MEDIA_TYPE_ACID || currentRayMedia == MEDIA_TYPE_ACID ),
            wasPortal );


        bool delaySplitOnNextTime = false;

        if ((h.geometryInstanceFlags & GEOM_INST_FLAG_NO_MEDIA_CHANGE) != 0)
        {
            throughput *= getMediaTransmittance(newRayMedia, 1.0);
            newRayMedia = currentRayMedia;

            delaySplitOnNextTime = (globalUniform.noBackfaceReflForNoMediaChange != 0) && isBackface(h.normalGeom, rayDir);
        }



        float3 rayOrigin = h.hitPosition;
        bool doSplit = !wasSplit;
        bool doRefraction;
        float3 refractionDir;
        float F;

        if (delaySplitOnNextTime)
        {
            doSplit = false;
            toRefract = true;
            isPixOdd = true;
        }

        if (toRefract && calcRefractionDirection(curIndexOfRefraction, newIndexOfRefraction, rayDir, normal, refractionDir))
        {
            doRefraction = isPixOdd;
            F = getFresnelSchlick(curIndexOfRefraction, newIndexOfRefraction, -rayDir, normal);
        }
        else
        {
            doRefraction = false;
            doSplit = false;
            F = 1.0;
        }

        if (doRefraction)
        {
            rayDir = refractionDir;
            throughput *= (1 - F);

            currentRayMedia = newRayMedia;
        }
        else if (isPortal)
        {
            const ShPortalInstance portal = portalInstances.g_portals[h.portalIndex];

            const float3 inCenter = portal.inPosition.xyz;
            const float3 inWorldOffset = h.hitPosition - inCenter;

            float3x3 inLookAt = lookAt(getPortalNormal(normal, inWorldOffset), globalUniform.worldUpVector.xyz);

            const float3 outCenter = portal.outPosition.xyz;
            const float3x3 outLookAt = lookAt(portal.outDirection.xyz,
                                              portal.outUp.xyz);

            rayDir = mul(outLookAt, mul(transpose(inLookAt), rayDir));

            const float2 localOffset = float2(dot(inWorldOffset, getColumn(inLookAt, 0)),
                                              dot(inWorldOffset, getColumn(inLookAt, 1)));

            rayOrigin = outCenter + localOffset.x * getColumn(outLookAt, 0) + localOffset.y * getColumn(outLookAt, 1);

            wasPortal = true;
        }
        else
        {
            rayDir = reflect(rayDir, normal);
            throughput *= F;
        }

        if (doSplit)
        {
            throughput *= 2;
            wasSplit = true;
        }


        if ((h.geometryInstanceFlags & GEOM_INST_FLAG_REFL_REFR_ALBEDO_MULT) != 0)
        {
            throughput *= h.albedo;
        }
        else if ((h.geometryInstanceFlags & GEOM_INST_FLAG_REFL_REFR_ALBEDO_ADD) != 0)
        {
            throughput += h.albedo;
        }


        currentPayload = traceReflectionRefractionRay(rayOrigin, rayDir, instIndex, h.geometryInstanceFlags, doRefraction);
        rayStatsMark(RAY_STATS_CATEGORY_REFLECTION_REFRACTION);


        if (!doesPayloadContainHitInfo(currentPayload))
        {
            throughput *= getMediaTransmittance(currentRayMedia, pow(abs(dot(rayDir, globalUniform.worldUpVector.xyz)), -3));

            uint4 q2SegFog1, q2SegFog2;
            q2FindFogVolumes(rayOrigin, rayDir, 0.0, 1e6, q2SegFog1, q2SegFog2);
            q2FogAccum = q2AlphaBlendPremultiplied(q2FogAccum, q2SegmentFog(q2SegFog1, q2SegFog2, 1e6));

            storeSky(pix, rayDir, true, throughput, wasSplit, q2FogAccum);
            return;
        }

        float rayLen;
        float emis;
        uint emisBlendCode;

        h = getHitInfoWithRayCone_ReflectionRefraction(
            currentPayload, rayCone,
            rayOrigin, rayDir, cameraRayDir,
            virtualPos,
            rayLen,
            motionCurToPrev, motionDepthLinearCurToPrev,
            emis,
            emisBlendCode,
            0.0f
        );

        uint4 q2SegFog1, q2SegFog2;
        q2FindFogVolumes(rayOrigin, rayDir, 0.0, rayLen, q2SegFog1, q2SegFog2);
        q2FogAccum = q2AlphaBlendPremultiplied(q2FogAccum, q2SegmentFog(q2SegFog1, q2SegFog2, rayLen));

        hitInfoWasOverwritten = true;
        throughput *= getMediaTransmittance(currentRayMedia, rayLen);
        propagateRayCone(rayCone, rayLen);
        fullPathLength += rayLen;
        q2LastSegmentLen = rayLen;
        prevHitPosition = h.hitPosition;
        screenEmission += h.albedo * emis * throughput;
        acidFog += getGlowingMediaFog(currentRayMedia, rayLen) * (doSplit ? 2.0 : 1.0);
    }


    if (!hitInfoWasOverwritten)
    {
        return;
    }


    framebufIsSky[pix]                      = (uint4)0;
    framebufAlbedo[getRegularPixFromCheckerboardPix(pix)] = float4(h.albedo, 0.0);
    framebufScreenEmisRT[getRegularPixFromCheckerboardPix(pix)] = float4(screenEmission + ( globalUniform.cameraMediaType != MEDIA_TYPE_ACID ? acidFog * 0.05 : (float3)0.0 ), 0.0);
    framebufAcidFogRT[getRegularPixFromCheckerboardPix(pix)] = float4(acidFog, 0);
    imageStoreNormal(                       pix, h.normal);
    imageStoreNormalGeometry(               pix, h.normalGeom);
    framebufMetallicRoughness[pix]          = float4(h.metallic, h.roughness, 0, 0);
    framebufDepthWorld[pix]                 = (float4)fullPathLength;
    framebufMotion[pix]                     = float4(motionCurToPrev, motionDepthLinearCurToPrev, 0.0);
    framebufSurfacePosition[pix]            = float4(h.hitPosition, asfloat(h.instCustomIndex));
    framebufVisibilityBuffer[pix]           = packVisibilityBuffer(currentPayload);
    framebufViewDirection[pix]              = float4(rayDir, 0.0);
    framebufThroughput[pix]                 = float4(throughput, wasSplit ? 1.0 : -1.0);

    const float q2HalfConeAngle = framebufQ2BounceThroughput_Sampled.Load(int3(pix, 0)).w;
    storeQ2GBuffer(pix, h.albedo, 1.0, (float2)0.0, (float3)1.0, h.metallic, h.roughness,
                   -fullPathLength, q2HalfConeAngle, q2LastSegmentLen,
                   (float3)0.0, 0.0, q2FogAccum, h.cluster);
}
#endif


#if defined(Q2_REFL_REFR_SHADER)
RAYGEN_PRIMARY_ENTRY_ATTR
void main()
{
    if (globalUniform.reflectRefractMaxDepth == 0)
    {
        return;
    }

    const int2 regularPix = (int2)DispatchRaysIndex().xy;
    const int2 pix = getCheckerboardPix(regularPix);
    const float2 inUV = getPixelUVWithJitter(regularPix);
    /* This pass owns the traced particle stand-ins' layer, and this zero is its per-frame clear:
       it runs before every early-out, so every regular pixel the composite can read is reset,
       and the writes of q2BlendParticleProxies below accumulate on top. A pixel that leaves the
       loop early keeps the zero, which is the composite's identity. */
    framebufQ2ParticleLayer[getRegularPixFromCheckerboardPix(pix)] = (float4)0.0;
    const float3 cameraRayDir = getRayDir(inUV);

    if (framebufIsSky.Load(pix).r != 0)
    {
        return;
    }

    const uint3 primaryToReflRefrBuf = framebufPrimaryToReflRefr_Sampled.Load(int3(pix, 0)).rgb;


    if (globalUniform.reflRefrEarlyOut != 0u)
    {
        const uint primaryFlags = primaryToReflRefrBuf.r;
        bool primaryNeedsReflRefr =
            (primaryFlags & (GEOM_INST_FLAG_MEDIA_TYPE_WATER | GEOM_INST_FLAG_MEDIA_TYPE_ACID |
                             GEOM_INST_FLAG_MEDIA_TYPE_GLASS)) != 0 ||
            (isPortalFromFlags(primaryFlags) && primaryToReflRefrBuf.b != PORTAL_INDEX_NONE);
        if (!primaryNeedsReflRefr && (primaryFlags & GEOM_INST_FLAG_REFLECT) != 0)
        {
            primaryNeedsReflRefr = framebufMetallicRoughness.Load(pix).g < globalUniform.minRoughness;
        }
        if (!primaryNeedsReflRefr)
        {
            return;
        }
    }

    ShHitInfo h;
    h.albedo                            = framebufAlbedo.Load(getRegularPixFromCheckerboardPix(pix)).rgb;
    h.hitPosition                       = framebufSurfacePosition.Load(pix).xyz;
    h.geometryInstanceFlags             = primaryToReflRefrBuf.r;
    h.portalIndex                       = primaryToReflRefrBuf.b;
    h.normalGeom                        = decodeNormal(framebufNormalGeometry.Load(pix).r);
    h.normal                            = decodeNormal(framebufNormal.Load(pix).r);
    h.metallic                          = framebufMetallicRoughness.Load(pix).r;
    h.roughness                         = framebufMetallicRoughness.Load(pix).g;
    const float3  motionBuf             = framebufMotion.Load(pix).rgb;
    float2        motionCurToPrev         = motionBuf.rg;
    float         motionDepthLinearCurToPrev = motionBuf.b;
    const float   firstHitDepthLinear     = framebufDepthWorld.Load(pix).r;
    float3        screenEmission          = framebufScreenEmisRT.Load(getRegularPixFromCheckerboardPix(pix)).rgb;
    float3        acidFog                 = framebufAcidFogRT.Load(getRegularPixFromCheckerboardPix(pix)).rgb;
    float3        throughput              = framebufThroughput.Load(pix).rgb;
    ShPayload currentPayload;
    currentPayload.instIdAndIndex       = primaryToReflRefrBuf.g;

    const float4 q2BaseColor              = framebufQ2BaseColor.Load(pix);
    /* The channel carries the primary surface's glass transparency (written by
       the primary pass), which the glass branch below absorbs the rays with. */
    h.transparency = clamp(q2BaseColor.a, 0.0, 1.0);
    const float4 q2BounceThroughput       = framebufQ2BounceThroughput.Load(pix);
    h.glassParams = q2BounceThroughput.xy;
    /* The primary pass stored the pane's glass colour beside its ior and thickness: one channel
       in the bounce throughput and the free tail of the metallic buffer. */
    const float3 paneGlassColor = float3(q2BounceThroughput.z, framebufQ2Metallic.Load(pix).zw);
    const float q2HalfConeAngle           = q2BounceThroughput.w;
    float4 q2Transparent                  = framebufQ2Transparent.Load(pix);

    float4 q2FogAccum                     = framebufQ2FogAccum.Load(pix);

    RayCone rayCone;
    rayCone.width = 0;
    rayCone.spreadAngle = globalUniform.cameraRayConeSpreadAngle;

    float fullPathLength = firstHitDepthLinear;
    float q2LastSegmentLen = 0.0;
    float3 prevHitPosition = h.hitPosition;
    bool wasSplit = false;
    bool wasPortal = false;
    float3 virtualPos = h.hitPosition;
    float3 rayDir = cameraRayDir;
    uint currentRayMedia = globalUniform.cameraMediaType;
    bool hitInfoWasOverwritten = false;
    bool pathReachedRefraction = false;
    const bool shaderGlassReflection = globalUniform.glassBlur != 0u &&
        (h.geometryInstanceFlags & GEOM_INST_FLAG_MEDIA_TYPE_GLASS) != 0u;
    float3 refrHitPosition = h.hitPosition;

    propagateRayCone(rayCone, firstHitDepthLinear);

    for (int i = 0; i < globalUniform.reflectRefractMaxDepth; i++)
    {
        const uint instIndex = unpackInstanceIdAndCustomIndex(currentPayload.instIdAndIndex).y;
        bool isPixOdd = isCheckerboardPixOdd(pix) != 0;

        uint newRayMedia = getNewRayMedia(i, currentRayMedia, h.geometryInstanceFlags);
        bool isPortal = isPortalFromFlags(h.geometryInstanceFlags) && h.portalIndex != PORTAL_INDEX_NONE;

        const bool primaryIsWater  = (h.geometryInstanceFlags & GEOM_INST_FLAG_MEDIA_TYPE_WATER) != 0;
        const bool primaryIsSlime  = (h.geometryInstanceFlags & GEOM_INST_FLAG_MEDIA_TYPE_ACID) != 0;
        const bool primaryIsGlass  = (h.geometryInstanceFlags & GEOM_INST_FLAG_MEDIA_TYPE_GLASS) != 0;
    /* the roughness of the surface this iteration leaves, for the next cone */
    float traversalBlur = 0.0f;
        const bool primaryIsChrome = (h.geometryInstanceFlags & GEOM_INST_FLAG_REFLECT) != 0 &&
                                     h.roughness < globalUniform.minRoughness;

        if (!primaryIsWater && !primaryIsSlime && !primaryIsGlass && !primaryIsChrome && !isPortal)
        {
            break;
        }

        const float3 normal = getNormal(h.hitPosition, h.normal, h.normalGeom, rayCone, rayDir,
                                        !isPortal && (newRayMedia == MEDIA_TYPE_WATER || currentRayMedia == MEDIA_TYPE_WATER ||
                                                      newRayMedia == MEDIA_TYPE_ACID || currentRayMedia == MEDIA_TYPE_ACID),
                                        wasPortal);

        float3 rayOrigin = h.hitPosition;
        bool doSplit = !wasSplit && !shaderGlassReflection;
        bool doRefraction = false;
        int correctMotionVector = 0;

        if (isPortal)
        {
            const ShPortalInstance portal = portalInstances.g_portals[h.portalIndex];

            const float3 inCenter = portal.inPosition.xyz;
            const float3 inWorldOffset = h.hitPosition - inCenter;

            float3x3 inLookAt = lookAt(getPortalNormal(normal, inWorldOffset), globalUniform.worldUpVector.xyz);

            const float3 outCenter = portal.outPosition.xyz;
            const float3x3 outLookAt = lookAt(portal.outDirection.xyz,
                                              portal.outUp.xyz);

            rayDir = mul(outLookAt, mul(transpose(inLookAt), rayDir));

            const float2 localOffset = float2(dot(inWorldOffset, getColumn(inLookAt, 0)),
                                              dot(inWorldOffset, getColumn(inLookAt, 1)));

            rayOrigin = outCenter + localOffset.x * getColumn(outLookAt, 0) + localOffset.y * getColumn(outLookAt, 1);

            wasPortal = true;
            doSplit = false;
        }
        else if (primaryIsWater || primaryIsSlime)
        {
            const float ior = getIndexOfRefraction(primaryIsWater ? MEDIA_TYPE_WATER : MEDIA_TYPE_ACID);
            const float3 reflected = reflect(rayDir, normal);
            const float nDotV = abs(dot(rayDir, normal));

            if (currentRayMedia == MEDIA_TYPE_WATER || currentRayMedia == MEDIA_TYPE_ACID)
            {
                const float3 refracted = refract(rayDir, normal, ior);
                float ndv = 1.0 - (1.0 - nDotV) * 3.0;
                if (ndv <= 0.0 || dot(refracted, refracted) == 0.0)
                {
                    rayDir = reflected;
                    correctMotionVector = 1;
                }
                else
                {
                    const float f = pow(1.0 - ndv, 5.0);
                    doRefraction = (i == 0) ? isPixOdd : isPixOdd;
                    if (doRefraction)
                    {
                        rayDir = refracted;
                        throughput *= (1.0 - f);
                        currentRayMedia = MEDIA_TYPE_VACUUM;
                        correctMotionVector = 2;
                    }
                    else
                    {
                        rayDir = reflected;
                        throughput *= f;
                        correctMotionVector = 1;
                    }
                    if (doSplit)
                    {
                        throughput *= 2.0;
                    }
                }
            }
            else
            {
                const float3 refracted = refract(rayDir, normal, 1.0 / ior);
                const float F = 0.1 + 0.9 * pow(1.0 - nDotV, 5.0);
                doSplit = (i == 0);
                doRefraction = isPixOdd;
                if (doRefraction)
                {
                    rayDir = refracted;
                    throughput *= (1.0 - F);
                    currentRayMedia = newRayMedia;
                    correctMotionVector = 2;
                }
                else
                {
                    rayDir = reflected;
                    throughput *= F;
                    correctMotionVector = 1;
                }
                if (doSplit)
                {
                    throughput *= 2.0;
                }
            }
        }
        else if (primaryIsGlass)
        {
            // per-material refraction: a material may carry its own index, and a
            // slab thickness that moves where the ray leaves the pane
            const float ior = (h.glassParams.x > 0.0f) ? clamp(h.glassParams.x, 1.0f, 5.0f)
                                                       : getIndexOfRefraction(MEDIA_TYPE_GLASS);
            const float thickness = max(h.glassParams.y, 0.0f);
            float3 glassGeomN = h.normalGeom;
            float3 glassN = normal;

            float gnDotV = dot(rayDir, glassGeomN);
            if (gnDotV > 0)
            {
                glassGeomN = -glassGeomN;
                glassN = -glassN;
                gnDotV = -gnDotV;
            }

            float3 entryN = globalUniform.glassBlur != 0u ? glassGeomN : glassN;

            float3 reflected = reflect(rayDir, entryN);
            if (globalUniform.glassBlur == 0u && !isPixOdd && !wasSplit && dot(reflected, glassGeomN) < 0.01)
            {
                entryN = glassGeomN;
                reflected = reflect(rayDir, entryN);
            }
            const float nDotV = dot(rayDir, -entryN);
            const float F0 = pow((1.0 - ior) / (1.0 + ior), 2.0);
            float F = F0 + (1.0 - F0) * pow(1.0 - abs(nDotV), 5.0);
            if (dot(reflected, glassGeomN) < 0.01)
            {
                F = 0.0;
            }

            /* The checkerboard split carries the pane's two images: one field refracts (the
               transmission) and the other reflects, both Fresnel-weighted, and CmQ2Interleave
               reconstructs the pair. The traced copy writes its own hit's motion into the
               upscaler's motion buffer further down, so the FSR/TAAU history follows the surface
               the pixel shows instead of the pane that happens to cover it. */
            doSplit = globalUniform.glassBlur == 0u && !wasSplit;
            doRefraction = globalUniform.glassBlur == 0u && h.transparency > 0.0f && (doSplit ? isPixOdd : true);
            if (doRefraction)
            {
                const float3 refr1 = refract(rayDir, entryN, 1.0 / ior);

                if (dot(refr1, refr1) == 0.0)
                {
                    /* the sampled facet passes nothing: the ray reflects off it */
                    rayDir = reflected;
                    throughput *= F;
                    correctMotionVector = 1;
                }
                else
                {
                    const float3 refr2 = refract(refr1, glassGeomN, ior);

                    if (dot(refr2, refr2) > 0.0)
                    {
                        rayDir = refr2;
                        if (thickness > 0.0)
                        {
                            /* the pane has depth: the ray leaves it at the virtual far
                               face, pulled back in front of whatever the pane covers --
                               a wall at the junction closer than the thickness would
                               leave the next ray starting inside it, and the culled
                               back face would read as a hole */
                            const float cosT = max(-dot(refr1, entryN), 0.1);
                            const uint clearanceMask = getReflectionRefractionCullMask(instIndex, h.geometryInstanceFlags, true);

                            rayOrigin += refr1 * traceClearance(rayOrigin, refr1, thickness / cosT, clearanceMask);
                        }
                        /* The pane's transmission: transparency is how much light passes (0
                           blocks it outright) and the diffuse texture times the glass colour
                           is the tint it is filtered into. */
                        throughput *= (1.0 - F);
                        throughput *= glassTransmissionFilter(h.albedo, paneGlassColor, h.transparency);
                        correctMotionVector = 2;
                    }
                    else
                    {
                        /* total internal reflection at the far face */
                        rayDir = reflect(rayDir, entryN);
                        throughput *= F;
                        correctMotionVector = 1;
                    }
                }
            }
            else
            {
                rayDir = reflected;
                throughput *= F;
                correctMotionVector = 1;
            }

            traversalBlur = 0.0;

            if (doSplit)
            {
                throughput *= 2.0;
            }
        }
        else
        {
            throughput *= h.albedo;
            rayDir = reflect(rayDir, normal);
            correctMotionVector = 1;
        }

        if (doSplit)
        {
            wasSplit = true;
        }

        currentPayload = traceReflectionRefractionRay(rayOrigin, rayDir, instIndex, h.geometryInstanceFlags, doRefraction);
        rayStatsMark(RAY_STATS_CATEGORY_REFLECTION_REFRACTION);

        if (!doesPayloadContainHitInfo(currentPayload))
        {
            const float skyLod = primaryIsGlass && globalUniform.glassBlur == 0u ? 0.0 : h.roughness * (SKY_MIP_COUNT - 1.0);
            const float3 env = getSkyVisibleFiltered(rayDir, skyLod);
            q2Transparent = q2AlphaBlendPremultiplied(float4(env * throughput, 1.0), q2Transparent);

            /* The segment leaves the pane and reaches the sky: the frame's particle stand-ins
               are the only thing between the two. */
            q2BlendParticleProxies(rayOrigin, rayDir, MAX_RAY_LENGTH * 2.0, throughput, pix);

            uint4 q2SegFog1, q2SegFog2;
            q2FindFogVolumes(rayOrigin, rayDir, 0.0, 1e6, q2SegFog1, q2SegFog2);
            q2FogAccum = q2AlphaBlendPremultiplied(q2SegmentFog(q2SegFog1, q2SegFog2, 1e6), q2FogAccum);

            if (correctMotionVector == 2 && isPixOdd)
            {
                const int2 refrPix = getRegularPixFromCheckerboardPix(pix);
                const int2 pairPix = int2(refrPix.x + (refrPix.x % 2 == 0 ? 1 : -1), refrPix.y);
                framebufDepthNdc[refrPix] = (float4)1.0;
                framebufDepthNdc[pairPix] = (float4)1.0;
            }

            storeSky(pix, rayDir, true, throughput, wasSplit, q2FogAccum);
            storeQ2GBuffer(pix, (float3)0.0, 1.0, (float2)0.0, (float3)1.0, 0.0, 1.0, -MAX_RAY_LENGTH * 2.0, q2HalfConeAngle, MAX_RAY_LENGTH * 2.0,
                           q2Transparent.rgb, q2Transparent.a, q2FogAccum, ~0u);

            /* The transport's spare RGB carries the segment's own origin to the
               god-rays reflection pass, whose sky continuation would otherwise
               march from the camera and count the primary segment twice. */
            framebufQ2GodRaysThroughputDist[pix] = float4(rayOrigin, MAX_RAY_LENGTH * 2.0);
            return;
        }

        float rayLen;
        float emis;
        uint emisBlendCode;

        h = getHitInfoWithRayCone_ReflectionRefraction(
            currentPayload, rayCone,
            rayOrigin, rayDir, cameraRayDir,
            virtualPos,
            rayLen,
            motionCurToPrev, motionDepthLinearCurToPrev,
            emis,
            emisBlendCode,
            traversalBlur
        );

        uint4 q2SegFog1, q2SegFog2;
        q2FindFogVolumes(rayOrigin, rayDir, 0.0, rayLen, q2SegFog1, q2SegFog2);
        q2FogAccum = q2AlphaBlendPremultiplied(q2SegmentFog(q2SegFog1, q2SegFog2, rayLen), q2FogAccum);

        /* The particle stand-ins between the pane and the surface the segment found: blended into
           the through-glass signal before the segment's material enters the compose, so the
           sprite's coverage hides the surface behind it exactly as an opaque-adjacent sprite does
           in the raster path. Every segment of the loop gets its own query: a stack of panes sees
           the sprites at each interface, at the cost of one more traversal on the pixels that
           recurse at all. */
        q2BlendParticleProxies(rayOrigin, rayDir, rayLen, throughput, pix);

        hitInfoWasOverwritten = true;
        if (correctMotionVector == 2 && isPixOdd)
        {
            pathReachedRefraction = true;
            refrHitPosition = h.hitPosition;
        }
        throughput *= getMediaTransmittance(currentRayMedia, rayLen);
        propagateRayCone(rayCone, rayLen);
        fullPathLength += rayLen;
        q2LastSegmentLen = rayLen;
        prevHitPosition = h.hitPosition;
        screenEmission += h.albedo * emis * throughput;
        acidFog += getGlowingMediaFog(currentRayMedia, rayLen) * (doSplit ? 2.0 : 1.0);
    }


    if (!hitInfoWasOverwritten)
    {
        return;
    }

    framebufIsSky[pix]                      = (uint4)0;
    framebufAlbedo[getRegularPixFromCheckerboardPix(pix)] = float4(h.albedo, 0.0);
    framebufScreenEmisRT[getRegularPixFromCheckerboardPix(pix)] = float4(screenEmission + ( globalUniform.cameraMediaType != MEDIA_TYPE_ACID ? acidFog * 0.05 : (float3)0.0 ), 0.0);
    framebufAcidFogRT[getRegularPixFromCheckerboardPix(pix)] = float4(acidFog, 0);
    imageStoreNormal(                       pix, h.normal);
    imageStoreNormalGeometry(               pix, h.normalGeom);
    framebufMetallicRoughness[pix]          = float4(h.metallic, h.roughness, 0, 0);
    framebufDepthWorld[pix]                 = (float4)fullPathLength;
    if (pathReachedRefraction)
    {
        const float4 refrClipPos = mul(mul(globalUniform.projection, globalUniform.view), float4(refrHitPosition, 1.0));
        const float refrDepthNdc = clamp(refrClipPos.z / refrClipPos.w + 1e-5, 0.0, 1.0);
        const int2 refrPix = getRegularPixFromCheckerboardPix(pix);
        const int2 pairPix = int2(refrPix.x + (refrPix.x % 2 == 0 ? 1 : -1), refrPix.y);
        framebufDepthNdc[refrPix] = (float4)refrDepthNdc;
        framebufDepthNdc[pairPix] = (float4)refrDepthNdc;
    }
    framebufMotion[pix]                     = float4(motionCurToPrev, motionDepthLinearCurToPrev, 0.0);
    /* The upscaler reprojects what the pixel shows, and for a pane that is the refracted hit,
       not the pane surface the primary pass wrote its own motion vector for. Without this the
       FSR/TAAU history follows the pane and smears the transmitted image. */
    framebufMotionDlss[getRegularPixFromCheckerboardPix(pix)] = float4(getMotionVectorForUpscaler(motionCurToPrev), 0.0, 0.0);
    framebufSurfacePosition[pix]            = float4(h.hitPosition, asfloat(h.instCustomIndex));
    framebufVisibilityBuffer[pix]           = packVisibilityBuffer(currentPayload);
    framebufViewDirection[pix]              = float4(rayDir, 0.0);
    framebufThroughput[pix]                 = float4(throughput, wasSplit ? 1.0 : -1.0);

    storeQ2GBuffer(pix, h.albedo, h.transparency, h.glassParams, h.glassColor, h.metallic, h.roughness,
                   -fullPathLength, q2HalfConeAngle, q2LastSegmentLen,
                   q2Transparent.rgb, q2Transparent.a, q2FogAccum, h.cluster);
}
#endif

#endif
