// Copyright (c) 2021-2022 Sultim Tsyrendashiev
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.



// HLSL counterpart of RaygenPrimary.inl: the three raygen entry points of the renderer (primary
// visibility, vkpt reflections/refractions and the Q2RTX-style reflect/refract pass) with the helper
// ray casts, the Q2RTX G-buffer store, the sky store, the water normal and the portal
// transformations. RtRaygenPrimary.rgen, RtRaygenReflRefr.rgen and RtQ2ReflectRefract.rgen are the
// three consumers of the GLSL one; each defines exactly one of the three entry point macros and
// includes this file afterwards, and this file keeps that shape.
//
// The include order is the golden's: ShaderCommonHLSLFunc.hlsli first (the accessor layer the whole
// base stands on), then RaygenCommon.hlsli, Q2Fog.hlsli and Q2Asvgf.hlsli, in the order the golden
// names them at :55-57. RaygenPrimary.inl is an unguarded fragment on the GLSL side because each
// consumer processes it exactly once; a diamond include of the ported header would not be, so this
// file carries the RAYGEN_PRIMARY_HLSLI_ guard that ShaderCommonHLSL.hlsli's guard rule asks for.
//
// The two blocks of the golden that only exist per configuration are kept word for word: the three
// way guard of :31-39 (exactly one entry point macro, plus MATERIAL_MAX_ALBEDO_LAYERS), the
// DESC_SET_* indices of :43-53 and LIGHT_SAMPLE_METHOD at :54.
//
// The one name this port adds is the entry point attribute macro below. HLSL needs
// [shader("raygeneration")] on the function that becomes the OpEntryPoint, while GLSL has no
// attribute at all; the macro keeps the attribute in the header (a consumer is then the golden's
// three lines and an include) and still lets a probe drop it, because the probe half has an entry
// point of its own and a translation unit must not turn the golden's main into a second one. The
// probe pair GLSL/RaygenPrimary.probe.rgen <-> Probes/RaygenPrimary.probe.rgen.hlsl is written that
// way and pins every matrix site of this file on folded constants; the sites are marked
// MATRIX SITE below, with the golden's line number.
//
// Spellings that had to change:
//   * vec2/vec3/vec4 -> float2/float3/float4, ivec2 -> int2, uvec3/uvec4 -> uint3/uint4, mat3 ->
//     float3x3. No declaration changes shape here except mat3, and the four "uint(...)" casts of
//     the golden become C style casts of the same truncating conversion.
//   * the single scalar constructors of the golden become casts: vec2(1.0) -> (float2)1.0 (three
//     sites in getWaterNormal), vec4(depth) -> (float4)depth, vec4(0) -> (float4)0,
//     vec4(1.0) -> (float4)1.0, vec4(0.0) -> (float4)0.0, vec4(MAX_RAY_LENGTH * 2.0) ->
//     (float4)(MAX_RAY_LENGTH * 2.0), vec4(UINT32_MAX) -> (float4)UINT32_MAX, vec3(0.0) ->
//     (float3)0.0, vec4(clamp(...)) -> (float4)clamp(...), ivec4(1) -> (uint4)1, uvec4(1) ->
//     (uint4)1, uvec4(cluster) -> (uint4)cluster, uvec4(getRandomSeed(...)) ->
//     (uint4)getRandomSeed(...), vec3(screenEmission) -> (float3)screenEmission. Constructors that
//     collect several distinct components are copied as they are.
//   * imageStore(img, pix, v) -> img[pix] = v and imageLoad(img, pix) -> img[pix], the two
//     substitutions of the whole pass (ShaderCommonHLSLFunc.hlsli). The cast of the stored value
//     keeps the golden's conversion, and the uimage2D/uimage2D targets take the uint4 the golden
//     stores.
//   * texelFetch(t, pix, 0) -> t.Load(int3(pix, 0)): the nine samplerless reads of
//     framebufPrimaryToReflRefr_Sampled, framebufAlbedo_Sampled, framebufSurfacePosition_Sampled,
//     framebufMetallicRoughness_Sampled, framebufMotion_Sampled, framebufDepthWorld_Sampled,
//     framebufScreenEmisRT_Sampled, framebufAcidFogRT_Sampled, framebufThroughput_Sampled,
//     framebufQ2FogAccum_Sampled, framebufQ2BaseColor_Sampled, framebufQ2BounceThroughput_Sampled
//     and framebufQ2Transparent_Sampled.
//   * gl_LaunchIDEXT.xy -> DispatchRaysIndex().xy: HLSL spells the raygen builtin as the
//     DispatchRaysIndex() intrinsic and hands back a uint3, so the ivec2(...) of the golden is the
//     same truncating conversion written as the (int2) cast.
//   * mix -> lerp (three sites in getWaterNormal, one in each storeQ2GBuffer call).
//   * mod(x, y) -> raygenPrimaryMod(x, y): GLSL mod() is x - y * floor(x / y), while the nearest
//     HLSL function, fmod(), truncates towards zero and would differ for a negative first operand.
//     The one call site of the golden, in getPortalNormal, takes globalUniform.time, so the two
//     agree on every input the shader can see today; the port does not rely on that and spells the
//     floor out. The helper is the first name added by this file.
//   * atan(y, x) -> atan2(y, x), the two argument form of the golden (getPortalNormal).
//   * uintBitsToFloat -> asfloat (two sites, framebufSurfacePosition).
//   * the matrix spellings of ShaderCommonHLSL.hlsli: the four truncating casts
//     mat3(globalUniform.view|viewPrev|projection|projectionPrev) -> (float3x3)globalUniform.* with
//     the product order kept, mul(M, v); the single index reads basis[0]/basis[1] and
//     inLookAt_Plain[0]/[1] and inLookAt[0]/[1] and outLookAt[0]/[1] -> getColumn(m, i); the
//     constructor mat3(right, up, forward) -> transpose(float3x3(right, up, forward)).
//   * the file's two "1 " literals, "1 - F" and "1 - src.a"-style promotions do not exist here, but
//     "1e6" and "~0u" are copied: HLSL reads them as double and uint exactly as GLSL does.
//
// What did not change: the three way configuration and its #errors, the DESC_SET_* block, the body
// and the order of every one of the eleven functions, every constant of the golden (MEDIA_TYPE_*,
// GEOM_INST_FLAG_*, PORTAL_INDEX_NONE, SURFACE_POSITION_INCORRECT, MAX_RAY_LENGTH,
// Q2_DEPTH_GRAD_MIN_STEP, SKY_MIP_COUNT, DEBUG_SHOW_FLAG_LUMA, RAY_STATS_CATEGORY_*), the two
// parameter lists of storeSky and the #if that selects between them, the amount and order of every
// arithmetic operation, the loop in the Q2 entry point, the early-outs, and the comment text of the
// golden.
//
// Measured with the probe pair (spirv-dis of Build/RaygenPrimary.probe.rgen.{glsl,hlsl}.spv, both
// halves additionally through spirv-opt -O, see the slot table of the probe for the numbers):
// the four casts and products, the ONB basis reads and the basis product, the lookAt constructor,
// the portal rotation and the local offset recomposition of both portal blocks, and the
// projection * view * vec4 chain of :1045 fold to the same numbers on the two sides, and every
// negative control of the probe folds to a different number on the same inputs.

#ifndef RAYGEN_PRIMARY_HLSLI_
#define RAYGEN_PRIMARY_HLSLI_


// This file was originally a raygen shader. But G-buffer decals are drawn
// on primary surfaces, but not in perfect reflections/refractions. Because
// 

// Must be defined:
// - either RAYGEN_PRIMARY_SHADER or RAYGEN_REFL_REFR_SHADER or Q2_REFL_REFR_SHADER
// - MATERIAL_MAX_ALBEDO_LAYERS

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

// The entry point attribute, see the header comment. A consumer of this header leaves it at the
// default and gets the three-way guarded main() as a raygen entry; the probe drops the attribute
// because it defines an entry point of its own.
#ifndef RAYGEN_PRIMARY_ENTRY_ATTR
    #define RAYGEN_PRIMARY_ENTRY_ATTR [shader("raygeneration")]
#endif

#include "ShaderCommonHLSLFunc.hlsli"
#include "RaygenCommon.hlsli"
#include "Q2Fog.hlsli"
#include "Q2Asvgf.hlsli"

float2 getMotionVectorForUpscaler(const float2 motionCurToPrev)
{
    return motionCurToPrev;
}

float2 getMotionForInfinitePoint(const int2 pix)
{
    // treat as a point with .w=0, i.e. at infinite distance
    float3 rayDir = getRayDir(getPixelUVWithJitter(pix));

    // MATRIX SITE :69, :70, :72, :73 -- the four truncating casts and the four products. The
    // operand order is the golden's, mul(M, v); the mirrored mul(v, M) multiplies by the transpose.
    float3 viewSpacePosCur   = mul((float3x3)globalUniform.view,     rayDir);
    float3 viewSpacePosPrev  = mul((float3x3)globalUniform.viewPrev, rayDir);

    float3 clipSpacePosCur   = mul((float3x3)globalUniform.projection,     viewSpacePosCur);
    float3 clipSpacePosPrev  = mul((float3x3)globalUniform.projectionPrev, viewSpacePosPrev);

    // don't divide by .w
    float3 ndcCur            = clipSpacePosCur.xyz;
    float3 ndcPrev           = clipSpacePosPrev.xyz;

    float2 screenSpaceCur    = ndcCur.xy  * 0.5 + 0.5;
    float2 screenSpacePrev   = ndcPrev.xy * 0.5 + 0.5;

    return screenSpacePrev - screenSpaceCur;
}

// Q2RTX-style path tracer G-buffer, written on the new Q2 core path only.
// These feed the Q2 ASVGF gradient pipeline (phase 4.4.2) and the Q2RTX
// reflections (phase 4.4.3). Q2ViewDepth is the ray distance and is NEGATIVE
// for reflection/refraction surfaces (Q2RTX reflect_refract convention) so
// the ASVGF filters don't bleed across reflection boundaries.
void storeQ2GBuffer(
    const int2 pix,
    const float3 baseColor, float specularFactor,
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
    framebufQ2BaseColor[pix]                = float4(baseColor, specularFactor);
    framebufQ2Metallic[pix]                 = float4(metallic, roughness, 0.0, 0.0);
    framebufQ2BounceThroughput[pix]         = float4(1.0, 1.0, 1.0, halfConeAngle);
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
            // was already in G-buffer after rasterization pass
            albedo = framebufAlbedo[getRegularPixFromCheckerboardPix(pix)].rgb;
        }
            
        framebufAlbedo[getRegularPixFromCheckerboardPix(pix)] = float4(albedo, 0.0);

        // Q2RTX-style G-buffer (sky = empty surface, env color in transparent)
        storeQ2GBuffer(pix, albedo, 0.0, 0.0, 1.0, MAX_RAY_LENGTH * 2.0, 0.0, MAX_RAY_LENGTH * 2.0, albedo, 1.0, fogAccum, ~0u);
    }

    float2 m = getMotionForInfinitePoint(pix);

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
    // if camera is not in vacuum, assume that new media is vacuum
    if (i == 0 && globalUniform.cameraMediaType != MEDIA_TYPE_VACUUM)
    {
       return MEDIA_TYPE_VACUUM;
    }

    return getMediaTypeFromFlags(geometryInstanceFlags);
}

float3 getWaterNormal(const RayCone rayCone, const float3 rayDir, const float3 normalGeom, const float3 position, bool wasPortal)
{
    // MATRIX SITE :182, :183, :190, :191, :229 -- the third consumer of getONB after Light.h. The
    // basis is the same logical matrix on both sides, so its columns are read with getColumn and
    // the product keeps the golden's operand order.
    const float3x3 basis = getONB(normalGeom);
    const float2 baseUV = float2(dot(position, getColumn(basis, 0)), dot(position, getColumn(basis, 1))); 


    // how much vertical flow to apply
    float verticality = 1.0 - abs(dot(normalGeom, globalUniform.worldUpVector.xyz));

    // project basis[0] and basis[1] on up vector
    float2 flowSpeedVertical = 10 * float2(dot(getColumn(basis, 0), globalUniform.worldUpVector.xyz), 
                                           dot(getColumn(basis, 1), globalUniform.worldUpVector.xyz));

    float2 flowSpeedHorizontal = (float2)1.0;


    const float uvScale = 0.05 / globalUniform.waterTextureAreaScale;
    float2 speed0 = uvScale * lerp(flowSpeedHorizontal, flowSpeedVertical, verticality) * globalUniform.waterWaveSpeed;
    float2 speed1 = -0.9 * speed0 * lerp(1.0, -0.1, verticality);


    // for texture sampling
    float derivU = globalUniform.waterTextureDerivativesMultiplier * 0.5 * uvScale * getWaterDerivU(rayCone, rayDir, normalGeom);

    // make water sharper if visible through the portal
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

    // MATRIX SITE :237 -- the three arguments are COLUMNS in GLSL and a constructor fills rows in
    // HLSL, so the golden's mat3(right, up, forward) is spelled transpose(float3x3(...)).
    return transpose(float3x3(right, up, forward));
}

// GLSL mod(x, y) is floor based; HLSL fmod() truncates towards zero instead, so the one call site
// of the golden (getPortalNormal) goes through this helper.
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

    // MATRIX SITE :252, :253, :254, :266 -- lookAt feeds the two column reads and the product.
    const float3x3 inLookAt_Plain = lookAt(-baseNormal, globalUniform.worldUpVector.xyz);
    const float2 localOffset_Plain = float2(dot(inWorldOffset, getColumn(inLookAt_Plain, 0)), 
                                            dot(inWorldOffset, getColumn(inLookAt_Plain, 1)));

    float distance = length(localOffset_Plain);
    float angle = atan2(localOffset_Plain.y, localOffset_Plain.x);

    float phase = sin(phaseScale * sqrt(distance) + angle + tm) + 1.0;
    phase *= waveScale;
    // less weight around center
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


    const uint currentRayMedia = globalUniform.cameraMediaType;


    // was no hit
    if (!doesPayloadContainHitInfo(primaryPayload))
    {
        float3 throughput = (float3)1.0;
        // throughput *= getMediaTransmittance(currentRayMedia, pow(abs(dot(cameraRayDir, globalUniform.worldUpVector.xyz)), -3));

        // Q2RTX-style fog over the primary segment, extended to the end of the volumes
        float4 q2FogAccum = (float4)0;
        if (globalUniform.coreQ2RTX != 0)
        {
            uint4 q2SkyFog1, q2SkyFog2;
            q2FindFogVolumes(cameraOrigin, cameraRayDir, 0.0, 1e6, q2SkyFog1, q2SkyFog2);
            q2FogAccum = q2SegmentFog(q2SkyFog1, q2SkyFog2, 1e6);
        }

        // if sky is a rasterized geometry, it was already rendered to albedo framebuf 
        storeSky(pix, cameraRayDir, globalUniform.skyType != SKY_TYPE_RASTERIZED_GEOMETRY, throughput, MAX_RAY_LENGTH * 2.0, q2FogAccum);
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
    // depth gradients is not 2d, to remove vertical/horizontal artifacts
    float depthGrad = length(gradDepth.xy);
    if (globalUniform.q2DepthGradMode != 0u)
    {
        // Q2RTX stores the reciprocal of the per-pixel depth change in IMG_PT_MOTION.w
        // (fwidth_depth), so that dist_z = |depth difference| * fwidth_depth is a
        // distance in pixels and does not grow with the view distance.
        depthGrad = 1.0 / max(Q2_DEPTH_GRAD_MIN_STEP, gradDepth.z);
    }
    framebufDepthGrad[pix]                  = (float4)depthGrad;
    framebufMotion[pix]                     = float4(motionCurToPrev, motionDepthLinearCurToPrev, 0.0);
    framebufSurfacePosition[pix]            = float4(h.hitPosition, asfloat(h.instCustomIndex));
    framebufVisibilityBuffer[pix]           = packVisibilityBuffer(primaryPayload);
    framebufViewDirection[pix]              = float4(cameraRayDir, 0.0);
    framebufThroughput[pix]                 = float4(throughput, 0.0);

    // save some info for refl/refr shader
    framebufPrimaryToReflRefr[pix]          = uint4(h.geometryInstanceFlags, primaryPayload.instIdAndIndex, h.portalIndex, emissionBlendCode);

    // save info for rasterization and upscalers (FSR/DLSS), but only about primary surface,
    // as reflections/refraction only may be losely represented via rasterization
    framebufDepthNdc[getRegularPixFromCheckerboardPix(pix)] = (float4)clamp(firstHitDepthNDC, 0.0, 1.0);
    framebufMotionDlss[getRegularPixFromCheckerboardPix(pix)] = float4(getMotionVectorForUpscaler(motionCurToPrev), 0.0, 0.0);

    // Q2RTX-style G-buffer. specular factor = dielectric F0 (0.04) to metal albedo.
    // Accumulate the fog over the primary segment (Q2RTX approach).
    uint4 q2Fog1, q2Fog2;
    q2FindFogVolumes(cameraOrigin, cameraRayDir, 0.0, firstHitDepthLinear, q2Fog1, q2Fog2);
    const float4 q2FogAccum = q2SegmentFog(q2Fog1, q2Fog2, firstHitDepthLinear);
    storeQ2GBuffer(pix, h.albedo, lerp(0.04, 1.0, h.metallic), h.metallic, h.roughness,
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



    // restore state from primary shader
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

    // Q2RTX-style accumulated fog from the primary pass; the reflection
    // segments are blended on top of it below (nearest fog in front).
    float4 q2FogAccum = framebufQ2FogAccum_Sampled.Load(int3(pix, 0));



    RayCone rayCone;
    rayCone.width = 0;
    rayCone.spreadAngle = globalUniform.cameraRayConeSpreadAngle;

    float fullPathLength = firstHitDepthLinear;
    // length of the last reflected/refracted segment (used by the god rays
    // reflection pass to march only along the reflected segment, Q2RTX-style)
    float q2LastSegmentLen = 0.0;
    float3 prevHitPosition = h.hitPosition;
    bool wasSplit = false;
    bool wasPortal = false;
    float3 virtualPos = h.hitPosition;
    float3 rayDir = cameraRayDir;
    uint currentRayMedia = globalUniform.cameraMediaType;
    // if there was no hitinfo from refl/refr, preserve primary hitinfo
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
            // apply small new media transmittance, and ignore the media (but not the refraction indices)
            throughput *= getMediaTransmittance(newRayMedia, 1.0);
            newRayMedia = currentRayMedia;
            
            // if reflections are disabled if viewing from inside of NO_MEDIA_CHANGE geometry
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
            // force refraction for all pixels
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
            // total internal reflection
            doRefraction = false;
            doSplit = false;
            F = 1.0;
        }
        
        if (doRefraction)
        {
            rayDir = refractionDir;
            throughput *= (1 - F);

            // change media
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

            // to local space; then to world space but at portal output
            // MATRIX SITE :563, :570, :572, :573, :575 -- one of the two identical portal blocks
            // (the other one is in the Q2 entry point below). The rotation keeps the golden's
            // operand order and needs the transpose of inLookAt; the column reads and the
            // recomposition go through getColumn.
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
        rayStatsAdd(RAY_STATS_CATEGORY_REFLECTION_REFRACTION, 1);

        
        if (!doesPayloadContainHitInfo(currentPayload))
        {
            throughput *= getMediaTransmittance(currentRayMedia, pow(abs(dot(rayDir, globalUniform.worldUpVector.xyz)), -3));

            // add the fog over this (missed, sky) segment to the accumulated fog
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
            emisBlendCode
        );

        // Accumulate the fog along this reflection/refraction segment
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

    // Q2RTX-style G-buffer. Negative depth so the ASVGF filters don't bleed
    // across reflection/refraction boundaries; the half-cone angle for the
    // accumulated-cone LOD is taken from the primary pass (Q2RTX convention).
    const float q2HalfConeAngle = framebufQ2BounceThroughput_Sampled.Load(int3(pix, 0)).w;
    storeQ2GBuffer(pix, h.albedo, lerp(0.04, 1.0, h.metallic), h.metallic, h.roughness,
                   -fullPathLength, q2HalfConeAngle, q2LastSegmentLen,
                   (float3)0.0, 0.0, q2FogAccum, h.cluster);
}
#endif


#if defined(Q2_REFL_REFR_SHADER)
// Q2RTX-style reflection/refraction pass (separate raygen, used on the Q2 core
// path). Ported from Q2RTX reflect_refract.rgen and adapted to the vkpt
// framework:
//   - material kinds come from the vkpt geometry instance flags for now (they
//     will switch to the Q2RTX .mat kinds together with the material system
//     port); screens/security cameras do not exist in Quake 1 and are skipped,
//   - the ray is traced with the vkpt payload/trace helper and the hit surface
//     is evaluated with the vkpt material code,
//   - the result is written to BOTH the vkpt G-buffer and the Q2RTX-style
//     G-buffer (negative view depth for reflections/refractions).
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

    // restore state from primary shader
    const uint3 primaryToReflRefrBuf = framebufPrimaryToReflRefr_Sampled.Load(int3(pix, 0)).rgb;

    // The loop below can only write anything if the primary surface is one of the
    // five kinds it handles, and at i == 0 that is decided by exactly the two
    // values above plus the roughness channel - the same ones the loop reads, so
    // the test below is the loop's first break condition. Pixels that fail it
    // leave before loading the whole G-buffer for nothing.
    if (globalUniform.reflRefrEarlyOut != 0u)
    {
        const uint primaryFlags = primaryToReflRefrBuf.r;
        bool primaryNeedsReflRefr =
            (primaryFlags & (GEOM_INST_FLAG_MEDIA_TYPE_WATER | GEOM_INST_FLAG_MEDIA_TYPE_ACID |
                             GEOM_INST_FLAG_MEDIA_TYPE_GLASS)) != 0 ||
            (isPortalFromFlags(primaryFlags) && primaryToReflRefrBuf.b != PORTAL_INDEX_NONE);
        if (!primaryNeedsReflRefr && (primaryFlags & GEOM_INST_FLAG_REFLECT) != 0)
        {
            primaryNeedsReflRefr = framebufMetallicRoughness_Sampled.Load(int3(pix, 0)).g < globalUniform.minRoughness;
        }
        if (!primaryNeedsReflRefr)
        {
            return;
        }
    }

    ShHitInfo h;
    h.albedo                            = framebufAlbedo_Sampled.Load(int3(getRegularPixFromCheckerboardPix(pix), 0)).rgb;
    h.hitPosition                       = framebufSurfacePosition_Sampled.Load(int3(pix, 0)).xyz;
    h.geometryInstanceFlags             = primaryToReflRefrBuf.r;
    h.portalIndex                       = primaryToReflRefrBuf.b;
    h.normalGeom                        = texelFetchNormalGeometry(pix);
    h.normal                            = texelFetchNormal(pix);
    h.metallic                          = framebufMetallicRoughness_Sampled.Load(int3(pix, 0)).r;
    h.roughness                         = framebufMetallicRoughness_Sampled.Load(int3(pix, 0)).g;
    const float3  motionBuf             = framebufMotion_Sampled.Load(int3(pix, 0)).rgb;
    float2        motionCurToPrev         = motionBuf.rg;
    float         motionDepthLinearCurToPrev = motionBuf.b;
    const float   firstHitDepthLinear     = framebufDepthWorld_Sampled.Load(int3(pix, 0)).r;
    float3        screenEmission          = framebufScreenEmisRT_Sampled.Load(int3(getRegularPixFromCheckerboardPix(pix), 0)).rgb;
    float3        acidFog                 = framebufAcidFogRT_Sampled.Load(int3(getRegularPixFromCheckerboardPix(pix), 0)).rgb;
    float3        throughput              = framebufThroughput_Sampled.Load(int3(pix, 0)).rgb;
    ShPayload currentPayload;
    currentPayload.instIdAndIndex       = primaryToReflRefrBuf.g;

    // Q2RTX-style G-buffer from the primary pass
    const float4 q2BaseColor              = framebufQ2BaseColor_Sampled.Load(int3(pix, 0));
    const float q2HalfConeAngle         = framebufQ2BounceThroughput_Sampled.Load(int3(pix, 0)).w;
    float4 q2Transparent                  = framebufQ2Transparent_Sampled.Load(int3(pix, 0));
    float4 q2FogAccum                     = framebufQ2FogAccum_Sampled.Load(int3(pix, 0));

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
        bool doSplit = !wasSplit;
        bool doRefraction = false;
        int correctMotionVector = 0; // 1 = reflection, 2 = refraction

        if (isPortal)
        {
            const ShPortalInstance portal = portalInstances.g_portals[h.portalIndex];

            const float3 inCenter = portal.inPosition.xyz;
            const float3 inWorldOffset = h.hitPosition - inCenter;

            float3x3 inLookAt = lookAt(getPortalNormal(normal, inWorldOffset), globalUniform.worldUpVector.xyz);

            const float3 outCenter = portal.outPosition.xyz;
            const float3x3 outLookAt = lookAt(portal.outDirection.xyz,
                                              portal.outUp.xyz);

            // to local space; then to world space but at portal output
            // MATRIX SITE :812, :819, :821, :822, :824 -- the second portal block, textually the
            // same statements as the block of the RAYGEN_REFL_REFR_SHADER entry point above.
            rayDir = mul(outLookAt, mul(transpose(inLookAt), rayDir));

            const float2 localOffset = float2(dot(inWorldOffset, getColumn(inLookAt, 0)),
                                              dot(inWorldOffset, getColumn(inLookAt, 1)));

            rayOrigin = outCenter + localOffset.x * getColumn(outLookAt, 0) + localOffset.y * getColumn(outLookAt, 1);

            wasPortal = true;
            doSplit = false;
        }
        else if (primaryIsWater || primaryIsSlime)
        {
            // Q2RTX water/slime: IOR 1.34, Fresnel, checkerboard split.
            const float ior = getIndexOfRefraction(primaryIsWater ? MEDIA_TYPE_WATER : MEDIA_TYPE_ACID);
            const float3 reflected = reflect(rayDir, normal);
            const float nDotV = abs(dot(rayDir, normal));

            if (currentRayMedia == MEDIA_TYPE_WATER || currentRayMedia == MEDIA_TYPE_ACID)
            {
                // Looking up from under water/slime: adjusted N.V, TIR.
                const float3 refracted = refract(rayDir, normal, ior);
                float ndv = 1.0 - (1.0 - nDotV) * 3.0;
                if (ndv <= 0.0 || dot(refracted, refracted) == 0.0)
                {
                    // Total internal reflection - single ray
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
                // Looking down on the water/slime surface
                const float3 refracted = refract(rayDir, normal, 1.0 / ior);
                const float F = 0.1 + 0.9 * pow(1.0 - nDotV, 5.0);
                doSplit = (i == 0);
                doRefraction = isPixOdd;
                if (doRefraction)
                {
                    rayDir = refracted;
                    throughput *= (1.0 - F);
                    currentRayMedia = newRayMedia; // enter water/slime
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
            // Q2RTX glass: IOR 1.52, thin glass (dual refraction), split.
            const float ior = getIndexOfRefraction(MEDIA_TYPE_GLASS);
            float3 glassGeomN = h.normalGeom;
            float3 glassN = normal;

            float gnDotV = dot(rayDir, glassGeomN);
            if (gnDotV > 0)
            {
                glassGeomN = -glassGeomN;
                glassN = -glassN;
                gnDotV = -gnDotV;
            }

            const float nDotV = dot(rayDir, -glassN);
            const float3 reflected = reflect(rayDir, glassN);
            float F = 0.05 + 0.95 * pow(1.0 - abs(nDotV), 5.0);
            if (dot(reflected, glassGeomN) < 0.01)
            {
                F = 0.0;
            }

            doSplit = (i == 0) && (F > 0.0);
            doRefraction = isPixOdd;
            if (doRefraction)
            {
                // infinitely thin glass: dual refraction (in via normal, out via flat geo normal)
                const float3 refr1 = refract(rayDir, glassN, 1.0 / ior);
                const float3 refr2 = refract(refr1, glassGeomN, ior);
                if (length(refr2) > 0.0)
                {
                    rayDir = refr2;
                }
                throughput *= (1.0 - F);
                throughput *= q2BaseColor.rgb;
                currentRayMedia = MEDIA_TYPE_VACUUM; // Q2RTX thin glass: no media change
                correctMotionVector = 2;
            }
            else
            {
                rayDir = reflected;
                throughput *= F;
                correctMotionVector = 1;
            }

            if (abs(dot(glassN, glassGeomN)) < 0.99999)
            {
                correctMotionVector = 0;
            }

            if (doSplit)
            {
                throughput *= 2.0;
            }
        }
        else
        {
            // chrome / smooth surface: reflection
            throughput *= h.albedo;
            rayDir = reflect(rayDir, normal);
            correctMotionVector = 1;
        }

        if (doSplit)
        {
            wasSplit = true;
        }

        currentPayload = traceReflectionRefractionRay(rayOrigin, rayDir, instIndex, h.geometryInstanceFlags, doRefraction);
        rayStatsAdd(RAY_STATS_CATEGORY_REFLECTION_REFRACTION, 1);

        if (!doesPayloadContainHitInfo(currentPayload))
        {
            // Reflection/refraction ray hit the sky: store an empty surface,
            // blend the environment into the accumulated transparency and use
            const float3 env = getSkyFiltered(rayDir, h.roughness * (SKY_MIP_COUNT - 1.0));
            q2Transparent = q2AlphaBlendPremultiplied(float4(env * throughput, 1.0), q2Transparent);

            uint4 q2SegFog1, q2SegFog2;
            q2FindFogVolumes(rayOrigin, rayDir, 0.0, 1e6, q2SegFog1, q2SegFog2);
            q2FogAccum = q2AlphaBlendPremultiplied(q2SegmentFog(q2SegFog1, q2SegFog2, 1e6), q2FogAccum);

            if (correctMotionVector == 2)
            {
                const int2 refrPix = getRegularPixFromCheckerboardPix(pix);
                const int2 pairPix = int2(refrPix.x + (refrPix.x % 2 == 0 ? 1 : -1), refrPix.y);
                framebufDepthNdc[refrPix] = (float4)1.0;
                framebufDepthNdc[pairPix] = (float4)1.0;
            }

            storeSky(pix, rayDir, true, throughput, wasSplit, q2FogAccum);
            // override the Q2RTX-style G-buffer: empty surface, negative depth,
            // environment blended into transparent (Q2RTX reflect_refract sky).
            storeQ2GBuffer(pix, (float3)0.0, 0.0, 0.0, 1.0, -MAX_RAY_LENGTH * 2.0, q2HalfConeAngle, MAX_RAY_LENGTH * 2.0,
                           q2Transparent.rgb, q2Transparent.a, q2FogAccum, ~0u);
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
            emisBlendCode
        );

        // Accumulate the fog along this reflection/refraction segment
        uint4 q2SegFog1, q2SegFog2;
        q2FindFogVolumes(rayOrigin, rayDir, 0.0, rayLen, q2SegFog1, q2SegFog2);
        q2FogAccum = q2AlphaBlendPremultiplied(q2SegmentFog(q2SegFog1, q2SegFog2, rayLen), q2FogAccum);

        hitInfoWasOverwritten = true;
        if (correctMotionVector == 2)
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
        // MATRIX SITE :1045 -- the full projection * view * vec4 chain, left associative, with the
        // 4x4 matrices as they are stored (no truncating cast here).
        const float4 refrClipPos = mul(mul(globalUniform.projection, globalUniform.view), float4(refrHitPosition, 1.0));
        const float refrDepthNdc = clamp(refrClipPos.z / refrClipPos.w + 1e-5, 0.0, 1.0);
        const int2 refrPix = getRegularPixFromCheckerboardPix(pix);
        const int2 pairPix = int2(refrPix.x + (refrPix.x % 2 == 0 ? 1 : -1), refrPix.y);
        framebufDepthNdc[refrPix] = (float4)refrDepthNdc;
        framebufDepthNdc[pairPix] = (float4)refrDepthNdc;
    }
    framebufMotion[pix]                     = float4(motionCurToPrev, motionDepthLinearCurToPrev, 0.0);
    framebufSurfacePosition[pix]            = float4(h.hitPosition, asfloat(h.instCustomIndex));
    framebufVisibilityBuffer[pix]           = packVisibilityBuffer(currentPayload);
    framebufViewDirection[pix]              = float4(rayDir, 0.0);
    framebufThroughput[pix]                 = float4(throughput, wasSplit ? 1.0 : -1.0);

    // Q2RTX-style G-buffer. Negative depth so the ASVGF filters don't bleed
    // across reflection/refraction boundaries.
    storeQ2GBuffer(pix, h.albedo, lerp(0.04, 1.0, h.metallic), h.metallic, h.roughness,
                   -fullPathLength, q2HalfConeAngle, q2LastSegmentLen,
                   q2Transparent.rgb, q2Transparent.a, q2FogAccum, h.cluster);
}
#endif

#endif // RAYGEN_PRIMARY_HLSLI_
