// Copyright (c) 2021 Sultim Tsyrendashiev
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

#ifndef RAYGEN_COMMON_HLSLI_
#define RAYGEN_COMMON_HLSLI_

// HLSL counterpart of RaygenCommon.h: the raygen-side common block of the shader base (cull masks,
// payload helpers, the primary/reflection/indirect/shadow ray casts, the sky lookups, the shading
// helpers and the ray statistics) that RaygenPrimary.inl, RtRaygenDirect.rgen and RtQ2Indirect.rgen
// are written against.
//
// The include order is load-bearing and is the golden's order: ShaderCommonHLSLFunc.hlsli first,
// then the two .inl counterparts, then RayCone.hlsli. The accessor header has to come first here
// because RayCone.hlsli calls getColumn() and because Surface.hlsli (through BRDF.hlsli) needs the
// constants that it defines. HitInfo.hlsli is included three times, once per HITINFO_INL_* macro,
// exactly as the golden includes HitInfo.inl, and it is deliberately unguarded.
//
// The golden's `#extension GL_EXT_ray_tracing : require` has no counterpart: dxc needs no extension
// declaration and the SPIR-V module declares SPV_KHR_ray_tracing itself.
//
// Spellings that had to change:
//   * Ray payloads: GLSL declares `g_payload` and `g_payloadShadow` as rayPayloadEXT globals with
//     layout(location = ...) at PAYLOAD_INDEX_DEFAULT / PAYLOAD_INDEX_SHADOW. HLSL has neither ray
//     payload globals nor payload locations: the payload has to be a local of the shader that calls
//     TraceRay, and dxc gives a location to each payload *type* in the order the types are first
//     used. The two payloads are therefore locals of tracePrimaryRay / traceReflectionRefractionRay
//     / traceIndirectRay and of traceShadowRay, declared in the golden's order (primary before
//     shadow), and resetPayload() takes the payload that it clears as an inout argument (it is
//     called from nowhere else). The probe reads the assigned locations out of the SPIR-V on the
//     HLSL half, where dxc emits OpDecorate ... Location 0 and ... Location 1 on the two payload
//     variables and so shows that they are 0 and 1, i.e. PAYLOAD_INDEX_DEFAULT and
//     PAYLOAD_INDEX_SHADOW. glslang encodes the payload index in the source qualifier only
//     (layout(location = ...)) and emits no such decoration, so there is nothing to read on the
//     golden half.
//   * traceRayEXT(topLevelAS, flags, mask, sbtOffset, sbtStride, missIndex, origin, tmin, direction,
//     tmax, payloadIndex) becomes TraceRay(topLevelAS, flags, mask, sbtOffset, sbtStride, missIndex,
//     rayDesc, payload): the four ray scalars are packed into a RayDesc and the payload index
//     disappears, because the payload is an argument.
//   * texture(samplerCube(t, s), d) -> t.SampleLevel(s, d, 0.0), textureLod(..., lod) ->
//     t.SampleLevel(s, d, lod) and nonuniformEXT(i) -> NonUniformResourceIndex(i), in front of the
//     texture index and of the sampler index, as the golden has it. The explicit lod of the first
//     form is forced: dxc rejects an implicit-lod sample outside fragment and compute shaders
//     ("sampling with implicit lod is only allowed in fragment and compute shaders"), so `.Sample`
//     cannot be written in a raygen at all; a raygen has no derivatives, so the implicit lod the
//     golden samples with is the base level, which is the 0.0 passed here.
//   * the ray flags keep the golden's sets, function for function, and change only their spelling:
//     gl_RayFlagsSkipClosestHitShaderEXT -> RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
//     gl_RayFlagsTerminateOnFirstHitEXT -> RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH (dxc has no
//     RAY_FLAG_TERMINATE_ON_FIRST_HIT; both names are the same flag bit, 4, of Vulkan's
//     VK_RAY_TRACING_TERMINATE_ON_FIRST_HIT_BIT_KHR and DXR's accept-first-hit flag) and
//     gl_RayFlagsCullFrontFacingTrianglesEXT -> RAY_FLAG_CULL_FRONT_FACING_TRIANGLES (bit 32). The
//     probe folds both spellings of the shadow ray's flag set and shows the same numbers.
//   * `readonly buffer B { T x[]; }` -> `StructuredBuffer<T> x;`; the single-instance storage block
//     `buffer RtRayStats { uint counts[N]; } rtStats;` -> `struct RtRayStats` plus
//     `RWStructuredBuffer<RtRayStats> rtStats;`, so `rtStats.counts[c]` is spelled
//     `rtStats[0].counts[c]` (a single-instance storage block has no HLSL spelling; the base spells
//     the light-list buffers the same way).
//   * atomicAdd(rtStats.counts[c], n) -> InterlockedAdd(rtStats[0].counts[c], n): the same atomic
//     add, with the return value that the golden's statement throws away.
//   * vecN(scalar) -> (floatN)scalar and mix -> lerp.
//   * matrices: the product keeps the golden's operand order, GLSL `M * v` becomes `mul(M, v)`,
//     because `mul(v, M)` multiplies by the transpose instead; and the truncating constructor
//     `mat3(m4)` becomes the cast `(float3x3)m4`, because dxc rejects `float3x3(m4)` with "too many
//     elements in vector initialization" and reads the cast as the upper left submatrix, exactly as
//     GLSL reads `mat3(mat4)` (the same spelling as the accepted VertexPreprocessPartial.hlsli). Both
//     re-spellings are part of the two sites that rotate the sky direction, each marked MATRIX SITE,
//     and the probe folds the golden's spelling and the HLSL one on literal operands and shows the
//     same three numbers, while `mul(v, M)` and an untransposed argument list fold to others.
//   * the Q2 light-list bit test in traceSunVisibility() is inlined here exactly as in the golden,
//     because this header is compiled before Q2LightLists.h; it reads the q2ClusterSkyVis buffer of
//     the accessor header. Do not include Q2LightLists.hlsli from here.

#include "ShaderCommonHLSLFunc.hlsli"



#if !defined(DESC_SET_TLAS) || \
    !defined(DESC_SET_GLOBAL_UNIFORM) || \
    !defined(DESC_SET_VERTEX_DATA) || \
    !defined(DESC_SET_TEXTURES) || \
    !defined(DESC_SET_RANDOM) || \
    !defined(DESC_SET_LIGHT_SOURCES)
        #error Descriptor set indices must be set!
#endif

#define LIGHT_SAMPLE_METHOD_NONE 0
#define LIGHT_SAMPLE_METHOD_DIRECT 1
#define LIGHT_SAMPLE_METHOD_INDIR 2
#define LIGHT_SAMPLE_METHOD_GRADIENTS 3
#define LIGHT_SAMPLE_METHOD_INITIAL 4
#define LIGHT_SAMPLE_METHOD_VOLUME 5
#if !defined(LIGHT_SAMPLE_METHOD)
    #error Light sampling method must be defined
#endif



#include "Surface.hlsli"
#include "Light.hlsli"
#include "Media.hlsli"
#include "RayCone.hlsli"
#include "TurbWarp.hlsli"

#define HITINFO_INL_PRIM
    #include "HitInfo.hlsli"
#undef HITINFO_INL_PRIM

#define HITINFO_INL_RFL
    #include "HitInfo.hlsli"
#undef HITINFO_INL_RFL

#define HITINFO_INL_INDIR
    #include "HitInfo.hlsli"
#undef HITINFO_INL_INDIR



[[vk::binding(BINDING_ACCELERATION_STRUCTURE_MAIN, DESC_SET_TLAS)]] RaytracingAccelerationStructure topLevelAS;

#ifdef DESC_SET_CUBEMAPS
[[vk::binding(BINDING_CUBEMAPS, DESC_SET_CUBEMAPS)]] TextureCube globalCubemaps[];
[[vk::binding(BINDING_CUBEMAPS_SAMPLER, DESC_SET_CUBEMAPS)]] SamplerState globalCubemaps_Sampler[];
#endif

#ifdef DESC_SET_RENDER_CUBEMAP
[[vk::binding(BINDING_RENDER_CUBEMAP, DESC_SET_RENDER_CUBEMAP)]] TextureCube renderCubemap;
[[vk::binding(BINDING_RENDER_CUBEMAP_SAMPLER, DESC_SET_RENDER_CUBEMAP)]] SamplerState renderCubemap_Sampler;
[[vk::binding(BINDING_RENDER_CUBEMAP_ENV, DESC_SET_RENDER_CUBEMAP)]] TextureCube renderCubemapEnv;
[[vk::binding(BINDING_RENDER_CUBEMAP_ENV_SAMPLER, DESC_SET_RENDER_CUBEMAP)]] SamplerState renderCubemapEnv_Sampler;
#endif

#ifdef DESC_SET_PORTALS
// The host binds the portal instances as a uniform buffer (PortalList.cpp:
// VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER), so the block keeps the golden's uniform-block spelling: a
// StructuredBuffer would make dxc emit a StorageBuffer descriptor, a descriptor kind the host does
// not create. CheckShaderProperties.py compares the pointee type and the NonWritable flag but not
// the storage class, so the mismatch is invisible to it.
struct PortalInstances_BT
{
    ShPortalInstance g_portals[PORTAL_MAX_COUNT];
};
[[vk::binding(BINDING_PORTAL_INSTANCES, DESC_SET_PORTALS)]] ConstantBuffer<PortalInstances_BT> portalInstances;
#endif


// The golden's rayPayloadEXT globals g_payload (PAYLOAD_INDEX_DEFAULT) and g_payloadShadow
// (PAYLOAD_INDEX_SHADOW) are declared as locals of the tracing functions below, in the golden's
// order; see the payload note in the header comment.



uint getPrimaryVisibilityCullMask()
{
    return globalUniform.rayCullMaskWorld | INSTANCE_MASK_REFRACT | INSTANCE_MASK_FIRST_PERSON;
}

uint getReflectionRefractionCullMask(uint surfInstCustomIndex, uint geometryInstanceFlags, bool isRefraction)
{
    uint world = globalUniform.rayCullMaskWorld | INSTANCE_MASK_REFRACT;

    if( ( geometryInstanceFlags & GEOM_INST_FLAG_IGNORE_REFRACT_AFTER ) != 0 )
    {
        // ignore refract geometry if requested
        world = world & ( ~INSTANCE_MASK_REFRACT );

        // if it's also a first-person geometry, then ignore everything first-person
        if ((surfInstCustomIndex & INSTANCE_CUSTOM_INDEX_FLAG_FIRST_PERSON) != 0)
        {
            return world;
        }
    }

    if ((surfInstCustomIndex & INSTANCE_CUSTOM_INDEX_FLAG_FIRST_PERSON) != 0)
    {
        // ignore first-person viewer -- on first-person
        return world | INSTANCE_MASK_FIRST_PERSON;
    }
    
    return isRefraction ? 
        // no first-person viewer in refractions
        world | INSTANCE_MASK_FIRST_PERSON :
        // no first-person in reflections
        world | INSTANCE_MASK_FIRST_PERSON_VIEWER;
}

uint getShadowCullMask(uint surfInstCustomIndex)
{
    const uint world = globalUniform.rayCullMaskWorld_Shadow;
    
    if ((surfInstCustomIndex & INSTANCE_CUSTOM_INDEX_FLAG_FIRST_PERSON) != 0)
    {
        // no first-person viewer shadows -- on first-person
        return world | INSTANCE_MASK_FIRST_PERSON;
    }
    else if ((surfInstCustomIndex & INSTANCE_CUSTOM_INDEX_FLAG_FIRST_PERSON_VIEWER) != 0)
    {
        // no first-person shadows -- on first-person viewer
        return world | INSTANCE_MASK_FIRST_PERSON_VIEWER;
    }
    else
    {
        // no first-person shadows -- on world
        return world | INSTANCE_MASK_FIRST_PERSON_VIEWER;
    }
}

uint getIndirectIlluminationCullMask(uint surfInstCustomIndex)
{
    const uint world = globalUniform.rayCullMaskWorld;
    
    if ((surfInstCustomIndex & INSTANCE_CUSTOM_INDEX_FLAG_FIRST_PERSON) != 0)
    {
        // no first-person viewer indirect illumination -- on first-person
        return world | INSTANCE_MASK_FIRST_PERSON;
    }
    else if ((surfInstCustomIndex & INSTANCE_CUSTOM_INDEX_FLAG_FIRST_PERSON_VIEWER) != 0)
    {
        // no first-person indirect illumination -- on first-person viewer
        return world | INSTANCE_MASK_FIRST_PERSON_VIEWER;
    }
    else
    {
        // no first-person indirect illumination -- on first-person viewer
        return world | INSTANCE_MASK_FIRST_PERSON_VIEWER;
    }
}



uint getAdditionalRayFlags()
{
    return globalUniform.rayCullBackFaces != 0 ? RAY_FLAG_CULL_FRONT_FACING_TRIANGLES : 0;
}



bool doesPayloadContainHitInfo(const ShPayload p)
{
    if (p.instIdAndIndex == UINT32_MAX || p.geomAndPrimIndex == UINT32_MAX)
    {
        return false;
    }

    int instanceId, instanceCustomIndex;
    unpackInstanceIdAndCustomIndex(p.instIdAndIndex, instanceId, instanceCustomIndex);

    if ((instanceCustomIndex & INSTANCE_CUSTOM_INDEX_FLAG_SKY) != 0)
    {
        return false;
    }

    return true;
}

void resetPayload(inout ShPayload g_payload)
{
    g_payload.baryCoords = (float2)0.0;
    g_payload.instIdAndIndex = UINT32_MAX;
    g_payload.geomAndPrimIndex = UINT32_MAX;
}

ShPayload tracePrimaryRay(float3 origin, float3 direction)
{
    ShPayload g_payload;
    resetPayload(g_payload);

    uint cullMask = getPrimaryVisibilityCullMask();

    // traceRayEXT()'s (origin, tmin, direction, tmax, payloadIndex) become a RayDesc and the payload
    // argument of TraceRay(), see the header comment.
    RayDesc rayDesc;
    rayDesc.Origin = origin;
    rayDesc.TMin = globalUniform.primaryRayMinDist;
    rayDesc.Direction = direction;
    rayDesc.TMax = globalUniform.rayLength;

    TraceRay(
        topLevelAS,
        getAdditionalRayFlags(), 
        cullMask, 
        0, 0,     // sbtRecordOffset, sbtRecordStride
        SBT_INDEX_MISS_DEFAULT, 
        rayDesc,
        g_payload);

    return g_payload; 
}

ShPayload traceReflectionRefractionRay(float3 origin, float3 direction, uint surfInstCustomIndex, uint geometryInstanceFlags, bool isRefraction)
{
    ShPayload g_payload;
    resetPayload(g_payload);

    uint cullMask = getReflectionRefractionCullMask(surfInstCustomIndex, geometryInstanceFlags, isRefraction);

    RayDesc rayDesc;
    rayDesc.Origin = origin;
    rayDesc.TMin = 0.001;
    rayDesc.Direction = direction;
    rayDesc.TMax = globalUniform.rayLength;

    TraceRay(
        topLevelAS,
        getAdditionalRayFlags(), 
        cullMask, 
        0, 0,     // sbtRecordOffset, sbtRecordStride
        SBT_INDEX_MISS_DEFAULT, 
        rayDesc,
        g_payload);

    return g_payload; 
}

ShPayload traceIndirectRay(uint surfInstCustomIndex, float3 surfPosition, float3 bounceDirection)
{
    ShPayload g_payload;
    resetPayload(g_payload);

    uint cullMask = getIndirectIlluminationCullMask(surfInstCustomIndex);

    RayDesc rayDesc;
    rayDesc.Origin = surfPosition;
    rayDesc.TMin = 0.001;
    rayDesc.Direction = bounceDirection;
    rayDesc.TMax = globalUniform.rayLength;

    TraceRay(
        topLevelAS,
        getAdditionalRayFlags(), 
        cullMask, 
        0, 0,     // sbtRecordOffset, sbtRecordStride
        SBT_INDEX_MISS_DEFAULT, 
        rayDesc,
        g_payload); 

    return g_payload; 
}



#ifdef DESC_SET_CUBEMAPS
// Get sky color for primary visibility, i.e. without skyColorMultiplier
float3 getSkyPrimary(float3 direction)
{
    uint skyType = globalUniform.skyType;

#ifdef DESC_SET_RENDER_CUBEMAP
    if (skyType == SKY_TYPE_RASTERIZED_GEOMETRY || skyType == SKY_TYPE_PROCEDURAL)
    {
        return renderCubemap.SampleLevel(renderCubemap_Sampler, direction, 0.0).rgb;
    }
#endif

    if (skyType == SKY_TYPE_CUBEMAP)
    {
        // MATRIX SITE :284 -- the golden rotates with mat3(skyCubemapRotationTransform) * direction;
        // mul(M, v) keeps that operand order (mul(v, M) would use the transpose) and the cast is the
        // golden's truncating mat3(mat4), see the matrix note in the header comment.
        direction = mul((float3x3)globalUniform.skyCubemapRotationTransform, direction);
        
        return globalCubemaps[NonUniformResourceIndex(globalUniform.skyCubemapIndex)].SampleLevel(globalCubemaps_Sampler[NonUniformResourceIndex(globalUniform.skyCubemapIndex)], direction, 0.0).rgb;
    }

    return globalUniform.skyColorDefault.xyz;
}

float3 getSky(float3 direction)
{
    float3 col = getSkyPrimary(direction);
    return col * globalUniform.skyColorMultiplier;
}

#define SKY_MIP_COUNT 11.0
#define SKY_DIFFUSE_BOUNCE_LOD 6.0

float3 getSkyFiltered(float3 direction, float lod)
{
    uint skyType = globalUniform.skyType;

#ifdef DESC_SET_RENDER_CUBEMAP
    if (skyType == SKY_TYPE_RASTERIZED_GEOMETRY)
    {
        return renderCubemap.SampleLevel(renderCubemap_Sampler, direction, lod).rgb;
    }

    if (skyType == SKY_TYPE_PROCEDURAL)
    {
        return renderCubemapEnv.SampleLevel(renderCubemapEnv_Sampler, direction, lod).rgb;
    }
#endif

    if (skyType == SKY_TYPE_CUBEMAP)
    {
        // MATRIX SITE :319 -- same re-spelling as in getSkyPrimary()
        direction = mul((float3x3)globalUniform.skyCubemapRotationTransform, direction);
        return globalCubemaps[NonUniformResourceIndex(globalUniform.skyCubemapIndex)].SampleLevel(globalCubemaps_Sampler[NonUniformResourceIndex(globalUniform.skyCubemapIndex)], direction, lod).rgb;
    }

    return globalUniform.skyColorDefault.xyz;
}

float3 getSkyFilteredMultiplied(float3 direction, float lod)
{
    return getSkyFiltered(direction, lod) * globalUniform.skyColorMultiplier;
}

float3 getSkyAmbientMultiplied(float3 direction, float lod)
{
    return getSkyFilteredMultiplied(direction, min(lod, max(globalUniform.skyAmbientLod, 0.0)));
}

float evalSkyNeePdf(const float3 n, const float3 direction)
{
    return clamp(dot(n, direction), 0.0, 1.0) * (1.0 / M_PI);
}
#endif



#if LIGHT_SAMPLE_METHOD != LIGHT_SAMPLE_METHOD_NONE

#define SHADOW_RAY_EPS       0.01
#define RAY_ORIGIN_LEAK_BIAS 0.01    // offset a bit towards a viewer to prevent light leaks from the other side of polygons

bool traceShadowRay(uint surfInstCustomIndex, float3 start, float3 end, bool ignoreFirstPersonViewer /* = false */)
{
    // prepare shadow payload
    ShPayloadShadow g_payloadShadow;
    g_payloadShadow.isShadowed = 1;  

    uint cullMask = getShadowCullMask(surfInstCustomIndex);

    if (ignoreFirstPersonViewer)
    {
        cullMask &= ~INSTANCE_MASK_FIRST_PERSON_VIEWER;
    }

    float3 l = end - start;
    float maxDistance = length(l);
    l /= maxDistance;

    RayDesc rayDesc;
    rayDesc.Origin = start;
    rayDesc.TMin = 0.001;
    rayDesc.Direction = l;
    rayDesc.TMax = maxDistance - SHADOW_RAY_EPS;

    TraceRay(
        topLevelAS, 
        RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | getAdditionalRayFlags(), 
        cullMask, 
        0, 0, 	// sbtRecordOffset, sbtRecordStride
        SBT_INDEX_MISS_SHADOW, 		// shadow missIndex
        rayDesc,
        g_payloadShadow);

    return g_payloadShadow.isShadowed == 1;
}

float traceVisibility(const Surface surf, const float3 lightPosition, uint lightIndex)
{
    const float3 start = surf.position + surf.toViewerDir * RAY_ORIGIN_LEAK_BIAS;
    const float3 end = lightPosition;

    const bool ignoreFirstPersonViewer = (globalUniform.lightIndexIgnoreFPVShadows == lightIndex);

    const bool isShadowed = traceShadowRay(surf.instCustomIndex, start, end, ignoreFirstPersonViewer);
    return float(!isShadowed);
}

float traceLightVisibility(const Surface surf, const LightSample light, uint lightIndex, out bool traced)
{
    const float3 l = safeNormalize(light.position - surf.position);
    traced = dot(surf.normal, l) > 0.0 && dot(surf.normalGeom, l) > 0.0;

    if (!traced)
    {
        return 0.0;
    }

    return traceVisibility(surf, light.position, lightIndex);
}

float traceSunVisibility(const Surface surf, const LightSample sunLight, out bool traced)
{
    const float3 l = safeNormalize(sunLight.position - surf.position);
    traced = dot(surf.normal, l) > 0.0 && dot(surf.normalGeom, l) > 0.0;

    if (!traced)
    {
        return 0.0;
    }

    /* The host proved for each cluster whether any sun ray can still leave it through the
       sky (Q2RTX's sky_visibility gate); where it cannot, the shadow ray always misses
       the sky, so the cluster keeps its darkness without paying for the ray. Clusters the
       host never classified keep tracing. This is the bit test of q2ClusterSeesSky from
       Q2LightLists.h, inlined because this header is compiled before that one. */
    const uint sunCluster = surf.cluster;
    if (sunCluster < uint(Q2_MAX_CLUSTERS) &&
        (q2ClusterSkyVis[sunCluster >> 5] & (1u << (sunCluster & 31u))) == 0u)
    {
        traced = false;
        return 0.0;
    }

    return traceVisibility(surf, sunLight.position, LIGHT_ARRAY_DIRECTIONAL_LIGHT_OFFSET);
}

float traceSkyVisibility(const Surface surf, const float3 skyDirection)
{
    const float3 start = surf.position + surf.normalGeom * 0.01;
    const float3 end   = start + skyDirection * globalUniform.rayLength;

    const bool isShadowed = traceShadowRay(surf.instCustomIndex, start, end, false);
    return float(!isShadowed);
}
#endif // LIGHT_SAMPLE_METHOD != LIGHT_SAMPLE_METHOD_NONE



#define LIGHT_CHROMA_BOOST 1.3
float3 boostChroma(const float3 c)
{
    const float lum = getLuminance(c);
    return max(lerp((float3)lum, c, LIGHT_CHROMA_BOOST), (float3)0.0);
}

#define EMISSION_CHROMA_BOOST 8.0
float3 boostEmissionChroma(const float3 c)
{
    const float lum = getLuminance(c);
    return max(lerp((float3)lum, c, EMISSION_CHROMA_BOOST), (float3)0.0);
}

void shade(const Surface surf, const LightSample light, float oneOverPdf, out float3 diffuse, out float3 specular)
{
    float3 l = safeNormalize(light.position - surf.position);
    float nl = dot(surf.normal, l);
    float ngl = dot(surf.normalGeom, l);

    if (nl <= 0 || ngl <= 0)
    {
        diffuse = specular = (float3)0;
        return;
    }

    const float3 color = boostChroma(light.color);
    diffuse  = light.dw * nl * color * evalBRDFLambertian(1.0);
    specular = light.dw * nl * color * evalBRDFSmithGGX(surf.normal, surf.toViewerDir, l, surf.roughness, surf.specularColor);

    diffuse  *= oneOverPdf;
    specular *= oneOverPdf;
}

// Diffuse-only variant of shade() for bounce paths that discard the specular output.
// Produces exactly the same `diffuse` value as shade(), minus the GGX evaluation.
void shadeDiffuse(const Surface surf, const LightSample light, float oneOverPdf, out float3 diffuse)
{
    float3 l = safeNormalize(light.position - surf.position);
    float nl = dot(surf.normal, l);
    float ngl = dot(surf.normalGeom, l);

    if (nl <= 0 || ngl <= 0)
    {
        diffuse = (float3)0;
        return;
    }

    const float3 color = boostChroma(light.color);
    diffuse = light.dw * nl * color * evalBRDFLambertian(1.0);

    diffuse *= oneOverPdf;
}


#define RAY_STATS_CATEGORY_PRIMARY            0
#define RAY_STATS_CATEGORY_REFLECTION_REFRACTION 1
#define RAY_STATS_CATEGORY_INDIRECT           2
// NEE visibility rays are counted per casting pass, because the two passes that
// trace them are timed separately (GPU_PASS_DIRECT and GPU_PASS_INDIRECT).
#define RAY_STATS_CATEGORY_SHADOW_DIRECT      3
#define RAY_STATS_CATEGORY_SHADOW_INDIRECT    4

struct RtRayStats
{
    uint counts[RAY_STATS_CATEGORY_COUNT];
};

[[vk::binding(0, DESC_SET_RAY_STATS)]] RWStructuredBuffer<RtRayStats> rtStats;

void rayStatsAdd(const uint category, const uint count)
{
    if ((globalUniform.debugShowFlags & DEBUG_SHOW_FLAG_RAY_STATS) != 0)
    {
        InterlockedAdd(rtStats[0].counts[category], count);
    }
}

#endif // RAYGEN_COMMON_HLSLI_
