// Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) 2026 QuakeRay contributors
//
// This file is a port of shader/god_rays.comp from Quake 2 RTX (https://github.com/NVIDIA/Q2RTX),
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
// Volumetric sunlight ("god rays") computed by ray marching through the
// shadow map. Ported from Quake 2 RTX (god_rays.comp), GPL v2.
// Runs at HALF render resolution and is bilaterally upscaled to full
// resolution by CmGodRaysFilter.comp (Q2RTX god_rays_filter.comp).
//
// HLSL counterpart of CmGodRays.comp.
//
// Spellings that had to change:
//   * layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) -> [numthreads(16, 16, 1)]
//     on the entry point, and gl_GlobalInvocationID -> SV_DispatchThreadID
//   * layout(push_constant, std140) uniform PushConstants { uint passIndex; } push -> the struct
//     in a ConstantBuffer with [[vk::push_constant]]; `push.passIndex` is read the same way
//   * the shadow map keeps the golden's split: texture2D at binding 0 and sampler at binding 1 of
//     DESC_SET_GOD_RAYS_SHADOW, as Texture2D<float4> and SamplerState, and
//     textureLod(sampler2D(texShadowMap, texShadowMap_Sampler), uv, 0) becomes
//     texShadowMap.SampleLevel(texShadowMap_Sampler, uv, 0)
//   * the golden's single instance storage block GodRaysParams_BT has no HLSL spelling, so it
//     becomes a struct of the same name read through one instance: `godRaysParams.shadowMapVP` is
//     `godRaysParams[0].shadowMapVP` here. The matrix keeps its shape (mat4 -> float4x4) and the
//     product below keeps the golden's operand order, `mul(m, v)` for `m * v`
//   * the matrix product of GetShadow, MATRIX SITE: the golden's
//     `godRaysParams.shadowMapVP * vec4(worldPos, 1)` becomes
//     `mul(godRaysParams[0].shadowMapVP, float4(worldPos, 1))`. The mirrored spelling
//     `mul(v, M)` would multiply by the transpose; the two fold to different numbers
//   * texelFetch(t, p, 0) -> t.Load(int3(p, 0)), imageLoad(img, p).rgb -> img[p].rgb,
//     imageStore(img, p, v) -> img[p] = v
//   * vecN(scalar) -> (floatN)scalar, vec2(ivec2) -> (float2)ivec2 and ivec2(vec2)/ivec2(x, y)
//     -> int2(...), the same truncating conversions
//   * any(greaterThanEqual(a, b)) -> any(a >= b), as in the other ports of the pass
//   * mix(a, b, s) -> lerp(a, b, s), the same definition under the HLSL name
//   * the file scope `const float PI` becomes `static const float PI`: a namespace scope const
//     without static is a uniform whose contents are only known at runtime in HLSL
//   * IntersectRayBox keeps its `out` parameters, with the HLSL keyword on both of them
//
// What did not change: the half resolution pixel and screen size, the rotated grid offset and the
// checkerboard pixel, the Q2 depth test of the reflection pass, the reflection segment start and
// the sky early out, the Henyey-Greenstein phase with its normalized form, the world box
// intersection and both box margins (0.75 -> 1.5 history is preserved), the medium dependent
// extinction factors and the water/acid branches, the blue noise jitter of the first step, the
// march loop with its break on `offset >= rayDistance`, the sun-ward shadow bias of 20 units,
// the inscatter accumulation `(shadow * phase * stepLength * density) * through` and the
// `* 0.0001` scaling of the final colour, and the accumulator rule of the reflection pass.

#define DESC_SET_GOD_RAYS_SHADOW 0
#define DESC_SET_GOD_RAYS_PARAMS 1
#define DESC_SET_FRAMEBUFFERS 2
#define DESC_SET_GLOBAL_UNIFORM 3
#define DESC_SET_RANDOM 4
#include "ShaderCommonHLSLFunc.hlsli"
#include "Random.hlsli"

// The GLSL original declares the workgroup size here; in HLSL it is an attribute of the
// entry point instead, so the shader itself carries [numthreads(16, 16, 1)].

// 0 = primary rays, 1 = reflection/refraction rays (Q2RTX god_rays pass_index)
struct GodRaysPush_BT
{
    uint passIndex;
};

[[vk::push_constant]] ConstantBuffer<GodRaysPush_BT> push;

[[vk::binding(0, DESC_SET_GOD_RAYS_SHADOW)]] Texture2D<float4> texShadowMap;
[[vk::binding(1, DESC_SET_GOD_RAYS_SHADOW)]] SamplerState texShadowMap_Sampler;

struct GodRaysParams_BT
{
    float4 sunDirection;
    float4 sunColor;
    float4 worldCenter;
    float4 worldHalfSizeInv;
    float4x4 shadowMapVP;
    float shadowMapDepthScale;
    float godRaysIntensity;
    float godRaysEccentricity;
    uint godRaysEnabled;
    float _padding;
};

[[vk::binding(0, DESC_SET_GOD_RAYS_PARAMS)]] StructuredBuffer<GodRaysParams_BT> godRaysParams;

float GetShadow(float3 worldPos)
{
    // MATRIX SITE: the golden's `m * v` keeps its operand order as `mul(m, v)`
    float4 uvz = mul(godRaysParams[0].shadowMapVP, float4(worldPos, 1));
    uvz.x = uvz.x * 0.5 + 0.5;
    uvz.y = 0.5 - 0.5 * uvz.y;

    float depthOpaque = texShadowMap.SampleLevel(texShadowMap_Sampler, uvz.xy, 0).x;

    // Q2RTX comparison (god_rays.comp GetShadow): depthOpaque > uvz.z, no bias
    // here. The anti-leak offset is applied by the caller, which shifts the
    // sample position along -sunDirection before projecting it.
    return depthOpaque > uvz.z ? 1.0 : 0.0;
}

static const float PI = 3.1415926535;

float ScatterPhase_HenyeyGreenstein(float cosa, float g)
{
    // "normalized" Henyey-Greenstein
    const float g_sqr = g * g;
    const float num = (1.0 - abs(g));
    const float denom = sqrt(max(1.0 - 2.0 * g * cosa + g_sqr, 0.0));
    const float frac = num / denom;
    const float scale = g_sqr + (1.0 - g_sqr) / (4.0 * PI);
    return scale * (frac * frac * frac);
}

float vmin(float3 v) { return min(v.x, min(v.y, v.z)); }
float vmax(float3 v) { return max(v.x, max(v.y, v.z)); }

bool IntersectRayBox(float3 origin, float3 direction, float3 mins, float3 maxs, out float tIn, out float tOut)
{
    float3 t1 = (mins - origin) / direction;
    float3 t2 = (maxs - origin) / direction;
    tIn = vmax(min(t1, t2));
    tOut = vmin(max(t1, t2));
    return tIn < tOut && tOut > 0;
}

float getDensity(float3 p)
{
    // 1.0 inside the world box, falls off to 0 over 25% of world size outside
    float3 bounds = clamp(3.0 - 2.0 * abs((p - godRaysParams[0].worldCenter.xyz) * godRaysParams[0].worldHalfSizeInv.xyz), (float3)0, (float3)1);
    return bounds.x * bounds.y * bounds.z;
}

float getStep(float t, float density)
{
    return max(1.0, lerp(20.0, 5.0, density));
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const int2 pixelPos = int2(dispatchThreadID.xy);
    // half-resolution pass (Q2RTX): each invocation computes one half-res pixel
    const int2 screenSize = int2((globalUniform.renderWidth + 1) / 2, (globalUniform.renderHeight + 1) / 2);
    if (any(pixelPos >= screenSize))
        return;

    // DIAGNOSTIC (rt_debugflags 2048): show the whole shadow map stretched to
    // the screen (grayscale = stored depth). This displays the "picture" the
    // shadow map actually contains: the level seen from the sun, with holes
    // where the sun shines through. If this picture is wrong, the shadow map
    // render is broken; if it is right, the sampling (VP in GetShadow) is.
    if ((globalUniform.debugShowFlags & DEBUG_SHOW_FLAG_GOD_RAYS) != 0)
    {
        if (godRaysParams[0].godRaysEnabled == 0)
        {
            framebufGodRays[pixelPos] = float4(0.0, 1.0, 0.0, 0.0);
            return;
        }
        if (getLuminance(godRaysParams[0].sunColor.rgb) == 0.0)
        {
            framebufGodRays[pixelPos] = float4(1.0, 1.0, 0.0, 0.0);
            return;
        }

        const float2 mapUv = ((float2)pixelPos + 0.5) / (float2)screenSize;
        const float dOpaque = texShadowMap.SampleLevel(texShadowMap_Sampler, mapUv, 0).x;
        framebufGodRays[pixelPos] = float4(dOpaque * 0.25, dOpaque * 0.25, dOpaque * 0.25, 0.0);
        return;
    }

    if (godRaysParams[0].godRaysEnabled == 0)
    {
        if (push.passIndex == 0)
        {
            framebufGodRays[pixelPos] = (float4)0;
        }
        return;
    }

    if (getLuminance(godRaysParams[0].sunColor.rgb) == 0.0)
    {
        if (push.passIndex == 0)
        {
            framebufGodRays[pixelPos] = (float4)0;
        }
        return;
    }

    // surface data for this pixel (checkerboard storage). Half-res pass
    // (Q2RTX): pick the matching full-res sub-pixel of this half-res pixel
    // with the rotated grid offset (Q2RTX GetRotatedGridOffset).
    const int2 fullResPix = pixelPos * 2 + int2(pixelPos.y & 1, 1 - (pixelPos.x & 1));
    const int2 cbPix = getCheckerboardPix(fullResPix);

    // Reflection pass (passIndex != 0): only process pixels whose surface is a
    // reflected/refracted ray (Q2ViewDepth < 0). Other pixels keep the result
    // of the primary pass.
    if (push.passIndex != 0)
    {
        const float q2Depth = framebufQ2ViewDepth_Sampled.Load(int3(cbPix, 0)).x;
        if (q2Depth >= 0.0)
        {
            return;
        }
    }

    const float3 surfacePos = framebufSurfacePosition_Sampled.Load(int3(cbPix, 0)).xyz;
    const float3 direction  = framebufViewDirection_Sampled.Load(int3(cbPix, 0)).xyz;
    const float3 throughput = framebufThroughput_Sampled.Load(int3(cbPix, 0)).xyz;

    // In the reflection pass the march follows the reflected segment only,
    // starting at the reflection point (surfacePos - direction * segmentLen);
    // the segment length is stored by the refl pass in Q2GodRaysThroughputDist.w.
    float rayDistance;
    if (push.passIndex != 0)
    {
        rayDistance = framebufQ2GodRaysThroughputDist_Sampled.Load(int3(cbPix, 0)).w;
    }
    else
    {
        rayDistance = framebufDepthWorld_Sampled.Load(int3(cbPix, 0)).x;
    }

    // Q2RTX marches sky pixels too, so the sun shafts are visible in the sky
    // around the sun (the camera ray crosses the world box). On the legacy path
    // the sky stays empty - there the legacy screen-space volumetric renders
    // the sky shafts. For sky pixels the stored surface position is invalid,
    // so the march starts from the camera.
    const bool isSky = rayDistance > MAX_RAY_LENGTH;
    if (isSky && globalUniform.coreQ2RTX == 0)
    {
        if (push.passIndex == 0)
        {
            framebufGodRays[pixelPos] = (float4)0;
        }
        return;
    }

    // start marching from where the ray enters the world box
    const float3 originalPos = isSky ? globalUniform.cameraPosition.xyz
                                     : surfacePos - direction * rayDistance;

    // Medium the camera is in (Q2RTX reads the surface material's medium; for
    // the primary segment the camera medium is the physically correct one).
    // In media the phase becomes more isotropic and the light is extinguished.
    const uint medium = globalUniform.cameraMediaType;

    // godRaysParams.sunDirection points TOWARD the sun (Q2RTX convention)
    const float cosa = dot(direction, godRaysParams[0].sunDirection.xyz);
    const float eccentricity = (medium == MEDIA_TYPE_VACUUM)
        ? godRaysParams[0].godRaysEccentricity
        : 0.5;
    const float phase = ScatterPhase_HenyeyGreenstein(cosa, eccentricity);

    float offset = 0.0;
    float tIn, tOut;
    const float3 halfSize = 1.0 / godRaysParams[0].worldHalfSizeInv.xyz;
    // Q2RTX marches a box of world_center +/- world_size*0.75 =
    // +/- halfSize*1.5, which matches the getDensity falloff edge (density
    // reaches 0 at |p-center| == 1.5*halfSize). The previous +/- 0.75*halfSize
    // box was half the size and culled the density falloff region.
    const float3 boxMins = godRaysParams[0].worldCenter.xyz - halfSize * 1.5;
    const float3 boxMaxs = godRaysParams[0].worldCenter.xyz + halfSize * 1.5;

    if (IntersectRayBox(originalPos, direction, boxMins, boxMaxs, tIn, tOut))
    {
        if (tIn > rayDistance)
        {
            // Q2RTX clears in both passes: a reflection whose segment misses
            // the world box shows no god rays at all
            framebufGodRays[pixelPos] = (float4)0;
            return;
        }

        offset = max(tIn, offset);
        rayDistance = min(tOut, rayDistance);
    }
    else
    {
        framebufGodRays[pixelPos] = (float4)0;
        return;
    }

    float3 currentPos = originalPos + direction * offset;
    float density = getDensity(currentPos);

    // extinction factors along the segment (Q2RTX get_extinction_factors)
    float3 extinctionFactors = (float3)0.0001;
    if (medium == MEDIA_TYPE_WATER)
    {
        extinctionFactors = -log(max(globalUniform.waterColorAndDensity.rgb, (float3)1e-6));
    }
    else if (medium == MEDIA_TYPE_ACID)
    {
        extinctionFactors = -log(max(globalUniform.acidColorAndDensity.rgb, (float3)1e-6));
        extinctionFactors *= max(1.0, sqrt(globalUniform.acidColorAndDensity.a));
    }
    // MEDIA_TYPE_GLASS: transparent, keep the vacuum extinction

    float3 through = throughput;
    float3 inscatter = (float3)0;

    // randomize the first step to break banding
    const uint seed = getRandomSeed(pixelPos, globalUniform.frameId);
    const float rnd = rndBlueNoise8(seed, 0).x;
    offset += getStep(offset, density) * (rnd - 1.0);

    while (true)
    {
        density = getDensity(currentPos);
        float stepLength = getStep(offset, density);

        offset += stepLength;

        if (offset >= rayDistance)
            break;

        currentPos = originalPos + direction * offset;

        // Q2RTX: shift the sample 20 world units away from the sun (deeper into
        // the shadow map) so occluders are effectively thickened and the shafts
        // don't leak through thin walls or at grazing angles. The final step
        // before the surface is not sampled (Q2RTX breaks here), which keeps
        // the surface-facing samples from being pushed behind the geometry.
        const float3 shadowBias = -godRaysParams[0].sunDirection.xyz * 20.0;
        const float shadow = GetShadow(currentPos + shadowBias);

        const float3 differentialInscatter = (shadow * phase * stepLength * density) * through;
        inscatter += differentialInscatter;
        through *= exp(-(stepLength * density) * extinctionFactors);
    }

    float3 inscatterColor = inscatter * godRaysParams[0].sunColor.rgb * godRaysParams[0].godRaysIntensity * 0.0001;

    // underwater the volumetric sunlight is much denser (Q2RTX pt_water_density)
    if (medium == MEDIA_TYPE_WATER)
    {
        inscatterColor *= globalUniform.waterColorAndDensity.a * 30.0;
    }

    if (push.passIndex != 0)
    {
        // accumulate the reflected-segment god rays on top of the primary result
        const float3 prevColor = framebufGodRays[pixelPos].rgb;
        framebufGodRays[pixelPos] = float4(inscatterColor + prevColor, 0);
    }
    else
    {
        framebufGodRays[pixelPos] = float4(inscatterColor, 0);
    }
}
