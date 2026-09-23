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

// HLSL counterpart of GLSL/RtRaygenDirect.rgen, the direct-lighting raygen: one or two light NEE
// samples chosen from the pixel's cluster plus the directional light, then the two unfiltered
// outputs and the view direction. The body is a statement by statement port into the already
// ported headers -- RaygenCommon.hlsli (fetch, shade, traceLightVisibility, traceSunVisibility,
// rayStatsAdd), Q2LightLists.hlsli (q2SampleClusterLights, q2AccumulateLightStats,
// q2GetIsGradient, q2RoughnessSquareToSpecPower), Light.hlsli (sampleLight,
// sampleDirectionalLight, decodeAsDirectionalLight), Random.hlsli (rnd16, rnd16_2),
// BRDF.hlsli (demodulateSpecular) and the generated accessor layer.
//
// This file is one of the two raygen files of the base that define their own main(); GLSL needs no
// attribute on it, HLSL needs [shader("raygeneration")], which is what makes the function the
// OpEntryPoint of the lib_6_3 module. The entry stays named main, so the blob name and the host
// are untouched.
//
// Spellings that had to change:
//   * ivec2 -> int2, vec3 -> float3, vec2 -> float2, uvec4 -> uint4, gl_LaunchIDEXT.xy ->
//     DispatchRaysIndex().xy, i.e. the golden's ivec2(gl_LaunchIDEXT.xy) becomes the truncating
//     (int2) cast of the uint3 the intrinsic hands back, as in RaygenPrimary.hlsli.
//   * imageStore(img, pix, v) -> img[pix] = v, and the uvec4(0) of the two sky stores becomes
//     (uint4)0: the scalar does not broadcast into a constructor in HLSL.
//   * texelFetch(t, pix, 0) -> t.Load(int3(pix, 0)), for the seed and the cluster of the pixel.
//   * vecN(scalar) -> (floatN)scalar (the two direct accumulators and every explicit 0.0 of the
//     golden), and the scalar conversions of the golden are C style casts of the same truncating
//     conversion: uint(x) -> (uint)x, int(x) -> (int)x, float(x) -> (float)x.
//   * mix -> lerp and the matrix rules of ShaderCommonHLSL.hlsli have no site here: the body
//     contains no product, no index and no constructor of a matrix.
//
// What did not change: the DESC_SET_* block, the LIGHT_SAMPLE_METHOD pin (DIRECT, the header's
// block is compiled for it), every #define (Q2_RNG_CELL_SELECT / Q2_RNG_LIGHT_POINT /
// Q2_RNG_SUN_DISK, Q2_DIRECT_MAX_SPHERE_SOLID_ANGLE, MATERIAL_MAX_ALBEDO_LAYERS 0), the salt
// arithmetic (saltBase + 1u, the Q2_RNG_LIGHT_POINT + 2*s stream, the single Q2_RNG_SUN_DISK
// stream), the neeLightSamples clamp to [1, 2] and the comment that justifies it, the
// accumulation order of the two loop outputs, the 1.0 / lightPdf and 1.0 / numNeeSamples
// reciprocals, the per-sample statistics call with its sample index, every early out, and the
// three stores of the tail.

// don't need albedo for direct illumination
#define MATERIAL_MAX_ALBEDO_LAYERS 0

#define DESC_SET_TLAS 0
#define DESC_SET_FRAMEBUFFERS 1
#define DESC_SET_GLOBAL_UNIFORM 2
#define DESC_SET_VERTEX_DATA 3
#define DESC_SET_TEXTURES 4
#define DESC_SET_RANDOM 5
#define DESC_SET_LIGHT_SOURCES 6
#define DESC_SET_RAY_STATS 11
#define LIGHT_SAMPLE_METHOD (LIGHT_SAMPLE_METHOD_DIRECT)
#include "RaygenCommon.hlsli"
#include "Q2Asvgf.hlsli"
#include "Q2LightLists.hlsli"

#define Q2_RNG_CELL_SELECT 200
#define Q2_RNG_LIGHT_POINT 208
#define Q2_RNG_SUN_DISK 212

#define Q2_DIRECT_MAX_SPHERE_SOLID_ANGLE (2.0 * M_PI)

[shader("raygeneration")]
void main()
{
    const int2 pix = (int2)DispatchRaysIndex().xy;

    const uint seed = framebufQ2RngSeed_Sampled.Load(int3(pix, 0)).r;

    const Surface surf = fetchGbufferSurface(pix);

    if (surf.isSky)
    {
        framebufUnfilteredDirect[pix] = (uint4)0;
        framebufUnfilteredSpecular[pix] = (uint4)0;
        return;
    }

    const bool isGradient = q2GetIsGradient(pix);
    const uint cluster = framebufQ2Cluster_Sampled.Load(int3(pix, 0)).r;

    const float alpha = square(surf.roughness);
    const float phongExp = q2RoughnessSquareToSpecPower(alpha);
    const float phongScale = min(100.0, 1.0 / (M_PI * max(alpha * alpha, 1e-4)));
    const float phongWeight = clamp(
        getLuminance(surf.specularColor) / (getLuminance(surf.specularColor) + getLuminance(surf.albedo)),
        0.0, 0.9);

    float3 directDiffuse = (float3)0.0;
    float3 directSpecular = (float3)0.0;

    // Halving the count stays unbiased because the accumulated result is divided
    // by it below, so it only trades noise for shadow rays. The value cannot
    // exceed 2: above it the light-point salt (Q2_RNG_LIGHT_POINT + 2*s) would
    // reach Q2_RNG_SUN_DISK and recycle that stream for a second purpose.
    const int numNeeSamples = clamp((int)globalUniform.neeLightSamples, 1, 2);

    for (int s = 0; s < numNeeSamples; s++)
    {
        const uint saltBase = (uint)Q2_RNG_CELL_SELECT + (uint)s * 4u;
        const float3 rng = float3(
            rnd16(seed, saltBase),
            rnd16(seed, saltBase + 1u),
            rnd16(seed, saltBase + 2u));

        uint lightIndex = LIGHT_INDEX_NONE;
        uint lightSlot = 0u;
        float lightPdf = 0.0;
        q2SampleClusterLights(cluster, surf.position, surf.normal, surf.toViewerDir,
                              phongExp, phongScale, phongWeight, isGradient, rng,
                              lightIndex, lightSlot, lightPdf);

        if (lightIndex != LIGHT_INDEX_NONE && lightPdf > 0.0)
        {
            const float2 pointRnd = rnd16_2(seed, (uint)Q2_RNG_LIGHT_POINT + (uint)s * 2u) * 0.99;
            LightSample light = sampleLight(lightSources[lightIndex], surf.position, pointRnd);

            if (lightSources[lightIndex].lightType == LIGHT_TYPE_SPHERE ||
                lightSources[lightIndex].lightType == LIGHT_TYPE_SPOT)
            {
                light.dw = min(light.dw, Q2_DIRECT_MAX_SPHERE_SOLID_ANGLE);
            }

            if (getLuminance(light.color) > 0.0)
            {
                bool lightTraced;
                const float vis = traceLightVisibility(surf, light, lightIndex, lightTraced);

                if (lightTraced)
                {
                    rayStatsAdd(RAY_STATS_CATEGORY_SHADOW_DIRECT, 1);
                }

                if (lightSources[lightIndex].lightType == LIGHT_TYPE_TRIANGLE ||
                    lightSources[lightIndex].lightType == LIGHT_TYPE_TEXTURED_AREA)
                {
                    q2AccumulateLightStats(cluster, lightSlot, surf.normal, vis, (uint)s);
                }

                float3 d, s;
                shade(surf, light, 1.0 / lightPdf, d, s);
                directDiffuse += d * vis;
                directSpecular += s * vis;
            }
        }
    }
    directDiffuse  *= (1.0 / (float)numNeeSamples);
    directSpecular *= (1.0 / (float)numNeeSamples);

    if (globalUniform.directionalLightExists != 0)
    {
        const DirectionalLight sun = decodeAsDirectionalLight(lightSources[LIGHT_ARRAY_DIRECTIONAL_LIGHT_OFFSET]);
        const float2 sunRnd = rnd16_2(seed, Q2_RNG_SUN_DISK) * 0.99;
        const LightSample sunLight = sampleDirectionalLight(sun, surf.position, sunRnd);

        bool sunTraced;
        const float sunVis = traceSunVisibility(surf, sunLight, sunTraced);

        if (sunTraced)
        {
            rayStatsAdd(RAY_STATS_CATEGORY_SHADOW_DIRECT, 1);
        }

        float3 d, s;
        shade(surf, sunLight, 1.0, d, s);
        directDiffuse += d * sunVis;
        directSpecular += s * sunVis;
    }

    imageStoreUnfilteredDirect(pix, directDiffuse);
    imageStoreUnfilteredSpecular(pix, demodulateSpecular(directSpecular, surf.specularColor));
    framebufViewDirection[pix] = float4(-surf.toViewerDir, MAX_RAY_LENGTH);
}
