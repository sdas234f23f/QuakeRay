// Copyright (C) 2018 Christoph Schied
// Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) 2026 QuakeRay contributors
//
// This file is a port of shader/indirect_lighting.rgen from Quake 2 RTX (https://github.com/NVIDIA/Q2RTX),
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
// Indirect lighting (one or two bounces) of the Q2RTX-style path tracer.
//
// HLSL counterpart of GLSL/RtQ2Indirect.rgen, the indirect raygen: the first-bounce specular ray
// with the sky-NEE weight, the cluster light NEE of the hit, the sun bounce with its range
// attenuation, the optional second diffuse bounce, and the three temporal outputs. The body is a
// statement by statement port into the already ported headers -- RaygenCommon.hlsli (the bounce
// and shadow ray casts, fetch, shadeDiffuse, boostChroma, getSkyAmbientMultiplied, evalSkyNeePdf,
// rayStatsAdd), Q2LightLists.hlsli (q2SampleClusterLights, q2AccumulateLightStats,
// q2ClusterSeesSky, q2GetIsGradient, q2RoughnessSquareToSpecPower), Q2Asvgf.hlsli
// (SKY_MIP_COUNT arrives from RaygenCommon.hlsli, the gradient constants from here),
// Light.hlsli, Random.hlsli, HitInfo.hlsli (getHitInfoBounce), Surface.hlsli
// (hitInfoToSurface_Indirect) and BRDF.hlsli.
//
// This file is one of the two raygen files of the base that define their own main(); the
// [shader("raygeneration")] attribute below is what makes it the OpEntryPoint of the lib_6_3
// module, and the entry stays named main, so the blob name and the host are untouched.
//
// Spellings that had to change:
//   * `layout (constant_id = 0) const uint maxAlbedoLayerCount = 0` ->
//     `[[vk::constant_id(0)]] const uint maxAlbedoLayerCount = 0`: the same SpecId 0, uint and
//     default 0 (the mechanism the accepted RaygenPrimary port and EfSimple.hlsli use).
//   * ivec2 -> int2, vec3 -> float3, vec2 -> float2, vec4 -> float4, gl_LaunchIDEXT.xy ->
//     DispatchRaysIndex().xy, i.e. the golden's ivec2(gl_LaunchIDEXT.xy) becomes the truncating
//     (int2) cast of the uint3 the intrinsic hands back, as in RaygenPrimary.hlsli.
//   * imageStore(img, pix, v) -> img[pix] = v, and texelFetch(t, pix, 0) -> t.Load(int3(pix, 0)).
//   * textureSize(framebufQ2RngSeed_Sampled, 0).y -> the seeded texture's own GetDimensions: HLSL
//     has no textureSize intrinsic, the size is a method of the texture and it hands back uints,
//     so the int pix.y comparison takes a (int) cast. Every framebuffer height of this renderer
//     is far below 2^31, so that cast is exact where the golden's ivec2 component is already an
//     int. This is the only site of the file that is not a one to one re-spelling.
//   * vecN(scalar) -> (floatN)scalar (the golden's explicit zero constants; the one three component
//     constructor of the file, float3( 10.0, 0.0, 0.0 ), is copied unchanged) and the scalar
//     conversions of the golden become C style casts of the same truncating conversion:
//     uint(x) -> (uint)x, int(x) -> (int)x, float(x) -> (float)x.
//   * mix -> lerp, atan(y, x) -> atan2(y, x), and every matrix site is inside the ported headers:
//     this file has no product, no index and no constructor of a matrix in its own text.
//
// What did not change: the whole GPL header, every #define and its value (the five RNG salts, the
// two bounce mip biases, Q2_INDIR_MAX_SPHERE_SOLID_ANGLE and the five rt_debugflags bits with
// their comment), the DESC_SET_* block and the LIGHT_SAMPLE_METHOD pin (INDIR), the order of the
// eleven functions, indirDbg / indirMaxSphereSolidAngle / indirSphotGain as the debug gates,
// every early out (the giBounceRays.x < 0.25 zero store, the half-res row out, the sky store, the
// sky NEE gate of sampleLambertian and of the luminance test), the RNG streams and their exact
// salts (Q2_RNG_INDIR_CELL_SELECT + 0/1/2u, Q2_RNG_INDIR_LIGHT_POINT, Q2_RNG_INDIR2_SUN_DISK,
// Q2_RNG_INDIR_SUN_DISK, Q2_RNG_INDIR_SKY_NEE, RANDOM_SALT_SPEC_BOUNCE(1), RANDOM_SALT_DIFF_BOUNCE(2)),
// the 0.99 of every random disk, the order of every accumulation, the 50.0 clamp, the 50.0
// second-bounce debug gain that is applied after that clamp, the skyBsdfWeight of the sky branch,
// the half-res scale and its site, the specular demodulation against the accumulated direct
// specular of the pixel, and the sanity store of the red SH.

[[vk::constant_id(0)]] const uint maxAlbedoLayerCount = 0;
#define MATERIAL_MAX_ALBEDO_LAYERS maxAlbedoLayerCount
#define FORCE_EVALBRDF_GGX_LOOSE
#define DESC_SET_TLAS 0
#define DESC_SET_FRAMEBUFFERS 1
#define DESC_SET_GLOBAL_UNIFORM 2
#define DESC_SET_VERTEX_DATA 3
#define DESC_SET_TEXTURES 4
#define DESC_SET_RANDOM 5
#define DESC_SET_LIGHT_SOURCES 6
#define DESC_SET_CUBEMAPS 7
#define DESC_SET_RENDER_CUBEMAP 8
#define DESC_SET_RAY_STATS 11
#define LIGHT_SAMPLE_METHOD (LIGHT_SAMPLE_METHOD_INDIR)
#include "RaygenCommon.hlsli"
#include "Q2Asvgf.hlsli"
#include "Q2LightLists.hlsli"

#define Q2_RNG_INDIR_CELL_SELECT 220
#define Q2_RNG_INDIR_LIGHT_POINT 223
#define Q2_RNG_INDIR_SUN_DISK    225
// The first-bounce salts are used by the first-bounce light NEE. The 230/233 pair is
// reserved for a second-bounce light NEE, which Q2RTX does not have and which this pass
// therefore no longer traces (only 235, the second-bounce sun disk, is live).
#define Q2_RNG_INDIR2_CELL_SELECT 230
#define Q2_RNG_INDIR2_LIGHT_POINT 233
#define Q2_RNG_INDIR2_SUN_DISK    235
#define Q2_RNG_INDIR_SKY_NEE      240

#define Q2_INDIR_MAX_SPHERE_SOLID_ANGLE 0.02

// Q2RTX pt_sun_bounce_range / sun_bounce: the sun contribution of a bounce fades out with
// the distance the bounce ray travelled, and what is left is scaled by a straight
// multiplier. Both come from the host through ShGlobalUniform.sunBounce (cvars
// rt_sun_bounce_range, 2000 in game units, and rt_sun_bounce_scale, 1.0). Without the range
// every bounce hit anywhere in the map adds full sun energy and pays for its own sun shadow
// ray, which is where the salt-like noise in dark areas came from.

// Indirect-light debug probes, driven by rt_debugflags bits 15..19. Those bits sit
// above the RgDebugDrawFlagBits enum, so they reach the shader only through the
// pass-through in VulkanDevice.cpp. They have no display branch: the final image stays
// normal, which makes them usable as A/B tests of the GI estimator without a rebuild.
//   1<<15 (32768)   sphere/spot dw clamp 0.0001 - a floor, so it binds for any light
//                   closer than ~142 u (sphere radius 0.8 u) and therefore darkens
//                   almost everywhere if this clamp controls the image at all
//   1<<16 (65536)   sphere/spot dw clamp like the direct path (2*pi, no clamp at all)
//   1<<17 (131072)  sphere/spot indirect NEE gain x100 (clamp applied first, then scaled)
//   1<<18 (262144)  second bounce contribution x50 (applied after the final clamp)
//   1<<19 (524288)  SANITY: flat red indirect channel, to prove the bits arrive here
// 1<<18 is inert unless rt_gi_level is 2 (the second bounce is the high GI level).
#define INDIR_DBG_CLAMP_FLOOR   ( 1u << 15 )
#define INDIR_DBG_CLAMP_FULL    ( 1u << 16 )
#define INDIR_DBG_SPHOT_X100    ( 1u << 17 )
#define INDIR_DBG_BOUNCE2_X50   ( 1u << 18 )
#define INDIR_DBG_SANITY        ( 1u << 19 )

bool indirDbg( const uint bit )
{
    return ( globalUniform.debugShowFlags & bit ) != 0u;
}

float indirMaxSphereSolidAngle()
{
    if ( indirDbg( INDIR_DBG_CLAMP_FULL ) )
    {
        return 2.0 * M_PI;
    }
    if ( indirDbg( INDIR_DBG_CLAMP_FLOOR ) )
    {
        return 0.0001;
    }
    return Q2_INDIR_MAX_SPHERE_SOLID_ANGLE;
}

// Debug gain for the sphere/spot indirect NEE contribution. light.dw enters shade()
// linearly (diffuse = dw * nl * color * brdf * oneOverPdf), so scaling it is a clean
// multiplicative gain on exactly the light types the clamp above applies to.
float indirSphotGain()
{
    return indirDbg( INDIR_DBG_SPHOT_X100 ) ? 100.0 : 1.0;
}

float3 getSpecularBounce(const uint seed, uint bounceIndex,
                         const float3 n, const float roughness, const float3 surfSpecularColor,
                         const float3 v,
                         out float oneOverSourcePdf)
{
    const float2 u = rndBlueNoise8( seed, RANDOM_SALT_SPEC_BOUNCE( bounceIndex ) ).xy;
    return sampleSmithGGX( n, v, roughness, u[ 0 ], u[ 1 ], oneOverSourcePdf );
}

float3 getDiffuseBounce(const uint seed, uint bounceIndex,
                        const float3 n, out float oneOverSourcePdf)
{
    const float2 u = rndBlueNoise8( seed, RANDOM_SALT_DIFF_BOUNCE( bounceIndex ) ).xy;
    return sampleLambertian( n, u[ 0 ], u[ 1 ], oneOverSourcePdf );
}

// Q2RTX sun_attenuation: a smooth quartic falloff over the bounce ray's own length, so far
// away bounces neither receive sun light nor pay for a sun shadow ray. The first bounce ray
// is the specular one, so its range shrinks with the primary roughness exactly like Q2RTX
// does for `is_specular_ray`; the diffuse second bounce uses the full range.
float q2SunBounceAttenuation(float hitDistance, float sunBounceRange)
{
    // A zero range means "no indirect sun". The floor keeps the division finite so that a
    // hit at distance zero cannot turn the whole term into a NaN.
    sunBounceRange = max( sunBounceRange, 1e-4 );
    return square( clamp( 1.0 - square( square( hitDistance / sunBounceRange ) ), 0.0, 1.0 ) );
}

#define FIRST_BOUNCE_MIP_BIAS 0
#define SECOND_BOUNCE_MIP_BIAS 32.0

Surface traceBounce(const float3 originPosition, float originRoughness, uint originInstCustomIndex,
                    const float3 bounceDir, float bounceMipBias)
{
    const ShPayload p = traceIndirectRay( originInstCustomIndex, originPosition, bounceDir );
    rayStatsAdd( RAY_STATS_CATEGORY_INDIRECT, 1 );

    if ( !doesPayloadContainHitInfo( p ) )
    {
        Surface s;
        s.isSky = true;
        return s;
    }

    return hitInfoToSurface_Indirect(
        getHitInfoBounce( p, originPosition, originRoughness, bounceMipBias ),
        bounceDir );
}

float3 processSecondDiffuseBounce(const uint seed, const Surface surf, const float3 bounceDir,
                                  float oneOverPdf)
{
    const Surface hitSurf = traceBounce(
        surf.position + surf.normalGeom * 0.01,
        surf.roughness, surf.instCustomIndex, bounceDir, SECOND_BOUNCE_MIP_BIAS );

    if ( hitSurf.isSky )
    {
        return getSkyAmbientMultiplied( bounceDir, SKY_DIFFUSE_BOUNCE_LOD ) * oneOverPdf;
    }

    // Q2RTX processes analytic lights only in the first bounce pass, so the second bounce
    // of the diffuse path is lit by the sky it can reach and by the sun alone: the
    // `get_direct_illumination` call sits behind `spec_bounce_index == 0` and the light
    // list would otherwise be sampled twice for the same pixel.
    const float sunAttenuation = q2SunBounceAttenuation(
        length( hitSurf.position - surf.position ), globalUniform.sunBounce.x );

    float3 directDiffuse = (float3)0.0;

    if ( globalUniform.directionalLightExists != 0 && sunAttenuation > 0.0 )
    {
        const DirectionalLight sun =
            decodeAsDirectionalLight( lightSources[ LIGHT_ARRAY_DIRECTIONAL_LIGHT_OFFSET ] );
        const float2 sunRnd = rnd16_2( seed, Q2_RNG_INDIR2_SUN_DISK ) * 0.99;
        const LightSample sunLight = sampleDirectionalLight( sun, hitSurf.position, sunRnd );

        if ( getLuminance( sunLight.color ) > 0.0 )
        {
            bool sunTraced;
            const float sunVis = traceSunVisibility( hitSurf, sunLight, sunTraced );

            if ( sunTraced )
            {
                rayStatsAdd( RAY_STATS_CATEGORY_SHADOW_INDIRECT, 1 );
            }

            float3 d;
            shadeDiffuse( hitSurf, sunLight, 1.0, d );
            directDiffuse += d * sunVis * sunAttenuation * globalUniform.sunBounce.y;
        }
    }

    const float3 reflected = boostChroma( directDiffuse * hitSurf.albedo );
    return reflected * oneOverPdf;
}

void q2ShadeIndirect(const Surface surf, const float3 hitPos, const float3 hitRadiance,
                     float oneOverPdf, out float3 diffuse, out float3 specular)
{
    const float3 l  = safeNormalize( hitPos - surf.position );
    const float nl  = dot( surf.normal, l );
    const float ngl = dot( surf.normalGeom, l );

    if ( nl <= 0.0 || ngl <= 0.0 )
    {
        diffuse = specular = (float3)0.0;
        return;
    }

    const float3 radiance = boostChroma( hitRadiance );

    diffuse  = nl * radiance * evalBRDFLambertian( 1.0 );
    specular = nl * radiance *
               evalBRDFSmithGGX( surf.normal, surf.toViewerDir, l, surf.roughness,
                                 surf.specularColor );

    diffuse  *= oneOverPdf;
    specular *= oneOverPdf;
}

float q2SkyNeeBsdfWeight( const Surface surf, const float3 direction )
{
    if ( globalUniform.skyNee == 0.0 )
    {
        return 1.0;
    }

    const float pdfBsdf = evalSpecularBouncePdf(
        surf.normal, surf.toViewerDir, surf.roughness, direction );
    const float pdfSky = evalSkyNeePdf( surf.normal, direction );

    return pdfBsdf / max( pdfBsdf + pdfSky, 1e-8 );
}

float3 q2SkyNee( const Surface surf, const uint seed )
{
    if ( globalUniform.skyNee == 0.0 )
    {
        return (float3)0.0;
    }

    /* A cluster the host proved cannot see the sky never hits the dome: skip the ray the
       same way traceSunVisibility skips the sun shadow ray (q2ClusterSeesSky). */
    if ( !q2ClusterSeesSky( surf.cluster ) )
    {
        return (float3)0.0;
    }

    const float2 u = rnd16_2( seed, Q2_RNG_INDIR_SKY_NEE );

    float oneOverPdfSky;
    const float3 skyDirection = sampleLambertian( surf.normal, u[ 0 ], u[ 1 ], oneOverPdfSky );

    const float nl = dot( surf.normal, skyDirection );
    const float ngl = dot( surf.normalGeom, skyDirection );
    const float pdfSky = evalSkyNeePdf( surf.normal, skyDirection );

    if ( nl <= 0.0 || ngl <= 0.0 || pdfSky <= 0.0 )
    {
        return (float3)0.0;
    }

    const float pdfBsdf = evalSpecularBouncePdf(
        surf.normal, surf.toViewerDir, surf.roughness, skyDirection );
    const float misWeight = pdfSky / max( pdfSky + pdfBsdf, 1e-8 );

    const float3 radiance = boostChroma(
        getSkyAmbientMultiplied( skyDirection, surf.roughness * ( SKY_MIP_COUNT - 1.0 ) ) );

    if ( getLuminance( radiance ) * misWeight <= 0.0 )
    {
        return (float3)0.0;
    }

    const float vis = traceSkyVisibility( surf, skyDirection );
    rayStatsAdd( RAY_STATS_CATEGORY_SHADOW_INDIRECT, 1 );

    return evalBRDFLambertian( 1.0 ) * nl * radiance * vis * oneOverPdfSky * misWeight;
}

[shader("raygeneration")]
void main()
{
    int2 pix = (int2)DispatchRaysIndex().xy;

    // Q2RTX pt_num_bounce_rays == 0: no indirect lighting at all. Storing zeros
    // keeps the denoiser and the composite pass consistent - they only ever add
    // the SH to the direct lighting.
    if ( globalUniform.giBounceRays.x < 0.25 )
    {
        imageStoreUnfilteredIndirectSH( pix, irradianceToSH( (float3)0.0, float3( 0.0, 0.0, 1.0 ) ) );
        return;
    }

    // Q2RTX Low GI (pt_num_bounce_rays == 0.5): the host dispatches only half of the
    // rows and the traced rows alternate every frame, so the denoiser fills the rest.
    // Every traced pixel therefore stands in for two, hence the doubled contribution
    // below (Q2RTX does the same with throughput.rgb *= 2).
    const bool halfRes = globalUniform.giBounceRays.x < 0.75;
    if ( halfRes )
    {
        pix.y = pix.y * 2 + (int)( globalUniform.frameId & 1u );

        // Odd render heights leave the last row of the half resolution grid without
        // a partner; such a pixel keeps its previous value, like the untraced rows.
        //
        // textureSize(framebufQ2RngSeed_Sampled, 0).y is the texture's own height here:
        // GetDimensions hands back uints and the golden's ivec2 component is an int, so the
        // comparison takes a (int) cast, exact for every height this renderer creates.
        uint seedTexWidth, seedTexHeight;
        framebufQ2RngSeed_Sampled.GetDimensions(seedTexWidth, seedTexHeight);

        if ( pix.y >= (int)seedTexHeight )
        {
            return;
        }
    }
    const float halfResScale = halfRes ? 2.0 : 1.0;

    const uint seed = framebufQ2RngSeed_Sampled.Load(int3(pix, 0)).r;

    Surface surf = fetchGbufferSurface( pix );
    surf.position += surf.toViewerDir * RAY_ORIGIN_LEAK_BIAS;

    if ( surf.isSky )
    {
        imageStoreUnfilteredIndirectSH( pix, irradianceToSH( (float3)0.0, float3( 0.0, 0.0, 1.0 ) ) );
        return;
    }

    float oneOverSourcePdf;
    const float3 bounceDir = getSpecularBounce(
        seed, 1, surf.normal, surf.roughness, surf.specularColor, surf.toViewerDir,
        oneOverSourcePdf );

    const Surface hitSurf = traceBounce(
        surf.position + surf.normalGeom * 0.01,
        surf.roughness, surf.instCustomIndex, bounceDir, FIRST_BOUNCE_MIP_BIAS );

    float3 hitPos;
    float3 hitRadiance;
    float skyBsdfWeight = 1.0;
    if ( hitSurf.isSky )
    {
        skyBsdfWeight = q2SkyNeeBsdfWeight( surf, bounceDir );
        hitPos      = surf.position + bounceDir * MAX_RAY_LENGTH;
        hitRadiance = getSkyAmbientMultiplied( bounceDir, surf.roughness * ( SKY_MIP_COUNT - 1.0 ) );
    }
    else
    {
        const bool isGradient = q2GetIsGradient( pix );
        const uint cluster    = hitSurf.cluster;

        const float alpha       = square( hitSurf.roughness );
        const float phongExp    = q2RoughnessSquareToSpecPower( alpha );
        const float phongScale  = min( 100.0, 1.0 / ( M_PI * max( alpha * alpha, 1e-4 ) ) );
        const float phongWeight = clamp(
            getLuminance( hitSurf.specularColor ) /
                ( getLuminance( hitSurf.specularColor ) + getLuminance( hitSurf.albedo ) ),
            0.0, 0.9 );

        float3 directDiffuse = (float3)0.0;
        {
            const float3 rng = float3(
                rnd16( seed, Q2_RNG_INDIR_CELL_SELECT ),
                rnd16( seed, Q2_RNG_INDIR_CELL_SELECT + 1u ),
                rnd16( seed, Q2_RNG_INDIR_CELL_SELECT + 2u ) );

            uint  lightIndex = LIGHT_INDEX_NONE;
            uint  lightSlot  = 0u;
            float lightPdf   = 0.0;
            q2SampleClusterLights( cluster, hitSurf.position, hitSurf.normal, hitSurf.toViewerDir,
                                   phongExp, phongScale, phongWeight, isGradient, rng,
                                   lightIndex, lightSlot, lightPdf );

            if ( lightIndex != LIGHT_INDEX_NONE && lightPdf > 0.0 )
            {
                const float2 pointRnd = rnd16_2( seed, Q2_RNG_INDIR_LIGHT_POINT ) * 0.99;
                LightSample light =
                    sampleLight( lightSources[ lightIndex ], hitSurf.position, pointRnd );

                if ( lightSources[ lightIndex ].lightType == LIGHT_TYPE_SPHERE ||
                     lightSources[ lightIndex ].lightType == LIGHT_TYPE_SPOT )
                {
                    light.dw = min( light.dw, indirMaxSphereSolidAngle() ) * indirSphotGain();
                }

                if ( getLuminance( light.color ) > 0.0 )
                {
                    bool lightTraced;
                    const float vis =
                        traceLightVisibility( hitSurf, light, lightIndex, lightTraced );

                    if ( lightTraced )
                    {
                        rayStatsAdd( RAY_STATS_CATEGORY_SHADOW_INDIRECT, 1 );
                    }

                    if ( lightSources[ lightIndex ].lightType == LIGHT_TYPE_TRIANGLE ||
                         lightSources[ lightIndex ].lightType == LIGHT_TYPE_TEXTURED_AREA )
                    {
                        q2AccumulateLightStats( cluster, lightSlot, hitSurf.normal, vis, 0u );
                    }

                    float3 d;
                    shadeDiffuse( hitSurf, light, 1.0 / lightPdf, d );
                    directDiffuse += d * vis;
                }
            }
        }

        // The first bounce ray is the specular one, so its sun bounce range shrinks with the
        // primary roughness the same way Q2RTX shortens it for is_specular_ray.
        const float sunAttenuation = q2SunBounceAttenuation(
            length( hitSurf.position - surf.position ),
            globalUniform.sunBounce.x * ( 1.0 - sqrt( surf.roughness ) ) );

        if ( globalUniform.directionalLightExists != 0 && sunAttenuation > 0.0 )
        {
            const DirectionalLight sun =
                decodeAsDirectionalLight( lightSources[ LIGHT_ARRAY_DIRECTIONAL_LIGHT_OFFSET ] );
            const float2 sunRnd = rnd16_2( seed, Q2_RNG_INDIR_SUN_DISK ) * 0.99;
            const LightSample sunLight = sampleDirectionalLight( sun, hitSurf.position, sunRnd );

            if ( getLuminance( sunLight.color ) > 0.0 )
            {
                bool sunTraced;
                const float sunVis = traceSunVisibility( hitSurf, sunLight, sunTraced );

                if ( sunTraced )
                {
                    rayStatsAdd( RAY_STATS_CATEGORY_SHADOW_INDIRECT, 1 );
                }

                float3 d;
                shadeDiffuse( hitSurf, sunLight, 1.0, d );
                directDiffuse += d * sunVis * sunAttenuation * globalUniform.sunBounce.y;
            }
        }

        float3 secondBounce = (float3)0.0;
        if ( globalUniform.indirSecondBounce != 0u )
        {
            float oneOverPdfSecond;
            const float3 bounceDirSecond =
                getDiffuseBounce( seed, 2u, hitSurf.normalGeom, oneOverPdfSecond );
            secondBounce = processSecondDiffuseBounce(
                seed, hitSurf, bounceDirSecond, oneOverPdfSecond );
            directDiffuse += secondBounce;
        }

        directDiffuse = clamp( directDiffuse, (float3)0.0, (float3)50.0 );

        if ( indirDbg( INDIR_DBG_BOUNCE2_X50 ) )
        {
            directDiffuse = clamp( directDiffuse - secondBounce, (float3)0.0, (float3)50.0 )
                + 50.0 * secondBounce;
        }

        hitPos = hitSurf.position;
        hitRadiance = boostChroma( directDiffuse * hitSurf.albedo );
    }

    float3 diffuse, specular;
    q2ShadeIndirect( surf, hitPos, hitRadiance * halfResScale, oneOverSourcePdf, diffuse, specular );

    {
        const float3 direct = texelFetchUnfilteredSpecular( pix );
        if ( getLuminance( direct ) < getLuminance( specular ) )
        {
            framebufViewDirection[pix] =
                float4( -surf.toViewerDir, length( hitPos - surf.position ) );
        }

        imageStoreUnfilteredSpecular( pix,
                                      direct + demodulateSpecular( specular, surf.specularColor ) );
    }

    {
        SH sh = irradianceToSH( diffuse * skyBsdfWeight, safeNormalize( hitPos - surf.position ) );
        accumulateSH( sh, irradianceToSH( q2SkyNee( surf, seed ) * halfResScale, surf.normal ), 1.0 );

        if ( indirDbg( INDIR_DBG_SANITY ) )
        {
            sh = irradianceToSH( float3( 10.0, 0.0, 0.0 ), surf.normal );
        }

        imageStoreUnfilteredIndirectSH( pix, sh );
    }
}
