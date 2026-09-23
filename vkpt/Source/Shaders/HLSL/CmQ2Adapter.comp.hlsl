// Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) 2026 QuakeRay contributors
//
// This file is a port of shader/compositing.comp from Quake 2 RTX (https://github.com/NVIDIA/Q2RTX),
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
// HLSL counterpart of CmQ2Adapter.comp. Pinned by the pair GLSL/CmQ2Adapter.comp <->
// this file, which pulls in BRDF.hlsli and Q2Asvgf.hlsli and pins them through the calls of
// this pass.
//
// Spellings that had to change:
//   * layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in
//     -> [numthreads(16, 16, 1)] on main, and gl_GlobalInvocationID -> SV_DispatchThreadID,
//     which only the entry point can receive
//   * const utexture2D buf -> const Texture2D<uint4> buf, the samplerless unsigned view of
//     the r32ui framebuffer, passed as a function parameter the way the other accessors of
//     the port pass a Texture2D. texelFetch(buf, p, 0) -> buf.Load(int3(p, 0))
//   * ivec2/vec2/vec3/vec4/uvec4 -> int2/float2/float3/float4/uint4; the reinterpret casts of
//     the golden become C style conversions: uvec4(q2PackRGBE(hf)) -> (uint4)q2PackRGBE(hf)
//   * vec3(0.0) -> (float3)0.0 and vec3(globalUniform.fixedAlbedo.x) -> (float3)x: dxc refuses
//     to widen a single scalar through a constructor, so the broadcast is spelled as a cast
//     (measured in Q2Asvgf.hlsli, where the same broadcast folds identically on both halves)
//   * texelFetch(framebufAlbedo_Sampled, ...) / texelFetch(framebufThroughput_Sampled, ...) /
//     texelFetch(framebufMetallicRoughness_Sampled, ...) -> .Load(int3(pix, 0)) on the same
//     sampled views, as in ShaderCommonHLSLFunc.hlsli; texelFetchUnfilteredDirect,
//     texelFetchUnfilteredSpecular and texelFetchUnfilteredIndirectSH are those accessors and
//     keep their names
//   * imageStore(img, pix, v) -> img[pix] = v, the storage image write of the port; the whole
//     vec4 of the Q2Color store is kept, including its zero alpha. The golden's identity
//     constructor vec4(lf.shY * Q2_STORAGE_SCALE_LF) is dropped, because a vec4 of a vec4 is
//     the same value; the four-component one keeps its float4 spelling
//   * int(globalUniform.renderWidth) -> (int)globalUniform.renderWidth
//
// What did not change: the firefly constants Q2_FIREFLY_FACTOR and Q2_FIREFLY_MIN_LUM and the
// firefly test with its 1.0 initial maximum and its nine neighbour bounds; every operand
// order, the diffuse + indirect and specular + ro_s sums of the unfiltered composite, the
// DEBUG_SHOW_FLAG_* branches with their else if chain, the throughput and
// getMaterialAmbient(albedo) products, the premultiplied blends of the Q2Transparent and
// Q2FogAccum framebuffers, the max((float3)0.0, illuminated) clamp of the Q2Color store and
// the Q2_STORAGE_SCALE_HF / _SPEC / _LF scalings of the filtered path; the Q2RTX channel
// order UnfilteredDirect -> HF, UnfilteredSpecular -> SPEC, UnfilteredIndirectSH -> LF.

//   UnfilteredDirect      -> HF   (direct diffuse irradiance, packed RGBE)
//   UnfilteredSpecular    -> SPEC (demodulated specular irradiance, packed RGBE)
//   UnfilteredIndirectSH  -> LF   (indirect diffuse, YCoCg luma SH + chroma)
//
// The ASVGF denoiser then operates on these channels (see CmQ2Temporal.comp,
// CmQ2AtrousLF.comp, CmQ2Atrous.comp).

#define DESC_SET_FRAMEBUFFERS 0
#define DESC_SET_GLOBAL_UNIFORM 1
#include "ShaderCommonHLSLFunc.hlsli"
#include "BRDF.hlsli"
#include "Q2Asvgf.hlsli"

#define Q2_FIREFLY_FACTOR 8.0
#define Q2_FIREFLY_MIN_LUM 1.0

float3 q2AntiFirefly(const int2 pix, const float3 center, const Texture2D<uint4> buf, const int width, const int height)
{
    float maxNeighborLum = 1.0f;
    float centerLum = getLuminance(center);

    if (centerLum < Q2_FIREFLY_MIN_LUM)
    {
        return center;
    }

    for (int yy = -1; yy <= 1; yy++)
    {
        for (int xx = -1; xx <= 1; xx++)
        {
            if (xx == 0 && yy == 0)
            {
                continue;
            }

            const int2 p = pix + int2(xx, yy);
            if (p.x < 0 || p.x >= width || p.y < 0 || p.y >= height)
            {
                continue;
            }

            const float3 c = decodeE5B9G9R9(buf.Load(int3(p, 0)).r);
            maxNeighborLum = max(maxNeighborLum, getLuminance(c));
        }
    }

    const float limit = maxNeighborLum * Q2_FIREFLY_FACTOR;
    if (centerLum > limit)
    {
        return center * (limit / centerLum);
    }

    return center;
}

// Composites the raw, unfiltered ReSTIR signal. Used when the denoiser is
// disabled (Q2RTX flt_enable = 0): the lighting channels come straight from the
// path tracer instead of the ASVGF reconstructions, so the image is noisy but
// reacts to any lighting change immediately.
void q2CompositeUnfiltered(const int2 pix)
{
    const float3 diffuse = texelFetchUnfilteredDirect(pix);
    const float3 specular = texelFetchUnfilteredSpecular(pix);
    const SH indir = texelFetchUnfilteredIndirectSH(pix);

    const float3 albedo = framebufAlbedo_Sampled.Load(int3(getRegularPixFromCheckerboardPix(pix), 0)).rgb;
    const float3 throughput = framebufThroughput_Sampled.Load(int3(pix, 0)).rgb;

    float3 illuminated;

    if (!isSkyPix(pix))
    {
        const float2 mrFb = framebufMetallicRoughness_Sampled.Load(int3(pix, 0)).xy;
        const float3 normal = texelFetchNormal(pix);

        const float metallic = mrFb.x;

        const float3 indirect = SHToIrradiance(indir, normal);

        const float3 ro_s = getSpecularColor(albedo, metallic);
        float3 ro_d = albedo * (1.0 - metallic);

        // Q2RTX flt_fixed_albedo: replace the diffuse albedo after filtering,
        // keeping the specular reflectivity textured, for a "no textures" mode.
        if (globalUniform.fixedAlbedo.x != 0.0)
        {
            ro_d = (float3)globalUniform.fixedAlbedo.x;
        }

        illuminated =
            (diffuse + indirect) * ro_d +
            specular * ro_s;

        if ((globalUniform.debugShowFlags & DEBUG_SHOW_FLAG_ONLY_DIRECT_DIFFUSE) != 0)
        {
            illuminated = diffuse * albedo;
        }
        else if ((globalUniform.debugShowFlags & DEBUG_SHOW_FLAG_ONLY_SPECULAR) != 0)
        {
            illuminated = specular * ro_s;
        }
        else if ((globalUniform.debugShowFlags & DEBUG_SHOW_FLAG_ONLY_INDIRECT_DIFFUSE) != 0)
        {
            illuminated = indirect;
        }

        illuminated *= throughput;
        illuminated *= getMaterialAmbient(albedo);
    }
    else
    {
        illuminated = albedo * throughput;
    }

    const float4 q2Transparent = framebufQ2Transparent_Sampled.Load(int3(pix, 0));
    illuminated = q2Transparent.rgb + illuminated * (1.0 - q2Transparent.a);

    const float4 fogAccum = framebufQ2FogAccum_Sampled.Load(int3(pix, 0));
    illuminated = fogAccum.rgb + illuminated * (1.0 - fogAccum.a);

    framebufQ2Color[pix] = float4(max((float3)0.0, illuminated), 0);
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const int2 pix = int2(dispatchThreadID.xy);
    const int width = (int)globalUniform.renderWidth;
    const int height = (int)globalUniform.renderHeight;

    if (pix.x >= width || pix.y >= height)
    {
        return;
    }

    if (globalUniform.fltEnable.x < 0.5)
    {
        q2CompositeUnfiltered(pix);
        return;
    }

    const float3 diffuse = q2AntiFirefly(pix, texelFetchUnfilteredDirect(pix), framebufUnfilteredDirect_Sampled, width, height);
    const float3 specular = q2AntiFirefly(pix, texelFetchUnfilteredSpecular(pix), framebufUnfilteredSpecular_Sampled, width, height);
    const SH indir = texelFetchUnfilteredIndirectSH(pix);

    const float3 hf = diffuse * Q2_STORAGE_SCALE_HF;
    framebufQ2ColorHF[pix] = (uint4)q2PackRGBE(hf);

    // SPEC: demodulated specular irradiance, scaled for storage
    const float3 spec = specular * Q2_STORAGE_SCALE_SPEC;
    framebufQ2ColorSpec[pix] = (uint4)q2PackRGBE(spec);

    // LF: indirect diffuse SH converted to the YCoCg luma-SH + chroma format
    const Q2SH lf = q2IrradianceToSH(indir.r, indir.g, indir.b);
    framebufQ2ColorLF_SH[pix] = lf.shY * Q2_STORAGE_SCALE_LF;
    framebufQ2ColorLF_COCG[pix] = float4(lf.CoCg * Q2_STORAGE_SCALE_LF, 0.0, 0.0);
}
