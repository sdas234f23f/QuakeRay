// Copyright (C) 2018 Christoph Schied
// Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) 2026 QuakeRay contributors
//
// This file is a port of shader/asvgf_atrous.comp from Quake 2 RTX (https://github.com/NVIDIA/Q2RTX),
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
// Ported from Q2RTX (GPL v2) shader/asvgf_atrous.comp. The final iteration
// composites the channels using the vkpt lighting model (matches the vkpt
// ReSTIR outputs that feed the ASVGF).
//

// HLSL counterpart of CmQ2Atrous.comp.
//
// A-trous wavelet filter for the HF (direct diffuse) and SPEC (specular)
// channels, with variance-guided luminance filtering. The last invocation
// also composites all channels into the final Q2Color image.
//
// Spellings that had to change:
//   * gl_GlobalInvocationID -> SV_DispatchThreadID. The entry point reads it once and passes the
//     value to filter_image, which in the golden reads the same builtin for itself: HLSL has no
//     builtin that a helper function can read, so ipos becomes the first parameter of
//     filter_image. The value is the one the golden computed in both places, and main computes it
//     the same way.
//   * the golden's local_size_x/y = 16 become the [numthreads(16, 16, 1)] attribute of the entry
//     point
//   * layout(constant_id = 0) const uint q2AtrousIteration -> [[vk::constant_id(0)]] const uint
//   * texture2D / utexture2D parameters and locals -> Texture2D<float4> / Texture2D<uint4>,
//     with texelFetch(t, p, 0) -> t.Load(int3(p, 0)) and imageStore(img, p, v) -> img[p] = v;
//     the three inputs of filter_image are RWTexture2D storage images instead, because this
//     pass also writes the images they read (A4.5). The GLSL half selects the same globals
//     with two flags, because glslang rejects an image format qualifier on a function
//     parameter
//   * mix -> lerp and fract -> frac
//   * vec2(p), vec2(0.5) and the vec3(0.0) of the two max() calls -> (float2)p, (float2)0.5 and
//     (float3)0.0, because HLSL has no one-argument vector constructor; the vec4(sh, 0, 0) and
//     vec4(x, 0) constructors of the golden keep their shape with the renamed type
//   * the vector comparisons keep their sense and only change spelling:
//     all(greaterThanEqual(a, b)) -> all(a >= b), any(lessThan(a, b)) -> any(a < b), and the
//     float(...) of the bool vector reduction becomes a (float) cast
//   * int(x) / uint(x) / float(x) -> (int)x / (uint)x / (float)x; that includes the truncating
//     int2(globalUniform.renderWidth, renderHeight) of the tap bounds, kept as a constructor
//     because both arguments are converted component-wise on either side
//   * the two local tap arrays keep their initializers and their dynamic indexing; the golden's
//     nested braces of the ivec2 array are written as int2 constructors, the same list
//
// What did not change: the early out of filter_image when both channel counts are exhausted, the
// moments-based luminance variance with its 1e-8 clamp, the two normal-weight clamps (8, 1024),
// the per-pixel checkerboard field boundaries of the 3x3 pattern, the wavelet kernel indexed by
// the absolute tap offset, the depth, luminance and normal weights with their exp and pow, the
// skipped out-of-range taps of interpolate_lf and its 4-tap depth/normal weighting, the storage
// scales of the last iteration and the whole compositing chain, including the roughness
// smoothstep, the fixed-albedo branch, the three debug-show flags and the two operand orders of
// `illuminated`.

#define DESC_SET_FRAMEBUFFERS 0
#define DESC_SET_GLOBAL_UNIFORM 1
#include "ShaderCommonHLSLFunc.hlsli"
#include "BRDF.hlsli"
#include "Q2Asvgf.hlsli"

[[vk::constant_id(0)]] const uint q2AtrousIteration = 0;

Q2SH q2LoadSH(Texture2D<float4> img_shY, Texture2D<float4> img_CoCg, int2 p)
{
    Q2SH r;
    r.shY = img_shY.Load(int3(p, 0));
    r.CoCg = img_CoCg.Load(int3(p, 0)).xy;
    return r;
}

void
filter_image(
    const int2 ipos,
    RWTexture2D<uint4> img_hf,
    RWTexture2D<uint4> img_spec,
    RWTexture2D<float4> img_moments,
    out float3 filtered_hf,
    out float3 filtered_spec,
    out float2 filtered_moments)
{
    float3 color_center_hf = q2UnpackRGBE(img_hf.Load( ipos ).r);
    float3 color_center_spec = q2UnpackRGBE(img_spec.Load( ipos ).r);
    float2 moments_center = img_moments.Load( ipos ).xy;

    if(Q2_ATROUS_ITERATIONS_HF <= q2AtrousIteration && Q2_ATROUS_ITERATIONS_SPEC <= q2AtrousIteration)
    {
        filtered_hf = color_center_hf;
        filtered_spec = color_center_spec;
        filtered_moments = moments_center;
        return;
    }

    float3 normal_center = texelFetchNormal(ipos);
    float depth_center = framebufQ2ViewDepth_Sampled.Load(int3( ipos, 0 )).r;
    float fwidth_depth = framebufDepthGrad_Sampled.Load(int3( ipos, 0 )).r;
    float roughness_center = framebufMetallicRoughness_Sampled.Load(int3( ipos, 0 )).y;

    float lum_mean_hf = 0;
    float sigma_l_hf = 0;

    float hist_len_hf = framebufQ2HistMomentsHF_Sampled.Load(int3( ipos, 0 )).b;

    if(Q2_FLT_ATROUS_LUM_HF != 0 && hist_len_hf > 1)
    {
        // Compute luminance variance from the statistical moments: Var(X) = E[X^2] - E[X]^2
        lum_mean_hf = moments_center.x;
        float lum_variance_hf = max(1e-8, moments_center.y - moments_center.x * moments_center.x);
        sigma_l_hf = min(hist_len_hf, Q2_FLT_ATROUS_LUM_HF) / (2.0 * lum_variance_hf);
    }
    else
    {
        // No history: ignore luminance, perform a depth-normal-guided bilateral blur
        sigma_l_hf = 0;
    }

    // reduce the HF filter sensitivity to normals when the lighting is invalidated
    float normal_weight_scale = clamp(hist_len_hf / 8, 0, 1);

    float normal_weight_hf = Q2_FLT_ATROUS_NORMAL_HF;
    normal_weight_hf *= normal_weight_scale;
    float normal_weight_spec = roughnessSquaredToSpecPower(square(roughness_center)) * Q2_FLT_ATROUS_NORMAL_SPEC;
    normal_weight_spec = clamp(normal_weight_spec, 8, 1024);
    normal_weight_spec *= normal_weight_scale;

    const int step_size = (int)(1u << q2AtrousIteration);

    float3 sum_color_hf = color_center_hf.rgb;
    float3 sum_color_spec = color_center_spec.rgb;
    float2 sum_moments = moments_center;

    float sum_w_hf = 1.0;
    float sum_w_spec = 1.0;

    // Boundaries for the checkerboard field, either left or right half of the screen
    const int width = (int)globalUniform.renderWidth;
    const int height = (int)globalUniform.renderHeight;
    int field_left = 0;
    int field_right = width / 2;
    if(ipos.x >= field_right)
    {
        field_left = field_right;
        field_right = width;
    }

    float spec_filter_width_scale = clamp(roughness_center * 30 - (float)q2AtrousIteration, 0, 1);

    // Compute the weighted average of color and moments from a sparse 3x3 pattern around the target pixel
    const int r = 1;
    for(int yy = -r; yy <= r; yy++)
    {
        for(int xx = -r; xx <= r; xx++)
        {
            int2 p = ipos + int2(xx, yy) * step_size;

            if(xx == 0 && yy == 0)
                continue;

            float w = (float)(all(p >= int2(field_left, 0))
                    && all(p < int2(field_right, height)));

            float3 normal = texelFetchNormal(p);

            float depth = framebufQ2ViewDepth_Sampled.Load(int3( p, 0 )).r;
            float roughness = framebufMetallicRoughness_Sampled.Load(int3( p, 0 )).y;

            float dist_z = abs(depth_center - depth) * fwidth_depth * Q2_FLT_ATROUS_DEPTH;
            w *= exp(-dist_z / (float)step_size);
            w *= Q2_WAVELET_KERNEL[abs(xx)][abs(yy)];

            float w_hf = w;

            float3 c_hf = q2UnpackRGBE(img_hf.Load( p ).r);
            float3 c_spec = q2UnpackRGBE(img_spec.Load( p ).r);
            float2 c_mom = img_moments.Load( p ).xy;
            float l_hf = getLuminance(c_hf.rgb);
            float dist_l_hf = abs(lum_mean_hf - l_hf);

            w_hf *= exp(- dist_l_hf * dist_l_hf * sigma_l_hf);

            float w_spec = w_hf;
            w_spec *= max(0, 1 - 10 * abs(roughness - roughness_center));
            w_spec *= spec_filter_width_scale;

            float NdotN = max(0.0, dot(normal_center, normal));

            if(normal_weight_hf > 0)
            {
                w_hf *= pow(NdotN, normal_weight_hf);
            }

            if(normal_weight_spec > 0)
            {
                w_spec *= pow(NdotN, normal_weight_spec);
            }

            if(Q2_ATROUS_ITERATIONS_HF <= q2AtrousIteration)
                w_hf = 0;

            if(Q2_ATROUS_ITERATIONS_SPEC <= q2AtrousIteration)
                w_spec = 0;

            sum_color_hf += c_hf.rgb * w_hf;
            sum_color_spec += c_spec.rgb * w_spec;
            sum_moments  += c_mom * w_hf;
            sum_w_hf     += w_hf;
            sum_w_spec   += w_spec;
        }
    }

    filtered_hf = sum_color_hf / sum_w_hf;
    filtered_spec = sum_color_spec / sum_w_spec;
    filtered_moments = sum_moments / sum_w_hf;
}

// Bilinear/bilateral interpolation of the LF channel data (denoised at 1/3 resolution)
Q2SH interpolate_lf(Texture2D<float4> img_lf_shY, Texture2D<float4> img_lf_CoCg, int2 ipos)
{
    float depth_center = framebufQ2ViewDepth_Sampled.Load(int3( ipos, 0 )).r;
    float fwidth_depth = framebufDepthGrad_Sampled.Load(int3( ipos, 0 )).r;
    float3 geo_normal_center = texelFetchNormalGeometry(ipos);

    float2 pos_lowres = ((float2)ipos + (float2)0.5) / Q2_GRAD_DWN - (float2)0.5;
    float2 pos_ld = floor(pos_lowres);
    float2 subpix = frac(pos_lowres - pos_ld);

    Q2SH sum_lf = q2InitSH();
    float sum_w = 0;

    // 4 bilinear taps
    const int2 off[4] = { int2( 0, 0 ), int2( 1, 0 ), int2( 0, 1 ), int2( 1, 1 ) };
    float w[4] = {
        (1.0 - subpix.x) * (1.0 - subpix.y),
        (subpix.x      ) * (1.0 - subpix.y),
        (1.0 - subpix.x) * (subpix.y      ),
        (subpix.x      ) * (subpix.y      )
    };
    for(int i = 0; i < 4; i++)
    {
        int2 p_lowres = (int2)pos_ld + off[i];
        int2 p_hires = p_lowres * Q2_GRAD_DWN + (int2)1;

        // Out-of-range taps are skipped rather than fetched with a zero weight: an
        // out-of-range texelFetch is undefined, and a NaN would survive the weights.
        // It also rejects the low-res texels that main() never writes when the render
        // size is not a multiple of Q2_GRAD_DWN.
        if(any(p_hires < (int2)0)
                || any(p_hires >= int2(globalUniform.renderWidth, globalUniform.renderHeight)))
            continue;

        float p_depth = framebufQ2ViewDepth_Sampled.Load(int3( p_hires, 0 )).r;
        float3 p_geo_normal = texelFetchNormalGeometry(p_hires);

        float p_w = w[i];

        float dist_depth = abs(p_depth - depth_center) * fwidth_depth;
        p_w *= exp(-dist_depth);
        p_w *= pow(max(0.0, dot(geo_normal_center, p_geo_normal)), 8);

        if(p_w > 0)
        {
            Q2SH p_lf = q2LoadSH(img_lf_shY, img_lf_CoCg, p_lowres);
            q2AccumulateSH(sum_lf, p_lf, p_w);
            sum_w += p_w;
        }
    }

    if(sum_w > 0)
    {
        float inv_w = 1 / sum_w;
        sum_lf.shY *= inv_w;
        sum_lf.CoCg *= inv_w;
    }
    else
    {
        // Nothing relevant among the 4 taps - use the full-res temporally filtered LF
        sum_lf = q2LoadSH(framebufQ2HistColorLF_SH_Sampled, framebufQ2HistColorLF_COCG_Sampled, ipos);
    }

    return sum_lf;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int2 ipos = int2(dispatchThreadID.xy);
    if(any(ipos >= int2(globalUniform.renderWidth, globalUniform.renderHeight)))
        return;

    float3 filtered_hf;
    float3 filtered_spec;
    float2 filtered_moments;

    // Read the HF/SPEC/moments inputs through their storage images, not the sampled views:
    // this pass also writes them (each iteration reads the field the previous one wrote),
    // and binding both views of one image in one set makes the SRV and the UAV disagree
    // about the image layout (A4.5). The packed RGBE words travel unchanged: the sampled
    // view was a typed r32ui view too, so q2UnpackRGBE decodes the same bits.
    switch(q2AtrousIteration)
    {
    case 0: filter_image(ipos, framebufQ2AtrousPingHF, framebufQ2AtrousPingSpec, framebufQ2AtrousPingMoments, filtered_hf, filtered_spec, filtered_moments); break;
    case 1: filter_image(ipos, framebufQ2HistColorHF,  framebufQ2AtrousPongSpec, framebufQ2AtrousPongMoments, filtered_hf, filtered_spec, filtered_moments); break;
    case 2: filter_image(ipos, framebufQ2AtrousPingHF, framebufQ2AtrousPingSpec, framebufQ2AtrousPingMoments, filtered_hf, filtered_spec, filtered_moments); break;
    case 3: filter_image(ipos, framebufQ2AtrousPongHF, framebufQ2AtrousPongSpec, framebufQ2AtrousPongMoments, filtered_hf, filtered_spec, filtered_moments); break;
    }

    switch(q2AtrousIteration)
    {
    case 0:
        framebufQ2HistColorHF[ipos] = (uint4)q2PackRGBE(filtered_hf);
        framebufQ2AtrousPongSpec[ipos] = (uint4)q2PackRGBE(filtered_spec);
        framebufQ2AtrousPongMoments[ipos] = float4(filtered_moments, 0, 0);
        break;
    case 1:
        framebufQ2AtrousPingHF[ipos] = (uint4)q2PackRGBE(filtered_hf);
        framebufQ2AtrousPingSpec[ipos] = (uint4)q2PackRGBE(filtered_spec);
        framebufQ2AtrousPingMoments[ipos] = float4(filtered_moments, 0, 0);
        break;
    case 2:
        framebufQ2AtrousPongHF[ipos] = (uint4)q2PackRGBE(filtered_hf);
        framebufQ2AtrousPongSpec[ipos] = (uint4)q2PackRGBE(filtered_spec);
        framebufQ2AtrousPongMoments[ipos] = float4(filtered_moments, 0, 0);
        break;
    }

    // Perform compositing on the last iteration
    if(q2AtrousIteration == 3)
    {
        Q2SH filtered_lf = interpolate_lf(framebufQ2AtrousPingLF_SH_Sampled, framebufQ2AtrousPingLF_COCG_Sampled, ipos);

        filtered_lf.shY /= Q2_STORAGE_SCALE_LF;
        filtered_lf.CoCg /= Q2_STORAGE_SCALE_LF;
        filtered_hf /= Q2_STORAGE_SCALE_HF;
        filtered_spec /= Q2_STORAGE_SCALE_SPEC;

        float3 albedo = framebufAlbedo_Sampled.Load(int3( getRegularPixFromCheckerboardPix(ipos), 0 )).rgb;
        float3 throughput = framebufThroughput_Sampled.Load(int3( ipos, 0 )).rgb;

        float3 illuminated;

        if(!isSkyPix(ipos))
        {
            const float2 mrFb = framebufMetallicRoughness_Sampled.Load(int3( ipos, 0 )).xy;
            const float3 normal = texelFetchNormal(ipos);

            const float metallic = mrFb.x;
            const float roughness = mrFb.y;

            const float3 diffuse = filtered_hf;
            const float3 indirect = q2SHToIrradiance(filtered_lf, normal);

            // to prevent noise, specular will be replaced with SH, if the surface is too rough
            const float3 specular = lerp(
                filtered_spec,
                indirect,
                smoothstep(
                        FAKE_ROUGH_SPECULAR_THRESHOLD,
                        FAKE_ROUGH_SPECULAR_THRESHOLD + FAKE_ROUGH_SPECULAR_LENGTH,
                        roughness
                ));

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
                illuminated = specular * getSpecularColor(albedo, metallic);
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

        const float4 q2Transparent = framebufQ2Transparent_Sampled.Load(int3( ipos, 0 ));
        illuminated = q2Transparent.rgb + illuminated * (1.0 - q2Transparent.a);

        // Blend the per-ray accumulated fog (Q2RTX approach: fog is traced in
        // the primary/refl passes and accumulated over each path segment), so
        // it is part of the denoised, temporally consistent signal.
        const float4 fogAccum = framebufQ2FogAccum_Sampled.Load(int3( ipos, 0 ));
        illuminated = fogAccum.rgb + illuminated * (1.0 - fogAccum.a);

        framebufQ2Color[ipos] = float4(max((float3)0.0, illuminated), 0);
    }
}
