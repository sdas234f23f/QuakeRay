// Copyright (C) 2018 Christoph Schied
// Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) 2026 QuakeRay contributors
//
// This file is a port of shader/asvgf_taau.comp from Quake 2 RTX (https://github.com/NVIDIA/Q2RTX),
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
// Ported from Q2RTX (GPL v2) shader/asvgf_taau.comp. Adapted to vkpt: the
// input is the already tonemapped Final, so the PQ transfer is not applied.
//
// Temporal anti-aliasing + upscale (TAAU) for the new Q2RTX core path.
// Reads the tonemapped Final at render resolution and outputs UpscaledPing at
// the upscaled resolution, replacing the FSR/DLSS upscalers on the new path.
//
// HLSL counterpart of CmQ2TAAU.comp.
//
// Spellings that had to change:
//   * layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in
//     -> [numthreads(16, 16, 1)] on main, and gl_GlobalInvocationID -> SV_DispatchThreadID,
//     which the golden already converts with ivec2(...), so the cast stays as
//     int2(dispatchThreadID.xy)
//   * the two file scope constants taa_motion_weight_mix and taa_variance go through static
//     const, as in Q2Asvgf.hlsli: a namespace scope const array in HLSL is a uniform whose
//     contents are only known at runtime, while both of these are compile time constants
//   * texture2D / sampler parameters -> Texture2D<float4> / SamplerState, texelFetch(t, p, 0)
//     -> t.Load(int3(p, 0)), textureSize(tex, 0) -> tex.GetDimensions, and
//     textureLod(sampler2D(tex, texSampler), uv, 0) -> tex.SampleLevel(texSampler, uv, 0): the
//     golden builds a temporary sampler2D out of the two separate descriptors of the split table,
//     which is exactly the pair SampleLevel takes, so the two halves read the same texel
//   * imageStore(framebufUpscaledPing, opos, ...) / imageStore(framebufQ2TaaHistory, opos, ...)
//     -> the same storage image writes through the subscript
//   * mix -> lerp; vecN(0) / vec2(0.5) -> (floatN)0 / (float2)0.5, because HLSL has no one
//     argument vector constructor; vec4(color_output, 1) keeps its shape with the renamed type
//   * int(x) -> (int)x in the four render size locals, and vec2(renderWidth, renderHeight) ->
//     float2(renderWidth, renderHeight) keeps the golden's conversion constructor
//   * all(greaterThanEqual(a, b)) -> all(a >= b), all(lessThan(a, b)) -> all(a < b), the same
//     comparisons handed back as a bool vector by the HLSL operators; any(isnan(color_prev))
//     keeps its shape
//   * const vec2 Sample[3] out parameters of BicubicCatmullRom -> out float2 Sample[3], the same
//     array of the golden, and the call site keeps its four arguments
//
// What did not change: the bicubic Catmull-Rom weights of BicubicCatmullRom with its
// tc/f/f2/f3 chain and the w0..w3 order, the 3x3 sample loop of sample_texture_catmull_rom and
// its float4 weight broadcast, get_sample_weight and its clamp, the whole body and order of main
// (the render/upscaled size locals, the render_to_output ratio, the jitter, the nearest and
// integer render positions, the 3x3 moments loop with its skip of the center and the two clamps,
// the anti sparkle scale, the 9.0 division, the longest motion vector search, the previous
// position in the history, the two range comparisons, the neighbourhood color clamping with
// taa_variance, the motion and sample weights, the pixel weight clamp and the lerp), and the two
// stores at the end.

#define DESC_SET_FRAMEBUFFERS 0
#define DESC_SET_GLOBAL_UNIFORM 1
#include "ShaderCommonHLSLFunc.hlsli"

// The GLSL original declares the workgroup size here; in HLSL it is an attribute of the entry
// point instead, so the shader itself carries [numthreads(16, 16, 1)].

static const float taa_motion_weight_mix = 0.1;
static const float taa_variance = 1.0;

void
BicubicCatmullRom(float2 UV, float2 texSize, out float2 Sample[3], out float2 Weight[3])
{
    const float2 invTexSize = 1.0 / texSize;

    float2 tc = floor(UV - 0.5) + 0.5;
    float2 f = UV - tc;
    float2 f2 = f * f;
    float2 f3 = f2 * f;

    float2 w0 = f2 - 0.5 * (f3 + f);
    float2 w1 = 1.5 * f3 - 2.5 * f2 + 1;
    float2 w3 = 0.5 * (f3 - f2);
    float2 w2 = 1 - w0 - w1 - w3;

    Weight[0] = w0;
    Weight[1] = w1 + w2;
    Weight[2] = w3;

    Sample[0] = tc - 1;
    Sample[1] = tc + w2 / Weight[1];
    Sample[2] = tc + 2;

    Sample[0] *= invTexSize;
    Sample[1] *= invTexSize;
    Sample[2] *= invTexSize;
}

/* uv is in pixel coordinates */
float4
sample_texture_catmull_rom(Texture2D<float4> tex, SamplerState texSampler, float2 uv)
{
    float4 sum = (float4)0;
    float2 sampleLoc[3], sampleWeight[3];
    uint texWidth, texHeight;
    tex.GetDimensions(texWidth, texHeight);
    BicubicCatmullRom(uv, float2(texWidth, texHeight), sampleLoc, sampleWeight);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            float2 uvv = float2(sampleLoc[j].x, sampleLoc[i].y);
            float4 c = tex.SampleLevel(texSampler, uvv, 0);
            sum += c * (float4)(sampleWeight[j].x * sampleWeight[i].y);
        }
    }
    return sum;
}

float get_sample_weight(float2 delta, float scale)
{
    return clamp(1.0 - scale * dot(delta, delta), 0, 1);
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const int2 opos = int2(dispatchThreadID.xy);

    const int renderWidth = (int)globalUniform.renderWidth;
    const int renderHeight = (int)globalUniform.renderHeight;
    const int outWidth = (int)globalUniform.upscaledRenderWidth;
    const int outHeight = (int)globalUniform.upscaledRenderHeight;

    if (opos.x >= outWidth || opos.y >= outHeight)
    {
        return;
    }

    // Map the output pixel to the render resolution
    const float2 render_to_output = float2(renderWidth, renderHeight) / float2(outWidth, outHeight);
    const float2 sub_pixel_jitter = float2(globalUniform.jitterX, globalUniform.jitterY);
    const float2 nearest_render_pos = (float2(opos) + (float2)0.5) * render_to_output - (float2)0.5 - sub_pixel_jitter;
    const int2 int_render_pos = int2(clamp(round(nearest_render_pos), (float2)0, float2(renderWidth - 1, renderHeight - 1)));

    const float3 color_center_raw = framebufFinal_Sampled.Load(int3(int_render_pos, 0)).rgb;

    const float flt_taa_anti_sparkle = 0.25;
    const float anti_sparkle_scale = pow(min(1.0, flt_taa_anti_sparkle), -0.25);

    float3 mom1 = (float3)0.0;
    float3 mom2 = (float3)0.0;
    {
        const int r = 1;
        for (int yy = -r; yy <= r; yy++)
        {
            for (int xx = -r; xx <= r; xx++)
            {
                if (xx == 0 && yy == 0)
                    continue;
                int2 p = int_render_pos + int2(xx, yy);
                p = clamp(p, (int2)0, int2(renderWidth - 1, renderHeight - 1));
                float3 c = framebufFinal_Sampled.Load(int3(p, 0)).rgb;
                mom1 += c;
                mom2 += c * c;
            }
        }
    }

    float3 color_center = min(color_center_raw, anti_sparkle_scale * mom1 / 8.0);

    mom1 += color_center;
    mom2 += color_center * color_center;
    {
        const float num_pix = 9.0;
        mom1 /= num_pix;
        mom2 /= num_pix;
    }

    float3 color_output = color_center;

    // Find the longest motion vector in a 3x3 window
    float2 motion;
    {
        float len = -1;
        const int r = 1;
        for (int yy = -r; yy <= r; yy++)
        {
            for (int xx = -r; xx <= r; xx++)
            {
                int2 p = int_render_pos + int2(xx, yy);
                p = clamp(p, (int2)0, int2(renderWidth - 1, renderHeight - 1));
                float2 m = framebufMotionDlss_Sampled.Load(int3(p, 0)).rg;
                float l = dot(m, m);
                if (l > len)
                {
                    len = l;
                    motion = m;
                }
            }
        }
    }

    // Calculate the previous position in the (upscaled) TAA history
    const float2 pos_prev = ((float2(opos) + (float2)0.5) / float2(outWidth, outHeight) + motion.xy)
        * float2(outWidth, outHeight);

    // Scale the motion for the weight calculation below
    motion *= float2(outWidth, outHeight);

    if (all(int2(pos_prev) >= (int2)1)
    && all(int2(pos_prev) < int2(outWidth, outHeight) - 1))
    {
        // Motion vector was valid - sample the previous frame's TAA output
        float3 color_prev = sample_texture_catmull_rom(framebufQ2TaaHistory_Prev_Sampled, framebufQ2TaaHistory_Prev_Sampler, pos_prev).rgb;

        if (!any(isnan(color_prev)))
        {
            // Neighbourhood color clamping (NCC)
            float3 sigma = sqrt(max((float3)0, mom2 - mom1 * mom1));
            float3 mi = mom1 - sigma * taa_variance;
            float3 ma = mom1 + sigma * taa_variance;
            color_prev = clamp(color_prev, mi, ma);

            const float motion_weight = smoothstep(0.0, 1.0, sqrt(dot(motion, motion)));
            const float sample_weight = get_sample_weight(nearest_render_pos - (float2)int_render_pos, (float)outWidth * (1.0 / (float)renderWidth));
            float pixel_weight = max(motion_weight, sample_weight) * taa_motion_weight_mix;
            pixel_weight = clamp(pixel_weight, 0, 1);

            color_output = lerp(color_prev, color_center, pixel_weight);
        }
    }

    framebufUpscaledPing[opos] = float4(color_output, 1);
    framebufQ2TaaHistory[opos] = float4(color_output, 1);
}
