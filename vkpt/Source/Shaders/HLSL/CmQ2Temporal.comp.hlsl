// Copyright (C) 2018 Christoph Schied
// Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) 2026 QuakeRay contributors
//
// This file is a port of shader/asvgf_temporal.comp from Quake 2 RTX (https://github.com/NVIDIA/Q2RTX),
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
// Ported from Q2RTX (GPL v2) shader/asvgf_temporal.comp. Adapted to the vkpt
// framework: gradients come from the Q2RTX-style gradient pipeline
// (CmQ2GradientReproject + CmQ2GradientImg + CmQ2GradientAtrous),
// depth derivative from DepthGrad, split surfaces via Throughput.a.
//


// HLSL counterpart of CmQ2Temporal.comp.
//
// Spellings that had to change:
//   * layout(local_size_x = GROUP_SIZE, local_size_y = GROUP_SIZE, local_size_z = 1) ->
//     [numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
//   * gl_GlobalInvocationID -> SV_DispatchThreadID, gl_LocalInvocationID -> SV_GroupThreadID,
//     gl_LocalInvocationIndex -> SV_GroupIndex, gl_WorkGroupID -> SV_GroupID. The four are entry
//     point semantics in HLSL, so preload and get_shared_data receive the ones they used to read
//     globally as parameters (groupID and linearIdx, groupThreadID); the arithmetic that reads
//     them is unchanged, ivec2(gl_WorkGroupID) is spelled int2(groupID.xy) and ivec2(gl_
//     LocalInvocationID) is spelled int2(groupThreadID.xy)
//   * shared -> groupshared, barrier() -> GroupMemoryBarrierWithGroupSync()
//   * the two texture2D parameters of q2LoadSH become Texture2D<float4>: the golden's q2LoadSH is
//     this file's own helper (the header recommends it), and neither half of the LF pair needs a
//     format tag to be loaded
//   * texelFetch(t, p, 0) -> t.Load(int3(p, 0)), imageStore(img, p, v) -> img[p] = v
//   * mix(a, b, s) -> lerp(a, b, s) at every one of the eleven sites, the same definition under
//     the HLSL name, as in BRDF.hlsli
//   * fract(x) -> frac(x): the x - floor(x) intrinsic under its HLSL name, as in
//     CmLuminanceHistogram.comp
//   * vecN(scalar) -> (floatN)scalar and uvec2(scalar) -> (uint2)scalar, because a scalar does
//     not broadcast through a constructor in HLSL: vec2(0.5), vec2(0.0) and vec3(0.0) and
//     vec4(0.0) of the golden become (float2)0.5, (float2)0.0, (float3)0.0 and (float4)0.0,
//     ivec2(0) becomes (int2)0, ivec2(FILTER_RADIUS) becomes (int2)FILTER_RADIUS, and
//     uvec2(1) becomes (uint2)1. The vecN(...) calls that collect several components keep their
//     argument lists and only change the type name (vec4(spatial_moments_hf, 1, 1) becomes
//     float4(spatial_moments_hf, 1, 1), and the two int literals still convert to floats)
//   * packHalf2x16(v) and unpackHalf2x16(u) have no single intrinsic in HLSL: f32tof16 converts
//     one float at a time and f16tof32 reads one half at a time, so the two helpers below
//     assemble the pair with the shift and the mask of the golden's layout (the low half first,
//     the high half second). dxc lowers the scalar f32tof16 to PackHalf2x16 of (x, 0) and the
//     scalar f16tof32 to UnpackHalf2x16(u).x, so the pair of them is exactly the instruction the
//     GLSL builtin emits. Neither compiler folds those two instructions, so the site is proven
//     at operand level and not on a folded number. The surrounding vec2/int2 constructors keep
//     their component lists: the s_normal_lum pair is
//     uint2(packHalf2x16(normal.xy), packHalf2x16(float2(normal.z, getLuminance(color_hf)))) and
//     the three unpack sites are float3(unpackHalf2x16(...), unpackHalf2x16(...).x) and
//     unpackHalf2x16(...).y
//   * uint(...) / int(...) / float(...) constructors -> C style casts with the same truncation:
//     int(floor(...)), int(globalUniform.renderWidth), float(linear_idx), float(SHARED_SIZE)
//     and uint(linear_idx) are the ones of this file
//   * uvec4(q2PackRGBE(...)) -> (uint4)q2PackRGBE(...): the one argument constructor of the
//     golden is the broadcast that a scalar cast performs in HLSL. uvec2(...) and uint2(...)
//     constructors of several components keep their argument lists
//   * Q2_STORE_SH and the Q2_* macros are used as they are; the macro body lives in Q2Asvgf.hlsli
//
// What did not change: the four shared buffers with their sizes and types, the GROUP_SIZE /
// FILTER_RADIUS / SHARED_SIZE values, every operand order and every clamp, max, min, pow, abs,
// dot, floor, square and length of the golden, the reprojection of both checkerboard modes with
// its q2CheckerToFlat / q2FlatToChecker conversions and the two integer divisions width / 2 and
// width / 2, the 2x2 bilinear tap loop with its four weights and the off[] table, the history
// validation (relative depth difference, both dot products, the 0.25 of the secondary surfaces),
// the two accumulation branches and their 1e-6 guards, the spatial 3x3 moment window, the
// antilag alphas and the history lengths with their pow(..., 10) and 256.0 clamp, the three
// alpha blends, the two fallback branches, the six stores, the 1/3 resolution downsample with
// the bilateral 3x3 window, and the final Q2_STORE_SH of the LF channel.
//
// Known caveat of the pair, the same one the fleet recorded for EfCommon.hlsli's divisions:
// t = ((float)linear_idx + 0.5) / (float)SHARED_SIZE keeps OpFDiv in the golden blob, which
// GenerateShaders.py builds with plain glslc, while dxc optimizes by default and rewrites the
// division into OpFMul by 0.0588235296 (measured: plain glslc on the golden keeps
// OpFDiv %float_17, the port has OpFMul %float_0_0588235296; with -O on both sides, as the
// property checker runs them, the golden rewrites it too and the two agree). The constant is the
// correctly rounded fl(1/17), but x * fl(1/17) and x / 17 differ by one ulp for many inputs
// (e.g. 44.538719177246094 / 17 = 2.619924545288086 against * = 2.619924783706665), so the two
// blobs can disagree by one ulp on t, which is enough to move a shared memory preload tap to a
// neighbouring texel. The port keeps the golden's spelling; the deviation is bounded by that
// one tap and cannot be removed without writing a hand-made reciprocal into the port.

#define DESC_SET_FRAMEBUFFERS 0
#define DESC_SET_GLOBAL_UNIFORM 1
#include "ShaderCommonHLSLFunc.hlsli"
#include "Q2Asvgf.hlsli"

#define GROUP_SIZE 15
// spatially compute variance in a 3x3 (radius = 1) or a 5x5 (radius = 2) window
#define FILTER_RADIUS 1
// size of the shared memory copies of color, depth, and normals
#define SHARED_SIZE (GROUP_SIZE + FILTER_RADIUS * 2)

// Store some color data in shared memory for efficient access and for downsampling
groupshared uint2 s_normal_lum[SHARED_SIZE][SHARED_SIZE];
groupshared float s_depth[SHARED_SIZE][SHARED_SIZE];
groupshared float4 s_lf_shy[GROUP_SIZE][GROUP_SIZE];
groupshared float2 s_lf_cocg[GROUP_SIZE][GROUP_SIZE];
groupshared float s_depth_width[GROUP_SIZE / Q2_GRAD_DWN][GROUP_SIZE / Q2_GRAD_DWN];

// The pair packing of the golden, assembled from the two scalar conversions because HLSL has no
// intrinsic that packs two halves into one uint. The layout is the GLSL one: the first component
// in the low 16 bits, the second in the high 16 bits.
uint packHalf2x16(const float2 v)
{
    return f32tof16(v.x) | (f32tof16(v.y) << 16);
}

float2 unpackHalf2x16(const uint u)
{
    return float2(f16tof32(u & 0xFFFF), f16tof32(u >> 16));
}

Q2SH q2LoadSH(Texture2D<float4> img_shY, Texture2D<float4> img_CoCg, int2 p)
{
    Q2SH r;
    r.shY = img_shY.Load(int3(p, 0));
    r.CoCg = img_CoCg.Load(int3(p, 0)).xy;
    return r;
}

// Preload the color data into shared memory
void
preload(const uint3 groupID, const uint linearIdx)
{
    int2 groupBase = int2(groupID.xy) * GROUP_SIZE - FILTER_RADIUS;

    // The size of these shared memory buffers is larger than the group size because we
    // use them for some spatial filtering. So a single load per thread is not enough.
    for(uint linear_idx = linearIdx; linear_idx < SHARED_SIZE * SHARED_SIZE; linear_idx += GROUP_SIZE * GROUP_SIZE)
    {
        float t = ((float)linear_idx + 0.5) / (float)SHARED_SIZE;
        int xx = (int)floor(frac(t) * (float)SHARED_SIZE);
        int yy = (int)floor(t);

        int2 ipos = groupBase + int2(xx, yy);
        float depth = framebufQ2ViewDepth_Sampled.Load(int3(ipos, 0)).r;
        float3 normal = texelFetchNormal(ipos);
        float3 color_hf = q2UnpackRGBE(framebufQ2ColorHF_Sampled.Load(int3(ipos, 0)).r);

        s_normal_lum[yy][xx] = uint2(packHalf2x16(normal.xy), packHalf2x16(float2(normal.z, getLuminance(color_hf))));
        s_depth[yy][xx] = depth;
    }
}

// Load the color and normal data from shared memory
void
get_shared_data(const uint3 groupThreadID, int2 offset, out float depth, out float3 normal, out float lum_hf)
{
    int2 addr = int2(groupThreadID.xy) + (int2)FILTER_RADIUS + offset;

    uint2 normal_lum = s_normal_lum[addr.y][addr.x];
    depth = s_depth[addr.y][addr.x];

    normal = float3(unpackHalf2x16(normal_lum.x), unpackHalf2x16(normal_lum.y).x);
    lum_hf = unpackHalf2x16(normal_lum.y).y;
}

[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID,
          uint groupIndex : SV_GroupIndex, uint3 groupID : SV_GroupID)
{
    preload(groupID, groupIndex);
    GroupMemoryBarrierWithGroupSync();

    int2 ipos = int2(dispatchThreadID.xy);
    float4 motion = framebufMotion_Sampled.Load(int3(ipos, 0));
    const float gradDepth = framebufDepthGrad_Sampled.Load(int3(ipos, 0)).r;

    // Find out if this pixel belongs to a checkerboard-split-path surface (water/glass)
    const float wasSplit = framebufThroughput_Sampled.Load(int3(ipos, 0)).a;
    const bool is_checkerboarded_surface = wasSplit > 0.5;

    // On a single GPU we can access both checkerboard fields for higher sampling quality
    const bool sample_across_fields = !is_checkerboarded_surface;

    const int width = (int)globalUniform.renderWidth;
    const int height = (int)globalUniform.renderHeight;
    const float inv_width = 1.0 / globalUniform.renderWidth;
    const float inv_height = 1.0 / globalUniform.renderHeight;

    float2 pos_prev;
    if (sample_across_fields)
    {
        // Reprojection in flat-screen coordinates
        pos_prev = ((float2(q2CheckerToFlat(ipos, width)) + (float2)0.5) * float2(inv_width, inv_height) + motion.xy) * float2(width, height);
    }
    else
    {
        // Reprojection in checkerboarded coordinates
        pos_prev = ((float2(ipos) + (float2)0.5) * float2(inv_width * 2.0, inv_height) + motion.xy) * float2(width / 2, height);
    }

    float motion_length = length(motion.xy * float2(width, height));

    // Load the parameters of the target pixel
    float depth_curr;
    float3 normal_curr;
    float lum_curr_hf;
    get_shared_data(groupThreadID, (int2)0, depth_curr, normal_curr, lum_curr_hf);

    float2 metal_rough = framebufMetallicRoughness_Sampled.Load(int3(ipos, 0)).xy;
    float shininess = clamp(2.0 / square(square(metal_rough.y)) - 2.0, 0.0, 32.0);

    float3 geo_normal_curr = texelFetchNormalGeometry(ipos);

    // Try to get the history sample for all channels, including HF moments
    bool temporal_sample_valid_diff = false;
    bool temporal_sample_valid_spec = false;
    Q2SH temporal_color_lf = q2InitSH();
    float3 temporal_color_hf = (float3)0.0;
    float4 temporal_color_histlen_spec = (float4)0.0;
    float4 temporal_moments_histlen_hf = (float4)0.0;
    {
        float temporal_sum_w_diff = 0.0;
        float temporal_sum_w_spec = 0.0;

        float2 pos_ld = floor(pos_prev - (float2)0.5);
        float2 subpix = frac(pos_prev - (float2)0.5 - pos_ld);

        int field_left = 0;
        int field_right = sample_across_fields ? width : (width / 2);
        if (!sample_across_fields && ipos.x >= width / 2)
        {
            field_left = field_right;
            field_right = width;
        }

        // Bilinear/bilateral filter
        const int2 off[4] = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } };
        float w[4] = {
            (1.0 - subpix.x) * (1.0 - subpix.y),
            (subpix.x      ) * (1.0 - subpix.y),
            (1.0 - subpix.x) * (subpix.y      ),
            (subpix.x      ) * (subpix.y      )
        };
        for(int i = 0; i < 4; i++)
        {
            int2 p = int2(pos_ld) + off[i];

            if(p.x < field_left || p.x >= field_right || p.y >= height || p.y < 0)
                continue;

            if (sample_across_fields)
            {
                // p is in flat coordinates - translate back into checkerboarded coordinates
                p = q2FlatToChecker(p, width);
            }

            float depth_prev = framebufQ2ViewDepth_Prev_Sampled.Load(int3(p, 0)).r;
            float3  normal_prev = texelFetchNormal_Prev(p);
            float3  geo_normal_prev = texelFetchNormalGeometry_Prev(p);

            // Q2ViewDepth is negative for reflection/refraction surfaces, and the
            // stored motion depth delta (motion.z) is in the positive primary
            // space - flip its sign so the motion-compensated match stays exact.
            const float motion_z = (depth_curr < 0.0) ? -motion.z : motion.z;
            float dist_depth = abs(depth_curr - depth_prev + motion_z) / abs(depth_curr);
            float dot_normals = dot(normal_curr, normal_prev);
            float dot_geo_normals = dot(geo_normal_curr, geo_normal_prev);

            if(depth_curr < 0)
            {
                // Reduce the filter sensitivity to depth for secondary surfaces,
                // because reflection/refraction motion vectors are often inaccurate.
                dist_depth *= 0.25;
            }

            if(dist_depth < 0.1 && dot_geo_normals > 0.5)
            {
                float w_diff = w[i] * max(dot_normals, 0);
                float w_spec = w[i] * pow(max(dot_normals, 0), shininess);

                Q2SH hist_color_lf = q2LoadSH(framebufQ2HistColorLF_SH_Prev_Sampled, framebufQ2HistColorLF_COCG_Prev_Sampled, p);
                q2AccumulateSH(temporal_color_lf, hist_color_lf, w_diff);

                temporal_color_hf           += q2UnpackRGBE(framebufQ2HistColorHF_Prev_Sampled.Load(int3(p, 0)).r) * w_diff;
                temporal_color_histlen_spec += framebufQ2FilteredSpec_Prev_Sampled.Load(int3(p, 0)) * w_spec;
                temporal_moments_histlen_hf += framebufQ2HistMomentsHF_Prev_Sampled.Load(int3(p, 0)).rgba * w_diff;
                temporal_sum_w_diff         += w_diff;
                temporal_sum_w_spec         += w_spec;
            }
        }

        if(temporal_sum_w_diff > 1e-6)
        {
            float inv_w_diff = 1.0 / temporal_sum_w_diff;
            temporal_color_lf.shY  *= inv_w_diff;
            temporal_color_lf.CoCg *= inv_w_diff;
            temporal_color_hf      *= inv_w_diff;
            temporal_moments_histlen_hf *= inv_w_diff;
            temporal_sample_valid_diff = true;
        }

        if(temporal_sum_w_spec > 1e-6)
        {
            float inv_w_spec = 1.0 / temporal_sum_w_spec;
            temporal_color_histlen_spec *= inv_w_spec;
            temporal_sample_valid_spec = true;
        }
    }

    // Compute spatial moments of the HF channel in a 3x3 window
    float2 spatial_moments_hf = float2(lum_curr_hf, lum_curr_hf * lum_curr_hf);

    {
        float spatial_sum_w_hf = 1.0;
        for(int yy = -FILTER_RADIUS; yy <= FILTER_RADIUS; yy++)
        {
            for(int xx = -FILTER_RADIUS; xx <= FILTER_RADIUS; xx++)
            {
                if(xx == 0 && yy == 0)
                    continue;

                float depth;
                float3 normal;
                float lum_p_hf;
                get_shared_data(groupThreadID, int2(xx, yy), depth, normal, lum_p_hf);

                float dist_z = abs(depth_curr - depth) * gradDepth;
                if(dist_z < 2.0)
                {
                    float w_hf = pow(max(0.0, dot(normal, normal_curr)), 128.0);

                    spatial_moments_hf += float2(lum_p_hf * w_hf, lum_p_hf * lum_p_hf * w_hf);
                    spatial_sum_w_hf  += w_hf;
                }
            }
        }

        spatial_moments_hf /= spatial_sum_w_hf;
    }

    // Load the target pixel colors for all channels
    Q2SH color_curr_lf = q2LoadSH(framebufQ2ColorLF_SH_Sampled, framebufQ2ColorLF_COCG_Sampled, ipos);
    float3 color_curr_hf = q2UnpackRGBE(framebufQ2ColorHF_Sampled.Load(int3(ipos, 0)).r);
    float3 color_curr_spec = q2UnpackRGBE(framebufQ2ColorSpec_Sampled.Load(int3(ipos, 0)).r);

    Q2SH out_color_lf;
    float3 out_color_hf;
    float4 out_color_histlen_spec;
    float4 out_moments_histlen_hf;

    // Load the gradients (produced by the Q2RTX-style gradient pipeline at
    // 1/3 res: CmQ2GradientReproject + Img + Atrous). After the 7 atrous
    // iterations the LF result is in Q2GradLFPong and the HF/SPEC result
    // (after 3 iterations) in Q2GradHFSpecPong. Clamped like Q2RTX.
    float grad_lf = framebufQ2GradLFPong_Sampled.Load(int3(ipos / Q2_GRAD_DWN, 0)).x;
    float2 grad_hf_spec = framebufQ2GradHFSpecPong_Sampled.Load(int3(ipos / Q2_GRAD_DWN, 0)).xy;
    grad_lf = clamp(grad_lf, 0.0, 1.0);
    grad_hf_spec = clamp(grad_hf_spec, (float2)0.0, (float2)1.0);

    if(temporal_sample_valid_diff)
    {
        float antilag_alpha_lf = clamp(lerp(1.0, Q2_FLT_ANTILAG_LF * grad_lf, Q2_FLT_TEMPORAL_LF), 0, 1);
        float antilag_alpha_hf = clamp(lerp(1.0, Q2_FLT_ANTILAG_HF * grad_hf_spec.x, Q2_FLT_TEMPORAL_HF), 0, 1);

        // Adjust the history length, taking the antilag factors into account
        float hist_len_hf = min(temporal_moments_histlen_hf.b * pow(1.0 - antilag_alpha_hf, 10) + 1.0, 256.0);
        float hist_len_lf = min(temporal_moments_histlen_hf.a * pow(1.0 - antilag_alpha_lf, 10) + 1.0, 256.0);

        // Compute the blending weights based on history length, so that the filter
        // converges faster (first frame weight 1.0, second 1/2, third 1/3, ...).
        float alpha_color_lf = max(Q2_FLT_MIN_ALPHA_COLOR_LF, 1.0 / hist_len_lf);
        float alpha_color_hf = max(Q2_FLT_MIN_ALPHA_COLOR_HF, 1.0 / hist_len_hf);
        float alpha_moments_hf = max(Q2_FLT_MIN_ALPHA_MOMENTS_HF, 1.0 / hist_len_hf);

        // Adjust the blending factors, taking the antilag factors into account again
        alpha_color_lf = lerp(alpha_color_lf, 1.0, antilag_alpha_lf);
        alpha_color_hf = lerp(alpha_color_hf, 1.0, antilag_alpha_hf);
        alpha_moments_hf = lerp(alpha_moments_hf, 1.0, antilag_alpha_hf);

        out_color_lf = q2MixSH(temporal_color_lf, color_curr_lf, alpha_color_lf);
        out_color_hf.rgb = lerp(temporal_color_hf.rgb, color_curr_hf.rgb, alpha_color_hf);

        out_moments_histlen_hf.rg = lerp(temporal_moments_histlen_hf.rg, spatial_moments_hf.rg, alpha_moments_hf);
        out_moments_histlen_hf.b = hist_len_hf;
        out_moments_histlen_hf.a = hist_len_lf;
    }
    else
    {
        // No valid history - just use the current color and spatial moments
        out_color_lf = color_curr_lf;
        out_color_hf.rgb = color_curr_hf;
        out_moments_histlen_hf = float4(spatial_moments_hf, 1, 1);
    }

    if(temporal_sample_valid_spec)
    {
        // Same sequence as above, only for the specular channel
        float antilag = grad_hf_spec.y * Q2_FLT_ANTILAG_SPEC + motion_length * Q2_FLT_ANTILAG_SPEC_MOTION;
        float antilag_alpha_spec = clamp(lerp(1.0, antilag, Q2_FLT_TEMPORAL_SPEC), 0, 1);
        float hist_len_spec = min(temporal_color_histlen_spec.a * pow(1.0 - antilag_alpha_spec, 10) + 1.0, 256.0);
        float alpha_color_spec = max(Q2_FLT_MIN_ALPHA_COLOR_SPEC, 1.0 / hist_len_spec);
        alpha_color_spec = lerp(alpha_color_spec, 1.0, antilag_alpha_spec);
        out_color_histlen_spec.rgb = lerp(temporal_color_histlen_spec.rgb, color_curr_spec.rgb, alpha_color_spec);
        out_color_histlen_spec.a = hist_len_spec;
    }
    else
    {
        out_color_histlen_spec = float4(color_curr_spec, 1);
    }

    // Store the outputs for further processing by the a-trous HF filter
    framebufQ2HistMomentsHF[ipos] = out_moments_histlen_hf;
    Q2_STORE_SH(framebufQ2HistColorLF_SH, framebufQ2HistColorLF_COCG, ipos, out_color_lf);
    framebufQ2AtrousPingHF[ipos] = (uint4)q2PackRGBE(out_color_hf);
    framebufQ2AtrousPingSpec[ipos] = (uint4)q2PackRGBE(out_color_histlen_spec.rgb);
    framebufQ2AtrousPingMoments[ipos] = float4(out_moments_histlen_hf.xy, 0, 0);
    framebufQ2FilteredSpec[ipos] = out_color_histlen_spec;

    GroupMemoryBarrierWithGroupSync();

    // Store the LF channel into shared memory for averaging
    s_lf_shy[groupThreadID.y][groupThreadID.x] = out_color_lf.shY;
    s_lf_cocg[groupThreadID.y][groupThreadID.x] = out_color_lf.CoCg;

    if(groupThreadID.x % Q2_GRAD_DWN == 1 && groupThreadID.y % Q2_GRAD_DWN == 1)
    {
        s_depth_width[groupThreadID.y / Q2_GRAD_DWN][groupThreadID.x / Q2_GRAD_DWN] = gradDepth;
    }

    GroupMemoryBarrierWithGroupSync();

    // Compute a 1/3-resolution version of the LF channel for this group.
    // Use a bilateral filter that takes the center pixel of each 3x3 square as the anchor.

    uint2 lowres_local_id;
    lowres_local_id.x = groupIndex % (GROUP_SIZE / Q2_GRAD_DWN);
    lowres_local_id.y = groupIndex / (GROUP_SIZE / Q2_GRAD_DWN);

    if(lowres_local_id.y >= (GROUP_SIZE / Q2_GRAD_DWN))
        return;

    uint2 center_shared_pos = lowres_local_id * Q2_GRAD_DWN + (uint2)1;
    float3 center_normal = float3(unpackHalf2x16(s_normal_lum[center_shared_pos.y + FILTER_RADIUS][center_shared_pos.x + FILTER_RADIUS].x),
                                  unpackHalf2x16(s_normal_lum[center_shared_pos.y + FILTER_RADIUS][center_shared_pos.x + FILTER_RADIUS].y).x);
    float center_depth = s_depth[center_shared_pos.y + FILTER_RADIUS][center_shared_pos.x + FILTER_RADIUS];
    float depth_width = s_depth_width[lowres_local_id.y][lowres_local_id.x];

    Q2SH center_lf;
    center_lf.shY = s_lf_shy[center_shared_pos.y][center_shared_pos.x];
    center_lf.CoCg = s_lf_cocg[center_shared_pos.y][center_shared_pos.x];

    float sum_w = 1;
    Q2SH sum_lf = center_lf;

    // Average the anchor pixel color with the relevant neighborhood
    for(int yy = -1; yy <= 1; yy++)
    {
        for(int xx = -1; xx <= 1; xx++)
        {
            if(yy == 0 && xx == 0)
                continue;

            float3 p_normal = float3(unpackHalf2x16(s_normal_lum[center_shared_pos.y + FILTER_RADIUS + yy][center_shared_pos.x + FILTER_RADIUS + xx].x),
                                     unpackHalf2x16(s_normal_lum[center_shared_pos.y + FILTER_RADIUS + yy][center_shared_pos.x + FILTER_RADIUS + xx].y).x);
            float p_depth = s_depth[center_shared_pos.y + FILTER_RADIUS + yy][center_shared_pos.x + FILTER_RADIUS + xx];

            float dist_depth = abs(p_depth - center_depth) * depth_width;
            if(dist_depth < 2)
            {
                float w = pow(max(dot(p_normal, center_normal), 0), 8);

                Q2SH p_lf;
                p_lf.shY = s_lf_shy[center_shared_pos.y + yy][center_shared_pos.x + xx];
                p_lf.CoCg = s_lf_cocg[center_shared_pos.y + yy][center_shared_pos.x + xx];

                q2AccumulateSH(sum_lf, p_lf, w);
                sum_w += w;
            }
        }
    }

    float inv_w = 1.0 / sum_w;
    sum_lf.shY  *= inv_w;
    sum_lf.CoCg *= inv_w;

    // Store the LF result for further processing by the a-trous LF filter
    int2 ipos_lowres = int2(groupID.xy) * (GROUP_SIZE / Q2_GRAD_DWN) + int2(lowres_local_id);

    Q2_STORE_SH(framebufQ2AtrousPingLF_SH, framebufQ2AtrousPingLF_COCG, ipos_lowres, sum_lf);
}
