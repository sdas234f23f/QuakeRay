// Copyright (C) 2018 Christoph Schied
// Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) 2026 QuakeRay contributors
//
// This file is a port of shader/asvgf_gradient_atrous.comp from Quake 2 RTX (https://github.com/NVIDIA/Q2RTX),
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
// Ported from Q2RTX (GPL v2) shader/asvgf_gradient_atrous.comp, adapted to the
// vkpt framework. A-trous wavelet filter for the lighting gradients. 7
// iterations: LF is blurred in all 7 (normalized in the last one), HF/SPEC are
// blurred in the first 3. Ping/pong over the Q2Grad* buffers.
//
// HLSL counterpart of CmQ2GradientAtrous.comp. Pinned by the pair
// GLSL/CmQ2GradientAtrous.comp <-> this file, which pulls in Q2Asvgf.hlsli and pins
// q2GetGradient, Q2_GRAD_DWN and Q2_WAVELET_KERNEL through this pass.
//
// Spellings that had to change:
//   * layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in
//     -> [numthreads(16, 16, 1)] on main
//   * gl_GlobalInvocationID -> SV_DispatchThreadID. The golden reads the builtin in both main
//     and q2FilterGradientImage, where the filter re-derives the very same value; HLSL has no
//     builtin that a helper can read, so main converts it once into an int2 and passes it as the
//     second parameter of q2FilterGradientImage. The parameter is added, the two derived values
//     are the golden's
//   * layout(push_constant, std140) uniform IterationInfo { uint iteration; } push
//     -> struct IterationInfo_BT with the single uint and
//     [[vk::push_constant]] ConstantBuffer<IterationInfo_BT> push, which lands the member on
//     offset 0 on both halves
//   * ivec2/vec2/vec4 -> int2/float2/float4; vec2(0) -> (float2)0, because HLSL has no one
//     argument vector constructor, and vec4(filtered_lf, 0, 0) keeps its shape with the renamed
//     type
//   * texture2D -> Texture2D<float4>, texelFetch(img, p, 0) -> img.Load(int3(p, 0)) and
//     imageStore(img, p, v) -> img[p] = v
//   * any(greaterThanEqual(p, grad_size)) -> any(p >= grad_size), the same comparison handed back
//     as a bool vector by the HLSL operators
//   * int(1u << push.iteration) -> (int)(1u << push.iteration), a C style cast that truncates
//     exactly as the GLSL constructor does
//
// What did not change: the whole body and the order of the two functions, the
// (int)globalUniform.renderWidth / renderHeight divided by Q2_GRAD_DWN of both grad_size
// computations, the 3x3 accumulation with the step size of 2^iteration, the out-of-range tap
// replacement by (float2)0 before the weight is applied, the Q2_WAVELET_KERNEL weight indexed by
// the absolute tap offsets, the sum_color / sum_w result, the seven way switch of the filter and
// the seven way switch of the stores, the ping/pong targets of every iteration and the
// q2GetGradient normalization of the last one.

#define DESC_SET_FRAMEBUFFERS 0
#define DESC_SET_GLOBAL_UNIFORM 1
#include "ShaderCommonHLSLFunc.hlsli"
#include "Q2Asvgf.hlsli"

struct IterationInfo_BT
{
    uint iteration;
};

[[vk::push_constant]] ConstantBuffer<IterationInfo_BT> push;

float2 q2FilterGradientImage(Texture2D<float4> img, const int2 ipos)
{
    int2 grad_size = int2((int)globalUniform.renderWidth, (int)globalUniform.renderHeight) / Q2_GRAD_DWN;

    float2 sum_color = (float2)0;
    float sum_w = 0;

    const int step_size = (int)(1u << push.iteration);

    const int r = 1;
    for (int yy = -r; yy <= r; yy++)
    {
        for (int xx = -r; xx <= r; xx++)
        {
            int2 p = ipos + int2(xx, yy) * step_size;

            float2 c = img.Load(int3(p, 0)).xy;

            if (any(p >= grad_size))
            {
                c = (float2)0;
            }

            float w = Q2_WAVELET_KERNEL[abs(xx)][abs(yy)];

            sum_color += c * w;
            sum_w += w;
        }
    }

    return sum_color / sum_w;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int2 ipos = int2(dispatchThreadID.xy);
    int2 grad_size = int2((int)globalUniform.renderWidth, (int)globalUniform.renderHeight) / Q2_GRAD_DWN;
    if (any(ipos >= grad_size))
    {
        return;
    }

    float2 filtered_lf = (float2)0;
    float2 filtered_hf_spec = (float2)0;
    switch (push.iteration)
    {
        case 0:
            filtered_lf = q2FilterGradientImage(framebufQ2GradLFPing_Sampled, ipos);
            filtered_hf_spec = q2FilterGradientImage(framebufQ2GradHFSpecPing_Sampled, ipos);
            break;
        case 1:
            filtered_lf = q2FilterGradientImage(framebufQ2GradLFPong_Sampled, ipos);
            filtered_hf_spec = q2FilterGradientImage(framebufQ2GradHFSpecPong_Sampled, ipos);
            break;
        case 2:
            filtered_lf = q2FilterGradientImage(framebufQ2GradLFPing_Sampled, ipos);
            filtered_hf_spec = q2FilterGradientImage(framebufQ2GradHFSpecPing_Sampled, ipos);
            break;
        case 3: filtered_lf = q2FilterGradientImage(framebufQ2GradLFPong_Sampled, ipos); break;
        case 4: filtered_lf = q2FilterGradientImage(framebufQ2GradLFPing_Sampled, ipos); break;
        case 5: filtered_lf = q2FilterGradientImage(framebufQ2GradLFPong_Sampled, ipos); break;
        case 6:
            filtered_lf = q2FilterGradientImage(framebufQ2GradLFPing_Sampled, ipos);
            // LF gradients are not normalized in the img shader - do it now
            filtered_lf.x = q2GetGradient(filtered_lf.x, filtered_lf.y);
            filtered_lf.y = 0;
            break;
    }

    switch (push.iteration)
    {
        case 0:
            framebufQ2GradLFPong[ipos] = float4(filtered_lf, 0, 0);
            framebufQ2GradHFSpecPong[ipos] = float4(filtered_hf_spec, 0, 0);
            break;
        case 1:
            framebufQ2GradLFPing[ipos] = float4(filtered_lf, 0, 0);
            framebufQ2GradHFSpecPing[ipos] = float4(filtered_hf_spec, 0, 0);
            break;
        case 2:
            framebufQ2GradLFPong[ipos] = float4(filtered_lf, 0, 0);
            framebufQ2GradHFSpecPong[ipos] = float4(filtered_hf_spec, 0, 0);
            break;
        case 3: framebufQ2GradLFPing[ipos] = float4(filtered_lf, 0, 0); break;
        case 4: framebufQ2GradLFPong[ipos] = float4(filtered_lf, 0, 0); break;
        case 5: framebufQ2GradLFPing[ipos] = float4(filtered_lf, 0, 0); break;
        case 6: framebufQ2GradLFPong[ipos] = float4(filtered_lf, 0, 0); break;
    }
}
