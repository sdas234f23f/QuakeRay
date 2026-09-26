// Copyright (C) 2018 Christoph Schied
// Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) 2026 QuakeRay contributors
//
// This file is a port of shader/asvgf_gradient_img.comp from Quake 2 RTX (https://github.com/NVIDIA/Q2RTX),
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
// Ported from Q2RTX (GPL v2) shader/asvgf_gradient_img.comp, adapted to the
// vkpt framework. Computes the HF/SPEC gradients (squared relative luminance
// difference between the current and the reprojected previous frame) and the
// LF gradient (relative luminance difference of the temporally accumulated LF
// history over a large region).
//
// HLSL counterpart of CmQ2GradientImg.comp. Pinned by the pair
// GLSL/CmQ2GradientImg.comp <-> this file, which pulls in Q2Asvgf.hlsli and pins its
// q2CheckerToFlat / q2FlatToChecker / q2GetGradient / q2UnpackRGBE through this pass.
//
// Spellings that had to change:
//   * layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in
//     -> [numthreads(16, 16, 1)] on main, and gl_GlobalInvocationID -> SV_DispatchThreadID
//   * ivec2/vec2/vec3/vec4 -> int2/float2/float3/float4
//   * texelFetch(t, p, 0) -> t.Load(int3(p, 0)): the motion, the LF color and the stratum
//     position are sampled views of ShaderCommonHLSL.hlsli, and the two imageStore targets
//     become storage image writes framebufQ2GradLFPing[ipos] and framebufQ2GradHFSpecPing[ipos].
//     The previous HF/SPEC luminance is read back through the second of those storage images
//     instead of its sampled view, because this shader binds that image in both roles (A4.5,
//     see the read's own comment)
//   * vec2(0.5) / vec2(0) -> (float2)0.5 / (float2)0: dxc refuses to widen a single scalar
//     through a constructor, and the same goes for the float(...) of the gradient type
//   * float2(ipos_flat) and int2(floor(...)) keep the golden's conversion-constructor
//     spelling, which dxc accepts for a vector argument of a different type
//   * uint(x) -> (uint)x in the two shifts of the stratum extraction; Q2_STRATUM_OFFSET_SHIFT,
//     Q2_STRATUM_OFFSET_MASK, Q2_GRAD_DWN and the q2GetGradient / getLuminance helpers all
//     keep their names, and u != 0u keeps its spelling as legal HLSL
//   * imageStore(framebufQ2GradLFPing, ipos, vec4(grad_lf, 0, 0)) -> the same float4 built by
//     the float4 constructor and stored through the subscript
//
// What did not change: the reprojection arithmetic of q2GetLFGradient with its
// sqrt-free checkerboard conversions and the out-of-range early out, the six-tap
// accumulation loop over Q2_GRAD_DWN squared with its literal indices, the grad_width /
// grad_height bounds, the 1.0 / renderWidth reciprocals, the stratum shift chain and the
// q2GetGradient calls on the two luminances; the returned (lum_curr, lum_prev) pair and the
// grad_lf, grad_hf, grad_spec accumulators of the golden.

#define DESC_SET_FRAMEBUFFERS 0
#define DESC_SET_GLOBAL_UNIFORM 1
#include "ShaderCommonHLSLFunc.hlsli"
#include "Q2Asvgf.hlsli"

// LF gradient is computed from the temporally accumulated history over a large
// screen region, which continuously detects invalid history and sudden flashes
// (the raw LF signal is too sparse for the per-sample HF approach).
float2 q2GetLFGradient(int2 ipos, int width, int height, float inv_width, float inv_height)
{
    float4 motion = framebufMotion_Sampled.Load(int3(ipos, 0));
    int2 ipos_flat = q2CheckerToFlat(ipos, width);
    float2 pos_prev_flat = (float2(ipos_flat) + (float2)0.5) * float2(inv_width, inv_height) + motion.xy;
    int2 pp_flat = int2(floor(pos_prev_flat * float2(width, height)));

    if (pp_flat.x < 0 || pp_flat.x >= width || pp_flat.y < 0 || pp_flat.y >= height)
    {
        return (float2)0;
    }

    int2 pp = q2FlatToChecker(pp_flat, width);

    float lum_curr = framebufQ2ColorLF_SH_Sampled.Load(int3(ipos, 0)).w;
    float lum_prev = framebufQ2HistColorLF_SH_Prev_Sampled.Load(int3(pp, 0)).w;

    // Return raw colors; normalization happens after the blur (gradient atrous).
    return float2(lum_curr, lum_prev);
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int2 ipos = int2(dispatchThreadID.xy);

    const int width = (int)globalUniform.renderWidth;
    const int height = (int)globalUniform.renderHeight;
    const int grad_width = width / Q2_GRAD_DWN;
    const int grad_height = height / Q2_GRAD_DWN;

    if (ipos.x >= grad_width || ipos.y >= grad_height)
    {
        return;
    }

    const float inv_width = 1.0 / globalUniform.renderWidth;
    const float inv_height = 1.0 / globalUniform.renderHeight;

    uint u = framebufQ2GradSmplPos_Sampled.Load(int3(ipos, 0)).r;

    float2 grad_lf = (float2)0;
    float grad_hf = 0;
    float grad_spec = 0;

    // Process the reprojected HF/SPEC sample
    if (u != 0u)
    {
        int2 grad_strata_pos = int2(
            (u >> (Q2_STRATUM_OFFSET_SHIFT * 0)) & Q2_STRATUM_OFFSET_MASK,
            (u >> (Q2_STRATUM_OFFSET_SHIFT * 1)) & Q2_STRATUM_OFFSET_MASK);

        int2 grad_sample_pos_curr = ipos * Q2_GRAD_DWN + grad_strata_pos;

        // Read the previous HF/SPEC luminance through the storage image, not the sampled
        // view: this shader also writes framebufQ2GradHFSpecPing, and binding both views of
        // one image in one set makes the SRV and the UAV disagree about the image layout
        // (A4.5). The invocation that reads this texel is the one that writes it at the end
        // of main, so the read-then-write is a same-thread access to the same texel; the
        // image is rg16f, so .xy are the same raw halfs the sampled view returned.
        float2 prev_hf_spec_lum = framebufQ2GradHFSpecPing.Load(ipos).xy;

        float3 curr_hf = q2UnpackRGBE(framebufQ2ColorHF_Sampled.Load(int3(grad_sample_pos_curr, 0)).r);
        float3 curr_spec = q2UnpackRGBE(framebufQ2ColorSpec_Sampled.Load(int3(grad_sample_pos_curr, 0)).r);

        grad_hf = q2GetGradient(getLuminance(curr_hf), prev_hf_spec_lum.x);
        grad_spec = q2GetGradient(getLuminance(curr_spec), prev_hf_spec_lum.y);
    }

    // Accumulate the LF luminances over the whole 3x3 square
    for (int yy = 0; yy < Q2_GRAD_DWN; yy++)
    {
        for (int xx = 0; xx < Q2_GRAD_DWN; xx++)
        {
            grad_lf += q2GetLFGradient(ipos * Q2_GRAD_DWN + int2(xx, yy), width, height, inv_width, inv_height);
        }
    }

    framebufQ2GradLFPing[ipos] = float4(grad_lf, 0, 0);
    framebufQ2GradHFSpecPing[ipos] = float4(grad_hf, grad_spec, 0, 0);
}
