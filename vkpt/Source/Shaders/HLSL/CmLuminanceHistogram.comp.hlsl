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


// HLSL counterpart of CmLuminanceHistogram.comp.
//
// Spellings that had to change:
//   * gl_GlobalInvocationID -> SV_DispatchThreadID and gl_LocalInvocationIndex -> SV_GroupIndex.
//     The second one is the flattened index of an invocation inside its group on both sides,
//     SPIR-V calls it LocalInvocationIndex, and the golden's ivec2(gl_GlobalInvocationID)
//     becomes int2(dispatchThreadID.xy)
//   * shared uint s_Histogram[N] -> groupshared uint s_Histogram[N], barrier() ->
//     GroupMemoryBarrierWithGroupSync()
//   * texelFetch(framebufPreFinal_Sampled, ipos, 0) -> framebufPreFinal_Sampled.Load(int3(ipos, 0))
//   * vec3(0.0) and vec2(0.5) -> (float3)0.0 and (float2)0.5: HLSL has no one argument vector
//     constructor, and its cast binds tighter than a binary operator
//   * uint(x) / int(x) -> (uint)x / (int)x, C style casts that truncate toward zero exactly as
//     the GLSL constructors do. The golden's uint(right_weight_F * FIXED_POINT_FRAC_MULTIPLIER)
//     converts the product, so the port keeps the parentheses: (uint)(right_weight_F * ...)
//   * fract(x) -> frac(x): the x - floor(x) intrinsic is spelled differently in HLSL, both
//     compile to GLSL.std.450 Fract
//   * any(greaterThanEqual(ipos, screenSize)) -> any(ipos >= screenSize): HLSL compares vectors
//     with the operators and hands back a bool vector for any()
//   * atomicAdd -> InterlockedAdd, as in Q2LightLists.hlsli: HLSL has no atomicAdd, and the two
//     argument InterlockedAdd is the same atomic add with the return value discarded. The GLSL
//     block has a single instance, so its tonemapping.histogram[i] is tonemapping[0].histogram[i]
//
// What did not change: every operand order, every clamp, max, length, log2, exp2 and the
// ternary of the golden, the shared histogram and the two barriers around it, and the
// fixed-point conversion, which still multiplies in float and converts afterwards. Every
// arithmetic site was folded with literal operands on both halves and the two sides agree,
// the control of each site folds to a different number.
//
// Differences that are not source differences, measured on the compiled pair:
//   * the port recomputes `validThread && linear_idx < HISTOGRAM_BINS` in the last block where
//     the golden reuses the value its optimizer commoned up out of the first block. The census
//     at a common fixpoint therefore reads one OpLogicalAnd less and two OpPhi plus one
//     OpULessThan more on the port. Same value, liveness only
//   * the two extra OpVectorShuffle on the port are the .xy of the dispatch id and its int2
//     conversion; glslang reaches the same components through OpBitcast and OpCompositeExtract
//   * the golden's `max` arrives as GLSL.std.450 FMax and the port's as NMax. Both return the
//     larger operand, and no NaN can reach the site: the guard `getLuminance(input_color) > 0.0`
//     is false for a NaN, so the whole block including the max is skipped
//   * glslang gives the workgroup atomicAdd device scope where dxc gives it workgroup scope.
//     A wider scope only makes the same add exclusive against more invocations, so the counter
//     value after the barrier is the same
//
// The port is pinned by the pair GLSL/CmLuminanceHistogram.comp <-> this file, which pulls in
// TonemappingUtils.hlsli and pins that header as well.

#define DESC_SET_FRAMEBUFFERS 0
#define DESC_SET_GLOBAL_UNIFORM 1
#define DESC_SET_TONEMAPPING 2
#define TONEMAPPING_BUFFER_WRITEABLE
#include "ShaderCommonHLSLFunc.hlsli"
#include "TonemappingUtils.hlsli"

// Q2RTX-style saliency-weighted histogram over log2 luminance (photographic
// stops) of the pre-final HDR image. Values are accumulated in fixed-point
// so that atomic additions don't lose precision.
// Ported from Quake 2 RTX (tone_mapping_histogram.comp), GPL v2.

groupshared uint s_Histogram[HISTOGRAM_BINS];

[numthreads(COMPUTE_LUM_HISTOGRAM_GROUP_SIZE_X, COMPUTE_LUM_HISTOGRAM_GROUP_SIZE_Y, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint linear_idx : SV_GroupIndex)
{
    const int2 ipos = int2(dispatchThreadID.xy);
    const int2 screenSize = int2(globalUniform.renderWidth, globalUniform.renderHeight);

    const bool validThread = !any(ipos >= screenSize);

    const float3 input_color = validThread ? framebufPreFinal_Sampled.Load(int3(ipos, 0)).rgb : (float3)0.0;

    // init shared histogram
    if (validThread && linear_idx < HISTOGRAM_BINS)
    {
        s_Histogram[linear_idx] = 0;
    }

    GroupMemoryBarrierWithGroupSync();

    // Ignore completely black pixels
    if (validThread && getLuminance(input_color) > 0.0)
    {
        const float lum = max(getLuminance(input_color), exp2(min_log_luminance));
        const float biased_log_luminance = log2(lum) * log_luminance_scale + log_luminance_bias;
        const float histogram_bin = clamp(biased_log_luminance * HISTOGRAM_BINS, 0.0, HISTOGRAM_BINS - 1.0);

        // Distribute luminance between two bins using a linear (tent) filter
        const uint left_bin = (uint)histogram_bin;
        const uint right_bin = left_bin + 1;

        // Pixel importance based on distance to the center of the screen
        const float weight = clamp(1.0 - length((float2)ipos / (float2)screenSize - (float2)0.5) * 1.5, 0.01, 1.0);

        const float right_weight_F = frac(histogram_bin) * weight;
        const float left_weight_F = weight - right_weight_F;

        // fixed-point for atomic addition
        const uint right_weight_U = (uint)(right_weight_F * FIXED_POINT_FRAC_MULTIPLIER);
        const uint left_weight_U = (uint)(left_weight_F * FIXED_POINT_FRAC_MULTIPLIER);

        InterlockedAdd(s_Histogram[left_bin], left_weight_U);
        if (right_bin < HISTOGRAM_BINS)
        {
            InterlockedAdd(s_Histogram[right_bin], right_weight_U);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // add workgroup histogram to the global histogram
    if (validThread && linear_idx < HISTOGRAM_BINS)
    {
        const int localBinValue = (int)s_Histogram[linear_idx];
        if (localBinValue != 0)
        {
            InterlockedAdd(tonemapping[0].histogram[linear_idx], localBinValue);
        }
    }
}
