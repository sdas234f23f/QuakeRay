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


// HLSL counterpart of CmLuminanceAvg.comp.
//
// Spellings that had to change:
//   * gl_GlobalInvocationID -> SV_DispatchThreadID and gl_LocalInvocationIndex -> SV_GroupIndex,
//     the flattened invocation index inside its group, exactly as in CmLuminanceHistogram.comp.hlsl;
//     the golden's ivec2(gl_GlobalInvocationID) becomes int2(dispatchThreadID.xy) and the bounds
//     test any(greaterThanEqual(ipos, screenSize)) becomes any(ipos >= screenSize)
//   * layout(local_size_x = COMPUTE_LUM_HISTOGRAM_BIN_COUNT, local_size_y = 1, local_size_z = 1) in
//     -> [numthreads(COMPUTE_LUM_HISTOGRAM_BIN_COUNT, 1, 1)]: the host dispatches one group of
//     COMPUTE_LUM_HISTOGRAM_BIN_COUNT invocations for this stage (Tonemapping.cpp:208)
//   * shared float s_Shared[HISTOGRAM_BINS] -> groupshared float s_Shared[HISTOGRAM_BINS] and
//     barrier() -> GroupMemoryBarrierWithGroupSync(), both in the three reduction helpers and in
//     main, in the golden's order
//   * the golden's writeable buffer block holds a single instance, so its `tonemapping.<field>`
//     becomes `tonemapping[0].<field>`; ShaderCommonHLSLFunc.hlsli declares it as
//     RWStructuredBuffer<ShTonemapping> behind TONEMAPPING_BUFFER_WRITEABLE, which this file
//     defines before the include exactly as the golden defines it before ShaderCommonGLSLFunc.h
//   * mix -> lerp, float(x)/uint(x)/int(x) -> (float)x/(uint)x/(int)x, and the golden's
//     parentheses stay around the converted operand, because a GLSL constructor converts its whole
//     argument: (float)(i * i), (int)noise_stop_bin and (float)(noise_stop_bin - 1). The implicit
//     int-to-float conversion of smoothstep's third argument is written out as (float)linear_idx,
//     which dxc accepts in both spellings and converts to the same value
//   * #version 460 and GL_EXT_control_flow_attributes disappear: no attribute of that extension is
//     used by the golden, and HLSL has no version directive
//
// What did not change: every operand order and every association of the golden, the three shared
// reduction helpers, the 16-iteration threshold loop, the 14-tap Gaussian kernel with its abs(dx)
// index and its clamp to [0, HISTOGRAM_BINS - 1], the step and smoothstep argument orders, the
// whole auto-exposure block and the noise-aware tone curve.
//
// Evidence, measured on the compiled pair (glslc -O against dxc, both disassembled):
//   * 1.0 + (float)3u / FIXED_POINT_FRAC_MULTIPLIER folds to 1.0234375 on both halves; the
//     control 1.0 + (float)(3u / FIXED_POINT_FRAC_MULTIPLIER) folds to 1
//   * the bin log luminance ((float)37 / (float)HISTOGRAM_BINS) * (max_log_luminance -
//     min_log_luminance) + min_log_luminance folds to -14.75 on both; the control with the
//     subtraction swapped folds to -33.25
//   * (float)(max_log_luminance - min_log_luminance) / HISTOGRAM_BINS folds to 0.25 on both, its
//     swapped control to -0.25
//   * clamp(((-6.0) * log_luminance_scale + log_luminance_bias) * HISTOGRAM_BINS, 0.0,
//     HISTOGRAM_BINS - 1.0) folds to 72 on both; the association control
//     clamp((-6.0) * log_luminance_scale + log_luminance_bias * HISTOGRAM_BINS, ...) to 95.8125
//   * (int)0.5 - 1 folds to -1 on both, its control (int)(0.5 - 1.0) to 0
//   * exp(-(float)(3 * 3) / (2.0 * 2.0 * 2.0)) folds to 0.324652463 on both, exp2(-6.0) to
//     0.015625, 1.5 / 1.3 to 1.15384614, 1.5 / max(0.0001, 1.3) to 1.15384614 and
//     (2.5 - 1.25) * 0.25 - 3.0 to -2.6875; every control (sign, operand order, removed
//     parenthesis) folds away from its site
//   * mix -> lerp: the shipped blobs hold the same four GLSL.std.450 FMix instructions with the
//     same operand order (measured with runtime operands), so no runtime value can differ. With
//     literal operands the two constant folders disagree in the last ulp: dxc's
//     lerp(0.3, 0.7, 0.25) is 0.399999976 against 0.400000006 from spirv-opt, which folds the
//     GLSL definition x*(1 - a) + y*a where dxc folds the HLSL definition x + s*(y - x); the
//     swapped-operand control folds to 0.599999964, so the site separates on both halves
//   * step, smoothstep and int abs do not fold on the dxc side: both halves emit Step, SmoothStep
//     and ConvertFToS + SAbs + ConvertSToF with the same operand order, so those sites are argued
//     at operand level, not measured
//   * division by FIXED_POINT_FRAC_MULTIPLIER = 128: dxc writes OpFMul by 0.0078125 where glslang
//     writes OpFDiv by 128.0, and glslc -O reaches the same multiply; 1/128 = 2^-7 is exact, so
//     the rewrite cannot move a value
//   * max -> the golden's FMax against the port's NMax, as in CmLuminanceHistogram.comp.hlsl: for
//     the two operands of this file (a fixed 0.0001 and a sum of finite bin weights) both return
//     the larger value
//
// Census of the shipped blobs: identical counts of every GLSL.std.450 instruction except the
// FMax/NMax pair, identical OpControlBarrier (31) and OpLoopMerge (14), identical arithmetic
// counters (OpFAdd 16, OpFMul 23, OpFDiv 8, OpFSub 9, OpFNegate 4, OpIAdd 13, OpISub 3) and the
// same float constants with the same multiplicities. The one structural difference: dxc lowers
// the golden's (lower_limit <= histogram_cdf) && (histogram_cdf_prev <= upper_limit) as two
// short-circuit selection regions where glslang emitted a single OpLogicalAnd; both operands are
// pure comparisons, so the two halves decide the same branch.
//
// The port is pinned by the pair GLSL/CmLuminanceAvg.comp <-> this file, which also pins the
// shared TonemappingUtils.hlsli and the ShTonemapping layout of the histogram buffer.

#define DESC_SET_FRAMEBUFFERS 0
#define DESC_SET_GLOBAL_UNIFORM 1
#define DESC_SET_TONEMAPPING 2
#define TONEMAPPING_BUFFER_WRITEABLE
#include "ShaderCommonHLSLFunc.hlsli"
#include "TonemappingUtils.hlsli"

// 128x1x1 dispatch, one invocation per histogram bin.

// Q2RTX-style noise-aware tone curve + auto-exposure computation.
// Ported from Quake 2 RTX (tone_mapping_curve.comp), GPL v2.

groupshared float s_Shared[HISTOGRAM_BINS];

// Shared-memory reductions keyed towards HISTOGRAM_BINS x 1 x 1 dispatches.

float computeSharedSum(float val, const uint linear_idx)
{
    s_Shared[linear_idx] = val;
    GroupMemoryBarrierWithGroupSync();
    for (uint k = 64; k >= 1; k /= 2)
    {
        if (linear_idx < k)
        {
            s_Shared[linear_idx] += s_Shared[linear_idx + k];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    val = s_Shared[0];
    GroupMemoryBarrierWithGroupSync();
    return val;
}

float computeSharedMax(float val, const uint linear_idx)
{
    s_Shared[linear_idx] = val;
    GroupMemoryBarrierWithGroupSync();
    for (uint k = 64; k >= 1; k /= 2)
    {
        if (linear_idx < k)
        {
            s_Shared[linear_idx] = max(s_Shared[linear_idx], s_Shared[linear_idx + k]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    val = s_Shared[0];
    GroupMemoryBarrierWithGroupSync();
    return val;
}

float computePrefixSum(float val, const uint linear_idx)
{
    s_Shared[linear_idx] = val;
    GroupMemoryBarrierWithGroupSync();
    for (uint k = 1; k < 128; k *= 2)
    {
        const uint block_idx = linear_idx / k;
        if ((block_idx % 2) == 1)
        {
            s_Shared[linear_idx] += s_Shared[k * block_idx - 1];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    val = s_Shared[linear_idx];
    GroupMemoryBarrierWithGroupSync();
    return val;
}

[numthreads(COMPUTE_LUM_HISTOGRAM_BIN_COUNT, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint localInvocationIndex : SV_GroupIndex)
{
    const int2 ipos = int2(dispatchThreadID.xy);
    if (any(ipos >= int2(globalUniform.renderWidth, globalUniform.renderHeight)))
        return;

    const int linear_idx = (int)localInvocationIndex;

    // Read histogram and normalize. Add bias to avoid an all-zero histogram.
    float original_hist = 1.0 + (float)tonemapping[0].histogram[linear_idx] / FIXED_POINT_FRAC_MULTIPLIER;
    const float hist_sum = computeSharedSum(original_hist, linear_idx);
    const float hist_max = computeSharedMax(original_hist, linear_idx);

    tonemapping[0].normalized[linear_idx] = original_hist / hist_max;

    original_hist /= hist_sum;

    // Log luminance of the bin this invocation is processing.
    const float bin_log_luminance = ((float)linear_idx / (float)HISTOGRAM_BINS) * (max_log_luminance - min_log_luminance) + min_log_luminance;

    // CDF-based estimate of the scene luminance (for eye adaptation).
    const float histogram_cdf = computePrefixSum(original_hist, linear_idx);
    const float histogram_cdf_prev = histogram_cdf - original_hist;

    const float lower_limit = tonemapping[0].tmLowPercentile * 0.01;
    const float upper_limit = tonemapping[0].tmHighPercentile * 0.01;

    float weight_sum = 0.0;
    float bin_sum = 0.0;

    if ((lower_limit <= histogram_cdf) && (histogram_cdf_prev <= upper_limit))
    {
        weight_sum = bin_log_luminance * original_hist;
        bin_sum = original_hist;
    }

    weight_sum = computeSharedSum(weight_sum, linear_idx);
    bin_sum = computeSharedSum(bin_sum, linear_idx);

    float log_target_lum = weight_sum / max(0.0001, bin_sum);
    log_target_lum = clamp(log_target_lum, log2(tonemapping[0].tmMinLuminance), log2(tonemapping[0].tmMaxLuminance));

    // Blend the luminance estimate over time (eye adaptation).
    if (tonemapping[0].resetCurve == 0)
    {
        float log_old_lum = tonemapping[0].adaptedLuminance;
        if (log_old_lum > 0.0)
        {
            log_old_lum = log2(log_old_lum);
        }

        const float speed = (log_old_lum < log_target_lum) ? tonemapping[0].tmExposureSpeedUp : tonemapping[0].tmExposureSpeedDown;
        log_target_lum = lerp(log_target_lum, log_old_lum, exp(-tonemapping[0].frameTime * speed));
    }

    if (linear_idx == 0)
    {
        const float adapted_luminance = exp2(log_target_lum);
        tonemapping[0].adaptedLuminance = adapted_luminance;
        // keep legacy field in sync (used by Exposure.h / RsWorld.frag)
        tonemapping[0].avgLuminance = adapted_luminance;
    }

    // Noise-aware tone mapping curve (Eilertsen, Mantiuk, Unger + NVIDIA mods).

    if (bin_log_luminance < tonemapping[0].tmNoiseStops)
        original_hist = 0;

    // Dynamic range of the display in log2 stops.
    const float r = tonemapping[0].tmDynRangeStops;
    const float delta = (float)(max_log_luminance - min_log_luminance) / HISTOGRAM_BINS;
    const float r_over_delta = r / delta;

    // Threshold so that slopes are never negative (iterative method, eqn. 17).
    float rcp_hist = (original_hist > 0.0 ? 1.0 / original_hist : 0.0);
    float thresh = 1e-16;
    float sum_recip;
    float len_omega;
    float thresh_passed;
    for (uint i = 0; i < 16; ++i)
    {
        thresh_passed = step(thresh, original_hist);
        len_omega = computeSharedSum(thresh_passed, linear_idx);
        sum_recip = computeSharedSum(rcp_hist * thresh_passed, linear_idx);
        thresh = (len_omega - r_over_delta) / sum_recip;
    }

    // Slopes from Equation (14).
    thresh_passed = step(thresh, original_hist);
    len_omega = computeSharedSum(thresh_passed, linear_idx);
    sum_recip = computeSharedSum(rcp_hist * thresh_passed, linear_idx);
    float my_slope = (1.0 + rcp_hist * (r_over_delta - len_omega) / sum_recip) * thresh_passed;

    // Blur slopes with a symmetric Gaussian kernel (14 half-weights from sigma).
    float gaussian_sum = 0.0;
    float weights[14];
    for (int i = 0; i < 14; i++)
    {
        const float kernel_value = exp(-(float)(i * i) / (2.0 * tonemapping[0].tmSlopeBlurSigma * tonemapping[0].tmSlopeBlurSigma));
        gaussian_sum += kernel_value * (i == 0 ? 1.0 : 2.0);
        weights[i] = kernel_value;
    }
    for (int i = 0; i < 14; i++)
    {
        weights[i] /= gaussian_sum;
    }

    s_Shared[linear_idx] = my_slope;
    GroupMemoryBarrierWithGroupSync();
    my_slope *= weights[0];
    for (int dx = -13; dx <= 13; dx++)
    {
        if (dx != 0)
        {
            my_slope += weights[abs(dx)] * s_Shared[clamp(linear_idx + dx, 0, HISTOGRAM_BINS - 1)];
        }
    }

    // Turn slopes into an inclusive prefix sum, then an exclusive, scaled one.
    float my_tonecurve = computePrefixSum(my_slope, linear_idx);
    my_tonecurve = (my_tonecurve - my_slope) * delta - r;

    // Interpolate values below tm_noise_stops toward the auto-exposure value.
    const float noise_stop_bin = clamp((tonemapping[0].tmNoiseStops * log_luminance_scale + log_luminance_bias) * HISTOGRAM_BINS, 0.0, HISTOGRAM_BINS - 1.0);
    if (linear_idx < noise_stop_bin)
    {
        // s_Shared still holds the prefix sums from computePrefixSum()
        const float my_tonecurve_at_ns = s_Shared[(int)noise_stop_bin - 1] * delta - r;
        const float bin_log_luminance_at_ns = ((float)(noise_stop_bin - 1) / (float)HISTOGRAM_BINS) * (max_log_luminance - min_log_luminance) + min_log_luminance;
        const float fudge = -(my_tonecurve_at_ns - bin_log_luminance_at_ns) / log_target_lum;

        const float tone_curve_ae = bin_log_luminance - log_target_lum * fudge;
        my_tonecurve = lerp(tone_curve_ae, my_tonecurve, lerp(smoothstep(0.5 * noise_stop_bin, noise_stop_bin, (float)linear_idx), 1.0, tonemapping[0].tmNoiseBlend));
    }

    // Blend the curve over time (eye adaptation).
    if (tonemapping[0].resetCurve == 0)
    {
        const float my_old_tonecurve = tonemapping[0].curve[linear_idx];

        const float blend_speed = (my_old_tonecurve < my_tonecurve) ? tonemapping[0].tmExposureSpeedUp : tonemapping[0].tmExposureSpeedDown;

        my_tonecurve = lerp(my_tonecurve, my_old_tonecurve, exp(-tonemapping[0].frameTime * blend_speed));
    }

    tonemapping[0].curve[linear_idx] = my_tonecurve;

    // Clear the histogram accumulator for the next frame.
    tonemapping[0].histogram[linear_idx] = 0;
}
