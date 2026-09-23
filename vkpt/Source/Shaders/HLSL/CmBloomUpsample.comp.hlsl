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
//
// http://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare
//
// HLSL counterpart of CmBloomUpsample.comp.
//
// Spellings that had to change:
//   * layout(local_size_x = ...) -> [numthreads(COMPUTE_BLOOM_UPSAMPLE_GROUP_SIZE_X,
//     COMPUTE_BLOOM_UPSAMPLE_GROUP_SIZE_Y, 1)] on the entry point, and
//     gl_GlobalInvocationID -> SV_DispatchThreadID
//   * layout (constant_id = 0) const uint stepIndex -> [[vk::constant_id(0)]] const uint
//     stepIndex: the same specialization constant with the same SpecId and the same default
//   * texture2D/sampler function parameters -> Texture2D<float4>/SamplerState parameters, and
//     textureLod(sampler2D(t, s), uv, 0) -> t.SampleLevel(s, uv, 0)
//   * the unsized local arrays of the golden get their size: vec2 offsets[] -> float2 offsets[9]
//     and float weights[] -> float weights[9]
//   * vec2(scalar) -> (float2)scalar and float(x) -> (float)x, as everywhere in the port;
//     vec2(upsampledPix) is the cast (float2)upsampledPix and vec2(offsets) with two elements
//     keeps its two element constructor
//   * the left operand of the golden's shifts gets an explicit `(int)` cast: `1 << <uint>` is
//     ambiguous to dxc (warning -Wambig-lit-shift) because the shift amount is a uint while the
//     literal is an int. `(int)1` keeps the golden's int result and folds to the same power of
//     two, and it is the only change that avoids the warning -- casting the shift amount instead
//     still warns.
//
// What did not change: the two reciprocal sizes and the shift `1 << (stepIndex + 1)` that builds
// them, the UV of the upsampled pixel, the nine offsets and the tent weights in their order, the
// accumulation `weights[i] * getSample(...)` and the switch that picks the source mip.

#define DESC_SET_FRAMEBUFFERS 0
#define DESC_SET_GLOBAL_UNIFORM 1
#include "ShaderCommonHLSLFunc.hlsli"

// The GLSL original declares the workgroup size here; in HLSL it is an attribute of the
// entry point instead, so the shader itself carries
// [numthreads(COMPUTE_BLOOM_UPSAMPLE_GROUP_SIZE_X, COMPUTE_BLOOM_UPSAMPLE_GROUP_SIZE_Y, 1)].

// A specialization constant, same as the GLSL `layout (constant_id = 0)`.
[[vk::constant_id(0)]] const uint stepIndex = 0;

// for upsampling, stepIndex is decreasing by 1 on each step

float2 getInverseSrcSize()
{
    return float2((float)((int)1 << (stepIndex + 1)) / globalUniform.renderWidth, (float)((int)1 << (stepIndex + 1)) / globalUniform.renderHeight);
}

float2 getInverseUpscampledSize()
{
    return float2((float)((int)1 << stepIndex) / globalUniform.renderWidth, (float)((int)1 << stepIndex) / globalUniform.renderHeight);
}

// get UV coords in [0..1] range
float2 getSrcUV(const int2 upsampledPix)
{
    return ((float2)upsampledPix + (float2)0.5) * getInverseUpscampledSize();
}

float3 getSample(Texture2D<float4> srcTexture, SamplerState srcSampler, const float2 uv)
{
    return srcTexture.SampleLevel(srcSampler, uv, 0).rgb;
}

float3 filterTent3x3(Texture2D<float4> srcTexture, SamplerState srcSampler, const float2 centerUV)
{
    const float2 invSrcSize = getInverseSrcSize();

    const float2 offsets[9] = 
    {
        float2(-1,-1), float2(0,-1), float2(1,-1),
        float2(-1, 0), float2(0, 0), float2(1, 0),
        float2(-1, 1), float2(0, 1), float2(1, 1),
    };

    const float weights[9] = 
    {
        1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
        2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0,
        1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
    };
    
    float3 r = (float3)0.0;

    for (int i = 0; i < 9; i++)
    {
        r += weights[i] * getSample(srcTexture, srcSampler, centerUV + offsets[i] * invSrcSize);
    }

    return r;
}

[numthreads(COMPUTE_BLOOM_UPSAMPLE_GROUP_SIZE_X, COMPUTE_BLOOM_UPSAMPLE_GROUP_SIZE_Y, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // each step upsamples source by 2
    const int2 upsampledPix = int2(dispatchThreadID.x, dispatchThreadID.y);
    const float2 srcUV = getSrcUV(upsampledPix);

    if (srcUV.x > 1.0 || srcUV.y > 1.0)
    {
        return;
    }

    switch (stepIndex)
    {
        case 4: framebufBloom_Mip4[upsampledPix]   = float4(filterTent3x3(framebufBloom_Mip5_Sampled, framebufBloom_Mip5_Sampler, srcUV), 0.0); break;
        case 3: framebufBloom_Mip3[upsampledPix]   = float4(filterTent3x3(framebufBloom_Mip4_Sampled, framebufBloom_Mip4_Sampler, srcUV), 0.0); break;
        case 2: framebufBloom_Mip2[upsampledPix]   = float4(filterTent3x3(framebufBloom_Mip3_Sampled, framebufBloom_Mip3_Sampler, srcUV), 0.0); break;
        case 1: framebufBloom_Mip1[upsampledPix]   = float4(filterTent3x3(framebufBloom_Mip2_Sampled, framebufBloom_Mip2_Sampler, srcUV), 0.0); break;
        case 0: framebufBloom_Result[upsampledPix] = float4(filterTent3x3(framebufBloom_Mip1_Sampled, framebufBloom_Mip1_Sampler, srcUV), 0.0); break;
    }
}

#if COMPUTE_BLOOM_STEP_COUNT != 5
    #error Recheck COMPUTE_BLOOM_STEP_COUNT
#endif
