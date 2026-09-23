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
// HLSL counterpart of CmBloomDownsample.comp.
//
// Spellings that had to change:
//   * layout(local_size_x = ...) -> [numthreads(COMPUTE_BLOOM_DOWNSAMPLE_GROUP_SIZE_X,
//     COMPUTE_BLOOM_DOWNSAMPLE_GROUP_SIZE_Y, 1)] on the entry point, and
//     gl_GlobalInvocationID -> SV_DispatchThreadID
//   * layout(constant_id = 0) const uint stepIndex -> [[vk::constant_id(0)]] const uint
//     stepIndex: the same specialization constant with the same SpecId and the same default
//   * texture2D/sampler function parameters -> Texture2D<float4>/SamplerState parameters, and
//     textureLod(sampler2D(t, s), uv, 0) -> t.SampleLevel(s, uv, 0)
//   * the unsized local arrays of the golden get their size: vec3 taps[] -> float3 taps[13] and
//     vec3 box[] -> float3 box[5]
//   * vec3(scalar) -> (float3)scalar and float(x) -> (float)x, as everywhere in the port;
//     vec2(downsampledPix) is the cast (float2)downsampledPix
//   * the left operand of the golden's shifts gets an explicit `(int)` cast: `1 << <uint>` is
//     ambiguous to dxc (warning -Wambig-lit-shift) because the shift amount is a uint while the
//     literal is an int. `(int)1` keeps the golden's int result and folds to the same power of
//     two, and it is the only change that avoids the warning -- casting the shift amount instead
//     still warns.
//
// What did not change: the two reciprocal sizes and the shift `1 << (stepIndex + 1)` that builds
// them, the UV of the downsampled pixel, the thirteen taps of the Karis 13-tap filter in their
// order, the five 2x2 boxes and the two branches with their weights, and the switch that picks
// the source mip. stepIndex == 0 selects the branch that weights by the partial Karis average.

#define DESC_SET_FRAMEBUFFERS 0
#define DESC_SET_GLOBAL_UNIFORM 1
#define DESC_SET_TONEMAPPING 2
#include "ShaderCommonHLSLFunc.hlsli"

// The GLSL original declares the workgroup size here; in HLSL it is an attribute of the
// entry point instead, so the shader itself carries
// [numthreads(COMPUTE_BLOOM_DOWNSAMPLE_GROUP_SIZE_X, COMPUTE_BLOOM_DOWNSAMPLE_GROUP_SIZE_Y, 1)].

// A specialization constant, same as the GLSL `layout(constant_id = 0)`.
[[vk::constant_id(0)]] const uint stepIndex = 0;

float2 getInverseSrcSize()
{
    return float2((float)((int)1 << stepIndex) / globalUniform.renderWidth, (float)((int)1 << stepIndex) / globalUniform.renderHeight);
}

float2 getInverseDownsampledSize()
{
    return float2((float)((int)1 << (stepIndex + 1)) / globalUniform.renderWidth, (float)((int)1 << (stepIndex + 1)) / globalUniform.renderHeight);
}

// get UV coords in [0..1] range
float2 getSrcUV(const int2 downsampledPix)
{
    return ((float2)downsampledPix + (float2)0.5) * getInverseDownsampledSize();
}

float3 getSample(Texture2D<float4> srcTexture, SamplerState srcSampler, const float2 uv)
{
    return srcTexture.SampleLevel(srcSampler, uv, 0).rgb;
}

float getKarisWeight(const float3 box4x4)
{
    return 1.0 / (1.0 + getLuminance(box4x4));
}

float3 downsample13tap(Texture2D<float4> srcTexture, SamplerState srcSampler, const float2 centerUV)
{
    const float2 invSrcSize = getInverseSrcSize();

    // line by line indexing, slide 153
    const float3 taps[13] = 
    {
        getSample(srcTexture, srcSampler, centerUV + float2(-2,-2) * invSrcSize),
        getSample(srcTexture, srcSampler, centerUV + float2( 0,-2) * invSrcSize),
        getSample(srcTexture, srcSampler, centerUV + float2( 2,-2) * invSrcSize),

        getSample(srcTexture, srcSampler, centerUV + float2(-1,-1) * invSrcSize),
        getSample(srcTexture, srcSampler, centerUV + float2( 1,-1) * invSrcSize),

        getSample(srcTexture, srcSampler, centerUV + float2(-2, 0) * invSrcSize),
        getSample(srcTexture, srcSampler, centerUV + float2( 0, 0) * invSrcSize),
        getSample(srcTexture, srcSampler, centerUV + float2( 2, 0) * invSrcSize),

        getSample(srcTexture, srcSampler, centerUV + float2(-1, 1) * invSrcSize),
        getSample(srcTexture, srcSampler, centerUV + float2( 1, 1) * invSrcSize),

        getSample(srcTexture, srcSampler, centerUV + float2(-2, 2) * invSrcSize),
        getSample(srcTexture, srcSampler, centerUV + float2( 0, 2) * invSrcSize),
        getSample(srcTexture, srcSampler, centerUV + float2( 2, 2) * invSrcSize),
    };

    // on the first downsample use Karis average
    if (stepIndex == 0)
    {
        const float3 box[5] =
        {
            0.25 * (taps[3] + taps[4] + taps[8]  + taps[9]), 
            0.25 * (taps[0] + taps[1] + taps[5]  + taps[6]), 
            0.25 * (taps[1] + taps[2] + taps[6]  + taps[7]), 
            0.25 * (taps[5] + taps[6] + taps[10] + taps[11]), 
            0.25 * (taps[6] + taps[7] + taps[11] + taps[12]), 
        };

        // weight by partial Karis average to reduce fireflies
        return 
            0.5   * getKarisWeight(box[0]) * box[0] + 
            0.125 * getKarisWeight(box[1]) * box[1] + 
            0.125 * getKarisWeight(box[2]) * box[2] + 
            0.125 * getKarisWeight(box[3]) * box[3] + 
            0.125 * getKarisWeight(box[4]) * box[4];
    }
    else
    {
        return 
            0.5   * (0.25 * (taps[3] + taps[4] + taps[8]  + taps[9]))  + 
            0.125 * (0.25 * (taps[0] + taps[1] + taps[5]  + taps[6]))  + 
            0.125 * (0.25 * (taps[1] + taps[2] + taps[6]  + taps[7]))  + 
            0.125 * (0.25 * (taps[5] + taps[6] + taps[10] + taps[11])) + 
            0.125 * (0.25 * (taps[6] + taps[7] + taps[11] + taps[12]));
    }
}

[numthreads(COMPUTE_BLOOM_DOWNSAMPLE_GROUP_SIZE_X, COMPUTE_BLOOM_DOWNSAMPLE_GROUP_SIZE_Y, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // each step downsamples source by 2
    const int2 downsampledPix = int2(dispatchThreadID.x, dispatchThreadID.y);
    const float2 srcUV = getSrcUV(downsampledPix);

    if (srcUV.x > 1.0 || srcUV.y > 1.0)
    {
        return;
    }
    
    switch (stepIndex)
    {
        case 0: framebufBloom_Mip1[downsampledPix] = float4(downsample13tap(framebufBloomInput_Sampled, framebufBloomInput_Sampler, srcUV), 0.0); break;
        case 1: framebufBloom_Mip2[downsampledPix] = float4(downsample13tap(framebufBloom_Mip1_Sampled, framebufBloom_Mip1_Sampler, srcUV), 0.0); break;
        case 2: framebufBloom_Mip3[downsampledPix] = float4(downsample13tap(framebufBloom_Mip2_Sampled, framebufBloom_Mip2_Sampler, srcUV), 0.0); break;
        case 3: framebufBloom_Mip4[downsampledPix] = float4(downsample13tap(framebufBloom_Mip3_Sampled, framebufBloom_Mip3_Sampler, srcUV), 0.0); break;
        case 4: framebufBloom_Mip5[downsampledPix] = float4(downsample13tap(framebufBloom_Mip4_Sampled, framebufBloom_Mip4_Sampler, srcUV), 0.0); break;
    }
}

#if COMPUTE_BLOOM_STEP_COUNT != 5
    #error Recheck COMPUTE_BLOOM_STEP_COUNT
#endif
