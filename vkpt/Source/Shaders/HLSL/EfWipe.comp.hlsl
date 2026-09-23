// Copyright (c) 2022 Sultim Tsyrendashiev
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


// HLSL counterpart of EfWipe.comp.

#define DESC_SET_FRAMEBUFFERS 0
#define DESC_SET_GLOBAL_UNIFORM 1
#define DESC_SET_RANDOM 2
#include "ShaderCommonHLSLFunc.hlsli"

// The GLSL original declares the workgroup size here; in HLSL it is an attribute of the
// entry point instead, so the shader itself carries
// [numthreads(COMPUTE_EFFECT_GROUP_SIZE_X, COMPUTE_EFFECT_GROUP_SIZE_Y, 1)].

// A specialization constant, same as the GLSL `layout(constant_id = 0)`.
[[vk::constant_id(0)]] const uint isSourcePing = 0;

#define EFFECT_SOURCE_IS_PING (isSourcePing != 0)
#include "EfCommon.hlsli"

struct WipePush_BT
{
    uint stripWidthInPixels;
    uint startFrameId;
    float beginTime;
    float endTime;
};

[[vk::push_constant]] ConstantBuffer<WipePush_BT> push;

[numthreads(COMPUTE_EFFECT_GROUP_SIZE_X, COMPUTE_EFFECT_GROUP_SIZE_Y, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const int2 pix = int2(dispatchThreadID.x, dispatchThreadID.y);
    int2 sz = effect_getFramebufSize();

    uint count = sz.x  / push.stripWidthInPixels;
    uint strip = pix.x / push.stripWidthInPixels;

    float rnd = effect_getRandomSample(int2(0, 0), push.startFrameId);

    float x = strip / float(count) + rnd * 10;
  
    const float maxDiff = 0.1f;
    float wave = maxDiff * sin(20 * x) * sin(10 * x) * sin(100 * x) + maxDiff;
    
    float progress = 
        max(globalUniform.time - push.beginTime, 0.0) / 
        max(push.endTime       - push.beginTime, 0.001);

    progress = clamp(progress - wave, 0.0, 1.0);
    progress *= progress;
    progress *= 2;

    
    int pix_offset = int(progress * sz.y);
    float3 c;

    if (pix.y - pix_offset < 0)
    {
        c = effect_loadFromSource(pix);
    }
    else
    {
        c = framebufWipeEffectSource[int2(pix.x, pix.y - pix_offset)].rgb;
    }
    
    effect_storeToTarget(c, pix);
}
