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


struct EffectChromaticAberration_PushConst
{
    float intensity;
};

#define EFFECT_PUSH_CONST_T EffectChromaticAberration_PushConst
#include "EfSimple.hlsli"

static const float BASE_RADIUS = 0.05;
static const int MAX_ITERATIONS = 8;

// Interpolate between a, b, c by t
float3 mix3(float3 a, float3 b, float3 c, float t)
{
    return lerp(
        lerp(a, b, clamp(t * 2, 0, 1)),
        c,
        clamp((t - 0.5) * 2, 0, 1));
}

float3 getAberrationColor(float t)
{
    return mix3(
        float3(1,0,0), 
        float3(1,1,1), 
        float3(0,0,1), 
        t);
}

[numthreads(COMPUTE_EFFECT_GROUP_SIZE_X, COMPUTE_EFFECT_GROUP_SIZE_Y, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const int2 pix = int2(dispatchThreadID.x, dispatchThreadID.y);
    
    if (!effect_isPixValid(pix))
    {
        return;
    }

    float2 c = effect_getCenteredFromPix(pix);
    float2 offset = BASE_RADIUS * getProgress() * push.custom.intensity * c * dot(c, c);

    int iterCount = clamp(int(length(effect_getFramebufSize() * offset / 2.0)), 3, MAX_ITERATIONS);
    float2 delta = offset / iterCount;

    float3 color = (float3)0.0;
    float3 weight = (float3)0.0;

    for (int i = 0; i < iterCount; i++)
    {
        float3 a = effect_loadFromSource_Centered(c - delta * i);
        float3 w = getAberrationColor((i + 0.5) / iterCount);

        color += a * w;
        weight += w;
    }

    effect_storeToTarget(color / weight, pix);
}
