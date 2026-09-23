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


struct EffectColorTint_PushConst
{
    // not vec4 for padding
    float intensity;
    float r;
    float g;
    float b;
};

#define EFFECT_PUSH_CONST_T EffectColorTint_PushConst
#include "EfSimple.hlsli"

float3 applyTint(float3 color)
{
    float3 tint = float3(push.custom.r, push.custom.g, push.custom.b);
    
    float t = push.custom.intensity * clamp(getLuminance(color), 0.05, 1.0) * getProgress();
    return lerp(color, tint, t);
}

#define APPLY_RADIAL_OFFSET 1

[numthreads(COMPUTE_EFFECT_GROUP_SIZE_X, COMPUTE_EFFECT_GROUP_SIZE_Y, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const int2 pix = int2(dispatchThreadID.x, dispatchThreadID.y);
    
    if (!effect_isPixValid(pix))
    {
        return;
    }

#if APPLY_RADIAL_OFFSET
    float2 c = effect_getCenteredFromPix(pix);
    c *= lerp(1, 0.985, getProgress());
    
    float3 rgb = lerp(effect_loadFromSource(pix), applyTint(effect_loadFromSource_Centered(c)), 0.5 * dot(c, c));
#else
    float3 rgb = applyTint(effect_loadFromSource(pix));
#endif

    effect_storeToTarget(rgb, pix);
}
