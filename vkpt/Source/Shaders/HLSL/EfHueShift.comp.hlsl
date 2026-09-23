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


#include "EfSimple.hlsli"

// GLSL mod(x, y) is floor based; HLSL fmod truncates towards zero instead, so the golden's mod()
// is spelled out as in RaygenPrimary.hlsli.
float hueShiftMod(float x, float y)
{
    return x - y * floor(x / y);
}

// http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
float3 hsv2rgb(float3 c)
{
    float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * lerp(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

[numthreads(COMPUTE_EFFECT_GROUP_SIZE_X, COMPUTE_EFFECT_GROUP_SIZE_Y, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const int2 pix = int2(dispatchThreadID.x, dispatchThreadID.y);
    
    if (!effect_isPixValid(pix))
    {
        return;
    }
    
    float3 color = effect_loadFromSource(pix);

    // sample albedo, so dark places will be visible too
    const int2 rendPix = int2(effect_getFramebufUV(pix) * float2(globalUniform.renderWidth, globalUniform.renderHeight));
    const float3 albedo = framebufAlbedo_Sampled.Load(int3(rendPix, 0)).rgb;


    float bw = getLuminance(color) + getLuminance(albedo) * 0.4;
    bw = clamp(bw * 1.5, 0.0, 1.0);

    const float h_scale = 0.7;
    const float h_offset = 0.65;
    float h = hueShiftMod(h_offset + bw * h_scale, 1.0);

    float3 dst = hsv2rgb(float3(h, 1.0, clamp(sqrt(bw)+0.1, 0.0, 1.0)));

    effect_storeToTarget(lerp(color, dst, getProgress()), pix);
}
