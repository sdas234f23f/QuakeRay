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


// HLSL counterpart of EfCommon.inl.


#if !defined(EFFECT_SOURCE_IS_PING) && !defined(EFFECT_SOURCE_IS_PONG) 
    #error Define EFFECT_SOURCE_IS_PING or EFFECT_SOURCE_IS_PONG to boolean value
#endif


int2 effect_getFramebufSize()
{
    uint w, h;
    framebufUpscaledPing.GetDimensions(w, h); // framebufUpscaledPong has the same size
    return int2(w, h);
}


float2 effect_getInverseFramebufSize()
{
    int2 sz = effect_getFramebufSize();
    return float2(1.0 / float(sz.x), 1.0 / float(sz.y));
}


int2 effect_clampPix(int2 pix)
{
    return clamp(pix, int2(0, 0), effect_getFramebufSize() - 1);
}


bool effect_isPixValid(int2 pix)
{
    return all(pix == effect_clampPix(pix));
}


// get UV coords in [0..1] range
float2 effect_getFramebufUV(int2 pix)
{
    return (float2(pix) + 0.5) * effect_getInverseFramebufSize();
}


// to [-1..1]
float2 effect_getCenteredFromPix(int2 pix)
{
    return effect_getFramebufUV(pix) * 2.0 - 1.0;
}


// from [-1..1]
int2 effect_getPixFromCentered(float2 centered)
{
    return int2((centered * 0.5 + 0.5) * float2(effect_getFramebufSize()));
}


float3 effect_loadFromSource(int2 pix)
{
    pix = effect_clampPix(pix);

    if (EFFECT_SOURCE_IS_PING)
    {
        return framebufUpscaledPing[pix].rgb;
    }
    else
    {
        return framebufUpscaledPong[pix].rgb;
    }
}


void effect_storeToTarget(const float3 value, int2 pix)
{
    pix = effect_clampPix(pix);

    if (EFFECT_SOURCE_IS_PING)
    {
        framebufUpscaledPong[pix] = float4(value, 0.0);
    }
    else
    {
        framebufUpscaledPing[pix] = float4(value, 0.0);
    }
}


void effect_storeUnmodifiedToTarget(int2 pix)
{
    effect_storeToTarget(effect_loadFromSource(pix), pix);
}


float3 effect_loadFromSource_Centered(float2 centered)
{
    return effect_loadFromSource(effect_getPixFromCentered(centered));
}


// Like the GLSL one, this block pulls Random.hlsli in itself, so the shader has to have included
// the generated header and Utils.hlsli first. The helpers are pinned by the Random probe; the
// block itself declares no descriptor.
#ifdef DESC_SET_RANDOM
#include "Random.hlsli"
float effect_getRandomSample(int2 pix, uint frameIndex)
{
    return rnd16(getRandomSeed(pix, frameIndex), RANDOM_SALT_POSTEFFECT);
}
#endif

// Need these functions as R10G11B10 doesn't allow negative values,
// and I/Q components can be <0
#define I_LIMIT 0.6
#define Q_LIMIT 0.55
float3 encodeYiqForStorage(float3 yiq)
{
    float i = clamp(yiq.y, -I_LIMIT, I_LIMIT);
    float q = clamp(yiq.z, -Q_LIMIT, Q_LIMIT);

    i += I_LIMIT;
    q += Q_LIMIT;

    i /= I_LIMIT * 2;
    q /= Q_LIMIT * 2;

    return float3(yiq.x, i, q);
}
float3 decodeYiqFromStorage(float3 yiqFromStorage)
{
    float i = clamp(yiqFromStorage.y, 0.0, 1.0);
    float q = clamp(yiqFromStorage.z, 0.0, 1.0);

    i *= I_LIMIT * 2;
    q *= Q_LIMIT * 2;

    i -= I_LIMIT;
    q -= Q_LIMIT;

    return float3(yiqFromStorage.x, i, q);
}
#undef I_LIMIT
#undef Q_LIMIT
