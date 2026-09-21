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

#pragma once

// HLSL counterpart of Random.h. Like the GLSL one it includes nothing itself: the shader has to
// pull in Utils.hlsli (M_PI, safePositiveRcp, UINT8_MAX) and the generated header
// (BLUE_NOISE_TEXTURE_*, BINDING_BLUE_NOISE) first.
//
// Spellings that had to change:
//   * uint(...) / float(...) constructor -> C style cast, same truncation and rounding
//   * texelFetch(t, ivec3(x, y, slice), 0) -> t.Load(int4(x, y, 0, slice)), because HLSL spells
//     the array slice as the 4th component and widens nothing implicitly, so the arguments are
//     cast: `t.Load(int4((int)offset.x, (int)offset.y, 0, (int)texIndex))`
//   * `r == 0 ? 0 : ...` inside a float expression -> `0.0`
//   * a matrix row can not be passed as an `out` argument, so getONB fills two locals and
//     assigns the rows (see there)
//
// getONB keeps the GLSL convention of storing the basis vectors in basis[0], basis[1], basis[2].
// A GLSL column is an HLSL row, so both spellings agree: the caller reads basis[i] and the
// product `basis * v` becomes `mul(v, basis)`.

#define RANDOM_SALT_DIFF_BOUNCE(bounceIndex) (8 + (bounceIndex))
#define RANDOM_SALT_SPEC_BOUNCE(bounceIndex) (12 + (bounceIndex))
#define RANDOM_SALT_POSTEFFECT 16
#define RANDOM_SALT_LIGHT_POINT 20
#define RANDOM_SALT_LIGHT_GRID_BASE 24
#define RANDOM_SALT_INITIAL_RESERVOIRS_BASE 48
#define RANDOM_SALT_LIGHT_CHOOSE_DIRECT_BASE 72
#define RANDOM_SALT_LIGHT_CHOOSE_INDIRECT_BASE 96
#define RANDOM_SALT_RESAMPLE_INDIRECT_BASE 132

// Sample disk uniformly
// u1, u2 -- uniform random numbers
float2 sampleDisk(float radius, float u1, float u2)
{
    // from [0,1] to [0,1)
    u1 *= 0.99;
    u2 *= 0.99;

    // polar mapping
    const float r = radius * sqrt(u1);
    const float phi = 2 * M_PI * u2;

    return float2(
        r * cos(phi),
        r * sin(phi)
    );

    // pdf = M_PI * radius * radius;
}

// Sample triangle uniformly
// u1, u2 -- uniform random numbers
// "Ray Tracing Gems", Chapter 16: Sampling Transformations Zoo,
// 16.5.2.1 Warping
float3 sampleTriangle(const float3 p0, const float3 p1, const float3 p2, float u1, float u2)
{
    // from [0,1] to [0,1)
    u1 *= 0.99;
    u2 *= 0.99;

    float beta = 1 - sqrt(u1);
    float gamma = (1 - beta) * u2;
    float alpha = 1 - beta - gamma;

    return alpha * p0 + beta * p1 + gamma * p2;
}

// Sample direction from cosine-weighted unit hemisphere oriented to Z axis
// u1, u2 -- uniform random numbers
float3 sampleHemisphere(float u1, float u2, out float oneOverPdf)
{
    // from [0,1] to [0,1)
    u1 *= 0.99;
    u2 *= 0.99;

    const float r = sqrt(u1);
    const float phi = 2 * M_PI * u2;

    const float z = sqrt(1 - u1);

    // clamp z, so max oneOverPdf is finite (currenty, 10pi)
    oneOverPdf = M_PI / max(z, 0.1);

    return float3(
        r * cos(phi),
        r * sin(phi),
        z
    );
}

// Sample a surface point on a unit sphere with given radius
// u1, u2 -- uniform random numbers
// "Ray Tracing Gems", Chapter 16: Sampling Transformations Zoo,
// Octathedral concentric uniform map
float3 sampleSphere(float u1, float u2)
{
    // from [0,1] to [0,1)
    u1 *= 0.99;
    u2 *= 0.99;

    u1 = 2 * u1 - 1;
    u2 = 2 * u2 - 1;

    const float d = 1 - (abs(u1) + abs(u2));
    const float r = 1 - abs(d);

    const float phi = r == 0 ? 0.0 : M_PI / 4 * ((abs(u2) - abs(u1)) / r + 1);
    const float f = r * sqrt(2 - r * r);

    return float3(
        f * sign(u1) * cos(phi),
        f * sign(u2) * sin(phi),
        sign(d) * (1 - r * r));

    // pdf = 1 / (4 * M_PI)
}

// "Building an Orthonormal Basis, Revisited"
void revisedONB(const float3 n, out float3 b1, out float3 b2)
{
    if(n.z < 0.0)
    {
        const float a = 1.0f / (1.0f - n.z);
        const float b = n.x * n.y * a;

        b1 = float3(1.0f - n.x * n.x * a, -b, n.x);
        b2 = float3(b, n.y * n.y * a - 1.0f, -n.y);
    }
    else
    {
        const float a = 1.0f / (1.0f + n.z);
        const float b = -n.x * n.y * a;

        b1 = float3(1.0f - n.x * n.x * a, b, -n.x);
        b2 = float3(b, 1.0f - n.y * n.y * a, -n.y);
    }
}

// "Building an Orthonormal Basis from a 3D Unit Vector Without Normalization", Frisvad
void frisvadONB(const float3 n, out float3 b1, out float3 b2)
{
    if(n.z < -0.9999999)
    {
        b1 = float3( 0.0, -1.0, 0.0);
        b2 = float3( -1.0, 0.0, 0.0);

        return;
    }

    const float a = 1.0 / (1.0 + n.z);
    const float b = -n.x * n.y * a;

    b1 = float3(1.0 - n.x * n.x * a, b, n.x);
    b2 = float3(b, 1.0 - n.y * n.y * a, -n.y);
}

float3x3 getONB(const float3 n)
{
    float3x3 basis;
    basis[2] = n;

    // a matrix row is not an lvalue that can be passed as an `out` argument, so the rows are
    // built in locals first
    float3 b1;
    float3 b2;
    revisedONB(n, b1, b2);
    //frisvadONB(n, b1, b2); // Note: buggy for VNDF

    basis[0] = b1;
    basis[1] = b2;

    return basis;
}

// Sample direction in a hemisphere oriented to a normal n
float3 sampleOrientedHemisphere(const float3 n, float u1, float u2, out float oneOverPdf)
{
    /*float3 a = sampleHemisphere(u1, u2, oneOverPdf);

    float3x3 basis = getONB(n);
    return normalize(mul(a, basis));*/

    // Ray Tracing Gems, Chapter 16 "Sampling Transformations Zoo"
    float a = 1 - 2 * u1;
    float b = sqrt(1 - a * a);
    float phi = 2 * M_PI * u2;

    // avoid grazing angles (perpendicular to normal), 
    // so r won't be close to zero
    a *= 0.98;
    b *= 0.98;

    float3 r = float3(
        n.x + b * cos(phi),
        n.y + b * sin(phi),
        n.z + a
    );
    r = normalize(r);

    float z = dot(r, n);
    oneOverPdf = M_PI * safePositiveRcp(z);

    return r;
}

uint packRandomSeed(uint textureIndex, uint2 offset)
{
    return
        (textureIndex << (BLUE_NOISE_TEXTURE_SIZE_POW * 2)) |
        (offset.y     << (BLUE_NOISE_TEXTURE_SIZE_POW    )) |
        offset.x;
}

void unpackRandomSeed(uint seed, out uint textureIndex, out uint2 offset)
{
    textureIndex = seed >> (BLUE_NOISE_TEXTURE_SIZE_POW * 2);
    offset.y     = (seed >> BLUE_NOISE_TEXTURE_SIZE_POW) & (BLUE_NOISE_TEXTURE_SIZE - 1);
    offset.x     = seed                                  & (BLUE_NOISE_TEXTURE_SIZE - 1);
}


#ifdef DESC_SET_RANDOM
// The bindless table split that ShaderCommonHLSLFunc.hlsli applies to textures does not touch the
// blue noise: it is a single texture2DArray read without a sampler, so the image is the only
// descriptor that the host binds here.
[[vk::binding(BINDING_BLUE_NOISE, DESC_SET_RANDOM)]]
Texture2DArray<float4> blueNoiseTextures;

#if BLUE_NOISE_TEXTURE_SIZE_POW * 2 > 31
    #error BLUE_NOISE_TEXTURE_SIZE_POW must be lower, around 6-8
#endif

// Blue noise random in [0..1] with 1/255 precision
float4 rndBlueNoise8(uint seed, uint salt)
{
    uint texIndex;
    uint2 offset;
    unpackRandomSeed(seed, texIndex, offset);

    texIndex = (texIndex + salt) % BLUE_NOISE_TEXTURE_COUNT;

    return blueNoiseTextures.Load(int4((int)offset.x, (int)offset.y, 0, (int)texIndex));
}
#endif // DESC_SET_RANDOM


// https://nullprogram.com/blog/2018/07/31/
uint wellonsLowBias32(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// Random in [0..1] with 1/65535 precision
float rnd16(uint seed, uint salt)
{
    uint rnd = wellonsLowBias32(seed + salt);
    return
        (float)(rnd & 0x0000FFFF) / (float)UINT16_MAX;
}

float2 rnd16_2(uint seed, uint salt)
{
    uint rnd = wellonsLowBias32(seed + salt);
    return float2(
        (float)(rnd & 0x0000FFFF) / (float)UINT16_MAX,
        (float)((rnd & 0xFFFF0000) >> 16) / (float)UINT16_MAX);
}

float4 rnd8_4(uint seed, uint salt)
{
    uint rnd = wellonsLowBias32(seed + salt);
    return float4(
        (float)(rnd & 0x000000FF) / (float)UINT8_MAX,
        (float)((rnd & 0x0000FF00) >> 8 ) / (float)UINT8_MAX,
        (float)((rnd & 0x00FF0000) >> 16) / (float)UINT8_MAX,
        (float)((rnd & 0xFF000000) >> 24) / (float)UINT8_MAX);
}

// https://gist.github.com/mpottinger/54d99732d4831d8137d178b4a6007d1a
uint3 murmurHash33(uint3 src) {
    const uint M = 0x5bd1e995u;
    uint3 h = uint3(1190494759u, 2147483647u, 3559788179u);
    src *= M; src ^= src >> 24u; src *= M;
    h *= M; h ^= src.x; h *= M; h ^= src.y; h *= M; h ^= src.z;
    h ^= h >> 13u; h *= M; h ^= h >> 15u;
    return h;
}

uint getRandomSeed(const int2 pix, uint frameIndex)
{
    uint3 hash = murmurHash33(uint3((uint)pix.x, (uint)pix.y, frameIndex));

    uint2 offset = uint2(
        hash.x % BLUE_NOISE_TEXTURE_SIZE,
        hash.y % BLUE_NOISE_TEXTURE_SIZE
    );
    uint texIndex = hash.z % BLUE_NOISE_TEXTURE_COUNT;

    return packRandomSeed(texIndex, offset);
}
