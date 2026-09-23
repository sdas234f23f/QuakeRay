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

// HLSL counterpart of RsDecal.frag.
//
// The golden's `gl_FragCoord` is the `SV_Position` input parameter and its `discard` is the same
// statement. The `flat in uint` input keeps its location through `[[vk::location(0)]]` and its
// flatness through `nointerpolation`, which is the decoration the golden's `flat` writes.

#define FRAMEBUF_IGNORE_ATTACHMENTS
#define DESC_SET_GLOBAL_UNIFORM 0
#define DESC_SET_FRAMEBUFFERS 1
#define DESC_SET_TEXTURES 2
#define DESC_SET_DECALS 3
#include "ShaderCommonHLSLFunc.hlsli"

// `inverse` is the one arithmetic that had to be re-spelled: HLSL has no such intrinsic, while the
// golden's built-in reaches SPIR-V as a single `OpExtInst ... MatrixInverse` that neither glslc nor
// spirv-opt expands, so there is no golden-side number to fold against. The adjugate below is the
// same computation that instruction stands for: the transposed cofactor matrix of the 3x3 minors,
// divided by the determinant, which the row constructor fills directly.
//
// The minors are spelled out with plain arithmetic rather than with `determinant`, because dxc
// lowers that intrinsic to another `OpExtInst` that does not fold either: the arithmetic form can
// be folded and compared against a reference, the intrinsic cannot.
float determinant3x3(float3 a, float3 b, float3 c)
{
    return a.x * (b.y * c.z - b.z * c.y) +
           a.y * (b.z * c.x - b.x * c.z) +
           a.z * (b.x * c.y - b.y * c.x);
}

float4x4 inverse4x4(float4x4 m)
{
    const float4 c0 = getColumn(m, 0);
    const float4 c1 = getColumn(m, 1);
    const float4 c2 = getColumn(m, 2);
    const float4 c3 = getColumn(m, 3);

    // The four 3x3 minors of the first column: they are the first row of the adjugate and, expanded
    // along that same column, the 4x4 determinant.
    const float d0 = determinant3x3(c1.yzw, c2.yzw, c3.yzw);
    const float d1 = determinant3x3(c1.xzw, c2.xzw, c3.xzw);
    const float d2 = determinant3x3(c1.xyw, c2.xyw, c3.xyw);
    const float d3 = determinant3x3(c1.xyz, c2.xyz, c3.xyz);

    // Entry (i, j) is (-1)^(i + j) times the determinant of the 3x3 minor that is left after
    // removing row i and column j of the transpose, which is the golden `inverse`'s adjugate.
    const float4x4 adjugate = float4x4(
        d0,
        -d1,
        d2,
        -d3,

        -determinant3x3(c0.yzw, c2.yzw, c3.yzw),
        determinant3x3(c0.xzw, c2.xzw, c3.xzw),
        -determinant3x3(c0.xyw, c2.xyw, c3.xyw),
        determinant3x3(c0.xyz, c2.xyz, c3.xyz),

        determinant3x3(c0.yzw, c1.yzw, c3.yzw),
        -determinant3x3(c0.xzw, c1.xzw, c3.xzw),
        determinant3x3(c0.xyw, c1.xyw, c3.xyw),
        -determinant3x3(c0.xyz, c1.xyz, c3.xyz),

        -determinant3x3(c0.yzw, c1.yzw, c2.yzw),
        determinant3x3(c0.xzw, c1.xzw, c2.xzw),
        -determinant3x3(c0.xyw, c1.xyw, c2.xyw),
        determinant3x3(c0.xyz, c1.xyz, c2.xyz));

    return adjugate / (c0.x * d0 - c0.y * d1 + c0.z * d2 - c0.w * d3);
}

void main(
    float4 fragCoord : SV_Position,
    [[vk::location(0)]] nointerpolation uint instanceIndex : TEXCOORD0,
    out float4 outAlbedo : SV_Target0)
{
    const int2 pix = getCheckerboardPix(int2(fragCoord.xy));

    const ShDecalInstance decal = decalInstances[instanceIndex];
    
    const float3 worldPosition = framebufSurfacePosition_Sampled.Load(int3(pix, 0)).xyz;
    const float4x4 worldToLocal = inverse4x4(decal.transform); // TODO: on CPU
    const float4 localPosition = mul(worldToLocal, float4(worldPosition, 1.0));

    // if not inside [-0.5, 0.5] box
    if (any(abs(localPosition.xyz) > (float3)0.5))
    {
        discard;
    }

    // Z points from surface to outside
    const float2 texCoord = localPosition.xy + 0.5;

    const float4 decalAlbedo = getTextureSample(decal.textureAlbedoAlpha, texCoord);

    outAlbedo = decalAlbedo;
}
