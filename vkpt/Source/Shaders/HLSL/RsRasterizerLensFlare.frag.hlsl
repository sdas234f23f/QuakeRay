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


// HLSL counterpart of RsRasterizerLensFlare.frag.

#define DESC_SET_TEXTURES 1
#include "ShaderCommonHLSLFunc.hlsli"

// A specialization constant, same as the GLSL `layout(constant_id = 0)`.
[[vk::constant_id(0)]] const uint alphaTest = 0;

#define ALPHA_THRESHOLD 0.5

struct LensFlareFragInput
{
    [[vk::location(0)]] float4 vertColor    : COLOR0;
    [[vk::location(1)]] float2 vertTexCoord : TEXCOORD0;
    // The golden's `flat in uint`: an integer fragment input has to be flat in SPIR-V as well.
    // dxc adds the decoration for an integer input by itself; the qualifier is written out
    // because it is the golden's intent and it costs nothing.
    [[vk::location(2)]] nointerpolation uint textureIndex : TEXCOORD1;
};

struct LensFlareFragOutput
{
    [[vk::location(0)]] float4 outColor          : SV_Target0;
    [[vk::location(1)]] float3 outScreenEmission : SV_Target1;
};

void main(LensFlareFragInput input, out LensFlareFragOutput output)
{
    output.outColor = input.vertColor * getTextureSample(input.textureIndex, input.vertTexCoord);
    output.outScreenEmission = (float3)0.0;

    if (alphaTest != 0)
    {
        if (output.outColor.a < ALPHA_THRESHOLD)
        {
            discard;
        }
    }
}
