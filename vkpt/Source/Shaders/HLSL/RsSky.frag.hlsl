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


// HLSL counterpart of RsSky.frag.

#define DESC_SET_TEXTURES 0
#include "ShaderCommonHLSLFunc.hlsli"

// The GLSL push constant block is the fragment half of the host's RasterizedPushConst: it
// starts at offset 64, so each member keeps its golden offset explicitly. The last member,
// emissionMultiplier, is declared by the golden but read by no one.
struct RasterizerFrag_BT
{
    [[vk::offset(64)]] float4 color;
    [[vk::offset(80)]] uint   textureIndex;
    [[vk::offset(84)]] uint   emissionTextureIndex;
    [[vk::offset(88)]] float  emissionMultiplier;
};

[[vk::push_constant]] ConstantBuffer<RasterizerFrag_BT> rasterizerFragInfo;

// A specialization constant, same as the GLSL `layout(constant_id = 0)`.
[[vk::constant_id(0)]] const uint alphaTest = 0;

#define ALPHA_THRESHOLD 0.5

struct RasterizerFragInput
{
    [[vk::location(0)]] float4 vertColor    : COLOR0;
    [[vk::location(1)]] float2 vertTexCoord : TEXCOORD0;
};

struct RasterizerFragOutput
{
    [[vk::location(0)]] float4 outColor : SV_Target0;
};

void main(RasterizerFragInput input, out RasterizerFragOutput output)
{
    float4 albedoAlpha = getTextureSample(rasterizerFragInfo.textureIndex, input.vertTexCoord);


    output.outColor = rasterizerFragInfo.color * input.vertColor * albedoAlpha;


    if (alphaTest != 0)
    {
        if (output.outColor.a < ALPHA_THRESHOLD)
        {
            discard;
        }
    }
}
