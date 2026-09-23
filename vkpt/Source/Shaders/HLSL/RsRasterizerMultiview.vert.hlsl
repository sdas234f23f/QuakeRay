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


// HLSL counterpart of RsRasterizerMultiview.vert.
//
// The golden's `#extension GL_EXT_multiview : require` has no HLSL spelling: the view index
// arrives through SV_ViewID, and dxc declares the very same MultiView capability for it as
// glslc declares for the GLSL extension, without an OpExtension on either side (the capability
// is core in the SPIR-V version Vulkan 1.2 asks for).

#define DESC_SET_GLOBAL_UNIFORM 1
#include "ShaderCommonHLSLFunc.hlsli"

// The push constant block of the golden; the host passes the very same bytes, so no member
// needs an explicit offset.
struct RasterizerVert_BT
{
    float4x4 model;
};

[[vk::push_constant]] ConstantBuffer<RasterizerVert_BT> rasterizerVertInfo;

// A specialization constant, same as the GLSL `layout(constant_id = 0)`.
[[vk::constant_id(0)]] const uint applyVertexColorGamma = 0;

struct MultiviewVertInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float4 color    : COLOR0;
    [[vk::location(2)]] float2 texCoord : TEXCOORD0;
};

struct MultiviewVertOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float4 outColor    : COLOR0;
    [[vk::location(1)]] float2 outTexCoord : TEXCOORD0;
};

MultiviewVertOutput main(MultiviewVertInput input, uint viewIndex : SV_ViewID)
{
    MultiviewVertOutput output;

    if (applyVertexColorGamma != 0)
    {
        output.outColor = float4(pow(input.color.rgb, (float3)2.2), input.color.a);
    }
    else
    {
        output.outColor = input.color;
    }

    output.outTexCoord = input.texCoord;

    const float4x4 viewProj = globalUniform.viewProjCubemap[viewIndex];
    output.position = mul(mul(viewProj, rasterizerVertInfo.model), float4(input.position, 1.0));

    return output;
}
