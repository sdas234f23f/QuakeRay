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


// HLSL counterpart of RsRasterizerLensFlare.vert.

#define DESC_SET_GLOBAL_UNIFORM 0
#define DESC_SET_LENS_FLARE_VERTEX_INSTANCES 2
#include "ShaderCommonHLSLFunc.hlsli"

// The GLSL block holds one runtime array, which keeps both its name and its `x[i]` syntax in HLSL.
// It is a writeable `buffer` in the golden, so the port is a RWStructuredBuffer.
[[vk::binding(BINDING_DRAW_LENS_FLARES_INSTANCES, DESC_SET_LENS_FLARE_VERTEX_INSTANCES)]]
RWStructuredBuffer<ShLensFlareInstance> lensFlareInstances;

// A specialization constant, same as the GLSL `layout(constant_id = 0)`.
[[vk::constant_id(0)]] const uint applyVertexColorGamma = 0;

// gl_InstanceIndex is SV_InstanceID, whose HLSL type is uint where the GLSL builtin is signed;
// only the sign differs in SPIR-V (the width does not), and the DXIL path rejects the signed
// spelling, so the port keeps the HLSL one. The golden's `flat out uint` needs no qualifier on
// the vertex side: dxc marks an integer varying as Flat by itself, where glslang leaves the
// vertex output bare and qualifies only the fragment input.
struct LensFlareVertInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float4 color    : COLOR0;
    [[vk::location(2)]] float2 texCoord : TEXCOORD0;
};

struct LensFlareVertOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float4 outColor        : COLOR0;
    [[vk::location(1)]] float2 outTexCoord     : TEXCOORD0;
    [[vk::location(2)]] uint   outTextureIndex : TEXCOORD1;
};

LensFlareVertOutput main(LensFlareVertInput input, uint instanceID : SV_InstanceID)
{
    LensFlareVertOutput output;

    if (applyVertexColorGamma != 0)
    {
        output.outColor = float4(pow(input.color.rgb, (float3)2.2), input.color.a);
    }
    else
    {
        output.outColor = input.color;
    }

    output.outTexCoord = input.texCoord;
    output.outTextureIndex = lensFlareInstances[instanceID].textureIndex;

    if (globalUniform.applyViewProjToLensFlares != 0)
    {
        output.position = mul(mul(globalUniform.projection, globalUniform.view), float4(input.position, 1.0));
    }
    else
    {
        output.position = float4(input.position, 1.0);
    }

    return output;
}
