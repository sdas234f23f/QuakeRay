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


// HLSL counterpart of RsRasterizer.vert.
//
// `gl_Position` is the `SV_Position` output parameter. A `layout (location = N)` varying has no
// counterpart in the HLSL language, so every one of them is pinned with `[[vk::location(N)]]` and
// carries a semantic for the DXIL path: the semantic index is NOT the location (measured: a lone
// `TEXCOORD1` output lands on location 0, the declaration order, while `[[vk::location(1)]]` on
// the same parameter lands on 1). The push constant block keeps the golden's name, and the
// specialization constant keeps the golden's `layout (constant_id = 0)`.

struct RasterizerVert_BT
{
    float4x4 viewProj;
};

[[vk::push_constant]] ConstantBuffer<RasterizerVert_BT> rasterizerVertInfo;

// A specialization constant, same as the GLSL `layout (constant_id = 0)`.
[[vk::constant_id(0)]] const uint applyVertexColorGamma = 0;

void main(
    [[vk::location(0)]] float3 position : POSITION0,
    [[vk::location(1)]] float4 color : COLOR0,
    [[vk::location(2)]] float2 texCoord : TEXCOORD0,
    [[vk::location(0)]] out float4 outColor : COLOR0,
    [[vk::location(1)]] out float2 outTexCoord : TEXCOORD1,
    out float4 outPosition : SV_Position)
{
    if (applyVertexColorGamma != 0)
    {
        outColor = float4(pow(color.rgb, (float3)2.2), color.a);
    }
    else
    {
        outColor = color;
    }

    outTexCoord = texCoord;
    outPosition = mul(rasterizerVertInfo.viewProj, float4(position, 1.0));
}
