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


// HLSL counterpart of RsDecal.vert.
//
// `gl_VertexIndex` and `gl_InstanceIndex` are the `SV_VertexID`/`SV_InstanceID` parameters, which
// HLSL spells as unsigned where GLSL spells them signed; every index they carry is a non-negative
// vertex or instance number, so the values agree. `gl_Position` is the `SV_Position` output
// parameter. The golden's `flat out uint` keeps its location through `[[vk::location(0)]]` and its
// flatness through `nointerpolation`, which is the decoration the golden's `flat` writes.
//
// The product keeps the golden's operand order through `mul(a, b)`.

#define DESC_SET_GLOBAL_UNIFORM 0
#define DESC_SET_DECALS 3
#include "ShaderCommonHLSLFunc.hlsli"

// Buffer-free [-0.5, 0.5] cube triangle strips 
float4 getPosition(uint vertexIndex)
{
    // https://twitter.com/donzanoid/status/616370134278606848
    // `SV_VertexID` is unsigned, so the literal carries the suffix the shift needs; `b` keeps the
    // golden's twelve bits either way.
    const uint b = 1u << (vertexIndex % 14);

    return float4(
        float((0x287A & b) != 0) - 0.5,
        float((0x02AF & b) != 0) - 0.5,
        float((0x31E3 & b) != 0) - 0.5,
        1.0
    );
}

void main(
    uint vertexIndex : SV_VertexID,
    uint instanceIndex : SV_InstanceID,
    [[vk::location(0)]] nointerpolation out uint outInstanceIndex : TEXCOORD0,
    out float4 outPosition : SV_Position)
{
    outInstanceIndex = instanceIndex;
    outPosition = mul(mul(mul(globalUniform.projection, globalUniform.view), decalInstances[instanceIndex].transform), getPosition(vertexIndex));
}
