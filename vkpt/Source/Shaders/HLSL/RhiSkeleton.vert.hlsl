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


// HLSL counterpart of RhiSkeleton.vert, the fullscreen triangle of the A1 "rhiframe" path: the host
// loads RhiSkeleton.vert.spv, so the blob keeps its name and only the source changes.
//
// Spellings that had to change:
//   * gl_VertexIndex becomes an int parameter with the SV_VertexID semantic. The checker arbitrated
//     the type: a uint parameter declares BuiltIn VertexIndex as uint32 while glslang declares
//     gl_VertexIndex as int32, which the checker reports as a mismatch; int declares int32 and
//     matches, and dxc accepts it on that semantic
//   * vUV becomes a member of the returned struct, [[vk::location(0)]] pinning the golden's
//     layout(location = 0) and TEXCOORD0 carrying the semantic HLSL requires
//   * gl_Position becomes the second member of that struct with an SV_Position semantic, the
//     builtin glslang reports as BuiltIn Position
//   * the bit arithmetic is copied as it is: ( vertexIndex << 1 ) & 2 and vertexIndex & 2 stay on
//     an int, so both halves emit a signed shift and a signed and, and the int to float conversion
//     inside float2(...) is the same convert both compilers write for the golden's vec2(...)
//
// The gl_PerVertex difference of ShadowMap.vert.hlsl applies here as well: glslang declares the
// Position builtin as a member of the output block, dxc as a plain output variable of the same type.
// It is a difference of the two front ends, not of this port, and the wave-6 hand-off reports it.
//
// What did not change: the single local vec2 uv with its expression, the assignment of uv to vUV
// and the position expression uv * 2.0 - 1.0 with the 0.0 and 1.0 components of the golden.

struct RhiSkeletonVertOutput
{
    [[vk::location(0)]] float2 vUV : TEXCOORD0;
    float4 position : SV_Position;
};

RhiSkeletonVertOutput main( int vertexIndex : SV_VertexID )
{
    const float2 uv = float2( ( vertexIndex << 1 ) & 2, vertexIndex & 2 );

    RhiSkeletonVertOutput o;
    o.vUV      = uv;
    o.position = float4( uv * 2.0 - 1.0, 0.0, 1.0 );
    return o;
}
