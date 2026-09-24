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


// HLSL counterpart of RhiPresent.frag, the present of the A2 "rhiframe" path: the host loads
// RhiPresent.frag.spv, so the blob keeps its name and only the source changes.
//
// Spellings that had to change:
//   * vUV becomes an entry point parameter with [[vk::location(0)]] and the TEXCOORD0 semantic that
//     HLSL requires, pinning the golden's layout(location = 0)
//   * the named output oColor becomes the return value of main with an SV_Target0 semantic and
//     [[vk::location(0)]]: a fragment shader output is a return value or an out parameter in HLSL,
//     and the location is what the host sees. The value stays the golden's vec4( display, 1.0 ),
//     computed from the same locals in the same order
//   * the golden's std140 block RhiPresentParams at set 0, binding 256 becomes the struct
//     RhiPresentParams_BT with the same single float4 member at the same offset and
//     [[vk::binding(256, 0)]] ConstantBuffer<RhiPresentParams_BT> params. 256 is NVRHI's Vulkan
//     binding for a constant buffer of slot 0; std140 and the HLSL packing rules place the member
//     at the same offset, so the two halves declare the same block layout
//   * the golden's texture2D at set 0, binding 0 and sampler at set 0, binding 128 keep those
//     bindings as Texture2D<float4> and SamplerState, NVRHI's offsets for a shader resource and a
//     sampler of slot 0, and textureLod( sampler2D( albedoTexture, albedoTexture_Sampler ), vUV,
//     0.0 ) becomes albedoTexture.SampleLevel( albedoTexture_Sampler, vUV, 0.0 ), the same
//     explicit LOD 0
//
// Unlike RhiSkeleton.frag, the present has no bindless table and no set 1: it samples exactly one
// image, the ALBEDO render target, so every resource it needs fits in set 0. The host loads this
// blob and RhiSkeleton.vert.spv, the same fullscreen triangle that already feeds the skeleton.
//
// What did not change: the LOD-0 sample of the ALBEDO image, the exposure multiply, the
// x / (1 + x) curve and the alpha of 1.0, with the golden's operand order.

struct RhiPresentParams_BT
{
    // x = exposure multiplier applied before the curve; y, z and w are unused and keep the block
    // one float4, the shape the skeleton's colour parameters have.
    float4 exposure;
};

// Set 0 holds all three resources at the bindings NVRHI assigns to slot 0: the constant buffer at
// 256, the texture at 0 and its sampler at 128.
[[vk::binding(256, 0)]] ConstantBuffer<RhiPresentParams_BT> params;

[[vk::binding(0, 0)]] Texture2D<float4> albedoTexture;
[[vk::binding(128, 0)]] SamplerState albedoTexture_Sampler;

float4 main( [[vk::location(0)]] float2 vUV : TEXCOORD0 ) : SV_Target0
{
    // LOD 0 is explicit, as in RhiSkeleton.frag: the target can differ in size from the source,
    // and an implicit LOD would make the presented image depend on the screen's derivative.
    const float3 albedo = albedoTexture.SampleLevel( albedoTexture_Sampler, vUV, 0.0 ).rgb;

    // The exposure scales the linear HDR value, then the monotone x / (1 + x) curve folds the
    // unbounded result into [0, 1) before it reaches the display attachment.
    const float3 exposed = albedo * params.exposure.x;
    const float3 display = exposed / (1.0 + exposed);

    return float4( display, 1.0 );
}
