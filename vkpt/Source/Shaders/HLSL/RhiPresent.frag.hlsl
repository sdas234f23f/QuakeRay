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

#include "Utils.hlsli"

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
//   * the golden's uimage2D at set 0, binding 384 - the packed unfiltered-direct image, read
//     through the storage image so no sampled descriptor of it is declared - keeps that binding as
//     RWTexture2D<uint4> with the r32ui format, NVRHI's offset for an unordered-access view of slot
//     0; its packed value goes through Utils.hlsli's decodeE5B9G9R9, so this twin includes
//     Utils.hlsli, as every other shader does
//
// Unlike RhiSkeleton.frag, the present has no bindless table and no set 1: it samples the ALBEDO
// render target and reads the packed unfiltered-direct image, so every resource it needs fits in
// set 0. The host loads this blob and RhiSkeleton.vert.spv, the same fullscreen triangle that
// already feeds the skeleton.
//
// The compose is the A4.2 diagnostic of the direct pass: the unpacked direct term is added to the
// albedo - sky pixels have a zero direct value, so they keep their color - and the sum goes through
// the same exposure multiply and x / (1 + x) curve. The A4.4 chain replaces this compose: when
// exposure.w is non-zero the sample already went through CmPrepareFinal, so it is display-referred
// linear color and is passed through raw.

struct RhiPresentParams_BT
{
    // x = exposure multiplier applied before the curve (unused in the display-referred mode);
    // y = vertical mirror of the sample coordinate (0 or 1, set per frame mode); z = non-zero
    // enables the direct-lighting term of the traced chain (kept zero in the display-referred
    // mode); w = the display-referred switch: non-zero passes the sample through raw - no exposure
    // multiply, no direct-lighting term and no x / (1 + x) curve - and zero keeps the diagnostic
    // compose. The block stays one float4, the shape the skeleton's colour parameters have.
    float4 exposure;
};

// Set 0 holds all four resources at the bindings NVRHI assigns to the slots: the constant buffer at
// 256, the ALBEDO texture at 0 and its sampler at 128, the storage image at 384 (the offset of an
// unordered-access view of slot 0).
[[vk::binding(256, 0)]] ConstantBuffer<RhiPresentParams_BT> params;

[[vk::binding(0, 0)]] Texture2D<float4> albedoTexture;
[[vk::binding(128, 0)]] SamplerState albedoTexture_Sampler;
[[vk::binding(384, 0), vk::image_format("r32ui")]] RWTexture2D<uint4> directTexture;

float4 main( [[vk::location(0)]] float2 vUV : TEXCOORD0 ) : SV_Target0
{
    // The sample coordinate: params.exposure.y mirrors it vertically. The traced modes' ALBEDO is
    // written by the engine's ray-tracing passes, whose pixel-to-UV convention puts the view's
    // first row in the image's first row, while the raster passes of the frame feed this present in
    // NVRHI's raster convention, which puts the view's first row in the image's last row (the two
    // were measured in RhiDebugTrace.rgen.hlsl:107-114). The host sets the flag per frame mode, so
    // one present serves both.
    const float2 uv = float2( vUV.x, lerp( vUV.y, 1.0 - vUV.y, params.exposure.y ) );

    // LOD 0 is explicit, as in RhiSkeleton.frag: the target can differ in size from the source,
    // and an implicit LOD would make the presented image depend on the screen's derivative.
    const float3 albedo = albedoTexture.SampleLevel( albedoTexture_Sampler, uv, 0.0 ).rgb;

    // The direct-lighting term of the traced chain: the image is checkerboard-packed, so this
    // screen pixel's trace is addressed through the same mapping the raygens write with
    // (getCheckerboardPix, ShaderCommonHLSLFunc.hlsli:299-307; the render width halves into the two
    // fields). params.exposure.z is set while the direct pass runs; the raster and debug modes leave
    // it zero and read nothing.
    float3 direct = (float3)0.0;
    if (params.exposure.z != 0.0)
    {
        uint width, height;
        albedoTexture.GetDimensions( width, height );

        const int2 p = clamp( int2( uv * float2( width, height ) ), int2( 0, 0 ), int2( width - 1, height - 1 ) );
        const int sep = (int)width / 2;
        const int odd = ( p.x + p.y % 2 ) % 2;
        const int2 cb = int2( odd * sep + p.x / 2, p.y );

        direct = decodeE5B9G9R9( directTexture.Load( cb ).r );
    }

    // The diagnostic compose: the exposure scales the linear HDR value, then the monotone
    // x / (1 + x) curve folds the unbounded result into [0, 1) before it reaches the display
    // attachment. The direct term is added to the albedo (sky pixels keep their color).
    const float3 illuminated = albedo * ( 1.0 + direct );
    const float3 exposed = illuminated * params.exposure.x;
    const float3 curved = exposed / (1.0 + exposed);

    // params.exposure.w is the display-referred switch: after CmPrepareFinal the sample is already
    // the display-referred linear image, so it must reach the attachment raw; the diagnostic path
    // keeps the compose above.
    const float3 display = params.exposure.w != 0.0 ? albedo : curved;

    return float4( display, 1.0 );
}
