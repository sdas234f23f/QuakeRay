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


// HLSL counterpart of RhiSkeleton.frag, the checkerboard of the A1 "rhiframe" path: the host loads
// RhiSkeleton.frag.spv, so the blob keeps its name and only the source changes.
//
// Spellings that had to change:
//   * mod( cell.x + cell.y, 2.0 ) is spelled out as x - y * floor( x / y ), as in RaygenPrimary.hlsli
//     and the two ported effect shaders: GLSL mod is floor based while HLSL fmod truncates towards
//     zero, and the two differ for a negative first argument, which floor( vUV * vec2(16.0, 9.0) )
//     produces for a negative vUV
//   * mix -> lerp, the same definition under the HLSL name
//   * vUV becomes an entry point parameter with [[vk::location(0)]] and the TEXCOORD0 semantic that
//     HLSL requires, pinning the golden's layout(location = 0)
//   * the named output oColor becomes the return value of main with an SV_Target0 semantic and
//     [[vk::location(0)]]: a fragment shader output is a return value or an out parameter in HLSL,
//     and the location is what the host sees. The value stays the golden's
//     vec4( color, 1.0 ), computed from the same locals in the same order
//   * the golden's std140 block RhiSkeletonParams at set 0, binding 256 becomes the struct
//     RhiSkeletonParams_BT with the same two float4 members in the same order and
//     [[vk::binding(256, 0)]] ConstantBuffer<RhiSkeletonParams_BT> params. 256 is NVRHI's Vulkan
//     binding for a constant buffer of slot 0; std140 and the HLSL packing rules place both float4
//     members at the same offsets, so the two halves declare the same block layout
//   * the golden's texture2D at set 0, binding 0 and sampler at set 0, binding 128 keep those
//     bindings as Texture2D<float4> and SamplerState, NVRHI's offsets for a shader resource and a
//     sampler of slot 0, and textureLod( sampler2D( whiteTexture, whiteTexture_Sampler ), vUV, 0.0 )
//     becomes whiteTexture.SampleLevel( whiteTexture_Sampler, vUV, 0.0 ), the same explicit LOD 0
//   * the mix keeps its operand order and becomes
//     lerp( params.darkColor.rgb, params.highlightColor.rgb, checker ): the two float3 literals
//     moved into the constant buffer, and the sampled texel multiplies the color right before the
//     return. The caller binds a 1x1 white texture, so every output pixel keeps the golden's value
//   * the bindless table both halves declare at set 1 keeps its split shape on the HLSL side:
//     [[vk::binding(0, 1)]] Texture2D<float4> rhiTextures[]; and
//     [[vk::binding(1, 1)]] SamplerState rhiTextures_Sampler[];, the same shape
//     ShaderCommonHLSLFunc.hlsli declares for the engine's table. The set index is NVRHI's, not the
//     pair's: in the legacy mode a pipeline's layouts keep the order the caller adds them in, with
//     nothing appended or reordered (vulkan-resource-bindings.cpp:1090-1099), and a bindless layout
//     turns its register spaces into Vulkan bindings counted from 0, in the order they are listed
//     (:119 and :145-161). The pass adds its regular layout first and the table second, so the
//     table is set 1: the textures at binding 0, the samplers at binding 1. main samples slot 0 of
//     the table and blends it over the checkerboard
//     in the top-right patch, so the table is visible in game and a wrong slot is obvious
//
// What did not change: the two locals of the checkerboard, the floor of vUV * vec2(16.0, 9.0) with
// the golden's factor pair in that order, the checker value, the gradient added to the color with
// 0.35, 0.15 and the constant 0.0, the caller's white texture multiply and the alpha of 1.0. The
// top-right patch is the only addition to the body.

// Set 0 holds all three resources at the bindings NVRHI assigns to slot 0: the constant buffer at
// 256, the texture at 0 and its sampler at 128. Set 1 is the bindless table described above.
struct RhiSkeletonParams_BT
{
    float4 darkColor;      // rgb = the dark cell color (0.04, 0.05, 0.08), a = unused
    float4 highlightColor; // rgb = the bright cell color (0.20, 0.35, 0.60), a = unused
};

[[vk::binding(256, 0)]] ConstantBuffer<RhiSkeletonParams_BT> params;

[[vk::binding(0, 0)]] Texture2D<float4> whiteTexture;
[[vk::binding(128, 0)]] SamplerState whiteTexture_Sampler;

// Set 1, the bindless table of the pass: one binding per register space, the same split
// ShaderCommonHLSLFunc.hlsli declares for the engine's table. It is not a slot the pass fills.
[[vk::binding(0, 1)]] Texture2D<float4> rhiTextures[];
[[vk::binding(1, 1)]] SamplerState rhiTextures_Sampler[];

// GLSL mod(x, y) is floor based; HLSL fmod truncates towards zero instead, so the golden's mod()
// is spelled out as in RaygenPrimary.hlsli.
float rhiSkeletonMod( float x, float y )
{
    return x - y * floor( x / y );
}

float4 main( [[vk::location(0)]] float2 vUV : TEXCOORD0 ) : SV_Target0
{
    // A checkerboard with a horizontal gradient: it only has to make it obvious
    // that the image on screen is produced by the RHI pass of the current frame.
    const float2 cell = floor( vUV * float2( 16.0, 9.0 ) );
    const float checker = rhiSkeletonMod( cell.x + cell.y, 2.0 );

    float3 color = lerp( params.darkColor.rgb, params.highlightColor.rgb, checker );
    color += float3( vUV.x * 0.35, vUV.y * 0.15, 0.0 );

    // The caller binds a 1x1 white texture, so this multiply leaves the A1 image unchanged.
    color *= whiteTexture.SampleLevel( whiteTexture_Sampler, vUV, 0.0 ).rgb;

    // The bindless table is sampled in the top-right patch so that the table is visible in game:
    // the texture replaces the checkerboard inside the patch, and a wrong slot or an unbound table
    // is immediately obvious.
    const float patchMask = smoothstep( 0.75, 0.77, vUV.x ) * smoothstep( 0.75, 0.77, vUV.y );
    color = lerp( color, rhiTextures[0].SampleLevel( rhiTextures_Sampler[0], vUV, 0.0 ).rgb, patchMask );

    return float4( color, 1.0 );
}
