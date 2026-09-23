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
//
// What did not change: the whole body with its two locals, the floor of vUV * vec2(16.0, 9.0) with
// the golden's factor pair in that order, the checker value, the two colors of the mix, the gradient
// added to the color with 0.35, 0.15 and the constant 0.0, and the alpha of 1.0.

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

    float3 color = lerp( float3( 0.04, 0.05, 0.08 ), float3( 0.20, 0.35, 0.60 ), checker );
    color += float3( vUV.x * 0.35, vUV.y * 0.15, 0.0 );

    return float4( color, 1.0 );
}
