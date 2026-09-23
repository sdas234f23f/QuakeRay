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


// HLSL counterpart of CmCheckerboard.comp.
//
// Spellings that had to change:
//   * gl_GlobalInvocationID.x/y -> SV_DispatchThreadID.x/y, and the golden's local_size_x/y =
//     COMPUTE_COMPOSE_GROUP_SIZE_X/Y become [numthreads(...)] on the entry point
//   * the parameter of resolveCheckerboard, GLSL texture2D src, becomes Texture2D<float4> src;
//     texelFetch(src, p, 0) -> src.Load(int3(p, 0)) and imageStore(img, p, v) -> img[p] = v
//   * mix(center, crossColor, 0.5) -> lerp(center, crossColor, 0.5)
//   * vec3(hdr, 0) / vec4(0) keep their shape, only the types are renamed: a scalar next to a
//     vector is broadcast into the missing component by HLSL as well
//   * float(!isLeftOutOfBounds(x)) -> (float)(!isLeftOutOfBounds(x)): HLSL converts a bool with a
//     cast, and the negation, not the conversion, is what turns the out-of-bounds tap off
//   * uint(globalUniform.renderWidth) -> (uint)globalUniform.renderWidth, and the comparison of
//     the golden stays UNSIGNED: the golden compares an int pixel with a uint width, so the port
//     converts the pixel with (uint) too instead of letting HLSL pick a signed comparison
//
// What did not change: the two out-of-bounds predicates and their equality tests, the four-tap
// sum of the cross color with its fixed 0.25 weights and the two unsigned taps at the end,
// the resolve condition and the three image stores.

#define DESC_SET_FRAMEBUFFERS 0
#define DESC_SET_GLOBAL_UNIFORM 1
#include "ShaderCommonHLSLFunc.hlsli"

bool isLeftOutOfBounds(int checkerboardPix_X)
{
    return 
        checkerboardPix_X == 0 ||
        checkerboardPix_X == getCheckerboardSeparatorX(); 
}

bool isRightOutOfBounds(int checkerboardPix_X)
{
    return 
        checkerboardPix_X == getCheckerboardSeparatorX() ||
        checkerboardPix_X == (int)globalUniform.renderWidth;
}

float3 resolveCheckerboard( Texture2D<float4> src, const int2 regularPix, const int2 checkerboardPix, const float3 center )
{
    float3 crossColor =
        0.25 * src.Load(int3(regularPix + int2( -1,  0 ), 0)).rgb * (float)( !isLeftOutOfBounds( checkerboardPix.x ) ) +
        0.25 * src.Load(int3(regularPix + int2(  1,  0 ), 0)).rgb * (float)( !isRightOutOfBounds( checkerboardPix.x ) ) +
        0.25 * src.Load(int3(regularPix + int2(  0,  1 ), 0)).rgb +
        0.25 * src.Load(int3(regularPix + int2(  0, -1 ), 0)).rgb;

    return lerp( center, crossColor, 0.5 );
}

[numthreads(COMPUTE_COMPOSE_GROUP_SIZE_X, COMPUTE_COMPOSE_GROUP_SIZE_Y, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const int2 pix             = int2( dispatchThreadID.x, dispatchThreadID.y );
    const int2 checkerboardPix = getCheckerboardPix( pix );

    if( (uint)pix.x >= (uint)globalUniform.renderWidth || (uint)pix.y >= (uint)globalUniform.renderHeight )
    {
        return;
    }

    // Color (PreFinal) is already resolved from checkerboard by CmQ2Interleave;
    // resolving it again here would double-blur split surfaces.
    float3 hdr  = framebufPreFinal_Sampled.Load(int3( pix, 0 )).rgb;
    float3 emis = framebufScreenEmisRT_Sampled.Load(int3( pix, 0 )).rgb;
    float3 fog  = framebufAcidFogRT_Sampled.Load(int3( pix, 0 )).rgb;

    if( needResolveCheckerboard( checkerboardPix ) )
    {
        // Emission and fog are written per-split-field in regular layout by the
        // ray gen; blend them across the split here.
        emis = resolveCheckerboard( framebufScreenEmisRT_Sampled, pix, checkerboardPix, emis );
        fog  = resolveCheckerboard( framebufAcidFogRT_Sampled, pix, checkerboardPix, fog );
    }

    framebufFinal[pix] = float4( hdr, 0 );
    framebufScreenEmission[pix] = float4( emis, 0 );
    framebufAcidFog[pix] = float4( fog, 0 );
}
