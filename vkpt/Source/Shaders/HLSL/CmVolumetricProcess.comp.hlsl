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


// HLSL counterpart of CmVolumetricProcess.comp.
//
// Spellings that had to change:
//   * layout(local_size_x = COMPUTE_VOLUMETRIC_GROUP_SIZE_X, local_size_y = ..., local_size_z = 1)
//     -> [numthreads(COMPUTE_VOLUMETRIC_GROUP_SIZE_X, COMPUTE_VOLUMETRIC_GROUP_SIZE_Y, 1)]
//   * gl_GlobalInvocationID -> SV_DispatchThreadID, and the truncating int(...) of the golden
//     becomes the (int) cast, which truncates identically
//   * layout(constant_id = 0) const float g_temporalWeight -> [[vk::constant_id(0)]]
//   * ivec3(x, y, z) -> int3(x, y, z), vec4(...) -> float4(...): same component lists, the only
//     change is the type name
//   * texelFetch(g_volumetric_Sampled, cell, 0) -> g_volumetric_Sampled.Load(int4(cell, 0)): for a
//     3D texture the mip level is the fourth component of Load
//   * imageStore(g_volumetric, cell, final) -> g_volumetric[cell] = final
//   * mix(final, prev, g_temporalWeight) -> lerp(final, prev, g_temporalWeight), the same
//     definition under the HLSL name, as in BRDF.hlsli
//   * saturate keeps its name: the GLSL side takes it from Utils.h, HLSL has it built in
//
// What did not change: the two function names and their parameter lists, the order of the
// components of every float4 (rgb then a), the arithmetic of accumulateScattering (the
// front.rgb + saturate(exp(-front.a)) * back.rgb and the front.a + back.a), the exp(-a) of
// store, the constant comparison g_temporalWeight > 0.0 and the temporal blend it guards, the
// VOLUMETRIC_SIZE_Z loop and the accumulate-then-store order of its body. The golden includes
// Random.h without calling anything from it, and this file includes Random.hlsli the same way.

#define DESC_SET_VOLUMETRIC 0
#define DESC_SET_GLOBAL_UNIFORM 1
#define DESC_SET_RANDOM 2
#include "ShaderCommonHLSLFunc.hlsli"
#include "Random.hlsli"
#include "Volumetric.hlsli"

// "Volumetric Fog and Lighting", Bart Wronski

// A specialization constant, same as the GLSL `layout(constant_id = 0)`. It is declared before
// the entry point because [numthreads] has to be attached to the function it describes.
[[vk::constant_id(0)]] const float g_temporalWeight = 0.95;

// rgb - in−scattered light accumulated so far
// a   - accumulated scattering coefficient
float4 accumulateScattering( const float4 front, const float4 back )
{
    float3 light = front.rgb + saturate( exp( -front.a ) ) * back.rgb;

    return float4( light.rgb, front.a + back.a );
}

void store( const int3 cell, const float4 colorAndDensity )
{
    float4 final = float4( colorAndDensity.rgb, exp( -colorAndDensity.a ) );

    if( g_temporalWeight > 0.0 )
    {
        float4 prev = volume_sample_Prev( cell );
        final       = lerp( final, prev, g_temporalWeight );
    }

    // rgb - in−scattered light accumulated so far
    // a   - scene light transmittance
    g_volumetric[cell] = final;
}

[numthreads(COMPUTE_VOLUMETRIC_GROUP_SIZE_X, COMPUTE_VOLUMETRIC_GROUP_SIZE_Y, 1)]
void main( uint3 dispatchThreadID : SV_DispatchThreadID )
{
    const int x = (int)dispatchThreadID.x;
    const int y = (int)dispatchThreadID.y;


    float4 accum = g_volumetric_Sampled.Load( int4( x, y, 0, 0 ) );
    store( int3( x, y, 0 ), accum );

    for( int z = 1; z < VOLUMETRIC_SIZE_Z; z++ )
    {
        const int3 cell = int3( x, y, z );
        const float4 v  = g_volumetric_Sampled.Load( int4( cell, 0 ) );

        accum = accumulateScattering( accum, v );

        store( cell, accum );
    }
}
