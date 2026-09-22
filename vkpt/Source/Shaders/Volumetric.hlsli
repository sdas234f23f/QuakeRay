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


// HLSL counterpart of Volumetric.h. Like the GLSL one it includes nothing itself: the shader has to
// pull in ShaderCommonHLSLFunc.hlsli first, which declares globalUniform under
// #ifdef DESC_SET_GLOBAL_UNIFORM, the volumetric storage image, the two sampled volumes and their
// samplers under #ifdef DESC_SET_VOLUMETRIC, and -- through ShaderCommonHLSL.hlsli -- Utils.hlsli,
// where safePositiveRcp and safeNormalize come from. CmVolumetricProcess.comp, CmPrepareFinal.comp
// and RsWorld.frag, the consumers of the GLSL one, do the same: they include ShaderCommonGLSLFunc.h
// before Volumetric.h. VOLUMETRIC_SIZE_X, VOLUMETRIC_SIZE_Y and VOLUMETRIC_SIZE_Z come from the
// shader as well; the GLSL one gets them from the same place.
//
// Spellings that had to change:
//   * the two matrix products keep the order of the operands, as ShaderCommonHLSL.hlsli prescribes:
//     `viewprojInv * ndc` became mul(viewprojInv, ndc) and `viewproj * vec4(world, 1.0)` became
//     mul(viewproj, float4(world, 1.0)). The matrices arrive as parameters and are declared
//     float4x4 for the mat4 of the golden, which is the declaration rule of that file; the header
//     builds no matrix, indexes none and transposes none, so no other matrix rule applies to it
//   * vec3(cell), the conversion of an int3 parameter to a float3, -> (float3)cell, a C style cast,
//     because a scalar does not broadcast into a constructor in HLSL (see Utils.hlsli). That
//     conversion in volume_getCenter_T is the only one of the header
//   * ivec3(...) -> int3(...): both truncate toward zero, so volume_toCellIndex keeps the
//     conversion of the golden and needs no floor
//   * mix(n, f, z) -> lerp(n, f, z), the same definition under the HLSL name, as in BRDF.hlsli
//   * textureLod(sampler3D(g_volumetric_Sampled, g_volumetric_Sampler), sp, 0.0) became
//     g_volumetric_Sampled.SampleLevel(g_volumetric_Sampler, sp, 0.0), and the same for the
//     previous volume: the descriptors are split into a sampled view and a sampler, so the combined
//     sampler3D(...) spelling of the golden does not exist in HLSL. The volumetric views are 3D, so
//     the getTextureSampleLod of ShaderCommonHLSLFunc.hlsli, which is declared for the 2D tables,
//     does not apply and the SampleLevel is spelled out here
//   * the #ifndef VOLUMETRIC_H_ / #define VOLUMETRIC_H_ guard -> #pragma once, as in BRDF.hlsli
//
// What did not change: the eight functions with their names and their parameter lists, the call
// graph of the golden (volume_sample and volume_sampleDithered through volume_toSamplePosition_T,
// volume_getCenter and volume_getCenter_Prev through volume_getCenter_T, volume_sample_Prev through
// volume_getCenter and volume_toSamplePosition_T), the fields of globalUniform that the two variants
// of each helper read -- volumeViewProj, volumeViewProj_Prev, cameraPosition, cameraPositionPrev,
// volumeCameraNear and volumeCameraFar -- the arithmetic of the round trip through the inverse view
// projection (worldpos.xyz *= safePositiveRcp(worldpos.w), worlddir = safeNormalize(worldpos.xyz -
// origin), ndc.xy /= ndc.w) and of the distance ramp, the clamp and the pow of the cell depth,
// VOLUMETRIC_DISTANCE_POW with its two spellings, pow(z, 1) and pow(z, 1.0 / 1), which both
// compilers promote, and the #if SHIPPING_HACK line that neutralizes ditherRadius, which stays a
// compile time switch here as it is there. The brace initializer of ndc is copied as it is, with
// its trailing comma: dxc takes a brace list for a vector the way glslang does, and it fills the
// components in the same order (verified by compiling the probe pair of this file).
//
// The #if !defined(...) || !defined(...) guard of the golden is copied verbatim, the bare #error
// included: dxc accepts a #error that has no message.
//
// Volumetric.h declares no descriptor of its own, so this file is pinned by the probe pair
// GLSL/Volumetric.probe.comp <-> Probes/Volumetric.probe.comp.hlsl.

#ifndef VOLUMETRIC_HLSLI_
#define VOLUMETRIC_HLSLI_
#if !defined( DESC_SET_GLOBAL_UNIFORM ) || !defined( DESC_SET_VOLUMETRIC )
    #error
#endif


#define VOLUMETRIC_DISTANCE_POW 1

float3 volume_getCenter_T( const int3 cell, const float4x4 viewprojInv, const float3 origin )
{
    float3 local =
        ( (float3)cell + 0.5 ) / float3( VOLUMETRIC_SIZE_X, VOLUMETRIC_SIZE_Y, VOLUMETRIC_SIZE_Z );

    float4 ndc = {
        local.x * 2.0 - 1.0,
        local.y * 2.0 - 1.0,
        0.1,
        1.0,
    };

    float4 worldpos = mul( viewprojInv, ndc );
    worldpos.xyz *= safePositiveRcp( worldpos.w );

    float3 worlddir = safeNormalize( worldpos.xyz - origin );

    float n = globalUniform.volumeCameraNear;
    float f = globalUniform.volumeCameraFar;

    float z    = clamp( local.z, 0.0, 1.0 );
    z          = pow( z, VOLUMETRIC_DISTANCE_POW );
    float dist = lerp( n, f, z );

    return origin + worlddir * dist;
}

float3 volume_toSamplePosition_T( const float3 world, const float4x4 viewproj, const float3 origin )
{
    float4 ndc = mul( viewproj, float4( world, 1.0 ) );
    ndc.xy /= ndc.w;

    float n = globalUniform.volumeCameraNear;
    float f = globalUniform.volumeCameraFar;

    float dist = length( world - origin );
    float z    = ( dist - n ) / ( f - n );
    z          = clamp( z, 0.0, 1.0 );
    z          = pow( z, 1.0 / VOLUMETRIC_DISTANCE_POW );

    return float3( 
        ndc.x * 0.5 + 0.5,
        ndc.y * 0.5 + 0.5,
        z );
}


float3 volume_getCenter( const int3 cell ) 
{
    return volume_getCenter_T(
        cell, globalUniform.volumeViewProjInv, globalUniform.cameraPosition.xyz );
}
float3 volume_getCenter_Prev( const int3 prevcell )
{
    return volume_getCenter_T(
        prevcell, globalUniform.volumeViewProjInv_Prev, globalUniform.cameraPositionPrev.xyz );
}


int3 volume_toCellIndex( const float3 samplePosition )
{
    return int3( samplePosition *
                 float3( VOLUMETRIC_SIZE_X, VOLUMETRIC_SIZE_Y, VOLUMETRIC_SIZE_Z ) );
}


float4 volume_sample( const float3 world ) 
{
    float3 sp = volume_toSamplePosition_T(
        world, globalUniform.volumeViewProj, globalUniform.cameraPosition.xyz );

    return g_volumetric_Sampled.SampleLevel( g_volumetric_Sampler, sp, 0.0 );
}

float4 volume_sample_Prev( const int3 curcell )
{
    float3 curworld = volume_getCenter( curcell );

    float3 spPrev = volume_toSamplePosition_T(
        curworld, globalUniform.volumeViewProj_Prev, globalUniform.cameraPositionPrev.xyz );

    return g_volumetric_Sampled_Prev.SampleLevel( g_volumetric_Sampler_Prev, spPrev, 0.0 );
}

float4 volume_sampleDithered( const float3 world, const float3 rnd01, float ditherRadius )
{
    float3 sp = volume_toSamplePosition_T(
        world, globalUniform.volumeViewProj, globalUniform.cameraPosition.xyz );

#if SHIPPING_HACK
    ditherRadius = 0.0;
#endif

    sp += ditherRadius * ( rnd01 * 2 - 1 ) /
          float3( VOLUMETRIC_SIZE_X, VOLUMETRIC_SIZE_Y, VOLUMETRIC_SIZE_Z );

    return g_volumetric_Sampled.SampleLevel( g_volumetric_Sampler, sp, 0.0 );
}

#endif // VOLUMETRIC_HLSLI_
