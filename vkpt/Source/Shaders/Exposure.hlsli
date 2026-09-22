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


// HLSL counterpart of Exposure.h. Like the GLSL one it includes nothing itself: the shader has to
// pull in ShaderCommonHLSLFunc.hlsli first, which declares tonemapping under
// #ifdef DESC_SET_TONEMAPPING and -- through ShaderCommonHLSL.hlsli -- Utils.hlsli, where square
// comes from. CmPrepareFinal.comp and RsWorld.frag, the consumers of the GLSL one, do the same:
// they include ShaderCommonGLSLFunc.h before Exposure.h.
//
// Spellings that had to change:
//   * tonemapping.avgLuminance -> tonemapping[0].avgLuminance: ShaderCommonHLSLFunc.hlsli declares
//     the same bytes as a one element StructuredBuffer, so the single instance that the GLSL block
//     has an implicit name for is its element 0. This is the only resource access of the header, it
//     reads no other field of the block and it writes nothing
//   * the #ifndef EXPOSURE_H_ / #define EXPOSURE_H_ guard -> #pragma once, as in BRDF.hlsli
//
// What did not change: the five functions with their names, their parameter lists and the order of
// the operands of every product, the implicit conversions of the golden (the int 100 that
// getManualEV100 passes to a float parameter, the int 100 that initializes S, and the int 100 and
// 1.0 / 16.0 / 1.0 / 125.0 of getCurrentEV100) are converted by dxc the way glslc converts them, and
// the ternary of getCurrentEV100 stays a ternary over the local false of the golden, log2, exp2,
// max and square being the same intrinsics on both sides. The file holds no matrix and no vector
// expression at all -- the golden has no mat, no index, no swizzle and no product of a matrix -- so
// none of the matrix rules of ShaderCommonHLSL.hlsli applies to it.
//
// The #ifndef DESC_SET_TONEMAPPING guard of the golden is copied verbatim, its message included.
//
// Exposure.h declares no descriptor of its own, so this file is pinned by the probe pair
// GLSL/Exposure.probe.comp <-> Probes/Exposure.probe.comp.hlsl.

#ifndef EXPOSURE_HLSLI_
#define EXPOSURE_HLSLI_
#ifndef DESC_SET_TONEMAPPING
    #error DESC_SET_TONEMAPPING is required
#endif

float getManualEV100( float aperture, float shutterTime, float iso )
{
    return log2( square( aperture ) / shutterTime * 100.0 / iso );
}

float getAutoEV100()
{
    const float lumAverage = max( 0.0, tonemapping[0].avgLuminance );
    const float S          = 100;
    const float K          = 12.5;
    return log2( lumAverage * S / K );
}

float getCurrentEV100()
{
    bool manual = false;
    return manual ? getManualEV100( 1.0 / 16.0, 1.0 / 125.0, 100 ) : getAutoEV100();
}

float ev100ToLuminousExposure( float ev100 )
{
    float maxLuminance = 1.2 * exp2( ev100 );
    return maxLuminance > 0.0 ? 1.0 / maxLuminance : 0.0;
}

float ev100ToLuminance( float ev100 )
{
    return exp2( ev100 - 3 );
}

#endif // EXPOSURE_HLSLI_
