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

// HLSL counterpart of Surface.inl. The GLSL file includes only BRDF.h, which is BRDF.hlsli here,
// and this one keeps that single include, so the shader has to pull in the rest itself, in the
// order RaygenCommon.h uses:
//   * ShaderCommonHLSLFunc.hlsli for isSkyPix, getRegularPixFromCheckerboardPix, texelFetchNormal,
//     texelFetchNormal_Prev, texelFetchNormalGeometry and texelFetchNormalGeometry_Prev
//   * Structs.hlsli for ShHitInfo and Utils.hlsli for getLuminance
//   * the generated framebuffers, which need DESC_SET_FRAMEBUFFERS to be defined
//   * DESC_SET_GLOBAL_UNIFORM, because that is what defines CHECKERBOARD_FULL_WIDTH and
//     CHECKERBOARD_FULL_HEIGHT for the two helpers above: without both, the two fetch functions
//     below do not exist at all
// FRAMEBUF_IGNORE_ATTACHMENTS has to stay undefined, and CHECKERBOARD_SEPARATOR_DIVISOR comes from
// ShaderCommonHLSL.hlsli.
//
// There is no matrix product, no matrix index, no matrix constructor and no transpose anywhere in
// Surface.inl, so none of the matrix rules of ShaderCommonHLSL.hlsli applies to this port. The only
// single element indices are metallicRoughness[0] and [1], which are components of a float2 and not
// a column of a matrix.
//
// Spellings that had to change:
//   * texelFetch(t, pix, 0) became t.Load(int3(pix, 0)): Load takes the texel coordinate only and
//     has no mip level argument, and the type of the texture decides the result type, so
//     framebufQ2Cluster_Sampled.Load(...).r still yields the uint that Surface::cluster wants
//   * floatBitsToUint became asuint
//   * vec3(1.0) and vec3(0.0) became (float3)1.0 and (float3)0.0, as HLSL does not broadcast a
//     scalar into a constructor
//
// What did not change: the four way compile time gating, the early return of the sky branch (and
// the members it leaves uninitialized, which stay uninitialized here for the same reason), the
// order in which the members are filled in, the swizzles, and the comment about the albedo layout.


#ifndef SURFACE_HLSLI_
#define SURFACE_HLSLI_
#include "BRDF.hlsli"


struct Surface
{
    float3  position;
    uint    instCustomIndex;
    float3  normalGeom;
    float   roughness;
    float3  normal;
    float3  albedo;
    bool    isSky;
    float3  specularColor;
    float   emission;
    float3  toViewerDir;
    uint    cluster;
};


#if defined(DESC_SET_FRAMEBUFFERS)
#if !defined(FRAMEBUF_IGNORE_ATTACHMENTS)
#if defined(CHECKERBOARD_FULL_WIDTH) && defined(CHECKERBOARD_FULL_HEIGHT)

Surface fetchGbufferSurface(const int2 pix)
{
    Surface s;
    s.isSky = isSkyPix(pix);

    if (s.isSky)
    {
        return s;
    }
    
    // framebufAlbedo ALWAYS uses regular layout because of the sky rasterization pass  
    s.albedo = framebufAlbedo_Sampled.Load(int3(getRegularPixFromCheckerboardPix(pix), 0)).rgb;
    s.emission = getLuminance(framebufScreenEmisRT_Sampled.Load(int3(getRegularPixFromCheckerboardPix(pix), 0)).rgb);   
    {
        float4 posEnc           = framebufSurfacePosition_Sampled.Load(int3(pix, 0));
        s.position              = posEnc.xyz;
        s.instCustomIndex       = asuint(posEnc.a);
    }
    {
        float2 metallicRoughness  = framebufMetallicRoughness_Sampled.Load(int3(pix, 0)).xy;
        s.specularColor           = getSpecularColor(s.albedo, metallicRoughness[0]);
        s.roughness               = metallicRoughness[1];
    }
    s.normalGeom                = texelFetchNormalGeometry(pix);
    s.normal                    = texelFetchNormal(pix);
    s.toViewerDir               = -framebufViewDirection_Sampled.Load(int3(pix, 0)).xyz;
    s.cluster                   = framebufQ2Cluster_Sampled.Load(int3(pix, 0)).r;
    return s;
}

Surface fetchGbufferSurface_NoAlbedoViewDir_Prev(const int2 pix)
{
    Surface s;
    s.isSky = false;
    s.albedo = (float3)1.0;
    s.emission = 0.0;
    {
        float4 posEnc           = framebufSurfacePosition_Prev_Sampled.Load(int3(pix, 0));
        s.position              = posEnc.xyz;
        s.instCustomIndex       = asuint(posEnc.a);
    }
    {
        float2 metallicRoughness  = framebufMetallicRoughness_Prev_Sampled.Load(int3(pix, 0)).xy;
        s.specularColor           = getSpecularColor(s.albedo, metallicRoughness[0]);
        s.roughness               = metallicRoughness[1];
    }
    s.normalGeom                = texelFetchNormalGeometry_Prev(pix);
    s.normal                    = texelFetchNormal_Prev(pix);
    s.toViewerDir               = (float3)0.0;
    return s;
}
#endif // CHECKERBOARD_FULL_WIDTH && CHECKERBOARD_FULL_HEIGHT
#endif // !FRAMEBUF_IGNORE_ATTACHMENTS
#endif // DESC_SET_FRAMEBUFFERS
       

Surface hitInfoToSurface_Indirect(const ShHitInfo h, const float3 rayDirection)
{
    Surface s;
    s.position = h.hitPosition;
    s.instCustomIndex = h.instCustomIndex;
    s.normalGeom = h.normalGeom;
    s.roughness = h.roughness;
    s.normal = h.normalGeom; // ignore precise normals for indirect
    s.albedo = h.albedo;
    s.isSky = false;
    s.specularColor = getSpecularColor(h.albedo, h.metallic);
    s.emission = h.emission;
    s.toViewerDir = -rayDirection;
    s.cluster = h.cluster;
    return s;
}

#endif // SURFACE_HLSLI_
