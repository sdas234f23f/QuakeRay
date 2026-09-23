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
//
// HLSL counterpart of CmCas.comp.
//
// Spellings that had to change:
//   * layout(local_size_x = 64) in -> [numthreads(64, 1, 1)] on main;
//     gl_LocalInvocationID -> SV_GroupThreadID and gl_WorkGroupID -> SV_GroupID, which HLSL only
//     hands to the entry point, so main converts the two builtins into the localID / groupID
//     parameters of applyCas and applySimpleSharp. The golden reads them inside those functions,
//     HLSL has no builtin a helper can read, and the values are the golden's
//   * layout(constant_id = 0) const uint isSourcePing and layout(constant_id = 1) const uint
//     useSimpleSharp -> [[vk::constant_id(0)]] const uint and [[vk::constant_id(1)]] const uint,
//     the same two spec constants with the same defaults
//   * layout(push_constant) uniform CasPush_BT { uvec4 con0; uvec4 con1; } push -> the struct
//     CasPush_BT with the same two members and [[vk::push_constant]] ConstantBuffer<CasPush_BT>
//     push, which lands them on offsets 0 and 16 on both halves
//   * #define A_GLSL 1 -> #define A_HLSL 1: CAS/ffx_a.h is the two language header of the AMD
//     portability layer, and its HLSL section provides the same AF3, ASU2, AU2, ARmp8x8 and
//     friends that the GLSL section provides for the golden. CAS/ffx_cas.h is included behind the
//     same five definitions (loadInput, storeOutput, CasLoad, CasInput and the two defines), so
//     the CAS implementation itself is the very same text on both halves
//   * texelFetch(t, p, 0) -> t.Load(int3(p, 0)) in loadInput, and imageStore(img, p, color) ->
//     img[p] = color in storeOutput
//   * the hexadecimal suffixes of the shifts keep their spelling ((gl_WorkGroupID.x << 4u) ->
//     (groupID.x << 4u)) and the ASU2(...) / AU2(...) constructors of the AMD header keep theirs,
//     both being HLSL constructors of the renamed types
//
// What did not change: the whole CAS integration body, its four 8x8 tiles per workgroup and the
// 2x2 foot-print loop of applyCas, the sharpening-only flag, the simple sharp path with its
// (1 + 4w) center and -w ring, the AF4(r, 0) of sharp, the two spec constant branches of main
// and the AMD headers themselves, which were already written for both languages.

#define DESC_SET_FRAMEBUFFERS 0
#include "ShaderCommonHLSLFunc.hlsli"

// The GLSL original declares the workgroup size here; in HLSL it is an attribute of the entry
// point instead, so the shader itself carries [numthreads(64, 1, 1)].

[[vk::constant_id(0)]] const uint isSourcePing = 0;
[[vk::constant_id(1)]] const uint useSimpleSharp = 0;

struct CasPush_BT
{
    uint4 con0;
    uint4 con1;
};

[[vk::push_constant]] ConstantBuffer<CasPush_BT> push;

    // CAS impl
    #define A_GPU 1
    #define A_HLSL 1
    #include "CAS/ffx_a.h"

    // CAS should operate in linear space 
    AF3 loadInput(const ASU2 p) 
    {    
        if (isSourcePing != 0)
        {
            return framebufUpscaledPing_Sampled.Load(int3(p, 0)).rgb;
        }
        else
        {
            return framebufUpscaledPong_Sampled.Load(int3(p, 0)).rgb;
        }               
    }

    void storeOutput(const ASU2 p, const AF4 color)
    {
        if (isSourcePing != 0)
        {
            framebufUpscaledPong[p] = color;
        }
        else
        {
            framebufUpscaledPing[p] = color;
        }            
    }

    AF3 CasLoad(const ASU2 p) 
    {
        return loadInput(p);
    }
    void CasInput(inout AF1 r,inout AF1 g,inout AF1 b){}
    #include "CAS/ffx_cas.h"

void applyCas(uint3 groupID, uint3 localID)
{
    const bool sharpenOnly = true;

    AU2 gxy=ARmp8x8(localID.x)+AU2(groupID.x<<4u,groupID.y<<4u);

    AF4 c;
    CasFilter(c.r,c.g,c.b,gxy,push.con0,push.con1,sharpenOnly);
    storeOutput(ASU2(gxy), c);

    gxy.x+=8u;
    CasFilter(c.r,c.g,c.b,gxy,push.con0,push.con1,sharpenOnly);
    storeOutput(ASU2(gxy), c);

    gxy.y+=8u;
    CasFilter(c.r,c.g,c.b,gxy,push.con0,push.con1,sharpenOnly);
    storeOutput(ASU2(gxy), c);

    gxy.x-=8u;
    CasFilter(c.r,c.g,c.b,gxy,push.con0,push.con1,sharpenOnly);
    storeOutput(ASU2(gxy), c);
}

AF4 sharp(const ASU2 p)
{
    const float w = 1.0;

    const AF3 r = 
        loadInput(p             )  * (1 + 4 * w) +
        loadInput(p + ASU2( 0, 1)) * (-w) +
        loadInput(p + ASU2( 1, 0)) * (-w) +
        loadInput(p + ASU2( 0,-1)) * (-w) +
        loadInput(p + ASU2(-1, 0)) * (-w);

    return AF4(r, 0);
}

void applySimpleSharp(uint3 groupID, uint3 localID)
{
    ASU2 gxy = ASU2(ARmp8x8(localID.x) + AU2(groupID.x << 4u, groupID.y << 4u));
    storeOutput(gxy, sharp(gxy));

    gxy.x += 8;
    storeOutput(gxy, sharp(gxy));

    gxy.y += 8;
    storeOutput(gxy, sharp(gxy));

    gxy.x -= 8;
    storeOutput(gxy, sharp(gxy));
}

[numthreads(64, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID)
{
    if (useSimpleSharp != 0)
    {
        applySimpleSharp(groupID, localID);
    }
    else 
    {
        applyCas(groupID, localID);
    }
}
