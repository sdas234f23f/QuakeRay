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


// HLSL counterpart of RsWorld.frag. The stage is a fragment shader, its entry point keeps the
// golden's name main, and the two outputs of the golden become the members of the returned struct.
//
// Spellings that had to change:
//   * the two input variables and the two output variables move into the entry point signature:
//     an HLSL parameter and an output struct member need a semantic, and [[vk::location(n)]]
//     pins the number the golden writes in layout(location = n). vertColor is at 0, vertTexCoord
//     at 1, outColor is SV_Target0 at 0 and outScreenEmission SV_Target1 at 1, which is the same
//     interface the checker compares against the golden
//   * gl_FragCoord.xyz / gl_FragCoord.xy -> fragCoord, an SV_Position input parameter: dxc
//     declares the same BuiltIn FragCoord as glslang does for the GLSL builtin
//   * ivec2( gl_FragCoord.xy ) -> int2( fragCoord.xy ): both truncate towards zero, and the
//     fragment coordinates are not negative here
//   * imageLoad( framebufPrimaryToReflRefr, pix ) -> framebufPrimaryToReflRefr[pix] and
//     imageStore( framebufPrimaryToReflRefr, pix, v ) -> framebufPrimaryToReflRefr[pix] = v, as
//     the generated header declares the view as RWTexture2D. The uint4 constructor of the golden
//     is copied as it is, its four components are the golden's three plus emisBlendCode
//   * textureLod( sampler3D( g_illuminationVolume_Sampled, g_illuminationVolume_Sampler ), sp, 0.0 )
//     -> g_illuminationVolume_Sampled.SampleLevel( g_illuminationVolume_Sampler, sp, 0.0 ): the
//     descriptors are split into a sampled view and a sampler
//   * layout(push_constant) with layout(offset = 64), 80 and 84 -> the same member offsets via
//     [[vk::offset]], because dxc packs members on its own rules and the golden's block leaves the
//     first 64 bytes to the vertex stage part of the same push constant range
//   * layout( constant_id = 0 ) const uint alphaTest -> [[vk::constant_id(0)]] const uint
//   * the two products of the ILLUMINATION_VOLUME branch keep the golden's order of operands:
//     globalUniform.invView * globalUniform.invProjection * ndc becomes
//     mul( mul( globalUniform.invView, globalUniform.invProjection ), ndc ), which is what the
//     transposed declaration of the matrices means by that expression. The branch is dead on both
//     halves -- ShaderCommonGLSLFunc.h and ShaderCommonHLSLFunc.hlsli both define
//     ILLUMINATION_VOLUME 0 -- so the checker never compiles it; the spelling is transcribed for
//     the day the switch flips, and the product itself is measured on a scratch probe with literal
//     matrices
//
// What did not change: the whole body of main with its order of statements, the checks
// (emissionTextureIndex != MATERIAL_NO_TEXTURE, alphaTest != 0, emisBlendCode != 0u), the
// arithmetic of the color (rasterizerFragInfo.color * vertColor * albedoAlpha), the emission
// conversion uint( emisSample.a * 255.0 + 0.5 ), the scaled emission
// rmeEmissionToScreenEmission( emis ) * albedoAlpha.rgb, the correction of outColor.rgb by
// ev100ToLuminousExposure( getCurrentEV100() ), and the two getTextureSample calls with the same
// arguments. ALPHA_THRESHOLD and EMISSION_CHANNEL keep the golden's names and values.
//
// The two headers of the golden are included here in the golden's order: ShaderCommonGLSLFunc.h ->
// ShaderCommonHLSLFunc.hlsli, Exposure.h -> Exposure.hlsli, Volumetric.h -> Volumetric.hlsli. Like
// the GLSL ones they include nothing themselves and expect the DESC_SET_* macros to be defined
// before them, which is why the same five defines are repeated here.

#define DESC_SET_TEXTURES       0
#define DESC_SET_GLOBAL_UNIFORM 1
#define DESC_SET_TONEMAPPING    2
#define DESC_SET_VOLUMETRIC     3
#define DESC_SET_FRAMEBUFFERS   4
#include "ShaderCommonHLSLFunc.hlsli"
#include "Exposure.hlsli"
#include "Volumetric.hlsli"

struct RasterizerFrag_BT
{
    [[vk::offset(64)]] float4 color;
    [[vk::offset(80)]] uint   textureIndex;
    [[vk::offset(84)]] uint   emissionTextureIndex;
};

[[vk::push_constant]] ConstantBuffer<RasterizerFrag_BT> rasterizerFragInfo;

[[vk::constant_id(0)]] const uint alphaTest = 0;

#define ALPHA_THRESHOLD 0.5
#define EMISSION_CHANNEL 2

struct RsWorldFragOutput
{
    [[vk::location(0)]] float4 outColor          : SV_Target0;
    [[vk::location(1)]] float3 outScreenEmission : SV_Target1;
};

RsWorldFragOutput main( [[vk::location(0)]] float4 vertColor    : TEXCOORD0,
                        [[vk::location(1)]] float2 vertTexCoord : TEXCOORD1,
                        float4 fragCoord : SV_Position )
{
    RsWorldFragOutput o;

    float4 albedoAlpha = getTextureSample( rasterizerFragInfo.textureIndex, vertTexCoord );
    o.outColor         = rasterizerFragInfo.color * vertColor * albedoAlpha;

#if ILLUMINATION_VOLUME
    float4 ndc = float4( fragCoord.xyz, 1.0 );
    ndc.xy   /= float2( globalUniform.renderWidth, globalUniform.renderHeight );
    ndc.xy = ndc.xy * 2.0 - 1.0;

    float4 worldpos = mul( mul( globalUniform.invView, globalUniform.invProjection ), ndc );
    worldpos.xyz /= worldpos.w;

    float3 sp = volume_toSamplePosition_T(
        worldpos.xyz, globalUniform.volumeViewProj, globalUniform.cameraPosition.xyz );
    float3 illum = g_illuminationVolume_Sampled.SampleLevel(
        g_illuminationVolume_Sampler, sp, 0.0 ).rgb;

    o.outColor.rgb *= illum;
#else
    o.outColor.rgb *= ev100ToLuminousExposure( getCurrentEV100() );
#endif

    float emis            = 0.0;
    uint  emisBlendCode   = 0u;
    if( rasterizerFragInfo.emissionTextureIndex != MATERIAL_NO_TEXTURE )
    {
        const float4 emisSample = getTextureSample( rasterizerFragInfo.emissionTextureIndex,
                                                    vertTexCoord );
        emis          = emisSample[ EMISSION_CHANNEL ];

        // Per-material rt_emis_blend override: the CPU packs it into the alpha of the
        // RME texture as ( mode + 1 ), while 0 means "not authored".
        emisBlendCode = uint( emisSample.a * 255.0 + 0.5 );
    }
    o.outScreenEmission = rmeEmissionToScreenEmission( emis ) * albedoAlpha.rgb;

    if( alphaTest != 0 )
    {
        if( o.outColor.a < ALPHA_THRESHOLD )
        {
            discard;
        }
    }

    // The compose pass reads the per-pixel mode from the alpha of
    // framebufPrimaryToReflRefr, where the traced path stores it for the surfaces
    // that it covers. Rasterized emissive surfaces must contribute the same value.
    if( emisBlendCode != 0u )
    {
        // The traced pass addresses this framebuffer in checkerboard space, so the
        // rasterized pixel has to be mapped the same way for the compose pass to
        // find the mode at the pixel it resolves.
        const int2  pix  = getCheckerboardPix( int2( fragCoord.xy ) );
        const uint4 prev = framebufPrimaryToReflRefr[ pix ];
        framebufPrimaryToReflRefr[ pix ] = uint4( prev.rgb, emisBlendCode );
    }

    return o;
}
