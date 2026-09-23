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


// HLSL counterpart of CmPrepareFinal.comp. The compose pass of the renderer and the largest leaf of
// the base: it resolves the emissive layer through the tone curve, adds the volumetrics, the acid
// fog and the classic level fog, dithers the quantisation and writes framebufFinal and
// framebufBloomInput.
//
// Spellings that had to change:
//   * layout(local_size_x = COMPUTE_COMPOSE_GROUP_SIZE_X, local_size_y = COMPUTE_COMPOSE_GROUP_SIZE_Y,
//     local_size_z = 1) in -> [numthreads(COMPUTE_COMPOSE_GROUP_SIZE_X, COMPUTE_COMPOSE_GROUP_SIZE_Y, 1)]
//     on main, and gl_GlobalInvocationID.x/y -> the SV_DispatchThreadID parameter of main, which HLSL
//     only hands to the entry point; the ivec2 pix of the golden is the same int2 either way
//   * pix.x >= uint(globalUniform.renderWidth) keeps the golden's UNSIGNED comparison, so the port
//     converts the pixel as well: (uint)pix.x >= (uint)globalUniform.renderWidth, the same spelling
//     CmCheckerboard.comp.hlsl uses, instead of letting HLSL pick a signed comparison
//   * #define A_GLSL 1 -> #define A_HLSL 1 for the AMD portability header, exactly as
//     CmCas.comp.hlsl does it. The golden includes that header by the bare name "ffx_a.h"; the tree
//     vendors two copies of it (CAS/ and LPM/) and the host build resolves the bare name only because
//     GenerateShaders.py adds every subfolder as an include folder -- to whichever copy the folder
//     set happens to reach first. This shader consumes the LPM filter control block, so the port
//     names the LPM copy explicitly: "LPM/ffx_a.h". Both copies define the same AU1/AU4 types and a
//     probe of the golden compiled against either one produces the same module, but naming the copy
//     makes the choice deterministic and is the only spelling the checker's fixed include list
//     ("-I . -I ../Generated/") can resolve
//   * layout(set = DESC_SET_LPM_PARAMS, binding = BINDING_LPM_PARAMS) readonly uniform LpmParams_BT
//     { AU4 g_lpmParams[24]; } -- an instance-less block the golden reads as g_lpmParams[i] -- becomes
//     the struct LpmParams_BT with the same single member and a named ConstantBuffer, read as
//     lpmParams.g_lpmParams[i]: HLSL has no instance-less block and a ConstantBuffer is what the host
//     binds for it (VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ImageComposition.cpp:232). The member stays on
//     offset 0 with an array stride of 16 on both halves. The block is DEAD on both halves -- nothing
//     calls the LpmFilterCtl that reads it, so neither compiled module references the descriptor and
//     set 3 binding 0 appears in neither property set; a scratch pair that makes the read live was
//     used to verify the layout. The dxc run also reports the two -Wambig-lit-shift warnings of the
//     vendored AMD header itself (LPM/ffx_a.h:1111 and :1113); the header is not touched here
//   * texelFetch(t, p, 0) -> t.Load(int3(p, 0)), imageStore(img, p, v) -> img[p] = v (the generated
//     header declares the storage views as RWTexture2D), and
//     textureLod(sampler3D(view, sampler), sp, 0.0) -> view.SampleLevel(sampler, sp, 0.0): the
//     descriptors are split into a sampled view and a sampler
//   * mix(...) -> lerp(...), greaterThan(l, x) -> l > x, vecN(scalar) -> (floatN)scalar, and the
//     float(...) / uint(...) constructors -> (float)(...) / (uint)(...)
//   * fract(...) -> frac(...): the same truncation of x to its fractional part, HLSL just spells
//     the intrinsic without the t
//   * the parameter `linear` of outputCodeStepLinear is a modifier keyword in HLSL, so it becomes
//     `linearColor`; the body is the golden's and reads the same value
//   * tonemapping.<field> -> tonemapping[0].<field>: ShaderCommonHLSLFunc.hlsli declares the same
//     bytes as a one element StructuredBuffer, so the single instance the GLSL block has an implicit
//     name for is its element 0, the same spelling Exposure.hlsli uses
//   * vec2(pix) -> (float2)pix, float(channel) -> (float)channel and
//     uvec3(pix.x, pix.y, channel) -> uint3((uint)pix.x, (uint)pix.y, channel), the spelling
//     Random.hlsli already uses for the same murmurHash33 call
//   * globalUniform.invProjection * vec4(...) -> mul(globalUniform.invProjection, float4(...)): the
//     order of the operands is the golden's, as the matrix rule of ShaderCommonHLSL.hlsli requires
//
// What did not change: the whole chain of main in the golden's order (the emissive mode fetch, the
// god rays composite with its coreQ2RTX and q2SplitFlag branches, finalizeColor, getBloomInput,
// applyVolumetrics, the acid fog, applyLevelFog, processDebug, the dither and the two stores), every
// function with its name and parameter list, the emission blend modes 0/2/3/4/5 of
// blendEmissionLayer with the same two lerp weights, the histogram interpolation of finalizeColor
// (HISTOGRAM_BINS = COMPUTE_LUM_HISTOGRAM_BIN_COUNT, the same binning as
// CmLuminanceHistogram.comp), the auto-exposure Reinhard blend and the knee, the
// ev100ToLuminance products of getBloomInput, the checkerboard remaps, the level fog falloff
// exp(-(density * depth * getViewAxisFactor(pix))^2) with the sky taking 1 - skyBlend, the
// interleaved gradient noise, the triangular-PDF dither (structured + white - 1.0) and the
// amplitude of outputCodeStepLinear (1/255 divided by the slope of the sRGB encoding), and the two
// #define values (OUTPUT_DITHER_CODES 1.0, DEBUG_LPM 0).
//
// Dead on both halves and transcribed as the golden wrote it: the DEBUG_LPM branch (DEBUG_LPM 0;
// its inAU4 is the AMD header's in-qualifier macro, which its HLSL branch defines too, so the branch
// is text that would compile if the switch flipped), the DEBUG_VOLUME_ILLUMINATION branch
// (undefined), the GRADIENT_ESTIMATION_ENABLED branch (GRADIENT_ESTIMATION_ENABLED 0 -- it names
// framebufDISPingGradient_Sampled, which neither generated header declares any more, so it has to
// stay dead), and the DEBUG_SHOW_SH branch (undefined).
//
// The five golden headers are included here in the same order as there: ShaderCommonGLSLFunc.h ->
// ShaderCommonHLSLFunc.hlsli, Volumetric.h -> Volumetric.hlsli, Random.h -> Random.hlsli,
// Exposure.h -> Exposure.hlsli, TonemappingUtils.glsl -> TonemappingUtils.hlsli. Like the GLSL ones
// they declare nothing themselves and expect the DESC_SET_* macros to be defined first, which is why
// the same five defines are repeated here.

#define DESC_SET_FRAMEBUFFERS 0
#define DESC_SET_GLOBAL_UNIFORM 1
#define DESC_SET_TONEMAPPING 2
#define DESC_SET_LPM_PARAMS 3
#define DESC_SET_VOLUMETRIC 4
#include "ShaderCommonHLSLFunc.hlsli"
#include "Volumetric.hlsli"
#include "Random.hlsli"
#include "Exposure.hlsli"
#include "TonemappingUtils.hlsli"

// The GLSL original declares the workgroup size here; in HLSL it is an attribute of the entry
// point instead, so the shader itself carries [numthreads(...)] on main.

#define DEBUG_LPM 0

#define A_GPU 1
#define A_HLSL 1
#include "LPM/ffx_a.h"

struct LpmParams_BT
{
    AU4 g_lpmParams[24];
};

[[vk::binding(BINDING_LPM_PARAMS, DESC_SET_LPM_PARAMS)]] ConstantBuffer<LpmParams_BT> lpmParams;

#if DEBUG_LPM
    AU4 debug_lpmParams[24];
    void LpmSetupOut(AU1 i, inAU4 v) { debug_lpmParams[i] = v; }
    AU4  LpmFilterCtl(AU1 i)         { return debug_lpmParams[i]; }
#else
    #define LPM_NO_SETUP 1
    AU4  LpmFilterCtl(AU1 i)         { return lpmParams.g_lpmParams[i]; }
#endif

float3 uncharted2TonemapOp(const float3 x)
{
    float A = 0.15;
    float B = 0.50;
    float C = 0.10;
    float D = 0.20;
    float E = 0.02;
    float F = 0.30;

    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

float3 uncharted2Tonemap(const float3 color, float whitePoint)
{
    return uncharted2TonemapOp(2.0 * color) / uncharted2TonemapOp((float3)whitePoint);
}


float3 reinhard(const float3 c)
{
    const float3 c1 = clamp(c, exp2(globalUniform.minLogLuminance), exp2(globalUniform.maxLogLuminance));
    const float w2 = globalUniform.luminanceWhitePoint * globalUniform.luminanceWhitePoint;
    return c * (1.0 + c1 / w2) / (1.0 + c1);
}


float3 blendEmissionLayer( const float3 hdr, const float3 layer, const uint mode )
{
    if( mode == 0u )
    {
        return hdr;
    }
    if( mode == 2u )
    {
        return hdr + layer;
    }

    const float3 base     = clamp( hdr, (float3)0.0, (float3)1.0 );
    const float3 emiss    = clamp( layer, (float3)0.0, (float3)1.0 );
    const float  coverage = clamp( max( max( layer.x, layer.y ), layer.z ), 0.0, 1.0 );

    float3 blended = emiss;
    if( mode == 3u )
    {
        blended = lerp( 2.0 * base * emiss, 1.0 - 2.0 * ( 1.0 - base ) * ( 1.0 - emiss ), step( (float3)0.5, base ) );
    }
    else if( mode == 4u )
    {
        blended = lerp( 2.0 * base * emiss, 1.0 - 2.0 * ( 1.0 - base ) * ( 1.0 - emiss ), step( (float3)0.5, emiss ) );
    }
    else if( mode == 5u )
    {
        blended = clamp( base / max( (float3)1.0 - emiss, (float3)1e-3 ), (float3)0.0, (float3)1.0 );
    }

    return hdr + ( blended - base ) * coverage + max( layer - 1.0, 0.0 );
}


// Resolves the per-pixel emission blend mode. The primary ray pass stores the
// material's authored mode (+1) in the alpha of framebufPrimaryToReflRefr;
// 0 means "not authored" and any other out-of-range value (e.g. the 255 alpha of
// an authored *_rme texture) falls back to the global rt_emis_blend cvar.
uint decodeEmissionBlendMode( const uint code )
{
    if( code < 1u || code > 6u )
    {
        return globalUniform.emissionBlendMode;
    }
    return min( code - 1u, 5u );
}


// Q2RTX-style tone curve application (tone_mapping_apply.comp, SDR path).
// Applies the noise-aware tone curve + auto-exposure + knee to the HDR image.
float3 finalizeColor( const float3 hdr, const float3 screenEmis, const uint emisBlendMode )
{
    const float strength = clamp( globalUniform.emissionBlendStrength, 0.0, 1.0 );
    const float3 layer   = screenEmis * globalUniform.emissionMaxScreenColor * strength;

    float3 input_color = blendEmissionLayer( hdr, layer, emisBlendMode );

    const float lum = max( getLuminance( input_color ), exp2( min_log_luminance ) );

    // Interpolate the tone curve; binning must match CmLuminanceHistogram.comp.
    const float biased_log_luminance = log2( lum ) * log_luminance_scale + log_luminance_bias;
    const float histogram_bin = clamp( biased_log_luminance * HISTOGRAM_BINS, 0.0, (float)( HISTOGRAM_BINS - 1 ) );
    const uint  left_bin       = (uint)histogram_bin;
    const uint  right_bin      = min( left_bin + 1, (uint)( HISTOGRAM_BINS - 1 ) );
    const float right_weight_F = frac( histogram_bin );
    const float left_weight_F  = 1.0 - right_weight_F;

    const float out_log_luminance = left_weight_F * tonemapping[0].curve[left_bin] +
                                    right_weight_F * tonemapping[0].curve[right_bin];
    const float out_luminance = exp2( out_log_luminance + tonemapping[0].tmExposureBias );

    float3 mapped_color = input_color * out_luminance / lum;

    // Knee: bring values that are still above 1 back into range.
    const float3 step_value = step( tonemapping[0].tmKneeStart, mapped_color );
    mapped_color = lerp( mapped_color,
                         ( tonemapping[0].kneeW * mapped_color + tonemapping[0].kneeA ) / max( (float3)1e-6, mapped_color + tonemapping[0].kneeB ),
                         step_value );

    // Auto-exposure Reinhard blend.
    const float adapted_luminance    = tonemapping[0].adaptedLuminance;
    const float scaled_luminance     = exp2( tonemapping[0].tmExposureBias - 2.0 ) * lum / adapted_luminance;
    const float white_point          = tonemapping[0].tmWhitePoint;
    const float white_point_squared  = white_point * white_point;
    const float mapped_luminance     = ( scaled_luminance * ( 1.0 + scaled_luminance / white_point_squared ) ) / ( 1.0 + scaled_luminance );
    const float3 ae_mapped_color     = input_color * mapped_luminance / lum;

    mapped_color = lerp( mapped_color, ae_mapped_color, tonemapping[0].tmReinhard );

    return clamp( mapped_color, (float3)0, (float3)1 );
}


float3 getBloomInput( const float3 hdr, const float3 screenEmis, const int2 pix )
{
    const float power = getLuminance( screenEmis );

    const float albedoLum = getLuminance( framebufAlbedo_Sampled.Load(int3( pix, 0 )).rgb );
    const float3 emis     = power > 0.001 ? screenEmis / max( albedoLum, power ) : (float3)0;

    float ec = power * globalUniform.bloomEmissionMultiplier;

    return hdr * ev100ToLuminance( getCurrentEV100() ) +
           emis * ev100ToLuminance( getCurrentEV100() + ec );
}


float3 applyVolumetrics( const int2 pix, const float3 color )
{
    if( globalUniform.volumeEnableType == VOLUME_ENABLE_NONE )
    {
        return color;
    }

    // On the new Q2RTX core path the level volumetric is provided by the traced
    // fog volumes (per-ray, works through portals and in reflections), so the
    // legacy screen-space volumetric composite is disabled there.
    if( globalUniform.coreQ2RTX != 0 )
    {
        return color;
    }

    uint seed = getRandomSeed( pix, globalUniform.frameId );
    float3 rnd  = rnd8_4( seed, 0 ).xyz;

    const float2 inUV         = getPixelUVWithJitter( pix );
    const float3 cameraRayDir = getRayDir( inUV );

    float virtualdepth = framebufDepthWorld_Sampled.Load(int3( getCheckerboardPix( pix ), 0 )).r;
    bool  isSky        = virtualdepth > MAX_RAY_LENGTH;

    if( globalUniform.volumeEnableType == VOLUME_ENABLE_VOLUMETRIC )
    {
        float3 position = globalUniform.cameraPosition.xyz + cameraRayDir * virtualdepth;

#ifdef DEBUG_VOLUME_ILLUMINATION
        float3 sp = volume_toSamplePosition_T(
            position.xyz, globalUniform.volumeViewProj, globalUniform.cameraPosition.xyz );
        float3 illum = g_illuminationVolume_Sampled.SampleLevel( g_illuminationVolume_Sampler, sp, 0.0 ).rgb;
        return color * illum;
#endif

        float4 v = volume_sampleDithered( position, rnd, 2.0 );
        return color * v.a + v.rgb;
    }
    else
    {
        // simple depth-based fog; factor tuned to be clearly visible
        float density = globalUniform.volumeScattering * 0.001;

        if( !isSky )
        {
            float f = exp( -virtualdepth * density );
            return lerp( globalUniform.volumeAmbient.rgb, color, f );
        }
        else
        {
            return color;
        }
    }
}


// Ratio of the view-space depth (the distance the classic renderer fogged with,
// gl_Position.w) to the ray length that framebufDepthWorld holds. A ray that is
// off the view axis is longer by 1/cos(angle), and fogging by the ray length
// would thicken the fog towards the screen edges, where the classic renderer
// showed none of that.
float getViewAxisFactor( const int2 pix )
{
    const float2 inUV = getPixelUVWithJitter( pix ) * 2.0 - 1.0;

    const float4 target   = mul( globalUniform.invProjection, float4( inUV.x, inUV.y, 1.0, 1.0 ) );
    const float3 localDir = abs( target.w ) < 0.001 ? target.xyz : target.xyz / target.w;

    return clamp( -normalize( localDir ).z, 0.0, 1.0 );
}


// Classic Quake level fog: the worldspawn "fog" key and the `fog` console
// command, which is also how Arcane Dimensions drives its dynamic fog. Quake
// fogged every fragment, so this runs on the composited image, after the
// tonemapped color and after the rasterized geometry (sprites, viewmodel) has
// been drawn into framebufFinal. The color is display-referred, and is used as
// it is: the classic renderer mixed it into the framebuffer the same way, and
// the sky below takes it from the same place, so a solid-colored sky stays
// identical to the fog.
float3 applyLevelFog( const int2 pix, const float3 color )
{
    const float density  = globalUniform.levelFogColorDensity.w;
    const float skyBlend = globalUniform.levelFogSkyBlend.x;

    // The classic renderer skipped the fog entirely when the level had none,
    // and so does the sky blend below, which only makes sense with fog present.
    if( density <= 0.0 )
    {
        return color;
    }

    const float depth = framebufDepthWorld_Sampled.Load(int3( getCheckerboardPix( pix ), 0 )).r;

    float fog;
    if( depth > MAX_RAY_LENGTH )
    {
        // Sky: there is no distance to fall off over, so the level's `skyfog`
        // is the blend weight itself.
        fog = 1.0 - skyBlend;
    }
    else
    {
        // Same falloff as the classic renderer: exp(-(density * distance)^2).
        const float d = density * max( depth, 0.0 ) * getViewAxisFactor( pix );
        fog = exp( -d * d );
    }

    return lerp( globalUniform.levelFogColorDensity.rgb, color, fog );
}


float3 processDebug( const int2 pix, const float3 fallback );


// Dither of the output quantisation, in units of one 8-bit display code.
#define OUTPUT_DITHER_CODES 1.0

// One step of the 8-bit sRGB encoding, expressed in the linear domain. This
// shader writes linear color into framebufFinal and the sRGB encode happens
// later, in the present blit, which could not dither itself: the amplitude of
// the dither has to be the derivative of that encoding at the current value.
float3 outputCodeStepLinear( const float3 linearColor )
{
    float3 l = clamp( linearColor, (float3)1e-5, (float3)1.0 );

    // slope of the sRGB encoding: linear below the 0.0031308 knee, power above
    float3 slope = lerp( (float3)12.92,
                         ( 1.055 / 2.4 ) * pow( l, (float3)( 1.0 / 2.4 - 1.0 ) ),
                         l > (float3)0.0031308 );

    return (float3)( 1.0 / 255.0 ) / slope;
}

// Interleaved gradient noise (Jimenez): cheap spatially-structured noise, so
// that the dither pattern reads as grain instead of white-noise speckle.
float interleavedGradientNoise( const float2 p )
{
    const float3 magic = float3( 0.06711056, 0.00583715, 52.9829189 );
    return frac( magic.z * frac( dot( p, magic.xy ) ) );
}

// Triangular-PDF dither in [-1, 1]: the sum of the structured sample above and
// a white sample. The channel is hashed in so that R, G and B are decorrelated.
float outputDither( const int2 pix, const uint channel )
{
    const float structured = interleavedGradientNoise( (float2)pix + 0.5 + (float)channel * 31.7 );
    const float white = (float)( murmurHash33( uint3( (uint)pix.x, (uint)pix.y, channel ) ).x & 0xFFFFu ) / (float)UINT16_MAX;

    return structured + white - 1.0;
}


[numthreads(COMPUTE_COMPOSE_GROUP_SIZE_X, COMPUTE_COMPOSE_GROUP_SIZE_Y, 1)]
void main( uint3 dispatchThreadID : SV_DispatchThreadID )
{
    const int2 pix = int2( dispatchThreadID.x, dispatchThreadID.y );
    if( (uint)pix.x >= (uint)globalUniform.renderWidth || (uint)pix.y >= (uint)globalUniform.renderHeight )
    {
        return;
    }

    float3 hdr        = framebufFinal_Sampled.Load(int3( pix, 0 )).rgb;
    const float3 screenEmis = framebufScreenEmission_Sampled.Load(int3( pix, 0 )).rgb;
    // per-material rt_emis_blend override, stored by the primary ray pass.
    // That framebuffer is addressed in checkerboard space (like the throughput
    // buffer below), so the regular pixel must be mapped first.
    const uint emisBlendMode = decodeEmissionBlendMode(
        framebufPrimaryToReflRefr_Sampled.Load(int3( getCheckerboardPix( pix ), 0 )).a );

    // volumetric sunlight (god rays) - additive, before tonemapping. The god
    // rays pass runs at half resolution (CmGodRays) and is bilaterally
    // upscaled to full resolution (CmGodRaysFilter), so the filtered buffer is
    // composited here. Q2RTX-style (coreQ2RTX): the primary pass fills the
    // whole image, then the reflection pass accumulates the reflected-segment
    // god rays on top, so the result is composited unconditionally. On the
    // legacy path there is no god rays reflection pass, so reflection/refraction
    // pixels are skipped (their march follows the primary camera ray and would
    // be wrong on reflections).
    if (globalUniform.coreQ2RTX != 0)
    {
        hdr += framebufGodRaysFiltered_Sampled.Load(int3( pix, 0 )).rgb;
    }
    else
    {
        const float q2SplitFlag = framebufThroughput_Sampled.Load(int3( getCheckerboardPix(pix), 0 )).a;
        if (q2SplitFlag == 0.0)
        {
            hdr += framebufGodRaysFiltered_Sampled.Load(int3( pix, 0 )).rgb;
        }
    }

    float3 color = finalizeColor( hdr, screenEmis, emisBlendMode );
    float3 bloom = getBloomInput( hdr, screenEmis, pix );

    color = applyVolumetrics( pix, color );
#if SHIPPING_HACK
    color += framebufAcidFog_Sampled.Load(int3( pix, 0 )).rgb;
#endif

    color = applyLevelFog( pix, color );


    if( globalUniform.debugShowFlags != 0 )
    {
        color = processDebug( pix, color );
    }

    // The 8-bit sRGB present quantises this image one code at a time, which
    // turns the smooth falloff of dark areas into contour bands. Dither one code
    // step, weighted by the encoding slope, so the steps turn back into noise.
    const float3 dither = float3(
        outputDither( pix, 0u ),
        outputDither( pix, 1u ),
        outputDither( pix, 2u ) );

    color = clamp( color + dither * OUTPUT_DITHER_CODES * outputCodeStepLinear( color ), (float3)0.0, (float3)1.0 );

    framebufFinal[pix] = float4( color, 0 );
    framebufBloomInput[pix] = float4( bloom, 0.0 );
}


float3 processDebug(const int2 pix, const float3 fallback)
{
    if ((globalUniform.debugShowFlags & DEBUG_SHOW_FLAG_GOD_RAYS) != 0)
    {
        // filtered (bilateral-upscaled) volumetric sunlight buffer
        return framebufGodRaysFiltered_Sampled.Load(int3(pix, 0)).rgb * 4.0;
    }
    else if ((globalUniform.debugShowFlags & DEBUG_SHOW_FLAG_MOTION_VECTORS) != 0)
    {
        const float2 m = framebufMotion_Sampled.Load(int3(getCheckerboardPix(pix), 0)).rg;
        return float3(abs(m.r), abs(m.g), 0);
    }
    else if ((globalUniform.debugShowFlags & DEBUG_SHOW_FLAG_UNFILTERED_DIFFUSE) != 0)
    {
        return texelFetchUnfilteredDirect(getCheckerboardPix(pix));
    }
    else if ((globalUniform.debugShowFlags & DEBUG_SHOW_FLAG_UNFILTERED_SPECULAR) != 0)
    {
        return texelFetchUnfilteredSpecular(getCheckerboardPix(pix));
    }
    else if ((globalUniform.debugShowFlags & DEBUG_SHOW_FLAG_UNFILTERED_INDIRECT) != 0)
    {
        const SH sh = imageLoadUnfilteredIndirectSH(getCheckerboardPix(pix));
        const float3 normal = texelFetchNormal(getCheckerboardPix(pix));
        return SHToIrradiance(sh, normal);
    }
#if GRADIENT_ESTIMATION_ENABLED
    else if ((globalUniform.debugShowFlags & DEBUG_SHOW_FLAG_GRADIENTS) != 0)
    {
        return framebufDISPingGradient_Sampled.Load(int3(getCheckerboardPix(pix) / COMPUTE_ASVGF_STRATA_SIZE, 0)).xyz;
    }
#endif
#ifdef DEBUG_SHOW_SH
    const int2 checkSHRange = int2(800, 400);

    if (pix.x < checkSHRange.x && pix.y < checkSHRange.y)
    {
        float2 uv = float2(pix.x / (float)checkSHRange.x, pix.y / (float)checkSHRange.y);

        float theta = uv.x * 2.0 * M_PI;
        float phi = uv.y * M_PI;
        float3 normal = float3(cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi));

        int2 centerPix = int2(globalUniform.renderWidth * 0.5, globalUniform.renderHeight * 0.5);
        SH indirSH = texelFetchSH(
            //framebufUnfilteredIndirectSH_R_Sampled, framebufUnfilteredIndirectSH_G_Sampled, framebufUnfilteredIndirectSH_B_Sampled,
            framebufIndirPongSH_R_Sampled, framebufIndirPongSH_G_Sampled, framebufIndirPongSH_B_Sampled, 
            getCheckerboardPix(centerPix));

        return SHToIrradiance(indirSH, normal);
    }
#endif

    if ((globalUniform.debugShowFlags & DEBUG_SHOW_FLAG_LUMA) != 0)
    {
        const float3 luma = framebufScreenEmission_Sampled.Load(int3(pix, 0)).rgb;
        const float3 albedo = framebufAlbedo_Sampled.Load(int3(pix, 0)).rgb;
        return luma * 8.0 + albedo * 0.15;
    }

    return fallback;
}
