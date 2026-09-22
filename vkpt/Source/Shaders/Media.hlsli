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


// HLSL counterpart of Media.h. Like the GLSL one it includes nothing itself: the shader has to pull
// in ShaderCommonHLSLFunc.hlsli (globalUniform, SHIPPING_HACK) and the generated
// Generated/ShaderCommonHLSL.hlsli (MEDIA_TYPE_*, GEOM_INST_FLAG_*, through that same layer) first.
//
// The spells that had to change are vec3 -> float3 and the scalar broadcast of the GLSL original:
// this dxc refuses `float3(0.0)` with `too few elements in vector initialization`, so a broadcast is
// written as the cast `(float3)0.0`. The out parameter of calcRefractionDirection, the media and
// geometry-instance flags and every helper name are as in the GLSL original, including the parameter
// that shadows the `distance` intrinsic.

#ifndef MEDIA_HLSLI_
#define MEDIA_HLSLI_
#ifndef DESC_SET_GLOBAL_UNIFORM
    #error DESC_SET_GLOBAL_UNIFORM must be defined
#endif

// Beer-Lambert extinction is derived as -log(medium color), so a medium color channel
// of exactly 0 would make the extinction infinite (and 0 * inf -> NaN at distance 0),
// blacking out everything seen through the medium. Clamp the log argument instead.
#define MEDIA_EXTINCTION_MIN_COLOR 1e-6


float getIndexOfRefraction(uint media)
{
    switch (media)
    {
        case MEDIA_TYPE_WATER:
            return globalUniform.indexOfRefractionWater;
        case MEDIA_TYPE_ACID:
            return globalUniform.indexOfRefractionWater;
        case MEDIA_TYPE_GLASS:
            return globalUniform.indexOfRefractionGlass;
        default:
            return 1.0;
    }
}

float3 getMediaTransmittance( uint media, float distance )
{
    float3 extinction = (float3)0.0;

    if( media == MEDIA_TYPE_WATER )
    {
        extinction = -log( max( globalUniform.waterColorAndDensity.rgb, (float3)MEDIA_EXTINCTION_MIN_COLOR ) );
    }
    else if( media == MEDIA_TYPE_ACID )
    {
        extinction = -log( max( globalUniform.acidColorAndDensity.rgb, (float3)MEDIA_EXTINCTION_MIN_COLOR ) );
        extinction *= max(1.0, sqrt( globalUniform.acidColorAndDensity.a ) );
    }

    return exp( -distance * extinction );
}

#if SHIPPING_HACK
float3 getGlowingMediaFog( uint media, float distance )
{
    if( media != MEDIA_TYPE_ACID )
    {
        return (float3)0.0;
    }

    float density = 0.00005 * globalUniform.acidColorAndDensity.a;

    float fog = exp( -distance * density );
    fog       = clamp( 1.0 - fog, 0.0, 1.0 );

    return fog * globalUniform.acidColorAndDensity.rgb;
}
#endif


// Ray Tracing Gems II. Chapter 8: Reflection and Refraction Formulas
// Returns false, if total internal reflection
bool calcRefractionDirection(float n1, float n2, const float3 I, const float3 N, out float3 T)
{
    float eta = n1 / n2; //  relative index of refraction
    float c1 = -dot(I, N); // cos(theta1)
    float w = eta * c1;
    float c2m = (w - eta) * (w + eta); // cos^2(theta2) - 1

    if (c2m < -1.0f)
    { 
        return false; // total internal reflection
    }

    T = eta * I + (w - sqrt(1.0f + c2m)) * N;
    return true;
}

uint getMediaTypeFromFlags(uint geometryInstanceFlags)
{
    if ((geometryInstanceFlags & GEOM_INST_FLAG_MEDIA_TYPE_WATER) != 0)
    {
        return MEDIA_TYPE_WATER;
    }
    else if ((geometryInstanceFlags & GEOM_INST_FLAG_MEDIA_TYPE_GLASS) != 0)
    {
        return MEDIA_TYPE_GLASS;
    }
    else if ((geometryInstanceFlags & GEOM_INST_FLAG_MEDIA_TYPE_ACID) != 0)
    {
        return MEDIA_TYPE_ACID;
    }
    else
    {
        return MEDIA_TYPE_VACUUM;
    }
}

bool isPortalFromFlags(uint geometryInstanceFlags)
{
    return (geometryInstanceFlags & GEOM_INST_FLAG_PORTAL) != 0;
}

bool isRefractFromFlags(uint geometryInstanceFlags)
{
    // if water, but water refraction is disabled, return false;
    // otherwise, check refract flag
    return 
        !(globalUniform.forceNoWaterRefraction != 0 && (geometryInstanceFlags & GEOM_INST_FLAG_MEDIA_TYPE_WATER) != 0) &&
        ((geometryInstanceFlags & GEOM_INST_FLAG_REFRACT) != 0);
}

bool isReflectFromFlags(uint geometryInstanceFlags)
{
    return (geometryInstanceFlags & GEOM_INST_FLAG_REFLECT) != 0;
}

#endif // MEDIA_HLSLI_
