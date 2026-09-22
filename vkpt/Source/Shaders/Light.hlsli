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


// HLSL counterpart of Light.h. Like the GLSL one it includes no resource layer of its own: the
// shader has to pull in ShaderCommonHLSLFunc.hlsli first, which declares globalUniform under
// #ifdef DESC_SET_GLOBAL_UNIFORM, talCdf and the other light source buffers under
// #ifdef DESC_SET_LIGHT_SOURCES, and getTextureSize/getTextureSampleLod under
// #ifdef DESC_SET_TEXTURES. RaygenCommon.h, the only consumer of the GLSL one, does the same: it
// includes ShaderCommonGLSLFunc.h and Surface.inl (which reaches Random.h through BRDF.h) before
// it includes Light.h. Only Random.hlsli (sampleDisk, sampleTriangle, sampleOrientedHemisphere,
// getONB) is pulled in here, because it brings nothing of its own either.
//
// Spellings that had to change:
//   * vec3(x) / vec2(x) with a single scalar -> (float3)x / (float2)x: the four places that build
//     a vector out of one scalar, which dxc refuses to widen in a constructor (see Utils.hlsli)
//   * vec2(ivec2) -> (float2)ivec2, e.g. `vec2(textureSize(...))` -> `(float2)getTextureSize(...)`
//   * the uint(...) / int(...) / float(...) constructors -> C style casts, same truncation and
//     rounding, and bool -> float in isSphereInFront is spelled (float)(... > ...)
//   * textureSize(globalTextures[nonuniformEXT(i)], 0) -> getTextureSize(i, 0), which hands back
//     the ivec2 of the GLSL one as an int2 (see ShaderCommonHLSLFunc.hlsli)
//   * floatBitsToUint -> asuint
//   * a single index of a matrix is a column in GLSL and a row in HLSL, so the two places that
//     build a normal out of the basis of getONB -- sampleDirectionalLight and sampleSpotLight --
//     index getColumn(basis, 0) and getColumn(basis, 1) instead of basis[0] and basis[1]. getONB
//     hands back the same matrix as the GLSL one element-wise, so those are the two basis vectors
//     as the GLSL has them (see Random.hlsli). This header performs no matrix product of its own
//     and builds no matrix of its own, so there is no other operand order to keep
//   * the #ifndef LIGHT_H_ / #define LIGHT_H_ guard -> #pragma once, as in BRDF.hlsli
//
// What did not change: every struct and its member order, every constant, the arithmetic of every
// weight and of every sample, the `a > 0.0 && !isnan(a) && !isinf(a)` guard of safeSolidAngle
// (dxc spells both intrinsics the way glslc does), the empty-entry branch of getTalCdfUv that the
// LUT distinguishes from a black sample, and the two switches, including the quirk of
// getLightWeight handing the directional light to decodeAsSphereLight, which the GLSL one does and
// which is harmless because getDirectionalLightWeight reads nothing but the colour.
//
// Light.h declares no descriptor of its own, so this file is pinned by the probe pair
// GLSL/Light.probe.comp <-> Probes/Light.probe.comp.hlsl.

#ifndef LIGHT_HLSLI_
#define LIGHT_HLSLI_
#include "Random.hlsli"

struct DirectionalLight
{
    float3 direction;
    float angularRadius;
    float3 color;
};

struct SphereLight
{
    float3 center;
    float radius;
    float3 color;
    float3 normal;
};

struct TriangleLight
{
    float3 position[3];
    float3 normal;
    float area;
    float3 color;
};

#define MAX_TEXTURED_AREA_LIGHT_VERTS 8

struct TexturedAreaLight
{
    float3 A;
    float3 B;
    float3 C;
    float3 normal;
    float area;
    float textureIndex;
    float meanEmiss;
    int numVerts;
    float2 uvVerts[MAX_TEXTURED_AREA_LIGHT_VERTS];
    float3 color;
};

struct SpotLight
{
    float3 center;
    float radius;
    float3 direction;
    float cosAngleInner;
    float3 color;
    float cosAngleOuter;
};

DirectionalLight decodeAsDirectionalLight(const ShLightEncoded encoded)
{
    DirectionalLight l;
    l.direction = encoded.data_0.xyz;
    l.angularRadius = encoded.data_0.w;
    l.color = encoded.color;

    return l;
}

SphereLight decodeAsSphereLight(const ShLightEncoded encoded)
{
    SphereLight l;
    l.center = encoded.data_0.xyz;
    l.radius = encoded.data_0.w;
    l.color = encoded.color;
    l.normal = encoded.data_1.xyz;

    return l;
}

TriangleLight decodeAsTriangleLight(const ShLightEncoded encoded)
{
    TriangleLight l;
    l.position[0] = encoded.data_0.xyz;
    l.position[1] = encoded.data_1.xyz;
    l.position[2] = encoded.data_2.xyz;
    l.color = encoded.color;

    l.normal = float3(
        encoded.data_0.w,
        encoded.data_1.w,
        encoded.data_2.w
    );
    // len is guaranteed to be > 0.0
    float len = length(l.normal);
    l.normal /= len;
    l.area = len * 0.5;

    return l;
}

TexturedAreaLight decodeAsTexturedAreaLight(const ShLightEncoded encoded)
{
    TexturedAreaLight l;
    l.A = encoded.data_0.xyz;
    l.textureIndex = encoded.data_0.w;
    l.B = encoded.data_1.xyz;
    l.meanEmiss = encoded.data_1.w;
    l.C = encoded.data_2.xyz;
    l.numVerts = clamp((int)encoded.data_2.w, 0, MAX_TEXTURED_AREA_LIGHT_VERTS);
    l.uvVerts[0] = encoded.data_3.xy;
    l.uvVerts[1] = encoded.data_3.zw;
    l.uvVerts[2] = encoded.data_4.xy;
    l.uvVerts[3] = encoded.data_4.zw;
    l.uvVerts[4] = encoded.data_5.xy;
    l.uvVerts[5] = encoded.data_5.zw;
    l.uvVerts[6] = encoded.data_6.xy;
    l.uvVerts[7] = encoded.data_6.zw;
    l.normal = encoded.data_7.xyz;
    l.area = encoded.data_7.w;
    l.color = encoded.color;

    return l;
}

SpotLight decodeAsSpotLight(const ShLightEncoded encoded)
{
    SpotLight l;
    l.center = encoded.data_0.xyz;
    l.radius = encoded.data_0.w;
    l.direction = encoded.data_1.xyz;
    l.color = encoded.color;
    l.cosAngleInner = encoded.data_2.x;
    l.cosAngleOuter = encoded.data_2.y;
    
    return l;
}

float getPolySpotFactor(const float3 lightNormal, const float3 lightToSurf)
{
    float ll = max(dot(lightNormal, lightToSurf), 0.0);
    return pow(ll, globalUniform.polyLightSpotlightFactor);
}

float getSpotFactor(float cosA, float cosAInner, float cosAOuter)
{
    return square(smoothstep(cosAOuter, cosAInner, cosA));
}

float isSphereInFront(const float3 planeNormal, const float3 planePos, const float3 sphereCenter, float sphereRadius)
{
    return (float)(dot(planeNormal, sphereCenter - planePos) > -sphereRadius);
}



// Veach, E. Robust Monte Carlo Methods for Light Transport Simulation
// The change of variables from solid angle measure to area integration measure
// Note: but without |dot(surfNormal, surfaceToLight)|
float getGeometryFactor(const float3 lightNormal, const float3 lightToSurface, float surfaceToLightDistance)
{
    return abs(dot(lightNormal, lightToSurface)) / square(surfaceToLightDistance);
}
float getGeometryFactorClamped(const float3 lightNormal, const float3 lightToSurface, float surfaceToLightDistance)
{
    return max(0.0, dot(lightNormal, lightToSurface)) * safePositiveRcp(square(surfaceToLightDistance));
}

float safeSolidAngle(float a)
{
    return a > 0.0 && !isnan(a) && !isinf(a) ? clamp(a, 0.0, 4.0 * M_PI) : 0.0;
}

float calcSolidAngleForSphere(float sphereRadius, float distanceToSphereCenter)
{
    // solid angle here is the spherical cap area on a unit sphere
    float sinTheta = sphereRadius / max(sphereRadius, distanceToSphereCenter);
    float cosTheta = sqrt(1.0 - sinTheta * sinTheta);
    return safeSolidAngle(2 * M_PI * (1.0 - cosTheta));
}

float calcSolidAngleForArea(float area, const float3 areaPosition, const float3 areaNormal, const float3 surfPosition)
{
    const DirectionAndLength areaLightToSurf = calcDirectionAndLength(areaPosition, surfPosition);
    // from area measure to solid angle measure
    return safeSolidAngle(area * getGeometryFactor(areaNormal, areaLightToSurf.dir, areaLightToSurf.len));
}



float getLightColorWeight(const float3 color)
{
    return clamp(getLuminance(color) * 0.1 + 0.9, 1.0, 10.0);
}

float getDirectionalLightWeight(const SphereLight l, const float3 cellCenter, float cellRadius)
{
    return 
        getLightColorWeight(l.color);
}

float getSphereLightWeight(const SphereLight l, const float3 cellCenter, float cellRadius)
{
    return 
        getLightColorWeight(l.color) * 
        calcSolidAngleForSphere(l.radius, max(length(l.center - cellCenter), cellRadius));
}

float getTriangleLightWeight(const TriangleLight l, const float3 cellCenter, float cellRadius)
{
    const float3 triCenter = 
        l.position[0] / 3.0 +
        l.position[1] / 3.0 +
        l.position[2] / 3.0;

    const float aprxTriRadius = 
        length(l.position[0] - triCenter) / 3.0 +
        length(l.position[1] - triCenter) / 3.0 +
        length(l.position[2] - triCenter) / 3.0;

    return 
        getLightColorWeight(l.color) * 
        calcSolidAngleForSphere(aprxTriRadius, max(length(triCenter - cellCenter), cellRadius)) *
        isSphereInFront(l.normal, triCenter, cellCenter, cellRadius);
}

float3 texturedAreaLightWorldPos(const TexturedAreaLight l, const float2 uv)
{
    return l.C + l.A * uv.x + l.B * uv.y;
}

float3 getTexturedAreaLightCenter(const TexturedAreaLight l)
{
    float2 uvCenter = (float2)0.0;
    for (int i = 0; i < l.numVerts; i++)
    {
        uvCenter += l.uvVerts[i];
    }
    uvCenter /= max((float)l.numVerts, 1.0);
    return texturedAreaLightWorldPos(l, uvCenter);
}

float2 sampleConvexPolygon(const float2 verts[MAX_TEXTURED_AREA_LIGHT_VERTS], int numVerts, float u1, float u2)
{
    if (numVerts < 3)
    {
        return verts[0];
    }

    float triArea[MAX_TEXTURED_AREA_LIGHT_VERTS - 2];
    float totalArea = 0.0;
    for (int i = 0; i < numVerts - 2; i++)
    {
        const float2 e1 = verts[i + 1] - verts[0];
        const float2 e2 = verts[i + 2] - verts[0];
        triArea[i] = 0.5 * abs(e1.x * e2.y - e1.y * e2.x);
        totalArea += triArea[i];
    }
    totalArea = max(totalArea, 1e-8);

    float r = u1 * totalArea;
    int t = numVerts - 3;
    float acc = 0.0;
    for (int i = 0; i < numVerts - 2; i++)
    {
        acc += triArea[i];
        if (r <= acc)
        {
            t = i;
            break;
        }
    }

    const float accBefore = acc - triArea[t];
    const float uTri = clamp((r - accBefore) / max(triArea[t], 1e-8), 0.0, 1.0);

    const float beta  = 1.0 - sqrt(uTri);
    const float gamma = (1.0 - beta) * u2;
    const float alpha = 1.0 - beta - gamma;

    return alpha * verts[0] + beta * verts[t + 1] + gamma * verts[t + 2];
}

float getTexturedAreaLightWeight(const TexturedAreaLight l, const float3 cellCenter, float cellRadius)
{
    const float3 center = getTexturedAreaLightCenter(l);

    float aprxRadius = 0.0;
    for (int i = 0; i < l.numVerts; i++)
    {
        aprxRadius = max(aprxRadius, length(texturedAreaLightWorldPos(l, l.uvVerts[i]) - center));
    }

    return 
        getLightColorWeight(l.color) * l.meanEmiss *
        calcSolidAngleForSphere(aprxRadius, max(length(center - cellCenter), cellRadius)) *
        isSphereInFront(l.normal, center, cellCenter, cellRadius);
}

float getSpotLightWeight(const SpotLight l, const float3 cellCenter, float cellRadius)
{
    return 
        getLightColorWeight(l.color) * 
        calcSolidAngleForSphere(l.radius, max(length(l.center - cellCenter), cellRadius)) *
        isSphereInFront(l.direction, l.center, cellCenter, cellRadius);
}




struct LightSample
{
    float3 position;
    float3 color;
    float dw;
};

LightSample emptyLightSample()
{
    LightSample r;
    r.position = (float3)0.0;
    r.color = (float3)0.0;
    r.dw = 0;
    return r;
}

LightSample sampleDirectionalLight(const DirectionalLight l, const float3 surfPosition, const float2 pointRnd)
{
    float3 lightNormal;
    {
        const float diskRadiusAtUnit = sin(max(0.01, l.angularRadius));
        const float2 disk = sampleDisk(diskRadiusAtUnit, pointRnd.x, pointRnd.y);
        const float3x3 basis = getONB(l.direction);

        lightNormal = normalize(l.direction + getColumn(basis, 0) * disk.x + getColumn(basis, 1) * disk.y);
    }

    LightSample r;
    r.position = surfPosition - lightNormal * MAX_RAY_LENGTH;
    r.color = l.color;
    r.dw = 1.0;
    
    return r;
}

LightSample sampleSphereLight(const SphereLight l, const float3 surfPosition, const float2 pointRnd)
{
    const DirectionAndLength toLightCenter = calcDirectionAndLength(surfPosition, l.center);

    // sample hemisphere visible to the surface point
    float ltHsOneOverPdf;
    const float3 lightNormal = sampleOrientedHemisphere(-toLightCenter.dir, pointRnd.x, pointRnd.y, ltHsOneOverPdf);

    LightSample r;
    r.position = l.center + lightNormal * l.radius;
    r.color = l.color;
    r.dw = calcSolidAngleForSphere(l.radius, toLightCenter.len);

    return r;
}

LightSample sampleTriangleLight(const TriangleLight l, const float3 surfPosition, const float2 pointRnd)
{
    LightSample r;
    r.position = sampleTriangle(l.position[0], l.position[1], l.position[2], pointRnd.x, pointRnd.y);
    r.color = l.color * getPolySpotFactor(l.normal, normalize(surfPosition - r.position));
    r.dw = calcSolidAngleForArea(l.area, r.position, l.normal, surfPosition);

    return r;
}

bool getTalCdfUv(const uint textureIndex, const float rnd, const float2 jitter, out float2 uv)
{
    const uint entryCount = (uint)TAL_CDF_LUT_ENTRIES;
    const uint index = min((uint)(rnd * (float)entryCount), entryCount - 1u);
    const uint packed = talCdf[textureIndex * entryCount + index];

    if (packed == TAL_CDF_EMPTY_ENTRY)
    {
        return false;
    }

    const float2 texSize = (float2)getTextureSize(textureIndex, 0);
    const float2 cellSize = 1.0 / min(texSize, (float2)TAL_CDF_GRID_MAX_SIZE);

    uv = float2((float)(packed & 0xFFFFu), (float)(packed >> 16u)) * (1.0 / 65535.0) + (jitter - 0.5) * cellSize;
    return true;
}

bool isUvInsideConvexPolygon(const float2 verts[MAX_TEXTURED_AREA_LIGHT_VERTS], const int numVerts, const float2 uv)
{
    if (numVerts < 3)
    {
        return true;
    }

    float sign = 0.0;
    for (int i = 0; i < numVerts; i++)
    {
        const float2 a = verts[i];
        const float2 b = verts[(i + 1) % numVerts];

        const float cross = (b.x - a.x) * (uv.y - a.y) - (b.y - a.y) * (uv.x - a.x);

        if (cross != 0.0)
        {
            if (sign == 0.0)
            {
                sign = cross > 0.0 ? 1.0 : -1.0;
            }
            else if (cross * sign < 0.0)
            {
                return false;
            }
        }
    }

    return true;
}

void getTalUvTiles(const TexturedAreaLight l, out float2 tileMin, out float2 tileMax)
{
    float2 uvMin = l.uvVerts[0];
    float2 uvMax = l.uvVerts[0];

    const int vertCount = min((int)l.numVerts, MAX_TEXTURED_AREA_LIGHT_VERTS);

    for (int i = 1; i < MAX_TEXTURED_AREA_LIGHT_VERTS; i++)
    {
        if (i >= vertCount)
        {
            break;
        }

        uvMin = min(uvMin, l.uvVerts[i]);
        uvMax = max(uvMax, l.uvVerts[i]);
    }

    tileMin = floor(uvMin);
    tileMax = max(ceil(uvMax) - 1.0, tileMin);
}

LightSample sampleTexturedAreaLight(const TexturedAreaLight l, const float3 surfPosition, const float2 pointRnd)
{
    LightSample r;

    const uint textureIndex = asuint(l.textureIndex);

    float2 uv = sampleConvexPolygon(l.uvVerts, l.numVerts, pointRnd.x, pointRnd.y);

    // The point is drawn uniformly over the light's polygon, so the emission is taken as it is
    // there: reading the mask and also scaling by the mean emission would count the mask twice,
    // and drawing the point from the CDF LUT would need that density divided out here to stay
    // unbiased. A mask that reads black is a valid zero sample, not a miss.
    float mask = 1.0;
    float emiss = l.meanEmiss;

    if (textureIndex != 0u)
    {
        mask = getTextureSampleLod(textureIndex, uv, 0.0).b;
        emiss = 1.0;
    }

    r.position = texturedAreaLightWorldPos(l, uv);

    const DirectionAndLength lightToSurf = calcDirectionAndLength(r.position, surfPosition);

    // Match Q2RTX sample_polygonal_lights: sample on the polygon (no normal offset),
    // soft edge attenuation via sqrt spot factor instead of a hard coplanar cull.
    const float spotlight = sqrt(max(0.0, dot(l.normal, lightToSurf.dir)));

    r.color = l.color * mask * spotlight;
    r.dw = safeSolidAngle(emiss * l.area * getGeometryFactorClamped(l.normal, lightToSurf.dir, lightToSurf.len));

    return r;
}

LightSample sampleSpotLight(const SpotLight l, const float3 surfPosition, const float2 pointRnd)
{
    LightSample r;
    {
        const float2 disk = sampleDisk(l.radius, pointRnd.x, pointRnd.y);
        const float3x3 basis = getONB(l.direction);

        r.position = l.center + getColumn(basis, 0) * disk.x + getColumn(basis, 1) * disk.y;
    }

    const DirectionAndLength toLightCenter = calcDirectionAndLength(surfPosition, l.center);
    const float cosA = max(dot(l.direction, -toLightCenter.dir), 0.0);
    
    r.color = l.color * getSpotFactor(cosA, l.cosAngleInner, l.cosAngleOuter);
    r.dw = calcSolidAngleForSphere(l.radius, toLightCenter.len);

    return r;
}



float getLightWeight(const ShLightEncoded encoded, const float3 cellCenter, float cellRadius)
{
    switch (encoded.lightType)
    {
        case LIGHT_TYPE_DIRECTIONAL:       return getDirectionalLightWeight    (decodeAsSphereLight          (encoded), cellCenter, cellRadius);
        case LIGHT_TYPE_SPHERE:            return getSphereLightWeight         (decodeAsSphereLight          (encoded), cellCenter, cellRadius);
        case LIGHT_TYPE_TRIANGLE:          return getTriangleLightWeight       (decodeAsTriangleLight        (encoded), cellCenter, cellRadius);
        case LIGHT_TYPE_SPOT:              return getSpotLightWeight           (decodeAsSpotLight            (encoded), cellCenter, cellRadius);
        case LIGHT_TYPE_TEXTURED_AREA:     return getTexturedAreaLightWeight   (decodeAsTexturedAreaLight    (encoded), cellCenter, cellRadius);
        default:                           return 0.0;
    }
}

LightSample sampleLight(const ShLightEncoded encoded, const float3 surfPosition, const float2 pointRnd)
{
    switch (encoded.lightType)
    {
        case LIGHT_TYPE_DIRECTIONAL:       return sampleDirectionalLight       (decodeAsDirectionalLight     (encoded), surfPosition, pointRnd);
        case LIGHT_TYPE_SPHERE:            return sampleSphereLight            (decodeAsSphereLight          (encoded), surfPosition, pointRnd);
        case LIGHT_TYPE_TRIANGLE:          return sampleTriangleLight          (decodeAsTriangleLight        (encoded), surfPosition, pointRnd);
        case LIGHT_TYPE_SPOT:              return sampleSpotLight              (decodeAsSpotLight            (encoded), surfPosition, pointRnd);
        case LIGHT_TYPE_TEXTURED_AREA:     return sampleTexturedAreaLight      (decodeAsTexturedAreaLight    (encoded), surfPosition, pointRnd);
        default:                           return emptyLightSample();
    }
}

#endif // LIGHT_HLSLI_
