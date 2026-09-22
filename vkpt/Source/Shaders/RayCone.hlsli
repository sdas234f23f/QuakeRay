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


// HLSL counterpart of RayCone.h. Like the GLSL one it includes nothing itself: the shader has to
// pull in Structs.hlsli (ShTriangle), ShaderCommonHLSLFunc.hlsli (the split texture table and its
// accessors) and define MATERIAL_MAX_ALBEDO_LAYERS first.
//
// Ray Tracing Gems 2. Chapter 7: Texture Coordinate Gradients Estimation for Ray Cones
//
// Spellings that had to change:
//   * GLSL mat3x2 -> float2x3, the transposed declaration every matrix of the port uses. A single
//     index is a column of the GLSL matrix and a row of the HLSL one, so the local copy of the
//     layer texture coordinates is spelled transpose(tri.layerTexCoord[i]) and the body below then
//     reads like the GLSL one again
//   * triangle.positions[i] is such a column read on a mat3, so it becomes
//     getColumn(tri.positions, i), the construction HitInfo.hlsli takes the geometric normal from
//   * textureGrad(sampler2D(getTexture(i), getTextureSampler(i)), uv, dx, dy) -> the split table
//     helper getTextureSampleGrad(i, uv, dx, dy) of ShaderCommonHLSLFunc.hlsli, with vec2(uDeriv,
//     0) as float2(uDeriv, 0.0), because a scalar argument does not broadcast into a vector
//   * `triangle` is a reserved word in HLSL, as the point/line/triangle primitives of the geometry
//     shader are, so the parameter of getTriangleUVDerivativesFromRayCone is called `tri`
//
// What did not change: the DerivativeSet members, the normalTerm / projectedConeWidth /
// visibleAreaRatio arithmetic that all three functions share, the `const` qualifiers and the fact
// that `u` is left untouched above MATERIAL_MAX_ALBEDO_LAYERS, as in GLSL.

#ifndef RAY_CONE_HLSLI_
#define RAY_CONE_HLSLI_
struct RayCone
{
    float width;
    float spreadAngle;
};

void propagateRayCone(inout RayCone c, float rayLength)
{
    // new cone width should increase by 2*RayLength*tan(SpreadAngle/2), but RayLength*SpreadAngle is a close approximation
    c.width	+= c.spreadAngle * rayLength;
    c.spreadAngle *= 2;
}

float4 getUVDerivativesFromRayCone(
    const RayCone rayCone, 
    const float3 rayDir, 
    const float3 worldNormal, 
    const float3 vertWorldPositions[3], 
    const float2 vertTexCoords[3])
{
    const float2 uv10 = vertTexCoords[1] - vertTexCoords[0];
    const float2 uv20 = vertTexCoords[2] - vertTexCoords[0];
    float quadUVArea = abs(uv10.x * uv20.y - uv20.x * uv10.y);

    const float3 edge10 = vertWorldPositions[1] - vertWorldPositions[0];
    const float3 edge20 = vertWorldPositions[2] - vertWorldPositions[0];
    const float3 faceNormal = cross(edge10, edge20);
    float quadArea = length(faceNormal);

    float normalTerm = abs(dot(rayDir, worldNormal));
    float projectedConeWidth = rayCone.width / normalTerm;
    float visibleAreaRatio = (projectedConeWidth * projectedConeWidth) / quadArea;

    float visibleUVArea = quadUVArea * visibleAreaRatio;
    float ULength = sqrt(visibleUVArea);

    return float4(ULength, 0.0, 0.0, ULength);
}

float getWaterDerivU(const RayCone rayCone, const float3 rayDir, const float3 worldNormal)
{
    // just any values
    const float quadArea = 1.0;
    const float quadUVArea = 1.0;

    float normalTerm = abs(dot(rayDir, worldNormal));
    float projectedConeWidth = rayCone.width / normalTerm;
    float visibleAreaRatio = (projectedConeWidth * projectedConeWidth) / quadArea;

    float visibleUVArea = quadUVArea * visibleAreaRatio;
    float ULength = sqrt(visibleUVArea);

    return ULength;
}

struct DerivativeSet
{
    float u[3];
};

DerivativeSet getTriangleUVDerivativesFromRayCone(
    const ShTriangle tri,
    const float3 worldNormal,
    const RayCone rayCone, 
    const float3 rayDir)
{
    const float3 edge10 = getColumn(tri.positions, 1) - getColumn(tri.positions, 0);
    const float3 edge20 = getColumn(tri.positions, 2) - getColumn(tri.positions, 0);
    const float3 faceNormal = cross(edge10, edge20);
    float quadArea = length(faceNormal);

    float normalTerm = abs(dot(rayDir, worldNormal));
    float projectedConeWidth = rayCone.width / normalTerm;
    float visibleAreaRatio = (projectedConeWidth * projectedConeWidth) / quadArea;


    DerivativeSet derivSet;

    for (int i = 0; i < MATERIAL_MAX_ALBEDO_LAYERS; i++)
    {
        const float3x2 vertTexCoords = transpose(tri.layerTexCoord[i]);

        const float2 uv10 = vertTexCoords[1] - vertTexCoords[0];
        const float2 uv20 = vertTexCoords[2] - vertTexCoords[0];
        float quadUVArea = abs(uv10.x * uv20.y - uv20.x * uv10.y);

        float visibleUVArea = quadUVArea * visibleAreaRatio;
        float ULength = sqrt(visibleUVArea);

        derivSet.u[i] = ULength;
    }

    return derivSet;
}

float4 getTextureSampleDerivU(uint textureIndex, const float2 texCoord, const float uDeriv)
{
    return getTextureSampleGrad(textureIndex, texCoord, float2(uDeriv, 0.0), float2(0.0, uDeriv));
}

float4 getTextureSampleDerivSet(uint textureIndex, const float2 texCoord, const DerivativeSet derivSet, int index)
{
    return getTextureSampleDerivU(textureIndex, texCoord, derivSet.u[index]);
}

#endif // RAY_CONE_HLSLI_
