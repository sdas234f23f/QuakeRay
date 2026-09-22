// Hand written counterpart of GLSL/RayCone.probe.comp.
//
// RayCone.hlsli declares no resource of its own: it takes a ShTriangle by value and reads the
// geometry out of the columns of it, and it takes the material maps out of the split texture table
// through the gradient sample. This pair pins that surface on its own, with every function of the
// header instantiated and every column it reads filled. ShaderCommonFunc.probe.comp.hlsl, which
// includes RayCone.hlsli as well, keeps pinning the header in passing. Both halves define the same
// descriptor set index, touch the same resources and call the same helpers, so
// CheckShaderProperties.py can compare what glslc and dxc make of them. Probes are not part of the
// shader build, so no probe blob is ever shipped.
//
// The triangle below is deliberately not symmetric, so a column read and a row read of positions
// give different numbers. The three columns of the GLSL positions and of the GLSL mat3x2 layer
// texture coordinates are written here as rows and transposed, because `m[i] = v` fills a row on
// this side, as ShaderCommonHLSL.hlsli prescribes.

#define DESC_SET_TEXTURES 2

#include "ShaderCommonHLSLFunc.hlsli"

// MATERIAL_MAX_ALBEDO_LAYERS is normally defined by the consuming shader, as a specialization
// constant or as 0. RayCone.hlsli requires it.
#define MATERIAL_MAX_ALBEDO_LAYERS 3
#include "RayCone.hlsli"

#define PROBE_DESC_SET 8

[[vk::binding(0, PROBE_DESC_SET)]] RWStructuredBuffer<float> probeOutput;

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float v = 0.0;

    const float2 uv = float2(0.5, 0.5);
    // The direction and the normal are deliberately not perpendicular, and are dyadic so that the dot
    // product is exact: normalTerm = |dot(dir, worldNormal)| = 0.6875. With a perpendicular pair the
    // cone-width chain divides by zero, every derivative helper folds to inf, and the fold can no
    // longer see whether `positions` was read by column or by row (measured: both variants give inf).
    const float3 dir = float3(0.25, 0.5, 0.75);
    const float3 worldNormal = float3(-0.125, 0.875, 0.375);

    // propagateRayCone, on the cone that the helpers below then take by value
    RayCone rayCone;
    rayCone.width = 0.001;
    rayCone.spreadAngle = 0.01;

    propagateRayCone(rayCone, 1.0);
    v += rayCone.width + rayCone.spreadAngle;

    // getUVDerivativesFromRayCone, over the three vertices of a triangle
    const float3 vertWorldPositions[3] = { float3(1.0, 2.0, 3.0), float3(4.0, 6.0, 9.0), float3(10.0, 11.0, 13.0) };
    const float2 vertTexCoords[3] = { float2(0.25, 0.5), float2(0.75, 0.125), float2(1.0, 0.875) };

    const float4 uvDeriv =
        getUVDerivativesFromRayCone(rayCone, dir, worldNormal, vertWorldPositions, vertTexCoords);
    v += uvDeriv.x + uvDeriv.y + uvDeriv.z + uvDeriv.w;

    // getWaterDerivU, the same arithmetic on a unit quad
    v += getWaterDerivU(rayCone, dir, worldNormal);

    // getTriangleUVDerivativesFromRayCone: the two column reads of positions, and the layer texture
    // coordinate matrices, both built as rows and transposed
    // `triangle` is a reserved word in HLSL, hence `tri`
    ShTriangle tri;
    tri.positions = transpose(float3x3(vertWorldPositions[0], vertWorldPositions[1], vertWorldPositions[2]));

    // every layer is filled, not just the first one
    for (int layer = 0; layer < MATERIAL_MAX_ALBEDO_LAYERS; layer++)
    {
        tri.layerTexCoord[layer] = transpose(float3x2(vertTexCoords[0], vertTexCoords[1], vertTexCoords[2]));
    }

    const DerivativeSet derivSet = getTriangleUVDerivativesFromRayCone(tri, worldNormal, rayCone, dir);
    v += derivSet.u[0] + derivSet.u[1] + derivSet.u[2];

    // getTextureSampleDerivU and getTextureSampleDerivSet, the gradient sample and the wrapper that
    // indexes the set of derivatives
    v += getTextureSampleDerivU(3, uv, 0.01).x;
    v += getTextureSampleDerivSet(4, uv, derivSet, 1).y;

    probeOutput[0] = v;
}
