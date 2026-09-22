// Structs.h has no shader stage of its own, so this probe exists to check that Structs.hlsli
// keeps the memory layout of Structs.h. Every struct is instantiated and every member is
// touched, and CheckShaderProperties.py compares the layouts that glslc and dxc derive from
// them. Probes are not part of the shader build, so no probe blob is ever shipped.

// The GLSL half reads the columns of the position matrices, so this half needs getColumn to read the
// same elements. Structs.hlsli does not use the helper itself.
#include "ShaderCommonHLSLFunc.hlsli"
#include "Structs.hlsli"

[[vk::binding(0, 0)]] StructuredBuffer<ShTriangle>        t;
[[vk::binding(1, 0)]] StructuredBuffer<ShHitInfo>         h;
[[vk::binding(2, 0)]] StructuredBuffer<ShPayload>         p;
[[vk::binding(3, 0)]] StructuredBuffer<ShPayloadShadow>   s;
[[vk::binding(4, 0)]] RWStructuredBuffer<float>           o;

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float v = 0.0;

    v += t[0].positions[0][0] + t[0].positions[1][1] + t[0].positions[2][2];
    v += getColumn(t[0].prevPositions, 0)[1] + getColumn(t[0].normals, 1)[2];
    v += getColumn(t[0].layerTexCoord[0], 1)[0] + getColumn(t[0].layerTexCoord[1], 0)[1]
        + getColumn(t[0].layerTexCoord[2], 1)[1];
    v += t[0].materialColors[2].w;
    v += float(t[0].materials[1].z) + float(t[0].vertexColors[2]);
    v += float(t[0].geometryInstanceFlags);
    v += t[0].tangent.y;
    v += t[0].geomRoughness + t[0].geomEmission + t[0].geomMetallicity;
    v += float(t[0].portalIndex) + float(t[0].cluster) + float(t[0].lightStyleIndices);

    v += h[0].albedo.y + h[0].metallic;
    v += h[0].normal.z + h[0].roughness;
    v += h[0].normalGeom.x + h[0].emission;
    v += h[0].hitPosition.z;
    v += float(h[0].instCustomIndex) + float(h[0].geometryInstanceFlags);
    v += float(h[0].portalIndex) + float(h[0].cluster);

    v += p[0].baryCoords.x + float(p[0].instIdAndIndex) + float(p[0].geomAndPrimIndex);
    v += float(s[0].isShadowed);

    o[0] = v;
}
