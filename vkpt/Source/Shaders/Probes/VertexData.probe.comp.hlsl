// Hand written counterpart of GLSL/VertexData.probe.comp.
//
// VertexData.inl is the geometry side of the shader base: eight storage buffers of the ray tracing
// data, and the accessors that turn a primitive into a ShTriangle. It has no stage of its own, so
// this pair exists to check its HLSL port against it. Both halves declare the same descriptor sets,
// touch the same eight resources and call the same helpers, so CheckShaderProperties.py can compare
// what glslc and dxc derive from them: the set and binding of every buffer, whether it is writable,
// the layout of ShVertex and ShGeometryInstance as the two compilers see them, and the properties of
// the two entry points. Probes are not part of the shader build, so no probe blob is ever shipped.
//
// The writeable variant is a pair of its own (VertexDataWriteable.probe.*): VERTEX_BUFFER_WRITEABLE
// turns the vertex buffers into RWStructuredBuffer on this side and drops `readonly` on the GLSL
// one, and it is the only other way through the header.

#define DESC_SET_GLOBAL_UNIFORM 0
#define DESC_SET_VERTEX_DATA    1

#include "ShaderCommonHLSLFunc.hlsli"

#define PROBE_DESC_SET 8

[[vk::binding(0, PROBE_DESC_SET)]] RWStructuredBuffer<float> probeOutput;

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float v = 0.0;

    // Every buffer of the set is read straight here: a buffer that only a helper touches can be
    // dropped from the module, and then the two halves would no longer describe the same set.
    v += float(g_staticVertices[0].cluster);
    v += float(g_dynamicVertices[0].lightStyles);
    v += float(staticIndices[0]);
    v += float(dynamicIndices[0]);
    v += geometryInstances[0].model[0][0];
    v += float(geomIndexPrevToCur[0]);
    v += float(g_dynamicVertices_Prev[0].packedColor);
    v += float(prevDynamicIndices[0]);

    // The accessors, static and dynamic alike
    v += getStaticVerticesPositions(0u).x + getStaticVerticesNormals(1u).y;
    v += getDynamicVerticesPositions(0u).z + getDynamicVerticesNormals(1u).x;
    v += getPrevDynamicVerticesPositions(0u).y;

    // Both spellings of the index lookup: with an index buffer, and without one
    const uint3 anyVertices = getVertIndicesStatic(0u, UINT32_MAX, 0u);
    const uint3 dynamicVertices = getVertIndicesDynamic(0u, UINT32_MAX, 0u);
    const uint3 prevVertices = getPrevVertIndicesDynamic(0u, UINT32_MAX, 0u);
    v += float(anyVertices.x + anyVertices.y + anyVertices.z);
    v += float(dynamicVertices.x + dynamicVertices.y + dynamicVertices.z);
    v += float(prevVertices.x + prevVertices.y + prevVertices.z);

    const uint3 indexedVertices = getVertIndicesStatic(0u, 0u, 0u);
    const uint3 indexedDynamicVertices = getVertIndicesDynamic(0u, 0u, 0u);
    const uint3 indexedPrevVertices = getPrevVertIndicesDynamic(0u, 0u, 0u);
    v += float(indexedVertices.x + indexedVertices.y + indexedVertices.z);
    v += float(indexedDynamicVertices.x + indexedDynamicVertices.y + indexedDynamicVertices.z);
    v += float(indexedPrevVertices.x + indexedPrevVertices.y + indexedPrevVertices.z);

    // getTangent, on the same shapes makeTriangle hands it. The GLSL half spells the texture
    // coordinates as mat3x2, three columns of two floats; float2x3 is its transposed declaration, so
    // the rows below are the columns of the GLSL constant, in the same order.
    const float3x3 localPos = float3x3(float3(0.0, 0.0, 0.0), float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0));
    const float2x3 texCoord = float2x3(float3(0.0, 1.0, 0.0), float3(0.0, 0.0, 1.0));
    v += getTangent(localPos, float3(0.0, 0.0, 1.0), texCoord).w;

    // makeTriangle needs three vertices; the texture coordinate layers and the coarse normal are
    // the interesting parts of it
    ShVertex a;
    a.position = float4(0.0, 0.0, 0.0, 1.0);
    a.normal = float4(0.0, 0.0, 1.0, 0.0);
    a.texCoord = float2(0.0, 0.0);
    a.texCoordLayer1 = float2(0.0, 1.0);
    a.texCoordLayer2 = float2(1.0, 0.0);
    a.packedColor = 0u;
    a.cluster = 0u;
    a.lightStyles = 0u;

    ShVertex b = a;
    b.position = float4(1.0, 0.0, 0.0, 1.0);
    b.texCoord = float2(1.0, 0.0);
    b.texCoordLayer1 = float2(1.0, 1.0);

    ShVertex c = a;
    c.position = float4(0.0, 1.0, 0.0, 1.0);
    c.texCoord = float2(0.0, 1.0);
    c.texCoordLayer2 = float2(0.0, 0.0);

    const ShTriangle made = makeTriangle(a, b, c);
    v += made.positions[0].x + made.positions[1].y + made.positions[2].z;
    v += made.normals[0].z + made.normals[1].x + made.normals[2].y;
    v += made.layerTexCoord[0][0].x + made.layerTexCoord[1][1].y + made.layerTexCoord[2][0].y;
    v += float(made.materials[0].x) + float(made.materials[1].y) + float(made.materials[2].z);
    v += made.materialColors[0].x + made.materialColors[1].y + made.materialColors[2].z;
    v += float(made.cluster) + float(made.lightStyleIndices) + float(made.vertexColors[0]);
    v += made.tangent.x + made.tangent.y + made.tangent.z + made.tangent.w;

    // getGeometryIndex, and the previous to current lookup of a geometry instance
    v += float(getGeometryIndex(0, 0));

    int curFrameGlobalGeomIndex;
    v += getCurrentGeometryIndexByPrev(0, 0, curFrameGlobalGeomIndex) ? 1.0 : 0.0;
    v += float(curFrameGlobalGeomIndex);

    // getTriangle, the whole ShTriangle out of one primitive
    const ShTriangle tri = getTriangle(0, 0, 0, 0);
    v += tri.positions[0].x + tri.prevPositions[1].y + tri.geomRoughness + tri.geomMetallicity;
    v += tri.normals[2].z + tri.tangent.w + tri.geomEmission + float(tri.portalIndex);
    v += tri.materialColors[2].z + float(tri.materials[1].y) + float(tri.geometryInstanceFlags);

    // The two position matrices
    const float3x3 curPositions = getOnlyCurPositions(0, 0, 0);
    const float3x3 prevPositions = getOnlyPrevPositions(0, 0, 0);
    v += curPositions[0].x + curPositions[1].y + curPositions[2].z;
    v += prevPositions[0].x + prevPositions[1].y + prevPositions[2].z;

    // The visibility buffer: its packing, and the three ways of taking it apart
    ShPayload p;
    p.instIdAndIndex = 0u;
    p.geomAndPrimIndex = 0u;
    p.baryCoords = float2(0.0, 0.0);

    const float4 packed = packVisibilityBuffer(p);
    v += packed.x + packed.y + packed.z + packed.w;
    v += float(unpackInstCustomIndexFromVisibilityBuffer(packed));

    int instanceID, instCustomIndex;
    int localGeomIndex, primIndex;
    float2 bary;
    unpackVisibilityBuffer(packed, instanceID, instCustomIndex, localGeomIndex, primIndex, bary);
    v += float(instanceID + instCustomIndex + localGeomIndex + primIndex) + bary.x + bary.y;

    float3 prevWorldPos = float3(0.0, 0.0, 0.0);
    v += unpackPrevVisibilityBuffer(packed, prevWorldPos) ? 1.0 : 0.0;
    v += prevWorldPos.x + prevWorldPos.y + prevWorldPos.z;

    v += getModelMatrix(0, 0)[0][0];

    // DESC_SET_GLOBAL_UNIFORM, reached through the wrapper of VertexData.hlsli
    v += globalUniform.renderWidth + globalUniform.instanceGeomInfoOffset[0].x;

    probeOutput[0] = v;
}
