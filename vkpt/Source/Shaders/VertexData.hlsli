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


// HLSL counterpart of VertexData.inl. Like the GLSL one it includes nothing itself: the shader has
// to pull in ShaderCommonHLSLFunc.hlsli (globalUniform, the index packers) first, and to define
// DESC_SET_VERTEX_DATA as well as DESC_SET_GLOBAL_UNIFORM, which owns the geometry instances.
//
// This is the first ported header that declares descriptors. VERTEX_BUFFER_WRITEABLE marks the pass
// that writes the smoothed normals back (CmVertexPreprocess.comp): the vertex buffers are then
// RWStructuredBuffer, the way a GLSL `buffer` without `readonly` is, while the index buffers and
// the read-only paths stay StructuredBuffer. Note that VertexData.inl keeps prevDynamicIndices
// writable in that case.
//
// Spellings that had to change:
//   * GLSL uvec3 -> uint3, and GLSL `vec3(v)` -> HLSL float3(v.x, v.y, v.z): a scalar does not
//     broadcast into a vector in HLSL, so a one-component construction lists every component
//   * every GLSL `m * v` became mul(m, v), the operand order of the GLSL source, and GLSL
//     mat3(inst.model) became (float3x3)inst.model, which dxc reads as the upper left submatrix,
//     exactly as GLSL does
//   * every matrix member that the GLSL body fills or reads column by column keeps those columns
//     here: a single index of an HLSL matrix is a row, so `tr.positions[i] = v` has no counterpart
//     and the columns are transposed into the member instead
//     (tr.positions = transpose(float3x3(a, b, c))), while a column that is read back is taken from
//     transpose, the way the getColumn helper of ShaderCommonHLSLFunc.hlsli spells it (that helper
//     is declared below this header, after it is included, so it can not be called from here)
//   * ShTriangle.layerTexCoord is three GLSL columns of two floats, so its transposed declaration
//     holds what those columns held: makeTriangle builds each of the two rows out of the three
//     vertices, and getTangent takes that shape and transposes it back for its own u1/u2
//   * uintBitsToFloat -> asfloat, floatBitsToUint -> asuint
//   * GLSL arrays whose size is deduced from the initializer (prevLocalPos, localPos) spell it out
//
// What did not change: the eight bindings and their sets, the packing of the visibility buffer, the
// order in which the vertex data is looked up, the const qualifiers, the compile time `#ifdef`
// gating, and the commented out getGeometryInstanceMaterialLayer at the end.

#ifndef VERTEX_DATA_HLSLI_
#define VERTEX_DATA_HLSLI_
#ifdef DESC_SET_GLOBAL_UNIFORM
#ifdef DESC_SET_VERTEX_DATA

#ifdef VERTEX_BUFFER_WRITEABLE
[[vk::binding(BINDING_VERTEX_BUFFER_STATIC, DESC_SET_VERTEX_DATA)]] RWStructuredBuffer<ShVertex> g_staticVertices;
#else
[[vk::binding(BINDING_VERTEX_BUFFER_STATIC, DESC_SET_VERTEX_DATA)]] StructuredBuffer<ShVertex> g_staticVertices;
#endif

#ifdef VERTEX_BUFFER_WRITEABLE
[[vk::binding(BINDING_VERTEX_BUFFER_DYNAMIC, DESC_SET_VERTEX_DATA)]] RWStructuredBuffer<ShVertex> g_dynamicVertices;
#else
[[vk::binding(BINDING_VERTEX_BUFFER_DYNAMIC, DESC_SET_VERTEX_DATA)]] StructuredBuffer<ShVertex> g_dynamicVertices;
#endif

[[vk::binding(BINDING_INDEX_BUFFER_STATIC, DESC_SET_VERTEX_DATA)]] StructuredBuffer<uint> staticIndices;

[[vk::binding(BINDING_INDEX_BUFFER_DYNAMIC, DESC_SET_VERTEX_DATA)]] StructuredBuffer<uint> dynamicIndices;

[[vk::binding(BINDING_GEOMETRY_INSTANCES, DESC_SET_VERTEX_DATA)]] StructuredBuffer<ShGeometryInstance> geometryInstances;

[[vk::binding(BINDING_GEOMETRY_INSTANCES_MATCH_PREV, DESC_SET_VERTEX_DATA)]] StructuredBuffer<int> geomIndexPrevToCur;

#ifdef VERTEX_BUFFER_WRITEABLE
[[vk::binding(BINDING_PREV_POSITIONS_BUFFER_DYNAMIC, DESC_SET_VERTEX_DATA)]] RWStructuredBuffer<ShVertex> g_dynamicVertices_Prev;
#else
[[vk::binding(BINDING_PREV_POSITIONS_BUFFER_DYNAMIC, DESC_SET_VERTEX_DATA)]] StructuredBuffer<ShVertex> g_dynamicVertices_Prev;
#endif

#ifdef VERTEX_BUFFER_WRITEABLE
[[vk::binding(BINDING_PREV_INDEX_BUFFER_DYNAMIC, DESC_SET_VERTEX_DATA)]] RWStructuredBuffer<uint> prevDynamicIndices;
#else
[[vk::binding(BINDING_PREV_INDEX_BUFFER_DYNAMIC, DESC_SET_VERTEX_DATA)]] StructuredBuffer<uint> prevDynamicIndices;
#endif

float3 getStaticVerticesPositions(uint index)
{
    return g_staticVertices[index].position.xyz;
}

float3 getStaticVerticesNormals(uint index)
{
    return g_staticVertices[index].normal.xyz;
}

float3 getDynamicVerticesPositions(uint index)
{
    return g_dynamicVertices[index].position.xyz;
}

float3 getDynamicVerticesNormals(uint index)
{
    return g_dynamicVertices[index].normal.xyz;
}

#ifdef VERTEX_BUFFER_WRITEABLE
void setStaticVerticesNormals(uint index, float3 value)
{
    g_staticVertices[index].normal = float4(value, 0.0);
}

void setDynamicVerticesNormals(uint index, float3 value)
{
    g_dynamicVertices[index].normal = float4(value, 0.0);
}
#endif // VERTEX_BUFFER_WRITEABLE

// Get indices in vertex buffer. If geom uses index buffer then it flattens them to vertex buffer indices.
uint3 getVertIndicesStatic(uint baseVertexIndex, uint baseIndexIndex, uint primitiveId)
{
    // if to use indices
    if (baseIndexIndex != UINT32_MAX)
    {
        return uint3(
            baseVertexIndex + staticIndices[baseIndexIndex + primitiveId * 3 + 0],
            baseVertexIndex + staticIndices[baseIndexIndex + primitiveId * 3 + 1],
            baseVertexIndex + staticIndices[baseIndexIndex + primitiveId * 3 + 2]);
    }
    else
    {
        return uint3(
            baseVertexIndex + primitiveId * 3 + 0,
            baseVertexIndex + primitiveId * 3 + 1,
            baseVertexIndex + primitiveId * 3 + 2);
    }
}

uint3 getVertIndicesDynamic(uint baseVertexIndex, uint baseIndexIndex, uint primitiveId)
{
    // if to use indices
    if (baseIndexIndex != UINT32_MAX)
    {
        return uint3(
            baseVertexIndex + dynamicIndices[baseIndexIndex + primitiveId * 3 + 0],
            baseVertexIndex + dynamicIndices[baseIndexIndex + primitiveId * 3 + 1],
            baseVertexIndex + dynamicIndices[baseIndexIndex + primitiveId * 3 + 2]);
    }
    else
    {
        return uint3(
            baseVertexIndex + primitiveId * 3 + 0,
            baseVertexIndex + primitiveId * 3 + 1,
            baseVertexIndex + primitiveId * 3 + 2);
    }
}

// Only for dynamic, static geom vertices are not changed.
uint3 getPrevVertIndicesDynamic(uint prevBaseVertexIndex, uint prevBaseIndexIndex, uint primitiveId)
{
    // if to use indices
    if (prevBaseIndexIndex != UINT32_MAX)
    {
        return uint3(
            prevBaseVertexIndex + prevDynamicIndices[prevBaseIndexIndex + primitiveId * 3 + 0],
            prevBaseVertexIndex + prevDynamicIndices[prevBaseIndexIndex + primitiveId * 3 + 1],
            prevBaseVertexIndex + prevDynamicIndices[prevBaseIndexIndex + primitiveId * 3 + 2]);
    }
    else
    {
        return uint3(
            prevBaseVertexIndex + primitiveId * 3 + 0,
            prevBaseVertexIndex + primitiveId * 3 + 1,
            prevBaseVertexIndex + primitiveId * 3 + 2);
    }
}

float3 getPrevDynamicVerticesPositions(uint index)
{
    return g_dynamicVertices_Prev[index].position.xyz;
}

// texCoord is the transposed declaration of the GLSL mat3x2, so transpose() gives back the three
// columns of two floats that the GLSL body subtracts.
float4 getTangent(const float3x3 localPos, const float3 normal, const float2x3 texCoord)
{
    const float3 e1 = transpose(localPos)[1] - transpose(localPos)[0];
    const float3 e2 = transpose(localPos)[2] - transpose(localPos)[0];

    const float3x2 tc = transpose(texCoord);
    const float2 u1 = tc[1] - tc[0];
    const float2 u2 = tc[2] - tc[0];

    const float invDet = 1.0 / (u1.x * u2.y - u2.x * u1.y);

    const float3 tangent   = normalize((e1 * u2.y - e2 * u1.y) * invDet);
    const float3 bitangent = normalize((e2 * u1.x - e1 * u2.x) * invDet);

    // Don't store bitangent, only store cross(normal, tangent) handedness.
    // If that cross product and bitangent have the same sign,
    // then handedness is 1, otherwise -1
    float handedness = float(dot(cross(normal, tangent), bitangent) > 0.0);
    handedness = handedness * 2.0 - 1.0;

    return float4(tangent, handedness);
}

ShTriangle makeTriangle(const ShVertex a, const ShVertex b, const ShVertex c)
{    
    ShTriangle tr;

    // the GLSL body fills both matrices column by column, which HLSL can not: the three columns
    // become the three rows of the constructor, transposed into the member
    tr.positions = transpose(float3x3(a.position.xyz, b.position.xyz, c.position.xyz));

    tr.normals = transpose(float3x3(a.normal.xyz, b.normal.xyz, c.normal.xyz));

    // three GLSL columns of two floats, written as the two rows of the transposed declaration
    tr.layerTexCoord[0][0] = float3(a.texCoord.x, b.texCoord.x, c.texCoord.x);
    tr.layerTexCoord[0][1] = float3(a.texCoord.y, b.texCoord.y, c.texCoord.y);

    tr.layerTexCoord[1][0] = float3(a.texCoordLayer1.x, b.texCoordLayer1.x, c.texCoordLayer1.x);
    tr.layerTexCoord[1][1] = float3(a.texCoordLayer1.y, b.texCoordLayer1.y, c.texCoordLayer1.y);

    tr.layerTexCoord[2][0] = float3(a.texCoordLayer2.x, b.texCoordLayer2.x, c.texCoordLayer2.x);
    tr.layerTexCoord[2][1] = float3(a.texCoordLayer2.y, b.texCoordLayer2.y, c.texCoordLayer2.y);

    tr.cluster = a.cluster;
    tr.lightStyleIndices = a.lightStyles;

    // get very coarse normal for triangle to determine bitangent's handedness
    tr.tangent = getTangent(tr.positions, safeNormalize(transpose(tr.normals)[0] + transpose(tr.normals)[1] + transpose(tr.normals)[2]), tr.layerTexCoord[0]);

    return tr;
}

// Get geometry index in "geometryInstances" array by instanceID, localGeometryIndex.
int getGeometryIndex(int instanceID, int localGeometryIndex)
{
    return globalUniform.instanceGeomInfoOffset[instanceID / 4][instanceID % 4] + localGeometryIndex;
}

bool getCurrentGeometryIndexByPrev(int prevInstanceID, int prevLocalGeometryIndex, out int curFrameGlobalGeomIndex)
{
    // get previous frame's global geom index
    const int prevFrameGeomIndex = globalUniform.instanceGeomInfoOffsetPrev[prevInstanceID / 4][prevInstanceID % 4] + prevLocalGeometryIndex;
    
    // try to find global geom index in current frame by it
    curFrameGlobalGeomIndex = geomIndexPrevToCur[prevFrameGeomIndex];

    // UINT32_MAX -- no prev to cur exist
    return curFrameGlobalGeomIndex != UINT32_MAX;
}

/*#define GEOMETRY_INSTANCE_TEXTURE_SET_SIZE 3

uvec3 getGeometryInstanceMaterialLayer(const ShGeometryInstance inst, int layer)
{
    return uvec3(
        inst.materials[layer * GEOMETRY_INSTANCE_TEXTURE_SET_SIZE + MATERIAL_ALBEDO_ALPHA_INDEX], 
        inst.materials[layer * GEOMETRY_INSTANCE_TEXTURE_SET_SIZE + MATERIAL_ROUGHNESS_METALLIC_EMISSION_INDEX], 
        inst.materials[layer * GEOMETRY_INSTANCE_TEXTURE_SET_SIZE + MATERIAL_NORMAL_INDEX]
    );
}*/

// localGeometryIndex is index of geometry in pGeometries in BLAS
// primitiveId is index of a triangle
ShTriangle getTriangle(int instanceID, int instanceCustomIndex, int localGeometryIndex, int primitiveId)
{
    ShTriangle tr;

    // get info about geometry by the index in pGeometries in BLAS with index "instanceID"
    const int globalGeometryIndex = getGeometryIndex(instanceID, localGeometryIndex);
    const ShGeometryInstance inst = geometryInstances[globalGeometryIndex];

    const bool isDynamic = (instanceCustomIndex & INSTANCE_CUSTOM_INDEX_FLAG_DYNAMIC) == INSTANCE_CUSTOM_INDEX_FLAG_DYNAMIC;

    if (isDynamic)
    {
        {
            const uint3 vertIndices = getVertIndicesDynamic(inst.baseVertexIndex, inst.baseIndexIndex, primitiveId);

            tr = makeTriangle(
                g_dynamicVertices[vertIndices[0]],
                g_dynamicVertices[vertIndices[1]],
                g_dynamicVertices[vertIndices[2]]);
        }

        // to world space
        tr.positions = transpose(float3x3(
            mul(inst.model, float4(transpose(tr.positions)[0], 1.0)).xyz,
            mul(inst.model, float4(transpose(tr.positions)[1], 1.0)).xyz,
            mul(inst.model, float4(transpose(tr.positions)[2], 1.0)).xyz));
        
        // dynamic     -- use prev model matrix and prev positions if exist
        const bool hasPrevInfo = inst.prevBaseVertexIndex != UINT32_MAX;

        if (hasPrevInfo)
        {
            const uint3 prevVertIndices = getPrevVertIndicesDynamic(inst.prevBaseVertexIndex, inst.prevBaseIndexIndex, primitiveId);

            const float4 prevLocalPos[3] =
            {
                float4(getPrevDynamicVerticesPositions(prevVertIndices[0]), 1.0),
                float4(getPrevDynamicVerticesPositions(prevVertIndices[1]), 1.0),
                float4(getPrevDynamicVerticesPositions(prevVertIndices[2]), 1.0)
            };

            tr.prevPositions = transpose(float3x3(
                mul(inst.prevModel, prevLocalPos[0]).xyz,
                mul(inst.prevModel, prevLocalPos[1]).xyz,
                mul(inst.prevModel, prevLocalPos[2]).xyz));
        }
        else
        {
            tr.prevPositions = tr.positions;
        }
    }
    else
    {
        {
            const uint3 vertIndices = getVertIndicesStatic(inst.baseVertexIndex, inst.baseIndexIndex, primitiveId);
        
            tr = makeTriangle(
                g_staticVertices[vertIndices[0]],
                g_staticVertices[vertIndices[1]],
                g_staticVertices[vertIndices[2]]);
        }

        const float4 prevLocalPos[3] =
        {
            float4(transpose(tr.positions)[0], 1.0),
            float4(transpose(tr.positions)[1], 1.0),
            float4(transpose(tr.positions)[2], 1.0)
        };

        // to world space
        tr.positions = transpose(float3x3(
            mul(inst.model, prevLocalPos[0]).xyz,
            mul(inst.model, prevLocalPos[1]).xyz,
            mul(inst.model, prevLocalPos[2]).xyz));
        
        const bool isMovable = (inst.flags & GEOM_INST_FLAG_IS_MOVABLE) != 0;
        const bool hasPrevInfo = inst.prevBaseVertexIndex != UINT32_MAX;

        // movable     -- use prev model matrix if exist
        // non-movable -- use current model matrix
        if (isMovable && hasPrevInfo)
        {
            // static geoms' local positions are constant, 
            // only model matrices are changing
            tr.prevPositions = transpose(float3x3(
                mul(inst.prevModel, prevLocalPos[0]).xyz,
                mul(inst.prevModel, prevLocalPos[1]).xyz,
                mul(inst.prevModel, prevLocalPos[2]).xyz));
        }
        else
        {
            tr.prevPositions = tr.positions;
        }
    }


    tr.materials[0] = uint3(inst.materials0A, inst.materials0B, inst.materials0C);
    tr.materials[1] = uint3(inst.materials1A, inst.materials1B, MATERIAL_NO_TEXTURE);
    tr.materials[2] = uint3(inst.materials2A, inst.materials2B, MATERIAL_NO_TEXTURE);

    tr.materialColors[0] = inst.materialColors[0];
    tr.materialColors[1] = inst.materialColors[1];
    tr.materialColors[2] = inst.materialColors[2];
        
    
    const float3x3 model3 = (float3x3)inst.model;

    // to world space
    tr.normals = transpose(float3x3(
        mul(model3, transpose(tr.normals)[0]),
        mul(model3, transpose(tr.normals)[1]),
        mul(model3, transpose(tr.normals)[2])));
    tr.tangent.xyz = mul(model3, tr.tangent.xyz);


    tr.geometryInstanceFlags = inst.flags;

    tr.geomRoughness = inst.defaultRoughness;
    tr.geomMetallicity = inst.defaultMetallicity;

    // use (first layer's color) * defaultEmission
    tr.geomEmission = inst.defaultEmission;

    tr.portalIndex = inst.portalIndex;

    return tr;
}

float3x3 getOnlyCurPositions(int globalGeometryIndex, int instanceCustomIndex, int primitiveId)
{
    float3x3 positions;

    const ShGeometryInstance inst = geometryInstances[globalGeometryIndex];

    const bool isDynamic = (instanceCustomIndex & INSTANCE_CUSTOM_INDEX_FLAG_DYNAMIC) == INSTANCE_CUSTOM_INDEX_FLAG_DYNAMIC;

    if (isDynamic)
    {
        const uint3 vertIndices = getVertIndicesDynamic(inst.baseVertexIndex, inst.baseIndexIndex, primitiveId);

        // to world space
        positions = transpose(float3x3(
            mul(inst.model, float4(getDynamicVerticesPositions(vertIndices[0]), 1.0)).xyz,
            mul(inst.model, float4(getDynamicVerticesPositions(vertIndices[1]), 1.0)).xyz,
            mul(inst.model, float4(getDynamicVerticesPositions(vertIndices[2]), 1.0)).xyz));
    }
    else
    {
        const uint3 vertIndices = getVertIndicesStatic(inst.baseVertexIndex, inst.baseIndexIndex, primitiveId);

        // to world space
        positions = transpose(float3x3(
            mul(inst.model, float4(getStaticVerticesPositions(vertIndices[0]), 1.0)).xyz,
            mul(inst.model, float4(getStaticVerticesPositions(vertIndices[1]), 1.0)).xyz,
            mul(inst.model, float4(getStaticVerticesPositions(vertIndices[2]), 1.0)).xyz));
    }
    
    return positions;
}

float3x3 getOnlyPrevPositions(int globalGeometryIndex, int instanceCustomIndex, int primitiveId)
{
    float3x3 prevPositions;

    const ShGeometryInstance inst = geometryInstances[globalGeometryIndex];

    const bool isDynamic = (instanceCustomIndex & INSTANCE_CUSTOM_INDEX_FLAG_DYNAMIC) == INSTANCE_CUSTOM_INDEX_FLAG_DYNAMIC;

    if (isDynamic)
    {
        // dynamic     -- use prev model matrix and prev positions if exist
        const bool hasPrevInfo = inst.prevBaseVertexIndex != UINT32_MAX;

        if (hasPrevInfo)
        {
            const uint3 prevVertIndices = getPrevVertIndicesDynamic(inst.prevBaseVertexIndex, inst.prevBaseIndexIndex, primitiveId);

            const float4 prevLocalPos[3] =
            {
                float4(getPrevDynamicVerticesPositions(prevVertIndices[0]), 1.0),
                float4(getPrevDynamicVerticesPositions(prevVertIndices[1]), 1.0),
                float4(getPrevDynamicVerticesPositions(prevVertIndices[2]), 1.0)
            };

            prevPositions = transpose(float3x3(
                mul(inst.prevModel, prevLocalPos[0]).xyz,
                mul(inst.prevModel, prevLocalPos[1]).xyz,
                mul(inst.prevModel, prevLocalPos[2]).xyz));
        }
        else
        {
            const uint3 vertIndices = getVertIndicesDynamic(inst.baseVertexIndex, inst.baseIndexIndex, primitiveId);
            
            const float4 localPos[3] =
            {
                float4(getDynamicVerticesPositions(vertIndices[0]), 1.0),
                float4(getDynamicVerticesPositions(vertIndices[1]), 1.0),
                float4(getDynamicVerticesPositions(vertIndices[2]), 1.0)
            };

            prevPositions = transpose(float3x3(
                mul(inst.model, localPos[0]).xyz,
                mul(inst.model, localPos[1]).xyz,
                mul(inst.model, localPos[2]).xyz));
        }
    }
    else
    {
        const uint3 vertIndices = getVertIndicesStatic(inst.baseVertexIndex, inst.baseIndexIndex, primitiveId);
        
        const float4 localPos[3] =
        {
            float4(getStaticVerticesPositions(vertIndices[0]), 1.0),
            float4(getStaticVerticesPositions(vertIndices[1]), 1.0),
            float4(getStaticVerticesPositions(vertIndices[2]), 1.0)
        };

        const bool isMovable = (inst.flags & GEOM_INST_FLAG_IS_MOVABLE) != 0;
        const bool hasPrevInfo = inst.prevBaseVertexIndex != UINT32_MAX;

        // movable     -- use prev model matrix if exist
        // non-movable -- use current model matrix
        if (isMovable && hasPrevInfo)
        {
            // static geoms' local positions are constant, 
            // only model matrices are changing
            prevPositions = transpose(float3x3(
                mul(inst.prevModel, localPos[0]).xyz,
                mul(inst.prevModel, localPos[1]).xyz,
                mul(inst.prevModel, localPos[2]).xyz));
        }
        else
        {
            prevPositions = transpose(float3x3(
                mul(inst.model, localPos[0]).xyz,
                mul(inst.model, localPos[1]).xyz,
                mul(inst.model, localPos[2]).xyz));
        }
    }

    return prevPositions;
}

float4 packVisibilityBuffer(const ShPayload p)
{
    return float4(asfloat(p.instIdAndIndex), asfloat(p.geomAndPrimIndex), p.baryCoords);
}

int unpackInstCustomIndexFromVisibilityBuffer(const float4 v)
{
    int instanceID, instCustomIndex;
    unpackInstanceIdAndCustomIndex(asuint(v[0]), instanceID, instCustomIndex);

    return instCustomIndex;
}

void unpackVisibilityBuffer(
    const float4 v,
    out int instanceID, out int instCustomIndex,
    out int localGeomIndex, out int primIndex,
    out float2 bary)
{
    unpackInstanceIdAndCustomIndex(asuint(v[0]), instanceID, instCustomIndex);
    unpackGeometryAndPrimitiveIndex(asuint(v[1]), localGeomIndex, primIndex);
    bary = float2(v[2], v[3]);
}

// v must be fetched from framebufVisibilityBuffer_Prev_Sampled
bool unpackPrevVisibilityBuffer(const float4 v, out float3 prevPos)
{
    int prevInstanceID, instCustomIndex;
    int prevLocalGeomIndex, primIndex;

    unpackInstanceIdAndCustomIndex(asuint(v[0]), prevInstanceID, instCustomIndex);
    unpackGeometryAndPrimitiveIndex(asuint(v[1]), prevLocalGeomIndex, primIndex);

    int curFrameGlobalGeomIndex;
    const bool matched = getCurrentGeometryIndexByPrev(prevInstanceID, prevLocalGeomIndex, curFrameGlobalGeomIndex);

    if (!matched)
    {
        return false;
    }

    const float3x3 prevVerts = getOnlyCurPositions(curFrameGlobalGeomIndex, instCustomIndex, primIndex);
    const float3 baryCoords = float3(1.0 - v[2] - v[3], v[2], v[3]);

    prevPos = mul(prevVerts, baryCoords);

    return true;
}

float4x4 getModelMatrix(int instanceID, int localGeometryIndex)
{
    int globalGeometryIndex = getGeometryIndex(instanceID, localGeometryIndex);
    return geometryInstances[globalGeometryIndex].model;
}
#endif // DESC_SET_VERTEX_DATA
#endif // DESC_SET_GLOBAL_UNIFORM

#endif // VERTEX_DATA_HLSLI_
