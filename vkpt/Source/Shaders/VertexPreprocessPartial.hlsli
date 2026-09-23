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


// HLSL counterpart of VertexPreprocessPartial.inl. It is a fragment and not a shader: like the GLSL
// one it includes nothing itself, so the shader that includes it has to pull in
// ShaderCommonHLSLFunc.hlsli first and to define exactly one of VERTEX_PREPROCESS_PARTIAL_DYNAMIC,
// VERTEX_PREPROCESS_PARTIAL_STATIC_ALL or VERTEX_PREPROCESS_PARTIAL_STATIC_MOVABLE. It is
// deliberately unguarded - CmVertexPreprocess.comp includes it once per preprocessing mode - and it
// reads two names out of the scope it is included into: tlasInstanceIndex, the geometry instance of
// this workgroup, and groupThreadID, which is the SV_GroupThreadID parameter of the entry point and
// where gl_LocalInvocationID comes from here.
//
// Spellings that had to change:
//   * GLSL mat4 -> float4x4, and GLSL mat3(model) -> (float3x3)model: dxc rejects float3x3(model)
//     ("too many elements in vector initialization") and reads the cast as the upper left submatrix,
//     exactly as GLSL reads mat3(mat4)
//   * GLSL gl_LocalInvocationID.x -> groupThreadID.x, as described above: a fragment has no access
//     to the invocation id by itself
//   * GLSL gl_WorkGroupSize.x -> COMPUTE_VERT_PREPROC_GROUP_SIZE_X: in HLSL the workgroup size is
//     an attribute of the entry point, so [numthreads] becomes the LocalSize execution mode and
//     nothing inside the shader is left that holds it; the constant is the same 256 that the GLSL
//     shader declares as local_size_x
//   * GLSL uvec3 -> uint3, and GLSL arrays whose size is deduced from the initializer (localPos)
//     spell that size out
//
// What did not change: the three modes and the buffers they alias, the int4 unpacking of
// instanceGeomInfoOffset and instanceGeomCount, the loop bounds and the increment, the movable
// early out, the normal sign expression (float(bool) is an HLSL cast), both triangle loops and the
// seven #undefs at the end.
//
// Two things in the golden are dead and are ported as dead, not "fixed": GET_NORMALS is defined for
// all three modes and never expanded, and model3 is built and never read.


#if defined(VERTEX_PREPROCESS_PARTIAL_DYNAMIC)

    #define GET_POSITIONS getDynamicVerticesPositions
    #define GET_NORMALS getDynamicVerticesNormals
    #define SET_NORMALS setDynamicVerticesNormals
    #define INDICES dynamicIndices

#elif defined(VERTEX_PREPROCESS_PARTIAL_STATIC_ALL) || defined(VERTEX_PREPROCESS_PARTIAL_STATIC_MOVABLE)

    #define GET_POSITIONS getStaticVerticesPositions
    #define GET_NORMALS getStaticVerticesNormals
    #define SET_NORMALS setStaticVerticesNormals
    #define INDICES staticIndices

#else
    #error
#endif




// translate from local to global geom index
const int geomIndexOffset = globalUniform.instanceGeomInfoOffset[tlasInstanceIndex / 4][tlasInstanceIndex % 4];
const int geomCount = globalUniform.instanceGeomCount[tlasInstanceIndex / 4][tlasInstanceIndex % 4];

for (uint localGeomIndex = groupThreadID.x; localGeomIndex < geomCount; localGeomIndex += COMPUTE_VERT_PREPROC_GROUP_SIZE_X)
{
    const ShGeometryInstance inst = geometryInstances[geomIndexOffset + localGeomIndex];

#if defined(VERTEX_PREPROCESS_PARTIAL_STATIC_MOVABLE)
    const bool isMovable = (inst.flags & GEOM_INST_FLAG_IS_MOVABLE) != 0;
    
    // ignore non-movable if preprocess mode allows only movable
    if (!isMovable)
    {
        continue;
    }
#endif

    const bool useIndices = inst.baseIndexIndex != UINT32_MAX;
    const bool genNormals = (inst.flags & GEOM_INST_FLAG_GENERATE_NORMALS) != 0;
    // -1 if normals should be inverted
    const float normalSign = float((inst.flags & GEOM_INST_FLAG_INVERTED_NORMALS) == 0) * 2.0 - 1.0;

    const float4x4 model = inst.model;
    const float3x3 model3 = (float3x3)model;


    if (useIndices)
    {
        for (uint tri = 0; tri < inst.indexCount / 3; tri++)
        {
            const uint i = inst.baseIndexIndex + tri * 3;

            const uint3 vertexIndices = uint3(
                inst.baseVertexIndex + INDICES[i + 0],
                inst.baseVertexIndex + INDICES[i + 1],
                inst.baseVertexIndex + INDICES[i + 2]);

            const float3 localPos[3] = 
            {
                GET_POSITIONS(vertexIndices[0]),
                GET_POSITIONS(vertexIndices[1]),
                GET_POSITIONS(vertexIndices[2])
            };

            float3 localNormal;
            
            if (genNormals)
            {
                localNormal = normalSign * normalize(cross(localPos[1] - localPos[0], localPos[2] - localPos[0]));
                
                SET_NORMALS(vertexIndices[0], localNormal);
                SET_NORMALS(vertexIndices[1], localNormal);
                SET_NORMALS(vertexIndices[2], localNormal);
            }
        }
    }
    else
    {
        for (uint tri = 0; tri < inst.vertexCount / 3; tri++)
        {
            const uint v = inst.baseVertexIndex + tri * 3;

            const uint3 vertexIndices = uint3(
                v + 0,
                v + 1,
                v + 2);

            const float3 localPos[3] = 
            {
                GET_POSITIONS(vertexIndices[0]),
                GET_POSITIONS(vertexIndices[1]),
                GET_POSITIONS(vertexIndices[2])
            };

            float3 localNormal;

            if (genNormals)
            {
                localNormal = normalSign * normalize(cross(localPos[1] - localPos[0], localPos[2] - localPos[0]));

                SET_NORMALS(vertexIndices[0], localNormal);
                SET_NORMALS(vertexIndices[1], localNormal);
                SET_NORMALS(vertexIndices[2], localNormal);
            }
        } 
    }
}


#undef GET_POSITIONS
#undef GET_NORMALS
#undef SET_NORMALS
#undef INDICES

#undef VERTEX_PREPROCESS_PARTIAL_STATIC_ALL
#undef VERTEX_PREPROCESS_PARTIAL_STATIC_MOVABLE
#undef VERTEX_PREPROCESS_PARTIAL_DYNAMIC
