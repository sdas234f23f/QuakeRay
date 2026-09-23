// Hand written counterpart of GLSL/VertexPreprocessPartial.probe.comp.
//
// The probe of VertexPreprocessPartial.hlsli. The fragment is included three times, once per
// preprocessing mode, which is how CmVertexPreprocess.comp uses it: it defines
// VERTEX_PREPROCESS_PARTIAL_DYNAMIC for the dynamic instances, VERTEX_PREPROCESS_PARTIAL_STATIC_ALL
// for the "all" mode of preprocessMode, and the movable mode is the one whose branch is commented
// out there. All three are pinned here at once, so each of the macros the fragment expands is
// expanded under every mode. The one accessor the golden aliases and never calls - GET_NORMALS - is
// called directly below, so that both normals buffers are part of this probe, too.
//
// The fragment reads tlasInstanceIndex and groupThreadID from the scope of the shader that includes
// it, so the entry point here declares them the way the consumer has to: the geometry instance of
// the workgroup as SV_GroupID and the invocation id as SV_GroupThreadID.
//
// probeOutput[1..14] are the folded controls. The golden's mat3(model) is built from a 4x4 whose
// sixteen elements are all distinct, so the off-diagonal reads below show both that the cast took
// the upper left submatrix and that no row or column was swapped on the way.

#define DESC_SET_GLOBAL_UNIFORM 0
#define DESC_SET_VERTEX_DATA    1
#define VERTEX_BUFFER_WRITEABLE

#include "ShaderCommonHLSLFunc.hlsli"

#define PROBE_DESC_SET 8

[[vk::binding(0, PROBE_DESC_SET)]] RWStructuredBuffer<float> probeOutput;

[numthreads(COMPUTE_VERT_PREPROC_GROUP_SIZE_X, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadID)
{
    uint tlasInstanceIndex = groupID.x;

    // The sum, named probeSum and not v: the fragment declares a v of its own in the non indexed
    // loop, and both loops of the fragment only write to the vertex buffers
    float probeSum = 0.0;

    // The dynamic mode, as CmVertexPreprocess.comp includes it
    {
        #define VERTEX_PREPROCESS_PARTIAL_DYNAMIC
        #include "VertexPreprocessPartial.hlsli"
    }

    // The static mode of preprocessMode == VERT_PREPROC_MODE_ALL
    {
        #define VERTEX_PREPROCESS_PARTIAL_STATIC_ALL
        #include "VertexPreprocessPartial.hlsli"
    }

    // The movable only branch, which the consumer has commented out: the early out is the only
    // part of the fragment that is guarded by this mode
    {
        #define VERTEX_PREPROCESS_PARTIAL_STATIC_MOVABLE
        #include "VertexPreprocessPartial.hlsli"
    }

    // GET_NORMALS, the accessor macro the golden defines and never expands
    probeSum += getDynamicVerticesNormals(0u).x + getStaticVerticesNormals(0u).x;

    probeOutput[0] = probeSum;

    // The same logical 4x4 on both sides: HLSL fills it row by row, GLSL column by column, and all
    // sixteen elements are distinct dyadic numbers, so every element below is identifiable on its
    // own. The upper left 3x3 is fully off diagonal.
    const float4x4 probeModel = float4x4(
        0.0625, 0.1250, 0.1875, 0.2500,
        0.3125, 0.3750, 0.4375, 0.5000,
        0.5625, 0.6250, 0.6875, 0.7500,
        0.8125, 0.8750, 0.9375, 1.0000);

    // The golden's mat3(model)
    const float3x3 probeModel3 = (float3x3)probeModel;

    // The diagonal elements keep the golden's spelling on both sides; an off diagonal one is taken
    // from getColumn here, because a single index of an HLSL matrix is a row
    probeOutput[1]  = probeModel3[0][0];                        // (0,0)
    probeOutput[2]  = probeModel3[1][1];                        // (1,1)
    probeOutput[3]  = getColumn(probeModel3, 0).y;              // (1,0)
    probeOutput[4]  = getColumn(probeModel3, 1).x;              // (0,1)
    probeOutput[5]  = getColumn(probeModel3, 1).z;              // (2,1)
    probeOutput[6]  = getColumn(probeModel3, 2).y;              // (1,2)
    probeOutput[7]  = getColumn(probeModel,  3).y;              // (1,3), outside the submatrix
    probeOutput[8]  = getColumn(probeModel,  0).w;              // (3,0), outside the submatrix

    // The normal sign of the golden, float((flags & GEOM_INST_FLAG_INVERTED_NORMALS) == 0) * 2.0 -
    // 1.0, folded for both polarities of that flag
    probeOutput[9]  = float((uint(GEOM_INST_FLAG_INVERTED_NORMALS) & GEOM_INST_FLAG_INVERTED_NORMALS) == 0) * 2.0 - 1.0;
    probeOutput[10] = float((0u                                  & GEOM_INST_FLAG_INVERTED_NORMALS) == 0) * 2.0 - 1.0;

    // and the genNormals test of the golden, folded with the flag set
    probeOutput[11] = float((uint(GEOM_INST_FLAG_GENERATE_NORMALS) & GEOM_INST_FLAG_GENERATE_NORMALS) != 0);

    // The triangle normal of the fragment, with the same three vertices under the same indices:
    // localPos[1] - localPos[0] and localPos[2] - localPos[0] of a triangle in the xy plane. The
    // three vertices are separate constants and not a const array: dxc keeps a const array in local
    // memory and loads it, while these get folded into the operands of the cross product
    const float3 probeLocalPos0 = float3(0.25, 0.25, 0.25);
    const float3 probeLocalPos1 = float3(0.75, 0.25, 0.25);
    const float3 probeLocalPos2 = float3(0.25, 0.75, 0.25);

    probeOutput[12] = -1.0 * normalize(cross(probeLocalPos1 - probeLocalPos0, probeLocalPos2 - probeLocalPos0)).z;
    probeOutput[13] = +1.0 * normalize(cross(probeLocalPos1 - probeLocalPos0, probeLocalPos2 - probeLocalPos0)).z;

    // GLSL gl_WorkGroupSize.x, which is COMPUTE_VERT_PREPROC_GROUP_SIZE_X here
    probeOutput[14] = float(COMPUTE_VERT_PREPROC_GROUP_SIZE_X);
}
