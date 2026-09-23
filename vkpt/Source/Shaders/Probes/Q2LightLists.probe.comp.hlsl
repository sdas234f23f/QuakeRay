// Hand written counterpart of GLSL/Q2LightLists.probe.comp.
//
// Q2LightLists.hlsli has no shader stage of its own, so this pair exists to check Q2LightLists.hlsli
// against its golden. Its two consumers, RtQ2Indirect.rgen and RtRaygenDirect.rgen, reach it after
// ShaderCommonGLSLFunc.h, Surface.inl (which brings in Random.h and Light.h) and Q2Asvgf.h, in that
// order, and the probe uses the same order: Q2LightLists.hlsli needs Q2_GRAD_DWN and the two
// Q2_STRATUM_OFFSET_ constants of Q2Asvgf.hlsli, and the decoders it calls come from Light.hlsli,
// whose samplers must be declared for its bodies to compile at all even though this probe calls none
// of them. The descriptor set numbers are the consumers' own, not the probe's:
//
//   1 = FRAMEBUFFERS    framebufQ2GradSmplPos_Sampled (binding 239), the only image the header
//                       reads, in q2GetIsGradient
//   2 = GLOBAL_UNIFORM  fltEnable.x (the gradient gate), frameId, q2LightStatsMode and lightCount
//   4 = TEXTURES        globalTextures / globalTextures_Sampler: the header reads neither, but
//                       Light.hlsli's decoders name them, so the set has to be declared for those
//                       bodies to compile at all
//   6 = LIGHT_SOURCES   q2LightListOffsets (4), q2LightListLights (5), q2LightStats (6, the one
//                       writable buffer), q2ClusterSkyVis (8) and lightSources (the encoded lights
//                       the mass of a cluster slot is taken from)
//
// The header writes exactly one buffer, q2LightStats, and it does so in the two branches of
// q2AccumulateLightStats (the += of Q2_LIGHT_STATS_NON_ATOMIC and the InterlockedAdd of every other
// mode). The two calls below reach one branch each, with the mode and the visibility that select
// them; their stores are not slots of the table, because the slots are the probe's own output and
// the two halves are read store by store. What pins the stores instead is the read back at
// probeOutput[46], which lets the two disassemblies be read for the atomic and for the
// read-modify-write of the same address.
//
// Everything the header can fold is folded on both halves, and the slots are arranged so that the
// folded number of a site can be read out of the two disassemblies side by side:
//
//   probeOutput[0]         accumulator over every call of the file; it pins reach, not a value
//   probeOutput[1]..[7]    the one value constant of the header (16) together with the four
//                          generated bounds its arithmetic is written against (8192, 128, 6, 3)
//                          and the two encoded-light constants the mass of a slot depends on
//   probeOutput[8]..[11]   the three light-list helpers, on a constant cluster, plus the early
//                          return of q2ClusterSeesSky for a cluster past Q2_MAX_CLUSTERS
//   probeOutput[12]..[13]  q2GetIsGradient, under the uniform gate of fltEnable.x
//   probeOutput[14]..[16]  q2RoughnessSquareToSpecPower, at its floor, at its asymptote and in
//                          between
//   probeOutput[17]..[22]  the six sides of q2GetPrimaryDirectionSide, one slot each, so that the
//                          two sign tests and the three axis tests are pinned separately
//   probeOutput[23]..[24]  q2GetLightStatsAddr at both ends of its range
//   probeOutput[25]..[27]  q2Phong, on the two trivial configurations and on a constant that
//                          survives the exponent
//   probeOutput[28]..[31]  q2SphericalTriArea: with phongWeight 0 the slot is the solid angle
//                          alone, with phongWeight 1 it is the solid angle times the specular term,
//                          which is where the matrix is read; [30] and [31] are the two early
//                          returns of the function
//   probeOutput[32]..[37]  q2LightSelectionMass: the two lights of the buffer, so that all four
//                          branches are reached with data dependent lightType, and one light per
//                          branch encoded with constants, so that each branch folds
//   probeOutput[38]..[45]  q2SampleClusterLights: both statistics slots, both statistics modes and
//                          the early return for a cluster past Q2_MAX_CLUSTERS
//   probeOutput[46]        q2AccumulateLightStats and the read back of the address it wrote
//
// The one site of the golden whose spelling has no equivalent in HLSL is its matrix write:
// q2SphericalTriArea writes through the columns of its positions parameter, positions[i] = ...
// which HLSL does not index that way. The port builds the three columns as locals and rebuilds the
// matrix with transpose(float3x3(positions0, positions1, positions2)); this half builds its three
// constant matrices the same way, so that the two halves agree on which of the three vectors is
// which column. probeOutput[29] is the slot that can tell a correct rebuild from a transposed one:
// its triangle is not symmetric and its columns are not interchangeable, so the column-summed light
// direction and the row-summed one are far apart (about 1.94 against about 0.70). This file has to
// be edited together with the GLSL one: what one side reaches and the other does not is reported as
// a mismatch, which is exactly the point. Probes are not part of the shader build, so no probe blob
// is ever shipped.
//
// The spellings this half had to change, on top of the ivec2 -> int2, vec3 -> float3 and mat3 ->
// float3x3 renaming: a single-scalar vector constructor is a C style cast here (vec3(0.0) ->
// (float3)0.0, vec4(0.0) -> (float4)0.0), because dxc refuses to widen one lane into a constructor,
// and the four ShLightEncoded initializations are brace initializers instead of the golden's struct
// constructor: HLSL has no constructor for a struct at all (measured on this file, dxc reports
// "constructors only defined for numeric base types" for ShLightEncoded(...)), so the ten members
// are handed over in the declaration order of the generated struct. Q2Asvgf.hlsli solved the same
// problem with the q2MakeSH helper for its own struct; here the values are constants, so the brace
// form keeps them in the file. The probe set is spelled PROBE_DESC_SET 8 here and as the literal
// set = 8 on the GLSL half; both bind the probe output at set 8 binding 0.
//
// Where the halves fold differently, measured with spirv-dis on both disassemblies: dxc keeps the
// NMax of q2RoughnessSquareToSpecPower as an extended instruction, so probeOutput[14], [15] and
// [16] are not single constants here; their operands are the very numbers the GLSL half folds
// (NMax(0.01, 5.99680185), NMax(0.01, -1.50001252) and NMax(0.01, 19998)). glslc reaches those
// folds only under `spirv-opt -O`: its raw output has no inlining and keeps an OpFunctionCall at
// the three slots. Every number the two halves do fold is the same on both sides, and the value
// of every slot the two front ends refuse to fold is not a number on either side: q2Phong stays
// an extended instruction on both (Pow, or NMax inside Pow, on Reflect), q2SphericalTriArea stays
// a call or an OpPhi over Normalize/Cross/Atan2, and the buffer and uniform reads stay loads.

#define DESC_SET_FRAMEBUFFERS 1
#define DESC_SET_GLOBAL_UNIFORM 2
#define DESC_SET_TEXTURES 4
#define DESC_SET_LIGHT_SOURCES 6

#include "ShaderCommonHLSLFunc.hlsli"
#include "Random.hlsli"
#include "Light.hlsli"
#include "Q2Asvgf.hlsli"
#include "Q2LightLists.hlsli"

#define PROBE_DESC_SET 8

[[vk::binding(0, PROBE_DESC_SET)]] RWStructuredBuffer<float> probeOutput;

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // The four points of the matrix site. The triangle is deliberately off axis, not symmetric and
    // not flat, and p is not on it, so that a rebuild that puts the three vectors in the wrong
    // place changes the result beyond any rounding
    const float3 tri0 = float3(2.0, 0.0, 1.0);
    const float3 tri1 = float3(0.0, 2.0, 1.0);
    const float3 tri2 = float3(-2.0, 1.0, 3.0);
    const float3x3 tri = transpose(float3x3(tri0, tri1, tri2));
    const float3 p = float3(2.0, 2.0, 2.0);
    const float3 n = float3(0.0, 0.0, -1.0);
    const float3 V = float3(1.0, 1.0, 0.0);

    // Two more triangles for the two early returns of q2SphericalTriArea: the first one lies behind
    // the plane of a surface whose normal looks away from it, the second one is seen from a point
    // inside its plane, so that the dot products of the first test stay positive and the second
    // test is the one that has to fire
    const float3x3 triBehind = transpose(float3x3(float3(2.0, 0.0, -1.0), float3(0.0, 2.0, -1.0),
        float3(-2.0, 1.0, -1.0)));
    const float3x3 triFront = transpose(float3x3(float3(2.0, 0.0, 1.0), float3(0.0, 2.0, 1.0),
        float3(-2.0, 1.0, 1.0)));

    const float phongExp = 4.0;
    const float phongScale = 1.0;
    const float3 rng = float3(0.25, 0.5, 0.75);

    // One light per branch of q2LightSelectionMass, encoded with constants so that the branch the
    // slot belongs to is the one that folds. ShLightEncoded has ten members -- color, lightType and
    // data_0..data_7, eight float4 -- so every initializer below passes exactly eight; a branch
    // reads only the fields it names, and every other field is left at zero. HLSL has no struct
    // constructor (see the header comment), so the ten values are written as a brace initializer in
    // the declaration order of the generated struct. The mass of the triangle branch reads
    // data_0..data_2 as its three positions and their w components as the unnormalized normal of
    // the triangle, the textured area branch reads its polygon out of data_0..data_4 and its plane
    // out of data_7, and the spot branch reads data_0 as center and radius, data_1 as direction and
    // data_2.xy as the two cosines
    const ShLightEncoded sphereEnc = {
        float3(0.5, 0.25, 0.125), uint(LIGHT_TYPE_SPHERE),
        float4(3.0, 0.0, 4.0, 1.5), float4(0.6, 0.8, 0.0, 0.0), (float4)0.0, (float4)0.0, (float4)0.0,
        (float4)0.0, (float4)0.0, (float4)0.0 };
    const ShLightEncoded spotEnc = {
        float3(0.25, 0.5, 0.75), uint(LIGHT_TYPE_SPOT),
        float4(3.0, 0.0, 4.0, 1.5), float4(0.0, -1.0, 0.0, 0.0), float4(0.9, 0.5, 0.0, 0.0),
        (float4)0.0, (float4)0.0, (float4)0.0, (float4)0.0, (float4)0.0 };
    const ShLightEncoded areaEnc = {
        float3(1.0, 1.0, 1.0), uint(LIGHT_TYPE_TEXTURED_AREA),
        float4(1.0, 0.0, 0.0, 2.0), float4(0.0, 2.0, 0.0, 5.0), float4(0.0, 0.0, 3.0, 3.0),
        float4(0.0, 0.0, 1.0, 0.0), float4(0.0, 1.0, 0.0, 0.0), (float4)0.0, (float4)0.0,
        float4(0.0, 0.0, 1.0, 6.0) };
    const ShLightEncoded triEnc = {
        float3(0.5, 0.5, 0.5), uint(LIGHT_TYPE_TRIANGLE),
        float4(2.0, 0.0, 1.0, 0.0), float4(0.0, 2.0, 1.0, 0.0), float4(-2.0, 1.0, 3.0, 6.0),
        (float4)0.0, (float4)0.0, (float4)0.0, (float4)0.0, (float4)0.0 };

    float v = 0.0;

    // The value constant of the header and the four generated bounds its arithmetic is written
    // against, together with the two encoded-light constants a mass depends on, each one on its
    // own so that a wrong number cannot hide behind a use
    probeOutput[1] = (float)Q2_MAX_BRUTEFORCE_SAMPLING;
    probeOutput[2] = (float)Q2_MAX_CLUSTERS;
    probeOutput[3] = (float)Q2_LIGHT_LIST_MAX_PER_CELL;
    probeOutput[4] = (float)Q2_LIGHT_LIST_STATS_SIDES;
    probeOutput[5] = (float)Q2_LIGHT_LIST_STATS_BUFFERS;
    probeOutput[6] = (float)LIGHT_INDEX_NONE;
    probeOutput[7] = (float)LIGHT_ARRAY_REGULAR_LIGHTS_OFFSET;

    // The three light-list helpers. Their offset buffer is the host's, so nothing folds here: the
    // slots pin the shape of the two reads, the subtraction of the count and the shift and mask of
    // the sky table. [11] is the one that does fold, because the early return of
    // q2ClusterSeesSky fires for any cluster the table cannot hold
    probeOutput[8] = (float)q2GetClusterLightCount(1u);
    probeOutput[9] = (float)q2GetClusterLight(1u, 2u);
    probeOutput[10] = (float)q2ClusterSeesSky(3u);
    probeOutput[11] = (float)q2ClusterSeesSky(uint(Q2_MAX_CLUSTERS) + 5u);

    // The gradient test, under a uniform that decides it before any read. The division of the
    // coordinate by Q2_GRAD_DWN and the two bit fields of the sample word are inside the gate, so a
    // difference in either one shows up as a difference in the shape of these two slots
    probeOutput[12] = (float)q2GetIsGradient(int2(10, 20));
    probeOutput[13] = (float)q2GetIsGradient(int2(0, 0));

    // The specular power: at its floor of 0.01, at its asymptote of 19998 for a zero roughness,
    // and in between, where the exact float is the front end's business and both halves have to
    // agree on it
    probeOutput[14] = q2RoughnessSquareToSpecPower(0.5);
    probeOutput[15] = q2RoughnessSquareToSpecPower(2.0);
    probeOutput[16] = q2RoughnessSquareToSpecPower(0.0);

    // The six sides, one per slot, so that each of the five comparisons of the function is pinned
    // by a number of its own and an inverted test cannot be compensated by another one
    probeOutput[17] = (float)q2GetPrimaryDirectionSide(float3(1.0, 0.0, 0.0));
    probeOutput[18] = (float)q2GetPrimaryDirectionSide(float3(-1.0, 0.0, 0.0));
    probeOutput[19] = (float)q2GetPrimaryDirectionSide(float3(0.0, 1.0, 0.0));
    probeOutput[20] = (float)q2GetPrimaryDirectionSide(float3(0.0, -1.0, 0.0));
    probeOutput[21] = (float)q2GetPrimaryDirectionSide(float3(0.0, 0.0, 1.0));
    probeOutput[22] = (float)q2GetPrimaryDirectionSide(float3(0.0, 0.0, -1.0));

    // The statistics address, at the first slot of the first cell and at the last slot of the
    // second cell on side 3, which is 1536 and 1566 for the layout the two multipliers fix
    probeOutput[23] = (float)q2GetLightStatsAddr(1u, 0u, 0u);
    probeOutput[24] = (float)q2GetLightStatsAddr(1u, 2u, 3u);

    // The phong term: 1.0 for a light that sits on the normal, 0.0 for one that sits behind the
    // surface, and 0.216 for a light in the tangent plane of a surface that looks along x, which
    // is the one of the three that can tell the order of the reflection
    probeOutput[25] = q2Phong(n, n, n, phongExp);
    probeOutput[26] = q2Phong(n, -n, n, phongExp);
    probeOutput[27] = q2Phong(float3(0.0, 0.0, 1.0), float3(-0.6, -0.8, 0.0),
        float3(1.0, 0.0, 0.0), 3.0);

    // The solid angle of a triangle, three times: alone (phongWeight 0 makes the brdf 1.0), with
    // the specular term of the same triangle on top of it (phongWeight 1), and for the two
    // triangles the early returns of the function are written for
    probeOutput[28] = q2SphericalTriArea(tri, p, n, V, phongExp, phongScale, 0.0);
    probeOutput[29] = q2SphericalTriArea(tri, p, n, V, phongExp, phongScale, 1.0);
    probeOutput[30] = q2SphericalTriArea(triBehind, (float3)0.0, float3(0.0, 0.0, 1.0), V,
        phongExp, phongScale, 1.0);
    probeOutput[31] = q2SphericalTriArea(triFront, (float3)0.0, float3(0.0, 0.0, 1.0), V,
        phongExp, phongScale, 1.0);

    // The mass of a light. The first two read the encoded lights of the buffer, so all four
    // branches are reached with a light type that comes from memory; the next four take a light
    // whose type is a constant, so each branch is also instantiated on its own
    probeOutput[32] = q2LightSelectionMass(lightSources[0], p, n, V, phongExp, phongScale, 0.5);
    probeOutput[33] = q2LightSelectionMass(lightSources[1], p, n, V, phongExp, phongScale, 0.5);
    probeOutput[34] = q2LightSelectionMass(sphereEnc, p, n, V, phongExp, phongScale, 0.5);
    probeOutput[35] = q2LightSelectionMass(spotEnc, p, n, V, phongExp, phongScale, 0.5);
    probeOutput[36] = q2LightSelectionMass(areaEnc, p, n, V, phongExp, phongScale, 0.5);
    probeOutput[37] = q2LightSelectionMass(triEnc, p, n, V, phongExp, phongScale, 1.0);

    // The sampler of the header. Its outputs are the three slots of the call. Both statistics slot
    // orders are exercised, so that the two frameSlot arithmetic expressions are reached, and the
    // gradient flag, which selects between them, is exercised in both states; the last call is the
    // early return for a cluster past Q2_MAX_CLUSTERS, which leaves the three outputs at the values
    // the function starts from
    uint li = 0u;
    uint slot = 0u;
    float pdf = 0.0;

    q2SampleClusterLights(1u, p, n, V, phongExp, phongScale, 1.0, true, rng, li, slot, pdf);
    probeOutput[38] = (float)li;
    probeOutput[39] = (float)slot;
    probeOutput[40] = pdf;

    q2SampleClusterLights(1u, p, n, V, phongExp, phongScale, 1.0, false, rng, li, slot, pdf);
    probeOutput[41] = (float)li;
    probeOutput[42] = (float)slot;
    probeOutput[43] = pdf;

    q2SampleClusterLights(uint(Q2_MAX_CLUSTERS) + 1u, p, n, V, phongExp, phongScale, 1.0, true, rng,
        li, slot, pdf);
    probeOutput[44] = (float)li;
    probeOutput[45] = pdf;

    // The statistics. The two calls differ in the mode and in the visibility, so both the
    // read-modify-write branch and the atomic branch are reached, and the sample index gate of
    // Q2_LIGHT_STATS_FIRST_SAMPLE is reached in both states as well. The read back of the same
    // address is the slot that shows what the write left there, and it is also what lets the two
    // disassemblies be compared on the address arithmetic of the frame base
    q2AccumulateLightStats(1u, 0u, n, 1.0, 0u);
    q2AccumulateLightStats(1u, 0u, -n, 0.0, 1u);
    probeOutput[46] = (float)q2LightStats[((globalUniform.frameId % uint(Q2_LIGHT_LIST_STATS_BUFFERS))
        * uint(Q2_MAX_CLUSTERS) * uint(Q2_LIGHT_LIST_MAX_PER_CELL)
        * uint(Q2_LIGHT_LIST_STATS_SIDES) * 2u) + q2GetLightStatsAddr(1u, 0u, 0u)];

    // Reach: every function of the header is called at least once above, and this accumulator is
    // the single store whose operand lists all of them
    v += (float)q2GetClusterLightCount(1u);
    v += (float)q2GetClusterLight(1u, 2u);
    v += (float)q2ClusterSeesSky(3u);
    v += (float)q2ClusterSeesSky(uint(Q2_MAX_CLUSTERS) + 5u);
    v += (float)q2GetIsGradient(int2(10, 20));
    v += (float)q2GetIsGradient(int2(0, 0));
    v += q2RoughnessSquareToSpecPower(0.5);
    v += q2RoughnessSquareToSpecPower(2.0);
    v += q2RoughnessSquareToSpecPower(0.0);
    v += (float)q2GetPrimaryDirectionSide(float3(1.0, 0.0, 0.0));
    v += (float)q2GetPrimaryDirectionSide(float3(-1.0, 0.0, 0.0));
    v += (float)q2GetPrimaryDirectionSide(float3(0.0, 1.0, 0.0));
    v += (float)q2GetPrimaryDirectionSide(float3(0.0, -1.0, 0.0));
    v += (float)q2GetPrimaryDirectionSide(float3(0.0, 0.0, 1.0));
    v += (float)q2GetPrimaryDirectionSide(float3(0.0, 0.0, -1.0));
    v += (float)q2GetLightStatsAddr(1u, 0u, 0u);
    v += (float)q2GetLightStatsAddr(1u, 2u, 3u);
    v += q2Phong(n, n, n, phongExp);
    v += q2Phong(n, -n, n, phongExp);
    v += q2Phong(float3(0.0, 0.0, 1.0), float3(-0.6, -0.8, 0.0), float3(1.0, 0.0, 0.0), 3.0);
    v += q2SphericalTriArea(tri, p, n, V, phongExp, phongScale, 0.0);
    v += q2SphericalTriArea(tri, p, n, V, phongExp, phongScale, 1.0);
    v += q2SphericalTriArea(triBehind, (float3)0.0, float3(0.0, 0.0, 1.0), V, phongExp,
        phongScale, 1.0);
    v += q2SphericalTriArea(triFront, (float3)0.0, float3(0.0, 0.0, 1.0), V, phongExp,
        phongScale, 1.0);
    v += q2LightSelectionMass(lightSources[0], p, n, V, phongExp, phongScale, 0.5);
    v += q2LightSelectionMass(lightSources[1], p, n, V, phongExp, phongScale, 0.5);
    v += q2LightSelectionMass(sphereEnc, p, n, V, phongExp, phongScale, 0.5);
    v += q2LightSelectionMass(spotEnc, p, n, V, phongExp, phongScale, 0.5);
    v += q2LightSelectionMass(areaEnc, p, n, V, phongExp, phongScale, 0.5);
    v += q2LightSelectionMass(triEnc, p, n, V, phongExp, phongScale, 1.0);
    v += (float)li + (float)slot + pdf;
    v += (float)probeOutput[46];

    probeOutput[0] = v;
}
