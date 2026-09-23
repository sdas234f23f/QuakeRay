// Hand written counterpart of GLSL/RaygenPrimary.probe.rgen.
//
// RaygenPrimary.inl has no stage of its own: it is the raygen side of the renderer, guarded by
// exactly one of RAYGEN_PRIMARY_SHADER, RAYGEN_REFL_REFR_SHADER and Q2_REFL_REFR_SHADER, and
// included by RtRaygenPrimary.rgen, RtRaygenReflRefr.rgen and RtQ2ReflectRefract.rgen. This pair
// pins one concrete configuration of it and says which: RAYGEN_PRIMARY_SHADER, the one
// RtRaygenPrimary.rgen ships. The other two entry points cannot be compiled in the same translation
// unit (the first #error of the golden forbids it), so they are covered the way every other
// configuration-dependent statement is: they were compiled as separate scratch configurations with
// both compilers (glslc and dxc, exit 0 each), and the matrix sites that live only in them, the
// portal block of RtRaygenReflRefr.rgen (:563-575) and the portal block plus the
// projection * view chain of RtQ2ReflectRefract.rgen (:812-824, :1045), are folded here as literal
// re-spellings with a negative control each.
//
// Why this probe is a .rgen and not a .comp: the golden's three entry points use the ray tracing
// builtins (rayPayloadEXT, traceRayEXT, gl_LaunchIDEXT), and a compute stage cannot parse them.
// Measured with the shipped SDK: `glslc --target-env=vulkan1.2` on a scratch compute shader that
// includes the golden reports `'rayPayloadEXT' : not supported in this stage: compute` and
// `'gl_LaunchIDEXT' : undeclared identifier`, exit 1, and `dxc -T cs_6_2` refuses the same bodies
// because TraceRay is a lib_6_3 intrinsic. The accepted RaygenCommon pair is a .rgen with the
// lib_6_3 profile for the same reason. The exact commands:
//
//   glslc --target-env=vulkan1.2 -I . -I ../Generated/ GLSL/RaygenPrimary.probe.rgen -o <out>
//   dxc -spirv -T lib_6_3 -fspv-target-env=vulkan1.2 -I . -I ../Generated/ Probes/RaygenPrimary.probe.rgen.hlsl -Fo <out>
//
// The golden's entry point is renamed out of the way and then called from this file's main, so that
// the whole primary entry point -- its tracePrimaryRay call, the getHitInfoPrimaryRay result, the
// sky branch, the Q2RTX G-buffer store and every imageStore it performs -- is instantiated and
// reachable in the module. The rename is the same mechanism on both halves: `#define main NAME`
// before the include and `#undef main` after it. The one HLSL-only piece is
// RAYGEN_PRIMARY_ENTRY_ATTR: the header puts [shader("raygeneration")] on its three entry points so
// that a shipping consumer is the golden's three lines plus an include, and this probe defines that
// macro empty because a translation unit must not turn the golden's renamed main into a second
// entry point (measured: with the attribute left on, dxc emits two OpEntryPoints, `main` and
// `raygenPrimaryPrimaryMain`, while glslang's renamed function is not an entry point at all, and
// the checker's entry point property differs by exactly that).
//
// Every other function of the golden is instantiated here directly: getMotionVectorForUpscaler,
// getMotionForInfinitePoint, storeQ2GBuffer, storeSky, getNewRayMedia, getWaterNormal, lookAt,
// getPortalNormal, isBackface and getNormal, the last one on both of its branches. The three
// functions of the pinned entry point itself are the primary main, so no function of the file is
// left uncalled.
//
// The arguments of the calls that cannot fold are chosen so that no divisor is zero and no
// normalized vector is zero.
//
// The folded sites below come in pairs: the correct spelling of the port, and the deliberately
// wrong arithmetic that the same inputs could have produced. Every operand is literal, dyadic
// (0.25, 0.5, -0.125, 1.5, ...), not symmetric and not axis aligned, so a transposed read, a row
// read or a swapped operand order folds to a different number and the agreement of the correct
// pair proves the operand order instead of assuming it. The comparison is taken from spirv-dis of
// both halves after `spirv-opt -O`, because glslc does not fold this much by default while dxc
// does.
//
// What folds and what does not was measured operation by operation on literals before the slots
// were chosen: matrix * vector, transpose(matrix), composite extract, dot, sqrt, division, atan2,
// sin/cos, min/max, exp2 and pow fold on both halves; cross(), normalize(), length(), floor() (and
// with it the golden's mod()) and matrix * matrix() do not fold on the dxc half at all, and cross()
// does not fold on the glslc half either. Four consequences shape this file:
//   * the shipped lookAt() call is instantiated (in the v accumulation at the end) but cannot fold,
//     because it is built from two cross products; the constructor respell of :237 is therefore
//     proven on the explicit component form of the same two cross products (slots 48..65), which
//     is the same arithmetic on both halves;
//   * the shipped normalize() call (:266) does not fold on the dxc half, so the fold slots 72..77
//     use the literal (0.25, 0.5, 1.0) as the operand in place of normalize(localN). The literal
//     that equals the actual normalized value is not used because its components are not dyadic
//     and the two folders then disagree in the last bit (measured before the substitution: glslc
//     0.46371302 against dxc 0.46371299); the intrinsic itself is textually identical on both
//     halves and is not one of the respellings this pair proves;
//   * the left associative :1045 chain cannot fold on the dxc half, because its inner
//     matrix * matrix has no folder there: slots 118..121 are that spelling as it ships and the
//     structure (OpMatrixTimesMatrix then OpVectorTimesMatrix) is visible in the disassembly,
//     while slots 122..125 are the right associative spelling, which does fold on the dxc half and
//     folds to the same number, because with these dyadic inputs both associations are exact; the
//     glslc half folds both spellings and proves that the two numbers are equal on these inputs;
//   * the golden's mod() (:250) folds on the glslc half to 1 and 3 and does not fold on the dxc
//     half (it keeps OpExtInst Floor); the truncating remainder of slot 80 is the separating
//     control on the glslc half (-1 against 3).
//
// The two row/column controls deserve a note of their own: for the ONB that getONB builds, row i
// and column i share their first two components for every input (b2.x equals b1.y by construction),
// so only the third component separates them; the control slots 35..37 and 46..47 carry all three
// components, and the numbers below show the separation in .z (0.25 against -0.25). A control that
// read only .x would have been blind, the way the first Light.hlsli control was.
//
// probeOutput slot map, identical in both halves, with the numbers measured from spirv-dis of
// Build/RaygenPrimary.probe.rgen.{glsl,hlsl}.spv after spirv-opt -O:
//
//    0        v, the accumulator (not folded)
//    1..3     :69 mat3(view) * d                      0.625, 3, 1.375
//    4..6     :69 d * mat3(view), control             0.375, 0.5, 3.125
//    7        :69 with the real globalUniform.view (not folded)
//    8..10    :70 mat3(viewPrev) * d                  1.25, -1, 2.375
//    11..13   :70 mirrored, control                   -0.25, 1.25, 1.375
//    14..16   :72 over the :69 result                 1.15625, 4.484375, 2.296875
//    17..19   :72 projection mirrored, control        1.53125, 4.921875, 1.171875
//    20..22   :73 over the :70 result                 2.265625, -1.5625, 2.90625
//    23..25   :73 projectionPrev mirrored, control    2.1875, -0.546875, 3.375
//    26..27   the motion of getMotionForInfinitePoint 0.5546875, -3.0234375
//    28       the motion with the operands swapped, control -0.5546875
//    29..31   :182 basis[0] read with getColumn       0.964285731, -0.0714285746, -0.25
//    32..34   :183 basis[1] read with getColumn       -0.0714285746, 0.857142866, -0.5
//    35..37   basis row 0, control                    0.964285731, -0.0714285746, 0.25
//    38..40   :229 basis * n                          1.57142854, 3.14285707, 1
//    41..43   n * basis, control                      0.071428597, 0.142857194, 3.5
//    44..45   :183 dot(position, basis[0/1])          0.0267857313, -0.946428537
//    46..47   dot(position, row 0/1), control         0.526785731, 0.0535714328
//    48..56   :237 the constructor respell on the explicit cross arithmetic, column by column:
//             -1, 0, 0.25 | -0.125, 1.0625, -0.5 | 0.25, 0.5, 1
//    57..65   the same without the transpose, control:
//             -1, -0.125, 0.25 | 0, 1.0625, 0.5 | 0.25, -0.5, 1
//    66..67   :253 dot(offset, inLookAt[0/1])         0.375, -0.875
//    68..69   the same with the rows, control         -0.125, 0
//    70       :257 atan2(localOffset.y, localOffset.x) -1.16590452
//    71       the same with the arguments swapped, control 2.73670077
//    72..74   :266 inLookAt * localN                  0.125, 0.875, 0.53125
//    75..77   localN * inLookAt, control              0.125, 0, 0.96875
//    78..79   :250 mod(5.0, 2.0), mod(-1.0, 4.0)      1, 3 (glslc; dxc keeps Floor)
//    80       the truncating remainder, control        -1 (glslc; dxc keeps Trunc)
//    81..83   :570 the portal rotation of block 1     0.828125, -0.046875, -1.40625
//    84..86   the same without transpose(inLookAt), control 0.546875, -2.015625, 1.125
//    87..89   the same with the two products swapped, control 0.8046875, -0.546875, -0.9375
//    90..91   :572 inLookAt[0/1] read with getColumn  0.375, -0.875
//    92..93   the same with the rows, control         -0.125, 0
//    94..96   :575 localOffset.x * outLookAt[0] + ... 0.25, -0.1875, -0.1875
//    97..99   the same with the rows of outLookAt, control 0.0625, -0.28125, 0.4375
//    100..102 :819 the portal rotation of block 2     -2.5625, 0.640625, -2.34375
//    103..105 the same without transpose(inLookAt), control -2.375, 1.4375, -1.9375
//    106..107 :821 inLookAt[0/1] read with getColumn  -0.5625, 0.6875
//    108..109 the same with the rows, control         -0.25, 0.625
//    110..112 :824 localOffset.x * outLookAt[0] + ... -0.3125, 0.875, -0.0625
//    113..115 the same with the rows of outLookAt, control 0.4375, -0.25, 0.4375
//    116..117 the two recompositions dotted with themselves 0.1328125, 0.8671875
//    118..121 :1045 projection * view * vec4(refrHitPosition, 1.0), left associative:
//             0.67578125, 0.3515625, 2.484375, 0.21875 (glslc only; dxc does not fold it)
//    122..125 the same, right associative, the same number on both halves:
//             0.67578125, 0.3515625, 2.484375, 0.21875
//    126..129 the same with the two matrices in the other order, control:
//             0.078125, 1.03125, 1.85546875, 0.25
//    130      :1045 with the real projection and view (not folded)
//
// The ten slots without the same number on both halves are exactly the ones named in the map:
// 0 and 130 (the accumulator and the shipped :1045 statement, runtime values), 7 (the shipped
// :69 statement, a runtime uniform), 78..80 (the dxc half keeps OpExtInst Floor and Trunc), and
// 118..121 (the dxc half keeps OpMatrixTimesMatrix for the inner product). Every other slot --
// 121 of the 131 -- is bit-identical on the two halves.
//
// The two halves spell the row reads as transpose(m)[i] (GLSL) and getColumn(transpose(m), i)
// (HLSL); both are the row of the same logical matrix, and the defect they guard against is an HLSL
// row read `m[i]` where the golden means a column.
//
// This file has to be edited together with its other half. Measured with spirv-dis on
// Build/RaygenPrimary.probe.rgen.{glsl,hlsl}.spv, both halves additionally through spirv-opt -O.

#define MATERIAL_MAX_ALBEDO_LAYERS 1

#define RAYGEN_PRIMARY_ENTRY_ATTR
#define RAYGEN_PRIMARY_SHADER
#define main raygenPrimaryPrimaryMain
#include "RaygenPrimary.hlsli"
#undef main

#define PROBE_DESC_SET 12

[[vk::binding(0, PROBE_DESC_SET)]] RWStructuredBuffer<float> probeOutput;

// The literal matrices. Each is the same logical matrix in both halves: mat4() takes columns in
// GLSL and float4x4() takes rows in HLSL, so the two argument lists are each other's transpose.
// Every upper-left 3x3 is off-diagonal and not symmetric, and no entry that a product divides by
// is zero.
//
//   view      rows: ( 0.5,   0.25,  -0.125, 0.75  ) (-0.25,  0.875,  0.5,   -0.5  )
//                   ( 0.125, -0.5,   0.75,  0.25  ) ( 0.0,   0.0,    0.0,    1.0  )
//   viewPrev  rows: ( 0.75, -0.125,  0.25,  0.5   ) ( 0.25,  0.5,   -0.75,  -0.25 )
//                   (-0.5,   0.125,  0.875, 0.125 ) ( 0.0,   0.0,    0.0,    1.0  )
//   proj      rows: ( 1.25,  0.125,  0.0,  -0.25  ) ( 0.25,  1.5,   -0.125,  0.5  )
//                   ( 0.0,   0.25,   1.125, 0.75  ) ( 0.0,   0.0,   -0.5,    1.0  )
//   projPrev  rows: ( 1.375,-0.25,   0.125, 0.25  ) ( 0.125, 1.125, -0.25,   0.75 )
//                   ( 0.25,  0.375,  1.25, -0.125 ) ( 0.0,   0.0,   -0.625,  1.0  )
#define PROBE_VIEW_M4     float4x4( 0.5, 0.25, -0.125, 0.75,    -0.25, 0.875, 0.5, -0.5,    0.125, -0.5, 0.75, 0.25,      0.0, 0.0, 0.0, 1.0 )
#define PROBE_VIEWPREV_M4 float4x4( 0.75, -0.125, 0.25, 0.5,    0.25, 0.5, -0.75, -0.25,   -0.5, 0.125, 0.875, 0.125,   0.0, 0.0, 0.0, 1.0 )
#define PROBE_PROJ_M4     float4x4( 1.25, 0.125, 0.0, -0.25,    0.25, 1.5, -0.125, 0.5,    0.0, 0.25, 1.125, 0.75,     0.0, 0.0, -0.5, 1.0 )
#define PROBE_PROJPREV_M4 float4x4( 1.375, -0.25, 0.125, 0.25,  0.125, 1.125, -0.25, 0.75,  0.25, 0.375, 1.25, -0.125,  0.0, 0.0, -0.625, 1.0 )

#define PROBE_DIRECTION float3(1.0, 2.0, 3.0)

// getONB / getWaterNormal inputs
#define PROBE_ONB_N     float3(0.25, 0.5, 0.75)
#define PROBE_ONB_V     float3(1.0, 2.0, 3.0)
#define PROBE_POSITION  float3(0.25, -0.5, 1.0)

// lookAt / getPortalNormal inputs: forward and worldUp are not parallel, so the cross products do
// not vanish, and neither is -PROBE_BASE_NORMAL parallel to worldUp.
#define PROBE_FORWARD       float3(0.25, 0.5, 1.0)
#define PROBE_WORLD_UP      float3(0.0, 1.0, 0.0)
#define PROBE_BASE_NORMAL   float3(0.5, -0.25, 0.75)
#define PROBE_WORLD_OFFSET  float3(0.25, -0.5, 1.0)
// The operand of the :266 product: the golden multiplies by normalize(localN), and normalize()
// does not fold on the dxc half (measured), so the probe substitutes the literal (0.25, 0.5, 1.0)
// for it. The literal that equals the actual normalized value is not used, because its
// components are not dyadic and the two front-end folders then disagree in the last bit
// (measured before the substitution: glslc 0.46371302 against dxc 0.46371299); with this operand
// every product and sum of the site is exact in single precision and both halves fold the same
// numbers.
#define PROBE_LOCAL_N       float3(0.25, 0.5, 1.0)

// The literal matrices of the two portal blocks. They stand in for the lookAt() results of :563,
// :566, :812 and :815, which cannot fold (see the header comment).
//
//   inLook1   rows: ( 0.5,   0.25, -0.125 ) (-0.25, 0.875,  0.5  ) ( 0.125, -0.5,  0.75 )
//   outLook1  rows: ( 0.75, -0.125, 0.25 ) ( 0.25,  0.5,   -0.75 ) (-0.5,   0.125, 0.875 )
//   inLook2   rows: (-0.5,   0.75,  0.25 ) ( 0.25,  0.5,   -0.5  ) ( 0.5,  -0.25,  0.75 )
//   outLook2  rows: ( 0.25, -1.0,   0.5  ) ( 0.5,   1.0,    0.125) ( 0.25, -0.5,   0.75 )
#define PROBE_INLOOK1_M3  float3x3( 0.5, 0.25, -0.125,    -0.25, 0.875, 0.5,     0.125, -0.5, 0.75 )
#define PROBE_OUTLOOK1_M3 float3x3( 0.75, -0.125, 0.25,    0.25, 0.5, -0.75,    -0.5, 0.125, 0.875 )
#define PROBE_INLOOK2_M3  float3x3( -0.5, 0.75, 0.25,      0.25, 0.5, -0.5,      0.5, -0.25, 0.75 )
#define PROBE_OUTLOOK2_M3 float3x3( 0.25, -1.0, 0.5,       0.5, 1.0, 0.125,     0.25, -0.5, 0.75 )

#define PROBE_RAY1          float3(1.0, -2.0, 0.5)
#define PROBE_OFFSET1       float3(0.25, -0.5, 1.0)
#define PROBE_LO1           float2(0.25, -0.5)

#define PROBE_RAY2          float3(0.5, 1.0, -2.0)
#define PROBE_OFFSET2       float3(0.5, 0.25, -0.75)
#define PROBE_LO2           float2(0.75, 0.5)

// :1045 inputs
#define PROBE_REFR_POSITION float3(0.5, -0.25, 1.5)

[shader("raygeneration")]
void main()
{
    // The pinned entry point, instantiated from here: the whole primary raygen body is reachable in
    // the module on both halves.
    raygenPrimaryPrimaryMain();

    float v = 0.0;

    // ---- the four cast/product sites of getMotionForInfinitePoint -----------------------------
    probeOutput[1] = mul((float3x3)PROBE_VIEW_M4, PROBE_DIRECTION).x;
    probeOutput[2] = mul((float3x3)PROBE_VIEW_M4, PROBE_DIRECTION).y;
    probeOutput[3] = mul((float3x3)PROBE_VIEW_M4, PROBE_DIRECTION).z;

    probeOutput[4] = mul(PROBE_DIRECTION, (float3x3)PROBE_VIEW_M4).x;
    probeOutput[5] = mul(PROBE_DIRECTION, (float3x3)PROBE_VIEW_M4).y;
    probeOutput[6] = mul(PROBE_DIRECTION, (float3x3)PROBE_VIEW_M4).z;

    // the statement as it ships, reading the real uniform: not folded, present for reach
    probeOutput[7] = mul((float3x3)globalUniform.view, PROBE_DIRECTION).x;

    probeOutput[8] = mul((float3x3)PROBE_VIEWPREV_M4, PROBE_DIRECTION).x;
    probeOutput[9] = mul((float3x3)PROBE_VIEWPREV_M4, PROBE_DIRECTION).y;
    probeOutput[10] = mul((float3x3)PROBE_VIEWPREV_M4, PROBE_DIRECTION).z;

    probeOutput[11] = mul(PROBE_DIRECTION, (float3x3)PROBE_VIEWPREV_M4).x;
    probeOutput[12] = mul(PROBE_DIRECTION, (float3x3)PROBE_VIEWPREV_M4).y;
    probeOutput[13] = mul(PROBE_DIRECTION, (float3x3)PROBE_VIEWPREV_M4).z;

    probeOutput[14] = mul((float3x3)PROBE_PROJ_M4, mul((float3x3)PROBE_VIEW_M4, PROBE_DIRECTION)).x;
    probeOutput[15] = mul((float3x3)PROBE_PROJ_M4, mul((float3x3)PROBE_VIEW_M4, PROBE_DIRECTION)).y;
    probeOutput[16] = mul((float3x3)PROBE_PROJ_M4, mul((float3x3)PROBE_VIEW_M4, PROBE_DIRECTION)).z;

    probeOutput[17] = mul(mul((float3x3)PROBE_VIEW_M4, PROBE_DIRECTION), (float3x3)PROBE_PROJ_M4).x;
    probeOutput[18] = mul(mul((float3x3)PROBE_VIEW_M4, PROBE_DIRECTION), (float3x3)PROBE_PROJ_M4).y;
    probeOutput[19] = mul(mul((float3x3)PROBE_VIEW_M4, PROBE_DIRECTION), (float3x3)PROBE_PROJ_M4).z;

    probeOutput[20] = mul((float3x3)PROBE_PROJPREV_M4, mul((float3x3)PROBE_VIEWPREV_M4, PROBE_DIRECTION)).x;
    probeOutput[21] = mul((float3x3)PROBE_PROJPREV_M4, mul((float3x3)PROBE_VIEWPREV_M4, PROBE_DIRECTION)).y;
    probeOutput[22] = mul((float3x3)PROBE_PROJPREV_M4, mul((float3x3)PROBE_VIEWPREV_M4, PROBE_DIRECTION)).z;

    probeOutput[23] = mul(mul((float3x3)PROBE_VIEWPREV_M4, PROBE_DIRECTION), (float3x3)PROBE_PROJPREV_M4).x;
    probeOutput[24] = mul(mul((float3x3)PROBE_VIEWPREV_M4, PROBE_DIRECTION), (float3x3)PROBE_PROJPREV_M4).y;
    probeOutput[25] = mul(mul((float3x3)PROBE_VIEWPREV_M4, PROBE_DIRECTION), (float3x3)PROBE_PROJPREV_M4).z;

    {
        const float3 ndcCur  = mul((float3x3)PROBE_PROJ_M4, mul((float3x3)PROBE_VIEW_M4, PROBE_DIRECTION));
        const float3 ndcPrev = mul((float3x3)PROBE_PROJPREV_M4, mul((float3x3)PROBE_VIEWPREV_M4, PROBE_DIRECTION));
        const float2 screenCur  = ndcCur.xy  * 0.5 + 0.5;
        const float2 screenPrev = ndcPrev.xy * 0.5 + 0.5;
        probeOutput[26] = (screenPrev - screenCur).x;
        probeOutput[27] = (screenPrev - screenCur).y;
        probeOutput[28] = (screenCur - screenPrev).x;
    }

    // ---- getONB, the basis reads and the basis product -----------------------------------------
    {
        const float3x3 basis = getONB(PROBE_ONB_N);

        probeOutput[29] = getColumn(basis, 0).x;
        probeOutput[30] = getColumn(basis, 0).y;
        probeOutput[31] = getColumn(basis, 0).z;

        probeOutput[32] = getColumn(basis, 1).x;
        probeOutput[33] = getColumn(basis, 1).y;
        probeOutput[34] = getColumn(basis, 1).z;

        probeOutput[35] = getColumn(transpose(basis), 0).x;
        probeOutput[36] = getColumn(transpose(basis), 0).y;
        probeOutput[37] = getColumn(transpose(basis), 0).z;

        probeOutput[38] = mul(basis, PROBE_ONB_V).x;
        probeOutput[39] = mul(basis, PROBE_ONB_V).y;
        probeOutput[40] = mul(basis, PROBE_ONB_V).z;

        probeOutput[41] = mul(PROBE_ONB_V, basis).x;
        probeOutput[42] = mul(PROBE_ONB_V, basis).y;
        probeOutput[43] = mul(PROBE_ONB_V, basis).z;

        probeOutput[44] = dot(PROBE_POSITION, getColumn(basis, 0));
        probeOutput[45] = dot(PROBE_POSITION, getColumn(basis, 1));
        probeOutput[46] = dot(PROBE_POSITION, getColumn(transpose(basis), 0));
        probeOutput[47] = dot(PROBE_POSITION, getColumn(transpose(basis), 1));
    }

    // ---- lookAt: the shipped call for reach, then the constructor on explicit cross terms ------
    {

        // cross(forward, worldUp) and cross(right, forward), written out component by component:
        // cross() does not fold on either half (measured), the component arithmetic does.
        const float3 r = float3(PROBE_FORWARD.y * PROBE_WORLD_UP.z - PROBE_FORWARD.z * PROBE_WORLD_UP.y,
                                PROBE_FORWARD.z * PROBE_WORLD_UP.x - PROBE_FORWARD.x * PROBE_WORLD_UP.z,
                                PROBE_FORWARD.x * PROBE_WORLD_UP.y - PROBE_FORWARD.y * PROBE_WORLD_UP.x);
        const float3 u = float3(r.y * PROBE_FORWARD.z - r.z * PROBE_FORWARD.y,
                                r.z * PROBE_FORWARD.x - r.x * PROBE_FORWARD.z,
                                r.x * PROBE_FORWARD.y - r.y * PROBE_FORWARD.x);

        const float3x3 la = transpose(float3x3(r, u, PROBE_FORWARD));
        probeOutput[48] = getColumn(la, 0).x;
        probeOutput[49] = getColumn(la, 0).y;
        probeOutput[50] = getColumn(la, 0).z;
        probeOutput[51] = getColumn(la, 1).x;
        probeOutput[52] = getColumn(la, 1).y;
        probeOutput[53] = getColumn(la, 1).z;
        probeOutput[54] = getColumn(la, 2).x;
        probeOutput[55] = getColumn(la, 2).y;
        probeOutput[56] = getColumn(la, 2).z;

        // the constructor with the transpose dropped, which is how the defect looks in HLSL; the
        // GLSL half spells the same value transpose(mat3(...))
        const float3x3 laNoTranspose = float3x3(r, u, PROBE_FORWARD);
        probeOutput[57] = getColumn(laNoTranspose, 0).x;
        probeOutput[58] = getColumn(laNoTranspose, 0).y;
        probeOutput[59] = getColumn(laNoTranspose, 0).z;
        probeOutput[60] = getColumn(laNoTranspose, 1).x;
        probeOutput[61] = getColumn(laNoTranspose, 1).y;
        probeOutput[62] = getColumn(laNoTranspose, 1).z;
        probeOutput[63] = getColumn(laNoTranspose, 2).x;
        probeOutput[64] = getColumn(laNoTranspose, 2).y;
        probeOutput[65] = getColumn(laNoTranspose, 2).z;
    }

    // ---- getPortalNormal: the two column reads and the product ---------------------------------
    {
        const float3x3 inLookAt = PROBE_INLOOK1_M3;

        probeOutput[66] = dot(PROBE_WORLD_OFFSET, getColumn(inLookAt, 0));
        probeOutput[67] = dot(PROBE_WORLD_OFFSET, getColumn(inLookAt, 1));
        probeOutput[68] = dot(PROBE_WORLD_OFFSET, getColumn(transpose(inLookAt), 0));
        probeOutput[69] = dot(PROBE_WORLD_OFFSET, getColumn(transpose(inLookAt), 1));

        // the golden's normalize(localN), with PROBE_LOCAL_N as the literal operand, see the
        // macro

        probeOutput[70] = atan2(dot(PROBE_WORLD_OFFSET, getColumn(inLookAt, 1)), dot(PROBE_WORLD_OFFSET, getColumn(inLookAt, 0)));
        probeOutput[71] = atan2(dot(PROBE_WORLD_OFFSET, getColumn(inLookAt, 0)), dot(PROBE_WORLD_OFFSET, getColumn(inLookAt, 1)));

        probeOutput[72] = mul(inLookAt, PROBE_LOCAL_N).x;
        probeOutput[73] = mul(inLookAt, PROBE_LOCAL_N).y;
        probeOutput[74] = mul(inLookAt, PROBE_LOCAL_N).z;

        probeOutput[75] = mul(PROBE_LOCAL_N, inLookAt).x;
        probeOutput[76] = mul(PROBE_LOCAL_N, inLookAt).y;
        probeOutput[77] = mul(PROBE_LOCAL_N, inLookAt).z;

        // the golden's mod(): glslc folds it, dxc keeps OpExtInst Floor (see the header comment)
        probeOutput[78] = raygenPrimaryMod(5.0, 2.0);
        probeOutput[79] = raygenPrimaryMod(-1.0, 4.0);
        probeOutput[80] = -1.0 - 4.0 * trunc(-1.0 / 4.0);
    }

    // ---- the portal rotation of the first block (:563-575) -------------------------------------
    {
        probeOutput[81] = mul(PROBE_OUTLOOK1_M3, mul(transpose(PROBE_INLOOK1_M3), PROBE_RAY1)).x;
        probeOutput[82] = mul(PROBE_OUTLOOK1_M3, mul(transpose(PROBE_INLOOK1_M3), PROBE_RAY1)).y;
        probeOutput[83] = mul(PROBE_OUTLOOK1_M3, mul(transpose(PROBE_INLOOK1_M3), PROBE_RAY1)).z;

        probeOutput[84] = mul(PROBE_OUTLOOK1_M3, mul(PROBE_INLOOK1_M3, PROBE_RAY1)).x;
        probeOutput[85] = mul(PROBE_OUTLOOK1_M3, mul(PROBE_INLOOK1_M3, PROBE_RAY1)).y;
        probeOutput[86] = mul(PROBE_OUTLOOK1_M3, mul(PROBE_INLOOK1_M3, PROBE_RAY1)).z;

        probeOutput[87] = mul(transpose(PROBE_INLOOK1_M3), mul(PROBE_OUTLOOK1_M3, PROBE_RAY1)).x;
        probeOutput[88] = mul(transpose(PROBE_INLOOK1_M3), mul(PROBE_OUTLOOK1_M3, PROBE_RAY1)).y;
        probeOutput[89] = mul(transpose(PROBE_INLOOK1_M3), mul(PROBE_OUTLOOK1_M3, PROBE_RAY1)).z;

        probeOutput[90] = dot(PROBE_OFFSET1, getColumn(PROBE_INLOOK1_M3, 0));
        probeOutput[91] = dot(PROBE_OFFSET1, getColumn(PROBE_INLOOK1_M3, 1));
        probeOutput[92] = dot(PROBE_OFFSET1, getColumn(transpose(PROBE_INLOOK1_M3), 0));
        probeOutput[93] = dot(PROBE_OFFSET1, getColumn(transpose(PROBE_INLOOK1_M3), 1));

        probeOutput[94] = (PROBE_LO1.x * getColumn(PROBE_OUTLOOK1_M3, 0) + PROBE_LO1.y * getColumn(PROBE_OUTLOOK1_M3, 1)).x;
        probeOutput[95] = (PROBE_LO1.x * getColumn(PROBE_OUTLOOK1_M3, 0) + PROBE_LO1.y * getColumn(PROBE_OUTLOOK1_M3, 1)).y;
        probeOutput[96] = (PROBE_LO1.x * getColumn(PROBE_OUTLOOK1_M3, 0) + PROBE_LO1.y * getColumn(PROBE_OUTLOOK1_M3, 1)).z;

        probeOutput[97] = (PROBE_LO1.x * getColumn(transpose(PROBE_OUTLOOK1_M3), 0) + PROBE_LO1.y * getColumn(transpose(PROBE_OUTLOOK1_M3), 1)).x;
        probeOutput[98] = (PROBE_LO1.x * getColumn(transpose(PROBE_OUTLOOK1_M3), 0) + PROBE_LO1.y * getColumn(transpose(PROBE_OUTLOOK1_M3), 1)).y;
        probeOutput[99] = (PROBE_LO1.x * getColumn(transpose(PROBE_OUTLOOK1_M3), 0) + PROBE_LO1.y * getColumn(transpose(PROBE_OUTLOOK1_M3), 1)).z;
    }

    // ---- the portal rotation of the second block (:812-824) ------------------------------------
    {
        probeOutput[100] = mul(PROBE_OUTLOOK2_M3, mul(transpose(PROBE_INLOOK2_M3), PROBE_RAY2)).x;
        probeOutput[101] = mul(PROBE_OUTLOOK2_M3, mul(transpose(PROBE_INLOOK2_M3), PROBE_RAY2)).y;
        probeOutput[102] = mul(PROBE_OUTLOOK2_M3, mul(transpose(PROBE_INLOOK2_M3), PROBE_RAY2)).z;

        probeOutput[103] = mul(PROBE_OUTLOOK2_M3, mul(PROBE_INLOOK2_M3, PROBE_RAY2)).x;
        probeOutput[104] = mul(PROBE_OUTLOOK2_M3, mul(PROBE_INLOOK2_M3, PROBE_RAY2)).y;
        probeOutput[105] = mul(PROBE_OUTLOOK2_M3, mul(PROBE_INLOOK2_M3, PROBE_RAY2)).z;

        probeOutput[106] = dot(PROBE_OFFSET2, getColumn(PROBE_INLOOK2_M3, 0));
        probeOutput[107] = dot(PROBE_OFFSET2, getColumn(PROBE_INLOOK2_M3, 1));
        probeOutput[108] = dot(PROBE_OFFSET2, getColumn(transpose(PROBE_INLOOK2_M3), 0));
        probeOutput[109] = dot(PROBE_OFFSET2, getColumn(transpose(PROBE_INLOOK2_M3), 1));

        probeOutput[110] = (PROBE_LO2.x * getColumn(PROBE_OUTLOOK2_M3, 0) + PROBE_LO2.y * getColumn(PROBE_OUTLOOK2_M3, 1)).x;
        probeOutput[111] = (PROBE_LO2.x * getColumn(PROBE_OUTLOOK2_M3, 0) + PROBE_LO2.y * getColumn(PROBE_OUTLOOK2_M3, 1)).y;
        probeOutput[112] = (PROBE_LO2.x * getColumn(PROBE_OUTLOOK2_M3, 0) + PROBE_LO2.y * getColumn(PROBE_OUTLOOK2_M3, 1)).z;

        probeOutput[113] = (PROBE_LO2.x * getColumn(transpose(PROBE_OUTLOOK2_M3), 0) + PROBE_LO2.y * getColumn(transpose(PROBE_OUTLOOK2_M3), 1)).x;
        probeOutput[114] = (PROBE_LO2.x * getColumn(transpose(PROBE_OUTLOOK2_M3), 0) + PROBE_LO2.y * getColumn(transpose(PROBE_OUTLOOK2_M3), 1)).y;
        probeOutput[115] = (PROBE_LO2.x * getColumn(transpose(PROBE_OUTLOOK2_M3), 0) + PROBE_LO2.y * getColumn(transpose(PROBE_OUTLOOK2_M3), 1)).z;

        // the same recompositions, summed so that the three components above are not the only
        // readers (and so that the two blocks are not dropped as dead stores)
        probeOutput[116] = dot(PROBE_LO1.x * getColumn(PROBE_OUTLOOK1_M3, 0) + PROBE_LO1.y * getColumn(PROBE_OUTLOOK1_M3, 1),
                               PROBE_LO1.x * getColumn(PROBE_OUTLOOK1_M3, 0) + PROBE_LO1.y * getColumn(PROBE_OUTLOOK1_M3, 1));
        probeOutput[117] = dot(PROBE_LO2.x * getColumn(PROBE_OUTLOOK2_M3, 0) + PROBE_LO2.y * getColumn(PROBE_OUTLOOK2_M3, 1),
                               PROBE_LO2.x * getColumn(PROBE_OUTLOOK2_M3, 0) + PROBE_LO2.y * getColumn(PROBE_OUTLOOK2_M3, 1));
    }

    // ---- :1045, the projection * view * vec4 chain ---------------------------------------------
    {
        const float4 refrClipPos = mul(mul(PROBE_PROJ_M4, PROBE_VIEW_M4), float4(PROBE_REFR_POSITION, 1.0));
        probeOutput[118] = refrClipPos.x;
        probeOutput[119] = refrClipPos.y;
        probeOutput[120] = refrClipPos.z;
        probeOutput[121] = refrClipPos.w;

        const float4 refrClipPosRightAssoc = mul(PROBE_PROJ_M4, mul(PROBE_VIEW_M4, float4(PROBE_REFR_POSITION, 1.0)));
        probeOutput[122] = refrClipPosRightAssoc.x;
        probeOutput[123] = refrClipPosRightAssoc.y;
        probeOutput[124] = refrClipPosRightAssoc.z;
        probeOutput[125] = refrClipPosRightAssoc.w;

        const float4 refrClipPosSwapped = mul(PROBE_VIEW_M4, mul(PROBE_PROJ_M4, float4(PROBE_REFR_POSITION, 1.0)));
        probeOutput[126] = refrClipPosSwapped.x;
        probeOutput[127] = refrClipPosSwapped.y;
        probeOutput[128] = refrClipPosSwapped.z;
        probeOutput[129] = refrClipPosSwapped.w;

        // the statement as it ships, reading the real uniform: not folded, present for reach
        probeOutput[130] = mul(mul(globalUniform.projection, globalUniform.view), float4(PROBE_REFR_POSITION, 1.0)).z;
    }

    // ---- the portal instances descriptor: read here so the pair compares its kind ----------------
    // The golden declares the block as `uniform readonly` and the host binds it as a uniform
    // buffer (PortalList.cpp), so the port keeps the block spelling; this read is what keeps the
    // descriptor live in the module, which is what makes the checker compare it at all. The two
    // slots are runtime buffer contents, so neither folds -- they are reach witnesses, like 130.
    probeOutput[131] = portalInstances.g_portals[0].inPosition.x;
    probeOutput[132] = portalInstances.g_portals[0].outDirection.z;

    // ---- the functions of the golden that the entry point does not call ------------------------
    // Every remaining function is instantiated from here, with arguments that keep every divisor
    // and every normalize operand away from zero.
    v += getMotionVectorForUpscaler(PROBE_DIRECTION.xy).x;
    v += getMotionForInfinitePoint(int2(3, 4)).x;
    v += (float)getNewRayMedia(0, MEDIA_TYPE_VACUUM, 0u);

    storeQ2GBuffer(
        int2(1, 1), float3(0.5, 0.25, 1.0), 0.25,
        0.5, 0.75,
        2.0,
        0.125, 3.0,
        float3(0.25, 0.5, 0.75), 0.5,
        float4(0.0, 0.0, 0.0, 1.0),
        7u);

    storeSky(int2(1, 1), PROBE_DIRECTION, false, float3(0.5, 0.25, 1.0), 3.0, float4(0.0, 0.0, 0.0, 1.0));

    RayCone rayCone;
    rayCone.width = 0.5;
    rayCone.spreadAngle = 0.25;
    v += getWaterNormal(rayCone, PROBE_DIRECTION, PROBE_ONB_N, PROBE_POSITION, false).x;
    v += getNormal(PROBE_POSITION, PROBE_ONB_V, PROBE_ONB_N, rayCone, PROBE_DIRECTION, true, false).y;
    v += getNormal(PROBE_POSITION, PROBE_ONB_V, PROBE_ONB_N, rayCone, PROBE_DIRECTION, false, false).z;

    v += getPortalNormal(PROBE_BASE_NORMAL, PROBE_WORLD_OFFSET).x;
    v += isBackface(PROBE_ONB_N, PROBE_DIRECTION) ? 1.0 : 0.0;
    v += getColumn(lookAt(PROBE_FORWARD, PROBE_WORLD_UP), 0).x;

    probeOutput[0] = v;
}
